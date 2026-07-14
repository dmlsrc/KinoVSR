"""Bounded, driver-owned VideoToolbox optical-flow services."""

from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class _Entry:
    service: Any
    users: int = 0
    use_lock: Any = field(default_factory=threading.Lock)


class VtFlowServices:
    """Own a small LRU of geometry-specific VTOpticalFlow services.

    A service owns mutable CVPixelBuffers and VTFrameProcessors, so leases for
    one geometry are serialized. Reservations count while callers wait for the
    per-service lock, preventing eviction or close underneath an active user.
    Construction, eviction cleanup, and native computation happen outside the
    cache lock. The owner closes the complete cache with its pipeline driver.

    Service configuration is fixed here (quality tier, window, and self-test),
    so geometry is the complete compatibility key. A future configurable tier
    must become part of the key or use a separate manager instance.
    """

    def __init__(self, max_geometries: int = 2) -> None:
        if isinstance(max_geometries, bool) or not isinstance(max_geometries, int):
            raise ValueError("max_geometries must be an integer")
        if max_geometries < 1:
            raise ValueError("max_geometries must be positive")
        self.max_geometries = max_geometries
        self._entries: OrderedDict[tuple[int, int], _Entry] = OrderedDict()
        self._creating: set[tuple[int, int]] = set()
        self._condition = threading.Condition()
        self._closed = False
        self._close_complete = False

    @staticmethod
    def _make_service(width: int, height: int) -> Any:
        from kinovsr.processors.mc import McTemporalDenoiser

        return McTemporalDenoiser(
            width,
            height,
            strength=0.0,
            window=1,
            self_test=True,
        )

    @property
    def size(self) -> int:
        with self._condition:
            return len(self._entries)

    def _reserve(self, key: tuple[int, int]) -> _Entry:
        evicted: Any = None
        while True:
            with self._condition:
                if self._closed:
                    raise RuntimeError("VT optical-flow services are closed")
                entry = self._entries.get(key)
                if entry is not None:
                    entry.users += 1
                    self._entries.move_to_end(key)
                    return entry
                if key in self._creating:
                    self._condition.wait()
                    continue
                if len(self._entries) + len(self._creating) >= self.max_geometries:
                    idle_key = next(
                        (candidate for candidate, item in self._entries.items() if item.users == 0),
                        None,
                    )
                    if idle_key is None:
                        self._condition.wait()
                        continue
                    evicted = self._entries.pop(idle_key).service
                self._creating.add(key)
                break

        try:
            if evicted is not None:
                evicted.close()
            service = self._make_service(*key)
        except BaseException:
            with self._condition:
                self._creating.remove(key)
                self._condition.notify_all()
            raise

        with self._condition:
            close_after_creation = self._closed
            if not close_after_creation:
                entry = _Entry(service=service, users=1)
                self._entries[key] = entry
                self._creating.remove(key)
                self._condition.notify_all()
                return entry

        active = RuntimeError("VT optical-flow services closed during construction")
        try:
            service.close()
        except BaseException as cleanup:
            _append_cleanup_context(active, cleanup)
        finally:
            with self._condition:
                self._creating.remove(key)
                self._condition.notify_all()
        raise active

    @contextmanager
    def borrow(self, width: int, height: int):
        """Yield one live, exclusively borrowed service for ``width x height``."""
        key = (int(width), int(height))
        if key[0] < 1 or key[1] < 1:
            raise ValueError("VT optical-flow geometry must be positive")
        entry = self._reserve(key)
        try:
            entry.use_lock.acquire()
        except BaseException:
            with self._condition:
                entry.users -= 1
                self._condition.notify_all()
            raise
        try:
            yield entry.service
        finally:
            entry.use_lock.release()
            with self._condition:
                entry.users -= 1
                self._condition.notify_all()

    def close(self) -> None:
        """Wait for active leases, then close every service exactly once."""
        with self._condition:
            if self._closed:
                while not self._close_complete:
                    self._condition.wait()
                return
            self._closed = True
            while self._creating or any(entry.users for entry in self._entries.values()):
                self._condition.wait()
            services = [entry.service for entry in self._entries.values()]
            self._entries.clear()

        failures: list[BaseException] = []
        try:
            for service in services:
                try:
                    service.close()
                except BaseException as exc:  # close the rest before delivery
                    failures.append(exc)
        finally:
            with self._condition:
                self._close_complete = True
                self._condition.notify_all()
        if failures:
            for cleanup in failures[1:]:
                _append_cleanup_context(failures[0], cleanup)
            raise failures[0]


def _append_cleanup_context(primary: BaseException, cleanup: BaseException) -> None:
    """Append cleanup without retaining the active exception as a back-link."""
    cursor = primary
    primary_ids = {id(primary)}
    while cursor.__context__ is not None:
        next_error = cursor.__context__
        if id(next_error) in primary_ids:
            cursor.__context__ = None
            break
        primary_ids.add(id(next_error))
        cursor = next_error
    tail = cursor

    if id(cleanup) in primary_ids:
        return
    cursor = cleanup
    cleanup_ids = {id(cleanup)}
    while cursor.__context__ is not None:
        next_error = cursor.__context__
        if id(next_error) in primary_ids or id(next_error) in cleanup_ids:
            cursor.__context__ = None
            break
        cleanup_ids.add(id(next_error))
        cursor = next_error
    tail.__context__ = cleanup


@contextmanager
def vt_flow_services_scope(
    services: VtFlowServices | None,
    *,
    max_geometries: int,
):
    """Borrow an injected manager or own one with cleanup-safe precedence."""
    if services is not None:
        yield services
        return
    owned = VtFlowServices(max_geometries)
    try:
        yield owned
    except BaseException as active:
        try:
            owned.close()
        except BaseException as cleanup:
            _append_cleanup_context(active, cleanup)
        raise
    else:
        owned.close()


__all__ = ["VtFlowServices", "vt_flow_services_scope"]
