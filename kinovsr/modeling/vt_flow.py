"""Bounded, driver-owned native optical-flow services (VT or Vision r1)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _Entry:
    service: Any
    borrowed: bool = False


class VtFlowServices:
    """Own a small LRU of (backend, geometry)-specific flow services.

    A service owns mutable CVPixelBuffers and native flow sessions, so a
    lease is exclusive per key.  Every product instance is owned by one
    pipeline driver and borrowed from a single thread (verified at runtime
    on flow-aligned runs), so there is no waiting: borrowing a key that
    is already borrowed, or needing an eviction victim while every entry is
    borrowed, is a caller bug and raises immediately.  The lock keeps the
    bookkeeping coherent for host embedders; it is not a scheduling
    primitive.

    Backends: "vt" (VTOpticalFlow Quality tier) and "vision" (Vision
    optical flow revision 1, Medium accuracy, pinned).  The remaining
    service configuration is fixed here (window and self-test), so
    (backend, geometry) is the complete compatibility key.
    """

    _BACKENDS = ("vt", "vision")

    def __init__(self, max_geometries: int = 2) -> None:
        if isinstance(max_geometries, bool) or not isinstance(max_geometries, int):
            raise ValueError("max_geometries must be an integer")
        if max_geometries < 1:
            raise ValueError("max_geometries must be positive")
        self.max_geometries = max_geometries
        self._entries: OrderedDict[tuple[str, int, int], _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False

    @staticmethod
    def _make_service(width: int, height: int, backend: str = "vt") -> Any:
        from kinovsr.processors.mc import McTemporalDenoiser

        return McTemporalDenoiser(
            width,
            height,
            strength=0.0,
            window=1,
            self_test=True,
            flow=backend,
        )

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    @contextmanager
    def borrow(self, width: int, height: int, backend: str = "vt"):
        """Yield one live, exclusively borrowed ``backend`` service for
        ``width x height``."""
        if backend not in self._BACKENDS:
            raise ValueError(
                f"flow backend must be one of {self._BACKENDS}, got {backend!r}")
        key = (str(backend), int(width), int(height))
        if key[1] < 1 or key[2] < 1:
            raise ValueError("optical-flow geometry must be positive")
        evicted: Any = None
        with self._lock:
            if self._closed:
                raise RuntimeError("optical-flow services are closed")
            entry = self._entries.get(key)
            if entry is not None:
                if entry.borrowed:
                    raise RuntimeError(
                        f"optical-flow service {key} is already borrowed")
                entry.borrowed = True
                self._entries.move_to_end(key)
            elif len(self._entries) >= self.max_geometries:
                idle = next(
                    (candidate for candidate, item in self._entries.items()
                     if not item.borrowed),
                    None,
                )
                if idle is None:
                    raise RuntimeError(
                        "every optical-flow service is borrowed; raise "
                        "max_geometries or release a lease first")
                evicted = self._entries.pop(idle).service

        if entry is None:
            # Construction and eviction cleanup run outside the lock; the
            # entry is published only after the service exists.
            if evicted is not None:
                evicted.close()
            service = self._make_service(key[1], key[2], key[0])
            redundant: Any = None
            failure: BaseException | None = None
            with self._lock:
                existing = self._entries.get(key)
                if self._closed:
                    redundant = service
                    failure = RuntimeError(
                        "optical-flow services closed during construction")
                elif existing is not None:
                    # A host raced the same geometry in; ours is redundant.
                    redundant = service
                    if existing.borrowed:
                        failure = RuntimeError(
                            f"optical-flow service {key} is already "
                            f"borrowed")
                    else:
                        existing.borrowed = True
                        self._entries.move_to_end(key)
                        entry = existing
                else:
                    entry = _Entry(service=service, borrowed=True)
                    self._entries[key] = entry
            if redundant is not None:
                try:
                    redundant.close()
                except BaseException as cleanup:  # noqa: BLE001 - chained
                    if failure is None:
                        raise
                    _append_cleanup_context(failure, cleanup)
            if failure is not None:
                raise failure

        try:
            yield entry.service
        finally:
            with self._lock:
                entry.borrowed = False

    def close(self) -> None:
        """Close every service exactly once; later calls are no-ops.

        Closing while a lease is live is a caller bug (the product driver
        closes on the borrowing thread after its last lease exits) and
        raises before anything is torn down.
        """
        with self._lock:
            if self._closed and not self._entries:
                return
            if any(entry.borrowed for entry in self._entries.values()):
                raise RuntimeError(
                    "cannot close optical-flow services while a service "
                    "is borrowed")
            self._closed = True
            services = [entry.service for entry in self._entries.values()]
            self._entries.clear()
        failures: list[BaseException] = []
        for service in services:
            try:
                service.close()
            except BaseException as exc:  # close the rest before delivery
                failures.append(exc)
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
