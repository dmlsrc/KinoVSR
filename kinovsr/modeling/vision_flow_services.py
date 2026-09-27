"""Bounded, driver-owned Vision optical-flow services (revision 1)."""

import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from kinovsr.processors.errors import append_cleanup_context

from .vision_flow import VisionFlowEngine, open_vision_flow_engine


@dataclass(slots=True)
class _Entry:
    service: VisionFlowEngine
    borrowed: bool = False


class VisionFlowServices:
    """Own a small LRU of geometry-specific Vision flow services.

    A service owns mutable CVPixelBuffers and native flow sessions, so a
    lease is exclusive per key.  Every product instance is owned by one
    pipeline driver and borrowed from a single thread (verified at runtime
    on flow-aligned runs), so there is no waiting: borrowing a key that
    is already borrowed, or needing an eviction victim while every entry is
    borrowed, is a caller bug and raises immediately.  The lock keeps the
    bookkeeping coherent for host embedders; it is not a scheduling
    primitive.

    Each service is a Vision optical flow revision 1 engine at Medium
    accuracy, pinned, that passed the up-front flow self-test, so the
    geometry is the complete compatibility key.
    """

    def __init__(self, max_geometries: int = 2) -> None:
        if isinstance(max_geometries, bool) or not isinstance(max_geometries, int):
            raise ValueError("max_geometries must be an integer")
        if max_geometries < 1:
            raise ValueError("max_geometries must be positive")
        self.max_geometries = max_geometries
        self._entries: OrderedDict[tuple[int, int], _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False

    @staticmethod
    def _make_service(width: int, height: int) -> VisionFlowEngine:
        return open_vision_flow_engine(width, height, consumer="flow_mode='vision'")

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    @contextmanager
    def borrow(self, width: int, height: int) -> Iterator[VisionFlowEngine]:
        """Yield one live, exclusively borrowed service for ``width x height``."""
        key = (int(width), int(height))
        if key[0] < 1 or key[1] < 1:
            raise ValueError("optical-flow geometry must be positive")
        evicted: VisionFlowEngine | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("optical-flow services are closed")
            entry = self._entries.get(key)
            if entry is not None:
                if entry.borrowed:
                    raise RuntimeError(f"optical-flow service {key} is already borrowed")
                entry.borrowed = True
                self._entries.move_to_end(key)
            elif len(self._entries) >= self.max_geometries:
                idle = next(
                    (candidate for candidate, item in self._entries.items() if not item.borrowed),
                    None,
                )
                if idle is None:
                    raise RuntimeError(
                        "every optical-flow service is borrowed; raise "
                        "max_geometries or release a lease first"
                    )
                evicted = self._entries.pop(idle).service

        if entry is None:
            # Construction and eviction cleanup run outside the lock; the
            # entry is published only after the service exists.
            if evicted is not None:
                evicted.close()
            service = self._make_service(key[0], key[1])
            redundant: VisionFlowEngine | None = None
            failure: BaseException | None = None
            with self._lock:
                existing = self._entries.get(key)
                if self._closed:
                    redundant = service
                    failure = RuntimeError("optical-flow services closed during construction")
                elif existing is not None:
                    # A host raced the same geometry in; ours is redundant.
                    redundant = service
                    if existing.borrowed:
                        failure = RuntimeError(f"optical-flow service {key} is already borrowed")
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
                except BaseException as cleanup:  # broad: chained
                    if failure is None:
                        raise
                    append_cleanup_context(failure, cleanup)
            if failure is not None:
                raise failure

        assert entry is not None  # every path that leaves it None raised above
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
                raise RuntimeError("cannot close optical-flow services while a service is borrowed")
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
                append_cleanup_context(failures[0], cleanup)
            raise failures[0]


@contextmanager
def vision_flow_services_scope(
    services: VisionFlowServices | None,
    *,
    max_geometries: int,
) -> Iterator[VisionFlowServices]:
    """Borrow an injected manager or own one with cleanup-safe precedence."""
    if services is not None:
        yield services
        return
    owned = VisionFlowServices(max_geometries)
    try:
        yield owned
    except BaseException as active:
        try:
            owned.close()
        except BaseException as cleanup:
            append_cleanup_context(active, cleanup)
        raise
    else:
        owned.close()


__all__ = ["VisionFlowServices", "vision_flow_services_scope"]
