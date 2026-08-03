"""Reference-counted payload ownership and readiness envelopes."""

from __future__ import annotations

import contextlib
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from kinovsr.processors import FrameUnit


class Completion(Protocol):
    def wait(self) -> None: ...


class StorageKind(Enum):
    """The backing-storage fact that is deliberately absent from FrameSpec."""

    PYTHON = "python"
    MLX_ARRAY = "mlx_array"
    HOST_BUFFER = "host_buffer"
    CV_PIXEL_BUFFER = "cv_pixel_buffer"
    IO_SURFACE = "io_surface"
    MTL_TEXTURE = "mtl_texture"
    MTL_BUFFER = "mtl_buffer"
    ANE_RAW_SLOT = "ane_raw_slot"


@dataclass(frozen=True, slots=True)
class StorageDescriptor:
    kind: StorageKind = StorageKind.PYTHON
    borrowed: bool = False
    reusable: bool = False
    label: str | None = None


PYTHON_STORAGE = StorageDescriptor()


@dataclass(frozen=True, slots=True)
class ImmediateCompletion:
    def wait(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class FutureCompletion:
    future: Future[Any]

    def wait(self) -> None:
        self.future.result()


IMMEDIATE = ImmediateCompletion()


class _LeaseState:
    __slots__ = (
        "bytes",
        "descriptor",
        "lock",
        "on_release",
        "payload",
        "references",
        "released",
    )

    def __init__(
        self,
        payload: Any,
        estimated_bytes: int,
        on_release: Any = None,
        descriptor: StorageDescriptor = PYTHON_STORAGE,
    ) -> None:
        self.payload = payload
        self.bytes = max(0, int(estimated_bytes))
        self.descriptor = descriptor
        self.on_release = on_release
        self.references = 1
        self.lock = threading.Lock()
        self.released = threading.Event()


class PayloadLease:
    """One independently releasable handle to shared payload storage."""

    __slots__ = ("_released", "_state")

    def __init__(
        self,
        payload: Any = None,
        *,
        estimated_bytes: int = 0,
        on_release: Any = None,
        descriptor: StorageDescriptor = PYTHON_STORAGE,
        _state: _LeaseState | None = None,
    ) -> None:
        self._state = (
            _LeaseState(payload, estimated_bytes, on_release, descriptor)
            if _state is None
            else _state
        )
        self._released = False

    @property
    def payload(self) -> Any:
        if self._released:
            raise RuntimeError("payload lease was already released")
        return self._state.payload

    @property
    def estimated_bytes(self) -> int:
        return self._state.bytes

    @property
    def descriptor(self) -> StorageDescriptor:
        return self._state.descriptor

    @property
    def references(self) -> int:
        with self._state.lock:
            return self._state.references

    @property
    def released(self) -> bool:
        return self._released

    def retain(self) -> PayloadLease:
        if self._released:
            raise RuntimeError("cannot retain a released payload lease")
        with self._state.lock:
            if self._state.references <= 0:
                raise RuntimeError("cannot retain released payload storage")
            self._state.references += 1
        return PayloadLease(_state=self._state)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        callback = None
        with self._state.lock:
            self._state.references -= 1
            if self._state.references < 0:
                raise RuntimeError("payload lease reference count underflow")
            if self._state.references == 0:
                callback, self._state.on_release = self._state.on_release, None
                self._state.payload = None
                self._state.released.set()
        if callback is not None:
            callback()

    def wait_released(self, timeout: float | None = None) -> bool:
        """Wait until every retained handle releases the shared storage."""
        return self._state.released.wait(timeout)

    def promote(
        self,
        copier: Any,
        *,
        descriptor: StorageDescriptor | None = None,
        estimated_bytes: int | None = None,
        on_release: Any = None,
    ) -> PayloadLease:
        """Copy borrowed/reusable storage into a new durable lease.

        The family supplies the storage-specific copier.  Keeping promotion
        explicit prevents the generic runtime from guessing how to retain an
        IOSurface, snapshot a persistent ANE backing, or copy an MTL resource.
        """
        if not callable(copier):
            raise TypeError("payload lease promotion requires a copier")
        payload = copier(self.payload)
        return PayloadLease(
            payload,
            estimated_bytes=(
                self.estimated_bytes
                if estimated_bytes is None
                else estimated_bytes
            ),
            on_release=on_release,
            descriptor=(
                descriptor
                if descriptor is not None
                else StorageDescriptor(kind=self.descriptor.kind)
            ),
        )

    def __enter__(self) -> PayloadLease:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:
        # Finalizers cannot report errors safely.
        with contextlib.suppress(BaseException):
            self.release()


@dataclass(slots=True)
class Envelope:
    unit: FrameUnit
    lease: PayloadLease
    readiness: Completion = IMMEDIATE
    sequence: int = 0

    @property
    def estimated_bytes(self) -> int:
        return self.lease.estimated_bytes

    def release(self) -> None:
        self.lease.release()
        # The lease state drops its payload at the last release, but a stale
        # transport Envelope can otherwise retain the same object through its
        # immutable FrameUnit. Clear this internal carrier as well; any unit
        # already handed to a consumer is a separate reference and follows the
        # public next-pull/close lifetime contract.
        self.unit = self.unit.with_payload(None)


__all__ = [
    "Completion",
    "Envelope",
    "FutureCompletion",
    "IMMEDIATE",
    "ImmediateCompletion",
    "PayloadLease",
    "StorageDescriptor",
    "StorageKind",
]
