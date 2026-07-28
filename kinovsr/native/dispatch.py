"""Depth-one dispatch pipelining shared by the accelerator backends."""

from __future__ import annotations

from typing import Any


class DispatchPipeline:
    """One accelerator dispatch in flight on a dedicated worker thread.

    The reusable overlap primitive for every ANE processor: submit a
    zero-argument job - a prediction plus its host-side buffer
    bookkeeping, pure Core ML / MPSGraph and Python, NEVER MLX, which
    stays on the caller's thread - keep working, and join before touching
    anything the job writes. One slot is deliberate: depth-one pipelining
    bounds memory and latency, keeps dispatches back to back (an ANE
    dispatch issued after 10 ms or more of host idleness pays a 15-23 ms
    power-state ramp no host-side warm-up avoids), and is all the
    concurrency a synchronous pull pipeline can put to use.
    """

    def __init__(self, name: str = "ane-dispatch"):
        self._name = name
        self._executor: Any = None
        self._future: Any = None

    @property
    def in_flight(self) -> bool:
        return self._future is not None

    def idle(self) -> bool:
        """True when a join would not block (nothing pending, or done)."""
        return self._future is None or self._future.done()

    def submit(self, job) -> None:
        if self._future is not None:
            raise RuntimeError("a dispatch is already in flight; join first")
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=self._name)
        self._future = self._executor.submit(job)

    def join(self) -> None:
        """Wait out the in-flight job; re-raises what it raised."""
        future, self._future = self._future, None
        if future is not None:
            future.result()

    def drain(self) -> None:
        """Absorb the in-flight job on an error path (its own error is
        suppressed so the primary error wins)."""
        import contextlib

        future, self._future = self._future, None
        if future is not None:
            with contextlib.suppress(BaseException):
                future.result()

    def close(self) -> None:
        self.drain()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


__all__ = ["DispatchPipeline"]
