"""Retain-safe ownership for native CVPixelBuffer pipeline outputs."""

from __future__ import annotations

from collections.abc import Iterator

from kinovsr.processors import FrameUnit, Layout, StreamSpec

_CV_LAYOUTS = frozenset({Layout.CV_NV12, Layout.CV_BGRA, Layout.CV_RGBA_HALF})


class OwnedCvOutputs:
    """Copy borrowed native outputs before exposing them to a host caller."""

    def __init__(self, run: Iterator[FrameUnit], frame_spec: object | None = None) -> None:
        self._run = run
        from kinovsr.media.pixel_buffers import copy_pixel_buffer

        self._copy = lambda payload: copy_pixel_buffer(
            payload, frame_spec=frame_spec
        )

    def __iter__(self) -> OwnedCvOutputs:
        return self

    def __next__(self) -> FrameUnit:
        to_raise: BaseException | None = None
        try:
            unit = next(self._run)
        except StopIteration:
            raise
        except BaseException as active:
            to_raise = self._close_after_failure(active)
        else:
            try:
                return unit.with_payload(self._copy(unit.payload))
            except BaseException as active:
                # A copy-side StopIteration is a failure, not source exhaustion.
                to_raise = self._close_after_failure(active)
        # Raise outside the handler so Python does not overwrite the explicit,
        # acyclic precedence chain constructed above.
        raise to_raise

    def _close_after_failure(self, active: BaseException) -> BaseException:
        try:
            self.close()
        except BaseException as cleanup:
            # Import lazily: scheduler imports this ownership wrapper.
            from .builder import _append_context

            active_is_interrupt = not isinstance(active, Exception)
            cleanup_is_interrupt = not isinstance(cleanup, Exception)
            if cleanup_is_interrupt and not active_is_interrupt:
                _append_context(cleanup, [active])
                return cleanup
            _append_context(active, [cleanup])
        return active

    def close(self) -> None:
        close = getattr(self._run, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> OwnedCvOutputs:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        exit_run = getattr(self._run, "__exit__", None)
        if callable(exit_run):
            exit_run(exc_type, exc, tb)
        else:
            self.close()


def retain_safe_outputs(
    run: Iterator[FrameUnit],
    output_spec: StreamSpec,
    *,
    retain_outputs: bool,
) -> Iterator[FrameUnit]:
    """Wrap native outputs when the caller is allowed to retain them."""
    if retain_outputs and output_spec.frame.layout in _CV_LAYOUTS:
        return OwnedCvOutputs(run, output_spec.frame)
    return run


__all__ = ["OwnedCvOutputs", "retain_safe_outputs"]
