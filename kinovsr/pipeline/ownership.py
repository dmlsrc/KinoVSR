"""Retain-safe ownership for native CVPixelBuffer pipeline outputs."""

from __future__ import annotations

from collections.abc import Iterator

from kinovsr.processors import FrameUnit, Layout, StreamSpec

_CV_LAYOUTS = frozenset({Layout.CV_NV12, Layout.CV_BGRA, Layout.CV_RGBA_HALF})


class OwnedCvOutputs:
    """Copy borrowed native outputs before exposing them to a host caller."""

    def __init__(self, run: Iterator[FrameUnit]) -> None:
        self._run = run
        from kinovsr.media.pixel_buffers import copy_pixel_buffer

        self._copy = copy_pixel_buffer

    def __iter__(self) -> OwnedCvOutputs:
        return self

    def __next__(self) -> FrameUnit:
        unit = next(self._run)
        return unit.with_payload(self._copy(unit.payload))

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
        return OwnedCvOutputs(run)
    return run


__all__ = ["OwnedCvOutputs", "retain_safe_outputs"]
