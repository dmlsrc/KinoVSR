"""Adapter from the family driver shape to the Processor protocol.

The learned families already speak a common streaming dialect:
``feed(frame, token) -> [(out, token), ...]``, ``flush() -> [...]``,
``reset()``, and optionally ``close()``. The token rides through the
family's internal delay untouched, which is exactly the timestamp
bookkeeping a typed pipeline needs: this adapter passes the whole input
:class:`~kinovsr.processors.units.FrameUnit` as the token, so a delayed
output re-emerges bound to the unit (PTS, duration) it was computed FROM,
no matter how deep the family's buffer is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from .boundaries import Boundary
from .protocol import PipelineContext
from .specs import StreamSpec, luma_coefficients
from .units import FrameUnit

# The shared denoise luma/chroma blend keys (planning 07). A denoise family
# adds these to its accepted keys and threads the parsed strengths into
# FeedFlushProcessor, which owns the recombination.
LUMA_CHROMA_KEYS = ("luma_strength", "chroma_strength")


def parse_luma_chroma(raw: Mapping[str, Any]) -> tuple[float, float]:
    """Parse the shared denoise ``luma_strength``/``chroma_strength`` keys.

    Both default to 1.0 (full denoiser effect, no split). Values are
    deliberately unclamped: >1 over-drives and <1 keeps original texture,
    matching the ``--denoise-luma-strength`` / ``--denoise-chroma-strength``
    CLI dials.
    """
    from kinovsr.config.helpers import typed_value

    return (typed_value(raw, "luma_strength", float, 1.0),
            typed_value(raw, "chroma_strength", float, 1.0))


class FeedFlushDriver(Protocol):
    """What the wrapped family object must provide."""

    def feed(self, frame: Any, token: Any = None) -> list: ...

    def flush(self) -> list: ...

    def reset(self) -> None: ...


class PerFrameDriver:
    """feed()/flush() shape over a per-frame engine (``denoise(x) -> x``).

    Several single-image families expose ``denoise``/``reset``/``close``
    without the streaming dialect; this adapter gives them the driver
    shape :class:`FeedFlushProcessor` pumps, with reset/close passing
    through when the engine has them.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def feed(self, frame: Any, token: Any = None) -> list:
        return [(self._engine.denoise(frame), token)]

    def flush(self) -> list:
        return []

    def reset(self) -> None:
        reset = getattr(self._engine, "reset", None)
        if callable(reset):
            reset()

    def close(self) -> None:
        close = getattr(self._engine, "close", None)
        if callable(close):
            close()


class FeedFlushProcessor:
    """Wrap a feed/flush family driver as a pipeline Processor.

    Construction is deferred to ``prepare`` via a zero-argument factory,
    so pipeline build stays cheap and weight loading happens at the
    documented lifecycle edge.

    Optional luma/chroma split: when a denoise family passes
    ``luma_strength``/``chroma_strength`` other than 1.0, each output is
    recombined against the input it was computed from with separate blend
    strengths per channel group (planning 07's shared keys). The token
    threading is exactly what makes this correct through a delay line -
    ``token`` is the input unit each output emerged from, so the blend
    pairs a delayed output with its own source frame, not the frame
    currently arriving. (Kr, Kb) bind from the input StreamSpec's color
    matrix at prepare so the split matches the clip's color space.
    """

    def __init__(self, make_driver: Any, *,
                 luma_strength: float = 1.0,
                 chroma_strength: float = 1.0) -> None:
        self._make_driver = make_driver
        self._driver: FeedFlushDriver | None = None
        self._luma_strength = float(luma_strength)
        self._chroma_strength = float(chroma_strength)
        # A blend closure bound at prepare when the split is active, else None.
        self._blend: Any = None

    def prepare(self, input_spec: StreamSpec,
                context: PipelineContext) -> None:
        if self._driver is None:
            self._driver = self._make_driver()
        if self._blend is None and (self._luma_strength != 1.0
                                    or self._chroma_strength != 1.0):
            from kinovsr.media.yuv import luma_chroma_blend

            kr, kb = luma_coefficients(input_spec.frame.color_matrix)
            al, ac = self._luma_strength, self._chroma_strength

            def blend(orig: Any, new: Any) -> Any:
                return luma_chroma_blend(orig, new, al, ac, kr, kb).astype(
                    new.dtype)

            self._blend = blend

    def process(self, unit: FrameUnit,
                context: PipelineContext) -> Iterable[FrameUnit]:
        for out, token in self._driver.feed(unit.payload, token=unit):
            if self._blend is not None:
                out = self._blend(token.payload, out)
            yield token.with_payload(out)

    def reset(self, boundary: Boundary,
              context: PipelineContext) -> None:
        if self._driver is not None:
            self._driver.reset()

    def flush(self, context: PipelineContext) -> Iterable[FrameUnit]:
        if self._driver is None:
            return
        for out, token in self._driver.flush():
            if self._blend is not None:
                out = self._blend(token.payload, out)
            yield token.with_payload(out)

    def close(self, context: PipelineContext) -> None:
        driver, self._driver = self._driver, None
        if driver is not None:
            close = getattr(driver, "close", None)
            if callable(close):
                close()


__all__ = [
    "LUMA_CHROMA_KEYS",
    "FeedFlushDriver",
    "FeedFlushProcessor",
    "PerFrameDriver",
    "parse_luma_chroma",
]
