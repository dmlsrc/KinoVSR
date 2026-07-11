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

from collections.abc import Iterable
from typing import Any, Protocol

from .boundaries import Boundary
from .protocol import PipelineContext
from .specs import StreamSpec
from .units import FrameUnit


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
    """

    def __init__(self, make_driver: Any) -> None:
        self._make_driver = make_driver
        self._driver: FeedFlushDriver | None = None

    def prepare(self, input_spec: StreamSpec,
                context: PipelineContext) -> None:
        if self._driver is None:
            self._driver = self._make_driver()

    def process(self, unit: FrameUnit,
                context: PipelineContext) -> Iterable[FrameUnit]:
        for out, token in self._driver.feed(unit.payload, token=unit):
            yield token.with_payload(out)

    def reset(self, boundary: Boundary,
              context: PipelineContext) -> None:
        if self._driver is not None:
            self._driver.reset()

    def flush(self, context: PipelineContext) -> Iterable[FrameUnit]:
        if self._driver is None:
            return
        for out, token in self._driver.flush():
            yield token.with_payload(out)

    def close(self, context: PipelineContext) -> None:
        driver, self._driver = self._driver, None
        if driver is not None:
            close = getattr(driver, "close", None)
            if callable(close):
                close()


__all__ = ["FeedFlushDriver", "FeedFlushProcessor", "PerFrameDriver"]
