"""Adapter from the family driver shape to the Processor protocol.

The learned families already speak a common streaming dialect:
``feed(frame, token) -> Iterable[(out, token)]``, ``flush() -> Iterable[...]``,
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

    def feed(self, frame: Any, token: Any = None) -> Iterable: ...

    def flush(self) -> Iterable: ...

    def reset(self) -> None: ...


class AsyncWindowHandle(Protocol):
    """One recurrent window in flight on an accelerator, driven cooperatively.

    A family net whose backend can run a whole reset-window off the GPU
    (today: Core ML on the Neural Engine) returns one of these from
    ``begin_window(frames)``. The contract every implementation must keep:

    - ``advance(block=False)`` makes whatever progress it can without
      waiting and returns True once the window is complete;
      ``advance(block=True)`` runs it to completion. ALL of the family's
      MLX work (input prep, output materialization) happens inside
      ``advance`` on the caller's thread - only the accelerator dispatches
      may run on a worker (``anemil.runtime.DispatchPipeline`` is the
      shared primitive for that).
    - ``advance_until_output(block=True)``, when present, stops after the
      next output is materialized (or the window completes). It lets a final
      window drain through downstream stages without first completing the
      entire accelerator batch.
    - ``outputs`` holds one entry per input frame, in input order, once
      complete.
    - a raised ``advance`` poisons the handle; the stream needs a reset.
    """

    def advance(self, block: bool = False) -> bool: ...

    @property
    def outputs(self) -> list: ...


class WindowWavefront:
    """Depth-one cross-window pipelining for accelerator-window drivers.

    While one window's accelerator dispatches are in flight, the driver
    keeps buffering input - so upstream keeps decoding - and the
    previously completed window's emissions have already flowed
    downstream. The pull scheduler's contract already allows this: a
    feed/flush driver may emit any input's output arbitrarily late, as
    long as tokens pair correctly.

    ``submit`` completes the window in flight first (that wait is the
    depth-one backpressure: at most one window's dispatches plus one
    window's buffered frames are ever held), starts the next, and returns
    the finished window's remaining emissions. ``poll`` opportunistically
    advances without blocking; ``available`` emits outputs materialized so
    far. ``drain`` advances only to the next output before yielding it
    downstream. ``barrier`` eagerly completes everything before a new
    window. ``abandon`` quiesces on reset/close paths.
    """

    def __init__(self) -> None:
        self._handle: AsyncWindowHandle | None = None
        self._collect: Any = None
        self._done = False

    @property
    def in_flight(self) -> bool:
        return self._handle is not None

    def poll(self) -> None:
        if self._handle is not None:
            self._done = self._handle.advance(block=False)

    def submit(self, begin: Any, collect: Any) -> Iterable:
        """Complete the in-flight window, then start the next.

        ``begin`` is a zero-argument callable returning the new window's
        :class:`AsyncWindowHandle` (it should also submit the first
        dispatch, so the accelerator is busy before this returns);
        ``collect(handle, complete)`` detaches newly available outputs and,
        at completion, validates them and resets shared network state.
        """
        out = self.barrier()
        self._handle = begin()
        self._collect = collect
        self._done = self._handle.advance(block=False)
        return out

    def available(self) -> Iterable:
        """Collect outputs materialized by nonblocking progress so far."""
        if self._handle is None:
            return []
        if self._done:
            return self._finish()
        return self._collect(self._handle, False)

    def drain(self) -> Iterable:
        """Yield each available output before advancing to the next one."""
        while self._handle is not None:
            available = iter(self.available())
            try:
                first = next(available)
            except StopIteration:
                pass
            else:
                yield first
                yield from available
                continue
            if self._handle is None:
                return
            advance = getattr(self._handle, "advance_until_output", None)
            if callable(advance):
                self._done = advance(block=True)
            else:
                self._done = self._handle.advance(block=True)

    def barrier(self) -> Iterable:
        """Complete and finalize the in-flight window, if any."""
        if self._handle is None:
            return []
        try:
            self._done = self._handle.advance(block=True)
        except BaseException:
            self._handle = None
            self._collect = None
            self._done = False
            raise
        return self._finish()

    def _finish(self) -> Iterable:
        handle, self._handle = self._handle, None
        collect, self._collect = self._collect, None
        self._done = False
        return collect(handle, True)

    def abandon(self) -> None:
        """Quiesce the in-flight window without emitting (reset/close)."""
        import contextlib

        handle, self._handle = self._handle, None
        self._collect = None
        self._done = False
        if handle is not None:
            with contextlib.suppress(BaseException):
                handle.advance(block=True)


class PerFrameDriver:
    """feed()/flush() shape over a per-frame engine (``denoise(x) -> x``).

    Several single-image families expose ``denoise``/``reset``/``close``
    without the streaming dialect; this adapter gives them the driver
    shape :class:`FeedFlushProcessor` pumps, with reset/close passing
    through when the engine has them. An engine whose ``denoise`` accepts
    a ``source`` keyword additionally receives the unit's raw-stream
    identity (``FrameUnit.source``: sync flag, GOP position, coded size),
    so per-frame families can key conditioning decisions to the source's
    coding structure.
    """

    def __init__(self, engine: Any) -> None:
        import inspect

        self._engine = engine
        # Construction stays lazy for engines without denoise (test stubs,
        # engines used only for lifecycle): feed() fails at call time for
        # those, exactly as before this inspection existed.
        denoise = getattr(engine, "denoise", None)
        parameters: Any = {}
        if denoise is not None:
            try:
                parameters = inspect.signature(denoise).parameters
            except (TypeError, ValueError):
                parameters = {}
        self._wants_source = "source" in parameters

    def feed(self, frame: Any, token: Any = None) -> list:
        if self._wants_source:
            source = getattr(token, "source", None)
            return [(self._engine.denoise(frame, source=source), token)]
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
        # The driver's optional pump() hook, bound at prepare. Emissions
        # flow downstream one at a time through the pull scheduler, and a
        # GOP-windowed driver may have a whole accelerator window in
        # flight behind them; pumping between consumptions keeps those
        # dispatches advancing while later stages (VT upscale, encode)
        # run on this thread. Without it a burst of window outputs
        # serializes the chain: the accelerator idles while downstream
        # works through the burst, then downstream idles while the next
        # window runs (measured additive at 640x480, 2026-07-23).
        self._pump: Any = None
        # The family's final report, stashed at close (see run_diagnostics).
        self._final_diagnostics: list[str] = []
        self._final_debug_images: dict[str, Any] = {}

    def prepare(self, input_spec: StreamSpec,
                context: PipelineContext) -> None:
        if self._driver is None:
            self._driver = self._make_driver()
        configure_gop = getattr(self._driver, "set_gop_policy", None)
        if callable(configure_gop):
            configure_gop(context.gop)
        # Accelerator-backed drivers can start loading their engine now,
        # before the first frame: the input geometry is already known, so
        # multi-second Core ML function loads overlap the source's startup
        # and first-window decode instead of serializing at the first
        # dispatch.
        preheat = getattr(self._driver, "preheat", None)
        if callable(preheat):
            geometry = input_spec.frame.geometry
            preheat(int(geometry.height), int(geometry.width))
        pump = getattr(self._driver, "pump", None)
        self._pump = pump if callable(pump) else None
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
            if self._pump is not None:
                # Downstream just consumed one emission; let the driver's
                # in-flight window progress before handing over the next.
                self._pump()

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
            if self._pump is not None:
                self._pump()

    def run_diagnostics(self) -> list[str]:
        """End-of-run diagnostic lines from the wrapped driver, when the
        family implements ``run_diagnostics()``. The run's iterator closes
        its stages on exhaustion, so ``close`` stashes the final report and
        this keeps answering after the driver is released."""
        if self._driver is None:
            return list(self._final_diagnostics)
        hook = getattr(self._driver, "run_diagnostics", None)
        return list(hook()) if callable(hook) else []

    def debug_images(self) -> Mapping[str, Any]:
        """End-of-run debug maps (suffix -> [0,1] (H,W) array) from the
        wrapped driver, for the run-level ``noise_map_debug`` PNG dump;
        stashed at close like :meth:`run_diagnostics`."""
        if self._driver is None:
            return dict(self._final_debug_images)
        hook = getattr(self._driver, "debug_images", None)
        return dict(hook()) if callable(hook) else {}

    def close(self, context: PipelineContext) -> None:
        driver, self._driver = self._driver, None
        if driver is not None:
            # Capture the family's final report before releasing it: the
            # session's diagnostics collection runs after the chain closed.
            hook = getattr(driver, "run_diagnostics", None)
            if callable(hook):
                self._final_diagnostics = list(hook())
            hook = getattr(driver, "debug_images", None)
            if callable(hook):
                self._final_debug_images = dict(hook())
            close = getattr(driver, "close", None)
            if callable(close):
                close()


__all__ = [
    "LUMA_CHROMA_KEYS",
    "AsyncWindowHandle",
    "FeedFlushDriver",
    "FeedFlushProcessor",
    "PerFrameDriver",
    "WindowWavefront",
    "parse_luma_chroma",
]
