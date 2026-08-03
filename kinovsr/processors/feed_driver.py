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
from concurrent.futures import Future
from dataclasses import dataclass
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
    - ``wait_until_ready()``, when present, blocks only for an in-flight
      native dispatch. It must not perform MLX work. The streaming executor
      uses that edge outside the serial MLX owner, then calls nonblocking
      ``advance`` on the owner to prepare and submit the next dispatch.
    - ``outputs`` holds one entry per input frame, in input order, once
      complete.
    - a raised ``advance`` poisons the handle; the stream needs a reset.
    """

    def advance(self, block: bool = False) -> bool: ...

    def wait_until_ready(self) -> None: ...

    @property
    def outputs(self) -> list: ...


class WindowWavefront:
    """Depth-one, owner-driven progress for accelerator-window drivers.

    While one window's accelerator dispatches are in flight, the driver
    keeps buffering input - so upstream keeps decoding - and the
    previously completed window's emissions have already flowed
    downstream. The pull scheduler's contract already allows this: a
    feed/flush driver may emit any input's output arbitrarily late, as
    long as tokens pair correctly.

    ``bind_background_submit`` lets the streaming executor queue a blocking
    advance on the driver's own affinity lane.  Once submitted, native
    dispatch therefore keeps advancing even while a downstream channel or
    writer consumes earlier output. ``submit`` completes the window in flight first (that wait is the
    depth-one backpressure: at most one window's dispatches plus one
    window's buffered frames are ever held), starts the next, and returns
    the finished window's remaining emissions. ``poll`` opportunistically
    advances without blocking; ``available`` emits outputs materialized so
    far. ``drain`` advances only to the next output before yielding it
    downstream. ``barrier`` eagerly completes everything before a new
    window. ``abandon`` quiesces on reset/close paths.
    """

    _POLL_INTERVAL_SECONDS = 0.002

    def __init__(self) -> None:
        self._handle: AsyncWindowHandle | None = None
        self._collect: Any = None
        self._done = False
        self._background_submit: Any = None
        self._completion_submit: Any = None
        self._owner_wait: Any = None
        self._progress: Future[Any] | None = None
        self._progress_phase: str | None = None
        self._progress_failure: BaseException | None = None

    def bind_background_submit(
        self,
        submit: Any,
        completion_submit: Any = None,
        owner_wait: Any = None,
    ) -> None:
        if not callable(submit):
            raise TypeError("window progress submitter must be callable")
        if completion_submit is not None and not callable(completion_submit):
            raise TypeError("window completion submitter must be callable")
        if owner_wait is not None and not callable(owner_wait):
            raise TypeError("window owner waiter must be callable")
        self._background_submit = submit
        self._completion_submit = completion_submit
        self._owner_wait = owner_wait

    @property
    def in_flight(self) -> bool:
        return self._handle is not None

    def poll(self) -> None:
        if self._handle is not None:
            if self._progress_failure is not None:
                raise self._progress_failure
            if self._progress is not None:
                return
            self._done = self._handle.advance(block=False)

    def _queue_lane_progress(
        self,
        handle: AsyncWindowHandle,
        *,
        delay: float = 0.0,
    ) -> None:
        def progress() -> bool:
            # One nonblocking dispatch transition per lane turn. Native work
            # remains in flight between polls, while immediate MLX bridges
            # queued by downstream run before the next delayed poll. This is
            # owner-driven (no downstream callback), but it avoids holding the
            # MLX lane across an accelerator wait or monopolizing the GPU in
            # the shadow of VideoToolbox.
            return bool(handle.advance(block=False))

        future = self._background_submit(progress, delay=delay)
        self._progress = future
        self._progress_phase = "lane"

        def completed(done: Future[Any]) -> None:
            if self._handle is not handle:
                return
            try:
                complete = bool(done.result())
            except BaseException as exc:  # re-delivered on the owner lane
                self._progress_failure = exc
                return
            self._done = complete
            if not complete and self._handle is handle:
                wait = getattr(handle, "wait_until_ready", None)
                if callable(wait) and self._completion_submit is not None:
                    self._queue_native_wait(handle, wait)
                else:
                    # Generic handles without an explicit native wait edge use
                    # a bounded fallback poll. Already-waiting lane calls run
                    # before the delayed poll.
                    self._queue_lane_progress(
                        handle,
                        delay=self._POLL_INTERVAL_SECONDS,
                    )

        future.add_done_callback(completed)

    def _queue_native_wait(
        self,
        handle: AsyncWindowHandle,
        wait: Any,
    ) -> None:
        future = self._completion_submit(wait)
        self._progress = future
        self._progress_phase = "wait"

        def completed(done: Future[Any]) -> None:
            if self._handle is not handle:
                return
            try:
                done.result()
            except BaseException as exc:
                self._progress_failure = exc
                return
            self._queue_lane_progress(handle)

        future.add_done_callback(completed)

    def submit(self, begin: Any, collect: Any) -> Iterable:
        """Complete the in-flight window, then start the next.

        ``begin`` is a zero-argument callable returning the new window's
        :class:`AsyncWindowHandle` (it should also submit the first
        dispatch, so the accelerator is busy before this returns);
        ``collect(handle, complete)`` detaches newly available outputs and,
        at completion, validates them and resets shared network state.
        """
        if self._owner_wait is not None and self._handle is not None:
            return self._handoff(begin, collect)
        out = self.barrier()
        self._begin(begin, collect)
        return out

    def _begin(self, begin: Any, collect: Any) -> None:
        """Start one handle and bind its owner-driven progress chain."""
        self._handle = begin()
        self._collect = collect
        self._done = False
        self._progress_failure = None
        if self._background_submit is None:
            self._progress = None
            self._done = self._handle.advance(block=False)
        else:
            handle = self._handle
            self._queue_lane_progress(handle)

    def _handoff(self, begin: Any, collect: Any) -> Iterable:
        """Progressively emit one window before starting its successor.

        This path runs only inside the streaming owner's cooperative turn.
        It preserves the logical one-window barrier while allowing outputs
        already admitted by the family algorithm to reach downstream before
        the final native dispatch completes.
        """
        handle = self._handle
        assert handle is not None
        while self._handle is handle and not self._done:
            if self._progress_failure is not None:
                raise self._progress_failure
            emitted = False
            for output in self._collect(handle, False):
                emitted = True
                yield output
            if emitted:
                continue
            output_count = len(getattr(handle, "outputs", ()))
            self._owner_wait(
                lambda handle=handle, output_count=output_count: (
                    self._handle is not handle
                    or self._done
                    or self._progress_failure is not None
                    or len(getattr(handle, "outputs", ())) > output_count
                )
            )
        if self._progress_failure is not None:
            raise self._progress_failure
        out = self._finish()
        self._begin(begin, collect)
        yield from out

    def available(self) -> Iterable:
        """Collect outputs materialized by nonblocking progress so far."""
        if self._handle is None:
            return []
        if self._progress_failure is not None:
            raise self._progress_failure
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
            if self._owner_wait is not None:
                handle = self._handle
                output_count = len(getattr(handle, "outputs", ()))
                self._owner_wait(
                    lambda handle=handle, output_count=output_count: (
                        self._handle is not handle
                        or self._done
                        or self._progress_failure is not None
                        or len(getattr(handle, "outputs", ())) > output_count
                    )
                )
                continue
            if self._progress is not None and not self._progress.done():
                if self._progress_phase == "wait":
                    self._progress.result()
                # This can occur when flush reaches drain in the same lane
                # callback that queued the background task. Advance inline to
                # the next output; the queued task later observes a completed
                # handle or continues from this exact serial state.
                advance = getattr(self._handle, "advance_until_output", None)
                if callable(advance):
                    self._done = advance(block=True)
                else:
                    self._done = self._handle.advance(block=True)
                continue
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
            if self._background_submit is None:
                self._done = self._handle.advance(block=True)
            elif self._progress_failure is not None:
                raise self._progress_failure
            if self._done:
                pass
            elif self._owner_wait is not None:
                handle = self._handle
                self._owner_wait(
                    lambda: (
                        self._handle is not handle
                        or self._done
                        or self._progress_failure is not None
                    )
                )
                if self._progress_failure is not None:
                    raise self._progress_failure
            else:
                if (
                    self._progress_phase == "wait"
                    and self._progress is not None
                    and not self._progress.done()
                ):
                    # Native completion waiting runs outside the MLX lane.
                    # Let that join finish before taking over the remaining
                    # serialized state inline; never join the same pipeline
                    # concurrently from two threads.
                    self._progress.result()
                # See drain(): a barrier may be reached inside the callback
                # that just queued progress, where waiting on that Future
                # would deadlock its own affinity lane.
                self._done = self._handle.advance(block=True)
        except BaseException:
            self._handle = None
            self._collect = None
            self._done = False
            self._progress = None
            self._progress_phase = None
            raise
        return self._finish()

    def _finish(self) -> Iterable:
        handle, self._handle = self._handle, None
        collect, self._collect = self._collect, None
        self._done = False
        self._progress = None
        self._progress_phase = None
        self._progress_failure = None
        return collect(handle, True)

    def abandon(self) -> None:
        """Quiesce the in-flight window without emitting (reset/close)."""
        import contextlib

        handle, self._handle = self._handle, None
        self._collect = None
        self._done = False
        self._progress = None
        self._progress_phase = None
        self._progress_failure = None
        if handle is not None:
            with contextlib.suppress(BaseException):
                handle.advance(block=True)


@dataclass(frozen=True, slots=True)
class _BridgeOutput:
    value: Any
    source_payload: Any


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
        self._driver_prepare_output: Any = None
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
        prepare_output = getattr(self._driver, "prepare_output", None)
        self._driver_prepare_output = (
            prepare_output if callable(prepare_output) else None
        )
        if self._blend is None and (self._luma_strength != 1.0
                                    or self._chroma_strength != 1.0):
            from kinovsr.media.yuv import luma_chroma_blend

            kr, kb = luma_coefficients(input_spec.frame.color_matrix)
            al, ac = self._luma_strength, self._chroma_strength

            def blend(orig: Any, new: Any) -> Any:
                return luma_chroma_blend(orig, new, al, ac, kr, kb).astype(
                    new.dtype)

            self._blend = blend

    def bind_background_submit(
        self,
        submit: Any,
        completion_submit: Any = None,
        owner_wait: Any = None,
    ) -> None:
        """Give an async-window driver its affinity-lane progress queue."""
        if self._driver is None:
            raise RuntimeError("cannot bind progress before prepare")
        bind = getattr(self._driver, "bind_background_submit", None)
        if callable(bind):
            if completion_submit is None and owner_wait is None:
                bind(submit)
            else:
                bind(submit, completion_submit, owner_wait)

    def prepare_input(
        self,
        unit: FrameUnit,
        context: PipelineContext,
    ) -> FrameUnit:
        if self._driver is None:
            raise RuntimeError("cannot bridge input before prepare")
        hook = getattr(self._driver, "prepare_input", None)
        if not callable(hook):
            return unit
        return unit.with_payload(hook(unit.payload))

    def prepare_output(
        self,
        unit: FrameUnit,
        context: PipelineContext,
    ) -> FrameUnit:
        hook = self._driver_prepare_output
        if hook is None:
            return unit
        bridged = unit.payload
        if not isinstance(bridged, _BridgeOutput):
            raise RuntimeError("driver output bridge lost its source payload")
        out = hook(bridged.value)
        if self._blend is not None:
            out = self._blend(bridged.source_payload, out)
        return unit.with_payload(out)

    def process(self, unit: FrameUnit,
                context: PipelineContext) -> Iterable[FrameUnit]:
        for out, token in self._driver.feed(unit.payload, token=unit):
            if self._driver_prepare_output is not None:
                out = _BridgeOutput(out, token.payload)
            elif self._blend is not None:
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
            if self._driver_prepare_output is not None:
                out = _BridgeOutput(out, token.payload)
            elif self._blend is not None:
                out = self._blend(token.payload, out)
            yield token.with_payload(out)

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
