"""Drive a built chain over frame units, deterministically.

The scheduler owns everything the processor contract promises
(:mod:`kinovsr.processors.protocol`): prepare-before-first-unit,
boundary-triggered resets, tail flushing, and exactly-once close on
success, cancellation, failure, or abandonment. It is pull-based:
:func:`run_chain` returns an owning :class:`ChainRun` iterator, closing
it cancels the run (valid even before the first pull), and iteration
provides natural backpressure.

Boundary delivery is scheduler-owned so families cannot get it wrong. At
a mid-stream boundary the stage's buffered tail is drained FIRST (with
pre-boundary state, as pre-boundary content), then the stage resets, then
the boundary unit processes - and the boundary re-attaches to the first
unit the stage emits afterward, so it keeps traveling even across
buffering stages. Families never see upstream boundaries in-band: units
are stripped before ``process`` (boundary information reaches a family
only through ``reset``), which makes double delivery impossible when a
family passes units through or fans one unit into many. A boundary a
family ADDS to its output (cut detection) is fresh information and rides
along untouched. The first unit of a run always carries STREAM_START
(synthesized here when the input endpoint did not).

Hot-path discipline (planning 06): per unit per stage the scheduler does
one boundary check, one Python call into the stage, and one loop over its
outputs. It never inspects payloads, never times per frame, and never
forces a host/device synchronization.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator

from kinovsr.processors import (
    Boundary,
    BoundaryKind,
    FrameUnit,
    PipelineContext,
    PipelineError,
    PipelineRuntimeError,
    Processor,
)

from .builder import BuildPlan, ResolvedStage, build_processors


def _wrap_stage_error(stage: ResolvedStage, exc: Exception) -> Exception:
    if isinstance(exc, PipelineError):
        return exc
    return PipelineRuntimeError(
        stage.name, f"{type(exc).__name__}: {exc}")


def _stage_stream(
    stage: ResolvedStage,
    processor: Processor,
    upstream: Iterator[FrameUnit],
    context: PipelineContext,
) -> Iterator[FrameUnit]:
    """One stage's view of the stream: lifecycle-correct unit flow.

    ``close`` deliberately does NOT happen here - the driver owns it, so
    it runs exactly once even when this generator never starts or dies
    mid-flight.
    """
    pending: list[Boundary] = []

    def attach(unit: FrameUnit) -> FrameUnit:
        if not pending:
            return unit
        stamped = FrameUnit(
            payload=unit.payload, pts=unit.pts, duration=unit.duration,
            boundaries=(*pending, *unit.boundaries))
        pending.clear()
        return stamped

    try:
        processor.prepare(stage.input_spec, context)
    except Exception as exc:
        raise _wrap_stage_error(stage, exc) from exc

    started = False
    for unit in upstream:
        if unit.boundaries:
            try:
                if started:
                    # Drain the pre-boundary tail with pre-boundary state.
                    for tail in processor.flush(context):
                        yield attach(tail)
                for boundary in unit.boundaries:
                    processor.reset(boundary, context)
            except Exception as exc:
                raise _wrap_stage_error(stage, exc) from exc
            pending.extend(unit.boundaries)
            # The family learns about boundaries through reset() only.
            unit = FrameUnit(payload=unit.payload, pts=unit.pts,
                             duration=unit.duration)
        started = True
        try:
            outputs = processor.process(unit, context)
            for produced in outputs:
                yield attach(produced)
        except Exception as exc:
            raise _wrap_stage_error(stage, exc) from exc

    try:
        for tail in processor.flush(context):
            yield attach(tail)
    except Exception as exc:
        raise _wrap_stage_error(stage, exc) from exc


def _ensure_stream_start(units: Iterable[FrameUnit]) -> Iterator[FrameUnit]:
    iterator = iter(units)
    first = next(iterator, None)
    if first is None:
        return
    if not any(b.kind is BoundaryKind.STREAM_START for b in first.boundaries):
        first = first.with_boundary(Boundary(BoundaryKind.STREAM_START))
    yield first
    yield from iterator


class ChainRun:
    """An owning iterator over a running chain.

    Cleanup is armed from CONSTRUCTION, not from the first pull: closing
    (or abandoning) a run before any unit was pulled still closes every
    stage exactly once - a plain generator would never have entered its
    own try/finally. Iteration exhaustion, a stage error, an explicit
    ``close``, context-manager exit, and garbage collection all funnel
    into the same exactly-once close path; a close failure surfaces on
    the success path and never masks an active error.
    """

    def __init__(
        self,
        built: tuple[tuple[ResolvedStage, Processor], ...],
        units: Iterable[FrameUnit],
        context: PipelineContext,
    ) -> None:
        self._built = built
        self._units: Iterable[FrameUnit] | None = units
        self._context = context
        self._stream: Iterator[FrameUnit] | None = None
        self._closed = False

    # -- iteration ---------------------------------------------------------

    def __iter__(self) -> ChainRun:
        return self

    def __next__(self) -> FrameUnit:
        if self._closed:
            raise StopIteration
        if self._stream is None:
            stream: Iterator[FrameUnit] = _ensure_stream_start(self._units)
            self._units = None
            for stage, processor in self._built:
                stream = _stage_stream(
                    stage, processor, stream,
                    self._context.for_stage(stage.name))
            self._stream = stream
        try:
            return next(self._stream)
        except StopIteration:
            self._close_all(active_error=False)   # may raise a close error
            raise
        except BaseException:
            self._close_all(active_error=True)
            raise

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Cancel the run; safe at any point, including before the first
        pull and repeatedly."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()
        self._close_all(active_error=False)

    def __enter__(self) -> ChainRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()
        self._close_all(active_error=exc_type is not None)

    def __del__(self) -> None:
        # A finalizer must never raise; close errors were the caller's to
        # collect via an explicit close().
        with contextlib.suppress(Exception):
            self.close()

    def _close_all(self, active_error: bool) -> None:
        if self._closed:
            return
        self._closed = True
        first_close_error: Exception | None = None
        for stage, processor in self._built:
            try:
                processor.close(self._context.for_stage(stage.name))
            except Exception as exc:  # noqa: BLE001 - collected, not lost
                if first_close_error is None:
                    first_close_error = _wrap_stage_error(stage, exc)
        if first_close_error is not None and not active_error:
            raise first_close_error


def run_chain(
    built: tuple[tuple[ResolvedStage, Processor], ...],
    units: Iterable[FrameUnit],
    context: PipelineContext,
) -> ChainRun:
    """Pull output units through the whole chain.

    Returns a :class:`ChainRun`: iterate for units, ``close()`` to cancel
    (valid even before the first pull), or use it as a context manager.
    Every stage instance closes exactly once - on completion, error,
    cancellation, or abandonment - in chain order, with per-stage
    contexts.
    """
    return ChainRun(built, units, context)


def run_plan(
    plan: BuildPlan,
    units: Iterable[FrameUnit],
    context: PipelineContext,
) -> ChainRun:
    """Build the plan's stages and run the chain (the common entry)."""
    return run_chain(build_processors(plan, context), units, context)


__all__ = ["ChainRun", "run_chain", "run_plan"]
