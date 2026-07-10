"""Drive a built chain over frame units, deterministically.

The scheduler owns everything the processor contract promises
(:mod:`kinovsr.processors.protocol`): prepare-before-first-unit,
boundary-triggered resets, tail flushing, and exactly-once close on
success, cancellation, or failure. It is pull-based: :func:`run_chain`
returns a generator, closing it cancels the run, and iteration provides
natural backpressure.

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


def run_chain(
    built: tuple[tuple[ResolvedStage, Processor], ...],
    units: Iterable[FrameUnit],
    context: PipelineContext,
) -> Iterator[FrameUnit]:
    """Pull output units through the whole chain.

    Closing the returned generator cancels the run; on cancellation,
    exception, or completion every stage instance is closed exactly once,
    in chain order, with per-stage contexts. A close failure never masks
    the original error.
    """
    closed = False

    def close_all(active_error: bool) -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        first_close_error: Exception | None = None
        for stage, processor in built:
            try:
                processor.close(context.for_stage(stage.name))
            except Exception as exc:  # noqa: BLE001 - collected, not lost
                if first_close_error is None:
                    first_close_error = _wrap_stage_error(stage, exc)
        if first_close_error is not None and not active_error:
            raise first_close_error

    try:
        stream: Iterator[FrameUnit] = _ensure_stream_start(units)
        for stage, processor in built:
            stream = _stage_stream(
                stage, processor, stream, context.for_stage(stage.name))
        yield from stream
    except BaseException:
        close_all(active_error=True)
        raise
    finally:
        close_all(active_error=False)


def run_plan(
    plan: BuildPlan,
    units: Iterable[FrameUnit],
    context: PipelineContext,
) -> Iterator[FrameUnit]:
    """Build the plan's stages and run the chain (the common entry)."""
    return run_chain(build_processors(plan, context), units, context)


__all__ = ["run_chain", "run_plan"]
