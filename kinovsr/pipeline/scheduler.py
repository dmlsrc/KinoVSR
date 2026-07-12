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
    Processor,
)

from .builder import (
    BuildPlan,
    ResolvedStage,
    _append_context,
    _wrap_stage_error,
    build_processors,
)


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
    into the same exactly-once close path.

    Exception precedence is fixed and documented: an ACTIVE error (a
    stage failure, a context-manager body error) always outranks cleanup
    failures; with no active error, the FIRST cleanup failure wins
    (generator close before processor close). A ``KeyboardInterrupt`` or
    ``SystemExit`` raised during cleanup outranks everything - after the
    remaining stages have still been closed. Outranked errors are
    preserved on the winner's ``__context__`` chain, never silently
    dropped.
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
        to_raise: BaseException | None
        try:
            return next(self._stream)
        except StopIteration:
            to_raise = self._deliver_cleanup(None)
            if to_raise is None:
                raise
        except BaseException as active:
            to_raise = self._deliver_cleanup(active)
        # Raise OUTSIDE the except block: raising a different exception while
        # one is being handled makes Python overwrite the raised exception's
        # __context__ with the handled one, which would clobber the cleanup
        # chain built above. Out here nothing is being handled.
        raise to_raise

    def _deliver_cleanup(
        self, active: BaseException | None,
    ) -> BaseException | None:
        """Close every stage and return the exception the caller should
        raise (or None to let the current one propagate). Builds the full
        precedence-correct context chain so no cleanup failure is dropped.

        ``active`` is the stage/body error in flight, or None on the
        stream-exhausted success path.
        """
        close_errors, interrupts = self._close_all()
        if active is None:
            ordered = [*close_errors, *interrupts]
            if not ordered:
                return None
            winner = interrupts[0] if interrupts else ordered[0]
            _append_context(winner, [c for c in ordered if c is not winner])
            return winner
        if interrupts:
            # an interrupt outranks the active error; the active error and
            # every close failure and later interrupt stay on the chain
            winner = interrupts[0]
            _append_context(winner, [active, *close_errors, *interrupts[1:]])
            return winner
        # the active error wins; ordinary close failures ride its context
        # chain instead of being dropped
        _append_context(active, close_errors)
        return active

    # -- lifecycle ---------------------------------------------------------

    @staticmethod
    def _is_interrupt(exc: BaseException) -> bool:
        return not isinstance(exc, Exception)   # KeyboardInterrupt, SystemExit

    def _shutdown(self) -> BaseException | None:
        """Close the stream, then every processor - unconditionally - and
        return the one cleanup failure to deliver (or None). Precedence:
        an interrupt beats ordinary failures; otherwise the FIRST failure
        (stream close happens before processor close) wins, with every
        other failure preserved on the winner's context chain."""
        stream, self._stream = self._stream, None
        stream_error: BaseException | None = None
        if stream is not None:
            try:
                stream.close()
            except BaseException as exc:  # noqa: BLE001 - re-delivered below
                stream_error = exc
        close_errors, interrupts = self._close_all()
        # Chronological order: the stream closed before the processors.
        # The first interrupt wins; failing that, the first failure wins.
        # Every other error is preserved on the winner's context chain.
        ordered = [c for c in (stream_error, *close_errors, *interrupts)
                   if c is not None]
        if not ordered:
            return None
        winner = next((c for c in ordered if self._is_interrupt(c)),
                      ordered[0])
        _append_context(winner, [c for c in ordered if c is not winner])
        return winner

    def close(self) -> None:
        """Cancel the run; safe at any point, including before the first
        pull and repeatedly."""
        failure = self._shutdown()
        if failure is not None:
            raise failure

    def __enter__(self) -> ChainRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        failure = self._shutdown()
        if failure is None:
            return
        if exc is not None and not self._is_interrupt(failure):
            # The active body error outranks ordinary cleanup failures; keep
            # the whole cleanup chain on its context (appended, not clobbered).
            _append_context(exc, [failure])
            return
        raise failure

    def __del__(self) -> None:
        # A finalizer must never raise - not even for interrupts; close
        # errors were the caller's to collect via an explicit close().
        with contextlib.suppress(BaseException):
            self.close()

    def _close_all(self) -> tuple[list[Exception], list[BaseException]]:
        """Close every stage exactly once; report failures instead of
        raising so callers own delivery precedence. Returns EVERY ordinary
        close error and EVERY cleanup interrupt (both wrapped/collected in
        chain order) - no stage's failure is dropped when a later stage also
        fails, and the first interrupt still wins precedence downstream."""
        if self._closed:
            return [], []
        self._closed = True
        close_errors: list[Exception] = []
        interrupts: list[BaseException] = []
        for stage, processor in self._built:
            try:
                processor.close(self._context.for_stage(stage.name))
            except Exception as exc:  # noqa: BLE001 - collected, not lost
                close_errors.append(_wrap_stage_error(stage, exc))
            except BaseException as exc:
                # KeyboardInterrupt/SystemExit mid-cleanup: keep closing the
                # remaining stages, then deliver every one (first wins).
                interrupts.append(exc)
        return close_errors, interrupts


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
