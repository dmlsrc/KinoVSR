"""Public scheduling facade over the bounded streaming runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from kinovsr.processors import FrameUnit, PipelineContext

from .builder import BuildPlan, ResolvedStage, _append_context, build_processors
from .ownership import retain_safe_outputs
from .streaming import StreamingChainRun

# Preserve the established public class name while making the streaming
# implementation the only production executor.
ChainRun = StreamingChainRun


def run_chain(
    built: tuple[tuple[ResolvedStage, object], ...],
    units: Iterable[FrameUnit],
    context: PipelineContext,
    *,
    finalizers: tuple[Callable[[], None], ...] = (),
    input_spec=None,
    source_bridge=None,
    terminal_consumer=None,
    trace: bool = False,
    trace_limit: int = 100_000,
) -> ChainRun:
    """Run a built chain through bounded source/stage/terminal channels."""
    return ChainRun(
        built,
        units,
        context,
        finalizers,
        input_spec=input_spec,
        source_bridge=source_bridge,
        terminal_consumer=terminal_consumer,
        trace=trace,
        trace_limit=trace_limit,
    )


def _close_after_failure(
    active: BaseException,
    close: Callable[[], None],
) -> BaseException:
    """Close an owner and return the precedence-correct exception winner."""
    try:
        close()
    except BaseException as cleanup:  # noqa: BLE001 - re-delivered by caller
        if not isinstance(cleanup, Exception):
            _append_context(cleanup, [active])
            return cleanup
        _append_context(active, [cleanup])
    return active


def run_plan(
    plan: BuildPlan,
    units: Iterable[FrameUnit],
    context: PipelineContext,
    *,
    retain_outputs: bool = True,
) -> Iterator[FrameUnit]:
    """Build and stream a plan with retain-safe host outputs by default."""
    from kinovsr.media.pixel_buffers import ci_cache_owner

    lease = ci_cache_owner()
    run: ChainRun | None = None
    active: BaseException | None = None
    try:
        built = build_processors(plan, context)
        run = run_chain(
            built,
            units,
            context,
            finalizers=(lease.close,),
            input_spec=plan.input_spec,
        )
        return retain_safe_outputs(
            run, plan.output_spec, retain_outputs=retain_outputs
        )
    except BaseException as exc:  # noqa: BLE001 - cleanup precedence below
        active = exc
    winner = _close_after_failure(
        active,
        run.close if run is not None else lease.close,
    )
    raise winner


__all__ = ["ChainRun", "run_chain", "run_plan"]
