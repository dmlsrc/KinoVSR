"""The host session: an opened, validated pipeline over caller units.

:func:`open_pipeline` resolves a pipeline config against a concrete
input :class:`~kinovsr.processors.specs.StreamSpec` and preflight-
validates every edge before any processing - typed errors surface at
open time, not mid-stream. The returned :class:`PipelineSession` is
bound to that spec: :meth:`~PipelineSession.process` pulls the caller's
frame units through the chain with natural backpressure, weights load
lazily at the first pull, and closing the iterator (or the session, or
leaving the ``with`` block) cancels the run and releases every stage
exactly once - the :class:`~kinovsr.pipeline.scheduler.ChainRun`
semantics, which also fix the exception-precedence rules.

A session runs once: stage instances are stateful, so a consumed
session refuses a second ``process`` instead of silently reusing state.
Opening a pipeline is cheap (pure resolution; no weights, no Metal
sessions) - open another for the next stream.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from kinovsr.processors import (
    FrameUnit,
    PipelineContext,
    PipelineError,
    StreamSpec,
)
from kinovsr.reporting import Reporter
from kinovsr.settings import Settings

from .builder import BuildPlan, build_processors, resolve_pipeline
from .scheduler import ChainRun, run_chain


class PipelineSession:
    """One validated pipeline, bound to an input spec, run at most once."""

    def __init__(self, plan: BuildPlan, context: PipelineContext) -> None:
        self._plan = plan
        self._context = context
        self._run: ChainRun | None = None
        self._consumed = False

    @property
    def plan(self) -> BuildPlan:
        return self._plan

    @property
    def input_spec(self) -> StreamSpec:
        return self._plan.input_spec

    @property
    def output_spec(self) -> StreamSpec:
        """The validated spec of the units :meth:`process` yields."""
        return self._plan.output_spec

    def process(self, units: Iterable[FrameUnit]) -> ChainRun:
        """Pull ``units`` through the chain; yields output FrameUnits.

        Returns an owning iterator: close it (or this session) to cancel
        at any point, including before the first pull. Stage instances
        are built now; weights and native sessions materialize at the
        first pull (each stage's ``prepare``).
        """
        if self._consumed:
            raise PipelineError(
                "this session was already consumed; open_pipeline again "
                "for the next stream (stage state is never reused)")
        self._consumed = True
        built = build_processors(self._plan, self._context)
        self._run = run_chain(built, units, self._context)
        return self._run

    def close(self) -> None:
        """Cancel the active run (if any); safe to call repeatedly."""
        run, self._run = self._run, None
        self._consumed = True
        if run is not None:
            run.close()

    def __enter__(self) -> PipelineSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_pipeline(
    config: Mapping[str, Any],
    input_spec: StreamSpec,
    *,
    settings: Settings | None = None,
    reporter: Reporter | None = None,
) -> PipelineSession:
    """Resolve and validate ``config`` against ``input_spec``; return a
    session ready to process units.

    Every stage edge is validated here (typed errors: unknown families,
    stage-config problems, stream-contract violations), so a session
    that opens will not fail preflight mid-stream. ``settings`` defaults
    to the environment-resolved product settings; ``reporter`` receives
    phase progress (default: none).
    """
    if settings is None:
        settings = Settings.from_env()
    plan = resolve_pipeline(config, input_spec=input_spec, settings=settings)
    context = (PipelineContext(settings=settings) if reporter is None
               else PipelineContext(settings=settings, reporter=reporter))
    return PipelineSession(plan, context)


__all__ = ["PipelineSession", "open_pipeline"]
