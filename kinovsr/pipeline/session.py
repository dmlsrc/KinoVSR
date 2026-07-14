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

from collections.abc import Iterable, Iterator, Mapping
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
from .ownership import retain_safe_outputs
from .scheduler import ChainRun, _close_after_failure, run_chain


class PipelineSession:
    """One validated pipeline, bound to an input spec, run at most once."""

    def __init__(self, plan: BuildPlan, context: PipelineContext) -> None:
        self._plan = plan
        self._context = context
        self._run: ChainRun | None = None
        self._consumed = False
        self._built: tuple = ()
        self._terminal_pool_binding: tuple[Any, int, int, int] | None = None

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

    def _bind_terminal_output_pool(
        self,
        pool: Any,
        pixel_format: int,
        width: int,
        height: int,
    ) -> None:
        """Offer a file writer pool to a compatible terminal native stage."""
        if self._consumed:
            raise PipelineError("output pool must be bound before processing")
        self._terminal_pool_binding = (
            pool, int(pixel_format), int(width), int(height))

    def process(self, units: Iterable[FrameUnit], *,
                retain_outputs: bool = True) -> Iterator[FrameUnit]:
        """Pull ``units`` through the chain; yields output FrameUnits.

        Returns an owning iterator: close it (or this session) to cancel
        at any point, including before the first pull. Stage instances
        are built now; weights and native sessions materialize at the
        first pull (each stage's ``prepare``).

        Output ownership (matters only for CVPixelBuffer layouts; MLX
        payloads are immutable values either way):

        - ``retain_outputs=True`` (default): each output CVPixelBuffer is a
          fresh, host-owned deep copy - safe to keep indefinitely, even
          after you feed or recycle the next input.
        - ``retain_outputs=False``: outputs are yielded as produced, so a
          payload may alias a borrowed input or a stage's reused buffer and
          is valid only until the next pull. Copy or hand it off before
          advancing. The file sink uses this (it consumes each unit into
          the encoder synchronously, so there is nothing to retain).
        """
        if self._consumed:
            raise PipelineError(
                "this session was already consumed; open_pipeline again "
                "for the next stream (stage state is never reused)")
        self._consumed = True
        binding, self._terminal_pool_binding = (
            self._terminal_pool_binding,
            None,
        )
        from kinovsr.media.pixel_buffers import ci_cache_owner

        lease = ci_cache_owner()
        run: ChainRun | None = None
        active: BaseException | None = None
        try:
            built = build_processors(self._plan, self._context)
            self._built = built
            run = run_chain(
                built,
                units,
                self._context,
                finalizers=(lease.close,),
            )
            self._run = run
            if binding is not None and built:
                hook = getattr(built[-1][1], "_bind_output_pool", None)
                if callable(hook):
                    hook(*binding)
            return retain_safe_outputs(
                run,
                self._plan.output_spec,
                retain_outputs=retain_outputs,
            )
        except BaseException as exc:  # noqa: BLE001 - cleanup precedence below
            active = exc
        self._run = None
        winner = _close_after_failure(
            active,
            run.close if run is not None else lease.close,
        )
        raise winner

    def stage_diagnostics(self) -> list[str]:
        """End-of-run diagnostic lines from the stages that ran.

        Each stage may expose ``run_diagnostics() -> list[str]`` (the
        family owns its own reporting - noise-map stats, gate openness,
        auto-QF reports - exactly the lines the inherited harness printed
        at end of run); stages without the hook contribute nothing. Call
        after draining :meth:`process`: the run's iterator closes stages
        on exhaustion, so hook-bearing stages stash their final report at
        close and keep answering afterward.
        """
        lines: list[str] = []
        for _stage, processor in self._built:
            hook = getattr(processor, "run_diagnostics", None)
            if callable(hook):
                lines.extend(hook())
        return lines

    def stage_debug_images(self) -> dict[str, Any]:
        """End-of-run debug maps (suffix -> [0,1] (H,W) array) from stages
        exposing ``debug_images()``; same lifecycle window as
        :meth:`stage_diagnostics`. Later stages win a suffix collision
        (one map of each kind per run, like the harness's single dump)."""
        images: dict[str, Any] = {}
        for _stage, processor in self._built:
            hook = getattr(processor, "debug_images", None)
            if callable(hook):
                images.update(hook())
        return images

    def close(self) -> None:
        """Cancel the active run (if any); safe to call repeatedly."""
        run, self._run = self._run, None
        self._terminal_pool_binding = None
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
    windowing: Any = None,
    publication_origin_pts: int | None = None,
) -> PipelineSession:
    """Resolve and validate ``config`` against ``input_spec``; return a
    session ready to process units.

    Every stage edge is validated here (typed errors: unknown families,
    stage-config problems, stream-contract violations), so a session
    that opens will not fail preflight mid-stream. ``settings`` defaults
    to the environment-resolved product settings; ``reporter`` receives
    phase progress (default: none). ``windowing`` is an optional
    GOP-aligned recurrent-window plan carried to every stage via
    :class:`PipelineContext` (see its docstring for the contract).
    """
    if settings is None:
        settings = Settings.from_env()
    plan = resolve_pipeline(config, input_spec=input_spec, settings=settings)
    kwargs: dict[str, Any] = {"settings": settings}
    if reporter is not None:
        kwargs["reporter"] = reporter
    if windowing is not None:
        kwargs["windowing"] = tuple(
            (int(p0), int(p1), int(e0), int(e1))
            for p0, p1, e0, e1 in windowing)
    if publication_origin_pts is not None:
        kwargs["publication_origin_pts"] = int(publication_origin_pts)
    return PipelineSession(plan, PipelineContext(**kwargs))


__all__ = ["PipelineSession", "open_pipeline"]
