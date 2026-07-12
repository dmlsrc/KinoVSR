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
    Layout,
    PipelineContext,
    PipelineError,
    StreamSpec,
)
from kinovsr.reporting import Reporter
from kinovsr.settings import Settings

from .builder import BuildPlan, build_processors, resolve_pipeline
from .scheduler import ChainRun, run_chain

# CV payload layouts whose buffers must be copied to give a retaining
# consumer an owned output (MLX arrays are immutable values, never copied).
_CV_LAYOUTS = frozenset({Layout.CV_NV12, Layout.CV_BGRA, Layout.CV_RGBA_HALF})


class _OwnedCvOutputs:
    """Wrap a :class:`ChainRun` so each yielded CVPixelBuffer output is a
    fresh, host-owned deep copy - the retain-safe default for CV-layout
    sessions. Iteration, cancellation, and context management delegate to the
    underlying run; only the payload is replaced.
    """

    def __init__(self, run: ChainRun) -> None:
        self._run = run
        from kinovsr.media.pixel_buffers import copy_pixel_buffer
        self._copy = copy_pixel_buffer

    def __iter__(self) -> _OwnedCvOutputs:
        return self

    def __next__(self) -> FrameUnit:
        unit = next(self._run)
        return unit.with_payload(self._copy(unit.payload))

    def close(self) -> None:
        self._run.close()

    def __enter__(self) -> _OwnedCvOutputs:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._run.__exit__(exc_type, exc, tb)


class PipelineSession:
    """One validated pipeline, bound to an input spec, run at most once."""

    def __init__(self, plan: BuildPlan, context: PipelineContext) -> None:
        self._plan = plan
        self._context = context
        self._run: ChainRun | None = None
        self._consumed = False
        self._built: tuple = ()

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
        built = build_processors(self._plan, self._context)
        self._built = built
        run = run_chain(built, units, self._context)
        self._run = run
        if retain_outputs and self._plan.output_spec.frame.layout in _CV_LAYOUTS:
            return _OwnedCvOutputs(run)
        return run

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
    return PipelineSession(plan, PipelineContext(**kwargs))


__all__ = ["PipelineSession", "open_pipeline"]
