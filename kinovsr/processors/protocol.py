"""The processor contract: lifecycle protocol, factory, and context.

Pure core, effectful edges (planning 05): a family's MLX math is a pure
function ``(params, inputs, state) -> (outputs, state)`` - anything
``mx.compile``d must not mutate attributes, read hidden mutable state, or
perform I/O. The lifecycle methods below are the deliberately impure
edges: they own sessions, buffers, and weight loading, and they pass
temporal state through the pure core as explicit values, so ``reset``
means "replace the state value".

Scheduler guarantees the implementations may rely on:

- ``prepare`` runs once, before the first unit, with the exact input spec
  this instance will see (compile warmup belongs here);
- ``reset`` runs before the first unit AFTER a boundary reaches this
  stage (including STREAM_START before the first unit of the run);
- ``flush`` drains the buffered tail. It runs at end of stream, and ALSO
  just before ``reset`` at a mid-stream boundary so the pre-boundary tail
  leaves with pre-boundary state; implementations must tolerate further
  ``process`` calls after a mid-stream flush;
- ``close`` runs exactly once - on success, cancellation, or failure.

Framework guarantees the scheduler owes the implementations: it never
re-enters a stage concurrently, never inserts work inside the stage's
compiled regions, and never forces a per-frame host/device sync.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from kinovsr.reporting import NullReporter, Reporter
from kinovsr.settings import Settings

from .boundaries import Boundary
from .capabilities import Capability, CapabilitySpec
from .specs import StreamSpec
from .units import FrameUnit


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """What every lifecycle call receives.

    A frozen value: per-stage variants are new instances via
    :meth:`for_stage`. Carries no mutable services on purpose - progress
    goes through the reporter contract, and nothing here may force MLX
    evaluation.
    """

    settings: Settings
    reporter: Reporter = NullReporter()
    stage_id: str | None = None

    def for_stage(self, stage_id: str) -> PipelineContext:
        return replace(self, stage_id=stage_id)


@runtime_checkable
class Processor(Protocol):
    """One configured stage instance's lifecycle."""

    def prepare(self, input_spec: StreamSpec,
                context: PipelineContext) -> None:
        """Bind to the concrete input contract; compile/warm the core."""

    def process(self, unit: FrameUnit,
                context: PipelineContext) -> Iterable[FrameUnit]:
        """Consume one unit; yield zero or more output units.

        Yielding nothing is how windowed/buffering stages accumulate;
        yielding several is how interpolation expands the timeline.
        Units arrive with upstream boundaries already stripped - boundary
        information reaches a stage only through ``reset`` - and the
        scheduler re-attaches them downstream. Emitting a NEW boundary on
        an output unit (cut detection) is the one legitimate reason to
        set ``boundaries`` here.
        """

    def reset(self, boundary: Boundary, context: PipelineContext) -> None:
        """Drop temporal state: replace the explicit state value(s) with
        their initial form. Never buffers across a boundary."""

    def flush(self, context: PipelineContext) -> Iterable[FrameUnit]:
        """Emit any buffered tail at end of stream."""

    def close(self, context: PipelineContext) -> None:
        """Release sessions/resources. Called exactly once, always."""


@runtime_checkable
class ProcessorFactory(Protocol):
    """A family's entry in the catalog: metadata plus two pure-ish steps.

    ``parse_config`` is a pure function over values: raw mapping in,
    typed frozen config out (raising StageConfigError on bad keys/values).
    ``build`` is the effectful step that may load weights and create the
    stage instance.
    """

    name: str
    capabilities: Mapping[Capability, CapabilitySpec]

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> object:
        ...

    def build(self, config: object, *,
              context: PipelineContext) -> Processor:
        ...


@runtime_checkable
class BracketFactory(Protocol):
    """A factory whose config can bracket the chain with a companion.

    When a capability's ``companion(config)`` returns a spec, the builder
    calls ``build_bracket`` instead of ``build`` and gets back both halves
    at once, so they can share state (a PTS-keyed capture buffer). The
    builder places ``pre`` at the stage's declared position and ``post`` at
    the last point the stream stays in the pre-pass's layout (for the
    MLX-domain restore, before any native-CV stage; the chain end if all
    stages keep that layout). A family only implements this when one of its
    capabilities declares a companion.
    """

    def build_bracket(
        self, config: object, *, context: PipelineContext,
    ) -> tuple[Processor, Processor]:
        """Return ``(pre, post)`` sharing state; ``post`` runs at the end."""


__all__ = [
    "BracketFactory",
    "PipelineContext",
    "Processor",
    "ProcessorFactory",
]
