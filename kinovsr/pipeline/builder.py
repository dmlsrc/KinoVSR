"""Chain resolution and preflight validation.

``resolve_pipeline`` turns a composed user config (one ordered ``pipeline``
list plus named stage tables) into a :class:`BuildPlan`: every stage
resolved to its family factory, capability, profile, and typed config, and
the :class:`~kinovsr.processors.StreamSpec` threaded from the input
endpoint through every stage to the output endpoint. Any ordering that
cannot run fails HERE, before a single frame moves, with an error naming
the offending stage - and for edge mismatches, both sides and every
mismatched field.

Resolution is pure: it reads values and returns values (the effectful
``factory.build`` calls happen later, in :func:`build_processors`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from kinovsr.config import validate_config
from kinovsr.config.merge import resolve_stage_config, split_stage_table
from kinovsr.processors import (
    BoundaryKind,
    Capability,
    CapabilitySpec,
    FieldViolation,
    PipelineContext,
    PipelineError,
    PipelineRuntimeError,
    Processor,
    ProcessorFactory,
    StageConfigError,
    StreamConstraint,
    StreamEdgeError,
    StreamSpec,
    UnknownFamilyError,
    UnknownStageError,
    coherence_violations,
    get_factory,
)
from kinovsr.settings import Settings

INPUT_ENDPOINT = "input"
OUTPUT_ENDPOINT = "output"


@dataclass(frozen=True, slots=True)
class OutputEndpointSpec:
    """What the output endpoint can represent (the last edge's consumer)."""

    accepts: StreamConstraint = StreamConstraint()
    name: str = OUTPUT_ENDPOINT


# The default output endpoint accepts anything (in-memory consumption).
_ACCEPT_ANYTHING_OUTPUT = OutputEndpointSpec()


@dataclass(frozen=True, slots=True)
class ResolvedStage:
    """One pipeline entry, fully resolved and edge-validated."""

    name: str                      # stage-table name (the instance id)
    position: int                  # index in the pipeline list
    family: str
    factory: ProcessorFactory
    capability: Capability
    capability_spec: CapabilitySpec
    profile: str | None
    config: object                 # the family's typed config
    input_spec: StreamSpec
    output_spec: StreamSpec


@dataclass(frozen=True, slots=True)
class BuildPlan:
    stages: tuple[ResolvedStage, ...]
    input_spec: StreamSpec
    output_spec: StreamSpec        # what reaches the output endpoint


def _resolve_capability(
    stage: str,
    factory: ProcessorFactory,
    capability_token: str | None,
    profile: str | None,
) -> Capability:
    """Pick the capability per the documented rules: explicit token wins;
    else an unambiguous profile selects; else a single-capability family
    is unambiguous; anything else is an explicit-selector error."""
    advertised = factory.capabilities
    if capability_token is not None:
        try:
            capability = Capability(capability_token)
        except ValueError:
            valid = ", ".join(c.value for c in Capability)
            raise UnknownStageError(
                stage, f"unknown capability {capability_token!r} "
                f"(valid: {valid})") from None
        if capability not in advertised:
            offered = ", ".join(sorted(c.value for c in advertised))
            raise UnknownStageError(
                stage, f"family {factory.name!r} does not offer capability "
                f"{capability.value!r} (offers: {offered})")
        return capability

    if profile is not None:
        matches = [c for c, spec in advertised.items()
                   if profile in spec.profiles]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(sorted(c.value for c in matches))
            raise UnknownStageError(
                stage, f"profile {profile!r} exists under several "
                f"capabilities ({names}); state capability explicitly")
        # fall through: unknown profile is reported against the resolved
        # capability below when unambiguous, or here when it is not

    if len(advertised) == 1:
        return next(iter(advertised))
    offered = ", ".join(sorted(c.value for c in advertised))
    raise UnknownStageError(
        stage, f"family {factory.name!r} offers several capabilities "
        f"({offered}); state capability (or a profile that selects one)")


def _resolve_stage(
    stage: str,
    table: Mapping[str, Any],
    settings: Settings,
) -> tuple[ProcessorFactory, Capability, CapabilitySpec, str | None, object]:
    selector, family_settings = split_stage_table(table)
    family = selector["processor"]          # presence guaranteed by validate_config
    try:
        factory = get_factory(family)
    except UnknownFamilyError as exc:
        raise UnknownStageError(stage, str(exc)) from exc

    profile = selector.get("profile")
    if profile is not None and not isinstance(profile, str):
        raise StageConfigError(stage, "profile must be a string")
    capability_token = selector.get("capability")
    capability = _resolve_capability(stage, factory, capability_token, profile)
    capability_spec = factory.capabilities[capability]

    if profile is not None and profile not in capability_spec.profiles:
        offered = ", ".join(capability_spec.profiles) or "(none)"
        raise UnknownStageError(
            stage, f"family {factory.name!r} capability "
            f"{capability.value!r} has no profile {profile!r} "
            f"(profiles: {offered})")

    # Profile presets: family-declared in M3 (moving onto manifests); the
    # hook is optional so contract-only fakes stay tiny.
    preset: Mapping[str, Any] | None = None
    profile_defaults = getattr(factory, "profile_defaults", None)
    if profile is not None and callable(profile_defaults):
        preset = profile_defaults(capability=capability, profile=profile)
    resolved = resolve_stage_config(None, preset, family_settings)

    try:
        config = factory.parse_config(
            resolved, capability=capability, profile=profile,
            settings=settings)
    except StageConfigError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise StageConfigError(stage, str(exc)) from exc
    return factory, capability, capability_spec, profile, config


def resolve_pipeline(
    config: Mapping[str, Any],
    *,
    input_spec: StreamSpec,
    settings: Settings,
    output: OutputEndpointSpec | None = None,
) -> BuildPlan:
    """Resolve and preflight-validate the whole chain; raise typed errors
    before any processing on the first problem found."""
    if output is None:
        output = _ACCEPT_ANYTHING_OUTPUT
    validate_config(dict(config))
    pipeline: list[str] = list(config.get("pipeline", []))

    stages: list[ResolvedStage] = []
    current = input_spec
    upstream = INPUT_ENDPOINT
    provided_boundaries: set[BoundaryKind] = {BoundaryKind.STREAM_START}

    for position, stage_name in enumerate(pipeline):
        factory, capability, cap_spec, profile, cfg = _resolve_stage(
            stage_name, config[stage_name], settings)

        broken = coherence_violations(current)
        if broken:
            raise StreamEdgeError(upstream, stage_name, broken,
                                  produced=current)
        violations = list(cap_spec.accepts.violations(current))
        missing = [k for k in cap_spec.requires_boundaries
                   if k not in provided_boundaries]
        for kind in missing:
            violations.append(_boundary_violation(kind))
        if violations:
            raise StreamEdgeError(upstream, stage_name, tuple(violations),
                                  produced=current)

        produced = cap_spec.produces(current, cfg)
        if cap_spec.is_tap and produced != current:
            raise StageConfigError(
                stage_name, f"family {factory.name!r} declares a tap but "
                f"its produces transform rewrites the stream contract")
        provided_boundaries.update(cap_spec.emits_boundaries)

        stages.append(ResolvedStage(
            name=stage_name, position=position, family=factory.name,
            factory=factory, capability=capability,
            capability_spec=cap_spec, profile=profile, config=cfg,
            input_spec=current, output_spec=produced))
        current = produced
        upstream = stage_name

    broken = coherence_violations(current)
    if broken:
        raise StreamEdgeError(upstream, output.name, broken, produced=current)
    final_violations = output.accepts.violations(current)
    if final_violations:
        raise StreamEdgeError(upstream, output.name, final_violations,
                              produced=current)

    return BuildPlan(stages=tuple(stages), input_spec=input_spec,
                     output_spec=current)


def _boundary_violation(kind: BoundaryKind) -> FieldViolation:
    return FieldViolation(
        "boundaries",
        f"an upstream provider of {kind.value}",
        "not provided by the input endpoint or any earlier stage")


def _wrap_stage_error(stage: ResolvedStage, exc: Exception) -> Exception:
    """Name the offending stage on a raw processor error, leaving an
    already-typed PipelineError untouched."""
    if isinstance(exc, PipelineError):
        return exc
    return PipelineRuntimeError(stage.name, f"{type(exc).__name__}: {exc}")


def _append_context(winner: BaseException,
                    losers: Iterable[BaseException | None]) -> None:
    """Relink the winner and every outranked error into ONE strict, acyclic
    ``__context__`` chain: winner first, then its existing context nodes,
    then each loser and its context nodes - each object exactly once.

    Collecting by identity and relinking linearly is what guarantees no
    cycle. A loser's existing chain can already point back at the winner
    (Python auto-sets ``__context__`` to the exception being handled when an
    error is raised mid-handling), which a naive append would close into a
    loop; flattening by id and terminating at ``None`` breaks it. The winner
    keeps its documented precedence; every outranked error stays reachable,
    none silently dropped.
    """
    ordered: list[BaseException] = []
    seen: set[int] = set()

    def collect(exc: BaseException | None) -> None:
        node = exc
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            ordered.append(node)
            node = node.__context__

    collect(winner)
    for loser in losers:
        collect(loser)
    # Pairwise adjacent (n-1 pairs); the lengths differ by one by design.
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        earlier.__context__ = later
    ordered[-1].__context__ = None


def build_processors(
    plan: BuildPlan, context: PipelineContext,
) -> tuple[tuple[ResolvedStage, Processor], ...]:
    """The effectful step: construct every stage instance from the plan.

    Instances come back paired with their resolved stage, in chain order,
    each built with a per-stage context. Duplicate pipeline entries get
    independent instances (state is never shared; immutable weights may be
    cached underneath by the family).

    Construction is transactional: if a later stage's build raises, every
    already-built instance is closed before the original error propagates,
    so a failing chain never leaks native sessions or weights."""
    built: list[tuple[ResolvedStage, Processor]] = []
    try:
        for stage in plan.stages:
            stage_context = context.for_stage(stage.name)
            built.append(
                (stage,
                 stage.factory.build(stage.config, context=stage_context)))
    except BaseException as build_error:
        interrupt: BaseException | None = None
        close_errors: list[BaseException] = []
        for stage, processor in built:
            try:
                processor.close(context.for_stage(stage.name))
            except Exception as exc:  # noqa: BLE001 - collected, chained below
                # Ordinary close failures lose to the build error but ride its
                # context chain so a leaked-resource failure is still visible.
                close_errors.append(_wrap_stage_error(stage, exc))
            except BaseException as exc:
                # KeyboardInterrupt/SystemExit during cleanup: finish
                # closing the remaining stages first, then deliver it
                # (chained onto the build error).
                if interrupt is None:
                    interrupt = exc
        if interrupt is not None:
            _append_context(interrupt, close_errors)
            raise interrupt from build_error
        _append_context(build_error, close_errors)
        raise
    return tuple(built)


__all__ = [
    "BuildPlan",
    "INPUT_ENDPOINT",
    "OUTPUT_ENDPOINT",
    "OutputEndpointSpec",
    "ResolvedStage",
    "build_processors",
    "resolve_pipeline",
]
