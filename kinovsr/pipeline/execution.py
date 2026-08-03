"""Physical execution metadata for the bounded streaming runtime.

The public pipeline remains a sequence of logical ``ResolvedStage`` values.
After preflight, each built processor receives an internal execution contract:
one affinity owner, finite buffering, resource claims, and an ordering rule.
Factories may contribute affinity/resource metadata without exposing physical
nodes through user config or ``--print-config``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from kinovsr.processors import (
    DType,
    FrameUnit,
    Layout,
    PipelineContext,
    StreamSpec,
)
from kinovsr.processors.capabilities import TemporalMode

from .leases import StorageDescriptor, StorageKind

if TYPE_CHECKING:
    from kinovsr.processors import Processor

    from .builder import ResolvedStage


class ResourceKind(Enum):
    GPU = "gpu"
    ANE = "ane"
    MEDIA = "media"
    CPU = "cpu"
    IO = "io"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    OPAQUE_NATIVE = "opaque_native"


class Ordering(Enum):
    SERIAL = "serial"


@dataclass(frozen=True, slots=True)
class AffinityKey:
    """One long-lived serial execution owner."""

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("affinity name must not be empty")


@dataclass(frozen=True, slots=True)
class ResourceClaim:
    resource: ResourceKind
    weight: int = 1
    exclusive: bool = False

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("resource claim weight must be positive")


@dataclass(frozen=True, slots=True)
class BufferingSpec:
    retained_input_units: int
    pending_output_units: int
    native_slots: int
    estimated_bytes: int

    def __post_init__(self) -> None:
        values = dataclasses.asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def channel_units(self) -> int:
        """Finite cross-island output capacity for the compatibility path."""
        return max(1, self.pending_output_units)


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    affinity: AffinityKey
    serial_key: str
    claims: tuple[ResourceClaim, ...]
    ordering: Ordering
    buffering: BufferingSpec
    max_emissions_per_input: int
    resource_handoff_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.serial_key:
            raise ValueError("serial_key must not be empty")
        if self.max_emissions_per_input <= 0:
            raise ValueError("max_emissions_per_input must be positive")
        if (
            not math.isfinite(self.resource_handoff_seconds)
            or self.resource_handoff_seconds < 0
        ):
            raise ValueError(
                "resource_handoff_seconds must be finite and nonnegative"
            )


@dataclass(frozen=True, slots=True)
class PhysicalOperation:
    """One logical-stage owner plus its optional storage bridges.

    A bridge is an ordinary :class:`ExecutionSpec`: the runtime invokes the
    processor's ``prepare_input``/``prepare_output`` hook on that affinity
    before/after the serial logical owner.  This keeps framework knowledge in
    the family while the scheduler only sees physical execution contracts.
    """

    stage: ResolvedStage
    processor: Processor
    context: PipelineContext
    execution: ExecutionSpec
    input_bridge: ExecutionSpec | None = None
    output_bridge: ExecutionSpec | None = None
    output_slots: int = 0
    output_storage: StorageDescriptor = StorageDescriptor()


_DTYPE_BYTES = {
    DType.FLOAT32: 4,
    DType.FLOAT16: 2,
    DType.UINT8: 1,
}


def estimate_frame_bytes(spec: StreamSpec) -> int:
    """Conservative payload charge derived only from the validated edge."""
    frame = spec.frame
    pixels = int(frame.geometry.width) * int(frame.geometry.height)
    if frame.layout is Layout.CV_NV12:
        return max(1, (pixels * 3 + 1) // 2)
    channels = 4 if frame.layout in {
        Layout.CV_BGRA,
        Layout.CV_RGBA_HALF,
    } else 3
    return max(1, pixels * channels * _DTYPE_BYTES[frame.dtype])


def storage_descriptor_for_spec(
    spec: StreamSpec,
    *,
    borrowed: bool = False,
    reusable: bool = False,
    label: str | None = None,
) -> StorageDescriptor:
    """Derive transport storage from the validated pixel-layout edge."""
    if spec.frame.layout is Layout.MLX_RGB_HWC:
        kind = StorageKind.MLX_ARRAY
    elif spec.frame.layout in {
        Layout.CV_BGRA,
        Layout.CV_NV12,
        Layout.CV_RGBA_HALF,
    }:
        kind = StorageKind.CV_PIXEL_BUFFER
    else:
        kind = StorageKind.PYTHON
    return StorageDescriptor(
        kind=kind,
        borrowed=borrowed,
        reusable=reusable,
        label=label,
    )


def _declared_affinity(stage: ResolvedStage) -> str | None:
    declared = getattr(stage.factory, "execution_affinity", None)
    if callable(declared):
        declared = declared(stage.config)
    if declared is None:
        return None
    if not isinstance(declared, str) or not declared:
        raise TypeError(
            f"factory {stage.family!r} execution_affinity must be a nonempty string"
        )
    return declared.format(stage=stage.name)


def _default_affinity(stage: ResolvedStage) -> AffinityKey:
    declared = _declared_affinity(stage)
    if declared is not None:
        return AffinityKey(declared)
    layouts = {stage.input_spec.frame.layout, stage.output_spec.frame.layout}
    if layouts == {Layout.MLX_RGB_HWC}:
        # All ordinary MLX stages share one long-lived owner.
        return AffinityKey("mlx")
    # Native compatibility stages keep independent serial owners. A family
    # that shares a native context across instances declares the same key.
    return AffinityKey(f"native:{stage.name}")


def _declared_resources(stage: ResolvedStage) -> tuple[ResourceClaim, ...] | None:
    declared = getattr(stage.factory, "execution_resources", None)
    if callable(declared):
        declared = declared(stage.config)
    if declared is None:
        return None
    claims: list[ResourceClaim] = []
    for value in declared:
        if isinstance(value, ResourceClaim):
            claims.append(value)
            continue
        try:
            kind = value if isinstance(value, ResourceKind) else ResourceKind(str(value))
        except ValueError as exc:
            raise ValueError(
                f"factory {stage.family!r} declared unknown execution resource {value!r}"
            ) from exc
        claims.append(ResourceClaim(kind))
    return tuple(claims)


def _default_resources(stage: ResolvedStage) -> tuple[ResourceClaim, ...]:
    declared = _declared_resources(stage)
    if declared is not None:
        return declared
    layouts = {stage.input_spec.frame.layout, stage.output_spec.frame.layout}
    if Layout.MLX_RGB_HWC in layouts:
        return (ResourceClaim(ResourceKind.GPU),)
    return (ResourceClaim(ResourceKind.OPAQUE_NATIVE),)


def _configured_window(config: object) -> int | None:
    for name in ("window", "window_size", "chunk_size"):
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def resolve_buffering(
    stage: ResolvedStage,
    context: PipelineContext,
    *,
    native_slots: int,
) -> BufferingSpec:
    """Resolve a finite compatibility bound from typed config and specs."""
    capability = stage.capability_spec
    radius = max(0, int(capability.temporal_radius))
    configured = _configured_window(stage.config)
    if context.gop is not None and radius:
        # Reactive windows retain their processing range plus split trim.
        retained = context.gop.max_window + 2
    elif configured is not None:
        retained = configured
    elif capability.temporal_mode is TemporalMode.CENTERED:
        retained = 2 * radius + 1
    elif capability.temporal_mode is TemporalMode.CAUSAL:
        retained = radius + 1
    else:
        retained = 1
    # A compatibility processor can synchronously expose a complete bounded
    # window as one input's emissions.  Its private egress must be able to
    # detach that burst from the owner before the two-unit cross-island edge
    # applies backpressure; otherwise a native window can only advance when a
    # downstream consumer returns to the old pull stack.  Per-frame/causal
    # operations retain the ordinary two-unit bound.
    pending = max(
        2,
        retained
        if capability.temporal_mode is TemporalMode.CENTERED
        else 2,
    )
    frame_bytes = max(
        estimate_frame_bytes(stage.input_spec),
        estimate_frame_bytes(stage.output_spec),
    )
    total_units = retained + pending + native_slots
    return BufferingSpec(
        retained_input_units=retained,
        pending_output_units=pending,
        native_slots=native_slots,
        estimated_bytes=frame_bytes * total_units,
    )


def resolve_execution(
    stage: ResolvedStage,
    context: PipelineContext,
) -> ExecutionSpec:
    affinity = _default_affinity(stage)
    native_slots = 0 if affinity.name == "mlx" else 2
    declared_slots = getattr(stage.factory, "execution_native_slots", None)
    if callable(declared_slots):
        declared_slots = declared_slots(stage.config)
    if declared_slots is not None:
        if (
            isinstance(declared_slots, bool)
            or not isinstance(declared_slots, int)
            or declared_slots < 0
        ):
            raise TypeError(
                f"factory {stage.family!r} execution_native_slots must be nonnegative"
            )
        native_slots = declared_slots
    buffering = resolve_buffering(stage, context, native_slots=native_slots)
    declared_buffering = getattr(stage.factory, "execution_buffering", None)
    if callable(declared_buffering):
        buffering = declared_buffering(
            stage.config,
            input_spec=stage.input_spec,
            output_spec=stage.output_spec,
            context=context,
            default=buffering,
        )
        if not isinstance(buffering, BufferingSpec):
            raise TypeError(
                f"factory {stage.family!r} execution_buffering must return "
                "BufferingSpec"
            )
    resource_handoff = getattr(
        stage.factory,
        "execution_resource_handoff_seconds",
        0.0,
    )
    if callable(resource_handoff):
        resource_handoff = resource_handoff(stage.config)
    if isinstance(resource_handoff, bool) or not isinstance(
        resource_handoff,
        int | float,
    ):
        raise TypeError(
            f"factory {stage.family!r} execution_resource_handoff_seconds "
            "must be a number"
        )
    return ExecutionSpec(
        affinity=affinity,
        serial_key=affinity.name,
        claims=_default_resources(stage),
        ordering=Ordering.SERIAL,
        buffering=buffering,
        max_emissions_per_input=buffering.pending_output_units,
        resource_handoff_seconds=float(resource_handoff),
    )


def _bridge_execution(
    stage: ResolvedStage,
    context: PipelineContext,
    *,
    side: str,
) -> ExecutionSpec | None:
    attribute = f"execution_{side}_affinity"
    declared = getattr(stage.factory, attribute, None)
    if callable(declared):
        declared = declared(stage.config)
    if declared is None:
        return None
    if not isinstance(declared, str) or not declared:
        raise TypeError(
            f"factory {stage.family!r} {attribute} must be a nonempty string"
        )
    edge = stage.input_spec if side == "input" else stage.output_spec
    # An MLX bridge declaration is conditional on that concrete validated
    # edge.  Head VT stages with native CV input therefore do not wake an MLX
    # lane merely because the family can also accept MLX mid-chain.
    if declared == "mlx" and edge.frame.layout is not Layout.MLX_RGB_HWC:
        return None
    frame_bytes = estimate_frame_bytes(edge)
    return ExecutionSpec(
        affinity=AffinityKey(declared.format(stage=stage.name)),
        serial_key=declared.format(stage=stage.name),
        claims=(
            ResourceClaim(ResourceKind.GPU),
            ResourceClaim(ResourceKind.MEMORY_BANDWIDTH),
        ),
        ordering=Ordering.SERIAL,
        buffering=BufferingSpec(
            retained_input_units=1,
            pending_output_units=1,
            native_slots=0,
            estimated_bytes=frame_bytes,
        ),
        max_emissions_per_input=1,
    )


def _output_slots(stage: ResolvedStage) -> int:
    declared = getattr(stage.factory, "execution_output_slots", 0)
    if callable(declared):
        declared = declared(stage.config)
    if (
        isinstance(declared, bool)
        or not isinstance(declared, int)
        or declared < 0
    ):
        raise TypeError(
            f"factory {stage.family!r} execution_output_slots must be "
            "a nonnegative integer"
        )
    return declared


def _output_storage(
    stage: ResolvedStage,
    *,
    output_slots: int,
) -> StorageDescriptor:
    declared = getattr(stage.factory, "execution_output_storage", None)
    if callable(declared):
        declared = declared(stage.config)
    if declared is not None:
        if not isinstance(declared, StorageDescriptor):
            raise TypeError(
                f"factory {stage.family!r} execution_output_storage must "
                "be a StorageDescriptor"
            )
        return declared
    return storage_descriptor_for_spec(
        stage.output_spec,
        borrowed=output_slots > 0,
        reusable=output_slots > 0,
        label=f"{stage.name}:output",
    )


class _FusedProcessor:
    """A same-affinity per-frame island with no internal thread hops."""

    def __init__(self, members: tuple[PhysicalOperation, ...]) -> None:
        self._members = members
        self._started = [False] * len(members)
        self._pending: list[list[object]] = [[] for _ in members]

    @staticmethod
    def _call(member: PhysicalOperation, callback):
        from .builder import _wrap_stage_error

        try:
            return callback()
        except Exception as exc:
            wrapped = _wrap_stage_error(member.stage, exc)
            if wrapped is exc:
                raise
            raise wrapped from exc

    @classmethod
    def _iterate(
        cls,
        member: PhysicalOperation,
        values: Iterable[FrameUnit],
    ) -> Iterator[FrameUnit]:
        iterator = iter(values)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                return
            except Exception as exc:
                from .builder import _wrap_stage_error

                wrapped = _wrap_stage_error(member.stage, exc)
                if wrapped is exc:
                    raise
                raise wrapped from exc

    def prepare(self, input_spec: StreamSpec, context: PipelineContext) -> None:
        for member in self._members:
            self._call(
                member,
                lambda member=member: member.processor.prepare(
                    member.stage.input_spec,
                    member.context,
                ),
            )

    def _attach_pending(self, index: int, unit: FrameUnit) -> FrameUnit:
        pending = self._pending[index]
        if not pending:
            return unit
        attached = dataclasses.replace(
            unit,
            boundaries=(*pending, *unit.boundaries),
        )
        pending.clear()
        return attached

    def _feed(self, index: int, unit: FrameUnit) -> Iterator[FrameUnit]:
        if index == len(self._members):
            yield unit
            return
        member = self._members[index]
        if unit.boundaries:
            if self._started[index]:
                flushed = self._call(
                    member,
                    lambda: member.processor.flush(member.context),
                )
                for tail in self._iterate(member, flushed):
                    tail = self._attach_pending(index, tail)
                    yield from self._feed(index + 1, tail)
            for boundary in unit.boundaries:
                self._call(
                    member,
                    lambda boundary=boundary: member.processor.reset(
                        boundary,
                        member.context,
                    ),
                )
            self._pending[index].extend(unit.boundaries)
            unit = dataclasses.replace(unit, boundaries=())
        self._started[index] = True
        produced = self._call(
            member,
            lambda: member.processor.process(unit, member.context),
        )
        for output in self._iterate(member, produced):
            output = self._attach_pending(index, output)
            yield from self._feed(index + 1, output)

    def process(
        self,
        unit: FrameUnit,
        context: PipelineContext,
    ) -> Iterator[FrameUnit]:
        yield from self._feed(0, unit)

    def flush(self, context: PipelineContext) -> Iterator[FrameUnit]:
        for index, member in enumerate(self._members):
            flushed = self._call(
                member,
                lambda member=member: member.processor.flush(member.context),
            )
            for output in self._iterate(member, flushed):
                output = self._attach_pending(index, output)
                yield from self._feed(index + 1, output)

    def reset(self, boundary, context: PipelineContext) -> None:
        for index, member in enumerate(self._members):
            self._call(
                member,
                lambda member=member: member.processor.reset(
                    boundary,
                    member.context,
                ),
            )
            self._started[index] = False
            self._pending[index].clear()

    def close(self, context: PipelineContext) -> None:
        from .builder import _append_context

        failures: list[BaseException] = []
        for member in reversed(self._members):
            try:
                self._call(
                    member,
                    lambda member=member: member.processor.close(member.context),
                )
            except BaseException as exc:  # noqa: BLE001 - precedence below
                failures.append(exc)
        if not failures:
            return
        winner = next(
            (failure for failure in failures if not isinstance(failure, Exception)),
            failures[0],
        )
        _append_context(
            winner,
            [failure for failure in failures if failure is not winner],
        )
        raise winner


def _merge_claims(
    operations: tuple[PhysicalOperation, ...],
) -> tuple[ResourceClaim, ...]:
    merged: dict[ResourceKind, ResourceClaim] = {}
    for operation in operations:
        for claim in operation.execution.claims:
            previous = merged.get(claim.resource)
            merged[claim.resource] = ResourceClaim(
                claim.resource,
                weight=max(claim.weight, previous.weight if previous else 0),
                exclusive=claim.exclusive or bool(previous and previous.exclusive),
            )
    return tuple(merged[kind] for kind in sorted(merged, key=lambda item: item.value))


def _fuse_group(
    operations: tuple[PhysicalOperation, ...],
) -> PhysicalOperation:
    first, last = operations[0], operations[-1]
    stage = dataclasses.replace(
        first.stage,
        name="+".join(operation.stage.name for operation in operations),
        output_spec=last.stage.output_spec,
    )
    buffering = BufferingSpec(
        retained_input_units=sum(
            operation.execution.buffering.retained_input_units
            for operation in operations
        ),
        pending_output_units=max(
            operation.execution.buffering.pending_output_units
            for operation in operations
        ),
        native_slots=sum(
            operation.execution.buffering.native_slots
            for operation in operations
        ),
        estimated_bytes=sum(
            operation.execution.buffering.estimated_bytes
            for operation in operations
        ),
    )
    execution = dataclasses.replace(
        first.execution,
        serial_key=first.execution.affinity.name,
        claims=_merge_claims(operations),
        buffering=buffering,
        max_emissions_per_input=max(
            operation.execution.max_emissions_per_input
            for operation in operations
        ),
        resource_handoff_seconds=max(
            operation.execution.resource_handoff_seconds
            for operation in operations
        ),
    )
    return PhysicalOperation(
        stage=stage,
        processor=_FusedProcessor(operations),
        context=first.context,
        execution=execution,
        output_storage=last.output_storage,
    )


def _fuse_operations(
    operations: tuple[PhysicalOperation, ...],
) -> tuple[PhysicalOperation, ...]:
    islands: list[PhysicalOperation] = []
    pending: list[PhysicalOperation] = []

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        group = tuple(pending)
        islands.append(group[0] if len(group) == 1 else _fuse_group(group))
        pending = []

    for operation in operations:
        fusible = (
            operation.input_bridge is None
            and operation.output_bridge is None
            and operation.stage.capability_spec.temporal_mode
            is TemporalMode.PER_FRAME
        )
        if not fusible:
            flush_pending()
            islands.append(operation)
            continue
        if (
            pending
            and pending[-1].execution.affinity
            != operation.execution.affinity
        ):
            flush_pending()
        pending.append(operation)
    flush_pending()
    return tuple(islands)


def expand_operations(
    built: tuple[tuple[ResolvedStage, Processor], ...],
    context: PipelineContext,
) -> tuple[PhysicalOperation, ...]:
    """Purely attach physical contracts after M7's resolved stage boundary."""
    operations: list[PhysicalOperation] = []
    for stage, processor in built:
        output_slots = _output_slots(stage)
        operations.append(
            PhysicalOperation(
                stage=stage,
                processor=processor,
                context=context.for_stage(stage.name),
                execution=resolve_execution(stage, context),
                input_bridge=_bridge_execution(stage, context, side="input"),
                output_bridge=_bridge_execution(stage, context, side="output"),
                output_slots=output_slots,
                output_storage=_output_storage(
                    stage,
                    output_slots=output_slots,
                ),
            )
        )
    return _fuse_operations(tuple(operations))


__all__ = [
    "AffinityKey",
    "BufferingSpec",
    "ExecutionSpec",
    "Ordering",
    "PhysicalOperation",
    "ResourceClaim",
    "ResourceKind",
    "estimate_frame_bytes",
    "expand_operations",
    "resolve_buffering",
    "resolve_execution",
    "storage_descriptor_for_spec",
]
