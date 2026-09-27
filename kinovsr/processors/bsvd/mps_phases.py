"""One-step BSVD windows executed by one direct MPSGraph ANECIR program.

One schedule-generic ANE entry handles fill, steady state, and drain. Its
sixteen logical recurrent tensors are packed into four persistent
``anec.state`` IOSurfaces, so no state crosses through Python, MLX, or a Core
ML prediction. Keeping one concrete program and stable request maps resident
avoids raw-runtime phase re-entry while the one-step cadence preserves
downstream GPU overlap.

The six skip rings remain reusable IOSurface bindings shared with that entry.
Both state and delayed features therefore cross dispatch boundaries without
host copies while the backend-neutral ``NoneFlowNet`` mirror remains the
source of truth for frame order and output visibility.
"""

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx

from kinovsr.native import mpsgraph as mg
from kinovsr.native import mpsgraph_state as mgs

from .mps_layout import _LINES, _PUSH_OUTPUTS, _SKIP_DEPTHS, _STATE_GROUPS, _net_keys
from .schedule import NoneFlowNet, StepRecord

_STEPS = 1
_DRAIN_STEPS = 16
_STATEFUL_CACHE_LABEL = "scheduled-stateful-generic1-direct-v3"
_UNIT_KEYS = ("u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7")


class _Binding(Protocol):
    def write(self, value: bytes | bytearray | memoryview | mx.array) -> None: ...


class _Entry(Protocol):
    def bind(self, name: str, *, shared: _Binding | None = None) -> _Binding: ...

    def write_feeds(self, values: Mapping[str, mx.array]) -> None: ...

    def begin_dispatch(
        self, bindings: Mapping[str, _Binding] | None = None
    ) -> Callable[[], None]: ...

    def read(self, wanted: set[str] | None = None) -> dict[str, mx.array]: ...


class _Executable(Protocol):
    @property
    def state_specs(self) -> tuple[object, ...]: ...

    def entry(self, name: str) -> _Entry: ...

    def prepare(self) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class _Pipeline(Protocol):
    @property
    def in_flight(self) -> bool: ...

    def idle(self) -> bool: ...

    def submit(self, job: Callable[[], object]) -> None: ...

    def join(self) -> None: ...

    def drain(self) -> None: ...


class _MpsNet(Protocol):
    input_channels: int
    _prefixes: tuple[str, str]
    _weights: dict[str, mx.array] | None
    _ane_fw_to_fw_signal: bool
    _ane_late_latch: bool
    _line_shapes: list[tuple[int, ...]]
    _state_slots: list[mg.TensorBinding]
    _state_bindings: dict[str, mg.TensorBinding]
    _rings: list[deque[_Binding]]
    _slots: list[list[_Binding]]
    _free: list[deque[_Binding]]
    _zero_bindings: list[_Binding | None]
    _discard_bindings: list[_Binding | None]
    _zero_frame: mx.array | None

    @property
    def _pipeline(self) -> _Pipeline: ...

    def _executable_cache(self, label: str, height: int, width: int) -> Path | None: ...


def stateful_cache_ready(net: _MpsNet, height: int, width: int) -> bool:
    """Whether this runtime's single-entry persistent cache is complete."""
    cache = net._executable_cache(_STATEFUL_CACHE_LABEL, height, width)
    return cache is not None and mgs.stateful_cache_ready(cache)


@dataclass
class _Emitted:
    out: object
    x0: object
    x1: object
    out_channels: int


@dataclass
class _Chunk[G]:
    graph: G
    frame_names: tuple[str, ...]
    gate_names: tuple[str, ...]
    left_gate_names: tuple[str, ...]
    output_names: tuple[str, ...]
    pop_names: tuple[tuple[str, ...], ...]
    push_names: tuple[tuple[str | None, ...], ...]


@dataclass
class _Action:
    frames: tuple[mx.array | None, ...]
    records: tuple[StepRecord, ...]


@dataclass
class _Resolved:
    frames: tuple[mx.array, ...]
    gates: tuple[mx.array, ...]
    left_gates: tuple[mx.array, ...]
    records: tuple[StepRecord, ...]


@dataclass
class _Prepared:
    job: Callable[[], None]
    records: tuple[StepRecord, ...]
    rings: list[deque[_Binding]]
    free: list[deque[_Binding]]


def _line_shapes(net: _MpsNet, height: int, width: int) -> tuple[tuple[int, ...], ...]:
    """Skip-ring shapes without compiling the ordinary graph."""

    assert net._weights is not None  # only close() clears it, after joining any preheat
    keys_by_net = tuple(_net_keys(prefix) for prefix in net._prefixes)
    line_shapes: list[tuple[int, ...]] = []
    for keys in keys_by_net:
        line_shapes.extend(
            (
                (1, 3, height, width),
                (
                    1,
                    int(net._weights[keys["inc3"] + ".weight"].shape[0]),
                    height,
                    width,
                ),
                (
                    1,
                    int(net._weights[keys["u0"] + ".weight"].shape[0]),
                    height // 2,
                    width // 2,
                ),
            )
        )
    return tuple(line_shapes)


def _state_specs(net: _MpsNet, height: int, width: int) -> tuple[mgs.StateTensorSpec, ...]:
    """Pack shape-compatible BiBuffer cells into four persistent ports."""

    unit_shapes = _unit_state_shapes(net, height, width)
    states = []
    for group, units in enumerate(_STATE_GROUPS):
        member = unit_shapes[units[0]]
        if any(unit_shapes[unit] != member for unit in units[1:]):
            raise RuntimeError(f"BSVD state group {group} mixes incompatible unit shapes")
        logical = (
            member[0],
            member[1] * len(units),
            member[2],
            member[3],
        )
        states.append(
            mgs.StateTensorSpec(
                f"state_group_{group}",
                logical,
                _packed_storage_shape(logical),
            )
        )
    return tuple(states)


def _unit_state_shapes(net: _MpsNet, height: int, width: int) -> tuple[tuple[int, ...], ...]:
    """Return the sixteen logical BiBuffer cell shapes in execution order."""

    assert net._weights is not None  # only close() clears it, after joining any preheat
    keys_by_net = tuple(_net_keys(prefix) for prefix in net._prefixes)
    shapes = []
    for unit in range(16):
        local = unit % 8
        key = keys_by_net[unit // 8][_UNIT_KEYS[local]]
        channels = int(net._weights[key + ".weight"].shape[0])
        divisor = 2 if local < 2 or local > 5 else 4
        shapes.append(
            (
                1,
                channels + channels // 8,
                height // divisor,
                width // divisor,
            )
        )
    return tuple(shapes)


def _packed_storage_shape(logical_shape: tuple[int, ...], *, limit: int = 255) -> tuple[int, ...]:
    """Balance a packed slab's ANE-safe storage away from its leading axis.

    ``safe_storage_shape`` first folds oversized logical axes into N. Packed
    groups can leave several factors there; move those factors back into a
    spatial axis when it remains under the ANE dimension limit. This preserves
    the ordinary cell layout while minimizing the physical leading dimension.
    """
    storage = list(mgs.safe_storage_shape(logical_shape, limit=limit))
    for axis in (2, 3, 1):
        while storage[0] > 1:
            factor = next(
                (
                    candidate
                    for candidate in range(2, storage[0] + 1)
                    if storage[0] % candidate == 0 and storage[axis] * candidate <= limit
                ),
                None,
            )
            if factor is None:
                break
            storage[0] //= factor
            storage[axis] *= factor
    result = tuple(storage)
    if prod(result) != prod(logical_shape):
        raise AssertionError("packed state storage changed element count")
    return result


def _build_chunk(net: _MpsNet, height: int, width: int) -> _Chunk[mgs.Program]:
    """Compile the schedule-generic one-step direct MPSGraph program."""

    assert net._weights is not None  # only close() clears it, after joining any preheat
    weights = net._weights
    keys_by_net = tuple(_net_keys(prefix) for prefix in net._prefixes)
    line_shapes = _line_shapes(net, height, width)
    state_specs = _state_specs(net, height, width)
    builder = mg.GraphBuilder(mg.FLOAT16)

    dynamic: set[str] = set()
    frame_names: list[str] = []
    gate_names: list[str] = []
    left_gate_names: list[str] = []
    pop_names: list[tuple[str, ...]] = []
    frame_tensors = []
    gate_tensors = []
    left_gate_tensors = []
    pop_tensors = []
    for step in range(_STEPS):
        frame_name = f"frame_{step}"
        gate_name = f"gate_{step}"
        left_gate_name = f"left_gate_{step}"
        frame_names.append(frame_name)
        gate_names.append(gate_name)
        left_gate_names.append(left_gate_name)
        frame_tensors.append(
            builder.placeholder((1, net.input_channels, height, width), frame_name)
        )
        gate_tensors.append(builder.placeholder((1, 16, 1, 1), gate_name))
        left_gate_tensors.append(builder.placeholder((1, 16, 1, 1), left_gate_name))
        step_names = []
        step_tensors = []
        for line in range(_LINES):
            name = f"skip_{step}_{line}"
            step_names.append(name)
            step_tensors.append(builder.placeholder(line_shapes[line], name))
            dynamic.add(name)
        pop_names.append(tuple(step_names))
        pop_tensors.append(tuple(step_tensors))

    # State ports trail every ordinary/dynamic port. This is the large-model
    # ABI exercised by MPSGraph's own mapped-product runtime.
    logical_states = mgs.state_placeholders(builder, state_specs)
    unit_shapes = _unit_state_shapes(net, height, width)
    state_values = {}
    state_shapes = {}
    for group, (state, units) in enumerate(zip(state_specs, _STATE_GROUPS, strict=True)):
        slab = logical_states[state.name]
        member = unit_shapes[units[0]]
        channels = member[1]
        for slot, unit in enumerate(units):
            if unit_shapes[unit] != member:
                raise RuntimeError(f"BSVD state group {group} mixes incompatible unit shapes")
            state_values[unit] = builder.slice_channels(
                slab,
                slot * channels,
                channels,
                f"state_group_{group}.unit_{unit}",
            )
            state_shapes[unit] = member

    def bibuffer(
        current: object,
        unit: int,
        key: str,
        gate: object,
        left_gate: object,
        tag: str,
    ) -> object:
        weight = weights[key + ".weight"]
        channels = int(weight.shape[0])
        fold = channels // 8
        packed = state_values[unit]
        if state_shapes[unit][1] != fold + channels:
            raise RuntimeError(f"BSVD state unit {unit} has an incompatible slab")
        left = builder.slice_channels(packed, 0, fold, tag + ".left")
        center = builder.slice_channels(packed, fold, channels, tag + ".center")
        right = builder.multiply(
            builder.slice_channels(current, 0, fold, tag + ".right"),
            gate,
            tag + ".right.gated",
        )
        tail = builder.slice_channels(
            center,
            2 * fold,
            channels - 2 * fold,
            tag + ".tail",
        )
        merged = builder.concat_channels([right, left, tail], tag + ".merged")
        next_left = builder.multiply(
            builder.slice_channels(center, fold, fold, tag + ".next_left"),
            left_gate,
            tag + ".next_left.gated",
        )
        state_values[unit] = builder.concat_channels([next_left, current], tag + ".state")
        result = builder.conv2d(
            merged,
            weight,
            weights[key + ".bias"],
            name=tag + ".conv",
        )
        return builder.clamp(result, 0.0, 6.0, tag + ".relu6")

    def emit_net(
        net_index: int,
        value: object,
        pops: tuple[object, ...],
        gate_vector: object,
        left_gate_vector: object,
        step: int,
    ) -> _Emitted:
        keys = keys_by_net[net_index]
        base = net_index * 8
        tag = f"s{step}.n{net_index}"
        h2, w2 = height // 2, width // 2
        h4, w4 = height // 4, width // 4

        def conv(
            current: object,
            key_name: str,
            name: str,
            *,
            stride: int = 1,
            relu6: bool = True,
        ) -> object:
            key = keys[key_name]
            result = builder.conv2d(
                current,
                weights[key + ".weight"],
                weights[key + ".bias"],
                stride=stride,
                name=f"{tag}.{name}",
            )
            if relu6:
                result = builder.clamp(result, 0.0, 6.0, f"{tag}.{name}.relu6")
            return result

        def unit_gate(unit: int, vector: object, name: str) -> object:
            return builder.slice_channels(vector, unit, 1, f"{tag}.u{unit}.{name}")

        def memory(
            current: object,
            local: int,
            key_name: str,
        ) -> object:
            unit = base + local
            return bibuffer(
                current,
                unit,
                keys[key_name],
                unit_gate(unit, gate_vector, "gate"),
                unit_gate(unit, left_gate_vector, "left_gate"),
                f"{tag}.u{unit}",
            )

        x0 = conv(value, "inc0", "inc0")
        x0 = conv(x0, "inc3", "inc3")
        x1 = conv(x0, "d0", "down0", stride=2)
        x1 = memory(x1, 0, "u0")
        x1 = memory(x1, 1, "u1")
        x2 = conv(x1, "d1", "down1", stride=2)
        x2 = memory(x2, 2, "u2")
        x2 = memory(x2, 3, "u3")
        middle = memory(x2, 4, "u4")
        middle = memory(middle, 5, "u5")

        up2_key = keys["up2"]
        up2_weight = weights[up2_key + ".weight"]
        up2 = builder.conv2d(middle, up2_weight, None, name=f"{tag}.up2")
        up2 = builder.pixel_shuffle_biased(
            up2,
            weights[up2_key + ".bias"],
            channels=int(up2_weight.shape[0]),
            height=h4,
            width=w4,
            name=f"{tag}.up2.shuffle",
        )
        merged = builder.add(up2, pops[2], f"{tag}.skip3")
        merged = memory(merged, 6, "u6")
        merged = memory(merged, 7, "u7")

        up1_key = keys["up1"]
        up1_weight = weights[up1_key + ".weight"]
        up1 = builder.conv2d(merged, up1_weight, None, name=f"{tag}.up1")
        up1 = builder.pixel_shuffle_biased(
            up1,
            weights[up1_key + ".bias"],
            channels=int(up1_weight.shape[0]),
            height=h2,
            width=w2,
            name=f"{tag}.up1.shuffle",
        )
        merged = builder.add(up1, pops[1], f"{tag}.skip2")
        prediction = conv(merged, "out0", "out0")
        prediction = conv(prediction, "out3", "out3", relu6=False)
        out_channels = int(weights[keys["out3"] + ".weight"].shape[0])
        head = builder.subtract(
            pops[0],
            builder.slice_channels(prediction, 0, 3, f"{tag}.prediction.head"),
            f"{tag}.residual",
        )
        if out_channels == 3:
            out = head
        else:
            out = builder.concat_channels(
                [
                    head,
                    builder.slice_channels(
                        prediction,
                        3,
                        out_channels - 3,
                        f"{tag}.prediction.tail",
                    ),
                ],
                f"{tag}.out",
            )
        return _Emitted(out, x0, x1, out_channels)

    targets: list[tuple[str, object, tuple[int, ...]]] = []
    output_names: list[str] = []
    push_names: list[tuple[str | None, ...]] = []

    for step in range(_STEPS):
        frame = frame_tensors[step]
        gate = gate_tensors[step]
        left_gate = left_gate_tensors[step]
        pops = pop_tensors[step]

        first = emit_net(0, frame, tuple(pops[:3]), gate, left_gate, step)
        second = emit_net(1, first.out, tuple(pops[3:]), gate, left_gate, step)
        output_name = f"out_{step}"
        targets.append(
            (
                output_name,
                second.out,
                (1, second.out_channels, height, width),
            )
        )
        output_names.append(output_name)

        pushed = {
            0: builder.slice_channels(frame, 0, 3, f"s{step}.frame.head"),
            1: first.x0,
            2: first.x1,
            3: builder.slice_channels(first.out, 0, 3, f"s{step}.n0.out3"),
            4: second.x0,
            5: second.x1,
        }
        step_push_names: list[str | None] = [None] * _LINES
        for line, _ordinary_name in _PUSH_OUTPUTS:
            name = f"skip_out_{step}_{line}"
            targets.append((name, pushed[line], line_shapes[line]))
            step_push_names[line] = name
            dynamic.add(name)
        push_names.append(tuple(step_push_names))

    state_results = {}
    for state, units in zip(state_specs, _STATE_GROUPS, strict=True):
        update = builder.concat_channels(
            [state_values[unit] for unit in units],
            state.name + ".pack",
        )
        result = mgs.state_result(builder, state, update)
        targets.append(result)
        state_results[result[0]] = state.name

    graph = mgs.Program(
        name="generic1",
        builder=builder,
        targets=targets,
        state_results=state_results,
        dynamic=dynamic,
    )
    return _Chunk(
        graph=graph,
        frame_names=tuple(frame_names),
        gate_names=tuple(gate_names),
        left_gate_names=tuple(left_gate_names),
        output_names=tuple(output_names),
        pop_names=tuple(pop_names),
        push_names=tuple(push_names),
    )


def _compile_stateful_executable(
    net: _MpsNet,
    height: int,
    width: int,
    states: tuple[mgs.StateTensorSpec, ...],
) -> _Executable:
    cache = net._executable_cache(_STATEFUL_CACHE_LABEL, height, width)
    if cache is None:
        raise RuntimeError("persistent MPSGraph windows require a versioned cache path")
    return cast(  # its entry is only handed bindings made by its own bind()
        _Executable,
        mgs.compile_stateful_direct(
            {
                "generic1": lambda: _build_chunk(net, height, width).graph,
            },
            states,
            dtype=mg.FLOAT16,
            cache_directory=cache,
            ane_fw_to_fw_signal=net._ane_fw_to_fw_signal,
            ane_late_latch=net._ane_late_latch,
        ),
    )


def preload_stateful_executable(net: _MpsNet, height: int, width: int) -> _Executable | None:
    """Open a warm direct single-program cache without building graphs."""
    cache = net._executable_cache(_STATEFUL_CACHE_LABEL, height, width)
    if cache is None or not mgs.stateful_cache_ready(cache):
        return None
    states = _state_specs(net, height, width)
    executable = _compile_stateful_executable(net, height, width, states)
    try:
        executable.prepare()
    except BaseException:
        executable.close()
        raise
    return executable


class ScheduledMpsPhaseSuite:
    """One direct entry with persistent ANE state and stable bindings."""

    def __init__(
        self,
        net: _MpsNet,
        height: int,
        width: int,
        *,
        executable: _Executable | None = None,
    ):

        self.net = net
        self.height = height
        self.width = width
        self.pipeline = net._pipeline
        self._views: dict[tuple[int, str, int], _Binding] = {}

        line_shapes = _line_shapes(net, height, width)
        self.states = _state_specs(net, height, width)
        if executable is not None and executable.state_specs != self.states:
            raise RuntimeError("preloaded MPSGraph entry has incompatible state tensors")
        self.executable = executable or _compile_stateful_executable(
            net,
            height,
            width,
            self.states,
        )
        generic = self.executable.entry("generic1")
        self.chunk = _Chunk(
            graph=generic,
            frame_names=tuple(f"frame_{step}" for step in range(_STEPS)),
            gate_names=tuple(f"gate_{step}" for step in range(_STEPS)),
            left_gate_names=tuple(f"left_gate_{step}" for step in range(_STEPS)),
            output_names=tuple(f"out_{step}" for step in range(_STEPS)),
            pop_names=tuple(
                tuple(f"skip_{step}_{line}" for line in range(_LINES)) for step in range(_STEPS)
            ),
            push_names=tuple(
                tuple(None if line == 0 else f"skip_out_{step}_{line}" for line in range(_LINES))
                for step in range(_STEPS)
            ),
        )
        net._line_shapes = list(line_shapes)
        net._state_slots = []
        net._state_bindings = {}

        net._slots = []
        net._free = []
        net._zero_bindings = []
        net._discard_bindings = []
        # Four extra slots keep every input and result of one unrolled
        # dispatch on distinct storage, including a full steady ring.
        for line, depth in enumerate(_SKIP_DEPTHS):
            name = self.chunk.pop_names[0][line]
            slots = [self.chunk.graph.bind(name) for _ in range(depth + _STEPS)]
            zero = self.chunk.graph.bind(name)
            zero.write(bytes(2 * prod(line_shapes[line])))
            net._slots.append(slots)
            net._free.append(deque(slots))
            net._zero_bindings.append(zero)
            net._discard_bindings.append(None if line == 0 else self.chunk.graph.bind(name))
        net._zero_frame = mx.zeros((1, net.input_channels, height, width), dtype=mx.float16)

    def _view(
        self,
        graph: _Entry,
        name: str,
        backing: _Binding | None,
    ) -> _Binding:
        key = (id(graph), name, id(backing))
        if key not in self._views:
            self._views[key] = graph.bind(name, shared=backing)
        return self._views[key]

    @staticmethod
    def _actions(frames: list[mx.array]) -> list[_Action]:
        mirror = NoneFlowNet()
        scheduled: list[tuple[mx.array | None, StepRecord]] = [
            (frame, mirror.step(True)) for frame in frames
        ]
        scheduled.extend((None, mirror.step(False)) for _ in range(_DRAIN_STEPS))
        while len(scheduled) % _STEPS:
            scheduled.append((None, mirror.step(False)))
        return [
            _Action(
                frames=tuple(frame for frame, _record in scheduled[index : index + _STEPS]),
                records=tuple(record for _frame, record in scheduled[index : index + _STEPS]),
            )
            for index in range(0, len(scheduled), _STEPS)
        ]

    def _resolve(self, action: _Action) -> _Resolved:
        assert self.net._zero_frame is not None  # set by __init__; only close() clears it
        frames = tuple(
            self.net._zero_frame
            if frame is None
            else mx.contiguous(mx.transpose(frame.astype(mx.float16), (0, 3, 1, 2)))
            for frame in action.frames
        )
        gates = tuple(
            mx.array(
                [0.0 if drained else 1.0 for drained in record.drained],
                dtype=mx.float16,
            ).reshape(1, 16, 1, 1)
            for record in action.records
        )
        left_gates = tuple(
            mx.array(
                [0.0 if unprimed else 1.0 for unprimed in record.unprimed],
                dtype=mx.float16,
            ).reshape(1, 16, 1, 1)
            for record in action.records
        )
        mx.eval(*frames, *gates, *left_gates)
        return _Resolved(frames, gates, left_gates, action.records)

    def _prepare(self, resolved: _Resolved) -> _Prepared:

        graph = self.chunk.graph
        values = {}
        for step in range(_STEPS):
            values[self.chunk.frame_names[step]] = resolved.frames[step]
            values[self.chunk.gate_names[step]] = resolved.gates[step]
            values[self.chunk.left_gate_names[step]] = resolved.left_gates[step]
        graph.write_feeds(values)

        bindings: dict[str, _Binding] = {}

        rings = [deque(line) for line in self.net._rings]
        free = [deque(line) for line in self.net._free]
        retired: list[list[_Binding]] = [[] for _ in range(_LINES)]
        discarded: list[list[_Binding]] = [[] for _ in range(_LINES)]
        slot: _Binding | None

        for step, record in enumerate(resolved.records):
            for line in range(_LINES):
                pop_name = self.chunk.pop_names[step][line]
                if record.pops[line]:
                    if not rings[line]:
                        raise RuntimeError(f"MPSGraph chunk skip ring {line} underran")
                    slot = rings[line].popleft()
                    retired[line].append(slot)
                else:
                    slot = self.net._zero_bindings[line]
                bindings[pop_name] = self._view(graph, pop_name, slot)

            if record.pushes[0]:
                if not free[0]:
                    raise RuntimeError("MPSGraph chunk skip ring 0 exhausted")
                slot = free[0].popleft()
                slot.write(resolved.frames[step][:, :3])
                rings[0].append(slot)

            for line in range(1, _LINES):
                if not free[line]:
                    raise RuntimeError(f"MPSGraph chunk skip ring {line} exhausted")
                slot = free[line].popleft()
                name = self.chunk.push_names[step][line]
                if name is None:
                    raise RuntimeError(f"MPSGraph chunk has no skip output {line}")
                bindings[name] = self._view(graph, name, slot)
                if record.pushes[line]:
                    rings[line].append(slot)
                else:
                    discarded[line].append(slot)

        for line in range(_LINES):
            free[line].extend(retired[line])
            free[line].extend(discarded[line])

        return _Prepared(
            graph.begin_dispatch(bindings),
            resolved.records,
            rings,
            free,
        )

    def _finish(self, prepared: _Prepared) -> list[mx.array]:
        self.net._rings = prepared.rings
        self.net._free = prepared.free
        wanted = {
            self.chunk.output_names[step]
            for step, record in enumerate(prepared.records)
            if record.out_real
        }
        outputs = self.chunk.graph.read(wanted)
        materialized = [
            mx.contiguous(
                mx.transpose(
                    outputs[self.chunk.output_names[step]],
                    (0, 2, 3, 1),
                )
            )
            for step, record in enumerate(prepared.records)
            if record.out_real
        ]
        if materialized:
            mx.eval(*materialized)
        return materialized

    def machine(self, frames: list[mx.array]) -> WindowMachine:
        return WindowMachine(self, frames)

    def reset(self) -> None:
        self.executable.reset()

    def close(self) -> None:
        self.executable.close()
        self._views.clear()


class WindowMachine:
    """Cooperatively drive one reset window, one ANE job in flight.

    ``wait_until_ready()`` joins only the native ANECIR job. The runtime
    can wait there outside the MLX owner, then return to that owner for the
    next nonblocking prepare/submit transition.
    """

    def __init__(self, suite: ScheduledMpsPhaseSuite, frames: list[mx.array]):
        if not frames:
            raise ValueError("MPSGraph window cannot be empty")
        self.outputs: list[mx.array] = []
        self._suite = suite
        self._count = len(frames)
        self._sequence = self._drive(frames)
        self._done = False
        self._failed = False

    def _drive(self, frames: list[mx.array]) -> Iterator[None]:
        suite = self._suite
        actions = suite._actions(frames)
        resolved = suite._resolve(actions[0])
        for index, _action in enumerate(actions):
            prepared = suite._prepare(resolved)
            suite.pipeline.submit(prepared.job)
            if index + 1 < len(actions):
                resolved = suite._resolve(actions[index + 1])
            yield
            self.outputs.extend(suite._finish(prepared))
        if len(self.outputs) != self._count:
            raise RuntimeError(
                f"MPSGraph window returned {len(self.outputs)} outputs for {self._count} frames"
            )

    def _advance(self, block: bool, stop_on_output: bool) -> bool:
        if self._failed:
            raise RuntimeError("MPSGraph window failed; reset the stream")
        pipeline = self._suite.pipeline
        output_count = len(self.outputs)
        try:
            while not self._done and (not stop_on_output or len(self.outputs) == output_count):
                if pipeline.in_flight:
                    if not block and not pipeline.idle():
                        return False
                    pipeline.join()
                try:
                    next(self._sequence)
                except StopIteration:
                    self._done = True
                if not block:
                    # Bound an owner turn to one native transition even when
                    # the next MPSGraph completion races ahead. This keeps the
                    # shared MLX/GPU lane fair to downstream physical bridges.
                    break
        except BaseException:
            self._failed = True
            pipeline.drain()
            raise
        return self._done

    def advance(self, block: bool = False) -> bool:
        return self._advance(block, stop_on_output=False)

    def wait_until_ready(self) -> None:
        """Join only the current native dispatch; perform no MLX work."""
        if self._failed:
            raise RuntimeError("MPSGraph window failed; reset the stream")
        pipeline = self._suite.pipeline
        if self._done or not pipeline.in_flight:
            return
        try:
            pipeline.join()
        except BaseException:
            self._failed = True
            pipeline.drain()
            raise

    def advance_until_output(self, block: bool = True) -> bool:
        return self._advance(block, stop_on_output=True)


__all__ = [
    "ScheduledMpsPhaseSuite",
    "WindowMachine",
    "preload_stateful_executable",
    "stateful_cache_ready",
]
