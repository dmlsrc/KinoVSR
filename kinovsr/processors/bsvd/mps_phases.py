"""Sparse BSVD windows compiled by MPSGraph and executed directly on ANE.

Three ANECIR entries implement an 8-11 frame sparse fill, a schedule-generic
one-frame middle, and a sixteen-step sparse drain. They share sixteen
persistent ``anec.state`` IOSurfaces, while the direct runtime retains an
entry until the semantic phase changes. Switching phases therefore
does not expose recurrent tensors to Python, MLX, or a Core ML prediction.
The static edge entries omit work that the generic graph would only gate to
zero; short windows retain the generic path until the network is fully
primed.

The six skip rings remain reusable IOSurface bindings shared across entries.
Both state and delayed features therefore cross dispatch boundaries without
host copies while the backend-neutral ``NoneFlowNet`` mirror remains the
source of truth for frame order and output visibility.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from math import prod
from typing import Any

import mlx.core as mx

from kinovsr.native import mpsgraph as mg
from kinovsr.native import mpsgraph_state as mgs

from .schedule import NoneFlowNet, StepRecord

_STEPS = 1
_DRAIN_STEPS = 16
_UNIT_KEYS = ("u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7")
_STATEFUL_CACHE_LABEL = "scheduled-stateful-direct-v2"


def stateful_cache_ready(net: Any, height: int, width: int) -> bool:
    """Whether every direct sparse-phase product is durably published."""
    cache = net._executable_cache(_STATEFUL_CACHE_LABEL, height, width)
    return cache is not None and mgs.stateful_cache_ready(cache)


@dataclass
class _Emitted:
    out: Any
    x0: Any
    x1: Any
    out_channels: int


@dataclass
class _Chunk:
    graph: Any
    frame_names: tuple[str, ...]
    gate_names: tuple[str, ...]
    left_gate_names: tuple[str, ...]
    output_names: tuple[str, ...]
    pop_names: tuple[tuple[str, ...], ...]
    push_names: tuple[tuple[str | None, ...], ...]


@dataclass
class _Action:
    frames: tuple[Any | None, ...]
    records: tuple[Any, ...]


@dataclass
class _Resolved:
    frames: tuple[Any, ...]
    gates: tuple[Any, ...]
    left_gates: tuple[Any, ...]
    records: tuple[Any, ...]


@dataclass
class _Prepared:
    job: Any
    records: tuple[Any, ...]
    rings: list[deque]
    free: list[deque]


@dataclass
class _EdgeChunk:
    graph: Any
    frame_names: tuple[str, ...]
    active_names: tuple[tuple[int, str], ...]
    skip_feeds: tuple[tuple[str, int], ...]
    ring_results: tuple[tuple[str, int, int | None], ...]
    output_names: tuple[str, ...]


@dataclass
class _PreparedFill:
    job: Any
    rings: list[deque]
    free: list[deque]
    unused: list[tuple[int, mgs.TensorBinding]]


def _storage_shapes(
    net: Any, height: int, width: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """State-slab and skip-line shapes without compiling the ordinary graph."""
    from .mps import _STATE_GROUPS, _net_keys

    keys_by_net = tuple(_net_keys(prefix) for prefix in net._prefixes)
    unit_shapes = {}
    for unit in range(16):
        local = unit % 8
        key = keys_by_net[unit // 8][_UNIT_KEYS[local]]
        channels = int(net._weights[key + ".weight"].shape[0])
        divisor = 2 if local < 2 or local > 5 else 4
        unit_shapes[unit] = (
            1,
            channels + channels // 8,
            height // divisor,
            width // divisor,
        )
    state_shapes = []
    for units in _STATE_GROUPS:
        first = unit_shapes[units[0]]
        if any(unit_shapes[unit] != first for unit in units):
            raise RuntimeError("BSVD state slab groups incompatible shapes")
        state_shapes.append((
            1, first[1] * len(units), first[2], first[3]))

    line_shapes = []
    for keys in keys_by_net:
        line_shapes.extend((
            (1, 3, height, width),
            (
                1,
                int(net._weights[
                    keys["inc3"] + ".weight"].shape[0]),
                height,
                width,
            ),
            (
                1,
                int(net._weights[
                    keys["u0"] + ".weight"].shape[0]),
                height // 2,
                width // 2,
            ),
        ))
    return tuple(state_shapes), tuple(line_shapes)


def _state_specs(
    net: Any, height: int, width: int
) -> tuple[mgs.StateTensorSpec, ...]:
    """The sixteen BiBuffer cells as independent persistent state ports."""
    from .mps import _net_keys

    keys_by_net = tuple(_net_keys(prefix) for prefix in net._prefixes)
    states = []
    for unit in range(16):
        local = unit % 8
        key = keys_by_net[unit // 8][_UNIT_KEYS[local]]
        channels = int(net._weights[key + ".weight"].shape[0])
        divisor = 2 if local < 2 or local > 5 else 4
        states.append(mgs.StateTensorSpec.create(
            f"state_{unit}",
            (
                1,
                channels + channels // 8,
                height // divisor,
                width // divisor,
            ),
        ))
    return tuple(states)


def _build_chunk(net: Any, height: int, width: int) -> _Chunk:
    """Compile the schedule-generic one-step direct MPSGraph program."""
    from .mps import (
        _LINES,
        _PUSH_OUTPUTS,
        _net_keys,
    )

    weights = net._weights
    keys_by_net = tuple(_net_keys(prefix) for prefix in net._prefixes)
    _state_slab_shapes, line_shapes = _storage_shapes(net, height, width)
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
        frame_tensors.append(builder.placeholder(
            (1, net.input_channels, height, width), frame_name))
        gate_tensors.append(builder.placeholder(
            (1, 16, 1, 1), gate_name))
        left_gate_tensors.append(builder.placeholder(
            (1, 16, 1, 1), left_gate_name))
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
    state_values = {
        unit: logical_states[state.name]
        for unit, state in enumerate(state_specs)
    }
    state_shapes = {
        unit: state.logical_shape for unit, state in enumerate(state_specs)
    }

    def bibuffer(
        current: Any,
        unit: int,
        key: str,
        gate: Any,
        left_gate: Any,
        tag: str,
    ) -> Any:
        weight = weights[key + ".weight"]
        channels = int(weight.shape[0])
        fold = channels // 8
        packed = state_values[unit]
        if state_shapes[unit][1] != fold + channels:
            raise RuntimeError(
                f"BSVD state unit {unit} has an incompatible slab")
        left = builder.slice_channels(
            packed, 0, fold, tag + ".left")
        center = builder.slice_channels(
            packed, fold, channels, tag + ".center")
        right = builder.multiply(
            builder.slice_channels(
                current, 0, fold, tag + ".right"),
            gate,
            tag + ".right.gated",
        )
        tail = builder.slice_channels(
            center,
            2 * fold,
            channels - 2 * fold,
            tag + ".tail",
        )
        merged = builder.concat_channels(
            [right, left, tail], tag + ".merged")
        next_left = builder.multiply(
            builder.slice_channels(
                center, fold, fold, tag + ".next_left"),
            left_gate,
            tag + ".next_left.gated",
        )
        state_values[unit] = builder.concat_channels(
            [next_left, current], tag + ".state")
        result = builder.conv2d(
            merged,
            weight,
            weights[key + ".bias"],
            name=tag + ".conv",
        )
        return builder.clamp(
            result, 0.0, 6.0, tag + ".relu6")

    def emit_net(
        net_index: int,
        value: Any,
        pops: tuple[Any, Any, Any],
        gate_vector: Any,
        left_gate_vector: Any,
        step: int,
    ) -> _Emitted:
        keys = keys_by_net[net_index]
        base = net_index * 8
        tag = f"s{step}.n{net_index}"
        h2, w2 = height // 2, width // 2
        h4, w4 = height // 4, width // 4

        def conv(
            current: Any,
            key_name: str,
            name: str,
            *,
            stride: int = 1,
            relu6: bool = True,
        ) -> Any:
            key = keys[key_name]
            result = builder.conv2d(
                current,
                weights[key + ".weight"],
                weights[key + ".bias"],
                stride=stride,
                name=f"{tag}.{name}",
            )
            if relu6:
                result = builder.clamp(
                    result, 0.0, 6.0, f"{tag}.{name}.relu6")
            return result

        def unit_gate(unit: int, vector: Any, name: str) -> Any:
            return builder.slice_channels(
                vector, unit, 1, f"{tag}.u{unit}.{name}")

        def memory(
            current: Any,
            local: int,
            key_name: str,
        ) -> Any:
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
        up2 = builder.conv2d(
            middle, up2_weight, None, name=f"{tag}.up2")
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
        up1 = builder.conv2d(
            merged, up1_weight, None, name=f"{tag}.up1")
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
        prediction = conv(
            prediction, "out3", "out3", relu6=False)
        out_channels = int(
            weights[keys["out3"] + ".weight"].shape[0])
        head = builder.subtract(
            pops[0],
            builder.slice_channels(
                prediction, 0, 3, f"{tag}.prediction.head"),
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

    targets: list[tuple[str, Any, tuple[int, ...]]] = []
    output_names: list[str] = []
    push_names: list[tuple[str | None, ...]] = []

    for step in range(_STEPS):
        frame = frame_tensors[step]
        gate = gate_tensors[step]
        left_gate = left_gate_tensors[step]
        pops = pop_tensors[step]

        first = emit_net(
            0, frame, tuple(pops[:3]), gate, left_gate, step)
        second = emit_net(
            1, first.out, tuple(pops[3:]), gate, left_gate, step)
        output_name = f"out_{step}"
        targets.append((
            output_name,
            second.out,
            (1, second.out_channels, height, width),
        ))
        output_names.append(output_name)

        pushed = {
            0: builder.slice_channels(
                frame, 0, 3, f"s{step}.frame.head"
            ),
            1: first.x0,
            2: first.x1,
            3: builder.slice_channels(
                first.out, 0, 3, f"s{step}.n0.out3"),
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
    for unit, state in enumerate(state_specs):
        result = mgs.state_result(builder, state, state_values[unit])
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


def _phase_records(
    *, draining: bool, steps: int
) -> tuple[tuple[Any, tuple[bool, bool, bool]], ...]:
    """Return the static NoneFlow topology for a reset-window edge."""
    mirror = NoneFlowNet()
    if draining:
        for _ in range(16):
            mirror.step(True)
    records = []
    for _ in range(steps):
        record = StepRecord()
        first = mirror.blocks[0](not draining, record)
        second = mirror.blocks[1](first, record)
        record.out_real = second
        records.append((record, (not draining, first, second)))
    return tuple(records)


def _build_edge(
    net: Any,
    height: int,
    width: int,
    *,
    draining: bool,
    ring_group_size: int = 1,
) -> _EdgeChunk:
    """Build the sparse 8-11-frame fill or complete sixteen-step drain."""
    from .mps import _SKIP_DEPTHS, _net_keys

    if ring_group_size < 1:
        raise ValueError("MPSGraph edge ring group must be positive")
    steps = 16 if draining else 11
    variable_fill = not draining
    weights = net._weights
    keys_by_net = tuple(_net_keys(prefix) for prefix in net._prefixes)
    _state_slab_shapes, line_shapes = _storage_shapes(net, height, width)
    state_specs = _state_specs(net, height, width)
    states_by_name = {state.name: state for state in state_specs}
    records = _phase_records(draining=draining, steps=steps)
    builder = mg.GraphBuilder(mg.FLOAT16)

    skip_feeds: list[tuple[str, int]] = []
    skip_feed_values: dict[str, Any] = {}
    frame_names: list[str] = []
    frame_values: dict[int, Any] = {}
    active_names: list[tuple[int, str]] = []
    active_tensors: dict[int, Any] = {}

    if draining:
        # Drain reads only the still-resident head of each skip ring. Declare
        # those ordinary inputs in their first-use order before state ports.
        for step, (record, block_flow) in enumerate(records):
            for net_index in range(2):
                base = net_index * 8
                skip_base = net_index * 3
                lines = []
                if not record.drained[base + 6]:
                    lines.append(skip_base + 2)
                if block_flow[net_index + 1]:
                    lines.extend((skip_base + 1, skip_base))
                for line in lines:
                    depth = _SKIP_DEPTHS[line]
                    if step >= depth:
                        continue
                    group = step // ring_group_size
                    name = f"skip_{line}_{group}"
                    if name in skip_feed_values:
                        continue
                    count = min(
                        ring_group_size, depth - group * ring_group_size)
                    channels = line_shapes[line][1]
                    shape = (
                        1,
                        channels * count,
                        line_shapes[line][2],
                        line_shapes[line][3],
                    )
                    skip_feed_values[name] = builder.placeholder(shape, name)
                    skip_feeds.append((name, line))
    else:
        for step in range(steps):
            frame_name = f"frame_{step}"
            frame_names.append(frame_name)
            frame_values[step] = builder.placeholder(
                (1, net.input_channels, height, width), frame_name)
            if step >= 8:
                active_name = f"active_{step}"
                active_tensors[step] = builder.placeholder(
                    (1, 1, 1, 1), active_name)
                active_names.append((step, active_name))

    # State ports trail every ordinary/dynamic port. Large MPSGraph mapped
    # products use this ABI when sharing persistent ANE state across entries.
    logical_states = mgs.state_placeholders(builder, state_specs)
    state_values = {
        unit: logical_states[state.name]
        for unit, state in enumerate(state_specs)
    }
    zero_cache: dict[tuple[int, ...], Any] = {}

    def zero(shape: tuple[int, ...]) -> Any:
        shape = tuple(int(value) for value in shape)
        if shape not in zero_cache:
            zero_cache[shape] = builder.constant(
                mx.zeros(shape, dtype=mx.float16),
                cache_key=("edge.zero", *shape),
            )
        return zero_cache[shape]

    def blend(old: Any, new: Any, active: Any, name: str) -> Any:
        delta = builder.subtract(new, old, name + ".delta")
        delta = builder.multiply(delta, active, name + ".active")
        return builder.add(old, delta, name)

    pushed_history: list[dict[int, Any]] = []
    real_pushes: list[dict[int, Any]] = []
    edge_outputs: list[tuple[str, Any, tuple[int, ...]]] = []
    queue_values: dict[int, list[Any]] = {1: [], 2: []}

    def skip_pop(step: int, line: int) -> Any:
        depth = _SKIP_DEPTHS[line]
        if step >= depth:
            return pushed_history[step - depth][line]
        if draining:
            group = step // ring_group_size
            slot = step % ring_group_size
            name = f"skip_{line}_{group}"
            if name not in skip_feed_values:
                raise AssertionError(
                    f"MPSGraph drain skip feed {name} was not declared")
            return builder.slice_channels(
                skip_feed_values[name],
                slot * line_shapes[line][1],
                line_shapes[line][1],
                f"{name}.{step}",
            )
        raise AssertionError(
            "MPSGraph fill unexpectedly required a cold skip-ring feed")

    def conv(
        value: Any,
        net_index: int,
        key_name: str,
        tag: str,
        *,
        stride: int = 1,
        relu6: bool = True,
    ) -> Any:
        key = keys_by_net[net_index][key_name]
        result = builder.conv2d(
            value,
            weights[key + ".weight"],
            weights[key + ".bias"],
            stride=stride,
            name=tag,
        )
        if relu6:
            result = builder.clamp(result, 0.0, 6.0, tag + ".relu6")
        return result

    def upsample(
        value: Any,
        net_index: int,
        key_name: str,
        tag: str,
        out_height: int,
        out_width: int,
    ) -> Any:
        key = keys_by_net[net_index][key_name]
        weight = weights[key + ".weight"]
        result = builder.conv2d(value, weight, None, name=tag)
        return builder.pixel_shuffle_biased(
            result,
            weights[key + ".bias"],
            channels=int(weight.shape[0]),
            height=out_height,
            width=out_width,
            name=tag + ".shuffle",
        )

    for step, (record, block_flow) in enumerate(records):
        before_state = dict(state_values)
        value = None if draining else frame_values[step]
        if (value is not None) != block_flow[0]:
            raise AssertionError("MPSGraph edge input topology drifted")
        step_outputs: dict[int, Any] = {}

        for net_index in range(2):
            base = net_index * 8
            keys = keys_by_net[net_index]
            if (value is not None) != block_flow[net_index]:
                raise AssertionError("MPSGraph edge block topology drifted")

            def bibuffer(
                current: Any | None,
                local: int,
                key_name: str,
                *,
                base: int = base,
                net_index: int = net_index,
                record: Any = record,
                step: int = step,
            ) -> Any | None:
                unit = base + local
                key = keys_by_net[net_index][key_name]
                channels = int(weights[key + ".weight"].shape[0])
                fold = channels // 8
                divisor = 2 if local < 2 or local > 5 else 4
                current_shape = (
                    1, channels, height // divisor, width // divisor)
                left_shape = (
                    1, fold, height // divisor, width // divisor)
                right_real = not record.drained[unit]
                center_real = not record.unprimed[unit]
                if (current is not None) != right_real:
                    raise AssertionError(
                        "MPSGraph edge BiBuffer topology drifted")
                if not center_real:
                    if current is not None:
                        state_values[unit] = builder.concat_channels(
                            [zero(left_shape), current],
                            f"s{step}.u{unit}.prime",
                        )
                    return None
                packed_state = state_values[unit]
                left = builder.slice_channels(
                    packed_state, 0, fold, f"s{step}.u{unit}.left")
                center = builder.slice_channels(
                    packed_state, fold, channels, f"s{step}.u{unit}.center")
                if current is None:
                    right = zero(left_shape)
                    next_center = zero(current_shape)
                else:
                    right = builder.slice_channels(
                        current, 0, fold, f"s{step}.u{unit}.right")
                    next_center = current
                tail = builder.slice_channels(
                    center,
                    2 * fold,
                    channels - 2 * fold,
                    f"s{step}.u{unit}.tail",
                )
                merged = builder.concat_channels(
                    [right, left, tail], f"s{step}.u{unit}.merged")
                next_left = builder.slice_channels(
                    center, fold, fold, f"s{step}.u{unit}.next_left")
                state_values[unit] = builder.concat_channels(
                    [next_left, next_center], f"s{step}.u{unit}.state")
                result = builder.conv2d(
                    merged,
                    weights[key + ".weight"],
                    weights[key + ".bias"],
                    name=f"s{step}.u{unit}.conv",
                )
                return builder.clamp(
                    result, 0.0, 6.0, f"s{step}.u{unit}.relu6")

            skip_base = net_index * 3
            if value is not None:
                step_outputs[skip_base] = builder.slice_channels(
                    value, 0, 3, f"s{step}.n{net_index}.skip1")
                x0 = conv(
                    value, net_index, "inc0", f"s{step}.n{net_index}.inc0")
                x0 = conv(
                    x0, net_index, "inc3", f"s{step}.n{net_index}.inc3")
                step_outputs[skip_base + 1] = x0
                x1 = conv(
                    x0,
                    net_index,
                    "d0",
                    f"s{step}.n{net_index}.down0",
                    stride=2,
                )
            else:
                x1 = None

            x1 = bibuffer(x1, 0, "u0")
            x1 = bibuffer(x1, 1, "u1")
            if record.pushes[skip_base + 2]:
                step_outputs[skip_base + 2] = x1
            x2 = (
                conv(
                    x1,
                    net_index,
                    "d1",
                    f"s{step}.n{net_index}.down1",
                    stride=2,
                )
                if x1 is not None else None
            )
            x2 = bibuffer(x2, 2, "u2")
            x2 = bibuffer(x2, 3, "u3")
            x2 = bibuffer(x2, 4, "u4")
            x2 = bibuffer(x2, 5, "u5")
            if x2 is not None:
                x2 = upsample(
                    x2,
                    net_index,
                    "up2",
                    f"s{step}.n{net_index}.up2",
                    height // 4,
                    width // 4,
                )

            if not record.drained[base + 6]:
                if x2 is None:
                    raise AssertionError("MPSGraph edge skip3 topology drifted")
                merged = builder.add(
                    x2,
                    skip_pop(step, skip_base + 2),
                    f"s{step}.n{net_index}.skip3",
                )
            else:
                merged = None
            merged = bibuffer(merged, 6, "u6")
            merged = bibuffer(merged, 7, "u7")
            if merged is not None:
                merged = upsample(
                    merged,
                    net_index,
                    "up1",
                    f"s{step}.n{net_index}.up1",
                    height // 2,
                    width // 2,
                )

            if block_flow[net_index + 1]:
                if merged is None:
                    raise AssertionError("MPSGraph edge output topology drifted")
                merged = builder.add(
                    merged,
                    skip_pop(step, skip_base + 1),
                    f"s{step}.n{net_index}.skip2",
                )
                prediction = conv(
                    conv(
                        merged,
                        net_index,
                        "out0",
                        f"s{step}.n{net_index}.out0",
                    ),
                    net_index,
                    "out3",
                    f"s{step}.n{net_index}.out3",
                    relu6=False,
                )
                out_channels = int(
                    weights[keys["out3"] + ".weight"].shape[0])
                head = builder.subtract(
                    skip_pop(step, skip_base),
                    builder.slice_channels(
                        prediction,
                        0,
                        3,
                        f"s{step}.n{net_index}.prediction.head",
                    ),
                    f"s{step}.n{net_index}.residual",
                )
                if out_channels == 3:
                    value = head
                else:
                    value = builder.concat_channels(
                        [
                            head,
                            builder.slice_channels(
                                prediction,
                                3,
                                out_channels - 3,
                                f"s{step}.n{net_index}.prediction.tail",
                            ),
                        ],
                        f"s{step}.n{net_index}.out",
                    )
            else:
                value = None

        expected_pushes = {
            line for line, pushed in enumerate(record.pushes) if pushed}
        if set(step_outputs) != expected_pushes:
            raise AssertionError(
                f"MPSGraph edge skip topology drifted at step {step}")

        if variable_fill and step >= 8:
            active = active_tensors[step]
            for unit in range(16):
                old = before_state[unit]
                new = state_values[unit]
                state_values[unit] = blend(
                    old, new, active, f"s{step}.u{unit}.commit")

        if not draining:
            for line in (1, 2):
                old_queue = queue_values[line]
                candidate = list(old_queue)
                if record.pops[line]:
                    if not candidate:
                        raise AssertionError(
                            f"MPSGraph edge queue {line} underrun")
                    candidate.pop(0)
                if record.pushes[line]:
                    candidate.append(step_outputs[line])
                if step >= 8:
                    if len(candidate) != len(old_queue):
                        raise AssertionError(
                            f"MPSGraph optional queue {line} changed depth")
                    active = active_tensors[step]
                    candidate = [
                        blend(
                            old,
                            new,
                            active,
                            f"s{step}.ring{line}.{slot}",
                        )
                        for slot, (old, new) in enumerate(
                            zip(old_queue, candidate, strict=True))
                    ]
                queue_values[line] = candidate

        if record.out_real:
            edge_outputs.append((
                f"out_{step}", value, (1, 3, height, width)))
        real_pushes.append(dict(step_outputs))
        for line, shape in enumerate(line_shapes):
            step_outputs.setdefault(line, zero(shape))
        pushed_history.append(step_outputs)

    targets: list[tuple[str, Any, tuple[int, ...]]] = []
    state_results = {}
    ring_results: list[tuple[str, int, int | None]] = []
    output_names: list[str] = []
    if draining:
        targets.extend(edge_outputs)
        output_names.extend(name for name, _tensor, _shape in edge_outputs)
    else:
        for unit, state in enumerate(state_specs):
            result = mgs.state_result(builder, state, state_values[unit])
            targets.append(result)
            state_results[result[0]] = state.name

        for line in (1, 2):
            if len(queue_values[line]) != _SKIP_DEPTHS[line]:
                raise AssertionError(
                    f"MPSGraph fill queue {line} has wrong final depth")
            for slot, tensor in enumerate(queue_values[line]):
                name = f"fill.ring{line}.{slot}"
                targets.append((name, tensor, line_shapes[line]))
                ring_results.append((name, line, None))

        for step in range(8, 11):
            for line in (3, 4, 5):
                tensor = real_pushes[step].get(line)
                if tensor is None:
                    continue
                name = f"fill.step{step}.ring{line}"
                targets.append((name, tensor, line_shapes[line]))
                ring_results.append((name, line, step))

    dynamic = {name for name, _line in skip_feeds}
    dynamic.update(name for name, _line, _step in ring_results)
    graph = mgs.Program(
        name="drain16" if draining else "fill8_11",
        builder=builder,
        targets=targets,
        state_results=state_results,
        dynamic=dynamic,
    )
    # Every graph must contract the same state set even when a phase only
    # reads it. The native compiler validates these feeds before lowering.
    if set(states_by_name) != {
        name for _tensor, _shape, name in builder.feeds
        if name in states_by_name
    }:
        raise AssertionError("MPSGraph edge state feeds changed")
    return _EdgeChunk(
        graph=graph,
        frame_names=tuple(frame_names),
        active_names=tuple(active_names),
        skip_feeds=tuple(skip_feeds),
        ring_results=tuple(ring_results),
        output_names=tuple(output_names),
    )


def _compile_stateful_executable(
    net: Any,
    height: int,
    width: int,
    states: tuple[mgs.StateTensorSpec, ...],
) -> Any:
    cache = net._executable_cache(_STATEFUL_CACHE_LABEL, height, width)
    if cache is None:
        raise RuntimeError(
            "persistent MPSGraph phases require a versioned cache path")
    return mgs.compile_stateful_direct(
        {
            "fill8_11": lambda: _build_edge(
                net, height, width, draining=False).graph,
            "generic1": lambda: _build_chunk(
                net, height, width).graph,
            "drain16": lambda: _build_edge(
                net, height, width, draining=True).graph,
        },
        states,
        dtype=mg.FLOAT16,
        cache_directory=cache,
        ane_fw_to_fw_signal=net._ane_fw_to_fw_signal,
        ane_late_latch=net._ane_late_latch,
    )


def preload_stateful_executable(
    net: Any, height: int, width: int
) -> Any | None:
    """Open a warm direct phase cache without rebuilding source graphs."""
    cache = net._executable_cache(_STATEFUL_CACHE_LABEL, height, width)
    if cache is None or not mgs.stateful_cache_ready(cache):
        return None
    states = _state_specs(net, height, width)
    executable = _compile_stateful_executable(
        net, height, width, states)
    try:
        executable.prepare()
    except BaseException:
        executable.close()
        raise
    return executable


class ScheduledMpsPhaseSuite:
    """Sparse fill/steady/drain entries sharing one persistent ANE state."""

    def __init__(
        self,
        net: Any,
        height: int,
        width: int,
        *,
        executable: Any | None = None,
    ):
        from .mps import _LINES, _SKIP_DEPTHS

        self.net = net
        self.height = height
        self.width = width
        self.pipeline = net._pipeline
        self._views: dict[tuple[int, str, int], mgs.TensorBinding] = {}

        _state_shapes, line_shapes = _storage_shapes(net, height, width)
        self.states = _state_specs(net, height, width)
        if executable is not None and executable.state_specs != self.states:
            raise RuntimeError(
                "preloaded MPSGraph phases have incompatible state tensors")
        self.executable = executable or _compile_stateful_executable(
            net, height, width, self.states,
        )
        generic = self.executable.entry("generic1")
        self.chunk = _Chunk(
            graph=generic,
            frame_names=tuple(f"frame_{step}" for step in range(_STEPS)),
            gate_names=tuple(f"gate_{step}" for step in range(_STEPS)),
            left_gate_names=tuple(
                f"left_gate_{step}" for step in range(_STEPS)),
            output_names=tuple(f"out_{step}" for step in range(_STEPS)),
            pop_names=tuple(
                tuple(f"skip_{step}_{line}" for line in range(_LINES))
                for step in range(_STEPS)),
            push_names=tuple(
                tuple(
                    None if line == 0 else f"skip_out_{step}_{line}"
                    for line in range(_LINES))
                for step in range(_STEPS)),
        )
        fill_entry = self.executable.entry("fill8_11")
        fill_ring_results = []
        for name in fill_entry.target_names:
            match = re.fullmatch(r"fill\.ring([0-9]+)\.([0-9]+)", name)
            optional = re.fullmatch(
                r"fill\.step([0-9]+)\.ring([0-9]+)", name)
            if match:
                fill_ring_results.append((name, int(match.group(1)), None))
            elif optional:
                fill_ring_results.append((
                    name, int(optional.group(2)), int(optional.group(1))))
        self.fill = _EdgeChunk(
            graph=fill_entry,
            frame_names=tuple(f"frame_{step}" for step in range(11)),
            active_names=tuple(
                (step, f"active_{step}") for step in range(8, 11)),
            skip_feeds=(),
            ring_results=tuple(fill_ring_results),
            output_names=(),
        )
        drain_entry = self.executable.entry("drain16")
        drain_feeds = []
        for name in drain_entry.feed_names:
            match = re.fullmatch(r"skip_([0-9]+)_([0-9]+)", name)
            if match:
                drain_feeds.append((name, int(match.group(1))))
        self.drain = _EdgeChunk(
            graph=drain_entry,
            frame_names=(),
            active_names=(),
            skip_feeds=tuple(drain_feeds),
            ring_results=(),
            output_names=tuple(
                name for name in drain_entry.target_names
                if name.startswith("out_")),
        )

        net._line_shapes = list(line_shapes)
        net._state_slots = []
        net._state_bindings = {}

        net._slots = []
        net._free = []
        net._zero_bindings = []
        net._discard_bindings = []
        # One extra slot keeps every input and result of one in-flight
        # dispatch on distinct storage, including a full steady ring.
        for line, depth in enumerate(_SKIP_DEPTHS):
            name = self.chunk.pop_names[0][line]
            slots = [
                self.chunk.graph.bind(name)
                for _ in range(depth + _STEPS)
            ]
            zero = self.chunk.graph.bind(name)
            zero.write(bytes(2 * prod(line_shapes[line])))
            net._slots.append(slots)
            net._free.append(deque(slots))
            net._zero_bindings.append(zero)
            net._discard_bindings.append(
                None if line == 0 else self.chunk.graph.bind(name))
        net._zero_frame = mx.zeros(
            (1, net.input_channels, height, width), dtype=mx.float16)

    def _view(
        self,
        graph: mgs.StatefulEntry,
        name: str,
        backing: mgs.TensorBinding,
    ) -> mgs.TensorBinding:
        key = (id(graph), name, id(backing))
        if key not in self._views:
            self._views[key] = graph.bind(
                name, shared=backing)
        return self._views[key]

    @staticmethod
    def _actions(frames: list[Any]) -> list[_Action]:
        mirror = NoneFlowNet()
        scheduled: list[tuple[Any | None, Any]] = [
            (frame, mirror.step(True)) for frame in frames
        ]
        for _ in range(_DRAIN_STEPS):
            scheduled.append((None, mirror.step(False)))
        while len(scheduled) % _STEPS:
            scheduled.append((None, mirror.step(False)))
        return [
            _Action(
                frames=tuple(
                    frame for frame, _record in scheduled[index:index + _STEPS]
                ),
                records=tuple(
                    record for _frame, record in scheduled[index:index + _STEPS]
                ),
            )
            for index in range(0, len(scheduled), _STEPS)
        ]

    @staticmethod
    def _middle_actions(
        frames: list[Any], fill_length: int
    ) -> list[_Action]:
        """All-real steps between sparse fill and sparse drain."""
        mirror = NoneFlowNet()
        for _ in range(fill_length):
            mirror.step(True)
        scheduled = [
            (frame, mirror.step(True)) for frame in frames[fill_length:]
        ]
        if len(scheduled) % _STEPS:
            raise AssertionError(
                "MPSGraph sparse fill did not align the generic middle")
        return [
            _Action(
                frames=tuple(
                    frame for frame, _record
                    in scheduled[index:index + _STEPS]),
                records=tuple(
                    record for _frame, record
                    in scheduled[index:index + _STEPS]),
            )
            for index in range(0, len(scheduled), _STEPS)
        ]

    def _prepare_fill(
        self, frames: list[Any], fill_length: int
    ) -> _PreparedFill:
        from .mps import _LINES

        graph = self.fill.graph
        resolved = [
            mx.contiguous(mx.transpose(
                frame.astype(mx.float16), (0, 3, 1, 2)))
            for frame in frames[:fill_length]
        ]
        resolved.extend(
            self.net._zero_frame
            for _ in range(len(self.fill.frame_names) - fill_length)
        )
        values = {
            name: resolved[step]
            for step, name in enumerate(self.fill.frame_names)
        }
        for step, name in self.fill.active_names:
            values[name] = (
                self.net._gate_on
                if step < fill_length else self.net._gate_off)
        mx.eval(*values.values())
        graph.write_feeds(values)

        bindings: dict[str, mgs.TensorBinding] = {}
        rings = [deque() for _ in range(_LINES)]
        free = [deque(line) for line in self.net._free]
        unused: list[tuple[int, mgs.TensorBinding]] = []

        for frame in resolved[fill_length - 8:fill_length]:
            if not free[0]:
                raise RuntimeError("MPSGraph fill ring 0 exhausted")
            slot = free[0].popleft()
            slot.write(frame[:, :3])
            rings[0].append(slot)

        for name, line, optional_step in self.fill.ring_results:
            if not free[line]:
                raise RuntimeError(
                    f"MPSGraph fill ring {line} exhausted")
            slot = free[line].popleft()
            bindings[name] = self._view(graph, name, slot)
            if optional_step is None or optional_step < fill_length:
                rings[line].append(slot)
            else:
                unused.append((line, slot))
        return _PreparedFill(
            graph.begin_dispatch(bindings), rings, free, unused)

    def _finish_fill(self, prepared: _PreparedFill) -> None:
        for line, slot in prepared.unused:
            prepared.free[line].append(slot)
        self.net._rings = prepared.rings
        self.net._free = prepared.free

    def _prepare_drain(self) -> Any:
        graph = self.drain.graph
        bindings: dict[str, mgs.TensorBinding] = {}
        rings = [deque(line) for line in self.net._rings]
        for name, line in self.drain.skip_feeds:
            if not rings[line]:
                raise RuntimeError(
                    f"MPSGraph drain skip ring {line} underran")
            bindings[name] = self._view(
                graph, name, rings[line].popleft())
        return graph.begin_dispatch(bindings)

    def _finish_drain(self) -> list[Any]:
        outputs = self.drain.graph.read(set(self.drain.output_names))
        materialized = [
            mx.contiguous(mx.transpose(outputs[name], (0, 2, 3, 1)))
            for name in self.drain.output_names
        ]
        if materialized:
            mx.eval(*materialized)
        return materialized

    def _resolve(self, action: _Action) -> _Resolved:
        frames = tuple(
            self.net._zero_frame
            if frame is None
            else mx.contiguous(mx.transpose(
                frame.astype(mx.float16), (0, 3, 1, 2)))
            for frame in action.frames
        )
        gates = tuple(
            mx.array(
                [0.0 if drained else 1.0
                 for drained in record.drained],
                dtype=mx.float16,
            ).reshape(1, 16, 1, 1)
            for record in action.records
        )
        left_gates = tuple(
            mx.array(
                [0.0 if unprimed else 1.0
                 for unprimed in record.unprimed],
                dtype=mx.float16,
            ).reshape(1, 16, 1, 1)
            for record in action.records
        )
        mx.eval(*frames, *gates, *left_gates)
        return _Resolved(
            frames, gates, left_gates, action.records)

    def _prepare(self, resolved: _Resolved) -> _Prepared:
        from .mps import _LINES

        graph = self.chunk.graph
        values = {}
        for step in range(_STEPS):
            values[self.chunk.frame_names[step]] = resolved.frames[step]
            values[self.chunk.gate_names[step]] = resolved.gates[step]
            values[self.chunk.left_gate_names[step]] = (
                resolved.left_gates[step])
        graph.write_feeds(values)

        bindings: dict[str, mgs.TensorBinding] = {}

        rings = [deque(line) for line in self.net._rings]
        free = [deque(line) for line in self.net._free]
        retired = [[] for _ in range(_LINES)]
        discarded = [[] for _ in range(_LINES)]

        for step, record in enumerate(resolved.records):
            for line in range(_LINES):
                pop_name = self.chunk.pop_names[step][line]
                if record.pops[line]:
                    if not rings[line]:
                        raise RuntimeError(
                            f"MPSGraph chunk skip ring {line} underran")
                    slot = rings[line].popleft()
                    retired[line].append(slot)
                else:
                    slot = self.net._zero_bindings[line]
                bindings[pop_name] = self._view(
                    graph, pop_name, slot)

            if record.pushes[0]:
                if not free[0]:
                    raise RuntimeError(
                        "MPSGraph chunk skip ring 0 exhausted")
                slot = free[0].popleft()
                slot.write(resolved.frames[step][:, :3])
                rings[0].append(slot)

            for line in range(1, _LINES):
                if not free[line]:
                    raise RuntimeError(
                        f"MPSGraph chunk skip ring {line} exhausted")
                slot = free[line].popleft()
                name = self.chunk.push_names[step][line]
                if name is None:
                    raise RuntimeError(
                        f"MPSGraph chunk has no skip output {line}")
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

    def _finish(self, prepared: _Prepared) -> list[Any]:
        self.net._rings = prepared.rings
        self.net._free = prepared.free
        wanted = {
            self.chunk.output_names[step]
            for step, record in enumerate(prepared.records)
            if record.out_real
        }
        outputs = self.chunk.graph.read(wanted)
        materialized = [
            mx.contiguous(mx.transpose(
                outputs[self.chunk.output_names[step]],
                (0, 2, 3, 1),
            ))
            for step, record in enumerate(prepared.records)
            if record.out_real
        ]
        if materialized:
            mx.eval(*materialized)
        return materialized

    def machine(self, frames: list[Any]) -> WindowMachine:
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

    def __init__(self, suite: ScheduledMpsPhaseSuite, frames: list[Any]):
        if not frames:
            raise ValueError("MPSGraph window cannot be empty")
        self.outputs: list[Any] = []
        self._suite = suite
        self._count = len(frames)
        self._sequence = self._drive(frames)
        self._done = False
        self._failed = False

    def _drive(self, frames: list[Any]):
        suite = self._suite
        if len(frames) < 16:
            actions = suite._actions(frames)
            resolved = suite._resolve(actions[0])
            for index, _action in enumerate(actions):
                prepared = suite._prepare(resolved)
                suite.pipeline.submit(prepared.job)
                if index + 1 < len(actions):
                    resolved = suite._resolve(actions[index + 1])
                yield
                self.outputs.extend(suite._finish(prepared))
        else:
            fill_length = 8 + len(frames) % _STEPS
            fill = suite._prepare_fill(frames, fill_length)
            suite.pipeline.submit(fill.job)
            actions = suite._middle_actions(frames, fill_length)
            resolved = suite._resolve(actions[0]) if actions else None
            yield
            suite._finish_fill(fill)

            for index, _action in enumerate(actions):
                if resolved is None:
                    raise AssertionError("MPSGraph middle action was not resolved")
                prepared = suite._prepare(resolved)
                suite.pipeline.submit(prepared.job)
                if index + 1 < len(actions):
                    resolved = suite._resolve(actions[index + 1])
                yield
                self.outputs.extend(suite._finish(prepared))

            suite.pipeline.submit(suite._prepare_drain())
            yield
            self.outputs.extend(suite._finish_drain())
        if len(self.outputs) != self._count:
            raise RuntimeError(
                f"MPSGraph window returned {len(self.outputs)} outputs "
                f"for {self._count} frames")

    def _advance(self, block: bool, stop_on_output: bool) -> bool:
        if self._failed:
            raise RuntimeError("MPSGraph window failed; reset the stream")
        pipeline = self._suite.pipeline
        output_count = len(self.outputs)
        try:
            while (
                not self._done
                and (
                    not stop_on_output
                    or len(self.outputs) == output_count
                )
            ):
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
