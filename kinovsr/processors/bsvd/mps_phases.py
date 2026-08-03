"""Four-step BSVD windows executed directly by MPSGraph on ANE.

The ordinary graph pays MPSGraph's explicit recurrent-boundary overhead
once per frame. This window path carries four temporal steps inside one
executable, reducing that toll without changing runtimes: it is MPSGraph
from graph construction through ANE dispatch, with no Core ML package,
model, state, or prediction involved.

One schedule-generic executable handles fill, steady state, and drain.
Per-step gates reproduce the backend-neutral ``NoneFlowNet`` schedule,
while the graph carries recurrent state internally across the four steps.
Keeping one unrolled ANE program resident is important: separate programs
for every static fill/drain phase measured well in isolation but thrashed
ANE program residency in full GOP-aligned video runs.

The four recurrent state slabs and six skip rings reuse the ordinary
MPSGraph graph's shared MTLBuffers. A chunk therefore hands state to the
next chunk without copying it through Python or MLX.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import prod
from typing import Any

import mlx.core as mx

from kinovsr.native import mpsgraph as mg

from .schedule import NoneFlowNet

_STEPS = 4
_DRAIN_STEPS = 16
_UNIT_KEYS = ("u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7")


@dataclass
class _Emitted:
    out: Any
    x0: Any
    x1: Any
    out_channels: int


@dataclass
class _Chunk:
    graph: mg.CompiledGraph
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


def _build_chunk(net: Any, height: int, width: int) -> _Chunk:
    """Compile the schedule-generic four-step direct MPSGraph unroll."""
    from .mps import (
        _LINES,
        _PUSH_OUTPUTS,
        _STATE_GROUPS,
        _net_keys,
    )

    weights = net._weights
    keys_by_net = tuple(_net_keys(prefix) for prefix in net._prefixes)
    state_slab_shapes, line_shapes = _storage_shapes(net, height, width)
    builder = mg.GraphBuilder(mg.FLOAT16)

    state_values: dict[int, Any] = {}
    state_shapes: dict[int, tuple[int, ...]] = {}
    state_feed_names: list[str] = []
    for group, units in enumerate(_STATE_GROUPS):
        shape = state_slab_shapes[group]
        name = f"g{group}.state"
        slab = builder.placeholder(shape, name)
        state_feed_names.append(name)
        channels = shape[1] // len(units)
        unit_shape = (1, channels, shape[2], shape[3])
        for slot, unit in enumerate(units):
            state_values[unit] = builder.slice_channels(
                slab,
                slot * channels,
                channels,
                f"g{group}.u{unit}.initial",
            )
            state_shapes[unit] = unit_shape

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
    dynamic: set[str] = set(state_feed_names)
    frame_names: list[str] = []
    gate_names: list[str] = []
    left_gate_names: list[str] = []
    output_names: list[str] = []
    pop_names: list[tuple[str, ...]] = []
    push_names: list[tuple[str | None, ...]] = []

    for step in range(_STEPS):
        frame_name = f"frame_{step}"
        frame = builder.placeholder(
            (1, net.input_channels, height, width), frame_name)
        frame_names.append(frame_name)
        gate_name = f"gate_{step}"
        left_gate_name = f"left_gate_{step}"
        gate = builder.placeholder(
            (1, 16, 1, 1), gate_name)
        left_gate = builder.placeholder(
            (1, 16, 1, 1), left_gate_name)
        gate_names.append(gate_name)
        left_gate_names.append(left_gate_name)

        step_pop_names = []
        pops = []
        for line in range(_LINES):
            name = f"skip_{step}_{line}"
            pops.append(builder.placeholder(
                line_shapes[line], name))
            step_pop_names.append(name)
            dynamic.add(name)
        pop_names.append(tuple(step_pop_names))

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

    for group, units in enumerate(_STATE_GROUPS):
        shape = state_slab_shapes[group]
        name = f"g{group}.state.next"
        targets.append((
            name,
            builder.concat_channels(
                [state_values[unit] for unit in units],
                f"g{group}.state.final",
            ),
            shape,
        ))
        dynamic.add(name)

    label = "scheduled-generic4"
    graph = mg.compile_graph(
        builder,
        targets,
        device=mg.DEVICE_ANE,
        dynamic=dynamic,
        synchronize_results=False,
        use_command_queue=net._use_command_queue,
        ane_fw_to_fw_signal=net._ane_fw_to_fw_signal,
        ane_late_latch=net._ane_late_latch,
        ane_streaming_session=net._ane_streaming_session,
        ane_energy_efficient=net._ane_energy_efficient,
        executable_cache=net._executable_cache(label, height, width),
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


class ScheduledMpsPhaseSuite:
    """One schedule-generic direct-MPSGraph executable for reset windows."""

    def __init__(self, net: Any, height: int, width: int):
        from .mps import _SKIP_DEPTHS

        self.net = net
        self.height = height
        self.width = width
        self.pipeline = net._pipeline
        self._views: dict[tuple[int, str, int], mg.TensorBinding] = {}

        state_shapes, line_shapes = _storage_shapes(net, height, width)
        self.chunk = _build_chunk(net, height, width)

        # Scheduled mode owns its storage through the generic executable.
        # It must not compile the ordinary one-step program just to obtain
        # same-shaped MTLBuffers: a fourth resident ANE program is outside
        # the measured-stable residency envelope at production geometry.
        net._line_shapes = list(line_shapes)
        net._state_slots = []
        net._state_bindings = {}
        for group, shape in enumerate(state_shapes):
            name = f"g{group}.state"
            slot = self.chunk.graph.bind(name)
            slot.write(bytes(2 * prod(shape)))
            net._state_slots.append(slot)

        net._slots = []
        net._free = []
        net._zero_bindings = []
        net._discard_bindings = []
        # Four extra slots keep every input and result of one unrolled
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
        graph: mg.CompiledGraph,
        name: str,
        backing: mg.TensorBinding,
    ) -> mg.TensorBinding:
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
        from .mps import _LINES, _STATE_GROUPS

        graph = self.chunk.graph
        values = {}
        for step in range(_STEPS):
            values[self.chunk.frame_names[step]] = resolved.frames[step]
            values[self.chunk.gate_names[step]] = resolved.gates[step]
            values[self.chunk.left_gate_names[step]] = (
                resolved.left_gates[step])
        graph.write_feeds(values)

        bindings: dict[str, mg.TensorBinding] = {}
        for group in range(len(_STATE_GROUPS)):
            slot = self.net._state_slots[group]
            for name in (
                f"g{group}.state",
                f"g{group}.state.next",
            ):
                bindings[name] = self._view(graph, name, slot)

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
        self.chunk.graph.reset()

    def close(self) -> None:
        self.chunk.graph.close()
        self._views.clear()


class WindowMachine:
    """Cooperatively drive one reset window, one four-step job in flight.

    ``wait_until_ready()`` joins only the native MPSGraph job. The runtime
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


__all__ = ["ScheduledMpsPhaseSuite", "WindowMachine"]
