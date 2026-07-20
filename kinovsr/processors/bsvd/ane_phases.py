"""Phase-specialized BSVD graphs for independently reset schedule windows.

The ordinary ANE graph deliberately computes every convolution on every
step and uses gates to reproduce BSVD's ``None`` fill/drain semantics.  That
is appropriate for a long continuous stream, but GOP-aligned windows spend
32 of roughly 46 steps filling or draining.  This module unrolls those phases
eight at a time and omits operations whose product value is ``None``.

Four functions (fill 0-7, fill 8-15, drain 0-7, drain 8-15) share one Core
ML asset and one weight blob.  They also share the shipping graph's MLState
handle and host skip rings, so the steady middle of a window can return to the
ordinary one-step runner without copying state.  The emitted-frame replay
gate below requires that handoff to remain exact.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import mlx.core as mx

from kinovsr.native.anemil import builder, runtime

from . import ane as base

_log = logging.getLogger("kinovsr.bsvd_ane")

PHASE_STEPS = 8
PHASE_GRAPH_VERSION = 4
PHASE_REPLAY_FRAMES = 20
CANARY_STEPS = 2
CANARY_DIVERGENCE_FLOOR = 4e-5
CANARY_REPLAY_TOLERANCE = 2e-5
_CANARY_NAME = "state_canary"
_FUNCTIONS = {
    ("fill", 0): "fill_00",
    ("fill", 8): "fill_08",
    ("drain", 0): "drain_00",
    ("drain", 8): "drain_08",
}
_VERIFIED: set[str] = set()


def _phase_records(start: int, draining: bool) -> list[tuple[Any, list[bool]]]:
    mirror = base._NoneFlowNet()
    # Sixteen real steps fully prime the boolean topology.  Additional
    # steady steps do not change it; drain to the requested phase afterward.
    for _ in range(16 if draining else start):
        mirror.step(True)
    if draining:
        for _ in range(start):
            mirror.step(False)
    records = []
    for _ in range(PHASE_STEPS):
        record = base._StepRecord()
        first = mirror.blocks[0](not draining, record)
        second = mirror.blocks[1](first, record)
        record.out_real = second
        records.append((record, [not draining, first, second]))
    return records


def _emit_chunk(params: dict, input_channels: int, height: int, width: int,
                start: int, draining: bool, blob: builder.BlobFile):
    state_shapes, skip_shapes = base._shapes(
        params, input_channels, height, width)
    graph = builder.Graph(blob)
    inputs = []
    for step in range(PHASE_STEPS):
        inputs.append(
            (f"frame_{step}", (1, input_channels, height, width)))
        # Once an eight-step chunk wraps a depth-four ring, the later
        # consumer reads the producer value directly inside the graph.
        # Do not register the later wrapped slots as external inputs.
        inputs.extend(
            (f"skip_{step}_{index}", shape)
            for index, shape in enumerate(skip_shapes)
            if step < base.SKIP_DEPTH[index % 3])
    states = [(f"st{index}", shape)
              for index, shape in enumerate(state_shapes)]
    for name, shape in inputs + states:
        graph.register_input(name, shape)

    initial_reads = [graph.read_state(name) for name, _shape in states]
    state_values = list(initial_reads)
    zero_scalar = graph.fp16_const(
        "phase_zero_scalar", mx.zeros((1, 1, 1, 1), dtype=mx.float16))
    zero_cache: dict[tuple[int, ...], str] = {}

    def zero_source(shape) -> str:
        key = tuple(int(value) for value in shape)
        source = zero_cache.get(key)
        if source is None:
            source = graph.fp16_const(
                f"phase_zero_{len(zero_cache)}",
                mx.zeros(key, dtype=mx.float16))
            zero_cache[key] = source
        return source

    def zero(shape, name: str) -> str:
        source = zero_source(shape)
        return graph.binary("mul", source, zero_scalar, "zero_alias",
                            name=name)

    def conv(x: str, block: str, key: str, *, name: str | None = None,
             relu6: bool = True) -> str:
        weight, bias, stride = params[block][key]
        return graph.conv2d(
            x, base._oihw(weight),
            None if bias is None else bias.astype(mx.float32),
            tag=f"{block}_{key}", stride=int(stride), relu6=relu6,
            relu6_name=name)

    def upsample(x: str, block: str, key: str) -> str:
        weight, bias, _stride = params[block][key]
        folded = base._fold_weights(
            base._oihw(weight), bias.astype(mx.float32))
        divisor = 4 if key == "u2" else 2
        rows, columns = height // divisor, width // divisor
        ones = graph.fp16_const(
            graph.n(f"{block}_{key}_ones"),
            mx.ones((1, 1, rows, columns), dtype=mx.float16))
        extended = graph.concat_channels(
            [x, ones], tag=f"{block}_{key}_ext")
        return graph.conv_transpose2d(
            extended, folded, tag=f"{block}_{key}_up")

    output_names: list[str] = []
    pushed_history: list[dict[int, str]] = []
    for step, (record, block_flow) in enumerate(
            _phase_records(start, draining)):
        next_states = list(state_values)
        x = None if draining else f"frame_{step}"
        if (x is not None) != block_flow[0]:
            raise AssertionError("phase input topology drifted")
        step_outputs: dict[int, str] = {}

        def skip_pop(index: int, *, step=step) -> str:
            depth = base.SKIP_DEPTH[index % 3]
            if step < depth:
                return f"skip_{step}_{index}"
            return pushed_history[step - depth][index]

        for block_index, block in enumerate(base.BLOCKS):
            state_base = block_index * 8
            if (x is not None) != block_flow[block_index]:
                raise AssertionError("phase block topology drifted")

            def bibuffer(value: str | None, local_index: int, key: str,
                         *, output_name: str | None = None,
                         state_base=state_base, record=record,
                         state_values=state_values, block=block, step=step,
                         next_states=next_states) -> str | None:
                index = state_base + local_index
                right_real = not record.drained[index]
                center_real = not record.unprimed[index]
                if (value is not None) != right_real:
                    raise AssertionError("phase BiBuffer topology drifted")
                read = state_values[index]
                channels = int(params[block][key][0].shape[3])
                fold = channels // 8
                divisor = (2 if local_index < 2 or local_index > 5 else 4)
                if not center_real:
                    if value is not None:
                        carry = zero(
                            (1, fold, height // divisor, width // divisor),
                            graph.n(f"s{step}_{block}_{key}_zero_carry"))
                        next_states[index] = graph.concat_channels(
                            [value, carry],
                            tag=f"s{step}_{block}_{key}_prime")
                    return None

                center = graph.slice_channels(
                    read, 0, channels, f"s{step}_{block}_{key}_center")
                left = graph.slice_channels(
                    read, channels, fold, f"s{step}_{block}_{key}_left")
                rows, columns = graph.shape[center][2:]
                if value is None:
                    right = zero(
                        (1, fold, rows, columns),
                        graph.n(f"s{step}_{block}_{key}_zero_right"))
                    new_center = zero(
                        (1, channels, rows, columns),
                        graph.n(f"s{step}_{block}_{key}_zero_center"))
                else:
                    right = graph.slice_channels(
                        value, 0, fold, f"s{step}_{block}_{key}_right")
                    new_center = value
                tail = graph.slice_channels(
                    center, 2 * fold, channels - 2 * fold,
                    f"s{step}_{block}_{key}_tail")
                packed = graph.concat_channels(
                    [right, left, tail], tag=f"s{step}_{block}_{key}_in")
                result = conv(packed, block, key, name=output_name)
                carry = graph.slice_channels(
                    center, fold, fold, f"s{step}_{block}_{key}_carry")
                next_states[index] = graph.concat_channels(
                    [new_center, carry],
                    tag=f"s{step}_{block}_{key}_state")
                return result

            skip_base = block_index * 3
            if x is not None:
                skip1_name = f"skip_out_{step}_{skip_base}"
                step_outputs[skip_base] = graph.slice_channels(
                    x, 0, 3, f"s{step}_{block}_skip1",
                    name=skip1_name)
                x0 = conv(x, block, "inc0")
                skip2_name = f"skip_out_{step}_{skip_base + 1}"
                x0 = conv(x0, block, "inc3", name=skip2_name)
                step_outputs[skip_base + 1] = x0
                x1 = conv(x0, block, "d0")
            else:
                x1 = None

            push3_name = (f"skip_out_{step}_{skip_base + 2}"
                          if record.pushes[skip_base + 2] else None)
            x1 = bibuffer(x1, 0, "d0c1")
            x1 = bibuffer(x1, 1, "d0c2", output_name=push3_name)
            if push3_name is not None:
                step_outputs[skip_base + 2] = x1

            x2 = conv(x1, block, "d1") if x1 is not None else None
            x2 = bibuffer(x2, 2, "d1c1")
            x2 = bibuffer(x2, 3, "d1c2")
            x2 = bibuffer(x2, 4, "u2c1")
            x2 = bibuffer(x2, 5, "u2c2")
            x2 = upsample(x2, block, "u2") if x2 is not None else None

            if not record.drained[state_base + 6]:
                if x2 is None:
                    raise AssertionError("phase skip3 topology drifted")
                merged = graph.binary(
                    "add", x2, skip_pop(skip_base + 2),
                    f"s{step}_{block}_skip3_add")
            else:
                merged = None
            merged = bibuffer(merged, 6, "u1c1")
            merged = bibuffer(merged, 7, "u1c2")
            x1o = upsample(merged, block, "u1") \
                if merged is not None else None

            if block_flow[block_index + 1]:
                if x1o is None:
                    raise AssertionError("phase output topology drifted")
                y = graph.binary(
                    "add", x1o, skip_pop(skip_base + 1),
                    f"s{step}_{block}_skip2_add")
                prediction = conv(
                    conv(y, block, "out0"), block, "out3", relu6=False)
                out_channels = int(params[block]["out3"][0].shape[0])
                final = block_index == len(base.BLOCKS) - 1
                out_name = f"out_{step}" if final else None
                if out_channels == 3:
                    x = graph.binary(
                        "sub", skip_pop(skip_base), prediction,
                        f"s{step}_{block}_minus", name=out_name)
                else:
                    head3 = graph.slice_channels(
                        prediction, 0, 3, f"s{step}_{block}_head3")
                    head = graph.binary(
                        "sub", skip_pop(skip_base), head3,
                        f"s{step}_{block}_minus")
                    rest = graph.slice_channels(
                        prediction, 3, out_channels - 3,
                        f"s{step}_{block}_rest")
                    x = graph.concat_channels(
                        [head, rest], tag=f"s{step}_{block}_next",
                        name=out_name)
            else:
                x = None

        state_values = next_states
        pushed = {index for index, value in enumerate(record.pushes) if value}
        if set(step_outputs) != pushed:
            raise AssertionError("phase skip-push topology drifted")
        if record.out_real:
            output_names.append(f"out_{step}")
        output_names.extend(
            f"skip_out_{step}_{index}" for index in sorted(pushed))
        # Missing pushes still participate in an in-graph depth-four pop.
        # Keep a shared zero value for that dataflow, but do not export or
        # physically write a dead output backing.
        for index, shape in enumerate(skip_shapes):
            if index not in step_outputs:
                step_outputs[index] = zero_source(shape)
        pushed_history.append(step_outputs)

    for index, value in enumerate(state_values):
        if value == initial_reads[index]:
            value = graph.binary(
                "mul", initial_reads[index], zero_scalar,
                f"phase_clear_st{index}")
        graph.update_state(f"st{index}", initial_reads[index], value)
    return graph, inputs, states, output_names


def _emit_program(params: dict, input_channels: int, height: int, width: int):
    blob = builder.BlobFile()
    functions = [("main", *base._emit_graph(
        params, input_channels, height, width, blob=blob))]
    for (kind, start), name in _FUNCTIONS.items():
        graph, inputs, states, outputs = _emit_chunk(
            params, input_channels, height, width, start,
            draining=kind == "drain", blob=blob)
        functions.append((name, graph, inputs, states, outputs))
    functions.append((_CANARY_NAME, *_emit_canary(
        params, input_channels, height, width, blob)))
    model_bytes = builder.finish_functions(
        functions, "KinoVSR BSVD ANE scheduled phases",
        default_function="main")
    return model_bytes, blob


def _emit_canary(params: dict, input_channels: int, height: int, width: int,
                 blob: builder.BlobFile):
    """Cheap state recurrence whose CPU and ANE results visibly diverge."""
    state_shapes, _skip_shapes = base._shapes(
        params, input_channels, height, width)
    graph = builder.Graph(blob)
    inputs = [("frame", (1, input_channels, height, width))]
    states = [(f"st{index}", shape)
              for index, shape in enumerate(state_shapes)]
    for name, shape in inputs + states:
        graph.register_input(name, shape)
    reads = [graph.read_state(name) for name, _shape in states]
    graph.slice_channels(
        reads[0], 0, 3, "canary_previous_state", name="out")

    def conv(x: str, key: str) -> str:
        weight, bias, stride = params["temp1"][key]
        return graph.conv2d(
            x, base._oihw(weight),
            None if bias is None else bias.astype(mx.float32),
            tag=f"canary_{key}", stride=int(stride), relu6=True)

    x = conv(conv(conv("frame", "inc0"), "inc3"), "d0")
    channels = int(params["temp1"]["d0c1"][0].shape[3])
    fold = channels // 8
    zero_scalar = graph.fp16_const(
        "canary_zero", mx.zeros((1, 1, 1, 1), dtype=mx.float16))
    carry = graph.slice_channels(
        reads[0], channels, fold, "canary_carry")
    carry = graph.binary("mul", carry, zero_scalar, "canary_zero_carry")
    state0 = graph.concat_channels([x, carry], tag="canary_state0")
    graph.update_state("st0", reads[0], state0)
    for index in range(1, len(states)):
        cleared = graph.binary(
            "mul", reads[index], zero_scalar, f"canary_clear_st{index}")
        graph.update_state(f"st{index}", reads[index], cleared)
    return graph, inputs, states, ["out"]


def _convert(params: dict, input_channels: int, height: int, width: int,
             directory: Path) -> Path:
    package = directory / f"scheduled8-v{PHASE_GRAPH_VERSION}.mlpackage"
    if package.is_dir():
        return package
    model_bytes, blob = _emit_program(
        params, input_channels, height, width)
    staging = package.with_name(f"{package.stem}.partial.mlpackage")
    shutil.rmtree(staging, ignore_errors=True)
    builder.write_package(staging, model_bytes, blob)
    staging.replace(package)
    return package


class ScheduledPhaseSuite:
    """Four phase functions sharing one MLState and the base skip rings."""

    def __init__(self, compiled: Path):
        # Load the default function first so every other function can attach
        # its model to the same MLState at construction.  Loading all functions
        # concurrently and replacing their states afterward transiently
        # allocated one complete BSVD state (~91 MiB at CIF) per function.
        self.runner = base.BsvdRunner(
            compiled, "ane", function_name="main")
        shared_state = self.runner.model._state

        def load(item):
            key, name = item
            return key, runtime.ModelRunner(
                compiled, "ane", dynamic=("skip_",),
                function_name=name, state=shared_state)

        load_items = [*_FUNCTIONS.items(), (_CANARY_NAME, _CANARY_NAME)]
        with ThreadPoolExecutor(max_workers=len(load_items)) as pool:
            self.models = dict(pool.map(load, load_items))
        first = self.models[("fill", 0)]
        self._spares = [
            [runtime.bind_array(first.dynamic_inputs[f"skip_0_{index}"])
             for _ in range(PHASE_STEPS)]
            for index in range(6)
        ]
        self.pipeline = runtime.DispatchPipeline("bsvd-ane-window")

    def set_state(self, state: Any) -> None:
        for model in self.models.values():
            model._state = state

    def dispatch(self, kind: str, start: int, frames: list[memoryview]) -> None:
        if len(frames) != PHASE_STEPS:
            raise ValueError(f"expected {PHASE_STEPS} phase frames")
        model = self.models[(kind, start)]
        for step, frame in enumerate(frames):
            model.input_view(f"frame_{step}")[:] = frame
        pushes = [record.pushes for record, _flow in _phase_records(
            start, draining=kind == "drain")]
        features, backings, selected = {}, {}, []
        for index, ring in enumerate(self.runner._rings):
            slots = []
            for step in range(PHASE_STEPS):
                slot = (self.runner._cursor[index] + step) % len(ring)
                pushed = pushes[step][index]
                slots.append((slot, pushed))
                input_name = f"skip_{step}_{index}"
                if input_name in model.dynamic_inputs:
                    features[input_name] = self.runner._input_multi(
                        index, slot)
                if pushed:
                    backings[f"skip_out_{step}_{index}"] = \
                        self._spares[index][step][2]
            selected.append(slots)
        model.predict_with(features, backings)
        for index, slots in enumerate(selected):
            ring = self.runner._rings[index]
            for step, (slot, pushed) in enumerate(slots):
                if pushed:
                    ring[slot], self._spares[index][step] = (
                        self._spares[index][step], ring[slot])
                self.runner._valid[index][slot] = pushed
            self.runner._cursor[index] = (
                self.runner._cursor[index] + PHASE_STEPS) % len(ring)

    def machine(self, frames: list, zero_frame: memoryview,
                snapshot: Callable[[Any], Any]) -> WindowMachine:
        """A cooperative driver for one window; see :class:`WindowMachine`."""
        return WindowMachine(self, frames, zero_frame, snapshot)

    def run(self, frames: list, zero_frame: memoryview,
            snapshot: Callable[[Any], Any]) -> list[Any]:
        """Run one window to completion (the synchronous convenience)."""
        driver = self.machine(frames, zero_frame, snapshot)
        driver.advance(block=True)
        return driver.outputs

    def close(self) -> None:
        self.pipeline.close()
        self.models.clear()
        self._spares.clear()


class WindowMachine:
    """Drives one reset-window through the phase suite, one dispatch in
    flight, advanced cooperatively from the CALLER's thread.

    This is the family-side implementation of the async window protocol
    (see ``kinovsr.processors.feed_driver.WindowWavefront``):
    ``advance(block=False)`` makes whatever progress it can without
    waiting - joining finished dispatches, materializing their outputs,
    prepping and submitting the next - and returns True once the window
    is complete and ``outputs`` holds one entry per input frame.
    ``advance(block=True)`` runs to completion. Every MLX operation
    (input prep, output materialization) happens inside ``advance`` on
    the caller's thread; only the Core ML dispatches run on the suite's
    pipeline worker.

    ``frames`` entries are input views or zero-arg callables producing
    them, so the host-side prep resolves in the shadow of an in-flight
    dispatch instead of serially before it.
    """

    def __init__(self, suite: ScheduledPhaseSuite, frames: list,
                 zero_frame: memoryview, snapshot: Callable[[Any], Any]):
        if len(frames) < 16:
            raise ValueError("phase specialization needs at least 16 frames")
        self.outputs: list[Any] = []
        self._suite = suite
        self._sequence = self._drive(frames, zero_frame, snapshot)
        self._done = False
        self._failed = False

    def _drive(self, frames: list, zero_frame: memoryview,
               snapshot: Callable[[Any], Any]):
        # A generator: each `yield` marks a dispatch in flight; `advance`
        # joins the pipeline before resuming, so everything between two
        # yields runs with the previous dispatch's results settled.
        suite = self._suite
        pipeline = suite.pipeline

        def resolve(index: int):
            entry = frames[index]
            return entry() if callable(entry) else entry

        suite.set_state(suite.runner.model._state)
        first = [resolve(index) for index in range(8)]
        pipeline.submit(lambda: suite.dispatch("fill", 0, first))
        second = [resolve(index) for index in range(8, 16)]
        yield
        # Fill steps 8-15 run as GATED SINGLES on the main function, NOT
        # the fill_08 phase function: re-executing that function fails
        # with ANE status=0x16 on its SECOND run per load at 640x480
        # (probed 2026-07-20; deterministic single-threaded, timing-
        # dependent otherwise - the cold CLI failed at window ONE because
        # the build replay had spent the one good execution). The bug is
        # specific to that function at that scale: fill_00, both drains,
        # the larger drain_00, and CIF-sized fill_08 all re-execute fine.
        # These eight steps only omitted 28 percent of compute anyway, so
        # the gated singles - the continuously verified path - cost about
        # 1.6 percent of a window and remove the trigger entirely.
        count = len(frames)
        pending = None
        fill_tail = _phase_records(8, draining=False)
        for step in range(8):
            record = fill_tail[step][0]
            write = _vector_bytes(
                [0.0 if record.primes[i] else 1.0 for i in range(16)])
            suite.runner.load_inputs(second[step], None, write)
            pipeline.submit(suite.runner.dispatch)
            if step == 0 and count > 16:
                pending = resolve(16)
            yield
        for index in range(16, count):
            view, pending = pending, None
            suite.runner.load_inputs(view)
            pipeline.submit(suite.runner.dispatch)
            if index + 1 < count:
                pending = resolve(index + 1)
            yield
            self.outputs.append(
                snapshot(suite.runner.model.output_array("out")))
        zeros = [zero_frame] * PHASE_STEPS
        pipeline.submit(lambda: suite.dispatch("drain", 0, zeros))
        yield
        pipeline.submit(lambda: suite.dispatch("drain", 8, zeros))
        drained = suite.models[("drain", 0)]
        self.outputs.extend(snapshot(drained.output_array(f"out_{step}"))
                            for step in range(PHASE_STEPS))
        yield
        drained = suite.models[("drain", 8)]
        self.outputs.extend(snapshot(drained.output_array(f"out_{step}"))
                            for step in range(PHASE_STEPS))

    def advance(self, block: bool = False) -> bool:
        """Progress the window; True once complete.

        Non-blocking calls only consume dispatches that have already
        finished; a blocking call runs the window to completion.
        """
        if self._failed:
            raise RuntimeError("window failed; reset the stream")
        pipeline = self._suite.pipeline
        try:
            while not self._done:
                if pipeline.in_flight:
                    if not block and not pipeline.idle():
                        return False
                    pipeline.join()
                try:
                    next(self._sequence)
                except StopIteration:
                    self._done = True
        except BaseException:
            self._failed = True
            pipeline.drain()
            raise
        return True


def _snapshot(value: Any) -> Any:
    copied = value.astype(mx.float32)
    mx.eval(copied)
    return copied


def _vector_bytes(values: list[float]) -> memoryview:
    vector = mx.array(values, dtype=mx.float16).reshape(1, 16, 1, 1)
    mx.eval(vector)
    return memoryview(mx.contiguous(vector)).cast("B")


def _drive_reference(runner: base.BsvdRunner, frames: list[Any]) -> list[Any]:
    mirror = base._NoneFlowNet()
    records = [mirror.step(True) for _ in frames]
    records.extend(mirror.step(False) for _ in range(16))
    zero_frame = bytes(len(runner.model.input_view("frame")))
    outputs = []
    for index, record in enumerate(records):
        frame = (memoryview(frames[index]).cast("B")
                 if index < len(frames) else zero_frame)
        gates = [0.0 if drained else 1.0 for drained in record.drained]
        writes = [0.0 if prime else 1.0 for prime in record.primes]
        result = runner.step(
            frame, _vector_bytes(gates), _vector_bytes(writes))
        for line, pushed in enumerate(record.pushes):
            if not pushed:
                runner.zero_last_push(line)
        if record.out_real:
            outputs.append(_snapshot(result))
    return outputs


def _replay_suite(suite: ScheduledPhaseSuite, input_channels: int,
                  height: int, width: int) -> list[Any]:
    frames = base._replay_frames(
        input_channels, height, width, PHASE_REPLAY_FRAMES)
    suite.runner.reset()
    outputs = suite.run(
        [memoryview(frame).cast("B") for frame in frames],
        memoryview(bytes(len(suite.runner.model.input_view("frame")))),
        _snapshot)
    suite.runner.reset()
    suite.set_state(suite.runner.model._state)
    return outputs


def _drive_canary_model(model: runtime.ModelRunner, input_channels: int,
                        height: int, width: int) -> list[Any]:
    model.reset_state()
    outputs = []
    for frame in base._replay_frames(
            input_channels, height, width, CANARY_STEPS):
        model.input_view("frame")[:] = memoryview(frame).cast("B")
        model.predict()
        outputs.append(_snapshot(model.output_array("out")))
    return outputs


def _drive_canary(compiled: Path, compute_units: str, input_channels: int,
                  height: int, width: int) -> list[Any]:
    model = runtime.ModelRunner(
        compiled, compute_units, function_name=_CANARY_NAME)
    return _drive_canary_model(model, input_channels, height, width)


def _mean_abs(left: list[Any], right: list[Any]) -> float:
    deltas = [mx.abs(a - b).mean()
              for a, b in zip(left, right, strict=True)]
    return float(mx.mean(mx.stack(deltas)))


def _verify_canary_load(suite: ScheduledPhaseSuite, directory: Path,
                        input_channels: int, height: int, width: int) -> None:
    record = json.loads((
        directory / f"scheduled8-v{PHASE_GRAPH_VERSION}-verify.json"
    ).read_text())
    if record.get("graph_version") != PHASE_GRAPH_VERSION:
        raise RuntimeError("scheduled phase cache has a different graph version")
    expected = mx.load(str(
        directory / f"scheduled8-v{PHASE_GRAPH_VERSION}-canary.safetensors"))
    actual = _drive_canary_model(
        suite.models[_CANARY_NAME], input_channels, height, width)
    reference = [expected[f"out_{index}"]
                 for index in range(len(actual))]
    drift = _mean_abs(actual, reference)
    tolerance = float(record.get(
        "canary_replay_tolerance", CANARY_REPLAY_TOLERANCE))
    if drift > tolerance:
        raise RuntimeError(
            f"scheduled phase ANE canary drift {drift:.3e} exceeds "
            f"{tolerance:.0e}; refusing a possible CPU "
            f"fallback")
    suite.runner.reset()
    suite.set_state(suite.runner.model._state)


def _verify_build(suite: ScheduledPhaseSuite, compiled: Path,
                  directory: Path, input_channels: int,
                  height: int, width: int) -> None:
    # Placement is gated on the DEFAULT function only. Loading a NAMED
    # function's MLComputePlan re-runs its own E5RT specialization (~4 s
    # per function on the serial compiler service, not shared with the
    # model loads - measured 2026-07-20: six named plans cost 24.3 s,
    # HALF the cold build, against 0.6 s for the default plan). The phase
    # functions are gated by something strictly stronger below: the EXACT
    # replay requires each of them to match this plan-gated main function
    # bit for bit, a realized-execution oracle no CPU-placed MLState graph
    # can satisfy, and the canary keeps its CPU-divergence check.
    placements = {"main": runtime.assert_all_ane(compiled)}
    frames = base._replay_frames(
        input_channels, height, width, PHASE_REPLAY_FRAMES)
    suite.runner.reset()
    expected = _drive_reference(suite.runner, frames)
    suite.runner.reset()
    actual = _replay_suite(suite, input_channels, height, width)
    maximum = max(float(mx.abs(left - right).max())
                  for left, right in zip(expected, actual, strict=True))
    if maximum != 0.0:
        raise RuntimeError(
            f"scheduled phase graph differs from the full graph (max abs "
            f"{maximum:.3e})")
    canary_ane = _drive_canary(
        compiled, "ane", input_channels, height, width)
    canary_cpu = _drive_canary(
        compiled, "cpu", input_channels, height, width)
    separation = _mean_abs(canary_ane, canary_cpu)
    if separation < CANARY_DIVERGENCE_FLOOR:
        raise RuntimeError(
            f"scheduled phase canary separates ANE and CPU by only "
            f"{separation:.3e}; refusing an inconclusive runtime oracle")
    mx.save_safetensors(
        str(directory / f"scheduled8-v{PHASE_GRAPH_VERSION}-replay"),
        {f"out_{index}": value for index, value in enumerate(actual)})
    mx.save_safetensors(
        str(directory / f"scheduled8-v{PHASE_GRAPH_VERSION}-canary"),
        {f"out_{index}": value
         for index, value in enumerate(canary_ane)})
    (directory / f"scheduled8-v{PHASE_GRAPH_VERSION}-verify.json").write_text(
        json.dumps({
            "graph_version": PHASE_GRAPH_VERSION,
            "phase_steps": PHASE_STEPS,
            "replay_frames": PHASE_REPLAY_FRAMES,
            "full_graph_max_abs": maximum,
            "canary_steps": CANARY_STEPS,
            "canary_separation": separation,
            "canary_divergence_floor": CANARY_DIVERGENCE_FLOOR,
            "canary_replay_tolerance": CANARY_REPLAY_TOLERANCE,
            "placements": placements,
        }, indent=2))


def build_suite(params: dict, input_channels: int, height: int, width: int,
                directory: Path) -> ScheduledPhaseSuite:
    """Build/load, place, replay-gate, and return the scheduled phase suite."""
    verify = directory / f"scheduled8-v{PHASE_GRAPH_VERSION}-verify.json"
    replay = directory / f"scheduled8-v{PHASE_GRAPH_VERSION}-replay.safetensors"
    canary = directory / f"scheduled8-v{PHASE_GRAPH_VERSION}-canary.safetensors"
    cold = not (verify.is_file() and replay.is_file() and canary.is_file())
    if cold:
        # The dominant cold cost is ANE specialization of the six
        # functions on the serial compiler service; without progress
        # lines the build reads as hung.
        _log.info("building BSVD ANE scheduled phases for %dx%d (one-time "
                  "per geometry, cached under %s)", width, height, directory)
    start = time.perf_counter()
    package = _convert(params, input_channels, height, width, directory)
    compiled = runtime.compile_package(package)
    if cold:
        _log.info("scheduled phases emitted and compiled in %.1fs; "
                  "specializing %d functions for the ANE (the slow part)",
                  time.perf_counter() - start, len(_FUNCTIONS) + 2)
    loaded = time.perf_counter()
    suite = ScheduledPhaseSuite(compiled)
    key = str(package)
    if cold:
        _log.info("scheduled phase functions loaded in %.1fs; verifying",
                  time.perf_counter() - loaded)
        gate = time.perf_counter()
        _verify_build(
            suite, compiled, directory, input_channels, height, width)
        _VERIFIED.add(key)
        _log.info("scheduled phases verified in %.1fs (total %.1fs)",
                  time.perf_counter() - gate, time.perf_counter() - start)
    elif key not in _VERIFIED:
        _verify_canary_load(
            suite, directory, input_channels, height, width)
        _VERIFIED.add(key)
    return suite


__all__ = ["PHASE_STEPS", "ScheduledPhaseSuite", "build_suite"]
