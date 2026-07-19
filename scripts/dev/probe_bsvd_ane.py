"""Build, validate and benchmark the fastest correct BSVD Core ML/ANE runner.

This is the consolidated result of a long investigation into why BSVD on the
Neural Engine was both slow and inaccurate. Three findings are baked in:

1. **The native MIL `pixel_shuffle` is wrong on the ANE when its input is a
   convolution result.** Measured error/std up to 1.09 at BSVD's u2 shape,
   where the same convolution alone is 9.63e-4 and an isolated `pixel_shuffle`
   is bit-exact. Inserting `relu`, `add` or `reshape` between them does not
   help, and the CPU is unaffected. That single operation pair accounted for
   the whole of BSVD's reported 22-35x numerical regression on ANE.

2. **Each `conv -> pixel_shuffle` pair folds exactly into one stride-two
   transposed convolution.** This avoids the broken operation rather than
   working around it, and is faster than the correct-but-slower
   reshape/transpose/reshape spelling. The fold is gated on a float64 proof
   before anything is converted.

3. **`MLState` works on the ANE at production scale.** Sixteen BiBuffer states
   convert, are reported ANE-preferred for every operation, and compute
   identically to explicit state I/O, while keeping 531 MB per step off the
   boundary. Note "ANE-preferred" is what `MLComputePlan` reports, which is a
   placement preference and not a realized runtime trace.

Correctness is gated before any timing, because a stateful graph that merely
loads is worth nothing: the fold is proven in float64, and the recurrence is
checked on CPU and ANE separately, since a state encoding can be right on one
compute unit and silently shifted on the other.

Two known constraints:

* The Core ML **CPU** path is about an order of magnitude worse than the ANE
  on this graph (about 3.0e-3 over 64 steps against 2.5e-4), so a deployment
  must not silently fall back to CPU. This script does not isolate the cause;
  a state-update defect is the likely candidate but CPU fp16 behavior
  generally is not ruled out here.
* **Small geometries fail at runtime.** 32x32 converts and reports every
  operation ANE-preferred, then the first prediction fails with
  `ANEProgramProcessRequestDirect() ... status=0x1d`. 96x128 and 480x640 are
  fine. Whatever the threshold is, do not validate this graph at a toy size
  and assume the result carries.

Run with the project venv, e.g. `"$KINOVSR_PYTHON" scripts/dev/probe_bsvd_ane.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from kinovsr.processors import bsvd as B

_log = logging.getLogger("kinovsr.dev.probe_bsvd_ane")

BLOCKS = ("temp1", "temp2")
# The eight BiBufferConvs of a DenBlock, in execution order, with the
# resolution divisor each runs at.
BIBUF = (("d0c1", 2), ("d0c2", 2), ("d1c1", 4), ("d1c2", 4),
         ("u2c1", 4), ("u2c2", 4), ("u1c1", 2), ("u1c2", 2))
# Push-to-pop latency of each DenBlock skip line, confirmed against the
# product's own dynamic queues.
SKIP_DEPTH = {"skip1": 8, "skip2": 8, "skip3": 4}
_MLMULTIARRAY_FLOAT16 = 0x10000 | 16


def scratch_root() -> Path:
    base = os.environ.get("SHARED_TEMP_DIR") or os.environ.get("TMPDIR") or "/tmp"
    return Path(base) / "kinovsr_bsvd_ane"


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------

def load_torch_parameters(variant: str):
    """Product BSVD weights as torch tensors in Core ML's NCHW convention."""
    params, input_channels = B.load_bsvd(
        B.default_weights_path(variant), dtype=mx.float32)
    converted = {}
    for block, table in params.items():
        entries = {}
        for key, (weight, bias, stride) in table.items():
            # Product weights are MLX-native NHWC (O, kH, kW, I).
            oihw = np.asarray(mx.transpose(weight, (0, 3, 1, 2)))
            entries[key] = (torch.from_numpy(oihw.copy()),
                            torch.from_numpy(np.asarray(bias).copy()),
                            int(stride))
        converted[block] = entries
    return converted, int(input_channels)


def state_shapes(params, input_channels: int, height: int, width: int):
    """Packed BiBuffer state shapes, then the three skip shapes per block."""
    states = []
    for block in BLOCKS:
        for key, divisor in BIBUF:
            channels = int(params[block][key][0].shape[1])
            states.append((1, channels + channels // 8,
                           height // divisor, width // divisor))
    skips = []
    for block in BLOCKS:
        skips.append((1, 3, height, width))
        skips.append((1, int(params[block]["inc3"][0].shape[0]), height, width))
        skips.append((1, int(params[block]["d0c1"][0].shape[1]),
                      height // 2, width // 2))
    return states, skips


# --------------------------------------------------------------------------
# The exact upsample fold
# --------------------------------------------------------------------------

def fold_weights(weight: torch.Tensor, bias: torch.Tensor, r: int = 2) -> torch.Tensor:
    """[Cout*r*r, Cin, k, k] + bias -> [Cin+1, Cout, k*r, k*r].

    Sub-pixel convolution and transposed convolution are equivalent. The
    appended input channel is fed a constant one and carries the per-phase
    bias, which a per-channel transposed-convolution bias cannot express.
    """
    c4, cin, k, _ = weight.shape
    cout = c4 // (r * r)
    folded = torch.zeros(cin + 1, cout, k * r, k * r, dtype=torch.float32)
    for out_c in range(cout):
        for i in range(r):
            for j in range(r):
                sub = weight[out_c * r * r + i * r + j]
                for a in range(k):
                    for b in range(k):
                        folded[:cin, out_c,
                               r * (k - 1 - a) + i,
                               r * (k - 1 - b) + j] = sub[:, a, b]
                folded[cin, out_c, 2 + i, 2 + j] = bias[out_c * r * r + i * r + j]
    return folded


def audit_fold(params, height: int, width: int) -> float:
    """Prove every upsample fold in float64 before converting anything."""
    worst = 0.0
    geometry = {"u2": (height // 4, width // 4), "u1": (height // 2, width // 2)}
    for block in BLOCKS:
        for key, (rows, columns) in geometry.items():
            weight, bias, _ = params[block][key]
            source = torch.randn(1, weight.shape[1], rows, columns, dtype=torch.float64)
            want = F.pixel_shuffle(
                F.conv2d(source, weight.double(), bias.double(), padding=1), 2)
            folded = fold_weights(weight.double(), bias.double()).double()
            extended = torch.cat(
                [source, torch.ones(1, 1, rows, columns, dtype=torch.float64)], dim=1)
            got = F.conv_transpose2d(extended, folded, stride=2, padding=2)
            worst = max(worst, float((want - got).abs().max()))
    return worst


# --------------------------------------------------------------------------
# One streaming step
# --------------------------------------------------------------------------

def conv(x, spec, relu6: bool = False):
    weight, bias, stride = spec
    value = F.conv2d(x, weight, bias, stride=stride, padding=1)
    return F.relu6(value) if relu6 else value


def bibuffer(x, state, spec, gate):
    """One BiBufferConv. `state` is packed as [center(C), left(C/8)].

    `gate` scales the `right` contribution: 1 for a normal step, 0 for a
    drained one. A drained `_BiBufferConv` differs from a normal one only in
    that `right` becomes zeros - its `_center` also goes `None`, but that only
    makes it return `None` on the following call, by which time the unit
    downstream is itself draining and ignores its input's value. So the centre
    written on a drained step is dead and needs no gate.
    """
    channels = int(spec[0].shape[1])
    fold = channels // 8
    center = state[:, :channels]
    left = state[:, channels: channels + fold]
    packed = torch.cat([x[:, :fold] * gate, left, center[:, 2 * fold:]], dim=1)
    value = conv(packed, spec, relu6=True)
    return value, torch.cat([x, center[:, fold: 2 * fold]], dim=1)


def mem_block(x, state_a, state_b, spec_a, spec_b, gate_a, gate_b):
    value, new_a = bibuffer(x, state_a, spec_a, gate_a)
    value, new_b = bibuffer(value, state_b, spec_b, gate_b)
    return value, new_a, new_b


class BsvdStep(torch.nn.Module):
    """One BSVD step: 16 BiBuffer states as buffers, skip FIFOs as I/O."""

    def __init__(self, params, folded, shapes):
        super().__init__()
        self.params = params
        self.folded = folded
        for index, shape in enumerate(shapes):
            self.register_buffer(f"st{index}", torch.zeros(*shape, dtype=torch.float16))

    def _states(self, block: int):
        # Read as fp32. The buffers must stay fp16 because `ct.StateType`
        # accepts nothing else, but tracing an fp16 conv2d runs PyTorch's CPU
        # reference path, which at 480x640 looks like a hang.
        return [getattr(self, f"st{block * 8 + i}").float() for i in range(8)]

    def _upsample(self, x, block: str, key: str):
        # Static ints: under tracing `x.shape` yields tensors and the torch
        # frontend cannot fold the resulting int cast.
        ones = torch.ones(1, 1, int(x.shape[2]), int(x.shape[3]), dtype=x.dtype)
        return F.conv_transpose2d(
            torch.cat([x, ones], dim=1), self.folded[block][key], stride=2, padding=2)

    def forward(self, frame, gate, sk0, sk1, sk2, sk3, sk4, sk5):
        gates = [gate[:, i: i + 1] for i in range(16)]
        popped = [sk0, sk1, sk2, sk3, sk4, sk5]
        pushed = []
        x = frame
        for index, block in enumerate(BLOCKS):
            p = self.params[block]
            states = self._states(index)
            g = index * 8
            skip1_pop, skip2_pop, skip3_pop = popped[index * 3: index * 3 + 3]

            pushed.append(x[:, :3])
            x0 = conv(conv(x, p["inc0"], relu6=True), p["inc3"], relu6=True)
            pushed.append(x0)

            x1 = conv(x0, p["d0"], relu6=True)
            x1, s0, s1 = mem_block(x1, states[0], states[1], p["d0c1"], p["d0c2"], gates[g + 0], gates[g + 1])
            pushed.append(x1)

            x2 = conv(x1, p["d1"], relu6=True)
            x2, s2, s3 = mem_block(x2, states[2], states[3], p["d1c1"], p["d1c2"], gates[g + 2], gates[g + 3])
            x2, s4, s5 = mem_block(x2, states[4], states[5], p["u2c1"], p["u2c2"], gates[g + 4], gates[g + 5])
            x2 = self._upsample(x2, block, "u2")

            merged, s6, s7 = mem_block(
                x2 + skip3_pop, states[6], states[7], p["u1c1"], p["u1c2"], gates[g + 6], gates[g + 7])
            x1o = self._upsample(merged, block, "u1")

            prediction = conv(conv(x1o + skip2_pop, p["out0"], relu6=True), p["out3"])
            head = skip1_pop[:, :3] - prediction[:, :3]
            x = head if prediction.shape[1] == 3 else torch.cat(
                [head, prediction[:, 3:]], dim=1)

            # Each write depends on its own read, which is the data coupling
            # the ANE compiler requires of an in-place state update.
            for offset, value in enumerate([s0, s1, s2, s3, s4, s5, s6, s7]):
                getattr(self, f"st{index * 8 + offset}")[:] = value.half()
        return tuple([x] + pushed)



# --------------------------------------------------------------------------
# Drain schedule
# --------------------------------------------------------------------------

def drain_schedule(length: int, steps: int = 16):
    """Per tail step: which units are gated, and which skip lines still push.

    Derived from the product's own `step(None)`, not from a rule, because the
    push side is non-monotonic for streams shorter than the 16-frame fill: a
    unit that was never primed can start pushing partway through the tail. The
    schedule is resolution-independent (verified 16x16 against 480x640), so it
    is measured at a trivial size and applied at any.

    The gate side does follow a closed form - unit `i` is gated at tail step
    `k` when `i <= k or i >= length + 1 + k`, verified for lengths 1 to 48 -
    but deriving both the same way avoids two sources of truth.
    """
    order = ("down0.c1", "down0.c2", "down1.c1", "down1.c2",
             "up2.c1", "up2.c2", "up1.c1", "up1.c2")
    net = B.BSVD(B.default_weights_path("c64"), dtype=mx.float32)
    units = [getattr(getattr(block, name.split(".")[0]), name.split(".")[1])
             for block in (net.temp1, net.temp2) for name in order]
    lines = [getattr(block, label) for block in (net.temp1, net.temp2)
             for label in ("skip1", "skip2", "skip3")]
    for index, unit in enumerate(units):
        unit._probe_id = index
    for index, line in enumerate(lines):
        line._probe_id = index

    record = {"gates": None, "pushes": None}
    original_call, original_push = B._BiBufferConv.__call__, B._MemSkip.push

    def patched_call(self, input_right):
        identity = getattr(self, "_probe_id", None)
        if identity is not None and record["gates"] is not None:
            record["gates"][identity] = input_right is None
        return original_call(self, input_right)

    def patched_push(self, value):
        identity = getattr(self, "_probe_id", None)
        if identity is not None and record["pushes"] is not None:
            record["pushes"][identity] = value is not None
        return original_push(self, value)

    # Patch the CLASS. Assigning `instance.__call__` does not intercept
    # `self.c1(x)`, because Python resolves `__call__` on the type, and the
    # resulting silence looks like a working experiment.
    B._BiBufferConv.__call__, B._MemSkip.push = patched_call, patched_push
    try:
        net.reset()
        generator = np.random.default_rng(3)
        for _ in range(length):
            frame = mx.array(generator.random((1, 16, 16, net.input_channels),
                                              dtype=np.float32))
            net.step(frame)
        gates, pushes, emitted = [], [], 0
        for _ in range(steps):
            record["gates"], record["pushes"] = [False] * 16, [False] * 6
            result = net.step(None)
            gates.append(list(record["gates"]))
            pushes.append(list(record["pushes"]))
            emitted += result is not None
    finally:
        B._BiBufferConv.__call__, B._MemSkip.push = original_call, original_push
    return gates, pushes, emitted


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------

def convert(model, shapes, skips, input_channels, height, width, directory: Path):
    import CoreML
    import coremltools as ct
    from Foundation import NSURL

    directory.mkdir(parents=True, exist_ok=True)
    package = directory / "model.mlpackage"
    compiled = directory / "model.mlmodelc"

    sample = tuple([torch.zeros(1, input_channels, height, width),
                    torch.ones(1, 16, 1, 1)]
                   + [torch.zeros(*shape) for shape in skips])
    with torch.no_grad():
        traced = torch.jit.trace(model, sample)

    names = ["frame", "gate"] + [f"skip_{i}" for i in range(len(skips))]
    outputs = ["out"] + [f"skip_out_{i}" for i in range(len(skips))]
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=tuple(t.shape), dtype=np.float16)
                for n, t in zip(names, sample, strict=True)],
        outputs=[ct.TensorType(name=n) for n in outputs],
        states=[ct.StateType(
            wrapped_type=ct.TensorType(shape=tuple(shape), dtype=np.float16),
            name=f"st{i}") for i, shape in enumerate(shapes)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    if package.is_dir():
        shutil.rmtree(package)
    staging = Path(tempfile.mkdtemp(prefix="package-", dir=str(directory))) / package.name
    converted.save(str(staging))
    staging.replace(package)

    url, error = CoreML.MLModel.compileModelAtURL_error_(
        NSURL.fileURLWithPath_(str(package)), None)
    if url is None:
        raise RuntimeError(f"Core ML compile failed: {error}")
    if compiled.is_dir():
        shutil.rmtree(compiled)
    staging = Path(tempfile.mkdtemp(prefix="compiled-", dir=str(directory))) / compiled.name
    shutil.copytree(Path(str(url.path())), staging)
    staging.replace(compiled)
    return compiled


def placement(compiled: Path) -> dict:
    from collections import Counter

    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(
        str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE)
    preferred = Counter()
    for function in plan.model_structure.program.functions.values():
        for operation in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is not None:
                preferred[type(usage.preferred_compute_device).__name__] += 1
    return dict(preferred)


# --------------------------------------------------------------------------
# Runner with pre-bound buffers and output backings
# --------------------------------------------------------------------------

class BackedRunner:
    """One dispatch per frame; MLState for BiBuffers, backings for skip I/O."""

    def __init__(self, compiled: Path):
        import CoreML
        from Foundation import NSURL

        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)
        model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
            NSURL.fileURLWithPath_(str(compiled)), config, None)
        if model is None:
            raise RuntimeError(f"model load failed: {error}")
        self._model = model
        self._state = model.newState()

        description = model.modelDescription()
        self._in = self._bind(CoreML, description.inputDescriptionsByName())
        self._out = self._bind(CoreML, description.outputDescriptionsByName())
        names_in = list(description.inputDescriptionsByName())
        names_out = list(description.outputDescriptionsByName())

        provider, error = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {n: CoreML.MLFeatureValue.featureValueWithMultiArray_(self._in[n][2])
             for n in names_in}, None)
        if provider is None:
            raise RuntimeError(f"feature provider failed: {error}")
        self._provider = provider
        options = CoreML.MLPredictionOptions.alloc().init()
        options.setOutputBackings_({n: self._out[n][2] for n in names_out})
        self._options = options

        ones = np.ones((1, 16, 1, 1), dtype=np.float16)
        self._in["gate"][1][:] = memoryview(ones).cast("B")

        self._fifos = []
        for index in range(6):
            depth = SKIP_DEPTH[("skip1", "skip2", "skip3")[index % 3]]
            width = len(self._in[f"skip_{index}"][1])
            self._fifos.append([bytearray(width) for _ in range(depth)])
        self._cursor = [0] * 6

    @staticmethod
    def _bind(CoreML, descriptions):
        bound = {}
        for name in descriptions:
            shape = [int(d) for d in descriptions[name].multiArrayConstraint().shape()]
            array = mx.zeros(shape, dtype=mx.float16)
            mx.eval(array)
            strides, acc = [1] * len(shape), 1
            for i in range(len(shape) - 1, -1, -1):
                strides[i], acc = acc, acc * shape[i]
            multi, error = CoreML.MLMultiArray.alloc().initWithDataPointer_shape_dataType_strides_deallocator_error_(
                memoryview(array), shape, _MLMULTIARRAY_FLOAT16, strides, None, None)
            if multi is None:
                raise RuntimeError(f"MLMultiArray wrap failed for {name}: {error}")
            bound[name] = (array, memoryview(array).cast("B"), multi)
        return bound

    def predict(self):
        result, error = self._model.predictionFromFeatures_usingState_options_error_(
            self._provider, self._state, self._options, None)
        if result is None:
            raise RuntimeError(f"prediction failed: {error}")
        return result

    def step(self, frame_bytes):
        self._in["frame"][1][:] = frame_bytes
        for index in range(6):
            self._in[f"skip_{index}"][1][:] = self._fifos[index][self._cursor[index]]
        self.predict()
        for index in range(6):
            self._fifos[index][self._cursor[index]][:] = self._out[f"skip_out_{index}"][1]
            self._cursor[index] = (self._cursor[index] + 1) % len(self._fifos[index])
        return self._out["out"][0]


# --------------------------------------------------------------------------
# Validation and timing
# --------------------------------------------------------------------------

def frames(count: int, channels: int, height: int, width: int):
    """Deterministic, temporally coherent inputs in [0,1]."""
    base = mx.random.uniform(shape=(1, height, width, channels), key=mx.random.key(20260718))
    out = []
    for index in range(count):
        noise = mx.random.uniform(shape=(1, height, width, channels),
                                  key=mx.random.key(1000 + index))
        frame = mx.clip(base * 0.8 + noise * 0.2 + index * 0.017, 0.0, 1.0)
        mx.eval(frame)
        out.append(frame)
    return out


def torch_reference(model, sequence, shapes, skips):
    """Reference from the same graph, on zeroed state.

    Compute is fp32; the recurrent buffers are fp16 and each update rounds
    through `.half()`, deliberately, because `ct.StateType` is fp16-only and
    the reference has to carry the same state precision the graph does. So
    this is NOT an fp32-throughout oracle - it isolates the Core ML/ANE
    execution of this graph, not the graph's own state quantization.

    It also shares `BsvdStep` with the model under test, so a transcription
    error in `BsvdStep` would pass. That was closed out of band by comparing
    against the product MLX path (2.8e-8), but this script does not re-gate
    it; treat a green run here as evidence about execution, not about the
    transcription.
    """
    reference = BsvdStep(model.params, model.folded, shapes).eval()
    carried = [torch.zeros(*shape) for shape in shapes]
    fifos = {i: [torch.zeros(*skips[i])
                 for _ in range(SKIP_DEPTH[("skip1", "skip2", "skip3")[i % 3]])]
             for i in range(len(skips))}
    for index, buffer in enumerate(carried):
        getattr(reference, f"st{index}")[:] = buffer.half()
    outputs = []
    with torch.no_grad():
        for frame in sequence:
            nchw = torch.from_numpy(np.ascontiguousarray(
                np.transpose(np.asarray(frame.astype(mx.float32)), (0, 3, 1, 2))))
            result = reference(nchw, torch.ones(1, 16, 1, 1),
                               *[fifos[i][-1] for i in range(len(skips))])
            outputs.append(np.transpose(result[0].numpy(), (0, 2, 3, 1)))
            for i, value in enumerate(result[1:]):
                fifos[i].insert(0, value)
                fifos[i].pop()
    return outputs


def drive(compiled: Path, units, sequence, skips):
    from coremltools.models import CompiledMLModel

    model = CompiledMLModel(str(compiled), units)
    state = model.make_state()
    fifos = {i: [np.zeros(tuple(skips[i]), dtype=np.float16)
                 for _ in range(SKIP_DEPTH[("skip1", "skip2", "skip3")[i % 3]])]
             for i in range(len(skips))}
    outputs = []
    for frame in sequence:
        feed = {"frame": np.ascontiguousarray(
            np.transpose(np.asarray(frame.astype(mx.float32)), (0, 3, 1, 2)),
            dtype=np.float16),
            "gate": np.ones((1, 16, 1, 1), dtype=np.float16)}
        for i in range(len(skips)):
            feed[f"skip_{i}"] = fifos[i][-1]
        result = model.predict(feed, state=state)
        outputs.append(np.transpose(
            np.asarray(result["out"], dtype=np.float32), (0, 2, 3, 1)))
        for i in range(len(skips)):
            fifos[i].insert(0, np.asarray(result[f"skip_out_{i}"], dtype=np.float16))
            fifos[i].pop()
    return outputs



def drain_tail(compiled, sequence, skips, channels, height, width,
               gates, pushes, emitted, gated: bool):
    """Feed the stream, then the tail, and return the tail outputs.

    With `gated` false this is what the graph can do without a drain flag -
    feed zero frames - which is what the tail currently costs.
    """
    import coremltools as ct
    from coremltools.models import CompiledMLModel

    model = CompiledMLModel(str(compiled), ct.ComputeUnit.CPU_AND_NE)
    state = model.make_state()
    fifos = {i: [np.zeros(tuple(skips[i]), dtype=np.float16)
                 for _ in range(SKIP_DEPTH[("skip1", "skip2", "skip3")[i % 3]])]
             for i in range(len(skips))}
    ones = np.ones((1, 16, 1, 1), dtype=np.float16)

    def advance(frame_nchw, gate, pushing):
        feed = {"frame": frame_nchw, "gate": gate}
        for i in range(len(skips)):
            feed[f"skip_{i}"] = fifos[i][-1]
        result = model.predict(feed, state=state)
        for i in range(len(skips)):
            value = (np.asarray(result[f"skip_out_{i}"], dtype=np.float16)
                     if pushing[i] else np.zeros_like(fifos[i][0]))
            fifos[i].insert(0, value)
            fifos[i].pop()
        return result

    every = [True] * 6
    for frame in sequence:
        advance(np.ascontiguousarray(
            np.transpose(np.asarray(frame.astype(mx.float32)), (0, 3, 1, 2)),
            dtype=np.float16), ones, every)

    zero = np.zeros((1, channels, height, width), dtype=np.float16)
    tail = []
    for step in range(emitted):
        gate = ones.copy()
        if gated:
            for unit in range(16):
                if gates[step][unit]:
                    gate[0, unit, 0, 0] = 0.0
        result = advance(zero, gate, pushes[step] if gated else every)
        tail.append(np.transpose(
            np.asarray(result["out"], dtype=np.float32), (0, 2, 3, 1)))
    return tail


def compare(reference, candidate, skip_first: int) -> dict:
    rows = [np.abs(a - b) for a, b in zip(reference[skip_first:], candidate[skip_first:], strict=True)]
    if not rows:
        return {}
    return {"mean_abs": float(np.mean([r.mean() for r in rows])),
            "max_abs": float(max(r.max() for r in rows))}


def median(call, warm: int = 5, iterations: int = 25) -> float:
    for _ in range(warm):
        call()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="c64")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    # 64, not 12: recurrent error keeps growing well past a short window, and
    # it grows at different rates for different backends, so a short run
    # misranks them. Over 12 steps the ANE looks 3.6x worse than MLX fp16;
    # over 64 it is slightly better, because MLX fp16 degrades 12x across that
    # span against the ANE's 3x.
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--directory", type=Path, default=None)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    height, width = arguments.height, arguments.width
    directory = arguments.directory or (
        scratch_root() / f"{arguments.variant}-{width}x{height}")

    params, channels = load_torch_parameters(arguments.variant)
    shapes, skips = state_shapes(params, channels, height, width)
    state_bytes = sum(int(np.prod(s)) for s in shapes) * 2

    worst = audit_fold(params, height, width)
    _log.info(f"fold algebra, all four upsamples, float64: max abs {worst:.2e}")
    if worst > 1e-9:
        _log.info("fold is not exact; refusing to continue")
        return 1

    folded = {block: {key: fold_weights(*params[block][key][:2])
                      for key in ("u2", "u1")} for block in BLOCKS}
    model = BsvdStep(params, folded, shapes).eval()
    compiled = convert(model, shapes, skips, channels, height, width, directory)
    _log.info(f"compiled: {compiled}")
    _log.info(f"device placement: {placement(compiled)}")
    _log.info(f"MLState carries {state_bytes / 1e6:.0f} MB, keeping "
          f"{2 * state_bytes / 1e6:.0f} MB per step off the boundary")

    sequence = frames(arguments.steps, channels, height, width)
    reference = torch_reference(model, sequence, shapes, skips)

    import coremltools as ct
    _log.info("\naccuracy against the same graph, fp32 compute / fp16 state (steps 2+):")
    results = {}
    for label, units in (("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
                         ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)):
        stats = compare(reference, drive(compiled, units, sequence, skips), 2)
        results[label] = stats
        note = "" if label == "CPU_AND_NE" else "   <- known Core ML CPU state defect"
        _log.info(f"  {label:11s} mean {stats['mean_abs']:.3e}  "
              f"max {stats['max_abs']:.3e}{note}")

    gates, pushes, emitted = drain_schedule(arguments.steps)
    _log.info("\ndrain: schedule derived for a %d-frame stream, %d tail outputs",
              arguments.steps, emitted)
    product = B.BSVD(B.default_weights_path(arguments.variant), dtype=mx.float32)
    product.reset()
    for frame in sequence:
        product.step(frame)
    truth = []
    for _ in range(emitted):
        result = product.step(None)
        truth.append(None if result is None else np.asarray(result))

    tails = {}
    for label in ("ungated", "gated"):
        tails[label] = drain_tail(compiled, sequence, skips, channels, height, width,
                                  gates, pushes, emitted, gated=(label == "gated"))
    _log.info("tail against the product's own fp32 step(None), worst of %d frames:", emitted)
    for label, tail in tails.items():
        worst = max(float(np.abs(w - g).mean())
                    for w, g in zip(truth, tail, strict=True) if w is not None)
        _log.info("  %-8s %.3e", label, worst)

    runner = BackedRunner(compiled)
    frame = memoryview(np.zeros((1, channels, height, width), dtype=np.float16)).cast("B")
    prediction_ms = median(runner.predict)
    step_ms = median(lambda: runner.step(frame))
    _log.info(f"\n{width}x{height} with output backings: "
          f"prediction {prediction_ms:.1f} ms, full step {step_ms:.1f} ms")

    report = {"variant": arguments.variant, "height": height, "width": width,
              "fold_algebra_max_abs": worst, "state_bytes": state_bytes,
              "accuracy": results, "prediction_ms": prediction_ms,
              "step_ms": step_ms}
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    _log.info(f"wrote {directory / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
