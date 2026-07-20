"""BSVD streaming denoise on the Neural Engine.

The opt-in ``--bsvd-backend ane`` implementation: the full per-step
convolution stack runs as one Core ML dispatch pinned to the ANE, with the
16 BiBuffer recurrences carried in MLState, the six skip delay lines held
host-side as copy-free MLX ring buffers (bound per dispatch through
``ModelRunner.predict_with``), and the ``conv -> pixel_shuffle`` pairs
folded into exact stride-two transposed convolutions (the native MIL
``pixel_shuffle`` miscomputes on the ANE after a convolution).

Standalone this path is slightly SLOWER than the MLX incumbent (about 91
vs 78 ms/frame at 640x480 c64); its value is composition - BSVD stops
occupying the GPU, so chains with other GPU stages can hide its cost.
Numerically it is slightly closer to fp32 than the shipping MLX fp16 path
(mean 2.5e-4 vs 2.8e-4, worst pixel 2.3x better, over 64 steps).

Fill and drain reproduce the product schedule exactly. The MLX network
propagates ``None`` through a 16-step fill and drains the same way; this
ordinary streaming graph always computes, so two (1,16,1,1) vector inputs
close the gap:
``write`` zeroes each unit's carried ``left`` fold on its priming step
(what actually leaks from a zero prologue is state, not output), and
``gate`` zeroes each unit's ``right`` contribution on its drained steps.
Both schedules, the skip-push gating, and the emit pattern come from a
boolean mirror of the product's own None propagation (`_NoneFlowNet`),
kept equivalent to the real network by test.

GOP-scheduled windows have their full frame list available up front.
For those, :mod:`.ane_phases` unrolls each fixed fill/drain half eight steps
at a time and omits the operations and skip outputs whose product value is
``None``.  The phase functions, ordinary step, host rings, and MLState are
one exact-replay-gated multifunction asset.

Safety: the Core ML CPU runtime miscomputes this graph's MLState updates
(coherent garbage, not noise), so a CPU fallback is silently wrong rather
than slow. Placement is refused unless every operation is ANE-preferred,
a build-time canary requires CPU and ANE runs to DISAGREE, the rebinding
runner must match the byte-copy reference bit-exactly, and every load
replays stored outputs.

Geometry envelope (probed 2026-07-19 on the shipped graph): the width the
graph runs at must be a multiple of 128, which keeps every pyramid
level's fp16 rows 64-byte aligned - the quarter-resolution level binds at
W/2 bytes per row. Widths off that grid (352, 320, 704, 32 all probed)
convert cleanly, report full ANE placement, then fail the FIRST
prediction with ANEProgramProcessRequestDirect status=0x1d; heights need
only the multiple-of-four the denoiser already guarantees (96, 100, 144,
232, 240, 288, 480 all pass at aligned widths). `AneBSVD` therefore
reflect-pads frames on the right up to the next 128 multiple and crops
the outputs back - CIF 352x288 runs as 384x288 - so callers see the
original geometry. Frames below 96 px on a side are refused (the
verified floor).

Conversion is first-party through :mod:`kinovsr.native.anemil` (protobuf
against the vendored Core ML schema; no coremltools, numpy, or torch) and
is cached per weights + geometry under the KinoVSR cache directory. The
first run at a new geometry emits and compiles the model - roughly a
minute at SD sizes - and later runs load it in about a second.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any

import mlx.core as mx

from kinovsr.native.anemil import runtime

_log = logging.getLogger("kinovsr.bsvd_ane")

BLOCKS = ("temp1", "temp2")
# The eight BiBufferConvs of a DenBlock in execution order, with the
# resolution divisor each runs at.
BIBUF = (("d0c1", 2), ("d0c2", 2), ("d1c1", 4), ("d1c2", 4),
         ("u2c1", 4), ("u2c2", 4), ("u1c1", 2), ("u1c2", 2))
SKIP_DEPTH = (8, 8, 4)  # skip1, skip2, skip3 push-to-pop latency

GRAPH_VERSION = 1
MIN_SIDE = 96
# The graph width must keep every pyramid level's fp16 rows 64-byte
# aligned (the quarter-res level is W/2 bytes per row): W % 128 == 0.
# Probed 2026-07-19: widths 128/256/384/640 run; 32/320/352/704 compile,
# place all-ANE, then fail the first prediction with status=0x1d.
ANE_WIDTH_QUANTUM = 128
CPU_DIVERGENCE_FLOOR = 1e-6
REPLAY_TOLERANCE = 5e-3
REPLAY_STEPS = 12  # past the depth-8 ring wraparound (first re-read at step 8)
REPLAY_SKIP = 2

_VERIFIED: set[str] = set()


def cache_root() -> Path:
    from kinovsr.settings import default_settings

    return Path(default_settings().cache_dir).expanduser() / "bsvd-ane"


def _cache_directory(params: dict, height: int, width: int) -> Path:
    return cache_root() / (
        f"{_weights_key(params)}-{width}x{height}-v{GRAPH_VERSION}")


def _weights_key(params: dict) -> str:
    digest = hashlib.sha256()
    for block in BLOCKS:
        for key in sorted(params[block]):
            weight, bias, stride = params[block][key]
            digest.update(f"{block}.{key}.{tuple(weight.shape)}.{stride}".encode())
            if bias is not None:
                digest.update(str(tuple(bias.shape)).encode())
    sample = params["temp1"]["inc0"][0]
    flat = sample.reshape(-1)[:64].astype(mx.float32)
    mx.eval(flat)
    digest.update(bytes(memoryview(mx.contiguous(flat)).cast("B")))
    return digest.hexdigest()[:16]


def _shapes(params: dict, input_channels: int, height: int, width: int):
    states = []
    for block in BLOCKS:
        for key, divisor in BIBUF:
            channels = int(params[block][key][0].shape[3])
            states.append((1, channels + channels // 8,
                           height // divisor, width // divisor))
    skips = []
    for block in BLOCKS:
        skips.append((1, 3, height, width))
        skips.append((1, int(params[block]["inc3"][0].shape[0]), height, width))
        skips.append((1, int(params[block]["d0c1"][0].shape[3]),
                      height // 2, width // 2))
    return states, skips


# --------------------------------------------------- float64 fold audit (MLX)

def _assert_true_float64() -> None:
    """Refuse the fold audit unless MLX CPU float64 is real double precision
    for the operation classes the audit uses (arithmetic, matmul, reductions,
    scatter, movement - transcendentals are NOT trusted and not used)."""
    tiny = 2.0 ** -40
    with mx.stream(mx.cpu):
        one_tiny = mx.array([1.0 + tiny], dtype=mx.float64)
        checks = {
            "add_sub": float((one_tiny + tiny - one_tiny)[0]) == tiny,
            "mul": abs(float((one_tiny * one_tiny - 1.0)[0]) - 2.0 ** -39)
                   < 2.0 ** -75,
            "sum": float(mx.sum(mx.array([1.0, tiny, -1.0],
                                         dtype=mx.float64))) == tiny,
            "matmul": float((mx.array([[1.0, tiny]], dtype=mx.float64)
                             @ mx.array([[1.0], [1.0]], dtype=mx.float64)
                             - 1.0)[0, 0]) == tiny,
            "scatter_add": float(mx.zeros((6,), dtype=mx.float64)
                                 .at[0:5:2].add(mx.array([tiny] * 3,
                                                         dtype=mx.float64))[2]
                                 ) == tiny,
            "movement": float(mx.pad(mx.transpose(one_tiny.reshape(1, 1)),
                                     ((0, 0), (1, 0)))[0, 1]) == 1.0 + tiny,
        }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            f"MLX CPU float64 is not double-precision for {failed}; refusing "
            f"the upsample fold audit.")


def _fold_weights(weight, bias, r: int = 2):
    """[Cout*r*r, Cin, k, k] + bias -> [Cin+1, Cout, k*r, k*r], any dtype.

    Sub-pixel convolution and transposed convolution are equivalent; the
    appended input channel is fed a constant one and carries the per-phase
    bias, which a per-channel transposed-convolution bias cannot express.
    """
    c4 = int(weight.shape[0])
    cin, k = int(weight.shape[1]), int(weight.shape[2])
    cout = c4 // (r * r)
    if k != 3 or r != 2:
        raise ValueError("fold is written for k=3, r=2")
    w6 = weight.reshape(cout, r, r, cin, k, k)[:, :, :, :, ::-1, ::-1]
    main = mx.transpose(w6, (3, 0, 4, 1, 5, 2)).reshape(
        cin, cout, k * r, k * r)
    plane = mx.pad(bias.reshape(cout, r, r), ((0, 0), (2, 2), (2, 2)))[None]
    return mx.concatenate([main, plane.astype(weight.dtype)], axis=0)


def _conv2d_f64(x, weight, bias):
    channels, h, w = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    padded = mx.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = mx.zeros((1, int(weight.shape[0]), h, w), dtype=x.dtype)
    for kh in range(3):
        for kw in range(3):
            window = padded[:, :, kh:kh + h, kw:kw + w].reshape(
                channels, h * w)
            out = out + (weight[:, :, kh, kw] @ window).reshape(1, -1, h, w)
    return out + bias.reshape(1, -1, 1, 1)


def _pixel_shuffle_f64(y, r: int = 2):
    n, c4, h, w = [int(s) for s in y.shape]
    c = c4 // (r * r)
    return mx.transpose(y.reshape(n, c, r, r, h, w),
                        (0, 1, 4, 2, 5, 3)).reshape(n, c, h * r, w * r)


def _conv_transpose_f64(x, weight, stride: int = 2, pad: int = 2):
    ci, h, w = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    k = int(weight.shape[2])
    oh = (h - 1) * stride - 2 * pad + k
    ow = (w - 1) * stride - 2 * pad + k
    out = mx.zeros((1, int(weight.shape[1]), oh, ow), dtype=x.dtype)
    for kh in range(k):
        for kw in range(k):
            oy0, ox0 = kh - pad, kw - pad
            iy_lo = max(0, -(oy0 // stride))
            iy_hi = min(h - 1, (oh - 1 - oy0) // stride)
            ix_lo = max(0, -(ox0 // stride))
            ix_hi = min(w - 1, (ow - 1 - ox0) // stride)
            if iy_lo > iy_hi or ix_lo > ix_hi:
                continue
            piece = x[:, :, iy_lo:iy_hi + 1, ix_lo:ix_hi + 1]
            rows, cols = int(piece.shape[2]), int(piece.shape[3])
            piece = (mx.transpose(weight[:, :, kh, kw])
                     @ piece.reshape(ci, rows * cols)).reshape(
                         1, -1, rows, cols)
            out = out.at[:, :,
                         oy0 + stride * iy_lo: oy0 + stride * iy_hi + 1: stride,
                         ox0 + stride * ix_lo: ox0 + stride * ix_hi + 1: stride
                         ].add(piece)
    return out


def _oihw(weight):
    """MLX OHWI conv weight (the product layout) -> contiguous fp32 OIHW."""
    return mx.contiguous(mx.transpose(weight, (0, 3, 1, 2)).astype(mx.float32))


def _audit_fold(params: dict, height: int, width: int) -> float:
    """Prove every upsample fold exact in float64 before converting."""
    _assert_true_float64()
    worst = 0.0
    geometry = {"u2": (height // 4, width // 4), "u1": (height // 2, width // 2)}
    with mx.stream(mx.cpu):
        for index, block in enumerate(BLOCKS):
            for key, (rows, columns) in geometry.items():
                weight = _oihw(params[block][key][0]).astype(mx.float64)
                bias = params[block][key][1].astype(mx.float64)
                source = (mx.random.uniform(
                    shape=(1, int(weight.shape[1]), rows, columns),
                    key=mx.random.key(20260719 + index)) * 2.0 - 1.0
                ).astype(mx.float64)
                want = _pixel_shuffle_f64(_conv2d_f64(source, weight, bias))
                folded = _fold_weights(weight, bias)
                extended = mx.concatenate(
                    [source, mx.ones((1, 1, rows, columns),
                                     dtype=mx.float64)], axis=1)
                got = _conv_transpose_f64(extended, folded)
                worst = max(worst, float(mx.abs(want - got).max()))
    return worst


# ----------------------------------------------------------------- emission

def _emit_graph(params: dict, input_channels: int, height: int, width: int,
                blob=None):
    """Emit one BSVD step and return its graph plus function signature."""
    from kinovsr.native.anemil import builder

    state_shapes, skip_shapes = _shapes(params, input_channels, height, width)
    g = builder.Graph(blob)

    g.register_input("frame", (1, input_channels, height, width))
    g.register_input("gate", (1, 16, 1, 1))
    g.register_input("write", (1, 16, 1, 1))
    for i, dims in enumerate(skip_shapes):
        g.register_input(f"skip_{i}", dims)
    for i, dims in enumerate(state_shapes):
        g.register_input(f"st{i}", dims)

    def conv(x, block, key, relu6=False, name=None):
        weight, bias, stride = params[block][key]
        return g.conv2d(x, _oihw(weight),
                        None if bias is None else bias.astype(mx.float32),
                        tag=f"{block}_{key}", stride=int(stride),
                        relu6=relu6, relu6_name=name)

    def upsample(x, block, key):
        weight, bias, _stride = params[block][key]
        folded = _fold_weights(_oihw(weight), bias.astype(mx.float32))
        divisor = 4 if key == "u2" else 2
        rows, cols = height // divisor, width // divisor
        ones = g.fp16_const(g.n(f"{block}_{key}_ones"),
                            mx.ones((1, 1, rows, cols), dtype=mx.float16))
        ext = g.concat_channels([x, ones], tag=f"{block}_{key}_ext")
        return g.conv_transpose2d(ext, folded, tag=f"{block}_{key}_up")

    gates = [g.slice_channels("gate", i, 1, f"gate{i}") for i in range(16)]
    writes = [g.slice_channels("write", i, 1, f"write{i}") for i in range(16)]

    x = "frame"
    pushed = []
    for block_index, block in enumerate(BLOCKS):
        base = block_index * 8
        block_reads = [g.read_state(f"st{base + i}") for i in range(8)]
        pending: list[tuple[int, str, str]] = []

        def bibuffer(x, index, block, key, gate_i, write_i, name=None, *,
                     block_reads=block_reads, pending=pending):
            reads = block_reads[index % 8]
            channels = int(params[block][key][0].shape[3])
            fold = channels // 8
            center = g.slice_channels(reads, 0, channels,
                                      f"{block}_{key}_center")
            left = g.slice_channels(reads, channels, fold,
                                    f"{block}_{key}_left")
            right = g.slice_channels(x, 0, fold, f"{block}_{key}_right")
            gated = g.binary("mul", right, gate_i, f"{block}_{key}_gated")
            tail = g.slice_channels(center, 2 * fold, channels - 2 * fold,
                                    f"{block}_{key}_tail")
            packed = g.concat_channels([gated, left, tail],
                                       tag=f"{block}_{key}_in")
            value = conv(packed, block, key, relu6=True, name=name)

            carry = g.slice_channels(center, fold, fold,
                                     f"{block}_{key}_carry")
            carry = g.binary("mul", carry, write_i, f"{block}_{key}_wgate")
            new_state = g.concat_channels([x, carry],
                                          tag=f"{block}_{key}_state")
            pending.append((index, reads, new_state))
            return value

        skip1_pop = f"skip_{block_index * 3}"
        skip2_pop = f"skip_{block_index * 3 + 1}"
        skip3_pop = f"skip_{block_index * 3 + 2}"

        pushed.append(g.slice_channels(x, 0, 3, "skip1_push",
                                       name=f"skip_out_{block_index * 3}"))
        x0 = conv(x, block, "inc0", relu6=True)
        x0 = conv(x0, block, "inc3", relu6=True,
                  name=f"skip_out_{block_index * 3 + 1}")
        pushed.append(x0)

        x1 = conv(x0, block, "d0", relu6=True)
        x1 = bibuffer(x1, base + 0, block, "d0c1",
                      gates[base + 0], writes[base + 0])
        x1 = bibuffer(x1, base + 1, block, "d0c2",
                      gates[base + 1], writes[base + 1],
                      name=f"skip_out_{block_index * 3 + 2}")
        pushed.append(x1)

        x2 = conv(x1, block, "d1", relu6=True)
        x2 = bibuffer(x2, base + 2, block, "d1c1",
                      gates[base + 2], writes[base + 2])
        x2 = bibuffer(x2, base + 3, block, "d1c2",
                      gates[base + 3], writes[base + 3])
        x2 = bibuffer(x2, base + 4, block, "u2c1",
                      gates[base + 4], writes[base + 4])
        x2 = bibuffer(x2, base + 5, block, "u2c2",
                      gates[base + 5], writes[base + 5])
        # The u2 conv itself is folded into the transposed convolution.
        x2 = upsample(x2, block, "u2")

        merged = g.binary("add", x2, skip3_pop, f"{block}_skip3_add")
        merged = bibuffer(merged, base + 6, block, "u1c1",
                          gates[base + 6], writes[base + 6])
        merged = bibuffer(merged, base + 7, block, "u1c2",
                          gates[base + 7], writes[base + 7])
        x1o = upsample(merged, block, "u1")

        y = g.binary("add", x1o, skip2_pop, f"{block}_skip2_add")
        prediction = conv(conv(y, block, "out0", relu6=True), block, "out3")

        out_channels = int(params[block]["out3"][0].shape[0])
        final = block_index == len(BLOCKS) - 1
        if out_channels == 3:
            x = g.binary("sub", skip1_pop, prediction, f"{block}_minus",
                         name="out" if final else None)
        else:
            head3 = g.slice_channels(prediction, 0, 3, f"{block}_head3")
            head = g.binary("sub", skip1_pop, head3, f"{block}_minus")
            rest = g.slice_channels(prediction, 3, out_channels - 3,
                                    f"{block}_rest")
            x = g.concat_channels([head, rest], tag=f"{block}_next",
                                  name="out" if final else None)

        for index, reads, new_state in pending:
            g.update_state(f"st{index}", reads, new_state)

    output_names = ["out"] + [f"skip_out_{i}" for i in range(len(pushed))]
    inputs = ([("frame", (1, input_channels, height, width)),
               ("gate", (1, 16, 1, 1)), ("write", (1, 16, 1, 1))]
              + [(f"skip_{i}", skip_shapes[i])
                 for i in range(len(skip_shapes))])
    states = [(f"st{i}", state_shapes[i]) for i in range(len(state_shapes))]
    return g, inputs, states, output_names


def _emit_program(params: dict, input_channels: int, height: int, width: int):
    """One BSVD step as an MLState mlprogram, on the verified spellings."""
    graph, inputs, states, output_names = _emit_graph(
        params, input_channels, height, width)
    model_bytes = graph.finish(
        inputs, states, output_names, "KinoVSR BSVD ANE")
    return model_bytes, graph.blob


def _convert(params: dict, input_channels: int, height: int, width: int,
             directory: Path) -> Path:
    from kinovsr.native.anemil import builder

    directory.mkdir(parents=True, exist_ok=True)
    package = directory / "model.mlpackage"
    if package.is_dir():
        return package
    model_bytes, blob = _emit_program(params, input_channels, height, width)
    staging = directory / "model.partial.mlpackage"
    shutil.rmtree(staging, ignore_errors=True)
    builder.write_package(staging, model_bytes, blob)
    staging.replace(package)
    return package


# ------------------------------------------------------------------ runtime

class BsvdRunner:
    """BSVD streaming with copy-free skip rings (depths 8/8/4).

    Each step binds the oldest slot of every ring directly as the skip
    input and a spare buffer as the push backing, then swaps references -
    no per-step byte copies. Bit-exact against `ByteCopyBsvdRunner`, which
    `_verify_build` gates at exact equality.
    """

    def __init__(self, compiled: Path, compute_units: str = "ane",
                 function_name: str | None = None, state: Any | None = None):
        self.model = runtime.ModelRunner(compiled, compute_units,
                                         dynamic=("skip_",),
                                         function_name=function_name,
                                         state=state)
        for required in ("frame", "gate", "write"):
            if required not in self.model.inputs:
                raise RuntimeError(f"model is missing input '{required}'")
        if "out" not in self.model.outputs:
            raise RuntimeError("model is missing output 'out'")
        self._skips = sum(1 for n in self.model.dynamic_inputs
                          if n.startswith("skip_"))
        ones = mx.ones((1, 16, 1, 1), dtype=mx.float16)
        mx.eval(ones)
        self._ones = memoryview(mx.contiguous(ones)).cast("B")
        self._rings = [
            [runtime.bind_array(self.model.dynamic_inputs[f"skip_{i}"])
             for _ in range(SKIP_DEPTH[i % 3])]
            for i in range(self._skips)]
        self._spares = [
            runtime.bind_array(self.model.dynamic_inputs[f"skip_{i}"])
            for i in range(self._skips)]
        # Ring slots are logically zero until a graph actually pushes them.
        # Keeping one immutable zero backing per line avoids physically
        # clearing hundreds of MiB at every independently reset window, and
        # lets phase-specialized graphs omit outputs for non-push steps.
        self._zeros = [
            runtime.bind_array(self.model.dynamic_inputs[f"skip_{i}"])
            for i in range(self._skips)]
        self._valid = [
            [False] * len(ring) for ring in self._rings]
        self._cursor = [0] * self._skips
        self.reset()

    def reset(self) -> None:
        self.model.reset_state()
        for valid in self._valid:
            valid[:] = [False] * len(valid)
        self._cursor = [0] * self._skips
        self.model.input_view("gate")[:] = self._ones
        self.model.input_view("write")[:] = self._ones

    def _input_multi(self, line: int, slot: int):
        if self._valid[line][slot]:
            return self._rings[line][slot][2]
        return self._zeros[line][2]

    def _bindings(self):
        features = {f"skip_{i}": self._input_multi(i, self._cursor[i])
                    for i in range(self._skips)}
        backings = {f"skip_out_{i}": self._spares[i][2]
                    for i in range(self._skips)}
        return features, backings

    def predict(self):
        """One dispatch with the current bindings, no ring rotation."""
        features, backings = self._bindings()
        return self.model.predict_with(features, backings)

    def load_inputs(self, frame_bytes, gate_bytes=None,
                    write_bytes=None) -> None:
        """Blit one step's inputs into the bound buffers (host-side only)."""
        self.model.input_view("frame")[:] = frame_bytes
        self.model.input_view("gate")[:] = (
            gate_bytes if gate_bytes is not None else self._ones)
        self.model.input_view("write")[:] = (
            write_bytes if write_bytes is not None else self._ones)

    def dispatch(self):
        """Predict on the loaded inputs and rotate the rings.

        Pure Core ML plus Python reference swaps - no MLX - so it is safe
        to run on a worker thread while the main thread owns every MLX
        operation (the AneBSVD pipelining arrangement).
        """
        self.predict()
        for i in range(self._skips):
            slot = self._cursor[i]
            self._rings[i][slot], self._spares[i] = (
                self._spares[i], self._rings[i][slot])
            self._valid[i][slot] = True
            self._cursor[i] = (slot + 1) % len(self._rings[i])
        return self.model.output_array("out")

    def step(self, frame_bytes, gate_bytes=None, write_bytes=None):
        self.load_inputs(frame_bytes, gate_bytes, write_bytes)
        return self.dispatch()

    def zero_last_push(self, line: int) -> None:
        """Logically zero the last slot (the product pushed nothing there)."""
        ring = self._rings[line]
        slot = (self._cursor[line] - 1) % len(ring)
        self._valid[line][slot] = False


class ByteCopyBsvdRunner:
    """Reference implementation: skip-FIFO byte rings through fixed
    bindings. Verification only - it pays the copy cost the rebinding
    runner removes."""

    def __init__(self, compiled: Path, compute_units: str = "ane",
                 function_name: str | None = None, state: Any | None = None):
        self.model = runtime.ModelRunner(
            compiled, compute_units, function_name=function_name, state=state)
        self._skips = sum(1 for n in self.model.inputs
                          if n.startswith("skip_"))
        ones = mx.ones((1, 16, 1, 1), dtype=mx.float16)
        mx.eval(ones)
        self._ones = memoryview(mx.contiguous(ones)).cast("B")
        self._fifos = [
            [bytearray(len(self.model.input_view(f"skip_{i}")))
             for _ in range(SKIP_DEPTH[i % 3])]
            for i in range(self._skips)]
        self._cursor = [0] * self._skips
        self.reset()

    def reset(self) -> None:
        self.model.reset_state()
        for ring in self._fifos:
            for slot in ring:
                slot[:] = bytes(len(slot))
        self._cursor = [0] * self._skips
        self.model.input_view("gate")[:] = self._ones
        self.model.input_view("write")[:] = self._ones

    def step(self, frame_bytes, gate_bytes=None, write_bytes=None):
        self.model.input_view("frame")[:] = frame_bytes
        self.model.input_view("gate")[:] = (
            gate_bytes if gate_bytes is not None else self._ones)
        self.model.input_view("write")[:] = (
            write_bytes if write_bytes is not None else self._ones)
        for i in range(self._skips):
            self.model.input_view(f"skip_{i}")[:] = \
                self._fifos[i][self._cursor[i]]
        self.model.predict()
        for i in range(self._skips):
            self._fifos[i][self._cursor[i]][:] = \
                self.model.output_view(f"skip_out_{i}")
            self._cursor[i] = (self._cursor[i] + 1) % len(self._fifos[i])
        return self.model.output_array("out")


# ------------------------------------------------------- verification gates

def _replay_frames(input_channels: int, height: int, width: int, count: int):
    base = mx.random.uniform(shape=(1, height, width, input_channels),
                             key=mx.random.key(20260718))
    out = []
    for index in range(count):
        noise = mx.random.uniform(shape=(1, height, width, input_channels),
                                  key=mx.random.key(1000 + index))
        frame = mx.clip(base * 0.8 + noise * 0.2 + index * 0.017, 0.0, 1.0)
        nchw = mx.contiguous(mx.transpose(frame, (0, 3, 1, 2))
                             .astype(mx.float16))
        mx.eval(nchw)
        out.append(nchw)
    return out


def _drive(runner, frames) -> list:
    runner.reset()
    outputs = []
    for frame in frames:
        out = runner.step(memoryview(frame).cast("B"))
        snap = out.astype(mx.float32)
        mx.eval(snap)
        outputs.append(snap)
    runner.reset()
    return outputs


def _mean_abs(a: list, b: list, skip_first: int) -> float:
    diffs = [mx.abs(x - y).mean() for x, y in
             zip(a[skip_first:], b[skip_first:], strict=True)]
    value = mx.mean(mx.stack(diffs))
    mx.eval(value)
    return float(value)


def _verify_build(compiled: Path, directory: Path, input_channels: int,
                  height: int, width: int, placement: dict) -> None:
    frames = _replay_frames(input_channels, height, width, REPLAY_STEPS)
    ane_runner = BsvdRunner(compiled, "ane")
    on_ane = _drive(ane_runner, frames)
    del ane_runner
    cpu_runner = BsvdRunner(compiled, "cpu")
    on_cpu = _drive(cpu_runner, frames)
    del cpu_runner
    separation = _mean_abs(on_ane, on_cpu, REPLAY_SKIP)
    if separation < CPU_DIVERGENCE_FLOOR:
        raise RuntimeError(
            f"CPU_AND_NE and CPU_ONLY agree to {separation:.3e}, so the "
            f"requested ANE run is executing on the CPU, where this graph's "
            f"state updates are miscomputed.")
    byte_copy = ByteCopyBsvdRunner(compiled, "ane")
    on_reference = _drive(byte_copy, frames)
    del byte_copy
    fifo_ab = max(float(mx.abs(a - b).max())
                  for a, b in zip(on_ane, on_reference, strict=True))
    if fifo_ab != 0.0:
        raise RuntimeError(
            f"rebinding runner differs from the byte-copy reference (max "
            f"abs {fifo_ab:.3e}); Core ML is not honoring the ring output "
            f"backings - refusing the build.")
    mx.save_safetensors(
        str(directory / "replay"),
        {f"out_{i}": on_ane[i] for i in range(REPLAY_SKIP, REPLAY_STEPS)})
    (directory / "verify.json").write_text(json.dumps({
        "graph_version": GRAPH_VERSION,
        "placement": placement,
        "canary_separation": separation,
        "fifo_ab_max_abs": fifo_ab,
        "replay_steps": REPLAY_STEPS,
        "replay_skip": REPLAY_SKIP,
        "replay_tolerance": REPLAY_TOLERANCE,
    }, indent=2))
    _log.info("bsvd-ane build verified: placement %s, canary %.3e, "
              "fifo A/B %.1e", placement, separation, fifo_ab)


def _verify_load(runner: BsvdRunner, directory: Path, input_channels: int,
                 height: int, width: int) -> None:
    record = json.loads((directory / "verify.json").read_text())
    if record.get("graph_version") != GRAPH_VERSION:
        raise RuntimeError("cache was built by a different graph version")
    stored = mx.load(str(directory / "replay.safetensors"))
    frames = _replay_frames(input_channels, height, width,
                            int(record["replay_steps"]))
    outputs = _drive(runner, frames)
    skip = int(record["replay_skip"])
    expected = [stored[f"out_{i}"] for i in range(skip, len(frames))]
    drift = _mean_abs(outputs[skip:], expected, 0)
    tolerance = float(record.get("replay_tolerance", REPLAY_TOLERANCE))
    if drift > tolerance:
        raise RuntimeError(
            f"replay drift {drift:.3e} exceeds {tolerance:.0e}; the model is "
            f"not producing the outputs it was verified with (CPU fallback "
            f"or a stale cache).")


def build_runner(params: dict, input_channels: int, height: int,
                 width: int) -> tuple[BsvdRunner, Path]:
    """Convert (cached), compile, gate, and construct a runner.

    Each caller gets its OWN runner - its own MLState and ring buffers -
    from the shared verified on-disk artifacts. The load replay oracle
    runs once per cache directory per process.
    """
    if height % 4 or width % 4:
        raise RuntimeError(f"{width}x{height} is not a multiple of four")
    if width % ANE_WIDTH_QUANTUM:
        raise RuntimeError(
            f"graph width {width} is not a multiple of "
            f"{ANE_WIDTH_QUANTUM}; unaligned widths convert and place "
            f"all-ANE, then fail the first prediction (status=0x1d). "
            f"AneBSVD pads frames to the quantum - use it, or pad first.")
    if min(height, width) < MIN_SIDE:
        raise RuntimeError(
            f"{width}x{height} is below the verified ANE floor ({MIN_SIDE} "
            f"px); small stateful graphs convert and then fail at the first "
            f"prediction.")
    directory = _cache_directory(params, height, width)
    complete = all((directory / name).exists() for name in
                   ("model.mlpackage", "verify.json", "replay.safetensors"))
    if not complete:
        _log.info("building BSVD ANE model for %dx%d (one-time per "
                  "geometry, cached under %s)", width, height, directory)
        worst = _audit_fold(params, height, width)
        if worst > 1e-9:
            raise RuntimeError(
                f"upsample fold audit failed in float64: {worst:.3e}")
        package = _convert(params, input_channels, height, width, directory)
        compiled = runtime.compile_package(package)
        placement = runtime.assert_all_ane(compiled)
        _verify_build(compiled, directory, input_channels, height, width,
                      placement)
        _VERIFIED.add(str(directory))
    compiled = runtime.compile_package(directory / "model.mlpackage")
    runner = BsvdRunner(compiled, "ane")
    if str(directory) not in _VERIFIED:
        _verify_load(runner, directory, input_channels, height, width)
        _VERIFIED.add(str(directory))
    return runner, directory


# ------------------------------------------- product None-flow mirror

class _StepRecord:
    """What one mirrored step observed, in schedule terms."""

    __slots__ = ("out_real", "unprimed", "primes", "drained", "pushes")

    def __init__(self):
        self.out_real = False
        self.unprimed = [False] * 16   # center held None at entry
        self.primes = [False] * 16     # center went None -> real this step
        self.drained = [False] * 16    # unit input was None this step
        self.pushes = [False] * 6      # skip line received a real push


class _NoneBiBuffer:
    """Boolean mirror of ``_BiBufferConv``'s None propagation."""

    def __init__(self, index: int):
        self.index = index
        self.center_real = None  # None: slot holds None; True: a real tensor

    def __call__(self, right_real: bool, record: _StepRecord) -> bool:
        record.drained[self.index] = not right_real
        record.unprimed[self.index] = self.center_real is None
        if self.center_real is None:
            if right_real:
                record.primes[self.index] = True
                self.center_real = True
            return False
        self.center_real = True if right_real else None
        return True


class _NoneSkip:
    """Boolean mirror of ``_MemSkip``."""

    def __init__(self, index: int):
        self.index = index
        self.items = 0

    def push(self, real: bool, record: _StepRecord) -> None:
        record.pushes[self.index] = real
        if real:
            self.items += 1

    def pop(self, trigger_real: bool) -> bool:
        if not trigger_real or self.items == 0:
            return False
        self.items -= 1
        return True


class _NoneDenBlock:
    """Boolean mirror of ``_DenBlock.__call__``'s None propagation."""

    def __init__(self, block_index: int):
        base = block_index * 8
        self.units = [_NoneBiBuffer(base + i) for i in range(8)]
        skip_base = block_index * 3
        self.skip1 = _NoneSkip(skip_base)
        self.skip2 = _NoneSkip(skip_base + 1)
        self.skip3 = _NoneSkip(skip_base + 2)

    def __call__(self, x_real: bool, record: _StepRecord) -> bool:
        self.skip1.push(x_real, record)
        x0 = x_real                                     # inc
        self.skip2.push(x0, record)
        x1 = self.units[1](self.units[0](x0, record), record)   # down0
        self.skip3.push(x1, record)
        x2 = self.units[3](self.units[2](x1, record), record)   # down1
        x2 = self.units[5](self.units[4](x2, record), record)   # up2 mem
        merged = x2 and self.skip3.pop(x2)              # none_add
        m = self.units[7](self.units[6](merged, record), record)  # up1 mem
        y = m and self.skip2.pop(m)                     # none_add -> out conv
        return y and self.skip1.pop(y)                  # none_minus


class _NoneFlowNet:
    """Boolean mirror of the product ``BSVD`` net's None propagation.

    Kept equivalent to the real network by test (the schedule is derived
    from instrumented product classes there and compared for stream
    lengths 1..48). The write/gate/push/emit schedules the ANE path needs
    are read off this mirror, so a graph that always computes reproduces
    the product's fill and drain behavior exactly.
    """

    def __init__(self):
        self.blocks = [_NoneDenBlock(0), _NoneDenBlock(1)]

    def step(self, real: bool) -> _StepRecord:
        record = _StepRecord()
        record.out_real = self.blocks[1](
            self.blocks[0](real, record), record)
        return record


# -------------------------------------------------------------- entry point

def _pad_width_reflect(x: Any, target: int) -> Any:
    """Reflect-pad NHWC ``x`` on the right to ``target`` columns."""
    width = int(x.shape[2])
    pad = target - width
    if pad == 0:
        return x
    mirror = x[:, :, width - 1 - pad:width - 1, :][:, :, ::-1, :]
    return mx.concatenate([x, mirror], axis=2)


class AneBSVD:
    """Drop-in for :class:`kinovsr.processors.bsvd.BSVD` on the ANE.

    Same contract shape: ``step(frame_or_none)`` once per stream item with
    NHWC (1, H, W, C) inputs padded to multiples of four, ``None`` through
    the fill, real outputs through the drain, ``reset()`` between streams.
    The engine is built lazily at the first real frame, when the geometry
    is known; construction failures raise (this backend is explicitly
    requested - there is no silent fallback).

    **Dispatches are pipelined one step deep.** ``step(k)`` collects the
    result of dispatch ``k-1`` and SUBMITS dispatch ``k`` to a worker
    thread, so the Core ML prediction runs while the caller does the rest
    of its per-frame work. That both hides the dispatch latency and keeps
    the ANE busy through the inter-frame gap - critical, because an ANE
    dispatch issued after >= 10 ms of host idleness pays a 15-23 ms
    power-state ramp that no host-side warm-up avoids (measured; the
    synchronous arrangement ran 57 ms/dispatch inside light pipelines
    against 34 ms hot). The worker runs ONLY ``BsvdRunner.dispatch`` -
    pure Core ML - and every MLX operation stays on the caller's thread.
    The pipelining adds one step to the output delay: ``SHIFT_NUM`` is 17
    (the network's 16 plus one dispatch in flight).

    Frames are reflect-padded on the right up to the ANE width quantum
    (multiples of 128 - see the module docstring) and outputs are cropped
    back, so callers see the original geometry throughout. The cache and
    the compiled model key on the PADDED size, so every source width that
    maps to the same quantum shares one model.
    """

    SHIFT_NUM = 17  # 16-step BiBuffer delay + 1 dispatch in flight
    TAIL_STEPS = 16

    def __init__(self, weights_path: str | Path, dtype: Any = mx.float16):
        if dtype != mx.float16:
            raise ValueError(
                "the BSVD ANE backend executes fp16 only; use "
                "--bsvd-dtype float16 or --bsvd-backend mlx")
        from . import load_bsvd

        self.dtype = mx.float16
        self.params, self.input_channels = load_bsvd(
            weights_path, dtype=mx.float32)
        self._runner: BsvdRunner | None = None
        self._directory: Path | None = None
        self._phase_suite: Any | None = None
        self._geometry: tuple[int, int] | None = None
        self._width = 0
        self._padded_width = 0
        self._zero_frame: bytes | None = None
        self._mirror = _NoneFlowNet()
        self._tail: list[dict] | None = None
        self._tail_cursor = 0
        self._window_complete = False
        self._worker: threading.Thread | None = None
        self._go = threading.Event()
        self._done = threading.Event()
        self._stop = False
        self._closed = False
        self._dirty = False
        self._pending: dict | None = None
        self._worker_error: BaseException | None = None

    def reset(self) -> None:
        self._require_open()
        self._join_pending(discard=True)
        if self._runner is not None and self._dirty:
            self._runner.reset()
            if self._phase_suite is not None:
                self._phase_suite.set_state(self._runner.model._state)
        self._dirty = False
        self._mirror = _NoneFlowNet()
        self._tail = None
        self._tail_cursor = 0
        self._window_complete = False

    def close(self) -> None:
        """Release the dispatch worker and the runner's large state buffers.

        The pipeline owns this object through an explicit lifecycle, so do
        not rely on a destructor: the worker's bound method retains ``self``
        (and therefore the Core ML state plus skip rings) until it is told to
        exit.  Cleanup remains complete even when an in-flight prediction
        failed; that error is re-raised after the worker has stopped.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._join_pending(discard=True)
        finally:
            self._stop = True
            self._go.set()
            worker = self._worker
            if worker is not None and worker is not threading.current_thread():
                worker.join()
            if self._phase_suite is not None:
                self._phase_suite.close()
            self._phase_suite = None
            self._worker = None
            self._runner = None
            self.params = {}
            self._geometry = None
            self._directory = None
            self._zero_frame = None
            self._tail = None
            self._mirror = None
            self._dirty = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("BSVD ANE backend is closed")

    def _configure_geometry(self, height: int, width: int) -> None:
        self._require_open()
        if self._geometry is not None:
            if self._geometry != (height, width):
                raise RuntimeError(
                    f"BSVD ANE stream changed resolution from "
                    f"{self._geometry} to {(height, width)}")
            return
        if min(height, width) < MIN_SIDE:
            raise RuntimeError(
                f"{width}x{height} is below the verified ANE floor "
                f"({MIN_SIDE} px per side); use --bsvd-backend mlx for "
                f"smaller frames.")
        padded = -(-width // ANE_WIDTH_QUANTUM) * ANE_WIDTH_QUANTUM
        self._geometry = (height, width)
        self._width = width
        self._padded_width = padded

    def _ensure_worker(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._worker_loop, name="bsvd-ane-dispatch", daemon=True)
        self._worker.start()

    def _ensure_runner(self, height: int, width: int) -> None:
        self._configure_geometry(height, width)
        if self._runner is None:
            self._runner, self._directory = build_runner(
                self.params, self.input_channels, height, self._padded_width)
            self._zero_frame = bytes(
                len(self._runner.model.input_view("frame")))
        self._ensure_worker()

    def _ensure_scheduled_runner(self, height: int, width: int) -> None:
        self._configure_geometry(height, width)
        if self._phase_suite is not None:
            return
        from .ane_phases import build_suite

        self._directory = _cache_directory(
            self.params, height, self._padded_width)
        self._phase_suite = build_suite(
            self.params, self.input_channels, height,
            self._padded_width, self._directory)
        self._runner = self._phase_suite.runner
        self._zero_frame = bytes(
            len(self._runner.model.input_view("frame")))

    # ------------------------------------------------ dispatch pipelining

    def _worker_loop(self) -> None:
        while True:
            self._go.wait()
            self._go.clear()
            if self._stop:
                return
            try:
                self._runner.dispatch()
            except BaseException as exc:  # noqa: BLE001 - carried to caller
                self._worker_error = exc
            self._done.set()

    def _submit(self, frame_bytes, gate_bytes, write_bytes, emit: bool,
                pushes: list | None = None) -> None:
        self._runner.load_inputs(frame_bytes, gate_bytes, write_bytes)
        self._pending = {"emit": emit, "pushes": pushes}
        self._dirty = True
        self._done.clear()
        self._go.set()

    def _join_pending(self, discard: bool = False):
        """Wait out the in-flight dispatch; return its emitted output.

        Push gating and output materialization happen here, on the
        caller's thread, strictly before the next submit reloads the
        input buffers and the next dispatch rewrites the out backing.
        """
        if self._pending is None:
            return None
        self._done.wait()
        pending, self._pending = self._pending, None
        if self._worker_error is not None:
            error, self._worker_error = self._worker_error, None
            raise error
        if discard:
            return None
        if pending["pushes"] is not None:
            for line, pushed in enumerate(pending["pushes"]):
                if not pushed:
                    self._runner.zero_last_push(line)
        return self._materialize_out() if pending["emit"] else None

    @staticmethod
    def _vector_bytes(values: list[float]):
        vector = mx.array(values, dtype=mx.float16).reshape(1, 16, 1, 1)
        mx.eval(vector)
        return memoryview(mx.contiguous(vector)).cast("B")

    def _materialize_array(self, raw):
        # Fresh copy, NCHW backing -> NHWC cropped to the caller's width:
        # the backing is rewritten by the next prediction, so no lazy
        # graph over it may leave this method.
        out = mx.contiguous(
            mx.transpose(raw, (0, 2, 3, 1))[:, :, :self._width, :])
        mx.eval(out)
        return out

    def _materialize_out(self):
        return self._materialize_array(
            self._runner.model.output_array("out"))

    def run_window(self, frames: list[Any]) -> list[Any]:
        """Run one independently reset window through phase-specialized ANE.

        Schedule windows are already fully buffered by ``BsvdDenoiser``.
        Unrolling their fixed 16-step fill and drain removes convolutions
        whose product value is ``None`` and halves the number of dispatches;
        the steady middle still uses the verified one-step runner.  The
        returned list is in input-frame order and has exactly ``len(frames)``
        entries.
        """
        self._require_open()
        if len(frames) < 16:
            raise ValueError("BSVD ANE phase path needs at least 16 frames")
        if self._dirty or self._pending is not None:
            raise RuntimeError("reset BSVD ANE before running a schedule window")
        first = frames[0]
        height, width = int(first.shape[1]), int(first.shape[2])
        self._ensure_scheduled_runner(height, width)

        packed = []
        for frame in frames:
            if (int(frame.shape[1]), int(frame.shape[2])) != (height, width):
                raise RuntimeError(
                    "BSVD ANE schedule window changed resolution")
            padded = _pad_width_reflect(
                frame.astype(mx.float16), self._padded_width)
            nchw = mx.contiguous(mx.transpose(padded, (0, 3, 1, 2)))
            mx.eval(nchw)
            packed.append(memoryview(nchw).cast("B"))
        outputs = self._phase_suite.run(
            packed, memoryview(self._zero_frame), self._materialize_array)
        if len(outputs) != len(frames):
            raise RuntimeError(
                f"BSVD ANE phase path returned {len(outputs)} outputs for "
                f"{len(frames)} frames")
        self._dirty = True
        self._window_complete = True
        return outputs

    def _assemble_tail(self) -> list[dict]:
        """Mirror the product's 16 drain steps and derive the schedule.

        ``write`` fires on a unit's priming step over the WHOLE timeline -
        units left unprimed by a stream shorter than the fill prime DURING
        the drain, so their write gate falls in the tail. The final tail
        step also writes-gates any unit still unprimed, matching the
        derivation the product schedule was verified against.
        """
        records = [self._mirror.step(False) for _ in range(self.TAIL_STEPS)]
        tail = []
        for k, record in enumerate(records):
            writes = [1.0] * 16
            for i in range(16):
                if record.unprimed[i] and (
                        k + 1 >= self.TAIL_STEPS
                        or not records[k + 1].unprimed[i]):
                    writes[i] = 0.0
            tail.append({
                "gate": [0.0 if record.drained[i] else 1.0
                         for i in range(16)],
                "write": writes,
                "pushes": list(record.pushes),
                "emit": record.out_real,
            })
        return tail

    def step(self, x: Any | None) -> Any | None:
        self._require_open()
        if self._window_complete:
            raise RuntimeError(
                "BSVD ANE schedule window is complete; reset() first")
        if self._runner is not None:
            self._ensure_worker()
        if x is None:
            if self._runner is None:
                return None            # drained before any input frame
            if self._tail is None:
                self._tail = self._assemble_tail()
                self._tail_cursor = 0
            out = self._join_pending()
            if self._tail_cursor < len(self._tail):
                entry = self._tail[self._tail_cursor]
                self._tail_cursor += 1
                self._submit(self._zero_frame,
                             self._vector_bytes(entry["gate"]),
                             self._vector_bytes(entry["write"]),
                             entry["emit"], entry["pushes"])
            return out

        if self._tail is not None:
            raise RuntimeError(
                "BSVD ANE received a real frame after draining began; "
                "reset() first")
        height, width = int(x.shape[1]), int(x.shape[2])
        self._ensure_runner(height, width)
        out = self._join_pending()
        record = self._mirror.step(True)
        write = [0.0 if record.primes[i] else 1.0 for i in range(16)]
        padded = _pad_width_reflect(x.astype(mx.float16),
                                    self._padded_width)
        nchw = mx.contiguous(mx.transpose(padded, (0, 3, 1, 2)))
        mx.eval(nchw)
        self._submit(memoryview(nchw).cast("B"), None,
                     self._vector_bytes(write), record.out_real)
        return out


__all__ = ["AneBSVD", "BsvdRunner", "ByteCopyBsvdRunner", "build_runner",
           "cache_root"]
