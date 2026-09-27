"""BSVD streaming denoise on the Neural Engine.

The opt-in ``--bsvd-backend ane`` implementation: the full per-step
convolution stack runs as one Core ML dispatch pinned to the ANE, with the
16 BiBuffer recurrences carried in MLState, the six skip delay lines held
host-side as copy-free MLX ring buffers (bound per dispatch through
``ModelRunner.predict_with``), and the upsamplers emitted as the reference
``conv -> pixel_shuffle`` in <=256-channel shuffle groups.  Native MIL
``pixel_shuffle`` is shape-dependent on this ANE: 256-channel inputs are
correct, wider inputs are badly corrupted, and the grouped form is exactly
equivalent (the channel->pixel mapping is block-diagonal).  The previously
shipped fold into stride-two transposed convolutions was bit-exact and
equally fast but broke Espresso's translator at large geometries (below).

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

Safety: the Core ML CPU runtime executes this fp16 ML Program's convolutions
with substantially less accurate accumulation than the ANE.  The MLState
read/slice-update/write machinery itself is bit-exact against equivalent
ordinary tensor I/O, but recurrent feedback amplifies the CPU convolution
error beyond the product tolerance.  A CPU fallback is therefore silently
wrong rather than merely slow. Placement is refused unless every operation
is ANE-preferred (the four value-exact ``island_*`` ops of large geometries
are the sole allowlisted exception), a build-time differential canary
fingerprints the CPU and ANE paths, the rebinding runner must match the
byte-copy reference bit-exactly, and every load replays stored outputs.

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

Large geometries (probed 2026-07-21): Espresso's MIL->EIR translator
crashes (std::bad_cast, surfaced as plan-build error -14) on large
stateful graphs - with the retired folded conv_transpose upsamplers any
state update derived from one broke the build above 1920x648, and even
with the current conv + grouped pixel-shuffle spelling (bit-exact and
speed-neutral with the folded form) a two-block translation unit breaks
above 1920x864. The emitted step therefore inserts a value-exact
float32 island between temp1 and temp2 above ``ISLAND_MIN_PIXELS``,
which makes Espresso translate the blocks as separate units; the four
``island_*`` ops legitimately run off-ANE (allowlisted in the placement
gate) and cost roughly 0.12 s/frame at 1080p. Islandless execution is
verified through 1920x864 (plus 1280x720 and 1664x900); the island form
is verified executing 64-frame streams at 1920x1080.

Island-geometry accuracy (measured 2026-07-23): the island OPS are
value-exact, but the islanded stateful graph as Core ML executes it at
1920x1080 lands ~1.5e-2 mean from the product fp32 net on settled
frames - two orders past this backend's usual ~3e-4 - and its build
gates never saw that because the replay oracle is self-referential.
Large geometries therefore route to the direct AppleNeuralEngine chain
(:mod:`.ane_direct`, ~3.5e-4 and faster); this islanded path remains
the small-geometry engine and the fallback when the private route is
unavailable.

Conversion is first-party through :mod:`kinovsr.native.anemil` (protobuf
against the vendored Core ML schema; no coremltools, numpy, or torch) and
is cached per weights + geometry under the KinoVSR cache directory. The
first run at a new geometry emits and compiles the model - roughly a
minute at SD sizes - and later runs load it in about a second.
"""

import hashlib
import json
import logging
import shutil
import struct
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx

from kinovsr.native.anemil import builder, runtime, schema
from kinovsr.settings import default_settings

_log = logging.getLogger("kinovsr.bsvd_ane")


def _vector_bytes(values: list[float]) -> bytes:
    """Pack one 16-lane gate/write schedule vector as fp16 bytes.

    These go straight into a bound byte view (``load_inputs`` blits them
    into ``input_view("gate")``), so the graph only ever wanted 32 bytes
    in the right layout - building them through ``mx.array`` + ``mx.eval``
    allocated and evaluated a GPU-backed array per dispatch just to read
    its bytes back out. ``mx.float16`` IS IEEE 754 binary16, which is what
    struct's ``e`` emits, and the schedule's values are exactly 0.0 and
    1.0, so this is byte-identical to the MLX spelling it replaced rather
    than merely close (asserted by test).

    Keeping the dispatch schedule free of MLX also leaves the door open
    for a dispatch loop that does not run on the MLX lane.
    """
    return struct.pack("<16e", *values)


BLOCKS = ("temp1", "temp2")
# The eight BiBufferConvs of a DenBlock in execution order, with the
# resolution divisor each runs at.
BIBUF = (
    ("d0c1", 2),
    ("d0c2", 2),
    ("d1c1", 4),
    ("d1c2", 4),
    ("u2c1", 4),
    ("u2c2", 4),
    ("u1c1", 2),
    ("u1c2", 2),
)
SKIP_DEPTH = (8, 8, 4)  # skip1, skip2, skip3 push-to-pop latency

GRAPH_VERSION = 2
MIN_SIDE = 96
# Above this many graph pixels the emitted step inserts a float32
# translation island between temp1 and temp2 (see _emit_graph). The
# islandless graph is verified through 1920x864 = 1,658,880 px; the
# first measured islandless failure is 1920x960 = 1,843,200 px, and the
# island form is verified loading at 1920x960/1008 and executing at
# 1920x1080. The island is always CORRECT (value-exact); the threshold
# only avoids its ~20% dispatch cost at small geometries.
ISLAND_MIN_PIXELS = 1920 * 864 + 1
# The graph width must keep every pyramid level's fp16 rows 64-byte
# aligned (the quarter-res level is W/2 bytes per row): W % 128 == 0.
# Probed 2026-07-19: widths 128/256/384/640 run; 32/320/352/704 compile,
# place all-ANE, then fail the first prediction with status=0x1d.
ANE_WIDTH_QUANTUM = 128
# The three-context phase schedule (main + fill_00 + drain_08) is stable
# through this measured production envelope.  At 640x480, adding either
# inner phase as a fourth resident ANE program makes program re-entry fail
# with ANEProgramProcessRequestDirect status=0x16; those steps therefore run
# through main.  With the v7 shuffle-spelled chunks the re-entry boundary
# was re-probed 2026-07-21 (cold gate plus six bit-exact reset windows per
# geometry): 640x480, 768x576, and 1024x576 pass with a steady ~1.45x
# window speedup over gated main; 1152x576 and 1024x648 (663K px, both
# equal-area) fail with 0x16 on the first re-entry, so the working-set
# boundary sits just above 590K px.  Geometries beyond the rectangle stay
# on the fully gated main function.
PHASE_MAX_WIDTH = 1024
PHASE_MAX_HEIGHT = 576
CPU_DIVERGENCE_FLOOR = 1e-6
REPLAY_TOLERANCE = 5e-3
REPLAY_STEPS = 12  # past the depth-8 ring wraparound (first re-read at step 8)
REPLAY_SKIP = 2

_VERIFIED: set[str] = set()

type _Params = dict[str, dict[str, tuple[mx.array, mx.array | None, int]]]


def cache_root() -> Path:

    return Path(default_settings().cache_dir).expanduser() / "bsvd-ane"


def _cache_directory(params: _Params, height: int, width: int) -> Path:
    return cache_root() / (f"{_weights_key(params)}-{width}x{height}-v{GRAPH_VERSION}")


def _weights_key(params: _Params) -> str:
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


def _shapes(
    params: _Params, height: int, width: int, blocks: tuple[str, ...] = BLOCKS
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    states = []
    for block in blocks:
        for key, divisor in BIBUF:
            channels = int(params[block][key][0].shape[3])
            states.append((1, channels + channels // 8, height // divisor, width // divisor))
    skips = []
    for block in blocks:
        skips.append((1, 3, height, width))
        skips.append((1, int(params[block]["inc3"][0].shape[0]), height, width))
        skips.append((1, int(params[block]["d0c1"][0].shape[3]), height // 2, width // 2))
    return states, skips


def _oihw(weight: mx.array) -> mx.array:
    """MLX OHWI conv weight (the product layout) -> contiguous fp32 OIHW."""
    return mx.contiguous(mx.transpose(weight, (0, 3, 1, 2)).astype(mx.float32))


# ----------------------------------------------------------------- emission


def _emit_graph(
    params: _Params,
    input_channels: int,
    height: int,
    width: int,
    blob: builder.BlobFile | None = None,
    *,
    blocks: tuple[str, ...] = BLOCKS,
    explicit_state: bool = False,
) -> tuple[
    builder.Graph,
    list[tuple[str, tuple[int, int, int, int]]],
    list[tuple[str, tuple[int, int, int, int]]],
    list[str],
]:
    """Emit one BSVD step and return its graph plus function signature.

    The default emission is the shipped stateful graph (MLState
    recurrence, fp32 island above ``ISLAND_MIN_PIXELS``) and is
    byte-stable across the ``blocks``/``explicit_state`` parameters.
    ``explicit_state=True`` emits the same step with the recurrence as
    ordinary tensors - ``st{i}`` stay inputs, each unit's would-be state
    update becomes an output - and no island; combined with a single
    entry in ``blocks`` this yields the per-block halves the direct
    AppleNeuralEngine backend chains (see ``ane_direct``).
    """

    state_shapes, skip_shapes = _shapes(params, height, width, blocks=blocks)
    island = (not explicit_state) and height * width >= ISLAND_MIN_PIXELS
    g = builder.Graph(blob)

    g.register_input("frame", (1, input_channels, height, width))
    g.register_input("gate", (1, 16, 1, 1))
    g.register_input("write", (1, 16, 1, 1))
    for i, dims in enumerate(skip_shapes):
        g.register_input(f"skip_{i}", dims)
    for i, dims in enumerate(state_shapes):
        g.register_input(f"st{i}", dims)
    if island:
        g.register_input("island_bias", (1, 1, 1, 1))

    def conv(x: str, block: str, key: str, relu6: bool = False, name: str | None = None) -> str:
        weight, bias, stride = params[block][key]
        return g.conv2d(
            x,
            _oihw(weight),
            None if bias is None else bias.astype(mx.float32),
            tag=f"{block}_{key}",
            stride=int(stride),
            relu6=relu6,
            relu6_name=name,
        )

    def upsample(x: str, block: str, key: str) -> str:
        """The reference conv + 2x pixel-shuffle, shuffled in <=256
        channel groups (block-diagonal, exactly equivalent; the wider
        native shuffle is numerically wrong on the ANE). This unfolded
        spelling is bit-exact with the retired folded conv_transpose
        form and dodges the Espresso MIL->EIR std::bad_cast that killed
        plan builds above 1920x648 whenever a state update derived from
        a conv_transpose (probed 2026-07-21)."""
        y = conv(x, block, key)
        channels = int(g.shape[y][1])
        if channels <= 256:
            return g.pixel_shuffle2x(y, tag=f"{block}_{key}_ps")
        half = channels // 2
        parts = [
            g.pixel_shuffle2x(
                g.slice_channels(y, start, half, f"{block}_{key}_grp{j}"),
                tag=f"{block}_{key}_ps{j}",
            )
            for j, start in enumerate((0, half))
        ]
        return g.concat_channels(parts, tag=f"{block}_{key}_cat")

    def translation_island(x: str) -> str:
        """cast fp32 -> add a runtime zero bias -> cast fp16 on the
        temp1 -> temp2 handoff. Value-exact (widening cast, +0.0, exact
        narrowing) and placed off-ANE, which makes Espresso translate
        the two blocks as separate MIL->EIR units - each fits at
        geometries where the whole graph's translation crashes with
        std::bad_cast (any update_state in a two-block unit above
        ~1.66M px). The bias is a model input so nothing can constant-
        fold the add away; ModelRunner's fixed bindings feed it zeros
        with no runner changes."""
        dims = g.shape[x]
        x32 = g.op(
            "cast",
            {"x": x, "dtype": g.const_str(g.n("island_cast32_dtype"), "fp32")},
            g.n("island_cast32"),
            dims,
            dtype=schema.FLOAT32,
        )
        bias32 = g.op(
            "cast",
            {"x": "island_bias", "dtype": g.const_str(g.n("island_bias_dtype"), "fp32")},
            g.n("island_bias32"),
            (1, 1, 1, 1),
            dtype=schema.FLOAT32,
        )
        sum32 = g.op("add", {"x": x32, "y": bias32}, g.n("island_add"), dims, dtype=schema.FLOAT32)
        return g.op(
            "cast",
            {"x": sum32, "dtype": g.const_str(g.n("island_cast16_dtype"), "fp16")},
            g.n("island_cast16"),
            dims,
        )

    gates = [g.slice_channels("gate", i, 1, f"gate{i}") for i in range(16)]
    writes = [g.slice_channels("write", i, 1, f"write{i}") for i in range(16)]

    x = "frame"
    pushed = []
    state_output_names: list[str] = []
    for block_index, block in enumerate(blocks):
        base = block_index * 8
        if explicit_state:
            block_reads = [f"st{base + i}" for i in range(8)]
        else:
            block_reads = [g.read_state(f"st{base + i}") for i in range(8)]
        pending: list[tuple[int, str, str]] = []

        def bibuffer(
            x: str,
            index: int,
            block: str,
            key: str,
            gate_i: str,
            write_i: str,
            name: str | None = None,
            *,
            block_reads: list[str] = block_reads,
            pending: list[tuple[int, str, str]] = pending,
        ) -> str:
            reads = block_reads[index % 8]
            channels = int(params[block][key][0].shape[3])
            fold = channels // 8
            center = g.slice_channels(reads, 0, channels, f"{block}_{key}_center")
            left = g.slice_channels(reads, channels, fold, f"{block}_{key}_left")
            right = g.slice_channels(x, 0, fold, f"{block}_{key}_right")
            gated = g.binary("mul", right, gate_i, f"{block}_{key}_gated")
            tail = g.slice_channels(center, 2 * fold, channels - 2 * fold, f"{block}_{key}_tail")
            packed = g.concat_channels([gated, left, tail], tag=f"{block}_{key}_in")
            value = conv(packed, block, key, relu6=True, name=name)

            carry = g.slice_channels(center, fold, fold, f"{block}_{key}_carry")
            carry = g.binary("mul", carry, write_i, f"{block}_{key}_wgate")
            new_state = g.concat_channels([x, carry], tag=f"{block}_{key}_state")
            pending.append((index, reads, new_state))
            return value

        skip1_pop = f"skip_{block_index * 3}"
        skip2_pop = f"skip_{block_index * 3 + 1}"
        skip3_pop = f"skip_{block_index * 3 + 2}"

        pushed.append(g.slice_channels(x, 0, 3, "skip1_push", name=f"skip_out_{block_index * 3}"))
        x0 = conv(x, block, "inc0", relu6=True)
        x0 = conv(x0, block, "inc3", relu6=True, name=f"skip_out_{block_index * 3 + 1}")
        pushed.append(x0)

        x1 = conv(x0, block, "d0", relu6=True)
        x1 = bibuffer(x1, base + 0, block, "d0c1", gates[base + 0], writes[base + 0])
        x1 = bibuffer(
            x1,
            base + 1,
            block,
            "d0c2",
            gates[base + 1],
            writes[base + 1],
            name=f"skip_out_{block_index * 3 + 2}",
        )
        pushed.append(x1)

        x2 = conv(x1, block, "d1", relu6=True)
        x2 = bibuffer(x2, base + 2, block, "d1c1", gates[base + 2], writes[base + 2])
        x2 = bibuffer(x2, base + 3, block, "d1c2", gates[base + 3], writes[base + 3])
        x2 = bibuffer(x2, base + 4, block, "u2c1", gates[base + 4], writes[base + 4])
        x2 = bibuffer(x2, base + 5, block, "u2c2", gates[base + 5], writes[base + 5])
        x2 = upsample(x2, block, "u2")

        merged = g.binary("add", x2, skip3_pop, f"{block}_skip3_add")
        merged = bibuffer(merged, base + 6, block, "u1c1", gates[base + 6], writes[base + 6])
        merged = bibuffer(merged, base + 7, block, "u1c2", gates[base + 7], writes[base + 7])
        x1o = upsample(merged, block, "u1")

        y = g.binary("add", x1o, skip2_pop, f"{block}_skip2_add")
        prediction = conv(conv(y, block, "out0", relu6=True), block, "out3")

        out_channels = int(params[block]["out3"][0].shape[0])
        final = block_index == len(blocks) - 1
        if out_channels == 3:
            x = g.binary(
                "sub", skip1_pop, prediction, f"{block}_minus", name="out" if final else None
            )
        else:
            head3 = g.slice_channels(prediction, 0, 3, f"{block}_head3")
            head = g.binary("sub", skip1_pop, head3, f"{block}_minus")
            rest = g.slice_channels(prediction, 3, out_channels - 3, f"{block}_rest")
            x = g.concat_channels([head, rest], tag=f"{block}_next", name="out" if final else None)

        for index, reads, new_state in pending:
            if explicit_state:
                state_output_names.append(new_state)
            else:
                g.update_state(f"st{index}", reads, new_state)

        if island and block_index == 0:
            x = translation_island(x)

    output_names = ["out"] + [f"skip_out_{i}" for i in range(len(pushed))]
    inputs = [
        ("frame", (1, input_channels, height, width)),
        ("gate", (1, 16, 1, 1)),
        ("write", (1, 16, 1, 1)),
    ] + [(f"skip_{i}", skip_shapes[i]) for i in range(len(skip_shapes))]
    if island:
        inputs.append(("island_bias", (1, 1, 1, 1)))
    if explicit_state:
        inputs += [(f"st{i}", state_shapes[i]) for i in range(len(state_shapes))]
        return g, inputs, [], output_names + state_output_names
    states = [(f"st{i}", state_shapes[i]) for i in range(len(state_shapes))]
    return g, inputs, states, output_names


def _emit_program(
    params: _Params, input_channels: int, height: int, width: int
) -> tuple[bytes, builder.BlobFile]:
    """One BSVD step as an MLState mlprogram, on the verified spellings."""
    graph, inputs, states, output_names = _emit_graph(params, input_channels, height, width)
    model_bytes = graph.finish(inputs, states, output_names, "KinoVSR BSVD ANE")
    return model_bytes, graph.blob


def _convert(
    params: _Params, input_channels: int, height: int, width: int, directory: Path
) -> Path:

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

    def __init__(
        self,
        compiled: Path,
        compute_units: str = "ane",
        function_name: str | None = None,
        state: object | None = None,
    ):
        self.model = runtime.ModelRunner(
            compiled, compute_units, dynamic=("skip_",), function_name=function_name, state=state
        )
        for required in ("frame", "gate", "write"):
            if required not in self.model.inputs:
                raise RuntimeError(f"model is missing input '{required}'")
        if "out" not in self.model.outputs:
            raise RuntimeError("model is missing output 'out'")
        self._skips = sum(1 for n in self.model.dynamic_inputs if n.startswith("skip_"))
        ones = mx.ones((1, 16, 1, 1), dtype=mx.float16)
        mx.eval(ones)
        self._ones = memoryview(mx.contiguous(ones)).cast("B")
        self._rings = [
            [
                runtime.bind_array(self.model.dynamic_inputs[f"skip_{i}"])
                for _ in range(SKIP_DEPTH[i % 3])
            ]
            for i in range(self._skips)
        ]
        self._spares = [
            runtime.bind_array(self.model.dynamic_inputs[f"skip_{i}"]) for i in range(self._skips)
        ]
        # Ring slots are logically zero until a graph actually pushes them.
        # Keeping one immutable zero backing per line avoids physically
        # clearing hundreds of MiB at every independently reset window, and
        # lets phase-specialized graphs omit outputs for non-push steps.
        self._zeros = [
            runtime.bind_array(self.model.dynamic_inputs[f"skip_{i}"]) for i in range(self._skips)
        ]
        self._valid = [[False] * len(ring) for ring in self._rings]
        self._cursor = [0] * self._skips
        self.reset()

    def reset(self, reuse_state: bool = False) -> None:
        """Reset the stream; ``reuse_state`` keeps the current MLState.

        The phase-specialized window graphs never READ pre-window state -
        every unit primes in-graph (SSA) before its first state read and
        each dispatch clears the states it did not write - so scheduled
        windows can share one MLState for the runner's lifetime. That is
        not a luxury: allocating a fresh state per window (265 MB at
        640x480) and dropping the old one into deferred release raced the
        next window's dispatches into ANEProgramProcessRequestDirect
        status=0x16 failures at production scale. The ORDINARY gated
        graph does read initial state through its fill, so continuous
        streams keep the default fresh (zeroed) state.
        """
        if not reuse_state:
            self.model.reset_state()
        for valid in self._valid:
            valid[:] = [False] * len(valid)
        self._cursor = [0] * self._skips
        self.model.input_view("gate")[:] = self._ones
        self.model.input_view("write")[:] = self._ones

    def _input_multi(self, line: int, slot: int) -> object:
        if self._valid[line][slot]:
            return self._rings[line][slot][2]
        return self._zeros[line][2]

    def _bindings(self) -> tuple[dict[str, object], dict[str, object]]:
        features = {f"skip_{i}": self._input_multi(i, self._cursor[i]) for i in range(self._skips)}
        backings = {f"skip_out_{i}": self._spares[i][2] for i in range(self._skips)}
        return features, backings

    def predict(self) -> object:
        """One dispatch with the current bindings, no ring rotation."""
        features, backings = self._bindings()
        return self.model.predict_with(features, backings)

    def load_inputs(
        self,
        frame_bytes: bytes | memoryview,
        gate_bytes: bytes | None = None,
        write_bytes: bytes | None = None,
    ) -> None:
        """Blit one step's inputs into the bound buffers (host-side only)."""
        self.model.input_view("frame")[:] = frame_bytes
        self.model.input_view("gate")[:] = gate_bytes if gate_bytes is not None else self._ones
        self.model.input_view("write")[:] = write_bytes if write_bytes is not None else self._ones

    def dispatch(self) -> mx.array:
        """Predict on the loaded inputs and rotate the rings.

        Pure Core ML plus Python reference swaps - no MLX - so it is safe
        to run on a worker thread while the main thread owns every MLX
        operation (the AneBSVD pipelining arrangement).
        """
        self.predict()
        for i in range(self._skips):
            slot = self._cursor[i]
            self._rings[i][slot], self._spares[i] = (self._spares[i], self._rings[i][slot])
            self._valid[i][slot] = True
            self._cursor[i] = (slot + 1) % len(self._rings[i])
        return self.model.output_array("out")

    def step(
        self,
        frame_bytes: bytes | memoryview,
        gate_bytes: bytes | None = None,
        write_bytes: bytes | None = None,
    ) -> mx.array:
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

    def __init__(
        self,
        compiled: Path,
        compute_units: str = "ane",
        function_name: str | None = None,
        state: object | None = None,
    ):
        self.model = runtime.ModelRunner(
            compiled, compute_units, function_name=function_name, state=state
        )
        self._skips = sum(1 for n in self.model.inputs if n.startswith("skip_"))
        ones = mx.ones((1, 16, 1, 1), dtype=mx.float16)
        mx.eval(ones)
        self._ones = memoryview(mx.contiguous(ones)).cast("B")
        self._fifos = [
            [bytearray(len(self.model.input_view(f"skip_{i}"))) for _ in range(SKIP_DEPTH[i % 3])]
            for i in range(self._skips)
        ]
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

    def step(
        self,
        frame_bytes: bytes | memoryview,
        gate_bytes: bytes | None = None,
        write_bytes: bytes | None = None,
    ) -> mx.array:
        self.model.input_view("frame")[:] = frame_bytes
        self.model.input_view("gate")[:] = gate_bytes if gate_bytes is not None else self._ones
        self.model.input_view("write")[:] = write_bytes if write_bytes is not None else self._ones
        for i in range(self._skips):
            self.model.input_view(f"skip_{i}")[:] = self._fifos[i][self._cursor[i]]
        self.model.predict()
        for i in range(self._skips):
            self._fifos[i][self._cursor[i]][:] = self.model.output_view(f"skip_out_{i}")
            self._cursor[i] = (self._cursor[i] + 1) % len(self._fifos[i])
        return self.model.output_array("out")


# ------------------------------------------------------- verification gates


def _replay_frames(input_channels: int, height: int, width: int, count: int) -> list[mx.array]:
    base = mx.random.uniform(shape=(1, height, width, input_channels), key=mx.random.key(20260718))
    out = []
    for index in range(count):
        noise = mx.random.uniform(
            shape=(1, height, width, input_channels), key=mx.random.key(1000 + index)
        )
        frame = mx.clip(base * 0.8 + noise * 0.2 + index * 0.017, 0.0, 1.0)
        nchw = mx.contiguous(mx.transpose(frame, (0, 3, 1, 2)).astype(mx.float16))
        mx.eval(nchw)
        out.append(nchw)
    return out


class _SteppedRunner(Protocol):
    def reset(self) -> None: ...

    def step(self, frame_bytes: bytes | memoryview) -> mx.array: ...


def _drive(runner: _SteppedRunner, frames: list[mx.array]) -> list[mx.array]:
    runner.reset()
    outputs = []
    for frame in frames:
        out = runner.step(memoryview(frame).cast("B"))
        snap = out.astype(mx.float32)
        mx.eval(snap)
        outputs.append(snap)
    runner.reset()
    return outputs


def _mean_abs(a: list[mx.array], b: list[mx.array], skip_first: int) -> float:
    diffs = [mx.abs(x - y).mean() for x, y in zip(a[skip_first:], b[skip_first:], strict=True)]
    value = mx.mean(mx.stack(diffs))
    mx.eval(value)
    return float(value)


def _verify_build(
    compiled: Path,
    directory: Path,
    input_channels: int,
    height: int,
    width: int,
    placement: dict[str, int],
) -> None:
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
            f"differential canary cannot distinguish the requested ANE run "
            f"from the CPU path. Core ML CPU fp16 convolution accumulation "
            f"exceeds this recurrent graph's numerical tolerance."
        )
    byte_copy = ByteCopyBsvdRunner(compiled, "ane")
    on_reference = _drive(byte_copy, frames)
    del byte_copy
    fifo_ab = max(float(mx.abs(a - b).max()) for a, b in zip(on_ane, on_reference, strict=True))
    if fifo_ab != 0.0:
        raise RuntimeError(
            f"rebinding runner differs from the byte-copy reference (max "
            f"abs {fifo_ab:.3e}); Core ML is not honoring the ring output "
            f"backings - refusing the build."
        )
    mx.save_safetensors(
        str(directory / "replay"), {f"out_{i}": on_ane[i] for i in range(REPLAY_SKIP, REPLAY_STEPS)}
    )
    (directory / "verify.json").write_text(
        json.dumps(
            {
                "graph_version": GRAPH_VERSION,
                "placement": placement,
                "canary_separation": separation,
                "fifo_ab_max_abs": fifo_ab,
                "replay_steps": REPLAY_STEPS,
                "replay_skip": REPLAY_SKIP,
                "replay_tolerance": REPLAY_TOLERANCE,
            },
            indent=2,
        )
    )
    _log.info(
        "bsvd-ane build verified: placement %s, canary %.3e, fifo A/B %.1e",
        placement,
        separation,
        fifo_ab,
    )


def _verify_load(
    runner: BsvdRunner, directory: Path, input_channels: int, height: int, width: int
) -> None:
    record = json.loads((directory / "verify.json").read_text())
    if record.get("graph_version") != GRAPH_VERSION:
        raise RuntimeError("cache was built by a different graph version")
    stored = cast(dict[str, mx.array], mx.load(str(directory / "replay.safetensors")))
    frames = _replay_frames(input_channels, height, width, int(record["replay_steps"]))
    outputs = _drive(runner, frames)
    skip = int(record["replay_skip"])
    expected = [stored[f"out_{i}"] for i in range(skip, len(frames))]
    drift = _mean_abs(outputs[skip:], expected, 0)
    tolerance = float(record.get("replay_tolerance", REPLAY_TOLERANCE))
    if drift > tolerance:
        raise RuntimeError(
            f"replay drift {drift:.3e} exceeds {tolerance:.0e}; the model is "
            f"not producing the outputs it was verified with (CPU fallback "
            f"or a stale cache)."
        )


def build_runner(
    params: _Params, input_channels: int, height: int, width: int
) -> tuple[BsvdRunner, Path]:
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
            f"AneBSVD pads frames to the quantum - use it, or pad first."
        )
    if min(height, width) < MIN_SIDE:
        raise RuntimeError(
            f"{width}x{height} is below the verified ANE floor ({MIN_SIDE} "
            f"px); small stateful graphs convert and then fail at the first "
            f"prediction."
        )
    directory = _cache_directory(params, height, width)
    complete = all(
        (directory / name).exists()
        for name in ("model.mlpackage", "verify.json", "replay.safetensors")
    )
    if not complete:
        _log.info(
            "building BSVD ANE model for %dx%d (one-time per geometry, cached under %s)",
            width,
            height,
            directory,
        )
        package = _convert(params, input_channels, height, width, directory)
        compiled = runtime.compile_package(package)
        # Geometries above ISLAND_MIN_PIXELS legitimately place the four
        # value-exact island_* ops off the ANE; everything else must be
        # ANE-preferred as before.
        placement = runtime.assert_all_ane(
            compiled, allow_prefixes=(("island_",) if height * width >= ISLAND_MIN_PIXELS else ())
        )
        _verify_build(compiled, directory, input_channels, height, width, placement)
        _VERIFIED.add(str(directory))
    compiled = runtime.compile_package(directory / "model.mlpackage")
    runner = BsvdRunner(compiled, "ane")
    if str(directory) not in _VERIFIED:
        _verify_load(runner, directory, input_channels, height, width)
        _VERIFIED.add(str(directory))
    return runner, directory


__all__ = ["BsvdRunner", "ByteCopyBsvdRunner", "build_runner", "cache_root"]
