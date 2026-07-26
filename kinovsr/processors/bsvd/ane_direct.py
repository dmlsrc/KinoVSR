"""BSVD on the direct AppleNeuralEngine route: two exact halves.

Above the Espresso translator ceiling the Core ML route needs a
value-exact fp32 island and pays roughly 0.12 s/frame for it at 1080p
(:mod:`.ane`). This module runs the SAME step as two per-block halves
on :mod:`kinovsr.native.anemil.direct`: ``temp1`` and ``temp2`` emitted
by the shared block emitter with the recurrence as ordinary tensors
(``_emit_graph(blocks=(block,), explicit_state=True)``), no island,
chained through one shared middle IOSurface with recurrent state and
skip-history carried by rotating surface HANDLES - no tensor bytes are
copied between frames or halves. The two programs alternate within a
frame (load ~13 ms / unload ~3 ms) and are never resident together.

Numerics: the private route is byte-identical to Core ML EXECUTING THE
SAME HALVES (proven bit-exact at 640x480, where Core ML can still
co-load them, and against frozen 1080p CRC references), and the halves
land at the normal ~3e-4 mean from the fp32 reference. The islanded
single graph the Core ML route must use at these geometries does NOT:
it was measured at ~1.5e-2 mean from the fp32 net at 1080p
(2026-07-23), self-consistent under its own replay gate, so the direct
route is both faster AND materially closer to the reference there. The
one-time build gate therefore scores the chain against the fp32 net,
not against the island replay.

Verification split (deliberate): the heavy gates run ONCE per
weights+geometry at build time and are cached; a warm start performs
only the ~40 ms private-framework preflight plus the ~1 s bundle-cached
device compiles, and the first dispatch's own error status stands in
for a load canary. There is no per-load replay on this route.

Selection: :func:`should_use` engages the route where the island would
otherwise be required, i.e. padded area >= ``ISLAND_MIN_PIXELS``. The
``bsvd_direct`` setting selects ``auto`` (default), ``off`` (always Core
ML), ``require`` (error if unavailable), or ``force`` (use it at any
geometry - probe/testing aid). It can be set through ``--bsvd-direct``,
the ``[settings]`` table, or ``KINOVSR_BSVD_DIRECT``.
"""
from __future__ import annotations

import json
import logging
import shutil
import struct
from pathlib import Path

import mlx.core as mx

from kinovsr.native.anemil import direct, runtime
from kinovsr.settings import default_settings

from . import ane

_log = logging.getLogger("kinovsr.bsvd_ane")

DIRECT_GRAPH_VERSION = 1
# The direct route pays two program switches per frame; it wins exactly
# where the single-graph route pays the fp32 island instead.
DIRECT_MIN_PIXELS = ane.ISLAND_MIN_PIXELS

_ONES_16 = [1.0] * 16
_HALF_ONE = struct.pack("<e", 1.0)


def _mode() -> str:
    value = (default_settings().bsvd_direct or "auto").strip().lower()
    if value not in ("auto", "off", "require", "force"):
        raise RuntimeError(
            f"bsvd_direct={value!r}; expected auto, off, require, "
            f"or force")
    return value


def should_use(height: int, padded_width: int) -> bool:
    """Whether the direct route should serve this geometry."""
    mode = _mode()
    if mode == "off":
        return False
    if mode == "force":
        return True
    large = height * padded_width >= DIRECT_MIN_PIXELS
    if not large:
        return False
    if direct.available():
        return True
    if mode == "require":
        direct.preflight()   # raises with the reason
    _log.warning(
        "direct ANE route unavailable; falling back to the Core ML island "
        "path for %dx%d - NOTE: at these geometries the islanded graph "
        "was measured ~1.5e-2 mean from the fp32 reference (2026-07-23), "
        "well outside the direct route's ~3e-4", padded_width, height)
    return False


# ---------------------------------------------------------------- convert

def _half_stem(block: str) -> str:
    return f"direct-{block}-v{DIRECT_GRAPH_VERSION}"


def warm_names() -> list[str]:
    """Cache artifacts a warm direct start needs (for preheat checks)."""
    return [f"{_half_stem(block)}.mlmodelc" for block in ane.BLOCKS] + [
        "direct-verify.json"]


def _convert_half(params: dict, block: str, input_channels: int,
                  height: int, width: int, directory: Path) -> Path:
    from kinovsr.native.anemil import builder

    package = directory / f"{_half_stem(block)}.mlpackage"
    if package.is_dir():
        return package
    graph, inputs, states, outputs = ane._emit_graph(
        params, input_channels, height, width,
        blocks=(block,), explicit_state=True)
    assert not states
    model_bytes = graph.finish(
        inputs, [], outputs, f"KinoVSR BSVD direct {block}")
    staging = directory / f"{_half_stem(block)}.partial.mlpackage"
    shutil.rmtree(staging, ignore_errors=True)
    builder.write_package(staging, model_bytes, graph.blob)
    staging.replace(package)
    return package


# ----------------------------------------------------------------- runner

class _SkipRing:
    """Rotating skip-history line, the direct-route BsvdRunner ring.

    Slot surfaces hold pushed history; a slot whose ``valid`` flag is
    off binds the shared zero surface instead (rings are logically zero
    until the graph pushes them - nothing is memset per window). The
    pool is seeded from the line's discovered port surfaces so no
    allocation is stranded.
    """

    __slots__ = ("slots", "valid", "cursor", "spare", "zero")

    def __init__(self, skip_in: direct.Port, skip_out: direct.Port,
                 depth: int):
        nbytes = skip_in.nbytes
        self.slots = [skip_in.surface] + [direct.Surface(nbytes)
                                          for _ in range(depth - 1)]
        self.valid = [False] * depth
        self.cursor = 0
        self.spare = skip_out.surface
        self.zero = direct.Surface(nbytes)
        self.zero.zero()

    def bind(self) -> direct.Surface:
        if self.valid[self.cursor]:
            return self.slots[self.cursor]
        return self.zero

    def rotate(self) -> None:
        slot = self.cursor
        self.slots[slot], self.spare = self.spare, self.slots[slot]
        self.valid[slot] = True
        self.cursor = (slot + 1) % len(self.slots)

    def zero_last_push(self) -> None:
        self.valid[(self.cursor - 1) % len(self.slots)] = False

    def reset(self) -> None:
        self.valid = [False] * len(self.slots)
        self.cursor = 0


class DirectChainRunner:
    """Two-half BSVD chain with the :class:`.ane.BsvdRunner` contract.

    ``step``/``reset``/``zero_last_push``/``load_inputs``/``dispatch``
    match the Core ML runner so :class:`.ane.AneBSVD` drives either
    interchangeably; ``model`` is this object (``input_view``,
    ``output_array``, ``reset_state``). ``dispatch`` is pure
    ObjC/byte-copy work and safe on the dispatch worker; every MLX
    OPERATION stays with the caller.
    """

    def __init__(self, temp1: direct.DirectModel, temp2: direct.DirectModel):
        self.model = self
        self._halves = (temp1, temp2)
        temp1.discover_ports()
        middle = temp1.output("out")
        temp2.discover_ports(shared={"frame": middle.surface})

        self._frame = temp1.input("frame")
        if not self._frame.is_contiguous():
            raise RuntimeError("direct temp1 frame port is not contiguous")
        self._vectors = [(half.input("gate"), half.input("write"))
                         for half in self._halves]

        # st{i} input <-> the unit's state output, matched by BiBuffer
        # token; surfaces swap roles every dispatch.
        self._states: list[tuple[direct.Port, direct.Port]] = []
        for half in self._halves:
            for index, (token, _divisor) in enumerate(ane.BIBUF):
                state_in = half.input(f"st{index}")
                state_out = next(
                    port for port in half.outputs
                    if "_state_" in port.name and token in port.name)
                if state_in.nbytes != state_out.nbytes:
                    raise RuntimeError(
                        f"{half.label} state {index} size mismatch")
                self._states.append((state_in, state_out))

        self._rings: list[tuple[direct.Port, direct.Port, _SkipRing]] = []
        for half in self._halves:
            for line in range(3):
                skip_in = half.input(f"skip_{line}")
                skip_out = half.output(f"skip_out_{line}")
                ring = _SkipRing(skip_in, skip_out, ane.SKIP_DEPTH[line])
                self._rings.append((skip_in, skip_out, ring))

        out = temp2.output("out")
        batches, channels, depth, height, width = (
            int(d) for d in out.logical_dims())
        if batches != 1 or depth != 1:
            raise RuntimeError(
                f"direct out port has batches={batches} depth={depth}; "
                f"expected 1/1")
        self._out_port = out
        self._out_array = mx.zeros((1, channels, height, width),
                                   dtype=mx.float16)
        mx.eval(self._out_array)
        self._out_view = memoryview(self._out_array).cast("B")
        self.reset()

    # ------------------------------------------------- runner contract

    def input_view(self, name: str) -> memoryview:
        if name != "frame":
            raise KeyError(name)
        return self._frame.surface.view()[: self._frame.logical_nbytes()]

    def output_array(self, name: str):
        if name != "out":
            raise KeyError(name)
        return self._out_array

    def reset_state(self) -> None:
        for state_in, state_out in self._states:
            state_in.surface.zero()
            state_out.surface.zero()

    def reset(self, reuse_state: bool = False) -> None:
        del reuse_state   # the direct route has no shared-window state
        self.reset_state()
        for _skip_in, _skip_out, ring in self._rings:
            ring.reset()

    def zero_last_push(self, line: int) -> None:
        self._rings[line][2].zero_last_push()

    def load_inputs(self, frame_bytes, gate_bytes=None,
                    write_bytes=None) -> None:
        """Blit one step's inputs (host side, caller's thread).

        ``gate``/``write`` arrive as the product's 16-lane fp16 vectors;
        temp1 consumes lanes 0-7 and temp2 lanes 8-15, each mapped onto
        its own half's lanes 0-7.
        """
        view = self._frame.surface.view()
        view[: len(frame_bytes)] = frame_bytes
        for half_index, (gate_port, write_port) in enumerate(self._vectors):
            offset = half_index * 8
            self._write_lanes(gate_port, gate_bytes, offset)
            self._write_lanes(write_port, write_bytes, offset)

    @staticmethod
    def _write_lanes(port: direct.Port, vector_bytes, offset: int) -> None:
        _batch_s, _depth_s, plane_s, _row_s = port.strides()
        view = port.surface.view()
        for lane in range(16):
            source_lane = offset + lane
            if vector_bytes is None or lane >= 8 or source_lane >= 16:
                value = _HALF_ONE
            else:
                start = source_lane * 2
                value = bytes(vector_bytes[start: start + 2])
            position = lane * plane_s
            view[position: position + 2] = value

    def dispatch(self):
        """Run both halves and rotate the recurrence (worker-thread safe)."""
        for half_index, half in enumerate(self._halves):
            for skip_in, skip_out, ring in self._rings[
                    half_index * 3: half_index * 3 + 3]:
                skip_in.surface = ring.bind()
                skip_out.surface = ring.spare
            half.dispatch()
            base = half_index * 8
            for state_in, state_out in self._states[base: base + 8]:
                state_in.surface, state_out.surface = (state_out.surface,
                                                       state_in.surface)
            for _skip_in, _skip_out, ring in self._rings[
                    half_index * 3: half_index * 3 + 3]:
                ring.rotate()
        self._gather_out()
        return self._out_array

    def step(self, frame_bytes, gate_bytes=None, write_bytes=None):
        self.load_inputs(frame_bytes, gate_bytes, write_bytes)
        return self.dispatch()

    def close(self) -> None:
        for half in self._halves:
            half.close()

    # ---------------------------------------------------------- output

    def _gather_out(self) -> None:
        """Copy the strided out surface into the persistent mx buffer.

        Byte copies only - no MLX operations - so this is safe inside
        the dispatch worker; the buffer is valid until the next
        dispatch, the same contract as the Core ML output backing.
        """
        port = self._out_port
        surface = port.surface
        surface.lock(readonly=True)
        try:
            source = surface.view()
            if port.is_contiguous():
                self._out_view[:] = source[: port.logical_nbytes()]
                return
            batches, channels, depth, height, width = port.logical_dims()
            batch_s, depth_s, plane_s, row_s = port.strides()
            row_bytes = width * 2
            position = 0
            for b in range(batches):
                for c in range(channels):
                    for d in range(depth):
                        base = b * batch_s + d * depth_s + c * plane_s
                        for h in range(height):
                            offset = base + h * row_s
                            self._out_view[position: position + row_bytes] = \
                                source[offset: offset + row_bytes]
                            position += row_bytes
        finally:
            surface.unlock(readonly=True)


# ------------------------------------------------------------------ build

def build_direct_runner(params: dict, input_channels: int, height: int,
                        width: int) -> tuple[DirectChainRunner, Path]:
    """Convert (cached), compile, gate once, and construct the chain.

    The gate runs once per cache directory (recorded in
    ``direct-verify.json``); warm starts skip it entirely and pay only
    the preflight plus the bundle-cached device compiles.
    """
    direct.preflight()
    directory = ane._cache_directory(params, height, width)
    directory.mkdir(parents=True, exist_ok=True)

    temp2_channels = int(params["temp1"]["out3"][0].shape[0])
    compiled = {}
    for block, channels in (("temp1", input_channels),
                            ("temp2", temp2_channels)):
        package = _convert_half(params, block, channels, height, width,
                                directory)
        compiled[block] = runtime.compile_package(package)

    verify_path = directory / "direct-verify.json"
    verified = (verify_path.exists() and json.loads(
        verify_path.read_text()).get("direct_version") ==
        DIRECT_GRAPH_VERSION)
    placements = {}
    if not verified:
        placements = {
            block: runtime.assert_all_ane(compiled[block])
            for block in compiled}

    runner = DirectChainRunner(
        direct.DirectModel(compiled["temp1"], "bsvd-direct-temp1"),
        direct.DirectModel(compiled["temp2"], "bsvd-direct-temp2"))
    if not verified:
        try:
            _verify_direct_build(runner, directory, placements,
                                 input_channels, height, width, params)
        except BaseException:
            runner.close()
            raise
    return runner, directory


# Build-gate protocol: past the depth-16 fill, then bound the settled
# emitted frames against the fp32 truth. The Core ML reference REPLAY is
# deliberately NOT the oracle here: at island geometries it fingerprints
# the islanded translation, which was measured 2026-07-23 at ~1.5e-2
# mean from the fp32 net at 1080p (self-consistent, so its own replay
# gate never saw it) while these halves sit at the normal ~3e-4. The
# fill-phase raw outputs are graph-spelling fingerprints, not product
# outputs, and are excluded.
_GATE_STEPS = 26
# The fp16 fill transient decays for a few steps past the first emitted
# frame (depth 16); step 20 still measured 9.5e-4 at 1080p while 21+
# sit at the 3-4e-4 envelope. The gate scores the settled window only.
_GATE_SETTLED = 22          # compare steps [_GATE_SETTLED, _GATE_STEPS)
_GATE_TOLERANCE = 8e-4      # the parity-test envelope vs the fp32 net


def _verify_direct_build(runner: DirectChainRunner, directory: Path,
                         placements: dict, input_channels: int,
                         height: int, width: int, params: dict) -> None:
    from . import BSVD, _DenBlock

    frames = ane._replay_frames(input_channels, height, width, _GATE_STEPS)
    outputs = ane._drive(runner, frames)

    reference = BSVD.__new__(BSVD)
    reference.dtype = mx.float32
    reference.params, reference.input_channels = params, input_channels
    reference.temp1 = _DenBlock(params["temp1"])
    reference.temp2 = _DenBlock(params["temp2"])
    worst = 0.0
    for index, frame in enumerate(frames):
        truth = reference.step(
            mx.transpose(frame.astype(mx.float32), (0, 2, 3, 1)))
        if index < _GATE_SETTLED or truth is None:
            continue
        truth = mx.transpose(truth, (0, 3, 1, 2))[:, :3]
        drift = float(mx.abs(outputs[index][:, :3] - truth).mean())
        mx.eval(mx.zeros(1))
        worst = max(worst, drift)
    if worst > _GATE_TOLERANCE:
        raise RuntimeError(
            f"direct chain drifts {worst:.3e} mean from the fp32 reference "
            f"over settled steps (tolerance {_GATE_TOLERANCE:.0e}); "
            f"refusing the build.")
    (directory / "direct-verify.json").write_text(json.dumps({
        "direct_version": DIRECT_GRAPH_VERSION,
        "graph_version": ane.GRAPH_VERSION,
        "placements": placements,
        "truth_worst_mean": worst,
        "gate_steps": _GATE_STEPS,
        "gate_settled": _GATE_SETTLED,
        "gate_tolerance": _GATE_TOLERANCE,
    }, indent=2))
    _log.info("bsvd direct build verified: %.2e mean vs the fp32 reference "
              "over settled steps, placements %s", worst, placements)


__all__ = ["DirectChainRunner", "DIRECT_MIN_PIXELS", "build_direct_runner",
           "should_use"]
