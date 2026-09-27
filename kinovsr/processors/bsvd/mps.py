"""BSVD on the Neural Engine through MPSGraph.

A second ANE backend, independent of the Core ML one in ``ane.py``. The
ordinary path compiles the whole two-stage network as one MPSGraph with
ANE placement (see :mod:`kinovsr.native.mpsgraph`). Its sixteen bi-buffer
states live in four shared Metal slabs, and its six skip delay lines rotate
reusable MPSGraph tensor bindings. A graph result therefore becomes a later
feed without snapshotting the large tensor through Python/MLX.

The host replays the shared None-flow mirror
(:mod:`kinovsr.processors.bsvd.schedule`) and reads three schedules off
it each step, which is what makes an always-computing static graph
reproduce the product's fill and drain semantics exactly:

* ``drained`` gates a unit's incoming right-fold to zero, matching
  ``_BiBufferConv``'s explicit zero drain;
* ``unprimed`` gates the in-graph left-fold update to zero, matching the
  reference's zeros-at-prime initialization (without it, the first real
  step after a unit primes would read a stale left fold);
* ``pushes``/``pops`` maintain the host-side skip rings: push values are
  graph outputs, pop values are graph inputs, and a pop never consumes a
  same-step push (the shallowest queue is four steps deep by the time
  its first pop fires), so feeding pops before the run is exact;
* ``out_real`` decides whether the frame result is read and emitted.

The ``conv -> pixel_shuffle`` pairs use the bias-after-shuffle spelling
from :meth:`kinovsr.native.mpsgraph.GraphBuilder.pixel_shuffle_biased`,
avoiding the ANE fused-bias defect documented there.

**Dispatches are pipelined one step deep**, like the Core ML backend:
``step(k)`` writes the feeds for step ``k`` (into the next parity's
buffers, overlapping the in-flight dispatch), collects dispatch ``k-1``,
and submits dispatch ``k`` - so the ANE never idles long enough to pay
its post-idle power ramp, and the prediction overlaps the caller's other
per-frame work.  This adds one step of output delay: ``SHIFT_NUM`` is 17
(the network's 16 plus one dispatch in flight).  The skip rings tolerate
it because a pop never needs a push younger than four steps.

Like the Core ML backend this executes fp16 only and fails loudly.
``--gop-align`` windows use one schedule-generic direct ANECIR entry. It
keeps persistent ANE state and stable IOSurface skip-ring bindings through
fill, steady state, drain, and window resets without calling Core ML or
switching raw programs. Its one-step cadence preserves downstream GPU overlap.
"""

import contextlib
import hashlib
import logging
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from pathlib import Path
from typing import ClassVar, cast

import mlx.core as mx

from kinovsr.native import mpsgraph as mg
from kinovsr.native import mpsgraph_state as mgs
from kinovsr.native.dispatch import (
    QOS_CLASS_USER_INITIATED,
    DispatchPipeline,
)
from kinovsr.processors.bsvd.schedule import NoneFlowNet, StepRecord
from kinovsr.processors.feed_driver import WindowRouteUnavailable
from kinovsr.settings import default_settings

from .mps_layout import (
    _LINES,
    _PUSH_OUTPUTS,
    _SKIP_DEPTHS,
    _STATE_GROUP_OF,
    _STATE_GROUPS,
    _UNITS,
    _net_keys,
)
from .mps_phases import (
    ScheduledMpsPhaseSuite,
    WindowMachine,
    _Binding,
    _Executable,
    preload_stateful_executable,
    stateful_cache_ready,
)
from .net import _block_prefix, _pad_width_reflect

_log = logging.getLogger(__name__)

_ANE_SESSION_MIN_PIXELS = 128 * 256
_PHASE_MAX_WIDTH = 1024
_PHASE_MAX_HEIGHT = 576
_GRAPH_CACHE_VERSION = 3
# The ordinary graph's first Neural Engine inference aborts the process
# (MPSGraph's ANERegion.mm assertion, Code=-19) when the graph width is a
# multiple of 32 but not of 128. A 2026-09-25 sweep on macOS 27 (M1 Max, 80
# geometries) found all 20 such widths it tried, from 32 to 1440, aborting,
# while all 10 multiples of 128 up to 1920 and 30 other widths ran; height
# (32 to 1080) did not matter, and the other widths compared (112 to 720)
# matched the MLX reference as closely as aligned ones. Frames in that class
# are reflect-padded on the right to the next multiple of 128, the quantum
# AneBSVD pads every width to, and the outputs are cropped back.
_ABORTING_WIDTH_STEP = 32
_ANE_WIDTH_QUANTUM = 128


def _graph_width(width: int) -> int:
    """The width the ordinary graph runs ``width``-wide frames at."""
    if width % _ABORTING_WIDTH_STEP or width % _ANE_WIDTH_QUANTUM == 0:
        return width
    return -(-width // _ANE_WIDTH_QUANTUM) * _ANE_WIDTH_QUANTUM


def _weights_key(weights: dict[str, mx.array], prefixes: tuple[str, str]) -> str:
    """Short deterministic cache key for one converted BSVD checkpoint."""
    digest = hashlib.sha256()
    for name in sorted(weights):
        value = weights[name]
        digest.update(f"{name}.{tuple(int(s) for s in value.shape)}".encode())
    sample = weights[f"{prefixes[0]}.inc.convblock.0.weight"].reshape(-1)[:64].astype(mx.float32)
    mx.eval(sample)
    digest.update(bytes(memoryview(mx.contiguous(sample)).cast("B")))
    return digest.hexdigest()[:16]


class _Emitted:
    """Tensors one ``_emit_net`` call contributes to the target list."""

    __slots__ = ("out", "out_channels", "x0", "x1")

    def __init__(self, out: object, x0: object, x1: object, out_channels: int) -> None:
        self.out = out
        self.x0 = x0
        self.x1 = x1
        self.out_channels = out_channels


class MpsGraphBSVD:
    """Drop-in for :class:`kinovsr.processors.bsvd.BSVD` via MPSGraph/ANE.

    Same contract shape: ``step(frame_or_none)`` once per stream item
    with NHWC (1, H, W, C) inputs whose sides are multiples of four
    (``BsvdDenoiser`` pads them so), ``None`` through the fill, real
    outputs through the drain, ``reset()`` between streams.  The graph is
    compiled lazily at the first real frame, when the geometry is known;
    construction failures raise - this backend is explicitly requested
    and never falls back. Widths the Neural Engine cannot run are padded
    and cropped back (see ``_graph_width``).
    """

    SHIFT_NUM = 17  # 16-step BiBuffer delay + 1 dispatch in flight
    MIN_WINDOW_FRAMES = 1
    # Set for the process when the direct schedule entry cannot be built
    # (macOS 27 places ANE regions in a form its lowering cannot read); every
    # later window then runs through the per-step graph, with the same output.
    _schedule_entry_unavailable: ClassVar[bool] = False

    def __init__(
        self,
        weights_path: str | Path,
        dtype: mx.Dtype = mx.float16,
        *,
        loopback_in_place: bool = True,
        synchronize_results: bool = False,
        use_command_queue: bool = True,
        ane_fw_to_fw_signal: bool = False,
        ane_late_latch: bool = False,
        ane_streaming_session: bool = True,
        ane_energy_efficient: bool = True,
    ):
        if dtype != mx.float16:
            raise ValueError(
                "the BSVD MPSGraph backend executes fp16 only; use "
                "--bsvd-dtype float16 or --bsvd-backend mlx"
            )

        self.dtype = mx.float16
        raw = cast(dict[str, mx.array], mx.load(str(weights_path)))
        self._prefixes = (_block_prefix(raw, 0), _block_prefix(raw, 1))
        self._loopback_in_place = loopback_in_place
        self._synchronize_results = synchronize_results
        self._use_command_queue = use_command_queue
        self._ane_fw_to_fw_signal = ane_fw_to_fw_signal
        self._ane_late_latch = ane_late_latch
        self._ane_streaming_session = ane_streaming_session
        self._ane_energy_efficient = ane_energy_efficient
        self._weights: dict[str, mx.array] | None = {
            key: value.astype(mx.float32) for key, value in raw.items()
        }
        self._weights_cache_key = _weights_key(self._weights, self._prefixes)
        inc0 = self._weights[f"{self._prefixes[0]}.inc.convblock.0.weight"]
        self.input_channels = int(inc0.shape[1])

        self._graph: mg.CompiledGraph | None = None
        self._phase_suite: ScheduledMpsPhaseSuite | None = None
        self._preheat: Future[_Executable | None] | None = None
        self._preheat_pool: ThreadPoolExecutor | None = None
        self._preheated_geometry: tuple[int, int] | None = None
        self._preheated_executable: _Executable | None = None
        self._geometry: tuple[int, int] | None = None
        self._frame_width = 0
        self._graph_width = 0
        self._line_shapes: list[tuple[int, ...]] = []
        self._state_slots: list[mg.TensorBinding] = []
        self._state_bindings: dict[str, mg.TensorBinding] = {}
        self._mirror = NoneFlowNet()
        self._rings: list[deque[_Binding]] = [deque() for _ in range(_LINES)]
        self._slots: list[list[_Binding]] = [[] for _ in range(_LINES)]
        self._free: list[deque[_Binding]] = [deque() for _ in range(_LINES)]
        self._zero_bindings: list[_Binding | None] = [None] * _LINES
        self._discard_bindings: list[_Binding | None] = [None] * _LINES
        self._gate_on = mx.ones((1, 1, 1, 1), dtype=mx.float16)
        self._gate_off = mx.zeros((1, 1, 1, 1), dtype=mx.float16)
        self._zero_frame: mx.array | None = None
        # MPSGraph does not expose _ANEInMemoryModel's integer program QoS.
        # Keep its submission worker at USER_INITIATED instead: a low caller
        # QoS also deprioritizes the accelerator request before it reaches
        # ANE's own queues.
        self._pipeline = DispatchPipeline(
            "bsvd-mpsgraph-dispatch",
            qos_class=QOS_CLASS_USER_INITIATED,
        )
        # The submitted-but-uncollected step: (record, bindings popped as
        # inputs, bindings receiving real pushed results).
        self._pending: tuple[StepRecord, list[_Binding | None], list[_Binding | None]] | None = None
        self._dirty = False
        self._closed = False

    # ----------------------------------------------------------- helpers

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("BSVD MPSGraph backend is closed")

    def _executable_cache(self, label: str, height: int, width: int) -> Path | None:
        # Tiny test/preview graphs compile faster than package bookkeeping
        # and do not benefit from the ANE streaming-session path.
        if height * width < _ANE_SESSION_MIN_PIXELS:
            return None

        root = Path(default_settings().cache_dir).expanduser()
        geometry = f"{self._weights_cache_key}-{width}x{height}-v{_GRAPH_CACHE_VERSION}"
        return root / "bsvd-mpsgraph" / geometry / label

    # ------------------------------------------------------- graph build

    def _bibuf(
        self,
        b: mg.GraphBuilder,
        x: object,
        unit: int,
        key: str,
        h: int,
        w: int,
        state_inputs: dict[int, tuple[object, tuple[int, ...], int]],
        state_updates: dict[int, list[object]],
    ) -> object:
        assert self._weights is not None
        weight = self._weights[key + ".weight"]
        channels = int(weight.shape[0])
        fold = channels // 8
        group, slot = _STATE_GROUP_OF[unit]
        state_channels = fold + channels
        if group not in state_inputs:
            shape = (1, state_channels * len(_STATE_GROUPS[group]), h, w)
            state_inputs[group] = (
                b.placeholder(shape, f"g{group}.state"),
                shape,
                state_channels,
            )
            state_updates[group] = [None] * len(_STATE_GROUPS[group])
        slab, slab_shape, slab_channels = state_inputs[group]
        if slab_shape[2:] != (h, w) or slab_channels != state_channels:
            raise RuntimeError(f"BSVD state group {group} mixes incompatible unit shapes")
        packed = b.slice_channels(slab, slot * state_channels, state_channels, f"u{unit}.state")
        left = b.slice_channels(packed, 0, fold, f"u{unit}.left")
        center = b.slice_channels(packed, fold, channels, f"u{unit}.center")
        gate = b.placeholder((1, 1, 1, 1), f"u{unit}.gate")
        leftgate = b.placeholder((1, 1, 1, 1), f"u{unit}.leftgate")
        right = b.multiply(b.slice_channels(x, 0, fold, f"u{unit}.r"), gate, f"u{unit}.rg")
        rest = b.slice_channels(center, 2 * fold, channels - 2 * fold, f"u{unit}.rest")
        merged = b.concat_channels([right, left, rest], f"u{unit}.cat")
        new_left = b.multiply(
            b.slice_channels(center, fold, fold, f"u{unit}.cslice"), leftgate, f"u{unit}.newleft"
        )
        state_updates[group][slot] = b.concat_channels([new_left, x], f"u{unit}.state.pack")
        y = b.conv2d(merged, weight, self._weights[key + ".bias"], name=f"u{unit}.conv")
        return b.clamp(y, 0.0, 6.0, f"u{unit}.relu6")

    def _emit_net(
        self,
        b: mg.GraphBuilder,
        net: int,
        x: object,
        hp: int,
        wp: int,
        state_inputs: dict[int, tuple[object, tuple[int, ...], int]],
        state_updates: dict[int, list[object]],
    ) -> _Emitted:
        keys = _net_keys(self._prefixes[net])
        assert self._weights is not None
        w = self._weights
        h2, w2, h4, w4 = hp // 2, wp // 2, hp // 4, wp // 4
        base = net * 8
        n = f"n{net}"

        def conv(x: object, key: str, stride: int = 1) -> object:
            return b.conv2d(
                x,
                w[keys[key] + ".weight"],
                w[keys[key] + ".bias"],
                stride=stride,
                name=f"{n}.{key}",
            )

        def relu6(x: object, tag: str) -> object:
            return b.clamp(x, 0.0, 6.0, f"{n}.{tag}")

        sp1 = b.placeholder((1, 3, hp, wp), f"s{net * 3}.pop")
        sp2_c = int(w[keys["inc3"] + ".weight"].shape[0])
        sp2 = b.placeholder((1, sp2_c, hp, wp), f"s{net * 3 + 1}.pop")
        sp3_c = int(w[keys["u0"] + ".weight"].shape[0])
        sp3 = b.placeholder((1, sp3_c, h2, w2), f"s{net * 3 + 2}.pop")

        x0 = relu6(conv(relu6(conv(x, "inc0"), "i0r"), "inc3"), "i3r")
        d = relu6(conv(x0, "d0", stride=2), "d0r")
        y = self._bibuf(b, d, base + 0, keys["u0"], h2, w2, state_inputs, state_updates)
        x1 = self._bibuf(b, y, base + 1, keys["u1"], h2, w2, state_inputs, state_updates)
        d = relu6(conv(x1, "d1", stride=2), "d1r")
        y = self._bibuf(b, d, base + 2, keys["u2"], h4, w4, state_inputs, state_updates)
        x2 = self._bibuf(b, y, base + 3, keys["u3"], h4, w4, state_inputs, state_updates)
        m = self._bibuf(b, x2, base + 4, keys["u4"], h4, w4, state_inputs, state_updates)
        m = self._bibuf(b, m, base + 5, keys["u5"], h4, w4, state_inputs, state_updates)
        up2_w = w[keys["up2"] + ".weight"]
        u = b.conv2d(m, up2_w, None, name=f"{n}.up2")
        x2u = b.pixel_shuffle_biased(
            u,
            w[keys["up2"] + ".bias"],
            channels=int(up2_w.shape[0]),
            height=h4,
            width=w4,
            name=f"{n}.ps2",
        )
        s3 = b.add(x2u, sp3, f"{n}.k3")
        m = self._bibuf(b, s3, base + 6, keys["u6"], h2, w2, state_inputs, state_updates)
        m = self._bibuf(b, m, base + 7, keys["u7"], h2, w2, state_inputs, state_updates)
        up1_w = w[keys["up1"] + ".weight"]
        u = b.conv2d(m, up1_w, None, name=f"{n}.up1")
        x1u = b.pixel_shuffle_biased(
            u,
            w[keys["up1"] + ".bias"],
            channels=int(up1_w.shape[0]),
            height=h2,
            width=w2,
            name=f"{n}.ps1",
        )
        s2 = b.add(x1u, sp2, f"{n}.k2")
        o = relu6(conv(s2, "out0"), "o0r")
        pred = conv(o, "out3")
        out_channels = int(w[keys["out3"] + ".weight"].shape[0])
        head = b.subtract(sp1, b.slice_channels(pred, 0, 3, f"{n}.ph"), f"{n}.res")
        if out_channels == 3:
            out = head
        else:
            out = b.concat_channels(
                [head, b.slice_channels(pred, 3, out_channels - 3, f"{n}.pt")], f"{n}.oc"
            )
        self._line_shapes += [(1, 3, hp, wp), (1, sp2_c, hp, wp), (1, sp3_c, h2, w2)]
        return _Emitted(out, x0, x1, out_channels)

    def _compile(self, hp: int, wp: int) -> None:
        if hp % 4 or wp % 4:
            raise ValueError(f"BSVD MPSGraph needs sides divisible by four, got {wp}x{hp}")
        builder = mg.GraphBuilder(mg.FLOAT16)
        self._line_shapes = []
        state_inputs: dict[int, tuple[object, tuple[int, ...], int]] = {}
        state_updates: dict[int, list[object]] = {}
        frame = builder.placeholder((1, self.input_channels, hp, wp), "frame")
        first = self._emit_net(builder, 0, frame, hp, wp, state_inputs, state_updates)
        second = self._emit_net(builder, 1, first.out, hp, wp, state_inputs, state_updates)
        targets: list[tuple[str, object, tuple[int, ...]]] = [
            ("out", second.out, (1, second.out_channels, hp, wp))
        ]
        for group in range(len(_STATE_GROUPS)):
            _slab, shape, _channels = state_inputs[group]
            updates = state_updates[group]
            if any(value is None for value in updates):
                raise RuntimeError(f"BSVD state group {group} was not fully updated")
            targets.append(
                (
                    f"g{group}.state.next",
                    builder.concat_channels(updates, f"g{group}.state.next.pack"),
                    shape,
                )
            )
        out3 = builder.slice_channels(first.out, 0, 3, "n0.out3")
        for name, tensor, shape in (
            ("n0.x0", first.x0, self._line_shapes[1]),
            ("n0.x1", first.x1, self._line_shapes[2]),
            ("n0.out3", out3, (1, 3, hp, wp)),
            ("n1.x0", second.x0, self._line_shapes[4]),
            ("n1.x1", second.x1, self._line_shapes[5]),
        ):
            targets.append((name, tensor, shape))
        dynamic = {f"s{line}.pop" for line in range(_LINES)}
        dynamic.update(name for _, name in _PUSH_OUTPUTS)
        for group in range(len(_STATE_GROUPS)):
            dynamic.add(f"g{group}.state")
            dynamic.add(f"g{group}.state.next")
        self._graph = mg.compile_graph(
            builder,
            targets,
            device=mg.DEVICE_ANE,
            dynamic=dynamic,
            synchronize_results=(self._synchronize_results),
            use_command_queue=(self._use_command_queue),
            ane_fw_to_fw_signal=(self._ane_fw_to_fw_signal),
            ane_late_latch=(self._ane_late_latch),
            ane_streaming_session=(
                self._ane_streaming_session and hp * wp >= _ANE_SESSION_MIN_PIXELS
            ),
            ane_energy_efficient=(self._ane_energy_efficient),
            executable_cache=(self._executable_cache("ordinary", hp, wp)),
        )
        self._state_slots = []
        self._state_bindings = {}
        for group in range(len(_STATE_GROUPS)):
            feed_name = f"g{group}.state"
            result_name = f"g{group}.state.next"
            slot = self._graph.bind(feed_name)
            slot.write(bytes(2 * prod(slot.shape)))
            self._state_slots.append(slot)
            self._state_bindings[feed_name] = slot
            self._state_bindings[result_name] = self._graph.bind(result_name, shared=slot)
        self._slots = []
        self._free = []
        self._zero_bindings = []
        self._discard_bindings = []
        for line, depth in enumerate(_SKIP_DEPTHS):
            name = f"s{line}.pop"
            slots: list[_Binding] = [self._graph.bind(name) for _ in range(depth + 1)]
            zero = self._graph.bind(name)
            zero.write(bytes(2 * prod(self._line_shapes[line])))
            self._slots.append(slots)
            self._free.append(deque(slots))
            self._zero_bindings.append(zero)
            self._discard_bindings.append(None if line == 0 else self._graph.bind(name))
        self._zero_frame = mx.zeros((1, self.input_channels, hp, wp), dtype=mx.float16)

    def _ensure_graph(self, hp: int, wp: int) -> None:
        if self._geometry is not None:
            if self._geometry != (hp, wp):
                raise RuntimeError(
                    f"BSVD MPSGraph stream changed resolution from {self._geometry} to {(hp, wp)}"
                )
            return
        graph_width = _graph_width(wp)
        if graph_width - wp >= wp:
            raise ValueError(
                f"BSVD MPSGraph cannot run {wp} px wide frames: the Neural Engine "
                f"needs {graph_width} px, more than reflect padding can supply; "
                f"use --bsvd-backend mlx"
            )
        self._compile(hp, graph_width)
        self._geometry = (hp, wp)
        self._frame_width = wp
        self._graph_width = graph_width

    def window_capable(self, height: int, width: int) -> bool:
        """Whether direct schedule chunks cover this padded geometry.

        The schedule entry does not pad, so widths the ordinary graph pads
        run their windows through ``step()``.
        """
        return (
            not type(self)._schedule_entry_unavailable
            and height % 4 == 0
            and width % 4 == 0
            and _graph_width(width) == width
            and height * width >= _ANE_SESSION_MIN_PIXELS
            and height <= _PHASE_MAX_HEIGHT
            and width <= _PHASE_MAX_WIDTH
        )

    def _join_preheat(self) -> None:
        future, self._preheat = self._preheat, None
        pool, self._preheat_pool = self._preheat_pool, None
        try:
            if future is not None:
                executable = future.result()
                if executable is not None:
                    self._preheated_executable = executable
        finally:
            if pool is not None:
                pool.shutdown(wait=False)

    def preheat(self, height: int, width: int, scheduled: bool = False) -> None:
        """Open a warm direct entry cache while source startup proceeds."""
        if (
            self._closed
            or not scheduled
            or self._preheat is not None
            or self._phase_suite is not None
            or self._preheated_executable is not None
            or not self.window_capable(height, width)
        ):
            return

        if not stateful_cache_ready(self, height, width):
            return

        self._preheated_geometry = (height, width)
        self._preheat_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bsvd-mpsgraph-preheat"
        )
        self._preheat = self._preheat_pool.submit(preload_stateful_executable, self, height, width)

    def begin_window(self, frames: list[mx.array]) -> WindowMachine:
        """Start one window on the persistent direct ANE entry."""
        self._require_open()
        if not frames:
            raise ValueError("BSVD MPSGraph window cannot be empty")
        if self._dirty or self._pending is not None:
            raise RuntimeError("reset BSVD MPSGraph before running a schedule window")
        first = frames[0]
        height, width = int(first.shape[1]), int(first.shape[2])
        if type(self)._schedule_entry_unavailable:
            raise WindowRouteUnavailable("MPSGraph schedule windows are unavailable here")
        if not self.window_capable(height, width):
            raise RuntimeError(
                f"{width}x{height} is outside the MPSGraph schedule-window "
                f"envelope (at most {_PHASE_MAX_WIDTH}x{_PHASE_MAX_HEIGHT}, "
                f"at a width the graph runs unpadded); route this window "
                f"through step() instead"
            )
        for frame in frames:
            if (int(frame.shape[1]), int(frame.shape[2])) != (height, width):
                raise RuntimeError("BSVD MPSGraph schedule window changed resolution")
        if self._geometry is not None and self._geometry != (height, width):
            raise RuntimeError(
                f"BSVD MPSGraph stream changed resolution from "
                f"{self._geometry} to {(height, width)}"
            )
        self._join_preheat()
        if self._phase_suite is None:
            executable = self._preheated_executable
            if executable is not None:
                if self._preheated_geometry != (height, width):
                    raise RuntimeError("preheated MPSGraph entry changed resolution")
                self._preheated_executable = None
            try:
                self._phase_suite = ScheduledMpsPhaseSuite(
                    self, height, width, executable=executable
                )
            except mgs.PlacedFormUnsupported as exc:
                if executable is not None:
                    executable.close()
                type(self)._schedule_entry_unavailable = True
                _log.warning(
                    "MPSGraph schedule windows are unavailable on this system (%s); "
                    "--gop-align windows run through the per-step graph instead, "
                    "which is slower and gives the same output",
                    exc,
                )
                raise WindowRouteUnavailable(str(exc)) from exc
            except BaseException:
                if executable is not None:
                    executable.close()
                raise
            self._geometry = (height, width)
            self._frame_width = self._graph_width = width
        self._dirty = True
        return self._phase_suite.machine(frames)

    # ---------------------------------------------------------- stepping

    def _fixed_feeds(self, frame: mx.array, record: StepRecord) -> dict[str, mx.array]:
        feeds: dict[str, mx.array] = {"frame": frame}
        for unit in range(_UNITS):
            feeds[f"u{unit}.gate"] = self._gate_off if record.drained[unit] else self._gate_on
            feeds[f"u{unit}.leftgate"] = self._gate_off if record.unprimed[unit] else self._gate_on
        return feeds

    def _pop_bindings(
        self, record: StepRecord
    ) -> tuple[dict[str, _Binding | None], list[_Binding | None]]:
        bindings: dict[str, _Binding | None] = dict(self._state_bindings)
        popped: list[_Binding | None] = [None] * _LINES
        for line in range(_LINES):
            if record.pops[line]:
                if not self._rings[line]:
                    raise RuntimeError(
                        f"BSVD MPSGraph skip ring {line} underran; the "
                        f"schedule mirror desynchronized"
                    )
                binding: _Binding | None = self._rings[line].popleft()
                popped[line] = binding
            else:
                binding = self._zero_bindings[line]
            bindings[f"s{line}.pop"] = binding
        return bindings, popped

    def _push_bindings(
        self, frame: mx.array, record: StepRecord, bindings: dict[str, _Binding | None]
    ) -> list[_Binding | None]:
        pushed: list[_Binding | None] = [None] * _LINES
        binding: _Binding | None
        if record.pushes[0]:
            if not self._free[0]:
                raise RuntimeError("BSVD MPSGraph skip ring 0 exhausted")
            binding = self._free[0].popleft()
            binding.write(frame[:, :3])
            pushed[0] = binding
        for line, name in _PUSH_OUTPUTS:
            if record.pushes[line]:
                if not self._free[line]:
                    raise RuntimeError(f"BSVD MPSGraph skip ring {line} exhausted")
                binding = self._free[line].popleft()
                pushed[line] = binding
            else:
                binding = self._discard_bindings[line]
            bindings[name] = binding
        return pushed

    def _collect(self) -> mx.array | None:
        """Join the in-flight dispatch; ring-push and emit its step."""
        if self._pending is None:
            return None
        (record, popped, pushed), self._pending = self._pending, None
        self._pipeline.join()
        assert self._graph is not None
        outputs = self._graph.read({"out"} if record.out_real else set())
        for line in range(_LINES):
            if popped[line] is not None:
                self._free[line].append(cast(_Binding, popped[line]))  # checked just above
            if pushed[line] is not None:
                self._rings[line].append(cast(_Binding, pushed[line]))  # checked just above
        if not record.out_real:
            return None
        out = mx.transpose(outputs["out"], (0, 2, 3, 1))
        if self._frame_width == self._graph_width:
            return out
        return out[:, :, : self._frame_width, :]

    def step(self, x: mx.array | None) -> mx.array | None:
        self._require_open()
        if x is None:
            if self._graph is None:
                return None  # drained before any input frame
            assert self._zero_frame is not None
            frame = self._zero_frame
            record = self._mirror.step(False)
        else:
            height, width = int(x.shape[1]), int(x.shape[2])
            self._ensure_graph(height, width)
            padded = _pad_width_reflect(x.astype(mx.float16), self._graph_width)
            frame = mx.contiguous(mx.transpose(padded, (0, 3, 1, 2)))
            record = self._mirror.step(True)
        assert self._graph is not None
        # Fixed feeds for THIS step go into the next parity's buffers while
        # the previous dispatch still runs. Dynamic pop bindings are chosen
        # before collecting its pushes - safe, a pop never needs a push
        # younger than four steps and the pipeline is one deep.
        self._graph.write_feeds(self._fixed_feeds(frame, record))
        bindings, popped = self._pop_bindings(record)
        previous = self._collect()
        pushed = self._push_bindings(frame, record, bindings)
        self._pending = (record, popped, pushed)
        self._pipeline.submit(
            self._graph.begin_dispatch(
                cast(dict[str, mg.TensorBinding], bindings)  # all made by self._graph.bind
            )
        )
        self._dirty = True
        return previous

    def reset(self) -> None:
        self._require_open()
        self._pipeline.drain()
        self._pending = None
        self._mirror = NoneFlowNet()
        self._dirty = False
        self._rings = [deque() for _ in range(_LINES)]
        self._free = [deque(slots) for slots in self._slots]
        for slot in self._state_slots:
            slot.write(bytes(2 * prod(slot.shape)))
        if self._phase_suite is not None:
            self._phase_suite.reset()
        if self._graph is not None:
            self._graph.reset()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            with contextlib.suppress(BaseException):
                self._join_preheat()
            self._pipeline.drain()
        finally:
            self._pipeline.close()
            self._pending = None
            if self._preheated_executable is not None:
                self._preheated_executable.close()
            self._preheated_executable = None
            self._preheated_geometry = None
            if self._phase_suite is not None:
                self._phase_suite.close()
            self._phase_suite = None
            if self._graph is not None:
                self._graph.close()
            self._graph = None
            self._weights = None
            self._rings = [deque() for _ in range(_LINES)]
            self._state_slots = []
            self._state_bindings = {}
            self._slots = [[] for _ in range(_LINES)]
            self._free = [deque() for _ in range(_LINES)]
            self._zero_bindings = [None] * _LINES
            self._discard_bindings = [None] * _LINES
            self._zero_frame = None


__all__ = ["MpsGraphBSVD"]
