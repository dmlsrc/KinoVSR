"""BSVD on the Neural Engine through MPSGraph.

A second ANE backend, independent of the Core ML one in ``ane.py``: the
whole two-stage network is one MPSGraph compiled with the ANE placement
pass (see :mod:`kinovsr.native.mpsgraph`).  The sixteen bi-buffer states
stay resident on the device through the compiled graph's loopback
bindings - each unit's arriving tensor is a graph result bound to become
the next step's ``center`` input, and the left-fold update is computed
in-graph as a gated slice of the current center - so a step's host
traffic is one frame in, per-unit gate scalars, the popped skip values
in, and the emitted frame plus pushed skip values out.

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

Like the Core ML backend this executes fp16 only and fails loudly; there
is no phase-specialized window path (``--gop-align`` windows route
through the ordinary per-step path).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import mlx.core as mx

from kinovsr.native import mpsgraph as mg
from kinovsr.native.dispatch import DispatchPipeline
from kinovsr.processors.bsvd.schedule import NoneFlowNet

_UNITS = 16
_LINES = 6
# Skip lines 1..5 push graph outputs; line 0 pushes the frame's own RGB
# head, which the host already holds.
_PUSH_OUTPUTS = ((1, "n0.x0"), (2, "n0.x1"), (3, "n0.out3"),
                 (4, "n1.x0"), (5, "n1.x1"))


def _net_keys(prefix: str) -> dict[str, str]:
    c = f"{prefix}.%s.convblock"
    return {
        "inc0": f"{c % 'inc'}.0", "inc3": f"{c % 'inc'}.3",
        "d0": f"{c % 'downc0'}.0",
        "u0": f"{c % 'downc0'}.3.c1.net", "u1": f"{c % 'downc0'}.3.c2.net",
        "d1": f"{c % 'downc1'}.0",
        "u2": f"{c % 'downc1'}.3.c1.net", "u3": f"{c % 'downc1'}.3.c2.net",
        "u4": f"{c % 'upc2'}.0.c1.net", "u5": f"{c % 'upc2'}.0.c2.net",
        "up2": f"{c % 'upc2'}.1",
        "u6": f"{c % 'upc1'}.0.c1.net", "u7": f"{c % 'upc1'}.0.c2.net",
        "up1": f"{c % 'upc1'}.1",
        "out0": f"{c % 'outc'}.0", "out3": f"{c % 'outc'}.3",
    }


class _Emitted:
    """Tensors one ``_emit_net`` call contributes to the target list."""

    __slots__ = ("out", "x0", "x1", "out_channels")

    def __init__(self, out, x0, x1, out_channels):
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
    and never falls back.
    """

    SHIFT_NUM = 17  # 16-step BiBuffer delay + 1 dispatch in flight

    def __init__(self, weights_path: str | Path, dtype: Any = mx.float16):
        if dtype != mx.float16:
            raise ValueError(
                "the BSVD MPSGraph backend executes fp16 only; use "
                "--bsvd-dtype float16 or --bsvd-backend mlx")
        from . import _block_prefix

        self.dtype = mx.float16
        raw = mx.load(str(weights_path))
        self._prefixes = (_block_prefix(raw, 0), _block_prefix(raw, 1))
        self._weights: dict[str, Any] | None = {
            key: value.astype(mx.float32) for key, value in raw.items()}
        inc0 = self._weights[f"{self._prefixes[0]}.inc.convblock.0.weight"]
        self.input_channels = int(inc0.shape[1])

        self._graph: mg.CompiledGraph | None = None
        self._geometry: tuple[int, int] | None = None
        self._line_shapes: list[tuple[int, ...]] = []
        self._mirror = NoneFlowNet()
        self._rings: list[deque] = [deque() for _ in range(_LINES)]
        self._zeros: dict[tuple[int, ...], Any] = {}
        self._gate_on = mx.ones((1, 1, 1, 1), dtype=mx.float16)
        self._gate_off = mx.zeros((1, 1, 1, 1), dtype=mx.float16)
        self._zero_frame: Any | None = None
        self._pipeline = DispatchPipeline("bsvd-mpsgraph-dispatch")
        # The submitted-but-uncollected step: (record, frame RGB head for
        # skip line 0 when that push is real, else None).
        self._pending: tuple[Any, Any | None] | None = None
        self._closed = False

    # ----------------------------------------------------------- helpers

    def _zero(self, shape: tuple[int, ...]) -> Any:
        if shape not in self._zeros:
            self._zeros[shape] = mx.zeros(shape, dtype=mx.float16)
        return self._zeros[shape]

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("BSVD MPSGraph backend is closed")

    # ------------------------------------------------------- graph build

    def _bibuf(self, b: mg.GraphBuilder, x: Any, unit: int, key: str,
               h: int, w: int, state: list) -> Any:
        weight = self._weights[key + ".weight"]
        channels = int(weight.shape[0])
        fold = channels // 8
        left = b.placeholder((1, fold, h, w), f"u{unit}.left")
        center = b.placeholder((1, channels, h, w), f"u{unit}.center")
        gate = b.placeholder((1, 1, 1, 1), f"u{unit}.gate")
        leftgate = b.placeholder((1, 1, 1, 1), f"u{unit}.leftgate")
        right = b.multiply(b.slice_channels(x, 0, fold, f"u{unit}.r"),
                           gate, f"u{unit}.rg")
        rest = b.slice_channels(center, 2 * fold, channels - 2 * fold,
                                f"u{unit}.rest")
        merged = b.concat_channels([right, left, rest], f"u{unit}.cat")
        new_left = b.multiply(
            b.slice_channels(center, fold, fold, f"u{unit}.cslice"),
            leftgate, f"u{unit}.newleft")
        state.append((f"u{unit}.arrival", x, (1, channels, h, w)))
        state.append((f"u{unit}.newleft", new_left, (1, fold, h, w)))
        y = b.conv2d(merged, weight, self._weights[key + ".bias"],
                     name=f"u{unit}.conv")
        return b.clamp(y, 0.0, 6.0, f"u{unit}.relu6")

    def _emit_net(self, b: mg.GraphBuilder, net: int, x: Any,
                  hp: int, wp: int, state: list) -> _Emitted:
        keys = _net_keys(self._prefixes[net])
        w = self._weights
        h2, w2, h4, w4 = hp // 2, wp // 2, hp // 4, wp // 4
        base = net * 8
        n = f"n{net}"

        def conv(x, key, stride=1):
            return b.conv2d(x, w[keys[key] + ".weight"],
                            w[keys[key] + ".bias"], stride=stride,
                            name=f"{n}.{key}")

        def relu6(x, tag):
            return b.clamp(x, 0.0, 6.0, f"{n}.{tag}")

        sp1 = b.placeholder((1, 3, hp, wp), f"s{net * 3}.pop")
        sp2_c = int(w[keys["inc3"] + ".weight"].shape[0])
        sp2 = b.placeholder((1, sp2_c, hp, wp), f"s{net * 3 + 1}.pop")
        sp3_c = int(w[keys["u0"] + ".weight"].shape[0])
        sp3 = b.placeholder((1, sp3_c, h2, w2), f"s{net * 3 + 2}.pop")

        x0 = relu6(conv(relu6(conv(x, "inc0"), "i0r"), "inc3"), "i3r")
        d = relu6(conv(x0, "d0", stride=2), "d0r")
        y = self._bibuf(b, d, base + 0, keys["u0"], h2, w2, state)
        x1 = self._bibuf(b, y, base + 1, keys["u1"], h2, w2, state)
        d = relu6(conv(x1, "d1", stride=2), "d1r")
        y = self._bibuf(b, d, base + 2, keys["u2"], h4, w4, state)
        x2 = self._bibuf(b, y, base + 3, keys["u3"], h4, w4, state)
        m = self._bibuf(b, x2, base + 4, keys["u4"], h4, w4, state)
        m = self._bibuf(b, m, base + 5, keys["u5"], h4, w4, state)
        up2_w = w[keys["up2"] + ".weight"]
        u = b.conv2d(m, up2_w, None, name=f"{n}.up2")
        x2u = b.pixel_shuffle_biased(
            u, w[keys["up2"] + ".bias"], channels=int(up2_w.shape[0]),
            height=h4, width=w4, name=f"{n}.ps2")
        s3 = b.add(x2u, sp3, f"{n}.k3")
        m = self._bibuf(b, s3, base + 6, keys["u6"], h2, w2, state)
        m = self._bibuf(b, m, base + 7, keys["u7"], h2, w2, state)
        up1_w = w[keys["up1"] + ".weight"]
        u = b.conv2d(m, up1_w, None, name=f"{n}.up1")
        x1u = b.pixel_shuffle_biased(
            u, w[keys["up1"] + ".bias"], channels=int(up1_w.shape[0]),
            height=h2, width=w2, name=f"{n}.ps1")
        s2 = b.add(x1u, sp2, f"{n}.k2")
        o = relu6(conv(s2, "out0"), "o0r")
        pred = conv(o, "out3")
        out_channels = int(w[keys["out3"] + ".weight"].shape[0])
        head = b.subtract(sp1, b.slice_channels(pred, 0, 3, f"{n}.ph"),
                          f"{n}.res")
        if out_channels == 3:
            out = head
        else:
            out = b.concat_channels(
                [head, b.slice_channels(pred, 3, out_channels - 3,
                                        f"{n}.pt")], f"{n}.oc")
        self._line_shapes += [(1, 3, hp, wp), (1, sp2_c, hp, wp),
                              (1, sp3_c, h2, w2)]
        return _Emitted(out, x0, x1, out_channels)

    def _compile(self, hp: int, wp: int) -> None:
        if hp % 4 or wp % 4:
            raise ValueError(
                f"BSVD MPSGraph needs sides divisible by four, got "
                f"{wp}x{hp}")
        builder = mg.GraphBuilder(mg.FLOAT16)
        self._line_shapes = []
        state: list[tuple[str, Any, tuple[int, ...]]] = []
        frame = builder.placeholder((1, self.input_channels, hp, wp),
                                    "frame")
        first = self._emit_net(builder, 0, frame, hp, wp, state)
        second = self._emit_net(builder, 1, first.out, hp, wp, state)
        targets = [("out", second.out, (1, second.out_channels, hp, wp))]
        targets += state
        out3 = builder.slice_channels(first.out, 0, 3, "n0.out3")
        for name, tensor, shape in (
                ("n0.x0", first.x0, self._line_shapes[1]),
                ("n0.x1", first.x1, self._line_shapes[2]),
                ("n0.out3", out3, (1, 3, hp, wp)),
                ("n1.x0", second.x0, self._line_shapes[4]),
                ("n1.x1", second.x1, self._line_shapes[5])):
            targets.append((name, tensor, shape))
        loopback = {}
        for unit in range(_UNITS):
            loopback[f"u{unit}.arrival"] = f"u{unit}.center"
            loopback[f"u{unit}.newleft"] = f"u{unit}.left"
        self._graph = mg.compile_graph(builder, targets,
                                       device=mg.DEVICE_ANE,
                                       loopback=loopback)
        self._zero_frame = mx.zeros((1, self.input_channels, hp, wp),
                                    dtype=mx.float16)

    def _ensure_graph(self, hp: int, wp: int) -> None:
        if self._geometry is not None:
            if self._geometry != (hp, wp):
                raise RuntimeError(
                    f"BSVD MPSGraph stream changed resolution from "
                    f"{self._geometry} to {(hp, wp)}")
            return
        self._compile(hp, wp)
        self._geometry = (hp, wp)

    # ---------------------------------------------------------- stepping

    def _feeds(self, frame: Any, record) -> dict[str, Any]:
        feeds: dict[str, Any] = {"frame": frame}
        for unit in range(_UNITS):
            feeds[f"u{unit}.gate"] = (
                self._gate_off if record.drained[unit] else self._gate_on)
            feeds[f"u{unit}.leftgate"] = (
                self._gate_off if record.unprimed[unit] else self._gate_on)
        for line in range(_LINES):
            if record.pops[line]:
                if not self._rings[line]:
                    raise RuntimeError(
                        f"BSVD MPSGraph skip ring {line} underran; the "
                        f"schedule mirror desynchronized")
                feeds[f"s{line}.pop"] = self._rings[line].popleft()
            else:
                feeds[f"s{line}.pop"] = self._zero(self._line_shapes[line])
        return feeds

    def _collect(self) -> Any | None:
        """Join the in-flight dispatch; ring-push and emit its step."""
        if self._pending is None:
            return None
        (record, frame_head), self._pending = self._pending, None
        self._pipeline.join()
        read = {name for line, name in _PUSH_OUTPUTS
                if record.pushes[line]}
        if record.out_real:
            read.add("out")
        outputs = self._graph.read(read)
        if record.pushes[0]:
            self._rings[0].append(frame_head)
        for line, name in _PUSH_OUTPUTS:
            if record.pushes[line]:
                self._rings[line].append(outputs[name])
        if not record.out_real:
            return None
        return mx.transpose(outputs["out"], (0, 2, 3, 1))

    def step(self, x: Any | None) -> Any | None:
        self._require_open()
        if x is None:
            if self._graph is None:
                return None            # drained before any input frame
            frame = self._zero_frame
            record = self._mirror.step(False)
        else:
            height, width = int(x.shape[1]), int(x.shape[2])
            self._ensure_graph(height, width)
            frame = mx.contiguous(mx.transpose(
                x.astype(mx.float16), (0, 3, 1, 2)))
            record = self._mirror.step(True)
        # Feeds for THIS step go into the next parity's buffers while the
        # previous dispatch still runs; its ring pushes land in _collect,
        # after this pop - safe, a pop never needs a push younger than
        # four steps and the pipeline is one deep.
        self._graph.write_feeds(self._feeds(frame, record))
        previous = self._collect()
        self._pending = (record,
                         frame[:, :3] if record.pushes[0] else None)
        self._pipeline.submit(self._graph.begin_dispatch())
        return previous

    def reset(self) -> None:
        self._require_open()
        self._pipeline.drain()
        self._pending = None
        self._mirror = NoneFlowNet()
        self._rings = [deque() for _ in range(_LINES)]
        if self._graph is not None:
            self._graph.reset()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._pipeline.drain()
        finally:
            self._pipeline.close()
            self._pending = None
            self._graph = None
            self._weights = None
            self._rings = [deque() for _ in range(_LINES)]
            self._zeros = {}
            self._zero_frame = None


__all__ = ["MpsGraphBSVD"]
