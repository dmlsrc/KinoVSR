"""TOFlow processors (MLX).

TOFlow's released Torch7 models are table-heavy, flow-warping task nets. The
converter in `LTX_2_MLX/videotoolbox/toflow/convert_t7_to_safetensors.py`
serializes the Torch7 module tree into JSON and tensors into safetensors; this
module interprets the small operator subset used by those checkpoints.

Runtime contract matches the other harness denoisers: RGB NHWC frames in [0,1]
go in, RGB NHWC frames in [0,1] come out. The TOFlow ImageNet normalization,
seven-frame sep ordering, and two-frame interpolation ordering are internal to
the processor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..safmn.net import _bicubic_up
from ..vsr_blocks import _bilinear, resize
from ..weights import resolve_weights as _resolve_weights

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_VARIANTS = {
    "denoise": "toflow_denoise.safetensors",
    "deblock": "toflow_deblock.safetensors",
    "sr": "toflow_sr.safetensors",
    "interp": "toflow_interp.safetensors",
}
_DEFAULT_VARIANT = "denoise"

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def default_weights_path(variant: str = _DEFAULT_VARIANT) -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)


def _graph_path_for(weights: Path, graph: Any = None) -> Path:
    if graph:
        p = Path(graph).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"TOFlow graph JSON not found: {p}")
    p = weights.with_suffix(".json")
    if p.is_file():
        return p
    raise FileNotFoundError(
        f"TOFlow graph JSON not found beside weights: {p}. Convert the source "
        ".t7 with LTX_2_MLX/videotoolbox/toflow/convert_t7_to_safetensors.py "
        "so both .safetensors and .json are present."
    )


def _cast_weights(w: dict[str, Any], dtype: Any) -> dict[str, Any]:
    out = {}
    for k, v in w.items():
        out[k] = v.astype(dtype) if v.dtype == mx.float32 else v
    return out


def _relu(x: Any) -> Any:
    return mx.maximum(x, 0)


def _conv(x: Any, weight: Any, bias: Any, attrs: dict) -> Any:
    sy, sx = int(attrs.get("dH", 1)), int(attrs.get("dW", 1))
    py, px = int(attrs.get("padH", 0)), int(attrs.get("padW", 0))
    groups = int(attrs.get("groups", 1))
    stride = sy if sy == sx else (sy, sx)
    pad = py if py == px else (py, px)
    y = mx.conv2d(x, weight, stride=stride, padding=pad, groups=groups)
    return y + bias if bias is not None else y


def _batchnorm(x: Any, p: dict[str, Any], attrs: dict) -> Any:
    eps = float(attrs.get("eps", 1e-5))
    xf = x.astype(mx.float32)
    weight = p["weight"].astype(mx.float32)
    bias = p["bias"].astype(mx.float32)
    mean = p["running_mean"].astype(mx.float32)
    var = p["running_var"].astype(mx.float32)
    y = (xf - mean) * mx.rsqrt(var + eps) * weight + bias
    return y.astype(x.dtype)


def _avgpool(x: Any, attrs: dict) -> Any:
    kh, kw = int(attrs["kH"]), int(attrs["kW"])
    dh, dw = int(attrs["dH"]), int(attrs["dW"])
    ph, pw = int(attrs.get("padH", 0)), int(attrs.get("padW", 0))
    if (kh, kw, dh, dw, ph, pw) != (2, 2, 2, 2, 0, 0):
        raise NotImplementedError(
            "TOFlow MLX avgpool currently supports only k=2 stride=2 no-pad"
        )
    n, h, w, c = x.shape
    h2, w2 = h // 2, w // 2
    return mx.mean(x[:, :h2 * 2, :w2 * 2, :].reshape(n, h2, 2, w2, 2, c),
                   axis=(2, 4))


def _nearest_up(x: Any, scale: int) -> Any:
    n, h, w, c = x.shape
    y = mx.broadcast_to(x[:, :, None, :, None, :], (n, h, scale, w, scale, c))
    return y.reshape(n, h * scale, w * scale, c)


def _as_channel_tensor(x: Any) -> Any:
    return x[..., None] if getattr(x, "ndim", 0) == 3 else x


def _join_channels(items: list[Any]) -> Any:
    return mx.concatenate([_as_channel_tensor(v) for v in items], axis=-1)


def _split_channels(x: Any) -> list[Any]:
    if x.ndim != 4:
        raise NotImplementedError("TOFlow SplitTable currently expects batched NHWC tensors")
    return [x[..., i] for i in range(int(x.shape[-1]))]


def _replicate(x: Any, attrs: dict) -> Any:
    nfeatures = int(attrs["nfeatures"])
    dim = int(attrs.get("dim", 1))
    if dim != 1:
        raise NotImplementedError(f"TOFlow Replicate dim {dim} is not supported")
    if x.ndim == 3:
        return mx.broadcast_to(x[..., None], (*x.shape, nfeatures))
    if x.ndim == 4 and int(x.shape[-1]) == 1:
        return mx.broadcast_to(x, (*x.shape[:-1], nfeatures))
    return mx.stack([x] * nfeatures, axis=-1)


def _sum_dim(x: Any, attrs: dict) -> Any:
    dim = int(attrs.get("dimension", 1))
    if dim != 1:
        raise NotImplementedError(f"TOFlow Sum dim {dim} is not supported")
    y = mx.sum(x, axis=-1)
    if bool(attrs.get("sizeAverage", False)):
        y = y / float(x.shape[-1])
    return y


def _mul_table(items: list[Any]) -> Any:
    acc = items[0]
    for item in items[1:]:
        acc = acc * item
    return acc


def _shuffle_get(x: list[Any], spec: Any) -> Any:
    if isinstance(spec, dict):
        return x[int(spec["i"]) - 1][int(spec["j"]) - 1]
    return x[int(spec) - 1]


def _shuffle_table(x: list[Any], idx: list[Any]) -> list[Any]:
    out = []
    for spec in idx:
        if isinstance(spec, list):
            out.append([_shuffle_get(x, child) for child in spec])
        else:
            out.append(_shuffle_get(x, spec))
    return out


def _warp_flow_new(pair: list[Any]) -> Any:
    img, flow = pair
    n, h, w, _ = img.shape
    gy, gx = mx.meshgrid(
        mx.arange(h, dtype=mx.float32),
        mx.arange(w, dtype=mx.float32),
        indexing="ij",
    )
    sx = gx[None] + flow[..., 0].astype(mx.float32)
    sy = gy[None] + flow[..., 1].astype(mx.float32)
    return _bilinear(img, sy, sx, "zeros")


def _node_signature(node: dict) -> tuple:
    """Structure + attrs + param NAMES (not weight keys): two Torch7 clones
    of the same module tree have equal signatures even though their param
    keys differ (clones share weights but keep per-clone batchnorm running
    stats)."""
    return (
        node["type"],
        json.dumps(node.get("attrs", {}), sort_keys=True),
        tuple(sorted(node.get("params", {}).keys())),
        tuple(_node_signature(c) for c in node.get("modules", ())),
    )


def _stack_tables(items: list[Any]) -> Any:
    if isinstance(items[0], list):
        return [_stack_tables([it[i] for it in items]) for i in range(len(items[0]))]
    return mx.concatenate(items, axis=0)


def _unstack_tables(x: Any, nb: int) -> list[Any]:
    if isinstance(x, list):
        parts = [_unstack_tables(xi, nb) for xi in x]
        return [[p[i] for p in parts] for i in range(nb)]
    step = int(x.shape[0]) // nb
    return [x[i * step:(i + 1) * step] for i in range(nb)]


class _TOFlowGraph:
    def __init__(self, weights_path: Path, graph_path: Path, dtype: Any):
        self.params = _cast_weights(mx.load(str(weights_path)), dtype)
        self.graph = json.loads(graph_path.read_text(encoding="utf-8"))
        if self.graph.get("format") != "toflow-mlx-graph-v1":
            raise ValueError(f"unsupported TOFlow graph format in {graph_path}")
        self.root = self.graph["root"]
        # Torch7 unrolls the per-neighbor flow module into N cloned branches
        # evaluated one by one at batch 1 -- the dominant cost of the whole
        # net (measured ~90% of a forward). The clones share every conv
        # weight and differ only in batchnorm running stats, so the N
        # branches can be evaluated ONCE as a batch: stack the branch inputs
        # along N, point the cloned tree at (N,1,1,C)-stacked batchnorm
        # stats, and split the outputs back. Exact same math, one kernel
        # per op instead of N.
        self._batch_par: dict[int, tuple[int, dict]] = {}
        self._prepare_batched_tables(self.root)
        # mx.compile per input-shape signature: the interpreter builds the
        # same static graph every frame, so trace once and replay (~8%:
        # fuses the elementwise batchnorm/relu/join chains; the conv-bound
        # bulk is untouched). fp32 compile reorders shift results < 3e-4.
        self._compiled: dict = {}

    def _prepare_batched_tables(self, node: dict) -> None:
        if node["type"] == "nn.ParallelTable":
            kids = node.get("modules", [])
            branches = [k for k in kids if k["type"] != "nn.Identity"]
            if (len(branches) >= 2 and len(branches) == len(kids) - 1
                    and kids[0]["type"] == "nn.Identity"):
                sig0 = _node_signature(branches[0])
                if all(_node_signature(k) == sig0 for k in branches[1:]):
                    virt = self._build_virtual(branches)
                    self._batch_par[id(node)] = (len(branches), virt)
        for c in node.get("modules", ()):
            self._prepare_batched_tables(c)

    def _build_virtual(self, branches: list[dict]) -> dict:
        def rec(nodes: list[dict]) -> dict:
            n0 = nodes[0]
            out: dict[str, Any] = {"type": n0["type"]}
            if n0.get("attrs"):
                out["attrs"] = n0["attrs"]
            params = n0.get("params", {})
            if params:
                newp = {}
                for name in params:
                    keys = [n["params"][name] for n in nodes]
                    if all(k == keys[0] for k in keys[1:]):
                        newp[name] = keys[0]           # shared weight
                    else:
                        vkey = f"__batched__{keys[0]}"
                        if vkey not in self.params:
                            stacked = mx.stack([self.params[k] for k in keys], axis=0)
                            if stacked.ndim == 2:      # (N,C) stats -> NHWC broadcast
                                stacked = stacked[:, None, None, :]
                            self.params[vkey] = stacked
                        newp[name] = vkey
                out["params"] = newp
            subs = [n.get("modules", ()) for n in nodes]
            if subs[0]:
                out["modules"] = [rec([s[i] for s in subs]) for i in range(len(subs[0]))]
            return out

        return rec(branches)

    def forward(self, inputs: list[Any]) -> Any:
        key = tuple((tuple(x.shape), str(x.dtype)) for x in inputs)
        fn = self._compiled.get(key)
        if fn is None:
            fn = mx.compile(lambda *xs: self._eval(self.root, list(xs)))
            self._compiled[key] = fn
        return fn(*inputs)

    def _params(self, node: dict) -> dict[str, Any]:
        return {name: self.params[key] for name, key in node.get("params", {}).items()}

    def _eval(self, node: dict, x: Any) -> Any:
        typ = node["type"]
        children = node.get("modules", ())
        attrs = node.get("attrs", {})

        if typ == "nn.Sequential":
            for child in children:
                x = self._eval(child, x)
            return x
        if typ == "nn.ConcatTable":
            return [self._eval(child, x) for child in children]
        if typ == "nn.ParallelTable":
            if not isinstance(x, list) or len(x) != len(children):
                raise ValueError("TOFlow ParallelTable input arity mismatch")
            batched = self._batch_par.get(id(node))
            if batched is not None:
                nb, virt = batched
                head = self._eval(children[0], x[0])
                y = self._eval(virt, _stack_tables(x[1:]))
                return [head, *_unstack_tables(y, nb)]
            return [self._eval(child, xi) for child, xi in zip(children, x, strict=True)]
        if typ == "nn.SelectTable":
            return x[int(attrs["index"]) - 1]
        if typ == "nn.Identity":
            return x
        if typ == "nn.JoinTable":
            dim = int(attrs["dimension"])
            if dim != 1:
                raise NotImplementedError(f"TOFlow JoinTable dim {dim} is not supported")
            return _join_channels(x)
        if typ == "nn.CAddTable":
            acc = x[0]
            for item in x[1:]:
                acc = acc + item
            return acc
        if typ == "nn.CMulTable":
            return _mul_table(x)
        if typ == "nn.CDivTable":
            return x[0] / x[1]
        if typ == "nn.AddConstant":
            return x + float(attrs["constant_scalar"])
        if typ == "nn.Mul":
            p = self._params(node)
            return x * p["weight"].reshape(())
        if typ == "nn.MulConstant":
            return x * float(attrs["constant_scalar"])
        if typ == "nn.ShuffleTable":
            return _shuffle_table(x, attrs["idx"])
        if typ == "nn.SplitTable":
            dim = int(attrs.get("dimension", 1))
            if dim != 1:
                raise NotImplementedError(f"TOFlow SplitTable dim {dim} is not supported")
            return _split_channels(x)
        if typ == "nn.Replicate":
            return _replicate(x, attrs)
        if typ == "nn.Sum":
            return _sum_dim(x, attrs)
        if typ == "nn.ReLU":
            return _relu(x)
        if typ == "nn.SpatialConvolution":
            p = self._params(node)
            return _conv(x, p["weight"], p.get("bias"), attrs)
        if typ == "nn.SpatialBatchNormalization":
            return _batchnorm(x, self._params(node), attrs)
        if typ == "nn.SpatialAveragePooling":
            return _avgpool(x, attrs)
        if typ == "nn.SpatialUpSamplingBilinear":
            scale = int(attrs.get("scale_factor", 2))
            if scale != 2:
                raise NotImplementedError(f"TOFlow bilinear scale {scale} is not supported")
            return resize(x, int(x.shape[1]) * scale, int(x.shape[2]) * scale, True)
        if typ == "nn.SpatialUpSamplingNearest":
            scale = int(attrs.get("scale_factor", 2))
            return _nearest_up(x, scale)
        if typ == "nn.WarpFlowNew":
            return _warp_flow_new(x)
        raise NotImplementedError(f"unsupported TOFlow module {typ}")


def _pad_axis_reflect(x: Any, axis: int, amount: int) -> Any:
    while amount > 0:
        dim = int(x.shape[axis])
        take = min(amount, max(1, dim - 1))
        if dim == 1:
            shape = list(x.shape)
            shape[axis] = take
            sl = [slice(None)] * x.ndim
            sl[axis] = slice(-1, None)
            pad = mx.broadcast_to(x[tuple(sl)], tuple(shape))
        elif axis == 1:
            pad = x[:, dim - 1 - take:dim - 1, :, :][:, ::-1, :, :]
        elif axis == 2:
            pad = x[:, :, dim - 1 - take:dim - 1, :][:, :, ::-1, :]
        else:
            raise ValueError("TOFlow reflection pad only supports H/W axes")
        x = mx.concatenate([x, pad], axis=axis)
        amount -= take
    return x


def _reflect_pad_to16(x: Any) -> tuple[Any, int, int]:
    _, h, w, _ = x.shape
    ph, pw = (-h) % 16, (-w) % 16
    if ph:
        x = _pad_axis_reflect(x, 1, ph)
    if pw:
        x = _pad_axis_reflect(x, 2, pw)
    return x, ph, pw


class TOFlow:
    """One converted TOFlow checkpoint.

    denoise/deblock/SR use seven-frame sep windows. interp uses a two-frame
    pair. The sep checkpoints run through the direct MLX forward (net.py:
    batched branches, folded batchnorm, padded fusion conv); anything whose
    structure does not match falls back to the graph interpreter.
    """

    NUM_FRAMES = 7

    def __init__(
        self,
        weights: Any = None,
        *,
        variant: str = _DEFAULT_VARIANT,
        graph: Any = None,
        dtype: Any = mx.float32,
        flow_scale: str = "full",
        engine: str = "auto",
    ):
        wp = resolve_weights(weights or variant)
        gp = _graph_path_for(wp, graph)
        self.dtype = dtype
        self.net = _TOFlowGraph(wp, gp, dtype=dtype)
        self.engine = "interp"
        if engine in ("auto", "direct"):
            try:
                from .net import TOFlowDirect
                raw = mx.load(str(wp))
                self.net = TOFlowDirect(self.net.graph, raw, dtype,
                                        flow_scale=flow_scale)
                self.engine = "direct"
            except ValueError:
                if engine == "direct":
                    raise
        self.variant = str((getattr(self.net, "graph", {}) or {}).get("variant")
                           or variant)
        self._mean = mx.array(_MEAN, dtype=mx.float32).reshape(1, 1, 1, 3)
        self._std = mx.array(_STD, dtype=mx.float32).reshape(1, 1, 1, 3)

    def _normalize(self, x: Any) -> Any:
        return ((x.astype(mx.float32) - self._mean) / self._std).astype(self.dtype)

    def _denormalize(self, x: Any) -> Any:
        return x.astype(mx.float32) * self._std + self._mean

    def _run_septuplet(self, frames: list[Any], *, residual_center: bool = False) -> Any:
        if len(frames) != self.NUM_FRAMES:
            raise ValueError(f"TOFlow needs {self.NUM_FRAMES} frames, got {len(frames)}")
        batched = [
            mx.clip((f if f.ndim == 4 else f[None])[..., :3].astype(mx.float32), 0.0, 1.0)
            for f in frames
        ]
        h, w = int(batched[0].shape[1]), int(batched[0].shape[2])
        padded = [_reflect_pad_to16(f)[0] for f in batched]
        hp, wp = int(padded[0].shape[1]), int(padded[0].shape[2])
        norm = [self._normalize(f) for f in padded]
        zero_flow = mx.zeros((1, hp // 8, wp // 8, 2), dtype=self.dtype)

        # Torch7 loader order: center, past three, future three, zero flow seed.
        inputs = [norm[3], norm[0], norm[1], norm[2], norm[4], norm[5], norm[6], zero_flow]
        out = self.net.forward(inputs)
        if residual_center:
            out = out + inputs[0]
        out = mx.clip(self._denormalize(out), 0.0, 1.0)
        return out[0, :h, :w, :].astype(mx.float32)

    def denoise_center(self, frames: list[Any]) -> Any:
        return self._run_septuplet(frames, residual_center=False)

    def sr_center(self, frames: list[Any]) -> Any:
        up = [
            _bicubic_up(mx.clip(f[None, ..., :3].astype(mx.float32), 0.0, 1.0), 4)[0]
            for f in frames
        ]
        return self._run_septuplet(up, residual_center=True)

    def interpolate_pair(self, left: Any, right: Any) -> Any:
        frames = [left, right]
        batched = [mx.clip(f[None, ..., :3].astype(mx.float32), 0.0, 1.0) for f in frames]
        h, w = int(batched[0].shape[1]), int(batched[0].shape[2])
        padded = [_reflect_pad_to16(f)[0] for f in batched]
        norm = [self._normalize(f) for f in padded]
        out = self.net.forward(norm)
        out = mx.clip(self._denormalize(out), 0.0, 1.0)
        return out[0, :h, :w, :].astype(mx.float32)


class TOFlowDenoiser:
    """Streaming seven-frame TOFlow denoise/deblock stage for vsr_harness."""

    def __init__(
        self,
        weights: Any = None,
        *,
        variant: str = _DEFAULT_VARIANT,
        graph: Any = None,
        strength: float = 1.0,
        dtype: Any = mx.float32,
        flow_scale: str = "full",
    ):
        if variant not in {"denoise", "deblock"}:
            raise ValueError("TOFlowDenoiser supports only denoise/deblock variants")
        wp = resolve_weights(weights or variant)
        if not wp.is_file():
            raise FileNotFoundError(
                f"TOFlow weights not found at {wp}. Convert the source .t7 with "
                "LTX_2_MLX/videotoolbox/toflow/convert_t7_to_safetensors.py "
                "or pass --toflow-weights."
            )
        self.net = TOFlow(wp, variant=variant, graph=graph, dtype=dtype,
                          flow_scale=flow_scale)
        # strength is a dry/wet residual blend; the reference network has NO
        # conditioning input, so values above 1.0 EXTRAPOLATE the residual
        # past the trained operating point (a boost the reference cannot
        # express) -- useful in moderation, amplifies model error with it
        self._strength = max(0.0, float(strength))
        self._radius = self.net.NUM_FRAMES // 2
        self._reset()

    def _reset(self) -> None:
        self._buf: list[tuple[Any, Any]] = []
        self._base = 0
        self._received = 0
        self._emitted = 0

    def reset(self) -> None:
        self._reset()

    def close(self) -> None:
        pass

    @staticmethod
    def _reflect(i: int, last: int) -> int:
        if i < 0:
            i = -i
        if i > last:
            i = 2 * last - i
        return max(0, min(last, i))

    def _frame(self, i: int, last: int) -> Any:
        return self._buf[self._reflect(i, last) - self._base][0]

    def _emit_one(self, last: int) -> tuple[Any, Any]:
        t = self._emitted
        window = [self._frame(t + d, last) for d in range(-self._radius, self._radius + 1)]
        out = self.net.denoise_center(window)
        center, tok = self._buf[t - self._base]
        if self._strength != 1.0:
            out = center.astype(mx.float32) + self._strength * (out - center.astype(mx.float32))
            out = mx.clip(out, 0.0, 1.0)
        mx.eval(out)
        self._emitted += 1
        keep = self._emitted - self._radius
        while self._base < keep and self._buf:
            self._buf.pop(0)
            self._base += 1
        return out, tok

    def feed(self, rgb: Any, token: Any = None) -> list[tuple[Any, Any]]:
        self._buf.append((mx.clip(rgb[..., :3].astype(mx.float32), 0.0, 1.0), token))
        self._received += 1
        last = self._received - 1
        ready = []
        while last - self._emitted >= self._radius:
            ready.append(self._emit_one(last))
        return ready

    def flush(self) -> list[tuple[Any, Any]]:
        last = self._received - 1
        out = []
        while self._emitted <= last:
            out.append(self._emit_one(last))
        self._reset()
        return out


class TOFlowSrUpscaler:
    """Streaming seven-frame TOFlow SR stage for vsr_harness."""

    SCALE = 4

    def __init__(
        self,
        weights: Any = None,
        *,
        graph: Any = None,
        dtype: Any = mx.float32,
        flow_scale: str = "full",
    ):
        wp = resolve_weights(weights or "sr")
        if not wp.is_file():
            raise FileNotFoundError(
                f"TOFlow SR weights not found at {wp}. Convert sr.t7 with "
                "LTX_2_MLX/videotoolbox/toflow/convert_t7_to_safetensors.py "
                "or pass --toflow-sr-weights."
            )
        self.net = TOFlow(wp, variant="sr", graph=graph, dtype=dtype,
                          flow_scale=flow_scale)
        self._radius = self.net.NUM_FRAMES // 2
        self._reset()

    def _reset(self) -> None:
        self._buf: list[tuple[Any, Any]] = []
        self._base = 0
        self._received = 0
        self._emitted = 0

    def reset(self) -> None:
        self._reset()

    def close(self) -> None:
        pass

    @staticmethod
    def _reflect(i: int, last: int) -> int:
        if i < 0:
            i = -i
        if i > last:
            i = 2 * last - i
        return max(0, min(last, i))

    def _frame(self, i: int, last: int) -> Any:
        return self._buf[self._reflect(i, last) - self._base][0]

    def _emit_one(self, last: int) -> tuple[Any, Any]:
        t = self._emitted
        window = [self._frame(t + d, last) for d in range(-self._radius, self._radius + 1)]
        out = self.net.sr_center(window)
        _, tok = self._buf[t - self._base]
        mx.eval(out)
        self._emitted += 1
        keep = self._emitted - self._radius
        while self._base < keep and self._buf:
            self._buf.pop(0)
            self._base += 1
        return out, tok

    def feed(self, rgb: Any, token: Any = None) -> list[tuple[Any, Any]]:
        self._buf.append((mx.clip(rgb[..., :3].astype(mx.float32), 0.0, 1.0), token))
        self._received += 1
        last = self._received - 1
        ready = []
        while last - self._emitted >= self._radius:
            ready.append(self._emit_one(last))
        return ready

    def flush(self) -> list[tuple[Any, Any]]:
        last = self._received - 1
        out = []
        while self._emitted <= last:
            out.append(self._emit_one(last))
        self._reset()
        return out


class TOFlowInterpolator:
    """Two-frame TOFlow interpolation helper.

    This exposes the released `interp.t7` model without wiring it into the harness
    FPS/audio path. `feed()` returns original/interpolated pairs and `flush()`
    returns the final original frame so callers can preserve duration.
    """

    def __init__(
        self,
        weights: Any = None,
        *,
        graph: Any = None,
        dtype: Any = mx.float32,
    ):
        wp = resolve_weights(weights or "interp")
        if not wp.is_file():
            raise FileNotFoundError(
                f"TOFlow interpolation weights not found at {wp}. Convert interp.t7 "
                "with LTX_2_MLX/videotoolbox/toflow/convert_t7_to_safetensors.py."
            )
        self.net = TOFlow(wp, variant="interp", graph=graph, dtype=dtype)
        self._prev: tuple[Any, Any] | None = None

    def reset(self) -> None:
        self._prev = None

    def close(self) -> None:
        pass

    def interpolate(self, left: Any, right: Any) -> Any:
        out = self.net.interpolate_pair(left, right)
        mx.eval(out)
        return out

    def feed(self, rgb: Any, token: Any = None) -> list[tuple[Any, Any]]:
        cur = mx.clip(rgb[..., :3].astype(mx.float32), 0.0, 1.0)
        if self._prev is None:
            self._prev = (cur, token)
            return []
        left, left_tok = self._prev
        mid = self.interpolate(left, cur)
        self._prev = (cur, token)
        return [(left, left_tok), (mid, (left_tok, token))]

    def flush(self) -> list[tuple[Any, Any]]:
        if self._prev is None:
            return []
        item = self._prev
        self._prev = None
        return [item]


__all__ = [
    "TOFlow",
    "TOFlowDenoiser",
    "TOFlowInterpolator",
    "TOFlowSrUpscaler",
    "default_weights_path",
    "resolve_weights",
]
