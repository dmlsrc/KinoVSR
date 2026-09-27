"""TOFlow processors (MLX).

TOFlow's released Torch7 models are table-heavy, flow-warping task nets. The
converter in `kinovsr/processors/toflow/convert_t7_to_safetensors.py`
serializes the Torch7 module tree into JSON and tensors into safetensors; this
module interprets the small operator subset used by those checkpoints.

Runtime contract matches the other harness denoisers: RGB NHWC frames in [0,1]
go in, RGB NHWC frames in [0,1] come out. The TOFlow ImageNet normalization,
seven-frame sep ordering, and two-frame interpolation ordering are internal to
the processor.
"""

import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinovsr.modeling.resample import bilinear_sample, resize
from kinovsr.modeling.upsample import bicubic_up as _bicubic_up
from kinovsr.modeling.weights import resolve_weights as _resolve_weights

from .net import TOFlowDirect, _Attrs, _Graph, _Node, _ShuffleRef

# An interpreter value: a tensor, or a Torch7 table (list) of values.
type _Value = mx.array | list[_Value]

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


def resolve_weights(spec: str | Path | None = None) -> Path:
    return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)


def _graph_path_for(weights: Path, graph: str | Path | None = None) -> Path:
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
        ".t7 with kinovsr/processors/toflow/convert_t7_to_safetensors.py "
        "so both .safetensors and .json are present."
    )


def _cast_weights(w: dict[str, mx.array], dtype: mx.Dtype) -> dict[str, mx.array]:
    out = {}
    for k, v in w.items():
        out[k] = v.astype(dtype) if v.dtype == mx.float32 else v
    return out


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _conv(x: mx.array, weight: mx.array, bias: mx.array | None, attrs: _Attrs) -> mx.array:
    sy, sx = int(attrs.get("dH", 1)), int(attrs.get("dW", 1))
    py, px = int(attrs.get("padH", 0)), int(attrs.get("padW", 0))
    groups = int(attrs.get("groups", 1))
    stride = sy if sy == sx else (sy, sx)
    pad = py if py == px else (py, px)
    y = mx.conv2d(x, weight, stride=stride, padding=pad, groups=groups)
    return y + bias if bias is not None else y


def _batchnorm(x: mx.array, p: dict[str, mx.array], attrs: _Attrs) -> mx.array:
    eps = float(attrs.get("eps", 1e-5))
    xf = x.astype(mx.float32)
    weight = p["weight"].astype(mx.float32)
    bias = p["bias"].astype(mx.float32)
    mean = p["running_mean"].astype(mx.float32)
    var = p["running_var"].astype(mx.float32)
    y = (xf - mean) * mx.rsqrt(var + eps) * weight + bias
    return y.astype(x.dtype)


def _avgpool(x: mx.array, attrs: _Attrs) -> mx.array:
    kh, kw = int(attrs["kH"]), int(attrs["kW"])
    dh, dw = int(attrs["dH"]), int(attrs["dW"])
    ph, pw = int(attrs.get("padH", 0)), int(attrs.get("padW", 0))
    if (kh, kw, dh, dw, ph, pw) != (2, 2, 2, 2, 0, 0):
        raise NotImplementedError("TOFlow MLX avgpool currently supports only k=2 stride=2 no-pad")
    n, h, w, c = x.shape
    h2, w2 = h // 2, w // 2
    return mx.mean(x[:, : h2 * 2, : w2 * 2, :].reshape(n, h2, 2, w2, 2, c), axis=(2, 4))


def _nearest_up(x: mx.array, scale: int) -> mx.array:
    n, h, w, c = x.shape
    y = mx.broadcast_to(x[:, :, None, :, None, :], (n, h, scale, w, scale, c))
    return y.reshape(n, h * scale, w * scale, c)


def _as_channel_tensor(x: mx.array) -> mx.array:
    return x[..., None] if getattr(x, "ndim", 0) == 3 else x


def _join_channels(items: list[mx.array]) -> mx.array:
    return mx.concatenate([_as_channel_tensor(v) for v in items], axis=-1)


def _split_channels(x: mx.array) -> list[_Value]:
    if x.ndim != 4:
        raise NotImplementedError("TOFlow SplitTable currently expects batched NHWC tensors")
    return [x[..., i] for i in range(int(x.shape[-1]))]


def _replicate(x: mx.array, attrs: _Attrs) -> mx.array:
    nfeatures = int(attrs["nfeatures"])
    dim = int(attrs.get("dim", 1))
    if dim != 1:
        raise NotImplementedError(f"TOFlow Replicate dim {dim} is not supported")
    if x.ndim == 3:
        return mx.broadcast_to(x[..., None], (*x.shape, nfeatures))
    if x.ndim == 4 and int(x.shape[-1]) == 1:
        return mx.broadcast_to(x, (*x.shape[:-1], nfeatures))
    return mx.stack([x] * nfeatures, axis=-1)


def _sum_dim(x: mx.array, attrs: _Attrs) -> mx.array:
    dim = int(attrs.get("dimension", 1))
    if dim != 1:
        raise NotImplementedError(f"TOFlow Sum dim {dim} is not supported")
    y = mx.sum(x, axis=-1)
    if bool(attrs.get("sizeAverage", False)):
        y = y / float(x.shape[-1])
    return y


def _mul_table(items: list[mx.array]) -> mx.array:
    acc = items[0]
    for item in items[1:]:
        acc = acc * item
    return acc


def _shuffle_get(x: list[_Value], spec: _ShuffleRef) -> _Value:
    if isinstance(spec, dict):
        return x[int(spec["i"]) - 1][int(spec["j"]) - 1]
    return x[int(spec) - 1]


def _shuffle_table(x: list[_Value], idx: list[_ShuffleRef | list[_ShuffleRef]]) -> list[_Value]:
    out: list[_Value] = []
    for spec in idx:
        if isinstance(spec, list):
            out.append([_shuffle_get(x, child) for child in spec])
        else:
            out.append(_shuffle_get(x, spec))
    return out


def _warp_flow_new(pair: list[mx.array]) -> mx.array:
    img, flow = pair
    _n, h, w, _ = img.shape
    gy, gx = mx.meshgrid(
        mx.arange(h, dtype=mx.float32),
        mx.arange(w, dtype=mx.float32),
        indexing="ij",
    )
    sx = gx[None] + flow[..., 0].astype(mx.float32)
    sy = gy[None] + flow[..., 1].astype(mx.float32)
    return bilinear_sample(img, sy, sx, "zeros")


def _node_signature(node: _Node) -> tuple[object, ...]:
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


def _stack_tables(items: list[_Value]) -> _Value:
    if isinstance(items[0], list):
        return [_stack_tables([it[i] for it in items]) for i in range(len(items[0]))]
    return mx.concatenate(cast(list[mx.array], items), axis=0)  # same structure as items[0]


def _unstack_tables(x: _Value, nb: int) -> list[_Value]:
    if isinstance(x, list):
        parts = [_unstack_tables(xi, nb) for xi in x]
        return [[p[i] for p in parts] for i in range(nb)]
    step = int(x.shape[0]) // nb
    return [x[i * step : (i + 1) * step] for i in range(nb)]


class _TOFlowGraph:
    def __init__(self, weights_path: Path, graph_path: Path, dtype: mx.Dtype):
        self.params = _cast_weights(cast(dict[str, mx.array], mx.load(str(weights_path))), dtype)
        self.graph = cast(_Graph, json.loads(graph_path.read_text(encoding="utf-8")))
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
        self._batch_par: dict[int, tuple[int, _Node]] = {}
        self._prepare_batched_tables(self.root)
        # mx.compile per input-shape signature: the interpreter builds the
        # same static graph every frame, so trace once and replay (~8%:
        # fuses the elementwise batchnorm/relu/join chains; the conv-bound
        # bulk is untouched). fp32 compile reorders shift results < 3e-4.
        self._compiled: dict[tuple[object, ...], Callable[[*tuple[mx.array, ...]], mx.array]] = {}

    def _prepare_batched_tables(self, node: _Node) -> None:
        if node["type"] == "nn.ParallelTable":
            kids = node.get("modules", [])
            branches = [k for k in kids if k["type"] != "nn.Identity"]
            if (
                len(branches) >= 2
                and len(branches) == len(kids) - 1
                and kids[0]["type"] == "nn.Identity"
            ):
                sig0 = _node_signature(branches[0])
                if all(_node_signature(k) == sig0 for k in branches[1:]):
                    virt = self._build_virtual(branches)
                    self._batch_par[id(node)] = (len(branches), virt)
        for c in node.get("modules", ()):
            self._prepare_batched_tables(c)

    def _build_virtual(self, branches: list[_Node]) -> _Node:
        def rec(nodes: list[_Node]) -> _Node:
            n0 = nodes[0]
            out: _Node = {"type": n0["type"]}
            if n0.get("attrs"):
                out["attrs"] = n0["attrs"]
            params = n0.get("params", {})
            if params:
                newp = {}
                for name in params:
                    keys = [n["params"][name] for n in nodes]
                    if all(k == keys[0] for k in keys[1:]):
                        newp[name] = keys[0]  # shared weight
                    else:
                        vkey = f"__batched__{keys[0]}"
                        if vkey not in self.params:
                            stacked = mx.stack([self.params[k] for k in keys], axis=0)
                            if stacked.ndim == 2:  # (N,C) stats -> NHWC broadcast
                                stacked = stacked[:, None, None, :]
                            self.params[vkey] = stacked
                        newp[name] = vkey
                out["params"] = newp
            subs = [n.get("modules", ()) for n in nodes]
            if subs[0]:
                out["modules"] = [rec([s[i] for s in subs]) for i in range(len(subs[0]))]
            return out

        return rec(branches)

    def forward(self, inputs: list[mx.array]) -> mx.array:
        key = tuple((tuple(x.shape), str(x.dtype)) for x in inputs)
        fn = self._compiled.get(key)
        if fn is None:
            fn = mx.compile(lambda *xs: cast(mx.array, self._eval(self.root, list(xs))))
            self._compiled[key] = fn
        return fn(*inputs)

    def _params(self, node: _Node) -> dict[str, mx.array]:
        return {name: self.params[key] for name, key in node.get("params", {}).items()}

    def _eval(self, node: _Node, x: _Value) -> _Value:
        typ = node["type"]
        children: Sequence[_Node] = node.get("modules", ())
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
            return _join_channels(cast(list[mx.array], x))  # module type fixes tensor vs table
        if typ == "nn.CAddTable":
            acc = cast(list[mx.array], x)[0]
            for item in cast(list[mx.array], x)[1:]:
                acc = acc + item
            return acc
        if typ == "nn.CMulTable":
            return _mul_table(cast(list[mx.array], x))
        if typ == "nn.CDivTable":
            return cast(list[mx.array], x)[0] / cast(list[mx.array], x)[1]
        if typ == "nn.AddConstant":
            return cast(mx.array, x) + float(attrs["constant_scalar"])
        if typ == "nn.Mul":
            p = self._params(node)
            return cast(mx.array, x) * p["weight"].reshape(())
        if typ == "nn.MulConstant":
            return cast(mx.array, x) * float(attrs["constant_scalar"])
        if typ == "nn.ShuffleTable":
            return _shuffle_table(cast(list[_Value], x), attrs["idx"])
        if typ == "nn.SplitTable":
            dim = int(attrs.get("dimension", 1))
            if dim != 1:
                raise NotImplementedError(f"TOFlow SplitTable dim {dim} is not supported")
            return _split_channels(cast(mx.array, x))
        if typ == "nn.Replicate":
            return _replicate(cast(mx.array, x), attrs)
        if typ == "nn.Sum":
            return _sum_dim(cast(mx.array, x), attrs)
        if typ == "nn.ReLU":
            return _relu(cast(mx.array, x))
        if typ == "nn.SpatialConvolution":
            p = self._params(node)
            return _conv(cast(mx.array, x), p["weight"], p.get("bias"), attrs)
        if typ == "nn.SpatialBatchNormalization":
            return _batchnorm(cast(mx.array, x), self._params(node), attrs)
        if typ == "nn.SpatialAveragePooling":
            return _avgpool(cast(mx.array, x), attrs)
        if typ == "nn.SpatialUpSamplingBilinear":
            scale = int(attrs.get("scale_factor", 2))
            if scale != 2:
                raise NotImplementedError(f"TOFlow bilinear scale {scale} is not supported")
            return resize(
                cast(mx.array, x),
                int(cast(mx.array, x).shape[1]) * scale,
                int(cast(mx.array, x).shape[2]) * scale,
                True,
            )
        if typ == "nn.SpatialUpSamplingNearest":
            scale = int(attrs.get("scale_factor", 2))
            return _nearest_up(cast(mx.array, x), scale)
        if typ == "nn.WarpFlowNew":
            return _warp_flow_new(cast(list[mx.array], x))
        raise NotImplementedError(f"unsupported TOFlow module {typ}")


def _pad_axis_reflect(x: mx.array, axis: int, amount: int) -> mx.array:
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
            pad = x[:, dim - 1 - take : dim - 1, :, :][:, ::-1, :, :]
        elif axis == 2:
            pad = x[:, :, dim - 1 - take : dim - 1, :][:, :, ::-1, :]
        else:
            raise ValueError("TOFlow reflection pad only supports H/W axes")
        x = mx.concatenate([x, pad], axis=axis)
        amount -= take
    return x


def _reflect_pad_to16(x: mx.array) -> tuple[mx.array, int, int]:
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
        weights: str | Path | None = None,
        *,
        variant: str = _DEFAULT_VARIANT,
        graph: str | Path | None = None,
        dtype: mx.Dtype = mx.float32,
        flow_scale: str = "full",
        engine: str = "auto",
    ):
        wp = resolve_weights(weights or variant)
        gp = _graph_path_for(wp, graph)
        self.dtype = dtype

        if flow_scale not in TOFlowDirect._FLOW_STARTS:
            raise ValueError(
                f"flow_scale must be one of {sorted(TOFlowDirect._FLOW_STARTS)}, got {flow_scale!r}"
            )
        self.net: _TOFlowGraph | TOFlowDirect = _TOFlowGraph(wp, gp, dtype=dtype)
        self.engine = "interp"
        if engine in ("auto", "direct"):
            try:
                raw = cast(dict[str, mx.array], mx.load(str(wp)))
                self.net = TOFlowDirect(self.net.graph, raw, dtype, flow_scale=flow_scale)
                self.engine = "direct"
            except ValueError:
                # structural mismatch (e.g. the two-frame interp checkpoint):
                # keep the interpreter. flow_scale was validated above, so a
                # bad value cannot silently land here.
                if engine == "direct":
                    raise
        self.variant = str((getattr(self.net, "graph", {}) or {}).get("variant") or variant)
        self._mean = mx.array(_MEAN, dtype=mx.float32).reshape(1, 1, 1, 3)
        self._std = mx.array(_STD, dtype=mx.float32).reshape(1, 1, 1, 3)

    def _normalize(self, x: mx.array) -> mx.array:
        return ((x.astype(mx.float32) - self._mean) / self._std).astype(self.dtype)

    def _denormalize(self, x: mx.array) -> mx.array:
        return x.astype(mx.float32) * self._std + self._mean

    def _prep_septuplet(self, frames: list[mx.array]) -> tuple[list[mx.array], int, int]:
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
        return inputs, h, w

    def _run_septuplet(self, frames: list[mx.array], *, residual_center: bool = False) -> mx.array:
        inputs, h, w = self._prep_septuplet(frames)
        out = self.net.forward(inputs)
        if residual_center:
            out = out + inputs[0]
        out = mx.clip(self._denormalize(out), 0.0, 1.0)
        return out[0, :h, :w, :].astype(mx.float32)

    def flow_septuplet(self, frames: list[mx.array]) -> mx.array:
        """The batched neighbor->center flow for a window (direct engine)."""
        inputs, _h, _w = self._prep_septuplet(frames)
        assert isinstance(self.net, TOFlowDirect)
        return self.net.forward_flow(inputs)

    def fuse_septuplet(
        self, frames: list[mx.array], flow: mx.array, *, residual_center: bool = False
    ) -> mx.array:
        """Warp + fusion with a precomputed flow (direct engine). Deblock
        passes do not move content, so chained passes can reuse pass 1's
        flow -- measured equivalent within 0.01 dB, ~55 dB agreement."""
        inputs, h, w = self._prep_septuplet(frames)
        assert isinstance(self.net, TOFlowDirect)
        out = self.net.forward_fuse(inputs, flow.astype(self.dtype))
        if residual_center:
            out = out + inputs[0]
        out = mx.clip(self._denormalize(out), 0.0, 1.0)
        return out[0, :h, :w, :].astype(mx.float32)

    def denoise_center(self, frames: list[mx.array]) -> mx.array:
        return self._run_septuplet(frames, residual_center=False)

    def sr_center(self, frames: list[mx.array]) -> mx.array:
        up = [
            _bicubic_up(mx.clip(f[None, ..., :3].astype(mx.float32), 0.0, 1.0), 4)[0]
            for f in frames
        ]
        return self._run_septuplet(up, residual_center=True)

    def interpolate_pair(self, left: mx.array, right: mx.array) -> mx.array:
        frames = [left, right]
        batched = [mx.clip(f[None, ..., :3].astype(mx.float32), 0.0, 1.0) for f in frames]
        h, w = int(batched[0].shape[1]), int(batched[0].shape[2])
        padded = [_reflect_pad_to16(f)[0] for f in batched]
        norm = [self._normalize(f) for f in padded]
        out = self.net.forward(norm)
        out = mx.clip(self._denormalize(out), 0.0, 1.0)
        return out[0, :h, :w, :].astype(mx.float32)


@dataclasses.dataclass(slots=True)
class _Stage:
    buf: list[tuple[mx.array, object]] = dataclasses.field(default_factory=list)
    base: int = 0
    received: int = 0
    emitted: int = 0


class TOFlowDenoiser:
    """Streaming seven-frame TOFlow denoise/deblock stage for the processing chain.

    passes > 1 runs an internal cascade (each pass consumes the previous
    pass's stream, like chaining the stage against itself) but computes the
    flow ONCE at pass 1 and reuses it: deblock passes do not move content,
    so the flow is pass-invariant (measured equivalent within 0.01 dB on
    static and moving fixtures, ~55 dB output agreement) -- later passes
    pay only warp + fusion. Latency is 3 * passes frames.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        variant: str = _DEFAULT_VARIANT,
        graph: str | Path | None = None,
        strength: float = 1.0,
        dtype: mx.Dtype = mx.float32,
        flow_scale: str = "full",
        passes: int = 1,
    ):
        if variant not in {"denoise", "deblock"}:
            raise ValueError("TOFlowDenoiser supports only denoise/deblock variants")
        wp = resolve_weights(weights or variant)
        if not wp.is_file():
            raise FileNotFoundError(
                f"TOFlow weights not found at {wp}. Convert the source .t7 with "
                "kinovsr/processors/toflow/convert_t7_to_safetensors.py "
                "or pass --toflow-weights."
            )
        self.net = TOFlow(wp, variant=variant, graph=graph, dtype=dtype, flow_scale=flow_scale)
        self._passes = max(1, int(passes))
        if self._passes > 1 and self.net.engine != "direct":
            raise ValueError(
                "multi-pass TOFlow needs the direct engine (flow reuse); this "
                "checkpoint fell back to the graph interpreter"
            )
        # strength is a dry/wet residual blend; the reference network has NO
        # conditioning input, so values above 1.0 EXTRAPOLATE the residual
        # past the trained operating point (a boost the reference cannot
        # express) -- useful in moderation, amplifies model error with it
        self._strength = max(0.0, float(strength))
        self._radius = self.net.NUM_FRAMES // 2
        self._reset()

    def _reset(self) -> None:
        self._stages: list[_Stage] = [_Stage() for _ in range(self._passes)]
        self._flows: dict[int, mx.array] = {}

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

    def _stage_frame(self, st: _Stage, i: int, last: int) -> mx.array:
        return st.buf[self._reflect(i, last) - st.base][0]

    def _stage_emit(self, s: int) -> tuple[mx.array, object]:
        st = self._stages[s]
        last = st.received - 1
        t = st.emitted
        window = [
            self._stage_frame(st, t + d, last) for d in range(-self._radius, self._radius + 1)
        ]
        if self._passes == 1:
            out = self.net.denoise_center(window)
        elif s == 0:
            flow = self.net.flow_septuplet(window)
            self._flows[t] = flow.astype(mx.float16)
            out = self.net.fuse_septuplet(window, flow)
        else:
            flow = self._flows[t]
            if s == self._passes - 1:
                del self._flows[t]
            out = self.net.fuse_septuplet(window, flow)
        center, tok = st.buf[t - st.base]
        if self._strength != 1.0:
            out = center.astype(mx.float32) + self._strength * (out - center.astype(mx.float32))
            out = mx.clip(out, 0.0, 1.0)
        mx.eval(out)
        st.emitted += 1
        keep = st.emitted - self._radius
        while st.base < keep and st.buf:
            st.buf.pop(0)
            st.base += 1
        return out, tok

    def _push(self, s: int, frame: mx.array, token: object) -> None:
        self._stages[s].buf.append((frame, token))
        self._stages[s].received += 1

    def _drain(self, flush: bool) -> list[tuple[mx.array, object]]:
        ready: list[tuple[mx.array, object]] = []
        for s, st in enumerate(self._stages):
            while (
                st.received - 1 - st.emitted >= (0 if flush else self._radius)
                and st.emitted <= st.received - 1
            ):
                out, tok = self._stage_emit(s)
                if s == self._passes - 1:
                    ready.append((out, tok))
                else:
                    self._push(s + 1, out, tok)
        return ready

    def feed(self, rgb: mx.array, token: object = None) -> list[tuple[mx.array, object]]:
        self._push(0, mx.clip(rgb[..., :3].astype(mx.float32), 0.0, 1.0), token)
        return self._drain(flush=False)

    def flush(self) -> list[tuple[mx.array, object]]:
        out = self._drain(flush=True)
        self._reset()
        return out


class TOFlowSrUpscaler:
    """Streaming seven-frame TOFlow SR stage for the processing chain."""

    SCALE = 4

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        graph: str | Path | None = None,
        dtype: mx.Dtype = mx.float32,
        flow_scale: str = "full",
    ):
        wp = resolve_weights(weights or "sr")
        if not wp.is_file():
            raise FileNotFoundError(
                f"TOFlow SR weights not found at {wp}. Convert sr.t7 with "
                "kinovsr/processors/toflow/convert_t7_to_safetensors.py "
                "or pass --toflow-sr-weights."
            )
        self.net = TOFlow(wp, variant="sr", graph=graph, dtype=dtype, flow_scale=flow_scale)
        self._radius = self.net.NUM_FRAMES // 2
        self._reset()

    def _reset(self) -> None:
        self._buf: list[tuple[mx.array, object]] = []
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

    def _frame(self, i: int, last: int) -> mx.array:
        return self._buf[self._reflect(i, last) - self._base][0]

    def _emit_one(self, last: int) -> tuple[mx.array, object]:
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

    def feed(self, rgb: mx.array, token: object = None) -> list[tuple[mx.array, object]]:
        self._buf.append((mx.clip(rgb[..., :3].astype(mx.float32), 0.0, 1.0), token))
        self._received += 1
        last = self._received - 1
        ready = []
        while last - self._emitted >= self._radius:
            ready.append(self._emit_one(last))
        return ready

    def flush(self) -> list[tuple[mx.array, object]]:
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
        weights: str | Path | None = None,
        *,
        graph: str | Path | None = None,
        dtype: mx.Dtype = mx.float32,
    ):
        wp = resolve_weights(weights or "interp")
        if not wp.is_file():
            raise FileNotFoundError(
                f"TOFlow interpolation weights not found at {wp}. Convert interp.t7 "
                "with kinovsr/processors/toflow/convert_t7_to_safetensors.py."
            )
        self.net = TOFlow(wp, variant="interp", graph=graph, dtype=dtype)
        self._prev: tuple[mx.array, object] | None = None

    def reset(self) -> None:
        self._prev = None

    def close(self) -> None:
        pass

    def interpolate(self, left: mx.array, right: mx.array) -> mx.array:
        out = self.net.interpolate_pair(left, right)
        mx.eval(out)
        return out

    def feed(self, rgb: mx.array, token: object = None) -> list[tuple[mx.array, object]]:
        cur = mx.clip(rgb[..., :3].astype(mx.float32), 0.0, 1.0)
        if self._prev is None:
            self._prev = (cur, token)
            return []
        left, left_tok = self._prev
        mid = self.interpolate(left, cur)
        self._prev = (cur, token)
        return [(left, left_tok), (mid, (left_tok, token))]

    def flush(self) -> list[tuple[mx.array, object]]:
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
