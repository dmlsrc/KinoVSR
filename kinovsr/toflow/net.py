"""Direct MLX forward for the TOFlow sep (denoise/deblock/sr) checkpoints.

The generic Torch7 graph interpreter in this package is faithful but pays for
generality: per-branch batchnorm chains as five-op fp32 passes, table
plumbing, and an unbatchable-by-default structure. This module extracts the
known sep topology out of the converted graph JSON once at load and runs it
as plain MLX:

    per neighbor (all six as ONE batch, weights shared):
      SpyNet-style coarse-to-fine flow, 4 levels (/8 -> full), per level a
      5-conv CNN (7x7) refining the x2-upsampled coarser flow after warping
      the neighbor; per-branch batchnorm stats fold into per-branch affine
      scale/shift applied after the shared convolutions.
    fusion: join(center + six warped) -> 9x9 conv (input channel-padded
      21 -> 32: measured 2.7x on MLX's conv path) -> 1x1 -> 1x1.

Extraction is structural with hard asserts; anything unexpected raises, and
the caller falls back to the interpreter. The interp checkpoint keeps the
interpreter (different, two-frame topology).

flow_scale: "full" (default) is the faithful network. "half" skips the
full-resolution flow refinement CNN -- the dominant cost of the whole net --
and "quarter" also skips the half-resolution level, each at a further
alignment fidelity cost (measured bills in the harness help; eyeball per
clip before relying on them).
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from ..vsr_blocks import _bilinear, resize

_LEVELS = 4


def _relu(x: Any) -> Any:
    return mx.maximum(x, 0)


def _warp(img: Any, flow: Any) -> Any:
    n, h, w, _ = img.shape
    gy, gx = mx.meshgrid(
        mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32),
        indexing="ij",
    )
    sx = gx[None] + flow[..., 0].astype(mx.float32)
    sy = gy[None] + flow[..., 1].astype(mx.float32)
    return _bilinear(img, sy, sx, "zeros")


def _avgpool2(x: Any) -> Any:
    n, h, w, c = x.shape
    h2, w2 = h // 2, w // 2
    return mx.mean(x[:, :h2 * 2, :w2 * 2, :].reshape(n, h2, 2, w2, 2, c),
                   axis=(2, 4))


def _up2(x: Any) -> Any:
    return resize(x, int(x.shape[1]) * 2, int(x.shape[2]) * 2, True)


class _Expect:
    """Tiny structural cursor over the graph JSON with hard asserts."""

    @staticmethod
    def typed(node: dict, typ: str) -> dict:
        if node["type"] != typ:
            raise ValueError(f"TOFlow direct: expected {typ}, got {node['type']}")
        return node

    @staticmethod
    def kids(node: dict, n: int | None = None) -> list:
        kids = list(node.get("modules", ()))
        if n is not None and len(kids) != n:
            raise ValueError(
                f"TOFlow direct: expected {n} children under {node['type']}, "
                f"got {len(kids)}"
            )
        return kids


def _fold_convs(graph: dict, params: dict) -> dict:
    """Extract per-level CNN weights + per-branch folded batchnorm affines
    and the fusion convs from the sep graph. Returns the flat param dict.

    Structure asserts follow the dump of the released checkpoints; any
    mismatch raises ValueError (caller falls back to the interpreter).
    """
    E = _Expect
    root = E.typed(graph["root"], "nn.Sequential")
    concat, par, fusion = E.kids(root, 3)
    E.typed(concat, "nn.ConcatTable")
    E.typed(par, "nn.ParallelTable")
    E.typed(fusion, "nn.Sequential")

    wiring = E.kids(concat)
    if E.typed(wiring[0], "nn.SelectTable")["attrs"]["index"] != 1:
        raise ValueError("TOFlow direct: fusion head is not the center frame")
    nb_index = []
    for w in wiring[1:]:
        sel_nb, sel_inner = E.kids(E.typed(w, "nn.ConcatTable"), 2)
        nb_index.append(int(E.typed(sel_nb, "nn.SelectTable")["attrs"]["index"]))
        inner = E.kids(E.typed(sel_inner, "nn.ConcatTable"), 3)
        idxs = [int(E.typed(s, "nn.SelectTable")["attrs"]["index"]) for s in inner]
        if idxs[0] != 1 or idxs[1] != nb_index[-1]:
            raise ValueError("TOFlow direct: unexpected branch input wiring")

    branches = E.kids(par)
    E.typed(branches[0], "nn.Identity")
    branches = branches[1:]
    if len(branches) != len(nb_index):
        raise ValueError("TOFlow direct: branch/wiring count mismatch")

    out: dict[str, Any] = {"nb_index": nb_index}

    def conv_of(node: dict) -> tuple[Any, Any]:
        E.typed(node, "nn.SpatialConvolution")
        p = node["params"]
        return params[p["weight"]], params[p["bias"]]

    def bn_of(node: dict) -> tuple[Any, Any, Any, Any, float]:
        E.typed(node, "nn.SpatialBatchNormalization")
        p = node["params"]
        eps = float(node.get("attrs", {}).get("eps", 1e-5))
        return (params[p["weight"]], params[p["bias"]],
                params[p["running_mean"]], params[p["running_var"]], eps)

    def cnn_nodes(seq: dict) -> list[dict]:
        kids = E.kids(E.typed(seq, "nn.Sequential"))
        types = [k["type"] for k in kids]
        want = (["nn.SpatialConvolution", "nn.SpatialBatchNormalization", "nn.ReLU"] * 4
                + ["nn.SpatialConvolution"])
        if types != want:
            raise ValueError(f"TOFlow direct: unexpected level CNN layout {types}")
        return kids

    def walk_branch(branch: dict) -> list[dict]:
        """Return the 4 level-CNN Sequentials, coarsest first."""
        seq = E.typed(branch, "nn.Sequential")
        par0, warp = E.kids(seq, 2)
        E.typed(warp, "nn.WarpFlowNew")
        _ident, rec = E.kids(E.typed(par0, "nn.ParallelTable"), 2)
        levels: list[dict] = []

        def rec_level(node: dict) -> None:
            kids = E.kids(E.typed(node, "nn.Sequential"))
            if len(kids) == 2:
                # coarsest: ConcatTable[Select3, Sequential[Join, CNN]], CAdd
                ct, cadd = kids
                E.typed(cadd, "nn.CAddTable")
                _sel, joincnn = E.kids(E.typed(ct, "nn.ConcatTable"), 2)
                join, cnn = E.kids(E.typed(joincnn, "nn.Sequential"), 2)
                E.typed(join, "nn.JoinTable")
                levels.append(cnn)
                return
            if len(kids) != 3:
                raise ValueError("TOFlow direct: unexpected level arity")
            ct_down, ct_warp, refine = kids
            sel1, sel2, coarser = E.kids(E.typed(ct_down, "nn.ConcatTable"), 3)
            inner = E.kids(E.typed(coarser, "nn.Sequential"))
            # [ParallelTable(pool,pool,Identity), REC, Upsample, MulConstant]
            if len(inner) != 4:
                raise ValueError("TOFlow direct: unexpected coarser wrapper")
            rec_level(inner[1])
            E.typed(ct_warp, "nn.ConcatTable")
            ct_ref, cadd = E.kids(E.typed(refine, "nn.Sequential"), 2)
            E.typed(cadd, "nn.CAddTable")
            _sel, joincnn = E.kids(E.typed(ct_ref, "nn.ConcatTable"), 2)
            join, cnn = E.kids(E.typed(joincnn, "nn.Sequential"), 2)
            E.typed(join, "nn.JoinTable")
            levels.append(cnn)

        rec_level(rec)
        if len(levels) != _LEVELS:
            raise ValueError(f"TOFlow direct: expected {_LEVELS} levels, got {len(levels)}")
        return levels[::-1]              # l0 = full res ... l3 = coarsest

    per_branch_levels = [walk_branch(b) for b in branches]

    # shared conv weights come from branch 0; per-branch BN stats fold into
    # (B,1,1,C) affines applied after the shared conv
    nb = len(branches)
    for lvl in range(_LEVELS):
        nodes0 = cnn_nodes(per_branch_levels[0][lvl])
        for ci in range(5):
            w0, b0 = conv_of(nodes0[ci * 3])
            for other in per_branch_levels[1:]:
                wx, _bx = conv_of(cnn_nodes(other[lvl])[ci * 3])
                if wx.shape != w0.shape:
                    raise ValueError("TOFlow direct: branch conv shape mismatch")
            out[f"l{lvl}.c{ci}.w"] = w0
            if ci == 4:
                out[f"l{lvl}.c{ci}.b"] = b0
                continue
            scales, shifts = [], []
            for br in range(nb):
                nodes = cnn_nodes(per_branch_levels[br][lvl])
                _wb, bb = conv_of(nodes[ci * 3])
                g, be, mean, var, eps = bn_of(nodes[ci * 3 + 1])
                s = (g.astype(mx.float32)
                     * mx.rsqrt(var.astype(mx.float32) + eps))
                t = (bb.astype(mx.float32) - mean.astype(mx.float32)) * s \
                    + be.astype(mx.float32)
                scales.append(s)
                shifts.append(t)
            out[f"l{lvl}.c{ci}.s"] = mx.stack(scales)[:, None, None, :]
            out[f"l{lvl}.c{ci}.t"] = mx.stack(shifts)[:, None, None, :]

    fk = E.kids(fusion)
    types = [k["type"] for k in fk]
    if types != ["nn.JoinTable", "nn.SpatialConvolution", "nn.ReLU",
                 "nn.SpatialConvolution", "nn.ReLU", "nn.SpatialConvolution"]:
        raise ValueError(f"TOFlow direct: unexpected fusion layout {types}")
    w0, b0 = conv_of(fk[1])
    cin = int(w0.shape[-1])
    pad_to = 32
    if cin < pad_to:
        w0 = mx.concatenate(
            [w0, mx.zeros((*w0.shape[:-1], pad_to - cin), dtype=w0.dtype)],
            axis=-1,
        )
    out["fusion.pad_to"] = pad_to
    out["fusion.c0.w"], out["fusion.c0.b"] = w0, b0
    out["fusion.c1.w"], out["fusion.c1.b"] = conv_of(fk[3])
    out["fusion.c2.w"], out["fusion.c2.b"] = conv_of(fk[5])
    mx.eval([v for v in out.values() if isinstance(v, mx.array)])
    return out


class TOFlowDirect:
    """Direct forward over params extracted by _fold_convs."""

    _FLOW_STARTS = {"full": 0, "half": 1, "quarter": 2}

    def __init__(self, graph: dict, params: dict, dtype: Any,
                 flow_scale: str = "full"):
        if flow_scale not in self._FLOW_STARTS:
            raise ValueError(
                f"flow_scale must be one of {sorted(self._FLOW_STARTS)}, "
                f"got {flow_scale!r}")
        self.p = {k: (v.astype(dtype) if isinstance(v, mx.array)
                      and v.dtype == mx.float32 else v)
                  for k, v in _fold_convs(graph, params).items()}
        self.dtype = dtype
        self.flow_scale = flow_scale
        self._compiled: dict = {}

    def _cnn(self, lvl: int, x: Any) -> Any:
        p = self.p
        for ci in range(4):
            x = mx.conv2d(x, p[f"l{lvl}.c{ci}.w"], padding=3)
            x = _relu(x * p[f"l{lvl}.c{ci}.s"] + p[f"l{lvl}.c{ci}.t"])
        return mx.conv2d(x, p[f"l{lvl}.c4.w"], padding=3) + p[f"l{lvl}.c4.b"]

    def _flow(self, center: Any, nb: Any) -> Any:
        """Coarse-to-fine flow aligning nb onto center; both (B,H,W,3)."""
        start = self._FLOW_STARTS[self.flow_scale]
        cs, ns = [center], [nb]
        for _ in range(_LEVELS - 1):
            cs.append(_avgpool2(cs[-1]))
            ns.append(_avgpool2(ns[-1]))
        b = int(center.shape[0])
        flow = mx.zeros((b, int(cs[-1].shape[1]), int(cs[-1].shape[2]), 2),
                        dtype=center.dtype)
        # zero seed at the coarsest level (/8), exactly like the loader input
        flow = self._cnn(_LEVELS - 1,
                         mx.concatenate([cs[-1], ns[-1], flow], axis=-1)) + flow
        for lvl in range(_LEVELS - 2, start - 1, -1):
            flow = _up2(flow) * 2.0
            warped = _warp(ns[lvl], flow).astype(center.dtype)
            flow = self._cnn(
                lvl, mx.concatenate([cs[lvl], warped, flow], axis=-1)) + flow
        for _ in range(start):
            flow = _up2(flow) * 2.0
        return flow

    def _forward_flow(self, *inputs: Any) -> Any:
        """Just the batched neighbor->center flow, (6,H,W,2)."""
        center = inputs[0]
        nbs = [inputs[i - 1] for i in self.p["nb_index"]]   # 1-based graph indices
        c6 = mx.concatenate([center] * len(nbs), axis=0)
        n6 = mx.concatenate(nbs, axis=0)
        return self._flow(c6, n6)

    def _forward_fuse(self, flow: Any, *inputs: Any) -> Any:
        """Warp + fusion given a precomputed flow. Deblock/denoise passes do
        not move content, so the flow between frames is pass-invariant:
        chained passes can reuse pass 1's flow (measured equivalent within
        0.01 dB on static and moving fixtures, ~55 dB output agreement) and
        skip the dominant cost of every later pass."""
        center = inputs[0]
        nbs = [inputs[i - 1] for i in self.p["nb_index"]]
        nb = len(nbs)
        n6 = mx.concatenate(nbs, axis=0)
        warped = _warp(n6, flow).astype(center.dtype)
        parts = [center] + [warped[i:i + 1] for i in range(nb)]
        x = mx.concatenate(parts, axis=-1)
        pad_to = self.p["fusion.pad_to"]
        cin = int(x.shape[-1])
        if cin < pad_to:
            x = mx.concatenate(
                [x, mx.zeros((*x.shape[:-1], pad_to - cin), dtype=x.dtype)],
                axis=-1,
            )
        x = _relu(mx.conv2d(x, self.p["fusion.c0.w"], padding=4)
                  + self.p["fusion.c0.b"])
        x = _relu(mx.conv2d(x, self.p["fusion.c1.w"]) + self.p["fusion.c1.b"])
        return mx.conv2d(x, self.p["fusion.c2.w"]) + self.p["fusion.c2.b"]

    def _forward(self, *inputs: Any) -> Any:
        return self._forward_fuse(self._forward_flow(*inputs), *inputs)

    def _get(self, name: str, fn: Any, key: tuple) -> Any:
        got = self._compiled.get((name, key))
        if got is None:
            got = mx.compile(fn)
            self._compiled[(name, key)] = got
        return got

    def forward(self, inputs: list[Any]) -> Any:
        key = tuple((tuple(x.shape), str(x.dtype)) for x in inputs)
        return self._get("all", self._forward, key)(*inputs)

    def forward_flow(self, inputs: list[Any]) -> Any:
        key = tuple((tuple(x.shape), str(x.dtype)) for x in inputs)
        return self._get("flow", self._forward_flow, key)(*inputs)

    def forward_fuse(self, inputs: list[Any], flow: Any) -> Any:
        key = (tuple((tuple(x.shape), str(x.dtype)) for x in inputs),
               tuple(flow.shape), str(flow.dtype))
        return self._get("fuse", self._forward_fuse, key)(flow, *inputs)
