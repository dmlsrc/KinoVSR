"""Shared BasicVSR backbone building blocks (MLX, NHWC).

Generic primitives used by BOTH the BasicVSR++ and RealBasicVSR ports: conv /
activation helpers, bilinear sampling + flow-warp + resize, the SPyNet optical-flow
pyramid, residual blocks, and pixel-shuffle. They live here so neither upscaler
depends on the other -- RealBasicVSR previously reached into basicvsrpp/net.py for
these, which coupled two sibling architectures.

Convention: NHWC throughout. Conv weights are pre-transposed to MLX's (O,kH,kW,I)
by each net's load_params; flow is (N,H,W,2) = (x-offset, y-offset), matching
flow_warp.
"""
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from .compile_cache import cached as _cached


def relu(x: Any) -> Any:
    return mx.maximum(x, 0)


def lrelu(x: Any, slope: float = 0.1) -> Any:
    return mx.where(x >= 0, x, x * slope)


def conv(x: Any, p: dict, key: str, stride: int = 1, pad: int = 1, groups: int = 1) -> Any:
    y = mx.conv2d(x, p[f"{key}.weight"], stride=stride, padding=pad, groups=groups)
    b = p.get(f"{key}.bias")
    return y if b is None else y + b


def _bilinear(x: Any, sy: Any, sx: Any, pad: str = "border") -> Any:
    """Sample x (N,H,W,C) at (sy,sx) (each (N,oH,oW)) -> (N,oH,oW,C). 'border'
    clamps out-of-range to the edge; 'zeros' returns 0 outside."""
    n, h, w, c = x.shape
    oh, ow = sy.shape[1], sy.shape[2]
    y0 = mx.floor(sy)
    x0 = mx.floor(sx)
    ly = (sy - y0)[..., None]
    lx = (sx - x0)[..., None]
    y0i = y0.astype(mx.int32)
    x0i = x0.astype(mx.int32)
    flat = x.reshape(n, h * w, c)

    def g(yi: Any, xi: Any) -> Any:
        idx = (mx.clip(yi, 0, h - 1) * w + mx.clip(xi, 0, w - 1)).reshape(n, oh * ow, 1)
        v = mx.take_along_axis(flat, mx.broadcast_to(idx, (n, oh * ow, c)), axis=1).reshape(n, oh, ow, c)
        if pad == "zeros":
            valid = ((yi >= 0) & (yi <= h - 1) & (xi >= 0) & (xi <= w - 1)).astype(x.dtype)
            v = v * valid[..., None]
        return v

    v00 = g(y0i, x0i)
    v01 = g(y0i, x0i + 1)
    v10 = g(y0i + 1, x0i)
    v11 = g(y0i + 1, x0i + 1)
    out = (1 - ly) * (1 - lx) * v00 + (1 - ly) * lx * v01 + ly * (1 - lx) * v10 + ly * lx * v11
    return out.astype(x.dtype)   # fp32 grid weights would otherwise upcast features


def flow_warp(x: Any, flow: Any, pad: str = "zeros") -> Any:
    """Warp x (N,H,W,C) by flow (N,H,W,2): out[p] = x[p + flow[p]]."""
    n, h, w, _ = x.shape
    gy, gx = mx.meshgrid(mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32), indexing="ij")
    sx = gx[None] + flow[..., 0]
    sy = gy[None] + flow[..., 1]
    return _bilinear(x, sy, sx, pad)


def box3(x: Any) -> Any:
    """Replicate-padded 3x3 mean for NHWC tensors.

    Used for cheap hidden-state cleanup/risk smoothing. Avoids MLX grouped
    conv's slow path for small depthwise filters.
    """
    _, h, w, _ = x.shape
    yp = mx.concatenate([x[:, :1], x, x[:, -1:]], axis=1)
    xp = mx.concatenate([yp[:, :, :1], yp, yp[:, :, -1:]], axis=2)
    acc = None
    for i in range(3):
        for j in range(3):
            t = xp[:, i:i + h, j:j + w, :]
            acc = t if acc is None else acc + t
    return (acc / 9.0).astype(x.dtype)


def history_improve_gate(curr: Any, prev: Any, flow: Any, dtype: Any,
                         strength: float = 1.0) -> Any:
    """Per-pixel history-admission gate in [0, strength], shape (N,H,W,1).

    Admits recurrent history only where the flow warp measurably IMPROVES the
    photometric residual against the current frame (versus using the previous
    frame unwarped) and the warped match is close. Near-static regions with no
    demonstrable improvement get ~0 history, which prevents the recurrence from
    locking and re-sharpening its own hallucinations (etching / propagation
    smear); regions with real, well-tracked motion pass through. A zero gate
    reproduces the nets' trained cold-start distribution (zero-initialized
    propagation features), so gating is in-distribution.

    Same formula and constants as RealViformer's driver gate: the improve ramp
    saturates at one 8-bit level (0.004) of residual improvement; the match
    falloff sigma is 0.035 (~9 levels). Arithmetic runs in fp32, cast at the end.
    """
    curr32 = curr.astype(mx.float32)
    prev32 = prev.astype(mx.float32)
    warped_prev = flow_warp(prev32, flow.astype(mx.float32), "border")
    resid_warp = mx.mean(mx.abs(curr32 - warped_prev), axis=-1, keepdims=True)
    resid_zero = mx.mean(mx.abs(curr32 - prev32), axis=-1, keepdims=True)
    improve = mx.clip((resid_zero - resid_warp) / 0.004, 0.0, 1.0)
    match = mx.exp(-((resid_warp / 0.035) ** 2))
    return (float(strength) * improve * match).astype(dtype)


def resize(x: Any, oh: int, ow: int, align_corners: bool) -> Any:
    """Bilinear resize NHWC x to (oh, ow) (edge-clamped), matching torch's
    align_corners True/False coordinate maps."""
    n, h, w, _ = x.shape
    if align_corners:
        ry = (h - 1) / (oh - 1) if oh > 1 else 0.0
        rx = (w - 1) / (ow - 1) if ow > 1 else 0.0
        sy1 = mx.arange(oh, dtype=mx.float32) * ry
        sx1 = mx.arange(ow, dtype=mx.float32) * rx
    else:
        sy1 = (mx.arange(oh, dtype=mx.float32) + 0.5) * (h / oh) - 0.5
        sx1 = (mx.arange(ow, dtype=mx.float32) + 0.5) * (w / ow) - 0.5
    sy = mx.broadcast_to(sy1.reshape(1, oh, 1), (n, oh, ow))
    sx = mx.broadcast_to(sx1.reshape(1, 1, ow), (n, oh, ow))
    return _bilinear(x, sy, sx, "border")


def make_lanczos_plan(n_in: int, n_out: int) -> tuple:
    """Precomputed 1-D Lanczos-3 resample plan: (indices, weights) with shape
    (taps, n_out), int32/float32. Built once (pure Python), applied per frame
    as `taps` gathers + a weighted sum -- GPU-resident, no per-frame CoreImage
    round trip, and no dense resample matrix (which would be >99% zeros).

    Handles both directions: downscales stretch the kernel by 1/scale for
    proper antialiasing. Tap indices are edge-clamped (replicate boundary);
    weights are normalized per output position (windowed sinc does not sum to
    exactly 1). At n_in == n_out the plan is an exact identity."""
    import math

    scale = n_out / n_in
    support = 3.0 * max(1.0, 1.0 / scale)
    taps = max(2, int(math.ceil(support * 2)))

    def lanczos3(t: float) -> float:
        t = abs(t)
        if t < 1e-9:
            return 1.0
        if t >= 3.0:
            return 0.0
        pt = math.pi * t
        return 3.0 * math.sin(pt) * math.sin(pt / 3.0) / (pt * pt)

    k_scale = min(1.0, scale)      # stretch the kernel when downscaling
    idx_rows: list = [[] for _ in range(taps)]
    w_rows: list = [[] for _ in range(taps)]
    for j in range(n_out):
        src = (j + 0.5) / scale - 0.5
        first = int(math.floor(src - support)) + 1
        ws = [lanczos3((src - (first + k)) * k_scale) for k in range(taps)]
        total = sum(ws) or 1.0
        for k in range(taps):
            idx_rows[k].append(min(max(first + k, 0), n_in - 1))
            w_rows[k].append(ws[k] / total)
    return (mx.array(idx_rows, dtype=mx.int32), mx.array(w_rows, dtype=mx.float32))


def resample_width(x: Any, plan: tuple) -> Any:
    """Apply a make_lanczos_plan along the width axis of (H, W, C) or
    (N, H, W, C); returns the same rank with the new width."""
    idx, w = plan
    axis = 1 if x.ndim == 3 else 2
    shape = [1] * x.ndim
    shape[axis] = int(w.shape[1])
    acc = None
    for k in range(int(idx.shape[0])):
        t = mx.take(x, idx[k], axis=axis) * w[k].reshape(shape)
        acc = t if acc is None else acc + t
    return acc


def _avgpool2(x: Any) -> Any:
    """2x2 average pool, stride 2 (input dims even)."""
    n, h, w, c = x.shape
    return x.reshape(n, h // 2, 2, w // 2, 2, c).mean(axis=(2, 4))


def _replicate_pad_to(x: Any, oh: int, ow: int) -> Any:
    """Replicate-pad bottom/right to an exact spatial size."""
    h, w = x.shape[1], x.shape[2]
    if h < oh:
        rows = mx.broadcast_to(x[:, -1:, :, :], (x.shape[0], oh - h, w, x.shape[3]))
        x = mx.concatenate([x, rows], axis=1)
    if w < ow:
        cols = mx.broadcast_to(x[:, :, -1:, :], (x.shape[0], x.shape[1], ow - w, x.shape[3]))
        x = mx.concatenate([x, cols], axis=2)
    return x


# ---- SPyNet ----------------------------------------------------------------
def pad_spynet_gates(p: dict) -> None:
    """Zero-pad each SPyNet basic module's FIRST conv from 8 to 16 input channels
    (in place, same key). C=8 fails MLX's implicit-GEMM gate (C<=4 or C%16==0,
    mlx conv.cpp) so the finest-level 7x7 conv ran on the ~2.5x-slower general
    kernel. spynet_flow appends matching zero channels to the module input, so the
    math is exact (zero columns x zero channels). Call after load_params; a no-op
    when the keys are absent or already padded."""
    for lvl in range(6):
        k = f"spynet.basic_module.{lvl}.basic_module.0.conv.weight"
        if k in p and p[k].shape[-1] == 8:
            w = p[k]
            p[k] = mx.concatenate(
                [w, mx.zeros((*w.shape[:3], 8), dtype=w.dtype)], axis=-1)


def _spynet_basic_module(x: Any, p: dict, lvl: int) -> Any:
    base = f"spynet.basic_module.{lvl}.basic_module"
    for j in (0, 1, 2, 3):
        x = relu(conv(x, p, f"{base}.{j}.conv", pad=3))
    return conv(x, p, f"{base}.4.conv", pad=3)


def spynet_flow(p: dict, ref: Any, supp: Any) -> Any:
    """Optical flow from ref to supp; both (N,H,W,3) in [0,1]. -> (N,H,W,2)."""
    n, h, w, _ = ref.shape
    w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
    h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
    ref = resize(ref, h_up, w_up, False)
    supp = resize(supp, h_up, w_up, False)
    mean, std = p["spynet.mean"], p["spynet.std"]
    refs = [(ref - mean) / std]
    supps = [(supp - mean) / std]
    levels = 6 if w_up > 32 else max(1, int(math.log2(w_up)))
    for _ in range(levels - 1):
        refs.append(_avgpool2(refs[-1]))
        supps.append(_avgpool2(supps[-1]))
    refs = refs[::-1]
    supps = supps[::-1]
    # Match BasicSR/RealViformer SPyNet exactly: the initial flow is half the
    # coarsest pyramid level, then every level starts with 2x upsampling and
    # replicate bottom/right padding if the coarsest level is odd.
    flow = mx.zeros((n, refs[0].shape[1] // 2, refs[0].shape[2] // 2, 2), dtype=ref.dtype)
    # Gate-padded first conv (pad_spynet_gates): append zero channels to match.
    inp_pad = p["spynet.basic_module.0.basic_module.0.conv.weight"].shape[-1] - 8
    for lvl in range(levels):
        flow_up = resize(flow, flow.shape[1] * 2, flow.shape[2] * 2, True) * 2.0
        flow_up = _replicate_pad_to(flow_up, refs[lvl].shape[1], refs[lvl].shape[2])
        warped = flow_warp(supps[lvl], flow_up, "border")
        parts = [refs[lvl], warped, flow_up]                          # (N,h,w,8)
        if inp_pad:
            parts.append(mx.zeros((*refs[lvl].shape[:3], inp_pad), dtype=ref.dtype))
        inp = mx.concatenate(parts, axis=-1)
        flow = flow_up + _spynet_basic_module(inp, p, lvl)
    flow = resize(flow, h, w, False)
    return mx.stack([flow[..., 0] * (w / w_up), flow[..., 1] * (h / h_up)], axis=-1)


_SPYNET_COMPILE_CACHE: dict = {}


def compiled_spynet_flow(p: dict, ref: Any, supp: Any) -> Any:
    """spynet_flow, mx.compiled + cached per checkpoint (~1.1x).

    The reference runs SPyNet in fp32; this port runs it in fp16, and compiling the
    fp16 path reorders ops so the flow shifts ~0.02 vs op-by-op (fp32 reorders <3e-4).
    That moves the final SR by <=0.012 max / ~6e-4 mean on [0,1] -- fp16 noise on a net
    that is already an fp16 approximation of the fp32 reference. Keyed by id(p)."""
    fn = _cached(_SPYNET_COMPILE_CACHE, id(p), lambda: mx.compile(lambda r, s: spynet_flow(p, r, s)))
    return fn(ref, supp)


# ---- residual blocks + pixel-shuffle ---------------------------------------
def _resblock(x: Any, p: dict, key: str) -> Any:
    """ResidualBlockNoBN: x + conv2(relu(conv1(x))), res_scale 1."""
    return x + conv(relu(conv(x, p, f"{key}.conv1")), p, f"{key}.conv2")


def _resblocks_with_input(x: Any, p: dict, prefix: str) -> Any:
    # Block count is read from the checkpoint, so c64n7 (7) and c128n25 (25) and
    # the 15-block restoration variants all load without a hardcoded count.
    x = lrelu(conv(x, p, f"{prefix}.main.0"))
    i = 0
    while f"{prefix}.main.2.{i}.conv1.weight" in p:
        x = _resblock(x, p, f"{prefix}.main.2.{i}")
        i += 1
    return x


_RESBLOCKS_COMPILE_CACHE: dict = {}


def compiled_resblocks(x: Any, p: dict, prefix: str) -> Any:
    """_resblocks_with_input, mx.compiled + cached per (checkpoint, prefix).

    The resblock stack fuses for ~1.05-1.07x over the op-by-op path, pure and
    byte-identical (profiled, MLX cache capped). For the recurrent VSR loops, where
    the stack runs many times per frame. Keyed by (id(p), prefix); the cache entry
    closes over p so its id stays stable. Do NOT call this from inside another
    mx.compile'd step (it would nest compiles) -- call _resblocks_with_input there.
    """
    fn = _cached(_RESBLOCKS_COMPILE_CACHE, (id(p), prefix),
                 lambda: mx.compile(lambda x: _resblocks_with_input(x, p, prefix)))
    return fn(x)


def _pixel_shuffle(x: Any, r: int) -> Any:
    """(N,H,W,C*r^2) -> (N,H*r,W*r,C), torch PixelShuffle channel order."""
    n, h, w, c4 = x.shape
    c = c4 // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * r, w * r, c)


def _pixelshuffle_pack(x: Any, p: dict, prefix: str, r: int = 2) -> Any:
    return _pixel_shuffle(conv(x, p, f"{prefix}.upsample_conv"), r)


# VTOpticalFlow sessions for flow_mode="vt", cached per frame geometry. The
# McTemporalDenoiser owns the session plumbing (Quality tier + the mandatory
# synthetic-shift self-test that catches VT's silent-zero sizes/orientations);
# strength/window are irrelevant here, it is used purely as a flow service.
_VT_FLOW_CACHE: dict = {}


def _vt_flow_service(w: int, h: int):
    key = (int(w), int(h))
    if key not in _VT_FLOW_CACHE:
        from kinovsr.processors.mc import McTemporalDenoiser   # noqa: I001  # lazy: pyobjc + VT session
        _VT_FLOW_CACHE[key] = McTemporalDenoiser(
            w, h, strength=0.0, window=1, self_test=True)
    return _VT_FLOW_CACHE[key]


def _vt_flows(frames: list) -> tuple:
    """VTOpticalFlow for both propagation directions, in _compute_flows'
    conventions: flows_forward[i] pulls frame i into frame i+1's geometry
    (anchored at i+1); flows_backward[i] pulls frame i+1 into frame i's.

    One VT call per direction per pair: the service's forward flow with
    (source=a, next=b) negated is the pull-flow anchored at b -- the mc
    denoiser's own empirically validated convention (warp(ref, -fwd)).
    """
    if frames[0].shape[0] != 1:
        raise ValueError("flow_mode='vt' supports batch-1 frames only")
    h, w = int(frames[0].shape[1]), int(frames[0].shape[2])
    svc = _vt_flow_service(w, h)
    dt = frames[0].dtype
    ff, fb = [], []
    for i in range(len(frames) - 1):
        a = frames[i][0].astype(mx.float32)
        b = frames[i + 1][0].astype(mx.float32)
        fwd_ab, _ = svc._compute_flows(b, [a])[0]   # source=a, next=b
        fwd_ba, _ = svc._compute_flows(a, [b])[0]   # source=b, next=a
        ff.append((-fwd_ab)[None].astype(dt))
        fb.append((-fwd_ba)[None].astype(dt))
        mx.eval(ff[-1], fb[-1])
    return ff, fb


def _compute_flows(frames: list, p: dict, flow_mode: str = "spynet") -> tuple:
    """flows_forward[i] = flow(i+1 -> i); flows_backward[i] = flow(i -> i+1).

    Each flow is materialized as computed: SPyNet upsizes to a multiple of 32 and
    builds the BasicSR pyramid, so holding all 2*(T-1) of them as one lazy graph
    spikes memory; per-flow eval keeps only the small (H,W,2) results alive."""
    if flow_mode == "zero":
        zeros = [mx.zeros((*frames[0].shape[:3], 2), dtype=frames[0].dtype)
                 for _ in range(len(frames) - 1)]
        if zeros:
            mx.eval(*zeros)
        return list(zeros), list(zeros)
    if flow_mode == "vt":
        return _vt_flows(frames)
    if flow_mode != "spynet":
        raise ValueError(
            f"unknown flow_mode {flow_mode!r}; expected 'spynet', 'zero', or 'vt'")
    fb, ff = [], []
    for i in range(len(frames) - 1):
        b = compiled_spynet_flow(p, frames[i], frames[i + 1])
        f = compiled_spynet_flow(p, frames[i + 1], frames[i])
        mx.eval(b, f)
        fb.append(b)
        ff.append(f)
    return ff, fb
