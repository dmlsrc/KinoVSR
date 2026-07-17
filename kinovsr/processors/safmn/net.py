"""MLX port of the SAFMN super-resolution family (Sun et al., "Spatially-Adaptive
Feature Modulation for Efficient Image Super-Resolution", ICCV 2023, and its
challenge-winning successors). Reimplemented from the reference architectures as a
spec; no upstream code is bundled. Two variants, auto-detected from the checkpoint:

- "real" (SAFMN_L_Real_LSDIR_x4, the paper's real-world model): dim 128 x 16
  AttBlocks; channel LayerNorm, 4-level SAFM pyramid (max-pool /2^i, per-level 3x3
  depthwise, nearest upsample, 1x1 aggregate, GELU gate) and a CCM FFN (3x3 expand
  + GELU + 1x1). Trained with the high-order Real-ESRGAN degradation (blur included)
  on LSDIR -- handles real video, motion blur and all.

Do NOT confuse "real" with the AIM 2025 challenge checkpoint also named "SAFMN-L"
(dim 96): that one was tuned on synthetically degraded stills for no-reference
perceptual metrics and hallucinates crusty texture over motion blur on real video
(verified against a reference-faithful torch computation -- it is the network, not
this port). The forward here runs it fine, but it is deliberately not a token.
- "light" (light_SAFMN++, 1st place fidelity track, AIS 2024 Real-Time 4K SR of
  compressed AVIF): dim 32 x 2 blocks; no norms, no biases, SimpleSAFM (single /8
  max-pool level, 3x3 depthwise, bilinear upsample, GELU gate) + CCM. Fidelity-
  trained on compressed content; ~45x fewer parameters.

- "purescale" / "purescale2x" / "purescale2x-sharp" (PureScale 2.0 by limitlesslab,
  https://github.com/limitlesslab/AI-upscaling-models): SAFMN-L retrained from
  scratch on the author's own curated real-world dataset with a FIXED SAFM branch --
  adaptive AVG pool (not max) and BICUBIC upsample (not nearest). The stock max+
  nearest combination broadcasts a hot activation as a constant block into the
  modulation gate, the known upstream lattice/blotch artifact; the fix requires
  retraining, which these checkpoints are. GAN models, JPEG-robust; "sharp" adds a
  deblurring component. The SAFM mode is inferred from the weights filename
  ("purescale" in the stem); checkpoint keys are identical to SAFMN-L. NOTE: these
  weights are licensed CC BY-NC-SA 4.0 (NON-COMMERCIAL use only) -- see
  ATTRIBUTION.md and weights/README.md; they are not distributed with this repo.

Layout: MLX-native NHWC; conv weights -> (O,kH,kW,I) at load; the depthwise convs
run as the 9-tap shift-add (their channel counts of 24/16 fail MLX's depthwise-gate
C%16 check -- see docs/VSR_PERFORMANCE_NOTES.md). Input is replicate-padded to a
multiple of 8 (the deepest pooling level) and the output cropped back.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from kinovsr.modeling.compile_cache import cached as _cached
from kinovsr.modeling.upsample import bicubic_up as _bicubic_up
from kinovsr.modeling.vsr_blocks import resize as _resize_bilinear
from kinovsr.modeling.weights import resolve_weights as _resolve_weights
from kinovsr.settings import default_settings

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
# Not bundled (download + convert; see weights/README.md).
_VARIANTS = {
    "light": "light_safmnpp.safetensors",             # light_SAFMN++, fidelity 4x (AIS 2024)
    "real": "safmn_l_real_lsdir_x4.safetensors",      # SAFMN_L_Real_LSDIR, real-world 4x
    "real2x": "safmn_l_real_lsdir_x2.safetensors",    # same family, 2x (HD -> 4K class)
    # PureScale 2.0 (limitlesslab) -- fixed-SAFM retrains, CC BY-NC-SA 4.0
    # (non-commercial only); see ATTRIBUTION.md / weights/README.md.
    "purescale": "safmn_purescale_x4.safetensors",            # real-world 4x
    "purescale2x": "safmn_purescale_x2.safetensors",          # real-world 2x
    "purescale2x-sharp": "safmn_purescale_sharper_x2.safetensors",  # 2x + deblur
}
_DEFAULT_VARIANT = "light"
_REPO = "https://github.com/sunny2109/SAFMN"
_PURESCALE_REPO = "https://github.com/limitlesslab/AI-upscaling-models"


def default_weights_path(variant: str = _DEFAULT_VARIANT) -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    """Variant token (real / light) or a path; falls back to $SAFMN_WEIGHTS."""
    if spec is None or spec == "":
        spec = default_settings().safmn_weights
    try:
        return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\nSAFMN weights are not bundled. Download a checkpoint from the SAFMN "
            f"repo ({_REPO}) and convert it (see weights/README.md), or point "
            f"$SAFMN_WEIGHTS / --safmn-weights at an existing .safetensors."
        ) from None


def _safm_mode_for(path: str | Path) -> str:
    """SAFM branch mode from the weights filename: PureScale checkpoints were
    TRAINED with avg pool + bicubic upsample ("fixed") and are key-identical to
    SAFMN-L, so the mode cannot be inferred from the tensors."""
    return "fixed" if "purescale" in Path(path).stem.lower() else "stock"


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load + lay out the checkpoint: conv weights -> NHWC (O,kH,kW,I); the depthwise
    weights (C,1,3,3) -> (C,3,3,1) for the shift-add; LayerNorm weight/bias stay 1-D.

    The SAFM branch mode is stamped into the dict as the plain string
    "__safm_mode__" (see _safm_mode_for) -- running a checkpoint with the wrong
    branch mode produces garbage."""
    resolved = resolve_weights(path)
    w = mx.load(str(resolved))
    p: dict = {}
    for k, v in w.items():
        a = mx.transpose(v, (0, 2, 3, 1)) if v.ndim == 4 else v
        p[k] = a.astype(dtype)
    if "to_feat.weight" not in p:
        raise ValueError("not a SAFMN checkpoint (missing to_feat.weight)")
    p["__safm_mode__"] = _safm_mode_for(resolved)
    return p


def _config(p: dict) -> tuple:
    """(variant, dim, n_blocks, scale, safm_mode, safm_up, pool_clamp) inferred from
    the weights. safm_mode is the trained pooling statistic ("stock" = max, "fixed" =
    avg -- NOT swappable, the gate calibrates to it); safm_up is the SAFM upsampler
    and defaults to the trained one (nearest for stock, bicubic for fixed) but is a
    mild shape-only choice that callers may override; pool_clamp (default 0 = off)
    winsorizes pooled SAFM features to mean +/- k*sigma per channel."""
    variant = "real" if "feats.0.norm1.weight" in p else "light"
    dim = int(p["to_feat.weight"].shape[0])
    i = 0
    while f"feats.{i}.conv1.proj.weight" in p or f"feats.{i}.norm1.weight" in p:
        i += 1
    scale = int(round((p["to_img.0.weight"].shape[0] / 3) ** 0.5))
    mode = p.get("__safm_mode__", "stock")
    up = "bicubic" if mode == "fixed" else "nearest"
    return variant, dim, i, scale, mode, up, 0.0


def _gelu(x: Any) -> Any:
    """Exact GELU (erf form), matching torch nn.GELU()."""
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


def _conv(x: Any, p: dict, key: str, pad: int = 0) -> Any:
    y = mx.conv2d(x, p[f"{key}.weight"], stride=1, padding=pad)
    b = p.get(f"{key}.bias")
    return y if b is None else y + b


def _dw3x3(x: Any, p: dict, key: str) -> Any:
    """3x3 depthwise (pad 1) as 9 shifted per-channel-scaled adds. The variants'
    depthwise channel counts (24 / 16) fail MLX's depthwise-gate C%16 check, and the
    shift-add is never pathological (see nafnet)."""
    w = p[f"{key}.weight"]
    h, wd = x.shape[1], x.shape[2]
    xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
    acc = None
    for i in range(3):
        for j in range(3):
            t = xp[:, i:i + h, j:j + wd, :] * w[:, i, j, 0]
            acc = t if acc is None else acc + t
    b = p.get(f"{key}.bias")
    return acc if b is None else acc + b


def _layernorm(x: Any, w: Any, b: Any, eps: float = 1e-6) -> Any:
    """Channel LayerNorm (torch channels_first): (x-mu)/sqrt(var+eps)*w + b over the
    channel axis, biased var, fp32 reduction. Manual on purpose -- mx.fast.layer_norm
    is transformer-shaped and slower for small-C/many-row norms."""
    xf = x.astype(mx.float32)
    mu = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mu) ** 2, axis=-1, keepdims=True)
    y = (xf - mu) * mx.rsqrt(var + eps)
    return (y * w.astype(mx.float32) + b.astype(mx.float32)).astype(x.dtype)


def _adaptive_bins(size: int, out: int) -> tuple:
    """torch adaptive-pooling bins: bin i covers [floor(i*size/out),
    ceil((i+1)*size/out)). Returns (idx (out,maxk) with clamped duplicate
    tail entries, weight (out,maxk) = 1/count for valid slots else 0)."""
    starts = [(i * size) // out for i in range(out)]
    ends = [-(-((i + 1) * size) // out) for i in range(out)]
    maxk = max(e - s for s, e in zip(starts, ends, strict=True))
    idx = [[min(s + j, e - 1) for j in range(maxk)]
           for s, e in zip(starts, ends, strict=True)]
    weight = [[1.0 / (e - s) if s + j < e else 0.0 for j in range(maxk)]
              for s, e in zip(starts, ends, strict=True)]
    return mx.array(idx, dtype=mx.int32), mx.array(weight, dtype=mx.float32)


def _pool_axis(x: Any, out: int, axis: int, op: str) -> Any:
    idx, weight = _adaptive_bins(x.shape[axis], out)
    maxk = idx.shape[1]
    g = mx.take(x, idx.reshape(-1), axis=axis)
    shape = list(x.shape)
    shape[axis:axis + 1] = [out, maxk]
    g = g.reshape(shape)
    if op == "max":
        return mx.max(g, axis=axis + 1)
    wshape = [1] * len(shape)
    wshape[axis], wshape[axis + 1] = out, maxk
    return mx.sum(g * weight.reshape(wshape).astype(g.dtype), axis=axis + 1)


def _adaptive_maxpool(x: Any, out_h: int, out_w: int) -> Any:
    """torch adaptive_max_pool2d on NHWC. Divisible sizes take the exact
    reshape fast path (the compiled hot path for multiple-of-8 frames);
    other sizes use torch's floor/ceil bin boundaries (max pooling needs no
    weights: the clamped duplicate tail entries cannot change a max)."""
    n, h, w, c = x.shape
    if h % out_h == 0 and w % out_w == 0:
        kh, kw = h // out_h, w // out_w
        return mx.max(x.reshape(n, out_h, kh, out_w, kw, c), axis=(2, 4))
    return _pool_axis(_pool_axis(x, out_h, 1, "max"), out_w, 2, "max")


def _adaptive_avgpool(x: Any, out_h: int, out_w: int) -> Any:
    """torch adaptive_avg_pool2d on NHWC (separable: per-axis bin counts
    factorize, so mean-of-means over the bins is the rectangle mean)."""
    n, h, w, c = x.shape
    if h % out_h == 0 and w % out_w == 0:
        kh, kw = h // out_h, w // out_w
        return mx.mean(x.reshape(n, out_h, kh, out_w, kw, c), axis=(2, 4))
    return _pool_axis(_pool_axis(x, out_h, 1, "avg"), out_w, 2, "avg")


def _nearest_up(x: Any, r: int) -> Any:
    n, h, w, c = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (n, h, r, w, r, c))
    return x.reshape(n, h * r, w * r, c)


def _nearest_to(x: Any, out_h: int, out_w: int) -> Any:
    """torch F.interpolate(mode='nearest') to an explicit size: src index =
    floor(dst * in/out). Integer-multiple targets take the broadcast path."""
    n, h, w, c = x.shape
    if out_h % h == 0 and out_w % w == 0 and out_h // h == out_w // w:
        return _nearest_up(x, out_h // h)
    ri = mx.array([(i * h) // out_h for i in range(out_h)], dtype=mx.int32)
    ci = mx.array([(j * w) // out_w for j in range(out_w)], dtype=mx.int32)
    return mx.take(mx.take(x, ri, axis=1), ci, axis=2)


def _pad_to(x: Any, out_h: int, out_w: int) -> Any:
    """Replicate-extend bottom/right rows/cols up to (out_h, out_w)."""
    n, h, w, c = x.shape
    if h < out_h:
        x = mx.concatenate(
            [x, mx.broadcast_to(x[:, h - 1:h], (n, out_h - h, w, c))], axis=1)
    if w < out_w:
        n2, h2, _, c2 = x.shape
        x = mx.concatenate(
            [x, mx.broadcast_to(x[:, :, w - 1:w], (n2, h2, out_w - w, c2))],
            axis=2)
    return x



def _pixel_shuffle(x: Any, r: int) -> Any:
    n, h, w, cr = x.shape
    c = cr // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * r, w * r, c)


# ---- "real" variant blocks (SAFMN-L) ----------------------------------------
def _pool_clamp(s: Any, k: float) -> Any:
    """Winsorize pooled SAFM features to mean +/- k*sigma per channel (spatial
    statistics). A hot activation that wins its max-pool cell is broadcast as a
    constant block into the modulation gate (the stock models' transient lattice);
    clamping only the outliers bounds that block's amplitude while leaving frames
    with no outliers numerically untouched -- unlike swapping the pooling, which
    shifts every value's statistics and breaks the trained gate.

    Frame-boundary pooled cells are EXEMPT (a one-cell margin, so 2/4/8 input px
    per level): synthetic border structures (letterbox bars, junk capture rows)
    saturate the features there, and that saturated response is self-limiting -- a
    quiet band in the output. Clamping it back into the plausible-texture range
    re-engages the GAN's texture machinery on garbage input and makes the border
    visibly bloom (measured on a junk border row; clamp direction does not matter).
    Statistics are computed over the same interior region, so an extreme border
    cannot inflate sigma and weaken mid-frame suppression."""
    n, h, w, c = s.shape
    if h <= 2 or w <= 2:
        return s
    sf = s.astype(mx.float32)
    core = sf[:, 1:-1, 1:-1]
    mu = mx.mean(core, axis=(1, 2), keepdims=True)
    sd = mx.sqrt(mx.mean((core - mu) ** 2, axis=(1, 2), keepdims=True))
    cl = mx.clip(core, mu - k * sd, mu + k * sd)
    mid = mx.concatenate([sf[:, 1:-1, :1], cl, sf[:, 1:-1, -1:]], axis=2)
    out = mx.concatenate([sf[:, :1], mid, sf[:, -1:]], axis=1)
    return out.astype(s.dtype)


def _safm(x: Any, p: dict, pre: str, mode: str = "stock", up: str = "nearest",
          clamp: float = 0.0) -> Any:
    """4-level spatially-adaptive feature modulation: chunk channels into 4, level i
    pools by 2^i, runs a per-level depthwise 3x3, upsamples back; the concatenated
    levels pass a 1x1 aggregate and gate x via GELU. mode="stock" pools with max
    (SAFMN paper, trained with nearest up); mode="fixed" pools with avg (PureScale
    retrains, trained with bicubic up -- kills the hot-pixel block-broadcast
    lattice). The POOLING is trained in and not swappable; the UPSAMPLER is a mild
    shape-only choice and may be overridden. clamp > 0 winsorizes the pooled
    features (see _pool_clamp)."""
    h, w = x.shape[1], x.shape[2]
    c4 = x.shape[-1] // 4
    outs = []
    for i in range(4):
        xc = x[..., i * c4:(i + 1) * c4]
        if i == 0:
            s = _dw3x3(xc, p, f"{pre}.mfr.0")
        else:
            # Reference: adaptive pooling to (h//2^i, w//2^i) on the un-padded
            # frame, then interpolate back to (h, w).
            ph, pw = max(1, h // 2 ** i), max(1, w // 2 ** i)
            s = (_adaptive_avgpool(xc, ph, pw) if mode == "fixed"
                 else _adaptive_maxpool(xc, ph, pw))
            if clamp > 0.0:
                s = _pool_clamp(s, clamp)
            s = _dw3x3(s, p, f"{pre}.mfr.{i}")
            # The PureScale retrains have no public arch source; keep
            # their trained-in scale-factor bicubic and replicate-extend
            # the sub-pixel remainder on non-multiple-of-2^i frames.
            s = (_pad_to(_bicubic_up(s, 2 ** i), h, w)
                 if up == "bicubic" else _nearest_to(s, h, w))
        outs.append(s)
    out = _conv(mx.concatenate(outs, axis=-1), p, f"{pre}.aggr")
    return _gelu(out) * x


def _ccm(x: Any, p: dict, pre: str) -> Any:
    return _conv(_gelu(_conv(x, p, f"{pre}.ccm.0", pad=1)), p, f"{pre}.ccm.2")


def _att_block_real(x: Any, p: dict, i: int, mode: str = "stock",
                    up: str = "nearest", clamp: float = 0.0) -> Any:
    x = _safm(_layernorm(x, p[f"feats.{i}.norm1.weight"], p[f"feats.{i}.norm1.bias"]),
              p, f"feats.{i}.safm", mode, up, clamp) + x
    return _ccm(
        _layernorm(x, p[f"feats.{i}.norm2.weight"], p[f"feats.{i}.norm2.bias"]),
        p,
        f"feats.{i}.ccm",
    ) + x


# ---- "light" variant blocks (light_SAFMN++) ----------------------------------
def _simple_safm(x: Any, p: dict, pre: str) -> Any:
    """Single-level modulation: 3x3 proj, split channels; one half max-pools to
    (H/8, W/8), runs a depthwise 3x3, bilinear-upsamples back and GELU-gates its
    source; concat with the other half, GELU, 1x1 out. No biases."""
    h, w = x.shape[1], x.shape[2]
    y = _conv(x, p, f"{pre}.proj", pad=1)
    d = y.shape[-1] // 2
    x0, x1 = y[..., :d], y[..., d:]
    s = _adaptive_maxpool(x0, max(1, h // 8), max(1, w // 8))
    s = _dw3x3(s, p, f"{pre}.dwconv")
    s = _resize_bilinear(s, h, w, False)
    s = _gelu(s) * x0
    z = mx.concatenate([x1, s], axis=-1)
    return _conv(_gelu(z), p, f"{pre}.out")


def _att_block_light(x: Any, p: dict, i: int) -> Any:
    y = _simple_safm(x, p, f"feats.{i}.conv1")
    return _conv(  # no per-block residual in the light net
        _gelu(_conv(y, p, f"feats.{i}.conv2.conv.0", pad=1)),
        p,
        f"feats.{i}.conv2.conv.2",
    )


def safmn(x: Any, p: dict, cfg: tuple | None = None) -> Any:
    """Upscale one batch. x: (N,H,W,3) in [0,1] -> (N, scale*H, scale*W, 3)."""
    if cfg is None:
        cfg = _config(p)
    variant, _dim, n_blocks, scale, safm_mode, safm_up, pool_clamp = cfg
    dt = p["to_feat.weight"].dtype
    # No pad: the reference runs arbitrary sizes through adaptive pooling
    # (its release harnesses feed odd frames un-padded), and the SAFM grids
    # differ frame-wide if pooling bins move.
    feat = _conv(x.astype(dt), p, "to_feat", pad=1)
    y = feat
    for i in range(n_blocks):
        y = (_att_block_real(y, p, i, safm_mode, safm_up, pool_clamp)
             if variant == "real" else _att_block_light(y, p, i))
    y = y + feat
    out = _pixel_shuffle(_conv(y, p, "to_img.0", pad=1), scale)
    return mx.clip(out, 0.0, 1.0)


_COMPILE_CACHE: dict = {}


def make_forward(p: dict, cfg: tuple | None = None, compile: bool = True):
    """Per-frame forward x -> upscaled image, mx.compiled once per checkpoint."""
    if cfg is None:
        cfg = _config(p)

    def run(x):
        return safmn(x, p, cfg=cfg)

    if not compile:
        return run
    return _cached(_COMPILE_CACHE, (id(p), cfg), lambda: mx.compile(run))


_log = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = load_params()
    cfg = _config(p)
    _log.info(f"loaded SAFMN: variant={cfg[0]} dim={cfg[1]} blocks={cfg[2]} scale={cfg[3]}x")
    mx.random.seed(0)
    x = mx.clip(mx.random.uniform(shape=(1, 64, 96, 3)), 0, 1)
    mx.eval(x)
    out = safmn(x, p, cfg)
    mx.eval(out)
    _log.info(f"{tuple(x.shape)} -> {tuple(out.shape)}, finite={bool(mx.all(mx.isfinite(out)))}")
