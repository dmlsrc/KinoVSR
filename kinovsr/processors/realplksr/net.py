"""MLX port of RealPLKSR (Partial Large Kernel CNN for efficient SR).

RealPLKSR is the GAN-stable rework of PLKSR (Lee et al., arXiv 2404.11848) that
the neosr author released on the PLKSR repo (issue #4) and the paper author
endorsed. Reimplemented from the spandrel reference architecture as a spec; no
upstream code is bundled. The trunk is a plain residual stack of PLKBlocks -- no
U-Net, no pooled modulation gate, so it structurally cannot produce SAFMN's
max-pool block lattice (ordinary GAN specks are still possible).

One PLKBlock:
  x_skip = x
  x = layer_norm(x)                 # LayerNorm variant only (channel LN); else Identity
  x = channel_mixer(x)              # DCCM: 3x3 (dim->2dim), Mish, 3x3 (2dim->dim)
  x = lk(x)                         # partial large kernel: 17x17 conv on the first
                                    #   split_ratio*dim channels only, rest passthrough
  x = attn(x)                       # EA: x * sigmoid(3x3(x))
  x = refine(x)                     # 1x1
  x = norm(x)                       # GroupNorm(4) variant only; else Identity
  return x + x_skip

Trunk: feats.0 (3x3, 3->dim), n_blocks PLKBlocks, dropout (eval no-op), feats.last
(3x3, dim -> out_ch*scale^2); then feats(x) + repeat_interleave(x, scale^2); then
to_img -> pixel-shuffle (GroupNorm checkpoints) or DySample (LayerNorm checkpoints).

Two checkpoint families, auto-detected (spandrel's detection rules):
- GroupNorm + PixelShuffle: e.g. 4xNomosWebPhoto_RealPLKSR (4x photo restoration).
  fp16-UNSAFE per the community (spandrel flags supports_half=False for this
  variant) -- run bf16 or fp32, or the measured fp16 islands.
- LayerNorm + DySample: e.g. 2xPublic_realplksr_dysample_layernorm_real[_nn] (2x).
  fp16-safe; the LayerNorm is what stabilizes half precision.

Layout: MLX-native NHWC; conv weights -> (O,kH,kW,I) at load. The partial LK conv
is a 17x17 dense conv on ~16 channels (split_ratio 0.25 of dim 64); channel norms
run their reductions in fp32.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from kinovsr.compile_cache import cached as _cached
from kinovsr.settings import default_settings
from kinovsr.vsr_blocks import _bilinear
from kinovsr.weights import resolve_weights as _resolve_weights

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
# Not bundled (download + convert; see weights/README.md).
_VARIANTS = {
    "nomos4x": "4xnomoswebphoto_realplksr.safetensors",              # 4x photo, GN+PixelShuffle
    "public2x": "2xpublic_realplksr_dysample_layernorm_real.safetensors",       # 2x, LN+DySample
    "public2x-nn": "2xpublic_realplksr_dysample_layernorm_real_nn.safetensors",  # 2x, no-noise train
}
_DEFAULT_VARIANT = "public2x"
_REPO = "https://github.com/Phhofm/models"


def default_weights_path(variant: str = _DEFAULT_VARIANT) -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    """Variant token or a path; falls back to $REALPLKSR_WEIGHTS."""
    if spec is None or spec == "":
        spec = default_settings().realplksr_weights
    try:
        return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, _DEFAULT_VARIANT)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n\nRealPLKSR weights are not bundled. Download a checkpoint from "
            f"Phhofm/models ({_REPO}) and convert it (see weights/README.md), or point "
            f"$REALPLKSR_WEIGHTS / --realplksr-weights at an existing .safetensors."
        ) from None


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load + lay out the checkpoint: 4D conv weights -> NHWC (O,kH,kW,I); 1-D
    norm/bias vectors unchanged. init_pos (DySample) keeps its trailing channel axis.
    """
    resolved = resolve_weights(path)
    w = mx.load(str(resolved))
    p: dict = {}
    for k, v in w.items():
        p[k] = (mx.transpose(v, (0, 2, 3, 1)) if v.ndim == 4 else v).astype(dtype)
    if "feats.0.weight" not in p:
        raise ValueError("not a RealPLKSR checkpoint (missing feats.0.weight)")
    return p


def _config(p: dict) -> tuple:
    """(dim, n_blocks, kernel_size, pdim, scale, layer_norm, dysample, groups)."""
    dim = int(p["feats.0.weight"].shape[0])
    last = max(int(k.split(".")[1]) for k in p if k.startswith("feats."))
    n_blocks = last - 2                                  # feats.0 conv + blocks + dropout + final
    kernel_size = int(p["feats.1.lk.conv.weight"].shape[1])
    pdim = int(p["feats.1.lk.conv.weight"].shape[0])
    out_final = int(p[f"feats.{last}.weight"].shape[0])
    scale = int(round((out_final / 3) ** 0.5))           # out_ch=in_ch=3, final = 3*scale^2
    layer_norm = "feats.1.layer_norm.weight" in p
    dysample = "to_img.init_pos" in p
    groups = int(p["to_img.offset.weight"].shape[0]) // (2 * scale * scale) if dysample else 0
    return dim, n_blocks, kernel_size, pdim, scale, layer_norm, dysample, groups


# --- primitives --------------------------------------------------------------
def _conv(x: Any, p: dict, key: str, pad: int = 0) -> Any:
    y = mx.conv2d(x, p[f"{key}.weight"], stride=1, padding=pad)
    b = p.get(f"{key}.bias")
    return y if b is None else y + b


def _mish(x: Any) -> Any:
    """x * tanh(softplus(x)) in x's own dtype. The softplus is the numerically
    stable form softplus(x) = max(x,0) + log1p(exp(-|x|)): exp(-|x|) is in (0,1]
    so it never overflows fp16, and tanh(softplus) in (0,1) keeps the product ~x.
    Running this in fp16 (rather than a fp32 island) is 2x faster for a ~4e-3
    difference vs fp32 -- measured negligible through the trunk (see the perf
    audit); no overflow because DCCM-mid activations stay ~10-30."""
    sp = mx.maximum(x, 0) + mx.log1p(mx.exp(-mx.abs(x)))
    return x * mx.tanh(sp)


def _layernorm(x: Any, w: Any, b: Any, eps: float = 1e-6) -> Any:
    """Channel LayerNorm (torch channels_first): normalize over the channel axis,
    biased var, fp32 reduction."""
    xf = x.astype(mx.float32)
    mu = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mu) ** 2, axis=-1, keepdims=True)
    y = (xf - mu) * mx.rsqrt(var + eps)
    return (y * w.astype(mx.float32) + b.astype(mx.float32)).astype(x.dtype)


def _groupnorm(x: Any, w: Any, b: Any, groups: int = 4, eps: float = 1e-5) -> Any:
    """torch GroupNorm(groups, C): per sample, per group, normalize over (H,W,C/g).

    fp32 reduction is load-bearing, not a nicety: refine-out activations reach
    ~370, so both mean(x) and mean(x^2) overflow fp16 (sum of ~1e5 elements, and
    x^2 ~ 1.4e5 > 65504). This reduces the CONTIGUOUS spatial axes (1,2) first,
    then the small channel-group axis -- 4.5x faster than a single strided reduce
    over (H,W,cg), which is the port's #1 hot op on the 4x GroupNorm variant (perf
    audit). Two-pass (mu, then (x-mu)^2) rather than E[x^2]-E[x]^2 so the variance
    can never go negative from cancellation."""
    n, h, wd, c = x.shape
    cg = c // groups
    cnt = h * wd * cg
    xf = x.astype(mx.float32).reshape(n, h, wd, groups, cg)
    mu = mx.sum(mx.sum(xf, axis=(1, 2), keepdims=True), axis=4, keepdims=True) / cnt
    d = xf - mu
    var = mx.sum(mx.sum(d * d, axis=(1, 2), keepdims=True), axis=4, keepdims=True) / cnt
    y = (d * mx.rsqrt(var + eps)).reshape(n, h, wd, c)
    return (y * w.astype(mx.float32) + b.astype(mx.float32)).astype(x.dtype)


def _pixel_shuffle(x: Any, r: int) -> Any:
    n, h, w, cr = x.shape
    c = cr // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * r, w * r, c)


def _dysample(x: Any, p: dict, scale: int, groups: int) -> Any:
    """DySample (Learning to Upsample by Learning to Sample, arXiv 2308.15085).

    Predicts a per-output-pixel sampling offset and bilinearly gathers the input.
    The reference builds normalized grid_sample coordinates
    coords = 2*(pixel_center + offset)/size - 1 and samples with align_corners=False;
    that round trip collapses to sampling the LR grid at (lr_index + offset) in pixel
    space, so we add the offset to the integer LR grid and call _bilinear (border
    clamp = the reference padding_mode). Offset channels are laid out [xy, groups,
    scale, scale] and pixel-shuffled to (groups, scale*H, scale*W); each of the
    scale^2 sub-pixels in an output block carries its own learned offset.
    """
    n, h, w, cin = x.shape
    gr2 = groups * scale * scale
    o = _conv(x, p, "to_img.offset")
    s = _conv(x, p, "to_img.scope")
    init_pos = p["to_img.init_pos"].reshape(-1)          # (2*gr2,)
    offset = o * mx.sigmoid(s) * 0.5 + init_pos
    off_x = offset[..., :gr2].astype(mx.float32)
    off_y = offset[..., gr2:].astype(mx.float32)
    jx = mx.arange(w, dtype=mx.float32).reshape(1, 1, w, 1)
    iy = mx.arange(h, dtype=mx.float32).reshape(1, h, 1, 1)
    sx_lr = off_x + jx                                   # (N,H,W,gr2), LR pixel coords
    sy_lr = off_y + iy

    def shuffle(a: Any) -> Any:
        a = a.reshape(n, h, w, groups, scale, scale)     # channel = [group, rh, rw]
        a = mx.transpose(a, (0, 3, 1, 4, 2, 5))          # (N, g, H, rh, W, rw)
        return a.reshape(n * groups, h * scale, w * scale)

    sx = shuffle(sx_lr)
    sy = shuffle(sy_lr)
    cg = cin // groups
    xg = mx.transpose(x.reshape(n, h, w, groups, cg), (0, 3, 1, 2, 4)).reshape(
        n * groups, h, w, cg)
    out = _bilinear(xg, sy, sx, "border")                # (N*g, rH, rW, cg)
    out = mx.transpose(out.reshape(n, groups, h * scale, w * scale, cg),
                       (0, 2, 3, 1, 4)).reshape(n, h * scale, w * scale, cin)
    return _conv(out, p, "to_img.end_conv")              # 1x1 -> out_ch


def _plk_block(x: Any, p: dict, i: int, ks: int, pdim: int,
               layer_norm: bool, groups: int) -> Any:
    pre = f"feats.{i}"
    x_skip = x
    if layer_norm:
        x = _layernorm(x, p[f"{pre}.layer_norm.weight"], p[f"{pre}.layer_norm.bias"])
    # DCCM channel mixer
    y = _conv(x, p, f"{pre}.channel_mixer.0", pad=1)
    y = _mish(y)
    x = _conv(y, p, f"{pre}.channel_mixer.2", pad=1)
    # partial large kernel: first pdim channels only
    x1 = _conv(x[..., :pdim], p, f"{pre}.lk.conv", pad=ks // 2)
    x = mx.concatenate([x1, x[..., pdim:]], axis=-1)
    # element-wise attention
    x = x * mx.sigmoid(_conv(x, p, f"{pre}.attn.f.0", pad=1))
    # refine + norm
    x = _conv(x, p, f"{pre}.refine")
    if not layer_norm:
        x = _groupnorm(x, p[f"{pre}.norm.weight"], p[f"{pre}.norm.bias"], groups=4)
    return x + x_skip


def realplksr(x: Any, p: dict, cfg: tuple | None = None) -> Any:
    """Upscale one batch. x: (N,H,W,3) in [0,1] -> (N, scale*H, scale*W, 3)."""
    if cfg is None:
        cfg = _config(p)
    dim, n_blocks, ks, pdim, scale, layer_norm, dysample, groups = cfg
    dt = p["feats.0.weight"].dtype
    x = x.astype(dt)
    y = _conv(x, p, "feats.0", pad=1)
    for i in range(1, n_blocks + 1):
        y = _plk_block(y, p, i, ks, pdim, layer_norm, groups)
    y = _conv(y, p, f"feats.{n_blocks + 2}", pad=1)      # final conv (dropout has no params)
    y = y + mx.repeat(x, scale * scale, axis=-1)         # repeat_interleave residual
    out = _dysample(y, p, scale, groups) if dysample else _pixel_shuffle(y, scale)
    return mx.clip(out, 0.0, 1.0)


_COMPILE_CACHE: dict = {}


def make_forward(p: dict, cfg: tuple | None = None, compile: bool = True):
    """Per-frame forward x -> upscaled image, mx.compiled once per checkpoint."""
    if cfg is None:
        cfg = _config(p)

    def run(x):
        return realplksr(x, p, cfg=cfg)

    if not compile:
        return run
    return _cached(_COMPILE_CACHE, (id(p), cfg), lambda: mx.compile(run))


if __name__ == "__main__":
    p = load_params()
    cfg = _config(p)
    print(f"loaded RealPLKSR: dim={cfg[0]} blocks={cfg[1]} ks={cfg[2]} pdim={cfg[3]} "
          f"scale={cfg[4]}x layer_norm={cfg[5]} dysample={cfg[6]} groups={cfg[7]}")
    mx.random.seed(0)
    x = mx.clip(mx.random.uniform(shape=(1, 48, 64, 3)), 0, 1)
    mx.eval(x)
    out = realplksr(x, p, cfg)
    mx.eval(out)
    print(f"{tuple(x.shape)} -> {tuple(out.shape)}, finite={bool(mx.all(mx.isfinite(out)))}")
