"""PVDD (pvdd0815) video denoiser, MLX-native reimplementation.

Architecture from Xu et al., "PVDD: A Practical Video Denoising Dataset with
Real-World Dynamic Scenes" (arXiv 2207.01356), MIT-licensed. One net serves all
six released checkpoints, parameterized by input channels (3 sRGB / 4 packed raw)
and a `level` flag (a noise-variance map fed to the cleaning U-Net).

Forward (per clip of T frames, batch B=1):
  1. ResUNet cleaning (up to 3 residual passes, early-stop on the residual mean);
  2. a strided feature extractor (down x4) feeding a bidirectional recurrent
     backbone whose temporal fusion is 3D shifted-window attention over 2-frame
     windows (self-attention blocks + a cross-attention collapse), plus residual
     conv trunks;
  3. pixel-shuffle x4 reconstruction with an input residual.

NHWC throughout; conv weights are transposed to OHWI at load. Reductions
(LayerNorm, softmax, pools) run in fp32 for fp16 safety.
"""
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

_GELU_C = math.sqrt(2.0 / math.pi)


# ---------------------------------------------------------------- primitives
def _gelu(x: Any) -> Any:
    """tanh-approx GELU, matching the reference (NOT erf)."""
    xf = x.astype(mx.float32)
    y = 0.5 * xf * (1.0 + mx.tanh(_GELU_C * (xf + 0.044715 * xf * xf * xf)))
    return y.astype(x.dtype)


def _lrelu(x: Any, slope: float) -> Any:
    return mx.where(x >= 0, x, x * slope)


def _conv(x: Any, p: dict, key: str, stride: int = 1, pad: int = 1) -> Any:
    y = mx.conv2d(x, p[f"{key}.weight"], stride=stride, padding=pad)
    b = p.get(f"{key}.bias")
    return y if b is None else y + b


def _linear(x: Any, p: dict, key: str) -> Any:
    y = x @ p[f"{key}.weight"].T
    b = p.get(f"{key}.bias")
    return y if b is None else y + b


def _layernorm(x: Any, p: dict, key: str) -> Any:
    """Standard LayerNorm over the last axis (weight + bias), fp32 reduction."""
    xf = x.astype(mx.float32)
    mu = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.mean((xf - mu) ** 2, axis=-1, keepdims=True)
    y = (xf - mu) * mx.rsqrt(var + 1e-5)
    y = y * p[f"{key}.weight"].astype(mx.float32) + p[f"{key}.bias"].astype(mx.float32)
    return y.astype(x.dtype)


def _pixelshuffle(x: Any, r: int = 2) -> Any:
    n, h, w, cr = x.shape
    c = cr // (r * r)
    x = x.reshape(n, h, w, c, r, r)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * r, w * r, c)


def _resblock(x: Any, p: dict, key: str) -> Any:
    """ResidualBlockNoBN: x + conv2(relu(conv1(x)))."""
    y = _conv(x, p, f"{key}.conv1")
    y = mx.maximum(y, 0)
    y = _conv(y, p, f"{key}.conv2")
    return x + y


def _resblocks(x: Any, p: dict, key: str, n: int) -> Any:
    for i in range(n):
        x = _resblock(x, p, f"{key}.{i}")
    return x


# ---------------------------------------------------------------- cleaning U-Net
def _resunet(x: Any, p: dict, pre: str, depth: int, num_block: int) -> Any:
    """ResUNet: conv_in -> depth strided downs -> trunk -> depth pixelshuffle ups
    (skip-concat) -> +conv_in feature -> conv_out. Returns the residual."""
    feat = _conv(x, p, f"{pre}.conv_in")
    feat_in = feat
    skips = [feat]
    for i in range(depth):
        feat = _conv(feat, p, f"{pre}.down_{i}.0", stride=2)
        feat = _lrelu(feat, 0.1)
        skips.append(feat)
    feat = _resblocks(feat, p, f"{pre}.trunk_blocks", num_block)
    for i in range(depth):
        feat = mx.concatenate([skips.pop(), feat], axis=-1)
        feat = _conv(feat, p, f"{pre}.up_{i}.0")
        feat = _pixelshuffle(feat, 2)
        feat = _lrelu(feat, 0.1)
    return _conv(feat + feat_in, p, f"{pre}.conv_out")


# ---------------------------------------------------------------- window attention
def _window_partition(x: Any, ws: tuple[int, int]) -> Any:
    """(B,D,H,W,C) -> (B*nW, D, wh, ww, C)."""
    B, D, H, W, C = x.shape
    x = x.reshape(B, D, H // ws[0], ws[0], W // ws[1], ws[1], C)
    x = mx.transpose(x, (0, 2, 4, 1, 3, 5, 6))
    return x.reshape(-1, D, ws[0], ws[1], C)


def _window_reverse(win: Any, ws: tuple[int, int], B: int, D: int, H: int, W: int) -> Any:
    x = win.reshape(B, H // ws[0], W // ws[1], D, ws[0], ws[1], -1)
    x = mx.transpose(x, (0, 3, 1, 4, 2, 5, 6))
    return x.reshape(B, D, H, W, -1)


def _get_window_size(hw, window, shift=None):
    uw = list(window)
    us = list(shift) if shift is not None else None
    for i in range(len(hw)):
        if hw[i] <= window[i]:
            uw[i] = hw[i]
            if shift is not None:
                us[i] = 0
    return (tuple(uw), tuple(us)) if shift is not None else tuple(uw)


def _rel_pos_bias(p: dict, pre: str, n1: int, n2: int) -> Any:
    """Gather (num_heads, N1, N2) relative-position bias from table + index."""
    table = p[f"{pre}.relative_position_bias_table"].astype(mx.float32)  # (K, nH)
    idx = p[f"{pre}.relative_position_index"].reshape(-1)                 # (N1*N2,)
    b = mx.take(table, idx, axis=0).reshape(n1, n2, -1)                   # (N1,N2,nH)
    return mx.transpose(b, (2, 0, 1))                                     # (nH,N1,N2)


def _attention(q: Any, kv: Any, p: dict, pre: str, num_heads: int, mask: Any | None) -> Any:
    """Window attention with relative-position bias. q:(B_,N1,C) kv:(B_,N2,C).
    Returns proj(attn @ v) + q  (the reference's in-module residual)."""
    q_copy = q
    B_, N1, C = q.shape
    N2 = kv.shape[1]
    hd = C // num_heads
    qh = _linear(q, p, f"{pre}.q").reshape(B_, N1, num_heads, hd).transpose(0, 2, 1, 3)
    kvh = _linear(kv, p, f"{pre}.kv").reshape(B_, N2, 2, num_heads, hd).transpose(2, 0, 3, 1, 4)
    k, v = kvh[0], kvh[1]
    scale = hd ** -0.5
    attn = (qh.astype(mx.float32) * scale) @ k.astype(mx.float32).transpose(0, 1, 3, 2)  # (B_,nH,N1,N2)
    attn = attn + _rel_pos_bias(p, pre, N1, N2)[None]
    if mask is not None:
        nW = mask.shape[0]
        attn = attn.reshape(B_ // nW, nW, num_heads, N1, N2) + mask[None, :, None].astype(mx.float32)
        attn = attn.reshape(B_, num_heads, N1, N2)
    attn = mx.softmax(attn, axis=-1, precise=True)
    x = (attn @ v.astype(mx.float32)).astype(q.dtype)          # (B_,nH,N1,hd)
    x = x.transpose(0, 2, 1, 3).reshape(B_, N1, C)
    return _linear(x, p, f"{pre}.proj") + q_copy


# ---------------------------------------------------------------- Mlp04 (SA/CA/TA)
def _mlp04(x: Any, p: dict, pre: str) -> Any:
    """Attention-augmented FFN. x: (B, D, H, W, C)."""
    B, D, H, W, C = x.shape
    x_in = _gelu(_linear(x, p, f"{pre}.proj_in"))          # (B,D,H,W,hf)
    hf = x_in.shape[-1]
    x2 = x_in.reshape(B * D, H, W, hf)                     # NHWC

    # spatial attention: conv over cat(max,mean) across channels
    mx_c = mx.max(x2, axis=-1, keepdims=True)
    mean_c = mx.mean(x2, axis=-1, keepdims=True)
    sa = _conv(mx.concatenate([mx_c, mean_c], axis=-1), p, f"{pre}.SA.spatial")  # (bn,H,W,1)
    sa = x2 * mx.sigmoid(sa)

    # channel attention: global avg pool -> 1x1 conv_du -> sigmoid gate
    y = mx.mean(x2.astype(mx.float32), axis=(1, 2), keepdims=True).astype(x2.dtype)  # (bn,1,1,hf)
    y = _conv(y, p, f"{pre}.CA.conv_du.0", pad=0)
    y = mx.maximum(y, 0)
    y = _conv(y, p, f"{pre}.CA.conv_du.2", pad=0)
    ca = x2 * mx.sigmoid(y)

    res = mx.concatenate([sa, ca], axis=-1)                # (bn,H,W,2hf)
    res = _conv(res, p, f"{pre}.conv1x1", pad=0) + x2      # (bn,H,W,hf)
    res = res.reshape(B, D, H, W, hf)

    # temporal attention across D frames: 1x1 conv over cat(max,mean) across D
    rf = res.reshape(B, D, H * W, hf)
    mx_t = mx.max(rf, axis=1, keepdims=True)               # (B,1,HW,hf)
    mean_t = mx.mean(rf, axis=1, keepdims=True)
    # TA.spatial is a 1x1 conv (2->1): weighted sum of [max, mean] + bias
    w = p[f"{pre}.TA.spatial.weight"]                      # OHWI (1,1,1,2)
    tb = p.get(f"{pre}.TA.spatial.bias")
    scale = mx_t * w[0, 0, 0, 0] + mean_t * w[0, 0, 0, 1]
    if tb is not None:
        scale = scale + tb
    scale = mx.sigmoid(scale)                              # (B,1,HW,hf)
    res = (rf * scale + rf).reshape(B, D, H, W, hf)
    return _linear(res, p, f"{pre}.proj_out")


# ---------------------------------------------------------------- transformer blocks
def _pad_hw(x: Any, ws: tuple[int, int]):
    _, _, H, W, _ = x.shape
    pb = (ws[0] - H % ws[0]) % ws[0]
    pr = (ws[1] - W % ws[1]) % ws[1]
    if pb or pr:
        x = mx.pad(x, [(0, 0), (0, 0), (0, pb), (0, pr), (0, 0)])
    return x, pb, pr


def _block_self(x: Any, p: dict, pre: str, num_heads: int, ws: tuple[int, int],
                shift: tuple[int, int], mask: Any | None) -> Any:
    """VSTSREncoderTransformerBlock: window self-attention over D frames."""
    B, D, H, W, C = x.shape
    ws, shift = _get_window_size((H, W), ws, shift)
    shortcut = x
    x = _layernorm(x, p, f"{pre}.norm1")
    x, pb, pr = _pad_hw(x, ws)
    _, _, Hp, Wp, _ = x.shape
    if any(shift):
        x = mx.roll(x, shift=(-shift[0], -shift[1]), axis=(2, 3))
        am = mask
    else:
        am = None
    xw = _window_partition(x, ws).reshape(-1, D * ws[0] * ws[1], C)
    aw = _attention(xw, xw, p, f"{pre}.attn", num_heads, am)
    aw = aw.reshape(-1, D, ws[0], ws[1], C)
    x = _window_reverse(aw, ws, B, D, Hp, Wp)
    if any(shift):
        x = mx.roll(x, shift=(shift[0], shift[1]), axis=(2, 3))
    if pb or pr:
        x = x[:, :, :H, :W, :]
    x = shortcut + x
    return x + _mlp04(_layernorm(x, p, f"{pre}.norm2"), p, f"{pre}.mlp")


def _block_cross(x: Any, p: dict, pre: str, num_heads: int, ws: tuple[int, int]) -> Any:
    """VSTSREncoderTransformerBlock02: cross-attention frame0<-frame1, D=2 -> 1.
    Assumes B=1 (matches the reference's broadcast)."""
    B, D, H, W, C = x.shape
    ws = _get_window_size((H, W), ws)
    shortcut = x[:, 0]                                    # (B,H,W,C)
    x = _layernorm(x, p, f"{pre}.norm1")
    x, pb, pr = _pad_hw(x, ws)
    _, _, Hp, Wp, _ = x.shape
    xw = _window_partition(x, ws)                         # (nWB,2,wh,ww,C)
    q = xw[:, 0].reshape(-1, ws[0] * ws[1], C)
    kv = xw[:, 1].reshape(-1, ws[0] * ws[1], C)
    aw = _attention(q, kv, p, f"{pre}.attn", num_heads, None)
    aw = aw.reshape(-1, 1, ws[0], ws[1], C)
    x = _window_reverse(aw, ws, B, 1, Hp, Wp)            # (B,1,Hp,Wp,C)
    if pb or pr:
        x = x[:, :, :H, :W, :]
    x = shortcut[:, None] + x                            # (B,1,H,W,C), B=1
    x = x + _mlp04(_layernorm(x, p, f"{pre}.norm2"), p, f"{pre}.mlp")
    return x[:, 0]                                       # (B,H,W,C)


def _region_ids(n: int, ws: int, shift: int) -> Any:
    """Swin region map along one axis: 0 on [0,n-ws), 1 on [n-ws,n-shift), 2 on rest."""
    idx = mx.arange(n)
    return mx.where(idx < n - ws, 0, mx.where(idx < n - shift, 1, 2)).astype(mx.float32)


def _shift_mask(D: int, H: int, W: int, ws: tuple[int, int], shift: tuple[int, int]) -> Any:
    Hp = int(math.ceil(H / ws[0])) * ws[0]
    Wp = int(math.ceil(W / ws[1])) * ws[1]
    hr = _region_ids(Hp, ws[0], shift[0])                       # (Hp,)
    wr = _region_ids(Wp, ws[1], shift[1])                       # (Wp,)
    region = hr[:, None] * 3.0 + wr[None, :]                    # (Hp,Wp)
    img = mx.broadcast_to(region[None, None, :, :, None], (1, D, Hp, Wp, 1))
    mw = _window_partition(img, ws).reshape(-1, D * ws[0] * ws[1])   # (nW, D*wh*ww)
    am = mw[:, None, :] - mw[:, :, None]
    return mx.where(am != 0, mx.array(-100.0, mx.float32), mx.array(0.0, mx.float32))


def _encoder_layer02(x: Any, p: dict, pre: str, depth: int, num_heads: int,
                     ws: tuple[int, int]) -> Any:
    """EncoderLayer02: (B,D,C,H,W) -> (B,C,H,W). depth self-blocks + cross last_blk."""
    B, D, C, H, W = x.shape
    x = mx.transpose(x, (0, 1, 3, 4, 2))                 # (B,D,H,W,C)
    shift = tuple(i // 2 for i in ws)
    ws_a, shift_a = _get_window_size((H, W), ws, shift)
    mask = _shift_mask(D, H, W, ws_a, shift_a) if any(shift_a) else None
    for i in range(depth):
        sh = (0, 0) if i % 2 == 0 else shift
        x = _block_self(x, p, f"{pre}.blocks.{i}", num_heads, ws, sh, mask)
    x = _block_cross(x, p, f"{pre}.last_blk", num_heads, ws)   # (B,H,W,C)
    return mx.transpose(x, (0, 3, 1, 2))                 # (B,C,H,W) logical -> NHWC handled by caller


# ---------------------------------------------------------------- full net
class PVDDConfig:
    def __init__(self, num_in=3, level=False, num_feat=64, num_block=3, num_block_f=3,
                 num_block_pre=3, depth=2, num_head=8, window_size=(8, 8),
                 dynamic_refine_thres=255.0):
        self.num_in = num_in
        self.level = level
        self.num_feat = num_feat
        self.num_block = num_block
        self.num_block_f = num_block_f
        self.num_block_pre = num_block_pre
        self.depth = depth
        self.num_head = num_head
        self.window_size = window_size
        self.thres = dynamic_refine_thres / 255.0


def _feat_extractor(x: Any, p: dict, cfg: PVDDConfig) -> Any:
    x = _lrelu(_conv(x, p, "feat_extractor.main.0", stride=2), 0.1)
    x = _lrelu(_conv(x, p, "feat_extractor.main.2", stride=2), 0.1)
    return _resblocks(x, p, "feat_extractor.main.4", cfg.num_block_f)


def _trunk(x: Any, p: dict, pre: str, cfg: PVDDConfig) -> Any:
    x = _lrelu(_conv(x, p, f"{pre}.main.0"), 0.2)
    return _resblocks(x, p, f"{pre}.main.2", cfg.num_block)


def _process(frames_nhwc: list, p: dict, cfg: PVDDConfig) -> list:
    """frames: list of T (1,H,W,C). Returns list of T denoised (1,H,W,C)."""
    T = len(frames_nhwc)
    feats = [_feat_extractor(f, p, cfg) for f in frames_nhwc]   # each (1,h/4,w/4,64)

    def sttb(pre, fa, fb):
        # stack two (1,h,w,C) frames -> (1,2,C,h,w) logical for encoder_layer02
        s = mx.stack([fa, fb], axis=1)                # (1,2,h,w,C)
        s = mx.transpose(s, (0, 1, 4, 2, 3))          # (1,2,C,h,w)
        out = _encoder_layer02(s, p, pre, cfg.depth, cfg.num_head, cfg.window_size)  # (1,C,h,w)
        return mx.transpose(out, (0, 2, 3, 1))        # (1,h,w,C)

    out_l = [None] * T
    feat_prop = feats[-1]
    for i in range(T - 1, -1, -1):
        feat_prop = sttb("backward_STTB", feats[i], feat_prop)
        feat_prop = _trunk(feat_prop, p, "backward_trunk", cfg)
        out_l[i] = feat_prop

    feat_prop = feats[0]
    outs = []
    for i in range(T):
        feat_prop = sttb("forward_STTB", feats[i], feat_prop)
        feat_prop = _trunk(feat_prop, p, "forward_trunk", cfg)
        out = mx.concatenate([out_l[i], feat_prop], axis=-1)    # (1,h,w,128)
        out = _lrelu(_conv(out, p, "fusion", pad=0), 0.1)
        out = _lrelu(_pixelshuffle(_conv(out, p, "upconv1"), 2), 0.1)
        out = _lrelu(_pixelshuffle(_conv(out, p, "upconv2"), 2), 0.1)
        out = _lrelu(_conv(out, p, "conv_hr"), 0.1)
        out = _conv(out, p, "conv_last") + frames_nhwc[i]
        outs.append(out)
    return outs


def pvdd_forward(frames_nhwc: list, p: dict, cfg: PVDDConfig,
                 noise_map: Any | None = None) -> list:
    """frames: list of T (1,H,W,C) in [0,1]. noise_map: (1,H,W,1) variance plane,
    or a list of T such planes for per-frame conditioning (GOP pulse gain)."""
    x = list(frames_nhwc)
    per_frame = isinstance(noise_map, (list, tuple))
    for _ in range(3):
        cleaned = []
        total = 0.0
        for i, f in enumerate(x):
            if noise_map is None:
                inp = f
            else:
                nm = noise_map[i] if per_frame else noise_map
                inp = mx.concatenate([f, nm], axis=-1)
            res = _resunet(inp, p, "clean_model", 2, cfg.num_block_pre)
            cleaned.append(f + res)
            total = total + mx.mean(mx.abs(res))
        x = cleaned
        if float((total / len(x)).item()) < cfg.thres:
            break
    return _process(x, p, cfg)
