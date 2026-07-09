"""MUSIQ (Multi-scale Image Quality Transformer), MLX port.

Ke, Wang, Wang, Milanfar, Yang -- "MUSIQ: Multi-scale Image Quality
Transformer" (ICCV 2021). Independent MLX reimplementation written from the
paper and the IQA-PyTorch reference read as a spec; no upstream code is
used. See ATTRIBUTION.md. Higher scores = better perceived quality (the
koniq checkpoint reads roughly 0-100).

The model scores a frame from a multi-scale representation: aspect-ratio-
preserving bicubic resizes to longer sides 224 and 384 plus the original
resolution, cut into 32px patches, each encoded by a weight-standardized
conv stem + one bottleneck block, then a 14-layer transformer over the
concatenated patch sequence with hash-based spatial embeddings, per-scale
embeddings, and a CLS head.

Parity-relevant semantics replicated exactly from the reference: TF-SAME
padding everywhere, unbiased-std weight standardization (folded into the
weights at conversion -- they are fixed at inference), torch-nearest hash
position indexing floor(i * G / count), torch bicubic (a = -0.75,
half-pixel centers, border clamp, no antialias), and the additive -1e3
(not -inf) attention mask.

Weights are not bundled (104 MB): convert the released checkpoint with
convert_musiq.py (see weights/README.md).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "musiq_koniq.safetensors"

PATCH = 32
GRID = 10
HIDDEN = 384
HEADS = 6
LAYERS = 14
SCALES = (224, 384)          # + original resolution as the last scale id


# ---- torch-parity bicubic (a = -0.75, half-pixel, border clamp) -------------

def _cubic_weights(t: Any) -> tuple[Any, Any, Any, Any]:
    # cubic convolution (Keys, a = -0.75), fraction t in [0,1):
    # w0/w3 are the negative lobes; the four weights sum to 1 exactly
    a = -0.75
    t2 = t * t
    t3 = t2 * t
    w0 = a * (t3 - 2 * t2 + t)
    w1 = (a + 2) * t3 - (a + 3) * t2 + 1
    w2 = -(a + 2) * t3 + (2 * a + 3) * t2 - a * t
    w3 = a * (t2 - t3)
    return w0, w1, w2, w3


def _resize_axis_cubic(x: Any, out_len: int, axis: int) -> Any:
    in_len = int(x.shape[axis])
    if in_len == out_len:
        return x
    scale = in_len / out_len
    dst = mx.arange(out_len).astype(mx.float32)
    src = (dst + 0.5) * scale - 0.5
    i1 = mx.floor(src)
    t = src - i1
    w0, w1, w2, w3 = _cubic_weights(t)
    idx = [mx.clip(i1 + d, 0, in_len - 1).astype(mx.int32) for d in (-1, 0, 1, 2)]
    xm = mx.moveaxis(x, axis, 0)
    y = (w0.reshape(-1, *([1] * (xm.ndim - 1))) * xm[idx[0]]
         + w1.reshape(-1, *([1] * (xm.ndim - 1))) * xm[idx[1]]
         + w2.reshape(-1, *([1] * (xm.ndim - 1))) * xm[idx[2]]
         + w3.reshape(-1, *([1] * (xm.ndim - 1))) * xm[idx[3]])
    return mx.moveaxis(y, 0, axis)


def resize_bicubic(img: Any, rh: int, rw: int) -> Any:
    """(h,w,c) torch-equivalent bicubic (align_corners=False, no antialias)."""
    return _resize_axis_cubic(_resize_axis_cubic(img, rh, 0), rw, 1)


# ---- preprocessing -----------------------------------------------------------

def _tf_same_pad_hw(x: Any, k: int, s: int, value: float = 0.0) -> Any:
    h, w = int(x.shape[0]), int(x.shape[1])
    pr = max((math.ceil(h / s) - 1) * s + k - h, 0)
    pc = max((math.ceil(w / s) - 1) * s + k - w, 0)
    if not pr and not pc:
        return x
    return mx.pad(x, ((pr // 2, pr - pr // 2), (pc // 2, pc - pc // 2), (0, 0)),
                  constant_values=value)


def _hse_index(count_h: int, count_w: int) -> Any:
    ih = mx.floor(mx.arange(count_h).astype(mx.float32) * GRID / count_h)
    iw = mx.floor(mx.arange(count_w).astype(mx.float32) * GRID / count_w)
    return (ih[:, None] * GRID + iw[None, :]).reshape(-1).astype(mx.int32)


def _patches_of(img: Any, scale_id: int, max_seq: int) -> tuple:
    """img (h,w,3) in [-1,1] -> (patches (P,32,32,3), pos, scale, mask)."""
    x = _tf_same_pad_hw(img, PATCH, PATCH)
    hp, wp = int(x.shape[0]), int(x.shape[1])
    ny, nx = hp // PATCH, wp // PATCH
    p = x.reshape(ny, PATCH, nx, PATCH, 3).transpose(0, 2, 1, 3, 4) \
        .reshape(ny * nx, PATCH, PATCH, 3)
    pos = _hse_index(ny, nx)
    n = ny * nx
    scale = mx.full((n,), scale_id, dtype=mx.int32)
    mask = mx.ones((n,), dtype=mx.float32)
    if 0 <= max_seq:
        if n < max_seq:
            pad = max_seq - n
            p = mx.concatenate([p, mx.zeros((pad, PATCH, PATCH, 3))], axis=0)
            pos = mx.concatenate([pos, mx.zeros((pad,), dtype=mx.int32)])
            scale = mx.concatenate([scale, mx.zeros((pad,), dtype=mx.int32)])
            mask = mx.concatenate([mask, mx.zeros((pad,))])
        else:
            p, pos, scale, mask = p[:max_seq], pos[:max_seq], scale[:max_seq], mask[:max_seq]
    return p, pos, scale, mask


def preprocess(img01: Any) -> tuple:
    """(h,w,3) RGB in [0,1] -> (patches (S,32,32,3), pos, scale, mask)."""
    x = (mx.array(img01).astype(mx.float32) - 0.5) * 2.0
    h, w = int(x.shape[0]), int(x.shape[1])
    parts = []
    for sid, side in enumerate(sorted(SCALES)):
        ratio = side / max(h, w)
        rh, rw = round(h * ratio), round(w * ratio)
        parts.append(_patches_of(resize_bicubic(x, rh, rw), sid,
                                 int(math.ceil(side / PATCH) ** 2)))
    parts.append(_patches_of(x, len(SCALES), -1))
    return tuple(mx.concatenate([p[i] for p in parts], axis=0) for i in range(4))


# ---- model -------------------------------------------------------------------

def _group_norm(x: Any, weight: Any, bias: Any, eps: float) -> Any:
    n, h, w, c = x.shape
    g = 32
    xf = x.astype(mx.float32).reshape(n, h * w, g, c // g)
    mu = mx.mean(xf, axis=(1, 3), keepdims=True)
    var = mx.var(xf, axis=(1, 3), keepdims=True)
    xf = (xf - mu) * mx.rsqrt(var + eps)
    return xf.reshape(n, h, w, c) * weight + bias


def _same_conv(x: Any, w: Any, stride: int) -> Any:
    k = int(w.shape[1])
    n, h, wd, _ = x.shape
    pr = max((math.ceil(h / stride) - 1) * stride + k - h, 0)
    pc = max((math.ceil(wd / stride) - 1) * stride + k - wd, 0)
    if pr or pc:
        x = mx.pad(x, ((0, 0), (pr // 2, pr - pr // 2),
                       (pc // 2, pc - pc // 2), (0, 0)))
    return mx.conv2d(x, w, stride=stride)


def _max_pool_3s2_same(x: Any) -> Any:
    n, h, w, c = x.shape
    pr = max((math.ceil(h / 2) - 1) * 2 + 3 - h, 0)
    pc = max((math.ceil(w / 2) - 1) * 2 + 3 - w, 0)
    x = mx.pad(x, ((0, 0), (pr // 2, pr - pr // 2), (pc // 2, pc - pc // 2),
                   (0, 0)), constant_values=-1e30)
    ho, wo = (int(x.shape[1]) - 1) // 2, (int(x.shape[2]) - 1) // 2
    outs = []
    for dy in range(3):
        for dx in range(3):
            outs.append(x[:, dy:dy + 2 * ho - 1:2, dx:dx + 2 * wo - 1:2, :])
    y = outs[0]
    for o in outs[1:]:
        y = mx.maximum(y, o)
    return y


class Musiq:
    """Batched scorer: frames in [0,1] RGB -> koniq quality scores."""

    def __init__(self, weights: Any = None):
        wp = Path(weights) if weights else WEIGHTS_PATH
        if not wp.is_file():
            raise FileNotFoundError(
                f"MUSIQ weights not found at {wp}. Convert the released "
                "checkpoint with kinovsr/musiq/convert_musiq.py "
                "(see weights/README.md)."
            )
        self.p = {k: v for k, v in mx.load(str(wp)).items()}
        self._compiled: dict = {}

    # stem over all patches of all frames at once: (B*S, 32, 32, 3)
    def _stem(self, patches: Any) -> Any:
        p = self.p
        x = _same_conv(patches, p["conv_root.weight"], 2)
        x = _group_norm(x, p["gn_root.weight"], p["gn_root.bias"], 1e-6)
        x = mx.maximum(x, 0)
        x = _max_pool_3s2_same(x)
        idt = _group_norm(_same_conv(x, p["block1.conv_proj.weight"], 1),
                          p["block1.gn_proj.weight"], p["block1.gn_proj.bias"], 1e-4)
        y = mx.maximum(_group_norm(_same_conv(x, p["block1.conv1.weight"], 1),
                                   p["block1.gn1.weight"], p["block1.gn1.bias"], 1e-4), 0)
        y = mx.maximum(_group_norm(_same_conv(y, p["block1.conv2.weight"], 1),
                                   p["block1.gn2.weight"], p["block1.gn2.bias"], 1e-4), 0)
        y = _group_norm(_same_conv(y, p["block1.conv3.weight"], 1),
                        p["block1.gn3.weight"], p["block1.gn3.bias"], 1e-4)
        return mx.maximum(y + idt, 0)

    def _forward(self, patches: Any, pos: Any, scale: Any, mask: Any) -> Any:
        """patches (B,S,32,32,3), pos/scale (S,), mask (S,) -> (B,) scores."""
        p = self.p
        b, s = int(patches.shape[0]), int(patches.shape[1])
        x = self._stem(patches.reshape(b * s, PATCH, PATCH, 3))
        x = x.reshape(b, s, -1)
        x = x @ p["embedding.weight"].T + p["embedding.bias"]

        te = "transformer_encoder."
        x = x + p[te + "posembed_input.position_emb"][0][pos]
        x = x + p[te + "scaleembed_input.scale_emb"][scale]
        cls = mx.broadcast_to(p[te + "cls"], (b, 1, HIDDEN))
        x = mx.concatenate([cls, x], axis=1)
        m = mx.concatenate([mx.ones((1,)), mask])
        m2 = m[:, None] * m[None, :]
        add_mask = mx.where(m2 == 0, mx.full(m2.shape, -1e3), mx.zeros(m2.shape))
        add_mask = add_mask[None, None]

        def ln(v, w, bias):
            return mx.fast.layer_norm(v, w, bias, 1e-6)

        n = s + 1
        for i in range(LAYERS):
            pf = te + f"transformer.encoderblock_{i}."
            y = ln(x, p[pf + "norm1.weight"], p[pf + "norm1.bias"])
            q = y @ p[pf + "attention.query.weight"].T + p[pf + "attention.query.bias"]
            k = y @ p[pf + "attention.key.weight"].T + p[pf + "attention.key.bias"]
            v = y @ p[pf + "attention.value.weight"].T + p[pf + "attention.value.bias"]
            hd = HIDDEN // HEADS
            q = q.reshape(b, n, HEADS, hd).transpose(0, 2, 1, 3)
            k = k.reshape(b, n, HEADS, hd).transpose(0, 2, 1, 3)
            v = v.reshape(b, n, HEADS, hd).transpose(0, 2, 1, 3)
            attn = mx.softmax(q @ k.transpose(0, 1, 3, 2) * hd ** -0.5 + add_mask,
                              axis=-1)
            y = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, HIDDEN)
            y = y @ p[pf + "attention.out.weight"].T + p[pf + "attention.out.bias"]
            x = x + y
            y = ln(x, p[pf + "norm2.weight"], p[pf + "norm2.bias"])
            y = y @ p[pf + "mlp.fc1.weight"].T + p[pf + "mlp.fc1.bias"]
            y = 0.5 * y * (1 + mx.erf(y / math.sqrt(2.0)))   # exact gelu
            y = y @ p[pf + "mlp.fc2.weight"].T + p[pf + "mlp.fc2.bias"]
            x = x + y
        x = ln(x, p[te + "encoder_norm.weight"], p[te + "encoder_norm.bias"])
        q = x[:, 0] @ p["head.weight"].T + p["head.bias"]
        return q.reshape(-1)

    def score_frames(self, frames: list, batch: int = 8) -> list:
        """Frames (h,w,3) RGB in [0,1] (same resolution) -> list of floats."""
        pre = [preprocess(f) for f in frames[:1]]
        pos, scale, mask = pre[0][1], pre[0][2], pre[0][3]
        out: list = []
        stack = [pre[0][0]] + [preprocess(f)[0] for f in frames[1:]]
        key = (tuple(stack[0].shape), len(stack))
        for i in range(0, len(stack), batch):
            chunk = mx.stack(stack[i:i + batch], axis=0)
            scores = self._forward(chunk, pos, scale, mask)
            mx.eval(scores)
            out += [float(v) for v in scores]
        return out

    def score(self, frame: Any) -> float:
        return self.score_frames([frame])[0]
