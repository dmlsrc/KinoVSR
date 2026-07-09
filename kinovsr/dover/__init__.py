"""DOVER-Mobile video-quality scorer (MLX port).

Disentangled Objective Video Quality Evaluator (Wu et al., ICCV 2023),
mobile variant: two ConvNeXt-V2-3D femto backbones score a *technical*
view (7x7 mosaic of 32px fragments -- sensitive to blocking, noise,
flicker) and an *aesthetic* view (whole frames resized to 224), fused
into one 0-1 score.  Trained on human opinions of user-generated video,
so unlike per-frame photo metrics it sees temporal artifacts: the 32
frames of each clip pass through the network together and the GRN units
normalize across time.

Port notes (see the weights README for conversion):

- The reference applies its (2D-derived) GRN to 5D activations, so the
  L2 statistic runs over (T, H) only -- not (T, H, W).  The checkpoint
  was trained with that behavior, so this port replicates it exactly.
- MLX has no grouped conv3d, so the depthwise (Lt, 7, 7) convs run as
  a temporal shift-and-sum of per-frame grouped ``mx.conv2d`` calls
  (Lt alternates 1 and 3 across blocks in the checkpoint; grouped 2D
  conv measured faster than spatial shift-and-add at all four stage
  shapes on M1 Max).
- The official test-time fragment layout draws random in-cell offsets
  (``torch.randint``) even at inference.  This port uses the centered
  offset instead, which is deterministic; on the fixtures used to
  validate the port the deviation sits inside the reference's own
  run-to-run spread (~+-0.001 fused).
- Sub-224p inputs are upscaled with size-derived (in/out) coordinate
  mapping rather than torch's scale_factor mapping; the two differ only
  in the third decimal of sampling positions.

Scores: higher is better.  ``fused`` is sigmoid-squashed to (0, 1);
``tech``/``aes`` are the raw branch outputs (typically -0.3 .. 0.3).
"""
from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx

_DEPTHS = (2, 2, 6, 2)
_MEAN = (123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375)
# Score-level fusion constants from the reference's evaluate_one_video.py:
# x = (tech - 0.1107)/0.07355*0.6104 + (aes + 0.08285)/0.03774*0.3896.
_FUSE = (0.1107, 0.07355, 0.6104, 0.08285, 0.03774, 0.3896)

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "dover_mobile.safetensors"


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))


def _ln(x: mx.array, w: mx.array, b: mx.array, eps: float = 1e-6) -> mx.array:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * mx.rsqrt(var + eps) * w + b


def _clip_offsets(n_frames: int, clip_len: int, n_clips: int, interval: int) -> list:
    ori = clip_len * interval
    if n_frames > ori - 1:
        avg = (n_frames - ori + 1) / n_clips
        return [int(i * avg + avg / 2.0) for i in range(n_clips)]
    return [0] * n_clips


def _sample_inds(n_frames: int, clip_len: int, n_clips: int, interval: int) -> list:
    offs = _clip_offsets(n_frames, clip_len, n_clips, interval)
    return [[(o + k * interval) % n_frames for k in range(clip_len)] for o in offs]


def _bilinear_matrix(n_in: int, n_out: int, antialias: bool) -> mx.array:
    """Row-stochastic (n_out, n_in) resize matrix, matching torch's
    separable bilinear resampler (triangle filter widened by the scale
    when antialiased downscaling; plain 2-tap otherwise)."""
    scale = n_in / n_out
    aa = antialias and scale >= 1.0
    support = scale if aa else 1.0
    invscale = 1.0 / scale if aa else 1.0
    rows = []
    for i in range(n_out):
        center = scale * (i + 0.5)
        jmin = max(int(center - support + 0.5), 0)
        jmax = min(int(center + support + 0.5), n_in)
        ws = [max(0.0, 1.0 - abs((j - center + 0.5) * invscale))
              for j in range(jmin, jmax)]
        tot = sum(ws)
        row = [0.0] * n_in
        for k, w in enumerate(ws):
            row[jmin + k] = w / tot
        rows.append(row)
    return mx.array(rows)


class DoverMobile:
    """Scores videos with DOVER-Mobile.  ``score(frames)`` takes a
    (T, H, W, 3) RGB array in 0-255 (uint8 or float) and returns
    ``{"tech": float, "aes": float, "fused": float}``."""

    def __init__(self, weights: Path | str | None = None) -> None:
        path = Path(weights) if weights is not None else WEIGHTS_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"DOVER-Mobile weights not found at {path}; see "
                f"{WEIGHTS_PATH.parent / 'README.md'} for download "
                "and conversion instructions.")
        self._p = mx.load(str(path))
        self._resize_mats: dict = {}
        self._tech = mx.compile(lambda x: self._branch(x, "technical"))
        self._aes = mx.compile(lambda x: self._branch(x, "aesthetic"))

    # -- network ------------------------------------------------------

    def _block(self, x: mx.array, pfx: str) -> mx.array:
        p = self._p
        n, t, h, w, c = x.shape
        wt = p[pfx + "dwconv.weight"]                    # (Lt,C,7,7,1)
        lt = wt.shape[0]
        if lt == 1:
            y = mx.conv2d(x.reshape(n * t, h, w, c), wt[0],
                          padding=3, groups=c)
        else:
            xp = mx.pad(x, ((0, 0), (lt // 2, lt // 2), (0, 0),
                            (0, 0), (0, 0)))
            y = None
            for layer_idx in range(lt):
                yl = mx.conv2d(
                    xp[:, layer_idx:layer_idx + t].reshape(n * t, h, w, c),
                    wt[layer_idx],
                    padding=3,
                    groups=c,
                )
                y = yl if y is None else y + yl
        y = y.reshape(n, t, h, w, c) + p[pfx + "dwconv.bias"]
        y = _ln(y, p[pfx + "norm.weight"], p[pfx + "norm.bias"])
        y = y @ p[pfx + "pwconv1.weight"] + p[pfx + "pwconv1.bias"]
        y = _gelu(y)
        # Reference GRN on 5D input: L2 over (T, H) only (see module doc).
        gx = mx.sqrt((y * y).sum(axis=(1, 2), keepdims=True))
        nx = gx / (gx.mean(axis=-1, keepdims=True) + 1e-6)
        y = p[pfx + "grn.gamma"] * (y * nx) + p[pfx + "grn.beta"] + y
        y = y @ p[pfx + "pwconv2.weight"] + p[pfx + "pwconv2.bias"]
        return x + y

    def _branch(self, x: mx.array, name: str) -> mx.array:
        p = self._p
        bb = name + "_backbone."
        for i in range(4):
            d = f"{bb}downsample_layers.{i}."
            if i == 0:
                x = mx.conv3d(x, p[d + "0.weight"], stride=(2, 4, 4))
                x = x + p[d + "0.bias"]
                x = _ln(x, p[d + "1.weight"], p[d + "1.bias"])
            else:
                x = _ln(x, p[d + "0.weight"], p[d + "0.bias"])
                x = mx.conv3d(x, p[d + "1.weight"], stride=(1, 2, 2))
                x = x + p[d + "1.bias"]
            for j in range(_DEPTHS[i]):
                x = self._block(x, f"{bb}stages.{i}.{j}.")
        x = _ln(x, p[bb + "norm.weight"], p[bb + "norm.bias"])
        hd = name + "_head."
        x = _gelu(x @ p[hd + "fc_hid.weight"] + p[hd + "fc_hid.bias"])
        x = x @ p[hd + "fc_last.weight"] + p[hd + "fc_last.bias"]
        return x.mean()

    # -- views --------------------------------------------------------

    def _resize(self, clip: mx.array, oh: int, ow: int,
                antialias: bool) -> mx.array:
        t, h, w, c = clip.shape
        kh, kw = (h, oh, antialias), (w, ow, antialias)
        for key in (kh, kw):
            if key not in self._resize_mats:
                self._resize_mats[key] = _bilinear_matrix(*key)
        x = mx.moveaxis(clip, 1, 3) @ self._resize_mats[kh].T   # (T,W,C,oh)
        x = mx.moveaxis(x, 3, 1)                                # (T,oh,W,C)
        x = mx.moveaxis(x, 2, 3) @ self._resize_mats[kw].T      # (T,oh,C,ow)
        return mx.moveaxis(x, 3, 2)                             # (T,oh,ow,C)

    def _fragments(self, clip: mx.array) -> mx.array:
        """(32, H, W, 3) 0-255 -> (32, 224, 224, 3) fp32 7x7 mosaic of
        32px crops, one centered crop per grid cell."""
        t, h, w, _ = clip.shape
        ratio = min(h / 224, w / 224)
        if ratio < 1:
            clip = self._resize(clip.astype(mx.float32),
                                int(h / ratio), int(w / ratio), False)
            t, h, w, _ = clip.shape
        hg = [min(h // 7 * i, h - 32) for i in range(7)]
        wg = [min(w // 7 * i, w - 32) for i in range(7)]
        ho = max((h // 7 - 32) // 2, 0)
        wo = max((w // 7 - 32) // 2, 0)
        rows = []
        for i in range(7):
            row = [clip[:, hg[i] + ho:hg[i] + ho + 32,
                        wg[j] + wo:wg[j] + wo + 32, :] for j in range(7)]
            rows.append(mx.concatenate(row, axis=2))
        return mx.concatenate(rows, axis=1).astype(mx.float32)

    # -- scoring ------------------------------------------------------

    def score(self, frames: mx.array) -> dict:
        n = frames.shape[0]
        mean = mx.array(_MEAN)
        std = mx.array(_STD)

        tech_clips = []
        for inds in _sample_inds(n, 32, 3, 2):
            clip = mx.take(frames, mx.array(inds), axis=0)
            tech_clips.append(self._fragments(clip))
        tech = (mx.stack(tech_clips) - mean) / std              # (3,32,224,224,3)

        ainds = [row[0] for row in _sample_inds(n, 1, 32, 2)]
        aclip = mx.take(frames, mx.array(ainds), axis=0).astype(mx.float32)
        aclip = self._resize(aclip, 224, 224, True)
        aes = ((aclip - mean) / std)[None]                      # (1,32,224,224,3)

        ts, as_ = self._tech(tech), self._aes(aes)
        mx.eval(ts, as_)
        ts, as_ = float(ts), float(as_)
        m0, s0, w0, m1, s1, w1 = _FUSE
        x = (ts - m0) / s0 * w0 + (as_ + m1) / s1 * w1
        return {"tech": ts, "aes": as_, "fused": 1.0 / (1.0 + math.exp(-x))}
