"""Spatial noise-map estimation for map-conditioned (FFDNet-style) denoisers.

Estimates a per-pixel noise **sigma map** (units: [0,1] luma sigma, the same scale
FastDVDnet and BSVD take as their 4th input channel; PVDD's level checkpoints take
noise **variance**, so square this map for them) from a short window of frames.

Method: temporal, not spatial. Consecutive-frame differences on luma cancel static
content (including fine texture, which spatial estimators misread as noise) and
leave motion plus temporally-varying noise. Per coarse block, a low quantile of
the pooled |diff| samples is scaled to sigma via the half-normal quantile factor
-- the low quantile makes the estimate robust to a minority of motion-contaminated
samples. Blocks that are motion-dominated anyway (all samples inflated) are capped
by a signal-dependence model (sigma as a function of block luma, fitted from the
quiet blocks), which also encodes the shadows-are-noisier structure of real
footage. The coarse map is smoothed and bilinearly-ish upsampled so the
conditioning signal stays smooth (map-trained nets were trained on smooth maps).

Because the estimate is differential it measures the *temporally fluctuating*
noise component -- exactly what temporal denoisers can remove. Static grain and
fixed-pattern noise do not register (they also cannot be removed temporally).

Public API:
  estimate_sigma_map(frames) -> (H,W,1) fp32 sigma map, or None if < 2 frames
  NoiseMapTracker            -> stateful wrapper: gain + EMA across windows
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

# Half-normal quantile factor: for d ~ N(0, sigma), quantile_p(|d|) = sigma *
# sqrt(2) * erfinv(p). At p = 0.25: sqrt(2) * erfinv(0.25) = 0.3186394.
_HALF_NORMAL_Q25 = 0.3186393639643751
# Rec.601 luma; the map conditions nets on broadcast-ish content.
_LUMA = (0.299, 0.587, 0.114)
# Map-conditioned nets are trained with per-CHANNEL AWGN sigma in the map plane,
# but the luma mix attenuates iid channel noise by sqrt(sum(c^2)) = 0.6686; scale
# the luma-domain estimate back to per-channel units.
_CHANNEL_FROM_LUMA = 1.0 / (0.299 ** 2 + 0.587 ** 2 + 0.114 ** 2) ** 0.5


def _to_luma_2d(frame: Any) -> Any:
    """(H,W,C) or (1,H,W,C), [0,1] -> (H,W) fp32 luma."""
    f = frame[0] if frame.ndim == 4 else frame
    f = f.astype(mx.float32)
    if f.shape[-1] == 1:
        return f[..., 0]
    return _LUMA[0] * f[..., 0] + _LUMA[1] * f[..., 1] + _LUMA[2] * f[..., 2]


def _edge_pad_hw(x: Any, ph: int, pw: int) -> Any:
    """Edge-replicate pad (H,W) or (K,H,W) on the bottom/right."""
    if ph:
        x = mx.concatenate([x, mx.broadcast_to(x[..., -1:, :], x.shape[:-2] + (ph, x.shape[-1]))], axis=-2)
    if pw:
        x = mx.concatenate([x, mx.broadcast_to(x[..., :, -1:], x.shape[:-1] + (pw,))], axis=-1)
    return x


def _box_smooth_coarse(g: Any) -> Any:
    """3x3 edge-padded box mean on a coarse (bh,bw) grid, applied twice."""
    bh, bw = g.shape
    for _ in range(2):
        p = mx.concatenate([g[:1], g, g[-1:]], axis=0)          # (bh+2, bw)
        p = mx.concatenate([p[:, :1], p, p[:, -1:]], axis=1)    # (bh+2, bw+2)
        acc = mx.zeros_like(g)
        for i in range(3):
            for j in range(3):
                acc = acc + p[i:i + bh, j:j + bw]
        g = acc / 9.0
    return g


def _box_blur_full(x: Any, k: int) -> Any:
    """Separable edge-padded box blur of odd width k on an (H,W) map."""
    h, w = x.shape
    r = k // 2
    x4 = x[None, :, :, None]
    x4 = mx.concatenate([mx.broadcast_to(x4[:, :1], (1, r, w, 1)), x4,
                         mx.broadcast_to(x4[:, -1:], (1, r, w, 1))], axis=1)
    x4 = mx.conv2d(x4, mx.full((1, k, 1, 1), 1.0 / k), stride=1, padding=0)
    x4 = mx.concatenate([mx.broadcast_to(x4[:, :, :1], (1, h, r, 1)), x4,
                         mx.broadcast_to(x4[:, :, -1:], (1, h, r, 1))], axis=2)
    x4 = mx.conv2d(x4, mx.full((1, 1, k, 1), 1.0 / k), stride=1, padding=0)
    return x4[0, :, :, 0]


def _select_runs(n: int, max_frames: int, run_len: int = 3) -> list[list[int]]:
    """Pick frame indices for diffing under a max_frames budget.

    Diffs must be between temporally ADJACENT frames (distant diffs measure
    motion, not noise), so when the window exceeds the budget we take short
    runs of `run_len` consecutive frames spread evenly across the window
    instead of just its head -- the estimate then reflects the whole window.
    """
    if n <= max_frames:
        return [list(range(n))]
    n_runs = max(2, max_frames // run_len)
    span = n - run_len
    starts = sorted({round(i * span / (n_runs - 1)) for i in range(n_runs)})
    return [list(range(s, s + run_len)) for s in starts]


def estimate_sigma_map(
    frames: list,
    block: int = 16,
    max_frames: int = 12,
    luma_cap_headroom: float = 2.0,
    luma_bins: int = 8,
    sigma_floor: float = 0.002,
    sigma_ceil: float = 0.25,
    smooth: bool = True,
) -> Any | None:
    """Estimate a per-pixel noise sigma map from a window of frames.

    frames: list of (H,W,C) or (1,H,W,C) arrays in [0,1] (any float dtype).
    Returns an (H,W,1) fp32 sigma map in [0,1] luma units, or None when fewer
    than 2 frames are given. Square it for variance-conditioned nets (PVDD).
    Windows longer than max_frames are sampled as spread runs of consecutive
    frames, so the estimate covers the whole window, not just its head.
    """
    if len(frames) < 2:
        return None
    runs = _select_runs(len(frames), max_frames)
    lum_all: list = []
    diffs: list = []
    for run in runs:
        lum = [_to_luma_2d(frames[i]) for i in run]
        lum_all.extend(lum)
        yr = mx.stack(lum, axis=0)
        diffs.append(mx.abs(yr[1:] - yr[:-1]) * (1.0 / 1.4142135623730951))
    y = mx.stack(lum_all, axis=0)                               # sampled lumas
    d = mx.concatenate(diffs, axis=0)                           # (K,H,W) adjacent diffs
    H, W = int(y.shape[1]), int(y.shape[2])
    K = d.shape[0]

    # block-pool |d| over space and time; low quantile -> sigma per block
    ph, pw = (-H) % block, (-W) % block
    d = _edge_pad_hw(d, ph, pw)
    ylum = _edge_pad_hw(mx.mean(y, axis=0), ph, pw)
    hp, wp = H + ph, W + pw
    bh, bw = hp // block, wp // block
    db = d.reshape(K, bh, block, bw, block)
    db = mx.transpose(db, (1, 3, 0, 2, 4)).reshape(bh, bw, K * block * block)
    n = db.shape[-1]
    q_idx = int(round(0.25 * (n - 1)))
    q = mx.sort(db, axis=-1)[..., q_idx]                        # (bh,bw)
    sig = q * (_CHANNEL_FROM_LUMA / _HALF_NORMAL_Q25)
    blum = mx.mean(ylum.reshape(bh, block, bw, block), axis=(1, 3))   # (bh,bw)

    # signal-dependence model: robust sigma-vs-luma from the quiet blocks, used to
    # cap motion-dominated blocks (their pooled quantile is inflated everywhere).
    sig_l = [float(v) for v in sig.reshape(-1).tolist()]
    lum_l = [float(v) for v in blum.reshape(-1).tolist()]
    def _p30(vals: list) -> float:
        s = sorted(vals)
        return s[int(0.30 * (len(s) - 1))]
    global_p30 = _p30(sig_l)
    model = [global_p30] * luma_bins
    for b in range(luma_bins):
        lo, hi = b / luma_bins, (b + 1) / luma_bins
        vals = [s for s, l in zip(sig_l, lum_l) if lo <= l < hi]
        if len(vals) >= 8:
            model[b] = _p30(vals)
    cap = [luma_cap_headroom * model[min(luma_bins - 1, max(0, int(l * luma_bins)))]
           for l in lum_l]
    sig = mx.minimum(sig, mx.array(cap, dtype=mx.float32).reshape(bh, bw))

    if smooth:
        sig = _box_smooth_coarse(sig)
    full = mx.repeat(mx.repeat(sig, block, axis=0), block, axis=1)[:H, :W]
    if smooth:
        full = _box_blur_full(full, block + 1)
    full = mx.clip(full, sigma_floor, sigma_ceil)
    return full[:, :, None].astype(mx.float32)


class NoiseMapTracker:
    """Stateful estimator: applies a gain and EMA-blends successive estimates so
    per-window maps do not pump. update(frames) returns the current (H,W,1) map
    (or None until enough frames have been seen); current() reads without update.
    """

    def __init__(self, gain: float = 1.0, ema: float = 0.5, min_frames: int = 8,
                 estimator: Any = None, **est_kwargs):
        if gain <= 0:
            raise ValueError(f"noise-map gain must be > 0; got {gain}")
        if not (0.0 < ema <= 1.0):
            raise ValueError(f"noise-map ema must be in (0, 1]; got {ema}")
        self.gain = float(gain)
        self.ema = float(ema)
        # windows shorter than this give high-variance estimates (a 6-frame
        # gop-align tail can read near zero); once a map exists, such windows
        # reuse it instead of updating. (For purely spatial estimators like
        # blockiness, pass min_frames=1.)
        self.min_frames = max(1, int(min_frames))
        # the map producer; defaults to the noise-sigma estimator. Pass
        # estimate_blockiness_map to track a deblocker mask instead.
        self.estimator = estimator or estimate_sigma_map
        self.est_kwargs = est_kwargs
        self._map: Any | None = None    # pre-gain EMA state

    def reset(self) -> None:
        self._map = None

    def update(self, frames: list) -> Any | None:
        if self._map is not None and len(frames) < self.min_frames:
            return self.current()
        est = self.estimator(frames, **self.est_kwargs)
        if est is None:
            return self.current()
        if self._map is None or self._map.shape != est.shape:
            self._map = est
        else:
            self._map = self.ema * est + (1.0 - self.ema) * self._map
        return self.current()

    def current(self) -> Any | None:
        if self._map is None:
            return None
        return self._map * self.gain


def _grid_phase(g: Any, period: int = 8) -> tuple[int, bool]:
    """Detect the coding-grid phase along an axis from gradient energy.

    g: (H, Wg) absolute gradients along the axis (Wg = W-1 for columns). Returns
    (phase, found). A real coding grid elevates exactly ONE phase; periodic
    texture whose period divides the grid (e.g. a 2px checker) elevates several
    phases equally and aliases into any phase test, so the grid counts as found
    only when the winner's margin over the runner-up is decisive (unimodal).
    """
    n = int(g.shape[1]) // period * period
    if n < period:
        return 0, False
    m = mx.mean(g[:, :n].reshape(g.shape[0], n // period, period), axis=(0, 1))  # (period,)
    vals = [float(v) for v in m.tolist()]
    order = sorted(range(period), key=lambda i: -vals[i])
    top1, top2 = vals[order[0]], vals[order[1]]
    found = top1 > 1e-6 and (top1 - top2) > 0.10 * top1
    return order[0], found


def estimate_blockiness_map(
    frames: list,
    period: int = 8,
    tile: int = 32,
    max_frames: int = 6,
    severity_scale: float = 0.03,
    smooth: bool = True,
) -> Any | None:
    """Estimate a per-pixel blockiness mask in [0, 1] from a window of frames.

    Blocking is a purely spatial artifact, so unlike the noise-sigma estimator
    this needs no adjacent frames -- it samples up to max_frames spread across
    the window. Per axis, the coding-grid phase is detected globally from
    gradient energy (period-8 grids; period-16 boundaries are a subset), then
    each tile scores the mean gradient ON grid boundaries minus the mean OFF
    them -- content texture raises both and cancels, so only grid-aligned
    discontinuities register. The two axes combine by geometric mean, so a real
    vertical or horizontal content edge (one axis only) scores zero; DCT
    blocking (always a 2D grid) survives. `severity_scale` is the boundary
    excess (luma units) mapped to mask 1.0.

    Returns (H,W,1) fp32 in [0,1], or None for an empty frame list.
    """
    if not frames:
        return None
    n = len(frames)
    take = list(range(n)) if n <= max_frames else \
        sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)})
    lum = [_to_luma_2d(frames[i]) for i in take]
    H, W = int(lum[0].shape[0]), int(lum[0].shape[1])
    y = mx.stack(lum, axis=0)                                   # (K,H,W)

    gv = mx.mean(mx.abs(y[:, :, 1:] - y[:, :, :-1]), axis=0)    # (H, W-1) col grads
    gh = mx.mean(mx.abs(y[:, 1:, :] - y[:, :-1, :]), axis=0)    # (H-1, W) row grads
    px, fx = _grid_phase(gv, period)
    py, fy = _grid_phase(mx.transpose(gh, (1, 0)), period)
    if not (fx and fy):
        # no decisive 2D coding grid anywhere -> nothing to deblock
        return mx.zeros((H, W, 1), dtype=mx.float32)

    def _tile_excess(g: Any, phase: int, transpose: bool) -> Any:
        """Per-tile min-over-boundary-lines minus interior mean, >= 0.

        Blocking elevates EVERY grid line crossing a tile; a content edge that
        happens to sit on the grid elevates one. Taking the minimum across the
        tile's boundary lines keeps the former and rejects the latter.
        """
        if transpose:
            g = mx.transpose(g, (1, 0))
        h, w = int(g.shape[0]), int(g.shape[1])
        ph, pw = (-h) % tile, (-w) % tile
        g = _edge_pad_hw(g, ph, pw)
        hp, wp = h + ph, w + pw
        gt = g.reshape(hp // tile, tile, wp // tile, tile)      # (ht,tr,wt,tc)
        lines = [gt[:, :, :, phase + k * period]
                 for k in range(max(1, (tile - phase) // period))
                 if phase + k * period < tile]
        line_means = mx.stack([mx.mean(l, axis=1) for l in lines], axis=0)  # (L,ht,wt)
        bmin = mx.min(line_means, axis=0)                       # (ht,wt)
        idx = mx.arange(tile)
        imask = ((idx % period) != phase).astype(mx.float32).reshape(1, 1, 1, tile)
        isum = mx.sum(gt * imask, axis=(1, 3))
        icnt = mx.sum(mx.broadcast_to(imask, gt.shape), axis=(1, 3))
        ex = mx.maximum(bmin - isum / mx.maximum(icnt, 1.0), 0.0)
        return mx.transpose(ex, (1, 0)) if transpose else ex

    bv = _tile_excess(gv, px, transpose=False)                  # (ht, wt-ish)
    bh = _tile_excess(gh, py, transpose=True)
    # tile grids can differ by one when (W-1) vs W pad differently; crop to match
    th = min(int(bv.shape[0]), int(bh.shape[0]))
    tw = min(int(bv.shape[1]), int(bh.shape[1]))
    b = mx.sqrt(bv[:th, :tw] * bh[:th, :tw])                    # 2D-grid evidence only
    if smooth:
        b = _box_smooth_coarse(b)
    full = mx.repeat(mx.repeat(b, tile, axis=0), tile, axis=1)[:H, :W]
    if smooth:
        full = _box_blur_full(full, tile + 1)
    mask = mx.clip(full / severity_scale, 0.0, 1.0)
    return mask[:, :, None].astype(mx.float32)


class PulseGain:
    """Per-frame noise-pulse gain for GOP-phase noise (I-frame grain refresh).

    Old encoders re-code the grain at every I-frame, so temporal noise pulses:
    elevated for the first frames after a keyframe, suppressed once P/B
    prediction settles. A static (per-window) map cannot express that, so this
    tracks a per-frame GLOBAL sigma (same robust low-quantile statistic as the
    map, one adjacent diff per frame) and returns its ratio to the running
    settled level -- the median of recent frames. Multiply the conditioning
    plane by the gain per frame (sigma planes by gain; variance planes by
    gain^2). Clamped to [lo, hi]; neutral 1.0 until enough history exists or at
    segment starts (first frame of a stream/window, where no adjacent diff is
    available).
    """

    def __init__(self, lo: float = 0.6, hi: float = 1.8, history: int = 48,
                 min_history: int = 8, sigma_floor: float = 0.002):
        if not (0.0 < lo <= 1.0 <= hi):
            raise ValueError(f"pulse gain bounds must satisfy 0 < lo <= 1 <= hi; got {lo}, {hi}")
        self.lo = float(lo)
        self.hi = float(hi)
        self.history = int(history)
        self.min_history = int(min_history)
        self.sigma_floor = float(sigma_floor)
        self.last = 1.0
        self.reset()

    def reset(self) -> None:
        self._prev: Any | None = None
        self._hist: list[float] = []
        self.last = 1.0

    def update(self, frame: Any, new_segment: bool = False) -> float:
        """Feed the next frame (temporally adjacent to the previous call unless
        new_segment=True); returns the clamped per-frame gain."""
        y = _to_luma_2d(frame)
        if new_segment or self._prev is None or self._prev.shape != y.shape:
            self._prev = y
            self.last = 1.0
            return self.last
        d = mx.abs(y - self._prev) * (1.0 / 1.4142135623730951)
        self._prev = y
        flat = mx.sort(d.reshape(-1))
        q = float(flat[int(0.25 * (flat.shape[0] - 1))])
        sigma_t = q * (_CHANNEL_FROM_LUMA / _HALF_NORMAL_Q25)
        self._hist.append(sigma_t)
        if len(self._hist) > self.history:
            self._hist.pop(0)
        if len(self._hist) < self.min_history:
            self.last = 1.0
            return self.last
        ref = sorted(self._hist)[len(self._hist) // 2]
        if ref < self.sigma_floor:
            self.last = 1.0
            return self.last
        self.last = max(self.lo, min(self.hi, sigma_t / ref))
        return self.last


__all__ = ["estimate_sigma_map", "estimate_blockiness_map", "NoiseMapTracker",
           "PulseGain"]
