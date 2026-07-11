"""Temporal state: NoiseMapTracker EMA blending and PulseGain
GOP-phase gain smoothing. Split of the noise_map module.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .estimate import (
    _frame_low_quantile_sigma,
    _to_luma_2d,
    estimate_sigma_map,
)


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
        sigma_t = _frame_low_quantile_sigma(d)
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


