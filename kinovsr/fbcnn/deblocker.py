"""Per-frame FBCNN JPEG-artifact-removal deblocker, RGB in / RGB out.

FBCNN is a single-image network (no temporal window), so this is a stateless per-frame
stage -- denoise(rgb) restores each frame independently. Pair it before the scaler to
strip JPEG / intra-block artifacts. Because it is single-image, BLIND mode
(quality=None) re-estimates the quality factor per frame, which can flicker on video as
the estimate drifts shot to shot; pass a fixed `quality` (a JPEG quality factor, lower =
stronger removal) for a temporally stable result. Unlike the temporal STDF deblocker it
does no noise averaging, so it is a pure deblocker, not a partial denoiser.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import net


class FbcnnDeblocker:
    """Stateless per-frame RGB JPEG-artifact deblocker (FBCNN color)."""

    MAP_REFRESH = 64   # frames between blockiness-mask refreshes

    def __init__(self, weights: Any = None, quality: Any = None, strength: float = 1.0,
                 compile: bool = True, dtype: Any = mx.float16,
                 blockiness_map: Any = None):
        self._p = net.load_params(weights, dtype=dtype)
        self._in_nc, self._nb = net._config(self._p)
        if self._in_nc != 3:
            raise ValueError(
                f"FbcnnDeblocker expects the RGB (color) FBCNN checkpoint (in_nc=3), got "
                f"{self._in_nc}; the grayscale variants would need a luma path not wired here.")
        # User-facing JPEG quality (1-100, lower = more compressed = stronger removal)
        # maps to the model's inverted-quality qf_input = 1 - quality/100; None = blind.
        self._quality = None if quality is None else float(quality)
        self._qf_input = None if quality is None else max(0.0, min(1.0, 1.0 - float(quality) / 100.0))
        # strength = linear dry/wet on the correction (the QF-independent knob); a fixed
        # quality also skips the QF predictor, so a pinned-quality run is the faster one.
        self._strength = float(strength)
        # optional blockiness tracker: per-pixel wet/dry mask on the correction.
        # FBCNN's strength is an output lerp, so full-strength net + outside
        # blend of mask*strength is exact.
        self._tracker = blockiness_map
        self._mask: Any = None
        self._recent: list = []
        self._since_refresh = 0
        self.last_blockiness_map: Any = None   # fp32 (H,W,1) (debug)
        net_strength = 1.0 if self._tracker is not None else self._strength
        self._fwd = net.make_forward(self._p, self._qf_input, net_strength, self._nb,
                                     compile=compile)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def denoise(self, rgb_f32: Any) -> Any:
        """Restore one RGB frame (H,W,3) in [0,1]; returns (H,W,3)."""
        a = rgb_f32 if rgb_f32.ndim == 4 else rgb_f32[None]
        inp = mx.clip(a[..., :3].astype(mx.float32), 0.0, 1.0)
        out = mx.clip(self._fwd(inp), 0.0, 1.0)
        if self._tracker is not None:
            self._recent.append(inp)
            if len(self._recent) > 6:
                self._recent.pop(0)
            if self._mask is None or self._since_refresh >= self.MAP_REFRESH:
                m = self._tracker.update(self._recent)
                if m is not None:
                    self.last_blockiness_map = m
                    self._mask = m[None]           # (1,H,W,1)
                self._since_refresh = 0
            else:
                self._since_refresh += 1
            if self._mask is not None:
                out = inp + (self._mask * self._strength) * (out - inp)
        mx.eval(out)
        return out[0]
