"""Per-frame driver for the MLX SAFMN upscaler.

Both SAFMN variants are single-image networks (no temporal state), so each frame
upscales independently and is emitted immediately; the feed()/flush() shape mirrors
the other upscalers so the harness wiring stays parallel.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from ..upscaler_base import to_rgb_batch

try:
    from . import net
except ImportError:   # running directly as a script
    import net


class SafmnUpscaler:
    """feed()/flush() driver for the per-frame SAFMN upscaler.

    safm_up: "auto" (default) runs the SAFM upsampler each checkpoint was trained
    with (nearest for the stock SAFMN models, bicubic for the PureScale retrains);
    "nearest"/"bicubic" force one -- a mild shape-only override, unlike the pooling
    statistic, which is trained in and follows the checkpoint unconditionally.
    pool_clamp > 0 winsorizes pooled SAFM features to mean +/- k*sigma per channel,
    a stock-weights mitigation for the transient hot-pixel block lattice (0 = off;
    frames with no outliers pass numerically untouched)."""

    def __init__(self, weights: Any = None, safm_up: str = "auto",
                 pool_clamp: float = 0.0,
                 dtype: Any = mx.float16, compile: bool = True):
        if safm_up not in {"auto", "nearest", "bicubic"}:
            raise ValueError(f"SAFMN safm_up must be auto/nearest/bicubic, got {safm_up!r}")
        pool_clamp = float(pool_clamp)
        if pool_clamp < 0.0:
            raise ValueError("SAFMN pool_clamp must be >= 0 (sigmas; 0 = off)")
        self._p = net.load_params(weights, dtype=dtype)
        cfg = net._config(self._p)
        if safm_up != "auto":
            cfg = cfg[:5] + (safm_up, cfg[6])
        if pool_clamp > 0.0:
            cfg = cfg[:6] + (pool_clamp,)
        self._cfg = cfg
        self.scale = self._cfg[3]
        self._fwd = net.make_forward(self._p, self._cfg, compile=compile)

    def reset(self) -> None:
        pass

    def feed(self, rgb: Any, token: Any = None) -> list:
        sr = self._fwd(to_rgb_batch(rgb))
        mx.eval(sr)
        return [(sr[0], token)]

    def flush(self) -> list:
        return []
