"""Streaming windowed wrapper around the recurrent BasicVSR++ 1x restoration net,
for use as a preprocessor (compression / noise / blur cleanup ahead of an upscaler).

Same sliding-window feed()/flush() machinery as the SR upscaler, but each emitted
frame is restored at the SAME resolution (net.restore, is_low_res_input=False).
Being temporal (bidirectional, second-order recurrent), it does what a per-frame
deblocker cannot: it enforces frame-to-frame consistency, so it does not pulse at
the GOP period the way a single-frame restorer does on inter-coded video.
"""
from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

from kinovsr.modeling.upscaler_base import WindowedUpscaler

try:
    from . import net
except ImportError:   # running directly as a script
    import net


class BasicVsrRestorer(WindowedUpscaler):
    """Windowed feed()/flush() driver for net.restore (1x recurrent restoration).

    weights: a restoration variant token (decompress_track1/2/3, denoise,
    deblur_dvd, deblur_gopro) or a .safetensors path. strength blends the restored
    frame with the original (1.0 = full restore, 0.0 = passthrough) so the cleanup
    can be dialed like the other preprocessors.
    """

    SCALE = 1

    def __init__(self, weights: Any = None, window: int = 14, trim: int = 2,
                 strength: float = 1.0, flow_mode: str = "spynet", ensemble: bool = False):
        if flow_mode not in ("spynet", "zero", "vt"):
            raise ValueError(
                f"BasicVSR++ restore flow_mode must be 'spynet', 'zero', or 'vt'; got {flow_mode!r}")
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"restore strength must be in [0, 1]; got {strength!r}")
        self._p = net.load_params(net.resolve_restore_weights(weights))
        if net.is_low_res_input(self._p):
            raise ValueError(
                "this checkpoint is a 4x-SR BasicVSR++ model, not a 1x-restoration one; "
                "use --upscale basicvsrpp for SR. Restoration tokens: "
                f"{list(net._RESTORE_VARIANTS)}")
        self._flow_mode = flow_mode
        self._strength = float(strength)
        self._ensemble = bool(ensemble)
        super().__init__(window=max(int(window), 2 * int(trim) + 1), trim=trim)

    def _upscale_window(self, frames: list) -> list:
        fn = net.restore_ensemble if self._ensemble else net.restore
        out = fn(frames, self._p, flow_mode=self._flow_mode)
        if self._strength != 1.0:
            s = self._strength
            out = [mx.clip(s * o + (1.0 - s) * f, 0.0, 1.0) for o, f in zip(out, frames, strict=True)]
            for o in out:
                mx.eval(o)
        return out


_log = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = BasicVsrRestorer("decompress_track1", window=10, trim=2)
    mx.random.seed(0)
    emitted: list = []
    for i in range(14):
        emitted.extend(r.feed(mx.random.uniform(shape=(60, 80, 3)), token=i))
    emitted.extend(r.flush())
    toks = [t for _, t in emitted]
    _log.info(f"restore: emitted {len(emitted)}/14, order ok: {toks == list(range(14))}, "
          f"frame shape {tuple(emitted[0][0].shape)}")
