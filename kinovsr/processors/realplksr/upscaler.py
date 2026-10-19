"""Per-frame driver for MLX RealPLKSR restoration and upscaling.

RealPLKSR is a single-image network (no temporal state), so each frame is
processed independently and emitted immediately; the feed()/flush() shape mirrors the
other upscalers so the harness wiring stays parallel.
"""

from pathlib import Path

import mlx.core as mx

from kinovsr.modeling.upscaler_base import to_rgb_batch

from . import net


class RealPlksrUpscaler:
    """feed()/flush() driver for the per-frame RealPLKSR processor.

    dtype defaults to fp16. The LayerNorm+DySample checkpoints (public2x) are
    fp16-safe; the GroupNorm+PixelShuffle checkpoints (nomos4x) are not fully
    fp16-stable per the community, so net.py keeps a single measured precision
    island -- the GroupNorm/LayerNorm reductions run in fp32 (the activations
    overflow an fp16 variance) -- regardless of the storage dtype. Pass
    dtype=mx.float32 to force a full fp32 run.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        dtype: mx.Dtype = mx.float16,
        compile: bool = True,
        strength: float = 1.0,
    ) -> None:
        self._p = net.load_params(weights, dtype=dtype)
        self._cfg = net._config(self._p)
        self.scale = self._cfg[4]
        self.strength = strength
        self._fwd = net.make_forward(self._p, self._cfg, compile=compile)

    def reset(self) -> None:
        pass

    def feed(self, rgb: mx.array, token: object = None) -> list[tuple[mx.array, object]]:
        source = to_rgb_batch(rgb)
        sr = self._fwd(source)
        if self.scale == 1 and self.strength != 1.0:
            sr = mx.clip(source + self.strength * (sr - source), 0.0, 1.0)
        mx.eval(sr)
        return [(sr[0], token)]

    def flush(self) -> list[tuple[mx.array, object]]:
        return []
