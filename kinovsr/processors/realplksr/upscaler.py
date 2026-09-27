"""Per-frame driver for the MLX RealPLKSR upscaler.

RealPLKSR is a single-image network (no temporal state), so each frame upscales
independently and is emitted immediately; the feed()/flush() shape mirrors the
other upscalers so the harness wiring stays parallel.
"""

from pathlib import Path

import mlx.core as mx

from kinovsr.modeling.upscaler_base import to_rgb_batch

from . import net


class RealPlksrUpscaler:
    """feed()/flush() driver for the per-frame RealPLKSR upscaler.

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
    ) -> None:
        self._p = net.load_params(weights, dtype=dtype)
        self._cfg = net._config(self._p)
        self.scale = self._cfg[4]
        self._fwd = net.make_forward(self._p, self._cfg, compile=compile)

    def reset(self) -> None:
        pass

    def feed(self, rgb: mx.array, token: object = None) -> list[tuple[mx.array, object]]:
        sr = self._fwd(to_rgb_batch(rgb))
        mx.eval(sr)
        return [(sr[0], token)]

    def flush(self) -> list[tuple[mx.array, object]]:
        return []
