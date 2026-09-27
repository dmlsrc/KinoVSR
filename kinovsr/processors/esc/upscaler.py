"""Per-frame driver for the MLX ESC-Real upscaler (single-image, no temporal state)."""

from pathlib import Path

import mlx.core as mx

from kinovsr.modeling.upscaler_base import to_rgb_batch

from . import net


class EscUpscaler:
    """feed()/flush() driver for the per-frame ESC-Real upscaler."""

    def __init__(
        self,
        weights: str | Path | None = None,
        dtype: mx.Dtype = mx.float16,
        compile: bool = True,
    ) -> None:
        self._p = net.load_params(weights, dtype=dtype)
        self._cfg = net._config(self._p)
        self.scale = self._cfg[6]
        self._fwd = net.make_forward(self._p, self._cfg, compile=compile)

    def reset(self) -> None:
        pass

    def feed(self, rgb: mx.array, token: object = None) -> list[tuple[mx.array, object]]:
        sr = self._fwd(to_rgb_batch(rgb))
        mx.eval(sr)
        return [(sr[0], token)]

    def flush(self) -> list[tuple[mx.array, object]]:
        return []
