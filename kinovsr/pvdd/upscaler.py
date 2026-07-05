"""Windowed streaming wrapper for the MLX PVDD denoiser.

PVDD is a bidirectional recurrent whole-clip denoiser (output resolution == input,
so SCALE = 1). The sliding-window feed()/flush() machinery lives in
../upscaler_base; this wrapper loads a variant's weights and denoises each window.
The `level` checkpoints take a noise-variance value (see LEVEL_PRESETS).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from ..upscaler_base import WindowedUpscaler
from . import PVDD, LEVEL_PRESETS, default_weights_path, _VARIANTS


class PvddDenoiser(WindowedUpscaler):
    """Windowed feed()/flush() driver for PVDD (SCALE 1, denoise-in-place)."""

    SCALE = 1

    def __init__(
        self,
        weights: Any = None,
        variant: str = "pvdd",
        window: int = 10,
        trim: int = 0,
        noise_variance: float | None = None,
        dtype: Any = mx.float16,
    ):
        if variant not in _VARIANTS:
            raise ValueError(f"PVDD variant must be one of {sorted(_VARIANTS)}; got {variant!r}")
        t, w = int(trim), int(window)
        if t < 0:
            raise ValueError(f"PVDD trim must be >= 0; got {t}")
        if w < 1:
            raise ValueError(f"PVDD window must be >= 1; got {w}")
        if t and w <= 2 * t:
            raise ValueError(
                "PVDD window must be greater than 2*trim so each window can emit "
                f"interior frames; got window={w}, trim={t}. Use trim=0 for "
                "reference-like non-overlapping chunks."
            )
        src = weights if weights else default_weights_path(variant)
        if isinstance(src, (str, Path)) and not Path(src).is_file():
            raise FileNotFoundError(
                f"PVDD weights not found at {src}. They are not bundled; convert the "
                "source .pth with scripts/pth_to_safetensors.py (see weights/README.md) "
                "or pass an explicit --pvdd-weights path."
            )
        self.net = PVDD(src, dtype=dtype)
        if noise_variance is not None:
            self._nv: float | None = float(noise_variance)
        elif self.net.is_level:
            self._nv = LEVEL_PRESETS["M"]
        else:
            self._nv = None
        super().__init__(window=w, trim=t)

    def _upscale_window(self, frames: list) -> list:
        return self.net.denoise_clip(frames, noise_variance=self._nv)

    def close(self) -> None:
        """Denoiser-protocol no-op (the harness calls close() on denoise stages)."""
        pass
