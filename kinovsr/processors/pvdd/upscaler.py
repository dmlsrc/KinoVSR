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

from kinovsr.modeling.upscaler_base import WindowedUpscaler

from . import _VARIANTS, LEVEL_PRESETS, PVDD, default_weights_path


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
        noise_map: Any | None = None,
        pulse: Any | None = None,
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
                "source .pth with kinovsr weights convert (see weights/README.md) "
                "or pass an explicit --pvdd-weights path."
            )
        self.net = PVDD(src, dtype=dtype)
        # optional NoiseMapTracker (sigma units): squared into the variance plane
        # the level checkpoints take. Only meaningful for level variants.
        self._tracker = noise_map
        # optional PulseGain: per-frame sigma gain, squared into the variance
        # planes (GOP-phase noise pulsing).
        self._pulse = pulse
        if (self._tracker is not None or self._pulse is not None) and not self.net.is_level:
            raise ValueError(
                "--noise-map / --noise-map-pulse need a level (non-blind) PVDD "
                f"variant; {variant!r} is blind. Use pvdd_level / pvdd_raw_level."
            )
        self.last_noise_map: Any = None   # fp32 (H,W,1) sigma actually used (debug)
        self._pulse_log: list[float] = []
        if noise_variance is not None:
            self._nv: float | None = float(noise_variance)
        elif self.net.is_level:
            self._nv = LEVEL_PRESETS["M"]
        else:
            self._nv = None
        super().__init__(window=w, trim=t)

    def _upscale_window(self, frames: list) -> list:
        gains = None
        if self._pulse is not None:
            from kinovsr.analysis.noise.track import source_since_sync

            tokens = getattr(self, "_window_tokens", None) or []
            # windows are separate segments: restart the diff chain each window
            gains = [self._pulse.update(
                        f, new_segment=(i == 0),
                        since_sync=source_since_sync(
                            tokens[i] if i < len(tokens) else None))
                     for i, f in enumerate(frames)]
            self._pulse_log.extend(gains)
        if self._tracker is not None:
            sig = self._tracker.update(frames)
            if sig is not None:
                self.last_noise_map = sig
                return self.net.denoise_clip(frames, noise_map=sig * sig,
                                             frame_gains=gains)
        return self.net.denoise_clip(frames, noise_variance=self._nv,
                                     frame_gains=gains)

    def run_diagnostics(self) -> list:
        from kinovsr.processors.conditioning import noise_map_diagnostics

        return noise_map_diagnostics(self)

    def debug_images(self) -> dict:
        from kinovsr.processors.conditioning import noise_map_debug_image

        return noise_map_debug_image(self)

    def close(self) -> None:
        """Denoiser-protocol no-op (the harness calls close() on denoise stages)."""
