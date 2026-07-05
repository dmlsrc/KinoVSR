"""PVDD real-world video denoiser (MLX).

One `pvdd0815` net serves all six released checkpoints (MIT, Xu et al. 2022):
  sRGB blind   -- pvdd / crvd / davis   (num_in=3, level=False)
  sRGB level   -- pvdd                  (num_in=3, level=True, noise-variance map)
  raw blind    -- pvdd                  (num_in=4, packed Bayer)
  raw level    -- pvdd                  (num_in=4, level=True)

Weights are not bundled; see weights/README.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from .net import PVDDConfig, pvdd_forward

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_VARIANTS = {
    "pvdd": "pvdd_srgb_nolevel.safetensors",
    "crvd": "crvd_srgb_nolevel.safetensors",
    "davis": "davis_srgb_nolevel.safetensors",
    "pvdd_level": "pvdd_srgb_level.safetensors",
    "pvdd_raw": "pvdd_raw_nolevel.safetensors",
    "pvdd_raw_level": "pvdd_raw_level.safetensors",
}
# S/M/L reference noise-variance levels for the level (non-blind) checkpoints.
LEVEL_PRESETS = {"S": 0.000687765, "M": 0.002191, "L": 0.005470}


def default_weights_path(variant: str = "pvdd") -> Path:
    return _WEIGHTS_DIR / _VARIANTS[variant]


def _pad4(f: Any) -> Any:
    """Edge-replicate the bottom/right of (1,H,W,C) up to a multiple of 4."""
    h, w = int(f.shape[1]), int(f.shape[2])
    ph, pw = (-h) % 4, (-w) % 4
    if ph:
        f = mx.concatenate([f, mx.broadcast_to(f[:, h - 1:h], (f.shape[0], ph, f.shape[2], f.shape[3]))], axis=1)
    if pw:
        f = mx.concatenate([f, mx.broadcast_to(f[:, :, w - 1:w], (f.shape[0], f.shape[1], pw, f.shape[3]))], axis=2)
    return f


def _to_params(raw: dict[str, Any], dtype: Any) -> tuple[dict[str, Any], PVDDConfig]:
    """Build the forward param dict from a raw (torch-layout) weight dict.

    Conv weights (4D, O,I,kH,kW) transpose to OHWI; Linear/tables/norms/biases
    stay; relative_position_index becomes int32; feat_STTB.* (unused pre-attention)
    is dropped. num_in and level are inferred from shapes.
    """
    num_in = int(raw["conv_last.weight"].shape[0])              # (num_in,64,3,3)
    clean_in = int(raw["clean_model.conv_in.weight"].shape[1])  # num_in(+1 if level)
    level = clean_in == num_in + 1
    p: dict[str, Any] = {}
    for k, v in raw.items():
        if k.startswith("feat_STTB"):
            continue
        if k.endswith("relative_position_index"):
            p[k] = v.astype(mx.int32)
            continue
        if v.ndim == 4:
            v = mx.transpose(v, (0, 2, 3, 1))                   # OHWI
        p[k] = v.astype(dtype)
    return p, PVDDConfig(num_in=num_in, level=level)


def load_pvdd(source: str | Path | dict, dtype: Any = mx.float32) -> tuple[dict, PVDDConfig]:
    """Load PVDD params from a .safetensors path or a raw weight dict."""
    if isinstance(source, dict):
        raw = source
    else:
        sp = Path(source)
        if sp.suffix in {".pth", ".pt"}:
            raise ValueError(
                f"PVDD weights must be .safetensors, got {sp}. Convert with "
                "scripts/pth_to_safetensors.py first (see weights/README.md)."
            )
        raw = mx.load(str(sp))
    return _to_params(raw, dtype)


class PVDD:
    """Stateless whole-clip PVDD denoiser. Call denoise_clip once per window."""

    def __init__(self, source: str | Path | dict, dtype: Any = mx.float32):
        self.dtype = dtype
        self.params, self.cfg = load_pvdd(source, dtype=dtype)

    @property
    def input_channels(self) -> int:
        return self.cfg.num_in

    @property
    def is_level(self) -> bool:
        return self.cfg.level

    def denoise_clip(self, frames_nhwc: list, noise_variance: float | None = None) -> list:
        """frames: list of (1,H,W,C) mx arrays in [0,1]. Returns denoised list.

        Inputs are edge-padded to a multiple of 4 so the x4 down/up feature path
        round-trips, then cropped back to the original size.
        """
        h0, w0 = int(frames_nhwc[0].shape[1]), int(frames_nhwc[0].shape[2])
        frames = [_pad4(f.astype(self.dtype)) for f in frames_nhwc]
        hp, wp = int(frames[0].shape[1]), int(frames[0].shape[2])
        nm = None
        if self.cfg.level:
            if noise_variance is None:
                noise_variance = LEVEL_PRESETS["M"]
            nm = mx.full((1, hp, wp, 1), float(noise_variance), dtype=self.dtype)
        outs = pvdd_forward(frames, self.params, self.cfg, noise_map=nm)
        return [o[:, :h0, :w0, :] for o in outs]


__all__ = ["PVDD", "load_pvdd", "default_weights_path", "LEVEL_PRESETS", "PVDDConfig"]
