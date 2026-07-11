"""Luma/chroma split weighting around a denoise engine (harness glue).

The engines themselves are processor families now
(kinovsr.processors.mc, kinovsr.processors.spatial, and the learned
families); this module keeps the channel-group re-weighting wrapper
the harness composes around any of them until the typed pipeline
owns that composition (the shared luma_strength/chroma_strength
vocabulary keys).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def luma_chroma_blend(orig: Any, new: Any, a_luma: float, a_chroma: float,
                      kr: float = 0.299, kb: float = 0.114) -> Any:
    """Recombine `orig` and `new` (both (H,W,3) RGB in [0,1]) with separate blend
    strengths for luma and chroma: the output luma is lerp(orig, new, a_luma) and the
    chroma is lerp(orig, new, a_chroma). a=1 takes the new (denoised) value, a=0 keeps the
    original; a_luma=a_chroma=1 returns `new` exactly. (kr, kb) are the ITU-R luma
    coefficients (default BT.601); pass the source matrix's so the split matches the clip's
    color space -- though they only affect the result when a_luma != a_chroma (otherwise
    the YCbCr basis cancels out of the lerp). Computed in float32 -- the chroma-scale
    divisions coarsen in fp16."""
    kg = 1.0 - kr - kb
    cb_s, cr_s = 2.0 * (1.0 - kb), 2.0 * (1.0 - kr)
    o = orig.astype(mx.float32)
    n = new.astype(mx.float32)

    def _yc(x):
        y = kr * x[..., 0:1] + kg * x[..., 1:2] + kb * x[..., 2:3]
        return y, (x[..., 2:3] - y) / cb_s, (x[..., 0:1] - y) / cr_s     # y, cb, cr

    yo, cbo, cro = _yc(o)
    yn, cbn, crn = _yc(n)
    y = yo + a_luma * (yn - yo)
    cb = cbo + a_chroma * (cbn - cbo)
    cr = cro + a_chroma * (crn - cro)
    r = y + cr_s * cr
    b = y + cb_s * cb
    g = (y - kr * r - kb * b) / kg
    return mx.clip(mx.concatenate([r, g, b], axis=-1), 0.0, 1.0)


class LumaChromaDenoiser:
    """Wrap any harness denoiser to apply separate luma/chroma blend strengths between
    its input and output -- e.g. denoise chroma hard (a_chroma=1) while keeping luma
    texture (a_luma<1), the split a single joint RGB sigma cannot do. The base still
    denoises RGB jointly; this only re-weights its effect per channel group on the way
    out.

    Threads the input frame through the base's token so delay-line denoisers (FastDVDnet)
    still pair each delayed output with its own input; per-frame denoisers (spatial / mc)
    blend in step. Presents the feed/flush interface either way."""

    def __init__(self, base: Any, luma_strength: float = 1.0, chroma_strength: float = 1.0,
                 kr: float = 0.299, kb: float = 0.114):
        self._base = base
        self._al = float(luma_strength)
        self._ac = float(chroma_strength)
        self._kr = float(kr)       # ITU-R luma coefficients of the source matrix
        self._kb = float(kb)

    def reset(self) -> None:
        self._base.reset()

    def close(self) -> None:
        if hasattr(self._base, "close"):
            self._base.close()

    def set_schedule(self, schedule: Any) -> None:
        if hasattr(self._base, "set_schedule"):
            self._base.set_schedule(schedule)

    def _blend(self, orig: Any, den: Any) -> Any:
        return luma_chroma_blend(orig, den, self._al, self._ac, self._kr, self._kb)

    def feed(self, rgb: Any, token: Any = None) -> list:
        if hasattr(self._base, "feed"):
            return [(self._blend(o, d), t)
                    for d, (o, t) in self._base.feed(rgb, token=(rgb, token))]
        return [(self._blend(rgb, self._base.denoise(rgb)), token)]

    def flush(self) -> list:
        if hasattr(self._base, "flush"):
            return [(self._blend(o, d), t) for d, (o, t) in self._base.flush()]
        return []
