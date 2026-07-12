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

from kinovsr.media.yuv import luma_chroma_blend


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
