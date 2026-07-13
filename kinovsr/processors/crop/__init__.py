"""A geometry-cropping preprocess family (bars and aspect windows).

The crop is declared, not detected: explicit bar counts and/or an aspect
window are pure functions of the input geometry, so ``produces`` computes the
output spec at resolve time and the builder validates the whole chain against
it. Auto bar detection is a probe-time analysis concern; the resolved counts
are written into this family's config before preflight.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from kinovsr.config.helpers import parse_edge_counts, reject_unknown_keys, typed_value
from kinovsr.processors.capabilities import Capability, CapabilitySpec
from kinovsr.processors.crop.geometry import compute_aspect_crop, crop_rgb
from kinovsr.processors.feed_driver import FeedFlushProcessor
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Domain,
    DType,
    Geometry,
    Layout,
    StreamConstraint,
    StreamSpec,
)
from kinovsr.settings import Settings

_ANCHOR_TOKENS = (
    "top-left", "top", "top-right", "left", "center", "right",
    "bottom-left", "bottom", "bottom-right",
)


@dataclasses.dataclass(frozen=True, slots=True)
class CropStageConfig:
    bars: tuple[int, int, int, int]     # (top, bottom, left, right)
    trim: tuple[int, int, int, int]     # junk-edge trim, folded after bars
    aspect: tuple[int, int] | None      # display aspect of the window
    anchor: str
    offset: tuple[int, int]             # (dx, dy), right/down positive


def _combined_crop(width: int, height: int, pixel_aspect,
                   config: CropStageConfig) -> tuple[int, int, int, int]:
    """Total (top, bottom, left, right) crop for a frame of WxH.

    ``trim`` (junk-edge bands cropped off rather than filled) folds
    additively with ``bars`` before the aspect window - the harness's
    bars -> trim -> aspect order. ``aspect`` is a DISPLAY aspect; on
    anamorphic sources the pixel aspect folds into the storage-domain
    target (display = storage * PAR), so 16:9 means 16:9 on screen
    regardless of storage shape.
    """
    from fractions import Fraction as _F

    t, b, left, r = (bar + trim for bar, trim
                     in zip(config.bars, config.trim, strict=True))
    inner_w, inner_h = width - left - r, height - t - b
    if inner_w < 2 or inner_h < 2:
        raise ValueError(
            f"bar crop {config.bars} leaves no picture in {width}x{height}")
    if config.aspect is not None:
        storage = (_F(config.aspect[0], config.aspect[1])
                   / _F(pixel_aspect))
        at, ab, al, ar = compute_aspect_crop(
            inner_w, inner_h, storage.numerator, storage.denominator,
            dx=config.offset[0], dy=config.offset[1], anchor=config.anchor)
        return (t + at, b + ab, left + al, r + ar)
    return (t, b, left, r)


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, CropStageConfig)
    geometry = spec.frame.geometry
    t, b, left, r = _combined_crop(
        geometry.width, geometry.height, geometry.pixel_aspect, config)
    frame = dataclasses.replace(
        spec.frame,
        geometry=Geometry(geometry.width - left - r,
                          geometry.height - t - b,
                          geometry.pixel_aspect))
    return dataclasses.replace(spec, frame=frame)


class _CropDriver:
    def __init__(self, crop: tuple[int, int, int, int]) -> None:
        self._crop = crop

    def feed(self, rgb: Any, token: Any = None) -> list:
        return [(crop_rgb(rgb, self._crop), token)]

    def flush(self) -> list:
        return []

    def reset(self) -> None:
        pass


class CropFactory:
    name = "crop"

    capabilities = {
        Capability.PREPROCESS: CapabilitySpec(
            capability=Capability.PREPROCESS,
            profiles=(),
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32, DType.FLOAT16),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            produces=_produces,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> CropStageConfig:
        reject_unknown_keys(raw, ("bars", "trim", "aspect", "anchor",
                                  "offset"))
        bars_spec = typed_value(raw, "bars", str)
        bars = parse_edge_counts(bars_spec) if bars_spec else (0, 0, 0, 0)
        trim_spec = typed_value(raw, "trim", str)
        trim = parse_edge_counts(trim_spec) if trim_spec else (0, 0, 0, 0)
        aspect_spec = typed_value(raw, "aspect", str)
        aspect: tuple[int, int] | None = None
        if aspect_spec:
            parts = aspect_spec.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"aspect must be W:H, got {aspect_spec!r}")
            aspect = (int(parts[0]), int(parts[1]))
            if aspect[0] <= 0 or aspect[1] <= 0:
                raise ValueError("aspect must be positive")
        if bars == trim == (0, 0, 0, 0) and aspect is None:
            raise ValueError(
                "state bars, trim, and/or aspect; a crop stage that crops "
                "nothing is config noise")
        anchor = typed_value(raw, "anchor", str, "center")
        if anchor not in _ANCHOR_TOKENS:
            raise ValueError(f"anchor must be one of {_ANCHOR_TOKENS}")
        offset_spec = typed_value(raw, "offset", str, "0,0")
        parts = offset_spec.split(",")
        if len(parts) != 2:
            raise ValueError(f"offset must be DX,DY, got {offset_spec!r}")
        return CropStageConfig(
            bars=bars, trim=trim, aspect=aspect, anchor=anchor,
            offset=(int(parts[0]), int(parts[1])))

    def build(self, config: CropStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        # The combined rect is a pure function of the validated input
        # geometry; bind it at prepare time through the driver factory.
        holder: dict[str, Any] = {}

        class _Processor(FeedFlushProcessor):
            def prepare(self, input_spec: StreamSpec,
                        context: PipelineContext) -> None:
                geometry = input_spec.frame.geometry
                holder["crop"] = _combined_crop(
                    geometry.width, geometry.height,
                    geometry.pixel_aspect, config)
                super().prepare(input_spec, context)

        return _Processor(lambda: _CropDriver(holder["crop"]))


FACTORY = CropFactory()

__all__ = ["FACTORY", "CropFactory", "CropStageConfig"]
