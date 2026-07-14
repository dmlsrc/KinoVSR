"""Per-frame CoreImage spatial denoise (CINoiseReduction).

Cheap, no temporal state; runs at native resolution before SR like
every denoise family. MLX-array in / MLX-array out, fp16 through
CoreImage so there is no 8-bit round trip.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import mlx.core as mx

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.media import pixel_buffers as _pb
from kinovsr.native.frameworks import Foundation, Quartz
from kinovsr.processors.capabilities import Capability, CapabilitySpec
from kinovsr.processors.feed_driver import (
    LUMA_CHROMA_KEYS,
    FeedFlushProcessor,
    parse_luma_chroma,
)
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Domain,
    DType,
    Layout,
    StreamConstraint,
    StreamSpec,
)
from kinovsr.settings import Settings


class SpatialDenoiser:
    """Per-frame CoreImage CINoiseReduction. Spatial only; no temporal state."""

    def __init__(self, strength: float = 0.5):
        # CINoiseReduction's inputNoiseLevel is ~0.0-0.1 in practice; map strength
        # onto a gentle range so strength=0.5 is a moderate clean.
        self.noise_level = 0.01 + 0.04 * float(strength)
        self.sharpness = 0.4

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def denoise(self, rgb_f32: Any) -> Any:
        # fp16 in / fp16 out: feed CoreImage a half-float CIImage and render back
        # to half-float (kCIFormatRGBAh), so no 8-bit quantization round trip.
        rgb_f32 = mx.clip(rgb_f32[..., :3].astype(mx.float32), 0.0, 1.0)
        h, w = int(rgb_f32.shape[0]), int(rgb_f32.shape[1])
        rgba = mx.concatenate(
            [rgb_f32.astype(mx.float16), mx.ones((h, w, 1), dtype=mx.float16)], axis=-1,
        )
        src = memoryview(mx.contiguous(rgba)).cast("B")
        buf = bytearray(w * h * 8)
        with _pb.ci_render_scope() as context:
            data = Foundation.NSData.dataWithBytes_length_(src, len(src))
            ci = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
                data, w * 8, (w, h), Quartz.kCIFormatRGBAh, _pb.srgb_colorspace(),
            )
            filt = Quartz.CIFilter.filterWithName_("CINoiseReduction")
            filt.setValue_forKey_(ci, "inputImage")
            filt.setValue_forKey_(float(self.noise_level), "inputNoiseLevel")
            filt.setValue_forKey_(float(self.sharpness), "inputSharpness")
            out = filt.valueForKey_("outputImage")
            context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
                out, buf, w * 8, ((0, 0), (w, h)),
                Quartz.kCIFormatRGBAh, _pb.srgb_colorspace(),
            )
        rgba_out = mx.array(memoryview(buf)).view(mx.float16).reshape(h, w, 4)
        return mx.contiguous(rgba_out[..., :3]).astype(mx.float32)


# ===========================================================================
# Processor family: a per-frame native spatial denoiser
# ===========================================================================

@dataclasses.dataclass(frozen=True, slots=True)
class SpatialStageConfig:
    strength: float
    luma_strength: float = 1.0
    chroma_strength: float = 1.0


def _passthrough(spec: StreamSpec, config: object) -> StreamSpec:
    return spec


class _SpatialDriver:
    """feed()/flush() shape over the per-frame engine."""

    def __init__(self, config: SpatialStageConfig) -> None:
        self._engine = SpatialDenoiser(strength=config.strength)

    def feed(self, rgb: Any, token: Any = None) -> list:
        return [(self._engine.denoise(rgb), token)]

    def flush(self) -> list:
        return []

    def reset(self) -> None:
        self._engine.reset()

    def close(self) -> None:
        self._engine.close()


class SpatialFactory:
    name = "spatial"

    capabilities = {
        Capability.DENOISE: CapabilitySpec(
            capability=Capability.DENOISE,
            profiles=(),
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32,),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            produces=_passthrough,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> SpatialStageConfig:
        reject_unknown_keys(raw, ("strength", *LUMA_CHROMA_KEYS))
        strength = typed_value(raw, "strength", float, 0.5)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        luma_strength, chroma_strength = parse_luma_chroma(raw)
        return SpatialStageConfig(strength=strength,
                                  luma_strength=luma_strength,
                                  chroma_strength=chroma_strength)

    def build(self, config: SpatialStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        return FeedFlushProcessor(
            lambda: _SpatialDriver(config),
            luma_strength=config.luma_strength,
            chroma_strength=config.chroma_strength)


FACTORY = SpatialFactory()
