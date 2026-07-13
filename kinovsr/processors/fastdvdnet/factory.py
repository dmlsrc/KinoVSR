"""FastDVDnet's processor factory: a self-buffered 5-frame denoiser.

The net consumes two future neighbors per output frame, paid as output
delay through its own buffer (CENTERED, radius 2). ``strength`` maps
onto the trained sigma range. Noise-map conditioning is typed stage config.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.processors.capabilities import (
    Capability,
    CapabilitySpec,
    TemporalMode,
)
from kinovsr.processors.conditioning import (
    NOISE_MAP_KEYS,
    NoiseMapConfig,
    build_conditioning,
    parse_noise_map,
)
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
)
from kinovsr.settings import Settings

_PROFILES = ("clipped", "standard")


@dataclasses.dataclass(frozen=True, slots=True)
class FastDvdStageConfig:
    weights_path: str | None
    variant: str
    strength: float
    luma_strength: float
    chroma_strength: float
    noise_map: NoiseMapConfig


class FastDvdFactory:
    name = "fastdvdnet"

    capabilities = {
        Capability.DENOISE: CapabilitySpec(
            capability=Capability.DENOISE,
            profiles=_PROFILES,
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32, DType.FLOAT16),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            temporal_mode=TemporalMode.CENTERED,
            temporal_radius=2,
            stateful=True,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> FastDvdStageConfig:
        reject_unknown_keys(
            raw, ("weights", "strength", *LUMA_CHROMA_KEYS, *NOISE_MAP_KEYS))
        strength = typed_value(raw, "strength", float, 0.5)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        luma_strength, chroma_strength = parse_luma_chroma(raw)
        return FastDvdStageConfig(
            weights_path=typed_value(raw, "weights", str)
            or settings.fastdvdnet_weights,
            variant=profile or "clipped",
            strength=strength,
            luma_strength=luma_strength,
            chroma_strength=chroma_strength,
            noise_map=parse_noise_map(raw),
        )

    def build(self, config: FastDvdStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            from . import FastDvdDenoiser

            tracker, pulse = build_conditioning(config.noise_map)
            return FastDvdDenoiser(
                config.weights_path,
                variant=config.variant,
                strength=config.strength,
                noise_map=tracker, map_refresh=config.noise_map.refresh,
                pulse=pulse, map_floor=config.noise_map.floor)

        return FeedFlushProcessor(
            make_driver,
            luma_strength=config.luma_strength,
            chroma_strength=config.chroma_strength)


FACTORY = FastDvdFactory()
