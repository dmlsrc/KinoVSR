"""Real-ESRGAN's processor factory: a stateless per-frame upscaler family.

Profiles resolve from the family manifest with per-profile scales
(x2plus is the 2x checkpoint). ``denoise_strength`` is the dni dial of
the general profile only - it blends against the wdn companion weight
the loader locates by filename.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.processors.capabilities import Capability, CapabilitySpec
from kinovsr.processors.feed_driver import FeedFlushProcessor
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Domain,
    DType,
    Layout,
    StreamConstraint,
    StreamSpec,
)
from kinovsr.settings import Settings

_PROFILES = ("general", "x4plus", "realesrnet", "bsrgan", "bsrnet",
             "x2plus", "anime", "animevideo", "esrgan")
_DEFAULT_PROFILE = "general"


@functools.cache
def _profile_scales() -> dict[str, int]:
    from kinovsr.modeling.weights import load_registered

    manifest = load_registered("realesrgan")
    return {name: int(profile.defaults["scale"])
            for name, profile in manifest.profiles.items()}


@dataclasses.dataclass(frozen=True, slots=True)
class RealEsrganStageConfig:
    weights_spec: str
    scale: int
    denoise_strength: float


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, RealEsrganStageConfig)
    frame = dataclasses.replace(
        spec.frame, geometry=spec.frame.geometry.scaled(config.scale))
    return dataclasses.replace(spec, frame=frame)


class RealEsrganFactory:
    name = "realesrgan"

    capabilities = {
        Capability.UPSCALE: CapabilitySpec(
            capability=Capability.UPSCALE,
            profiles=_PROFILES,
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
    ) -> RealEsrganStageConfig:
        reject_unknown_keys(raw, ("weights", "scale", "denoise_strength"))
        weights = typed_value(raw, "weights", str) or settings.realesrgan_weights
        denoise_strength = typed_value(raw, "denoise_strength", float, 1.0)
        if not 0.0 <= denoise_strength <= 1.0:
            raise ValueError("denoise_strength must be in [0, 1]")
        scale = typed_value(raw, "scale", int)
        scales = _profile_scales()
        token = profile or (weights if weights in scales else None)
        if scale is None:
            if token is None and weights is not None:
                raise ValueError(
                    "state scale when weights is an explicit path "
                    "(profiles declare it)")
            scale = scales[token or _DEFAULT_PROFILE]
        return RealEsrganStageConfig(
            weights_spec=weights or profile or _DEFAULT_PROFILE,
            scale=scale, denoise_strength=denoise_strength)

    def build(self, config: RealEsrganStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            from .upscaler import RealEsrganUpscaler

            driver = RealEsrganUpscaler(
                config.weights_spec,
                denoise_strength=config.denoise_strength)
            if driver.scale != config.scale:
                raise ValueError(
                    f"checkpoint scale {driver.scale}x does not match the "
                    f"declared scale {config.scale}x")
            return driver

        return FeedFlushProcessor(make_driver)


FACTORY = RealEsrganFactory()
