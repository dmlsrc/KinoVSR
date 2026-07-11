"""ESC's processor factory: a stateless per-frame 4x upscaler.

Profiles resolve from the family manifest; each declares its scale so
``produces`` stays resolve-time pure. An explicit ``weights`` path must
state ``scale`` (the build validates it against the checkpoint).
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

_PROFILES = ("gan", "mse")
_DEFAULT_PROFILE = "gan"


@functools.cache
def _profile_scales() -> dict[str, int]:
    from kinovsr.modeling.weights import load_registered

    manifest = load_registered("esc")
    return {name: int(profile.defaults["scale"])
            for name, profile in manifest.profiles.items()}


@dataclasses.dataclass(frozen=True, slots=True)
class EscStageConfig:
    weights_spec: str
    scale: int


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, EscStageConfig)
    frame = dataclasses.replace(
        spec.frame, geometry=spec.frame.geometry.scaled(config.scale))
    return dataclasses.replace(spec, frame=frame)


class EscFactory:
    name = "esc"

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
    ) -> EscStageConfig:
        reject_unknown_keys(raw, ("weights", "scale"))
        weights = typed_value(raw, "weights", str) or settings.esc_weights
        scale = typed_value(raw, "scale", int)
        scales = _profile_scales()
        token = profile or (weights if weights in scales else None)
        if scale is None:
            if token is None and weights is not None:
                raise ValueError(
                    "state scale when weights is an explicit path "
                    "(profiles declare it)")
            scale = scales[token or _DEFAULT_PROFILE]
        return EscStageConfig(
            weights_spec=weights or profile or _DEFAULT_PROFILE,
            scale=scale)

    def build(self, config: EscStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            from .upscaler import EscUpscaler

            driver = EscUpscaler(config.weights_spec)
            if driver.scale != config.scale:
                raise ValueError(
                    f"checkpoint scale {driver.scale}x does not match the "
                    f"declared scale {config.scale}x")
            return driver

        return FeedFlushProcessor(make_driver)


FACTORY = EscFactory()
