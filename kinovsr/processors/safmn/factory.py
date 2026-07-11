"""SAFMN's processor factory: a stateless per-frame upscaler family.

Profiles resolve from the family manifest with per-profile scales (the
2x and 4x checkpoints share one family). The SAFM branch mode follows
the checkpoint filename, so explicit ``weights`` paths must keep
"purescale" in the stem for those retrains; ``safm_up`` and
``pool_clamp`` are the family's creative/mitigation dials.
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

_PROFILES = ("light", "real", "real2x", "purescale", "purescale2x",
             "purescale2x-sharp")
_DEFAULT_PROFILE = "light"
_SAFM_UP = ("auto", "nearest", "bicubic")


@functools.cache
def _profile_scales() -> dict[str, int]:
    from kinovsr.modeling.weights import load_registered

    manifest = load_registered("safmn")
    return {name: int(profile.defaults["scale"])
            for name, profile in manifest.profiles.items()}


@dataclasses.dataclass(frozen=True, slots=True)
class SafmnStageConfig:
    weights_spec: str
    scale: int
    safm_up: str
    pool_clamp: float


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, SafmnStageConfig)
    frame = dataclasses.replace(
        spec.frame, geometry=spec.frame.geometry.scaled(config.scale))
    return dataclasses.replace(spec, frame=frame)


class SafmnFactory:
    name = "safmn"

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
    ) -> SafmnStageConfig:
        reject_unknown_keys(raw, ("weights", "scale", "safm_up", "pool_clamp"))
        weights = typed_value(raw, "weights", str) or settings.safmn_weights
        safm_up = typed_value(raw, "safm_up", str, "auto")
        if safm_up not in _SAFM_UP:
            raise ValueError(f"safm_up must be one of {_SAFM_UP}")
        pool_clamp = typed_value(raw, "pool_clamp", float, 0.0)
        if pool_clamp < 0.0:
            raise ValueError("pool_clamp must be >= 0 (0 = off)")
        scale = typed_value(raw, "scale", int)
        scales = _profile_scales()
        token = profile or (weights if weights in scales else None)
        if scale is None:
            if token is None and weights is not None:
                raise ValueError(
                    "state scale when weights is an explicit path "
                    "(profiles declare it)")
            scale = scales[token or _DEFAULT_PROFILE]
        return SafmnStageConfig(
            weights_spec=weights or profile or _DEFAULT_PROFILE,
            scale=scale, safm_up=safm_up, pool_clamp=pool_clamp)

    def build(self, config: SafmnStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            from . import SafmnUpscaler

            driver = SafmnUpscaler(config.weights_spec,
                                   safm_up=config.safm_up,
                                   pool_clamp=config.pool_clamp)
            if driver.scale != config.scale:
                raise ValueError(
                    f"checkpoint scale {driver.scale}x does not match the "
                    f"declared scale {config.scale}x")
            return driver

        return FeedFlushProcessor(make_driver)


FACTORY = SafmnFactory()
