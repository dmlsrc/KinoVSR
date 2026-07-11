"""FBCNN's processor factory: a stateless per-frame JPEG deblocker.

``quality`` mirrors the flag grammar: "auto" (per-tile comb measurement
with ``quality_fallback`` where the comb declines), "blind" (the net's
own estimator), or a number 1-100 pinning one global QF. Blockiness-map
conditioning stays harness-wired until conditioning becomes stage
config.
"""

from __future__ import annotations

import dataclasses
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
)
from kinovsr.settings import Settings

_PROFILES = ("color",)


def _parse_quality(spec: str) -> Any:
    q = spec.strip().lower()
    if q == "auto":
        return "auto"
    if q in ("blind", "none"):
        return None
    try:
        value = float(q)
    except ValueError:
        raise ValueError(
            f"bad quality {spec!r}: expected 'auto', 'blind', or a "
            f"number 1-100") from None
    if not 1.0 <= value <= 100.0:
        raise ValueError("quality must be in [1, 100]")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class FbcnnStageConfig:
    weights_path: str | None
    quality: Any                 # "auto" | None (blind) | float QF
    quality_fallback: float
    strength: float


class FbcnnFactory:
    name = "fbcnn"

    capabilities = {
        Capability.DEBLOCK: CapabilitySpec(
            capability=Capability.DEBLOCK,
            profiles=_PROFILES,
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32, DType.FLOAT16),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> FbcnnStageConfig:
        reject_unknown_keys(
            raw, ("weights", "strength", "quality", "quality_fallback"))
        strength = typed_value(raw, "strength", float, 1.0)
        if strength < 0.0:
            raise ValueError("strength must be >= 0")
        quality_fallback = typed_value(raw, "quality_fallback", float, 50.0)
        if not 1.0 <= quality_fallback <= 100.0:
            raise ValueError("quality_fallback must be in [1, 100]")
        return FbcnnStageConfig(
            weights_path=typed_value(raw, "weights", str)
            or settings.fbcnn_weights,
            quality=_parse_quality(typed_value(raw, "quality", str, "auto")),
            quality_fallback=quality_fallback,
            strength=strength,
        )

    def build(self, config: FbcnnStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            from . import FbcnnDeblocker

            return FbcnnDeblocker(
                config.weights_path,
                quality=config.quality,
                strength=config.strength,
                quality_fallback=config.quality_fallback)

        return FeedFlushProcessor(make_driver)


FACTORY = FbcnnFactory()
