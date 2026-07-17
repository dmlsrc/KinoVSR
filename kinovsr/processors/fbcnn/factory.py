"""FBCNN's processor factory: a stateless per-frame JPEG deblocker.

``quality`` mirrors the flag grammar: "auto" (per-tile comb measurement
with ``quality_fallback`` where the comb declines), "blind" (the net's
own estimator), or a number 1-100 pinning one global QF. Blockiness-map
conditioning is typed stage config (the shared kinovsr.processors.conditioning
helper builds the tracker the deblocker consumes).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.processors.capabilities import Capability, CapabilitySpec
from kinovsr.processors.conditioning import (
    DEBLOCK_MAP_KEYS,
    DeblockMapConfig,
    build_blockiness_tracker,
    parse_deblock_map,
)
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


def _parse_quality(spec: Any) -> Any:
    # Grammar is 'auto' | 'blind' | a number. The number arrives as a str
    # from a quoted TOML value, but as a float from the flag CLI (untyped
    # registry dials are floatified) and as an int/float from unquoted TOML.
    if isinstance(spec, bool) or not isinstance(spec, (str, int, float)):
        raise ValueError(
            f"bad quality {spec!r}: expected 'auto', 'blind', or a "
            f"number 1-100")
    if isinstance(spec, (int, float)):
        value = float(spec)
    else:
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
    gop: bool
    deblock_map: DeblockMapConfig


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
            raw, ("weights", "strength", "quality", "quality_fallback",
                  "gop", *DEBLOCK_MAP_KEYS))
        strength = typed_value(raw, "strength", float, 1.0)
        if strength < 0.0:
            raise ValueError("strength must be >= 0")
        quality_fallback = typed_value(raw, "quality_fallback", float, 50.0)
        if not 1.0 <= quality_fallback <= 100.0:
            raise ValueError("quality_fallback must be in [1, 100]")
        return FbcnnStageConfig(
            weights_path=typed_value(raw, "weights", str)
            or settings.fbcnn_weights,
            quality=_parse_quality(raw.get("quality", "auto")),
            quality_fallback=quality_fallback,
            strength=strength,
            gop=typed_value(raw, "gop", bool, True),
            deblock_map=parse_deblock_map(raw),
        )

    def build(self, config: FbcnnStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            from kinovsr.processors.feed_driver import PerFrameDriver

            from . import FbcnnDeblocker

            return PerFrameDriver(FbcnnDeblocker(
                config.weights_path,
                quality=config.quality,
                strength=config.strength,
                quality_fallback=config.quality_fallback,
                gop=config.gop,
                blockiness_map=build_blockiness_tracker(config.deblock_map)))

        return FeedFlushProcessor(make_driver)


FACTORY = FbcnnFactory()
