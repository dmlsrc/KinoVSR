"""STDF's processor factory: a self-buffered 7-frame luma deblocker.

The deformable fusion runs on Y only; the RGB<->Y split needs the
stream's luma coefficients, so the build binds (Kr, Kb) from the input
StreamSpec's color matrix at prepare time. Blockiness-map conditioning
stays harness-wired until conditioning becomes stage config.
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
    StreamSpec,
    luma_coefficients,
)
from kinovsr.settings import Settings

_PROFILES = ("mfqev2", "vimeo90k")


@dataclasses.dataclass(frozen=True, slots=True)
class StdfStageConfig:
    weights_spec: str
    strength: float
    deblock_map: DeblockMapConfig


class StdfFactory:
    name = "stdf"

    capabilities = {
        Capability.DEBLOCK: CapabilitySpec(
            capability=Capability.DEBLOCK,
            profiles=_PROFILES,
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32, DType.FLOAT16),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            temporal_mode=TemporalMode.CENTERED,
            temporal_radius=3,
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
    ) -> StdfStageConfig:
        reject_unknown_keys(raw, ("weights", "strength", *DEBLOCK_MAP_KEYS))
        strength = typed_value(raw, "strength", float, 1.0)
        if strength < 0.0:
            raise ValueError("strength must be >= 0")
        return StdfStageConfig(
            weights_spec=typed_value(raw, "weights", str)
            or settings.stdf_weights or profile or "mfqev2",
            strength=strength,
            deblock_map=parse_deblock_map(raw),
        )

    def build(self, config: StdfStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        holder: dict[str, Any] = {}

        def make_driver() -> Any:
            from .deblocker import StdfDeblocker

            kr, kb = holder["coef"]
            return StdfDeblocker(
                config.weights_spec, strength=config.strength, kr=kr, kb=kb,
                blockiness_map=build_blockiness_tracker(config.deblock_map))

        class _Processor(FeedFlushProcessor):
            def prepare(self, input_spec: StreamSpec,
                        context: PipelineContext) -> None:
                holder["coef"] = luma_coefficients(input_spec.frame.color_matrix)
                super().prepare(input_spec, context)

        return _Processor(make_driver)


FACTORY = StdfFactory()
