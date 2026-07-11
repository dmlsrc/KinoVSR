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
from kinovsr.processors.feed_driver import FeedFlushProcessor
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    ColorMatrix,
    Domain,
    DType,
    Layout,
    StreamConstraint,
    StreamSpec,
)
from kinovsr.settings import Settings

_PROFILES = ("mfqev2", "vimeo90k")

# ITU-R (Kr, Kb) luma coefficients per stream color matrix.
_LUMA_COEF = {
    ColorMatrix.BT601: (0.299, 0.114),
    ColorMatrix.BT709: (0.2126, 0.0722),
    ColorMatrix.BT2020: (0.2627, 0.0593),
}


@dataclasses.dataclass(frozen=True, slots=True)
class StdfStageConfig:
    weights_spec: str
    strength: float


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
        reject_unknown_keys(raw, ("weights", "strength"))
        strength = typed_value(raw, "strength", float, 1.0)
        if strength < 0.0:
            raise ValueError("strength must be >= 0")
        return StdfStageConfig(
            weights_spec=typed_value(raw, "weights", str)
            or settings.stdf_weights or profile or "mfqev2",
            strength=strength,
        )

    def build(self, config: StdfStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        holder: dict[str, Any] = {}

        def make_driver() -> Any:
            from .deblocker import StdfDeblocker

            kr, kb = holder["coef"]
            return StdfDeblocker(config.weights_spec,
                                 strength=config.strength, kr=kr, kb=kb)

        class _Processor(FeedFlushProcessor):
            def prepare(self, input_spec: StreamSpec,
                        context: PipelineContext) -> None:
                holder["coef"] = _LUMA_COEF[input_spec.frame.color_matrix]
                super().prepare(input_spec, context)

        return _Processor(make_driver)


FACTORY = StdfFactory()
