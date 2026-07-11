"""BSVD's processor factory: the stateful streaming (causal) prover.

The network carries a 16-step bidirectional-buffer delay; the driver's
token plumbing pairs each delayed output with the input it was computed
from, so the FeedFlush adapter emits units on their true timestamps.
Noise-map conditioning stays harness-level until the family migration
(M4) brings the tracker along.
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
    Domain,
    DType,
    Layout,
    StreamConstraint,
)
from kinovsr.settings import Settings

_PROFILES = ("c64", "c32")
_DTYPES = {"float16", "float32"}


@dataclasses.dataclass(frozen=True, slots=True)
class BsvdStageConfig:
    weights_path: str | None    # explicit path; None = the variant default
    variant: str
    strength: float
    dtype: str


class BsvdFactory:
    name = "bsvd"

    capabilities = {
        Capability.DENOISE: CapabilitySpec(
            capability=Capability.DENOISE,
            profiles=_PROFILES,
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32, DType.FLOAT16),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            # Bidirectional within its own buffer: future context is
            # self-buffered and paid as the 16-frame output delay.
            temporal_mode=TemporalMode.CENTERED,
            temporal_radius=16,
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
    ) -> BsvdStageConfig:
        reject_unknown_keys(raw, ("weights", "strength", "dtype"))
        strength = typed_value(raw, "strength", float, 0.5)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        dtype = typed_value(raw, "dtype", str, "float16")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        return BsvdStageConfig(
            weights_path=typed_value(raw, "weights", str)
            or settings.bsvd_weights,
            variant=profile or "c64",
            strength=strength,
            dtype=dtype,
        )

    def build(self, config: BsvdStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            import mlx.core as mx

            from . import BsvdDenoiser

            dtype = mx.float16 if config.dtype == "float16" else mx.float32
            return BsvdDenoiser(
                config.weights_path, variant=config.variant,
                strength=config.strength, dtype=dtype)

        return FeedFlushProcessor(make_driver)


FACTORY = BsvdFactory()
