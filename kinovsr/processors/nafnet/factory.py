"""NAFNet's processor factory: per-frame restoration with a guard state.

The capability selects the checkpoint class - DEBLUR (gopro/gopro32),
DENOISE (sidd/sidd32), RESTORE (reds) - all single-image nets; the
out-of-domain guard keeps rolling lockout state across frames, so the
stage is CAUSAL stateful even though the net itself is per-frame.
``guard_fall`` may be omitted (None derives from the ramp).
"""

from __future__ import annotations

import dataclasses
import logging
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

_log = logging.getLogger(__name__)

_POOLS = ("auto", "local", "global")
_GUARDS = ("auto", "off", "residual", "control", "control-source", "fast",
           "reject")

_ACCEPTS = StreamConstraint(
    layouts=(Layout.MLX_RGB_HWC,),
    dtypes=(DType.FLOAT32, DType.FLOAT16),
    domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
)

_CAPABILITY_PROFILES = {
    Capability.DEBLUR: ("gopro", "gopro32"),
    Capability.DENOISE: ("sidd", "sidd32"),
    Capability.RESTORE: ("reds",),
}


def _spec_for(capability: Capability) -> CapabilitySpec:
    return CapabilitySpec(
        capability=capability,
        profiles=_CAPABILITY_PROFILES[capability],
        accepts=_ACCEPTS,
        temporal_mode=TemporalMode.CAUSAL,
        temporal_radius=1,
        stateful=True,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class NafnetStageConfig:
    weights_spec: str
    variant: str
    strength: float
    pool: str
    guard: str
    guard_threshold: float
    guard_fast_fraction: float
    guard_lockout: int
    guard_ramp: int
    guard_fall: int | None


class NafnetFactory:
    name = "nafnet"

    capabilities = {
        Capability.DEBLUR: _spec_for(Capability.DEBLUR),
        Capability.DENOISE: _spec_for(Capability.DENOISE),
        Capability.RESTORE: _spec_for(Capability.RESTORE),
    }

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> NafnetStageConfig:
        reject_unknown_keys(
            raw, ("weights", "strength", "pool", "guard", "guard_threshold",
                  "guard_fast_fraction", "guard_lockout", "guard_ramp",
                  "guard_fall"))
        variant = profile or _CAPABILITY_PROFILES[capability][0]
        strength = typed_value(raw, "strength", float, 1.0)
        if strength < 0.0:
            raise ValueError("strength must be >= 0")
        pool = typed_value(raw, "pool", str, "auto")
        if pool not in _POOLS:
            raise ValueError(f"pool must be one of {_POOLS}")
        guard = typed_value(raw, "guard", str, "auto")
        if guard not in _GUARDS:
            raise ValueError(f"guard must be one of {_GUARDS}")
        guard_lockout = typed_value(raw, "guard_lockout", int, 48)
        if guard_lockout < 0:
            raise ValueError("guard_lockout must be >= 0")
        guard_ramp = typed_value(raw, "guard_ramp", int, 12)
        if guard_ramp < 0:
            raise ValueError("guard_ramp must be >= 0")
        guard_fall = typed_value(raw, "guard_fall", int)
        if guard_fall is not None and guard_fall < 0:
            raise ValueError("guard_fall must be >= 0")
        return NafnetStageConfig(
            weights_spec=typed_value(raw, "weights", str)
            or settings.nafnet_weights or variant,
            variant=variant,
            strength=strength,
            pool=pool,
            guard=guard,
            guard_threshold=typed_value(raw, "guard_threshold", float, 0.12),
            guard_fast_fraction=typed_value(
                raw, "guard_fast_fraction", float, 0.85),
            guard_lockout=guard_lockout,
            guard_ramp=guard_ramp,
            guard_fall=guard_fall,
        )

    def build(self, config: NafnetStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            from kinovsr.processors.feed_driver import PerFrameDriver

            from . import NafnetRestorer

            return PerFrameDriver(NafnetRestorer(
                config.weights_spec,
                strength=config.strength,
                pool_mode=config.pool,
                variant=config.variant,
                guard_mode=config.guard,
                residual_guard=config.guard_threshold,
                guard_fast_fraction=config.guard_fast_fraction,
                guard_lockout_frames=config.guard_lockout,
                guard_ramp_frames=config.guard_ramp,
                guard_fall_frames=config.guard_fall,
                progress_message=_log.warning))

        return FeedFlushProcessor(make_driver)


FACTORY = NafnetFactory()
