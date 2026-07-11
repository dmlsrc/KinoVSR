"""RealBasicVSR's processor factory: a windowed real-world 4x upscaler.

Iterative cleaning then BasicVSR propagation over sliding windows whose
future half is self-buffered and paid as output delay (CENTERED, radius
= the default window). The clean_*/residual/flow dials are the
documented artifact mitigations.
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
    StreamSpec,
)
from kinovsr.settings import Settings

_PROFILES = ("x4",)
_FLOWS = ("spynet", "zero", "vt")
_GATES = ("off", "improve")
_SCALE = 4


@dataclasses.dataclass(frozen=True, slots=True)
class RealBasicVsrStageConfig:
    weights_spec: str
    window: int
    trim: int
    clean_threshold: float
    clean_iters: int
    residual_strength: float
    flow_consistency: float
    flow: str
    history_strength: float
    history_gate: str


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, RealBasicVsrStageConfig)
    frame = dataclasses.replace(
        spec.frame, geometry=spec.frame.geometry.scaled(_SCALE))
    return dataclasses.replace(spec, frame=frame)


class RealBasicVsrFactory:
    name = "realbasicvsr"

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
            temporal_mode=TemporalMode.CENTERED,
            temporal_radius=14,
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
    ) -> RealBasicVsrStageConfig:
        reject_unknown_keys(
            raw, ("weights", "window", "trim", "clean_threshold",
                  "clean_iters", "residual_strength", "flow_consistency",
                  "flow", "history_strength", "history_gate"))
        window = typed_value(raw, "window", int, 14)
        if window < 1:
            raise ValueError("window must be >= 1")
        trim = typed_value(raw, "trim", int, 0)
        if trim < 0:
            raise ValueError("trim must be >= 0")
        if trim and window <= 2 * trim:
            raise ValueError(
                "window must be greater than 2*trim so each window can "
                "emit interior frames (trim = 0 for reference-like "
                "non-overlapping chunks)")
        clean_iters = typed_value(raw, "clean_iters", int, 3)
        if clean_iters < 0:
            raise ValueError("clean_iters must be >= 0")
        flow_consistency = typed_value(raw, "flow_consistency", float, 0.0)
        if not 0.0 <= flow_consistency <= 1.0:
            raise ValueError("flow_consistency must be in [0, 1]")
        flow = typed_value(raw, "flow", str, "spynet")
        if flow not in _FLOWS:
            raise ValueError(f"flow must be one of {_FLOWS}")
        gate = typed_value(raw, "history_gate", str, "off")
        if gate not in _GATES:
            raise ValueError(f"history_gate must be one of {_GATES}")
        history_strength = typed_value(raw, "history_strength", float, 1.0)
        if history_strength < 0.0:
            raise ValueError("history_strength must be >= 0")
        return RealBasicVsrStageConfig(
            weights_spec=typed_value(raw, "weights", str)
            or settings.realbasicvsr_weights or profile or "x4",
            window=window,
            trim=trim,
            clean_threshold=typed_value(raw, "clean_threshold", float, 5.0),
            clean_iters=clean_iters,
            residual_strength=typed_value(raw, "residual_strength", float, 1.0),
            flow_consistency=flow_consistency,
            flow=flow,
            history_strength=history_strength,
            history_gate=gate,
        )

    def build(self, config: RealBasicVsrStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            from .upscaler import RealBasicVsrUpscaler

            return RealBasicVsrUpscaler(
                config.weights_spec,
                window=config.window,
                trim=config.trim,
                dynamic_refine_thres=config.clean_threshold,
                clean_iters=config.clean_iters,
                residual_strength=config.residual_strength,
                flow_consistency=config.flow_consistency,
                flow_mode=config.flow,
                history_strength=config.history_strength,
                history_gate=config.history_gate)

        return FeedFlushProcessor(make_driver)


FACTORY = RealBasicVsrFactory()
