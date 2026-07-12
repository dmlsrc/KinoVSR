"""RealViformer's processor factory: a causal recurrent 4x upscaler.

Streams frame by frame with temporal state (CAUSAL, like mc's recursive
history): no source lookahead, resets at boundaries. ``window`` is the
reference inference's chunked state reset (0 = never reset), and the
history_* dials are the etch-mitigation policy documented on the flags.
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
_DTYPES = {"float16", "float32"}
_FLOWS = ("spynet", "zero", "vt")
_GATES = ("off", "improve", "holistic")
_SCALE = 4


@dataclasses.dataclass(frozen=True, slots=True)
class RealViformerStageConfig:
    weights_spec: str
    window: int
    dtype: str
    flow: str
    history_strength: float
    history_gate: str
    history_cleanup: float
    history_gate_drop: float
    history_risk_decay: float
    history_static_cap: float


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, RealViformerStageConfig)
    frame = dataclasses.replace(
        spec.frame, geometry=spec.frame.geometry.scaled(_SCALE))
    return dataclasses.replace(spec, frame=frame)


class RealViformerFactory:
    name = "realviformer"

    capabilities = {
        Capability.UPSCALE: CapabilitySpec(
            capability=Capability.UPSCALE,
            profiles=_PROFILES,
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32, DType.FLOAT16),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
                # _pad4 reflect-pads to a multiple of 4; reflect needs each
                # side strictly greater than its pad (up to 3), so sides < 3
                # crash on the first frame. Reject them at open instead.
                min_side=3,
            ),
            produces=_produces,
            temporal_mode=TemporalMode.CAUSAL,
            temporal_radius=1,
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
    ) -> RealViformerStageConfig:
        reject_unknown_keys(
            raw, ("weights", "window", "dtype", "flow", "history_strength",
                  "history_gate", "history_cleanup", "history_gate_drop",
                  "history_risk_decay", "history_static_cap"))
        window = typed_value(raw, "window", int, 100)
        if window < 0:
            raise ValueError("window must be >= 0 (0 = never reset)")
        dtype = typed_value(raw, "dtype", str, "float16")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        flow = typed_value(raw, "flow", str, "spynet")
        if flow not in _FLOWS:
            raise ValueError(f"flow must be one of {_FLOWS}")
        gate = typed_value(raw, "history_gate", str, "off")
        if gate not in _GATES:
            raise ValueError(f"history_gate must be one of {_GATES}")
        risk_decay = typed_value(raw, "history_risk_decay", float, 0.8)
        if not 0.0 <= risk_decay < 1.0:
            raise ValueError("history_risk_decay must be in [0, 1)")
        history_strength = typed_value(raw, "history_strength", float, 1.0)
        if history_strength < 0.0:
            raise ValueError("history_strength must be >= 0")
        cleanup = typed_value(raw, "history_cleanup", float, 0.25)
        if not 0.0 <= cleanup <= 1.0:
            raise ValueError("history_cleanup must be in [0, 1]")
        gate_drop = typed_value(raw, "history_gate_drop", float, 0.85)
        if not 0.0 <= gate_drop <= 1.0:
            raise ValueError("history_gate_drop must be in [0, 1]")
        static_cap = typed_value(raw, "history_static_cap", float, 0.0)
        if not 0.0 <= static_cap <= 1.0:
            raise ValueError("history_static_cap must be in [0, 1]")
        return RealViformerStageConfig(
            weights_spec=typed_value(raw, "weights", str)
            or settings.realviformer_weights or profile or "x4",
            window=window,
            dtype=dtype,
            flow=flow,
            history_strength=history_strength,
            history_gate=gate,
            history_cleanup=cleanup,
            history_gate_drop=gate_drop,
            history_risk_decay=risk_decay,
            history_static_cap=static_cap,
        )

    def build(self, config: RealViformerStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            import mlx.core as mx

            from .upscaler import RealViformerUpscaler

            dtype = mx.float16 if config.dtype == "float16" else mx.float32
            return RealViformerUpscaler(
                config.weights_spec,
                window=config.window,
                dtype=dtype,
                flow_mode=config.flow,
                history_strength=config.history_strength,
                history_gate=config.history_gate,
                history_cleanup=config.history_cleanup,
                history_gate_drop=config.history_gate_drop,
                history_risk_decay=config.history_risk_decay,
                history_static_cap=config.history_static_cap)

        return FeedFlushProcessor(make_driver)


FACTORY = RealViformerFactory()
