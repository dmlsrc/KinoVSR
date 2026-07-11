"""TOFlow's processor factory: one family, three checkpoint-selected jobs.

The capability picks the checkpoint family: DENOISE and DEBLOCK run the
seven-frame TOFlowDenoiser on the matching converted .t7 (CENTERED,
radius 3), UPSCALE runs the released SR checkpoint (4x residual over a
bicubic base, same seven-frame window). Every checkpoint is a
safetensors + same-stem JSON graph pair; ``graph`` overrides the JSON.
The interp checkpoint stays API-only (no stage surface yet).
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

_DTYPES = {"float16", "float32"}
_FLOW_SCALES = ("full", "half", "quarter")
_SR_SCALE = 4

_ACCEPTS = StreamConstraint(
    layouts=(Layout.MLX_RGB_HWC,),
    dtypes=(DType.FLOAT32, DType.FLOAT16),
    domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
)


@dataclasses.dataclass(frozen=True, slots=True)
class ToflowStageConfig:
    capability: Capability
    weights_spec: str | None    # None = the capability's bundled token
    graph_path: str | None
    strength: float
    passes: int
    flow_scale: str
    dtype: str


def _produces_sr(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, ToflowStageConfig)
    frame = dataclasses.replace(
        spec.frame, geometry=spec.frame.geometry.scaled(_SR_SCALE))
    return dataclasses.replace(spec, frame=frame)


class ToflowFactory:
    name = "toflow"

    capabilities = {
        Capability.DENOISE: CapabilitySpec(
            capability=Capability.DENOISE,
            profiles=("denoise",),
            accepts=_ACCEPTS,
            temporal_mode=TemporalMode.CENTERED,
            temporal_radius=3,
            stateful=True,
        ),
        Capability.DEBLOCK: CapabilitySpec(
            capability=Capability.DEBLOCK,
            profiles=("deblock",),
            accepts=_ACCEPTS,
            temporal_mode=TemporalMode.CENTERED,
            temporal_radius=3,
            stateful=True,
        ),
        Capability.UPSCALE: CapabilitySpec(
            capability=Capability.UPSCALE,
            profiles=("sr",),
            accepts=_ACCEPTS,
            produces=_produces_sr,
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
    ) -> ToflowStageConfig:
        if capability is Capability.UPSCALE:
            reject_unknown_keys(raw, ("weights", "graph", "dtype"))
            weights = (typed_value(raw, "weights", str)
                       or settings.toflow_sr_weights)
            graph = typed_value(raw, "graph", str) or settings.toflow_sr_graph
            strength, passes, flow_scale = 1.0, 1, "full"
            dtype = typed_value(raw, "dtype", str, "float32")
        else:
            reject_unknown_keys(
                raw, ("weights", "graph", "strength", "passes", "flow_scale",
                      "dtype"))
            weights = (typed_value(raw, "weights", str)
                       or settings.toflow_weights)
            graph = typed_value(raw, "graph", str) or settings.toflow_graph
            strength = typed_value(raw, "strength", float, 1.0)
            if strength < 0.0:
                raise ValueError("strength must be >= 0")
            passes = typed_value(raw, "passes", int, 1)
            if passes < 1:
                raise ValueError("passes must be >= 1")
            flow_scale = typed_value(raw, "flow_scale", str, "full")
            if flow_scale not in _FLOW_SCALES:
                raise ValueError(f"flow_scale must be one of {_FLOW_SCALES}")
            dtype = typed_value(raw, "dtype", str, "float32")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        return ToflowStageConfig(
            capability=capability,
            weights_spec=weights or profile,
            graph_path=graph,
            strength=strength,
            passes=passes,
            flow_scale=flow_scale,
            dtype=dtype,
        )

    def build(self, config: ToflowStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            import mlx.core as mx

            dtype = mx.float16 if config.dtype == "float16" else mx.float32
            if config.capability is Capability.UPSCALE:
                from . import TOFlowSrUpscaler

                return TOFlowSrUpscaler(
                    config.weights_spec, graph=config.graph_path, dtype=dtype)
            from . import TOFlowDenoiser

            variant = ("deblock" if config.capability is Capability.DEBLOCK
                       else "denoise")
            return TOFlowDenoiser(
                config.weights_spec,
                variant=variant,
                flow_scale=config.flow_scale,
                passes=config.passes,
                graph=config.graph_path,
                strength=config.strength,
                dtype=dtype)

        return FeedFlushProcessor(make_driver)


FACTORY = ToflowFactory()
