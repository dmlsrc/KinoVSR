"""BasicVSR++'s processor factory: windowed 4x upscale and 1x restore.

One family, two capabilities selected by checkpoint class: UPSCALE runs
the SR checkpoints (geometry x4), RESTORE runs the NTIRE decompress /
denoise / deblur checkpoints at 1x with a dry/wet strength. Both are
bidirectional over sliding windows whose future half is self-buffered
and paid as output delay (CENTERED, radius = the default window).
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

_SR_PROFILES = ("reds4", "vimeo90k_bi", "vimeo90k_bd", "ntire_vsr")
_SR_DEFAULT = "vimeo90k_bd"
_RESTORE_PROFILES = ("decompress_track1", "decompress_track2",
                     "decompress_track3", "denoise", "deblur_dvd",
                     "deblur_gopro")
_RESTORE_DEFAULT = "decompress_track1"
_FLOWS = ("spynet", "zero", "vt", "vision")
_GATES = ("off", "improve")
_SCALE = 4

_ACCEPTS = StreamConstraint(
    layouts=(Layout.MLX_RGB_HWC,),
    dtypes=(DType.FLOAT32, DType.FLOAT16),
    domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
)


@dataclasses.dataclass(frozen=True, slots=True)
class BasicVsrppStageConfig:
    capability: Capability
    weights_spec: str
    window: int
    trim: int
    strength: float             # restore only; SR has no dry/wet dial
    flow: str
    history_strength: float
    history_gate: str
    ensemble: bool


def _produces_sr(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, BasicVsrppStageConfig)
    frame = dataclasses.replace(
        spec.frame, geometry=spec.frame.geometry.scaled(_SCALE))
    return dataclasses.replace(spec, frame=frame)


class BasicVsrppFactory:
    name = "basicvsrpp"

    capabilities = {
        Capability.UPSCALE: CapabilitySpec(
            capability=Capability.UPSCALE,
            profiles=_SR_PROFILES,
            accepts=_ACCEPTS,
            produces=_produces_sr,
            temporal_mode=TemporalMode.CENTERED,
            temporal_radius=14,
            stateful=True,
        ),
        Capability.RESTORE: CapabilitySpec(
            capability=Capability.RESTORE,
            profiles=_RESTORE_PROFILES,
            accepts=_ACCEPTS,
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
    ) -> BasicVsrppStageConfig:
        restore = capability is Capability.RESTORE
        keys = ["weights", "window", "trim", "flow", "ensemble"]
        if restore:
            keys.append("strength")
        else:
            keys += ["history_strength", "history_gate"]
        reject_unknown_keys(raw, tuple(keys))
        window = typed_value(raw, "window", int, 14)
        if window < 1:
            raise ValueError("window must be >= 1")
        trim = typed_value(raw, "trim", int, 2)
        if trim < 0:
            raise ValueError("trim must be >= 0")
        if trim and window <= 2 * trim:
            # The runtime constructor inflates window to max(window, 2*trim+1);
            # reject the contradiction at open instead of silently buffering a
            # huge window mid-stream (matches realbasicvsr/pvdd).
            raise ValueError(
                "window must be greater than 2*trim so each window keeps a "
                f"processed interior after trimming {trim} warm-up frames per "
                f"edge; got window={window}, trim={trim}")
        flow = typed_value(raw, "flow", str, "spynet")
        if flow not in _FLOWS:
            raise ValueError(f"flow must be one of {_FLOWS}")
        gate = typed_value(raw, "history_gate", str, "off") if not restore else "off"
        if gate not in _GATES:
            raise ValueError(f"history_gate must be one of {_GATES}")
        strength = typed_value(raw, "strength", float, 1.0) if restore else 1.0
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        history_strength = (typed_value(raw, "history_strength", float, 1.0)
                            if not restore else 1.0)
        if history_strength < 0.0:
            raise ValueError("history_strength must be >= 0")
        if restore:
            weights = (typed_value(raw, "weights", str)
                       or settings.restore_weights
                       or profile or _RESTORE_DEFAULT)
        else:
            weights = (typed_value(raw, "weights", str)
                       or settings.basicvsrpp_weights
                       or profile or _SR_DEFAULT)
        return BasicVsrppStageConfig(
            capability=capability,
            weights_spec=weights,
            window=window,
            trim=trim,
            strength=strength,
            flow=flow,
            history_strength=history_strength,
            history_gate=gate,
            ensemble=typed_value(raw, "ensemble", bool, False),
        )

    def build(self, config: BasicVsrppStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            if config.capability is Capability.RESTORE:
                from .restorer import BasicVsrRestorer

                return BasicVsrRestorer(
                    config.weights_spec,
                    window=config.window,
                    trim=config.trim,
                    strength=config.strength,
                    flow_mode=config.flow,
                    ensemble=config.ensemble)
            from .upscaler import BasicVsrUpscaler

            return BasicVsrUpscaler(
                config.weights_spec,
                window=config.window,
                trim=config.trim,
                flow_mode=config.flow,
                history_strength=config.history_strength,
                history_gate=config.history_gate,
                ensemble=config.ensemble)

        return FeedFlushProcessor(make_driver)


FACTORY = BasicVsrppFactory()
