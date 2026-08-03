"""BSVD's processor factory: the stateful streaming (causal) prover.

The network carries a 16-step bidirectional-buffer delay; the driver's
token plumbing pairs each delayed output with the input it was computed
from, so the FeedFlush adapter emits units on their true timestamps.
Noise-map conditioning is typed stage config (M6): the shared
kinovsr.processors.conditioning helper turns the noise_map* keys into the
tracker/pulse the engine consumes.
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
    NOISE_MAP_KEYS,
    NoiseMapConfig,
    build_conditioning,
    parse_noise_map,
)
from kinovsr.processors.feed_driver import (
    LUMA_CHROMA_KEYS,
    FeedFlushProcessor,
    parse_luma_chroma,
)
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
_BACKENDS = {"mlx", "ane", "mpsgraph"}


@dataclasses.dataclass(frozen=True, slots=True)
class BsvdStageConfig:
    weights_path: str | None    # explicit path; None = the variant default
    variant: str
    strength: float
    dtype: str
    backend: str
    luma_strength: float
    chroma_strength: float
    noise_map: NoiseMapConfig


class BsvdFactory:
    name = "bsvd"
    execution_affinity = "mlx"
    execution_native_slots = 4

    @staticmethod
    def execution_resource_handoff_seconds(config: BsvdStageConfig) -> float:
        """Yield scarce SoC resources only while a downstream user is live.

        Back-to-back recurrent submissions make opaque VideoToolbox work
        substantially slower even when BSVD's realized placement remains ANE.
        MPSGraph's four-step jobs need a proportionally larger handoff than
        Core ML's single-step dispatches. The runtime activates this metadata
        only under downstream pressure, leaving isolated throughput unchanged.
        """
        return {
            "ane": 0.010,
            "mpsgraph": 0.040,
        }.get(config.backend, 0.0)

    @staticmethod
    def execution_resources(config: BsvdStageConfig) -> tuple[str, ...]:
        if config.backend in {"ane", "mpsgraph"}:
            return ("gpu", "ane", "memory_bandwidth")
        return ("gpu", "memory_bandwidth")

    @staticmethod
    def execution_buffering(
        config: BsvdStageConfig,
        *,
        input_spec: Any,
        output_spec: Any,
        context: PipelineContext,
        default: Any,
    ) -> Any:
        """Declare BSVD's delay, window wavefront, and native slot budget."""
        from kinovsr.pipeline.execution import BufferingSpec, estimate_frame_bytes

        warmup = 9 if config.noise_map.mode == "auto" else 0
        if context.gop is None:
            retained = 16 + warmup + 1
            pending = 2
        else:
            # One accelerator window can run while the next reactive window
            # buffers. The completed window's bounded emissions are a separate
            # egress charge rather than a third retained input copy.
            retained = 2 * (context.gop.max_window + 2) + warmup
            pending = context.gop.max_window + 2
        native_slots = 4 if config.backend in {"ane", "mpsgraph"} else 0
        frame_bytes = max(
            estimate_frame_bytes(input_spec),
            estimate_frame_bytes(output_spec),
        )
        # The recurrent feature/skip state is opaque to the transport layer;
        # charge two frame-equivalent slabs per native slot conservatively.
        estimated = frame_bytes * (
            retained + pending + 2 * native_slots
        )
        return BufferingSpec(
            retained_input_units=retained,
            pending_output_units=pending,
            native_slots=native_slots,
            estimated_bytes=estimated,
        )

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
        reject_unknown_keys(
            raw, ("weights", "strength", "dtype", "backend",
                  *LUMA_CHROMA_KEYS, *NOISE_MAP_KEYS))
        strength = typed_value(raw, "strength", float, 0.5)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        dtype = typed_value(raw, "dtype", str, "float16")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        backend = (typed_value(raw, "backend", str)
                   or settings.bsvd_backend or "mlx")
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {sorted(_BACKENDS)}")
        luma_strength, chroma_strength = parse_luma_chroma(raw)
        return BsvdStageConfig(
            weights_path=typed_value(raw, "weights", str)
            or settings.bsvd_weights,
            variant=profile or "c64",
            strength=strength,
            dtype=dtype,
            backend=backend,
            luma_strength=luma_strength,
            chroma_strength=chroma_strength,
            noise_map=parse_noise_map(raw),
        )

    def build(self, config: BsvdStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            import mlx.core as mx

            from . import BsvdDenoiser

            dtype = mx.float16 if config.dtype == "float16" else mx.float32
            tracker, pulse = build_conditioning(config.noise_map)
            return BsvdDenoiser(
                config.weights_path, variant=config.variant,
                strength=config.strength, dtype=dtype,
                noise_map=tracker, map_refresh=config.noise_map.refresh,
                pulse=pulse, map_floor=config.noise_map.floor,
                backend=config.backend)

        return FeedFlushProcessor(
            make_driver,
            luma_strength=config.luma_strength,
            chroma_strength=config.chroma_strength)


FACTORY = BsvdFactory()
