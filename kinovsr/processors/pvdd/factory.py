"""PVDD's processor factory: a windowed bidirectional real-noise denoiser.

Runs in sliding windows (default 10) whose future half is self-buffered
and paid as output delay (CENTERED). The level variants take a noise
dial: ``noise_preset`` S/M/L maps to the reference variance levels, and
``noise_variance`` overrides it exactly (sigma^2, not sigma); blind
variants ignore both. Noise-map conditioning stays harness-wired until
conditioning becomes stage config.
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
    NOISE_MAP_TRACKER_KEYS,
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

# The raw variants (pvdd_raw, pvdd_raw_level) expect packed-Bayer input
# (num_in=4); no accepted layout can feed them, so they stay manifest
# documented but off the runtime surface until a raw layout exists.
_PROFILES = ("pvdd", "crvd", "davis", "pvdd_level")
_DTYPES = {"float16", "float32"}
_PRESETS = ("off", "S", "M", "L")


@dataclasses.dataclass(frozen=True, slots=True)
class PvddStageConfig:
    weights_path: str | None
    variant: str
    window: int
    trim: int
    noise_variance: float | None
    dtype: str
    noise_map: NoiseMapConfig
    luma_strength: float
    chroma_strength: float


class PvddFactory:
    name = "pvdd"

    capabilities = {
        Capability.DENOISE: CapabilitySpec(
            capability=Capability.DENOISE,
            profiles=_PROFILES,
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32, DType.FLOAT16),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            temporal_mode=TemporalMode.CENTERED,
            temporal_radius=10,
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
    ) -> PvddStageConfig:
        reject_unknown_keys(
            raw, ("weights", "window", "trim", "noise_preset",
                  "noise_variance", "dtype", *LUMA_CHROMA_KEYS,
                  *NOISE_MAP_TRACKER_KEYS))
        window = typed_value(raw, "window", int, 10)
        if window < 2:
            raise ValueError("window must be >= 2")
        trim = typed_value(raw, "trim", int, 0)
        if not 0 <= trim < window / 2:
            raise ValueError("trim must be in [0, window/2)")
        dtype = typed_value(raw, "dtype", str, "float16")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        preset = typed_value(raw, "noise_preset", str, "M")
        if preset not in _PRESETS:
            raise ValueError(f"noise_preset must be one of {_PRESETS}")
        noise_variance = typed_value(raw, "noise_variance", float)
        if noise_variance is None and preset != "off":
            from . import LEVEL_PRESETS

            noise_variance = LEVEL_PRESETS[preset]
        variant = profile or "pvdd"
        noise_map = parse_noise_map(raw)
        # Conditioning needs a level (non-blind) checkpoint; the blind
        # variants take no map input. The engine reconfirms from the loaded
        # checkpoint's is_level, but a known-blind profile is knowable now.
        if (noise_map.mode == "auto" or noise_map.pulse) \
                and "level" not in variant:
            raise ValueError(
                f"noise-map conditioning needs a level PVDD variant; "
                f"{variant!r} is blind (use pvdd_level)")
        luma_strength, chroma_strength = parse_luma_chroma(raw)
        return PvddStageConfig(
            weights_path=typed_value(raw, "weights", str)
            or settings.pvdd_weights,
            variant=variant,
            window=window,
            trim=trim,
            noise_variance=noise_variance,
            dtype=dtype,
            noise_map=noise_map,
            luma_strength=luma_strength,
            chroma_strength=chroma_strength,
        )

    def build(self, config: PvddStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> Any:
            import mlx.core as mx

            from .upscaler import PvddDenoiser

            dtype = mx.float16 if config.dtype == "float16" else mx.float32
            tracker, pulse = build_conditioning(config.noise_map)
            return PvddDenoiser(
                config.weights_path,
                variant=config.variant,
                window=config.window,
                trim=config.trim,
                noise_variance=config.noise_variance,
                dtype=dtype,
                noise_map=tracker, pulse=pulse)

        return FeedFlushProcessor(
            make_driver,
            luma_strength=config.luma_strength,
            chroma_strength=config.chroma_strength)


FACTORY = PvddFactory()
