"""RealPLKSR's processor factory: per-frame restoration and upscaling.

Named profiles resolve from the family
manifest: each profile's ``defaults`` declares its scale, so the
``produces`` geometry transform stays pure (no checkpoint I/O at resolve
time). An explicit ``weights`` path must state ``scale`` unless it is
one of the known tokens.
"""

import dataclasses
import functools
from collections.abc import Mapping
from typing import ClassVar

import mlx.core as mx

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.modeling.weights import load_registered
from kinovsr.processors.capabilities import Capability, CapabilitySpec
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

from .upscaler import RealPlksrUpscaler

_UPSCALE_PROFILES = ("public2x", "public2x-nn", "nomos4x")
_RESTORE_PROFILES = ("deh264",)
_DEFAULT_UPSCALE_PROFILE = "public2x"
_DEFAULT_RESTORE_PROFILE = "deh264"
_DTYPES = {"float16", "float32"}
_ACCEPTS = StreamConstraint(
    layouts=(Layout.MLX_RGB_HWC,),
    dtypes=(DType.FLOAT32, DType.FLOAT16),
    domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
)


@functools.cache
def _profile_scales() -> dict[str, int]:
    """Profile -> scale, from the family manifest (loaded once)."""

    manifest = load_registered("realplksr")
    return {name: int(profile.defaults["scale"]) for name, profile in manifest.profiles.items()}


@dataclasses.dataclass(frozen=True, slots=True)
class RealPlksrStageConfig:
    weights_spec: str  # token or path handed to net.resolve_weights
    scale: int
    dtype: str
    strength: float = 1.0


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, RealPlksrStageConfig)
    frame = dataclasses.replace(spec.frame, geometry=spec.frame.geometry.scaled(config.scale))
    return dataclasses.replace(spec, frame=frame)


class RealPlksrFactory:
    name = "realplksr"

    capabilities: ClassVar[dict[Capability, CapabilitySpec]] = {
        Capability.UPSCALE: CapabilitySpec(
            capability=Capability.UPSCALE,
            profiles=_UPSCALE_PROFILES,
            accepts=_ACCEPTS,
            produces=_produces,
        ),
        Capability.RESTORE: CapabilitySpec(
            capability=Capability.RESTORE,
            profiles=_RESTORE_PROFILES,
            accepts=_ACCEPTS,
        ),
    }

    def profile_defaults(self, *, capability: Capability, profile: str) -> Mapping[str, object]:  # noqa: ARG002 - hook signature

        return load_registered(self.name).profiles[profile].defaults

    def parse_config(
        self,
        raw: Mapping[str, object],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> RealPlksrStageConfig:
        restore = capability is Capability.RESTORE
        keys = (
            ("weights", "scale", "dtype", "strength") if restore else ("weights", "scale", "dtype")
        )
        reject_unknown_keys(raw, keys)
        dtype = typed_value(raw, "dtype", str, "float16")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        strength = typed_value(raw, "strength", float, 1.0) if restore else 1.0
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        weights = typed_value(raw, "weights", str) or settings.realplksr_weights
        scale = typed_value(raw, "scale", int)
        scales = _profile_scales()
        token = profile or (weights if weights in scales else None)
        default_profile = _DEFAULT_RESTORE_PROFILE if restore else _DEFAULT_UPSCALE_PROFILE
        if scale is None:
            if token is None and weights is not None:
                raise ValueError(
                    "state scale when weights is an explicit path (profiles declare it)"
                )
            scale = scales[token or default_profile]
        valid_scales = (1,) if restore else (2, 4)
        if scale not in valid_scales:
            expected = "1" if restore else "2 or 4"
            raise ValueError(f"scale must be {expected} for {capability.value}")
        return RealPlksrStageConfig(
            weights_spec=weights or profile or default_profile,
            scale=scale,
            dtype=dtype,
            strength=strength,
        )

    def build(
        self,
        config: RealPlksrStageConfig,
        *,
        context: PipelineContext,  # noqa: ARG002 - protocol signature
    ) -> FeedFlushProcessor:
        def make_driver() -> RealPlksrUpscaler:

            dtype = mx.float16 if config.dtype == "float16" else mx.float32
            driver = RealPlksrUpscaler(config.weights_spec, dtype=dtype, strength=config.strength)
            if driver.scale != config.scale:
                raise ValueError(
                    f"checkpoint scale {driver.scale}x does not match the "
                    f"declared scale {config.scale}x"
                )
            return driver

        return FeedFlushProcessor(make_driver)


FACTORY = RealPlksrFactory()
