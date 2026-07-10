"""RealPLKSR's processor factory: the stateless per-frame prover.

Profiles are the existing product tokens and resolve from the family
manifest: each profile's ``defaults`` declares its scale, so the
``produces`` geometry transform stays pure (no checkpoint I/O at resolve
time). An explicit ``weights`` path must state ``scale`` unless it is
one of the known tokens.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys, typed_value
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

_PROFILES = ("public2x", "public2x-nn", "nomos4x")
_DEFAULT_PROFILE = "public2x"
_DTYPES = {"float16", "float32"}


@functools.cache
def _profile_scales() -> dict[str, int]:
    """Profile -> scale, from the family manifest (loaded once)."""
    from kinovsr.modeling.weights import load_registered

    manifest = load_registered("realplksr")
    return {name: int(profile.defaults["scale"])
            for name, profile in manifest.profiles.items()}


@dataclasses.dataclass(frozen=True, slots=True)
class RealPlksrStageConfig:
    weights_spec: str          # token or path handed to net.resolve_weights
    scale: int
    dtype: str


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, RealPlksrStageConfig)
    frame = dataclasses.replace(
        spec.frame, geometry=spec.frame.geometry.scaled(config.scale))
    return dataclasses.replace(spec, frame=frame)


class RealPlksrFactory:
    name = "realplksr"

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
        ),
    }

    def profile_defaults(self, *, capability: Capability,
                         profile: str) -> Mapping[str, Any]:
        from kinovsr.modeling.weights import load_registered

        return load_registered(self.name).profiles[profile].defaults

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> RealPlksrStageConfig:
        reject_unknown_keys(raw, ("weights", "scale", "dtype"))
        dtype = typed_value(raw, "dtype", str, "float16")
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        weights = typed_value(raw, "weights", str) or settings.realplksr_weights
        scale = typed_value(raw, "scale", int)
        scales = _profile_scales()
        token = profile or (weights if weights in scales else None)
        if scale is None:
            if token is None and weights is not None:
                raise ValueError(
                    "state scale = 2 or 4 when weights is an explicit "
                    "path (profiles declare it)")
            scale = scales[token or _DEFAULT_PROFILE]
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        return RealPlksrStageConfig(
            weights_spec=weights or profile or _DEFAULT_PROFILE,
            scale=scale, dtype=dtype)

    def build(self, config: RealPlksrStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver():
            import mlx.core as mx

            from .upscaler import RealPlksrUpscaler

            dtype = mx.float16 if config.dtype == "float16" else mx.float32
            driver = RealPlksrUpscaler(config.weights_spec, dtype=dtype)
            if driver.scale != config.scale:
                raise ValueError(
                    f"checkpoint scale {driver.scale}x does not match the "
                    f"declared scale {config.scale}x")
            return driver

        return FeedFlushProcessor(make_driver)


FACTORY = RealPlksrFactory()
