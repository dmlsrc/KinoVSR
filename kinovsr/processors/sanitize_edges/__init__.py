"""An edge-band replicate-fill preprocess family.

Thin factory over the shared edge_sanitize implementation: the declared
edge bands are overwritten with the adjacent interior row/column before
any learned stage sees the frame, so nets trained on photographic
content never see synthetic border structure (junk capture rows,
stabilization edges). Geometry is untouched.

The bands are declared, not detected - junk DETECTION is probe-time,
like crop's bar detection. Fill semantics: this family is the
``extend`` mode (replicated content reaches the output). The flat CLI's
``restore`` mode - composite the ORIGINAL border back over the
processed frame - spans the chain (a pre-stage here plus a post-stage
at output geometry) and stays harness-owned until the typed pipeline
can express that pairing.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.edge_sanitize import parse_edges_spec, sanitize_rgb
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


@dataclasses.dataclass(frozen=True, slots=True)
class SanitizeEdgesStageConfig:
    edges: tuple[int, int, int, int]    # (top, bottom, left, right)


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    # Geometry is known at resolve time: bands that consume an entire
    # axis fail preflight instead of after processing begins.
    assert isinstance(config, SanitizeEdgesStageConfig)
    top, bottom, left, right = config.edges
    geometry = spec.frame.geometry
    if top + bottom >= geometry.height or left + right >= geometry.width:
        raise ValueError(
            f"edge bands {config.edges} leave no interior in a "
            f"{geometry.width}x{geometry.height} stream")
    return spec


class _SanitizeDriver:
    def __init__(self, edges: tuple[int, int, int, int]) -> None:
        self._edges = edges

    def feed(self, rgb: Any, token: Any = None) -> list:
        return [(sanitize_rgb(rgb, self._edges), token)]

    def flush(self) -> list:
        return []

    def reset(self) -> None:
        pass


class SanitizeEdgesFactory:
    name = "sanitize_edges"

    capabilities = {
        Capability.PREPROCESS: CapabilitySpec(
            capability=Capability.PREPROCESS,
            profiles=(),
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32, DType.FLOAT16),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            produces=_produces,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> SanitizeEdgesStageConfig:
        reject_unknown_keys(raw, ("edges", "fill"))
        edges_spec = typed_value(raw, "edges", str)
        if not edges_spec:
            raise ValueError(
                "state edges as T,B,L,R (junk detection is probe-time; "
                "a sanitize stage with no bands is config noise)")
        edges = parse_edges_spec(edges_spec)
        if edges == (0, 0, 0, 0):
            raise ValueError("edges declare no bands")
        fill = typed_value(raw, "fill", str, "extend")
        if fill != "extend":
            raise ValueError(
                "fill must be 'extend' here; 'restore' pairs a pre-stage "
                "with a post-composite at output geometry and stays "
                "harness-owned, 'trim' is the crop family's job")
        return SanitizeEdgesStageConfig(edges=edges)

    def build(self, config: SanitizeEdgesStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        return FeedFlushProcessor(lambda: _SanitizeDriver(config.edges))


FACTORY = SanitizeEdgesFactory()

__all__ = ["FACTORY", "SanitizeEdgesFactory", "SanitizeEdgesStageConfig"]
