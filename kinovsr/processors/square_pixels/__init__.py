"""A square-pixel geometry preprocess family (anamorphic neutralization).

Physically resamples the width (Lanczos-3, at source resolution) so an
anamorphic source becomes 1:1, then tags the output pixel aspect 1:1.
The target width is a pure function of the input geometry - width scaled
by the pixel aspect, rounded to even - so ``produces`` computes the
output spec at resolve time and the builder validates the rest of the
chain against it. Parameterless: there is nothing to state but "make the
pixels square", so on an already-square source the stage is a clean
no-op (it still tags 1:1, which is already the case).

This is the one geometry op that rewrites ``pixel_aspect``; the passive
PAR carry (``FileSource`` reads it, ``FileSink`` tags it) neutralizes
here. It runs AFTER the PAR-aware crops, matching the harness order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys
from kinovsr.modeling.vsr_blocks import make_lanczos_plan, resample_width
from kinovsr.processors.capabilities import Capability, CapabilitySpec
from kinovsr.processors.feed_driver import FeedFlushProcessor
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Domain,
    DType,
    Geometry,
    Layout,
    StreamConstraint,
    StreamSpec,
)
from kinovsr.settings import Settings


@dataclasses.dataclass(frozen=True, slots=True)
class SquarePixelsStageConfig:
    """No knobs: the op is fully determined by the input pixel aspect."""


def _square_width(geometry: Geometry) -> int:
    """The width that makes ``geometry``'s pixels 1:1.

    ``width * pixel_aspect`` rounded to an even luma width (the 4:2:x
    encoder needs it even). Returns the current width unchanged when the
    source is already square or the target degenerates below 2px, so the
    stage is a safe no-op on non-anamorphic sources.
    """
    par = geometry.pixel_aspect
    if par == 1:
        return geometry.width
    sq_w = round(geometry.width * par)
    sq_w -= sq_w % 2
    if sq_w < 2:
        return geometry.width
    return sq_w


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, SquarePixelsStageConfig)
    geometry = spec.frame.geometry
    # Always tag 1:1, whether or not the width actually changed.
    frame = dataclasses.replace(
        spec.frame,
        geometry=Geometry(_square_width(geometry), geometry.height,
                          Fraction(1)))
    return dataclasses.replace(spec, frame=frame)


class _SquareDriver:
    def __init__(self, plan: tuple | None) -> None:
        self._plan = plan

    def feed(self, rgb: Any, token: Any = None) -> list:
        if self._plan is None:      # already square / degenerate: pass through
            return [(rgb, token)]
        return [(resample_width(rgb, self._plan), token)]

    def flush(self) -> list:
        return []

    def reset(self) -> None:
        pass


class SquarePixelsFactory:
    name = "square_pixels"

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
    ) -> SquarePixelsStageConfig:
        reject_unknown_keys(raw, ())
        return SquarePixelsStageConfig()

    def build(self, config: SquarePixelsStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        # The resample plan is a pure function of the validated input
        # geometry; the width it targets MUST equal what `produces`
        # declared, so both go through `_square_width`.
        holder: dict[str, Any] = {}

        class _Processor(FeedFlushProcessor):
            def prepare(self, input_spec: StreamSpec,
                        context: PipelineContext) -> None:
                geometry = input_spec.frame.geometry
                sq_w = _square_width(geometry)
                holder["plan"] = (
                    make_lanczos_plan(geometry.width, sq_w)
                    if sq_w != geometry.width else None)
                super().prepare(input_spec, context)

        return _Processor(lambda: _SquareDriver(holder["plan"]))


FACTORY = SquarePixelsFactory()

__all__ = ["FACTORY", "SquarePixelsFactory", "SquarePixelsStageConfig"]
