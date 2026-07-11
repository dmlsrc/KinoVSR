"""VideoToolbox as a processor family: the native-session prover.

M3 exposes the interpolate capability (VTFrameRateConversion). It is the
capability that exercises everything the typed pipeline exists for: a
native session with real lifecycle, one-to-many unit emission, a
rewritten cadence with regenerated timestamps, and preserved clip
duration - which is what keeps copied audio synchronized. Spatial VT
scaling migrates here in M4.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys
from kinovsr.processors.boundaries import Boundary
from kinovsr.processors.capabilities import Capability, CapabilitySpec
from kinovsr.processors.errors import MediaError
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Cardinality,
    Layout,
    StreamConstraint,
    StreamSpec,
    TimestampPolicy,
)
from kinovsr.processors.units import FrameUnit
from kinovsr.settings import Settings

_PROFILES = ("normal", "high")


@dataclasses.dataclass(frozen=True, slots=True)
class VtInterpolateConfig:
    target_fps: Fraction
    mode: str


def _parse_fps(value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError("target_fps must be a number")
    fps = Fraction(str(value))
    if fps <= 0:
        raise ValueError("target_fps must be positive")
    return fps


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, VtInterpolateConfig)
    timeline = dataclasses.replace(
        spec.timeline,
        cadence=config.target_fps,
        timestamp_policy=TimestampPolicy.REGENERATED,
        cardinality=(Cardinality.ONE_TO_MANY
                     if config.target_fps > spec.timeline.cadence
                     else Cardinality.MANY_TO_ONE),
    )
    return dataclasses.replace(spec, timeline=timeline)


class VtInterpolateProcessor:
    """Wrap VtfrcSession with grid-exact regenerated timestamps.

    Output unit ``m`` sits at ``m / target_fps``; PTS/duration are that
    instant expressed in the stream's integer time base, with durations
    computed as successive-grid differences so no drift accumulates.
    The session's target-index grid is monotonic across mid-stream
    drains, so a hard cut (scheduler: flush, then reset) keeps timestamps
    strictly increasing while interpolation never crosses the cut.

    Payload contract: source CVPixelBuffers must be IOSurface-backed
    (every real reader, decoder, and pool in this package produces those;
    a bare ``CVPixelBufferCreate`` buffer crashes inside VTFrameProcessor
    natively, not catchably).
    """

    def __init__(self, config: VtInterpolateConfig) -> None:
        self._config = config
        self._session: Any = None
        self._time_base: Fraction | None = None
        self._source_index = 0
        self._target_index = 0
        # The regenerated grid is anchored at the FIRST input unit's PTS,
        # so a stream that legitimately starts at a nonzero origin keeps
        # its alignment with sibling streams (audio) instead of being
        # silently re-based to zero.
        self._origin: int | None = None

    def _grid_ticks(self, target_index: int) -> int:
        return round(target_index / self._config.target_fps
                     / self._time_base)

    def _emit(self, payload: Any) -> FrameUnit:
        m = self._target_index
        self._target_index += 1
        pts = self._grid_ticks(m)
        origin = self._origin or 0
        return FrameUnit(payload=payload, pts=origin + pts,
                         duration=self._grid_ticks(m + 1) - pts)

    def prepare(self, input_spec: StreamSpec,
                context: PipelineContext) -> None:
        from kinovsr.native.temporal import VtfrcSession

        cadence = input_spec.timeline.cadence
        geometry = input_spec.frame.geometry
        self._time_base = input_spec.timeline.time_base
        try:
            self._session = VtfrcSession(
                geometry.width, geometry.height,
                source_fps=float(cadence),
                target_fps=float(self._config.target_fps),
                mode=self._config.mode)
        except (RuntimeError, SystemExit) as exc:
            raise MediaError(
                f"VTFrameRateConversion session unavailable: {exc}") from exc

    def process(self, unit: FrameUnit,
                context: PipelineContext) -> Iterable[FrameUnit]:
        if self._origin is None:
            self._origin = unit.pts
        index = self._source_index
        self._source_index += 1
        for payload in self._session.feed(unit.payload, index):
            yield self._emit(payload)

    def reset(self, boundary: Boundary,
              context: PipelineContext) -> None:
        # The scheduler drained the pre-boundary tail via flush(), which
        # clears the buffered pair inside the session; the target grid
        # deliberately keeps counting so PTS stays monotonic.
        pass

    def flush(self, context: PipelineContext) -> Iterable[FrameUnit]:
        if self._session is None:
            return
        for payload in self._session.drain():
            yield self._emit(payload)

    def close(self, context: PipelineContext) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.close()


class VideoToolboxFactory:
    name = "videotoolbox"

    capabilities = {
        Capability.INTERPOLATE: CapabilitySpec(
            capability=Capability.INTERPOLATE,
            profiles=_PROFILES,
            accepts=StreamConstraint(
                layouts=(Layout.CV_BGRA, Layout.CV_RGBA_HALF,
                         Layout.CV_NV12),
                cadences=(Fraction,),      # CFR sources only
            ),
            produces=_produces,
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
    ) -> VtInterpolateConfig:
        reject_unknown_keys(raw, ("target_fps",))
        if "target_fps" not in raw:
            raise ValueError("target_fps is required for interpolation")
        return VtInterpolateConfig(
            target_fps=_parse_fps(raw["target_fps"]),
            mode=profile or "normal")

    def build(self, config: VtInterpolateConfig, *,
              context: PipelineContext) -> VtInterpolateProcessor:
        return VtInterpolateProcessor(config)


FACTORY = VideoToolboxFactory()

__all__ = [
    "FACTORY",
    "VideoToolboxFactory",
    "VtInterpolateConfig",
    "VtInterpolateProcessor",
]
