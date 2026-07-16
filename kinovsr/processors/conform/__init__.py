"""Explicit CFR conform: map a carried timeline onto a uniform grid.

Old footage arrives with dropped frames, splices, and jitter; the file
pipeline CARRIES that clock exactly. When the deliverable must be
constant-rate anyway (editorial timelines, players that mishandle
gaps), this stage maps each target grid slot to the NEAREST source
frame - duplicating across gaps, dropping under jitter bursts - and
REGENERATES timestamps on the grid. No frame is synthesized: every
output frame is an original source frame (frame synthesis is the
interpolation stage's job). The dup/drop/max-shift ledger is reported
in the run diagnostics, so the normalization is declared, never
silent.

Streaming shape: one frame of lookahead. A slot belongs to whichever
neighboring source frame is nearest in time, so slots up to the
midpoint of a pair emit when the pair completes; flush drains the
final frame's slots through its real duration.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.media.timing import grid_ticks, rational_cadence
from kinovsr.processors.boundaries import Boundary
from kinovsr.processors.capabilities import (
    Capability,
    CapabilitySpec,
    TemporalMode,
)
from kinovsr.processors.errors import MediaError
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Cardinality,
    StreamConstraint,
    StreamSpec,
    TimestampPolicy,
)
from kinovsr.processors.units import FrameUnit


@dataclasses.dataclass(frozen=True, slots=True)
class ConformStageConfig:
    fps: Fraction | None      # None = auto (the source's nominal rate)


def _resolve_cadence(spec: StreamSpec, config: ConformStageConfig) -> Fraction:
    if config.fps is not None:
        return config.fps
    timeline = spec.timeline
    if timeline.nominal_cadence is not None:
        return timeline.nominal_cadence
    if isinstance(timeline.cadence, Fraction):
        return timeline.cadence
    raise MediaError(
        "conform fps 'auto' cannot resolve a rate for this source (no "
        "nominal cadence); give an explicit rate, e.g. fps=25")


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, ConformStageConfig)
    cadence = _resolve_cadence(spec, config)
    reference = (spec.timeline.cadence
                 if isinstance(spec.timeline.cadence, Fraction)
                 else spec.timeline.nominal_cadence)
    timeline = dataclasses.replace(
        spec.timeline,
        cadence=cadence,
        timestamp_policy=TimestampPolicy.REGENERATED,
        cardinality=(
            Cardinality.ONE_TO_MANY
            if reference is None or cadence > reference
            else Cardinality.MANY_TO_ONE
        ),
        nominal_cadence=None,
    )
    return dataclasses.replace(spec, timeline=timeline)


class CfrConformProcessor:
    """Nearest-slot dup/drop mapping with a printed ledger."""

    def __init__(self, config: ConformStageConfig) -> None:
        self._config = config
        self._time_base: Fraction | None = None
        self._cadence: Fraction | None = None
        self._origin: int | None = None
        self._publication_origin_pts: int | None = None
        self._slot: int | None = None
        self._prev: tuple[FrameUnit, Fraction] | None = None
        self._received = 0
        self._emitted = 0
        self._dups = 0
        self._drops = 0
        self._max_shift = Fraction(0)

    def prepare(self, input_spec: StreamSpec,
                context: PipelineContext) -> None:
        self._time_base = input_spec.timeline.time_base
        self._cadence = _resolve_cadence(input_spec, self._config)
        self._publication_origin_pts = context.publication_origin_pts

    def _slot_time(self, m: int) -> Fraction:
        assert self._cadence is not None
        return Fraction(m) / self._cadence

    def _emit(self, unit: FrameUnit, m: int,
              source_time: Fraction) -> FrameUnit:
        assert self._time_base is not None and self._cadence is not None
        origin = self._origin or 0
        pts = grid_ticks(m, self._cadence, self._time_base)
        duration = grid_ticks(m + 1, self._cadence, self._time_base) - pts
        shift = abs(self._slot_time(m) - source_time)
        if shift > self._max_shift:
            self._max_shift = shift
        self._emitted += 1
        return unit.retimed(origin + pts, duration)

    def _emit_through(self, boundary_time: Fraction) -> Iterable[FrameUnit]:
        """Emit the held frame into every slot up to ``boundary_time``."""
        assert (self._prev is not None and self._slot is not None
                and self._cadence is not None)
        unit, unit_time = self._prev
        # Bound the fanout BEFORE emitting (the interpolation stage uses
        # the same ceiling): a recorder clock jump of an hour would
        # otherwise push ~100k duplicates of one frame through every
        # downstream stage and the encoder before the ledger ever prints.
        fanout = (math.floor(boundary_time * self._cadence)
                  - self._slot + 1)
        if fanout > 10_000:
            raise MediaError(
                f"conform would emit {fanout} duplicates of the frame at "
                f"{float(unit_time):.6g}s (a clock jump, not a cadence "
                f"gap); trim the window with --start/--end so it excludes "
                f"the discontinuity, or process the segments separately")
        count = 0
        while self._slot_time(self._slot) <= boundary_time:
            yield self._emit(unit, self._slot, unit_time)
            self._slot += 1
            count += 1
        if count == 0:
            self._drops += 1
        elif count > 1:
            self._dups += count - 1

    def process(self, unit: FrameUnit,
                context: PipelineContext) -> Iterable[FrameUnit]:
        assert self._time_base is not None and self._cadence is not None
        if self._origin is None:
            self._origin = (unit.pts
                            if self._publication_origin_pts is None
                            else self._publication_origin_pts)
        t = Fraction(unit.pts - self._origin) * self._time_base
        self._received += 1
        if self._prev is None:
            if self._slot is None:
                self._slot = math.ceil(t * self._cadence)
            self._prev = (unit, t)
            return
        prev_unit, prev_time = self._prev
        if t <= prev_time:
            raise MediaError(
                f"conform requires strictly increasing timestamps; got "
                f"{float(prev_time):.6g}s then {float(t):.6g}s")
        # Slots up to the midpoint are nearest to the held frame; the
        # boundary slot exactly at the midpoint rounds down (earlier
        # frame wins ties, matching round-half-down on shifts).
        yield from self._emit_through((prev_time + t) / 2)
        self._prev = (unit, t)

    def reset(self, boundary: Boundary, context: PipelineContext) -> None:
        # The scheduler drained the pre-boundary tail via flush(); the
        # slot grid deliberately keeps counting so PTS stays monotonic.
        pass

    def flush(self, context: PipelineContext) -> Iterable[FrameUnit]:
        if self._prev is None:
            return
        assert self._time_base is not None and self._cadence is not None
        unit, unit_time = self._prev
        duration_ticks = unit.duration if unit.duration > 0 else 0
        end = unit_time + Fraction(duration_ticks) * self._time_base
        # The final frame covers slots whose MIDPOINT lies inside the
        # footage: a slot start scraping the very end of the last frame's
        # duration would emit a degenerate duplicate that the
        # duration-preserving clamp then crushes to a few ticks.
        half = 1 / (2 * self._cadence)
        if end - half > unit_time:
            yield from self._emit_through(end - half)
        elif end > unit_time:
            # Degenerate final frame (shorter than half a slot): it still
            # owns its own slot when nothing else claimed it.
            yield from self._emit_through(unit_time)
        self._prev = None

    def close(self, context: PipelineContext) -> None:
        pass

    def run_diagnostics(self) -> list[str]:
        assert self._cadence is not None or self._received == 0
        if self._received == 0:
            return []
        return [
            f"[conform] {self._emitted} frames on the "
            f"{float(self._cadence):g} fps grid from {self._received} "
            f"source frames: {self._dups} duplicated, {self._drops} "
            f"dropped, max shift "
            f"{float(self._max_shift) * 1000.0:.2f} ms"]


class ConformFactory:
    name = "conform"

    capabilities = {
        Capability.PREPROCESS: CapabilitySpec(
            capability=Capability.PREPROCESS,
            profiles=(),
            # Payloads pass through untouched: any layout, any dtype.
            accepts=StreamConstraint(),
            produces=_produces,
            temporal_mode=TemporalMode.CENTERED,
            temporal_radius=1,
            stateful=True,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, object],
        *,
        capability: Capability,
        profile: str | None,
        settings: Any,
    ) -> ConformStageConfig:
        reject_unknown_keys(raw, ("fps",))
        spec = typed_value(raw, "fps", str, "auto")
        text = str(spec).strip().lower()
        if text == "auto":
            return ConformStageConfig(fps=None)
        try:
            fps = rational_cadence(text)
        except ValueError as exc:
            raise ValueError(
                f"conform fps must be 'auto' or a positive rate; got "
                f"{spec!r}") from exc
        return ConformStageConfig(fps=fps)

    def build(self, config: ConformStageConfig, *,
              context: PipelineContext) -> CfrConformProcessor:
        return CfrConformProcessor(config)


FACTORY = ConformFactory()
