"""The timestamped unit that actually travels along a chain.

A :class:`FrameUnit` is a frozen value: payload plus integer PTS/duration
in the stream's declared time base, plus any boundaries that begin at this
unit. The payload is an MLX array or a CVPixelBuffer per the edge's
:class:`~kinovsr.processors.specs.Layout`; the unit itself never inspects
it (no per-frame host/device synchronization in framework code).

Replacing payload or timestamps produces a new unit (`with_payload`,
`retimed`); boundaries are preserved by default because rewriting a frame
does not erase the discontinuity it starts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from .boundaries import Boundary


@dataclass(frozen=True, slots=True)
class FrameUnit:
    payload: Any                 # mx.array or CVPixelBuffer, per edge Layout
    pts: int                     # in the stream's TimelineSpec.time_base
    duration: int                # same time base; 0 = unknown/unspecified
    boundaries: tuple[Boundary, ...] = ()

    def with_payload(self, payload: Any) -> FrameUnit:
        """Same instant, new payload (the per-frame processing shape)."""
        return dataclasses.replace(self, payload=payload)

    def retimed(self, pts: int, duration: int | None = None) -> FrameUnit:
        """Same payload on a rewritten timeline (interpolation emits these)."""
        return dataclasses.replace(
            self, pts=pts,
            duration=self.duration if duration is None else duration)

    def with_boundary(self, boundary: Boundary) -> FrameUnit:
        return dataclasses.replace(
            self, boundaries=(*self.boundaries, boundary))


__all__ = ["FrameUnit"]
