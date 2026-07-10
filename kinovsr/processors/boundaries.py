"""Boundary signals: semantic discontinuities that travel with frames.

A boundary rides on the first :class:`~kinovsr.processors.units.FrameUnit`
AFTER the discontinuity. Before that unit reaches a downstream stateful
processor, the scheduler calls ``reset(boundary, context)`` on it, so
temporal state never bleeds across a hard cut. A stateful stage placed
before the relevant boundary emitter is rejected at build time unless the
input endpoint supplies equivalent boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BoundaryKind(Enum):
    # An editorial hard cut: downstream temporal state must reset because
    # pre-cut context is the wrong context.
    HARD_CUT = "hard_cut"
    # The very first unit of the stream. Emitted by input endpoints so
    # stateful processors get exactly one reset-to-initial call per run
    # through the same code path as any other boundary.
    STREAM_START = "stream_start"


@dataclass(frozen=True, slots=True)
class Boundary:
    kind: BoundaryKind
    # Source display index of the first frame after the discontinuity,
    # when the emitter knows it (cut detectors do; live sources may not).
    source_index: int | None = None


__all__ = ["Boundary", "BoundaryKind"]
