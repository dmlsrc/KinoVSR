"""Stacked, phase-accurate progress bars for KinoVSR.

Single home for the project's progress UI primitives.  Previously lived
inside the VSR harness, then moved here as a small reusable UI primitive:

* `scripts/vsr_harness.py` decode / VSR / output frame bars.
* `kinovsr.encode` VT-encode frame bar, alone or stacked with caller-owned
  upstream work.

Keeping the progress UI here under a dedicated `progress/` package means
callers don't have to import PyObjC-touching modules just to draw a bar.

Public surface:

    from kinovsr.progress import PhaseBar, StackedPhaseBars

Submodule layout:

    bars.py     PhaseBar / StackedPhaseBars implementation.
"""

from __future__ import annotations

from .bars import (
    PhaseBar,
    StackedPhaseBars,
)

__all__ = [
    "PhaseBar",
    "StackedPhaseBars",
]
