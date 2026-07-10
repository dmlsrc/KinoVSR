"""Host-neutral progress reporting - the contract, not the terminal.

Runtime code reports phase progress through a :class:`Reporter`; it never
imports Rich or owns stdout. The CLI hands runtime code a Rich-backed
reporter (:class:`kinovsr.ui.progress.RichReporter`); a host engine may
hand in its own implementation to drive its own UI; tests use
:class:`RecordingReporter`; the default is a no-op.

Log messages are deliberately NOT part of this protocol: they already have
a host-neutral surface in stdlib :mod:`logging` (configured by
:mod:`kinovsr.ui.logging` for the CLI). The reporter carries what logging
cannot: live phase progress with totals, units, and completion.

A phase is keyed by its name string. Names are expected to be unique among
concurrently active phases (the M3 scheduler will use per-instance stage
ids); reusing a name after its phase ended starts a new phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Reporter(Protocol):
    """What runtime code may assume about a progress sink."""

    def phase_start(self, phase: str, *, total: float | None = None,
                    unit: str = "it") -> None:
        """A phase began. ``total`` is in ``unit`` steps; ``None`` = unknown."""

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        """The phase progressed by ``advance`` steps."""

    def phase_end(self, phase: str) -> None:
        """The phase finished (successfully or not); release its display."""


class NullReporter:
    """The default sink: silence. Keeps reporter plumbing unconditional."""

    def phase_start(self, phase: str, *, total: float | None = None,
                    unit: str = "it") -> None:
        pass

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        pass

    def phase_end(self, phase: str) -> None:
        pass


@dataclass
class RecordingReporter:
    """Captures events as ``(kind, phase, payload)`` tuples, for tests."""

    events: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def phase_start(self, phase: str, *, total: float | None = None,
                    unit: str = "it") -> None:
        self.events.append(("start", phase, {"total": total, "unit": unit}))

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        self.events.append(("advance", phase, {"advance": advance}))

    def phase_end(self, phase: str) -> None:
        self.events.append(("end", phase, {}))
