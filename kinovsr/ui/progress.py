"""Progress bars - thin layer over ``rich.progress``, plus the Rich reporter.

A :class:`rich.progress.Progress` renders stacked bars as an internal
table, so cross-row column alignment, label / count / percentage / time /
ETA columns, and the ``with Progress() as p:`` context all come for free.
One custom column adds math-consistent wall-clock pace.

**Redraw discipline matters.** macOS hardware-accelerates the terminal -
every redraw burns Terminal and WindowServer GPU cycles. The inherited
``StackedPhaseBars`` throttles to 1 Hz; we match that by default via
``refresh_per_second=1.0`` rather than rich's default of 10 Hz. Long clips
run thousands of frames; 1 Hz keeps the redraw cost invisible next to the
per-frame work.

Runtime code should not use this module directly: it reports through a
:class:`kinovsr.reporting.Reporter`, and the CLI wires a
:class:`RichReporter` in front of these bars.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from .console import get_console

_log = logging.getLogger(__name__)


class WallClockPaceColumn(ProgressColumn):
    """Pace as ``X.X s/<unit>`` (slow) or ``X.X <unit>/s`` (fast).

    Computed as ``completed / elapsed_from_start`` so it is
    math-consistent with the ``elapsed`` and ``count`` columns -
    ``pace x elapsed`` reproduces ``completed`` exactly. Rich's built-in
    speed column uses a sliding window, which does not have this property.

    The ``unit`` (``"frame"``, ``"step"``, ...) is read from the task's
    ``fields`` dict; defaults to ``"it"`` if unset.
    """

    def render(self, task) -> Text:
        if task.elapsed is None or task.completed == 0:
            return Text("measuring", style="dim")
        sec_per_unit = task.elapsed / task.completed
        unit = task.fields.get("unit", "it")
        if sec_per_unit >= 1.0:
            return Text(f"{sec_per_unit:>6.1f} s/{unit}")
        return Text(f"{1.0 / sec_per_unit:>6.1f} {unit}/s")


def make_progress(
    *, refresh_per_second: float = 1.0, console: Console | None = None,
) -> Progress:
    """Build a configured ``Progress`` ready for ``with`` use.

    ``refresh_per_second`` defaults to ``1.0`` for the redraw-cost reason
    in the module docstring. ``console`` defaults to the shared console;
    tests pass their own recording console.
    """
    return Progress(
        TextColumn("  {task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("| RUN"),
        TimeElapsedColumn(),
        TextColumn("| ETA"),
        TimeRemainingColumn(),
        TextColumn("|"),
        WallClockPaceColumn(),
        console=console or get_console(),
        refresh_per_second=refresh_per_second,
        transient=False,
    )


@contextmanager
def track_phase(
    progress: Progress,
    description: str,
    *,
    total: float,
    unit: str = "it",
    logger: logging.Logger | None = None,
) -> Iterator[TaskID]:
    """Add a task to ``progress``, yield its id, log a summary on exit.

    On exit (success or error) :func:`log_phase_summary` writes one INFO
    line - total wall time, iterations, and seconds per iteration - so the
    milestone persists after the (non-transient) bar and lands in any log
    file. Pass ``logger`` to attribute it to the calling subsystem rather
    than ``kinovsr.ui.progress``.
    """
    task_id = progress.add_task(description, total=total, unit=unit)
    try:
        yield task_id
    finally:
        log_phase_summary(progress, task_id, logger=logger)


def log_phase_summary(
    progress: Progress, task_id: TaskID, *, logger: logging.Logger | None = None,
) -> None:
    """Log a one-line summary of a task: total time, count, time per unit.

    Reads the finished task's ``elapsed`` and ``completed`` straight from
    rich, so the numbers match the live :class:`WallClockPaceColumn`. A
    no-op if the task id is unknown; logs a short note if the task never
    advanced.
    """
    task = next((t for t in progress.tasks if t.id == task_id), None)
    if task is None:
        return
    out = logger or _log
    iters = int(task.completed)
    unit = task.fields.get("unit", "it")
    if iters <= 0:
        out.info(f"{task.description}: no iterations recorded")
        return
    elapsed = task.elapsed or 0.0
    out.info(
        f"{task.description}: {iters} {unit} in {elapsed:.1f}s "
        f"({elapsed / iters:.3f}s/{unit})"
    )


class RichReporter:
    """A :class:`kinovsr.reporting.Reporter` that drives stacked Rich bars.

    One bar per active phase, keyed by phase name. ``phase_end`` logs the
    phase summary and retires the bar's bookkeeping (the non-transient row
    stays on screen). Use as a context manager so the underlying
    ``Progress`` starts and stops with the run::

        with RichReporter() as reporter:
            run_pipeline(..., reporter=reporter)
    """

    def __init__(self, *, progress: Progress | None = None) -> None:
        self._progress = progress or make_progress()
        self._tasks: dict[str, TaskID] = {}

    def __enter__(self) -> RichReporter:
        self._progress.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._progress.stop()

    # -- Reporter protocol -------------------------------------------------

    def phase_start(self, phase: str, *, total: float | None = None,
                    unit: str = "it") -> None:
        self._tasks[phase] = self._progress.add_task(phase, total=total, unit=unit)

    def phase_advance(self, phase: str, advance: float = 1.0) -> None:
        task_id = self._tasks.get(phase)
        if task_id is not None:
            self._progress.advance(task_id, advance)

    def phase_end(self, phase: str) -> None:
        task_id = self._tasks.pop(phase, None)
        if task_id is not None:
            log_phase_summary(self._progress, task_id)
