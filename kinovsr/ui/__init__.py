"""Rich-backed CLI presentation - the terminal implementation of reporting.

One shared ``Console`` backs both the stdlib ``logging`` framework (via a
``RichHandler``; subsystems are logger names) and the stacked progress
bars, so they interleave cleanly. Runtime modules emit via their own
``logging.getLogger(__name__)`` and report progress through a
:class:`kinovsr.reporting.Reporter`; only the CLI imports this package to
wire the terminal implementations up.
"""

from .console import get_console
from .logging import (
    configure_logging,
    configure_logging_from_settings,
    configure_machine_output,
)
from .progress import (
    RichReporter,
    WallClockPaceColumn,
    log_phase_summary,
    make_progress,
    track_phase,
)

__all__ = [
    "RichReporter",
    "WallClockPaceColumn",
    "configure_logging",
    "configure_logging_from_settings",
    "configure_machine_output",
    "get_console",
    "log_phase_summary",
    "make_progress",
    "track_phase",
]
