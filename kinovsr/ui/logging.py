"""Rich-backed logging - themed levels, per-subsystem control.

Terminal output goes through the stdlib :mod:`logging` framework rendered
by a single :class:`rich.logging.RichHandler`, so progress bars, the color
theme, the verbosity level, and per-subsystem filtering stay coordinated on
one console.

Each module gets its own logger via ``logging.getLogger(__name__)``. The
logger *name* (``kinovsr.native.encode``, ``kinovsr.pipeline``, ...) is the
subsystem, so output can be leveled or silenced per subsystem::

    logging.getLogger("kinovsr.native").setLevel(logging.WARNING)

Call :func:`configure_logging` once at CLI startup (typically via
:func:`configure_logging_from_settings`). Until then the library stays
silent at INFO, as library logging convention expects; warnings and errors
still reach stderr via logging's last-resort handler.

Direct terminal writes are forbidden in package modules and developer tools:
they clobber live progress bars and bypass severity, verbosity, file logging,
and per-module filtering. Machine-readable subprocess protocols use dedicated
logging handlers when stdout is part of the transport contract.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from rich.logging import RichHandler

from .console import get_console

if TYPE_CHECKING:
    from ..settings import Settings

# Parent logger for the whole package; every ``kinovsr.*`` logger propagates
# up to it, so one handler here renders all subsystems.
_ROOT_LOGGER = "kinovsr"


def configure_machine_output(
    logger_name: str,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure one isolated logger for a machine-readable stdout protocol.

    Human-facing logs remain on the shared Rich stderr console. This logger
    deliberately emits only ``record.message`` so a parent process can parse
    each line without timestamps, level labels, or duplicate propagation.
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _level_for(verbosity: int, quiet: bool) -> int:
    """Map a verbosity dial + quiet flag to a logging level.

    ``quiet`` wins and floors output at WARNING, so warnings and errors
    always get through - unlike a blanket mute, which is the bug a real
    level avoids. Otherwise ``verbosity`` 0 is INFO and 1+ is DEBUG.
    """
    if quiet:
        return logging.WARNING
    if verbosity >= 1:
        return logging.DEBUG
    return logging.INFO


def configure_logging(
    verbosity: int = 0,
    *,
    quiet: bool = False,
    show_date: bool = False,
    log_file: str | Path | None = None,
    log_file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Install the rich logging handler on the ``kinovsr`` logger; return it.

    Idempotent: clears the handlers it previously attached and rebuilds, so
    it is safe to call again after re-parsing settings. ``verbosity`` 0 =
    INFO (default), 1+ = DEBUG; ``quiet`` floors the console at WARNING.

    The console timestamp is time-only by default (``[HH:MM:SS]``);
    ``show_date=True`` prefixes the calendar date.

    Pass ``log_file`` to also tee records to a sidecar file at
    ``log_file_level`` (default DEBUG), independent of the console level -
    the console can sit at INFO while the file keeps the full DEBUG trace.
    Each file line carries the date, level, and subsystem (logger name).
    """
    console_level = _level_for(verbosity, quiet)
    time_format = "[%Y-%m-%d %H:%M:%S]" if show_date else "[%H:%M:%S]"
    console_handler = RichHandler(
        console=get_console(),
        show_time=True,
        show_path=False,  # the log call's file:line is dev noise in production
        markup=False,  # callers pass plain text, not rich markup
        rich_tracebacks=True,
        log_time_format=time_format,
    )
    console_handler.setLevel(console_level)

    logger = logging.getLogger(_ROOT_LOGGER)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    handler_levels = [console_level]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)
        handler_levels.append(log_file_level)

    # The logger gates before its handlers, so it must sit at the most
    # verbose level any handler wants; otherwise a DEBUG file would miss
    # records the INFO console already dropped at the logger.
    logger.setLevel(min(handler_levels))
    return logger


def configure_logging_from_settings(settings: Settings) -> logging.Logger:
    """Configure logging from :class:`~kinovsr.settings.Settings`.

    ``settings.verbose`` maps to verbosity 1 (DEBUG); ``settings.quiet``
    floors the console at WARNING and wins over verbose.
    """
    return configure_logging(
        verbosity=1 if settings.verbose else 0,
        quiet=settings.quiet,
    )
