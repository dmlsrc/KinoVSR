"""Reporter contract and Rich UI: logging config, pace column, reporter bars."""

from __future__ import annotations

import io
import logging

import pytest
from rich.console import Console

from kinovsr.reporting import NullReporter, RecordingReporter, Reporter
from kinovsr.settings import Settings
from kinovsr.ui import (
    RichReporter,
    WallClockPaceColumn,
    configure_logging,
    configure_logging_from_settings,
    configure_machine_output,
    get_console,
    make_progress,
    track_phase,
)


@pytest.fixture
def quiet_progress():
    """A Progress rendering into a throwaway buffer instead of the terminal."""
    console = Console(file=io.StringIO(), width=120, force_terminal=False)
    return make_progress(console=console)


@pytest.fixture(autouse=True)
def _reset_kinovsr_logger():
    yield
    logger = logging.getLogger("kinovsr")
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)


# ---- reporter contract -------------------------------------------------------


def test_reporter_protocol_matches_implementations():
    assert isinstance(NullReporter(), Reporter)
    assert isinstance(RecordingReporter(), Reporter)
    assert isinstance(RichReporter(progress=make_progress(
        console=Console(file=io.StringIO()))), Reporter)


def test_recording_reporter_captures_ordered_events():
    r = RecordingReporter()
    r.phase_start("denoise", total=8, unit="frame")
    r.phase_advance("denoise")
    r.phase_advance("denoise", 2.0)
    r.phase_end("denoise")
    assert r.events == [
        ("start", "denoise", {"total": 8, "unit": "frame"}),
        ("advance", "denoise", {"advance": 1.0}),
        ("advance", "denoise", {"advance": 2.0}),
        ("end", "denoise", {}),
    ]


def test_null_reporter_is_silent_noop():
    r = NullReporter()
    r.phase_start("x", total=None)
    r.phase_advance("x")
    r.phase_end("x")


# ---- logging configuration -----------------------------------------------------


def test_configure_logging_levels():
    assert configure_logging(0).handlers[0].level == logging.INFO
    assert configure_logging(1).handlers[0].level == logging.DEBUG
    assert configure_logging(1, quiet=True).handlers[0].level == logging.WARNING


def test_configure_logging_is_idempotent():
    configure_logging(0)
    logger = configure_logging(0)
    assert len(logger.handlers) == 1


def test_configure_logging_tees_to_file(tmp_path):
    log_file = tmp_path / "run.log"
    logger = configure_logging(0, log_file=log_file)
    # Console at INFO, file at DEBUG: the logger itself must sit at DEBUG.
    assert logger.level == logging.DEBUG
    logging.getLogger("kinovsr.test_ui").debug("debug goes to the file")
    for handler in logger.handlers:
        handler.flush()
    text = log_file.read_text()
    assert "debug goes to the file" in text
    assert "kinovsr.test_ui" in text  # subsystem name recorded


def test_configure_logging_from_settings_maps_fields():
    logger = configure_logging_from_settings(Settings(verbose=True))
    assert logger.handlers[0].level == logging.DEBUG
    logger = configure_logging_from_settings(Settings(verbose=True, quiet=True))
    assert logger.handlers[0].level == logging.WARNING  # quiet wins


def test_machine_output_is_plain_and_does_not_propagate():
    stream = io.StringIO()
    logger = configure_machine_output("kinovsr.test_ui.machine", stream=stream)
    logger.info('{"ok": true}')
    assert stream.getvalue() == '{"ok": true}\n'
    assert logger.propagate is False
    logger.handlers.clear()


def test_get_console_is_a_singleton_on_stderr():
    assert get_console() is get_console()
    assert get_console().stderr is True


# ---- pace column ------------------------------------------------------------------


class _Task:
    def __init__(self, elapsed, completed, unit=None):
        self.elapsed = elapsed
        self.completed = completed
        self.fields = {} if unit is None else {"unit": unit}


def test_pace_column_measuring_before_first_completion():
    assert WallClockPaceColumn().render(_Task(None, 0)).plain == "measuring"
    assert WallClockPaceColumn().render(_Task(3.0, 0)).plain == "measuring"


def test_pace_column_slow_is_seconds_per_unit():
    text = WallClockPaceColumn().render(_Task(30.0, 10, unit="frame")).plain
    assert text.strip() == "3.0 s/frame"


def test_pace_column_fast_is_units_per_second():
    text = WallClockPaceColumn().render(_Task(2.0, 50, unit="frame")).plain
    assert text.strip() == "25.0 frame/s"


def test_pace_column_is_wall_clock_consistent():
    task = _Task(20.0, 8, unit="step")
    text = WallClockPaceColumn().render(task).plain
    sec_per_unit = float(text.strip().split(" ")[0])
    assert sec_per_unit * task.completed == pytest.approx(task.elapsed)


# ---- phase tracking and the Rich reporter ------------------------------------------


def test_track_phase_logs_summary(quiet_progress, caplog):
    with caplog.at_level(logging.INFO, logger="kinovsr.ui.progress"), \
            quiet_progress, \
            track_phase(quiet_progress, "denoise", total=4, unit="frame") as task:
        for _ in range(4):
            quiet_progress.advance(task)
    assert any("denoise: 4 frame in" in r.message for r in caplog.records)


def test_track_phase_notes_zero_iterations(quiet_progress, caplog):
    with caplog.at_level(logging.INFO, logger="kinovsr.ui.progress"), \
            quiet_progress, track_phase(quiet_progress, "empty", total=4):
        pass
    assert any("empty: no iterations recorded" in r.message for r in caplog.records)


def test_rich_reporter_drives_bars_and_logs_on_end(quiet_progress, caplog):
    with caplog.at_level(logging.INFO, logger="kinovsr.ui.progress"), \
            RichReporter(progress=quiet_progress) as reporter:
        reporter.phase_start("upscale", total=3, unit="frame")
        for _ in range(3):
            reporter.phase_advance("upscale")
        reporter.phase_end("upscale")
    assert any("upscale: 3 frame in" in r.message for r in caplog.records)


def test_rich_reporter_tolerates_unknown_phase(quiet_progress):
    with RichReporter(progress=quiet_progress) as reporter:
        reporter.phase_advance("never-started")
        reporter.phase_end("never-started")


def test_rich_reporter_double_end_is_safe(quiet_progress):
    with RichReporter(progress=quiet_progress) as reporter:
        reporter.phase_start("x", total=1)
        reporter.phase_end("x")
        reporter.phase_end("x")
