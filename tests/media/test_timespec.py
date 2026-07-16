"""Trim-window resolution: fps grid for uniform clocks, table bisect off it."""

from __future__ import annotations

from fractions import Fraction

import pytest

from kinovsr.media.timespec import resolve_trim
from kinovsr.media.timing import SampleTiming, analyze_sample_table

pytestmark = pytest.mark.unit


def _gapped_table():
    # A 25 fps grid with dropped slots: frames at 0, 1, 2, 5, 6, 10, 11.
    pts = [Fraction(i, 25) for i in (0, 1, 2, 5, 6, 10, 11)]
    return analyze_sample_table(
        [SampleTiming(pts=value, duration=Fraction(1, 25)) for value in pts],
        nominal_cadence=25,
        source_tick=Fraction(1, 25000),
    )


def test_uniform_time_forms_keep_the_fps_grid():
    assert resolve_trim("2s", "4s", 25.0, 200) == (50, 100)


def test_frame_forms_stay_ordinals_with_a_table():
    table = _gapped_table()
    assert resolve_trim("2", "5f", 25.0, 200, table=table) == (2, 5)


def test_time_start_picks_the_frame_on_screen_at_that_instant():
    table = _gapped_table()
    # t=0.12s falls inside the gap after the frame at 2/25=0.08s: that
    # frame is still on screen, so the window starts at table position 2.
    start, end = resolve_trim("0.12s", None, 25.0, 200, table=table)
    assert (start, end) == (2, None)


def test_time_end_excludes_frames_at_or_after_the_instant():
    table = _gapped_table()
    # t=0.2s is exactly the frame at 5/25: an end boundary excludes it.
    start, end = resolve_trim(None, "0.2s", 25.0, 200, table=table)
    assert (start, end) == (0, 3)


def test_table_total_overrides_the_probe_estimate():
    table = _gapped_table()
    # The probe-estimate total (here: wrong, too large) must not admit
    # windows past the table's real sample count.
    start, end = resolve_trim("1f", "100f", 25.0, 700, table=table)
    assert (start, end) == (1, 7)


def test_windows_past_the_table_are_rejected():
    table = _gapped_table()
    with pytest.raises(ValueError, match="past the input length"):
        resolve_trim("7f", None, 25.0, 700, table=table)


def test_bad_specs_still_raise_cleanly_with_a_table():
    table = _gapped_table()
    with pytest.raises(ValueError, match="negative"):
        resolve_trim("-2.5", None, 25.0, 200, table=table)
