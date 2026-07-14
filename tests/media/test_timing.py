"""Exact source classification and long-run rational timestamp grids."""

from __future__ import annotations

from fractions import Fraction

import pytest

from kinovsr.media import pixel_buffers
from kinovsr.media.timing import analyze_sample_timing, grid_ticks

pytestmark = pytest.mark.unit


def test_variable_sample_intervals_are_not_reconstructed_from_nominal_fps():
    pts = [Fraction(0), Fraction(1, 30), Fraction(1, 10),
           Fraction(2, 15), Fraction(7, 30)]
    timing = analyze_sample_timing(
        [(value, Fraction(1, 30)) for value in pts],
        nominal_cadence=15,
        source_tick=Fraction(1, 30000),
    )
    assert timing.sample_count == 5
    assert timing.cadence is None
    assert timing.duration == Fraction(4, 15)


def test_consistent_observed_grid_can_correct_an_inaccurate_nominal_rate():
    timing = analyze_sample_timing(
        [(Fraction(i, 30), Fraction(1, 30)) for i in range(5)],
        nominal_cadence=15,
        source_tick=Fraction(1, 30000),
    )
    assert timing.sample_count == 5
    assert timing.cadence == Fraction(30)
    assert timing.duration == Fraction(1, 6)


def test_single_sample_uses_its_duration_when_nominal_rate_is_missing():
    timing = analyze_sample_timing(
        [(Fraction(10), Fraction(1, 24))],
        nominal_cadence=None,
        source_tick=Fraction(1, 24000),
    )
    assert timing.sample_count == 1
    assert timing.cadence == Fraction(24)
    assert timing.first_pts == Fraction(10)
    assert timing.duration == Fraction(1, 24)


def test_duplicate_display_timestamps_are_rejected_as_variable():
    timing = analyze_sample_timing(
        [(Fraction(0), None), (Fraction(0), Fraction(1, 30)),
         (Fraction(1, 30), None)],
        nominal_cadence=30,
        source_tick=Fraction(1, 30000),
    )
    assert timing.sample_count == 3
    assert timing.cadence is None


def test_coarse_exact_clock_does_not_hide_a_dropped_frame():
    tick = Fraction(1, 25)
    timing = analyze_sample_timing(
        [(tick * value, tick) for value in (0, 1, 3, 4)],
        nominal_cadence=25,
        source_tick=tick,
    )
    assert timing.cadence is None
    assert timing.duration == Fraction(1, 5)


def test_coarse_exact_clock_still_accepts_an_unbroken_grid():
    tick = Fraction(1, 25)
    timing = analyze_sample_timing(
        [(tick * value, tick) for value in range(4)],
        nominal_cadence=25,
        source_tick=tick,
    )
    assert timing.cadence == Fraction(25)
    assert timing.duration == Fraction(4, 25)


@pytest.mark.parametrize("nominal", [None, 15])
def test_uniform_two_tick_grid_can_replace_missing_or_wrong_nominal(nominal):
    tick = Fraction(1, 50)
    timing = analyze_sample_timing(
        [(Fraction(value, 25), Fraction(1, 25)) for value in range(4)],
        nominal_cadence=nominal,
        source_tick=tick,
    )
    assert timing.cadence == Fraction(25)
    assert timing.duration == Fraction(4, 25)


def test_long_coarse_grid_recovers_ntsc_instead_of_rounded_nominal():
    cadence = Fraction(30000, 1001)
    tick = Fraction(1, 600)
    samples = [
        (round(Fraction(index) / cadence / tick) * tick, None)
        for index in range(3000)
    ]
    timing = analyze_sample_timing(
        samples, nominal_cadence=30, source_tick=tick)
    assert timing.cadence == cadence
    assert abs(timing.duration - Fraction(3000) / cadence) <= tick


def test_variable_final_sample_duration_is_not_silently_truncated():
    timing = analyze_sample_timing(
        [(Fraction(0), Fraction(1, 30)),
         (Fraction(1, 30), Fraction(1, 30)),
         (Fraction(2, 30), Fraction(1))],
        nominal_cadence=30,
        source_tick=Fraction(1, 30000),
    )
    assert timing.cadence is None
    assert timing.duration == Fraction(16, 15)


def test_one_source_tick_ntsc_quantization_is_still_cfr():
    cadence = Fraction(30000, 1001)
    tick = Fraction(1, 30000)
    samples = [
        (Fraction(round(Fraction(i) / cadence / tick)) * tick, None)
        for i in range(120)
    ]
    timing = analyze_sample_timing(
        samples, nominal_cadence=float(cadence), source_tick=tick)
    assert timing.cadence == cadence
    assert timing.sample_count == 120


@pytest.mark.parametrize("cadence", [
    Fraction(30000, 1001),
    Fraction(60000, 1001),
])
def test_one_hour_ntsc_grid_never_accumulates_rounding_drift(cadence):
    scale = pixel_buffers.VIDEO_TIME_SCALE
    time_base = Fraction(1, scale)
    frame_count = round(3600 * cadence)
    previous = -1
    for index in range(frame_count + 1):
        current = pixel_buffers.frame_ticks(index, cadence)
        assert current > previous
        previous = current

    exact_end = Fraction(frame_count) / cadence
    encoded_end = Fraction(previous, scale)
    assert abs(encoded_end - exact_end) <= time_base
    assert previous == grid_ticks(frame_count, cadence, time_base)
    assert pixel_buffers.frame_pts(frame_count, cadence).value == previous
