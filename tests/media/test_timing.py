"""Exact source classification and long-run rational timestamp grids."""

from __future__ import annotations

from fractions import Fraction

import pytest

from kinovsr.media import pixel_buffers
from kinovsr.media.timing import (
    SampleTiming,
    TimingVerdict,
    analyze_sample_table,
    analyze_sample_timing,
    grid_ticks,
)

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


# ------------------------------------------------------------- table


def _records(pairs, sync=None, sizes=None):
    return [SampleTiming(
        pts=pts,
        duration=duration,
        is_sync=None if sync is None else sync[i],
        coded_size=None if sizes is None else sizes[i],
    ) for i, (pts, duration) in enumerate(pairs)]


def test_exact_grid_classifies_exact_cfr():
    table = analyze_sample_table(
        _records([(Fraction(i, 25), Fraction(1, 25)) for i in range(6)]),
        nominal_cadence=25,
        source_tick=Fraction(1, 25000),
    )
    assert table.verdict is TimingVerdict.EXACT_CFR
    assert table.cadence == Fraction(25)
    assert table.grid_cadence == Fraction(25)
    assert table.timing() == analyze_sample_timing(
        [(Fraction(i, 25), Fraction(1, 25)) for i in range(6)],
        nominal_cadence=25,
        source_tick=Fraction(1, 25000),
    )


def test_one_tick_quantized_grid_classifies_cfr_not_exact():
    # NTSC on a 1/600 clock: 20.02 ticks per frame, so adjacent stamps
    # alternate 20- and 21-tick intervals around the true grid.
    cadence = Fraction(30000, 1001)
    tick = Fraction(1, 600)
    pairs = [
        (round(Fraction(i) / cadence / tick) * tick, None)
        for i in range(3000)
    ]
    table = analyze_sample_table(
        _records(pairs), nominal_cadence=30, source_tick=tick)
    assert table.verdict is TimingVerdict.CFR
    assert table.cadence == cadence
    assert table.grid_cadence == cadence


def test_exact_tick_ntsc_grid_classifies_exact_cfr():
    # A 1/30000 clock represents 30000/1001 exactly (1001 ticks per
    # frame), so nothing alternates and the grid is exactly uniform.
    cadence = Fraction(30000, 1001)
    tick = Fraction(1, 30000)
    pairs = [
        (Fraction(round(Fraction(i) / cadence / tick)) * tick, None)
        for i in range(120)
    ]
    table = analyze_sample_table(
        _records(pairs), nominal_cadence=float(cadence), source_tick=tick)
    assert table.verdict is TimingVerdict.EXACT_CFR
    assert table.cadence == cadence


def test_dropped_frame_on_exact_clock_classifies_gapped_cfr():
    tick = Fraction(1, 25)
    table = analyze_sample_table(
        _records([(tick * value, tick) for value in (0, 1, 3, 4)]),
        nominal_cadence=25,
        source_tick=tick,
    )
    assert table.verdict is TimingVerdict.GAPPED_CFR
    assert table.cadence is None            # legacy CFR accept unchanged
    assert table.grid_cadence == Fraction(25)
    assert table.timing().cadence is None   # the file pipeline's view


def test_splice_gap_classifies_gapped_cfr():
    pts = [Fraction(i, 25) for i in range(5)]
    pts += [Fraction(i, 25) for i in range(105, 110)]
    table = analyze_sample_table(
        _records([(value, Fraction(1, 25)) for value in pts]),
        nominal_cadence=25,
        source_tick=Fraction(1, 25000),
    )
    assert table.verdict is TimingVerdict.GAPPED_CFR
    assert table.grid_cadence == Fraction(25)


def test_gapped_grid_recovers_from_modal_delta_without_nominal():
    tick = Fraction(1, 30000)
    pts = [Fraction(i, 30) for i in (0, 1, 2, 4, 5, 6, 8, 9)]
    table = analyze_sample_table(
        _records([(value, None) for value in pts]),
        nominal_cadence=None,
        source_tick=tick,
    )
    assert table.verdict is TimingVerdict.GAPPED_CFR
    assert table.grid_cadence == Fraction(30)


def test_multiple_of_grid_intervals_classify_gapped_not_vfr():
    # The reclassification of the module's leading VFR fixture: every
    # interval is an exact multiple of 1/30, so the clock is a gapped
    # 30 fps grid; the legacy view still reports variable.
    pts = [Fraction(0), Fraction(1, 30), Fraction(1, 10),
           Fraction(2, 15), Fraction(7, 30)]
    table = analyze_sample_table(
        _records([(value, Fraction(1, 30)) for value in pts]),
        nominal_cadence=15,
        source_tick=Fraction(1, 30000),
    )
    assert table.verdict is TimingVerdict.GAPPED_CFR
    assert table.grid_cadence == Fraction(30)
    assert table.timing().cadence is None


def test_duplicate_timestamps_classify_vfr():
    table = analyze_sample_table(
        _records([(Fraction(0), None), (Fraction(0), Fraction(1, 30)),
                  (Fraction(1, 30), None)]),
        nominal_cadence=30,
        source_tick=Fraction(1, 30000),
    )
    assert table.verdict is TimingVerdict.VFR
    assert table.grid_cadence is None


def test_non_multiple_intervals_classify_vfr():
    pts = [Fraction(0), Fraction(1, 30), Fraction(1, 30) + Fraction(1, 24),
           Fraction(1, 30) + Fraction(1, 24) + Fraction(1, 30)]
    table = analyze_sample_table(
        _records([(value, None) for value in pts]),
        nominal_cadence=30,
        source_tick=Fraction(1, 90000),
    )
    assert table.verdict is TimingVerdict.VFR
    assert table.grid_cadence is None


def test_uniform_pts_with_variable_durations_stay_vfr():
    # The duration cross-check rejected this as CFR; gapped rules must
    # not quietly re-admit it (every interval multiple is one).
    table = analyze_sample_table(
        _records([(Fraction(0), Fraction(1, 30)),
                  (Fraction(1, 30), Fraction(1, 30)),
                  (Fraction(2, 30), Fraction(1))]),
        nominal_cadence=30,
        source_tick=Fraction(1, 30000),
    )
    assert table.verdict is TimingVerdict.VFR
    assert table.grid_cadence is None


def test_sync_flags_and_coded_sizes_are_carried_in_display_order():
    pairs = [(Fraction(i, 25), Fraction(1, 25)) for i in range(4)]
    sync = [True, False, False, True]
    sizes = [9000, 1200, 1100, 8800]
    decode_order = [0, 2, 1, 3]
    table = analyze_sample_table(
        _records([pairs[i] for i in decode_order],
                 sync=[sync[i] for i in decode_order],
                 sizes=[sizes[i] for i in decode_order]),
        nominal_cadence=25,
        source_tick=Fraction(1, 25000),
    )
    assert [sample.pts for sample in table.samples] == [p for p, _ in pairs]
    assert [sample.is_sync for sample in table.samples] == sync
    assert [sample.coded_size for sample in table.samples] == sizes
    assert table.keyframe_indices == (0, 3)


def test_frame_gop_reports_absolute_ordinals_and_lengths():
    pairs = [(Fraction(i, 25), Fraction(1, 25)) for i in range(5)]
    table = analyze_sample_table(
        _records(pairs, sync=[True, False, False, True, False]),
        nominal_cadence=25,
        source_tick=Fraction(1, 25000),
    )
    assert table.frame_gop(0) == (0, 3)
    assert table.frame_gop(2) == (2, 3)
    assert table.frame_gop(3) == (0, 2)
    assert table.frame_gop(4) == (1, 2)
    with pytest.raises(IndexError):
        table.frame_gop(5)


def test_frame_gop_counts_an_open_leading_gop_from_zero():
    pairs = [(Fraction(i, 25), Fraction(1, 25)) for i in range(4)]
    table = analyze_sample_table(
        _records(pairs, sync=[False, False, True, False]),
        nominal_cadence=25,
        source_tick=Fraction(1, 25000),
    )
    assert table.frame_gop(1) == (1, 2)
    assert table.frame_gop(2) == (0, 2)


def test_frame_gop_is_none_without_sync_flags():
    pairs = [(Fraction(i, 25), None) for i in range(3)]
    table = analyze_sample_table(
        _records(pairs),
        nominal_cadence=25,
        source_tick=Fraction(1, 25000),
    )
    assert table.frame_gop(1) is None


def test_absent_sync_flags_yield_no_keyframe_indices():
    table = analyze_sample_table(
        _records([(Fraction(i, 25), None) for i in range(3)]),
        nominal_cadence=25,
        source_tick=Fraction(1, 25000),
    )
    assert all(sample.is_sync is None for sample in table.samples)
    assert table.keyframe_indices == ()


@pytest.mark.parametrize("pairs,nominal,tick", [
    ([(Fraction(i, 30), Fraction(1, 30)) for i in range(5)],
     15, Fraction(1, 30000)),
    ([(Fraction(1, 25) * v, Fraction(1, 25)) for v in (0, 1, 3, 4)],
     25, Fraction(1, 25)),
    ([(Fraction(0), None), (Fraction(0), Fraction(1, 30)),
      (Fraction(1, 30), None)],
     30, Fraction(1, 30000)),
    ([(Fraction(10), Fraction(1, 24))], None, Fraction(1, 24000)),
])
def test_pair_view_matches_table_timing(pairs, nominal, tick):
    assert analyze_sample_timing(
        pairs, nominal_cadence=nominal, source_tick=tick,
    ) == analyze_sample_table(
        _records(pairs), nominal_cadence=nominal, source_tick=tick,
    ).timing()


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


def test_short_final_duration_is_a_container_end_artifact():
    # Duration-preserving writers clamp the last sample; editors cut
    # mid-interval. A SHORT final duration must not demote the clock.
    timing = analyze_sample_timing(
        [(Fraction(0), Fraction(1, 30)),
         (Fraction(1, 30), Fraction(1, 30)),
         (Fraction(2, 30), Fraction(1, 30000))],
        nominal_cadence=30,
        source_tick=Fraction(1, 30000),
    )
    assert timing.cadence == Fraction(30)
    # The long-tail pin (silent-truncation test above) still holds: only
    # SHORTER-than-interval final durations are tolerated.


def test_frame_gop_scales_to_long_clips():
    # keyframe_indices is a stored field: per-frame GOP metadata over a
    # feature-length table must be near-instant, not quadratic (the
    # recomputing-property version burned ~26e9 comparisons on 90 min).
    import time

    count = 120_000
    pairs = [(Fraction(i, 30), Fraction(1, 30)) for i in range(count)]
    sync = [i % 30 == 0 for i in range(count)]
    table = analyze_sample_table(
        _records(pairs, sync=sync),
        nominal_cadence=30,
        source_tick=Fraction(1, 30000),
    )
    t0 = time.perf_counter()
    assert table.frame_gop(count - 1) == ((count - 1) % 30, 30)
    for index in range(0, count, 997):
        table.frame_gop(index)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"frame_gop sampling took {elapsed:.2f}s"


def test_epoch_unwrapper_rebases_resets_and_ignores_reordering():
    from kinovsr.media.timing import EpochUnwrapper

    # Reordering dips pass through untouched.
    u = EpochUnwrapper()
    dips = [Fraction(10), Fraction(10) + Fraction(2, 30),
            Fraction(10) + Fraction(1, 30),        # B-frame style dip
            Fraction(10) + Fraction(3, 30)]
    assert [u.push(p) for p in dips] == dips
    assert u.resets == 0

    # A hard reset to zero rebases the new epoch one frame interval
    # after the previous epoch's end.
    u = EpochUnwrapper()
    a = [Fraction(10) + Fraction(k, 30) for k in range(4)]
    out = [u.push(p) for p in a]
    assert out == a
    b = [Fraction(0), Fraction(1, 30), Fraction(2, 30)]
    out_b = [u.push(p) for p in b]
    assert u.resets == 1
    expected_start = a[-1] + Fraction(1, 30)
    assert out_b == [expected_start + p for p in b]
    assert out_b[0] > out[-1]
