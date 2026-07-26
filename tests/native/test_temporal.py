"""Exact target-grid phase math for VideoToolbox frame conversion."""

from __future__ import annotations

from fractions import Fraction

import pytest

from kinovsr.native.temporal import VtfrcSession

pytestmark = pytest.mark.unit


def _bare_session(source=Fraction(25), target=Fraction(40)):
    session = VtfrcSession.__new__(VtfrcSession)
    session.source_cadence = source
    session.target_cadence = target
    session._next_target_index = None
    session._submission_needs_random = False
    return session


def test_negative_context_indices_keep_target_zero_at_the_public_inpoint():
    session = _bare_session()
    session._next_target_index = -6

    emitted = []
    for source_index in range(-4, 1):
        prev_t = Fraction(source_index) / session.source_cadence
        curr_t = Fraction(source_index + 1) / session.source_cadence
        indices = session._targets_between(prev_t, curr_t)
        emitted.extend(indices)
        if indices:
            session._next_target_index = indices[-1] + 1

    assert emitted == list(range(-6, 2))
    assert session._phases_between(
        [0, 1], Fraction(0), Fraction(1, 25)) == [0.0, 0.625]


def test_time_based_pairs_reproduce_the_index_grid_exactly():
    # The index shim's times (N/source) must produce the same targets and
    # bit-identical phases as the historical integer arithmetic.
    session = _bare_session(Fraction(30000, 1001), Fraction(60))
    got = []
    for n in range(24):
        prev_t = Fraction(n) / session.source_cadence
        curr_t = Fraction(n + 1) / session.source_cadence
        indices = session._targets_between(prev_t, curr_t)
        phases = session._phases_between(indices, prev_t, curr_t)
        got.append((indices, phases))
        if indices:
            session._next_target_index = indices[-1] + 1
    # legacy formulation, computed independently
    import math as m
    legacy_next = None
    for n, (indices, phases) in enumerate(got):
        lower = m.ceil(Fraction(n) * session.target_cadence / session.source_cadence)
        upper = m.ceil(Fraction(n + 1) * session.target_cadence / session.source_cadence)
        start = lower if legacy_next is None else max(lower, legacy_next)
        expect = list(range(start, upper))
        assert indices == expect
        src_time = Fraction(n) / session.source_cadence
        denom = 1 / session.source_cadence
        expect_ph = []
        for mm in expect:
            p = float((Fraction(mm) / session.target_cadence - src_time) / denom)
            expect_ph.append(min(max(p, 0.0), 1.0 - 1e-9))
        assert phases == expect_ph
        if expect:
            legacy_next = expect[-1] + 1


def test_non_uniform_pair_brackets_by_real_times():
    # A dropped-frame gap: source frames at 0 and 3/30 on a 30 fps grid,
    # target 30 fps. The pair must emit targets 0, 1, 2 with phases at
    # thirds - interpolation ACROSS the gap, impossible on index math.
    session = _bare_session(Fraction(30), Fraction(30))
    indices = session._targets_between(Fraction(0), Fraction(3, 30))
    assert indices == [0, 1, 2]
    phases = session._phases_between(indices, Fraction(0), Fraction(3, 30))
    assert phases[0] == 0.0
    assert abs(phases[1] - 1 / 3) < 1e-12
    assert abs(phases[2] - 2 / 3) < 1e-12


def test_non_increasing_times_are_refused():
    session = _bare_session()
    with pytest.raises(RuntimeError, match="strictly increasing"):
        session._phases_between([0], Fraction(1, 25), Fraction(1, 25))


def test_duplicate_source_times_are_refused_not_dropped():
    # A tied stamp used to swap the buffered frame out through the
    # empty-target fast path, silently discarding a source frame.
    session = _bare_session(Fraction(30), Fraction(60))
    session._prev_src_pb = object()
    session._prev_time = Fraction(0)
    with pytest.raises(RuntimeError, match="strictly increasing"):
        list(session.feed_at(object(), Fraction(0)))


def test_reset_requires_a_drained_pair_and_marks_next_submission_random():
    session = _bare_session()
    session._prev_src_pb = object()
    with pytest.raises(RuntimeError, match=r"drain\(\)"):
        session.reset_temporal_context()

    session._prev_src_pb = None
    session.reset_temporal_context()
    assert session._submission_needs_random is True


def test_pair_without_targets_marks_next_submission_random():
    session = _bare_session(Fraction(24), Fraction(12))
    session._prev_src_pb = object()
    session._prev_time = Fraction(1, 24)
    session._next_target_index = 1
    next_buffer = object()

    assert list(session.feed_at(next_buffer, Fraction(2, 24))) == []
    assert session._prev_src_pb is next_buffer
    assert session._submission_needs_random is True


def test_empty_drain_marks_reused_session_random():
    session = _bare_session(Fraction(24), Fraction(12))
    session._prev_src_pb = object()
    session._prev_time = Fraction(1, 24)
    session._next_target_index = 1

    assert list(session.drain()) == []
    assert session._prev_src_pb is None
    assert session._submission_needs_random is True
