"""Exact target-grid phase math for VideoToolbox frame conversion."""

from __future__ import annotations

from fractions import Fraction

import pytest

from kinovsr.native.temporal import VtfrcSession

pytestmark = pytest.mark.unit


def test_negative_context_indices_keep_target_zero_at_the_public_inpoint():
    session = VtfrcSession.__new__(VtfrcSession)
    session.source_cadence = Fraction(25)
    session.target_cadence = Fraction(40)
    session._next_target_index = -6

    emitted = []
    for source_index in range(-4, 1):
        indices = session._target_indices_in_pair(source_index)
        emitted.extend(indices)
        if indices:
            session._next_target_index = indices[-1] + 1

    assert emitted == list(range(-6, 2))
    assert session._phases_for_targets([0, 1], 0) == [0.0, 0.625]
