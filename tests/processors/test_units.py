"""FrameUnit and Boundary value semantics."""

import dataclasses

import pytest

from kinovsr.processors import Boundary, BoundaryKind, FrameUnit

pytestmark = pytest.mark.unit


def test_units_are_frozen_values():
    unit = FrameUnit(payload=object(), pts=0, duration=960)
    with pytest.raises(dataclasses.FrozenInstanceError):
        unit.pts = 1


def test_with_payload_keeps_time_and_boundaries():
    cut = Boundary(BoundaryKind.HARD_CUT, source_index=12)
    unit = FrameUnit(payload="a", pts=960, duration=960, boundaries=(cut,))
    replaced = unit.with_payload("b")
    assert replaced.payload == "b"
    assert (replaced.pts, replaced.duration) == (960, 960)
    assert replaced.boundaries == (cut,)
    assert unit.payload == "a"  # original untouched


def test_retimed_rewrites_pts_and_optionally_duration():
    unit = FrameUnit(payload="a", pts=0, duration=960)
    shifted = unit.retimed(480)
    assert (shifted.pts, shifted.duration) == (480, 960)
    halved = unit.retimed(480, 480)
    assert (halved.pts, halved.duration) == (480, 480)


def test_with_boundary_appends():
    unit = FrameUnit(payload="a", pts=0, duration=1)
    start = Boundary(BoundaryKind.STREAM_START)
    cut = Boundary(BoundaryKind.HARD_CUT, source_index=0)
    both = unit.with_boundary(start).with_boundary(cut)
    assert both.boundaries == (start, cut)
