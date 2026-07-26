"""Flow destination selection: prefer the advertised shape above the floor."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def _attrs(width, height):
    from kinovsr.native.frameworks import Quartz

    return {
        Quartz.kCVPixelBufferWidthKey: width,
        Quartz.kCVPixelBufferHeightKey: height,
    }


def test_advertised_shape_is_used_when_it_clears_the_writer_floor():
    """854x480 advertises 240x135; forcing full size over-warps the scaler.

    Vectors are in destination-buffer coordinates, so the destination shape
    scales the field the SR scaler consumes. Measured edge temporal instability
    was 4.89 advertised versus 8.57 forced, at equal spatial detail.
    """
    from kinovsr.native.optical_flow import select_flow_destination_geometry

    assert select_flow_destination_geometry(_attrs(240, 135), 854, 480) == (
        240,
        135,
        True,
    )


def test_advertised_shape_below_the_floor_is_repaired_minimally():
    """A 640x480 source advertises 160x120; height 120 is never written.

    Raise only the offending dimension. Replacing the whole advertisement with
    the source size measured far worse: on a 640x360 clip, edge temporal
    instability was 4.28 at the repaired 160x128 against 7.96 at a forced
    640x360, where non-temporal Image mode scores 4.62.
    """
    from kinovsr.native.optical_flow import select_flow_destination_geometry

    assert select_flow_destination_geometry(_attrs(160, 120), 640, 480) == (
        160,
        128,
        False,
    )
    assert select_flow_destination_geometry(_attrs(160, 90), 640, 360) == (
        160,
        128,
        False,
    )


def test_missing_or_unusable_advertisement_falls_back():
    from kinovsr.native.optical_flow import select_flow_destination_geometry

    for advertised in (None, {}, _attrs("bad", "worse")):
        assert select_flow_destination_geometry(advertised, 854, 480) == (
            854,
            480,
            False,
        )


def test_portrait_repair_keeps_the_advertised_orientation():
    """A portrait source still advertises a landscape flow shape.

    The advertisement already carries VT's rotation-normalized orientation, so
    repairing it needs no axis swap; only the source-sized last resort does.
    """
    from kinovsr.native.optical_flow import select_flow_destination_geometry

    assert select_flow_destination_geometry(_attrs(160, 120), 480, 640) == (
        160,
        128,
        False,
    )


def test_source_sized_last_resort_is_rotation_normalized():
    """With no usable advertisement, fall back to landscape source geometry."""
    from kinovsr.native.optical_flow import select_flow_destination_geometry

    assert select_flow_destination_geometry(None, 480, 640) == (640, 480, False)


def test_tiny_source_fallback_is_raised_to_the_floor():
    from kinovsr.native.optical_flow import select_flow_destination_geometry

    assert select_flow_destination_geometry(_attrs(16, 12), 64, 48) == (
        128,
        128,
        False,
    )
