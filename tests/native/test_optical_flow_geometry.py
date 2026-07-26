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


def test_advertised_shape_below_the_floor_falls_back_to_full_size():
    """A 640x480 source advertises 160x120; height 120 is never written."""
    from kinovsr.native.optical_flow import select_flow_destination_geometry

    assert select_flow_destination_geometry(_attrs(160, 120), 640, 480) == (
        640,
        480,
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


def test_portrait_fallback_stays_rotation_normalized():
    """Portrait sources write flow in landscape coordinates."""
    from kinovsr.native.optical_flow import select_flow_destination_geometry

    assert select_flow_destination_geometry(_attrs(160, 120), 480, 640) == (
        640,
        480,
        False,
    )


def test_tiny_source_fallback_is_raised_to_the_floor():
    from kinovsr.native.optical_flow import select_flow_destination_geometry

    assert select_flow_destination_geometry(_attrs(16, 12), 64, 48) == (
        128,
        128,
        False,
    )
