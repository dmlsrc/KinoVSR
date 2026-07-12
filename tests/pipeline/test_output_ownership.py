"""Session output ownership (finding #1). An empty pipeline is the sharpest
case: with no stage, the output payload IS the borrowed input buffer unless
the session copies it. The default (retain_outputs=True) must hand back a
fresh owned buffer; retain_outputs=False is the zero-copy opt-out that
yields the borrowed input as-is. Skipped without pyobjc.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from kinovsr.pipeline import open_pipeline
from kinovsr.processors import (
    FrameUnit,
    Geometry,
    Layout,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.integration


def _have_pyobjc() -> bool:
    try:
        from kinovsr.native import compat as _compat
        _compat.require_pyobjc()
        return True
    except Exception:
        return False


def _cv_spec() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(8, 8),
            layout=Layout.CV_RGBA_HALF),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


def _cv_unit():
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native.compat import Quartz
    attrs = {
        Quartz.kCVPixelBufferPixelFormatTypeKey: pb.PIX_RGBAHALF,
        Quartz.kCVPixelBufferWidthKey: 8,
        Quartz.kCVPixelBufferHeightKey: 8,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }
    buf = pb.make_pixel_buffer_from_attrs(8, 8, attrs)
    return FrameUnit(payload=buf, pts=0, duration=960), buf


@pytest.mark.skipif(not _have_pyobjc(), reason="pyobjc unavailable")
def test_default_session_hands_back_owned_cv_outputs():
    unit, buf = _cv_unit()
    session = open_pipeline({"pipeline": []}, _cv_spec(), settings=Settings())
    with session, session.process([unit]) as run:
        out = list(run)
    assert len(out) == 1
    # Owned: a fresh buffer, not the borrowed input.
    assert out[0].payload is not buf


@pytest.mark.skipif(not _have_pyobjc(), reason="pyobjc unavailable")
def test_zero_copy_session_aliases_the_borrowed_input():
    unit, buf = _cv_unit()
    session = open_pipeline({"pipeline": []}, _cv_spec(), settings=Settings())
    with session, session.process([unit], retain_outputs=False) as run:
        out = list(run)
    assert len(out) == 1
    # Zero-copy: the pass-through payload IS the borrowed input buffer.
    assert out[0].payload is buf
