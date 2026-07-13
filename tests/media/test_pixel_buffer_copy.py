"""copy_pixel_buffer: a format-agnostic CVPixelBuffer deep copy that hands a
session consumer an OWNED output (finding #1). Proves the copy is a distinct
buffer whose contents survive the source being overwritten - packed
(RGBAHalf) and planar (NV12) both, so both plane-copy paths are covered.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from kinovsr.media import pixel_buffers as pb

pytestmark = pytest.mark.integration

_H = _W = 8

def _make(fmt_name: str):
    from kinovsr.native.frameworks import Quartz
    attrs = {
        Quartz.kCVPixelBufferPixelFormatTypeKey: getattr(pb, fmt_name),
        Quartz.kCVPixelBufferWidthKey: _W,
        Quartz.kCVPixelBufferHeightKey: _H,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }
    return pb.make_pixel_buffer_from_attrs(_W, _H, attrs)


def _frame(scale: float):
    a = np.arange(_H * _W * 4, dtype=np.float32) / (_H * _W * 4) * scale
    return mx.array(a.astype(np.float16).reshape(_H, _W, 4))


def _rgb_bytes(buf) -> bytes:
    return bytes(memoryview(mx.contiguous(pb.read_pixel_buffer_rgb(buf))))


@pytest.mark.parametrize("fmt_name", ["PIX_RGBAHALF", "PIX_NV12"])
def test_copy_is_distinct_and_owns_its_pixels(fmt_name):
    src = _make(fmt_name)
    pb.upload_frame_to_buffer(_frame(1.0), src)
    before = _rgb_bytes(src)

    dst = pb.copy_pixel_buffer(src)
    assert dst is not src
    # The copy holds the same pixels the source had.
    assert _rgb_bytes(dst) == before

    # Overwriting the SOURCE must not change the copy - that is the whole
    # point of the retain-safe default (the input owner may recycle its
    # buffer after handing it in).
    pb.upload_frame_to_buffer(_frame(0.25), src)
    assert _rgb_bytes(src) != before          # source really changed
    assert _rgb_bytes(dst) == before          # copy is unaffected
