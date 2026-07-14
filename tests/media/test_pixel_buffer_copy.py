"""copy_pixel_buffer: a format-agnostic CVPixelBuffer deep copy that hands a
session consumer an OWNED output (finding #1). Proves the copy is a distinct
buffer whose contents survive the source being overwritten - packed
(RGBAHalf) and planar (NV12) both, so both plane-copy paths are covered.
"""
from __future__ import annotations

import struct

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


def test_copy_propagates_public_attachments_and_drops_private_ones():
    from kinovsr.native.frameworks import Foundation, Quartz

    src = _make("PIX_RGBAHALF")
    mastering = struct.pack(
        ">HHHHHHHHII",
        8_500, 39_850,
        6_550, 2_300,
        35_400, 14_600,
        15_635, 16_450,
        10_000_000, 50,
    )
    content_light = struct.pack(">HH", 1_000, 400)
    propagated = {
        Quartz.kCVImageBufferColorPrimariesKey:
            Quartz.kCVImageBufferColorPrimaries_P3_D65,
        Quartz.kCVImageBufferTransferFunctionKey:
            Quartz.kCVImageBufferTransferFunction_ITU_R_2100_HLG,
        Quartz.kCVImageBufferYCbCrMatrixKey:
            Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020,
        Quartz.kCVImageBufferMasteringDisplayColorVolumeKey:
            Foundation.NSData.dataWithBytes_length_(mastering, len(mastering)),
        Quartz.kCVImageBufferContentLightLevelInfoKey:
            Foundation.NSData.dataWithBytes_length_(content_light, len(content_light)),
        Quartz.kCVImageBufferCleanApertureKey: {
            Quartz.kCVImageBufferCleanApertureWidthKey: 8,
            Quartz.kCVImageBufferCleanApertureHeightKey: 8,
            Quartz.kCVImageBufferCleanApertureHorizontalOffsetKey: 0,
            Quartz.kCVImageBufferCleanApertureVerticalOffsetKey: 0,
        },
        Quartz.kCVImageBufferPixelAspectRatioKey: {
            Quartz.kCVImageBufferPixelAspectRatioHorizontalSpacingKey: 4,
            Quartz.kCVImageBufferPixelAspectRatioVerticalSpacingKey: 3,
        },
        "KinoVSRTestPropagated": "kept",
    }
    for key, value in propagated.items():
        Quartz.CVBufferSetAttachment(
            src, key, value, Quartz.kCVAttachmentMode_ShouldPropagate
        )
    Quartz.CVBufferSetAttachment(
        src,
        "KinoVSRTestPrivate",
        "dropped",
        Quartz.kCVAttachmentMode_ShouldNotPropagate,
    )
    source_attachments = dict(
        Quartz.CVBufferCopyAttachments(
            src, Quartz.kCVAttachmentMode_ShouldPropagate
        )
        or {}
    )
    assert set(propagated) <= set(source_attachments)

    dst = pb.copy_pixel_buffer(src)
    copied = dict(
        Quartz.CVBufferCopyAttachments(
            dst, Quartz.kCVAttachmentMode_ShouldPropagate
        )
        or {}
    )

    for key, value in propagated.items():
        assert copied[key] == value
    private = Quartz.CVBufferCopyAttachments(
        dst, Quartz.kCVAttachmentMode_ShouldNotPropagate
    )
    assert not private or "KinoVSRTestPrivate" not in private
