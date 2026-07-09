"""Chroma-fidelity net for the CVPixelBuffer upload path (pixel_buffers.py).

upload_frame_to_buffer is where a decoded frame crosses into CoreVideo: a direct
fp16 memcpy for RGBAHalf, or a CoreImage sRGB->BT.709 render for NV12 (4:2:0,
chroma-subsampled). The numpy -> MLX-native rewrite must not perturb a single
channel value, so this pins the upload -> read-back round trip (sha256 of the
recovered RGB) for every (frame dtype, destination format) pair.

Each case runs with BOTH a numpy frame and the equivalent mlx frame: the hash
must match the golden for both, which proves (a) chroma is unchanged vs the
numpy implementation and (b) the mlx path is byte-faithful to the numpy path.
Captured from the numpy implementation. Skipped where pyobjc/VideoToolbox is
unavailable.
"""
from __future__ import annotations

import hashlib

import mlx.core as mx
import numpy as np
import pytest

from kinovsr import pixel_buffers as pb

_H = _W = 8


def _have_pyobjc() -> bool:
    try:
        from kinovsr import _compat
        _compat.require_pyobjc()
        return True
    except Exception:
        return False


def _np_frame(kind: str) -> np.ndarray:
    if kind == "u8":
        return (np.arange(_H * _W * 3, dtype=np.int32) % 256).astype(np.uint8).reshape(_H, _W, 3)
    return (np.arange(_H * _W * 4, dtype=np.float32) / (_H * _W * 4)).astype(np.float16).reshape(_H, _W, 4)


def _make_buffer(fmt: int):
    from kinovsr._compat import Quartz
    attrs = {
        Quartz.kCVPixelBufferPixelFormatTypeKey: fmt,
        Quartz.kCVPixelBufferWidthKey: _W,
        Quartz.kCVPixelBufferHeightKey: _H,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }
    return pb.make_pixel_buffer_from_attrs(_W, _H, attrs)


# (case id, frame kind, destination format name, golden sha256[:24])
_CASES = [
    ("u8_to_nv12", "u8", "PIX_NV12", "54343cfac1b73415c6894809"),
    ("u8_to_rgbahalf", "u8", "PIX_RGBAHALF", "8b4a544837a1a0280fa8a7c8"),
    ("f16_to_nv12", "f16", "PIX_NV12", "4036f9b99ded79ff15df0cb9"),
    ("f16_to_rgbahalf", "f16", "PIX_RGBAHALF", "b0aa18ac9534756d56021442"),
]


@pytest.mark.skipif(not _have_pyobjc(), reason="pyobjc / VideoToolbox unavailable")
@pytest.mark.parametrize("container", ["numpy", "mlx"])
@pytest.mark.parametrize("case_id,kind,fmt_name,golden", _CASES, ids=[c[0] for c in _CASES])
def test_upload_chroma_roundtrip(container, case_id, kind, fmt_name, golden):
    frame = _np_frame(kind)
    if container == "mlx":
        frame = mx.array(frame)
    buf = _make_buffer(getattr(pb, fmt_name))
    pb.upload_frame_to_buffer(frame, buf)
    rgb = pb.read_pixel_buffer_rgb(buf)
    assert hashlib.sha256(bytes(memoryview(mx.contiguous(rgb)))).hexdigest()[:24] == golden


@pytest.mark.skipif(not _have_pyobjc(), reason="pyobjc / VideoToolbox unavailable")
def test_retype_range_copy_reinterprets_not_rescales():
    """--source-range machinery: same bytes, different range interpretation.

    A video-range-typed 10-bit buffer with Y at code 0 (below video black) and
    neutral chroma must, after _retype_range_copy to the full-range type, hold
    byte-identical planes -- and the YUV->RGB conversion must then read the
    SAME code values differently: below-black (negative, extended-range fp16)
    under the video-range identity, exactly black under full-range.
    """
    from kinovsr import video_reader as vr
    from kinovsr._compat import Quartz, vt

    src = _make_buffer(vr._YUV10_VIDEO)
    # Y plane: code 0. Chroma plane: neutral (10-bit 512, left-justified in
    # 16-bit words = 0x8000, little-endian on disk).
    Quartz.CVPixelBufferLockBaseAddress(src, 0)
    try:
        for p, word in ((0, b"\x00\x00"), (1, b"\x00\x80")):
            rows = Quartz.CVPixelBufferGetHeightOfPlane(src, p)
            bpr = Quartz.CVPixelBufferGetBytesPerRowOfPlane(src, p)
            mv = Quartz.CVPixelBufferGetBaseAddressOfPlane(src, p).as_buffer(rows * bpr)
            mv[:] = word * (rows * bpr // 2)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(src, 0)

    dst = vr._retype_range_copy(src, vr._YUV10_FULL)
    assert Quartz.CVPixelBufferGetPixelFormatType(dst) == vr._YUV10_FULL

    # Planes byte-identical: reinterpretation, not rescale.
    Quartz.CVPixelBufferLockBaseAddress(src, 1)
    Quartz.CVPixelBufferLockBaseAddress(dst, 1)
    try:
        for p in (0, 1):
            rows = Quartz.CVPixelBufferGetHeightOfPlane(src, p)
            sbpr = Quartz.CVPixelBufferGetBytesPerRowOfPlane(src, p)
            dbpr = Quartz.CVPixelBufferGetBytesPerRowOfPlane(dst, p)
            smv = Quartz.CVPixelBufferGetBaseAddressOfPlane(src, p).as_buffer(rows * sbpr)
            dmv = Quartz.CVPixelBufferGetBaseAddressOfPlane(dst, p).as_buffer(rows * dbpr)
            row = min(sbpr, dbpr)
            for r in range(rows):
                assert bytes(smv[r * sbpr:r * sbpr + row]) == bytes(dmv[r * dbpr:r * dbpr + row])
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(dst, 1)
        Quartz.CVPixelBufferUnlockBaseAddress(src, 1)

    # Semantic check via VT's own converter, gray so the matrix guess is moot.
    err, xfer = vt.VTPixelTransferSessionCreate(None, None)
    assert err == 0 and xfer is not None
    means = {}
    for name, buf in (("video", src), ("full", dst)):
        rgb_buf = _make_buffer(pb.PIX_RGBAHALF)
        assert vt.VTPixelTransferSessionTransferImage(xfer, buf, rgb_buf) == 0
        rgb = pb.read_buffer_rgb_f32(rgb_buf)
        means[name] = float(mx.mean(rgb))
    assert means["video"] < -0.03   # (0 - 64) / 876 = -0.073 below-black
    assert abs(means["full"]) < 0.02  # 0 / 1023 = exactly black
