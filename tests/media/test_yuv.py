"""luma_chroma_blend recombination + the 4:2:2 chroma decimation siting."""
from __future__ import annotations

import mlx.core as mx
import pytest

from kinovsr.media import pixel_buffers as pb
from kinovsr.media import yuv
from kinovsr.media.yuv import luma_chroma_blend
from kinovsr.native.frameworks import Quartz

pytestmark = pytest.mark.unit

# Two distinct (1, 2, 3) RGB frames in [0, 1]; every channel differs.
ORIG = mx.array([[[0.80, 0.10, 0.40], [0.20, 0.90, 0.55]]], dtype=mx.float32)
NEW = mx.array([[[0.30, 0.35, 0.30], [0.60, 0.42, 0.50]]], dtype=mx.float32)

_KR, _KB = 0.299, 0.114
_KG = 1.0 - _KR - _KB
_CB_S, _CR_S = 2.0 * (1.0 - _KB), 2.0 * (1.0 - _KR)


def _ycc(x):
    y = _KR * x[..., 0] + _KG * x[..., 1] + _KB * x[..., 2]
    return y, (x[..., 2] - y) / _CB_S, (x[..., 0] - y) / _CR_S


def _close(a, b, atol=1e-5):
    return bool(mx.all(mx.abs(a - b) <= atol).item())


def test_full_strength_returns_new():
    # a_luma = a_chroma = 1 is an exact YCbCr round-trip of `new`.
    assert _close(luma_chroma_blend(ORIG, NEW, 1.0, 1.0), NEW)


def test_zero_strength_returns_orig():
    assert _close(luma_chroma_blend(ORIG, NEW, 0.0, 0.0), ORIG)


def test_luma_only_takes_new_luma_and_keeps_original_chroma():
    # a_luma=1 (new luma), a_chroma=0 (original chroma): the split a single
    # joint RGB blend cannot do.
    out = luma_chroma_blend(ORIG, NEW, 1.0, 0.0)
    y_out, cb_out, cr_out = _ycc(out)
    y_new, _, _ = _ycc(NEW)
    _, cb_orig, cr_orig = _ycc(ORIG)
    assert _close(y_out, y_new)
    assert _close(cb_out, cb_orig)
    assert _close(cr_out, cr_orig)


def test_output_is_clipped_to_the_unit_range():
    over = mx.array([[[2.0, -1.0, 0.5], [1.5, 0.5, -0.3]]], dtype=mx.float32)
    out = luma_chroma_blend(ORIG, over, 1.0, 1.0)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_coefficients_only_matter_when_luma_differs_from_chroma():
    # Equal strengths collapse to a plain RGB lerp: the YCbCr basis cancels,
    # so BT.601 and BT.709 coefficients give the same answer.
    same_601 = luma_chroma_blend(ORIG, NEW, 0.5, 0.5, 0.299, 0.114)
    same_709 = luma_chroma_blend(ORIG, NEW, 0.5, 0.5, 0.2126, 0.0722)
    assert _close(same_601, same_709)
    # With luma != chroma the coefficients change the result.
    split_601 = luma_chroma_blend(ORIG, NEW, 0.2, 1.0, 0.299, 0.114)
    split_709 = luma_chroma_blend(ORIG, NEW, 0.2, 1.0, 0.2126, 0.0722)
    assert not _close(split_601, split_709)


def _active_yuv_bytes(buffer, width):
    planes = []
    Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
    try:
        for plane in range(Quartz.CVPixelBufferGetPlaneCount(buffer)):
            rows = Quartz.CVPixelBufferGetHeightOfPlane(buffer, plane)
            bpr = Quartz.CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
            base = Quartz.CVPixelBufferGetBaseAddressOfPlane(buffer, plane)
            raw = base.as_buffer(rows * bpr)
            row_bytes = width * 2
            planes.append(b"".join(
                bytes(raw[row * bpr:row * bpr + row_bytes])
                for row in range(rows)))
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)
    return tuple(planes)


@pytest.mark.parametrize("matrix", [
    Quartz.kCVImageBufferYCbCrMatrix_ITU_R_601_4,
    Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2,
    Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020,
])
@pytest.mark.parametrize("full_range", [False, True])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_direct_yuv_matches_the_legacy_rgbahalf_boundary(
        matrix, full_range, dtype):
    width, height = 8, 6
    values = mx.array([
        -0.1, 0.0, 0.001, 0.12345, 0.5, 0.87654, 0.999, 1.1,
    ], dtype=dtype)
    rgb = mx.stack([
        mx.broadcast_to(values[None, :, None], (height, width, 1)),
        mx.broadcast_to(values[::-1][None, :, None], (height, width, 1)),
        mx.broadcast_to(mx.roll(values, 2)[None, :, None], (height, width, 1)),
    ], axis=-1).reshape(height, width, 3)
    rgba = pb.make_pixel_buffer_from_attrs(width, height, {
        "PixelFormatType": pb.PIX_RGBAHALF,
        "Width": width, "Height": height,
        "IOSurfaceProperties": {}, "MetalCompatibility": True,
    })
    alpha = mx.ones((height, width, 1), dtype=mx.float16)
    pb.write_fp16_rgba(mx.concatenate([
        rgb[..., :3].astype(mx.float16), alpha], axis=-1), rgba)
    staged = pb.read_rgbahalf_rgb(rgba)

    attrs = {
        "PixelFormatType": yuv.pixel_format(full_range),
        "Width": width, "Height": height,
        "IOSurfaceProperties": {}, "MetalCompatibility": True,
    }
    legacy = pb.make_pixel_buffer_from_attrs(width, height, attrs)
    direct = pb.make_pixel_buffer_from_attrs(width, height, attrs)
    yuv.rgb_to_yuv422_10(staged, legacy, matrix, full_range)
    equivalent = rgb[..., :3].astype(mx.float16).astype(mx.float32)
    yuv.rgb_to_yuv422_10(equivalent, direct, matrix, full_range)

    assert _active_yuv_bytes(direct, width) == _active_yuv_bytes(legacy, width)


def test_chroma_decimation_is_cosited_121():
    """4:2:2 chroma must be the left/co-sited [1,2,1]/4 filter: HEVC cannot
    signal any other siting for 4:2:2, and decoders assume co-sited."""
    from kinovsr.media.yuv import _compiled_planes

    mx.random.seed(9)
    H, W = 4, 16
    rgb = mx.random.uniform(shape=(H, W, 3))
    Kr, Kb = 0.2126, 0.0722
    _, chroma = _compiled_planes(Kr, Kb, False)(rgb)
    cb = (chroma.reshape(H, W // 2, 2)[..., 0] >> 6).astype(mx.float32)

    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = Kr * R + (1.0 - Kr - Kb) * G + Kb * B
    c = ((B - Y) / (2.0 * (1.0 - Kb))) * 896.0 + 512.0
    left = mx.concatenate([c[:, :1], c[:, 1:-1:2]], axis=1)
    want = mx.clip(mx.round((left + 2.0 * c[:, 0::2] + c[:, 1::2]) * 0.25),
                   0, 1023)
    assert float(mx.max(mx.abs(cb - want))) <= 1.0

    # constants are preserved exactly (the filter weights sum to 1)
    flat = mx.full((2, 8, 3), 0.25)
    _, chroma_c = _compiled_planes(Kr, Kb, False)(flat)
    cbc = (chroma_c.reshape(2, 4, 2)[..., 0] >> 6).astype(mx.float32)
    assert float(mx.max(cbc) - mx.min(cbc)) == 0.0
