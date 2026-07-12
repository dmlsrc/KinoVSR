"""luma_chroma_blend: per-channel-group recombination of two RGB frames."""
from __future__ import annotations

import mlx.core as mx
import pytest

from kinovsr.media.yuv import luma_chroma_blend

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
