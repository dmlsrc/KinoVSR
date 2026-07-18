"""vImage histogram binding: exact counts on known planes."""

from __future__ import annotations

import struct

import pytest

from kinovsr.native.vimage import histogram_planarf

pytestmark = pytest.mark.integration


def _plane(values):
    return bytearray(struct.pack(f"<{len(values)}f", *values))


def test_constant_plane_lands_in_one_bin():
    w, h = 8, 4
    hist = histogram_planarf(_plane([0.5] * (w * h)), w, h, 1024)
    assert sum(hist) == w * h
    assert hist[511] + hist[512] == w * h  # 0.5 sits on the 1024-bin midline


def test_two_level_plane_splits_exactly():
    w, h = 16, 2
    values = [0.125] * 20 + [0.875] * 12
    hist = histogram_planarf(_plane(values), w, h, 8)
    assert hist == [0, 20, 0, 0, 0, 0, 0, 12]


def test_size_mismatch_raises():
    with pytest.raises(ValueError, match="bytes of fp32"):
        histogram_planarf(bytearray(10), 4, 4, 256)
