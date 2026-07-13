"""Crop geometry and slicing tests."""

import mlx.core as mx
import pytest


def test_crop_rgb():
    from kinovsr.processors.crop.geometry import crop_rgb

    fr = mx.arange(6 * 8 * 3, dtype=mx.float32).reshape(6, 8, 3)
    out = crop_rgb(fr, (1, 2, 3, 0))
    assert out.shape == (3, 5, 3)
    assert float(mx.max(mx.abs(out - fr[1:4, 3:]))) == 0.0
    batched = crop_rgb(fr[None], (1, 2, 3, 0))
    assert batched.shape == (1, 3, 5, 3)


def test_compute_aspect_crop():
    from kinovsr.processors.crop.geometry import compute_aspect_crop

    # 16:9 window on a 4:3 frame: full width, centered vertically.
    assert compute_aspect_crop(640, 480, 16, 9) == (60, 60, 0, 0)
    # 9:16 portrait extract from 16:9: nearest even width, centered.
    t, b, left, r = compute_aspect_crop(1920, 1080, 9, 16)
    assert (left + r) + (1920 - left - r) == 1920
    w = 1920 - left - r
    h = 1080 - t - b
    assert w % 2 == 0 and h % 2 == 0
    assert abs(w / h - 9 / 16) < 0.01
    assert abs(left - r) <= 2  # centered
    # offsets shift and clamp
    t2, b2, l2, r2 = compute_aspect_crop(1920, 1080, 9, 16, dx=-10000)
    assert l2 == 0 and r2 == left + r
    # same aspect as the frame: no crop
    assert compute_aspect_crop(640, 480, 4, 3) == (0, 0, 0, 0)
    with pytest.raises(ValueError, match="positive"):
        compute_aspect_crop(640, 480, 0, 9)


def test_compute_aspect_crop_anchors():
    from kinovsr.processors.crop.geometry import compute_aspect_crop

    # 16:9 on 4:3 (640x480 -> 640x360): the vertical slack is 120.
    assert compute_aspect_crop(640, 480, 16, 9, anchor="top") == (0, 120, 0, 0)
    assert compute_aspect_crop(640, 480, 16, 9, anchor="bottom") == (120, 0, 0, 0)
    assert compute_aspect_crop(640, 480, 16, 9, anchor="center") == (60, 60, 0, 0)
    # corners on a window with slack in both axes: 1:1 on 640x480 -> 480x480.
    assert compute_aspect_crop(640, 480, 1, 1, anchor="top-left") == (0, 0, 0, 160)
    assert compute_aspect_crop(640, 480, 1, 1, anchor="bottom-right") == (0, 0, 160, 0)
    assert compute_aspect_crop(640, 480, 1, 1, anchor="right") == (0, 0, 160, 0)
    # offset nudges from the anchor and clamps.
    assert compute_aspect_crop(640, 480, 16, 9, anchor="bottom", dy=-20) == (100, 20, 0, 0)
    assert compute_aspect_crop(640, 480, 16, 9, anchor="bottom", dy=50) == (120, 0, 0, 0)
    with pytest.raises(ValueError, match="anchor"):
        compute_aspect_crop(640, 480, 16, 9, anchor="middle-ish")


def test_compute_aspect_crop_picks_closest_even_fit():
    from kinovsr.processors.crop.geometry import compute_aspect_crop

    # 16:9 in storage px on 348x288: even boxes can only approximate;
    # 344x194 (-0.26%) beats 346x194 (+0.32%) and 348x194 (+0.90%).
    assert compute_aspect_crop(348, 288, 16, 9) == (47, 47, 2, 2)
    # Display 16:9 on 128:117 anamorphic pixels = storage 13:8 (16*117:9*128
    # reduced): full width, 214 rows.
    assert compute_aspect_crop(348, 288, 16 * 117, 9 * 128) == (37, 37, 0, 0)
