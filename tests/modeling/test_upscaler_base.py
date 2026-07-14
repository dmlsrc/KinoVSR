"""Shared upscaler boundary tests."""

import mlx.core as mx
import pytest


def test_to_rgb_batch_clips_decode_overshoot():
    from kinovsr.modeling.upscaler_base import to_rgb_batch

    # decoded RGBAHalf carries legal YUV->RGB overshoot; every learned
    # upscaler entry must clip it (measured 56x confetti-speck area unclipped)
    fr = mx.array([[[-0.14, 0.5, 1.25, 1.0]] * 4] * 3).reshape(3, 4, 4)
    out = to_rgb_batch(fr)
    assert out.shape == (1, 3, 4, 3)
    assert float(mx.min(out)) >= 0.0
    assert float(mx.max(out)) <= 1.0
    # in-range values untouched
    assert abs(float(out[0, 0, 0, 1]) - 0.5) < 1e-7


@pytest.mark.parametrize(("n_frames", "min_window", "max_window"), [
    (100, 0, 64),
    (100, 24, 0),
    (100, -1, 64),
    (100, 64, 24),
    (100.0, 24, 64),
    (100, 24.0, 64),
])
def test_plan_gop_windows_rejects_invalid_bounds(
        n_frames, min_window, max_window):
    from kinovsr.modeling.upscaler_base import plan_gop_windows

    with pytest.raises(ValueError):
        plan_gop_windows(
            [0, 12, 24], n_frames, min_window, max_window)


def test_plan_gop_windows_dense_keyframes_tile_with_forward_scan():
    from kinovsr.modeling.upscaler_base import plan_gop_windows

    n_frames = 16_000
    windows = plan_gop_windows(
        list(range(n_frames)), n_frames, min_window=16, max_window=64)
    emits = [(emit_start, emit_end)
             for _, _, emit_start, emit_end in windows]
    assert emits[0][0] == 0
    assert emits[-1][1] == n_frames
    assert all(left[1] == right[0]
               for left, right in zip(emits[:-1], emits[1:], strict=True))
