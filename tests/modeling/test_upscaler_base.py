"""Shared upscaler boundary tests."""

import mlx.core as mx


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
