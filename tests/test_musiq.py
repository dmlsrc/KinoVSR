"""MUSIQ MLX port tests (skipped when the converted weights are absent).

The port was validated against an own-torch oracle written from the read
spec: end-to-end score parity 3e-4 on a 0-100 scale. These tests pin a
deterministic synthetic score and the metric's core direction (blur scores
lower), so regressions in preprocessing (bicubic, patching, hash positions)
or the net surface as score drift.
"""
import pytest

import mlx.core as mx

from kinovsr.musiq import WEIGHTS_PATH, Musiq

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.is_file(), reason="MUSIQ weights not converted"
)


def _synthetic():
    mx.random.seed(7)
    base = mx.random.uniform(shape=(240, 320))
    k = mx.ones((1, 5, 5, 1)) / 25.0
    base = mx.conv2d(base[None, :, :, None], k, padding=2)[0, :, :, 0]
    return mx.clip(mx.stack([base, base * 0.9 + 0.05, base * 1.1], axis=-1), 0, 1)


def test_pinned_score_and_blur_direction():
    m = Musiq()
    img = _synthetic()
    s_sharp = m.score(img)
    assert abs(s_sharp - 32.924) < 0.1, s_sharp
    blur_k = mx.broadcast_to((mx.ones((5, 5)) / 25)[None, :, :, None],
                             (3, 5, 5, 1)) * mx.eye(3).reshape(3, 1, 1, 3)
    blurred = mx.clip(mx.conv2d(img[None], blur_k, padding=2)[0], 0, 1)
    s_blur = m.score(blurred)
    assert s_blur < s_sharp - 1.0, (s_blur, s_sharp)


def test_batched_matches_single():
    m = Musiq()
    img = _synthetic()
    single = m.score(img)
    batched = m.score_frames([img, img, img], batch=3)
    assert all(abs(b - single) < 1e-3 for b in batched), (single, batched)
