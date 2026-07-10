"""DOVER-Mobile MLX port tests (skipped when converted weights are absent).

The port was validated against an own-torch oracle written from the read
spec: branch-score parity 1e-7 on injected identical views, end-to-end
agreement to 4 decimals on real fixtures.  These tests pin deterministic
synthetic scores and the metric's temporal direction (low-frequency
luma flicker scores lower), so regressions in sampling, the fragment
mosaic, the antialiased resize, or the net surface as score drift.

Note the sampling interval is 2: a flicker alternating every frame is
invisible to DOVER by construction (all sampled frames share a phase),
which is why the direction test pulses with period 5.
"""
import math

import mlx.core as mx
import pytest

from kinovsr.dover import WEIGHTS_PATH, DoverMobile

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.is_file(), reason="DOVER-Mobile weights not converted"
)


def _clip(n=128, h=280, w=360, gains=None):
    yy = mx.arange(h).reshape(h, 1, 1).astype(mx.float32)
    xx = mx.arange(w).reshape(1, w, 1).astype(mx.float32)
    base = (110 + 60 * mx.sin(yy / 9.0) * mx.cos(xx / 13.0)
            + 40 * mx.sin((yy + xx) / 23.0))
    frames = []
    for i in range(n):
        f = mx.broadcast_to(base, (h, w, 1)) * mx.ones((1, 1, 3))
        x0 = 20 + (i * 2) % (w - 80)
        box = mx.zeros((h, w, 1))
        box[60:120, x0:x0 + 60] = 60.0
        f = f + box
        if gains is not None:
            f = f * gains[i]
        frames.append(mx.clip(f, 0, 255))
    return mx.stack(frames).astype(mx.uint8)


def test_pinned_scores_and_flicker_direction():
    m = DoverMobile()
    clean = m.score(_clip())
    assert abs(clean["tech"] - 0.1005) < 0.01
    assert abs(clean["aes"] - (-0.0820)) < 0.01
    assert abs(clean["fused"] - 0.4810) < 0.02

    gains = [1.0 + 0.10 * math.sin(2 * math.pi * i / 5.0) for i in range(128)]
    flick = m.score(_clip(gains=gains))
    assert flick["fused"] < clean["fused"] - 0.003
    assert flick["tech"] < clean["tech"]


def test_short_video_wrap_path():
    # 40 frames < the 63 needed for spread offsets: exercises the
    # zero-offset + index-wrap path of the reference sampler.
    m = DoverMobile()
    s = m.score(_clip(n=40))
    assert 0.0 < s["fused"] < 1.0
    assert math.isfinite(s["tech"]) and math.isfinite(s["aes"])
