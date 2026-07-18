"""Native smoke coverage for the vtme (trackability) cut-detect mode.

The discriminator under test: consecutive frames stay block-coherent
even when content moves or global gain swings; a cut free-associates
the matcher. Runs only where VTMotionEstimationSession exists.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinovsr.processors.cut_detect import CutDetector


def _textured(h: int, w: int, fx: float = 12.9898, fy: float = 78.233):
    ys, xs = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    xf, yf = xs.astype(mx.float32), ys.astype(mx.float32)
    n = mx.sin(xf * fx + yf * fy) * 43758.5453
    hashn = n - mx.floor(n)
    rgb = mx.clip(mx.stack([
        0.5 + (hashn - 0.5) * 0.5,
        0.45 + (hashn - 0.5) * 0.4,
        0.55 + (hashn - 0.5) * 0.45,
    ], axis=-1), 0.0, 1.0).astype(mx.float32)
    mx.eval(rgb)
    return rgb


def _detector(w: int = 256, h: int = 144) -> CutDetector:
    det = CutDetector("vtme", 0.07)
    probe = mx.zeros((h, w, 3), dtype=mx.float32)
    try:
        det.is_cut(probe)  # builds the session lazily
    except (RuntimeError, SystemExit) as exc:
        det.close()
        pytest.skip(str(exc))
    det.reset_state()
    return det


@pytest.mark.integration
def test_cut_fires_and_motion_flicker_do_not():
    h, w, dx = 144, 256, 4
    sheet_a = _textured(h, w + dx)
    scene_a = sheet_a[:, :w]
    scene_a_moved = sheet_a[:, dx:dx + w]
    scene_b = _textured(h, w, fx=7.1234, fy=41.717)
    det = _detector(w, h)
    try:
        assert det.is_cut(scene_a) is False              # first frame
        assert det.is_cut(scene_a_moved) is False        # global motion
        assert det.is_cut(scene_b) is True               # hard cut
        dark = mx.clip(scene_b * 0.75, 0, 1)
        bright = mx.clip(scene_b * 1.30, 0, 1)
        mx.eval(dark, bright)
        assert det.is_cut(dark) is False                 # gain swing down
        assert det.is_cut(bright) is False               # gain swing up
    finally:
        det.close()


@pytest.mark.integration
def test_reset_state_restarts_comparison_and_keeps_session():
    h, w = 144, 256
    scene_a = _textured(h, w)
    scene_b = _textured(h, w, fx=7.1234, fy=41.717)
    det = _detector(w, h)
    try:
        assert det.is_cut(scene_a) is False
        engine = det._engine
        det.reset_state()
        # Post-reset the next frame has no comparison partner: no fire
        # even against unrelated content, and the session is reused.
        assert det.is_cut(scene_b) is False
        assert det._engine is engine
        assert det.is_cut(scene_a) is True               # cut still detected
    finally:
        det.close()
    assert det._engine is None
