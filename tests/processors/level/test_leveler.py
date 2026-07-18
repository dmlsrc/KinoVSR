"""Behavior tests for the histogram leveler driver.

vImage runs on every macOS install (ctypes into Accelerate), so these
are not availability-guarded; they exercise the real native histogram.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from kinovsr.processors.level import HistogramLeveler

pytestmark = pytest.mark.integration

_LUMA = mx.array([0.2126, 0.7152, 0.0722])


def _textured(h=96, w=128, seed=3):
    mx.random.seed(seed)
    rgb = mx.clip(mx.random.uniform(0, 1, (h, w, 3)) * 0.5 + 0.25, 0, 1)
    mx.eval(rgb)
    return rgb


def _run(driver, frames):
    outs = []
    for f in frames:
        outs.extend(driver.feed(f, token=object()))
    outs.extend(driver.flush())
    return [o for o, _ in outs]


def _pumping(frames):
    means = [float(mx.mean(f[..., :3] @ _LUMA)) for f in frames]
    return sum(abs(means[i] - means[i - 1])
               for i in range(1, len(means))) / (len(means) - 1)


def test_oscillatory_pumping_is_removed():
    base = _textured()
    frames = [mx.clip(base * (1.0 + 0.12 * math.sin(2 * math.pi * t / 4)),
                      0, 1) for t in range(24)]
    mx.eval(*frames)
    out = _run(HistogramLeveler(window=5, deadband=0.003), frames)
    assert len(out) == len(frames)
    assert _pumping(out) < _pumping(frames) / 5.0


def test_stable_content_passes_through_bit_exact():
    base = _textured()
    out = _run(HistogramLeveler(), [base] * 12)
    assert all(o is base for o in out)


def test_fade_is_followed_not_fought():
    base = _textured()
    n = 24
    faded = [mx.clip(base * (1.0 - 0.7 * t / (n - 1)), 0, 1)
             for t in range(n)]
    mx.eval(*faded)
    out = _run(HistogramLeveler(window=5, deadband=0.003), faded)
    # The centered reference tracks the trend: output luma must stay
    # close to the intended fade, not get pulled toward a constant.
    for o, f in zip(out, faded, strict=True):
        in_mean = float(mx.mean(f[..., :3] @ _LUMA))
        out_mean = float(mx.mean(o[..., :3] @ _LUMA))
        assert abs(out_mean - in_mean) < 0.02, (in_mean, out_mean)


def test_reset_drops_the_window():
    base = _textured()
    dark = mx.clip(base * 0.5, 0, 1)
    mx.eval(dark)
    driver = HistogramLeveler(window=2, deadband=0.003)
    for _ in range(5):
        driver.feed(base, token=object())
    driver.reset()
    # Post-reset the old bright window is gone: dark frames form their
    # own reference and pass the deadband untouched.
    out = _run(driver, [dark] * 8)
    assert all(o is dark for o in out)


def test_diagnostics_report_corrected_counts():
    base = _textured()
    frames = [mx.clip(base * (1.0 + (0.1 if t % 2 else -0.1)), 0, 1)
              for t in range(12)]
    mx.eval(*frames)
    driver = HistogramLeveler(window=3, deadband=0.003)
    _run(driver, frames)
    lines = driver.run_diagnostics()
    assert len(lines) == 1 and "pumping meter" in lines[0]
    # The first and last frame self-reference (symmetric boundary
    # shrink, m=0) and are never corrected; every interior frame is.
    assert "corrected 10/12" in lines[0]
