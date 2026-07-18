"""Native smoke coverage for the VTMotionEstimationSession flow backend."""

from __future__ import annotations

import mlx.core as mx
import pytest


def _textured(h: int, w: int) -> mx.array:
    """Flow-friendly texture shared with the Vision tests: irrational trig
    value noise (block-distinctive for the matcher, no repeats inside the
    search window) plus Gaussian blob anchors."""
    ys, xs = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    xf, yf = xs.astype(mx.float32), ys.astype(mx.float32)
    n = mx.sin(xf * 12.9898 + yf * 78.233) * 43758.5453
    hashn = n - mx.floor(n)
    s2 = (0.22 * min(w, h)) ** 2

    def blob(cx, cy, sign):
        return sign * mx.exp(-((xf - cx * w) ** 2 + (yf - cy * h) ** 2) / s2)

    blobs = (blob(0.30, 0.35, 1.0) + blob(0.72, 0.60, -1.0)
             + blob(0.45, 0.80, 0.8) + blob(0.80, 0.20, -0.7))
    base = 0.5 + (hashn - 0.5) * 0.5 + blobs * 0.25
    rgb = mx.clip(mx.stack([
        base,
        0.45 + (hashn - 0.5) * 0.4 + blobs * 0.22,
        0.55 + (hashn - 0.5) * 0.45 + blobs * 0.20,
    ], axis=-1), 0.0, 1.0).astype(mx.float32)
    mx.eval(rgb)
    return rgb


def _make_engine(w: int, h: int):
    from kinovsr.modeling.vtme_flow import VtmeFlowEngine

    try:
        return VtmeFlowEngine(w, h)
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))


@pytest.mark.integration
@pytest.mark.parametrize("w,h", [(256, 144), (853, 479), (360, 640)])
def test_engine_convention_and_accuracy_on_translation(w, h):
    dx = 5
    sheet = _textured(h, w + dx)
    a = sheet[:, :w]
    b = sheet[:, dx:dx + w]        # window slides right: content moves LEFT
    engine = _make_engine(w, h)
    try:
        flow = engine.compute(a, b)
        assert flow.shape == (h, w, 2)
        assert flow.dtype == mx.float32
        core = flow[h // 4:-h // 4, w // 4:-w // 4]
        mean_x = float(mx.mean(core[..., 0]))
        mean_y = float(mx.mean(core[..., 1]))
        # a[p] ~= b[p + flow[p]]: the content is at b[p - dx], so flow ~ -dx
        assert abs(mean_x + dx) < 1.0, mean_x
        assert abs(mean_y) < 0.75, mean_y
    finally:
        engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.compute(a, b)


@pytest.mark.integration
def test_flat_regions_read_zero():
    w, h = 256, 144
    engine = _make_engine(w, h)
    try:
        flat = mx.full((h, w, 3), 0.5)
        mx.eval(flat)
        flow = engine.compute(flat, flat)
        assert float(mx.max(mx.abs(flow))) == 0.0
    finally:
        engine.close()


@pytest.mark.integration
def test_mc_denoiser_runs_on_vtme_flow():
    from kinovsr.processors.mc import McTemporalDenoiser

    try:
        d = McTemporalDenoiser(96, 64, strength=0.5, window=1,
                               occlusion=True, flow="vtme", self_test=True)
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    try:
        mx.random.seed(0)
        base = _textured(64, 96)
        out1 = d.denoise(mx.clip(
            base + 0.05 * mx.random.normal(shape=base.shape), 0, 1))
        out2 = d.denoise(mx.clip(
            base + 0.05 * mx.random.normal(shape=base.shape), 0, 1))
        mx.eval(out1, out2)
        assert out2.shape == (64, 96, 3)
        assert bool(mx.all(mx.isfinite(out2)))
        # Static content: the flow must unlock a real share of the blend
        # (all-zero or wrong-sign flow reads ~0 here). The absolute level
        # is throttled by the photometric gate at this noise sigma.
        assert d.gate_openness > 0.1
    finally:
        d.close()
