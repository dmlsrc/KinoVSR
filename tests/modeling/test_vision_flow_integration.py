"""Native smoke coverage for the Vision r1 optical-flow backend."""

from __future__ import annotations

import mlx.core as mx
import pytest


def _textured(h: int, w: int) -> mx.array:
    """Flow-friendly texture: irrational trig value noise plus Gaussian
    blob anchors. Modular-arithmetic hashes admit EXACT false matches at
    small displacement combinations and must not be used as flow-test
    content (see McTemporalDenoiser._self_test_frames_vision)."""
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


@pytest.mark.integration
def test_engine_convention_and_accuracy_on_translation():
    from kinovsr.modeling.vision_flow import VisionFlowEngine

    h, w, dx = 144, 256, 5
    sheet = _textured(h, w + dx)
    a = sheet[:, :w]
    b = sheet[:, dx:dx + w]        # window slides right: content moves LEFT
    try:
        engine = VisionFlowEngine(w, h)
    except RuntimeError as exc:
        pytest.skip(str(exc))
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
def test_mc_denoiser_runs_on_vision_flow():
    from kinovsr.processors.mc import McTemporalDenoiser

    try:
        d = McTemporalDenoiser(96, 64, strength=0.5, window=1,
                               occlusion=True, flow="vision", self_test=True)
    except RuntimeError as exc:
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


@pytest.mark.integration
def test_sr_flow_mode_vision_computes_both_directions():
    from kinovsr.modeling.vsr_blocks import _compute_flows
    from kinovsr.modeling.vt_flow import VtFlowServices

    height, width = 144, 192
    sheet = _textured(height, width + 3)
    first = sheet[:, :width][None].astype(mx.float16)
    second = sheet[:, 3:3 + width][None].astype(mx.float16)
    mx.eval(first, second)
    services = VtFlowServices(1)
    try:
        forward, backward = _compute_flows(
            [first, second], {}, flow_mode="vision",
            vt_flow_services=services)
        mx.eval(forward[0], backward[0])
        assert forward[0].shape == backward[0].shape == (1, height, width, 2)
        assert forward[0].dtype == mx.float16
        # forward[0] pulls frame 0 into frame 1's geometry: content moved
        # LEFT by 3, so the pull-flow points +3 in x
        core = forward[0][0, height // 4:-height // 4, width // 4:-width // 4]
        assert abs(float(mx.mean(core[..., 0].astype(mx.float32))) - 3.0) < 1.0
        assert services.size == 1
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    finally:
        services.close()
    assert services.size == 0
