"""Shared VSR modeling primitive tests."""

import mlx.core as mx
import pytest

from kinovsr.modeling.vsr_blocks import _compute_flows, box3, history_improve_gate


def test_compute_flows_zero_mode_returns_zero_fields():
    frames = [
        mx.zeros((1, 4, 5, 3), dtype=mx.float16),
        mx.ones((1, 4, 5, 3), dtype=mx.float16),
    ]

    flows_forward, flows_backward = _compute_flows(frames, {}, flow_mode="zero")
    mx.eval(flows_forward[0], flows_backward[0])

    assert flows_forward[0].shape == (1, 4, 5, 2)
    assert flows_backward[0].shape == (1, 4, 5, 2)
    assert float(mx.sum(mx.abs(flows_forward[0]))) == 0.0
    assert float(mx.sum(mx.abs(flows_backward[0]))) == 0.0


def test_compute_flows_rejects_unknown_flow_mode():
    frames = [
        mx.zeros((1, 4, 5, 3), dtype=mx.float16),
        mx.ones((1, 4, 5, 3), dtype=mx.float16),
    ]

    with pytest.raises(ValueError, match="unknown flow_mode"):
        _compute_flows(frames, {}, flow_mode="bogus")


def test_history_improve_gate_closes_on_static_content():
    # Identical frames + zero flow: warping cannot improve the residual, so the
    # gate must close (this is the anti-etch property).
    mx.random.seed(0)
    curr = mx.random.uniform(shape=(1, 12, 16, 3))
    flow = mx.zeros((1, 12, 16, 2))
    gate = history_improve_gate(curr, curr, flow, mx.float32)
    mx.eval(gate)
    assert gate.shape == (1, 12, 16, 1)
    assert float(mx.max(gate)) == 0.0


def test_history_improve_gate_opens_on_well_tracked_motion():
    # prev shifted by exactly +2 px, flow pointing back at it: the warp
    # reconstructs curr almost exactly while the unwarped residual is large,
    # so interior gate values saturate toward strength.
    mx.random.seed(1)
    prev = mx.random.uniform(shape=(1, 16, 24, 3))
    curr = mx.roll(prev, 2, axis=2)
    flow = mx.concatenate(
        [mx.full((1, 16, 24, 1), -2.0), mx.zeros((1, 16, 24, 1))], axis=-1)
    gate = history_improve_gate(curr, prev, flow, mx.float32, strength=0.75)
    mx.eval(gate)
    interior = gate[:, 2:-2, 4:-4]
    assert float(mx.min(interior)) > 0.7
    assert float(mx.max(gate)) <= 0.75 + 1e-6


def test_box3_replicate_padded_mean():
    vals = mx.arange(3 * 3).reshape(1, 3, 3, 1).astype(mx.float32)
    out = box3(vals)
    mx.eval(out)

    # Replicate padding around:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # makes the top-left 3x3 neighbourhood [0,0,1; 0,0,1; 3,3,4].
    assert abs(float(out[0, 0, 0, 0]) - (12.0 / 9.0)) < 1e-6
    assert abs(float(out[0, 1, 1, 0]) - 4.0) < 1e-6


def test_lanczos_resample_plan_properties():
    from kinovsr.modeling.vsr_blocks import make_lanczos_plan, resample_width

    # identity when sizes match
    plan = make_lanczos_plan(12, 12)
    x = mx.random.uniform(shape=(4, 12, 3))
    assert float(mx.max(mx.abs(resample_width(x, plan) - x))) < 1e-6

    # constants preserved exactly on up AND down (weights sum to 1)
    up = make_lanczos_plan(87, 95)      # the 128:117 PAR ratio reduced
    dn = make_lanczos_plan(95, 87)
    const = mx.full((3, 87, 2), 0.4)
    out = resample_width(const, up)
    assert out.shape == (3, 95, 2)
    assert float(mx.max(mx.abs(out - 0.4))) < 1e-5
    constd = mx.full((3, 95, 2), 0.4)
    outd = resample_width(constd, dn)
    assert outd.shape == (3, 87, 2)
    assert float(mx.max(mx.abs(outd - 0.4))) < 1e-5

    # downscale kernel widens for antialiasing (more taps than upscale)
    assert dn[0].shape[0] > up[0].shape[0]

    # a linear ramp is reproduced closely in the interior on upscale
    ramp = mx.broadcast_to(mx.arange(87, dtype=mx.float32)[None, :, None], (2, 87, 1))
    r = resample_width(ramp, up)
    j = mx.arange(95, dtype=mx.float32)
    expect = (j + 0.5) * (87.0 / 95.0) - 0.5
    err = mx.abs(r[0, :, 0] - expect)[8:-8]
    assert float(mx.max(err)) < 0.02

    # batched 4D input works too
    x4 = mx.random.uniform(shape=(2, 4, 87, 3))
    assert resample_width(x4, up).shape == (2, 4, 95, 3)
