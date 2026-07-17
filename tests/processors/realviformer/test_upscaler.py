"""RealViformer upscaler policy tests."""

import mlx.core as mx
import pytest

from kinovsr.processors.realviformer.upscaler import RealViformerUpscaler


def test_rejects_unknown_flow_mode_before_loading_weights():
    with pytest.raises(ValueError, match="RealViformer"):
        RealViformerUpscaler(flow_mode="bogus")


def test_rejects_bad_history_controls_before_loading_weights():
    with pytest.raises(ValueError, match="history_gate"):
        RealViformerUpscaler(history_gate="bogus")
    with pytest.raises(ValueError, match="history_strength"):
        RealViformerUpscaler(history_strength=-0.1)
    with pytest.raises(ValueError, match="history_cleanup"):
        RealViformerUpscaler(history_cleanup=-0.1)
    with pytest.raises(ValueError, match="history_gate_drop"):
        RealViformerUpscaler(history_gate_drop=1.1)
    with pytest.raises(ValueError, match="history_risk_decay"):
        RealViformerUpscaler(history_risk_decay=1.0)
    with pytest.raises(ValueError, match="history_static_cap"):
        RealViformerUpscaler(history_static_cap=-0.1)


def _holistic_test_upscaler(**kwargs):
    up = RealViformerUpscaler.__new__(RealViformerUpscaler)
    up._history_strength = kwargs.get("history_strength", 1.0)
    up._history_cleanup = kwargs.get("history_cleanup", 0.25)
    up._history_gate_drop = kwargs.get("history_gate_drop", 0.85)
    up._history_risk_decay = kwargs.get("history_risk_decay", 0.0)
    up._history_static_cap = kwargs.get("history_static_cap", 0.0)
    up._risk = None
    return up


def test_realviformer_holistic_policy_is_gentle_on_static_content():
    mx.random.seed(2)
    up = _holistic_test_upscaler()
    curr = mx.random.uniform(shape=(1, 12, 16, 3))
    flow = mx.zeros((1, 12, 16, 2))
    warped = mx.random.uniform(shape=(1, 12, 16, 48))

    cleaned, gate = up._holistic_history_policy(curr, curr, flow, warped, mx.float32)
    mx.eval(cleaned, gate)

    assert gate.shape == (1, 12, 16, 1)
    assert cleaned.shape == warped.shape
    assert 0.14 < float(mx.min(gate)) < 0.16
    assert 0.14 < float(mx.max(gate)) < 0.16
    assert float(mx.mean(mx.abs(cleaned - warped))) > 0.0


def test_realviformer_holistic_policy_opens_on_well_tracked_motion():
    mx.random.seed(3)
    up = _holistic_test_upscaler()
    prev = mx.random.uniform(shape=(1, 16, 24, 3))
    curr = mx.roll(prev, 2, axis=2)
    flow = mx.concatenate(
        [mx.full((1, 16, 24, 1), -2.0), mx.zeros((1, 16, 24, 1))], axis=-1)
    warped = mx.random.uniform(shape=(1, 16, 24, 48))

    _, gate = up._holistic_history_policy(curr, prev, flow, warped, mx.float32)
    mx.eval(gate)

    interior = gate[:, 2:-2, 4:-4]
    assert float(mx.median(interior)) > 0.95


def test_realviformer_pad4_matches_reference_left_top_reflect():
    vals = mx.arange(5 * 6).reshape(1, 5, 6, 1)

    padded, pad_top, pad_left = RealViformerUpscaler._pad4(vals)
    mx.eval(padded)

    assert (pad_top, pad_left) == (3, 2)
    assert padded.shape == (1, 8, 8, 1)

    row_indices = [3, 2, 1, 0, 1, 2, 3, 4]
    col_indices = [2, 1, 0, 1, 2, 3, 4, 5]
    expected = [
        [r * 6 + c for c in col_indices]
        for r in row_indices
    ]
    assert padded[0, :, :, 0].tolist() == expected


def test_vision_flow_mode_passes_the_driver_guard():
    # Regression: the driver-level flow guard lagged the factory token
    # list when the vision backend landed, killing runs at build time.
    # Validation passes and the failure is weight resolution, not flow.
    with pytest.raises(FileNotFoundError):
        RealViformerUpscaler(flow_mode="vision",
                             weights="/nonexistent/weights.safetensors")
