import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from LTX_2_MLX.videotoolbox.basicvsrpp.upscaler import BasicVsrUpscaler
from LTX_2_MLX.videotoolbox.realbasicvsr.upscaler import RealBasicVsrUpscaler
from LTX_2_MLX.videotoolbox.realviformer.upscaler import RealViformerUpscaler
from LTX_2_MLX.videotoolbox.vsr_blocks import _compute_flows, box3, history_improve_gate

ROOT = Path(__file__).resolve().parents[1]


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


@pytest.mark.parametrize(
    ("cls", "name"),
    [
        (BasicVsrUpscaler, "BasicVSR"),
        (RealBasicVsrUpscaler, "RealBasicVSR"),
        (RealViformerUpscaler, "RealViformer"),
    ],
)
def test_upscaler_wrappers_reject_unknown_flow_mode_before_loading_weights(cls, name):
    with pytest.raises(ValueError, match=name):
        cls(flow_mode="bogus")


def test_realviformer_rejects_bad_history_controls_before_loading_weights():
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


def test_realbasicvsr_rejects_bad_history_controls_before_loading_weights():
    with pytest.raises(ValueError, match="history_gate"):
        RealBasicVsrUpscaler(history_gate="bogus")
    with pytest.raises(ValueError, match="history_strength"):
        RealBasicVsrUpscaler(history_strength=-0.1)


def test_basicvsrpp_rejects_bad_history_controls_before_loading_weights():
    with pytest.raises(ValueError, match="history_gate"):
        BasicVsrUpscaler(history_gate="bogus")
    with pytest.raises(ValueError, match="history_strength"):
        BasicVsrUpscaler(history_strength=-0.1)


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


def test_pth_converter_refuses_ambiguous_params_dict(tmp_path):
    torch = pytest.importorskip("torch")
    ckpt = tmp_path / "ambiguous.pth"
    out = tmp_path / "ambiguous.safetensors"
    torch.save(
        {
            "params": {"w": torch.ones(1)},
            "params_ema": {"w": torch.zeros(1)},
        },
        ckpt,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "pth_to_safetensors.py"),
            str(ckpt),
            "-o",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "checkpoint carries BOTH 'params' and 'params_ema'" in result.stderr
    assert not out.exists()
