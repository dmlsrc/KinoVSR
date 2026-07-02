import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from LTX_2_MLX.videotoolbox.vsr_blocks import _compute_flows
from LTX_2_MLX.videotoolbox.basicvsrpp.upscaler import BasicVsrUpscaler
from LTX_2_MLX.videotoolbox.realbasicvsr.upscaler import RealBasicVsrUpscaler
from LTX_2_MLX.videotoolbox.realviformer.upscaler import RealViformerUpscaler


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
