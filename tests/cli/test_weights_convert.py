"""Weights-converter CLI integration tests."""

import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

def test_pth_converter_refuses_ambiguous_params_dict(tmp_path):
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
            "-m",
            "kinovsr.cli.main",
            "weights",
            "convert",
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
