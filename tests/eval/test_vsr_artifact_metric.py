from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_metric_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "compare_vsr_artifacts.py"
    spec = importlib.util.spec_from_file_location("compare_vsr_artifacts", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_desra_contrast_is_one_for_identical_flat_frames() -> None:
    metric = load_metric_module()
    kernel = metric.gaussian_kernel()
    frame = np.full((24, 32, 3), 0.5, dtype=np.float32)

    contrast, texture = metric.desra_contrast(frame, frame.copy(), kernel)

    assert np.allclose(contrast, 1.0)
    assert np.allclose(texture, 0.0)


def test_desra_contrast_flags_texture_energy_mismatch() -> None:
    metric = load_metric_module()
    kernel = metric.gaussian_kernel()
    ref = np.full((32, 32, 3), 0.5, dtype=np.float32)
    cand = ref.copy()
    cand[:, ::2, :] = 0.25
    cand[:, 1::2, :] = 0.75

    contrast, _ = metric.desra_contrast(ref, cand, kernel)
    risk = 1.0 - contrast

    assert float(risk.mean()) > 0.20
    assert float(risk.max()) > 0.25


def test_flat_weight_downweights_busy_reference_regions() -> None:
    metric = load_metric_module()
    texture = np.concatenate(
        [
            np.full((8, 8), 0.01, dtype=np.float32),
            np.full((8, 8), 0.20, dtype=np.float32),
        ],
        axis=1,
    )

    weight = metric.flat_weight_from_texture(texture, busy_floor=0.25)

    assert float(weight[:, :8].mean()) > 0.95
    assert float(weight[:, 8:].mean()) < 0.35
