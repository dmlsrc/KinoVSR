from __future__ import annotations

import mlx.core as mx


def load_metric_module():
    from kinovsr.cli.commands import artifacts

    return artifacts


def test_desra_contrast_is_one_for_identical_flat_frames() -> None:
    metric = load_metric_module()
    kernel = metric.gaussian_kernel()
    frame = mx.full((24, 32, 3), 0.5, dtype=mx.float32)

    contrast, texture = metric.desra_contrast(frame, frame, kernel)

    assert bool(mx.allclose(contrast, mx.array(1.0)))
    assert bool(mx.allclose(texture, mx.array(0.0)))


def test_desra_contrast_flags_texture_energy_mismatch() -> None:
    metric = load_metric_module()
    kernel = metric.gaussian_kernel()
    ref = mx.full((32, 32, 3), 0.5, dtype=mx.float32)
    cand = mx.contiguous(ref)
    cand[:, ::2, :] = 0.25
    cand[:, 1::2, :] = 0.75

    contrast, _ = metric.desra_contrast(ref, cand, kernel)
    risk = 1.0 - contrast

    assert float(risk.mean()) > 0.20
    assert float(risk.max()) > 0.25


def test_flat_weight_downweights_busy_reference_regions() -> None:
    metric = load_metric_module()
    texture = mx.concatenate(
        [
            mx.full((8, 8), 0.01, dtype=mx.float32),
            mx.full((8, 8), 0.20, dtype=mx.float32),
        ],
        axis=1,
    )

    weight = metric.flat_weight_from_texture(texture, busy_floor=0.25)

    assert float(weight[:, :8].mean()) > 0.95
    assert float(weight[:, 8:].mean()) < 0.35
