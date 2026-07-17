"""SAFMN model and upscaler tests."""

import mlx.core as mx
import pytest


def _torch_bins(size, out):
    """The torch adaptive-pooling bin boundaries, straight from the spec."""
    return [((i * size) // out, -(-((i + 1) * size) // out))
            for i in range(out)]


def test_adaptive_maxpool_matches_torch_bins_on_odd_sizes():
    from kinovsr.processors.safmn.net import _adaptive_maxpool

    mx.random.seed(3)
    x = mx.random.uniform(shape=(1, 61, 90, 2))       # 486-class odd height
    out_h, out_w = 61 // 8, 90 // 8                   # 7 x 11 bins
    got = _adaptive_maxpool(x, out_h, out_w)
    assert got.shape == (1, out_h, out_w, 2)
    for bi, (hs, he) in enumerate(_torch_bins(61, out_h)):
        for bj, (ws_, we) in enumerate(_torch_bins(90, out_w)):
            want = mx.max(x[:, hs:he, ws_:we, :], axis=(1, 2))
            assert float(mx.max(mx.abs(got[:, bi, bj, :] - want))) < 1e-6


def test_adaptive_avgpool_matches_torch_bins_on_odd_sizes():
    from kinovsr.processors.safmn.net import _adaptive_avgpool

    mx.random.seed(4)
    x = mx.random.uniform(shape=(1, 13, 22, 3))
    out_h, out_w = 3, 5
    got = _adaptive_avgpool(x, out_h, out_w)
    for bi, (hs, he) in enumerate(_torch_bins(13, out_h)):
        for bj, (ws_, we) in enumerate(_torch_bins(22, out_w)):
            want = mx.mean(x[:, hs:he, ws_:we, :], axis=(1, 2))
            assert float(mx.max(mx.abs(got[:, bi, bj, :] - want))) < 1e-6


def test_adaptive_pools_keep_the_divisible_fast_path_exact():
    from kinovsr.processors.safmn.net import (
        _adaptive_avgpool,
        _adaptive_maxpool,
    )

    mx.random.seed(5)
    x = mx.random.uniform(shape=(1, 24, 32, 4))
    ref_max = mx.max(x.reshape(1, 3, 8, 4, 8, 4), axis=(2, 4))
    ref_avg = mx.mean(x.reshape(1, 3, 8, 4, 8, 4), axis=(2, 4))
    assert float(mx.max(mx.abs(_adaptive_maxpool(x, 3, 4) - ref_max))) == 0.0
    assert float(mx.max(mx.abs(_adaptive_avgpool(x, 3, 4) - ref_avg))) < 1e-7


def test_nearest_to_matches_torch_index_formula():
    from kinovsr.processors.safmn.net import _nearest_to

    x = mx.arange(7 * 11, dtype=mx.float32).reshape(1, 7, 11, 1)
    out = _nearest_to(x, 61, 90)
    assert out.shape == (1, 61, 90, 1)
    for i in (0, 30, 60):
        for j in (0, 45, 89):
            src = x[0, (i * 7) // 61, (j * 11) // 90, 0]
            assert float(out[0, i, j, 0]) == float(src)
    # integer-multiple targets reduce to the repeat path
    up = _nearest_to(x, 14, 22)
    assert float(mx.max(mx.abs(up[:, ::2, ::2] - x))) == 0.0


def test_safmn_bicubic_up_matches_torch_reference():
    from kinovsr.processors.safmn.net import _bicubic_up

    # Reference computed once with torch F.interpolate(scale_factor=2,
    # mode="bicubic", align_corners=False) on this exact input.
    vals = [v * 0.7 - 2.0 for v in range(12)]
    x = mx.array(vals, dtype=mx.float32).reshape(1, 3, 4, 1)
    expected = [
        [-2.3691406, -2.1613283, -1.8277344, -1.3874999, -1.1031251, -0.66289067, -0.32929698, -0.12148452],
        [-1.5378907, -1.3300781, -0.99648434, -0.55624998, -0.27187502, 0.16835934, 0.50195312, 0.70976561],
        [-0.20351566, 0.0042968662, 0.33789068, 0.77812499, 1.0625000, 1.5027344, 1.8363283, 2.0441408],
        [1.6558595, 1.8636718, 2.1972656, 2.6374998, 2.9218750, 3.3621094, 3.6957033, 3.9035158],
        [2.9902344, 3.1980467, 3.5316405, 3.9718747, 4.2562499, 4.6964846, 5.0300775, 5.2378907],
        [3.8214846, 4.0292969, 4.3628907, 4.8031244, 5.0874996, 5.5277343, 5.8613276, 6.0691404],
    ]
    out = _bicubic_up(x, 2)
    mx.eval(out)
    assert out.shape == (1, 6, 8, 1)
    for r in range(6):
        for c in range(8):
            assert abs(float(out[0, r, c, 0]) - expected[r][c]) < 2e-6, (r, c)


def test_safmn_bicubic_up_reproduces_constants_and_ramps():
    from kinovsr.processors.safmn.net import _bicubic_up

    # Weights sum to 1 -> constants reproduce exactly, including at the
    # replicate-padded borders.
    const = mx.full((1, 4, 4, 2), 0.37, dtype=mx.float32)
    out = _bicubic_up(const, 4)
    mx.eval(out)
    assert float(mx.max(mx.abs(out - 0.37))) < 1e-6

    # On a linear ramp the A=-0.75 cubic kernel has a fixed alternating phase
    # bias of -/+ 3/64 at r=2 (it does not reproduce linears exactly; torch
    # produces these same values). Interior rows, away from tap clamping:
    ramp = mx.broadcast_to(mx.arange(8, dtype=mx.float32)[None, :, None, None], (1, 8, 4, 1))
    out = _bicubic_up(ramp, 2)
    mx.eval(out)
    inner = out[0, 4:12, 2:6, 0]  # away from H borders
    linear = 1.75 + mx.arange(8, dtype=mx.float32) * 0.5
    bias = mx.array([-1.0, 1.0] * 4, dtype=mx.float32) * (3.0 / 64.0)
    expect = linear + bias
    assert float(mx.max(mx.abs(inner - expect[:, None]))) < 1e-5


def test_safmn_safm_mode_inferred_from_filename():
    from kinovsr.processors.safmn.net import _VARIANTS, _safm_mode_for

    assert _safm_mode_for("safmn_purescale_x4.safetensors") == "fixed"
    assert _safm_mode_for("/a/b/Safmn_PureScale_sharper_x2.safetensors") == "fixed"
    assert _safm_mode_for("safmn_l_real_lsdir_x4.safetensors") == "stock"
    assert _safm_mode_for("light_safmnpp.safetensors") == "stock"
    for token in ("purescale", "purescale2x", "purescale2x-sharp"):
        assert token in _VARIANTS
        assert _safm_mode_for(_VARIANTS[token]) == "fixed"
    for token in ("light", "real", "real2x"):
        assert _safm_mode_for(_VARIANTS[token]) == "stock"


def test_safmn_config_carries_safm_mode_and_trained_upsampler():
    from kinovsr.processors.safmn.net import _config

    p = {
        "to_feat.weight": mx.zeros((128, 3, 3, 3)),
        "feats.0.norm1.weight": mx.zeros((128,)),
        "feats.1.norm1.weight": mx.zeros((128,)),
        "to_img.0.weight": mx.zeros((48, 3, 3, 128)),
    }
    assert _config(p) == ("real", 128, 2, 4, "stock", "nearest", 0.0)
    p["__safm_mode__"] = "fixed"
    assert _config(p) == ("real", 128, 2, 4, "fixed", "bicubic", 0.0)


def test_safmn_upscaler_validates_args_before_loading_weights():
    from kinovsr.processors.safmn.upscaler import SafmnUpscaler

    with pytest.raises(ValueError, match="safm_up"):
        SafmnUpscaler(safm_up="bogus")
    with pytest.raises(ValueError, match="pool_clamp"):
        SafmnUpscaler(pool_clamp=-1.0)


def test_safmn_pool_clamp_touches_only_interior_outliers():
    from kinovsr.processors.safmn.net import _pool_clamp

    mx.random.seed(4)
    s = mx.random.normal(shape=(1, 12, 16, 4)) * 0.1
    spiked = s[:]
    spiked[0, 6, 8, 2] = 25.0    # interior spike, far beyond k sigma
    spiked[0, 0, 5, 3] = 25.0    # frame-boundary spike (exempt margin)
    out = _pool_clamp(spiked, 4.0)
    mx.eval(out)

    # The interior spike is bounded...
    assert float(out[0, 6, 8, 2]) < 25.0
    # ...to roughly mu + 4 sigma of its channel's INTERIOR statistics...
    core = spiked.astype(mx.float32)[:, 1:-1, 1:-1, 2]
    mu = float(mx.mean(core))
    sd = float(mx.sqrt(mx.mean((core - mu) ** 2)))
    assert abs(float(out[0, 6, 8, 2]) - (mu + 4.0 * sd)) < 1e-4
    # ...an untouched channel passes through numerically unchanged...
    assert float(mx.max(mx.abs(out[..., 0] - spiked[..., 0]))) < 1e-6
    # ...and the frame-boundary margin is exempt (synthetic borders saturate
    # there; clamping them re-engages texture hallucination).
    assert float(out[0, 0, 5, 3]) == 25.0

    # Degenerate pooled maps (no interior) pass through whole.
    tiny = mx.full((1, 2, 2, 1), 9.0)
    assert float(mx.max(mx.abs(_pool_clamp(tiny, 4.0) - tiny))) == 0.0
