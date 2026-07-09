"""Reference-free unit tests for the MLX RealPLKSR port.

The real checkpoints are not bundled, so these build tiny synthetic checkpoints
(both variants) to exercise the load path, variant detection, and output geometry
without weights or torch. The one numerically load-bearing piece -- DySample's
coordinate math -- is pinned against an independent anchor: with a zero predicted
offset, DySample must reduce to a nearest-neighbor upsample (each output block
samples its single source pixel). Full torch parity on the real weights lives in
$SHARED_TEMP_DIR/trace_analysis/realplksr_parity.py (both variants, ~2e-6).
"""
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from kinovsr.realplksr import net


def _synth(path, *, scale, layer_norm, dysample, dim=8, n_blocks=2, ks=3):
    """Write a tiny torch-layout RealPLKSR checkpoint (random weights)."""
    pdim = max(1, dim // 4)
    mx.random.seed(0)

    def w(*shape):
        return mx.random.normal(shape=shape) * 0.05

    d = {"feats.0.weight": w(dim, 3, 3, 3), "feats.0.bias": mx.zeros((dim,))}
    for i in range(1, n_blocks + 1):
        pre = f"feats.{i}"
        if layer_norm:
            d[f"{pre}.layer_norm.weight"] = mx.ones((dim,))
            d[f"{pre}.layer_norm.bias"] = mx.zeros((dim,))
        d[f"{pre}.channel_mixer.0.weight"] = w(dim * 2, dim, 3, 3)
        d[f"{pre}.channel_mixer.0.bias"] = mx.zeros((dim * 2,))
        d[f"{pre}.channel_mixer.2.weight"] = w(dim, dim * 2, 3, 3)
        d[f"{pre}.channel_mixer.2.bias"] = mx.zeros((dim,))
        d[f"{pre}.lk.conv.weight"] = w(pdim, pdim, ks, ks)
        d[f"{pre}.lk.conv.bias"] = mx.zeros((pdim,))
        d[f"{pre}.attn.f.0.weight"] = w(dim, dim, 3, 3)
        d[f"{pre}.attn.f.0.bias"] = mx.zeros((dim,))
        d[f"{pre}.refine.weight"] = w(dim, dim, 1, 1)
        d[f"{pre}.refine.bias"] = mx.zeros((dim,))
        if not layer_norm:
            d[f"{pre}.norm.weight"] = mx.ones((dim,))
            d[f"{pre}.norm.bias"] = mx.zeros((dim,))
    last = n_blocks + 2
    d[f"feats.{last}.weight"] = w(3 * scale * scale, dim, 3, 3)
    d[f"feats.{last}.bias"] = mx.zeros((3 * scale * scale,))
    if dysample:
        g = 4
        oc = 2 * g * scale * scale
        cin = 3 * scale * scale
        d["to_img.offset.weight"] = w(oc, cin, 1, 1)
        d["to_img.offset.bias"] = mx.zeros((oc,))
        d["to_img.scope.weight"] = mx.zeros((oc, cin, 1, 1))
        d["to_img.init_pos"] = mx.zeros((1, oc, 1, 1))
        d["to_img.end_conv.weight"] = w(3, cin, 1, 1)
        d["to_img.end_conv.bias"] = mx.zeros((3,))
    mx.save_safetensors(str(path), d)


@pytest.mark.parametrize("scale,ln,dys", [(4, False, False), (2, True, True)])
def test_config_detection_and_geometry(scale, ln, dys):
    with tempfile.TemporaryDirectory() as td:
        ck = Path(td) / "m.safetensors"
        _synth(ck, scale=scale, layer_norm=ln, dysample=dys, n_blocks=2)
        p = net.load_params(str(ck), dtype=mx.float32)
        dim, n_blocks, ks, pdim, det_scale, det_ln, det_dys, groups = net._config(p)
        assert (det_scale, det_ln, det_dys) == (scale, ln, dys)
        assert n_blocks == 2 and dim == 8
        assert groups == (4 if dys else 0)
        x = mx.clip(mx.random.uniform(shape=(1, 12, 16, 3)), 0, 1)
        out = net.realplksr(x, p, net._config(p))
        mx.eval(out)
        assert out.shape == (1, 12 * scale, 16 * scale, 3)
        assert bool(mx.all(mx.isfinite(out)))
        assert float(mx.min(out)) >= 0.0 and float(mx.max(out)) <= 1.0


def test_dysample_zero_offset_is_nearest_upsample():
    """DySample with a zero predicted offset must sample each output block's single
    source pixel -- i.e. a nearest-neighbor upsample of its input. This pins the
    normalize -> grid_sample(align_corners=False) coordinate collapse used by the
    MLX port (sample_pixel = lr_index + offset) without needing torch."""
    scale, groups = 2, 4
    cin = 3 * scale * scale  # 12
    mx.random.seed(1)
    # zero offset+scope+init_pos so offset == 0 everywhere; identity end_conv (per
    # group channel picks its own channel) so the gather is observable per channel.
    oc = 2 * groups * scale * scale
    p = {
        "to_img.offset.weight": mx.zeros((oc, 1, 1, cin)),
        "to_img.offset.bias": mx.zeros((oc,)),
        "to_img.scope.weight": mx.zeros((oc, 1, 1, cin)),
        "to_img.init_pos": mx.zeros((1, 1, 1, oc)),
        # end_conv = identity on the first 3 of 12 channels (1x1)
        "to_img.end_conv.weight": mx.concatenate(
            [mx.eye(3), mx.zeros((3, cin - 3))], axis=1).reshape(3, 1, 1, cin),
        "to_img.end_conv.bias": mx.zeros((3,)),
    }
    x = mx.random.uniform(shape=(1, 4, 5, cin))
    out = net._dysample(x, p, scale, groups)
    mx.eval(out)
    assert out.shape == (1, 8, 10, 3)
    # nearest upsample of x[..., :3] by `scale`
    xc = x[..., :3]
    nn = mx.broadcast_to(xc[:, :, None, :, None, :], (1, 4, scale, 5, scale, 3)).reshape(1, 8, 10, 3)
    assert float(mx.max(mx.abs(out - nn))) < 1e-4


def test_mish_stable_no_fp16_overflow():
    """Stable Mish softplus must not overflow fp16 for large inputs (naive
    log(1+exp(x)) would inf past x~11)."""
    x = mx.array([[[-20.0, -1.0, 0.0, 5.0, 40.0, 100.0]]], dtype=mx.float16)
    y = net._mish(x)
    mx.eval(y)
    assert bool(mx.all(mx.isfinite(y)))
    # mish(x) -> x for large x, -> 0^- for large negative
    assert abs(float(y[0, 0, 5]) - 100.0) < 0.5
    assert abs(float(y[0, 0, 0])) < 1e-3
