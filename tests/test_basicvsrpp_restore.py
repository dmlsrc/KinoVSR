"""Reference-free unit tests for the BasicVSR++ 1x-restoration additions.

The full recurrent net needs SPyNet/deform weights, so end-to-end parity lives in
$SHARED_TEMP_DIR/trace_analysis/basicvsrpp_1x_parity.py (2.7e-6 vs own-torch). Here
we pin the weight-free, torch-free pieces: the bicubic-1/4 flow downsample, the
variant detection, the mult-of-4 padding (the fix from ckkelvinchan/BasicVSR++
issue #24), the geometric transforms, and the 8-way self-ensemble's round-trip.
"""
import mlx.core as mx

from LTX_2_MLX.videotoolbox.basicvsrpp import net


def test_bicubic_down4_shape_and_constant():
    # weights sum to 1, so a constant is preserved exactly; shape is /4.
    x = mx.full((1, 16, 24, 3), 0.4)
    y = net._bicubic_down4(x)
    assert y.shape == (1, 4, 6, 3)
    assert float(mx.max(mx.abs(y - 0.4))) < 1e-6


def test_bicubic_down4_reproduces_linear_ramp():
    # torch bicubic downsample-by-4 maps output col j to input coord 4j+1.5; the
    # Keys cubic (A=-0.75) reproduces a linear ramp exactly there.
    w = 24
    ramp = mx.broadcast_to(mx.arange(w, dtype=mx.float32)[None, None, :, None], (1, 8, w, 1))
    y = net._bicubic_down4(ramp)
    expect = 4.0 * mx.arange(6, dtype=mx.float32) + 1.5
    assert float(mx.max(mx.abs(y[0, 0, :, 0] - expect))) < 1e-3


def test_is_low_res_input_detection():
    # SR checkpoints have feat_extract.main.0; 1x-restoration have the strided
    # feat_extract.0 stem instead.
    assert net.is_low_res_input({"feat_extract.main.0.weight": mx.zeros((1,))}) is True
    assert net.is_low_res_input({"feat_extract.0.weight": mx.zeros((1,))}) is False


def test_pad_mult4_replicate_and_crop_geometry():
    # issue #24: pad input to a multiple of 4, crop back. Replicate-pad, interior kept.
    mx.random.seed(0)
    x = mx.random.uniform(shape=(1, 50, 66, 3))
    y = net._pad_mult4(x)
    assert y.shape == (1, 52, 68, 3)
    assert float(mx.max(mx.abs(y[:, :50, :66] - x))) == 0.0
    # already-multiple input is untouched
    x4 = mx.random.uniform(shape=(1, 48, 64, 3))
    assert net._pad_mult4(x4).shape == (1, 48, 64, 3)


def test_geo_transforms_involution():
    mx.random.seed(1)
    x = mx.random.uniform(shape=(1, 6, 8, 3))
    for m in ("v", "h", "t"):
        assert float(mx.max(mx.abs(net._geo_tf(net._geo_tf(x, m), m) - x))) < 1e-6
    assert net._geo_tf(x, "t").shape == (1, 8, 6, 3)


def test_spatial_ensemble_identity_roundtrip():
    # With an identity model, the 8-way ensemble must reconstruct the input exactly
    # -- this validates that each variant's forward transform is correctly inverted
    # and averaged (the tricky part of the reference scheme). Non-square to exercise
    # the transpose variants.
    mx.random.seed(2)
    frames = [mx.clip(mx.random.uniform(shape=(1, 8, 12, 3)), 0, 1) for _ in range(3)]
    out = net._spatial_ensemble(frames, lambda fl: fl)
    assert len(out) == len(frames)
    for o, f in zip(out, frames, strict=True):
        assert o.shape == f.shape
        assert float(mx.max(mx.abs(o - f))) < 1e-5


def test_restore_variant_tokens_resolve_names():
    # every documented token maps to a distinct filename
    v = net._RESTORE_VARIANTS
    assert {"decompress_track1", "decompress_track2", "decompress_track3",
            "denoise", "deblur_dvd", "deblur_gopro"} <= set(v)
    assert len(set(v.values())) == len(v)
