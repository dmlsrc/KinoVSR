"""Reference-free unit tests for the BasicVSR++ 1x-restoration additions.

The full recurrent net needs SPyNet/deform weights, so end-to-end parity lives in
$SHARED_TEMP_DIR/trace_analysis/basicvsrpp_1x_parity.py (2.7e-6 vs own-torch). Here
we pin the weight-free, torch-free pieces: the bicubic-1/4 flow downsample, the
variant detection, the mult-of-4 padding (the fix from ckkelvinchan/BasicVSR++
issue #24), the geometric transforms, and the 8-way self-ensemble's round-trip.
"""
import mlx.core as mx

from kinovsr.modeling.upscaler_base import plan_gop_windows
from kinovsr.processors.basicvsrpp import net


def _emit_tiles(windows, n):
    emit = [(a, b) for _, _, a, b in windows]
    return (emit == sorted(emit) and emit[0][0] == 0 and emit[-1][1] == n
            and all(emit[i][1] == emit[i + 1][0] for i in range(len(emit) - 1)))


def test_plan_gop_windows_tiles_and_anchors():
    kf = list(range(0, 300, 12))  # GOP 12
    w = plan_gop_windows(kf, 300, min_window=24, max_window=64)
    assert _emit_tiles(w, 300)
    # every emit boundary is a keyframe; each GOP-aligned window re-processes only
    # the single closing keyframe (proc = emit + 1)
    kfset = set(kf)
    for p0, p1, e0, e1 in w:
        assert e0 in kfset            # emit starts on a keyframe
        # +1 closing keyframe for the backward anchor, except the clip-end tail
        assert p1 - p0 == (e1 - e0) + (1 if e1 < 300 else 0)
    assert all((e1 - e0) >= 24 for *_, e0, e1 in w[:-1])  # >= min_window (except tail)


def test_plan_gop_windows_single_keyframe_fallback():
    # open-GOP clip (one keyframe): fixed max_window tiling with internal trim
    w = plan_gop_windows([0], 100, min_window=24, max_window=60)
    assert _emit_tiles(w, 100)
    assert len(w) == 2  # 100 / 60 -> 2 sub-windows


def test_plan_gop_windows_long_gop_splits():
    # a 250-frame GOP must split under max_window with trim at internal splits,
    # while the 250 boundary stays keyframe-anchored
    w = plan_gop_windows([0, 250], 300, min_window=24, max_window=64)
    assert _emit_tiles(w, 300)
    # internal split sub-windows carry trim on both sides: max_window + 2*trim
    assert all((p1 - p0) <= 64 + 2 * 2 for p0, p1, *_ in w)


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


def test_windowed_schedule_mode_emits_every_frame_once():
    from kinovsr.modeling.upscaler_base import WindowedUpscaler

    class _Stub(WindowedUpscaler):
        SCALE = 1
        def __init__(self):
            super().__init__(window=10, trim=2)
        def _upscale_window(self, frames):
            return list(frames)   # identity

    for n, kf in [(60, list(range(0, 60, 12))), (37, list(range(0, 37, 12))), (100, [0])]:
        sched = plan_gop_windows(kf, n, 24, 64)
        up = _Stub()
        up.set_schedule(sched)
        em = []
        for i in range(n):
            em += up.feed(mx.full((1, 1, 1, 3), i / 200.0), token=i)
        em += up.flush()
        toks = [t for _, t in em]
        vals = [round(f[0, 0, 0].item() * 200) for f, _ in em]
        assert toks == list(range(n))          # every frame emitted once, in order
        assert vals == list(range(n))          # each output is the right input frame


def test_windowed_no_schedule_unchanged():
    # set_schedule(None) keeps the fixed window/trim path
    from kinovsr.modeling.upscaler_base import WindowedUpscaler

    class _Stub(WindowedUpscaler):
        SCALE = 1
        def __init__(self):
            super().__init__(window=8, trim=2)
        def _upscale_window(self, frames):
            return list(frames)

    up = _Stub()
    up.set_schedule(None)
    em = []
    for i in range(20):
        em += up.feed(mx.full((1, 1, 1, 3), i / 200.0), token=i)
    em += up.flush()
    assert [t for _, t in em] == list(range(20))


def test_restore_variant_tokens_resolve_names():
    # every documented token maps to a distinct filename
    v = net._RESTORE_VARIANTS
    assert {"decompress_track1", "decompress_track2", "decompress_track3",
            "denoise", "deblur_dvd", "deblur_gopro"} <= set(v)
    assert len(set(v.values())) == len(v)


def test_restore_vision_flow_mode_passes_the_driver_guard():
    # Regression: the restore driver's flow guard lagged the factory token
    # list when the vision backend landed ("--restore decompress_track1"
    # with flow=vision died at build time).
    import pytest

    from kinovsr.processors.basicvsrpp.restorer import BasicVsrRestorer

    with pytest.raises(FileNotFoundError):
        BasicVsrRestorer(weights="/nonexistent/weights.safetensors",
                         flow_mode="vision")
