"""Reference-free unit tests for the PVDD denoiser port.

End-to-end numerical parity (MLX vs own-torch, ~5e-7) lives in
$SHARED_TEMP_DIR/trace_analysis/pvdd_mlx_parity.py. Here we pin the weight-free,
torch-free pieces: window partition/reverse, the window-size shrink rule, the
mult-of-4 pad geometry, tanh-GELU, pixel shuffle, the shift mask, the relative-
position-bias gather, and variant (num_in/level) detection from a tiny fake dict.
"""
import math

import mlx.core as mx

from kinovsr.pvdd import net
from kinovsr.pvdd import load_pvdd


def test_window_partition_reverse_roundtrip():
    mx.random.seed(0)
    x = mx.random.uniform(shape=(1, 2, 16, 24, 5))   # B,D,H,W,C
    ws = (8, 8)
    win = net._window_partition(x, ws)
    assert win.shape == (1 * (16 // 8) * (24 // 8), 2, 8, 8, 5)
    back = net._window_reverse(win, ws, 1, 2, 16, 24)
    assert float(mx.max(mx.abs(back - x))) < 1e-6


def test_get_window_size_shrinks_on_small_input():
    # H,W larger than the window keep it; a small axis shrinks and zeroes its shift
    ws, ss = net._get_window_size((16, 5), (8, 8), (4, 4))
    assert ws == (8, 5) and ss == (4, 0)
    assert net._get_window_size((32, 40), (8, 8)) == (8, 8)


def test_pad4_geometry_and_crop_roundtrip():
    from kinovsr.pvdd import _pad4
    mx.random.seed(1)
    x = mx.random.uniform(shape=(1, 50, 66, 3))
    y = _pad4(x)
    assert y.shape == (1, 52, 68, 3)                 # up to next multiple of 4
    assert float(mx.max(mx.abs(y[:, :50, :66] - x))) == 0.0   # interior preserved
    # replicate (edge), so the pad rows equal the last real row
    assert float(mx.max(mx.abs(y[:, 50:, :66] - x[:, 49:50, :66]))) == 0.0
    x4 = mx.random.uniform(shape=(1, 48, 64, 3))
    assert _pad4(x4).shape == (1, 48, 64, 3)         # already a multiple: untouched


def test_gelu_tanh_matches_formula():
    x = mx.linspace(-3, 3, 41)
    got = net._gelu(x)
    c = math.sqrt(2.0 / math.pi)
    want = 0.5 * x * (1.0 + mx.tanh(c * (x + 0.044715 * x ** 3)))
    assert float(mx.max(mx.abs(got - want))) < 1e-6
    # differs from the erf GELU (this is the tanh approximation the net was trained on)
    erf_gelu = 0.5 * x * (1.0 + mx.erf(x * 0.7071067811865476))
    assert float(mx.max(mx.abs(got - erf_gelu))) > 1e-4


def test_pixelshuffle_channels_to_space():
    # channel block [0,1,2,3] must land as the 2x2 spatial tile (torch PixelShuffle)
    x = mx.arange(4, dtype=mx.float32).reshape(1, 1, 1, 4)
    y = net._pixelshuffle(x, 2)
    assert y.shape == (1, 2, 2, 1)
    assert [int(v) for v in y.reshape(-1)] == [0, 1, 2, 3]


def test_shift_mask_zero_within_region():
    # same-region token pairs get 0, cross-region get -100; diagonal always 0
    m = net._shift_mask(2, 16, 24, (8, 8), (4, 4))
    nW, n, n2 = m.shape
    assert n == n2 == 2 * 8 * 8
    diag = mx.array([m[w, i, i] for w in range(nW) for i in range(0, n, 17)])
    assert float(mx.max(mx.abs(diag))) == 0.0
    assert bool(mx.all((m == 0.0) | (m == -100.0)))   # only same/cross-region values
    assert float(mx.min(m)) == -100.0                 # some cross-region pairs exist


def test_rel_pos_bias_gather():
    # table[index] gathered then transposed to (heads, N1, N2)
    table = mx.arange(6 * 2, dtype=mx.float32).reshape(6, 2)   # K=6, heads=2
    idx = mx.array([[0, 1], [2, 5]], dtype=mx.int32)           # N1=N2=2
    p = {"a.relative_position_bias_table": table, "a.relative_position_index": idx}
    b = net._rel_pos_bias(p, "a", 2, 2)
    assert b.shape == (2, 2, 2)
    # head 0 picks column 0 of the gathered rows: idx -> rows [[0,1],[2,5]] col0 = [[0,2],[4,10]]
    assert [[float(b[0, i, j]) for j in range(2)] for i in range(2)] == [[0.0, 2.0], [4.0, 10.0]]


def _fake_raw(num_in, level):
    """Minimal raw (torch-layout) dict with just the keys detection reads + a couple
    to exercise the transpose loop."""
    extra = 1 if level else 0
    return {
        "conv_last.weight": mx.zeros((num_in, 64, 3, 3)),
        "conv_last.bias": mx.zeros((num_in,)),
        "clean_model.conv_in.weight": mx.zeros((64, num_in + extra, 3, 3)),
        "feat_extractor.main.0.weight": mx.zeros((64, num_in, 3, 3)),
        "backward_STTB.blocks.0.attn.q.weight": mx.zeros((64, 64)),      # Linear: kept as-is
        "backward_STTB.blocks.0.attn.relative_position_index": mx.zeros((128, 128)),
        "feat_STTB.blocks.0.norm1.weight": mx.zeros((64,)),             # dropped at load
    }


def test_variant_detection_and_load_shapes():
    for num_in, level in [(3, False), (3, True), (4, False), (4, True)]:
        p, cfg = load_pvdd(_fake_raw(num_in, level), dtype=mx.float32)
        assert cfg.num_in == num_in and cfg.level is level
        # conv weight transposed OHWI; Linear left (O,I); index -> int; feat_STTB dropped
        assert p["conv_last.weight"].shape == (num_in, 3, 3, 64)
        assert p["backward_STTB.blocks.0.attn.q.weight"].shape == (64, 64)
        assert p["backward_STTB.blocks.0.attn.relative_position_index"].dtype == mx.int32
        assert not any(k.startswith("feat_STTB") for k in p)
