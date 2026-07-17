"""Sanitize-edge pixel-operation tests."""

import mlx.core as mx
import pytest


def test_sanitize_rgb_replaces_bands_and_keeps_dims():
    from kinovsr.processors.sanitize_edges.ops import sanitize_rgb

    fr = mx.broadcast_to(mx.arange(10, dtype=mx.float32)[:, None, None], (10, 8, 3))
    out = sanitize_rgb(fr, (2, 1, 0, 0))
    mx.eval(out)
    assert out.shape == fr.shape
    assert float(mx.max(mx.abs(out[0] - fr[2]))) == 0.0   # top rows <- first interior
    assert float(mx.max(mx.abs(out[1] - fr[2]))) == 0.0
    assert float(mx.max(mx.abs(out[-1] - fr[-2]))) == 0.0  # bottom row <- neighbor
    assert float(mx.max(mx.abs(out[2:-1] - fr[2:-1]))) == 0.0  # interior untouched

    with pytest.raises(ValueError, match="interior"):
        sanitize_rgb(fr, (5, 5, 0, 0))


def test_restore_borders_composites_original():
    from kinovsr.processors.sanitize_edges.ops import restore_borders

    mx.random.seed(9)
    src = mx.random.uniform(shape=(8, 10, 3))
    out = mx.random.uniform(shape=(16, 20, 3))       # processed at 2x
    res = restore_borders(out, src, (0, 1, 2, 0), feather=0)
    mx.eval(res)

    assert res.shape == out.shape
    # bottom band (2 out rows) = nearest-2x of src's last row
    assert float(mx.max(mx.abs(res[14] - mx.repeat(src[7], 2, axis=0)))) < 1e-6
    assert float(mx.max(mx.abs(res[15] - res[14]))) < 1e-6
    # left band (4 out cols) = nearest-2x of src's first 2 cols
    assert float(mx.max(mx.abs(res[0, 0] - src[0, 0]))) < 1e-6
    assert float(mx.max(mx.abs(res[0, 3] - src[0, 1]))) < 1e-6
    # interior untouched
    assert float(mx.max(mx.abs(res[2:14, 4:] - out[2:14, 4:]))) < 1e-6

    # uint8 sources normalize to [0,1]
    src8 = mx.full((8, 10, 3), 255, dtype=mx.uint8)
    res = restore_borders(out, src8, (1, 0, 0, 0), feather=0)
    assert abs(float(mx.mean(res[0])) - 1.0) < 1e-6


def test_restore_borders_resamples_non_integer_geometry():
    # The anamorphic case: --square-pixels resamples the width between the
    # restore capture (576x720) and the composite (576x1024, PAR 64/45), so
    # the width ratio is non-integer and differs from the height ratio.
    from kinovsr.processors.sanitize_edges.ops import restore_borders

    mx.random.seed(11)
    src = mx.random.uniform(shape=(576, 720, 3))
    out = mx.random.uniform(shape=(576, 1024, 3))
    res = restore_borders(out, src, (6, 6, 0, 0), feather=0)
    mx.eval(res)

    assert res.shape == out.shape
    # top band: output col j takes the globally nearest source col
    for j in (0, 500, 1023):
        sj = (j * 720) // 1024
        assert float(mx.max(mx.abs(res[0, j] - src[0, sj]))) < 1e-6
        assert float(mx.max(mx.abs(res[5, j] - src[5, sj]))) < 1e-6
    # bottom band restored from the source's last rows
    assert float(mx.max(mx.abs(res[575, 0] - src[575, 0]))) < 1e-6
    # interior untouched
    assert float(mx.max(mx.abs(res[6:570] - out[6:570]))) < 1e-6

    # per-axis non-integer scale (15/8 rows, 2x cols) composites too
    small_src = mx.random.uniform(shape=(8, 10, 3))
    res2 = restore_borders(mx.zeros((15, 20, 3)), small_src, (1, 0, 0, 0),
                           feather=0)
    zone = max(1, round(1 * 15 / 8))                 # one source row -> 2 out
    for r in range(zone):
        sr = (r * 8) // 15
        assert float(mx.max(mx.abs(res2[r, 0] - small_src[sr, 0]))) < 1e-6


def test_restore_borders_feather_ramps_into_content():
    from kinovsr.processors.sanitize_edges.ops import restore_borders

    src = mx.zeros((8, 6, 3), dtype=mx.float32)
    out = mx.ones((16, 12, 3), dtype=mx.float32)   # processed at 2x
    res = restore_borders(out, src, (1, 0, 0, 0), feather=2)
    mx.eval(res)

    col = [float(res[r, 5, 0]) for r in range(8)]
    # band rows (2 at 2x) fully restored to source (0), then a linear ramp
    # across the 4-row feather zone, then untouched processed content (1).
    assert col[0] == 0.0 and col[1] == 0.0
    expect = [0.125, 0.375, 0.625, 0.875]
    for got, want in zip(col[2:6], expect, strict=True):
        assert abs(got - want) < 1e-6
    assert col[6] == 1.0 and col[7] == 1.0
