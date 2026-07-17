"""ESC window-attention relative-position bias matches the reference
orientation: bias[q, k] indexes the trained table at rel = k - q."""

import mlx.core as mx
import pytest

from kinovsr.processors.esc.net import _expand_rpe

pytestmark = pytest.mark.unit


def _reference_bias(table, window):
    """Direct transcription of the reference create_table_idxs mapping."""
    side = 2 * window - 1
    heads = table.shape[0]
    n = window * window
    out = [[[0.0] * n for _ in range(n)] for _ in range(heads)]
    for head in range(heads):
        for q in range(n):
            for k in range(n):
                q_h, q_w = q // window, q % window
                k_h, k_w = k // window, k % window
                rel = ((k_h - q_h + window - 1) * side
                       + (k_w - q_w + window - 1))
                out[head][q][k] = float(table[head, rel])
    return mx.array(out)


def test_expand_rpe_matches_reference_orientation():
    window, heads = 4, 3
    side = 2 * window - 1
    mx.random.seed(0)
    table = mx.random.normal(shape=(heads, side * side))

    dense = _expand_rpe(table, heads)

    assert dense.shape == (heads, window * window, window * window)
    assert mx.allclose(dense, _reference_bias(table, window)).item()
    # the trained table is not symmetric, so the transposed expansion (the
    # regressed orientation) must NOT match
    assert not mx.allclose(dense, dense.transpose(0, 2, 1)).item()
