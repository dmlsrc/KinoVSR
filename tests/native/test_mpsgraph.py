"""Generic MPSGraph machinery: builder ops, named I/O, safe pixel shuffle.

These run on the Metal device so they exercise the graph plumbing without
depending on ANE placement; the BSVD backend tests cover the ANE path end
to end.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinovsr.native import mpsgraph as mg

pytestmark = pytest.mark.unit


def _pixel_shuffle_reference(x: mx.array, block: int) -> mx.array:
    n, c4, h, w = x.shape
    c = c4 // (block * block)
    y = x.reshape(n, c, block, block, h, w)
    y = mx.transpose(y, (0, 1, 4, 2, 5, 3))
    return y.reshape(n, c, h * block, w * block)


class TestGraphBuilder:
    def test_rejects_unknown_dtype(self):
        with pytest.raises(ValueError, match="MPSDataType"):
            mg.GraphBuilder(123)

    def test_named_feeds_and_results_round_trip(self):
        b = mg.GraphBuilder(mg.FLOAT32)
        left = b.placeholder((1, 2, 4, 4), "left")
        right = b.placeholder((1, 2, 4, 4), "right")
        total = b.add(left, right, "total")
        diff = b.subtract(left, right, "diff")
        graph = mg.compile_graph(
            b, [("total", total, (1, 2, 4, 4)), ("diff", diff, (1, 2, 4, 4))],
            device=mg.DEVICE_GPU)
        a = mx.arange(32, dtype=mx.float32).reshape(1, 2, 4, 4)
        c = mx.ones((1, 2, 4, 4), dtype=mx.float32) * 2.0
        out = graph.run({"left": a, "right": c})
        assert mx.array_equal(out["total"], a + c)
        assert mx.array_equal(out["diff"], a - c)
        assert set(graph.feed_names) == {"left", "right"}

    def test_pixel_shuffle_biased_matches_the_reference_arrangement(self):
        """The bias must land per PRE-shuffle channel: output channel c at
        spatial phase (i, j) receives bias[c*4 + 2i + j].  This is the
        spelling that dodges the ANE fused-bias defect documented in the
        module; here it is checked for plain correctness on the GPU."""
        keys = mx.random.split(mx.random.key(9), 3)
        x = mx.random.normal((1, 8, 6, 10), key=keys[0])
        weight = mx.random.normal((8, 8, 3, 3), key=keys[1]) * 0.2
        bias = mx.random.normal((8,), key=keys[2])

        b = mg.GraphBuilder(mg.FLOAT32)
        src = b.placeholder((1, 8, 6, 10), "src")
        conv = b.conv2d(src, weight, None, name="conv")
        shuffled = b.pixel_shuffle_biased(
            conv, bias, channels=8, height=6, width=10, name="ps")
        graph = mg.compile_graph(
            b, [("ps", shuffled, (1, 2, 12, 20)),
                ("conv", conv, (1, 8, 6, 10))], device=mg.DEVICE_GPU)
        out = graph.run({"src": x})

        expected = _pixel_shuffle_reference(
            out["conv"] + bias.reshape(1, 8, 1, 1), 2)
        assert mx.max(mx.abs(out["ps"] - expected)).item() < 1e-5

    def test_pixel_shuffle_biased_rejects_indivisible_channels(self):
        b = mg.GraphBuilder(mg.FLOAT16)
        src = b.placeholder((1, 6, 4, 4), "src")
        with pytest.raises(ValueError, match="divisible"):
            b.pixel_shuffle_biased(src, mx.zeros((6,)), channels=6,
                                   height=4, width=4, name="ps")
