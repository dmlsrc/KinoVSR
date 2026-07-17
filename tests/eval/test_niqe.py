"""NIQE feature regressions: the AGGD alpha features must live on the
reciprocal gamma curve (the inverted match froze them at the grid max)."""

import mlx.core as mx
import pytest

from kinovsr.eval.niqe import _GAM, PATCH, image_features

pytestmark = pytest.mark.unit


def _smoothed_noise(k: int) -> mx.array:
    mx.random.seed(11)
    base = mx.random.uniform(shape=(2 * PATCH, 2 * PATCH))
    kern = mx.ones((1, k, k, 1)) / float(k * k)
    r = k // 2
    padded = mx.pad(base[None, :, :, None], [(0, 0), (r, r), (r, r), (0, 0)])
    return mx.conv2d(padded, kern)[0, :, :, 0]


def test_aggd_alpha_features_vary_and_respond_to_blur():
    sharp = image_features(_smoothed_noise(3))
    softer = image_features(_smoothed_noise(15))
    assert sharp.shape[0] > 0 and sharp.shape[1] == 36

    # AGGD alpha features: scale-1 cols 2/6/10/14, scale-2 cols 20/24/28/32
    cols = [2, 6, 10, 14, 20, 24, 28, 32]
    a_sharp = sharp[:, cols]
    a_soft = softer[:, cols]
    gmax = float(_GAM[-1])

    # the regressed matching saturated every AGGD alpha at the grid max
    assert float(mx.max(a_sharp)) < gmax
    assert float(mx.max(a_soft)) < gmax
    # and being constant, they could not respond to blur; now they must
    assert float(mx.max(mx.abs(a_sharp - a_soft))) > 1e-3
