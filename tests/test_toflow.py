"""TOFlow interpreter tests (weights are bundled, so these run everywhere).

Pins the two runtime optimizations against the plain interpretation: the
batched evaluation of the cloned per-neighbor branches must be exact (same
math, stacked), and the compiled forward must match within fp32 reorder
noise. Small frames keep the runtime in check.
"""
import mlx.core as mx

from LTX_2_MLX.videotoolbox.toflow import TOFlowDenoiser

H, W, T = 96, 128, 10


def _clip():
    mx.random.seed(5)
    base = mx.random.uniform(shape=(H, W, 3)) * 0.5 + 0.25
    return [mx.clip(base + 0.05 * mx.random.normal(shape=(H, W, 3)), 0, 1)
            for _ in range(T)]


def _run(den, clip):
    outs = []
    for i, f in enumerate(clip):
        outs += den.feed(f, token=i)
    outs += den.flush()
    assert [tok for _o, tok in outs] == list(range(T))
    return [o for o, _t in outs]


def test_batched_branches_match_plain_interpretation():
    clip = _clip()
    fast = TOFlowDenoiser(variant="denoise")
    assert fast.net.net._batch_par, "cloned-branch batching did not engage"
    plain = TOFlowDenoiser(variant="denoise")
    plain.net.net._batch_par = {}
    plain.net.net._compiled = {}
    plain.net.net.forward = lambda inputs: plain.net.net._eval(
        plain.net.net.root, inputs)
    a = _run(fast, clip)
    b = _run(plain, clip)
    worst = max(float(mx.max(mx.abs(x - y))) for x, y in zip(a, b))
    assert worst < 1e-3, f"batched/compiled diverged by {worst}"


def test_latency_and_flush():
    clip = _clip()
    den = TOFlowDenoiser(variant="denoise")
    n = 0
    for f in clip:
        n += len(den.feed(f))
    assert n == T - 3                     # 7-frame window: 3 frames lookahead
    n += len(den.flush())
    assert n == T
