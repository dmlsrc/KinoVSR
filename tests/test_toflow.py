"""TOFlow tests (weights are bundled, so these run everywhere).

Pins the direct MLX forward (net.py) against the plain Torch7-graph
interpretation, the half-resolution flow option against the faithful
network, and the streaming latency contract. Small frames keep the runtime
in check.
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


def test_direct_forward_matches_plain_interpretation():
    from LTX_2_MLX.videotoolbox.toflow import (
        _TOFlowGraph,
        _graph_path_for,
        resolve_weights,
    )
    clip = _clip()
    fast = TOFlowDenoiser(variant="denoise")
    assert fast.net.engine == "direct"
    plain = TOFlowDenoiser(variant="denoise")
    wp = resolve_weights("denoise")
    g = _TOFlowGraph(wp, _graph_path_for(wp), dtype=mx.float32)
    g._batch_par = {}
    g._compiled = {}
    g.forward = lambda inputs: g._eval(g.root, inputs)
    plain.net.net = g
    plain.net.engine = "interp"
    a = _run(fast, clip)
    b = _run(plain, clip)
    worst = max(float(mx.max(mx.abs(x - y))) for x, y in zip(a, b))
    assert worst < 1e-3, f"direct forward diverged from interpreter by {worst}"


def test_half_flow_tracks_full():
    clip = _clip()
    full = TOFlowDenoiser(variant="deblock")
    assert full.net.engine == "direct"
    half = TOFlowDenoiser(variant="deblock", flow_scale="half")
    a = _run(full, clip)
    b = _run(half, clip)
    worst = max(float(mx.max(mx.abs(x - y))) for x, y in zip(a, b))
    mean = max(float(mx.mean(mx.abs(x - y))) for x, y in zip(a, b))
    # half flow skips the full-res refinement: outputs stay close but not
    # identical (at real resolutions they agree at ~35 dB; this tiny frame
    # exaggerates pyramid differences). Catastrophic divergence = broke.
    assert mean < 0.035 and worst < 0.4, f"half flow diverged ({mean}, {worst})"


def test_latency_and_flush():
    clip = _clip()
    den = TOFlowDenoiser(variant="denoise")
    n = 0
    for f in clip:
        n += len(den.feed(f))
    assert n == T - 3                     # 7-frame window: 3 frames lookahead
    n += len(den.flush())
    assert n == T
