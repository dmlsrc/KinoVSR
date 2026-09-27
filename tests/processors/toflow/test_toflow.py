"""TOFlow tests (weights are bundled, so these run everywhere).

Pins the direct MLX forward (net.py) against the plain Torch7-graph
interpretation, the half-resolution flow option against the faithful
network, and the streaming latency contract. Small frames keep the runtime
in check.
"""

import copy
import json
import re
from pathlib import Path

import mlx.core as mx
import pytest

from kinovsr.processors import toflow as toflow_family
from kinovsr.processors.toflow import TOFlow, TOFlowDenoiser, _graph_path_for, resolve_weights
from kinovsr.processors.toflow.net import TOFlowDirect

H, W, T = 96, 128, 10


def _clip():
    mx.random.seed(5)
    base = mx.random.uniform(shape=(H, W, 3)) * 0.5 + 0.25
    return [mx.clip(base + 0.05 * mx.random.normal(shape=(H, W, 3)), 0, 1) for _ in range(T)]


def _run(den, clip):
    outs = []
    for i, f in enumerate(clip):
        outs += den.feed(f, token=i)
    outs += den.flush()
    assert [tok for _o, tok in outs] == list(range(T))
    return [o for o, _t in outs]


def test_direct_forward_matches_plain_interpretation():
    from kinovsr.processors.toflow import (
        _graph_path_for,
        _TOFlowGraph,
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
    worst = max(float(mx.max(mx.abs(x - y))) for x, y in zip(a, b, strict=True))
    assert worst < 1e-3, f"direct forward diverged from interpreter by {worst}"


def test_reduced_flow_tracks_full():
    clip = _clip()
    full = TOFlowDenoiser(variant="deblock")
    assert full.net.engine == "direct"
    a = _run(full, clip)
    for scale in ("half", "quarter"):
        red = TOFlowDenoiser(variant="deblock", flow_scale=scale)
        b = _run(red, clip)
        worst = max(float(mx.max(mx.abs(x - y))) for x, y in zip(a, b, strict=True))
        mean = max(float(mx.mean(mx.abs(x - y))) for x, y in zip(a, b, strict=True))
        # reduced flow skips fine refinement levels: outputs stay close but
        # not identical (at real resolutions they agree at ~35 dB; this tiny
        # frame exaggerates pyramid differences). Catastrophic = broke.
        assert mean < 0.05, f"{scale} diverged ({mean}, {worst})"
    assert worst < 0.5, f"{scale} diverged ({mean}, {worst})"


def test_passes_cascade_matches_explicit_chain():
    clip = _clip()
    frames = clip
    for _ in range(2):
        den = TOFlowDenoiser(variant="deblock", flow_scale="quarter")
        frames = _run(den, frames)
    cas = _run(TOFlowDenoiser(variant="deblock", flow_scale="quarter", passes=2), clip)
    worst = max(float(mx.max(mx.abs(x - y))) for x, y in zip(frames, cas, strict=True))
    mean = max(float(mx.mean(mx.abs(x - y))) for x, y in zip(frames, cas, strict=True))
    # the cascade reuses pass-1 flow instead of recomputing it on cleaned
    # frames; measured equivalent at real resolutions (~55 dB agreement,
    # PSNR within 0.01 dB) -- this tiny frame amplifies the flow delta, so
    # the thresholds only pin "not broken"
    assert mean < 0.01, f"cascade diverged from explicit chain ({mean}, {worst})"
    assert worst < 0.15, f"cascade diverged from explicit chain ({mean}, {worst})"


def test_latency_and_flush():
    clip = _clip()
    den = TOFlowDenoiser(variant="denoise")
    n = 0
    for f in clip:
        n += len(den.feed(f))
    assert n == T - 3  # 7-frame window: 3 frames lookahead
    n += len(den.flush())
    assert n == T


def _first(node, typ):
    if node.get("type") == typ:
        return node
    for kid in node.get("modules", ()):
        found = _first(kid, typ)
        if found is not None:
            return found
    return None


def _mutated_sep_graph(kind):
    graph = json.loads(_graph_path_for(resolve_weights("denoise")).read_text())
    graph = copy.deepcopy(graph)
    if kind == "selecttable_without_attrs":
        del _first(graph["root"], "nn.SelectTable")["attrs"]
    elif kind == "biasless_conv":
        del _first(graph["root"], "nn.SpatialConvolution")["params"]["bias"]
    else:
        _first(graph["root"], "nn.ConcatTable")["modules"] = []
    return graph


@pytest.mark.parametrize("kind", ["selecttable_without_attrs", "biasless_conv", "empty_concat"])
def test_unusual_sep_graphs_fall_back_to_the_interpreter(tmp_path, kind):
    # These shapes raised KeyError or IndexError out of the auto engine, which
    # promises the interpreter for any graph the direct forward cannot take.
    weights = resolve_weights("denoise")
    graph = _mutated_sep_graph(kind)
    with pytest.raises(ValueError, match="TOFlow direct"):
        TOFlowDirect(graph, dict(mx.load(str(weights))), mx.float32)
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    assert TOFlow(weights=str(weights), graph=path).engine == "interp"
    with pytest.raises(ValueError, match="TOFlow direct"):
        TOFlow(weights=str(weights), graph=path, engine="direct")


REPO = Path(__file__).resolve().parents[3]


def test_missing_file_messages_name_the_converter_that_exists(tmp_path, monkeypatch):
    # The messages used to name kinovsr/toflow/convert_t7_to_safetensors.py,
    # a path the family moved away from. They appear when a family's weights
    # file is absent, or its graph JSON beside it.
    missing = tmp_path / "missing.safetensors"
    monkeypatch.setattr(toflow_family, "resolve_weights", lambda _spec=None: missing)
    messages = []
    for build in (
        TOFlowDenoiser,
        toflow_family.TOFlowSrUpscaler,
        toflow_family.TOFlowInterpolator,
    ):
        with pytest.raises(FileNotFoundError) as error:
            build()
        messages.append(str(error.value))
    missing.write_bytes(b"")
    with pytest.raises(FileNotFoundError) as error:
        _graph_path_for(missing)
    messages.append(str(error.value))
    for message in messages:
        named = re.search(r"kinovsr/\S+\.py", message)
        assert named is not None, message
        assert (REPO / named.group(0)).is_file(), message
