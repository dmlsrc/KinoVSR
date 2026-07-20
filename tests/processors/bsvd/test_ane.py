"""BSVD on the Neural Engine: the None-flow schedule mirror, backend
selection plumbing, and (slow) end-to-end parity against the MLX net."""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinovsr.processors import bsvd as B
from kinovsr.processors.bsvd import ane as A

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Schedule mirror vs the instrumented product classes
# --------------------------------------------------------------------------

def _synthetic_block(cin: int, seed: int) -> dict:
    """A structurally faithful _DenBlock param table with tiny channels.

    The None-propagation schedule depends only on the architecture, never
    on weight values or channel widths, so eight-channel random weights at
    16x16 reproduce it exactly and keep the instrumented truth run fast
    and independent of the real checkpoint.
    """
    keys = mx.random.split(mx.random.key(seed), 16)
    index = 0

    def conv(out_channels: int, in_channels: int, stride: int = 1):
        nonlocal index
        weight = mx.random.normal(
            shape=(out_channels, 3, 3, in_channels), key=keys[index]) * 0.1
        bias = mx.zeros((out_channels,))
        index += 1
        return weight.astype(mx.float32), bias, stride

    return {
        "inc0": conv(8, cin), "inc3": conv(8, 8),
        "d0": conv(8, 8, stride=2), "d0c1": conv(8, 8), "d0c2": conv(8, 8),
        "d1": conv(8, 8, stride=2), "d1c1": conv(8, 8), "d1c2": conv(8, 8),
        "u2c1": conv(8, 8), "u2c2": conv(8, 8), "u2": conv(32, 8),
        "u1c1": conv(8, 8), "u1c2": conv(8, 8), "u1": conv(32, 8),
        "out0": conv(8, 8), "out3": conv(4, 8),
    }


def _instrumented_schedule(length: int, steps: int = 16):
    """Derive the schedule from the real product classes, instrumented the
    way scripts/dev/probe_bsvd_ane.py derives it: patch the CLASS (an
    instance attribute does not intercept ``self.c1(x)``), record which
    units see None, which are unprimed, and which skip pushes are real."""
    temp1 = B._DenBlock(_synthetic_block(4, seed=11))
    temp2 = B._DenBlock(_synthetic_block(4, seed=17))
    order = ("down0.c1", "down0.c2", "down1.c1", "down1.c2",
             "up2.c1", "up2.c2", "up1.c1", "up1.c2")
    units = [getattr(getattr(block, name.split(".")[0]), name.split(".")[1])
             for block in (temp1, temp2) for name in order]
    lines = [getattr(block, label) for block in (temp1, temp2)
             for label in ("skip1", "skip2", "skip3")]
    for probe_id, unit in enumerate(units):
        unit._probe_id = probe_id
    for probe_id, line in enumerate(lines):
        line._probe_id = probe_id

    record = {"gates": None, "pushes": None, "unprimed": None}
    original_call = B._BiBufferConv.__call__
    original_push = B._MemSkip.push

    def patched_call(self, input_right):
        identity = getattr(self, "_probe_id", None)
        if identity is not None and record["gates"] is not None:
            record["gates"][identity] = input_right is None
        if identity is not None and record["unprimed"] is not None:
            record["unprimed"][identity] = self._center is None
        return original_call(self, input_right)

    def patched_push(self, value):
        identity = getattr(self, "_probe_id", None)
        if identity is not None and record["pushes"] is not None:
            record["pushes"][identity] = value is not None
        return original_push(self, value)

    B._BiBufferConv.__call__ = patched_call
    B._MemSkip.push = patched_push
    try:
        unprimed, fill_emits = [], []
        for i in range(length):
            frame = mx.random.uniform(shape=(1, 16, 16, 4),
                                      key=mx.random.key(300 + i))
            record["unprimed"] = [False] * 16
            result = temp2(temp1(frame))
            unprimed.append(list(record["unprimed"]))
            fill_emits.append(result is not None)
        gates, pushes, emits = [], [], []
        for _ in range(steps):
            record["gates"], record["pushes"] = [False] * 16, [False] * 6
            record["unprimed"] = [False] * 16
            result = temp2(temp1(None))
            unprimed.append(list(record["unprimed"]))
            gates.append(list(record["gates"]))
            pushes.append(list(record["pushes"]))
            emits.append(result is not None)
    finally:
        B._BiBufferConv.__call__ = original_call
        B._MemSkip.push = original_push
    total = len(unprimed)
    writes = [[0.0 if (unprimed[k][i] and (k + 1 >= total
                                           or not unprimed[k + 1][i]))
               else 1.0 for i in range(16)] for k in range(total)]
    return {"gates": gates, "pushes": pushes, "emits": emits,
            "writes": writes, "fill_emits": fill_emits}


def _mirror_schedule(length: int):
    """The same schedule as the ANE backend derives it: the boolean mirror
    for the fill, and the production ``_assemble_tail`` for the drain."""
    mirror = A._NoneFlowNet()
    fill_writes, fill_emits = [], []
    for _ in range(length):
        record = mirror.step(True)
        fill_writes.append(
            [0.0 if record.primes[i] else 1.0 for i in range(16)])
        fill_emits.append(record.out_real)
    shim = object.__new__(A.AneBSVD)
    shim._mirror = mirror
    tail = A.AneBSVD._assemble_tail(shim)
    return {"fill_writes": fill_writes, "fill_emits": fill_emits,
            "tail": tail}


class TestNoneFlowMirror:
    @pytest.mark.parametrize(
        "length", [1, 2, 3, 4, 7, 8, 15, 16, 17, 24, 33, 48])
    def test_schedule_matches_the_instrumented_product_classes(self, length):
        truth = _instrumented_schedule(length)
        mine = _mirror_schedule(length)

        assert mine["fill_emits"] == truth["fill_emits"]
        assert mine["fill_writes"] == truth["writes"][:length]
        assert [entry["emit"] for entry in mine["tail"]] == truth["emits"]
        assert [entry["pushes"] for entry in mine["tail"]] == truth["pushes"]
        for k, entry in enumerate(mine["tail"]):
            want_gate = [0.0 if truth["gates"][k][i] else 1.0
                         for i in range(16)]
            assert entry["gate"] == want_gate, f"tail step {k}"
            assert entry["write"] == truth["writes"][length + k], \
                f"tail step {k}"

    def test_every_input_is_emitted_exactly_once(self):
        for length in (1, 5, 16, 20):
            truth = _instrumented_schedule(length)
            emitted = sum(truth["fill_emits"]) + sum(truth["emits"])
            assert emitted == length


# --------------------------------------------------------------------------
# Geometry envelope: width alignment and reflect padding
# --------------------------------------------------------------------------

class TestGeometry:
    def test_pad_width_reflect_mirrors_the_right_edge(self):
        x = mx.arange(2 * 3 * 6 * 1, dtype=mx.float32).reshape(2, 3, 6, 1)
        padded = A._pad_width_reflect(x, 9)
        mx.eval(padded)
        assert padded.shape == (2, 3, 9, 1)
        assert float(mx.abs(padded[:, :, :6] - x).max()) == 0.0
        # Reflection excludes the edge column: [.. 3 4 5] -> [.. 3 4 5 4 3 2]
        want = x[:, :, 2:5, :][:, :, ::-1, :]
        assert float(mx.abs(padded[:, :, 6:] - want).max()) == 0.0
        same = A._pad_width_reflect(x, 6)
        assert same is x

    def test_unaligned_graph_width_is_refused_by_build_runner(self):
        # Geometry checks run before the params are touched, so no weights
        # are needed to assert the guard.
        with pytest.raises(RuntimeError, match="multiple of 128"):
            A.build_runner(None, 4, 288, 352)

    def test_below_floor_is_refused(self):
        with pytest.raises(RuntimeError, match="verified ANE floor"):
            A.build_runner(None, 4, 92, 128)


# --------------------------------------------------------------------------
# Backend selection plumbing
# --------------------------------------------------------------------------

class TestBackendSelection:
    def _parse(self, raw, settings=None):
        from kinovsr.processors.bsvd.factory import FACTORY
        from kinovsr.processors.capabilities import Capability
        from kinovsr.settings import Settings

        return FACTORY.parse_config(
            raw, capability=Capability.DENOISE, profile=None,
            settings=settings or Settings())

    def test_default_backend_is_mlx(self):
        assert self._parse({}).backend == "mlx"

    def test_stage_table_selects_ane(self):
        assert self._parse({"backend": "ane"}).backend == "ane"

    def test_settings_default_flows_when_table_is_silent(self):
        from kinovsr.settings import Settings

        settings = Settings(bsvd_backend="ane")
        assert self._parse({}, settings).backend == "ane"
        assert self._parse({"backend": "mlx"}, settings).backend == "mlx"

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="backend must be one of"):
            self._parse({"backend": "gpu"})

    def test_denoiser_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="unknown BSVD backend"):
            B.BsvdDenoiser(B.default_weights_path("c64"), backend="npu")

    def test_ane_backend_is_fp16_only(self):
        with pytest.raises(ValueError, match="fp16 only"):
            A.AneBSVD("unused.safetensors", dtype=mx.float32)

    def test_cli_flag_is_published_to_the_process_default(self, monkeypatch):
        from kinovsr.cli.args import build_parser
        from kinovsr.cli.config import assemble

        monkeypatch.delenv("BSVD_BACKEND", raising=False)
        args = build_parser().parse_args(
            ["--video", "clip.mp4", "--bsvd-backend", "ane"])
        invocation = assemble(args)
        assert invocation.settings.bsvd_backend == "ane"

    def test_cli_default_is_mlx(self, monkeypatch):
        from kinovsr.cli.args import build_parser
        from kinovsr.cli.config import assemble

        monkeypatch.delenv("BSVD_BACKEND", raising=False)
        args = build_parser().parse_args(["--video", "clip.mp4"])
        assert assemble(args).settings.bsvd_backend == "mlx"


# --------------------------------------------------------------------------
# End-to-end parity against the MLX net (slow: converts a real model)
# --------------------------------------------------------------------------

def _real_frames(count: int, channels: int, height: int, width: int):
    base = mx.random.uniform(shape=(1, height, width, channels),
                             key=mx.random.key(20260719))
    frames = []
    for index in range(count):
        noise = mx.random.uniform(shape=(1, height, width, channels),
                                  key=mx.random.key(500 + index))
        frame = mx.clip(base * 0.85 + noise * 0.15 + index * 0.011, 0.0, 1.0)
        frame = frame.astype(mx.float16)
        mx.eval(frame)
        frames.append(frame)
    return frames


@pytest.fixture(scope="module")
def _module_cache(tmp_path_factory):
    """One conversion cache for the whole module: BSVD ANE conversion at
    96x128 costs tens of seconds, so the streams below share it."""
    import os

    from kinovsr.settings import _reset_default_settings as reset_settings

    cache = tmp_path_factory.mktemp("bsvd-ane-cache")
    previous = os.environ.get("KINOVSR_CACHE_DIR")
    os.environ["KINOVSR_CACHE_DIR"] = str(cache)
    reset_settings()
    A._VERIFIED.clear()
    yield cache
    if previous is None:
        os.environ.pop("KINOVSR_CACHE_DIR", None)
    else:
        os.environ["KINOVSR_CACHE_DIR"] = previous
    reset_settings()
    A._VERIFIED.clear()


@pytest.fixture(scope="module")
def ane_net(_module_cache):
    pytest.importorskip("CoreML")
    weights = B.default_weights_path("c64")
    if not weights.is_file():
        pytest.skip(f"bsvd weights not available at {weights}")
    net = A.AneBSVD(weights, dtype=mx.float16)
    try:
        net._ensure_runner(96, 128)
    except Exception as exc:  # noqa: BLE001 - environment, not correctness
        pytest.skip(f"BSVD ANE engine unavailable here: {exc}")
    net.reset()
    return net


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_weights
class TestAneParity:
    def _drive_both(self, ane_net, length: int):
        """Drive both nets; the ANE emissions lag the reference by exactly
        one step (the in-flight dispatch), with identical content order."""
        reference = B.BSVD(B.default_weights_path("c64"), dtype=mx.float32)
        frames = _real_frames(length, ane_net.input_channels, 96, 128)
        ane_net.reset()
        reference.reset()
        ane_outs, ref_outs = [], []
        for index, frame in enumerate(frames + [None] * 17):
            ane_out = ane_net.step(frame)
            ref_out = reference.step(
                None if frame is None else frame.astype(mx.float32))
            if ane_out is not None:
                ane_outs.append((index, ane_out))
            if ref_out is not None:
                ref_outs.append((index, ref_out))
        ane_net.reset()
        assert len(ane_outs) == len(ref_outs) == length
        assert ([i for i, _ in ane_outs]
                == [i + 1 for i, _ in ref_outs]), "one-step dispatch lag"
        pairs = []
        for (_, ane_out), (_, ref_out) in zip(ane_outs, ref_outs,
                                              strict=True):
            delta = mx.abs(ane_out.astype(mx.float32) - ref_out)
            mx.eval(delta)
            pairs.append((float(delta.mean()), float(delta.max())))
        return pairs

    def test_full_stream_matches_the_product_fp32_net(self, ane_net):
        pairs = self._drive_both(ane_net, 24)
        means = [mean for mean, _ in pairs]
        # Doc-20 measured the gated schedule at ~3e-4 mean against the
        # product's own fp32 output, and the UNGATED failure mode at
        # 9.5e-4 and up - the bound sits between them.
        assert means[0] < 8e-4, f"first emitted frame off: {means[0]:.3e}"
        assert max(means) < 8e-4, f"worst frame {max(means):.3e}"

    def test_sub_fill_clip_emits_entirely_through_the_drain(self, ane_net):
        pairs = self._drive_both(ane_net, 4)
        means = [mean for mean, _ in pairs]
        assert max(means) < 8e-4, f"worst frame {max(means):.3e}"

    @pytest.mark.usefixtures("_module_cache")
    def test_cif_geometry_runs_padded_and_matches(self):
        """352x288 (CIF) is the geometry that fails UNPADDED with the ANE
        status=0x1d alignment error; through the reflect-pad path it must
        run and match the fp32 reference. The right band differs by
        boundary CONTEXT (reflected content vs the frame edge the MLX
        path sees), so it gets its own bound; the interior must sit at
        the same ~3e-4 parity as aligned geometries (measured worst
        3.2e-4 interior, 3.5e-3 band, 9.0e-4 full frame)."""
        weights = B.default_weights_path("c64")
        if not weights.is_file():
            pytest.skip(f"bsvd weights not available at {weights}")
        net = A.AneBSVD(weights, dtype=mx.float16)
        reference = B.BSVD(weights, dtype=mx.float32)
        frames = _real_frames(20, net.input_channels, 288, 352)
        try:
            net._ensure_runner(288, 352)
        except Exception as exc:  # noqa: BLE001 - environment, not correctness
            pytest.skip(f"BSVD ANE engine unavailable here: {exc}")
        assert net._padded_width == 384
        net.reset()
        ane_outs, ref_outs = [], []
        for frame in frames + [None] * 17:
            ane_out = net.step(frame)
            ref_out = reference.step(
                None if frame is None else frame.astype(mx.float32))
            if ane_out is not None:
                ane_outs.append(ane_out)
            if ref_out is not None:
                ref_outs.append(ref_out)
        full, interior, band = [], [], []
        for ane_out, ref_out in zip(ane_outs, ref_outs, strict=True):
            assert ane_out.shape == ref_out.shape
            delta = mx.abs(ane_out.astype(mx.float32) - ref_out)
            mx.eval(delta)
            full.append(float(delta.mean()))
            interior.append(float(delta[:, :, :352 - 64, :].mean()))
            band.append(float(delta[:, :, 352 - 64:, :].mean()))
        assert len(full) == 20
        assert max(interior) < 8e-4, f"interior {max(interior):.3e}"
        assert max(band) < 8e-3, f"right band {max(band):.3e}"
        assert max(full) < 2e-3, f"full frame {max(full):.3e}"

    def test_denoiser_backend_parity(self, ane_net):
        ane = B.BsvdDenoiser(strength=0.4, backend="ane")
        mlx_ref = B.BsvdDenoiser(strength=0.4, backend="mlx")
        frames = [f[0].astype(mx.float32)[..., :3]
                  for f in _real_frames(8, 3, 96, 128)]
        for frame in frames:
            mx.eval(frame)
        ane_out, ref_out = [], []
        for index, frame in enumerate(frames):
            ane_out += ane.feed(frame, token=index)
            ref_out += mlx_ref.feed(frame, token=index)
        ane_out += ane.flush()
        ref_out += mlx_ref.flush()
        assert [token for _, token in ane_out] == list(range(8))
        assert [token for _, token in ref_out] == list(range(8))
        deltas = [float(mx.abs(a - r).mean().item())
                  for (a, _), (r, _) in zip(ane_out, ref_out, strict=True)]
        # Both backends run fp16 here, each ~3e-4 from fp32, so their
        # mutual distance stays within a few fp16 quanta.
        assert max(deltas) < 2e-3, f"worst frame {max(deltas):.3e}"
