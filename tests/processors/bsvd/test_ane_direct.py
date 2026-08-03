"""BSVD direct AppleNeuralEngine route: half-emission contract, engine
selection, ring semantics, and (slow) parity through the private path."""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinovsr.processors import bsvd as B
from kinovsr.processors.bsvd import ane as A
from kinovsr.processors.bsvd import ane_direct as D
from kinovsr.settings import Settings


def _synthetic_block(cin: int, seed: int) -> dict:
    """Tiny structurally faithful _DenBlock params (see test_ane)."""
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


def _synthetic_params() -> tuple[dict, int]:
    return {"temp1": _synthetic_block(4, seed=11),
            "temp2": _synthetic_block(4, seed=17)}, 4


@pytest.mark.unit
class TestHalfEmission:
    def test_port_contract(self):
        params, cin = _synthetic_params()
        for block in A.BLOCKS:
            graph, inputs, states, outputs = A._emit_graph(
                params, cin, 96, 128, blocks=(block,), explicit_state=True)
            del graph
            assert states == [], "explicit-state halves carry no MLState"
            names = [name for name, _shape in inputs]
            assert names[:3] == ["frame", "gate", "write"]
            assert [f"skip_{i}" for i in range(3)] == [
                n for n in names if n.startswith("skip_")]
            assert [f"st{i}" for i in range(8)] == [
                n for n in names if n.startswith("st")]
            assert outputs[0] == "out"
            assert [f"skip_out_{i}" for i in range(3)] == [
                n for n in outputs if n.startswith("skip_out_")]
            state_outputs = [n for n in outputs if "_state_" in n]
            assert len(state_outputs) == 8
            for (token, _divisor), name in zip(A.BIBUF, state_outputs,
                                               strict=True):
                assert token in name, (token, name)

    def test_halves_never_emit_the_island(self):
        params, cin = _synthetic_params()
        height, width = 1080, 1920   # far above ISLAND_MIN_PIXELS
        _graph, inputs, _states, outputs = A._emit_graph(
            params, cin, height, width, blocks=("temp1",),
            explicit_state=True)
        assert "island_bias" not in [name for name, _shape in inputs]
        assert not any("island" in name for name in outputs)

    def test_default_emission_is_parameter_stable(self):
        params, cin = _synthetic_params()
        plain = A._emit_graph(params, cin, 96, 128)
        spelled = A._emit_graph(params, cin, 96, 128,
                                blocks=A.BLOCKS, explicit_state=False)
        plain_bytes = plain[0].finish(plain[1], plain[2], plain[3], "t")
        spelled_bytes = spelled[0].finish(spelled[1], spelled[2],
                                          spelled[3], "t")
        assert plain_bytes == spelled_bytes


@pytest.mark.unit
class TestSelection:
    LARGE = (1080, 1920)
    SMALL = (480, 640)

    def _mode(self, monkeypatch, value):
        monkeypatch.setattr(
            D, "default_settings", lambda: Settings(bsvd_direct=value))

    def test_off_never_engages(self, monkeypatch):
        self._mode(monkeypatch, "off")
        assert not D.should_use(*self.LARGE)

    def test_force_engages_everywhere(self, monkeypatch):
        self._mode(monkeypatch, "force")
        assert D.should_use(*self.SMALL)

    def test_auto_small_geometry_stays_on_core_ml(self, monkeypatch):
        self._mode(monkeypatch, "auto")
        assert not D.should_use(*self.SMALL)

    def test_auto_large_geometry_engages_when_available(self, monkeypatch):
        self._mode(monkeypatch, "auto")
        monkeypatch.setattr(D.direct, "available", lambda: True)
        assert D.should_use(*self.LARGE)

    def test_auto_large_geometry_falls_back_when_unavailable(
            self, monkeypatch):
        self._mode(monkeypatch, "auto")
        monkeypatch.setattr(D.direct, "available", lambda: False)
        assert not D.should_use(*self.LARGE)

    def test_require_raises_when_unavailable(self, monkeypatch):
        self._mode(monkeypatch, "require")
        monkeypatch.setattr(D.direct, "available", lambda: False)

        def refuse():
            raise D.direct.DirectUnavailable("test refusal")
        monkeypatch.setattr(D.direct, "preflight", refuse)
        with pytest.raises(D.direct.DirectUnavailable):
            D.should_use(*self.LARGE)

    def test_unknown_mode_is_rejected(self, monkeypatch):
        self._mode(monkeypatch, "sometimes")
        with pytest.raises(RuntimeError, match="bsvd_direct"):
            D.should_use(*self.LARGE)


class _FakeSurface:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes
        self.zeroed = False

    def zero(self):
        self.zeroed = True


class _FakePort:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes
        self.surface = _FakeSurface(nbytes)


@pytest.mark.unit
class TestSkipRing:
    def test_ring_matches_a_fifo_model(self, monkeypatch):
        """The ring must present: the push from ``depth`` dispatches ago
        when that push was real, the zero surface otherwise."""
        monkeypatch.setattr(D.direct, "Surface", _FakeSurface)
        depth = 4
        ring = D._SkipRing(_FakePort(64), _FakePort(64), depth)
        assert ring.zero.zeroed
        pushes = [True, False, True, True, False, True, True, True, False,
                  True]
        history: list[tuple[object, bool]] = []
        for index, pushed in enumerate(pushes):
            bound = ring.bind()
            if index < depth or not history[index - depth][1]:
                assert bound is ring.zero, f"step {index} must bind zero"
            else:
                assert bound is history[index - depth][0], (
                    f"step {index} must bind the push from {index - depth}")
            writing = ring.spare
            ring.rotate()
            if not pushed:
                ring.zero_last_push()
            history.append((writing, pushed))

    def test_reset_invalidates_without_reallocating(self, monkeypatch):
        monkeypatch.setattr(D.direct, "Surface", _FakeSurface)
        ring = D._SkipRing(_FakePort(64), _FakePort(64), 3)
        for _ in range(5):
            ring.rotate()
        slots_before = list(ring.slots)
        ring.reset()
        assert ring.valid == [False, False, False]
        assert ring.cursor == 0
        assert ring.slots == slots_before


class _FakeVectorPort:
    """A (1,16,1,1) vector port with a 4-byte plane stride."""

    def __init__(self):
        self.backing = bytearray(16 * 4)

        class _Surface:
            def view(inner):  # noqa: N805 - tiny stub
                return memoryview(self.backing)
        self.surface = _Surface()

    @staticmethod
    def strides():
        return 256, 256, 4, 2


@pytest.mark.unit
class TestWriteLanes:
    def _lanes(self, backing: bytearray) -> list[bytes]:
        return [bytes(backing[lane * 4: lane * 4 + 2]) for lane in range(16)]

    def test_lane_split_and_ones_padding(self):
        import struct

        vector = b"".join(struct.pack("<e", float(v)) for v in range(16))
        one = struct.pack("<e", 1.0)
        for offset in (0, 8):
            port = _FakeVectorPort()
            D.DirectChainRunner._write_lanes(port, vector, offset)
            lanes = self._lanes(port.backing)
            for lane in range(8):
                assert lanes[lane] == vector[
                    (offset + lane) * 2: (offset + lane) * 2 + 2]
            assert lanes[8:] == [one] * 8

    def test_none_means_all_ones(self):
        import struct

        port = _FakeVectorPort()
        D.DirectChainRunner._write_lanes(port, None, 0)
        assert self._lanes(port.backing) == [struct.pack("<e", 1.0)] * 16


# --------------------------------------------------------------------------
# End-to-end parity through the private dispatch (slow: real hardware)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _direct_cache(tmp_path_factory):
    import os

    from kinovsr.settings import _reset_default_settings as reset_settings

    cache = tmp_path_factory.mktemp("bsvd-ane-direct-cache")
    previous = {name: os.environ.get(name)
                for name in ("KINOVSR_CACHE_DIR", "KINOVSR_BSVD_DIRECT")}
    os.environ["KINOVSR_CACHE_DIR"] = str(cache)
    os.environ["KINOVSR_BSVD_DIRECT"] = "force"
    reset_settings()
    A._VERIFIED.clear()
    yield cache
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    reset_settings()
    A._VERIFIED.clear()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_weights
class TestDirectParity:
    @pytest.mark.usefixtures("_direct_cache")
    def test_forced_direct_stream_matches_the_fp32_net(self):
        pytest.importorskip("CoreML")
        from kinovsr.native.anemil import direct

        if not direct.available():
            pytest.skip("private ANE dispatch unavailable here")
        weights = B.default_weights_path("c64")
        if not weights.is_file():
            pytest.skip(f"bsvd weights not available at {weights}")

        net = A.AneBSVD(weights, dtype=mx.float16)
        try:
            net._ensure_runner(96, 128)
        except Exception as exc:  # noqa: BLE001 - environment, not correctness
            net.close()
            pytest.skip(f"direct BSVD engine unavailable here: {exc}")
        assert isinstance(net._runner, D.DirectChainRunner)

        reference = B.BSVD(weights, dtype=mx.float32)
        base = mx.random.uniform(shape=(1, 96, 128, net.input_channels),
                                 key=mx.random.key(20260723))
        frames = []
        for index in range(24):
            noise = mx.random.uniform(
                shape=(1, 96, 128, net.input_channels),
                key=mx.random.key(900 + index))
            frame = mx.clip(base * 0.85 + noise * 0.15 + index * 0.011,
                            0.0, 1.0).astype(mx.float16)
            mx.eval(frame)
            frames.append(frame)
        try:
            ane_outs, ref_outs = [], []
            for frame in frames + [None] * 17:
                out = net.step(frame)
                ref = reference.step(
                    None if frame is None else frame.astype(mx.float32))
                if out is not None:
                    ane_outs.append(out)
                if ref is not None:
                    ref_outs.append(ref)
            assert len(ane_outs) == len(ref_outs) == len(frames)
            means = []
            for out, ref in zip(ane_outs, ref_outs, strict=True):
                delta = mx.abs(out.astype(mx.float32) - ref)
                mx.eval(delta)
                means.append(float(delta.mean()))
            assert max(means) < 8e-4, f"worst frame {max(means):.3e}"
        finally:
            net.close()
