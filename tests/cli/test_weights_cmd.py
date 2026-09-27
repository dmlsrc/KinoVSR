"""The kinovsr weights subcommand: listing, verification, exit codes."""

import array
import io
import pickle
import sys
import types
import zipfile
from collections import OrderedDict
from pathlib import Path

import pytest

from kinovsr.cli.commands.weights import run_weights_command
from kinovsr.cli.main import main

pytestmark = pytest.mark.unit


def test_list_all_owners_exits_zero():
    assert run_weights_command(["list"]) == 0


def test_verify_this_checkout_passes():
    # spynet is bundled (present + artifact hash recorded); the external
    # families report their install state without failing the run.
    assert run_weights_command(["verify"]) == 0


def test_verify_single_owner():
    assert run_weights_command(["verify", "spynet"]) == 0


def test_unknown_owner_exits_two():
    assert run_weights_command(["verify", "nosuchfamily"]) == 2


def test_main_routes_the_weights_subcommand():
    assert main(["weights", "list"]) == 0


def test_factory_profiles_match_their_manifests():
    from kinovsr.modeling.weights import load_registered
    from kinovsr.processors.bsvd.factory import FACTORY as bsvd_factory
    from kinovsr.processors.capabilities import Capability
    from kinovsr.processors.realplksr.factory import FACTORY as realplksr_factory

    bsvd_profiles = bsvd_factory.capabilities[Capability.DENOISE].profiles
    assert set(bsvd_profiles) == set(load_registered("bsvd").profiles)

    plksr_profiles = realplksr_factory.capabilities[Capability.UPSCALE].profiles
    assert set(plksr_profiles) == set(load_registered("realplksr").profiles)


class TestConvertSourceResolution:
    """The converter looks in weights-src/ when the input is not a path."""

    def _resolve(self):
        from kinovsr.cli.commands.weights_convert import _resolve_source

        return _resolve_source

    def test_literal_path_wins(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "weights-src" / "fam").mkdir(parents=True)
        (tmp_path / "weights-src" / "fam" / "x.pth").write_bytes(b"src")
        (tmp_path / "x.pth").write_bytes(b"local")
        assert self._resolve()("x.pth").read_bytes() == b"local"

    def test_bare_name_resolves_uniquely(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "weights-src" / "fam" / "x.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"src")
        resolved = self._resolve()("x.pth")
        assert resolved is not None
        assert resolved.read_bytes() == b"src"

    def test_family_relative_path_resolves(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "weights-src" / "fam" / "x.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"src")
        assert self._resolve()("fam/x.pth") == Path("weights-src/fam/x.pth")

    def test_ambiguous_name_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for fam in ("a", "b"):
            d = tmp_path / "weights-src" / fam
            d.mkdir(parents=True)
            (d / "x.pth").write_bytes(b"src")
        with pytest.raises(SystemExit, match="ambiguous"):
            self._resolve()("x.pth")

    def test_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert self._resolve()("nope.pth") is None

    def test_weights_src_input_requires_output(self, tmp_path, monkeypatch):
        from kinovsr.cli.commands.weights_convert import run_convert

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "weights-src" / "fam" / "x.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"not a real checkpoint")
        assert run_convert(["x.pth"]) == 2


# --- synthesized torch-format checkpoints, no torch installed ---------------
# A minimal fake ``torch`` module tree exists only while *writing* the fixture
# checkpoints (pickle resolves globals at dump time); the converter under test
# never sees it and stays torch-free.


class _FakeStorage:
    def __init__(self, key: str, numel: int) -> None:
        self.key = key
        self.numel = numel


class _FakeTensor:
    def __init__(self, storage: _FakeStorage, shape: tuple[int, ...]) -> None:
        self.storage = storage
        self.shape = shape

    def __reduce_ex__(self, protocol: int):
        stride = []
        accumulated = 1
        for extent in reversed(self.shape):
            stride.append(accumulated)
            accumulated *= extent
        rebuild = sys.modules["torch._utils"]._rebuild_tensor_v2
        return (
            rebuild,
            (self.storage, 0, self.shape, tuple(reversed(stride)), False, None),
        )


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    utils = types.ModuleType("torch._utils")

    def _rebuild_tensor_v2(*_args):  # never called; exists to be pickled by name
        raise AssertionError("fixture rebuild function must not execute")

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2

    torch_module = types.ModuleType("torch")

    class FloatStorage:
        pass

    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"
    torch_module.FloatStorage = FloatStorage
    torch_module._utils = utils
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch._utils", utils)


class _StoragePickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _FakeStorage):
            storage_type = sys.modules["torch"].FloatStorage
            return ("storage", storage_type, obj.key, "cpu", obj.numel)
        return None


def _write_zip_checkpoint(
    path: Path, tensors: dict[str, tuple[tuple[int, ...], list[float]]]
) -> None:
    """Write a torch>=1.6-style zip checkpoint of float32 tensors."""
    state = OrderedDict()
    storages: dict[str, bytes] = {}
    for index, (key, (shape, values)) in enumerate(tensors.items()):
        storage_key = str(index)
        storages[storage_key] = array.array("f", values).tobytes()
        state[key] = _FakeTensor(_FakeStorage(storage_key, len(values)), shape)
    buffer = io.BytesIO()
    _StoragePickler(buffer, protocol=2).dump(state)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", buffer.getvalue())
        archive.writestr("archive/version", "3\n")
        for storage_key, blob in storages.items():
            archive.writestr(f"archive/data/{storage_key}", blob)


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("true",))


def _evil_zip_checkpoint(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"weight": _Evil()}, protocol=2))


class TestConvertStaticScan:
    """Step 1's scan targets the bytes that actually get unpickled: the
    data.pkl member(s) of a zip checkpoint (torch >= 1.6), or the raw
    stream of a legacy checkpoint."""

    def _scan(self):
        from kinovsr.modeling.pickle_scan import scan_checkpoint_globals

        return scan_checkpoint_globals

    def _suspicious(self):
        from kinovsr.modeling.pickle_scan import suspicious_globals

        return suspicious_globals

    def test_zip_checkpoint_with_os_system_is_flagged(self, tmp_path):
        source = tmp_path / "evil.pth"
        _evil_zip_checkpoint(source)
        refs = self._scan()(source)
        assert refs, "scan must read the data.pkl member, not the zip container bytes"
        assert self._suspicious()(refs)

    def test_legacy_stream_with_os_system_is_flagged(self, tmp_path):
        source = tmp_path / "evil_legacy.pth"
        source.write_bytes(pickle.dumps({"weight": _Evil()}, protocol=2))
        assert self._suspicious()(self._scan()(source))

    def test_benign_zip_checkpoint_scans_clean(self, tmp_path, monkeypatch):
        _install_fake_torch(monkeypatch)
        source = tmp_path / "benign.pth"
        _write_zip_checkpoint(source, {"conv.weight": ((2, 3), [0.5] * 6)})
        refs = self._scan()(source)
        assert "torch._utils _rebuild_tensor_v2" in refs
        assert self._suspicious()(refs) == []

    def test_convert_blocks_malicious_zip_and_force_fails_closed(self, tmp_path):
        from kinovsr.cli.commands.weights_convert import run_convert

        source = tmp_path / "evil.pth"
        _evil_zip_checkpoint(source)
        out = tmp_path / "out.safetensors"
        assert run_convert([str(source), "-o", str(out)]) == 2
        # --force skips only the scan; the restricted unpickler still refuses.
        assert run_convert([str(source), "-o", str(out), "--force"]) == 1
        assert not out.exists()

    def test_convert_round_trips_a_benign_zip_checkpoint(self, tmp_path, monkeypatch):
        import mlx.core as mx

        from kinovsr.cli.commands.weights_convert import run_convert

        _install_fake_torch(monkeypatch)
        source = tmp_path / "model.pth"
        values = [float(i) for i in range(6)]
        _write_zip_checkpoint(source, {"module.conv.weight": ((2, 3), values)})
        out = tmp_path / "model.safetensors"
        assert run_convert([str(source), "-o", str(out)]) == 0
        loaded = dict(mx.load(str(out)))
        assert list(loaded) == ["conv.weight"]
        assert loaded["conv.weight"].tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


class TestConvertOutputPath:
    """MLX appends .safetensors to any other output name. The converter then
    reloaded the name it was given: an unknown-format error for weights.bin,
    or a stale file's arrays logged as the verified result."""

    @pytest.mark.parametrize("name", ["weights.bin", "weights", "weights.SAFETENSORS"])
    def test_non_safetensors_output_is_refused_before_any_work(self, tmp_path, monkeypatch, name):
        from kinovsr.cli.commands.weights_convert import run_convert

        _install_fake_torch(monkeypatch)
        source = tmp_path / "model.pth"
        _write_zip_checkpoint(source, {"conv.weight": ((2, 3), [0.5] * 6)})
        assert run_convert([str(source), "-o", str(tmp_path / name)]) == 2
        assert sorted(path.name for path in tmp_path.iterdir()) == ["model.pth"]

    def test_a_stale_output_is_left_alone(self, tmp_path, monkeypatch):
        import mlx.core as mx

        from kinovsr.cli.commands.weights_convert import run_convert

        _install_fake_torch(monkeypatch)
        source = tmp_path / "model.pth"
        _write_zip_checkpoint(source, {"conv.weight": ((2, 3), [0.5] * 6)})
        stale = tmp_path / "old.npz"
        mx.savez(str(stale), earlier=mx.zeros((1,)))
        assert run_convert([str(source), "-o", str(stale)]) == 2
        assert list(dict(mx.load(str(stale)))) == ["earlier"]
        assert not (tmp_path / "old.npz.safetensors").exists()
