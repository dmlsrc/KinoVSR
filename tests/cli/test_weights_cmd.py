"""The kinovsr weights subcommand: listing, verification, exit codes."""

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

    plksr_profiles = realplksr_factory.capabilities[
        Capability.UPSCALE].profiles
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
        assert resolved is not None and resolved.read_bytes() == b"src"

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
