"""Settings: env resolution, precedence, CLI bridge, and the default accessor."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pytest

from kinovsr import settings as settings_mod
from kinovsr.settings import (
    Settings,
    _reset_default_settings,
    add_argparse_args,
    default_settings,
    settings_from_args,
)

ALL_ENV_VARS = [
    "KINOVSR_VERBOSE", "KINOVSR_QUIET",
    "SHARED_TEMP_DIR", "TMPDIR", "KINOVSR_MLX_CACHE_LIMIT_GB",
    "KINOVSR_CACHE_DIR", "XDG_CACHE_HOME",
    "KINOVSR_SPYNET_BACKEND", "SPYNET_BACKEND",
    "KINOVSR_BSVD_BACKEND", "BSVD_BACKEND", "KINOVSR_BSVD_DIRECT",
    "BASICVSRPP_WEIGHTS", "BASICVSRPP_RESTORE_WEIGHTS", "BSVD_WEIGHTS",
    "ESC_WEIGHTS", "FASTDVD_WEIGHTS", "FBCNN_WEIGHTS", "NAFNET_WEIGHTS",
    "PVDD_WEIGHTS", "REALBASICVSR_WEIGHTS", "REALESRGAN_WEIGHTS",
    "REALPLKSR_WEIGHTS", "REALVIFORMER_WEIGHTS", "SAFMN_WEIGHTS",
    "SPYNET_WEIGHTS", "STDF_WEIGHTS", "TOFLOW_WEIGHTS", "TOFLOW_GRAPH",
    "TOFLOW_SR_WEIGHTS", "TOFLOW_SR_GRAPH",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts with none of the settings env vars set and no cache."""
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    _reset_default_settings()
    yield
    _reset_default_settings()


# ---- construction and env reads ------------------------------------------


def test_bare_settings_reads_no_environment(monkeypatch):
    monkeypatch.setenv("NAFNET_WEIGHTS", "gopro32")
    monkeypatch.setenv("KINOVSR_VERBOSE", "1")
    s = Settings()
    assert s.nafnet_weights is None
    assert s.verbose is False


def test_from_env_reads_declared_variables(monkeypatch):
    monkeypatch.setenv("NAFNET_WEIGHTS", "gopro32")
    monkeypatch.setenv("TOFLOW_GRAPH", "/some/graph.json")
    monkeypatch.setenv("KINOVSR_BSVD_DIRECT", "require")
    s = Settings.from_env()
    assert s.nafnet_weights == "gopro32"
    assert s.toflow_graph == "/some/graph.json"
    assert s.bsvd_direct == "require"
    assert s.bsvd_weights is None


def test_prefixed_backend_env_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("SPYNET_BACKEND", "mlx")
    monkeypatch.setenv("KINOVSR_SPYNET_BACKEND", "ane")
    monkeypatch.setenv("BSVD_BACKEND", "ane")
    monkeypatch.setenv("KINOVSR_BSVD_BACKEND", "mpsgraph")
    settings = Settings.from_env()
    assert settings.spynet_backend == "ane"
    assert settings.bsvd_backend == "mpsgraph"


def test_cache_dir_fallback_is_isolated_and_ordered(monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert Settings.from_env().cache_dir == "/xdg/KinoVSR"
    monkeypatch.setenv("KINOVSR_CACHE_DIR", "/kinovsr")
    assert Settings.from_env().cache_dir == "/kinovsr"


def test_weight_fields_stay_strings_for_variant_tokens(monkeypatch):
    monkeypatch.setenv("REALESRGAN_WEIGHTS", "x4plus")
    s = Settings.from_env()
    assert isinstance(s.realesrgan_weights, str)
    assert s.realesrgan_weights == "x4plus"


def test_every_declared_env_var_is_consumed(monkeypatch):
    weight_vars = [v for v in ALL_ENV_VARS if v.endswith(("_WEIGHTS", "_GRAPH"))]
    for name in weight_vars:
        monkeypatch.setenv(name, f"token-{name.lower()}")
    s = Settings.from_env()
    values = {getattr(s, f.name) for f in dataclasses.fields(Settings)
              if f.name.endswith(("_weights", "_graph"))}
    assert values == {f"token-{name.lower()}" for name in weight_vars}


def test_fallback_chain_order(monkeypatch):
    monkeypatch.setenv("TMPDIR", "/from/tmpdir")
    s = Settings.from_env()
    assert s.shared_temp_dir == Path("/from/tmpdir")
    monkeypatch.setenv("SHARED_TEMP_DIR", "/durable/scratch")
    s = Settings.from_env()
    assert s.shared_temp_dir == Path("/durable/scratch")


def test_fallback_chain_literal_terminal():
    s = Settings.from_env()
    assert s.shared_temp_dir == Path("/tmp")


def test_unset_template_falls_through_to_default():
    s = Settings.from_env()
    assert s.mlx_cache_limit_gb is None
    assert s.nafnet_weights is None


def test_float_and_path_parsing(monkeypatch, tmp_path):
    monkeypatch.setenv("KINOVSR_MLX_CACHE_LIMIT_GB", "1.5")
    monkeypatch.setenv("SHARED_TEMP_DIR", str(tmp_path))
    s = Settings.from_env()
    assert s.mlx_cache_limit_gb == 1.5
    assert s.shared_temp_dir == tmp_path


# ---- boolean semantics (truthy-string, preserves KINOVSR_VERBOSE) ---------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "anything", "2"])
def test_bool_truthy_values(monkeypatch, raw):
    monkeypatch.setenv("KINOVSR_VERBOSE", raw)
    assert Settings.from_env().verbose is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", "  0  "])
def test_bool_falsy_values(monkeypatch, raw):
    monkeypatch.setenv("KINOVSR_VERBOSE", raw)
    assert Settings.from_env().verbose is False


def test_bool_empty_string_is_unresolved_not_true(monkeypatch):
    monkeypatch.setenv("KINOVSR_QUIET", "")
    assert Settings.from_env().quiet is False


# ---- immutability and overrides --------------------------------------------


def test_settings_is_frozen():
    s = Settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.verbose = True


def test_with_overrides_skips_none():
    s = Settings(nafnet_weights="gopro")
    out = s.with_overrides(nafnet_weights=None, verbose=True)
    assert out.nafnet_weights == "gopro"
    assert out.verbose is True
    assert s.verbose is False  # original untouched


# ---- CLI bridge -------------------------------------------------------------


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_argparse_args(parser)
    return parser.parse_args(argv)


def test_cli_flags_override_env(monkeypatch):
    monkeypatch.setenv("NAFNET_WEIGHTS", "gopro")
    args = _parse(["--nafnet-weights", "sidd32"])
    s = settings_from_args(args, Settings.from_env())
    assert s.nafnet_weights == "sidd32"


def test_bsvd_direct_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("KINOVSR_BSVD_DIRECT", "off")
    args = _parse(["--bsvd-direct", "force"])
    s = settings_from_args(args, Settings.from_env())
    assert s.bsvd_direct == "force"


def test_cli_unset_flags_keep_base(monkeypatch):
    monkeypatch.setenv("TOFLOW_WEIGHTS", "denoise")
    args = _parse([])
    s = settings_from_args(args, Settings.from_env())
    assert s.toflow_weights == "denoise"


def test_cli_bool_pair(monkeypatch):
    monkeypatch.setenv("KINOVSR_VERBOSE", "1")
    args = _parse(["--no-verbose"])
    s = settings_from_args(args, Settings.from_env())
    assert s.verbose is False
    args = _parse(["--quiet"])
    s = settings_from_args(args, Settings.from_env())
    assert s.quiet is True


def test_cli_values_are_verbatim_no_template_expansion(monkeypatch):
    monkeypatch.setenv("SHARED_TEMP_DIR", "/real/scratch")
    args = _parse(["--nafnet-weights", "{{SHARED_TEMP_DIR}}/x.safetensors"])
    s = settings_from_args(args, Settings.from_env())
    assert s.nafnet_weights == "{{SHARED_TEMP_DIR}}/x.safetensors"


def test_every_field_gets_a_flag():
    parser = argparse.ArgumentParser()
    add_argparse_args(parser)
    flags = {a for action in parser._actions for a in action.option_strings}
    for f in dataclasses.fields(Settings):
        assert "--" + f.name.replace("_", "-") in flags


# ---- process-default accessor ----------------------------------------------


def test_default_settings_is_cached(monkeypatch):
    monkeypatch.setenv("NAFNET_WEIGHTS", "gopro")
    first = default_settings()
    assert first.nafnet_weights == "gopro"
    monkeypatch.setenv("NAFNET_WEIGHTS", "sidd")
    assert default_settings() is first
    assert default_settings().nafnet_weights == "gopro"
    _reset_default_settings()
    assert default_settings().nafnet_weights == "sidd"


def test_settings_is_the_only_env_reader_in_the_package():
    """M2 acceptance: runtime code reads no environment outside Settings.from_env().

    Scans every module in the kinovsr package for os.environ/getenv references;
    only settings.py may contain them.
    """
    pkg_root = Path(settings_mod.__file__).parent
    offenders = []
    for py in sorted(pkg_root.rglob("*.py")):
        if py.name == "settings.py":
            continue
        text = py.read_text()
        if "os.environ" in text or "os.getenv" in text or "getenv(" in text:
            offenders.append(str(py.relative_to(pkg_root)))
    assert offenders == []
