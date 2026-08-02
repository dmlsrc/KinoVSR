"""Structural conformance of the option registry to the vocabulary.

Planning 07-cli-vocabulary.md: the same key name means the same concept in
every family, canonical flags are the composition of family and key, and
settings-backed flags stay None-default so the env/TOML trifecta layers show
through.
"""

import subprocess
import sys
from dataclasses import fields

import pytest

from kinovsr.cli._registry import REGISTRY
from kinovsr.cli.args import build_parser
from kinovsr.cli.options import (
    FAMILY_KEYS,
    SHARED_KEYS,
    option_roles,
    vocabulary_violations,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


def test_registry_has_no_vocabulary_violations():
    assert vocabulary_violations(REGISTRY) == []


def test_registry_loading_keeps_family_implementations_lazy():
    code = """
import sys
import kinovsr.cli._registry
from kinovsr.processors.catalog import catalog_entries

families = {entry.name for entry in catalog_entries()}
loaded = sorted(
    name for name in sys.modules
    if name.startswith("kinovsr.processors.")
    and name.split(".", 3)[2] in families
)
if loaded:
    raise SystemExit("family implementations loaded by CLI registry: " + repr(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=False)
    assert result.returncode == 0, result.stderr


def test_family_flags_compose_family_and_key():
    for opt in REGISTRY:
        if opt.family is None or opt.key is None:
            continue
        expected = "--" + f"{opt.family}_{opt.key}".replace("_", "-")
        assert opt.flag == expected, (
            f"{opt.flag}: family={opt.family!r} key={opt.key!r} should "
            f"spell {expected}")


def test_settings_backed_dests_are_settings_fields():
    names = {f.name for f in fields(Settings)}
    for opt in REGISTRY:
        if opt.settings_backed:
            assert opt.resolved_dest in names, opt.flag


def test_registry_flags_are_unique():
    flags = [opt.flag for opt in REGISTRY]
    assert len(flags) == len(set(flags))


def test_compositional_flags_are_registry_owned_and_parser_complete():
    expected = {
        "--upscale", "--denoise", "--deblock", "--restore", "--nafnet",
        "--deflicker", "--cut-detect", "--level", "--crop-bars",
        "--crop-aspect", "--square-pixels", "--sanitize-edges",
        "--target-fps", "--conform-cfr", "--preprocess-order",
        "--denoise-first",
    }
    owned = {opt.flag for opt in REGISTRY if opt.compositional}
    parsed = {
        flag
        for action in build_parser()._actions
        for flag in action.option_strings
    }
    assert owned == expected
    assert owned <= parsed


def test_foundation_tables_cover_the_27_run_options_once():
    rows = [opt for opt in REGISTRY if opt.config_table is not None]
    assert len(rows) == 27
    assert {opt.config_table for opt in rows} == {
        "input", "output", "diagnostics"}
    dests = [opt.resolved_dest for opt in rows]
    assert len(dests) == len(set(dests))


def test_every_registry_row_has_an_explicit_config_role():
    assert all(option_roles(opt) for opt in REGISTRY)
    assert set().union(*(option_roles(opt) for opt in REGISTRY)) == {
        "composition", "dial", "run", "settings"}


def test_shared_and_family_keys_do_not_overlap():
    for family, keys in FAMILY_KEYS.items():
        overlap = keys & SHARED_KEYS
        assert not overlap, f"{family}: {overlap} shadow shared keys"


def test_every_family_specific_key_is_used():
    for family, keys in FAMILY_KEYS.items():
        used = {opt.key for opt in REGISTRY if opt.family == family}
        unused = keys - used
        assert not unused, f"{family}: {sorted(unused)}"


def test_every_family_with_weights_has_a_profile_or_selector():
    """Weights-capable learned families expose the profile selector too
    (nafnet and restore take the profile as the slot value itself)."""
    by_family: dict[str, set[str]] = {}
    for opt in REGISTRY:
        if opt.family and opt.key:
            by_family.setdefault(opt.family, set()).add(opt.key)
    # nafnet/restore take the profile as the slot value; toflow_sr has one
    # checkpoint; fbcnn has one checkpoint (its QF dial is conditioning,
    # not a preset); mc's weights key is flow_weights; slots are chains.
    exempt = {"nafnet", "restore", "toflow_sr", "fbcnn",
              "mc", "deblock", "denoise"}
    for family, keys in by_family.items():
        if "weights" in keys and family not in exempt:
            assert "profile" in keys, f"{family} has weights but no profile"
