"""Structural conformance of the option registry to the vocabulary.

Planning 07-cli-vocabulary.md: the same key name means the same concept in
every family, canonical flags are the composition of family and key, legacy
spellings are hidden aliases, and settings-backed flags stay None-default so
the env/TOML trifecta layers show through.
"""

import subprocess
import sys
from dataclasses import fields

import pytest

from kinovsr.cli._registry import REGISTRY
from kinovsr.cli.options import FAMILY_KEYS, SHARED_KEYS, vocabulary_violations
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


def test_registry_has_no_vocabulary_violations():
    assert vocabulary_violations(REGISTRY) == []


def test_registry_loading_keeps_family_implementations_lazy():
    code = """
import sys
import kinovsr.cli._registry

families = {
    "basicvsrpp", "bsvd", "crop", "cut_detect", "deflicker", "esc",
    "fastdvdnet", "fbcnn", "mc", "metalfx", "nafnet", "pvdd",
    "realbasicvsr", "realesrgan", "realplksr", "realviformer", "safmn",
    "sanitize_edges", "spatial", "square_pixels", "stdf", "toflow",
}
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


def test_aliases_differ_and_map_to_canonical_dests():
    canonical_flags = {o.flag for o in REGISTRY}
    for opt in REGISTRY:
        for alias in opt.aliases:
            assert alias != opt.flag
            assert alias not in canonical_flags, (
                f"{alias} is both an alias and a canonical flag")


def test_every_expected_legacy_spelling_is_carried():
    """The renames table from planning 07 - each legacy flag parses."""
    alias_map = {a: o.flag for o in REGISTRY for a in o.aliases}
    expected = {
        "--spatial-mode": "--upscale",
        "--restore-flow-mode": "--restore-flow",
        "--basicvsrpp-flow-mode": "--basicvsrpp-flow",
        "--realbasicvsr-flow-mode": "--realbasicvsr-flow",
        "--realviformer-flow-mode": "--realviformer-flow",
        "--basicvsrpp-variant": "--basicvsrpp-profile",
        "--bsvd-variant": "--bsvd-profile",
        "--toflow-variant": "--toflow-profile",
        "--pvdd-variant": "--pvdd-profile",
        "--fastdvd-variant": "--fastdvdnet-profile",
        "--fastdvd-weights": "--fastdvdnet-weights",
        "--realesrgan-denoise": "--realesrgan-denoise-strength",
        "--realbasicvsr-dynamic-refine-thres": "--realbasicvsr-clean-threshold",
    }
    for legacy, canonical in expected.items():
        assert alias_map.get(legacy) == canonical


def test_shared_and_family_keys_do_not_overlap():
    for family, keys in FAMILY_KEYS.items():
        overlap = keys & SHARED_KEYS
        assert not overlap, f"{family}: {overlap} shadow shared keys"


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
