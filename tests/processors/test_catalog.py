"""Catalog behavior: lazy import, duplicate rejection, conformance."""

import sys
import types
from pathlib import Path

import pytest

from kinovsr.processors import catalog
from kinovsr.processors.capabilities import (
    Capability,
    CapabilitySpec,
    preserve_stream,
)
from kinovsr.processors.errors import PipelineError
from kinovsr.processors.specs import StreamConstraint

pytestmark = pytest.mark.unit


class FakeFactory:
    def __init__(self, name):
        self.name = name
        self.capabilities = {
            Capability.DENOISE: CapabilitySpec(
                capability=Capability.DENOISE,
                profiles=("default",),
                accepts=StreamConstraint(),
                produces=preserve_stream,
            ),
        }

    def parse_config(self, raw, *, capability, profile, settings):
        return dict(raw)

    def build(self, config, *, context):
        raise NotImplementedError


@pytest.fixture
def clean_catalog(monkeypatch):
    monkeypatch.setattr(catalog, "_CATALOG", {})
    monkeypatch.setattr(catalog, "_loaded", {})
    return catalog


def install_fake_module(monkeypatch, module_name, **attrs):
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def test_register_and_lazy_get(clean_catalog, monkeypatch):
    install_fake_module(monkeypatch, "fake_family_mod",
                        FACTORY=FakeFactory("fakefam"))
    clean_catalog.register("fakefam", "fake_family_mod:FACTORY")
    assert clean_catalog.available_families() == ("fakefam",)
    factory = clean_catalog.get_factory("fakefam")
    assert factory.name == "fakefam"
    # cached: same object back
    assert clean_catalog.get_factory("fakefam") is factory


def test_registration_does_not_import(clean_catalog):
    clean_catalog.register("neverloaded", "kinovsr_no_such_module:F")
    # no import happened at registration; only get_factory would fail
    assert "kinovsr_no_such_module" not in sys.modules


def test_duplicate_name_rejected(clean_catalog):
    clean_catalog.register("dup", "a_mod:F")
    with pytest.raises(ValueError, match="registered twice"):
        clean_catalog.register("dup", "b_mod:F")


def test_malformed_target_rejected(clean_catalog):
    with pytest.raises(ValueError, match="module:attribute"):
        clean_catalog.register("bad", "no_colon_here")


def test_unknown_family_suggests_and_lists(clean_catalog, monkeypatch):
    install_fake_module(monkeypatch, "fake_family_mod",
                        FACTORY=FakeFactory("fastdvdnet"))
    clean_catalog.register("fastdvdnet", "fake_family_mod:FACTORY")
    with pytest.raises(catalog.UnknownFamilyError) as exc:
        clean_catalog.get_factory("fastdvdnt")
    message = str(exc.value)
    assert "did you mean 'fastdvdnet'" in message
    assert "available: fastdvdnet" in message


def test_name_mismatch_is_a_conformance_error(clean_catalog, monkeypatch):
    install_fake_module(monkeypatch, "fake_family_mod",
                        FACTORY=FakeFactory("other_name"))
    clean_catalog.register("fakefam", "fake_family_mod:FACTORY")
    with pytest.raises(PipelineError, match="does not match factory name"):
        clean_catalog.get_factory("fakefam")


def test_missing_attribute_is_reported(clean_catalog, monkeypatch):
    install_fake_module(monkeypatch, "fake_family_mod")
    clean_catalog.register("fakefam", "fake_family_mod:MISSING")
    with pytest.raises(PipelineError, match="does not exist"):
        clean_catalog.get_factory("fakefam")


def test_empty_capabilities_rejected(clean_catalog, monkeypatch):
    factory = FakeFactory("fakefam")
    factory.capabilities = {}
    install_fake_module(monkeypatch, "fake_family_mod", FACTORY=factory)
    clean_catalog.register("fakefam", "fake_family_mod:FACTORY")
    with pytest.raises(PipelineError, match="declares no capabilities"):
        clean_catalog.get_factory("fakefam")


def test_real_catalog_targets_stay_inside_kinovsr():
    for entry in catalog.catalog_entries():
        module_name = entry.factory_target.partition(":")[0]
        assert module_name.startswith("kinovsr."), (
            entry.name, entry.factory_target)


def test_cli_contributions_are_unique_and_family_local():
    processors_dir = Path(catalog.__file__).parent
    claimed_orders: dict[int, str] = {}
    for entry in catalog.catalog_entries():
        assert entry.cli_options, f"{entry.name} has no family-local CLI contribution"
        path = entry.cli_options_path
        assert path is not None
        assert (processors_dir / path).is_file(), (entry.name, path)
        for contribution in entry.cli_options:
            assert contribution.order not in claimed_orders, (
                contribution.order, entry.name,
                claimed_orders.get(contribution.order))
            claimed_orders[contribution.order] = entry.name
