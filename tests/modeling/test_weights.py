"""Manifest schema validation, registry loading, and verification."""

import pytest

from kinovsr.modeling.weights import (
    ManifestError,
    load_manifest,
    load_registered,
    registered_owners,
    verify_manifest,
)

pytestmark = pytest.mark.unit


def write_manifest(tmp_path, body):
    path = tmp_path / "manifest.toml"
    path.write_text(body)
    return path


GOOD = """\
schema_version = 1

[owner]
kind = "processor"
name = "fakefam"

[profiles.default]
capabilities = ["denoise"]
weights = ["main"]

[profiles.default.defaults]
strength = 0.25

[weights.main]
path = "weights/main.safetensors"
distribution = "external"
source = "https://example.invalid/main.pth"
source_sha256 = "aa"
license = "MIT"
"""


class TestSchema:
    def test_good_manifest_loads(self, tmp_path):
        manifest = load_manifest(write_manifest(tmp_path, GOOD))
        assert manifest.kind == "processor"
        assert manifest.name == "fakefam"
        assert manifest.profiles["default"].defaults == {"strength": 0.25}
        asset = manifest.weights["main"]
        assert asset.path == tmp_path / "weights/main.safetensors"
        assert asset.license == "MIT"

    def test_schema_version_required(self, tmp_path):
        with pytest.raises(ManifestError, match="schema_version"):
            load_manifest(write_manifest(
                tmp_path, GOOD.replace("schema_version = 1",
                                       "schema_version = 2")))

    def test_owner_kind_validated(self, tmp_path):
        with pytest.raises(ManifestError, match="owner.kind"):
            load_manifest(write_manifest(
                tmp_path, GOOD.replace('kind = "processor"',
                                       'kind = "plugin"')))

    def test_bundled_requires_artifact_hash(self, tmp_path):
        body = GOOD.replace('distribution = "external"',
                            'distribution = "bundled"')
        with pytest.raises(ManifestError, match="artifact_sha256"):
            load_manifest(write_manifest(tmp_path, body))

    def test_external_requires_source(self, tmp_path):
        body = GOOD.replace(
            'source = "https://example.invalid/main.pth"\n', "")
        with pytest.raises(ManifestError, match="source"):
            load_manifest(write_manifest(tmp_path, body))

    def test_profile_referencing_unknown_weights(self, tmp_path):
        body = GOOD.replace('weights = ["main"]', 'weights = ["nope"]')
        with pytest.raises(ManifestError, match="unknown weights id 'nope'"):
            load_manifest(write_manifest(tmp_path, body))

    def test_component_manifests_carry_no_profiles(self, tmp_path):
        body = GOOD.replace('kind = "processor"', 'kind = "component"')
        with pytest.raises(ManifestError, match="not profiles"):
            load_manifest(write_manifest(tmp_path, body))


class TestRegistry:
    def test_registered_owners(self):
        assert registered_owners() == ("bsvd", "realplksr", "spynet")

    @pytest.mark.parametrize("owner", ["bsvd", "realplksr", "spynet"])
    def test_every_registered_manifest_loads_and_matches(self, owner):
        manifest = load_registered(owner)
        assert manifest.name == owner

    def test_unknown_owner(self):
        with pytest.raises(ManifestError, match="no manifest registered"):
            load_registered("basicvsrpp")


class TestVerify:
    def test_bundled_spynet_verifies_on_this_checkout(self):
        reports = verify_manifest(load_registered("spynet"))
        assert len(reports) == 1
        report = reports[0]
        assert report.distribution == "bundled"
        assert report.present
        assert report.hash_checked and report.hash_ok

    def test_missing_external_reports_source(self, tmp_path):
        manifest = load_manifest(write_manifest(tmp_path, GOOD))
        report = verify_manifest(manifest)[0]
        assert not report.present
        assert "download from https://example.invalid/main.pth" in report.note
        assert report.hash_ok is None

    def test_artifact_hash_mismatch_detected(self, tmp_path):
        body = GOOD.replace(
            'distribution = "external"',
            'distribution = "bundled"').replace(
            'source_sha256 = "aa"',
            f'artifact_sha256 = "{"0" * 64}"')
        manifest = load_manifest(write_manifest(tmp_path, body))
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        (weights_dir / "main.safetensors").write_bytes(b"junk")
        report = verify_manifest(manifest)[0]
        assert report.present and report.hash_checked
        assert report.hash_ok is False
        assert "MISMATCH" in report.note

    def test_present_without_artifact_hash_states_the_chain(self, tmp_path):
        manifest = load_manifest(write_manifest(
            tmp_path, GOOD.replace('source_sha256 = "aa"',
                                   f'source_sha256 = "{"a" * 64}"')))
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        (weights_dir / "main.safetensors").write_bytes(b"data")
        report = verify_manifest(manifest)[0]
        assert report.present and not report.hash_checked
        assert "artifact hash not recorded" in report.note
        assert "source sha256" in report.note
