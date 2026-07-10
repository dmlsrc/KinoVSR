"""Weight manifests: parsing, resolution, and verification.

Every weights owner - processor family, shared component, evaluator -
carries a package-data ``manifest.toml`` (planning schema, version 1).
This module loads and validates that schema and implements the two
read-only operations the CLI exposes:

- ``list``: profiles and weight assets with bundled/external status;
- ``verify``: presence always; ``artifact_sha256`` compared whenever the
  manifest records one; otherwise the report states the provenance chain
  it cannot verify rather than implying a hash check that did not run.

Nothing here downloads. The registry of first-party manifests is
explicit, like the processor catalog.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_KINDS = ("processor", "component", "evaluator")

# Explicit first-party manifest registry: owner name -> manifest path
# relative to the kinovsr package root. Extended as families migrate.
MANIFESTS = {
    "bsvd": "bsvd/manifest.toml",
    "realplksr": "realplksr/manifest.toml",
    "spynet": "spynet/manifest.toml",
}

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class ManifestError(Exception):
    """A manifest file violates the schema; message names file and field."""


@dataclass(frozen=True, slots=True)
class WeightAsset:
    asset_id: str
    path: Path                    # absolute, anchored at the manifest dir
    distribution: str             # "bundled" | "external"
    source: str | None = None
    source_sha256: str | None = None
    artifact_sha256: str | None = None
    license: str | None = None


@dataclass(frozen=True, slots=True)
class ManifestProfile:
    name: str
    capabilities: tuple[str, ...]
    weights: tuple[str, ...]      # asset ids
    components: tuple[str, ...] = ()
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Manifest:
    kind: str
    name: str
    root: Path
    profiles: dict[str, ManifestProfile]
    weights: dict[str, WeightAsset]


@dataclass(frozen=True, slots=True)
class VerifyReport:
    owner: str
    asset_id: str
    path: Path
    distribution: str
    present: bool
    hash_checked: bool
    hash_ok: bool | None          # None when no artifact hash is recorded
    note: str


def _require(condition: bool, where: str, message: str) -> None:
    if not condition:
        raise ManifestError(f"{where}: {message}")


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    where = str(path)
    try:
        data = tomllib.loads(path.read_text())
    except OSError as exc:
        raise ManifestError(f"{where}: cannot read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{where}: invalid TOML: {exc}") from exc

    _require(data.get("schema_version") == 1, where,
             "schema_version must be 1")
    owner = data.get("owner")
    _require(isinstance(owner, dict), where, "[owner] table is required")
    kind, name = owner.get("kind"), owner.get("name")
    _require(kind in _KINDS, where,
             f"owner.kind must be one of {list(_KINDS)}")
    _require(isinstance(name, str) and bool(name), where,
             "owner.name must be a non-empty string")

    weights: dict[str, WeightAsset] = {}
    for asset_id, table in (data.get("weights") or {}).items():
        aw = f"{where}: [weights.{asset_id}]"
        _require(isinstance(table, dict), aw, "must be a table")
        rel = table.get("path")
        _require(isinstance(rel, str) and bool(rel), aw,
                 "path is required")
        distribution = table.get("distribution")
        _require(distribution in ("bundled", "external"), aw,
                 "distribution must be 'bundled' or 'external'")
        artifact = table.get("artifact_sha256")
        if distribution == "bundled":
            _require(isinstance(artifact, str) and len(artifact) == 64, aw,
                     "bundled weights must record artifact_sha256")
        if distribution == "external":
            _require(isinstance(table.get("source"), str), aw,
                     "external weights must record their source")
        weights[asset_id] = WeightAsset(
            asset_id=asset_id,
            path=(path.parent / rel).resolve(),
            distribution=distribution,
            source=table.get("source"),
            source_sha256=table.get("source_sha256"),
            artifact_sha256=artifact,
            license=table.get("license"),
        )

    profiles: dict[str, ManifestProfile] = {}
    for profile_name, table in (data.get("profiles") or {}).items():
        pw = f"{where}: [profiles.{profile_name}]"
        _require(isinstance(table, dict), pw, "must be a table")
        asset_ids = tuple(table.get("weights") or ())
        for asset_id in asset_ids:
            _require(asset_id in weights, pw,
                     f"references unknown weights id {asset_id!r}")
        profiles[profile_name] = ManifestProfile(
            name=profile_name,
            capabilities=tuple(table.get("capabilities") or ()),
            weights=asset_ids,
            components=tuple(table.get("components") or ()),
            defaults=dict(table.get("defaults") or {}),
        )
    if kind == "component":
        _require(not profiles, where,
                 "component manifests carry weights, not profiles")

    return Manifest(kind=kind, name=name, root=path.parent,
                    profiles=profiles, weights=weights)


def load_registered(owner: str) -> Manifest:
    rel = MANIFESTS.get(owner)
    if rel is None:
        known = ", ".join(sorted(MANIFESTS))
        raise ManifestError(
            f"no manifest registered for {owner!r} (registered: {known})")
    manifest = load_manifest(_PACKAGE_ROOT / rel)
    _require(manifest.name == owner, rel,
             f"owner.name {manifest.name!r} does not match registry key")
    return manifest


def registered_owners() -> tuple[str, ...]:
    return tuple(sorted(MANIFESTS))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest: Manifest) -> list[VerifyReport]:
    reports: list[VerifyReport] = []
    for asset in manifest.weights.values():
        present = asset.path.is_file()
        hash_checked = False
        hash_ok: bool | None = None
        if not present:
            note = ("missing (bundled - the package is incomplete)"
                    if asset.distribution == "bundled"
                    else f"not installed; download from {asset.source}")
        elif asset.artifact_sha256:
            hash_checked = True
            hash_ok = _sha256(asset.path) == asset.artifact_sha256
            note = ("artifact sha256 verified" if hash_ok
                    else "ARTIFACT HASH MISMATCH")
        else:
            chain = (f"source sha256 {asset.source_sha256[:12]}... recorded "
                     f"for the upstream file"
                     if asset.source_sha256 else "no hashes recorded")
            note = f"artifact hash not recorded ({chain})"
        reports.append(VerifyReport(
            owner=manifest.name, asset_id=asset.asset_id, path=asset.path,
            distribution=asset.distribution, present=present,
            hash_checked=hash_checked, hash_ok=hash_ok, note=note))
    return reports


__all__ = [
    "MANIFESTS",
    "Manifest",
    "ManifestError",
    "ManifestProfile",
    "VerifyReport",
    "WeightAsset",
    "load_manifest",
    "load_registered",
    "registered_owners",
    "verify_manifest",
]
