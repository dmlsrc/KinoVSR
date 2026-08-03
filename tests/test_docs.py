"""Product documentation hygiene.

Product docs must stand alone: no machine-local paths, no private
planning-repo references, no retired commands, plain ASCII, and the
generated processor matrix in sync with the manifests it derives from.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]

_FORBIDDEN = (
    ("machine-local path", re.compile(r"/Users/|/private/tmp|/tmp/")),
    ("private planning reference", re.compile(
        r"KinoVSR-planning|KinoMLX-planning")),
    ("retired harness command", re.compile(r"vsr_harness")),
)


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_docs_are_standalone_and_ascii(doc: Path):
    text = doc.read_text(encoding="utf-8")
    for label, pattern in _FORBIDDEN:
        hits = [line for line in text.splitlines() if pattern.search(line)]
        assert not hits, f"{doc.name}: {label}: {hits[:3]}"
    non_ascii = {char for char in text if ord(char) > 127}
    assert not non_ascii, f"{doc.name}: non-ASCII characters {non_ascii!r}"


def test_every_weights_directory_documents_and_attributes():
    missing = []
    for weights_dir in sorted(REPO.glob("kinovsr/**/weights")):
        if not weights_dir.is_dir():
            continue
        for required in ("README.md", "Attribution.md"):
            if not (weights_dir / required).is_file():
                missing.append(str((weights_dir / required).relative_to(REPO)))
    assert not missing, missing


def test_processor_matrix_matches_manifests(tmp_path, monkeypatch):
    """docs/PROCESSORS.md is generated; regenerating must be a no-op."""
    spec = importlib.util.spec_from_file_location(
        "gen_processor_matrix",
        REPO / "scripts" / "dev" / "gen_processor_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = module.OUT.read_text(encoding="utf-8")
    monkeypatch.setattr(module, "OUT", tmp_path / "PROCESSORS.md")
    module.main()
    regenerated = module.OUT.read_text(encoding="utf-8")
    assert regenerated == committed, (
        "docs/PROCESSORS.md is stale; regenerate with "
        "scripts/dev/gen_processor_matrix.py")
