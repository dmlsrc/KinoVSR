"""Static contract: package and developer tools emit through logging."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
ROOTS = (REPO / "kinovsr", REPO / "scripts" / "dev")


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def test_source_uses_logging_instead_of_direct_terminal_output():
    offenders: list[str] = []
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _dotted_name(node.func)
                direct = name == "print" or name in {
                    "sys.stdout.write",
                    "sys.stderr.write",
                }
                console = name is not None and name.endswith(".print")
                if direct or console:
                    rel = path.relative_to(REPO)
                    offenders.append(f"{rel}:{node.lineno}: {name}")
    assert offenders == []


def test_output_helpers_are_not_print_analogs():
    offenders: list[str] = []
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if "print" in node.name or node.name == "_say":
                    rel = path.relative_to(REPO)
                    offenders.append(f"{rel}:{node.lineno}: {node.name}")
    assert offenders == []
