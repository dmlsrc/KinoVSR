"""Import layering of the kinovsr package, checked from source.

A processor family is reached only through the processor catalog: code
outside a family never imports it, and families never import each other.
Shared layers (modeling, media, native, analysis, config, the processor
core, pipeline) own what several families use, and a module never imports
another package's private (leading-underscore) names.

The checks parse every module's AST; nothing here imports product code.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PACKAGE = Path(__file__).resolve().parents[1] / "kinovsr"
_FAMILIES = frozenset(
    path.name for path in (_PACKAGE / "processors").iterdir() if (path / "__init__.py").is_file()
)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(_PACKAGE.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _owner(module: str) -> str:
    """The unit that owns a module: its family, else its top-level package."""
    parts = module.split(".")
    if len(parts) >= 3 and parts[1] == "processors" and parts[2] in _FAMILIES:
        return ".".join(parts[:3])
    return ".".join(parts[:2])


def _imports(path: Path) -> list[tuple[int, str, list[str]]]:
    """(line, absolute module, imported names) for every import statement."""
    importer = _module_name(path)
    base = importer.split(".") if path.name == "__init__.py" else importer.split(".")[:-1]
    found: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name, []) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = base[: len(base) - (node.level - 1)]
                module = ".".join([*anchor, *([node.module] if node.module else [])])
            else:
                module = node.module or ""
            found.append((node.lineno, module, [alias.name for alias in node.names]))
    return found


def _modules() -> list[Path]:
    return sorted(_PACKAGE.rglob("*.py"))


def test_families_are_imported_only_by_themselves() -> None:
    findings = []
    for path in _modules():
        importer = _module_name(path)
        for line, module, names in _imports(path):
            if not module.startswith("kinovsr."):
                continue
            for target in (module, *(f"{module}.{name}" for name in names)):
                owner = _owner(target)
                if owner.startswith("kinovsr.processors.") and owner != _owner(importer):
                    findings.append(f"{importer}:{line} imports {target}")
                    break
    assert not findings, "family imported outside itself (use the catalog):\n" + "\n".join(findings)


def test_private_names_stay_in_their_package() -> None:
    findings = []
    for path in _modules():
        importer = _module_name(path)
        for line, module, names in _imports(path):
            if not module.startswith("kinovsr.") or _owner(module) == _owner(importer):
                continue
            private = [n for n in names if n.startswith("_") and not n.startswith("__")]
            private_segments = [part for part in module.split(".") if part.startswith("_")]
            if private or private_segments:
                findings.append(f"{importer}:{line} imports {module} {private or ''}".rstrip())
    assert not findings, "private names imported across packages:\n" + "\n".join(findings)
