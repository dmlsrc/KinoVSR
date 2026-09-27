"""Runtime-only environment contract (M2 acceptance).

NumPy and OpenCV belong to optional evaluation/developer extras, PyAV to the
optional ffmpeg/evaluation extras, and torch and safetensors to the developer
extra: a clean core runtime environment must import kinovsr, its API, and the
``kinovsr`` entry point, and render ``kinovsr --help``, without them. The
subprocess installs a meta-path blocker BEFORE any import, which is the
reliable way to prove absence (the test process itself already has numpy
loaded). The same imports load no processor family: the catalog loads a
family by name only when a run or a family probe selects it.

Imports live at module scope (Ruff PLC0415). The one exception is a choice
point that loads an optional extra only on the path that needs it, marked
``# noqa: PLC0415 - <extra> extra``; the second test pins what such a marker
may cover.
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]

_BLOCKED_PROBE = r"""
import sys

BLOCKED = {"numpy", "av", "cv2", "torch", "safetensors"}

class Blocker:
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in BLOCKED:
            raise ImportError(f"{name} blocked: runtime-only environment")
        return None

sys.meta_path.insert(0, Blocker())

import kinovsr
import kinovsr.api
import kinovsr.cli.main
from kinovsr.cli.args import build_parser

text = build_parser().format_help()
assert "--upscale" in text
assert "--fastdvdnet-profile" in text
offenders = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
assert not offenders, f"blocked modules imported anyway: {offenders}"

import pathlib

import kinovsr.processors

root = pathlib.Path(kinovsr.processors.__file__).parent
families = {path.name for path in root.iterdir() if (path / "__init__.py").is_file()}
loaded = sorted(
    m for m in sys.modules if m.startswith("kinovsr.processors.") and m.split(".")[2] in families
)
assert not loaded, f"processor families loaded at startup: {loaded}"
print("runtime-only import ok")
"""

# Import names each optional extra provides; the keys are extras declared in
# pyproject.toml [project.optional-dependencies].
_EXTRA_MODULES = {
    "ffmpeg": frozenset({"av"}),
    "eval": frozenset({"av", "numpy", "cv2"}),
}
_MARK = re.compile(r"#\s*noqa:\s*PLC0415\s*-\s*(?P<extra>[\w-]+) extra\b")


def test_import_and_help_without_optional_extras():
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCKED_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "runtime-only import ok" in proc.stdout


def _module_path(module: str) -> Path | None:
    base = _REPO.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _module_scope_imports(module: str) -> set[str]:
    """Top-level packages a kinovsr module imports at module scope."""
    path = _module_path(module)
    if path is None:
        return set()
    found: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _loaded_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    loaded = []
    for alias in node.names:
        submodule = f"{node.module}.{alias.name}"
        loaded.append(submodule if _module_path(submodule) else node.module)
    return loaded


def test_function_level_imports_are_optional_extra_choice_points():
    """Each PLC0415 exception names a declared extra and loads a module that
    needs it: the extra's own package, or a kinovsr module importing one."""
    pyproject = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert set(_EXTRA_MODULES) <= set(pyproject["project"]["optional-dependencies"])

    problems = []
    marked = 0
    for path in sorted([*(_REPO / "kinovsr").rglob("*.py"), *(_REPO / "scripts").rglob("*.py")]):
        text = path.read_text(encoding="utf-8")
        if "PLC0415" not in text:
            continue
        rel = path.relative_to(_REPO).as_posix()
        lines = text.splitlines()
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if "PLC0415" not in lines[node.lineno - 1]:
                continue
            marked += 1
            where = f"{rel}:{node.lineno}"
            match = _MARK.search(lines[node.lineno - 1])
            if match is None or match["extra"] not in _EXTRA_MODULES:
                problems.append(f"{where}: the reason must name an optional extra")
                continue
            if node.lineno != node.end_lineno:
                problems.append(f"{where}: keep an exception import on one line")
                continue
            if isinstance(node, ast.ImportFrom) and node.level:
                problems.append(f"{where}: use an absolute import at an extra's choice point")
                continue
            needs = _EXTRA_MODULES[match["extra"]]
            for module in _loaded_modules(node):
                top = module.split(".")[0]
                if top in needs:
                    continue
                if top == "kinovsr" and _module_scope_imports(module) & needs:
                    continue
                problems.append(f"{where}: {module} does not need the {match['extra']} extra")
    assert marked, "expected the optional-extra choice points to carry PLC0415 markers"
    assert not problems, "\n".join(problems)
