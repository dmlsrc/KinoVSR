"""Shared config helpers for synthetic VSR artifact fixture tools."""
from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
DEFAULT_CONFIG_CANDIDATES = (
    TOOL_DIR / "vsr_artifacts.local.toml",
    TOOL_DIR / "vsr_artifacts.local.json",
)


def default_shared_temp() -> Path:
    if value := os.environ.get("SHARED_TEMP_DIR"):
        return Path(value)
    return REPO_ROOT / "tmp"


def load_config(path: Path | None) -> tuple[dict[str, Any], Path | None]:
    if path is None:
        for candidate in DEFAULT_CONFIG_CANDIDATES:
            if candidate.exists():
                path = candidate
                break
    if path is None:
        return {}, None
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"unsupported config extension {path.suffix!r}; use .toml or .json")
    if not isinstance(data, dict):
        raise SystemExit(f"config must contain an object/table: {path}")
    return data, path


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise SystemExit(f"config section [{name}] must be a table/object")
    return section


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def config_get(section: dict[str, Any], args: Any, key: str, default: Any = None) -> Any:
    cli_value = getattr(args, key, None)
    if cli_value is not None:
        return cli_value
    return section.get(key, default)


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)
