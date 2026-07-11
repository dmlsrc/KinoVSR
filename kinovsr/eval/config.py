"""Shared config helpers for VSR evaluation scripts."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

# Local configs are user files: discovered in the working directory and
# beside the sweep under scripts/dev/vsr_eval_sweep/ (where the example
# lives) - never inside the installed package.
_SWEEP_DIR = (Path(__file__).resolve().parents[2]
              / "scripts" / "dev" / "vsr_eval_sweep")
DEFAULT_CONFIG_CANDIDATES = (
    Path.cwd() / "vsr_eval.local.toml",
    Path.cwd() / "vsr_eval.local.json",
    _SWEEP_DIR / "vsr_eval.local.toml",
    _SWEEP_DIR / "vsr_eval.local.json",
)


def default_shared_temp() -> Path:
    # Settings is the package's only environment reader; its
    # shared_temp_dir carries the SHARED_TEMP_DIR convention.
    from kinovsr.settings import default_settings

    return default_settings().shared_temp_dir


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
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return path.resolve()


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


def as_str_list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return [] if default is None else default[:]
    if isinstance(value, str):
        return value.split()
    return [str(v) for v in value]
