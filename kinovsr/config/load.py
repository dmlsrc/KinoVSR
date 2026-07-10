"""TOML loading and base/specific composition."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .merge import merge_configs
from .validate import ConfigError


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one TOML file, wrapping errors with the filename."""
    p = Path(path).expanduser()
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read config {p}: {exc}") from exc
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"invalid TOML in {p}: {exc}") from exc


def compose_config(
    base_paths: list[str | Path] | None,
    specific_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compose repeatable base configs plus one specific overlay, in order.

    ``--base-config`` files apply first, in the order given; the
    ``--config`` file applies last. Missing arguments are simply skipped,
    so a run with no config files resolves to an empty mapping.
    """
    layers = [load_config(p) for p in (base_paths or [])]
    if specific_path is not None:
        layers.append(load_config(specific_path))
    return merge_configs(*layers)
