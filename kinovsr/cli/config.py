"""Args-to-config assembly: flags + TOML into one resolved invocation.

Order of authority for global settings (the trifecta):

```text
dataclass defaults < environment < base TOML(s) < specific TOML < CLI flags
```

Only the ``[settings]`` table is consumed here in M2. Pipeline lists and
stage tables are composed and structurally validated by
:mod:`kinovsr.config`, but executing them requires the M3 builder, so
their presence is an explicit error rather than a silent no-op.

Assembly normalizes whitespace in comma-separated processor chains before
the flag surface is converted to typed pipeline configuration.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinovsr.config import ConfigError, compose_config, validate_config
from kinovsr.processors.catalog import available_families
from kinovsr.settings import (
    Settings,
    set_default_settings,
    settings_from_args,
)


@dataclass(frozen=True)
class Invocation:
    """A fully resolved CLI invocation: settings, stage options, config.

    ``options`` is the flat namespace of canonical option destinations consumed
    by the flag-to-pipeline assembler. ``config`` is the composed TOML document;
    an explicit ``pipeline`` list bypasses flag assembly and supplies the typed
    pipeline directly.
    """

    settings: Settings
    options: argparse.Namespace
    config: dict[str, Any] = dataclasses.field(default_factory=dict)


def normalize_chain(value: str) -> str:
    """Normalize and validate a comma-chain value (``off`` passes)."""
    if not value or value == "off":
        return value
    names = [n.strip() for n in value.split(",")]
    available = set(available_families())
    unknown = [name for name in names if name and name not in available]
    if unknown:
        raise ConfigError(
            f"unknown processor family in chain: {', '.join(unknown)}")
    return ",".join(n for n in names if n)


def _reject_profile_tokens_in_weight_flags(args: argparse.Namespace) -> None:
    """Keep CLI weight overrides path-only; named checkpoints use profiles."""
    from ._registry import REGISTRY

    selectors = {}
    for opt in REGISTRY:
        tokens = (opt.choices if opt.key == "profile"
                  else opt.profile_tokens)
        if opt.family is not None and tokens:
            selectors[opt.family] = (opt.flag, frozenset(tokens))
    for opt in REGISTRY:
        if opt.family is None or opt.key != "weights":
            continue
        value = getattr(args, opt.resolved_dest, None)
        selector = selectors.get(opt.family)
        if value is None or selector is None or value not in selector[1]:
            continue
        raise ConfigError(
            f"{opt.flag} expects a checkpoint path; use "
            f"{selector[0]} {value} to select that profile")


def _settings_from_table(base: Settings, table: dict[str, Any]) -> Settings:
    """Apply a TOML ``[settings]`` table on top of ``base``, typed."""
    known = {f.name: f for f in dataclasses.fields(Settings)}
    overrides: dict[str, Any] = {}
    for key, value in table.items():
        f = known.get(key)
        if f is None:
            raise ConfigError(f"[settings] unknown key {key!r}")
        annotation = f.type if isinstance(f.type, str) else str(f.type)
        if "bool" in annotation:
            if not isinstance(value, bool):
                raise ConfigError(f"[settings] {key} must be a boolean")
            overrides[key] = value
        elif "Path" in annotation:
            if not isinstance(value, str):
                raise ConfigError(f"[settings] {key} must be a path string")
            overrides[key] = Path(value).expanduser()
        elif "float" in annotation:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"[settings] {key} must be a number")
            overrides[key] = float(value)
        elif "int" in annotation:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"[settings] {key} must be an integer")
            overrides[key] = value
        else:
            if not isinstance(value, str):
                raise ConfigError(f"[settings] {key} must be a string")
            overrides[key] = value
    return base.with_overrides(**overrides)


def assemble(args: argparse.Namespace,
             base: Settings | None = None) -> Invocation:
    """Resolve one invocation from parsed args and config files.

    A config that declares a ``pipeline`` list is a typed-pipeline run:
    the stage tables are structurally validated here and resolved by the
    pipeline builder at run time (the run command routes it through
    :func:`kinovsr.pipeline.run_file`).
    """
    config = compose_config(args.base_config, args.config)
    validate_config(config)

    # Normalize comma-chain whitespace before flag-to-pipeline assembly.
    args.denoise = normalize_chain(args.denoise)
    args.deblock = normalize_chain(args.deblock)
    _reject_profile_tokens_in_weight_flags(args)

    settings = base if base is not None else Settings.from_env()
    settings = _settings_from_table(settings, config.get("settings", {}))
    settings = settings_from_args(args, settings)
    # Publish the resolved result so consumers too deep to be handed the
    # instance (model internals below the threading layers) see the config
    # table and command line, not just the environment.
    set_default_settings(settings)
    return Invocation(settings=settings, options=args, config=config)
