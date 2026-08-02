"""Args-to-config assembly: flags + TOML into one resolved invocation.

Order of authority for global settings (the trifecta):

```text
dataclass defaults < environment < base TOML(s) < specific TOML < CLI flags
```

The reserved foundation tables and their CLI flags resolve here. Stage tables
remain data until the source probe supplies the dimensions needed by the flag
assembler; both chain authoring surfaces then share one dial/override merge.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinovsr.config import (
    ConfigError,
    apply_set_overrides,
    compose_config,
    validate_config,
)
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
        elif "int" in annotation:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"[settings] {key} must be an integer")
            overrides[key] = value
        elif "float" in annotation:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"[settings] {key} must be a number")
            overrides[key] = float(value)
        else:
            if not isinstance(value, str):
                raise ConfigError(f"[settings] {key} must be a string")
            overrides[key] = value
    return base.with_overrides(**overrides)


def _settings_table(settings: Settings) -> dict[str, Any]:
    """Materialize defaults/env/TOML/flags so printed config is reproducible."""
    table = {}
    for field in dataclasses.fields(Settings):
        value = getattr(settings, field.name)
        if value is None:
            continue                    # TOML has no null value
        table[field.name] = str(value) if isinstance(value, Path) else value
    return table


def _foundation_value(opt, value: Any) -> Any:
    """Validate one reserved-table value using its parser registry row."""
    where = f"[{opt.config_table}] {opt.resolved_dest}"
    if opt.kind == "flag":
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be a boolean")
    elif opt.type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer")
    elif opt.type is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where} must be a number")
        value = float(value)
    elif not isinstance(value, str):
        raise ConfigError(f"{where} must be a string")
    if opt.choices is not None and value not in opt.choices:
        raise ConfigError(
            f"{where} must be one of {list(opt.choices)} (got {value!r})")
    return value


def _foundation_options(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    apply_cli: bool,
) -> dict[str, Any]:
    """Resolve and materialize the 27 run options from registry metadata."""
    from ._registry import REGISTRY

    rows = [opt for opt in REGISTRY if opt.config_table is not None]
    allowed: dict[str, set[str]] = {}
    out = dict(config)
    tables: dict[str, dict[str, Any]] = {}
    for opt in rows:
        table_name = opt.config_table
        assert table_name is not None
        allowed.setdefault(table_name, set()).add(opt.resolved_dest)
        if table_name not in tables:
            tables[table_name] = dict(out.get(table_name, {}))

    for table_name, table in tables.items():
        unknown = set(table) - allowed[table_name]
        if unknown:
            names = ", ".join(repr(name) for name in sorted(unknown))
            raise ConfigError(f"[{table_name}] unknown key(s): {names}")

    for opt in rows:
        table_name = opt.config_table
        assert table_name is not None
        key = opt.resolved_dest
        table = tables[table_name]
        value = table.get(key, opt.default)
        cli_value = getattr(args, key, opt.default)
        if apply_cli and cli_value is not None and cli_value != opt.default:
            value = cli_value
        if value is not None:
            value = _foundation_value(opt, value)
            table[key] = value
        else:
            table.pop(key, None)
        setattr(args, key, value)

    out.update(tables)
    return out


def assemble(args: argparse.Namespace,
             base: Settings | None = None) -> Invocation:
    """Resolve one invocation from parsed args and config files.

    A config that declares a ``pipeline`` list is a typed-pipeline run:
    the stage tables are structurally validated here and resolved by the
    pipeline builder at run time (the run command routes it through
    :func:`kinovsr.pipeline.run_file`).
    """
    config = compose_config(args.base_config, args.config)
    # Without an explicit pipeline, config files may provide partial tables
    # for the stage names the flag assembler will create after probing.  The
    # final merged config is validated strictly there.
    validate_config(config, allow_stage_fragments="pipeline" not in config)

    # Normalize comma-chain whitespace before flag-to-pipeline assembly.
    args.denoise = normalize_chain(args.denoise)
    args.deblock = normalize_chain(args.deblock)
    _reject_profile_tokens_in_weight_flags(args)

    settings = base if base is not None else Settings.from_env()
    settings = _settings_from_table(settings, config.get("settings", {}))
    settings = settings_from_args(args, settings)
    config = dict(config)
    config["settings"] = _settings_table(settings)
    config = _foundation_options(config, args, apply_cli=True)

    # ``--set`` is the final layer for every table. Apply it once here so
    # input/settings overrides affect probing and runtime setup, then again
    # after the dimension-dependent chain is assembled so stage overrides
    # remain last as well (the operation is idempotent).
    config = apply_set_overrides(config, getattr(args, "set", None))
    validate_config(config, allow_stage_fragments="pipeline" not in config)
    config = _foundation_options(config, args, apply_cli=False)
    settings = _settings_from_table(Settings(), config.get("settings", {}))
    # Publish the resolved result so consumers too deep to be handed the
    # instance (model internals below the threading layers) see the config
    # table and command line, not just the environment.
    set_default_settings(settings)
    return Invocation(settings=settings, options=args, config=config)
