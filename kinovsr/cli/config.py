"""Args-to-config assembly: flags + TOML into one resolved invocation.

Order of authority for global settings (the trifecta):

```text
dataclass defaults < environment < base TOML(s) < specific TOML < CLI flags
```

Only the ``[settings]`` table is consumed here in M2. Pipeline lists and
stage tables are composed and structurally validated by
:mod:`kinovsr.config`, but executing them requires the M3 builder, so
their presence is an explicit error rather than a silent no-op.

Assembly also owns the compatibility normalizations that keep legacy
spellings working (planning 07-cli-vocabulary.md): family-name aliases in
chain values (``fastdvd`` -> ``fastdvdnet``) and the deprecated
``--deblock-weights`` fill-in for chained deblockers.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinovsr.config import ConfigError, compose_config, validate_config
from kinovsr.settings import Settings, settings_from_args

# Chain-token aliases: accepted spelling -> canonical family name.
FAMILY_ALIASES = {
    "fastdvd": "fastdvdnet",
}

# Families a legacy --deblock-weights value may fill (when the family-level
# weights value is unset).
_DEBLOCK_WEIGHT_FAMILIES = ("stdf", "fbcnn", "toflow")


@dataclass(frozen=True)
class Invocation:
    """A fully resolved CLI invocation: settings, stage options, config.

    ``options`` is the transitional flat namespace of canonical option
    destinations consumed by the inherited orchestration. ``config`` is
    the composed TOML document; when it declares a ``pipeline`` list the
    run routes through the typed pipeline instead.
    """

    settings: Settings
    options: argparse.Namespace
    config: dict[str, Any] = dataclasses.field(default_factory=dict)


def normalize_chain(value: str) -> str:
    """Canonicalize family names in a comma-chain value ('off' passes)."""
    if not value or value == "off":
        return value
    names = [n.strip() for n in value.split(",")]
    return ",".join(FAMILY_ALIASES.get(n, n) for n in names if n)


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

    # Chain-token family aliases.
    args.denoise = normalize_chain(args.denoise)
    args.deblock = normalize_chain(args.deblock)

    # Deprecated --deblock-weights fills chained deblockers that got no
    # family-level value (never overrides one).
    if getattr(args, "deblock_weights", None):
        chain = [] if args.deblock == "off" else args.deblock.split(",")
        for family in _DEBLOCK_WEIGHT_FAMILIES:
            dest = f"{family}_weights"
            if family in chain and getattr(args, dest, None) is None:
                setattr(args, dest, args.deblock_weights)

    settings = base if base is not None else Settings.from_env()
    settings = _settings_from_table(settings, config.get("settings", {}))
    settings = settings_from_args(args, settings)
    return Invocation(settings=settings, options=args, config=config)
