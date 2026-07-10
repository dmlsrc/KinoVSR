"""Typed ``--set <table>.<key>=<value>`` CLI overrides.

Values are TOML-shaped so a family receives the same types whether a value
came from a config file or the command line: booleans, integers, floats,
arrays, and inline tables parse exactly as TOML scalars; an unquoted token
that is not valid TOML is a plain string; quoting preserves a
numeric-looking string (``--set x.pin='"42"'`` stays the string ``"42"``).
No ``{{VAR}}`` substitution is performed on CLI input.

The target path is ``table.key`` - the first dot separates the top-level
table (a stage name, or a reserved table like ``output``); any further
dots descend into nested mappings.
"""

from __future__ import annotations

import tomllib
from typing import Any

from .validate import ConfigError


def _parse_toml_value(raw: str) -> Any:
    """Parse a value with TOML scalar semantics; fall back to plain string."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def parse_set_argument(argument: str) -> tuple[list[str], Any]:
    """Parse one ``--set`` argument into (key path, typed value).

    Splits on the first ``=`` (values may contain ``=``) and on dots in
    the key path. The path needs at least a table and a key.
    """
    key_part, sep, value_part = argument.partition("=")
    if not sep:
        raise ConfigError(
            f"--set {argument!r}: expected <table>.<key>=<value>")
    path = [p.strip() for p in key_part.strip().split(".")]
    if len(path) < 2 or not all(path):
        raise ConfigError(
            f"--set {argument!r}: key path must be <table>.<key> "
            f"(got {key_part.strip()!r})")
    return path, _parse_toml_value(value_part)


def apply_set_overrides(
    config: dict[str, Any], set_arguments: list[str] | None,
) -> dict[str, Any]:
    """Return a new config with each ``--set`` applied, in order.

    Later ``--set`` arguments win over earlier ones and over config files,
    matching the documented precedence (CLI is the last layer).
    """
    out = dict(config)
    for argument in set_arguments or []:
        path, value = parse_set_argument(argument)
        node = out
        for i, key in enumerate(path[:-1]):
            existing = node.get(key)
            if existing is None:
                fresh: dict[str, Any] = {}
                node[key] = fresh
                node = fresh
            elif isinstance(existing, dict):
                copied = dict(existing)
                node[key] = copied
                node = copied
            else:
                raise ConfigError(
                    f"--set {argument!r}: {'.'.join(path[:i + 1])} is not a "
                    f"table (found {type(existing).__name__})")
        node[path[-1]] = value
    return out
