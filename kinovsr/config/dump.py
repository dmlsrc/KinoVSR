"""Small deterministic TOML writer for resolved KinoVSR config data."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .validate import ConfigError

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(value: str) -> str:
    return value if _BARE_KEY.fullmatch(value) else json.dumps(value, ensure_ascii=False)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str | Path):
        return json.dumps(str(value), ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "-inf" if value < 0 else "inf"
        return repr(value)
    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        items = ", ".join(
            f"{_key(str(key))} = {_value(item)}" for key, item in value.items())
        return "{ " + items + " }"
    raise ConfigError(
        f"cannot print config value {value!r} ({type(value).__name__}) as TOML")


def dump_config(config: Mapping[str, Any]) -> str:
    """Serialize a resolved config as parseable TOML, root values first."""
    lines = [
        f"{_key(str(key))} = {_value(value)}"
        for key, value in config.items()
        if not isinstance(value, Mapping)
    ]
    for name, table in config.items():
        if not isinstance(table, Mapping):
            continue
        if lines:
            lines.append("")
        lines.append(f"[{_key(str(name))}]")
        lines.extend(
            f"{_key(str(key))} = {_value(value)}"
            for key, value in table.items()
        )
    return "\n".join(lines) + "\n"


__all__ = ["dump_config"]
