"""Small helpers for family config parsers."""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping
from typing import Any


def reject_unknown_keys(raw: Mapping[str, Any],
                        known: Iterable[str]) -> None:
    """Raise ValueError naming the first unknown key, with a suggestion.

    Families call this at the top of ``parse_config``; the builder wraps
    the ValueError into a StageConfigError carrying the stage name, which
    yields the documented error shape:

        [deblock-final] unknown key 'flowscale'; did you mean 'flow_scale'?
    """
    known = sorted(known)
    for key in raw:
        if key not in known:
            hint = ""
            close = difflib.get_close_matches(key, known, n=1)
            if close:
                hint = f"; did you mean {close[0]!r}?"
            raise ValueError(
                f"unknown key {key!r}{hint} (known: {', '.join(known)})")


def typed_value(raw: Mapping[str, Any], key: str, kind: type,
                default: Any = None) -> Any:
    """Fetch ``raw[key]`` as ``kind`` (int upgrades to float), or default."""
    if key not in raw:
        return default
    value = raw[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise ValueError(f"{key} must be a {kind.__name__}, got {value!r}")
    return value
