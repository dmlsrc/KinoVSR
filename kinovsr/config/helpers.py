"""Small helpers for family config parsers."""

import difflib
from collections.abc import Iterable, Mapping
from typing import overload


def reject_unknown_keys(raw: Mapping[str, object], known: Iterable[str]) -> None:
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
            raise ValueError(f"unknown key {key!r}{hint} (known: {', '.join(known)})")


@overload
def typed_value[T](raw: Mapping[str, object], key: str, kind: type[T]) -> T | None: ...


@overload
def typed_value[T, D](raw: Mapping[str, object], key: str, kind: type[T], default: D) -> T | D: ...


def typed_value[T, D](
    raw: Mapping[str, object], key: str, kind: type[T], default: D | None = None
) -> T | D | None:
    """Fetch ``raw[key]`` as ``kind`` (int upgrades to float), or default."""
    if key not in raw:
        return default
    value = raw[key]
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise ValueError(f"{key} must be a {kind.__name__}, got {value!r}")
    return value


def parse_edge_counts(spec: str) -> tuple[int, int, int, int]:
    """Parse a nonnegative ``T,B,L,R`` pixel-count value."""
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"edge spec must be T,B,L,R (four integers), got {spec!r}")
    values = [int(part) for part in parts]
    if any(value < 0 for value in values):
        raise ValueError(f"edge counts must be >= 0, got {spec!r}")
    return values[0], values[1], values[2], values[3]
