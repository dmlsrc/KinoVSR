"""The three merge rules, and per-stage precedence resolution.

Rule 1: tables (mappings) merge recursively; later keys win, untouched
keys survive. Rule 2: scalars and arrays replace - no appending or
splicing. Rule 3 is a consequence of rule 2 that deserves its own name:
``pipeline`` is an array, so an overlay that wants a different chain
restates the whole list. There is no ``enabled`` flag and no merge-by-id;
the stage name is the id and inclusion is literal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Stage-table keys owned by the framework; everything else in a stage table
# belongs to the processor family that parses it.
RESERVED_STAGE_KEYS = ("processor", "capability", "profile")


def _merge_two(base: Any, overlay: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(overlay, Mapping):
        out: dict[str, Any] = dict(base)
        for key, value in overlay.items():
            if key in out:
                out[key] = _merge_two(out[key], value)
            else:
                out[key] = value
        return out
    return overlay


def merge_configs(*configs: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings left to right under the three rules."""
    out: dict[str, Any] = {}
    for cfg in configs:
        out = _merge_two(out, cfg)
    return out


def split_stage_table(table: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a stage table into (reserved selector keys, family settings).

    The framework reads only ``processor``, ``capability``, and ``profile``;
    the second mapping is handed verbatim to the family's config parser.
    """
    selector = {k: table[k] for k in RESERVED_STAGE_KEYS if k in table}
    settings = {k: v for k, v in table.items() if k not in RESERVED_STAGE_KEYS}
    return selector, settings


def resolve_stage_config(
    family_defaults: Mapping[str, Any] | None,
    profile_preset: Mapping[str, Any] | None,
    stage_settings: Mapping[str, Any] | None,
    set_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one stage's settings mapping under the documented precedence:

    ``family defaults < profile < merged stage table < typed --set``

    Every layer is optional; each merges under the same three rules. The
    result is what the family's config parser receives.
    """
    return merge_configs(
        family_defaults or {},
        profile_preset or {},
        stage_settings or {},
        set_overrides or {},
    )
