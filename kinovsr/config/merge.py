"""The three recursive config merge rules.

Rule 1: tables (mappings) merge recursively; later keys win, untouched
keys survive. Rule 2: scalars and arrays replace - no appending or
splicing. Rule 3 is a consequence of rule 2 that deserves its own name:
``pipeline`` is an array, so an overlay that wants a different chain
restates the whole list. There is no ``enabled`` flag and no merge-by-id;
the stage name is the id and inclusion is literal.
"""

from collections.abc import Mapping

# Stage-table keys owned by the framework; everything else in a stage table
# belongs to the processor family that parses it.
RESERVED_STAGE_KEYS = ("processor", "capability", "profile")


def _merge_tables(base: Mapping[str, object], overlay: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = dict(base)
    for key, value in overlay.items():
        if key in out:
            out[key] = _merge_two(out[key], value)
        else:
            out[key] = value
    return out


def _merge_two(base: object, overlay: object) -> object:
    if isinstance(base, Mapping) and isinstance(overlay, Mapping):
        return _merge_tables(base, overlay)
    return overlay


def merge_configs(*configs: Mapping[str, object]) -> dict[str, object]:
    """Merge mappings left to right under the three rules."""
    out: dict[str, object] = {}
    for cfg in configs:
        out = _merge_tables(out, cfg)
    return out


def split_stage_table(table: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Split a stage table into (reserved selector keys, family settings).

    The framework reads only ``processor``, ``capability``, and ``profile``;
    the second mapping is handed verbatim to the family's config parser.
    """
    selector = {k: table[k] for k in RESERVED_STAGE_KEYS if k in table}
    settings = {k: v for k, v in table.items() if k not in RESERVED_STAGE_KEYS}
    return selector, settings
