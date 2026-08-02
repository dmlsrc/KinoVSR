"""Structural validation of a composed config.

This is the config-shape layer only: the pipeline builder (M3) adds
semantic validation (known families, capabilities, profiles, typed stream
edges). Everything here fails before any processing starts, with the
offending name in the message.
"""

from __future__ import annotations

from typing import Any

# Top-level tables owned by the foundation; everything else at top level is
# a stage table. ``pipeline`` is the one non-table top-level key.
RESERVED_TABLES = ("settings", "input", "output", "diagnostics")


class ConfigError(Exception):
    """A user-config problem, reported before any processing starts."""


def validate_config(
    config: dict[str, Any], *, allow_stage_fragments: bool = False,
) -> None:
    """Validate the composed config's structure.

    - top-level keys are ``pipeline`` (a list of strings) or tables;
    - every ``pipeline`` entry names a top-level stage table;
    - reserved tables are not listed as stages;
    - every listed stage table selects a ``processor`` (a string);
    - optional ``capability``/``profile`` selectors are strings.

    Duplicate pipeline entries are legal: each occurrence builds an
    independent instance from the shared table.
    """
    pipeline = config.get("pipeline", [])
    if not isinstance(pipeline, list) or not all(isinstance(s, str) for s in pipeline):
        raise ConfigError("pipeline must be an array of stage names")

    for key, value in config.items():
        if key == "pipeline":
            continue
        if not isinstance(value, dict):
            raise ConfigError(
                f"top-level key {key!r} must be a table (stage tables and the "
                f"reserved tables {list(RESERVED_TABLES)} are tables; only "
                f"'pipeline' is a list)")
        if key in RESERVED_TABLES or key in pipeline:
            continue
        processor = value.get("processor")
        if allow_stage_fragments and processor is None:
            continue
        if not isinstance(processor, str) or not processor:
            raise ConfigError(
                f"unknown top-level table [{key}]: an unlisted stage table "
                "must select processor = \"<family>\"")

    for name in pipeline:
        if name in RESERVED_TABLES:
            raise ConfigError(
                f"pipeline lists {name!r}, which is a reserved foundation "
                f"table, not a stage")
        table = config.get(name)
        if table is None:
            raise ConfigError(
                f"pipeline lists {name!r} but no [{name}] stage table exists")
        processor = table.get("processor")
        if not isinstance(processor, str) or not processor:
            raise ConfigError(
                f"[{name}] must select a processor family "
                f"(processor = \"<family>\")")
        for selector in ("capability", "profile"):
            if selector in table and not isinstance(table[selector], str):
                raise ConfigError(
                    f"[{name}] {selector} must be a string")
