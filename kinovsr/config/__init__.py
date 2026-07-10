"""User configuration: TOML composition, typed CLI overrides, validation.

A config is data, not a language: global tables for the foundation
(``settings``, ``input``, ``output``, ``diagnostics``), one ordered
``pipeline`` list, and one named table per stage. Composition follows
exactly three rules (see :mod:`kinovsr.config.merge`):

1. tables merge recursively (later wins per key);
2. scalars and arrays replace;
3. the ``pipeline`` list is the sole source of stage order and inclusion -
   restate it to add, remove, or reorder stages.
"""

from .load import compose_config, load_config
from .merge import merge_configs, resolve_stage_config, split_stage_table
from .overrides import apply_set_overrides, parse_set_argument
from .validate import RESERVED_TABLES, ConfigError, validate_config

__all__ = [
    "RESERVED_TABLES",
    "ConfigError",
    "apply_set_overrides",
    "compose_config",
    "load_config",
    "merge_configs",
    "parse_set_argument",
    "resolve_stage_config",
    "split_stage_table",
    "validate_config",
]
