"""Compose foundation, slot, conditioning, and family CLI option rows.

Family contributions live beside their processor implementations; shared
foundation and cross-family slot rows live in :mod:`foundation_options`. This
module is the single ordered composition point consumed by the parser and the
vocabulary conformance tests.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from kinovsr.processors.catalog import catalog_entries

from .foundation_options import (
    DEBLOCK_SLOT_OPTIONS,
    DEBLOCK_STRENGTH_OPTIONS,
    DENOISE_SLOT_OPTIONS,
    PREPROCESS_SLOT_OPTIONS,
    RUNTIME_OPTIONS,
    SOURCE_AND_OUTPUT_OPTIONS,
    UPSCALE_SLOT_OPTIONS,
)
from .options import Opt


def _load_contribution(path: str, *names: str) -> tuple[list[Opt], ...]:
    """Load data-only family rows without importing family implementations.

    A normal family submodule import executes the family package initializer
    first. Several families define their runtime there, which would initialize
    MLX or native frameworks just to build CLI help. Loading the adjacent data
    contribution by file path preserves the lazy catalog boundary while keeping
    row ownership inside the family directory.
    """
    source = Path(__file__).parents[1] / "processors" / path
    module_name = "_kinovsr_cli_options_" + path.replace("/", "_").removesuffix(".py")
    spec = spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CLI option contribution {source}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(getattr(module, name) for name in names)


(DEBLOCK_MAP_OPTIONS, NOISE_MAP_OPTIONS) = _load_contribution(
    "conditioning_cli_options.py", "DEBLOCK_MAP_OPTIONS", "NOISE_MAP_OPTIONS")


def _family_fragments() -> list[tuple[int, list[Opt]]]:
    fragments = []
    for entry in catalog_entries():
        if not entry.cli_options:
            continue
        path = entry.cli_options_path
        assert path is not None
        exports = tuple(item.export for item in entry.cli_options)
        option_lists = _load_contribution(path, *exports)
        fragments.extend(
            (item.order, options)
            for item, options in zip(
                entry.cli_options, option_lists, strict=True)
        )
    return fragments


_FRAGMENTS = [
    (10, SOURCE_AND_OUTPUT_OPTIONS),
    (60, PREPROCESS_SLOT_OPTIONS),
    (90, DEBLOCK_SLOT_OPTIONS),
    (120, DEBLOCK_STRENGTH_OPTIONS),
    (130, DEBLOCK_MAP_OPTIONS),
    (150, DENOISE_SLOT_OPTIONS),
    (260, NOISE_MAP_OPTIONS),
    (300, UPSCALE_SLOT_OPTIONS),
    (390, RUNTIME_OPTIONS),
    *_family_fragments(),
]
_orders = [order for order, _options in _FRAGMENTS]
if len(_orders) != len(set(_orders)):
    raise RuntimeError("duplicate CLI option fragment order in processor catalog")

REGISTRY = [
    option
    for _order, options in sorted(_FRAGMENTS, key=lambda fragment: fragment[0])
    for option in options
]
