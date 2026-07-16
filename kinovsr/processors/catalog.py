"""The explicit, lazy, first-party processor catalog.

One package-owned index maps each stable family name to its factory target and
data-only CLI option contributions. This is not a discovery system: adding a
family means adding one catalog entry and its tests. Importing this module
imports no family code; a runtime module loads only when selected, and CLI
assembly reads contribution data without executing family package initializers.

Names are canonical (planning 07); retired family spellings are rejected
before lookup.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from importlib import import_module

from .errors import PipelineError
from .protocol import ProcessorFactory


@dataclass(frozen=True, slots=True)
class CliOptionContribution:
    """One ordered option-list export from a family's data-only module."""

    order: int
    export: str


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """All package-owned metadata for one first-party processor family."""

    name: str
    factory_target: str
    cli_options: tuple[CliOptionContribution, ...] = ()

    @property
    def cli_options_path(self) -> str | None:
        if not self.cli_options:
            return None
        return f"{self.name}/cli_options.py"


def _entry(
    name: str,
    factory_target: str,
    *cli_options: tuple[int, str],
) -> CatalogEntry:
    return CatalogEntry(
        name=name,
        factory_target=factory_target,
        cli_options=tuple(
            CliOptionContribution(order, export)
            for order, export in cli_options
        ),
    )


# This is the one explicit first-party family index. CLI composition consumes
# the data-only contribution metadata without importing any family runtime.
_CATALOG: dict[str, CatalogEntry] = {
    entry.name: entry
    for entry in (
        _entry(
            "basicvsrpp",
            "kinovsr.processors.basicvsrpp.factory:FACTORY",
            (70, "BASICVSRPP_RESTORE_OPTIONS"),
            (320, "BASICVSRPP_UPSCALE_OPTIONS"),
        ),
        _entry(
            "bsvd",
            "kinovsr.processors.bsvd.factory:FACTORY",
            (190, "BSVD_STRENGTH_OPTIONS"),
            (230, "BSVD_OPTIONS"),
        ),
        _entry(
            "crop",
            "kinovsr.processors.crop:FACTORY",
            (30, "CROP_OPTIONS")),
        _entry(
            "conform",
            "kinovsr.processors.conform:FACTORY",
            (85, "CONFORM_OPTIONS"),
        ),
        _entry(
            "cut_detect",
            "kinovsr.processors.cut_detect:FACTORY",
            (50, "CUT_OPTIONS"),
        ),
        _entry(
            "deflicker",
            "kinovsr.processors.deflicker:FACTORY",
            (80, "DEFLICKER_OPTIONS"),
        ),
        _entry(
            "esc", "kinovsr.processors.esc.factory:FACTORY",
            (360, "ESC_OPTIONS"),
        ),
        _entry(
            "fastdvdnet",
            "kinovsr.processors.fastdvdnet.factory:FACTORY",
            (180, "FASTDVDNET_STRENGTH_OPTIONS"),
            (220, "FASTDVDNET_OPTIONS"),
        ),
        _entry(
            "fbcnn",
            "kinovsr.processors.fbcnn.factory:FACTORY",
            (110, "FBCNN_WEIGHT_OPTIONS"),
            (140, "FBCNN_OPTIONS"),
        ),
        _entry(
            "mc",
            "kinovsr.processors.mc:FACTORY",
            (170, "MC_STRENGTH_OPTIONS"),
            (270, "MC_OPTIONS"),
        ),
        _entry(
            "metalfx",
            "kinovsr.processors.metalfx:FACTORY",
            (310, "METALFX_OPTIONS"),
        ),
        _entry(
            "nafnet",
            "kinovsr.processors.nafnet.factory:FACTORY",
            (290, "NAFNET_OPTIONS"),
        ),
        _entry(
            "pvdd",
            "kinovsr.processors.pvdd.factory:FACTORY",
            (210, "PVDD_STRENGTH_OPTIONS"),
            (250, "PVDD_OPTIONS"),
        ),
        _entry(
            "realbasicvsr",
            "kinovsr.processors.realbasicvsr.factory:FACTORY",
            (330, "REALBASICVSR_OPTIONS"),
        ),
        _entry(
            "realesrgan",
            "kinovsr.processors.realesrgan.factory:FACTORY",
            (340, "REALESRGAN_OPTIONS"),
        ),
        _entry(
            "realplksr",
            "kinovsr.processors.realplksr.factory:FACTORY",
            (370, "REALPLKSR_OPTIONS"),
        ),
        _entry(
            "realviformer",
            "kinovsr.processors.realviformer.factory:FACTORY",
            (350, "REALVIFORMER_OPTIONS"),
        ),
        _entry(
            "safmn",
            "kinovsr.processors.safmn.factory:FACTORY",
            (380, "SAFMN_OPTIONS"),
        ),
        _entry(
            "sanitize_edges",
            "kinovsr.processors.sanitize_edges:FACTORY",
            (20, "SANITIZE_EDGE_OPTIONS"),
        ),
        _entry(
            "spatial",
            "kinovsr.processors.spatial:FACTORY",
            (160, "SPATIAL_OPTIONS"),
        ),
        _entry(
            "square_pixels",
            "kinovsr.processors.square_pixels:FACTORY",
            (40, "SQUARE_PIXELS_OPTIONS"),
        ),
        _entry(
            "stdf",
            "kinovsr.processors.stdf.factory:FACTORY",
            (100, "STDF_OPTIONS"),
        ),
        _entry(
            "toflow",
            "kinovsr.processors.toflow.factory:FACTORY",
            (200, "TOFLOW_STRENGTH_OPTIONS"),
            (240, "TOFLOW_OPTIONS"),
            (280, "TOFLOW_UPSCALE_OPTIONS"),
        ),
        _entry(
            "videotoolbox",
            "kinovsr.processors.videotoolbox:FACTORY",
            (15, "VIDEOTOOLBOX_OPTIONS"),
        ),
    )
}

_loaded: dict[str, ProcessorFactory] = {}


class UnknownFamilyError(PipelineError, LookupError):
    """No such family in the first-party catalog."""

    def __init__(self, name: str) -> None:
        self.name = name
        hint = ""
        close = difflib.get_close_matches(name, _CATALOG, n=1)
        if close:
            hint = f"; did you mean {close[0]!r}?"
        available = ", ".join(sorted(_CATALOG)) or "(none yet)"
        super().__init__(
            f"unknown processor family {name!r}{hint} "
            f"(available: {available})")


def register(name: str, target: str) -> None:
    """Add one family. Duplicate names are rejected loudly - a duplicate
    means two modules claim the same user-facing name."""
    if name in _CATALOG:
        raise ValueError(
            f"processor family {name!r} registered twice "
            f"({_CATALOG[name].factory_target!r} and {target!r})")
    if ":" not in target:
        raise ValueError(
            f"catalog target for {name!r} must be 'module:attribute', "
            f"got {target!r}")
    _CATALOG[name] = CatalogEntry(name=name, factory_target=target)


def available_families() -> tuple[str, ...]:
    return tuple(sorted(_CATALOG))


def catalog_entries() -> tuple[CatalogEntry, ...]:
    """Return the first-party entries in stable family-name order."""
    return tuple(_CATALOG[name] for name in sorted(_CATALOG))


def get_factory(name: str) -> ProcessorFactory:
    """Resolve a family name to its factory, importing it on first use."""
    factory = _loaded.get(name)
    if factory is not None:
        return factory
    entry = _CATALOG.get(name)
    if entry is None:
        raise UnknownFamilyError(name)
    target = entry.factory_target
    module_name, _, attribute = target.partition(":")
    module = import_module(module_name)
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise PipelineError(
            f"catalog target {target!r} for family {name!r} does not "
            f"exist") from exc
    if getattr(factory, "name", None) != name:
        raise PipelineError(
            f"catalog name {name!r} does not match factory name "
            f"{getattr(factory, 'name', None)!r} from {target!r}")
    if not factory.capabilities:
        raise PipelineError(
            f"family {name!r} declares no capabilities")
    _loaded[name] = factory
    return factory


__all__ = [
    "CatalogEntry",
    "CliOptionContribution",
    "UnknownFamilyError",
    "available_families",
    "catalog_entries",
    "get_factory",
    "register",
]
