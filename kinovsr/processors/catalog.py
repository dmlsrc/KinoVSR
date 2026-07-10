"""The explicit, lazy, first-party processor catalog.

One mapping from stable family name to the import target of its factory.
This is package-owned data, not a discovery system: adding a family means
adding one line here (and its tests). Importing this module imports no
family code; a family module loads only when selected, so ``import
kinovsr`` stays light and broken optional deps only surface for the
families that need them.

Names are canonical (planning 07): chain-token aliases like ``fastdvd``
are normalized in CLI assembly before they reach the catalog.
"""

from __future__ import annotations

import difflib
from importlib import import_module

from .errors import PipelineError
from .protocol import ProcessorFactory

# family name -> "module:attribute" of its ProcessorFactory instance.
# Populated as families gain factories (M3 proves three; M4 migrates the
# rest). The name on the left MUST equal the factory's own .name.
_FACTORY_TARGETS: dict[str, str] = {
    "bsvd": "kinovsr.bsvd.factory:FACTORY",
    "realplksr": "kinovsr.realplksr.factory:FACTORY",
    "videotoolbox": "kinovsr.processors.videotoolbox:FACTORY",
}

_loaded: dict[str, ProcessorFactory] = {}


class UnknownFamilyError(PipelineError, LookupError):
    """No such family in the first-party catalog."""

    def __init__(self, name: str) -> None:
        self.name = name
        hint = ""
        close = difflib.get_close_matches(name, _FACTORY_TARGETS, n=1)
        if close:
            hint = f"; did you mean {close[0]!r}?"
        available = ", ".join(sorted(_FACTORY_TARGETS)) or "(none yet)"
        super().__init__(
            f"unknown processor family {name!r}{hint} "
            f"(available: {available})")


def register(name: str, target: str) -> None:
    """Add one family. Duplicate names are rejected loudly - a duplicate
    means two modules claim the same user-facing name."""
    if name in _FACTORY_TARGETS:
        raise ValueError(
            f"processor family {name!r} registered twice "
            f"({_FACTORY_TARGETS[name]!r} and {target!r})")
    if ":" not in target:
        raise ValueError(
            f"catalog target for {name!r} must be 'module:attribute', "
            f"got {target!r}")
    _FACTORY_TARGETS[name] = target


def available_families() -> tuple[str, ...]:
    return tuple(sorted(_FACTORY_TARGETS))


def get_factory(name: str) -> ProcessorFactory:
    """Resolve a family name to its factory, importing it on first use."""
    factory = _loaded.get(name)
    if factory is not None:
        return factory
    target = _FACTORY_TARGETS.get(name)
    if target is None:
        raise UnknownFamilyError(name)
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
    "UnknownFamilyError",
    "available_families",
    "get_factory",
    "register",
]
