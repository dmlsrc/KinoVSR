"""Guarded imports for optional developer dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def require_numpy(purpose: str) -> Any:
    """Return NumPy or raise a clear optional-dependency error."""
    try:
        return import_module("numpy")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"NumPy is required for {purpose}. Install the developer extras for "
            "diagnostics, converters, and tests."
        ) from exc
