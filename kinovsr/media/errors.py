"""Classification and normalization for native bridge failures.

PyObjC status failures do not share a built-in Python base class with the
``RuntimeError`` values raised by KinoVSR's explicit native checks. Keep the
bridge import lazy so importing the public API remains framework-light.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any


def is_objc_error(exc: BaseException) -> bool:
    """Whether ``exc`` is PyObjC's native bridge exception."""
    try:
        import objc
    except ImportError:
        return False
    return isinstance(exc, objc.error)


def is_native_operation_error(
    exc: BaseException,
    *,
    allow_value_error: bool = False,
) -> bool:
    """Whether an exception from a known native call is operational."""
    return (
        isinstance(exc, (OSError, RuntimeError))
        or (allow_value_error and isinstance(exc, ValueError))
        or is_objc_error(exc)
    )


@contextlib.contextmanager
def media_operation(
    operation: str,
    path: Any,
    *,
    allow_value_error: bool = False,
) -> Iterator[None]:
    """Normalize one explicit native or filesystem operation.

    The single normalize idiom every boundary shares: typed
    ``PipelineError`` values pass through untouched (no double wrap),
    operational failures become ``MediaError`` with the operation and
    path, and programmer errors plus process-control exceptions
    deliberately pass through unchanged.  Keep the scope at each call
    site narrow: ``RuntimeError`` is the established failure type of
    the native media adapters, but can also describe an internal
    invariant elsewhere.
    """
    from kinovsr.processors.errors import MediaError, PipelineError

    try:
        yield
    except PipelineError:
        raise
    except Exception as exc:
        if is_native_operation_error(
                exc, allow_value_error=allow_value_error):
            raise MediaError(f"{operation} {path}: {exc}") from exc
        raise


__all__ = ["is_native_operation_error", "is_objc_error", "media_operation"]
