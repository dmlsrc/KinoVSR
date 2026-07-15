"""Internal classification for native bridge failures.

PyObjC status failures do not share a built-in Python base class with the
``RuntimeError`` values raised by KinoVSR's explicit native checks. Keep the
bridge import lazy so importing the public API remains framework-light.
"""

from __future__ import annotations


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


__all__ = ["is_native_operation_error", "is_objc_error"]
