"""Darwin filesystem operations whose atomicity is part of publication."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path

_RENAME_EXCL = 0x00000004
_LIBC = ctypes.CDLL(None, use_errno=True)


def rename_exclusive(source: Path, destination: Path) -> None:
    """Atomically rename ``source`` only when ``destination`` is absent.

    KinoVSR is a macOS-native application. Darwin's ``renamex_np`` closes the
    check/use race that ``Path.exists()`` followed by ``Path.replace()`` would
    leave at requested artifact paths.
    """
    try:
        renamex = _LIBC.renamex_np
    except AttributeError as exc:  # pragma: no cover - non-Darwin guard
        raise OSError(
            errno.ENOTSUP,
            "atomic exclusive rename is unavailable on this platform",
            os.fspath(source),
        ) from exc
    renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renamex(
        os.fsencode(source),
        os.fsencode(destination),
        _RENAME_EXCL,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(
            error,
            os.strerror(error),
            os.fspath(source),
            os.fspath(destination),
        )
