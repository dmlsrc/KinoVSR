"""Darwin filesystem operations whose atomicity is part of publication."""

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
        # Positional slot 4 is winerror (ignored on POSIX); the destination
        # belongs in filename2 so str(exc) reports both paths.
        raise OSError(
            error,
            os.strerror(error),
            os.fspath(source),
            None,
            os.fspath(destination),
        )


def partial_name(destination: Path, attempt: int = 1) -> Path:
    """``<stem>.partial<suffix>`` beside ``destination``; ``.partial-N`` for later attempts."""
    counter = "" if attempt == 1 else f"-{attempt}"
    return destination.with_name(f"{destination.stem}.partial{counter}{destination.suffix}")


def create_partial_output(destination: Path, *, attempts: int = 1000) -> Path:
    """Create, exclusively, the visible file a video is written to before it is final.

    The name is ``<stem>.partial<suffix>`` beside ``destination``, or the first
    free ``.partial-N``, so an earlier run's leftover is never overwritten. The
    writer renames it over the destination once finished; a run that fails or
    is interrupted leaves it where it is.
    """
    for attempt in range(1, attempts + 1):
        candidate = partial_name(destination, attempt)
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate
    raise FileExistsError(
        errno.EEXIST, f"no free partial name in {attempts} attempts", os.fspath(destination)
    )
