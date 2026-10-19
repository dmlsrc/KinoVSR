"""Keep Core ML's builds for the tests out of the user's own caches.

Core ML keeps compiled bundles of the models it loads in an E5 cache in the
user's Library and never prunes it. It files a Neural Engine program by the
path the model loaded from (a small CPU-placed model by its content), and most
Neural Engine tests build their models in fresh pytest temporary directories,
so most of what they add is never loaded again: one run of the whole suite
added 138 entries, 268 MiB, and the next run 134 more.

A session therefore points Core ML at ``~/Library/Caches/KinoVSR/tests-home``
through CFFIXED_USER_HOME, which Foundation reads at its first home lookup and
child processes inherit. The home outlives the run, because the loads that do
repeat took the Neural Engine lane from 48.6 s to 43.1 s on the second run,
and its cache is cleared once it passes E5_CACHE_LIMIT. Every session holds a
shared lock on the home while it runs, so only a session that finds no other
one there can clear it. The home stays on APFS: with Core ML's cache on the
tmpfs RAM disk that holds pytest's temporary directories, Core ML placed some
of a model's operations on the CPU.
"""

import fcntl
import os
import shutil
from pathlib import Path

import pytest

E5_CACHE = "com.apple.e5rt.e5bundlecache"
E5_CACHE_LIMIT = 2 << 30
_HOME_LOCK = pytest.StashKey[int]()


def _size(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    # A home chosen outside the session is kept: a sandbox's, or the one a
    # parent session passed to its test subprocesses or xdist workers.
    if "CFFIXED_USER_HOME" in os.environ:
        return
    home = Path.home() / "Library" / "Caches" / "KinoVSR" / "tests-home"
    home.mkdir(parents=True, exist_ok=True)
    lock = os.open(home / "sessions.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass  # another session is running; the cache stays as it is
    else:
        caches = home / "Library" / "Caches"
        if caches.is_dir() and _size(caches) > E5_CACHE_LIMIT:
            for cache in caches.glob(f"*/{E5_CACHE}"):
                shutil.rmtree(cache, ignore_errors=True)
    fcntl.flock(lock, fcntl.LOCK_SH)
    config.stash[_HOME_LOCK] = lock
    os.environ["CFFIXED_USER_HOME"] = str(home)


def pytest_unconfigure(config: pytest.Config) -> None:
    lock = config.stash.get(_HOME_LOCK, None)
    if lock is not None:
        os.environ.pop("CFFIXED_USER_HOME", None)
        os.close(lock)
