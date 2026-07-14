"""Native VideoToolbox VSR process-global behavior."""

import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_native_stderr_suppression_holds_a_process_global_lock(monkeypatch):
    from kinovsr.native import vsr

    monkeypatch.setattr(
        vsr, "default_settings", lambda: SimpleNamespace(verbose=False))
    monkeypatch.setattr(vsr.os, "dup", lambda _fd: 10)
    monkeypatch.setattr(vsr.os, "open", lambda _path, _flags: 11)
    monkeypatch.setattr(vsr.os, "dup2", lambda _src, _dst: None)
    monkeypatch.setattr(vsr.os, "close", lambda _fd: None)

    entered = threading.Event()
    release = threading.Event()
    errors = []

    def hold_suppression():
        try:
            with vsr._suppress_native_stderr():
                entered.set()
                if not release.wait(timeout=2):
                    raise AssertionError("test did not release suppression")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=hold_suppression)
    thread.start()
    assert entered.wait(timeout=2)
    assert not vsr._NATIVE_STDERR_LOCK.acquire(blocking=False)
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == []

    # The lock is reentrant for nested construction in one thread and is
    # released after the outer context exits.
    with vsr._suppress_native_stderr(), vsr._suppress_native_stderr():
        pass
    assert vsr._NATIVE_STDERR_LOCK.acquire(blocking=False)
    vsr._NATIVE_STDERR_LOCK.release()
