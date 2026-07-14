"""Failure propagation and bounded cancellation for the native writer."""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace

import pytest

from kinovsr.native import writer as writer_module
from kinovsr.native.writer import AVWriter

pytestmark = pytest.mark.unit


class _NativeWriter:
    def __init__(self, *, complete: bool = False) -> None:
        self.complete = complete
        self.status_value = 1
        self.cancel_count = 0
        self.error_value = "native-error"

    def status(self):
        return self.status_value

    def error(self):
        return self.error_value

    def endSessionAtSourceTime_(self, _end):
        return None

    def finishWritingWithCompletionHandler_(self, callback):
        if self.complete:
            self.status_value = 2
            callback()

    def cancelWriting(self):
        self.cancel_count += 1
        self.status_value = 4


class _VideoInput:
    def __init__(self) -> None:
        self.finished = 0

    def markAsFinished(self):
        self.finished += 1


def _writer(*, native: _NativeWriter | None = None) -> AVWriter:
    result = AVWriter.__new__(AVWriter)
    result._state_lock = threading.RLock()
    result._state = "writing"
    result._failure = None
    result._finish_done = threading.Event()
    result._native_cancelled = False
    result._audio_done = threading.Event()
    result._audio_done.set()
    result._audio_progress = [0]
    result._explicit_end_ticks = 1
    result._yuv_feed = False
    result.writer = native or _NativeWriter()
    result.video_input = _VideoInput()
    result.audio_input = None
    result.audio_track = None
    result.frame_count = 0
    result.fps = 25.0
    result.label = "test"
    return result


def _patch_native_context(monkeypatch) -> None:
    monkeypatch.setattr(
        writer_module, "autorelease_pool", contextlib.nullcontext)
    monkeypatch.setattr(
        writer_module,
        "CoreMedia",
        SimpleNamespace(CMTimeMake=lambda value, scale: (value, scale)),
    )


def test_audio_callback_failure_reaches_finish_caller(monkeypatch):
    native = _NativeWriter()
    result = _writer(native=native)
    result._audio_done.clear()

    class _AudioInput:
        def __init__(self) -> None:
            self.stopped = 0

        def isReadyForMoreMediaData(self):
            return True

        def appendSampleBuffer_(self, _buffer):
            return False

        def stopRequestingMediaData(self):
            self.stopped += 1

    class _Track:
        n_samples = 16

        @staticmethod
        def make_sample_buffer(start, end):
            return (start, end)

    result.audio_input = _AudioInput()
    result.audio_track = _Track()
    result._pump_audio(n_samples=16, chunk_frames=4)

    assert result._audio_done.is_set()
    with pytest.raises(
            RuntimeError,
            match=r"audio appendSampleBuffer failed at 0: status=1.*native-error"):
        result.finish()
    assert native.cancel_count == 1
    assert result.audio_input.stopped == 1


def test_missing_finish_callback_times_out_and_cancels(monkeypatch):
    _patch_native_context(monkeypatch)
    native = _NativeWriter(complete=False)
    result = _writer(native=native)
    result.FINISH_TIMEOUT_S = 0.001

    with pytest.raises(RuntimeError, match="finish callback did not arrive"):
        result.finish()
    assert native.cancel_count == 1
    assert result._state == "cancelled"

    result.cancel()
    assert native.cancel_count == 1


def test_external_cancel_wakes_in_flight_finish(monkeypatch):
    _patch_native_context(monkeypatch)
    finish_started = threading.Event()

    class _BlockingNativeWriter(_NativeWriter):
        def finishWritingWithCompletionHandler_(self, _callback):
            finish_started.set()

    native = _BlockingNativeWriter()
    result = _writer(native=native)
    result.FINISH_TIMEOUT_S = 5.0
    failures = []

    def finish():
        try:
            result.finish()
        except BaseException as exc:  # asserted on the parent thread
            failures.append(exc)

    thread = threading.Thread(target=finish)
    thread.start()
    assert finish_started.wait(timeout=1.0)
    result.cancel()
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert "status=4" in str(failures[0])


def test_concurrent_finish_timeout_cancels_first_finisher(monkeypatch):
    _patch_native_context(monkeypatch)
    finish_started = threading.Event()

    class _BlockingNativeWriter(_NativeWriter):
        def finishWritingWithCompletionHandler_(self, _callback):
            finish_started.set()

    native = _BlockingNativeWriter()
    result = _writer(native=native)
    result.FINISH_TIMEOUT_S = 5.0
    first_failures = []

    def finish_first():
        try:
            result.finish()
        except BaseException as exc:
            first_failures.append(exc)

    thread = threading.Thread(target=finish_first)
    thread.start()
    assert finish_started.wait(timeout=1.0)
    result.FINISH_TIMEOUT_S = 0.001

    with pytest.raises(RuntimeError, match="concurrent finish"):
        result.finish()
    thread.join(timeout=0.5)

    assert not thread.is_alive()
    assert len(first_failures) == 1
    assert native.cancel_count == 1


def test_finish_is_idempotent_after_native_completion(monkeypatch):
    _patch_native_context(monkeypatch)
    native = _NativeWriter(complete=True)
    result = _writer(native=native)

    result.finish()
    result.finish()

    assert result.video_input.finished == 1
    assert result._state == "finished"
    assert native.cancel_count == 0


def test_cancelled_audio_callback_does_not_touch_track():
    native = _NativeWriter()
    result = _writer(native=native)
    result._audio_done.clear()
    touched = []

    class _AudioInput:
        def isReadyForMoreMediaData(self):
            return True

        def stopRequestingMediaData(self):
            return None

    class _Track:
        n_samples = 4

        @staticmethod
        def make_sample_buffer(start, end):
            touched.append((start, end))

    result.audio_input = _AudioInput()
    result.audio_track = _Track()
    result.cancel()
    result._pump_audio(n_samples=4, chunk_frames=4)

    assert touched == []
    assert result._audio_done.is_set()
