"""Failure propagation and bounded cancellation for the native writer."""

from __future__ import annotations

import contextlib
import threading
from fractions import Fraction
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
    result._audio_complete = False
    result._explicit_end_ticks = 1
    result._yuv_feed = False
    result.writer = native or _NativeWriter()
    result.video_input = _VideoInput()
    result.audio_input = None
    result.audio_track = None
    result.frame_count = 0
    result.cadence = Fraction(25)
    result.fps = 25.0
    result.label = "test"
    return result


def test_default_append_uses_the_exact_rational_cadence(monkeypatch):
    """The native encode helper's implicit-PTS path must not multiply a
    rounded one-frame NTSC duration across a long sequence."""
    from kinovsr.media import pixel_buffers

    cadence = Fraction(30000, 1001)
    frame_index = round(3600 * cadence)
    captured = []

    class _Adaptor:
        def appendPixelBuffer_withPresentationTime_(self, _buffer, pts):
            captured.append(pts)
            return True

    result = _writer()
    result.cadence = cadence
    result.fps = float(cadence)
    result.frame_count = frame_index
    result.adaptor = _Adaptor()
    result._wait_for_ready = lambda _input, _label: None
    monkeypatch.setattr(
        writer_module, "autorelease_pool", contextlib.nullcontext)

    result.append(object())

    assert len(captured) == 1
    expected = pixel_buffers.frame_ticks(frame_index, cadence)
    assert captured[0].value == expected
    assert captured[0].timescale == pixel_buffers.VIDEO_TIME_SCALE


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


def test_audio_pump_accepts_a_short_final_pull(monkeypatch):
    result = _writer()
    result._audio_done.clear()
    appended = []

    class _AudioInput:
        finished = 0

        @staticmethod
        def isReadyForMoreMediaData():
            return True

        @staticmethod
        def appendSampleBuffer_(sample):
            appended.append(sample)
            return True

        @classmethod
        def markAsFinished(cls):
            cls.finished += 1

    class _Track:
        n_samples = 4

        @staticmethod
        def make_sample_buffer(start, end):
            if start >= 3:
                return None
            return start, min(end, 3)

    monkeypatch.setattr(
        writer_module,
        "CoreMedia",
        SimpleNamespace(
            CMSampleBufferGetNumSamples=lambda sample: sample[1] - sample[0],
        ),
    )
    result.audio_input = _AudioInput()
    result.audio_track = _Track()

    result._pump_audio(n_samples=4, chunk_frames=4)

    assert appended == [(0, 3)]
    assert result._audio_progress == [3]
    assert result._audio_complete
    assert result._audio_done.is_set()
    assert result.audio_input.finished == 1


def test_audio_pump_does_not_decode_under_backpressure():
    result = _writer()
    result._audio_done.clear()
    pulls = []

    class _AudioInput:
        @staticmethod
        def isReadyForMoreMediaData():
            return False

    class _Track:
        n_samples = 4

        @staticmethod
        def make_sample_buffer(start, end):
            pulls.append((start, end))

    result.audio_input = _AudioInput()
    result.audio_track = _Track()

    result._pump_audio(n_samples=4, chunk_frames=4)

    assert pulls == []
    assert not result._audio_done.is_set()


def test_final_append_marks_audio_finished_before_readiness_drops(monkeypatch):
    result = _writer()
    result._audio_done.clear()

    class _AudioInput:
        def __init__(self):
            self.ready = True
            self.finished = 0

        def isReadyForMoreMediaData(self):
            return self.ready

        def appendSampleBuffer_(self, _sample):
            self.ready = False
            return True

        def markAsFinished(self):
            self.finished += 1

    class _Track:
        n_samples = 4

        @staticmethod
        def make_sample_buffer(start, end):
            return start, end

    monkeypatch.setattr(
        writer_module,
        "CoreMedia",
        SimpleNamespace(
            CMSampleBufferGetNumSamples=lambda sample: sample[1] - sample[0],
        ),
    )
    result.audio_input = _AudioInput()
    result.audio_track = _Track()

    result._pump_audio(n_samples=4, chunk_frames=4)

    assert result._audio_progress == [4]
    assert result.audio_input.finished == 1
    assert result._audio_complete
    assert result._audio_done.is_set()


def test_audio_pump_bounds_a_one_hour_track_to_ready_chunk(monkeypatch):
    result = _writer()
    result._audio_done.clear()
    pulls = []

    class _AudioInput:
        def __init__(self):
            self.ready = True

        def isReadyForMoreMediaData(self):
            return self.ready

        def appendSampleBuffer_(self, _sample):
            self.ready = False
            return True

    class _Track:
        n_samples = 48_000 * 3600

        @staticmethod
        def make_sample_buffer(start, end):
            pulls.append((start, end))
            return start, end

    monkeypatch.setattr(
        writer_module,
        "CoreMedia",
        SimpleNamespace(
            CMSampleBufferGetNumSamples=lambda sample: sample[1] - sample[0],
        ),
    )
    result.audio_input = _AudioInput()
    result.audio_track = _Track()

    result._pump_audio(n_samples=_Track.n_samples, chunk_frames=12_000)

    assert pulls == [(0, 12_000)]
    assert result._audio_progress == [12_000]
    assert not result._audio_complete
    assert not result._audio_done.is_set()


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
    closed = []

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

        @staticmethod
        def close():
            closed.append(True)

    result.audio_input = _AudioInput()
    result.audio_track = _Track()
    result.cancel()
    result._pump_audio(n_samples=4, chunk_frames=4)

    assert touched == []
    assert closed == [True]
    assert result._audio_done.is_set()
