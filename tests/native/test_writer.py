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
        self.cancelled = threading.Event()
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
        self.cancelled.set()


class _VideoInput:
    def __init__(self) -> None:
        self.finished = 0

    def markAsFinished(self):
        self.finished += 1


def _writer(*, native: _NativeWriter | None = None) -> AVWriter:
    result = AVWriter.__new__(AVWriter)
    result._state_lock = threading.RLock()
    result._audio_pump_lock = threading.Lock()
    result._video_append_lock = threading.Lock()
    result._native_mutations = set()
    result._state = "writing"
    result._failure = None
    result._finish_done = threading.Event()
    result._native_cancelled = False
    result._native_cancel_in_progress = False
    result._native_finished = False
    result._native_finish_done = None
    result._cancel_attempt = None
    result._audio_callbacks_inflight = 0
    result._audio_callbacks_done = threading.Event()
    result._audio_callbacks_done.set()
    result._audio_track_close_pending = False
    result._audio_track_closed = False
    result._audio_track_closing = False
    result._audio_close_attempt = None
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


def test_direct_mlx_rgb_preserves_rounding_and_append_order(monkeypatch):
    import mlx.core as mx

    result = _writer()
    result._yuv_feed = True
    result.accepts_mlx_rgb = True
    result._yuv_matrix = "matrix"
    result._yuv_full = False
    events = []
    pool = object()
    yuv_buffer = object()
    converted = []

    class _Adaptor:
        @staticmethod
        def pixelBufferPool():
            return pool

        @staticmethod
        def appendPixelBuffer_withPresentationTime_(buffer, pts):
            events.append("append")
            assert buffer is yuv_buffer
            assert pts == (7, 24000)
            return True

    result.adaptor = _Adaptor()
    result._wait_for_ready = lambda _input, _label: events.append("ready")
    _patch_native_context(monkeypatch)

    def pull(actual_pool):
        events.append("pool")
        return yuv_buffer if actual_pool is pool else None

    def convert(rgb, dst, matrix, full):
        events.append("convert")
        converted.append((rgb, dst, matrix, full))

    monkeypatch.setattr(writer_module._pb, "pool_create_buffer", pull)
    monkeypatch.setattr(
        writer_module._pb, "read_rgbahalf_rgb",
        lambda _buffer: pytest.fail("direct MLX input read RGBAHalf"))
    monkeypatch.setattr(writer_module._yuv, "rgb_to_yuv422_10", convert)

    rgb = mx.array([[[0.12345, 0.5, 0.98765]]], dtype=mx.float32)
    result.append_mlx_rgb(rgb, pts_ticks=7, duration_ticks=3)

    assert events == ["ready", "pool", "convert", "append"]
    assert len(converted) == 1
    converted_rgb, converted_buffer, matrix, full = converted[0]
    assert converted_rgb.dtype == mx.float32
    assert mx.array_equal(
        converted_rgb, rgb.astype(mx.float16).astype(mx.float32)).item()
    assert (converted_buffer, matrix, full) == (yuv_buffer, "matrix", False)
    assert result._explicit_end_ticks == 10
    assert result.frame_count == 1


def test_direct_mlx_conversion_failure_poison_writer_and_finish(monkeypatch):
    import mlx.core as mx

    native = _NativeWriter()
    result = _writer(native=native)
    result._yuv_feed = True
    result.accepts_mlx_rgb = True
    result._yuv_matrix = "matrix"
    result._yuv_full = False
    result._wait_for_ready = lambda _input, _label: None
    result.adaptor = type("_Adaptor", (), {
        "pixelBufferPool": staticmethod(object),
    })()
    _patch_native_context(monkeypatch)
    monkeypatch.setattr(
        writer_module._pb, "pool_create_buffer", lambda _pool: object())
    monkeypatch.setattr(
        writer_module._yuv, "rgb_to_yuv422_10",
        lambda *_args: (_ for _ in ()).throw(ValueError("conversion failed")))

    with pytest.raises(ValueError, match="conversion failed"):
        result.append_mlx_rgb(mx.zeros((1, 2, 3)))

    assert result._state == "failed"
    with pytest.raises(ValueError, match="conversion failed"):
        result.finish()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1


def test_cancel_during_video_conversion_prevents_native_append(monkeypatch):
    result = _writer()
    conversion_started = threading.Event()
    release_conversion = threading.Event()
    appended = []

    class _Adaptor:
        @staticmethod
        def appendPixelBuffer_withPresentationTime_(buffer, pts):
            appended.append((buffer, pts))
            return True

    result._yuv_feed = True
    result.adaptor = _Adaptor()
    result._wait_for_ready = lambda _input, _label: None
    monkeypatch.setattr(
        writer_module._pb,
        "read_rgbahalf_rgb",
        lambda _payload: object(),
    )

    def convert(_rgb):
        conversion_started.set()
        assert release_conversion.wait(timeout=2)
        return object()

    result._rgb_to_yuv_buffer = convert
    failures = []

    def append():
        try:
            result.append(object())
        except BaseException as exc:
            failures.append(exc)

    append_thread = threading.Thread(target=append)
    append_thread.start()
    assert conversion_started.wait(timeout=1)

    result.cancel()
    release_conversion.set()
    append_thread.join(timeout=1)

    assert not append_thread.is_alive()
    assert appended == []
    assert len(failures) == 1
    assert "cannot append while writer is cancelled" in str(failures[0])


def test_video_append_interrupt_poisons_writer_before_pts_can_repeat(
    monkeypatch,
):
    result = _writer()
    native_calls = []

    class _Adaptor:
        @staticmethod
        def appendPixelBuffer_withPresentationTime_(_buffer, pts):
            native_calls.append(pts)
            raise KeyboardInterrupt("interrupted after native side effect")

    result.adaptor = _Adaptor()
    result._wait_for_ready = lambda _input, _label: None
    _patch_native_context(monkeypatch)

    with pytest.raises(KeyboardInterrupt, match="native side effect"):
        result.append(object())

    assert result._state == "failed"
    assert result.frame_count == 0
    assert len(native_calls) == 1

    with pytest.raises(KeyboardInterrupt, match="native side effect"):
        result.append(object())
    assert len(native_calls) == 1


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

        @staticmethod
        def close():
            return None

    result.audio_input = _AudioInput()
    result.audio_track = _Track()
    result._pump_audio(n_samples=16, chunk_frames=4)

    assert result._audio_done.is_set()
    with pytest.raises(
            RuntimeError,
            match=r"audio appendSampleBuffer failed at 0: status=1.*native-error"):
        result.finish()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1
    assert result.audio_input.stopped == 0


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
    assert result._state == "cancelled"
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1

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
    assert "cancelled during native finish callback" in str(failures[0])


def test_cancel_returns_while_native_finish_registration_is_blocked(
    monkeypatch,
):
    _patch_native_context(monkeypatch)
    registration_entered = threading.Event()
    release_registration = threading.Event()

    class _BlockingRegistrationWriter(_NativeWriter):
        def finishWritingWithCompletionHandler_(self, _callback):
            registration_entered.set()
            assert release_registration.wait(timeout=2)

    native = _BlockingRegistrationWriter()
    result = _writer(native=native)
    failures = []

    def finish():
        try:
            result.finish()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=finish)
    thread.start()
    assert registration_entered.wait(timeout=1)

    started = writer_module.time.monotonic()
    result.cancel()
    elapsed = writer_module.time.monotonic() - started

    assert elapsed < 0.2
    release_registration.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert result._state == "cancelled"
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1


def test_cancel_waits_for_admitted_mutation_before_native_cancel():
    entered = threading.Event()
    release = threading.Event()
    order = []

    class _ObservedWriter(_NativeWriter):
        def cancelWriting(self):
            order.append("cancel")
            super().cancelWriting()

    native = _ObservedWriter()
    result = _writer(native=native)
    returned = []

    def mutation():
        order.append("mutation-enter")
        entered.set()
        assert release.wait(timeout=2)
        order.append("mutation-exit")

    thread = threading.Thread(
        target=lambda: returned.append(result._run_native_mutation(
            mutation,
            states=("writing",),
        )),
    )
    thread.start()
    assert entered.wait(timeout=1)

    started = writer_module.time.monotonic()
    result.cancel()
    elapsed = writer_module.time.monotonic() - started

    assert elapsed < 0.2
    assert native.cancel_count == 0
    release.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert returned == [(True, None)]
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert order == ["mutation-enter", "mutation-exit", "cancel"]


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
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1


def test_concurrent_finish_native_failure_cancels_blocked_finisher(
    monkeypatch,
):
    _patch_native_context(monkeypatch)
    mark_entered = threading.Event()
    release_mark = threading.Event()
    native = _NativeWriter()
    result = _writer(native=native)
    result.FINISH_TIMEOUT_S = 5.0

    class _BlockingVideoInput:
        @staticmethod
        def markAsFinished():
            mark_entered.set()
            assert release_mark.wait(timeout=2)

    result.video_input = _BlockingVideoInput()
    first_failures = []

    def finish_first():
        try:
            result.finish()
        except BaseException as exc:
            first_failures.append(exc)

    thread = threading.Thread(target=finish_first)
    thread.start()
    assert mark_entered.wait(timeout=1)
    native.status_value = 3

    with pytest.raises(RuntimeError, match="status=3.*concurrent finish"):
        result.finish()

    assert native.cancel_count == 0
    assert thread.is_alive()
    release_mark.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(first_failures) == 1
    assert "status=3" in str(first_failures[0])
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1


def test_finish_delivers_first_failure_when_native_call_raises_later(
    monkeypatch,
):
    _patch_native_context(monkeypatch)
    first = ValueError("first audio failure")
    result = None

    class _TwoFailureWriter(_NativeWriter):
        def endSessionAtSourceTime_(self, _end):
            assert result is not None
            result._record_failure(first)
            raise RuntimeError("later end failure")

    native = _TwoFailureWriter()
    result = _writer(native=native)

    with pytest.raises(ValueError, match="first audio failure") as caught:
        result.finish()

    assert caught.value is first
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "later end failure" in str(caught.value.__cause__)
    assert result._wait_for_cancel_cleanup(timeout=1.0)
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


def test_finished_state_waits_for_terminal_track_cleanup(monkeypatch):
    _patch_native_context(monkeypatch)
    native = _NativeWriter(complete=True)
    result = _writer(native=native)
    close_entered = threading.Event()
    release_close = threading.Event()

    class _Track:
        n_samples = 0

        @staticmethod
        def close():
            close_entered.set()
            assert release_close.wait(timeout=2)

    result.audio_track = _Track()
    first_failures = []
    second_failures = []
    second_done = threading.Event()

    def finish(errors, done=None):
        try:
            result.finish()
        except BaseException as exc:
            errors.append(exc)
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=finish, args=(first_failures,))
    first.start()
    assert close_entered.wait(timeout=1)
    with result._state_lock:
        assert result._state == "closing"

    second = threading.Thread(
        target=finish,
        args=(second_failures, second_done),
    )
    second.start()
    assert not second_done.wait(timeout=0.05)

    release_close.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert first_failures == []
    assert second_failures == []
    assert result._state == "finished"


def test_finish_retries_a_transient_audio_track_close_failure(monkeypatch):
    _patch_native_context(monkeypatch)
    native = _NativeWriter(complete=True)
    result = _writer(native=native)

    class _Track:
        n_samples = 0

        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("close failed")

    track = _Track()
    result.audio_track = track

    result.finish()

    assert result._state == "finished"
    assert track.close_calls == 2
    assert result._audio_track_closed
    assert not result._audio_track_close_pending

    result.finish()
    assert track.close_calls == 2


def test_finish_preserves_primary_failure_over_cancel_interrupt():
    class _InterruptingCancelWriter(_NativeWriter):
        def cancelWriting(self):
            self.cancel_count += 1
            raise KeyboardInterrupt("cleanup interrupted")

    native = _InterruptingCancelWriter()
    result = _writer(native=native)
    primary = ValueError("audio failed")
    result._record_failure(primary)

    with pytest.raises(ValueError, match="audio failed") as caught:
        result.finish()

    assert caught.value is primary
    assert caught.value.__cause__ is None
    with pytest.raises(KeyboardInterrupt, match="cleanup interrupted"):
        result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1
    assert any(
        "native writer cancellation failed" in note
        for note in getattr(primary, "__notes__", [])
    )


def test_later_boundary_failure_cannot_cycle_or_replace_primary_cause():
    root = OSError("root cause")
    primary = ValueError("first writer failure")
    primary.__cause__ = root
    result = _writer()
    result._record_failure(primary)
    later_failure = None

    try:
        raise RuntimeError("later boundary failure") from primary
    except RuntimeError as later:
        later_failure = later
        with pytest.raises(ValueError, match="first writer failure") as caught:
            result._raise_after_cancel(later)

    assert caught.value is primary
    assert primary.__cause__ is root
    assert primary.__context__ is None
    assert later_failure.__cause__ is primary
    assert any(
        "later boundary failure" in note
        for note in getattr(primary, "__notes__", [])
    )
    assert result._wait_for_cancel_cleanup(timeout=1.0)


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
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert closed == [True]
    assert result._audio_done.is_set()


def test_cancel_after_callback_admission_prevents_track_read_and_append():
    result = _writer()
    result._audio_done.clear()
    readiness_entered = threading.Event()
    release_readiness = threading.Event()
    reads = []
    appends = []
    closes = []

    class _AudioInput:
        def isReadyForMoreMediaData(self):
            readiness_entered.set()
            assert release_readiness.wait(timeout=2)
            return True

        def appendSampleBuffer_(self, sample):
            appends.append(sample)
            return True

        @staticmethod
        def stopRequestingMediaData():
            raise AssertionError("cancel must not wait on callback stop")

    class _Track:
        n_samples = 4

        @staticmethod
        def make_sample_buffer(start, end):
            reads.append((start, end))
            return start, end

        @staticmethod
        def close():
            closes.append(True)
            if len(closes) == 1:
                raise RuntimeError("transient close failure")

    result.audio_input = _AudioInput()
    result.audio_track = _Track()
    callback = threading.Thread(
        target=result._pump_audio,
        kwargs={"n_samples": 4, "chunk_frames": 4},
    )
    callback.start()
    assert readiness_entered.wait(timeout=1)

    result.cancel()
    release_readiness.set()
    callback.join(timeout=1)

    assert not callback.is_alive()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert reads == []
    assert appends == []
    assert closes == [True, True]
    assert result._failure is None
    assert result._audio_callbacks_inflight == 0


def test_cancel_revokes_audio_finish_not_yet_started():
    result = _writer()
    result._audio_done.clear()
    readiness_entered = threading.Event()
    release_readiness = threading.Event()
    marks = []

    class _AudioInput:
        @staticmethod
        def isReadyForMoreMediaData():
            readiness_entered.set()
            assert release_readiness.wait(timeout=2)
            return True

        @staticmethod
        def markAsFinished():
            marks.append(True)

    class _Track:
        n_samples = 0

        @staticmethod
        def close():
            return None

    result.audio_input = _AudioInput()
    result.audio_track = _Track()
    callback = threading.Thread(
        target=result._pump_audio,
        kwargs={"n_samples": 0, "chunk_frames": 4},
    )
    callback.start()
    assert readiness_entered.wait(timeout=1)

    result.cancel()
    release_readiness.set()
    callback.join(timeout=1)

    assert not callback.is_alive()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert marks == []
    assert result._failure is None


def test_cancel_returns_while_audio_read_is_blocked_and_defers_close():
    result = _writer()
    result._audio_done.clear()
    read_entered = threading.Event()
    release_read = threading.Event()
    appends = []
    closes = []

    class _AudioInput:
        @staticmethod
        def isReadyForMoreMediaData():
            return True

        @staticmethod
        def appendSampleBuffer_(sample):
            appends.append(sample)
            return True

        @staticmethod
        def stopRequestingMediaData():
            raise AssertionError("cancel must not wait on callback stop")

    class _Track:
        n_samples = 4

        @staticmethod
        def make_sample_buffer(start, end):
            read_entered.set()
            assert release_read.wait(timeout=2)
            return start, end

        @staticmethod
        def close():
            closes.append(True)

    result.audio_input = _AudioInput()
    result.audio_track = _Track()
    callback = threading.Thread(
        target=result._pump_audio,
        kwargs={"n_samples": 4, "chunk_frames": 4},
    )
    callback.start()
    assert read_entered.wait(timeout=1)

    started = writer_module.time.monotonic()
    result.cancel()
    elapsed = writer_module.time.monotonic() - started

    assert elapsed < 0.2
    assert closes == []
    release_read.set()
    callback.join(timeout=1)

    assert not callback.is_alive()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert appends == []
    assert closes == [True]


def test_cancel_returns_while_audio_append_is_blocked(monkeypatch):
    result = _writer()
    result._audio_done.clear()
    append_entered = threading.Event()
    release_append = threading.Event()
    closes = []

    class _AudioInput:
        @staticmethod
        def isReadyForMoreMediaData():
            return True

        @staticmethod
        def appendSampleBuffer_(_sample):
            append_entered.set()
            assert release_append.wait(timeout=2)
            return True

        @staticmethod
        def stopRequestingMediaData():
            return None

    class _Track:
        n_samples = 4

        @staticmethod
        def make_sample_buffer(start, end):
            return start, end

        @staticmethod
        def close():
            closes.append(True)

    monkeypatch.setattr(
        writer_module,
        "CoreMedia",
        SimpleNamespace(
            CMSampleBufferGetNumSamples=lambda sample: sample[1] - sample[0],
        ),
    )
    result.audio_input = _AudioInput()
    result.audio_track = _Track()
    callback = threading.Thread(
        target=result._pump_audio,
        kwargs={"n_samples": 4, "chunk_frames": 4},
    )
    callback.start()
    assert append_entered.wait(timeout=1)

    started = writer_module.time.monotonic()
    result.cancel()
    elapsed = writer_module.time.monotonic() - started

    assert elapsed < 0.2
    assert closes == []
    assert result.writer.cancel_count == 0
    release_append.set()
    callback.join(timeout=1)

    assert not callback.is_alive()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert result._audio_progress == [0]
    assert closes == [True]


def test_cancel_returns_while_video_append_is_blocked(monkeypatch):
    result = _writer()
    append_entered = threading.Event()
    release_append = threading.Event()
    failures = []

    class _Adaptor:
        @staticmethod
        def appendPixelBuffer_withPresentationTime_(_buffer, _pts):
            append_entered.set()
            assert release_append.wait(timeout=2)
            return True

    result.adaptor = _Adaptor()
    result._wait_for_ready = lambda _input, _label: None
    _patch_native_context(monkeypatch)

    def append():
        try:
            result.append(object())
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=append)
    thread.start()
    assert append_entered.wait(timeout=1)

    started = writer_module.time.monotonic()
    result.cancel()
    elapsed = writer_module.time.monotonic() - started

    assert elapsed < 0.2
    assert result.writer.cancel_count == 0
    release_append.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert result.frame_count == 0
    assert len(failures) == 1
    assert "cannot append while writer is cancelled" in str(failures[0])


def test_finish_reports_native_failure_before_waiting_for_audio(
    monkeypatch,
):
    _patch_native_context(monkeypatch)
    native = _NativeWriter()
    native.status_value = 3
    result = _writer(native=native)
    result.AUDIO_TIMEOUT_S = 5.0
    result._audio_done.clear()

    class _AudioInput:
        @staticmethod
        def stopRequestingMediaData():
            return None

    class _Track:
        n_samples = 4

        @staticmethod
        def close():
            return None

    result.audio_input = _AudioInput()
    result.audio_track = _Track()

    started = writer_module.time.monotonic()
    with pytest.raises(RuntimeError, match="status=3.*video input finish"):
        result.finish()
    elapsed = writer_module.time.monotonic() - started

    assert elapsed < 0.2
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1


def test_event_wait_rechecks_native_status_after_each_poll():
    native = _NativeWriter()
    result = _writer(native=native)
    waits = []

    class _Event:
        @staticmethod
        def is_set():
            return False

        @staticmethod
        def wait(timeout):
            waits.append(timeout)
            native.status_value = 3
            return False

    with pytest.raises(RuntimeError, match="status=3.*polling test"):
        result._wait_for_event(
            _Event(),
            timeout=5.0,
            what="polling test",
        )

    assert waits == [result.STATUS_POLL_S]


def test_native_mutation_token_retires_on_base_exception():
    result = _writer()

    def interrupt():
        raise KeyboardInterrupt("native mutation interrupted")

    with pytest.raises(KeyboardInterrupt, match="native mutation interrupted"):
        result._run_native_mutation(interrupt, states=("writing",))

    assert result._native_mutations == set()


def test_cancel_recovers_interrupted_native_token_retirement():
    class _InterruptingSet(set):
        interrupted = False

        def discard(self, value):
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt("retirement interrupted")
            super().discard(value)

    native = _NativeWriter()
    result = _writer(native=native)
    result._native_mutations = _InterruptingSet()

    with pytest.raises(KeyboardInterrupt, match="retirement interrupted"):
        result._run_native_mutation(lambda: None, states=("writing",))

    assert len(result._native_mutations) == 1
    result.cancel()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert result._native_mutations == set()
    assert native.cancel_count == 1


def test_native_finish_wait_reports_failure_without_missing_callback_timeout(
    monkeypatch,
):
    _patch_native_context(monkeypatch)

    class _FailingNativeWriter(_NativeWriter):
        def finishWritingWithCompletionHandler_(self, _callback):
            self.status_value = 3

    native = _FailingNativeWriter()
    result = _writer(native=native)
    result.FINISH_TIMEOUT_S = 5.0

    started = writer_module.time.monotonic()
    with pytest.raises(RuntimeError, match="status=3.*native finish callback"):
        result.finish()
    elapsed = writer_module.time.monotonic() - started

    assert elapsed < 0.2
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1


def test_native_cancel_failure_remains_retryable():
    class _FlakyCancelWriter(_NativeWriter):
        def cancelWriting(self):
            self.cancel_count += 1
            if self.cancel_count == 1:
                raise RuntimeError("cancel failed")
            self.status_value = 4

    native = _FlakyCancelWriter()
    result = _writer(native=native)

    result.cancel()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 2
    assert result._native_cancelled

    result.cancel()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 2


def test_cancel_coordinator_publishes_unexpected_base_exception():
    native = _NativeWriter()
    result = _writer(native=native)
    original_wait = result._wait_for_native_mutations
    interrupted = False

    def interrupt_once():
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("coordinator interrupted")
        original_wait()

    result._wait_for_native_mutations = interrupt_once
    result.cancel()

    with pytest.raises(KeyboardInterrupt, match="coordinator interrupted"):
        result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 0

    result.cancel()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1


def test_native_cancel_retries_preserve_the_first_failure():
    class _PersistentlyFailingWriter(_NativeWriter):
        def cancelWriting(self):
            self.cancel_count += 1
            if self.cancel_count == 1:
                raise RuntimeError("first cancel failure")
            if self.cancel_count == 2:
                raise RuntimeError("second cancel failure")
            self.status_value = 4

    native = _PersistentlyFailingWriter()
    result = _writer(native=native)

    result.cancel()
    with pytest.raises(RuntimeError, match="first cancel failure") as caught:
        result._wait_for_cancel_cleanup(timeout=1.0)

    assert any(
        "second cancel failure" in note
        for note in getattr(caught.value, "__notes__", [])
    )
    assert native.cancel_count == 2

    result.cancel()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 3


def test_audio_close_failure_is_retryable_on_later_cancel():
    result = _writer()

    class _Track:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls <= 2:
                raise RuntimeError(f"close failure {self.close_calls}")

    track = _Track()
    result.audio_track = track

    result.cancel()
    with pytest.raises(RuntimeError, match="close failure 1") as caught:
        result._wait_for_cancel_cleanup(timeout=1.0)
    assert track.close_calls == 2
    assert any(
        "close failure 2" in note
        for note in getattr(caught.value, "__notes__", [])
    )
    assert not result._audio_track_closed

    result.cancel()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert track.close_calls == 3
    assert result._audio_track_closed


def test_audio_close_interrupt_is_retryable_on_later_cancel():
    result = _writer()

    class _Track:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise KeyboardInterrupt("close interrupted")

    track = _Track()
    result.audio_track = track

    result.cancel()
    with pytest.raises(KeyboardInterrupt, match="close interrupted"):
        result._wait_for_cancel_cleanup(timeout=1.0)
    assert track.close_calls == 1

    result.cancel()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert track.close_calls == 2
    assert result._audio_track_closed


def test_blocked_native_cancel_does_not_retain_audio_track():
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    track_closed = threading.Event()

    class _BlockingNativeWriter(_NativeWriter):
        def cancelWriting(self):
            self.cancel_count += 1
            cancel_entered.set()
            assert release_cancel.wait(timeout=2)
            self.status_value = 4

    class _Track:
        @staticmethod
        def close():
            track_closed.set()

    native = _BlockingNativeWriter()
    result = _writer(native=native)
    result.audio_track = _Track()

    result.cancel()

    assert cancel_entered.wait(timeout=1.0)
    assert track_closed.wait(timeout=1.0)
    assert not result._wait_for_cancel_cleanup(timeout=0.01)
    release_cancel.set()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1


def test_background_cleanup_workers_enter_autorelease_pools(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def observed_pool():
        entered.append(threading.current_thread().name)
        yield

    class _Track:
        @staticmethod
        def close():
            return None

    monkeypatch.setattr(writer_module, "autorelease_pool", observed_pool)
    result = _writer()
    result.audio_track = _Track()

    result.cancel()
    assert result._wait_for_cancel_cleanup(timeout=1.0)

    assert any(name.startswith("kinovsr-close-") for name in entered)
    assert any(name.startswith("kinovsr-cancel-") for name in entered)


def test_pool_entry_failure_is_published_after_cleanup_retry(monkeypatch):
    enters = 0

    class _Pool:
        def __enter__(self):
            nonlocal enters
            enters += 1
            if enters == 1:
                raise KeyboardInterrupt("pool entry interrupted")
            return self

        @staticmethod
        def __exit__(_exc_type, _exc, _tb):
            return False

    monkeypatch.setattr(writer_module, "autorelease_pool", _Pool)
    native = _NativeWriter()
    result = _writer(native=native)

    result.cancel()

    with pytest.raises(KeyboardInterrupt, match="pool entry interrupted"):
        result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1
    assert result._native_cancelled


def test_persistent_pool_entry_failure_uses_unpooled_cleanup(monkeypatch):
    entries = 0

    class _Pool:
        def __enter__(self):
            nonlocal entries
            entries += 1
            raise KeyboardInterrupt(f"pool entry {entries} interrupted")

        @staticmethod
        def __exit__(_exc_type, _exc, _tb):
            return False

    monkeypatch.setattr(writer_module, "autorelease_pool", _Pool)
    native = _NativeWriter()
    result = _writer(native=native)

    result.cancel()

    with pytest.raises(KeyboardInterrupt, match="pool entry 1 interrupted") as caught:
        result._wait_for_cancel_cleanup(timeout=1.0)
    assert entries == 2
    assert native.cancel_count == 1
    assert result._native_cancelled
    assert any(
        "pool entry 2 interrupted" in note
        for note in getattr(caught.value, "__notes__", [])
    )


def test_claim_exit_interruption_still_cleans_and_publishes(monkeypatch):
    class _InterruptingExitLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            self._lock.release()
            raise KeyboardInterrupt("claim exit interrupted")

    monkeypatch.setattr(
        writer_module, "autorelease_pool", contextlib.nullcontext)
    generation = writer_module._AsyncAttempt()
    generation.claim_lock = _InterruptingExitLock()
    operations = 0

    def cleanup():
        nonlocal operations
        operations += 1

    writer_module._run_with_autorelease_pool(generation, cleanup)

    assert operations == 1
    assert generation.done.is_set()
    assert isinstance(generation.failure, KeyboardInterrupt)
    assert "claim exit interrupted" in str(generation.failure)


def test_claim_entry_interruption_is_recovered_under_runner_lock(monkeypatch):
    class _InterruptingEntryLock:
        @staticmethod
        def __enter__():
            raise KeyboardInterrupt("claim entry interrupted")

        @staticmethod
        def __exit__(_exc_type, _exc, _tb):
            pytest.fail("failed claim entry must not invoke __exit__")

    monkeypatch.setattr(
        writer_module, "autorelease_pool", contextlib.nullcontext)
    generation = writer_module._AsyncAttempt()
    generation.claim_lock = _InterruptingEntryLock()
    operations = 0

    def cleanup():
        nonlocal operations
        operations += 1

    writer_module._run_with_autorelease_pool(generation, cleanup)

    assert operations == 1
    assert generation.claimed
    assert generation.done.is_set()
    assert isinstance(generation.failure, KeyboardInterrupt)
    assert "claim entry interrupted" in str(generation.failure)


def test_completed_generation_cannot_run_again_without_shared_claim(monkeypatch):
    class _InterruptingEntryLock:
        @staticmethod
        def __enter__():
            raise KeyboardInterrupt("late claim entry interrupted")

        @staticmethod
        def __exit__(_exc_type, _exc, _tb):
            pytest.fail("failed claim entry must not invoke __exit__")

    monkeypatch.setattr(
        writer_module, "autorelease_pool", contextlib.nullcontext)
    generation = writer_module._AsyncAttempt()
    generation.done.set()
    generation.claim_lock = _InterruptingEntryLock()
    operations = 0

    def cleanup():
        nonlocal operations
        operations += 1

    writer_module._run_with_autorelease_pool(generation, cleanup)

    assert operations == 0
    assert not generation.claimed


def test_pool_exit_failure_precedes_generation_publication(monkeypatch):
    exits = 0

    class _Pool:
        @staticmethod
        def __enter__():
            return None

        def __exit__(self, _exc_type, _exc, _tb):
            nonlocal exits
            exits += 1
            if exits == 1:
                raise KeyboardInterrupt("pool exit interrupted")
            return False

    monkeypatch.setattr(writer_module, "autorelease_pool", _Pool)
    native = _NativeWriter()
    result = _writer(native=native)

    result.cancel()

    with pytest.raises(KeyboardInterrupt, match="pool exit interrupted"):
        result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1
    assert result._cancel_attempt.done.is_set()


def test_waiter_recovers_interrupted_final_publication():
    cancel_entered = threading.Event()
    release_cancel = threading.Event()

    class _BlockingNativeWriter(_NativeWriter):
        def cancelWriting(self):
            self.cancel_count += 1
            cancel_entered.set()
            assert release_cancel.wait(timeout=2)
            self.status_value = 4

    class _InterruptingEvent:
        def __init__(self):
            self._event = threading.Event()
            self._interruptions = 0

        def is_set(self):
            return self._event.is_set()

        def wait(self, timeout=None):
            return self._event.wait(timeout=timeout)

        def set(self):
            if self._interruptions < 2:
                self._interruptions += 1
                raise KeyboardInterrupt("publication interrupted")
            self._event.set()

    native = _BlockingNativeWriter()
    result = _writer(native=native)
    result.cancel()
    assert cancel_entered.wait(timeout=1.0)
    generation = result._cancel_attempt
    generation.done = _InterruptingEvent()

    release_cancel.set()
    with pytest.raises(KeyboardInterrupt, match="publication interrupted"):
        result._wait_for_cancel_cleanup(timeout=1.0)
    assert generation.done.is_set()
    assert native.cancel_count == 1

    result.cancel()
    assert result._wait_for_cancel_cleanup(timeout=1.0)


def test_concurrent_cancel_retries_failed_inflight_native_attempt():
    first_entered = threading.Event()
    release_first = threading.Event()

    class _ConcurrentCancelWriter(_NativeWriter):
        def cancelWriting(self):
            self.cancel_count += 1
            if self.cancel_count == 1:
                first_entered.set()
                assert release_first.wait(timeout=2)
                raise RuntimeError("first cancel failed")
            self.status_value = 4

    native = _ConcurrentCancelWriter()
    result = _writer(native=native)
    first_failures = []
    second_failures = []

    def cancel(errors):
        try:
            result.cancel()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=cancel, args=(first_failures,))
    second = threading.Thread(target=cancel, args=(second_failures,))
    first.start()
    assert first_entered.wait(timeout=1)
    second.start()
    second.join(timeout=0.2)

    assert not second.is_alive()
    assert second_failures == []

    release_first.set()
    first.join(timeout=1)

    assert not first.is_alive()
    assert first_failures == []
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 2
    assert result._native_cancelled


def test_concurrent_cancel_does_not_wait_behind_track_close():
    result = _writer()
    close_entered = threading.Event()
    release_close = threading.Event()

    class _Track:
        @staticmethod
        def close():
            close_entered.set()
            assert release_close.wait(timeout=2)

    result.audio_track = _Track()
    first_failures = []
    second_failures = []

    def cancel(errors):
        try:
            result.cancel()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=cancel, args=(first_failures,))
    first.start()
    assert close_entered.wait(timeout=1)

    second = threading.Thread(target=cancel, args=(second_failures,))
    second.start()
    second.join(timeout=0.2)

    assert not second.is_alive()
    assert second_failures == []

    release_close.set()
    first.join(timeout=1)
    assert not first.is_alive()
    assert first_failures == []
    assert result._wait_for_cancel_cleanup(timeout=1.0)


def test_cancel_start_interruption_after_launch_keeps_live_owner(monkeypatch):
    cancel_entered = threading.Event()
    release_cancel = threading.Event()

    class _BlockingNativeWriter(_NativeWriter):
        def cancelWriting(self):
            self.cancel_count += 1
            cancel_entered.set()
            assert release_cancel.wait(timeout=2)
            self.status_value = 4

    real_thread = threading.Thread

    class _InterruptAfterLaunch(real_thread):
        interrupted = False

        def start(self):
            super().start()
            if not self.interrupted:
                type(self).interrupted = True
                raise KeyboardInterrupt("start interrupted after launch")

    monkeypatch.setattr(writer_module.threading, "Thread", _InterruptAfterLaunch)
    native = _BlockingNativeWriter()
    result = _writer(native=native)

    with pytest.raises(KeyboardInterrupt, match="after launch"):
        result.cancel()

    assert cancel_entered.wait(timeout=1.0)
    assert not result._wait_for_cancel_cleanup(timeout=0)
    result.cancel()
    assert native.cancel_count == 1

    release_cancel.set()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1


def test_cancel_does_not_recover_live_owner_delayed_before_claim(monkeypatch):
    target_entered = threading.Event()
    release_target = threading.Event()
    real_thread = threading.Thread

    class _DelayedTarget(real_thread):
        def run(self):
            target_entered.set()
            assert release_target.wait(timeout=2)
            super().run()

    monkeypatch.setattr(writer_module.threading, "Thread", _DelayedTarget)
    native = _NativeWriter()
    result = _writer(native=native)

    result.cancel()
    assert target_entered.wait(timeout=1.0)
    assert not result._wait_for_cancel_cleanup(timeout=0.05)
    assert not result._cancel_attempt.claimed
    assert native.cancel_count == 0

    release_target.set()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1


def test_ambiguous_thread_launch_and_fallback_claim_once(monkeypatch):
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    count_lock = threading.Lock()
    active = 0
    max_active = 0

    class _BlockingNativeWriter(_NativeWriter):
        def cancelWriting(self):
            nonlocal active, max_active
            with count_lock:
                self.cancel_count += 1
                active += 1
                max_active = max(max_active, active)
            cancel_entered.set()
            try:
                assert release_cancel.wait(timeout=2)
                self.status_value = 4
            finally:
                with count_lock:
                    active -= 1

    real_thread = threading.Thread

    class _LaunchWithoutMetadata(real_thread):
        def start(self):
            real_thread(target=self._target, args=self._args).start()
            raise KeyboardInterrupt("launch metadata interrupted")

    monkeypatch.setattr(writer_module.threading, "Thread", _LaunchWithoutMetadata)
    native = _BlockingNativeWriter()
    result = _writer(native=native)

    with pytest.raises(KeyboardInterrupt, match="metadata interrupted"):
        result.cancel()
    assert cancel_entered.wait(timeout=1.0)

    result.cancel()
    assert native.cancel_count == 1
    assert max_active == 1

    release_cancel.set()
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1
    assert max_active == 1


def test_later_cancel_retries_generation_after_all_launchers_fail(monkeypatch):
    class _Track:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    real_thread = threading.Thread

    class _NeverStarts(real_thread):
        starts = 0

        def start(self):
            type(self).starts += 1
            raise KeyboardInterrupt("thread executor state ambiguous")

    def fail_dispatch(*_args):
        raise RuntimeError("dispatch executor unavailable")

    monkeypatch.setattr(writer_module.threading, "Thread", _NeverStarts)
    monkeypatch.setattr(
        writer_module.libdispatch,
        "dispatch_get_global_queue",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        writer_module.libdispatch, "dispatch_async", fail_dispatch)
    native = _NativeWriter()
    track = _Track()
    result = _writer(native=native)
    result.audio_track = track

    with pytest.raises(KeyboardInterrupt, match="executor state ambiguous"):
        result.cancel()
    cancel_generation = result._cancel_attempt
    close_generation = result._audio_close_attempt
    assert _NeverStarts.starts == 4
    assert not result._wait_for_cancel_cleanup(timeout=0.05)
    assert native.cancel_count == 0
    assert track.close_calls == 0

    for _ in range(5):
        with contextlib.suppress(KeyboardInterrupt):
            result.cancel()
    assert _NeverStarts.starts == 8
    assert len(cancel_generation.threads) <= 2
    assert len(close_generation.threads) <= 2

    monkeypatch.setattr(writer_module.threading, "Thread", real_thread)
    result.cancel()

    assert result._cancel_attempt is cancel_generation
    assert result._audio_close_attempt is close_generation
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1
    assert track.close_calls == 1


def test_later_cancel_retries_transient_same_executor_failure(monkeypatch):
    real_thread = threading.Thread

    class _ToggleThread(real_thread):
        available = False
        starts = 0

        def start(self):
            type(self).starts += 1
            if not self.available:
                raise RuntimeError("thread capacity exhausted")
            super().start()

    def fail_dispatch(*_args):
        raise RuntimeError("dispatch capacity exhausted")

    monkeypatch.setattr(writer_module.threading, "Thread", _ToggleThread)
    monkeypatch.setattr(
        writer_module.libdispatch,
        "dispatch_get_global_queue",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        writer_module.libdispatch, "dispatch_async", fail_dispatch)
    result = _writer()

    for _ in range(2):
        with pytest.raises(RuntimeError, match="thread capacity exhausted"):
            result.cancel()
    generation = result._cancel_attempt
    assert _ToggleThread.starts == 4
    assert len(generation.threads) <= 2
    assert not generation.ambiguous_launch
    assert generation.launch_failed

    _ToggleThread.available = True
    result.cancel()

    assert result._cancel_attempt is generation
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert result.writer.cancel_count == 1


def test_relaunched_cancel_generation_preserves_earlier_failure():
    earlier = KeyboardInterrupt("earlier launch failure")
    generation = writer_module._AsyncAttempt()
    generation.failure = earlier
    generation.ambiguous_launch = True
    generation.launch_failed = True
    result = _writer()
    result._cancel_attempt = generation

    result.cancel()

    with pytest.raises(KeyboardInterrupt, match="earlier launch failure"):
        result._wait_for_cancel_cleanup(timeout=1.0)
    assert result._cancel_attempt is generation
    assert generation.failure is earlier
    assert result.writer.cancel_count == 1


def test_relaunched_audio_generation_preserves_earlier_failure():
    class _Track:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    earlier = KeyboardInterrupt("earlier audio launch failure")
    generation = writer_module._AsyncAttempt()
    generation.failure = earlier
    track = _Track()
    result = _writer()

    result._audio_close_worker_main(track, generation)

    assert track.close_calls == 1
    assert generation.failure is earlier


def test_audio_close_start_interruption_does_not_double_close(monkeypatch):
    close_entered = threading.Event()
    release_close = threading.Event()

    class _Track:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            close_entered.set()
            assert release_close.wait(timeout=2)

    real_thread = threading.Thread

    class _InterruptAfterLaunch(real_thread):
        interrupted = False

        def start(self):
            super().start()
            if not self.interrupted:
                type(self).interrupted = True
                raise KeyboardInterrupt("close start interrupted")

    monkeypatch.setattr(writer_module.threading, "Thread", _InterruptAfterLaunch)
    result = _writer()
    track = _Track()
    result.audio_track = track

    with pytest.raises(KeyboardInterrupt, match="close start interrupted"):
        result._ensure_audio_close_worker()
    assert close_entered.wait(timeout=1.0)

    first = result._audio_close_attempt
    assert result._ensure_audio_close_worker() is first
    assert track.close_calls == 1

    release_close.set()
    assert result._wait_for_audio_close(timeout=1.0)
    assert track.close_calls == 1


def test_native_cancel_interrupt_remains_retryable():
    class _InterruptingCancelWriter(_NativeWriter):
        def cancelWriting(self):
            self.cancel_count += 1
            if self.cancel_count == 1:
                raise KeyboardInterrupt
            self.status_value = 4

    native = _InterruptingCancelWriter()
    result = _writer(native=native)

    result.cancel()
    first_generation = result._cancel_attempt

    with pytest.raises(KeyboardInterrupt):
        result._wait_for_cancel_cleanup(timeout=1.0)
    assert native.cancel_count == 1
    assert not result._native_cancelled

    result.cancel()
    second_generation = result._cancel_attempt
    assert second_generation is not first_generation
    assert result._wait_for_cancel_cleanup(timeout=1.0)
    assert isinstance(first_generation.failure, KeyboardInterrupt)
    assert native.cancel_count == 2
    assert result._native_cancelled


def test_constructor_cancels_writer_when_session_start_raises(
    monkeypatch,
    tmp_path,
):
    native = _NativeWriter()

    class _NativeAlloc:
        @staticmethod
        def initWithURL_fileType_error_(_url, _file_type, _error):
            return native, None

    class _NativeClass:
        @staticmethod
        def alloc():
            return _NativeAlloc()

    class _VideoInput:
        def setExpectsMediaDataInRealTime_(self, _value):
            return None

        def setMediaTimeScale_(self, _value):
            return None

    video_input = _VideoInput()

    class _InputClass:
        @staticmethod
        def assetWriterInputWithMediaType_outputSettings_(_kind, _settings):
            return video_input

    class _Adaptor:
        pass

    class _AdaptorClass:
        @staticmethod
        def assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes_(
            _input,
            _attrs,
        ):
            return _Adaptor()

    native.canAddInput_ = lambda _input: True
    native.addInput_ = lambda _input: None
    native.setMovieTimeScale_ = lambda _scale: None
    native.startWriting = lambda: True

    def fail_session(_time):
        raise RuntimeError("session start failed")

    native.startSessionAtSourceTime_ = fail_session
    fake_av = SimpleNamespace(
        AVAssetWriter=_NativeClass,
        AVFileTypeMPEG4="mp4",
        AVAssetWriterInput=_InputClass,
        AVMediaTypeVideo="video",
        AVAssetWriterInputPixelBufferAdaptor=_AdaptorClass,
    )
    fake_foundation = SimpleNamespace(
        NSURL=SimpleNamespace(fileURLWithPath_=lambda path: path),
    )
    monkeypatch.setattr(writer_module, "av", fake_av)
    monkeypatch.setattr(writer_module, "Foundation", fake_foundation)
    monkeypatch.setattr(
        writer_module,
        "hevc_video_settings",
        lambda *_args, **_kwargs: {},
    )
    _patch_native_context(monkeypatch)

    with pytest.raises(RuntimeError, match="session start failed"):
        AVWriter(
            tmp_path / "out.mp4",
            width=16,
            height=16,
            fps=24,
            source_pixel_format=writer_module._pb.PIX_NV12,
        )

    assert native.cancelled.wait(timeout=1.0)
    assert native.cancel_count == 1


def test_constructor_retries_transient_cleanup_before_losing_ownership(
    monkeypatch,
    tmp_path,
):
    class _FlakyNativeWriter(_NativeWriter):
        def cancelWriting(self):
            self.cancel_count += 1
            if self.cancel_count == 1:
                raise RuntimeError("native cancel failed")
            self.status_value = 4
            self.cancelled.set()

    class _FlakyTrack:
        def __init__(self):
            self.close_calls = 0
            self.closed = threading.Event()

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("track close failed")
            self.closed.set()

    native = _FlakyNativeWriter()
    track = _FlakyTrack()

    def fail_construct(self, *_args, **_kwargs):
        self.writer = native
        self.audio_track = track
        raise ValueError("construction failed")

    monkeypatch.setattr(AVWriter, "_construct", fail_construct)
    monkeypatch.setattr(
        writer_module, "autorelease_pool", contextlib.nullcontext)

    with pytest.raises(ValueError, match="construction failed") as caught:
        AVWriter(
            tmp_path / "out.mp4",
            width=16,
            height=16,
            fps=24,
            source_pixel_format=writer_module._pb.PIX_NV12,
        )

    notes = getattr(caught.value, "__notes__", [])
    assert notes == []
    assert native.cancelled.wait(timeout=1.0)
    assert track.closed.wait(timeout=1.0)
    assert native.cancel_count == 2
    assert track.close_calls == 2


def test_constructor_recovers_prelaunch_cleanup_worker_failure(
    monkeypatch,
    tmp_path,
):
    native = _NativeWriter()

    class _Track:
        def __init__(self):
            self.closed = threading.Event()
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            self.closed.set()

    track = _Track()

    def fail_construct(self, *_args, **_kwargs):
        self.writer = native
        self.audio_track = track
        raise ValueError("construction failed")

    real_thread = threading.Thread

    class _FailFirstStart(real_thread):
        starts = 0

        def start(self):
            type(self).starts += 1
            if self.starts == 1:
                raise KeyboardInterrupt("worker did not launch")
            super().start()

    monkeypatch.setattr(AVWriter, "_construct", fail_construct)
    monkeypatch.setattr(
        writer_module, "autorelease_pool", contextlib.nullcontext)
    monkeypatch.setattr(writer_module.threading, "Thread", _FailFirstStart)

    with pytest.raises(ValueError, match="construction failed") as caught:
        AVWriter(
            tmp_path / "out.mp4",
            width=16,
            height=16,
            fps=24,
            source_pixel_format=writer_module._pb.PIX_NV12,
        )

    assert any(
        "worker did not launch" in note
        for note in getattr(caught.value, "__notes__", [])
    )
    assert native.cancelled.wait(timeout=1.0)
    assert track.closed.wait(timeout=1.0)
    assert native.cancel_count == 1
    assert track.close_calls == 1


def test_constructor_uses_dispatch_fallback_when_threads_never_start(
    monkeypatch,
    tmp_path,
):
    native = _NativeWriter()

    class _Track:
        def __init__(self):
            self.closed = threading.Event()

        def close(self):
            self.closed.set()

    track = _Track()

    def fail_construct(self, *_args, **_kwargs):
        self.writer = native
        self.audio_track = track
        raise ValueError("construction failed")

    real_thread = threading.Thread

    class _NeverStarts(real_thread):
        @staticmethod
        def start():
            raise RuntimeError("thread creation unavailable")

    monkeypatch.setattr(AVWriter, "_construct", fail_construct)
    monkeypatch.setattr(
        writer_module, "autorelease_pool", contextlib.nullcontext)
    monkeypatch.setattr(writer_module.threading, "Thread", _NeverStarts)

    with pytest.raises(ValueError, match="construction failed") as caught:
        AVWriter(
            tmp_path / "out.mp4",
            width=16,
            height=16,
            fps=24,
            source_pixel_format=writer_module._pb.PIX_NV12,
        )

    assert any(
        "thread creation unavailable" in note
        for note in getattr(caught.value, "__notes__", [])
    )
    assert native.cancelled.wait(timeout=1.0)
    assert track.closed.wait(timeout=1.0)
    assert native.cancel_count == 1


def test_constructor_cleanup_survives_pool_entry_interruption(
    monkeypatch,
    tmp_path,
):
    native = _NativeWriter()
    entries = 0

    class _Pool:
        def __enter__(self):
            nonlocal entries
            entries += 1
            if entries == 2:
                # Entry one is the constructor's outer pool. Entry two is the
                # cleanup worker; its wrapper must retry with entry three.
                raise KeyboardInterrupt("worker pool entry interrupted")
            return self

        @staticmethod
        def __exit__(_exc_type, _exc, _tb):
            return False

    def fail_construct(self, *_args, **_kwargs):
        self.writer = native
        raise ValueError("construction failed")

    monkeypatch.setattr(AVWriter, "_construct", fail_construct)
    monkeypatch.setattr(writer_module, "autorelease_pool", _Pool)

    with pytest.raises(ValueError, match="construction failed"):
        AVWriter(
            tmp_path / "out.mp4",
            width=16,
            height=16,
            fps=24,
            source_pixel_format=writer_module._pb.PIX_NV12,
        )

    assert native.cancelled.wait(timeout=1.0)
    assert native.cancel_count == 1


def test_constructor_delivers_first_failure_when_setup_raises_later(
    monkeypatch,
    tmp_path,
):
    native = _NativeWriter()
    first = ValueError("first callback failure")

    def fail_construct(self, *_args, **_kwargs):
        self.writer = native
        self._record_failure(first)
        raise RuntimeError("later setup failure")

    monkeypatch.setattr(AVWriter, "_construct", fail_construct)
    monkeypatch.setattr(
        writer_module, "autorelease_pool", contextlib.nullcontext)

    with pytest.raises(ValueError, match="first callback failure") as caught:
        AVWriter(
            tmp_path / "out.mp4",
            width=16,
            height=16,
            fps=24,
            source_pixel_format=writer_module._pb.PIX_NV12,
        )

    assert caught.value is first
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "later setup failure" in str(caught.value.__cause__)
    assert native.cancelled.wait(timeout=1.0)
    assert native.cancel_count == 1
