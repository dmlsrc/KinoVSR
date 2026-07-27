"""Native VideoToolbox VSR process-global behavior."""

import json
import logging
import subprocess
import sys
import tempfile
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_uploaded_source_gets_matrix_attachment_after_pixel_upload(monkeypatch):
    from kinovsr.native import vsr

    events = []
    session = object.__new__(vsr.VsrSession)
    session._make_src_buffer = lambda: "source-buffer"
    session._tag_source_matrix = (
        lambda buffer: events.append(("matrix", buffer))
    )
    monkeypatch.setattr(
        vsr._pb,
        "upload_frame_to_buffer",
        lambda frame, buffer: events.append(("upload", frame, buffer)),
    )

    assert session._upload_src_buffer("frame") == "source-buffer"
    assert events == [
        ("upload", "frame", "source-buffer"),
        ("matrix", "source-buffer"),
    ]


def test_source_matrix_attachment_only_sets_matrix(monkeypatch):
    from kinovsr.native import vsr
    from kinovsr.native.frameworks import Quartz

    calls = []
    monkeypatch.setattr(
        Quartz,
        "CVBufferSetAttachment",
        lambda *args: calls.append(args),
    )
    session = object.__new__(vsr.VsrSession)
    session._source_matrix = Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2
    buffer = object()

    session._tag_source_matrix(buffer)

    assert calls == [
        (
            buffer,
            Quartz.kCVImageBufferYCbCrMatrixKey,
            Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2,
            Quartz.kCVAttachmentMode_ShouldPropagate,
        )
    ]


def test_flow_submission_isolates_source_and_uses_sequential_after_reset():
    from kinovsr.native import vsr

    class Executor:
        def __init__(self):
            self.calls = []

        def submit(self, fn, prev, cur, slot, submission_mode, index):
            self.calls.append(
                (fn, prev, cur, slot, submission_mode, index)
            )
            return "future"

    executor = Executor()
    session = object.__new__(vsr.VsrSession)
    session._flow_executor = executor
    session._flow_needs_random = True
    session._isolate_flow_source_frame = lambda frame: f"isolated-{frame}"

    for index in range(4):
        session._start_flow_future(
            f"prev-{index}",
            f"cur-{index}",
            index % 2,
            index,
        )

    random_mode = vsr.vt.VTOpticalFlowParametersSubmissionModeRandom
    sequential_mode = vsr.vt.VTOpticalFlowParametersSubmissionModeSequential
    assert [call[1] for call in executor.calls] == [
        "isolated-prev-0",
        "isolated-prev-1",
        "isolated-prev-2",
        "isolated-prev-3",
    ]
    assert [call[2] for call in executor.calls] == [
        "cur-0",
        "cur-1",
        "cur-2",
        "cur-3",
    ]
    assert [call[4] for call in executor.calls] == [
        random_mode,
        sequential_mode,
        sequential_mode,
        sequential_mode,
    ]


def test_flow_source_isolation_copies_into_bounded_pool(monkeypatch):
    from kinovsr.native import vsr

    events = []

    class SourceFrame:
        def buffer(self):
            return "shared-source"

        def presentationTimeStamp(self):
            return "pts"

    class FrameInit:
        def initWithBuffer_presentationTimeStamp_(self, buffer, pts):
            return ("flow-frame", buffer, pts)

    class FrameClass:
        @staticmethod
        def alloc():
            return FrameInit()

    monkeypatch.setattr(
        vsr._pb,
        "pool_create_buffer_bounded",
        lambda pool, limit: (
            events.append(("acquire", pool, limit))
            or "isolated-source"
        ),
    )
    monkeypatch.setattr(
        vsr._pb,
        "copy_pixel_buffer_into",
        lambda source, destination: events.append(
            ("copy", source, destination)
        ),
    )
    monkeypatch.setattr(
        vsr,
        "vt",
        SimpleNamespace(VTFrameProcessorFrame=FrameClass),
    )

    session = object.__new__(vsr.VsrSession)
    session._flow_src_pool = "flow-source-pool"

    assert session._isolate_flow_source_frame(SourceFrame()) == (
        "flow-frame",
        "isolated-source",
        "pts",
    )
    assert events == [
        (
            "acquire",
            "flow-source-pool",
            vsr.EXPLICIT_FLOW_SOURCE_POOL_LIMIT,
        ),
        ("copy", "shared-source", "isolated-source"),
    ]


def test_vision_vsr_uses_high_backward_flow_and_zero_forward(monkeypatch):
    from kinovsr.native import vision_flow, vsr

    assert vsr.VISION_VSR_ACCURACY == "high"
    calls = []
    conversions = []

    def generate(from_buffer, to_buffer, *, accuracy):
        calls.append((from_buffer, to_buffer, accuracy))
        return "vision-backward"

    class Converter:
        def convert(self, source, destination):
            conversions.append((source, destination))

    class Frame:
        def __init__(self, buffer):
            self._buffer = buffer

        def buffer(self):
            return self._buffer

    monkeypatch.setattr(vision_flow, "generate_vision_flow", generate)
    monkeypatch.setattr(vsr, "autorelease_pool", nullcontext)

    session = object.__new__(vsr.VsrSession)
    session._flow_backend = "vision"
    session._flow_zero_pair = ("zero-forward", "zero-backward")
    session._flow_pairs = (
        ("old-forward-0", "old-backward-0"),
        ("old-forward-1", "old-backward-1"),
    )
    session._vision_flow_converter = Converter()
    session._vision_flow_destinations = (
        "converted-backward-0",
        "converted-backward-1",
    )

    session._run_explicit_flow(
        Frame("previous"),
        Frame("current"),
        slot=1,
        submission_mode="ignored",
        frame_index=7,
    )

    assert calls == [("current", "previous", vsr.VISION_VSR_ACCURACY)]
    assert conversions == [
        ("vision-backward", "converted-backward-1"),
    ]
    assert session._flow_pairs == (
        ("old-forward-0", "old-backward-0"),
        ("zero-forward", "converted-backward-1"),
    )


def test_vision_vsr_log_uses_the_request_accuracy_constant(
        monkeypatch, caplog):
    from kinovsr.native import vision_flow, vsr

    buffers = iter(object() for _ in range(4))
    converter_calls = []

    class Converter:
        def __init__(self, *args, **kwargs):
            converter_calls.append((args, kwargs))

    monkeypatch.setattr(
        vsr._pb,
        "make_pixel_buffer_from_attrs",
        lambda *_args, **_kwargs: next(buffers),
    )
    monkeypatch.setattr(
        vsr.VsrSession,
        "_zero_flow_pair",
        lambda _self, _pair: None,
    )
    monkeypatch.setattr(vision_flow, "VisionFlowToVtConverter", Converter)
    monkeypatch.setattr(vsr, "VISION_VSR_ACCURACY", "medium")

    session = object.__new__(vsr.VsrSession)
    session.in_w = 640
    session.in_h = 480
    try:
        with caplog.at_level(logging.INFO, logger=vsr.__name__):
            session._start_vision_flow()
        assert "Vision revision 1 Medium" in caplog.text
        assert "full estimate 640x480 -> VT grid 160x120" in caplog.text
        assert converter_calls == [
            (
                (640, 480, 160, 120),
                {"rotate_counterclockwise": False},
            )
        ]
    finally:
        session._flow_executor.shutdown()


def test_explicit_flow_finish_waits_processes_and_clears_pending(
        monkeypatch):
    from kinovsr.native import vsr

    events = []

    class Future:
        def result(self):
            events.append("flow")

    session = object.__new__(vsr.VsrSession)
    session._explicit_flow = True
    session._flow_future = Future()
    session._flow_pending_frame = "frame"
    session._flow_pending_index = 7
    session._flow_pending_slot = 1
    session._process_precomputed_frame = (
        lambda frame, index, slot:
        events.append(("vsr", frame, index, slot)) or "output"
    )
    monkeypatch.setattr(vsr, "autorelease_pool", nullcontext)

    assert session.finish_pending_upscale() == "output"
    assert events == ["flow", ("vsr", "frame", 7, 1)]
    assert session._flow_future is None
    assert session._flow_pending_frame is None
    assert session._flow_pending_index is None
    assert session._flow_pending_slot is None


def test_explicit_flow_reset_requires_drain_and_rearms_random_modes():
    from kinovsr.native import vsr

    session = object.__new__(vsr.VsrSession)
    session._flow_pending_frame = object()
    with pytest.raises(RuntimeError, match="finish_pending_upscale"):
        session.reset_temporal_context()

    zeroed = []
    session._flow_pending_frame = None
    session._prev_src_frame = object()
    session._prev_dst_frame = object()
    session._flow_needs_random = False
    session._vsr_needs_random = False
    session._flow_backend = "vt"
    session._flow_pairs = (("forward-0", "backward-0"),)
    session._flow_zero_pair = None
    session._zero_flow_pair = zeroed.append

    session.reset_temporal_context()

    assert session._prev_src_frame is None
    assert session._prev_dst_frame is None
    assert session._flow_needs_random
    assert session._vsr_needs_random
    assert zeroed == [("forward-0", "backward-0")]


def test_explicit_flow_close_waits_before_ending_both_sessions():
    from kinovsr.native import vsr

    events = []

    class Executor:
        def shutdown(self, *, wait):
            events.append(("executor", wait))

    class Processor:
        def __init__(self, name):
            self.name = name

        def endSession(self):
            events.append(self.name)

    class Converter:
        def close(self):
            events.append("converter")

    session = object.__new__(vsr.VsrSession)
    session.processor = Processor("vsr")
    session._flow_processor = Processor("flow")
    session._flow_executor = Executor()
    session._prev_src_frame = object()
    session._prev_dst_frame = object()
    session._flow_future = object()
    session._flow_pending_frame = object()
    session._flow_pending_index = 1
    session._flow_pending_slot = 0
    session._flow_pairs = (("forward", "backward"),)
    session._flow_zero_pair = None
    session._vision_flow_converter = Converter()
    session._vision_flow_destinations = ("converted",)
    session._flow_config = object()
    session._flow_src_pool = None
    session.config = object()
    session._xfer = None
    session._src_pool = None
    session._dst_pool = None
    session._owns_dst_pool = True
    session.flush_pools = lambda: events.append("pools")

    session.close()

    assert events == [
        ("executor", True),
        "converter",
        "flow",
        "vsr",
        "pools",
    ]
    assert session.processor is None
    assert session._flow_processor is None
    assert session._flow_executor is None
    assert session._flow_pairs is None
    assert session._vision_flow_converter is None
    assert session._vision_flow_destinations is None


def test_verbose_native_stderr_path_performs_no_descriptor_operations(
        monkeypatch):
    from kinovsr.native import vsr

    class FailLock:
        def __enter__(self):
            pytest.fail("verbose path acquired stderr lock")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        vsr, "default_settings", lambda: SimpleNamespace(verbose=True))
    monkeypatch.setattr(vsr, "_NATIVE_STDERR_LOCK", FailLock())
    monkeypatch.setattr(
        vsr, "_duplicate_stderr",
        lambda: pytest.fail("verbose path duplicated fd 2"))

    entered = False
    with vsr._suppress_native_stderr():
        entered = True
    assert entered


def test_native_stderr_open_failure_closes_saved_descriptor(monkeypatch):
    from kinovsr.native import vsr

    failure = OSError("injected /dev/null open failure")
    closed = []
    monkeypatch.setattr(
        vsr, "default_settings", lambda: SimpleNamespace(verbose=False))
    monkeypatch.setattr(vsr, "_duplicate_stderr", lambda: 10)
    monkeypatch.setattr(
        vsr, "_open_devnull", lambda: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(
        vsr, "_redirect_stderr", lambda _fd: pytest.fail("unexpected restore"))
    monkeypatch.setattr(vsr, "_close_fd", closed.append)

    with pytest.raises(OSError) as caught, vsr._suppress_native_stderr():
        pytest.fail("body entered after setup failure")

    assert caught.value is failure
    assert closed == [10]


def test_native_stderr_preserves_body_and_orders_cleanup_notes(monkeypatch):
    from kinovsr.native import vsr

    body = RuntimeError("body failed")
    body.add_note("existing note")
    cause = ValueError("existing cause")
    context = LookupError("existing context")
    body.__cause__ = cause
    body.__context__ = context
    restore_failure = OSError("restore failed")
    devnull_close_failure = KeyboardInterrupt("devnull close failed")
    saved_close_failure = SystemExit("saved close failed")
    dup2_calls = []
    close_calls = []

    monkeypatch.setattr(
        vsr, "default_settings", lambda: SimpleNamespace(verbose=False))
    monkeypatch.setattr(vsr, "_duplicate_stderr", lambda: 10)
    monkeypatch.setattr(vsr, "_open_devnull", lambda: 11)

    def redirect(source):
        dup2_calls.append((source, 2))
        if len(dup2_calls) == 2:
            raise restore_failure

    def close(fd):
        close_calls.append(fd)
        if fd == 11:
            raise devnull_close_failure
        raise saved_close_failure

    monkeypatch.setattr(vsr, "_redirect_stderr", redirect)
    monkeypatch.setattr(vsr, "_close_fd", close)

    with pytest.raises(RuntimeError) as caught, vsr._suppress_native_stderr():
        raise body

    assert caught.value is body
    assert caught.value.__cause__ is cause
    assert caught.value.__context__ is context
    assert dup2_calls == [(11, 2), (10, 2)]
    assert close_calls == [11, 10]
    assert body.__notes__ == [
        "existing note",
        "restore stderr also failed: OSError: restore failed",
        "close /dev/null also failed: KeyboardInterrupt: devnull close failed",
        "close saved stderr also failed: SystemExit: saved close failed",
    ]


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_native_stderr_preserves_process_control_body(monkeypatch, failure_type):
    from kinovsr.native import vsr

    failure = failure_type("injected body interruption")
    closed = []
    monkeypatch.setattr(
        vsr, "default_settings", lambda: SimpleNamespace(verbose=False))
    monkeypatch.setattr(vsr, "_duplicate_stderr", lambda: 10)
    monkeypatch.setattr(vsr, "_open_devnull", lambda: 11)
    monkeypatch.setattr(vsr, "_redirect_stderr", lambda _fd: None)
    monkeypatch.setattr(vsr, "_close_fd", closed.append)

    with pytest.raises(
            failure_type) as caught, vsr._suppress_native_stderr():
        raise failure

    assert caught.value is failure
    assert closed == [11, 10]


def test_native_stderr_real_fd_failure_matrix_in_subprocess():
    script = r'''
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

from kinovsr.native import vsr

vsr.default_settings = lambda: SimpleNamespace(verbose=False)
real_dup2 = os.dup2
real_close = os.close
real_open_devnull = vsr._open_devnull
real_redirect = vsr._redirect_stderr
real_close_fd = vsr._close_fd

def fd_set():
    result = []
    for fd in range(3, 256):
        try:
            os.fstat(fd)
        except OSError:
            continue
        result.append(fd)
    return result

open_failure = RuntimeError("open failed")
before_open = fd_set()
vsr._open_devnull = lambda: (_ for _ in ()).throw(open_failure)
try:
    with vsr._suppress_native_stderr():
        raise AssertionError("open failure entered body")
except BaseException as caught:
    open_identity = caught is open_failure
finally:
    vsr._open_devnull = real_open_devnull
after_open = fd_set()

body = RuntimeError("body failed")
restore_failure = OSError("restore failed")
devnull_failure = KeyboardInterrupt("devnull close failed")
saved_failure = SystemExit("saved close failed")
dup2_calls = 0
close_calls = []

def redirect_then_fail_restore(source):
    global dup2_calls
    dup2_calls += 1
    real_dup2(source, 2)
    if dup2_calls == 2:
        raise restore_failure

def close_then_fail(fd):
    close_calls.append(fd)
    real_close(fd)
    if len(close_calls) == 1:
        raise devnull_failure
    raise saved_failure

before_body = fd_set()
vsr._redirect_stderr = redirect_then_fail_restore
vsr._close_fd = close_then_fail
try:
    with vsr._suppress_native_stderr():
        raise body
except BaseException as caught:
    body_identity = caught is body
    notes = list(getattr(caught, "__notes__", ()))
finally:
    vsr._redirect_stderr = real_redirect
    vsr._close_fd = real_close_fd
after_body = fd_set()

control_failure = KeyboardInterrupt("body interrupted")
before_control = fd_set()
try:
    with vsr._suppress_native_stderr():
        raise control_failure
except BaseException as caught:
    control_identity = caught is control_failure
after_control = fd_set()

devnull_close_primary = OSError("devnull close failed")
close_calls = []

def fail_first_close_after_effect(fd):
    close_calls.append(fd)
    real_close(fd)
    if len(close_calls) == 1:
        raise devnull_close_primary

before_devnull_close = fd_set()
vsr._close_fd = fail_first_close_after_effect
try:
    with vsr._suppress_native_stderr():
        pass
except BaseException as caught:
    devnull_close_identity = caught is devnull_close_primary
finally:
    vsr._close_fd = real_close_fd
after_devnull_close = fd_set()

saved_close_primary = OSError("saved close failed")
close_calls = []

def fail_second_close_after_effect(fd):
    close_calls.append(fd)
    real_close(fd)
    if len(close_calls) == 2:
        raise saved_close_primary

before_saved_close = fd_set()
vsr._close_fd = fail_second_close_after_effect
try:
    with vsr._suppress_native_stderr():
        pass
except BaseException as caught:
    saved_close_identity = caught is saved_close_primary
finally:
    vsr._close_fd = real_close_fd
after_saved_close = fd_set()

before_nested = fd_set()
with vsr._suppress_native_stderr(), vsr._suppress_native_stderr():
    pass
after_nested = fd_set()

holder_entered = threading.Event()
release_holder = threading.Event()
contender_started = threading.Event()
contender_entered = threading.Event()
thread_errors = []

def hold_suppression():
    try:
        with vsr._suppress_native_stderr():
            holder_entered.set()
            if not release_holder.wait(timeout=2):
                raise AssertionError("holder release timed out")
    except BaseException as exc:
        thread_errors.append(exc)

def contend_for_suppression():
    try:
        contender_started.set()
        with vsr._suppress_native_stderr():
            contender_entered.set()
    except BaseException as exc:
        thread_errors.append(exc)

before_contended = fd_set()
holder = threading.Thread(target=hold_suppression, daemon=True)
contender = threading.Thread(target=contend_for_suppression, daemon=True)
holder.start()
if not holder_entered.wait(timeout=2):
    raise AssertionError("holder did not enter")
contender.start()
if not contender_started.wait(timeout=2):
    raise AssertionError("contender did not start")
time.sleep(0.05)
contender_blocked = not contender_entered.is_set()
release_holder.set()
holder.join(timeout=2)
contender.join(timeout=2)
threads_finished = not holder.is_alive() and not contender.is_alive()
after_contended = fd_set()
os.write(2, b"stderr-restored-marker\n")

print(json.dumps({
    "open_identity": open_identity,
    "open_fd_match": before_open == after_open,
    "body_identity": body_identity,
    "body_fd_match": before_body == after_body,
    "close_count": len(close_calls),
    "notes": notes,
    "control_identity": control_identity,
    "control_fd_match": before_control == after_control,
    "devnull_close_identity": devnull_close_identity,
    "devnull_close_fd_match": before_devnull_close == after_devnull_close,
    "saved_close_identity": saved_close_identity,
    "saved_close_fd_match": before_saved_close == after_saved_close,
    "nested_fd_match": before_nested == after_nested,
    "contender_blocked": contender_blocked,
    "threads_finished": threads_finished,
    "thread_errors": [repr(exc) for exc in thread_errors],
    "contended_fd_match": before_contended == after_contended,
}))
'''
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryFile(mode="w+") as stderr:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            check=False,
            timeout=10,
        )
        stderr.seek(0)
        captured_stderr = stderr.read()

    assert result.returncode == 0, captured_stderr
    report = json.loads(result.stdout)
    assert report == {
        "open_identity": True,
        "open_fd_match": True,
        "body_identity": True,
        "body_fd_match": True,
        "close_count": 2,
        "notes": [
            "restore stderr also failed: OSError: restore failed",
            "close /dev/null also failed: KeyboardInterrupt: devnull close failed",
            "close saved stderr also failed: SystemExit: saved close failed",
        ],
        "control_identity": True,
        "control_fd_match": True,
        "devnull_close_identity": True,
        "devnull_close_fd_match": True,
        "saved_close_identity": True,
        "saved_close_fd_match": True,
        "nested_fd_match": True,
        "contender_blocked": True,
        "threads_finished": True,
        "thread_errors": [],
        "contended_fd_match": True,
    }
    assert "stderr-restored-marker" in captured_stderr


def test_native_stderr_suppression_holds_a_process_global_lock(monkeypatch):
    from kinovsr.native import vsr

    monkeypatch.setattr(
        vsr, "default_settings", lambda: SimpleNamespace(verbose=False))
    monkeypatch.setattr(vsr, "_duplicate_stderr", lambda: 10)
    monkeypatch.setattr(vsr, "_open_devnull", lambda: 11)
    monkeypatch.setattr(vsr, "_redirect_stderr", lambda _fd: None)
    monkeypatch.setattr(vsr, "_close_fd", lambda _fd: None)

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
