"""DispatchPipeline: the one-slot async primitive every ANE processor
shares for continuous and windowed dispatch overlap."""

from __future__ import annotations

import threading
import time

import pytest

from kinovsr.native.anemil.runtime import DispatchPipeline
from kinovsr.native.dispatch import QOS_CLASS_USER_INITIATED

pytestmark = pytest.mark.unit


class TestDispatchPipeline:
    def test_submit_join_runs_the_job_off_thread(self):
        pipeline = DispatchPipeline("test")
        seen = []
        pipeline.submit(lambda: seen.append(threading.current_thread().name))
        assert pipeline.in_flight
        pipeline.join()
        assert not pipeline.in_flight
        assert len(seen) == 1
        assert seen[0] != threading.current_thread().name
        pipeline.close()

    def test_idle_reports_without_consuming(self):
        pipeline = DispatchPipeline("test")
        assert pipeline.idle()
        release = threading.Event()
        pipeline.submit(release.wait)
        assert not pipeline.idle()
        release.set()
        pipeline.join()
        assert pipeline.idle()
        pipeline.close()

    def test_join_reraises_the_job_error(self):
        pipeline = DispatchPipeline("test")

        def boom():
            raise ValueError("dispatch failed")

        pipeline.submit(boom)
        with pytest.raises(ValueError, match="dispatch failed"):
            pipeline.join()
        assert not pipeline.in_flight   # the slot is free again
        pipeline.close()

    def test_drain_suppresses_the_job_error(self):
        pipeline = DispatchPipeline("test")

        def boom():
            raise ValueError("dispatch failed")

        pipeline.submit(boom)
        pipeline.drain()
        assert not pipeline.in_flight
        pipeline.close()

    def test_double_submit_is_refused(self):
        pipeline = DispatchPipeline("test")
        release = threading.Event()
        pipeline.submit(release.wait)
        with pytest.raises(RuntimeError, match="already in flight"):
            pipeline.submit(lambda: None)
        release.set()
        pipeline.join()
        pipeline.close()

    def test_close_is_idempotent_and_waits_for_the_job(self):
        pipeline = DispatchPipeline("test")
        done = []
        pipeline.submit(lambda: (time.sleep(0.02), done.append(True)))
        pipeline.close()
        assert done == [True]
        pipeline.close()

    def test_optional_worker_qos_is_applied(self):
        import ctypes

        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libc.pthread_self.restype = ctypes.c_void_p
        getter = libc.pthread_get_qos_class_np
        getter.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
        ]
        getter.restype = ctypes.c_int
        seen = []

        def capture():
            qos_class = ctypes.c_uint()
            relative = ctypes.c_int()
            assert getter(
                libc.pthread_self(),
                ctypes.byref(qos_class),
                ctypes.byref(relative),
            ) == 0
            seen.append((qos_class.value, relative.value))

        pipeline = DispatchPipeline(
            "test", qos_class=QOS_CLASS_USER_INITIATED)
        pipeline.submit(capture)
        pipeline.join()
        pipeline.close()
        assert seen == [(QOS_CLASS_USER_INITIATED, 0)]
