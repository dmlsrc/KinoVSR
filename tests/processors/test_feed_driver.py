"""FeedFlushProcessor: token pairing and the optional luma/chroma split.

The split is applied here, in the driver adapter, precisely because the
token threading is what pairs a delayed output with the input frame it was
computed from - not the frame currently arriving.
"""
from __future__ import annotations

from fractions import Fraction

import mlx.core as mx
import pytest

from kinovsr.media.yuv import luma_chroma_blend
from kinovsr.processors import (
    FrameUnit,
    Geometry,
    PipelineContext,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.processors.feed_driver import FeedFlushProcessor, parse_luma_chroma
from kinovsr.processors.specs import ColorMatrix, luma_coefficients
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()
CTX = PipelineContext(settings=SETTINGS)


def stream(matrix: str = "bt601") -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            matrix, full_range=False, geometry=Geometry(4, 4)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


def gray_unit(value: float, pts: int) -> FrameUnit:
    return FrameUnit(
        payload=mx.full((4, 4, 3), value, dtype=mx.float32),
        pts=pts, duration=1)


def colored_unit(pts: int = 0) -> FrameUnit:
    row = [[0.70, 0.20, 0.55], [0.15, 0.85, 0.40],
           [0.50, 0.50, 0.10], [0.30, 0.60, 0.90]]
    return FrameUnit(payload=mx.array([row, row, row, row], dtype=mx.float32),
                     pts=pts, duration=1)


def _close(a, b, atol=1e-5):
    return bool(mx.all(mx.abs(a - b) <= atol).item())


class _HalveDelayDriver:
    """A stand-in windowed denoiser: emit 0.5*input for each frame, delayed
    by ``delay`` frames, each bound to the token it was computed from. The
    flush drains the tail."""

    def __init__(self, delay: int = 0) -> None:
        self._delay = delay
        self._buf: list = []

    def feed(self, frame, token=None) -> list:
        self._buf.append((frame * 0.5, token))
        out = []
        while len(self._buf) > self._delay:
            out.append(self._buf.pop(0))
        return out

    def flush(self) -> list:
        out, self._buf = self._buf, []
        return out

    def reset(self) -> None:
        self._buf = []


def run(proc: FeedFlushProcessor, spec: StreamSpec, units: list) -> list:
    proc.prepare(spec, CTX)
    out: list = []
    for unit in units:
        out.extend(proc.process(unit, CTX))
    out.extend(proc.flush(CTX))
    return out


class TestNoSplit:
    def test_strengths_of_one_bind_no_blend(self):
        proc = FeedFlushProcessor(lambda: _HalveDelayDriver(0))
        out = run(proc, stream(), [gray_unit(0.4, 0)])
        assert proc._blend is None
        # raw driver output passes through untouched: 0.5 * 0.4
        assert _close(out[0].payload, mx.full((4, 4, 3), 0.2))

    def test_default_strengths_are_one(self):
        proc = FeedFlushProcessor(lambda: _HalveDelayDriver(0))
        assert (proc._luma_strength, proc._chroma_strength) == (1.0, 1.0)


class TestSplitApplies:
    def test_gray_frame_blends_luma_only(self):
        # Equal strengths on a gray frame move only luma (chroma is zero):
        # 0.4 + 0.6*(0.2 - 0.4) = 0.28.
        proc = FeedFlushProcessor(lambda: _HalveDelayDriver(0),
                                  luma_strength=0.6, chroma_strength=0.6)
        out = run(proc, stream(), [gray_unit(0.4, 0)])
        assert _close(out[0].payload, mx.full((4, 4, 3), 0.28))

    def test_output_matches_the_reference_blend(self):
        # Colored frame, luma != chroma, so the coefficients matter; the
        # adapter must match luma_chroma_blend with the stream's own coefs.
        proc = FeedFlushProcessor(lambda: _HalveDelayDriver(0),
                                  luma_strength=0.3, chroma_strength=1.0)
        unit = colored_unit()
        out = run(proc, stream("bt709"), [unit])
        kr, kb = luma_coefficients(ColorMatrix.BT709)
        expected = luma_chroma_blend(unit.payload, unit.payload * 0.5,
                                     0.3, 1.0, kr, kb)
        assert _close(out[0].payload, expected)


class TestTokenPairing:
    def test_delayed_output_blends_against_its_own_input(self):
        # delay=2: an output emerges two frames after the input it came from.
        # Blending against the arriving frame instead of the source frame
        # would use the wrong value, which distinct per-frame values detect.
        proc = FeedFlushProcessor(lambda: _HalveDelayDriver(2),
                                  luma_strength=0.6, chroma_strength=0.6)
        values = [0.1, 0.2, 0.3, 0.5]
        units = [gray_unit(v, pts=i) for i, v in enumerate(values)]
        out = run(proc, stream(), units)
        assert [u.pts for u in out] == [0, 1, 2, 3]
        # each output is 0.7 * its OWN input value (gray luma-only blend)
        for got, v in zip(out, values, strict=True):
            assert _close(got.payload, mx.full((4, 4, 3), v * 0.7))


class TestCoefficientBinding:
    def test_coefficients_come_from_the_input_spec_matrix(self):
        unit = colored_unit()
        results = {}
        for matrix in ("bt601", "bt709", "bt2020"):
            proc = FeedFlushProcessor(lambda: _HalveDelayDriver(0),
                                      luma_strength=0.2, chroma_strength=1.0)
            results[matrix] = run(proc, stream(matrix), [unit])[0].payload
        # a luma!=chroma split on a chromatic frame resolves differently per
        # matrix, so the adapter is reading the coefficients from the spec.
        assert not _close(results["bt601"], results["bt709"])
        assert not _close(results["bt709"], results["bt2020"])

    def test_luma_coefficients_pin_the_itu_r_values(self):
        assert luma_coefficients(ColorMatrix.BT601) == (0.299, 0.114)
        assert luma_coefficients(ColorMatrix.BT709) == (0.2126, 0.0722)
        assert luma_coefficients(ColorMatrix.BT2020) == (0.2627, 0.0593)


class TestDtype:
    def test_blend_preserves_the_driver_output_dtype(self):
        class _F16Driver:
            def feed(self, frame, token=None):
                return [((frame * 0.5).astype(mx.float16), token)]

            def flush(self):
                return []

            def reset(self):
                pass

        proc = FeedFlushProcessor(lambda: _F16Driver(),
                                  luma_strength=0.5, chroma_strength=0.5)
        out = run(proc, stream(), [gray_unit(0.4, 0)])
        # blend computes in float32 but stores in the stage's declared dtype,
        # so blend-on matches blend-off (the driver output dtype).
        assert out[0].payload.dtype == mx.float16


class TestParseHelper:
    def test_defaults_to_full_effect(self):
        assert parse_luma_chroma({}) == (1.0, 1.0)

    def test_values_pass_through_including_overdrive(self):
        assert parse_luma_chroma(
            {"luma_strength": 0.4, "chroma_strength": 1.5}) == (0.4, 1.5)


class TestWindowing:
    """PipelineContext.windowing drives schedule-capable drivers at prepare."""

    class _SchedDriver(_HalveDelayDriver):
        def __init__(self):
            super().__init__(0)
            self.schedule = None

        def set_schedule(self, schedule):
            self.schedule = schedule

    def test_schedule_reaches_a_capable_driver(self):
        driver = self._SchedDriver()
        proc = FeedFlushProcessor(lambda: driver)
        ctx = PipelineContext(
            settings=SETTINGS, windowing=((0, 9, 0, 8), (8, 17, 8, 16)))
        proc.prepare(stream(), ctx)
        assert driver.schedule == [(0, 9, 0, 8), (8, 17, 8, 16)]

    def test_per_frame_driver_is_skipped(self):
        # No set_schedule on the driver: the plan is ignored, not an error.
        proc = FeedFlushProcessor(lambda: _HalveDelayDriver(0))
        ctx = PipelineContext(settings=SETTINGS, windowing=((0, 4, 0, 4),))
        out = run(proc, stream(), [gray_unit(0.4, 0)])
        del out
        proc.prepare(stream(), ctx)   # idempotent, still no error

    def test_no_windowing_leaves_continuous_mode(self):
        driver = self._SchedDriver()
        FeedFlushProcessor(lambda: driver).prepare(stream(), CTX)
        assert driver.schedule is None


class TestRunDiagnostics:
    """The end-of-run diagnostics hook: family-owned lines, forwarded by the
    adapter, collected by the session after the stream drains."""

    class _Reporting(_HalveDelayDriver):
        def run_diagnostics(self):
            return ["[fake] processed everything"]

    def test_forwards_the_driver_report(self):
        proc = FeedFlushProcessor(lambda: self._Reporting(0))
        run(proc, stream(), [gray_unit(0.4, 0)])
        assert proc.run_diagnostics() == ["[fake] processed everything"]

    def test_silent_driver_reports_nothing(self):
        proc = FeedFlushProcessor(lambda: _HalveDelayDriver(0))
        run(proc, stream(), [gray_unit(0.4, 0)])
        assert proc.run_diagnostics() == []

    def test_unprepared_processor_reports_nothing(self):
        assert FeedFlushProcessor(lambda: self._Reporting(0)
                                  ).run_diagnostics() == []


class _ScriptedHandle:
    """An async window handle needing `advances_needed` non-blocking pokes,
    or one blocking call, to complete."""

    def __init__(self, outputs, advances_needed=0):
        self.outputs = outputs
        self._remaining = advances_needed
        self.error = None
        self.blocking_calls = 0
        self.nonblocking_calls = 0
        self.done = False

    def advance(self, block=False):
        if self.error is not None:
            raise self.error
        if block:
            self.blocking_calls += 1
            self.done = True
            return True
        self.nonblocking_calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            return False
        self.done = True
        return True


class TestWindowWavefront:
    """Depth-one cross-window pipelining: the shared async window protocol
    any accelerator-backed family implements via ``begin_window``."""

    def test_submit_completes_the_previous_window_first(self):
        from kinovsr.processors.feed_driver import WindowWavefront

        wavefront = WindowWavefront()
        first = _ScriptedHandle(["w1"])
        order = []

        out = wavefront.submit(lambda: first,
                               lambda h: order.append(1) or ["e1"])
        assert out == []                       # nothing was in flight
        assert wavefront.in_flight
        assert first.nonblocking_calls == 1    # kicked, never blocked on

        second = _ScriptedHandle(["w2"])
        out = wavefront.submit(lambda: second,
                               lambda h: order.append(2) or ["e2"])
        assert out == ["e1"]                   # depth-one backpressure
        assert first.blocking_calls == 1
        assert order == [1]

        assert wavefront.barrier() == ["e2"]
        assert order == [1, 2]
        assert not wavefront.in_flight
        assert wavefront.barrier() == []       # idempotent when empty

    def test_poll_advances_without_blocking(self):
        from kinovsr.processors.feed_driver import WindowWavefront

        wavefront = WindowWavefront()
        handle = _ScriptedHandle(["w"], advances_needed=3)
        wavefront.submit(lambda: handle, lambda h: list(h.outputs))
        for _ in range(5):
            wavefront.poll()
        assert handle.blocking_calls == 0
        assert handle.done                     # polls alone completed it
        assert wavefront.barrier() == ["w"]

    def test_finalize_receives_the_completed_handle(self):
        from kinovsr.processors.feed_driver import WindowWavefront

        wavefront = WindowWavefront()
        handle = _ScriptedHandle(["a", "b"])
        wavefront.submit(lambda: handle,
                         lambda h: [value.upper() for value in h.outputs])
        assert wavefront.barrier() == ["A", "B"]

    def test_submit_propagates_the_previous_window_error(self):
        from kinovsr.processors.feed_driver import WindowWavefront

        wavefront = WindowWavefront()
        failing = _ScriptedHandle([])
        wavefront.submit(lambda: failing, lambda h: [])
        failing.error = ValueError("window failed")
        with pytest.raises(ValueError, match="window failed"):
            wavefront.submit(lambda: _ScriptedHandle([]), lambda h: [])
        assert not wavefront.in_flight         # failed window was consumed

    def test_abandon_suppresses_the_window_error(self):
        from kinovsr.processors.feed_driver import WindowWavefront

        wavefront = WindowWavefront()
        handle = _ScriptedHandle(["w"])
        wavefront.submit(lambda: handle, lambda h: list(h.outputs))
        handle.error = ValueError("late failure")
        wavefront.abandon()                    # swallowed: teardown path
        assert not wavefront.in_flight


class TestPreheatHook:
    def test_prepare_passes_the_input_geometry(self):
        calls = []

        class Driver:
            def feed(self, frame, token=None):
                return []

            def flush(self):
                return []

            def reset(self):
                return None

            def preheat(self, height, width):
                calls.append((height, width))

        spec = StreamSpec(
            frame=frame_spec_for_matrix(
                "bt601", full_range=False, geometry=Geometry(64, 48)),
            timeline=TimelineSpec(
                time_base=Fraction(1, 24000), cadence=Fraction(25)))
        proc = FeedFlushProcessor(lambda: Driver())
        proc.prepare(spec, CTX)
        assert calls == [(48, 64)]   # (height, width) from Geometry(w, h)

    def test_drivers_without_the_hook_are_untouched(self):
        proc = FeedFlushProcessor(lambda: _HalveDelayDriver(0))
        proc.prepare(stream(), CTX)   # must not raise
