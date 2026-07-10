"""Scheduler behavior: lifecycle order, boundaries, flush, cleanup."""

from fractions import Fraction

import pytest

from kinovsr.pipeline import run_chain
from kinovsr.pipeline.builder import ResolvedStage
from kinovsr.processors import (
    Boundary,
    BoundaryKind,
    Capability,
    CapabilitySpec,
    FrameUnit,
    Geometry,
    PipelineContext,
    PipelineRuntimeError,
    StreamConstraint,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
    preserve_stream,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()
CONTEXT = PipelineContext(settings=SETTINGS)


def spec() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(64, 48)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


class Recorder:
    """A processor that records its lifecycle and passes units through."""

    def __init__(self, name="stage", log=None):
        self.name = name
        self.log = log if log is not None else []
        self.seen_boundaries = []

    def _note(self, event, detail=""):
        self.log.append((self.name, event, detail))

    def prepare(self, input_spec, context):
        self._note("prepare", context.stage_id)

    def process(self, unit, context):
        self.seen_boundaries.append(unit.boundaries)
        self._note("process", unit.pts)
        yield unit

    def reset(self, boundary, context):
        self._note("reset", boundary.kind.value)

    def flush(self, context):
        self._note("flush")
        return ()

    def close(self, context):
        self._note("close")


class Buffering(Recorder):
    """Holds ``depth`` units; emits FIFO once full; drains on flush."""

    def __init__(self, name="buffer", log=None, depth=2):
        super().__init__(name, log)
        self.depth = depth
        self.held = []

    def process(self, unit, context):
        self.seen_boundaries.append(unit.boundaries)
        self._note("process", unit.pts)
        self.held.append(unit)
        if len(self.held) > self.depth:
            yield self.held.pop(0)

    def reset(self, boundary, context):
        super().reset(boundary, context)
        self.held.clear()

    def flush(self, context):
        self._note("flush")
        drained, self.held = self.held, []
        yield from drained


class Doubling(Recorder):
    """Emits each unit twice (interpolation-shaped fan-out)."""

    def process(self, unit, context):
        self.seen_boundaries.append(unit.boundaries)
        self._note("process", unit.pts)
        yield unit
        yield unit.retimed(unit.pts + 1)


class Exploding(Recorder):
    def __init__(self, name="boom", log=None, at_pts=2):
        super().__init__(name, log)
        self.at_pts = at_pts

    def process(self, unit, context):
        if unit.pts == self.at_pts:
            raise ValueError("kaboom")
        yield unit


def stage_for(processor, name=None) -> ResolvedStage:
    s = spec()
    return ResolvedStage(
        name=name or processor.name, position=0, family="fake",
        factory=None, capability=Capability.PREPROCESS,
        capability_spec=CapabilitySpec(
            capability=Capability.PREPROCESS, profiles=(),
            accepts=StreamConstraint(), produces=preserve_stream),
        profile=None, config=None, input_spec=s, output_spec=s)


def units(n, start_pts=0):
    return [FrameUnit(payload=f"frame{i}", pts=start_pts + i, duration=1)
            for i in range(n)]


def chain(*processors):
    return tuple((stage_for(p), p) for p in processors)


class TestLifecycle:
    def test_order_prepare_reset_process_flush_close(self):
        log = []
        stage = Recorder("a", log)
        out = list(run_chain(chain(stage), units(2), CONTEXT))
        assert [u.payload for u in out] == ["frame0", "frame1"]
        assert log == [
            ("a", "prepare", "a"),
            ("a", "reset", "stream_start"),
            ("a", "process", 0),
            ("a", "process", 1),
            ("a", "flush", ""),
            ("a", "close", ""),
        ]

    def test_empty_stream_still_prepares_flushes_closes(self):
        log = []
        stage = Recorder("a", log)
        assert list(run_chain(chain(stage), [], CONTEXT)) == []
        events = [e for _, e, _ in log]
        assert events == ["prepare", "flush", "close"]

    def test_stream_start_not_duplicated_when_endpoint_provides(self):
        log = []
        stage = Recorder("a", log)
        first = FrameUnit(payload="x", pts=0, duration=1).with_boundary(
            Boundary(BoundaryKind.STREAM_START))
        list(run_chain(chain(stage), [first], CONTEXT))
        resets = [e for e in log if e[1] == "reset"]
        assert resets == [("a", "reset", "stream_start")]

    def test_pull_based_laziness(self):
        pulled = []

        def source():
            for unit in units(10):
                pulled.append(unit.pts)
                yield unit

        stream = run_chain(chain(Recorder("a")), source(), CONTEXT)
        next(stream)
        assert len(pulled) <= 2  # no eager drain of the source
        stream.close()


class TestBoundaries:
    @staticmethod
    def cut_at(items, index, source_index=None):
        out = list(items)
        out[index] = out[index].with_boundary(
            Boundary(BoundaryKind.HARD_CUT, source_index=source_index))
        return out

    def test_families_never_see_inband_boundaries(self):
        a, b = Recorder("a"), Recorder("b")
        feed = self.cut_at(units(4), 2)
        list(run_chain(chain(a, b), feed, CONTEXT))
        assert all(bs == () for bs in a.seen_boundaries)
        assert all(bs == () for bs in b.seen_boundaries)

    def test_hard_cut_resets_every_downstream_stage_once(self):
        log = []
        a, b = Recorder("a", log), Recorder("b", log)
        feed = self.cut_at(units(4), 2)
        list(run_chain(chain(a, b), feed, CONTEXT))
        cuts = [(stage, e) for stage, e, d in log if d == "hard_cut"]
        assert cuts == [("a", "reset"), ("b", "reset")]

    def test_cut_reset_happens_before_the_cut_frame_processes(self):
        log = []
        a = Recorder("a", log)
        feed = self.cut_at(units(4), 2)
        list(run_chain(chain(a), feed, CONTEXT))
        cut_reset = log.index(("a", "reset", "hard_cut"))
        frame2 = log.index(("a", "process", 2))
        assert cut_reset < frame2

    def test_buffering_stage_drains_pre_cut_tail_before_reset(self):
        log = []
        buffer = Buffering("buf", log, depth=2)
        feed = self.cut_at(units(5), 3)
        out = list(run_chain(chain(buffer), feed, CONTEXT))
        # nothing lost: all five frames come out, in order
        assert [u.pts for u in out] == [0, 1, 2, 3, 4]
        # the mid-stream flush happened before the reset
        flush_i = log.index(("buf", "flush", ""))
        reset_i = log.index(("buf", "reset", "hard_cut"))
        assert flush_i < reset_i
        # and the boundary rode the first POST-cut output, not a tail unit
        cut_units = [u.pts for u in out
                     if any(b.kind is BoundaryKind.HARD_CUT
                            for b in u.boundaries)]
        assert cut_units == [3]

    def test_boundary_crosses_a_buffering_stage_to_reset_downstream(self):
        log = []
        buffer = Buffering("buf", log, depth=2)
        after = Recorder("after", log)
        feed = self.cut_at(units(6), 3)
        out = list(run_chain(chain(buffer, after), feed, CONTEXT))
        assert [u.pts for u in out] == [0, 1, 2, 3, 4, 5]
        # downstream reset happens after the pre-cut tail (0,1,2) was
        # processed downstream and before the cut frame (3) is
        downstream = [(e, d) for stage, e, d in log if stage == "after"]
        reset_i = downstream.index(("reset", "hard_cut"))
        processed_before = [d for e, d in downstream[:reset_i]
                            if e == "process"]
        assert processed_before == [0, 1, 2]  # the pre-cut tail only
        assert ("process", 3) in downstream[reset_i:]

    def test_fanout_attaches_boundary_to_first_output_only(self):
        doubler = Doubling("x2")
        after = Recorder("after")
        feed = self.cut_at(units(3), 1)
        out = list(run_chain(chain(doubler, after), feed, CONTEXT))
        flagged = [i for i, u in enumerate(out)
                   if any(b.kind is BoundaryKind.HARD_CUT
                          for b in u.boundaries)]
        assert len(out) == 6
        assert flagged == [2]  # first emission for source frame 1

    def test_family_emitted_boundary_resets_downstream(self):
        class Cutter(Recorder):
            def process(self, unit, context):
                self.seen_boundaries.append(unit.boundaries)
                if unit.pts == 2:
                    unit = unit.with_boundary(
                        Boundary(BoundaryKind.HARD_CUT, source_index=2))
                yield unit

        log = []
        cutter = Cutter("cutter", log)
        after = Recorder("after", log)
        list(run_chain(chain(cutter, after), units(4), CONTEXT))
        downstream_cuts = [d for stage, e, d in log
                           if stage == "after" and e == "reset"]
        assert downstream_cuts == ["stream_start", "hard_cut"]


class TestFlushCascade:
    def test_upstream_tail_flows_through_downstream_stages(self):
        buffer = Buffering("buf", depth=3)
        after = Recorder("after")
        out = list(run_chain(chain(buffer, after), units(3), CONTEXT))
        # everything was held until end-of-stream flush, then flowed on
        assert [u.pts for u in out] == [0, 1, 2]
        assert [d for _, e, d in after.log if e == "process"] == [0, 1, 2]


class TestCleanup:
    def test_close_exactly_once_on_success(self):
        log = []
        a, b = Recorder("a", log), Recorder("b", log)
        list(run_chain(chain(a, b), units(2), CONTEXT))
        assert [x for x in log if x[1] == "close"] == [
            ("a", "close", ""), ("b", "close", "")]

    def test_cancel_mid_run_closes_exactly_once(self):
        log = []
        a, b = Recorder("a", log), Recorder("b", log)
        stream = run_chain(chain(a, b), units(10), CONTEXT)
        assert next(stream).pts == 0
        stream.close()
        closes = [x for x in log if x[1] == "close"]
        assert closes == [("a", "close", ""), ("b", "close", "")]

    def test_stage_exception_is_wrapped_and_still_closes_all(self):
        log = []
        boom = Exploding("boom", log, at_pts=1)
        after = Recorder("after", log)
        with pytest.raises(PipelineRuntimeError, match=r"\[boom\].*kaboom"):
            list(run_chain(chain(boom, after), units(4), CONTEXT))
        closes = [x[0] for x in log if x[1] == "close"]
        assert closes == ["boom", "after"]

    def test_close_failure_on_success_path_is_raised(self):
        class BadClose(Recorder):
            def close(self, context):
                super().close(context)
                raise RuntimeError("close failed")

        bad = BadClose("bad")
        with pytest.raises(PipelineRuntimeError, match=r"\[bad\].*close"):
            list(run_chain(chain(bad), units(1), CONTEXT))
        assert [x for x in bad.log if x[1] == "close"] == [
            ("bad", "close", "")]

    def test_close_failure_does_not_mask_stage_error(self):
        class BadCloseBoom(Exploding):
            def close(self, context):
                raise RuntimeError("close failed too")

        boom = BadCloseBoom("boom", at_pts=0)
        with pytest.raises(PipelineRuntimeError, match="kaboom"):
            list(run_chain(chain(boom), units(1), CONTEXT))


@pytest.mark.slow
def test_scheduler_overhead_sanity():
    """A loose in-process guard against gross framework regressions; the
    real gate protocol is scripts/dev/bench_scheduler.py (planning 06)."""
    import statistics
    import time

    stages = chain(*(Recorder(f"s{i}") for i in range(8)))
    for recorder in (p for _, p in stages):
        recorder.log = []  # keep the log tiny; we only time
    feed = (FrameUnit(payload=None, pts=i, duration=1) for i in range(150))
    stream = run_chain(stages, feed, CONTEXT)
    for _ in range(30):
        next(stream)
    samples = []
    for _ in range(120):
        t0 = time.perf_counter_ns()
        next(stream)
        samples.append(time.perf_counter_ns() - t0)
    stream.close()
    median_ms_per_stage = statistics.median(samples) / 8 / 1e6
    assert median_ms_per_stage < 0.05, median_ms_per_stage


class TestOwnershipBeforeIteration:
    """Cleanup must not depend on the first pull: a plain generator never
    enters its own try/finally when closed or abandoned unstarted."""

    def test_close_before_first_pull_closes_processors(self):
        log = []
        a, b = Recorder("a", log), Recorder("b", log)
        stream = run_chain(chain(a, b), units(5), CONTEXT)
        stream.close()  # never iterated
        closes = [x for x in log if x[1] == "close"]
        assert closes == [("a", "close", ""), ("b", "close", "")]
        assert list(stream) == []  # closed run yields nothing
        assert closes == [x for x in log if x[1] == "close"]  # still once

    def test_abandonment_closes_processors(self):
        log = []
        a = Recorder("a", log)
        stream = run_chain(chain(a), units(5), CONTEXT)
        next(stream)
        del stream  # CPython refcount finalizes immediately
        assert [x for x in log if x[1] == "close"] == [("a", "close", "")]

    def test_context_manager_closes_once(self):
        log = []
        a = Recorder("a", log)
        with run_chain(chain(a), units(2), CONTEXT) as stream:
            assert next(stream).pts == 0
        assert [x for x in log if x[1] == "close"] == [("a", "close", "")]

    def test_context_manager_does_not_mask_body_error(self):
        class BadClose(Recorder):
            def close(self, context):
                super().close(context)
                raise RuntimeError("close failed")

        def drive_and_fail(stream):
            with stream:
                next(stream)
                raise ValueError("body error")

        bad = BadClose("bad")
        stream = run_chain(chain(bad), units(2), CONTEXT)
        with pytest.raises(ValueError, match="body error"):
            drive_and_fail(stream)
        assert [x for x in bad.log if x[1] == "close"] == [
            ("bad", "close", "")]


class TestCleanupUnderInterrupts:
    """The close guarantee holds even for BaseException and a failing
    generator close: every stage still closes, nothing is masked silently."""

    def test_interrupt_during_close_still_closes_the_rest(self):
        log = []

        class InterruptingClose(Recorder):
            def close(self, context):
                super().close(context)
                raise KeyboardInterrupt

        first = InterruptingClose("first", log)
        second = Recorder("second", log)
        stream = run_chain(chain(first, second), units(2), CONTEXT)
        next(stream)
        with pytest.raises(KeyboardInterrupt):
            stream.close()
        closes = [x[0] for x in log if x[1] == "close"]
        assert closes == ["first", "second"]

    def test_abandonment_before_first_pull_closes(self):
        log = []
        stage = Recorder("a", log)
        stream = run_chain(chain(stage), units(3), CONTEXT)
        del stream  # never iterated
        assert [x for x in log if x[1] == "close"] == [("a", "close", "")]

    def test_repeated_close_is_idempotent(self):
        log = []
        stage = Recorder("a", log)
        stream = run_chain(chain(stage), units(3), CONTEXT)
        next(stream)
        stream.close()
        stream.close()
        assert [x for x in log if x[1] == "close"] == [("a", "close", "")]

    def test_failing_stream_close_still_closes_processors(self):
        log = []
        stage = Recorder("a", log)
        stream = run_chain(chain(stage), units(3), CONTEXT)
        next(stream)

        class BadStream:
            def close(self):
                raise RuntimeError("generator refused to die")

        stream._stream = BadStream()
        with pytest.raises(RuntimeError, match="refused to die"):
            stream.close()
        assert [x for x in log if x[1] == "close"] == [("a", "close", "")]


class TestCleanupPrecedence:
    """Combined-failure precedence: the active error outranks cleanup
    noise; with no active error the FIRST cleanup failure wins; outranked
    errors stay on the winner's context chain."""

    class _BadStream:
        def close(self):
            raise RuntimeError("stream-close failed")

    def test_body_error_outranks_failing_stream_close(self):
        def drive_and_fail(stream):
            with stream:
                next(stream)
                stream._stream = self._BadStream()
                raise ValueError("body error")

        log = []
        stage = Recorder("a", log)
        stream = run_chain(chain(stage), units(3), CONTEXT)
        with pytest.raises(ValueError, match="body error") as exc:
            drive_and_fail(stream)
        # processors still closed, and the cleanup failure is preserved
        # on the delivered error's context chain
        assert [x for x in log if x[1] == "close"] == [("a", "close", "")]
        context = exc.value.__context__
        assert isinstance(context, RuntimeError)
        assert "stream-close failed" in str(context)

    def test_first_cleanup_failure_wins_on_explicit_close(self):
        class BadClose(Recorder):
            def close(self, context):
                super().close(context)
                raise ValueError("processor-close failed")

        stage = BadClose("a")
        stream = run_chain(chain(stage), units(3), CONTEXT)
        next(stream)
        stream._stream = self._BadStream()
        with pytest.raises(RuntimeError, match="stream-close failed") as exc:
            stream.close()
        assert [x for x in stage.log if x[1] == "close"] == [
            ("a", "close", "")]
        context = exc.value.__context__
        assert context is not None
        assert "processor-close failed" in str(context)
