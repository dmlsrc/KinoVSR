"""BSVD on the Neural Engine: the None-flow schedule mirror, backend
selection plumbing, and (slow) end-to-end parity against the MLX net."""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinovsr.processors import bsvd as B
from kinovsr.processors.bsvd import ane as A
from kinovsr.processors.bsvd import ane_phases as P

# --------------------------------------------------------------------------
# Schedule mirror vs the instrumented product classes
# --------------------------------------------------------------------------

def _synthetic_block(cin: int, seed: int) -> dict:
    """A structurally faithful _DenBlock param table with tiny channels.

    The None-propagation schedule depends only on the architecture, never
    on weight values or channel widths, so eight-channel random weights at
    16x16 reproduce it exactly and keep the instrumented truth run fast
    and independent of the real checkpoint.
    """
    keys = mx.random.split(mx.random.key(seed), 16)
    index = 0

    def conv(out_channels: int, in_channels: int, stride: int = 1):
        nonlocal index
        weight = mx.random.normal(
            shape=(out_channels, 3, 3, in_channels), key=keys[index]) * 0.1
        bias = mx.zeros((out_channels,))
        index += 1
        return weight.astype(mx.float32), bias, stride

    return {
        "inc0": conv(8, cin), "inc3": conv(8, 8),
        "d0": conv(8, 8, stride=2), "d0c1": conv(8, 8), "d0c2": conv(8, 8),
        "d1": conv(8, 8, stride=2), "d1c1": conv(8, 8), "d1c2": conv(8, 8),
        "u2c1": conv(8, 8), "u2c2": conv(8, 8), "u2": conv(32, 8),
        "u1c1": conv(8, 8), "u1c2": conv(8, 8), "u1": conv(32, 8),
        "out0": conv(8, 8), "out3": conv(4, 8),
    }


def _instrumented_schedule(length: int, steps: int = 16):
    """Derive the schedule from the real product classes, instrumented the
    way scripts/dev/probe_bsvd_ane.py derives it: patch the CLASS (an
    instance attribute does not intercept ``self.c1(x)``), record which
    units see None, which are unprimed, and which skip pushes are real."""
    temp1 = B._DenBlock(_synthetic_block(4, seed=11))
    temp2 = B._DenBlock(_synthetic_block(4, seed=17))
    order = ("down0.c1", "down0.c2", "down1.c1", "down1.c2",
             "up2.c1", "up2.c2", "up1.c1", "up1.c2")
    units = [getattr(getattr(block, name.split(".")[0]), name.split(".")[1])
             for block in (temp1, temp2) for name in order]
    lines = [getattr(block, label) for block in (temp1, temp2)
             for label in ("skip1", "skip2", "skip3")]
    for probe_id, unit in enumerate(units):
        unit._probe_id = probe_id
    for probe_id, line in enumerate(lines):
        line._probe_id = probe_id

    record = {"gates": None, "pushes": None, "unprimed": None}
    original_call = B._BiBufferConv.__call__
    original_push = B._MemSkip.push

    def patched_call(self, input_right):
        identity = getattr(self, "_probe_id", None)
        if identity is not None and record["gates"] is not None:
            record["gates"][identity] = input_right is None
        if identity is not None and record["unprimed"] is not None:
            record["unprimed"][identity] = self._center is None
        return original_call(self, input_right)

    def patched_push(self, value):
        identity = getattr(self, "_probe_id", None)
        if identity is not None and record["pushes"] is not None:
            record["pushes"][identity] = value is not None
        return original_push(self, value)

    B._BiBufferConv.__call__ = patched_call
    B._MemSkip.push = patched_push
    try:
        unprimed, fill_emits = [], []
        for i in range(length):
            frame = mx.random.uniform(shape=(1, 16, 16, 4),
                                      key=mx.random.key(300 + i))
            record["unprimed"] = [False] * 16
            result = temp2(temp1(frame))
            unprimed.append(list(record["unprimed"]))
            fill_emits.append(result is not None)
        gates, pushes, emits = [], [], []
        for _ in range(steps):
            record["gates"], record["pushes"] = [False] * 16, [False] * 6
            record["unprimed"] = [False] * 16
            result = temp2(temp1(None))
            unprimed.append(list(record["unprimed"]))
            gates.append(list(record["gates"]))
            pushes.append(list(record["pushes"]))
            emits.append(result is not None)
    finally:
        B._BiBufferConv.__call__ = original_call
        B._MemSkip.push = original_push
    total = len(unprimed)
    writes = [[0.0 if (unprimed[k][i] and (k + 1 >= total
                                           or not unprimed[k + 1][i]))
               else 1.0 for i in range(16)] for k in range(total)]
    return {"gates": gates, "pushes": pushes, "emits": emits,
            "writes": writes, "fill_emits": fill_emits}


def _mirror_schedule(length: int):
    """The same schedule as the ANE backend derives it: the boolean mirror
    for the fill, and the production ``_assemble_tail`` for the drain."""
    mirror = A._NoneFlowNet()
    fill_writes, fill_emits = [], []
    for _ in range(length):
        record = mirror.step(True)
        fill_writes.append(
            [0.0 if record.primes[i] else 1.0 for i in range(16)])
        fill_emits.append(record.out_real)
    shim = object.__new__(A.AneBSVD)
    shim._mirror = mirror
    tail = A.AneBSVD._assemble_tail(shim)
    return {"fill_writes": fill_writes, "fill_emits": fill_emits,
            "tail": tail}


@pytest.mark.unit
class TestNoneFlowMirror:
    @pytest.mark.parametrize(
        "length", [1, 2, 3, 4, 7, 8, 15, 16, 17, 24, 33, 48])
    def test_schedule_matches_the_instrumented_product_classes(self, length):
        truth = _instrumented_schedule(length)
        mine = _mirror_schedule(length)

        assert mine["fill_emits"] == truth["fill_emits"]
        assert mine["fill_writes"] == truth["writes"][:length]
        assert [entry["emit"] for entry in mine["tail"]] == truth["emits"]
        assert [entry["pushes"] for entry in mine["tail"]] == truth["pushes"]
        for k, entry in enumerate(mine["tail"]):
            want_gate = [0.0 if truth["gates"][k][i] else 1.0
                         for i in range(16)]
            assert entry["gate"] == want_gate, f"tail step {k}"
            assert entry["write"] == truth["writes"][length + k], \
                f"tail step {k}"

    def test_every_input_is_emitted_exactly_once(self):
        for length in (1, 5, 16, 20):
            truth = _instrumented_schedule(length)
            emitted = sum(truth["fill_emits"]) + sum(truth["emits"])
            assert emitted == length


# --------------------------------------------------------------------------
# Geometry envelope: width alignment and reflect padding
# --------------------------------------------------------------------------

@pytest.mark.unit
class TestGeometry:
    def test_pad_width_reflect_mirrors_the_right_edge(self):
        x = mx.arange(2 * 3 * 6 * 1, dtype=mx.float32).reshape(2, 3, 6, 1)
        padded = A._pad_width_reflect(x, 9)
        mx.eval(padded)
        assert padded.shape == (2, 3, 9, 1)
        assert float(mx.abs(padded[:, :, :6] - x).max()) == 0.0
        # Reflection excludes the edge column: [.. 3 4 5] -> [.. 3 4 5 4 3 2]
        want = x[:, :, 2:5, :][:, :, ::-1, :]
        assert float(mx.abs(padded[:, :, 6:] - want).max()) == 0.0
        same = A._pad_width_reflect(x, 6)
        assert same is x

    def test_unaligned_graph_width_is_refused_by_build_runner(self):
        # Geometry checks run before the params are touched, so no weights
        # are needed to assert the guard.
        with pytest.raises(RuntimeError, match="multiple of 128"):
            A.build_runner(None, 4, 288, 352)

    def test_below_floor_is_refused(self):
        with pytest.raises(RuntimeError, match="verified ANE floor"):
            A.build_runner(None, 4, 92, 128)

    def test_phase_window_envelope_is_the_verified_1024x576_rectangle(self):
        net = object.__new__(A.AneBSVD)

        assert net.window_capable(480, 640)
        assert net.window_capable(576, 768)
        assert net.window_capable(576, 1024)
        assert net.window_capable(576, 897)  # padded to 1024
        assert net.window_capable(540, 960)  # qHD, padded to 1024
        assert not net.window_capable(648, 1024)  # 0x16 on re-entry
        assert not net.window_capable(576, 1025)  # padded to 1152: 0x16
        assert not net.window_capable(577, 1024)


# --------------------------------------------------------------------------
# Three-context scheduled window design
# --------------------------------------------------------------------------

class _ImmediatePipeline:
    def __init__(self):
        self.in_flight = False

    def submit(self, call):
        assert not self.in_flight
        call()
        self.in_flight = True

    def idle(self):
        return True

    def join(self):
        assert self.in_flight
        self.in_flight = False

    def drain(self):
        self.in_flight = False


class _ScheduledFakeRunner:
    def __init__(self, events):
        self.events = events
        self.loads = []
        self.zeroed = []
        self.model = type(
            "Model", (), {
                "_state": object(),
                "output_array": lambda self, name: name,
            })()

    def load_inputs(self, frame, gates=None, writes=None):
        self.loads.append((frame, gates, writes))

    def dispatch(self):
        self.events.append("main")

    def zero_last_push(self, line):
        self.zeroed.append(line)


class _ScheduledFakeSuite:
    def __init__(self):
        self.events = []
        self.pipeline = _ImmediatePipeline()
        self.runner = _ScheduledFakeRunner(self.events)
        self.models = {
            ("drain", 8): type(
                "PhaseModel", (), {
                    "output_array": lambda self, name: name,
                })(),
        }
        self.states = []

    def set_state(self, state):
        self.states.append(state)

    def dispatch(self, kind, start, frames):
        self.events.append((kind, start, len(frames)))


@pytest.mark.unit
class TestScheduledPhases:
    def test_asset_keeps_only_the_two_stable_specializations(self):
        assert P._FUNCTIONS == {
            ("fill", 0): "fill_00",
            ("drain", 8): "drain_08",
        }

    def test_schedule_vectors_are_byte_identical_to_the_mlx_spelling(self):
        """The 16-lane gate/write vectors are struct-packed rather than
        built through mx.array. mx.float16 IS IEEE 754 binary16 and the
        schedule's values are exactly 0.0/1.0, so the bytes must match
        the MLX spelling this replaced EXACTLY - the graph reads them as
        raw bytes, so a near-miss would be a silently wrong gate."""
        for values in ([0.0] * 16, [1.0] * 16, [0.0, 1.0] * 8,
                       [1.0 if index % 3 else 0.0 for index in range(16)]):
            vector = mx.array(values, dtype=mx.float16).reshape(1, 16, 1, 1)
            mx.eval(vector)
            expected = bytes(memoryview(mx.contiguous(vector)).cast("B"))
            assert len(expected) == 32
            assert bytes(A._vector_bytes(values)) == expected

    def test_phase_and_step_paths_share_one_vector_spelling(self):
        """One byte layout for both the window path and the ordinary
        step, so the two can never drift apart."""
        assert P._vector_bytes is A._vector_bytes

    def test_inner_phases_run_as_gated_main_steps(self, monkeypatch):
        monkeypatch.setattr(P, "_vector_bytes", tuple)
        suite = _ScheduledFakeSuite()
        frames = [bytes([index]) for index in range(20)]
        machine = P.WindowMachine(suite, frames, b"zero", lambda value: value)

        assert machine.advance(block=True)
        assert suite.events[0] == ("fill", 0, 8)
        assert suite.events[-1] == ("drain", 8, 8)
        assert suite.events[1:-1] == ["main"] * 20
        assert len(machine.outputs) == len(frames)

        gate_write_modes = [
            (gates is not None, writes is not None)
            for _frame, gates, writes in suite.runner.loads
        ]
        assert gate_write_modes == (
            [(False, True)] * 8
            + [(False, False)] * 4
            + [(True, True)] * 8
        )

    def test_blocking_progress_can_stop_at_each_materialized_output(
            self, monkeypatch):
        monkeypatch.setattr(P, "_vector_bytes", tuple)
        suite = _ScheduledFakeSuite()
        frames = [bytes([index]) for index in range(20)]
        machine = P.WindowMachine(suite, frames, b"zero", lambda value: value)

        assert not machine.advance_until_output(block=True)
        assert len(machine.outputs) == 1
        assert not machine.advance_until_output(block=True)
        assert len(machine.outputs) == 2
        while not machine.advance_until_output(block=True):
            pass
        assert len(machine.outputs) == len(frames)

    def test_warm_suite_load_does_not_reverify_or_require_a_canary(
            self, monkeypatch, tmp_path):
        stem = f"scheduled8-v{P.PHASE_GRAPH_VERSION}"
        package = tmp_path / f"{stem}.mlpackage"
        package.mkdir()
        (tmp_path / f"{stem}-verify.json").write_text("{}")
        (tmp_path / f"{stem}-replay.safetensors").touch()
        expected = object()

        monkeypatch.setattr(P, "_convert", lambda *_args: package)
        monkeypatch.setattr(P.runtime, "compile_package", lambda path: path)
        monkeypatch.setattr(P, "ScheduledPhaseSuite", lambda _path: expected)
        monkeypatch.setattr(
            P, "_verify_build",
            lambda *_args: pytest.fail("warm load unexpectedly reverified"))

        actual = P.build_suite({}, 4, 480, 640, tmp_path)
        assert actual is expected


# --------------------------------------------------------------------------
# Backend selection plumbing
# --------------------------------------------------------------------------

@pytest.mark.unit
class TestBackendSelection:
    def _parse(self, raw, settings=None):
        from kinovsr.processors.bsvd.factory import FACTORY
        from kinovsr.processors.capabilities import Capability
        from kinovsr.settings import Settings

        return FACTORY.parse_config(
            raw, capability=Capability.DENOISE, profile=None,
            settings=settings or Settings())

    def test_default_backend_is_mlx(self):
        assert self._parse({}).backend == "mlx"

    def test_stage_table_selects_ane(self):
        assert self._parse({"backend": "ane"}).backend == "ane"

    def test_settings_default_flows_when_table_is_silent(self):
        from kinovsr.settings import Settings

        settings = Settings(bsvd_backend="ane")
        assert self._parse({}, settings).backend == "ane"
        assert self._parse({"backend": "mlx"}, settings).backend == "mlx"

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="backend must be one of"):
            self._parse({"backend": "gpu"})

    def test_denoiser_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="unknown BSVD backend"):
            B.BsvdDenoiser(B.default_weights_path("c64"), backend="npu")

    def test_ane_backend_is_fp16_only(self):
        with pytest.raises(ValueError, match="fp16 only"):
            A.AneBSVD("unused.safetensors", dtype=mx.float32)

    def test_cli_flag_is_published_to_the_process_default(self, monkeypatch):
        from kinovsr.cli.args import build_parser
        from kinovsr.cli.config import assemble

        monkeypatch.delenv("BSVD_BACKEND", raising=False)
        args = build_parser().parse_args(
            ["--video", "clip.mp4", "--bsvd-backend", "ane"])
        invocation = assemble(args)
        assert invocation.settings.bsvd_backend == "ane"

    def test_cli_default_is_mlx(self, monkeypatch):
        from kinovsr.cli.args import build_parser
        from kinovsr.cli.config import assemble

        monkeypatch.delenv("BSVD_BACKEND", raising=False)
        args = build_parser().parse_args(["--video", "clip.mp4"])
        assert assemble(args).settings.bsvd_backend == "mlx"


# --------------------------------------------------------------------------
# Explicit lifecycle: worker ownership and cheap repeated resets
# --------------------------------------------------------------------------

class _FakeRunner:
    def __init__(self, error: BaseException | None = None):
        self.error = error
        self.dispatches = 0
        self.resets = 0
        self.reuse_flags: list[bool] = []

    def load_inputs(self, *_args):
        pass

    def dispatch(self):
        self.dispatches += 1
        if self.error is not None:
            raise self.error

    def reset(self, reuse_state: bool = False):
        self.resets += 1
        self.reuse_flags.append(reuse_state)


def _lifecycle_net(monkeypatch) -> A.AneBSVD:
    monkeypatch.setattr(B, "load_bsvd", lambda *_args, **_kwargs: ({}, 4))
    return A.AneBSVD("unused.safetensors", dtype=mx.float16)


@pytest.mark.unit
class TestAneLifecycle:
    def test_runner_uses_logical_zero_slots(self):
        runner = object.__new__(A.BsvdRunner)
        zero_multi = object()
        ring_multi = object()
        runner._zeros = [(None, None, zero_multi)]
        runner._rings = [[(None, None, ring_multi)] * 4]
        runner._valid = [[False] * 4]
        runner._cursor = [1]

        assert runner._input_multi(0, 1) is zero_multi
        runner._valid[0][1] = True
        assert runner._input_multi(0, 1) is ring_multi

        # The last pushed slot is cursor - 1; invalidation must not need to
        # touch its (potentially very large) physical buffer.
        runner._valid[0][0] = True
        runner.zero_last_push(0)
        assert runner._valid[0] == [False, True, False, False]

    def test_clean_reset_does_not_rezero_runner(self, monkeypatch):
        net = _lifecycle_net(monkeypatch)
        runner = _FakeRunner()
        net._runner = runner

        net.reset()
        assert runner.resets == 0
        net._dirty = True
        net.reset()
        assert runner.resets == 1
        net.reset()
        assert runner.resets == 1
        net.close()

    def test_scheduled_reset_reuses_the_state(self, monkeypatch):
        """Scheduled windows share one MLState across resets: allocating a
        fresh state per window (265 MB at 640x480) raced its deferred
        release into ANE status=0x16 failures. A later per-step stream
        must first zero the reused state, because the ordinary gated
        graph reads state through fill; continuous resets stay fresh."""
        net = _lifecycle_net(monkeypatch)
        runner = _FakeRunner()
        runner.model = type(
            "Model", (), {"_state": object(),
                          "reset_state": lambda self: None})()
        net._runner = runner

        class Suite:
            def __init__(self):
                self.states = []
                self.pipeline = type("P", (), {"drain": lambda self: None})()

            def set_state(self, state):
                self.states.append(state)

            def close(self):
                pass

        net._phase_suite = Suite()
        net._dirty = True
        net.reset()
        assert runner.reuse_flags == [True]
        assert net._state_needs_zero
        net.close()

        continuous = _lifecycle_net(monkeypatch)
        continuous_runner = _FakeRunner()
        continuous._runner = continuous_runner
        continuous._dirty = True
        continuous.reset()
        assert continuous_runner.reuse_flags == [False]
        assert not continuous._state_needs_zero
        continuous.close()

    def test_close_stops_worker_and_releases_runner(self, monkeypatch):
        net = _lifecycle_net(monkeypatch)
        runner = _FakeRunner()
        net._runner = runner

        net._submit(b"", b"", b"", emit=False)
        net.close()

        assert runner.dispatches == 1
        assert not net._pipeline.in_flight
        assert net._runner is None
        net.close()  # lifecycle cleanup is idempotent
        with pytest.raises(RuntimeError, match="closed"):
            net.step(None)

    def test_close_stops_worker_when_prediction_failed(self, monkeypatch):
        net = _lifecycle_net(monkeypatch)
        net._runner = _FakeRunner(RuntimeError("prediction failed"))
        net._submit(b"", b"", b"", emit=False)

        with pytest.raises(RuntimeError, match="prediction failed"):
            net.close()
        assert not net._pipeline.in_flight
        assert net._runner is None


@pytest.mark.unit
class TestDenoiserLifecycle:
    def test_close_delegates_and_releases_stream_buffers(self):
        class Net:
            def __init__(self):
                self.closes = 0

            def close(self):
                self.closes += 1

        from kinovsr.processors.feed_driver import WindowWavefront

        net = Net()
        denoiser = object.__new__(B.BsvdDenoiser)
        denoiser.net = net
        denoiser._wavefront = WindowWavefront()
        denoiser._nm = object()
        denoiser._tokens = [object()]
        denoiser._warm = [object()]
        denoiser._recent = [object()]
        denoiser._gop = None

        denoiser.close()
        denoiser.close()

        assert net.closes == 1
        assert denoiser.net is None
        assert denoiser._nm is None
        assert not denoiser._tokens
        assert not denoiser._warm
        assert not denoiser._recent

    def test_continuous_flush_yields_before_advancing_the_next_tail_step(self):
        """Drain work must stay interleaved with downstream consumption."""

        class Net:
            SHIFT_NUM = 0

        denoiser = object.__new__(B.BsvdDenoiser)
        denoiser.net = Net()
        denoiser._gop = None
        denoiser._warm = []
        denoiser._received = 2
        denoiser._emitted = 0
        events = []

        def step(_frame):
            index = denoiser._emitted
            events.append(f"step-{index}")
            denoiser._emitted += 1
            return [(f"out-{index}", f"token-{index}")]

        denoiser._step = step
        denoiser._reset_state = lambda: events.append("reset-state")
        denoiser._reset_conditioning = (
            lambda *, clear_debug: events.append(
                f"reset-conditioning-{clear_debug}"
            )
        )

        drain = denoiser.flush()
        assert events == []
        assert next(drain) == ("out-0", "token-0")
        assert events == ["step-0"]
        events.append("downstream")
        assert next(drain) == ("out-1", "token-1")
        assert events == ["step-0", "downstream", "step-1"]
        assert list(drain) == []
        assert events[-2:] == ["reset-state", "reset-conditioning-False"]

    def test_gop_window_uses_backend_async_path_when_available(self):
        """Windows flow through the WindowWavefront one deep: submitting
        window k+1 completes and emits window k, the net resets only
        between windows (never mid-flight), and the flush barrier emits
        the last window."""
        from kinovsr.processors.feed_driver import WindowWavefront

        class Handle:
            def __init__(self, frames):
                self._results = [frame + 100 for frame in frames]
                self.outputs = []

            def advance(self, block=False):
                if block:
                    self.outputs = list(self._results)
                    return True
                return False

        class Net:
            def __init__(self):
                self.resets = 0
                self.windows = []

            def reset(self):
                self.resets += 1

            def begin_window(self, frames):
                self.windows.append(list(frames))
                return Handle(frames)

        denoiser = object.__new__(B.BsvdDenoiser)
        denoiser.net = Net()
        denoiser._tracker = None
        denoiser._wavefront = WindowWavefront()
        denoiser._pulse_gain = lambda _x, **_kwargs: 1.0
        denoiser._with_nm = lambda x, _nm, gain: x
        denoiser._emit = lambda value, token: (value, token)

        first = list(denoiser._run_window(
            list(range(16)), [f"a{i}" for i in range(16)], 3, 7))
        assert first == []                    # window 1 is in flight
        assert denoiser.net.resets == 0       # never reset mid-flight

        second = list(denoiser._run_window(
            list(range(50, 66)), [f"b{i}" for i in range(16)], 0, 2))
        assert second == [(103, "a3"), (104, "a4"),
                          (105, "a5"), (106, "a6")]
        assert denoiser.net.resets == 1

        tail = list(denoiser._wavefront.barrier())
        assert tail == [(150, "b0"), (151, "b1")]
        assert denoiser.net.resets == 2
        assert denoiser.net.windows == [list(range(16)),
                                        list(range(50, 66))]

    def test_short_window_fallback_barriers_the_wavefront_first(self):
        """A window too short for the phase path emits inline through the
        per-step loop; the async window still in flight must complete and
        emit BEFORE it - its frames precede this window's - and the net
        must never be reset under an in-flight window (that inverted
        emission order in the field and corrupted shared runner state)."""
        from kinovsr.processors.feed_driver import WindowWavefront

        class Handle:
            def __init__(self, frames):
                self._results = [frame + 100 for frame in frames]
                self.outputs = []

            def advance(self, block=False):
                if block:
                    self.outputs = list(self._results)
                    return True
                return False

        class Net:
            SHIFT_NUM = 0

            def __init__(self):
                self.events = []

            def reset(self):
                self.events.append("reset")

            def begin_window(self, frames):
                self.events.append("begin")
                return Handle(frames)

            def step(self, x):
                return x + 200

        denoiser = object.__new__(B.BsvdDenoiser)
        denoiser.net = Net()
        denoiser._tracker = None
        denoiser._wavefront = WindowWavefront()
        denoiser._pulse_gain = lambda _x, **_kwargs: 1.0
        denoiser._with_nm = lambda x, _nm, gain: x
        denoiser._emit = lambda value, token: (value, token)

        first = list(denoiser._run_window(
            list(range(16)), [f"a{i}" for i in range(16)], 0, 16))
        assert first == []                    # async window in flight

        short_iter = denoiser._run_window([70, 71], ["b0", "b1"], 0, 2)
        first_deferred = next(short_iter)
        assert first_deferred == (100, "a0")
        # The prior window reaches downstream before the short fallback starts
        # stepping.
        assert denoiser.net.events == ["begin", "reset"]
        short = [first_deferred, *short_iter]
        # The deferred async window's emissions come FIRST, then the
        # short window's inline emissions.
        assert short[:16] == [(100 + i, f"a{i}") for i in range(16)]
        assert short[16:] == [(270, "b0"), (271, "b1")]
        # finalize's reset (completing the async window) precedes the
        # sync path's own enter/exit resets; nothing resets while a
        # window flies.
        assert denoiser.net.events == ["begin", "reset", "reset", "reset"]


# --------------------------------------------------------------------------
# End-to-end parity against the MLX net (slow: converts a real model)
# --------------------------------------------------------------------------

def _real_frames(count: int, channels: int, height: int, width: int):
    base = mx.random.uniform(shape=(1, height, width, channels),
                             key=mx.random.key(20260719))
    frames = []
    for index in range(count):
        noise = mx.random.uniform(shape=(1, height, width, channels),
                                  key=mx.random.key(500 + index))
        frame = mx.clip(base * 0.85 + noise * 0.15 + index * 0.011, 0.0, 1.0)
        frame = frame.astype(mx.float16)
        mx.eval(frame)
        frames.append(frame)
    return frames


@pytest.fixture(scope="module")
def _module_cache(tmp_path_factory):
    """One conversion cache for the whole module: BSVD ANE conversion at
    96x128 costs tens of seconds, so the streams below share it."""
    import os

    from kinovsr.settings import _reset_default_settings as reset_settings

    cache = tmp_path_factory.mktemp("bsvd-ane-cache")
    previous = os.environ.get("KINOVSR_CACHE_DIR")
    os.environ["KINOVSR_CACHE_DIR"] = str(cache)
    reset_settings()
    A._VERIFIED.clear()
    yield cache
    if previous is None:
        os.environ.pop("KINOVSR_CACHE_DIR", None)
    else:
        os.environ["KINOVSR_CACHE_DIR"] = previous
    reset_settings()
    A._VERIFIED.clear()


@pytest.fixture(scope="module")
def ane_net(_module_cache):
    pytest.importorskip("CoreML")
    weights = B.default_weights_path("c64")
    if not weights.is_file():
        pytest.skip(f"bsvd weights not available at {weights}")
    net = A.AneBSVD(weights, dtype=mx.float16)
    try:
        net._ensure_runner(96, 128)
    except Exception as exc:  # noqa: BLE001 - environment, not correctness
        pytest.skip(f"BSVD ANE engine unavailable here: {exc}")
    net.reset()
    return net


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_weights
class TestAneParity:
    def _drive_both(self, ane_net, length: int):
        """Drive both nets; the ANE emissions lag the reference by exactly
        one step (the in-flight dispatch), with identical content order."""
        reference = B.BSVD(B.default_weights_path("c64"), dtype=mx.float32)
        frames = _real_frames(length, ane_net.input_channels, 96, 128)
        ane_net.reset()
        reference.reset()
        ane_outs, ref_outs = [], []
        for index, frame in enumerate(frames + [None] * 17):
            ane_out = ane_net.step(frame)
            ref_out = reference.step(
                None if frame is None else frame.astype(mx.float32))
            if ane_out is not None:
                ane_outs.append((index, ane_out))
            if ref_out is not None:
                ref_outs.append((index, ref_out))
        ane_net.reset()
        assert len(ane_outs) == len(ref_outs) == length
        assert ([i for i, _ in ane_outs]
                == [i + 1 for i, _ in ref_outs]), "one-step dispatch lag"
        pairs = []
        for (_, ane_out), (_, ref_out) in zip(ane_outs, ref_outs,
                                              strict=True):
            delta = mx.abs(ane_out.astype(mx.float32) - ref_out)
            mx.eval(delta)
            pairs.append((float(delta.mean()), float(delta.max())))
        return pairs

    def test_full_stream_matches_the_product_fp32_net(self, ane_net):
        pairs = self._drive_both(ane_net, 24)
        means = [mean for mean, _ in pairs]
        # Doc-20 measured the gated schedule at ~3e-4 mean against the
        # product's own fp32 output, and the UNGATED failure mode at
        # 9.5e-4 and up - the bound sits between them.
        assert means[0] < 8e-4, f"first emitted frame off: {means[0]:.3e}"
        assert max(means) < 8e-4, f"worst frame {max(means):.3e}"

    def test_sub_fill_clip_emits_entirely_through_the_drain(self, ane_net):
        pairs = self._drive_both(ane_net, 4)
        means = [mean for mean, _ in pairs]
        assert max(means) < 8e-4, f"worst frame {max(means):.3e}"

    @pytest.mark.usefixtures("_module_cache")
    def test_cif_geometry_runs_padded_and_matches(self):
        """352x288 (CIF) is the geometry that fails UNPADDED with the ANE
        status=0x1d alignment error; through the reflect-pad path it must
        run and match the fp32 reference. The right band differs by
        boundary CONTEXT (reflected content vs the frame edge the MLX
        path sees), so it gets its own bound; the interior must sit at
        the same ~3e-4 parity as aligned geometries (measured worst
        3.2e-4 interior, 3.5e-3 band, 9.0e-4 full frame)."""
        weights = B.default_weights_path("c64")
        if not weights.is_file():
            pytest.skip(f"bsvd weights not available at {weights}")
        net = A.AneBSVD(weights, dtype=mx.float16)
        reference = B.BSVD(weights, dtype=mx.float32)
        # Four real frames exercise CIF padding, emission, and drain. The
        # aligned 24-frame test above owns the full recurrent schedule.
        frames = _real_frames(4, net.input_channels, 288, 352)
        try:
            net._ensure_runner(288, 352)
        except Exception as exc:  # noqa: BLE001 - environment, not correctness
            pytest.skip(f"BSVD ANE engine unavailable here: {exc}")
        assert net._padded_width == 384
        net.reset()
        ane_outs, ref_outs = [], []
        for frame in frames + [None] * 17:
            ane_out = net.step(frame)
            ref_out = reference.step(
                None if frame is None else frame.astype(mx.float32))
            if ane_out is not None:
                ane_outs.append(ane_out)
            if ref_out is not None:
                ref_outs.append(ref_out)
        full, interior, band = [], [], []
        for ane_out, ref_out in zip(ane_outs, ref_outs, strict=True):
            assert ane_out.shape == ref_out.shape
            delta = mx.abs(ane_out.astype(mx.float32) - ref_out)
            mx.eval(delta)
            full.append(float(delta.mean()))
            interior.append(float(delta[:, :, :352 - 64, :].mean()))
            band.append(float(delta[:, :, 352 - 64:, :].mean()))
        assert len(full) == 4
        assert max(interior) < 8e-4, f"interior {max(interior):.3e}"
        assert max(band) < 8e-3, f"right band {max(band):.3e}"
        assert max(full) < 2e-3, f"full frame {max(full):.3e}"

    @pytest.mark.usefixtures("_module_cache")
    def test_gop_window_matches_the_product(self):
        """The phase-specialized window path (fill/drain functions + the
        pipelined steady middle) must reproduce the product's reset-window
        semantics: reset, N frames, 16 drains. Emissions are compared in
        order against the fp32 reference; 21 frames exercises the steady
        middle across a non-multiple-of-eight window."""
        weights = B.default_weights_path("c64")
        if not weights.is_file():
            pytest.skip(f"bsvd weights not available at {weights}")
        net = A.AneBSVD(weights, dtype=mx.float16)
        frames = _real_frames(21, net.input_channels, 96, 128)
        try:
            outputs = net.run_window(frames)
        except Exception as exc:  # noqa: BLE001 - environment, not correctness
            net.close()
            pytest.skip(f"BSVD ANE phase suite unavailable here: {exc}")
        reference = B.BSVD(weights, dtype=mx.float32)
        reference.reset()
        expected = []
        for frame in frames + [None] * 16:
            out = reference.step(
                None if frame is None else frame.astype(mx.float32))
            if out is not None:
                expected.append(out)
        assert len(outputs) == len(expected) == 21
        means = []
        for got, want in zip(outputs, expected, strict=True):
            assert got.shape == want.shape
            delta = mx.abs(got.astype(mx.float32) - want)
            mx.eval(delta)
            means.append(float(delta.mean()))
        net.close()
        assert max(means) < 8e-4, f"worst frame {max(means):.3e}"

    def test_denoiser_backend_parity(self, ane_net):
        ane = B.BsvdDenoiser(strength=0.4, backend="ane")
        mlx_ref = B.BsvdDenoiser(strength=0.4, backend="mlx")
        frames = [f[0].astype(mx.float32)[..., :3]
                  for f in _real_frames(8, 3, 96, 128)]
        for frame in frames:
            mx.eval(frame)
        ane_out, ref_out = [], []
        for index, frame in enumerate(frames):
            ane_out += ane.feed(frame, token=index)
            ref_out += mlx_ref.feed(frame, token=index)
        ane_out.extend(ane.flush())
        ref_out.extend(mlx_ref.flush())
        assert [token for _, token in ane_out] == list(range(8))
        assert [token for _, token in ref_out] == list(range(8))
        deltas = [float(mx.abs(a - r).mean().item())
                  for (a, _), (r, _) in zip(ane_out, ref_out, strict=True)]
        # Both backends run fp16 here, each ~3e-4 from fp32, so their
        # mutual distance stays within a few fp16 quanta.
        assert max(deltas) < 2e-3, f"worst frame {max(deltas):.3e}"
