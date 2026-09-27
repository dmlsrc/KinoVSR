"""BSVD MPSGraph backend: selection plumbing and (slow) parity against the
MLX net through fill, steady state, and drain."""

import gc
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from kinovsr.native import mpsgraph_state as mgs
from kinovsr.processors import bsvd as B
from kinovsr.processors.bsvd import cli_options, factory, mps, mps_phases
from kinovsr.processors.bsvd.mps import MpsGraphBSVD
from kinovsr.processors.bsvd.mps_phases import ScheduledMpsPhaseSuite
from kinovsr.processors.bsvd.net import _pad_width_reflect

# --------------------------------------------------------------------------
# Selection plumbing
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestBackendSelection:
    def test_factory_and_cli_agree_on_the_backend_set(self):
        row = next(opt for opt in cli_options.BSVD_OPTIONS if opt.flag == "--bsvd-backend")
        assert set(row.choices) == factory._BACKENDS
        assert "mpsgraph" in factory._BACKENDS

    def test_unknown_backend_is_rejected_with_the_full_choice_list(self):
        with pytest.raises(ValueError, match="mpsgraph"):
            B.BsvdDenoiser(backend="coreml")

    def test_fp32_is_refused(self):
        with pytest.raises(ValueError, match="fp16 only"):
            MpsGraphBSVD(B.default_weights_path("c64"), dtype=mx.float32)

    def test_unaligned_geometry_is_refused(self):
        net = MpsGraphBSVD(B.default_weights_path("c64"))
        frame = mx.zeros((1, 94, 128, net.input_channels), dtype=mx.float16)
        with pytest.raises(ValueError, match="divisible by four"):
            net.step(frame)

    def test_warm_scheduled_preheat_attaches_in_background(self, monkeypatch, tmp_path):
        executable = object()
        net = object.__new__(MpsGraphBSVD)
        net._closed = False
        net._preheat = None
        net._preheat_pool = None
        net._preheated_geometry = None
        net._preheated_executable = None
        net._phase_suite = None
        net.window_capable = lambda height, width: True
        net._executable_cache = lambda label, height, width: tmp_path
        monkeypatch.setattr(mgs, "stateful_cache_ready", lambda path: True)
        monkeypatch.setattr(
            "kinovsr.processors.bsvd.mps.preload_stateful_executable",
            lambda owner, height, width: executable,
        )

        net.preheat(128, 256, scheduled=True)
        net._join_preheat()

        assert net._preheated_geometry == (128, 256)
        assert net._preheated_executable is executable

    def test_scheduled_cache_uses_one_direct_lifecycle_entry(self, monkeypatch, tmp_path):
        captured = {}
        executable = object()

        def executable_cache(label, height, width):
            captured["cache"] = (label, height, width)
            return tmp_path

        net = SimpleNamespace(
            _ane_fw_to_fw_signal=False,
            _ane_late_latch=False,
            _executable_cache=executable_cache,
        )

        def compile_stateful(factories, states, **kwargs):
            captured["entries"] = tuple(factories)
            captured["states"] = states
            captured["kwargs"] = kwargs
            return executable

        monkeypatch.setattr(mgs, "compile_stateful_direct", compile_stateful)
        states = (object(),)

        result = mps_phases._compile_stateful_executable(net, 480, 640, states)

        assert result is executable
        assert captured["entries"] == ("generic1",)
        assert captured["cache"] == ("scheduled-stateful-generic1-direct-v3", 480, 640)
        assert captured["states"] is states
        assert captured["kwargs"]["cache_directory"] == tmp_path

    def test_packed_state_storage_uses_four_balanced_ane_ports(self):
        from kinovsr.processors.bsvd.mps_layout import _net_keys

        prefixes = ("first", "second")
        weights = {}
        for prefix in prefixes:
            keys = _net_keys(prefix)
            for local, name in enumerate(mps_phases._UNIT_KEYS):
                channels = 128 if local < 2 or local > 5 else 256
                weights[keys[name] + ".weight"] = SimpleNamespace(shape=(channels,))
        states = mps_phases._state_specs(
            SimpleNamespace(_prefixes=prefixes, _weights=weights),
            480,
            640,
        )

        assert tuple(state.name for state in states) == (
            "state_group_0",
            "state_group_1",
            "state_group_2",
            "state_group_3",
        )
        assert tuple(state.storage_shape for state in states) == (
            (8, 144, 240, 160),
            (4, 144, 240, 160),
            (8, 144, 240, 160),
            (4, 144, 240, 160),
        )

    def test_window_reset_keeps_concrete_tensor_data_views(self):
        resets = []
        suite = object.__new__(ScheduledMpsPhaseSuite)
        suite.executable = SimpleNamespace(reset=lambda: resets.append(True))
        view = object()
        suite._views = {(1, "skip", 2): view}

        suite.reset()

        assert resets == [True]
        assert suite._views == {(1, "skip", 2): view}

    @pytest.mark.parametrize("count", [16, 17, 18, 19, 20, 23, 24, 63])
    def test_one_step_window_actions_cover_fill_and_drain_exactly(self, count):
        actions = ScheduledMpsPhaseSuite._actions(list(range(count)))
        assert all(len(action.frames) == len(action.records) == 1 for action in actions)
        assert sum(record.out_real for action in actions for record in action.records) == count
        flattened = [frame for action in actions for frame in action.frames]
        assert flattened[:count] == list(range(count))
        assert all(frame is None for frame in flattened[count:])
        assert len(flattened) >= count + 16
        assert len(flattened) == count + 16


# --------------------------------------------------------------------------
# Geometry envelope: widths the Neural Engine aborts on
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestGeometry:
    @pytest.mark.parametrize(
        ("width", "graph_width"),
        [
            (128, 128),
            (132, 132),
            (144, 144),
            (160, 256),
            (320, 384),
            (336, 336),
            (352, 384),
            (480, 512),
            (720, 720),
            (960, 1024),
            (1440, 1536),
            (1920, 1920),
        ],
    )
    def test_only_multiples_of_32_off_the_128_grid_are_padded(self, width, graph_width):
        assert mps._graph_width(width) == graph_width

    def test_padded_widths_run_their_windows_through_step(self):
        net = object.__new__(MpsGraphBSVD)

        assert net.window_capable(240, 336)
        assert net.window_capable(480, 640)
        assert not net.window_capable(240, 320)  # the graph pads it to 384
        assert not net.window_capable(540, 960)  # qHD, padded to 1024

    def test_frames_too_narrow_to_pad_are_refused(self):
        net = MpsGraphBSVD(B.default_weights_path("c64"))
        frame = mx.zeros((1, 96, 64, net.input_channels), dtype=mx.float16)
        with pytest.raises(ValueError, match="reflect padding"):
            net.step(frame)


# --------------------------------------------------------------------------
# Window route fallback: the schedule entry cannot be built on this system
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestWindowRouteFallback:
    def test_a_refused_window_runs_through_the_per_step_path(self):
        # Where the direct schedule entry cannot be built (MPSGraph does not
        # place the program as one mpsx.ane procedure), every --gop-align
        # window used to fail the run instead.
        from kinovsr.processors.feed_driver import WindowRouteUnavailable, WindowWavefront

        class Net:
            SHIFT_NUM = 0
            MIN_WINDOW_FRAMES = 1

            def __init__(self):
                self.events = []

            def reset(self):
                self.events.append("reset")

            def window_capable(self, height, width):
                return True

            def begin_window(self, frames):
                self.events.append("begin")
                raise WindowRouteUnavailable("no schedule entry here")

            def step(self, x):
                self.events.append("step")
                return x + 200

        denoiser = object.__new__(B.BsvdDenoiser)
        denoiser.net = Net()
        denoiser._tracker = None
        denoiser._wavefront = WindowWavefront()
        denoiser._pulse_gain = lambda _x, **_kwargs: 1.0
        denoiser._with_nm = lambda x, _nm, gain: x
        denoiser._emit = lambda value, token: (value, token)
        frames = [mx.zeros((1, 4, 4, 4)) + i for i in range(3)]

        out = list(denoiser._run_window(frames, ["a", "b", "c"], 0, 3))
        assert [(float(value[0, 0, 0, 0]), token) for value, token in out] == [
            (200.0, "a"),
            (201.0, "b"),
            (202.0, "c"),
        ]
        assert denoiser.net.events == ["begin", "reset", "step", "step", "step", "reset"]

    def test_the_net_refuses_every_window_once_the_entry_is_unavailable(self, monkeypatch):
        from kinovsr.processors.feed_driver import WindowRouteUnavailable

        monkeypatch.setattr(MpsGraphBSVD, "_schedule_entry_unavailable", True)
        net = object.__new__(MpsGraphBSVD)
        net._closed = False
        net._dirty = False
        net._pending = None
        assert not net.window_capable(480, 640)
        with pytest.raises(WindowRouteUnavailable):
            net.begin_window([mx.zeros((1, 480, 640, 4), dtype=mx.float16)])


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_weights
def test_gop_aligned_windows_run_at_a_window_capable_geometry(tmp_path, monkeypatch):
    # 256x128 is inside the schedule-window envelope. Where the direct entry
    # could not be built, the run used to fail at the first window; on
    # macOS 27 the windows run on the entry itself.
    import av

    from kinovsr.pipeline import run_file
    from kinovsr.settings import Settings
    from kinovsr.settings import _reset_default_settings as reset_settings

    # At this size the graphs cache their executables: keep them in tmp_path.
    monkeypatch.setenv("KINOVSR_CACHE_DIR", str(tmp_path / "cache"))
    reset_settings()
    monkeypatch.setattr(MpsGraphBSVD, "_schedule_entry_unavailable", False, raising=False)
    source = tmp_path / "clip.mp4"
    out = av.open(str(source), "w")
    vs = out.add_stream("mpeg4", rate=25)
    vs.width, vs.height, vs.pix_fmt = 256, 128, "yuv420p"
    for index in range(20):
        frame = av.VideoFrame(256, 128, "gray")
        frame.planes[0].update(bytes([index * 8]) * (256 * 128))
        for pkt in vs.encode(frame.reformat(format="yuv420p")):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()

    config = {"pipeline": ["d"], "d": {"processor": "bsvd", "backend": "mpsgraph"}}
    try:
        result = run_file(
            config, video=source, output=tmp_path / "out.mp4", settings=Settings(), gop_align=True
        )
    finally:
        reset_settings()
    assert result.frames_out == 20
    assert (tmp_path / "cache").is_dir()
    assert not MpsGraphBSVD._schedule_entry_unavailable


# --------------------------------------------------------------------------
# End-to-end parity with the MLX reference
# --------------------------------------------------------------------------


def _stream(height: int, width: int, count: int, sigma: float) -> list:
    keys = mx.random.split(mx.random.key(3), count + 1)
    base = mx.random.uniform(shape=(1, height + 8, width + 8, 3), key=keys[0])
    frames = []
    for t in range(count):
        crop = base[:, t % 8 : t % 8 + height, t % 8 : t % 8 + width, :]
        noisy = mx.clip(
            crop + mx.random.normal((1, height, width, 3), key=keys[t + 1]) * sigma, 0.0, 1.0
        )
        plane = mx.full((1, height, width, 1), sigma)
        frames.append(mx.concatenate([noisy, plane], axis=-1).astype(mx.float16))
    return frames


def _own_mpsgraph_archives() -> set[str]:
    """This process's MPSGraph archives in the per-user temp directory."""
    temp = subprocess.run(
        ["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True, check=True
    ).stdout.strip()
    root = Path(temp) / "com.apple.MetalPerformanceShadersGraph"
    return {path.name for path in root.glob(f"mpsgraph-{os.getpid()}-*")}


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_weights
class TestParityWithTheMlxReference:
    def test_fill_steady_and_drain_match(self):
        """The full emitted sequence must match the reference frame for
        frame (the shared schedule mirror guarantees the count and order;
        the backend's one-in-flight dispatch shifts WHEN each frame
        surfaces by exactly one step, which the token plumbing absorbs
        via ``SHIFT_NUM``), agreeing to accelerator-precision tolerances.

        Tolerance basis: measured worst-case against the MLX fp16 net at
        96x128 over 24 frames was max 0.0015 / mean 2.1e-4 (fp16 ANE vs
        fp16 GPU numerics); the gates below carry ~4x headroom.
        """
        height, width, count = 96, 128, 24
        frames = _stream(height, width, count, sigma=30.0 / 255.0)
        weights = B.default_weights_path("c64")
        reference = B.BSVD(weights, dtype=mx.float16)
        net = MpsGraphBSVD(weights)
        assert net.SHIFT_NUM == reference.SHIFT_NUM + 1

        expected, actual = [], []
        for t in range(count + net.SHIFT_NUM + 4):
            x = frames[t] if t < count else None
            out = reference.step(x)
            if out is not None:
                expected.append(out)
            out = net.step(x)
            if out is not None:
                actual.append(out)
        assert len(expected) == len(actual) == count

        worst_max = worst_mean = 0.0
        for a, b in zip(expected, actual, strict=True):
            delta = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
            worst_max = max(worst_max, mx.max(delta).item())
            worst_mean = max(worst_mean, mx.mean(delta).item())
        assert worst_max < 6e-3, f"max abs {worst_max}"
        assert worst_mean < 1e-3, f"mean abs {worst_mean}"

        with pytest.raises(RuntimeError, match="changed resolution"):
            net.step(mx.zeros((1, height + 4, width, net.input_channels), dtype=mx.float16))
        net.close()

    def test_aborting_width_runs_padded_and_matches(self):
        """160 px is a multiple of 32 but not of 128, so the unpadded graph's
        first inference aborted the process. It now
        runs 256 wide on reflect-padded frames, cropped back, and must match
        the fp32 net run on the same padded frames at the aligned tolerance.
        The fp32 reference keeps the bound width-independent: the MLX fp16
        net's own error doubles at widths of 176 and more.
        """
        height, width, count = 96, 160, 20
        frames = _stream(height, width, count, sigma=30.0 / 255.0)
        weights = B.default_weights_path("c64")
        reference = B.BSVD(weights, dtype=mx.float32)
        net = MpsGraphBSVD(weights)

        expected, actual = [], []
        for t in range(count + net.SHIFT_NUM + 4):
            x = frames[t] if t < count else None
            padded = None if x is None else _pad_width_reflect(x.astype(mx.float32), 256)
            out = reference.step(padded)
            if out is not None:
                expected.append(out[:, :, :width])
            out = net.step(x)
            if out is not None:
                actual.append(out)
        assert net._graph_width == 256
        assert len(expected) == len(actual) == count

        worst_max = worst_mean = 0.0
        for a, b in zip(expected, actual, strict=True):
            assert b.shape == a.shape
            assert b.shape[2] == width
            delta = mx.abs(a - b.astype(mx.float32))
            worst_max = max(worst_max, mx.max(delta).item())
            worst_mean = max(worst_mean, mx.mean(delta).item())
        assert worst_max < 6e-3, f"max abs {worst_max}"
        assert worst_mean < 1e-3, f"mean abs {worst_mean}"
        net.close()

    def test_schedule_window_runs_on_the_direct_entry_and_matches(self, tmp_path, monkeypatch):
        """macOS 27 places the ANE region as an mpsx.ane procedure, which the
        direct entry refused, so every --gop-align window ran through
        step(). A recurrent window now runs on the entry and must match the
        fp32 net at the aligned tolerance."""
        from kinovsr.settings import _reset_default_settings as reset_settings

        # At this size the graphs cache their executables: keep them in tmp_path.
        monkeypatch.setenv("KINOVSR_CACHE_DIR", str(tmp_path / "cache"))
        reset_settings()
        monkeypatch.setattr(MpsGraphBSVD, "_schedule_entry_unavailable", False, raising=False)
        height, width, count = 128, 256, 20
        frames = _stream(height, width, count, sigma=30.0 / 255.0)
        weights = B.default_weights_path("c64")
        archives = _own_mpsgraph_archives()
        net = MpsGraphBSVD(weights)
        try:
            machine = net.begin_window(frames)
            machine.advance(block=True)
            actual = list(machine.outputs)
        finally:
            net.close()
            reset_settings()
        del machine, net
        gc.collect()
        # The entry was built on this pool-less main thread; its build
        # executables stayed alive, and MPSGraph kept their temp archives.
        assert _own_mpsgraph_archives() <= archives

        reference = B.BSVD(weights, dtype=mx.float32)
        expected = []
        for t in range(count + reference.SHIFT_NUM + 4):
            out = reference.step(frames[t].astype(mx.float32) if t < count else None)
            if out is not None:
                expected.append(out)
        assert len(expected) == len(actual) == count

        worst_max = worst_mean = 0.0
        for a, b in zip(expected, actual, strict=True):
            assert b.shape == a.shape
            delta = mx.abs(a - b.astype(mx.float32))
            worst_max = max(worst_max, mx.max(delta).item())
            worst_mean = max(worst_mean, mx.mean(delta).item())
        assert worst_max < 6e-3, f"max abs {worst_max}"
        assert worst_mean < 1e-3, f"mean abs {worst_mean}"

    def test_reset_restarts_the_stream_cleanly(self):
        height, width, count = 96, 128, 18
        frames = _stream(height, width, count, sigma=30.0 / 255.0)
        net = MpsGraphBSVD(B.default_weights_path("c64"))

        def run_stream() -> list:
            outs = []
            for t in range(count + net.SHIFT_NUM + 4):
                out = net.step(frames[t] if t < count else None)
                if out is not None:
                    outs.append(out)
            return outs

        first = run_stream()
        net.reset()
        second = run_stream()
        assert len(first) == len(second) == count
        for a, b in zip(first, second, strict=True):
            assert mx.array_equal(a, b), "reset left state behind"
        net.close()
