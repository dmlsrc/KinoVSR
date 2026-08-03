"""BSVD MPSGraph backend: selection plumbing and (slow) parity against the
MLX net through fill, steady state, and drain."""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinovsr.processors import bsvd as B
from kinovsr.processors.bsvd import cli_options, factory
from kinovsr.processors.bsvd.mps import MpsGraphBSVD
from kinovsr.processors.bsvd.mps_phases import ScheduledMpsPhaseSuite

# --------------------------------------------------------------------------
# Selection plumbing
# --------------------------------------------------------------------------

@pytest.mark.unit
class TestBackendSelection:
    def test_factory_and_cli_agree_on_the_backend_set(self):
        row = next(opt for opt in cli_options.BSVD_OPTIONS
                   if opt.flag == "--bsvd-backend")
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

    @pytest.mark.parametrize("count", [16, 17, 18, 19, 20, 23, 24, 63])
    def test_four_step_window_actions_cover_fill_and_drain_exactly(
        self, count
    ):
        actions = ScheduledMpsPhaseSuite._actions(list(range(count)))
        assert all(
            len(action.frames) == len(action.records) == 4
            for action in actions
        )
        assert sum(
            record.out_real
            for action in actions
            for record in action.records
        ) == count
        flattened = [
            frame for action in actions for frame in action.frames]
        assert flattened[:count] == list(range(count))
        assert all(frame is None for frame in flattened[count:])
        assert len(flattened) >= count + 16
        assert len(flattened) < count + 20


# --------------------------------------------------------------------------
# End-to-end parity with the MLX reference
# --------------------------------------------------------------------------

def _stream(height: int, width: int, count: int, sigma: float) -> list:
    keys = mx.random.split(mx.random.key(3), count + 1)
    base = mx.random.uniform(shape=(1, height + 8, width + 8, 3),
                             key=keys[0])
    frames = []
    for t in range(count):
        crop = base[:, t % 8:t % 8 + height, t % 8:t % 8 + width, :]
        noisy = mx.clip(
            crop + mx.random.normal((1, height, width, 3),
                                    key=keys[t + 1]) * sigma, 0.0, 1.0)
        plane = mx.full((1, height, width, 1), sigma)
        frames.append(mx.concatenate([noisy, plane],
                                     axis=-1).astype(mx.float16))
    return frames


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
            net.step(mx.zeros((1, height + 4, width, net.input_channels),
                              dtype=mx.float16))
        net.close()

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
