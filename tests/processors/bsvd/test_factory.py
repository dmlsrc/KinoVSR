"""BSVD factory: the stateful causal-streaming proving processor."""

from fractions import Fraction

import pytest

from kinovsr.processors import (
    Boundary,
    BoundaryKind,
    Capability,
    FrameUnit,
    Geometry,
    PipelineContext,
    StreamSpec,
    TemporalMode,
    TimelineSpec,
    frame_spec_for_matrix,
    get_factory,
)
from kinovsr.processors.bsvd.factory import FACTORY
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw, profile=None):
    return FACTORY.parse_config(
        raw, capability=Capability.DENOISE, profile=profile,
        settings=SETTINGS)


def stream() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(64, 48)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


class TestParse:
    def test_defaults_and_profile(self):
        config = parse({})
        assert (config.variant, config.strength, config.dtype) == (
            "c64", 0.5, "float16")
        assert parse({}, profile="c32").variant == "c32"

    def test_strength_bounds(self):
        assert parse({"strength": 0.03}).strength == 0.03
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            parse({"strength": 1.5})

    def test_unknown_key_suggests(self):
        with pytest.raises(ValueError, match="did you mean 'strength'"):
            parse({"stregnth": 0.2})

    def test_declares_centered_stateful(self):
        # Bidirectional within its own buffer: CENTERED per the
        # taxonomy, with the 16-frame delay as the future reach.
        spec = FACTORY.capabilities[Capability.DENOISE]
        assert spec.temporal_mode is TemporalMode.CENTERED
        assert spec.temporal_radius == 16
        assert spec.stateful
        assert get_factory("bsvd") is FACTORY

    def test_noise_map_defaults_to_a_constant_no_op(self):
        from kinovsr.processors.conditioning import NoiseMapConfig

        assert parse({}).noise_map == NoiseMapConfig()

    def test_noise_map_keys_parse_onto_the_config(self):
        config = parse({"noise_map": "auto", "noise_map_gain": 1.2,
                        "noise_map_pulse": True})
        assert config.noise_map.mode == "auto"
        assert config.noise_map.gain == 1.2
        assert config.noise_map.pulse is True

    def test_bad_noise_map_value_is_rejected_at_parse(self):
        with pytest.raises(ValueError, match="noise_map must"):
            parse({"noise_map": "blur"})


@pytest.mark.requires_weights
@pytest.mark.integration
class TestStreaming:
    @staticmethod
    def run(frames, config=None):
        import mlx.core as mx

        from kinovsr.pipeline import resolve_pipeline, run_plan

        table = {"processor": "bsvd", "strength": 0.3}
        table.update(config or {})
        try:
            plan = resolve_pipeline(
                {"pipeline": ["den"], "den": table},
                input_spec=stream(), settings=SETTINGS)
            units = [
                FrameUnit(payload=mx.random.uniform(
                    shape=(48, 64, 3)).astype(mx.float32) * 0.5 + 0.25,
                    pts=i * 960, duration=960)
                if isinstance(i, int) else i
                for i in frames
            ]
            context = PipelineContext(settings=SETTINGS)
            return list(run_plan(plan, units, context))
        except FileNotFoundError as exc:
            pytest.skip(f"bsvd weights not available: {exc}")

    def test_delayed_outputs_keep_their_input_timestamps(self):
        out = self.run(range(6))
        # every input comes back out (the flush drains the 16-step delay),
        # bound to its original pts, in order
        assert [u.pts for u in out] == [i * 960 for i in range(6)]
        assert all(u.payload.shape == (48, 64, 3) for u in out)

    def test_hard_cut_mid_stream_loses_nothing(self):
        import mlx.core as mx

        cut_unit = FrameUnit(
            payload=mx.random.uniform(shape=(48, 64, 3)).astype(mx.float32)
            * 0.5 + 0.25,
            pts=3 * 960, duration=960,
            boundaries=(Boundary(BoundaryKind.HARD_CUT, source_index=3),))
        out = self.run([0, 1, 2, cut_unit, 4, 5])
        assert [u.pts for u in out] == [i * 960 for i in range(6)]
        flagged = [u.pts for u in out
                   if any(b.kind is BoundaryKind.HARD_CUT
                          for b in u.boundaries)]
        assert flagged == [3 * 960]

    def test_gop_policy_windows_mlx_and_skips_discarded_tail_steps(self):
        import mlx.core as mx

        from kinovsr.analysis.noise.track import NoiseMapTracker, PulseGain
        from kinovsr.processors import GopWindowPolicy
        from kinovsr.processors.bsvd import BsvdDenoiser, default_weights_path
        from kinovsr.processors.units import SourceFrameInfo

        weights = default_weights_path("c64")
        if not weights.is_file():
            pytest.skip(f"bsvd weights not available at {weights}")

        driver = BsvdDenoiser(
            weights, strength=0.3, noise_map=NoiseMapTracker(min_frames=2),
            pulse=PulseGain(), map_refresh=3, backend="mlx")
        original_step = driver.net.step
        step_count = 0

        def counted_step(frame):
            nonlocal step_count
            step_count += 1
            return original_step(frame)

        driver.net.step = counted_step
        pairs = [
            (mx.full((16, 16, 3), 0.2 + i / 100, dtype=mx.float32),
             FrameUnit(None, i, 1, source=SourceFrameInfo(i, i in {0, 8, 16})))
            for i in range(18)
        ]

        def run(policy):
            before = step_count
            driver.set_gop_policy(policy)
            output = [item for frame, token in pairs
                      for item in driver.feed(frame, token=token)]
            output.extend(driver.flush())
            return output, step_count - before

        try:
            plain, plain_steps = run(None)
            aligned, aligned_steps = run(GopWindowPolicy(4, 12))
            assert driver._gop is not None
            assert [token.pts for _, token in plain] == list(range(18))
            assert [token.pts for _, token in aligned] == list(range(18))
            assert plain_steps == 18 + driver.net.SHIFT_NUM
            # [0,9) and [8,17) discard their shared anchors; [16,18) is tail.
            assert aligned_steps == (9 + 16) * 2 + (2 + 16) - 2
            assert driver.last_noise_map is not None
            assert any(
                float(mx.max(mx.abs(left - right))) > 0.0
                for (left, _), (right, _) in zip(plain, aligned, strict=True)
            )
        finally:
            driver.close()

    def test_luma_chroma_split_reweights_the_output(self):
        # End to end through the real scheduler and denoiser: the same input
        # frames run with and without the split must stay frame-aligned but
        # differ, since keeping most of the original luma (luma_strength=0.2)
        # is exactly what a single joint sigma cannot do.
        import mlx.core as mx

        units = [
            FrameUnit(
                payload=mx.random.uniform(shape=(48, 64, 3)).astype(mx.float32)
                * 0.5 + 0.25,
                pts=i * 960, duration=960)
            for i in range(5)]
        full = self.run(list(units))
        split = self.run(
            list(units),
            config={"luma_strength": 0.2, "chroma_strength": 1.0})
        assert [u.pts for u in full] == [u.pts for u in split]
        deltas = [float(mx.max(mx.abs(f.payload - s.payload)).item())
                  for f, s in zip(full, split, strict=True)]
        assert max(deltas) > 1e-3

    def test_noise_map_auto_conditions_the_output(self):
        # End to end: the estimated per-pixel sigma map replaces the constant
        # sigma, so an auto run diverges from the constant run on the same
        # frames while staying frame-aligned. The c64 checkpoint is 4-channel,
        # so it accepts a map; 16 frames clears the tracker's warm-up.
        import mlx.core as mx

        units = [
            FrameUnit(
                payload=mx.random.uniform(shape=(48, 64, 3)).astype(mx.float32)
                * 0.5 + 0.25,
                pts=i * 960, duration=960)
            for i in range(16)]
        constant = self.run(list(units))
        auto = self.run(list(units), config={"noise_map": "auto"})
        assert [u.pts for u in constant] == [u.pts for u in auto]
        deltas = [float(mx.max(mx.abs(c.payload - a.payload)).item())
                  for c, a in zip(constant, auto, strict=True)]
        assert max(deltas) > 1e-3
