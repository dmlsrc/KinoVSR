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

    def test_declares_causal_stateful(self):
        spec = FACTORY.capabilities[Capability.DENOISE]
        assert spec.temporal_mode is TemporalMode.CAUSAL
        assert spec.stateful
        assert get_factory("bsvd") is FACTORY


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
