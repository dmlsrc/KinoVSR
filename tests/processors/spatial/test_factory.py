"""spatial factory: a per-frame native denoise family."""

from fractions import Fraction

import pytest

from kinovsr.processors import (
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
from kinovsr.processors.spatial import FACTORY, SpatialStageConfig
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw):
    return FACTORY.parse_config(
        raw, capability=Capability.DENOISE, profile=None, settings=SETTINGS)


def stream() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(64, 48)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


class TestParse:
    def test_default_strength(self):
        assert parse({}) == SpatialStageConfig(strength=0.5)

    def test_bounds(self):
        with pytest.raises(ValueError, match="strength"):
            parse({"strength": 1.5})


class TestSpec:
    def test_per_frame_stateless(self):
        spec = FACTORY.capabilities[Capability.DENOISE]
        assert spec.temporal_mode is TemporalMode.PER_FRAME
        assert not spec.stateful
        assert spec.produces(stream(), parse({})) == stream()

    def test_catalog_resolves(self):
        assert get_factory("spatial") is FACTORY


@pytest.mark.integration
def test_per_frame_denoise_through_the_chain():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan

    mx.random.seed(0)
    n = 3
    # Light noise on structured content - CINoiseReduction's design
    # regime (its inputNoiseLevel maps to sigma ~0.01-0.05; heavier
    # noise trips its sharpening pass instead).
    clean = mx.broadcast_to(
        mx.linspace(0.2, 0.8, 64)[None, :, None], (48, 64, 3)
    ).astype(mx.float32)
    units = [FrameUnit(
        payload=mx.clip(
            clean + 0.02 * mx.random.normal(shape=(48, 64, 3)), 0, 1),
        pts=i * 960, duration=960) for i in range(n)]
    plan = resolve_pipeline(
        {"pipeline": ["den"],
         "den": {"processor": "spatial", "strength": 0.8}},
        input_spec=stream(), settings=SETTINGS)
    out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))

    assert [u.pts for u in out] == [i * 960 for i in range(n)]
    # every frame independently lands closer to the clean structure
    for original, unit in zip(units, out, strict=True):
        before = float(mx.mean(mx.abs(original.payload - clean)))
        after = float(mx.mean(mx.abs(unit.payload - clean)))
        assert after < before
