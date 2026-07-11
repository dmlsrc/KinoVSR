"""sanitize_edges factory: a declared edge replicate-fill family."""

from fractions import Fraction

import pytest

from kinovsr.processors import (
    Capability,
    FrameUnit,
    Geometry,
    PipelineContext,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
    get_factory,
)
from kinovsr.processors.sanitize_edges import FACTORY
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw):
    return FACTORY.parse_config(
        raw, capability=Capability.PREPROCESS, profile=None,
        settings=SETTINGS)


def stream() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(64, 48)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


class TestParse:
    def test_edges_required(self):
        with pytest.raises(ValueError, match="probe-time"):
            parse({})
        with pytest.raises(ValueError, match="no bands"):
            parse({"edges": "0,0,0,0"})

    def test_restore_and_trim_are_not_this_stage(self):
        with pytest.raises(ValueError, match="harness-owned"):
            parse({"edges": "2,0,0,0", "fill": "restore"})

    def test_catalog_resolves(self):
        assert get_factory("sanitize_edges") is FACTORY


def test_bands_replicate_and_geometry_is_untouched():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan

    frame = mx.broadcast_to(
        mx.linspace(0, 1, 48)[:, None, None], (48, 64, 3)
    ).astype(mx.float32)
    # poison the top two rows
    poisoned = mx.concatenate(
        [mx.ones((2, 64, 3), dtype=mx.float32), frame[2:]], axis=0)
    units = [FrameUnit(payload=poisoned, pts=0, duration=960)]
    plan = resolve_pipeline(
        {"pipeline": ["san"],
         "san": {"processor": "sanitize_edges", "edges": "2,0,0,0"}},
        input_spec=stream(), settings=SETTINGS)
    out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))
    assert out[0].payload.shape == (48, 64, 3)
    assert plan.output_spec == stream()
    # bands now replicate the first interior row; the poison is gone
    result = out[0].payload
    assert bool(mx.array_equal(result[0], result[2]))
    assert bool(mx.array_equal(result[1], result[2]))
    assert float(result[0, 0, 0]) != 1.0
