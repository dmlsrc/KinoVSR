"""crop factory: a declared geometry-cropping preprocess family."""

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
from kinovsr.processors.crop import FACTORY
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw):
    return FACTORY.parse_config(
        raw, capability=Capability.PREPROCESS, profile=None,
        settings=SETTINGS)


def stream(width=640, height=480) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(width, height)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


class TestParse:
    def test_cropping_nothing_is_rejected(self):
        with pytest.raises(ValueError, match="config noise"):
            parse({})

    def test_anchor_and_aspect_validation(self):
        with pytest.raises(ValueError, match="anchor"):
            parse({"aspect": "16:9", "anchor": "middle"})
        with pytest.raises(ValueError, match="W:H"):
            parse({"aspect": "16x9"})


class TestSpec:
    def test_bars_shrink_geometry(self):
        spec = FACTORY.capabilities[Capability.PREPROCESS]
        out = spec.produces(stream(), parse({"bars": "60,60,0,0"}))
        assert out.frame.geometry == Geometry(640, 360)

    def test_aspect_window_16_9(self):
        spec = FACTORY.capabilities[Capability.PREPROCESS]
        out = spec.produces(stream(), parse({"aspect": "16:9"}))
        assert out.frame.geometry == Geometry(640, 360)

    def test_catalog_resolves(self):
        assert get_factory("crop") is FACTORY


def test_crop_through_the_chain():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan

    units = [FrameUnit(
        payload=mx.broadcast_to(
            mx.linspace(0, 1, 480)[:, None, None], (480, 640, 3)
        ).astype(mx.float32),
        pts=i * 960, duration=960) for i in range(3)]
    plan = resolve_pipeline(
        {"pipeline": ["c"],
         "c": {"processor": "crop", "bars": "60,60,0,0"}},
        input_spec=stream(), settings=SETTINGS)
    out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))
    assert all(u.payload.shape == (360, 640, 3) for u in out)
    assert plan.output_spec.frame.geometry == Geometry(640, 360)
    # the kept region is the interior (top band removed => first row
    # equals the source's row 60 value)
    import math
    assert math.isclose(float(out[0].payload[0, 0, 0]), 60 / 479,
                        rel_tol=1e-4)
