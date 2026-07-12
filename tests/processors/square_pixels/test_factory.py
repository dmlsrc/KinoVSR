"""square_pixels factory: a declared anamorphic-neutralization family."""

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
from kinovsr.processors.square_pixels import FACTORY
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw):
    return FACTORY.parse_config(
        raw, capability=Capability.PREPROCESS, profile=None,
        settings=SETTINGS)


def stream(width=160, height=128, par=Fraction(1)) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False,
            geometry=Geometry(width, height, par)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 25000), cadence=Fraction(25)))


class TestParse:
    def test_takes_no_keys(self):
        # Parameterless: any key is a config error.
        with pytest.raises(ValueError, match="unknown key"):
            parse({"scale": "2"})
        assert isinstance(parse({}), type(parse({})))

    def test_catalog_resolves(self):
        assert get_factory("square_pixels") is FACTORY


class TestSpec:
    def test_anamorphic_widens_and_squares(self):
        spec = FACTORY.capabilities[Capability.PREPROCESS]
        out = spec.produces(stream(par=Fraction(2, 1)), parse({}))
        # 160 * 2 = 320, pixels now 1:1
        assert out.frame.geometry == Geometry(320, 128, Fraction(1))

    def test_target_width_is_forced_even(self):
        spec = FACTORY.capabilities[Capability.PREPROCESS]
        # 102 * 3/2 = 153 (odd) -> clamped down to 152
        out = spec.produces(stream(width=102, par=Fraction(3, 2)), parse({}))
        assert out.frame.geometry.width == 152
        assert out.frame.geometry.pixel_aspect == Fraction(1)

    def test_already_square_is_a_noop_retag(self):
        spec = FACTORY.capabilities[Capability.PREPROCESS]
        out = spec.produces(stream(par=Fraction(1)), parse({}))
        assert out.frame.geometry == Geometry(160, 128, Fraction(1))


def test_square_pixels_resamples_through_the_chain():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan

    units = [FrameUnit(
        payload=mx.ones((128, 160, 3), dtype=mx.float32),
        pts=i * 1000, duration=1000) for i in range(3)]
    plan = resolve_pipeline(
        {"pipeline": ["s"], "s": {"processor": "square_pixels"}},
        input_spec=stream(par=Fraction(2, 1)), settings=SETTINGS)
    out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))
    assert all(u.payload.shape == (128, 320, 3) for u in out)
    assert plan.output_spec.frame.geometry == Geometry(320, 128, Fraction(1))


def test_already_square_source_passes_through_unresampled():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan

    units = [FrameUnit(payload=mx.ones((128, 160, 3), dtype=mx.float32),
                       pts=0, duration=1000)]
    plan = resolve_pipeline(
        {"pipeline": ["s"], "s": {"processor": "square_pixels"}},
        input_spec=stream(par=Fraction(1)), settings=SETTINGS)
    out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))
    # no resample: same object shape, and the width is unchanged
    assert out[0].payload.shape == (128, 160, 3)
    assert plan.output_spec.frame.geometry == Geometry(160, 128, Fraction(1))
