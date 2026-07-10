"""RealPLKSR factory: the stateless per-frame proving processor."""

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
from kinovsr.realplksr.factory import FACTORY, RealPlksrStageConfig
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw, profile=None, settings=SETTINGS):
    return FACTORY.parse_config(
        raw, capability=Capability.UPSCALE, profile=profile,
        settings=settings)


def stream(width=64, height=48) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(width, height)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


class TestParse:
    def test_defaults(self):
        config = parse({})
        assert config == RealPlksrStageConfig(
            weights_spec="public2x", scale=2, dtype="float16")

    def test_profile_declares_scale(self):
        assert parse({}, profile="nomos4x").scale == 4
        assert parse({}, profile="public2x-nn").scale == 2

    def test_weights_token_infers_scale(self):
        config = parse({"weights": "nomos4x"})
        assert config.scale == 4
        assert config.weights_spec == "nomos4x"

    def test_weights_path_requires_scale(self):
        with pytest.raises(ValueError, match="state scale"):
            parse({"weights": "/w/custom.safetensors"})
        config = parse({"weights": "/w/custom.safetensors", "scale": 2})
        assert config.scale == 2

    def test_settings_env_weights_are_honored(self):
        settings = Settings(realplksr_weights="nomos4x")
        assert parse({}, settings=settings).weights_spec == "nomos4x"

    def test_unknown_key_suggests(self):
        with pytest.raises(ValueError, match="did you mean 'dtype'"):
            parse({"dtpye": "float16"})

    def test_bad_dtype_and_scale(self):
        with pytest.raises(ValueError, match="dtype"):
            parse({"dtype": "bfloat16"})
        with pytest.raises(ValueError, match="scale must be"):
            parse({"weights": "/w/x.st", "scale": 3})


class TestSpec:
    def test_produces_scales_geometry(self):
        spec = FACTORY.capabilities[Capability.UPSCALE]
        out = spec.produces(stream(), parse({}, profile="nomos4x"))
        assert out.frame.geometry.width == 256
        assert out.frame.geometry.height == 192
        assert out.timeline == stream().timeline

    def test_catalog_resolves(self):
        assert get_factory("realplksr") is FACTORY


@pytest.mark.requires_weights
@pytest.mark.integration
def test_end_to_end_upscale_through_the_chain():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan

    config = {
        "pipeline": ["up"],
        "up": {"processor": "realplksr", "profile": "public2x"},
    }
    try:
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        context = PipelineContext(settings=SETTINGS)
        frames = [FrameUnit(payload=mx.zeros((48, 64, 3), dtype=mx.float32),
                            pts=i * 960, duration=960) for i in range(3)]
        out = list(run_plan(plan, frames, context))
    except FileNotFoundError as exc:
        pytest.skip(f"realplksr weights not available: {exc}")
    assert [u.pts for u in out] == [0, 960, 1920]
    assert all(u.payload.shape == (96, 128, 3) for u in out)
    assert plan.output_spec.frame.geometry.width == 128
