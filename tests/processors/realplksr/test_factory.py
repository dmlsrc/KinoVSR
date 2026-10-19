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
from kinovsr.processors.realplksr.factory import FACTORY, RealPlksrStageConfig
from kinovsr.processors.realplksr.net import resolve_weights
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw, profile=None, settings=SETTINGS, capability=Capability.UPSCALE):
    return FACTORY.parse_config(raw, capability=capability, profile=profile, settings=settings)


def stream(width=64, height=48) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix("bt709", full_range=False, geometry=Geometry(width, height)),
        timeline=TimelineSpec(time_base=Fraction(1, 24000), cadence=Fraction(25)),
    )


def _require_weights(token: str) -> None:
    """Skip unless the checkpoint for ``token`` is installed.

    Resolved before running: the runtime reports a stage's missing weights as a
    PipelineRuntimeError, so a FileNotFoundError never reaches the test.
    """
    try:
        resolve_weights(token)
    except FileNotFoundError as exc:
        pytest.skip(f"realplksr {token} weights not available: {exc}")


class TestParse:
    def test_defaults(self):
        config = parse({})
        assert config == RealPlksrStageConfig(weights_spec="public2x", scale=2, dtype="float16")

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

    def test_restore_profile_and_strength(self):
        config = parse({}, capability=Capability.RESTORE)
        assert config == RealPlksrStageConfig(
            weights_spec="deh264", scale=1, dtype="float16", strength=1.0
        )
        half = parse({"strength": 0.5}, profile="deh264", capability=Capability.RESTORE)
        assert half.strength == 0.5
        assert half.scale == 1
        assert (
            parse(
                {"weights": "/w/custom.safetensors", "scale": 1}, capability=Capability.RESTORE
            ).scale
            == 1
        )

    def test_restore_rejects_upscale_scale_and_strength_out_of_range(self):
        with pytest.raises(ValueError, match="scale must be 1 for restore"):
            parse({"scale": 2}, capability=Capability.RESTORE)
        with pytest.raises(ValueError, match="scale must be 2 or 4 for upscale"):
            parse({"scale": 1})
        with pytest.raises(ValueError, match="strength must be in"):
            parse({"strength": 1.5}, capability=Capability.RESTORE)
        with pytest.raises(ValueError, match="unknown key"):
            parse({"strength": 0.5})

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

    def test_restore_preserves_geometry_and_timeline(self):
        spec = FACTORY.capabilities[Capability.RESTORE]
        out = spec.produces(stream(), parse({}, capability=Capability.RESTORE))
        assert out == stream()
        assert spec.profiles == ("deh264",)

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
    _require_weights("public2x")
    plan = resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)
    context = PipelineContext(settings=SETTINGS)
    frames = [
        FrameUnit(payload=mx.zeros((48, 64, 3), dtype=mx.float32), pts=i * 960, duration=960)
        for i in range(3)
    ]
    out = list(run_plan(plan, frames, context))
    assert [u.pts for u in out] == [0, 960, 1920]
    assert all(u.payload.shape == (96, 128, 3) for u in out)
    assert plan.output_spec.frame.geometry.width == 128


@pytest.mark.requires_weights
@pytest.mark.integration
def test_end_to_end_restore_through_the_chain():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan

    config = {
        "pipeline": ["cleanup"],
        "cleanup": {
            "processor": "realplksr",
            "capability": "restore",
            "profile": "deh264",
            "strength": 0.5,
        },
    }
    _require_weights("deh264")
    plan = resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)
    frames = [
        FrameUnit(payload=mx.full((48, 64, 3), 0.25, dtype=mx.float32), pts=i * 960, duration=960)
        for i in range(3)
    ]
    out = list(run_plan(plan, frames, PipelineContext(settings=SETTINGS)))
    assert [u.pts for u in out] == [0, 960, 1920]
    assert all(u.payload.shape == (48, 64, 3) for u in out)
    assert all(bool(mx.all(mx.isfinite(u.payload))) for u in out)
    assert plan.output_spec.frame.geometry == stream().frame.geometry
