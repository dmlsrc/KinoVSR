"""mc factory: a causal motion-compensated temporal denoise family."""

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
from kinovsr.processors.mc import FACTORY, McStageConfig
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw):
    return FACTORY.parse_config(
        raw, capability=Capability.DENOISE, profile=None, settings=SETTINGS)


def stream() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(192, 144)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


class TestParse:
    def test_defaults_match_the_family_tuning(self):
        config = parse({})
        assert config == McStageConfig(
            strength=0.5, window=0, sigma=0.06, gate="smooth",
            clamp=False, occlusion=False, confidence=False,
            flow="vt", flow_weights=None)

    def test_flow_vocabulary(self):
        assert parse({"flow": "spynet"}).flow == "spynet"
        with pytest.raises(ValueError, match="flow must be"):
            parse({"flow": "zero"})

    def test_bounds(self):
        with pytest.raises(ValueError, match="sigma"):
            parse({"sigma": 0.0})
        with pytest.raises(ValueError, match="gate"):
            parse({"gate": "median"})


class TestSpec:
    def test_causal_zero_delay_stateful(self):
        spec = FACTORY.capabilities[Capability.DENOISE]
        assert spec.temporal_mode is TemporalMode.CAUSAL
        assert spec.stateful
        assert spec.produces(stream(), parse({})) == stream()

    def test_catalog_resolves(self):
        assert get_factory("mc") is FACTORY


@pytest.mark.integration
def test_recursive_denoise_through_the_chain():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan
    from kinovsr.processors.errors import MediaError

    mx.random.seed(0)
    # VT optical flow silently returns zeros on very small frames (the
    # engine's self-test guards it); use a geometry inside its regime.
    # The base must be SMOOTH: per-pixel random texture reads as huge
    # residual to the photometric gate, which then (correctly) refuses
    # to blend - the gate is the feature, so feed it honest static
    # content.
    yy = mx.linspace(0.0, 1.0, 144)[:, None, None]
    xx = mx.linspace(0.0, 1.0, 192)[None, :, None]
    base = mx.broadcast_to(
        0.25 + 0.25 * yy + 0.25 * xx, (144, 192, 3)).astype(mx.float32)
    base = mx.contiguous(base)
    n = 6
    frames = [mx.clip(base + 0.05 * mx.random.normal(shape=base.shape), 0, 1)
              for _ in range(n)]
    units = [FrameUnit(payload=f, pts=i * 960, duration=960)
             for i, f in enumerate(frames)]
    plan = resolve_pipeline(
        {"pipeline": ["den"], "den": {"processor": "mc", "strength": 0.8}},
        input_spec=stream(), settings=SETTINGS)
    try:
        out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))
    except (MediaError, RuntimeError, SystemExit) as exc:
        pytest.skip(f"VT optical flow unavailable: {exc}")

    assert [u.pts for u in out] == [i * 960 for i in range(n)]
    # static noisy content: later frames should sit closer to the clean
    # base than the raw input does (temporal integration worked)
    def err(x):
        return float(mx.mean(mx.abs(x - base)))
    assert err(out[-1].payload) < err(frames[-1]) * 0.9


def test_settings_spynet_weights_reach_the_config():
    settings = Settings(spynet_weights="/w/custom_spynet.safetensors")
    config = FACTORY.parse_config(
        {"flow": "spynet"}, capability=Capability.DENOISE, profile=None,
        settings=settings)
    assert config.flow_weights == "/w/custom_spynet.safetensors"
