"""VideoToolbox spatial upscale: the fast/balanced/image modes, and the
MLX->CV bridge that lets a native scaler follow an MLX preprocessing chain."""

from fractions import Fraction

import pytest

from kinovsr.processors import (
    Boundary,
    BoundaryKind,
    Capability,
    Domain,
    DType,
    FrameUnit,
    Geometry,
    Layout,
    MediaError,
    PipelineContext,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
    get_factory,
)
from kinovsr.processors.videotoolbox import FACTORY
from kinovsr.settings import Settings

SETTINGS = Settings()
CONTEXT = PipelineContext(settings=SETTINGS)

W, H = 256, 256   # within every mode's input cap; a size the HQ scaler accepts
TB = Fraction(1, 24000)


def parse(raw, profile=None):
    return FACTORY.parse_config(
        raw, capability=Capability.UPSCALE, profile=profile, settings=SETTINGS)


def mlx_stream(w=W, h=H) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix("bt709", full_range=False,
                                    geometry=Geometry(w, h)),
        timeline=TimelineSpec(time_base=TB, cadence=Fraction(24)))


@pytest.mark.unit
class TestParseAndSpec:
    def test_profile_is_the_mode_default_balanced(self):
        assert parse({}).mode == "balanced"
        assert parse({}, profile="fast").mode == "fast"
        assert parse({}, profile="image").mode == "image"

    def test_bad_profile_rejected(self):
        with pytest.raises(ValueError, match="profile must be one of"):
            parse({}, profile="ultra")

    def test_no_keys_accepted(self):
        with pytest.raises(ValueError, match="unknown"):
            parse({"scale": 2})

    @pytest.mark.parametrize("mode,scale,layout,dtype,domain", [
        ("fast", 2, Layout.CV_NV12, DType.UINT8, Domain.CODED),
        ("balanced", 4, Layout.CV_RGBA_HALF, DType.FLOAT16, Domain.UNIT),
        ("image", 4, Layout.CV_RGBA_HALF, DType.FLOAT16, Domain.UNIT),
    ])
    def test_produces_the_native_cv_frame(self, mode, scale, layout, dtype,
                                          domain):
        spec = FACTORY.capabilities[Capability.UPSCALE]
        up = spec.produces(mlx_stream(), parse({}, profile=mode))
        assert up.frame.layout is layout
        assert up.frame.dtype is dtype
        assert up.frame.domain is domain
        assert up.frame.geometry.width == W * scale
        assert up.frame.geometry.height == H * scale
        # spatial: the timeline is untouched
        assert up.timeline == mlx_stream().timeline

    def test_accepts_mlx_the_bridge(self):
        spec = FACTORY.capabilities[Capability.UPSCALE]
        assert spec.accepts.layouts == (Layout.MLX_RGB_HWC,)
        assert spec.stateful  # balanced threads a prev-frame chain

    @pytest.mark.parametrize("mode,max_w,max_h", [
        ("fast", 960, 960), ("balanced", 1920, 1080), ("image", 1920, 1080),
    ])
    def test_size_cap_rejected_at_open_time(self, mode, max_w, max_h):
        spec = FACTORY.capabilities[Capability.UPSCALE]
        with pytest.raises(ValueError, match="accepts input up to"):
            spec.produces(mlx_stream(max_w + 2, max_h + 2), parse({}, profile=mode))

    def test_catalog_resolves_both_capabilities(self):
        assert get_factory("videotoolbox") is FACTORY
        assert Capability.UPSCALE in FACTORY.capabilities
        assert Capability.INTERPOLATE in FACTORY.capabilities

    def test_bridge_shape_resolves(self):
        # MLX preprocessing then a native upscale: the exact harness shape.
        from kinovsr.pipeline import resolve_pipeline

        plan = resolve_pipeline(
            {"pipeline": ["den", "up"],
             "den": {"processor": "spatial", "strength": 0.3},
             "up": {"processor": "videotoolbox", "capability": "upscale",
                    "profile": "balanced"}},
            input_spec=mlx_stream(), settings=SETTINGS)
        assert plan.stages[0].output_spec.frame.layout is Layout.MLX_RGB_HWC
        assert plan.output_spec.frame.layout is Layout.CV_RGBA_HALF


def mlx_units(n, w=W, h=H):
    import mlx.core as mx

    ticks = round(1 / Fraction(24) / TB)
    return [
        FrameUnit(payload=mx.random.uniform(shape=(h, w, 3)).astype(mx.float32),
                  pts=i * ticks, duration=ticks)
        for i in range(n)
    ]


@pytest.mark.integration
class TestUpscale:
    @staticmethod
    def run(mode, units, cut_at=None):
        from kinovsr.pipeline import resolve_pipeline, run_plan

        if cut_at is not None:
            units = list(units)
            units[cut_at] = units[cut_at].with_boundary(
                Boundary(BoundaryKind.HARD_CUT, source_index=cut_at))
        config = {"pipeline": ["up"],
                  "up": {"processor": "videotoolbox", "capability": "upscale",
                         "profile": mode}}
        plan = resolve_pipeline(config, input_spec=mlx_stream(), settings=SETTINGS)
        try:
            return list(run_plan(plan, units, CONTEXT)), plan
        except MediaError as exc:
            pytest.skip(str(exc))   # mode unsupported on this device

    @pytest.mark.parametrize("mode,scale", [
        ("balanced", 4), ("image", 4), ("fast", 2)])
    def test_bridges_mlx_to_native_cv_at_scale(self, mode, scale):
        import mlx.core as mx

        units = mlx_units(4)
        out, plan = self.run(mode, units)
        assert len(out) == 4
        # spatial: PTS preserved (keeps audio + any restore companion in sync)
        assert [u.pts for u in out] == [u.pts for u in units]
        # the payload is now a native CVPixelBuffer, not an MLX array
        assert not isinstance(out[0].payload, mx.array)
        assert plan.output_spec.frame.geometry.width == W * scale

    def test_hard_cut_resets_the_temporal_chain(self):
        # balanced threads prev-src/prev-dst; the scheduler resets it on a cut,
        # which must run cleanly and lose no frames.
        out, _ = self.run("balanced", mlx_units(5), cut_at=2)
        assert len(out) == 5
