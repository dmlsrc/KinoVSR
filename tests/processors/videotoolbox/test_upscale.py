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


def cv_stream(layout, w=W, h=H) -> StreamSpec:
    import dataclasses

    dtype = DType.UINT8 if layout is Layout.CV_NV12 else DType.FLOAT16
    domain = Domain.CODED if layout is Layout.CV_NV12 else Domain.UNIT
    base = mlx_stream(w, h)
    return dataclasses.replace(
        base, frame=dataclasses.replace(
            base.frame, layout=layout, dtype=dtype, domain=domain))


@pytest.mark.unit
class TestParseAndSpec:
    def test_profile_is_the_mode_default_balanced(self):
        assert parse({}).mode == "balanced"
        assert parse({}).flow == "vt"
        assert parse({}, profile="fast").mode == "fast"
        assert parse({}, profile="image").mode == "image"

    def test_balanced_accepts_vision_flow(self):
        config = parse({"flow": "vision"}, profile="balanced")
        assert config.mode == "balanced"
        assert config.flow == "vision"

    def test_bad_flow_rejected(self):
        with pytest.raises(ValueError, match="flow must be one of"):
            parse({"flow": "internal"}, profile="balanced")

    @pytest.mark.parametrize("mode", ["fast", "image"])
    def test_flow_is_balanced_only(self, mode):
        with pytest.raises(ValueError, match="only valid.*balanced"):
            parse({"flow": "vision"}, profile=mode)

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

    def test_accepts_mlx_and_the_native_cv_sources(self):
        # MLX is the bridge; the CV layouts are the zero-copy decode->VSR
        # path (each mode's own session format, enforced in produces).
        spec = FACTORY.capabilities[Capability.UPSCALE]
        assert set(spec.accepts.layouts) == {
            Layout.MLX_RGB_HWC, Layout.CV_RGBA_HALF, Layout.CV_NV12}
        assert spec.stateful  # balanced threads a prev-frame chain

    @pytest.mark.parametrize("mode,layout", [
        ("fast", Layout.CV_NV12),
        ("balanced", Layout.CV_RGBA_HALF),
        ("image", Layout.CV_RGBA_HALF),
    ])
    def test_native_cv_source_resolves(self, mode, layout):
        spec = FACTORY.capabilities[Capability.UPSCALE]
        cv = cv_stream(layout)
        up = spec.produces(cv, parse({}, profile=mode))
        assert up.frame.geometry.width == W * (2 if mode == "fast" else 4)

    def test_mode_layout_mismatch_rejected_at_resolve(self):
        spec = FACTORY.capabilities[Capability.UPSCALE]
        with pytest.raises(ValueError, match="native cv_rgba_half"):
            spec.produces(cv_stream(Layout.CV_NV12), parse({}, profile="balanced"))
        with pytest.raises(ValueError, match="native cv_nv12"):
            spec.produces(cv_stream(Layout.CV_RGBA_HALF), parse({}, profile="fast"))

    def test_preferred_source_layout_hook(self):
        assert FACTORY.preferred_source_layout(
            capability=Capability.UPSCALE, profile="fast") is Layout.CV_NV12
        assert FACTORY.preferred_source_layout(
            capability=Capability.UPSCALE,
            profile="balanced") is Layout.CV_RGBA_HALF
        # A head interpolate decodes straight into the FRC's CV currency
        # (MLX stays accepted mid-chain via the upload bridge).
        assert FACTORY.preferred_source_layout(
            capability=Capability.INTERPOLATE,
            profile="normal") is Layout.CV_RGBA_HALF

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

    def test_crop_composes_before_the_native_upscale(self):
        # Parity gap C3: crop is MLX-only; now that the upscale accepts MLX,
        # a crop (MLX->MLX) threads its geometry through into the native scale.
        from kinovsr.pipeline import resolve_pipeline

        plan = resolve_pipeline(
            {"pipeline": ["crop", "up"],
             "crop": {"processor": "crop", "bars": "4,4,0,0"},
             "up": {"processor": "videotoolbox", "capability": "upscale",
                    "profile": "balanced"}},
            input_spec=mlx_stream(100, 100), settings=SETTINGS)
        assert [s.name for s in plan.stages] == ["crop", "up"]
        assert plan.output_spec.frame.layout is Layout.CV_RGBA_HALF
        # cropped 100x92, then 4x
        assert plan.output_spec.frame.geometry.width == 400
        assert plan.output_spec.frame.geometry.height == 368


def mlx_units(n, w=W, h=H):
    import mlx.core as mx

    ticks = round(1 / Fraction(24) / TB)
    return [
        FrameUnit(payload=mx.random.uniform(shape=(h, w, 3)).astype(mx.float32),
                  pts=i * ticks, duration=ticks)
        for i in range(n)
    ]


@pytest.mark.unit
class TestBalancedFlowPipeline:
    class Session:
        instances = []

        def __init__(self, *_args, **kwargs):
            self.kwargs = kwargs
            self.out_w = W * 4
            self.out_h = H * 4
            self.dst_attrs = {"PixelFormatType": 123}
            self.submissions = []
            self.outputs = iter(("out-0", None, "out-1"))
            self.finish_output = "out-2"
            self.reset_count = 0
            self.closed = False
            self.__class__.instances.append(self)

        def submit_upscale_buffer_to_buffer(self, payload, index):
            self.submissions.append((payload, index))
            return next(self.outputs)

        def finish_pending_upscale(self):
            output, self.finish_output = self.finish_output, None
            return output

        def reset_temporal_context(self):
            self.reset_count += 1

        def close(self):
            self.closed = True

    @staticmethod
    def _prepared(monkeypatch, *, flow="vt"):
        from kinovsr.native import vsr
        from kinovsr.processors import videotoolbox
        from kinovsr.processors.videotoolbox import (
            VtUpscaleConfig,
            VtUpscaleProcessor,
        )

        TestBalancedFlowPipeline.Session.instances.clear()
        monkeypatch.setattr(vsr, "VsrSession", TestBalancedFlowPipeline.Session)
        monkeypatch.setattr(
            videotoolbox,
            "apply_output_pool",
            lambda *_args: None,
        )
        processor = VtUpscaleProcessor(
            VtUpscaleConfig(mode="balanced", flow=flow)
        )
        processor.prepare(cv_stream(Layout.CV_RGBA_HALF), CONTEXT)
        return processor, TestBalancedFlowPipeline.Session.instances[-1]

    def test_balanced_uses_explicit_flow_and_preserves_delayed_unit_order(
            self, monkeypatch):
        processor, session = self._prepared(monkeypatch)
        units = [
            FrameUnit(payload=f"in-{index}", pts=index * 1000, duration=1000)
            for index in range(3)
        ]

        assert session.kwargs["explicit_flow"] is True
        assert session.kwargs["flow_backend"] == "vt"
        assert [out.payload for out in processor.process(units[0], CONTEXT)] == [
            "out-0"
        ]
        assert list(processor.process(units[1], CONTEXT)) == []
        delayed = list(processor.process(units[2], CONTEXT))
        assert [(out.payload, out.pts) for out in delayed] == [
            ("out-1", 1000)
        ]
        tail = list(processor.flush(CONTEXT))
        assert [(out.payload, out.pts) for out in tail] == [
            ("out-2", 2000)
        ]
        assert session.submissions == [
            ("in-0", 0),
            ("in-1", 1),
            ("in-2", 2),
        ]

        processor.reset(
            Boundary(BoundaryKind.HARD_CUT, source_index=3),
            CONTEXT,
        )
        assert session.reset_count == 1
        processor.close(CONTEXT)
        assert session.closed

    def test_vision_backend_reaches_the_native_session(self, monkeypatch):
        processor, session = self._prepared(monkeypatch, flow="vision")
        try:
            assert session.kwargs["explicit_flow"] is True
            assert session.kwargs["flow_backend"] == "vision"
        finally:
            processor.close(CONTEXT)

    def test_reset_rejects_an_undrained_delayed_frame(self, monkeypatch):
        processor, _session = self._prepared(monkeypatch)
        first = FrameUnit(payload="in-0", pts=0, duration=1000)
        second = FrameUnit(payload="in-1", pts=1000, duration=1000)
        assert list(processor.process(first, CONTEXT))
        assert list(processor.process(second, CONTEXT)) == []

        with pytest.raises(RuntimeError, match="flushed before"):
            processor.reset(
                Boundary(BoundaryKind.HARD_CUT, source_index=2),
                CONTEXT,
            )
        processor.close(CONTEXT)


@pytest.mark.integration
class TestUpscale:
    @staticmethod
    def run(mode, units, cut_at=None, *, flow=None):
        from kinovsr.pipeline import resolve_pipeline, run_plan

        if cut_at is not None:
            units = list(units)
            units[cut_at] = units[cut_at].with_boundary(
                Boundary(BoundaryKind.HARD_CUT, source_index=cut_at))
        up = {
            "processor": "videotoolbox",
            "capability": "upscale",
            "profile": mode,
        }
        if flow is not None:
            up["flow"] = flow
        config = {"pipeline": ["up"], "up": up}
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

    def test_balanced_vision_flow_backend_runs_through_pipeline(self):
        units = mlx_units(4)
        out, _ = self.run("balanced", units, flow="vision")
        assert len(out) == len(units)
        assert [unit.pts for unit in out] == [unit.pts for unit in units]

    @pytest.mark.parametrize("mode,layout,scale", [
        ("fast", Layout.CV_NV12, 2),
        ("balanced", Layout.CV_RGBA_HALF, 4),
    ])
    def test_zero_copy_cv_source(self, mode, layout, scale):
        # A source already decoded in the mode's session format feeds
        # upscale_buffer_to_buffer with no MLX round trip - the harness's
        # mainstream decode->VSR path.
        from kinovsr.media import pixel_buffers as _pb
        from kinovsr.pipeline import resolve_pipeline, run_plan

        fmt = _pb.PIX_NV12 if layout is Layout.CV_NV12 else _pb.PIX_RGBAHALF
        units = [FrameUnit(
            payload=_pb.make_pixel_buffer_from_attrs(W, H, {
                "PixelFormatType": fmt, "IOSurfaceProperties": {}}),
            pts=i * 1000, duration=1000) for i in range(3)]
        config = {"pipeline": ["up"],
                  "up": {"processor": "videotoolbox", "capability": "upscale",
                         "profile": mode}}
        plan = resolve_pipeline(config, input_spec=cv_stream(layout),
                                settings=SETTINGS)
        try:
            out = list(run_plan(plan, units, CONTEXT))
        except MediaError as exc:
            pytest.skip(str(exc))
        assert len(out) == 3
        assert plan.output_spec.frame.geometry.width == W * scale
