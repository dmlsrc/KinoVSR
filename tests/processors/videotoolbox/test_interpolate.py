"""VideoToolbox interpolation: one-to-many, monotonic PTS, rewritten
cadence, preserved clip duration - the native-session proving processor."""

import hashlib
from fractions import Fraction

import pytest

from kinovsr.processors import (
    Boundary,
    BoundaryKind,
    Capability,
    Cardinality,
    Domain,
    DType,
    FrameUnit,
    Geometry,
    Layout,
    MediaError,
    PipelineContext,
    StreamSpec,
    TimelineSpec,
    TimestampPolicy,
    frame_spec_for_matrix,
    get_factory,
)
from kinovsr.processors.videotoolbox import FACTORY
from kinovsr.settings import Settings

SETTINGS = Settings()
CONTEXT = PipelineContext(settings=SETTINGS)

W, H, SRC_FPS = 128, 96, 24
TB = Fraction(1, 24000)


def parse(raw, profile=None):
    return FACTORY.parse_config(
        raw, capability=Capability.INTERPOLATE, profile=profile,
        settings=SETTINGS)


def bgra_stream() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(W, H),
            layout=Layout.CV_BGRA, dtype=DType.UINT8),
        timeline=TimelineSpec(time_base=TB, cadence=Fraction(SRC_FPS)))


@pytest.mark.unit
class TestParseAndSpec:
    def test_target_fps_required_and_typed(self):
        with pytest.raises(ValueError, match="target_fps is required"):
            parse({})
        with pytest.raises(ValueError, match="positive"):
            parse({"target_fps": 0})
        assert parse({"target_fps": 60}).target_fps == Fraction(60)
        assert parse({"target_fps": 59.94}).target_fps == Fraction("59.94")

    def test_profile_is_the_mode(self):
        assert parse({"target_fps": 60}).mode == "normal"
        assert parse({"target_fps": 60}, profile="high").mode == "high"

    def test_produces_rewrites_the_timeline_explicitly(self):
        spec = FACTORY.capabilities[Capability.INTERPOLATE]
        up = spec.produces(bgra_stream(), parse({"target_fps": 60}))
        assert up.timeline.cadence == Fraction(60)
        assert up.timeline.timestamp_policy is TimestampPolicy.REGENERATED
        assert up.timeline.cardinality is Cardinality.ONE_TO_MANY
        assert up.frame.layout is Layout.CV_RGBA_HALF
        assert up.frame.dtype is DType.FLOAT16
        assert up.frame.domain is Domain.UNIT
        assert up.frame.geometry == bgra_stream().frame.geometry
        assert up.frame.color_matrix == bgra_stream().frame.color_matrix
        down = spec.produces(bgra_stream(), parse({"target_fps": 12}))
        assert down.timeline.cardinality is Cardinality.MANY_TO_ONE

    def test_catalog_resolves(self):
        assert get_factory("videotoolbox") is FACTORY


def test_reset_marks_native_frc_references_discontinuous():
    from kinovsr.processors.videotoolbox import VtInterpolateProcessor

    class Session:
        reset_calls = 0

        def reset_temporal_context(self):
            self.reset_calls += 1

    processor = VtInterpolateProcessor(parse({"target_fps": 60}))
    session = Session()
    processor._session = session

    processor.reset(
        Boundary(BoundaryKind.HARD_CUT, source_index=4),
        CONTEXT,
    )

    assert session.reset_calls == 1


def source_units(n):
    from kinovsr.media import pixel_buffers as _pb

    # VTFrameProcessor requires IOSurface-backed source buffers (the real
    # readers and pools always produce them; a bare CVPixelBuffer crashes
    # inside VT, not catchably).
    attrs = {
        "PixelFormatType": _pb.PIX_BGRA,
        "IOSurfaceProperties": {},
        "MetalCompatibility": True,
    }
    ticks = round(1 / Fraction(SRC_FPS) / TB)
    return [
        FrameUnit(
            payload=_pb.make_pixel_buffer_from_attrs(W, H, attrs),
            pts=i * ticks, duration=ticks)
        for i in range(n)
    ]


def cv_digest(buffer) -> str:
    from kinovsr.native.frameworks import Quartz

    width = int(Quartz.CVPixelBufferGetWidth(buffer))
    height = int(Quartz.CVPixelBufferGetHeight(buffer))
    row_bytes = int(Quartz.CVPixelBufferGetBytesPerRow(buffer))
    Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
    try:
        view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(
            row_bytes * height
        )
        active = b"".join(
            bytes(view[row * row_bytes:row * row_bytes + width * 8])
            for row in range(height)
        )
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)
    return hashlib.sha256(active).hexdigest()


def run_interpolation(units, target_fps, cut_at=None):
    from kinovsr.pipeline import resolve_pipeline, run_plan

    if cut_at is not None:
        units = list(units)
        units[cut_at] = units[cut_at].with_boundary(
            Boundary(BoundaryKind.HARD_CUT, source_index=cut_at))
    config = {
        "pipeline": ["interp"],
        "interp": {"processor": "videotoolbox", "capability": "interpolate",
                   "target_fps": target_fps},
    }
    plan = resolve_pipeline(config, input_spec=bgra_stream(),
                            settings=SETTINGS)
    try:
        return list(run_plan(plan, units, CONTEXT)), plan
    except MediaError as exc:
        pytest.skip(str(exc))


@pytest.mark.integration
class TestInterpolation:
    def test_one_to_many_grid_exact_and_duration_preserving(self):
        n = 8
        out, plan = run_interpolation(source_units(n), 60)
        # 8 frames at 24fps span 1/3 s; the 60fps grid holds m/60 < 8/24
        # -> exactly 20 output frames
        assert len(out) == 20
        assert [u.pts for u in out] == [m * 400 for m in range(20)]
        assert all(u.duration == 400 for u in out)
        # monotonic, gapless, and duration-preserving: the output clip
        # spans exactly the source window (what keeps audio in sync)
        assert sum(u.duration for u in out) == n * 1000
        assert plan.output_spec.timeline.cadence == Fraction(60)

    def test_downsample_grid(self):
        out, _ = run_interpolation(source_units(8), 12)
        assert [u.pts for u in out] == [m * 2000 for m in range(4)]
        assert sum(u.duration for u in out) == 8 * 1000

    def test_hard_cut_keeps_grid_monotonic_and_complete(self):
        out, _ = run_interpolation(source_units(8), 60, cut_at=4)
        # the pre-cut tail drains via flush, the post-cut segment starts
        # a fresh pair, and the target grid never restarts
        assert [u.pts for u in out] == [m * 400 for m in range(20)]
        flagged = [i for i, u in enumerate(out)
                   if any(b.kind is BoundaryKind.HARD_CUT
                          for b in u.boundaries)]
        assert len(flagged) == 1
        assert sum(u.duration for u in out) == 8 * 1000

    def test_cancel_closes_the_native_session(self):
        from kinovsr.pipeline import resolve_pipeline
        from kinovsr.pipeline.builder import build_processors
        from kinovsr.pipeline.scheduler import run_chain

        config = {
            "pipeline": ["interp"],
            "interp": {"processor": "videotoolbox",
                       "capability": "interpolate", "target_fps": 60},
        }
        plan = resolve_pipeline(config, input_spec=bgra_stream(),
                                settings=SETTINGS)
        built = build_processors(plan, CONTEXT)
        processor = built[0][1]
        stream = run_chain(built, source_units(8), CONTEXT)
        try:
            first = next(stream)
        except MediaError as exc:
            pytest.skip(str(exc))
        assert first.pts == 0
        stream.close()
        assert processor._session is None  # closed exactly once

    def test_default_host_output_survives_pool_reuse_and_session_close(self):
        from kinovsr.pipeline import open_pipeline

        config = {
            "pipeline": ["interp"],
            "interp": {
                "processor": "videotoolbox",
                "capability": "interpolate",
                "target_fps": 60,
            },
        }
        session = open_pipeline(
            config, bgra_stream(), settings=SETTINGS
        )
        try:
            with session, session.process(source_units(12)) as run:
                first = next(run)
                original = cv_digest(first.payload)
                later = list(run)
                assert len(later) == 29
                assert cv_digest(first.payload) == original
        except MediaError as exc:
            pytest.skip(str(exc))

        assert cv_digest(first.payload) == original


@pytest.mark.integration
class TestTimelineOrigin:
    """A nonzero source origin must be preserved, not silently re-based
    to zero - re-basing would desync the video from sibling streams."""

    @staticmethod
    def shifted(units, offset):
        return [u.retimed(u.pts + offset) for u in units]

    def test_nonzero_origin_anchors_the_grid(self):
        origin = 48000  # 2 seconds into the timeline, grid-aligned
        out, _ = run_interpolation(
            self.shifted(source_units(8), origin), 60)
        assert [u.pts for u in out] == [origin + m * 400 for m in range(20)]
        assert sum(u.duration for u in out) == 8 * 1000

    def test_non_grid_aligned_origin_is_kept_exactly(self):
        origin = 5007   # not a multiple of any frame duration
        out, _ = run_interpolation(
            self.shifted(source_units(8), origin), 60)
        assert out[0].pts == origin
        assert [u.pts for u in out] == [origin + m * 400 for m in range(20)]

    def test_negative_host_origin_is_not_mistaken_for_file_context(self):
        origin = -5007
        out, _ = run_interpolation(
            self.shifted(source_units(8), origin), 60)
        assert out[0].pts == origin
        assert [u.pts for u in out] == [origin + m * 400 for m in range(20)]

    def test_origin_survives_a_hard_cut(self):
        origin = 24000
        out, _ = run_interpolation(
            self.shifted(source_units(8), origin), 60, cut_at=4)
        assert [u.pts for u in out] == [origin + m * 400 for m in range(20)]
