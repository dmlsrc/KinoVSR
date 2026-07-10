"""Protocol conformance, context values, and edge-error rendering."""

import dataclasses
from fractions import Fraction

import pytest

from kinovsr.processors import (
    Boundary,
    BoundaryKind,
    FieldViolation,
    FrameUnit,
    Geometry,
    PipelineContext,
    Processor,
    StreamEdgeError,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


class MinimalProcessor:
    """The smallest structurally-conformant stage: passthrough."""

    def prepare(self, input_spec, context):
        pass

    def process(self, unit, context):
        yield unit

    def reset(self, boundary, context):
        pass

    def flush(self, context):
        return ()

    def close(self, context):
        pass


class TestProtocol:
    def test_minimal_processor_conforms(self):
        assert isinstance(MinimalProcessor(), Processor)

    def test_non_processor_does_not_conform(self):
        assert not isinstance(object(), Processor)

    def test_passthrough_yields_the_unit(self):
        context = PipelineContext(settings=Settings())
        unit = FrameUnit(payload="x", pts=0, duration=1)
        stage = MinimalProcessor()
        stage.prepare(None, context)
        assert list(stage.process(unit, context)) == [unit]
        stage.reset(Boundary(BoundaryKind.STREAM_START), context)
        assert list(stage.flush(context)) == []
        stage.close(context)


class TestContext:
    def test_context_is_frozen(self):
        context = PipelineContext(settings=Settings())
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.stage_id = "x"

    def test_for_stage_returns_new_value(self):
        context = PipelineContext(settings=Settings())
        staged = context.for_stage("denoise-1")
        assert staged.stage_id == "denoise-1"
        assert context.stage_id is None
        assert staged.settings is context.settings
        assert staged.reporter is context.reporter


class TestEdgeError:
    def test_names_both_sides_and_fields(self):
        spec = StreamSpec(
            frame=frame_spec_for_matrix(
                "bt709", full_range=False, geometry=Geometry(64, 64)),
            timeline=TimelineSpec(
                time_base=Fraction(1, 24000), cadence=Fraction(25)))
        error = StreamEdgeError(
            upstream="denoise",
            downstream="upscale",
            violations=(
                FieldViolation("frame.layout", "cv_nv12", "mlx_rgb_hwc"),
                FieldViolation("lookahead_available", "true", "false"),
            ),
            produced=spec,
        )
        message = str(error)
        assert "invalid edge denoise -> upscale:" in message
        assert ("frame.layout: upscale accepts cv_nv12; "
                "denoise produces mlx_rgb_hwc") in message
        assert "lookahead_available" in message
        assert "64x64" in message  # the produced-stream summary line
        assert error.upstream == "denoise"
        assert error.violations[0].field == "frame.layout"
