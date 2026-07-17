"""Spec values, constraint checking, coherence, and the foundation bridge."""

import dataclasses
from fractions import Fraction

import pytest

from kinovsr.processors import (
    ColorMatrix,
    ColorPrimaries,
    ColorRange,
    Domain,
    DType,
    FrameSpec,
    Geometry,
    Layout,
    StreamConstraint,
    StreamSpec,
    TimelineSpec,
    TransferFunction,
    coherence_violations,
    describe_spec,
    frame_spec_for_matrix,
)

pytestmark = pytest.mark.unit


def mlx_stream(width=640, height=480, *, lookahead=True,
               cadence=Fraction(25)) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False,
            geometry=Geometry(width, height)),
        timeline=TimelineSpec(time_base=Fraction(1, 24000), cadence=cadence),
        seekable=True,
        lookahead_available=lookahead,
    )


class TestSpecValues:
    def test_specs_are_frozen(self):
        spec = mlx_stream()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.frame.layout = Layout.CV_BGRA
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.seekable = False

    def test_geometry_scaled_preserves_pixel_aspect(self):
        geo = Geometry(720, 480, Fraction(8, 9))
        up = geo.scaled(4)
        assert (up.width, up.height) == (2880, 1920)
        assert up.pixel_aspect == Fraction(8, 9)

    def test_describe_is_one_line(self):
        text = describe_spec(mlx_stream())
        assert "\n" not in text
        assert "640x480" in text and "bt709" in text and "25fps" in text


class TestFoundationBridge:
    @pytest.mark.parametrize(
        ("token", "matrix", "primaries", "transfer"),
        [
            ("bt601", ColorMatrix.BT601, ColorPrimaries.SMPTE_C,
             TransferFunction.BT709),
            ("709", ColorMatrix.BT709, ColorPrimaries.BT709,
             TransferFunction.BT709),
            ("bt2020", ColorMatrix.BT2020, ColorPrimaries.BT2020,
             TransferFunction.BT2020),
        ])
    def test_matrix_families(self, token, matrix, primaries, transfer):
        frame = frame_spec_for_matrix(
            token, full_range=True, geometry=Geometry(64, 64))
        assert frame.color_matrix is matrix
        assert frame.color_primaries is primaries
        assert frame.transfer_function is transfer
        assert frame.color_range is ColorRange.FULL

    def test_unknown_token_raises(self):
        with pytest.raises(ValueError, match="unknown color matrix"):
            frame_spec_for_matrix("bt470", full_range=False,
                                  geometry=Geometry(64, 64))

    def test_cv_layout_forces_its_dtype(self):
        frame = frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(64, 64),
            layout=Layout.CV_RGBA_HALF, dtype=DType.FLOAT32)
        assert frame.dtype is DType.FLOAT16


class TestConstraint:
    def test_accepting_constraint_reports_nothing(self):
        constraint = StreamConstraint(
            layouts=(Layout.MLX_RGB_HWC,),
            dtypes=(DType.FLOAT32, DType.FLOAT16),
            domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
        )
        assert constraint.violations(mlx_stream()) == ()

    def test_layout_and_domain_mismatches_are_both_reported(self):
        constraint = StreamConstraint(
            layouts=(Layout.CV_NV12,),
            domains=(Domain.CODED,),
        )
        violations = constraint.violations(mlx_stream())
        assert {v.field for v in violations} == {
            "frame.layout", "frame.domain"}
        layout_violation = next(v for v in violations
                                if v.field == "frame.layout")
        assert layout_violation.accepted == "cv_nv12"
        assert layout_violation.actual == "mlx_rgb_hwc"

    def test_geometry_bounds(self):
        constraint = StreamConstraint(min_side=96, max_side=960)
        assert constraint.violations(mlx_stream(640, 480)) == ()
        too_small = constraint.violations(mlx_stream(640, 64))
        too_big = constraint.violations(mlx_stream(4096, 2160))
        assert [v.field for v in too_small] == ["frame.geometry"]
        assert "min side >= 96" in too_small[0].accepted
        assert [v.field for v in too_big] == ["frame.geometry"]

    def test_lookahead_requirement(self):
        constraint = StreamConstraint(requires_lookahead=True)
        assert constraint.violations(mlx_stream(lookahead=True)) == ()
        violations = constraint.violations(mlx_stream(lookahead=False))
        assert [v.field for v in violations] == ["lookahead_available"]

class TestCoherence:
    def test_valid_spec_is_coherent(self):
        assert coherence_violations(mlx_stream()) == ()

    def test_cv_layout_with_wrong_dtype_is_incoherent(self):
        frame = FrameSpec(
            layout=Layout.CV_BGRA, dtype=DType.FLOAT32,
            color_range=ColorRange.VIDEO, color_matrix=ColorMatrix.BT709,
            color_primaries=ColorPrimaries.BT709,
            transfer_function=TransferFunction.BT709,
            domain=Domain.CODED, geometry=Geometry(64, 64))
        spec = StreamSpec(frame=frame, timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))
        violations = coherence_violations(spec)
        assert [v.field for v in violations] == ["frame.dtype"]

    def test_coded_domain_cannot_ride_mlx_rgb(self):
        frame = dataclasses.replace(
            mlx_stream().frame, domain=Domain.CODED)
        spec = StreamSpec(frame=frame, timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))
        violations = coherence_violations(spec)
        assert [v.field for v in violations] == ["frame.domain"]
