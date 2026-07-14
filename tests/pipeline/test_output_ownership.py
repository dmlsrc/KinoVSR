"""Session output ownership (finding #1). An empty pipeline is the sharpest
case: with no stage, the output payload IS the borrowed input buffer unless
the session copies it. The default (retain_outputs=True) must hand back a
fresh owned buffer; retain_outputs=False is the zero-copy opt-out that
yields the borrowed input as-is.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from kinovsr.pipeline import open_pipeline
from kinovsr.processors import (
    FrameUnit,
    Geometry,
    Layout,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.integration

def _cv_spec(matrix: str = "bt709") -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            matrix, full_range=False, geometry=Geometry(8, 8),
            layout=Layout.CV_RGBA_HALF),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


def _cv_unit():
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native.frameworks import Quartz
    attrs = {
        Quartz.kCVPixelBufferPixelFormatTypeKey: pb.PIX_RGBAHALF,
        Quartz.kCVPixelBufferWidthKey: 8,
        Quartz.kCVPixelBufferHeightKey: 8,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
    }
    buf = pb.make_pixel_buffer_from_attrs(8, 8, attrs)
    return FrameUnit(payload=buf, pts=0, duration=960), buf


def test_default_session_hands_back_owned_cv_outputs():
    unit, buf = _cv_unit()
    session = open_pipeline({"pipeline": []}, _cv_spec(), settings=Settings())
    with session, session.process([unit]) as run:
        out = list(run)
    assert len(out) == 1
    # Owned: a fresh buffer, not the borrowed input.
    assert out[0].payload is not buf


def test_zero_copy_session_aliases_the_borrowed_input():
    unit, buf = _cv_unit()
    session = open_pipeline({"pipeline": []}, _cv_spec(), settings=Settings())
    with session, session.process([unit], retain_outputs=False) as run:
        out = list(run)
    assert len(out) == 1
    # Zero-copy: the pass-through payload IS the borrowed input buffer.
    assert out[0].payload is buf


@pytest.mark.parametrize("matrix_token", ["bt709", "bt2020"])
def test_retained_copy_reconciles_modeled_attachments_to_output_spec(matrix_token):
    from kinovsr.native.frameworks import Quartz

    unit, buf = _cv_unit()
    conflicting = {
        Quartz.kCVImageBufferColorPrimariesKey:
            Quartz.kCVImageBufferColorPrimaries_P3_D65,
        Quartz.kCVImageBufferTransferFunctionKey:
            Quartz.kCVImageBufferTransferFunction_ITU_R_2100_HLG,
        Quartz.kCVImageBufferYCbCrMatrixKey:
            Quartz.kCVImageBufferYCbCrMatrix_ITU_R_601_4,
        Quartz.kCVImageBufferPixelAspectRatioKey: {
            Quartz.kCVImageBufferPixelAspectRatioHorizontalSpacingKey: 4,
            Quartz.kCVImageBufferPixelAspectRatioVerticalSpacingKey: 3,
        },
        "KinoVSRTestUnmodeled": "preserved",
    }
    for key, value in conflicting.items():
        Quartz.CVBufferSetAttachment(
            buf, key, value, Quartz.kCVAttachmentMode_ShouldPropagate
        )

    session = open_pipeline(
        {"pipeline": []}, _cv_spec(matrix_token), settings=Settings()
    )
    with session, session.process([unit]) as run:
        retained = list(run)[0].payload
    attachments = dict(
        Quartz.CVBufferCopyAttachments(
            retained, Quartz.kCVAttachmentMode_ShouldPropagate
        )
        or {}
    )

    expected_primaries = (
        Quartz.kCVImageBufferColorPrimaries_ITU_R_2020
        if matrix_token == "bt2020"
        else Quartz.kCVImageBufferColorPrimaries_ITU_R_709_2
    )
    expected_matrix = (
        Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020
        if matrix_token == "bt2020"
        else Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2
    )
    assert attachments[Quartz.kCVImageBufferColorPrimariesKey] == expected_primaries
    assert attachments[Quartz.kCVImageBufferTransferFunctionKey] == (
        Quartz.kCVImageBufferTransferFunction_ITU_R_709_2
    )
    assert attachments[Quartz.kCVImageBufferYCbCrMatrixKey] == expected_matrix
    aspect = attachments[Quartz.kCVImageBufferPixelAspectRatioKey]
    assert aspect[Quartz.kCVImageBufferPixelAspectRatioHorizontalSpacingKey] == 1
    assert aspect[Quartz.kCVImageBufferPixelAspectRatioVerticalSpacingKey] == 1
    assert attachments["KinoVSRTestUnmodeled"] == "preserved"


def test_run_plan_is_retain_safe_by_default():
    from kinovsr.pipeline import resolve_pipeline, run_plan
    from kinovsr.processors import PipelineContext

    settings = Settings()
    plan = resolve_pipeline(
        {"pipeline": []}, input_spec=_cv_spec(), settings=settings
    )
    unit, buf = _cv_unit()

    out = list(run_plan(plan, [unit], PipelineContext(settings=settings)))

    assert out[0].payload is not buf


def test_run_plan_can_expose_borrowed_outputs_explicitly():
    from kinovsr.pipeline import resolve_pipeline, run_plan
    from kinovsr.processors import PipelineContext

    settings = Settings()
    plan = resolve_pipeline(
        {"pipeline": []}, input_spec=_cv_spec(), settings=settings
    )
    unit, buf = _cv_unit()

    out = list(
        run_plan(
            plan,
            [unit],
            PipelineContext(settings=settings),
            retain_outputs=False,
        )
    )

    assert out[0].payload is buf
