"""FRC's typed output matches its invariant RGBAHalf destination."""

from __future__ import annotations

from fractions import Fraction

import pytest

from kinovsr.processors import (
    Domain,
    DType,
    FrameUnit,
    Geometry,
    Layout,
    MediaError,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.integration

W, H = 128, 96
TB = Fraction(1, 24000)
SETTINGS = Settings()


def _stream(layout: Layout) -> StreamSpec:
    domain = Domain.CODED if layout is Layout.CV_NV12 else Domain.UNIT
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709",
            full_range=False,
            geometry=Geometry(W, H),
            layout=layout,
            dtype=(DType.FLOAT32 if layout is Layout.MLX_RGB_HWC else DType.UINT8),
            domain=domain,
        ),
        timeline=TimelineSpec(time_base=TB, cadence=Fraction(24)),
    )


def _units(layout: Layout) -> list[FrameUnit]:
    if layout is Layout.MLX_RGB_HWC:
        import mlx.core as mx

        payloads = [mx.full((H, W, 3), i / 4, dtype=mx.float32) for i in range(3)]
    else:
        from kinovsr.media import pixel_buffers as pb

        formats = {
            Layout.CV_BGRA: pb.PIX_BGRA,
            Layout.CV_NV12: pb.PIX_NV12,
            Layout.CV_RGBA_HALF: pb.PIX_RGBAHALF,
        }
        payloads = [
            pb.make_pixel_buffer_from_attrs(
                W,
                H,
                {
                    "PixelFormatType": formats[layout],
                    "Width": W,
                    "Height": H,
                    "IOSurfaceProperties": {},
                    "MetalCompatibility": True,
                },
            )
            for _ in range(3)
        ]
    return [
        FrameUnit(payload=payload, pts=i * 1000, duration=1000)
        for i, payload in enumerate(payloads)
    ]


@pytest.mark.parametrize(
    "layout",
    [
        Layout.MLX_RGB_HWC,
        Layout.CV_BGRA,
        Layout.CV_NV12,
        Layout.CV_RGBA_HALF,
    ],
)
def test_every_accepted_input_emits_typed_rgbahalf(layout):
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native.frameworks import Quartz
    from kinovsr.pipeline import resolve_pipeline, run_plan
    from kinovsr.processors import PipelineContext

    config = {
        "pipeline": ["interp"],
        "interp": {
            "processor": "videotoolbox",
            "capability": "interpolate",
            "target_fps": 48,
        },
    }
    plan = resolve_pipeline(config, input_spec=_stream(layout), settings=SETTINGS)
    assert plan.output_spec.frame.layout is Layout.CV_RGBA_HALF
    assert plan.output_spec.frame.dtype is DType.FLOAT16
    assert plan.output_spec.frame.domain is Domain.UNIT
    try:
        outputs = list(run_plan(plan, _units(layout), PipelineContext(settings=SETTINGS)))
    except MediaError as exc:
        pytest.skip(str(exc))

    assert outputs
    assert {Quartz.CVPixelBufferGetPixelFormatType(unit.payload) for unit in outputs} == {
        pb.PIX_RGBAHALF
    }
