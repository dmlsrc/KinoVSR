"""Writer-pool offers never outlive terminal VideoToolbox preparation."""

from __future__ import annotations

from fractions import Fraction

import pytest

from kinovsr.processors import (
    DType,
    Geometry,
    Layout,
    MediaError,
    PipelineContext,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.processors.videotoolbox import (
    VtInterpolateConfig,
    VtInterpolateProcessor,
    VtUpscaleConfig,
    VtUpscaleProcessor,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit
CONTEXT = PipelineContext(settings=Settings())


def _stream(layout: Layout) -> StreamSpec:
    dtype = DType.UINT8 if layout is not Layout.CV_RGBA_HALF else DType.FLOAT16
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709",
            full_range=False,
            geometry=Geometry(128, 96),
            layout=layout,
            dtype=dtype,
        ),
        timeline=TimelineSpec(time_base=Fraction(1, 24000), cadence=Fraction(24)),
    )


class _NativeSession:
    def __init__(self, *_args, **_kwargs) -> None:
        self.dst_attrs = {"PixelFormatType": 123}
        self.out_w = 256
        self.out_h = 192
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_interpolate_consumes_pool_offer_during_prepare(monkeypatch):
    from kinovsr.native import temporal
    from kinovsr.processors import videotoolbox

    processor = VtInterpolateProcessor(VtInterpolateConfig(target_fps=Fraction(60), mode="normal"))
    pool = object()
    processor._bind_output_pool(pool, 123, 128, 96)
    observed = []
    monkeypatch.setattr(temporal, "VtfrcSession", _NativeSession)

    def apply(_session, binding, width, height):
        assert processor._output_pool_binding is None
        observed.append((binding, width, height))

    monkeypatch.setattr(videotoolbox, "apply_output_pool", apply)

    processor.prepare(_stream(Layout.CV_BGRA), CONTEXT)
    processor.close(CONTEXT)

    assert observed == [((pool, 123, 128, 96), 128, 96)]
    assert processor._output_pool_binding is None


def test_upscale_consumes_pool_offer_during_prepare(monkeypatch):
    from kinovsr.native import vsr
    from kinovsr.processors import videotoolbox

    processor = VtUpscaleProcessor(VtUpscaleConfig(mode="fast"))
    pool = object()
    processor._bind_output_pool(pool, 123, 256, 192)
    observed = []
    monkeypatch.setattr(vsr, "VsrSession", _NativeSession)

    def apply(_session, binding, width, height):
        assert processor._output_pool_binding is None
        observed.append((binding, width, height))

    monkeypatch.setattr(videotoolbox, "apply_output_pool", apply)

    processor.prepare(_stream(Layout.CV_NV12), CONTEXT)
    processor.close(CONTEXT)

    assert observed == [((pool, 123, 256, 192), 256, 192)]
    assert processor._output_pool_binding is None


def test_failed_prepare_and_unprepared_close_drop_pool_offers(monkeypatch):
    from kinovsr.native import temporal

    class Failure:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("native construction failed")

    monkeypatch.setattr(temporal, "VtfrcSession", Failure)
    processor = VtInterpolateProcessor(VtInterpolateConfig(target_fps=Fraction(60), mode="normal"))
    processor._bind_output_pool(object(), 123, 128, 96)

    with pytest.raises(MediaError, match="native construction failed"):
        processor.prepare(_stream(Layout.CV_BGRA), CONTEXT)
    assert processor._output_pool_binding is None

    unprepared = VtUpscaleProcessor(VtUpscaleConfig(mode="fast"))
    unprepared._bind_output_pool(object(), 123, 256, 192)
    unprepared.close(CONTEXT)
    assert unprepared._output_pool_binding is None


def test_failed_mlx_upload_pool_is_wrapped_and_cleared(monkeypatch):
    from kinovsr.processors import videotoolbox

    class Failure:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("upload pool failed")

    monkeypatch.setattr(videotoolbox, "MlxUploadPool", Failure)
    processor = VtInterpolateProcessor(VtInterpolateConfig(target_fps=Fraction(60), mode="normal"))

    with pytest.raises(MediaError, match="upload pool failed"):
        processor.prepare(_stream(Layout.MLX_RGB_HWC), CONTEXT)

    assert processor._session is None
    assert processor._upload_bridge is None


def test_failed_pool_binding_closes_constructed_sessions(monkeypatch):
    from kinovsr.native import temporal, vsr
    from kinovsr.processors import videotoolbox

    sessions = []

    class NativeSession(_NativeSession):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            sessions.append(self)

    def fail_binding(*_args):
        raise RuntimeError("pool binding failed")

    monkeypatch.setattr(temporal, "VtfrcSession", NativeSession)
    monkeypatch.setattr(vsr, "VsrSession", NativeSession)
    monkeypatch.setattr(videotoolbox, "apply_output_pool", fail_binding)

    interpolate = VtInterpolateProcessor(
        VtInterpolateConfig(target_fps=Fraction(60), mode="normal")
    )
    upscale = VtUpscaleProcessor(VtUpscaleConfig(mode="fast"))
    with pytest.raises(MediaError, match="pool binding failed"):
        interpolate.prepare(_stream(Layout.CV_BGRA), CONTEXT)
    with pytest.raises(MediaError, match="pool binding failed"):
        upscale.prepare(_stream(Layout.CV_NV12), CONTEXT)

    assert len(sessions) == 2
    assert all(session.closed for session in sessions)
    assert interpolate._session is None
    assert upscale._session is None
