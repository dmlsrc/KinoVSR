"""The legacy native encoder preserves VideoToolbox destination padding."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_native_encoder_forwards_producer_attrs_before_pool_binding(monkeypatch, tmp_path):
    import numpy as np

    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native import encode
    from kinovsr.native.frameworks import Quartz

    dst_attrs = {
        "PixelFormatType": pb.PIX_NV12,
        "Width": 16,
        "Height": 16,
        Quartz.kCVPixelBufferExtendedPixelsBottomKey: 8,
    }
    offered_pool = object()
    captured = {}

    class Session:
        def __init__(self, *_args, **_kwargs):
            self.dst_attrs = dst_attrs
            self.bound_pool = None
            captured["session"] = self

        def use_dst_pool(self, pool):
            self.bound_pool = pool

        def upscale_to_buffer(self, _frame, _index):
            return object()

        def flush_pools(self):
            pass

        def close(self):
            pass

    class Adaptor:
        def pixelBufferPool(self):
            return offered_pool

    class Writer:
        def __init__(self, output_path, *_args, **kwargs):
            self.output_path = output_path
            self.adaptor = Adaptor()
            captured.update(kwargs)

        def append(self, _buffer):
            pass

        def finish(self):
            self.output_path.touch()

    monkeypatch.setattr(encode, "VsrSession", Session)
    monkeypatch.setattr(encode, "AVWriter", Writer)

    output = encode.encode_video_videotoolbox(
        [np.zeros((8, 8, 3), dtype=np.float32)],
        tmp_path / "output.mp4",
        fps=24,
        vsr_spatial_mode="fast",
    )

    assert output == tmp_path / "output.mp4"
    assert captured["source_attrs"] is dst_attrs
    assert captured["source_pixel_format"] == pb.PIX_NV12
    assert captured["session"].bound_pool is offered_pool
