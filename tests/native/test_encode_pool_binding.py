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
    owner_events = []

    class Lease:
        def __enter__(self):
            owner_events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            owner_events.append("exit")
            return False

    class Session:
        def __init__(self, *_args, **_kwargs):
            self.dst_attrs = dst_attrs
            self.bound_pool = None
            self.flush_count = 0
            captured["session"] = self

        def use_dst_pool(self, pool):
            self.bound_pool = pool

        def upscale_to_buffer(self, _frame, _index):
            return object()

        def flush_pools(self):
            self.flush_count += 1

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
    monkeypatch.setattr(pb, "ci_cache_owner", Lease)
    frame = np.zeros((8, 8, 3), dtype=np.float32)

    output = encode.encode_video_videotoolbox(
        [frame] * 64,
        tmp_path / "output.mp4",
        fps=24,
        vsr_spatial_mode="fast",
    )

    assert output == tmp_path / "output.mp4"
    assert captured["source_attrs"] is dst_attrs
    assert captured["source_pixel_format"] == pb.PIX_NV12
    assert captured["session"].bound_pool is offered_pool
    assert owner_events == ["enter", "exit"]
    # Central Core Image accounting replaced the old per-64-input cleanup;
    # bounded VideoToolbox pools retain their normal reuse set until close.
    assert captured["session"].flush_count == 0


def test_native_encoder_closes_core_image_owner_on_preflight_failure(monkeypatch, tmp_path):
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native import encode

    events = []

    class Lease:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")
            return False

    monkeypatch.setattr(pb, "ci_cache_owner", Lease)

    with pytest.raises(ValueError, match="empty frames"):
        encode.encode_video_videotoolbox(
            [],
            tmp_path / "output.mp4",
            fps=24,
        )

    assert events == ["enter", "exit"]


def test_native_encoder_closes_core_image_owner_on_append_failure(monkeypatch, tmp_path):
    import numpy as np

    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native import encode

    events = []

    class Lease:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")
            return False

    class Writer:
        def __init__(self, *_args, **_kwargs):
            self.adaptor = object()

        def append(self, _buffer):
            raise RuntimeError("append failed")

        def finish(self):
            events.append("finish")

        def cancel(self):
            events.append("cancel")

    monkeypatch.setattr(pb, "ci_cache_owner", Lease)
    monkeypatch.setattr(encode, "AVWriter", Writer)
    monkeypatch.setattr(
        encode,
        "_allocate_writer_src_buffer",
        lambda *_args: object(),
    )
    monkeypatch.setattr(pb, "upload_frame_to_buffer", lambda *_args: None)

    with pytest.raises(RuntimeError, match="append failed"):
        encode.encode_video_videotoolbox(
            [np.zeros((8, 8, 3), dtype=np.float32)],
            tmp_path / "output.mp4",
            fps=24,
        )

    # A failed encode loop must DISCARD the writer, never finalize it:
    # finish() here would leave a truncated but playable file at the
    # requested destination and could mask the loop's error.
    assert events == ["enter", "cancel", "exit"]
