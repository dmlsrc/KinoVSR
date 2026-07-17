"""Encode-loop failure must discard partial output, not finalize it."""

from __future__ import annotations

import pytest

from kinovsr.native.encode import _discard_failed_output

pytestmark = pytest.mark.unit


class _RecordingWriter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def cancel(self) -> None:
        self.calls.append("cancel")


class _WedgedWriter:
    def cancel(self) -> None:
        raise RuntimeError("native cancel wedged")


def test_discard_cancels_writer_and_removes_partial(tmp_path):
    writer = _RecordingWriter()
    partial = tmp_path / "out.mp4"
    partial.write_bytes(b"truncated")

    _discard_failed_output(writer, partial)

    assert writer.calls == ["cancel"]
    assert not partial.exists()


def test_discard_survives_cancel_failure_and_still_removes(tmp_path, caplog):
    partial = tmp_path / "out.mp4"
    partial.write_bytes(b"truncated")

    _discard_failed_output(_WedgedWriter(), partial)

    assert not partial.exists()


def test_discard_tolerates_missing_partial(tmp_path):
    writer = _RecordingWriter()

    _discard_failed_output(writer, tmp_path / "never-created.mp4")

    assert writer.calls == ["cancel"]


# ---- temp-then-publish: a pre-existing deliverable survives failure ----

def _stub_native(monkeypatch, writer_cls, session_cls=None):
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native import encode

    class Lease:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(pb, "ci_cache_owner", Lease)
    monkeypatch.setattr(encode, "AVWriter", writer_cls)
    monkeypatch.setattr(
        encode, "_allocate_writer_src_buffer", lambda *_args: object())
    monkeypatch.setattr(pb, "upload_frame_to_buffer", lambda *_args: None)
    if session_cls is not None:
        monkeypatch.setattr(encode, "VsrSession", session_cls)
    return encode


class _Adaptor:
    def pixelBufferPool(self):
        return object()


class _PathWriter:
    instances: list["_PathWriter"] = []

    def __init__(self, output_path, *_args, **kwargs):
        from pathlib import Path

        self.output_path = Path(output_path)
        self.label = kwargs.get("label", "video")
        self.calls: list[str] = []
        self.adaptor = _Adaptor()
        type(self).instances.append(self)

    def append(self, _buffer):
        self.calls.append("append")

    def finish(self):
        self.calls.append("finish")
        self.output_path.write_bytes(b"encoded")

    def cancel(self):
        self.calls.append("cancel")


def _frame():
    import numpy as np

    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_loop_failure_preserves_preexisting_destination(monkeypatch, tmp_path):
    class Writer(_PathWriter):
        instances = []

        def append(self, _buffer):
            raise RuntimeError("append failed")

    encode = _stub_native(monkeypatch, Writer)
    dest = tmp_path / "out.mp4"
    dest.write_bytes(b"precious")

    with pytest.raises(RuntimeError, match="append failed"):
        encode.encode_video_videotoolbox([_frame()], dest, fps=24)

    assert dest.read_bytes() == b"precious"
    writer = Writer.instances[-1]
    assert writer.output_path != dest
    assert "cancel" in writer.calls
    assert list(tmp_path.glob(".*.partial")) == []


def test_finish_failure_preserves_preexisting_destination(monkeypatch, tmp_path):
    class Writer(_PathWriter):
        instances = []

        def finish(self):
            raise RuntimeError("finalize wedged")

    encode = _stub_native(monkeypatch, Writer)
    dest = tmp_path / "out.mp4"
    dest.write_bytes(b"precious")

    with pytest.raises(RuntimeError, match="finalize wedged"):
        encode.encode_video_videotoolbox([_frame()], dest, fps=24)

    assert dest.read_bytes() == b"precious"
    assert list(tmp_path.glob(".*.partial")) == []


def test_success_publishes_finished_file_onto_destination(monkeypatch, tmp_path):
    class Writer(_PathWriter):
        instances = []

    encode = _stub_native(monkeypatch, Writer)
    dest = tmp_path / "out.mp4"
    dest.write_bytes(b"stale")

    out = encode.encode_video_videotoolbox([_frame()], dest, fps=24)

    assert out == dest
    assert dest.read_bytes() == b"encoded"
    assert list(tmp_path.glob(".*.partial")) == []


def test_companion_finish_failure_publishes_primary(monkeypatch, tmp_path):
    from kinovsr.media import pixel_buffers as pb

    class Session:
        def __init__(self, *_args, **_kwargs):
            self.dst_attrs = {
                "PixelFormatType": pb.PIX_NV12, "Width": 16, "Height": 16}

        def use_dst_pool(self, _pool):
            pass

        def upscale_to_buffer(self, _frame, _index):
            return object()

        def flush_pools(self):
            pass

        def close(self):
            pass

    class Writer(_PathWriter):
        instances = []

        def finish(self):
            if self.label == "encode_orig":
                raise RuntimeError("companion finalize wedged")
            super().finish()

    encode = _stub_native(monkeypatch, Writer, session_cls=Session)
    dest = tmp_path / "out.mp4"

    with pytest.raises(RuntimeError, match="companion finalize wedged"):
        encode.encode_video_videotoolbox(
            [_frame()], dest, fps=24, vsr_spatial_mode="fast",
            vsr_save_original=True)

    # the primary result is published; only the companion's partial is gone
    assert dest.read_bytes() == b"encoded"
    assert not (tmp_path / "out_orig.mp4").exists()
    assert list(tmp_path.glob(".*.partial")) == []
    companion = [w for w in Writer.instances if w.label == "encode_orig"][-1]
    assert "cancel" in companion.calls
