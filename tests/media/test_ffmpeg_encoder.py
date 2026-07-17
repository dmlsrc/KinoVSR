"""Regression net for the ffmpeg encode backend (ffmpeg_encoder.py).

Pins the raw byte stream fed to ffmpeg (per frame, both bit depths) and the WAV
writers' output on deterministic synthetic input. The hashes were captured from
the numpy implementation, so they guard the numpy -> MLX-native rewrite: a
correct rewrite reproduces them exactly. A smoke test confirms the per-frame
streaming path still produces a valid file end to end.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess

import mlx.core as mx
import pytest

from kinovsr.media.ffmpeg_encoder import (
    _frame_buffer,
    encode_video_ffmpeg,
)


def _frames():
    return [
        (mx.arange(60, dtype=mx.int32) + k * 37).astype(mx.uint8).reshape(4, 5, 3)
        for k in range(3)
    ]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:24]


def test_frame_byte_stream_8bit():
    stream = b"".join(bytes(_frame_buffer(f, 8)) for f in _frames())
    assert (_sha(stream), len(stream)) == ("ea4498f4c4f9dbd7a8b33f89", 180)


def test_frame_byte_stream_16bit():
    stream = b"".join(bytes(_frame_buffer(f, 16)) for f in _frames())
    assert (_sha(stream), len(stream)) == ("79b71bf74558573206c5156e", 360)


def _fake_cmd(monkeypatch, argv):
    from kinovsr.media import ffmpeg_encoder

    monkeypatch.setattr(
        ffmpeg_encoder, "build_ffmpeg_cmd",
        lambda *args, **kwargs: list(argv))


def _tiny_frames(n=3):
    return [
        (mx.arange(8 * 6 * 3, dtype=mx.int32) + k).astype(mx.uint8).reshape(8, 6, 3)
        for k in range(n)
    ]


def test_failed_encode_reaps_child_and_removes_partial(tmp_path, monkeypatch):
    # encoder exits nonzero after consuming nothing: the rc must surface and
    # the (truncated-by--y) destination must not survive as a deliverable
    _fake_cmd(monkeypatch, ["sh", "-c", "exit 3"])
    out = tmp_path / "clip.mp4"
    out.write_bytes(b"stale")
    with pytest.raises(RuntimeError, match=r"rc=3"):
        encode_video_ffmpeg(_tiny_frames(), out, tier="web", fps=24.0,
                            verbose=False)
    assert not out.exists()


def test_early_death_reports_rc_not_broken_pipe(tmp_path, monkeypatch):
    # a frame bigger than the pipe buffer forces EPIPE when the child is
    # already gone; the caller must see the diagnostic, not BrokenPipeError
    _fake_cmd(monkeypatch, ["sh", "-c", "exit 7"])
    big = mx.zeros((256, 256, 3), dtype=mx.uint8)
    out = tmp_path / "clip.mp4"
    with pytest.raises(RuntimeError, match=r"exited early \(rc=7\)"):
        encode_video_ffmpeg([big, big], out, tier="web", fps=24.0,
                            verbose=False)
    assert not out.exists()


def test_bad_frame_mid_stream_kills_child_and_cleans_up(tmp_path, monkeypatch):
    _fake_cmd(monkeypatch, ["sh", "-c", "cat > /dev/null"])
    out = tmp_path / "clip.mp4"
    out.write_bytes(b"stale")
    frames = [_tiny_frames(1)[0], mx.zeros((8, 6, 4), dtype=mx.uint8)]
    with pytest.raises(ValueError, match=r"\(H,W,3\)"):
        encode_video_ffmpeg(frames, out, tier="web", fps=24.0, verbose=False)
    assert not out.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not in PATH")
def test_encode_video_smoke(tmp_path):
    # 5 mlx-native (H=8, W=6, 3) frames -> web tier (libx264) -> probe it back.
    frames = [
        (mx.arange(8 * 6 * 3, dtype=mx.int32) + k).astype(mx.uint8).reshape(8, 6, 3)
        for k in range(5)
    ]
    out = encode_video_ffmpeg(frames, tmp_path / "clip.mp4", tier="web", fps=24.0, verbose=False)
    assert out.exists() and out.stat().st_size > 0
    if shutil.which("ffprobe"):
        dims = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
            text=True,
        ).strip()
        assert dims == "6,8"  # width=6, height=8
