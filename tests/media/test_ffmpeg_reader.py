"""Tests for the ffmpeg (PyAV) compatibility reader.

Self-contained: the fixture clip is muxed by PyAV itself (mpeg4 in MP4, fixed
GOP, tagged 601), so no binary fixtures and no dependency on files AVFoundation
can or cannot read.
"""
import math
from fractions import Fraction
from types import SimpleNamespace

import av
import mlx.core as mx
import pytest

from kinovsr.media import ffmpeg_reader as fr
from kinovsr.media.pixel_buffers import (
    PIX_BGRA,
    PIX_RGBAHALF,
    read_buffer_rgb_f32,
)

W, H, N, FPS, GOP = 160, 128, 24, 25, 8


def test_explicit_timing_origin_preserves_nonzero_stream_start_indexing():
    stream = SimpleNamespace(
        start_time=10000,
        time_base=Fraction(1, 1000),
    )
    assert fr._pts_index(10000, stream, 25.0) == 0
    assert fr._pts_index(
        10000, stream, 25.0, origin=Fraction(10)) == 0
    assert fr._pts_index(
        10040, stream, 25.0, origin=Fraction(10)) == 1


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Mux a tiny mpeg4 clip with a known GOP and a moving gradient."""
    path = tmp_path_factory.mktemp("ffr") / "clip.mp4"
    out = av.open(str(path), "w")
    vs = out.add_stream("mpeg4", rate=FPS)
    vs.width, vs.height = W, H
    vs.pix_fmt = "yuv420p"
    vs.options = {"g": str(GOP), "bf": "0", "qscale": "2"}
    for t in range(N):
        # smooth ramp + a moving bar; distinct mean per frame for identity checks
        base = bytearray()
        for _y in range(H):
            row = bytes(min(255, (x + 2 * t) % 256) for x in range(W))
            base += row
        frame = av.VideoFrame(W, H, "gray")
        frame.planes[0].update(bytes(base))
        frame = frame.reformat(format="yuv420p")
        for pkt in vs.encode(frame):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()
    return path


def test_probe_video_shape(clip):
    w, h, fps, n, transform, par = fr.probe_video(clip)
    assert (w, h) == (W, H)
    assert abs(fps - FPS) < 0.01
    assert n == N
    assert par is None                      # square pixels
    assert transform is not None            # identity CGAffineTransform


def test_keyframes_match_decoded_positions(clip):
    """The metadata keyframe list must equal the key positions in the reader's
    own decode order (self-consistency is the contract gop-align relies on)."""
    kf = fr.keyframe_display_indices(clip)
    assert kf[0] == 0 and kf == sorted(kf)
    c = av.open(str(clip))
    vs = [s for s in c.streams if s.type == "video"][0]
    decoded = [i for i, f in enumerate(c.decode(vs)) if f.key_frame]
    c.close()
    assert kf == decoded
    assert all(b - a == GOP for a, b in zip(kf, kf[1:], strict=False))   # fixed GOP respected


def test_frames_stream_and_trim(clip):
    # full stream arrives complete and in order via the RGBAHalf path
    frames = [pb for chunk in fr.iter_video_buffer_chunks(clip, PIX_RGBAHALF, chunk_size=5)
              for pb in chunk]
    assert len(frames) == N
    f0 = read_buffer_rgb_f32(frames[0])
    assert f0.shape == (H, W, 3)
    assert bool(mx.all(mx.isfinite(f0)))
    # frame-exact trim: [10, 14) yields 4 frames matching the full stream's
    trimmed = [pb for chunk in fr.iter_video_buffer_chunks(
        clip, PIX_RGBAHALF, chunk_size=5, start_frame=10, end_frame=14) for pb in chunk]
    assert len(trimmed) == 4
    a = read_buffer_rgb_f32(frames[10])
    b = read_buffer_rgb_f32(trimmed[0])
    assert float(mx.max(mx.abs(a - b))) < 1e-6


def test_bgra_path_matches_rgbahalf(clip):
    ch = next(iter(fr.iter_video_buffer_chunks(clip, PIX_BGRA, chunk_size=1)))
    hh = next(iter(fr.iter_video_buffer_chunks(clip, PIX_RGBAHALF, chunk_size=1)))
    a = read_buffer_rgb_f32(ch[0])
    b = read_buffer_rgb_f32(hh[0])
    assert float(mx.max(mx.abs(a - b))) < (1.5 / 255.0)   # 8-bit rounding only


def test_unsupported_format_raises(clip):
    from kinovsr.media.pixel_buffers import PIX_NV12
    with pytest.raises(ValueError):
        next(iter(fr.iter_video_buffer_chunks(clip, PIX_NV12)))


def test_probe_color_shape(clip):
    c = fr.probe_color(clip)
    assert set(c) == {"primaries", "transfer", "matrix", "full_range", "tagged"}
    assert isinstance(c["full_range"], bool) and isinstance(c["tagged"], bool)


def test_forced_color_range_differs(clip):
    """Forcing full-range read must change the pixels of a video-range clip."""
    a = read_buffer_rgb_f32(next(iter(fr.iter_forced_color_chunks(
        clip, PIX_RGBAHALF, "ITU_R_601_4", False, chunk_size=1)))[0])
    b = read_buffer_rgb_f32(next(iter(fr.iter_forced_color_chunks(
        clip, PIX_RGBAHALF, "ITU_R_601_4", True, chunk_size=1)))[0])
    assert float(mx.max(mx.abs(a - b))) > 0.02


def test_audio_track_decodes(tmp_path):
    """A muxed sine track round-trips into an AudioTrack at the right length."""
    path = tmp_path / "tone.mp4"
    out = av.open(str(path), "w")
    vs = out.add_stream("mpeg4", rate=FPS)
    vs.width, vs.height = W, H
    vs.pix_fmt = "yuv420p"
    asr = out.add_stream("aac", rate=48000, layout="mono")
    dur_s = 1.0
    n_samp = int(48000 * dur_s)
    import array
    sine = array.array("f", (0.25 * math.sin(2 * math.pi * 440 * i / 48000)
                             for i in range(n_samp)))
    # the AAC encoder takes fixed 1024-sample PLANAR float frames
    step = 1024
    for off in range(0, n_samp - step + 1, step):
        af = av.AudioFrame(format="fltp", layout="mono", samples=step)
        af.sample_rate = 48000
        af.pts = off
        af.planes[0].update(sine[off:off + step].tobytes())
        for pkt in asr.encode(af):
            out.mux(pkt)
    for pkt in asr.encode():
        out.mux(pkt)
    for _t in range(int(FPS * dur_s)):
        vf = av.VideoFrame(W, H, "gray").reformat(format="yuv420p")
        for pkt in vs.encode(vf):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()

    at = fr.read_audio_track(path)
    assert at is not None
    assert at.sample_rate == 48000 and at.channels == 1
    # AAC pads to frame boundaries; length within ~2 AAC frames of the source
    assert abs(at.n_samples - n_samp) < 4096


def test_coded_frame_sizes(clip):
    sizes = fr.coded_frame_sizes(clip)
    assert len(sizes) == N
    assert all(s > 0 for s in sizes)
    # the fixed-GOP fixture makes every keyframe the local maximum
    kf = fr.keyframe_display_indices(clip)
    p_sizes = [s for i, s in enumerate(sizes) if i not in kf]
    assert min(sizes[i] for i in kf) > max(p_sizes)
    # cross-reader parity: the native reader must account the same bytes
    try:
        from kinovsr.media import video_reader as nvr
        native = nvr.coded_frame_sizes(clip)
    except Exception:
        pytest.skip("native reader unavailable for this container")
    assert sum(native) == sum(sizes)
    assert len(native) == len(sizes)
