"""Tests for the ffmpeg (PyAV) compatibility reader.

Self-contained: the fixture clip is muxed by PyAV itself (mpeg4 in MP4, fixed
GOP, tagged 601), so no binary fixtures and no dependency on files AVFoundation
can or cannot read.
"""
import math
import subprocess
import sys
from array import array
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


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_direct_readers_reject_invalid_chunks_before_open(chunk_size):
    from pathlib import Path

    from kinovsr.media import video_reader

    missing = Path("must-not-open.mp4")
    with pytest.raises(ValueError, match="positive integer"):
        list(fr.iter_video_buffer_chunks(
            missing, PIX_BGRA, chunk_size=chunk_size))
    with pytest.raises(ValueError, match="positive integer"):
        list(video_reader.iter_video_buffer_chunks(
            missing, PIX_BGRA, chunk_size=chunk_size))


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
    chunks = list(fr.iter_video_buffer_chunks(
        clip, PIX_RGBAHALF, chunk_size=5))
    assert max(map(len, chunks)) == 5
    frames = [pb for chunk in chunks for pb in chunk]
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
    dur_s = 3.0
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

    window = fr.read_audio_track_window(
        path,
        start_sec=Fraction(1, 4),
        end_sec=Fraction(3, 4),
        max_duration_sec=Fraction(1, 4),
    )
    assert window is not None
    assert window.n_samples == 12_000
    assert window._source is None
    try:
        window_raw = window._read_interleaved(0, window.n_samples)
    finally:
        window.close()
    assert len(window_raw) == 48_000

    # Seeking with packet preroll must be sample-aligned and numerically
    # equivalent to decoding the full track then slicing. Separate codec
    # instances can differ by a few float32 rounding ulps.
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="fltp", layout=stream.layout, rate=stream.rate)
        decoded = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                decoded.append(
                    mx.array(memoryview(resampled.planes[0]).cast("B"))
                    .view(mx.float32)[:resampled.samples])
    expected = mx.concatenate(decoded)[12_000:24_000]
    actual = mx.array(memoryview(window_raw).cast("f"))
    assert mx.max(mx.abs(actual - expected)).item() < 1e-5

    late = fr.read_audio_track_window(
        path,
        start_sec=Fraction(2),
        end_sec=Fraction(9, 4),
    )
    assert late is not None and late.n_samples == 12_000
    try:
        late_raw = late._read_interleaved(0, late.n_samples)
    finally:
        late.close()
    late_expected = mx.concatenate(decoded)[96_000:108_000]
    late_actual = mx.array(memoryview(late_raw).cast("f"))
    assert mx.max(mx.abs(late_actual - late_expected)).item() < 1e-5


def _write_mkv_audio_timeline(path, spans, *, video_frames=0):
    """Mux AAC spans whose starts are absolute 48 kHz sample positions."""
    out = av.open(str(path), "w")
    video = None
    if video_frames:
        video = out.add_stream("mpeg4", rate=25)
        video.width = video.height = 64
        video.pix_fmt = "yuv420p"
        video.options = {"bf": "0"}
    audio = out.add_stream("aac", rate=48_000, layout="mono")
    for start, count in spans:
        for offset in range(0, count, 1024):
            n = min(1024, count - offset)
            samples = array("f", [0.25]) * n
            frame = av.AudioFrame(format="fltp", layout="mono", samples=n)
            frame.sample_rate = 48_000
            frame.pts = start + offset
            frame.time_base = Fraction(1, 48_000)
            frame.planes[0].update(samples.tobytes())
            for packet in audio.encode(frame):
                out.mux(packet)
    for packet in audio.encode():
        out.mux(packet)
    if video is not None:
        for _ in range(video_frames):
            frame = av.VideoFrame(64, 64, "gray")
            frame.planes[0].update(bytes(64 * 64))
            for packet in video.encode(frame.reformat(format="yuv420p")):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
    out.close()


def test_audio_window_handles_coarse_mkv_clock_without_fake_gaps(tmp_path):
    path = tmp_path / "coarse.mkv"
    _write_mkv_audio_timeline(path, [(0, 48_000)])

    track = fr.read_audio_track_window(path, end_sec=Fraction(1))
    assert track is not None
    try:
        raw = track._read_interleaved(0, 48_000)
    finally:
        track.close()

    assert len(raw) == 48_000 * 4


def test_audio_window_preserves_timestamp_gap_as_bounded_silence(tmp_path):
    path = tmp_path / "gap.mkv"
    _write_mkv_audio_timeline(
        path,
        [(0, 24_000), (48_000, 48_000)],
        video_frames=50,
    )

    track = fr.read_audio_track_window(path, end_sec=Fraction(2))
    assert track is not None
    try:
        raw = track._read_interleaved(0, 96_000)
    finally:
        track.close()

    assert len(raw) == 96_000 * 4
    samples = memoryview(raw).cast("f")
    assert all(sample == 0.0 for sample in samples[30_000:42_000])
    assert max(abs(sample) for sample in samples[55_000:56_000]) > 0.01


def test_audio_window_ends_cleanly_when_audio_is_shorter_than_video(tmp_path):
    path = tmp_path / "short.mkv"
    _write_mkv_audio_timeline(path, [(0, 24_000)], video_frames=50)

    track = fr.read_audio_track_window(path, end_sec=Fraction(2))
    assert track is not None
    try:
        raw = track._read_interleaved(0, 96_000)
        actual = len(raw) // 4
        assert 24_000 <= actual < 30_000
        assert track._read_interleaved(actual, 96_000) == b""
    finally:
        track.close()


def test_late_audio_window_work_does_not_scale_with_source_duration(tmp_path):
    results = []
    paths = []
    for duration in (300, 3600):
        path = tmp_path / f"sparse-{duration}.mkv"
        _write_mkv_audio_timeline(
            path,
            [(0, 1024), ((duration - 1) * 48_000, 48_000)],
        )
        track = fr.read_audio_track_window(
            path,
            start_sec=Fraction(duration - 10),
            end_sec=Fraction(duration),
        )
        assert track is not None and track._source is None
        try:
            raw = track._read_interleaved(0, 10 * 48_000)
            decoded = track._source._decoded_frame_count
        finally:
            track.close()
        results.append((len(raw), decoded))
        paths.append((path, duration))

    expected_bytes = 10 * 48_000 * 4
    assert [size for size, _ in results] == [expected_bytes, expected_bytes]
    assert max(decoded for _, decoded in results) <= 64
    assert abs(results[0][1] - results[1][1]) <= 1

    probe = (
        "import resource,sys;"
        "from fractions import Fraction;"
        "from pathlib import Path;"
        "from kinovsr.media.ffmpeg_reader import read_audio_track_window;"
        "d=int(sys.argv[2]);"
        "t=read_audio_track_window(Path(sys.argv[1]),"
        "start_sec=Fraction(d-10),end_sec=Fraction(d));"
        "raw=t._read_interleaved(0,10*48000);"
        "print(len(raw),t._source._decoded_frame_count,"
        "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);"
        "t.close()"
    )
    process_results = []
    for path, duration in paths:
        completed = subprocess.run(
            [sys.executable, "-c", probe, str(path), str(duration)],
            check=True,
            capture_output=True,
            text=True,
        )
        process_results.append(
            tuple(int(value) for value in completed.stdout.splitlines()[-1].split()))

    assert [item[0] for item in process_results] == [
        expected_bytes, expected_bytes]
    assert max(item[1] for item in process_results) <= 64
    # Fresh-process max RSS must not scale with the 12x source-duration delta.
    assert abs(process_results[0][2] - process_results[1][2]) <= 64 * 1024 * 1024


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


def test_corrupt_media_raises_operational_error(tmp_path):
    junk = tmp_path / "garbage.mp4"
    junk.write_bytes(b"\x00" * 4096)

    with pytest.raises(RuntimeError):
        fr.probe_video(junk)


def test_missing_file_keeps_oserror_typing(tmp_path):
    with pytest.raises(OSError):
        fr.probe_video(tmp_path / "absent.mp4")


# ------------------------------------------------- zero-byte repeat markers

class _InjectingContainer:
    """Wrap a real container, splicing zero-size packets into demux().

    Ogg/Theora represents "repeat the previous frame" as a ZERO-BYTE
    packet with a real timestamp, and libavformat's muxers refuse to
    write such packets, so the condition cannot be synthesized as a
    self-contained file; splicing at the demux seam exercises the exact
    reader path the real streams hit.
    """

    def __init__(self, container, position: str, pts_delta: int):
        self._c = container
        self._position = position          # "tail" | "head"
        self._delta = pts_delta

    def __getattr__(self, name):
        return getattr(self._c, name)

    def demux(self, *args, **kwargs):
        def marker(pts):
            pkt = av.Packet(0)
            pkt.pts = pts
            pkt.dts = pts
            return pkt

        emitted_head = False
        last_pts = None
        for pkt in self._c.demux(*args, **kwargs):
            if pkt.size > 0 and pkt.pts is not None:
                if self._position == "head" and not emitted_head:
                    emitted_head = True
                    yield marker(pkt.pts)
                last_pts = pkt.pts
            elif (pkt.size == 0 and pkt.pts is None
                    and self._position == "tail" and last_pts is not None):
                yield marker(last_pts + self._delta)   # before the sentinel
            yield pkt


def _patched_open(monkeypatch, position: str):
    real_open = fr._open_video

    def open_video(path):
        container, vs = real_open(path)
        c = av.open(str(path))
        s = [x for x in c.streams if x.type == "video"][0]
        pts = [p.pts for p in c.demux(s) if p.size > 0][:2]
        c.close()
        delta = int(pts[1] - pts[0])
        return _InjectingContainer(container, position, delta), vs

    monkeypatch.setattr(fr, "_open_video", open_video)


def test_zero_byte_marker_rematerializes_previous_frame(
        clip, monkeypatch, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="kinovsr.media.ffmpeg_reader")
    _patched_open(monkeypatch, "tail")
    frames = [pb for chunk in fr.iter_video_buffer_chunks(
        clip, PIX_BGRA, chunk_size=7) for pb in chunk]
    assert len(frames) == N + 1
    dup = read_buffer_rgb_f32(frames[N])
    prev = read_buffer_rgb_f32(frames[N - 1])
    assert float(mx.max(mx.abs(dup - prev))) == 0.0
    text = caplog.text
    assert "re-materializing each as a duplicate" in text
    assert "1 duplicate frame(s) re-materialized" in text


def test_zero_byte_marker_without_predecessor_is_dropped(
        clip, monkeypatch, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="kinovsr.media.ffmpeg_reader")
    _patched_open(monkeypatch, "head")
    frames = [pb for chunk in fr.iter_video_buffer_chunks(
        clip, PIX_BGRA, chunk_size=7) for pb in chunk]
    assert len(frames) == N                       # nothing extra, no crash
    assert "no usable predecessor" in caplog.text
    assert "0 duplicate frame(s) re-materialized" in caplog.text
    assert "1 dropped" in caplog.text


def test_sample_table_records_zero_byte_slots(clip, monkeypatch):
    _patched_open(monkeypatch, "tail")
    table = fr.read_sample_table(clip)
    assert len(table.samples) == N + 1
    assert table.samples[-1].coded_size == 0
