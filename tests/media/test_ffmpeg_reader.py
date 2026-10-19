"""Tests for the ffmpeg (PyAV) compatibility reader.

Self-contained: the fixture clip is muxed by PyAV itself (mpeg4 in MP4, fixed
GOP, tagged 601), so no binary fixtures and no dependency on files AVFoundation
can or cannot read.
"""

import itertools
import math
import subprocess
import sys
from array import array
from fractions import Fraction
from types import SimpleNamespace

import av
import mlx.core as mx
import pytest
import Quartz

from kinovsr.media import ffmpeg_reader as fr
from kinovsr.media import video_reader
from kinovsr.media.pixel_buffers import (
    PIX_BGRA,
    PIX_RGBAHALF,
    VIDEO_TIME_SCALE,
    make_pixel_buffer_from_attrs,
    read_buffer_rgb_f32,
)
from kinovsr.native.writer import HEVC_PROFILE_MAIN422_10, AVWriter

W, H, N, FPS, GOP = 160, 128, 24, 25, 8


def test_explicit_timing_origin_preserves_nonzero_stream_start_indexing():
    stream = SimpleNamespace(
        start_time=10000,
        time_base=Fraction(1, 1000),
    )
    assert fr._pts_index(10000, stream, 25.0) == 0
    assert fr._pts_index(10000, stream, 25.0, origin=Fraction(10)) == 0
    assert fr._pts_index(10040, stream, 25.0, origin=Fraction(10)) == 1


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_direct_readers_reject_invalid_chunks_before_open(chunk_size):
    from pathlib import Path

    missing = Path("must-not-open.mp4")
    with pytest.raises(ValueError, match="positive integer"):
        list(fr.iter_video_buffer_chunks(missing, PIX_BGRA, chunk_size=chunk_size))
    with pytest.raises(ValueError, match="positive integer"):
        list(video_reader.iter_video_buffer_chunks(missing, PIX_BGRA, chunk_size=chunk_size))


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
    assert par is None  # square pixels
    assert transform is not None  # identity CGAffineTransform


# Display matrices (a, b, c, d, tx, ty) as AVFoundation writes them: each quarter
# turn carries the translation that keeps the picture in the positive quadrant.
_ORIENTATIONS = {
    "rotated 90": (0.0, 1.0, -1.0, 0.0, float(H), 0.0),
    "rotated 180": (-1.0, 0.0, 0.0, -1.0, float(W), float(H)),
    "rotated 270": (0.0, -1.0, 1.0, 0.0, 0.0, float(W)),
    "mirrored": (-1.0, 0.0, 0.0, 1.0, float(W), 0.0),
}


@pytest.mark.integration
@pytest.mark.parametrize("case", sorted(_ORIENTATIONS))
def test_probe_carries_rotation_and_mirroring_like_the_native_probe(tmp_path, case):
    # PyAV 19 iterates a frame's side data by type key, which the display
    # matrix read did not expect: every rotated or mirrored file probed as
    # upright, and its output lost the orientation.
    path = tmp_path / "oriented.mp4"
    writer = AVWriter(
        path,
        width=W,
        height=H,
        fps=FPS,
        source_pixel_format=PIX_RGBAHALF,
        profile=HEVC_PROFILE_MAIN422_10,
        label="probe",
        transform=Quartz.CGAffineTransformMake(*_ORIENTATIONS[case]),
    )
    frame = make_pixel_buffer_from_attrs(
        W, H, {"PixelFormatType": PIX_RGBAHALF, "IOSurfaceProperties": {}}
    )
    ticks = VIDEO_TIME_SCALE // FPS
    for index in range(3):
        writer.append(frame, pts_ticks=ticks * index, duration_ticks=ticks)
    writer.finish()

    def parts(transform):
        return (transform.a, transform.b, transform.c, transform.d, transform.tx, transform.ty)

    native = parts(video_reader.probe_video(path)[4])
    assert native == _ORIENTATIONS[case]
    assert parts(fr.probe_video(path)[4]) == native


def test_keyframes_match_decoded_positions(clip):
    """The metadata keyframe list must equal the key positions in the reader's
    own decode order (self-consistency is the contract gop-align relies on)."""
    kf = fr.keyframe_display_indices(clip)
    assert kf[0] == 0
    assert kf == sorted(kf)
    c = av.open(str(clip))
    vs = next(s for s in c.streams if s.type == "video")
    decoded = [i for i, f in enumerate(c.decode(vs)) if f.key_frame]
    c.close()
    assert kf == decoded
    assert all(b - a == GOP for a, b in itertools.pairwise(kf))  # fixed GOP respected


def test_frames_stream_and_trim(clip):
    # full stream arrives complete and in order via the RGBAHalf path
    chunks = list(fr.iter_video_buffer_chunks(clip, PIX_RGBAHALF, chunk_size=5))
    assert max(map(len, chunks)) == 5
    frames = [pb for chunk in chunks for pb in chunk]
    assert len(frames) == N
    f0 = read_buffer_rgb_f32(frames[0])
    assert f0.shape == (H, W, 3)
    assert bool(mx.all(mx.isfinite(f0)))
    # frame-exact trim: [10, 14) yields 4 frames matching the full stream's
    trimmed = [
        pb
        for chunk in fr.iter_video_buffer_chunks(
            clip, PIX_RGBAHALF, chunk_size=5, start_frame=10, end_frame=14
        )
        for pb in chunk
    ]
    assert len(trimmed) == 4
    a = read_buffer_rgb_f32(frames[10])
    b = read_buffer_rgb_f32(trimmed[0])
    assert float(mx.max(mx.abs(a - b))) < 1e-6


def test_bgra_path_matches_rgbahalf(clip):
    ch = next(iter(fr.iter_video_buffer_chunks(clip, PIX_BGRA, chunk_size=1)))
    hh = next(iter(fr.iter_video_buffer_chunks(clip, PIX_RGBAHALF, chunk_size=1)))
    a = read_buffer_rgb_f32(ch[0])
    b = read_buffer_rgb_f32(hh[0])
    assert float(mx.max(mx.abs(a - b))) < (1.5 / 255.0)  # 8-bit rounding only


def test_unsupported_format_raises(clip):
    from kinovsr.media.pixel_buffers import PIX_NV12

    with pytest.raises(ValueError, match="supports RGBAHalf and BGRA outputs"):
        next(iter(fr.iter_video_buffer_chunks(clip, PIX_NV12)))


def test_probe_color_shape(clip):
    c = fr.probe_color(clip)
    assert set(c) == {"primaries", "transfer", "matrix", "full_range", "tagged"}
    assert isinstance(c["full_range"], bool)
    assert isinstance(c["tagged"], bool)


def write_tagged_clip(path, primaries: int, trc: int, colorspace: int) -> None:
    """Mux a two-frame mpeg4 clip carrying the given libav color tags."""
    out = av.open(str(path), "w")
    vs = out.add_stream("mpeg4", rate=FPS)
    vs.width, vs.height = W, H
    vs.pix_fmt = "yuv420p"
    context = vs.codec_context
    context.color_primaries, context.color_trc, context.colorspace = primaries, trc, colorspace
    for _ in range(2):
        for pkt in vs.encode(av.VideoFrame(W, H, "gray").reformat(format="yuv420p")):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()


# One field varied from a BT.709 base: (primaries, trc, matrix) per case.
_CODE_CASES = {
    "bt470bg primaries": (5, 1, 1),
    "smpte240m primaries": (7, 1, 1),
    "dci-p3 primaries": (11, 1, 1),
    "display-p3 primaries": (12, 1, 1),
    "p22 primaries": (22, 1, 1),
    "smpte240m transfer": (1, 7, 1),
    "linear transfer": (1, 8, 1),
    "pq transfer": (1, 16, 1),
    "smpte428 transfer": (1, 17, 1),
    "hlg transfer": (1, 18, 1),
    "bt470bg matrix": (1, 1, 5),
    "smpte240m matrix": (1, 1, 7),
    "bt2020c matrix": (1, 1, 10),
}


@pytest.mark.parametrize("case", sorted(_CODE_CASES))
def test_both_readers_resolve_each_color_code_alike(tmp_path, case):
    # The ffmpeg reader mapped fewer H.273 codes than CoreMedia names, so a PQ,
    # HLG, P3 or PAL-primaries source got other output tags through it.
    from kinovsr.media import color

    path = tmp_path / "tagged.mp4"
    write_tagged_clip(path, *_CODE_CASES[case])
    ffmpeg_side = color.resolve(fr.probe_color(path))
    native_side = color.resolve(video_reader.probe_color(path))
    assert tuple(map(str, ffmpeg_side)) == tuple(map(str, native_side))


def test_srgb_transfer_reads_as_the_native_reader_reports_it(tmp_path):
    # color_trc 13 used to read as "sRGB", which matches no CoreVideo or
    # AVFoundation token, so the output was tagged BT.709.
    path = tmp_path / "srgb.mp4"
    write_tagged_clip(path, primaries=1, trc=13, colorspace=1)
    expected = Quartz.kCVImageBufferTransferFunction_sRGB
    assert fr.probe_color(path)["transfer"] == expected
    assert video_reader.probe_color(path)["transfer"] == expected


def test_forced_color_range_differs(clip):
    """Forcing full-range read must change the pixels of a video-range clip."""
    a = read_buffer_rgb_f32(
        next(
            iter(
                fr.iter_forced_color_chunks(clip, PIX_RGBAHALF, "ITU_R_601_4", False, chunk_size=1)
            )
        )[0]
    )
    b = read_buffer_rgb_f32(
        next(
            iter(fr.iter_forced_color_chunks(clip, PIX_RGBAHALF, "ITU_R_601_4", True, chunk_size=1))
        )[0]
    )
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

    sine = array.array("f", (0.25 * math.sin(2 * math.pi * 440 * i / 48000) for i in range(n_samp)))
    # the AAC encoder takes fixed 1024-sample PLANAR float frames
    step = 1024
    for off in range(0, n_samp - step + 1, step):
        af = av.AudioFrame(format="fltp", layout="mono", samples=step)
        af.sample_rate = 48000
        af.pts = off
        af.planes[0].update(sine[off : off + step].tobytes())
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
    assert at.sample_rate == 48000
    assert at.channels == 1
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
        resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=stream.rate)
        decoded = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                decoded.append(
                    mx.array(memoryview(resampled.planes[0]).cast("B")).view(mx.float32)[
                        : resampled.samples
                    ]
                )
    expected = mx.concatenate(decoded)[12_000:24_000]
    actual = mx.array(memoryview(window_raw).cast("f"))
    assert mx.max(mx.abs(actual - expected)).item() < 1e-5

    late = fr.read_audio_track_window(
        path,
        start_sec=Fraction(2),
        end_sec=Fraction(9, 4),
    )
    assert late is not None
    assert late.n_samples == 12_000
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
        assert track is not None
        assert track._source is None
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
            tuple(int(value) for value in completed.stdout.splitlines()[-1].split())
        )

    assert [item[0] for item in process_results] == [expected_bytes, expected_bytes]
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
    with pytest.raises(OSError, match=r"absent\.mp4"):
        fr.probe_video(tmp_path / "absent.mp4")


def test_non_utf8_metadata_tag_does_not_reject_the_file(tmp_path):
    # Old QuickTime writers store tag text as raw Mac Roman, so a copyright
    # sign arrives as the lone byte 0xa9, which is not valid UTF-8.
    marker = b"KINOVSR-TAG"
    source = tmp_path / "tagged.mp4"
    out = av.open(str(source), "w")
    out.metadata["comment"] = marker.decode()
    vs = out.add_stream("mpeg4", rate=FPS)
    vs.width, vs.height = W, H
    vs.pix_fmt = "yuv420p"
    for _ in range(3):
        for pkt in vs.encode(av.VideoFrame(W, H, "yuv420p")):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()
    data = source.read_bytes()
    assert data.count(marker) == 1
    path = tmp_path / "mac-roman-tag.mp4"
    path.write_bytes(data.replace(marker, b"\xa9" + marker[1:]))

    w, h, _fps, n, _transform, _par = fr.probe_video(path)
    assert (w, h, n) == (W, H, 3)
    frames = [b for chunk in fr.iter_video_buffer_chunks(path, PIX_BGRA) for b in chunk]
    assert len(frames) == 3


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
        self._position = position  # "tail" | "head"
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
            elif (
                pkt.size == 0
                and pkt.pts is None
                and self._position == "tail"
                and last_pts is not None
            ):
                yield marker(last_pts + self._delta)  # before the sentinel
            yield pkt


def _patched_open(monkeypatch, position: str):
    real_open = fr._open_video

    def open_video(path):
        container, vs = real_open(path)
        c = av.open(str(path))
        s = next(x for x in c.streams if x.type == "video")
        pts = [p.pts for p in c.demux(s) if p.size > 0][:2]
        c.close()
        delta = int(pts[1] - pts[0])
        return _InjectingContainer(container, position, delta), vs

    monkeypatch.setattr(fr, "_open_video", open_video)


def test_zero_byte_marker_rematerializes_previous_frame(clip, monkeypatch, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="kinovsr.media.ffmpeg_reader")
    _patched_open(monkeypatch, "tail")
    frames = [
        pb for chunk in fr.iter_video_buffer_chunks(clip, PIX_BGRA, chunk_size=7) for pb in chunk
    ]
    assert len(frames) == N + 1
    dup = read_buffer_rgb_f32(frames[N])
    prev = read_buffer_rgb_f32(frames[N - 1])
    assert float(mx.max(mx.abs(dup - prev))) == 0.0
    text = caplog.text
    assert "re-materializing each as a duplicate" in text
    assert "1 duplicate frame(s) re-materialized" in text


def test_zero_byte_marker_without_predecessor_is_dropped(clip, monkeypatch, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="kinovsr.media.ffmpeg_reader")
    _patched_open(monkeypatch, "head")
    frames = [
        pb for chunk in fr.iter_video_buffer_chunks(clip, PIX_BGRA, chunk_size=7) for pb in chunk
    ]
    assert len(frames) == N  # nothing extra, no crash
    assert "no usable predecessor" in caplog.text
    assert "0 duplicate frame(s) re-materialized" in caplog.text
    assert "1 dropped" in caplog.text


def test_sample_table_records_zero_byte_slots(clip, monkeypatch):
    _patched_open(monkeypatch, "tail")
    table = fr.read_sample_table(clip)
    assert len(table.samples) == N + 1
    assert table.samples[-1].coded_size == 0


def test_a_stream_without_a_time_base_is_an_operational_error():
    # PyAV reports an unset time base as the falsy AVRational(0, 1); a guard
    # that tested for None let it through as a zero Fraction.
    stream = SimpleNamespace(time_base=av.AVRational(0, 1), type="video", index=0)
    with pytest.raises(RuntimeError, match="video stream #0 has no time base"):
        fr._time_base(stream)
    stamped = SimpleNamespace(time_base=av.AVRational(1, 90000))
    assert fr._time_base(stamped) == Fraction(1, 90000)


# ------------------------------------------------- packed-bitstream AVI

_VOP = b"\x00\x00\x01\xb6"
# DivX 5.01's user data; the trailing "p" declares a packed bitstream.
_DIVX_PACKED = b"\x00\x00\x01\xb2DivX501b484p"
_PACKED_FRAMES = 30
# Slot s of the packed AVI below shows this frame.
_PACKED_SLOTS = [0, *range(_PACKED_FRAMES), _PACKED_FRAMES - 1]
_X264_B_PYRAMID = {"bf": "2", "g": "12", "crf": "10", "x264-params": "b-pyramid=normal:scenecut=0"}


def _n_vop(increment: int) -> bytes:
    """A not-coded P-VOP (5-bit time increment at 25 fps) with DivX's stuffing byte."""
    bits = value = 0
    for field, width in ((0b01, 2), (0, 1), (1, 1), (increment, 5), (1, 1), (0, 1)):
        value, bits = (value << width) | field, bits + width
    pad = 8 - bits % 8
    value, bits = (value << pad) | ((1 << (pad - 1)) - 1), bits + pad
    return _VOP + value.to_bytes(bits // 8, "big") + b"\x7f"


def _write_gray(path, codec, options):
    """Encode ``_PACKED_FRAMES`` frames, frame t flat gray 8t + 4."""
    out = av.open(str(path), "w")
    vs = out.add_stream(codec, rate=FPS)
    vs.width, vs.height = W, H
    vs.pix_fmt = "yuv420p"
    vs.options = options
    for t in range(_PACKED_FRAMES):
        frame = av.VideoFrame(W, H, "gray")
        frame.planes[0].update(bytes([8 * t + 4]) * (W * H))
        for pkt in vs.encode(frame.reformat(format="yuv420p")):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()


@pytest.fixture(scope="module")
def packed_avi(tmp_path_factory):
    """A DivX-style packed-bitstream AVI whose frame t is flat gray 8t + 4.

    FFmpeg writes B frames unpacked, so its encode (one B frame, closed GOPs)
    is repacked as DivX 5 writes it: the I frame alone, a one-byte delay
    frame, each reference picture with the B frame that follows it in decode
    order in one packet and an N-VOP placeholder in the B frame's slot, and
    the packed-bitstream user data in the extradata and the first packet. A
    one-byte drop frame ends the file. In VfW terms slot 0 shows frame 0, the
    delay slot repeats it, slot s shows frame s - 1, and the last slot
    repeats the frame before it (``_PACKED_SLOTS``).
    """
    folder = tmp_path_factory.mktemp("packed")
    source = folder / "unpacked.avi"
    # Closed GOPs need scene-cut detection off in FFmpeg's encoder.
    options = {
        "g": str(GOP),
        "bf": "1",
        "qscale": "2",
        "flags": "+cgop",
        "sc_threshold": "1000000000",
    }
    _write_gray(source, "mpeg4", options)
    c = av.open(str(source))
    s = next(x for x in c.streams if x.type == "video")
    coded = [(bytes(p), p.is_keyframe) for p in c.demux(s) if p.size]
    path = folder / "packed.avi"
    out = av.open(str(path), "w", format="avi")
    target = out.add_stream_from_template(s)
    first = coded[0][0]
    headers = first[: first.find(_VOP)] + _DIVX_PACKED
    target.codec_context.extradata = headers
    slots: list[tuple[bytes, bool]] = []
    i = 0
    while i < len(coded):
        data, key = coded[i]
        j = i + 1
        while j < len(coded) and coded[j][0][coded[j][0].find(_VOP) + 4] >> 6 == 2:  # a B-VOP
            j += 1
        if i == 0:
            data = headers + data[data.find(_VOP) :]
        slots.append((data + b"".join(d for d, _ in coded[i + 1 : j]), key))
        if i == 0:
            slots.append((b"\x7f", False))
        slots.extend((_n_vop(len(slots) % FPS), False) for _ in range(j - i - 1))
        i = j
    slots.append((b"\x7f", False))
    for index, (data, key) in enumerate(slots):
        pkt = av.Packet(data)
        pkt.pts = pkt.dts = index
        pkt.time_base = target.time_base
        pkt.is_keyframe = key
        pkt.stream = target
        out.mux(pkt)
    out.close()
    c.close()
    assert len(slots) == len(_PACKED_SLOTS)
    return path


def _gray_frames(path, start=0, end=None):
    """Each delivered (label, frame), the frame read back from its gray level."""
    table = fr.read_sample_table(path)
    read = []
    for chunk in fr.iter_video_buffer_chunks(
        path, PIX_RGBAHALF, chunk_size=3, start_frame=start, end_frame=end, table=table
    ):
        assert len(chunk) <= 3
        for buffer, label in chunk:
            level = float(read_buffer_rgb_f32(buffer)[0, 0, 0]) * 255
            read.append((label, round((level - 4) / 8)))
    return read


def test_packed_avi_reads_every_slot_in_display_order(packed_avi, caplog):
    import logging

    # libavcodec stamps each packed pair's pictures with each other's slots
    # and leaves the drop frames without a picture.
    caplog.set_level(logging.INFO, logger="kinovsr.media.ffmpeg_reader")
    assert fr.read_sample_table(packed_avi).decode_order_stamps
    assert _gray_frames(packed_avi) == list(enumerate(_PACKED_SLOTS))
    assert "2 slot(s) without a picture" in caplog.text


@pytest.mark.parametrize("start", [1, 2, 5, 8, 9, 16, 28])
def test_packed_avi_window_after_a_seek(packed_avi, start):
    end = min(start + 5, len(_PACKED_SLOTS))
    expected = [(slot, _PACKED_SLOTS[slot]) for slot in range(start, end)]
    assert _gray_frames(packed_avi, start, end) == expected


def test_unpacked_mpeg4_avi_reads_as_before(packed_avi):
    # FFmpeg's AVI demuxer stamps unpacked B frames with display times.
    unpacked = packed_avi.parent / "unpacked.avi"
    assert not fr.read_sample_table(unpacked).decode_order_stamps
    assert _gray_frames(unpacked) == [(t, t) for t in range(_PACKED_FRAMES)]


def test_h264_avi_reads_in_display_order(tmp_path):
    # AVI keeps no display times and FFmpeg cannot derive H.264's, so its
    # B-frame pyramids arrive with permuted stamps. With open GOPs, the B
    # frames shown before each keyframe cannot decode after a seek to it,
    # so a window starting there must be read from further back.
    if "libx264" not in av.codecs_available:
        pytest.skip("this PyAV build has no libx264")
    path = tmp_path / "h264.avi"
    options = {**_X264_B_PYRAMID, "x264-params": "b-pyramid=normal:open-gop=1:scenecut=0"}
    _write_gray(path, "libx264", options)
    assert fr.read_sample_table(path).decode_order_stamps
    assert _gray_frames(path) == [(t, t) for t in range(_PACKED_FRAMES)]
    for start in range(1, _PACKED_FRAMES - 3):
        assert _gray_frames(path, start, start + 3) == [(t, t) for t in range(start, start + 3)]


def test_flv_with_zeroed_composition_offsets_reads_in_display_order(tmp_path):
    # FLV carries H.264's composition offsets; a muxer that zeroes them
    # leaves decode-order stamps, which permute like AVI's.
    if "libx264" not in av.codecs_available:
        pytest.skip("this PyAV build has no libx264")
    source, zeroed = tmp_path / "h264.flv", tmp_path / "zeroed.flv"
    _write_gray(source, "libx264", _X264_B_PYRAMID)
    c = av.open(str(source))
    s = next(x for x in c.streams if x.type == "video")
    out = av.open(str(zeroed), "w", format="flv")
    target = out.add_stream_from_template(s)
    for pkt in c.demux(s):
        if pkt.size:
            pkt.pts = pkt.dts
            pkt.stream = target
            out.mux(pkt)
    out.close()
    c.close()
    assert not fr.read_sample_table(source).decode_order_stamps
    assert fr.read_sample_table(zeroed).decode_order_stamps
    for path in (source, zeroed):
        assert _gray_frames(path) == [(t, t) for t in range(_PACKED_FRAMES)]


# ---------------------------------------------------------- coded pictures


def _decoded_kinds(path):
    """Each picture's type in the decoder's output order, which is display order."""
    c = av.open(str(path))
    s = next(x for x in c.streams if x.type == "video")
    kinds = [fr._PICTURE_KINDS.get(int(f.pict_type), "?") for f in c.decode(s)]
    c.close()
    return kinds


def _pictures(path, start=0, end=None):
    return list(fr.iter_coded_pictures(path, start_frame=start, end_frame=end))


def test_coded_pictures_follow_the_frames(clip):
    pictures = _pictures(clip)
    assert [index for index, _ in pictures] == list(range(N))
    kinds = [picture.kind for _, picture in pictures]
    assert kinds == _decoded_kinds(clip)
    assert [i for i, kind in enumerate(kinds) if kind == "I"] == fr.keyframe_display_indices(clip)
    assert all(picture.reference for _, picture in pictures)  # the clip has no B frames
    assert _pictures(clip, 10, 14) == pictures[10:14]


def test_packed_avi_slots_report_the_picture_they_show(packed_avi):
    # Slot s shows frame _PACKED_SLOTS[s] of the encode the file was repacked from; the slot
    # of a drop frame reports the picture it repeats.
    shown = _decoded_kinds(packed_avi.parent / "unpacked.avi")
    assert {"I", "P", "B"} <= set(shown)
    pictures = _pictures(packed_avi)
    assert [index for index, _ in pictures] == list(range(len(_PACKED_SLOTS)))
    assert [picture.kind for _, picture in pictures] == [shown[f] for f in _PACKED_SLOTS]
    for start in (1, 2, 5, 8, 9, 16, 28):
        end = min(start + 5, len(_PACKED_SLOTS))
        assert _pictures(packed_avi, start, end) == pictures[start:end]


def test_h264_avi_coded_pictures_follow_display_order(tmp_path):
    if "libx264" not in av.codecs_available:
        pytest.skip("this PyAV build has no libx264")
    path = tmp_path / "h264.avi"
    options = {**_X264_B_PYRAMID, "x264-params": "b-pyramid=normal:open-gop=1:scenecut=0"}
    _write_gray(path, "libx264", options)
    pictures = _pictures(path)
    kinds = [picture.kind for _, picture in pictures]
    assert kinds == _decoded_kinds(path)
    assert {"B", "P"} <= set(kinds)
    for start in range(1, _PACKED_FRAMES - 3):
        assert _pictures(path, start, start + 3) == pictures[start : start + 3]


def test_coded_pictures_carry_their_quantizers(packed_avi):
    pictures = [picture for _, picture in _pictures(packed_avi.parent / "unpacked.avi")]
    # libavcodec exports no parameters for a picture it drains at the end of the stream.
    missing = [i for i, picture in enumerate(pictures) if picture.quantizer is None]
    assert missing in ([], [len(pictures) - 1])
    exported = [p for p in pictures if p.quantizer is not None]
    assert all(p.quantizer > 0 for p in exported)

    def mean(kind):
        values = [p.quantizer for p in exported if p.kind == kind]
        return sum(values) / len(values)

    # FFmpeg's MPEG-4 encoder quantizes B pictures more coarsely than P pictures.
    assert mean("B") > mean("P")


def test_a_repeat_marker_reports_the_picture_it_repeats(clip, monkeypatch):
    _patched_open(monkeypatch, "tail")
    pictures = _pictures(clip)
    assert [index for index, _ in pictures] == list(range(N + 1))
    assert pictures[N][1] == pictures[N - 1][1]
