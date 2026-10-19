"""Trimmed reads deliver exactly the frames a full read does, in every container.

Every synthetic frame carries its own display index as 16 black or white bands,
one 8-pixel block column each, so the index survives lossy coding. A delivered
frame is right only when its label equals the index in its pixels; no reader
timestamp is trusted.
"""

import logging
import math
from array import array
from fractions import Fraction

import av
import CoreMedia
import mlx.core as mx
import pytest

from kinovsr.media import ffmpeg_reader as fr
from kinovsr.media import video_reader
from kinovsr.media.pixel_buffers import PIX_BGRA, read_buffer_rgb_f32
from kinovsr.media.reader import auto_reader

W, H, BITS, N, FPS = 128, 64, 16, 96, 25
MPEG = {"g": "12", "bf": "2", "qscale": "2"}
X264 = {"x264-params": "scenecut=0:bframes=2:b-adapt=0:keyint=12:min-keyint=12", "crf": "10"}

CASES = {
    # name: (container format, video codec, options, (audio codec, rate, layout) or None)
    "mpeg-ps": ("mpeg", "mpeg1video", MPEG, None),
    "mpeg-ps-mp2": ("mpeg", "mpeg1video", MPEG, ("mp2", 48000, "stereo")),
    "mpeg-ts": ("mpegts", "mpeg2video", MPEG, None),
    "m2v": ("mpeg2video", "mpeg2video", MPEG, None),
    "h264-es": ("h264", "libx264", X264, None),
    "rm": ("rm", "rv20", {"g": "12", "qscale": "2"}, None),
    "mkv": ("matroska", "mpeg4", MPEG, None),
    "avi": ("avi", "mpeg4", MPEG, None),
    "mp4-aac": ("mp4", "mpeg4", MPEG, ("aac", 48000, "stereo")),
    "rm-ac3": ("rm", "rv10", {"g": "12", "qscale": "2"}, ("ac3", 48000, "stereo")),
}


def _pixels(t: int) -> bytes:
    row = bytearray(W)
    for b in range(BITS):
        row[b * 8 : b * 8 + 8] = bytes([235 if (t >> b) & 1 else 20]) * 8
    return bytes(row) * H


def _write(path, fmt, codec, options, audio):
    out = av.open(str(path), "w", format=fmt)
    vs = out.add_stream(codec, rate=FPS)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    vs.options = dict(options)
    ast = fifo = None
    if audio is not None:
        codec_name, rate, layout = audio
        ast = out.add_stream(codec_name, rate=rate, layout=layout)
        fifo = av.AudioFifo()
    written = 0
    for t in range(N):
        frame = av.VideoFrame(W, H, "gray")
        frame.planes[0].update(_pixels(t))
        for packet in vs.encode(frame.reformat(format="yuv420p")):
            out.mux(packet)
        if ast is not None:
            upto = (t + 1) * rate // FPS
            # A rising sweep never repeats, so a misplaced window cannot match.
            sweep = [
                0.3 * math.sin(math.pi * i * (440 + i / 1200) / rate) for i in range(written, upto)
            ]
            channels = len(ast.layout.channels)
            tone = av.AudioFrame(format="flt", layout=layout, samples=upto - written)
            tone.planes[0].update(array("f", (v for v in sweep for _ in range(channels))).tobytes())
            tone.rate, tone.pts, written = rate, written, upto
            fifo.write(tone)
            while fifo.samples >= ast.codec_context.frame_size:
                for packet in ast.encode(fifo.read(ast.codec_context.frame_size)):
                    out.mux(packet)
    for packet in vs.encode():
        out.mux(packet)
    if ast is not None:
        tail = fifo.read()  # None when the frames divided evenly
        for packet in ast.encode(tail) if tail is not None else ():
            out.mux(packet)
        for packet in ast.encode():
            out.mux(packet)
    out.close()


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    folder = tmp_path_factory.mktemp("trimmed")
    made = {}
    for name, (fmt, codec, options, audio) in CASES.items():
        if codec not in av.codecs_available:
            continue
        made[name] = folder / f"{name}.{'m2v' if fmt == 'mpeg2video' else fmt}"
        _write(made[name], fmt, codec, options, audio)
    return made


def _clip(clips, name):
    if name not in clips:
        pytest.skip(f"{CASES[name][1]} encoder unavailable")
    return clips[name]


def _index(buffer) -> int:
    row = read_buffer_rgb_f32(buffer)[H // 2]
    columns = mx.array([b * 8 + 4 for b in range(BITS)])
    bits = (mx.mean(row[columns], axis=-1) > 0.5).tolist()
    return sum(1 << b for b, on in enumerate(bits) if on)


def _read(reader, path, table, start=0, end=None, limit=None):
    """(label, index in the pixels) of each frame the reader delivers."""
    got = []
    frames = reader.iter_video_buffer_chunks(
        path, PIX_BGRA, 4, start_frame=start, end_frame=end, table=table
    )
    try:
        for chunk in frames:
            for buffer, label in chunk:
                got.append((label, _index(buffer)))
                if len(got) == limit:
                    return got
    finally:
        frames.close()
    return got


@pytest.mark.parametrize("name", list(CASES))
def test_full_read_labels_every_frame_by_its_content(clips, name):
    path = _clip(clips, name)
    table = fr.read_sample_table(path)
    assert table.sample_count == N
    assert _read(fr, path, table) == [(i, i) for i in range(N)]


@pytest.mark.parametrize("name", list(CASES))
def test_every_trimmed_start_delivers_its_own_frame(clips, name):
    path = _clip(clips, name)
    table = fr.read_sample_table(path)
    wrong = {}
    for start in range(1, N - 1):
        got = _read(fr, path, table, start=start, limit=2)
        if got != [(start, start), (start + 1, start + 1)]:
            wrong[start] = got
    assert not wrong


@pytest.mark.parametrize("name", list(CASES))
def test_trimmed_windows_end_exactly(clips, name):
    path = _clip(clips, name)
    table = fr.read_sample_table(path)
    for start, end in [(5, 17), (40, 41), (N - 7, N)]:
        assert _read(fr, path, table, start=start, end=end) == [(i, i) for i in range(start, end)]


def test_raw_streams_number_pictures_in_bitstream_order(clips):
    table = fr.read_sample_table(_clip(clips, "m2v"))
    assert table.positional_stamps
    assert table.cadence == FPS
    assert not fr.read_sample_table(_clip(clips, "mpeg-ps")).positional_stamps


def test_auto_reader_identifies_program_streams_by_their_content(clips, tmp_path):
    disguised = {"looks-like.mp4": "mpeg-ps", "looks-like.mpg": "mkv", "looks-like.vob": "mpeg-ts"}
    for alias, name in disguised.items():
        (tmp_path / alias).write_bytes(_clip(clips, name).read_bytes())
    assert auto_reader(tmp_path / "looks-like.mp4") is fr
    assert auto_reader(tmp_path / "looks-like.mpg") is video_reader
    assert auto_reader(tmp_path / "looks-like.vob") is video_reader
    assert auto_reader(tmp_path / "absent.mpg") is video_reader


def _rewind_one_pes_stamp(data: bytes) -> bytes:
    """Stamp one mid-file video PES two frames early, as a faulty muxer can.

    The pictures it stamps then share their stamps with the two before them.
    """
    ticks = 2 * 90000 // FPS
    headers = []
    at = 0
    while (at := data.find(b"\x00\x00\x01\xe0", at + 1)) >= 0:
        field = at + 6
        while data[field] == 0xFF:  # MPEG-1 stuffing
            field += 1
        if data[field] & 0xC0 == 0x40:  # STD buffer size
            field += 2
        if data[field] >> 4 in (2, 3):
            headers.append(field)
    patched = bytearray(data)
    field = headers[len(headers) // 2]
    for offset in range(0, 10 if data[field] >> 4 == 3 else 5, 5):
        b = patched[field + offset : field + offset + 5]
        stamp = ((b[0] >> 1) & 7) << 30 | b[1] << 22 | (b[2] >> 1) << 15 | b[3] << 7 | b[4] >> 1
        stamp -= ticks
        patched[field + offset : field + offset + 5] = bytes(
            [
                (b[0] & 0xF0) | ((stamp >> 30) & 7) << 1 | 1,
                (stamp >> 22) & 0xFF,
                ((stamp >> 15) & 0x7F) << 1 | 1,
                (stamp >> 7) & 0xFF,
                (stamp & 0x7F) << 1 | 1,
            ]
        )
    return bytes(patched)


def test_contradictory_pes_stamps_number_pictures_in_bitstream_order(clips, tmp_path, caplog):
    path = tmp_path / "rewound.mpg"
    path.write_bytes(_rewind_one_pes_stamp(_clip(clips, "mpeg-ps").read_bytes()))
    with caplog.at_level(logging.WARNING, logger="kinovsr.media.ffmpeg_reader"):
        table = fr.read_sample_table(path)
    assert table.positional_stamps
    assert "share a presentation timestamp" in caplog.text
    assert table.cadence == FPS
    assert _read(fr, path, table) == [(i, i) for i in range(N)]
    for start in range(1, N - 1, 5):
        assert _read(fr, path, table, start=start, limit=2) == [(start, start), (start + 1,) * 2]


class _LateSeek:
    """Forward a real container, but land every seek a second past its target."""

    def __init__(self, container):
        self._c = container

    def __getattr__(self, name):
        return getattr(self._c, name)

    def seek(self, offset, *args, stream=None, **kwargs):
        late = 1_000_000 if stream is None else int(1 / Fraction(stream.time_base))
        return self._c.seek(offset + late, *args, stream=stream, **kwargs)


def test_a_seek_that_lands_late_falls_back_to_walking_from_the_head(clips, monkeypatch, caplog):
    path = _clip(clips, "mkv")
    table = fr.read_sample_table(path)
    real_open = fr._open_video
    monkeypatch.setattr(
        fr, "_open_video", lambda p: (lambda c, s: (_LateSeek(c), s))(*real_open(p))
    )
    with caplog.at_level(logging.INFO, logger="kinovsr.media.ffmpeg_reader"):
        assert _read(fr, path, table, start=50, limit=3) == [(50, 50), (51, 51), (52, 52)]
    assert "walking the packets from the head instead" in caplog.text


def _audio_window_matches_full_decode(path):
    full = fr.read_audio_track(path)
    assert full is not None
    rate, channels = full.sample_rate, full.channels
    reference = array("f", full._read_interleaved(0, full.n_samples))
    full.close()
    for start in range(rate // 2, full.n_samples - rate // 4, rate // 3):
        window = fr.read_audio_track_window(path, start_sec=Fraction(start, rate))
        assert window is not None
        got = array("f", window._read_interleaved(0, rate // 5))
        offset = window._offset
        window.close()
        want = reference[offset * channels : offset * channels + len(got)]
        assert len(got) == len(want), start
        # A decoder started inside the stream settles within its preroll but
        # can keep a rounding difference (MP2's is about 3e-5); a misplaced or
        # silenced window differs by the signal's own size.
        assert max(abs(a - b) for a, b in zip(got, want, strict=True)) < 1e-4, start


def test_audio_windows_in_a_program_stream_match_the_full_decode(clips):
    _audio_window_matches_full_decode(_clip(clips, "mpeg-ps-mp2"))


def test_audio_windows_in_realmedia_match_the_full_decode(clips):
    _audio_window_matches_full_decode(_clip(clips, "rm-ac3"))


def test_a_late_audio_seek_falls_back_to_walking_from_the_head(clips, monkeypatch, caplog):
    path = _clip(clips, "mp4-aac")
    real_open = fr._open_container
    monkeypatch.setattr(fr, "_open_container", lambda p, op: _LateSeek(real_open(p, op)))
    with caplog.at_level(logging.INFO, logger="kinovsr.media.ffmpeg_reader"):
        _audio_window_matches_full_decode(path)
    assert "the audio seek for sample" in caplog.text


# ------------------------------------------------------------ native reader


def _native_table(path):
    try:
        return video_reader.read_sample_table(path)
    except RuntimeError as exc:
        pytest.skip(f"AVFoundation cannot read {path.name}: {exc}")


def test_native_trimmed_read_without_an_end_reaches_the_last_frame(clips):
    # AVFoundation's duration for an MPEG program stream ends before its last
    # frames start, so a time range bounded by it dropped them.
    path = _clip(clips, "mpeg-ps")
    table = _native_table(path)
    assert _read(video_reader, path, table, start=N - 6) == [(i, i) for i in range(N - 6, N)]


class _LateRangeCoreMedia:
    """Forward CoreMedia, but start every reader time range a second late."""

    def __getattr__(self, name):
        return getattr(CoreMedia, name)

    def CMTimeRangeMake(self, start, duration):
        late = CoreMedia.CMTimeAdd(start, CoreMedia.CMTimeMake(1, 1))
        return CoreMedia.CMTimeRangeMake(late, duration)


def test_native_range_that_starts_late_rereads_from_the_head(clips, monkeypatch, caplog):
    path = _clip(clips, "mpeg-ps")
    table = _native_table(path)
    monkeypatch.setattr(video_reader, "CoreMedia", _LateRangeCoreMedia())
    with caplog.at_level(logging.INFO, logger="kinovsr.media.video_reader"):
        assert _read(video_reader, path, table, start=40, limit=2) == [(40, 40), (41, 41)]
    assert "reading from the head instead" in caplog.text
