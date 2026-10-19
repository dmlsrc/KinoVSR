"""Bounded audio-window and streaming-cursor contracts."""

from array import array
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

pytestmark = pytest.mark.unit


def test_sample_window_rounds_relative_cap_independently():
    from kinovsr.media.audio import _sample_window

    # round(.06 * 10) == 1 and the relative cap is another one sample.
    # Rounding the combined absolute end (.12 * 10) would incorrectly select
    # an empty [1, 1) window under ties-to-even.
    assert _sample_window(
        sample_rate=10,
        total_samples=100,
        start_sec=Fraction(3, 50),
        end_sec=Fraction(1),
        max_duration_sec=Fraction(3, 50),
    ) == (1, 2)


def test_sample_window_keeps_broadcast_boundary_exact():
    from kinovsr.media.audio import _sample_window

    start, stop = _sample_window(
        sample_rate=44_100,
        total_samples=10_000_000,
        start_sec=Fraction(840 * 1001, 24_000),
        end_sec=Fraction(841 * 1001, 24_000),
        max_duration_sec=None,
    )

    assert start == 1_545_044
    assert stop == round(Fraction(841 * 1001 * 44_100, 24_000))


def test_in_memory_trim_is_a_zero_copy_sample_view():
    from kinovsr.media.audio import AudioTrack

    track = AudioTrack(mx.arange(20, dtype=mx.float32).reshape(2, 10), 10)
    view = track.trimmed(Fraction(1, 5), Fraction(1, 2))

    assert view.n_samples == 3
    assert view._parent is track
    assert "_bytes" not in view.__dict__
    bytes_per_frame = 4 * track.channels
    assert (
        bytes(view._read_interleaved(0, 3))
        == track._bytes[2 * bytes_per_frame : 5 * bytes_per_frame]
    )


def test_legacy_reader_is_rejected_before_unbounded_decode():
    from kinovsr.media.audio import AudioTrack, read_audio_track_from_video

    calls = []

    class LegacyReader:
        @staticmethod
        def read_audio_track(path):
            calls.append(path)
            return AudioTrack(mx.arange(100, dtype=mx.float32)[None], 10)

    with pytest.raises(RuntimeError, match="refusing unbounded"):
        read_audio_track_from_video(
            Path("legacy.mov"),
            LegacyReader,
            start_sec=Fraction(3, 50),
            end_sec=Fraction(1),
            max_duration_sec=Fraction(3, 50),
        )

    assert calls == []


def test_streaming_track_is_lazy_and_forks_independent_cursors():
    from kinovsr.media.audio import StreamingAudioTrack

    sources = []

    class Source:
        def __init__(self):
            self.reads = []
            self.closed = False
            sources.append(self)

        def read_frames(self, start, end):
            self.reads.append((start, end))
            return bytes((end - start) * 8)

        def close(self):
            self.closed = True

    track = StreamingAudioTrack(
        sample_rate=48_000,
        channels=2,
        n_samples=20,
        source_factory=Source,
        offset=100,
    )
    first, second = track.fork(), track.fork()

    assert sources == []
    assert len(first._read_interleaved(0, 4)) == 32
    assert len(second._read_interleaved(4, 7)) == 24
    assert [source.reads for source in sources] == [
        [(100, 104)],
        [(104, 107)],
    ]
    first.close()
    second.close()
    assert all(source.closed for source in sources)
    assert track._source is None


def test_native_audio_cursor_seeks_before_bounded_allocation(monkeypatch):
    from kinovsr.media.audio import _native_streaming_audio_track

    opened = []
    capacities = []

    class Format:
        @staticmethod
        def sampleRate():
            return 10.0

        @staticmethod
        def channelCount():
            return 1

    class Channel:
        def __init__(self, capacity):
            self.data = bytearray(capacity * 4)

        def as_buffer(self, _frames):
            return self.data

    class Buffer:
        def __init__(self, capacity):
            capacities.append(capacity)
            self.capacity = capacity
            self.frames = 0
            self.channel = Channel(capacity)

        def frameLength(self):
            return self.frames

        def floatChannelData(self):
            return [self.channel]

    class BufferBuilder:
        @staticmethod
        def initWithPCMFormat_frameCapacity_(_format, capacity):
            return Buffer(capacity)

    class BufferClass:
        @staticmethod
        def alloc():
            return BufferBuilder()

    class File:
        def __init__(self):
            self.positions = []
            opened.append(self)

        @staticmethod
        def processingFormat():
            return Format()

        @staticmethod
        def length():
            return 100

        def setFramePosition_(self, position):
            self.positions.append(position)

        @staticmethod
        def readIntoBuffer_frameCount_error_(buffer, count, _error):
            buffer.frames = count
            buffer.channel.data[:] = array("f", range(count)).tobytes()
            return True, None

    class FileBuilder:
        @staticmethod
        def initForReading_error_(_url, _error):
            return File(), None

    class FileClass:
        @staticmethod
        def alloc():
            return FileBuilder()

    fake_av = SimpleNamespace(
        AVAudioFile=FileClass,
        AVAudioPCMBuffer=BufferClass,
    )
    fake_foundation = SimpleNamespace(
        NSURL=SimpleNamespace(fileURLWithPath_=lambda path: path),
    )
    monkeypatch.setattr("kinovsr.media.audio.av", fake_av)
    monkeypatch.setattr("kinovsr.media.audio.Foundation", fake_foundation)

    track = _native_streaming_audio_track(
        Path("long.mov"),
        start_sec=Fraction(1, 5),
        end_sec=Fraction(4, 5),
        max_duration_sec=Fraction(3, 10),
    )
    assert track is not None
    assert track.n_samples == 3
    assert len(opened) == 1
    assert capacities == []

    try:
        assert len(track._read_interleaved(0, 3)) == 12
    finally:
        track.close()

    assert len(opened) == 2
    assert opened[1].positions == [2]
    assert capacities == [3]


def test_native_audio_cursor_ends_cleanly_at_and_past_the_end(tmp_path):
    # Since macOS 27.0.1, AVAudioFile fails a read that starts at or past the
    # end with kAudioFileEndOfFileError; 27.0 returned NO without an error.
    from kinovsr.media.audio import _AVAudioFileSource, write_wav_float32

    ramp = mx.arange(1000, dtype=mx.float32)[None] / 1000
    path = tmp_path / "ramp.wav"
    write_wav_float32(ramp, path, 48_000)
    source = _AVAudioFileSource(path, 48_000, 1)
    try:
        assert source.read_frames(1000, 1016) == b""
        assert source.read_frames(1100, 1116) == b""
        tail = memoryview(source.read_frames(990, 1010)).cast("f")
        assert tail.tolist() == ramp[0, 990:].tolist()
        # The writer's next pull starts where the short read stopped.
        assert source.read_frames(1000, 1016) == b""
    finally:
        source.close()


def test_native_audio_cursor_takes_only_the_end_of_file_status_as_the_end():
    import Foundation

    from kinovsr.media.audio import _is_end_of_file

    def error(domain, code):
        return Foundation.NSError.errorWithDomain_code_userInfo_(domain, code, None)

    assert _is_end_of_file(None)
    assert _is_end_of_file(error(Foundation.NSOSStatusErrorDomain, -39))
    assert not _is_end_of_file(error(Foundation.NSOSStatusErrorDomain, -40))
    assert not _is_end_of_file(error(Foundation.NSCocoaErrorDomain, -39))


def test_streaming_sidecar_uses_a_fixed_pcm_pull_budget(tmp_path):
    from kinovsr.media.audio import _SIDECAR_PCM_BUDGET, StreamingAudioTrack

    sources = []
    bytes_per_frame = 8
    chunk_frames = _SIDECAR_PCM_BUDGET // bytes_per_frame

    class Source:
        def __init__(self):
            self.reads = []
            self.closed = False
            sources.append(self)

        def read_frames(self, start, end):
            self.reads.append((start, end))
            return bytes((end - start) * bytes_per_frame)

        def close(self):
            self.closed = True

    track = StreamingAudioTrack(
        sample_rate=48_000,
        channels=2,
        n_samples=chunk_frames + 1,
        source_factory=Source,
    )
    track.save_wav(tmp_path / "bounded.wav")

    assert track._source is None
    assert len(sources) == 1
    assert sources[0].closed
    assert sources[0].reads == [
        (0, chunk_frames),
        (chunk_frames, chunk_frames + 1),
    ]


def test_ffmpeg_clock_quantization_snaps_but_preserves_edits():
    from kinovsr.media.ffmpeg_reader import _audio_segment_start

    tick = Fraction(48)  # 1 ms at 48 kHz
    assert _audio_segment_start(1008, 1024, tick) == 1024
    assert _audio_segment_start(2064, 2032, tick) == 2032
    assert _audio_segment_start(30_000, 2048, tick) == 30_000
    assert _audio_segment_start(1024, 2048, tick) == 1024


@pytest.fixture
def ffmpeg_audio_segments():
    """Feed timestamped PCM frames through the real bounded FFmpeg cursor."""
    from kinovsr.media.ffmpeg_reader import _FFmpegAudioSource

    sources = []

    def make(segments, *, stream_time_base=Fraction(1, 48_000)):
        frames = [
            SimpleNamespace(
                pts=start,
                time_base=Fraction(1, 48_000),
                samples=count,
                planes=(array("f", [value]) * count,),
            )
            for start, count, value in segments
        ]
        container = SimpleNamespace(decode=lambda _stream: iter(frames), close=lambda: None)
        source = _FFmpegAudioSource(Path("timestamped-audio.mkv"), 48_000, 1, "mono", Fraction(0))
        source._container = container
        source._stream = SimpleNamespace(time_base=stream_time_base)
        resampler = SimpleNamespace(resample=lambda frame: [] if frame is None else [frame])
        source._segments = iter(source._segment_iter(resampler))
        source._cursor = 0
        sources.append(source)
        return source

    yield make
    for source in sources:
        source.close()


def test_ffmpeg_cursor_trims_six_sample_overlap_at_sample_12500(ffmpeg_audio_segments):
    source = ffmpeg_audio_segments([(0, 12_500, 1.0), (12_494, 12, 2.0), (12_506, 4, 3.0)])

    first = source.read_frames(0, 8192)
    second = source.read_frames(8192, 12_510)
    samples = memoryview(first + second).cast("f")

    assert len(samples) == 12_510
    assert list(samples[12_494:12_500]) == [1.0] * 6
    assert list(samples[12_500:12_506]) == [2.0] * 6
    assert list(samples[12_506:12_510]) == [3.0] * 4


def test_ffmpeg_cursor_corrects_compensating_vorbis_timestamps(ffmpeg_audio_segments):
    source = ffmpeg_audio_segments([(0, 576, 1.0), (1024, 576, 2.0), (1152, 128, 3.0)])

    samples = memoryview(source.read_frames(0, 1280)).cast("f")

    assert len(samples) == 1280
    assert list(samples[576:1152]) == [2.0] * 576
    assert list(samples[1152:1280]) == [3.0] * 128


def test_ffmpeg_cursor_uses_container_clock_after_resampling(ffmpeg_audio_segments):
    source = ffmpeg_audio_segments(
        [(0, 1045, 1.0), (1034, 1115, 2.0)],
        stream_time_base=Fraction(1, 1000),
    )

    samples = memoryview(source.read_frames(0, 2160)).cast("f")

    assert len(samples) == 2160
    assert list(samples[1045:2160]) == [2.0] * 1115


def test_ffmpeg_missing_start_time_uses_first_packet_origin(monkeypatch):
    from kinovsr.media import ffmpeg_reader

    stream = SimpleNamespace(
        type="audio",
        start_time=None,
        time_base=Fraction(1, 1000),
        rate=48_000,
        codec_context=SimpleNamespace(sample_rate=48_000),
        layout=SimpleNamespace(channels=[object()], name="mono"),
    )

    class Container:
        def __init__(self):
            self.streams = [stream]
            self.format = SimpleNamespace(name="matroska,webm", flags=0)
            self.seeks = []
            self.closed = False

        @staticmethod
        def demux(_stream):
            return iter([SimpleNamespace(pts=10_000, size=1)])

        def seek(self, timestamp, **_kwargs):
            self.seeks.append(timestamp)

        def close(self):
            self.closed = True

    probe_container = Container()
    origin = ffmpeg_reader._audio_origin(probe_container, stream)
    assert origin == Fraction(10)
    assert ffmpeg_reader._audio_sample_position(10_000, Fraction(1, 1000), origin, 48_000) == 0

    decode_container = Container()
    monkeypatch.setattr(ffmpeg_reader.av, "open", lambda _path, **_kwargs: decode_container)
    monkeypatch.setattr(ffmpeg_reader.av, "AudioResampler", lambda **_kwargs: object())
    source = ffmpeg_reader._FFmpegAudioSource(Path("offset.mkv"), 48_000, 1, "mono", origin)

    source._reset(96_000)

    # A two-second relative target seeks one second before it for fixed
    # decoder preroll, on the real ten-second absolute source clock.
    assert decode_container.seeks == [11_000]
    source.close()
    assert decode_container.closed


def test_ffmpeg_cursor_uses_container_seek_when_audio_stream_seek_fails(monkeypatch):
    import av

    from kinovsr.media import ffmpeg_reader

    stream = SimpleNamespace(
        type="audio",
        start_time=0,
        time_base=Fraction(1, 48_000),
        rate=48_000,
        layout=SimpleNamespace(channels=[object()], name="mono"),
        codec_context=SimpleNamespace(sample_rate=48_000),
    )

    class Container:
        def __init__(self):
            self.streams = [stream]
            self.format = SimpleNamespace(name="avi", flags=0)
            self.seeks = []

        def seek(self, timestamp, **kwargs):
            self.seeks.append((timestamp, kwargs))
            if "stream" in kwargs:
                raise av.error.PermissionError(1, "Operation not permitted")

        def close(self):
            pass

    container = Container()
    monkeypatch.setattr(ffmpeg_reader.av, "open", lambda _path, **_kwargs: container)
    monkeypatch.setattr(ffmpeg_reader.av, "AudioResampler", lambda **_kwargs: object())
    source = ffmpeg_reader._FFmpegAudioSource(
        Path("unseekable.avi"), 48_000, 1, "mono", Fraction(0)
    )

    source._reset(96_000)
    assert container.seeks == [
        (48_000, {"stream": stream, "backward": True, "any_frame": False}),
        (1_000_000, {"backward": True, "any_frame": False}),
    ]
    source.close()


def test_ffmpeg_audio_window_normalizes_uncommon_sample_rate(monkeypatch):
    from kinovsr.media import ffmpeg_reader

    stream = SimpleNamespace(
        type="audio",
        start_time=0,
        time_base=Fraction(1, 11_024),
        rate=11_024,
        duration=22_048,
        layout=SimpleNamespace(channels=[object()], name="mono"),
        codec_context=SimpleNamespace(sample_rate=11_024),
    )
    container = SimpleNamespace(streams=[stream], duration=None, close=lambda: None)
    monkeypatch.setattr(ffmpeg_reader, "_open_container", lambda *_args: container)

    track = ffmpeg_reader.read_audio_track_window(Path("camera.avi"), end_sec=Fraction(1))
    assert track is not None
    assert (track.sample_rate, track.n_samples) == (48_000, 48_000)
    source = track._source_factory()
    assert source._source_sample_rate == 11_024


def test_ffmpeg_cursor_preserves_gap_as_silence_and_ends_cleanly():
    from kinovsr.media.ffmpeg_reader import _FFmpegAudioSource

    source = _FFmpegAudioSource(Path("gap.mkv"), 48_000, 1, "mono", Fraction(0))
    source._cursor = 0
    source._segments = iter(
        [
            (0, 2, array("f", [1.0, 2.0]).tobytes()),
            (4, 6, array("f", [5.0, 6.0]).tobytes()),
        ]
    )

    raw = source.read_frames(0, 8)
    samples = array("f")
    samples.frombytes(raw)

    assert samples.tolist() == [1.0, 2.0, 0.0, 0.0, 5.0, 6.0]
    assert source.read_frames(6, 8) == b""
