"""File endpoints: probe-to-spec, unit grids, mux policy, and the
file-to-file runs that close M3's deferred acceptance.

The interpolation case is the audio-synchronization file proof: duration
preservation was proven in-memory in M3; here the same chain runs
against a real container with a real audio track and the output file's
video and audio timelines must agree.
"""

import math

import pytest

av = pytest.importorskip("av")

from fractions import Fraction  # noqa: E402

from kinovsr.pipeline import FileSource, run_file  # noqa: E402
from kinovsr.processors.errors import MediaError  # noqa: E402
from kinovsr.processors.specs import Domain, DType, Layout  # noqa: E402
from kinovsr.settings import Settings  # noqa: E402

pytestmark = pytest.mark.integration

W, H, N, FPS = 160, 128, 24, 25
SAMPLE_RATE = 48000
SETTINGS = Settings()


def _write_clip(path, *, with_audio: bool) -> None:
    out = av.open(str(path), "w")
    vs = out.add_stream("mpeg4", rate=FPS)
    vs.width, vs.height = W, H
    vs.pix_fmt = "yuv420p"
    vs.options = {"g": "8", "bf": "0", "qscale": "2"}
    audio = out.add_stream("aac", rate=SAMPLE_RATE) if with_audio else None

    for t in range(N):
        rows = bytearray()
        for _y in range(H):
            rows += bytes(min(255, (x + 2 * t) % 256) for x in range(W))
        frame = av.VideoFrame(W, H, "gray")
        frame.planes[0].update(bytes(rows))
        for pkt in vs.encode(frame.reformat(format="yuv420p")):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)

    if audio is not None:
        total = SAMPLE_RATE * N // FPS
        chunk = 1024
        for start in range(0, total, chunk):
            n = min(chunk, total - start)
            pcm = b"".join(
                int(12000 * math.sin(
                    2 * math.pi * 440 * (start + i) / SAMPLE_RATE)
                    ).to_bytes(2, "little", signed=True)
                for i in range(n))
            af = av.AudioFrame(format="s16", layout="mono", samples=n)
            af.planes[0].update(pcm)
            af.sample_rate = SAMPLE_RATE
            af.pts = start
            for pkt in audio.encode(af):
                out.mux(pkt)
        for pkt in audio.encode():
            out.mux(pkt)
    out.close()


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("run_file") / "clip.mp4"
    _write_clip(path, with_audio=False)
    return path


@pytest.fixture(scope="module")
def clip_with_audio(tmp_path_factory):
    path = tmp_path_factory.mktemp("run_file_audio") / "clip.mp4"
    _write_clip(path, with_audio=True)
    return path


def _stream_seconds(path):
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        video_s = float(video.duration * video.time_base)
        frames = sum(1 for _ in container.decode(video=0))
        audio_s = None
        if container.streams.audio:
            track = container.streams.audio[0]
            audio_s = float(track.duration * track.time_base)
        return video_s, audio_s, frames, (video.width, video.height)


class TestFileSource:
    def test_probe_produces_concrete_spec(self, clip):
        source = FileSource(clip)
        frame = source.spec.frame
        assert frame.layout is Layout.MLX_RGB_HWC
        assert frame.dtype is DType.FLOAT32
        assert frame.domain is Domain.UNIT
        assert (frame.geometry.width, frame.geometry.height) == (W, H)
        assert source.spec.timeline.cadence == Fraction(FPS)
        assert source.spec.timeline.time_base == Fraction(1, 24000)
        assert source.spec.seekable and source.spec.lookahead_available
        assert source.frame_count == N

    def test_units_ride_the_cadence_grid(self, clip):
        source = FileSource(clip, max_frames=5)
        units = list(source.units())
        assert [u.pts for u in units] == [i * 960 for i in range(5)]
        assert all(u.duration == 960 for u in units)
        assert all(u.payload.shape == (H, W, 3) for u in units)

    def test_empty_window_is_rejected(self, clip):
        with pytest.raises(MediaError, match="empty frame window"):
            FileSource(clip, start=N + 5)

    def test_audio_with_offset_window_is_refused(self, clip_with_audio):
        source = FileSource(clip_with_audio, start=4)
        with pytest.raises(MediaError, match="audio carry"):
            source.audio_track()


def test_passthrough_file_to_file(clip, tmp_path):
    result = run_file(
        {"pipeline": []}, video=clip, output=tmp_path / "out.mp4",
        settings=SETTINGS)
    assert result.frames_in == result.frames_out == N
    video_s, _, frames, size = _stream_seconds(result.path)
    assert frames == N
    assert size == (W, H)
    assert abs(video_s - N / FPS) < 1.0 / FPS


def test_windowed_run(clip, tmp_path):
    result = run_file(
        {"pipeline": []}, video=clip, output=tmp_path / "win.mp4",
        settings=SETTINGS, start=4, max_frames=8)
    assert result.frames_out == 8
    _, _, frames, _ = _stream_seconds(result.path)
    assert frames == 8


def test_learned_chain_through_endpoints(clip, tmp_path):
    config = {
        "pipeline": ["up"],
        "up": {"processor": "metalfx", "scale": 2},
    }
    try:
        result = run_file(
            config, video=clip, output=tmp_path / "sr.mp4",
            settings=SETTINGS, max_frames=6)
    except MediaError as exc:
        if "not supported" in str(exc):
            pytest.skip(f"MetalFX unavailable: {exc}")
        raise
    assert result.frames_out == 6
    assert result.output_spec.frame.geometry.width == W * 2
    _, _, frames, size = _stream_seconds(result.path)
    assert frames == 6
    assert size == (W * 2, H * 2)


def test_interpolation_preserves_duration_and_carries_audio(
        clip_with_audio, tmp_path):
    """M3 acceptance closure: the one-to-many cadence rewrite through a
    real container keeps the video timeline equal to the source window
    and the carried audio in sync with it."""
    config = {
        "pipeline": ["fps"],
        "fps": {"processor": "videotoolbox", "profile": "normal",
                "target_fps": 50},
    }
    result = run_file(
        config, video=clip_with_audio, output=tmp_path / "interp.mp4",
        settings=SETTINGS, layout=Layout.CV_RGBA_HALF, audio=True)

    source_seconds = N / FPS
    assert result.frames_out == 2 * N
    video_s, audio_s, frames, size = _stream_seconds(result.path)
    assert frames == 2 * N
    assert size == (W, H)
    # Duration preserved through the cadence rewrite...
    assert abs(video_s - source_seconds) < 1.0 / FPS
    # ...and the muxed audio agrees with the video timeline (the actual
    # synchronization proof; AAC priming allows sub-frame skew).
    assert audio_s is not None, "audio track missing from the output"
    assert abs(audio_s - video_s) < 0.05
