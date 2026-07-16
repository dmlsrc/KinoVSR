"""File endpoints: probe-to-spec, unit grids, mux policy, and the
file-to-file runs that close M3's deferred acceptance.

The interpolation case is the audio-synchronization file proof: duration
preservation was proven in-memory in M3; here the same chain runs
against a real container with a real audio track and the output file's
video and audio timelines must agree.
"""

import logging
import math
from array import array
from fractions import Fraction
from pathlib import Path

import av
import pytest

from kinovsr.pipeline import FileSource, run_file
from kinovsr.processors.errors import MediaError
from kinovsr.processors.specs import Domain, DType, Layout
from kinovsr.settings import Settings

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


def _write_vfr_clip(path) -> None:
    """Five samples at 0, 1/30, 3/30, 4/30, and 7/30 seconds."""
    out = av.open(str(path), "w")
    stream = out.add_stream("mpeg4", rate=30)
    stream.width, stream.height = 64, 64
    stream.pix_fmt = "yuv420p"
    stream.options = {"g": "30", "bf": "0"}
    for index, pts in enumerate((0, 1, 3, 4, 7)):
        frame = av.VideoFrame(64, 64, "gray")
        frame.planes[0].update(bytes([index * 40]) * (64 * 64))
        frame = frame.reformat(format="yuv420p")
        frame.pts = pts
        frame.time_base = Fraction(1, 30)
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()


def _write_staggered_tracks(path) -> None:
    """One-second video at t=10 beside one-second audio at t=0."""
    out = av.open(str(path), "w")
    video = out.add_stream("mpeg4", rate=25)
    video.width = video.height = 64
    video.pix_fmt = "yuv420p"
    video.options = {"bf": "0"}
    audio = out.add_stream("aac", rate=SAMPLE_RATE, layout="mono")

    for index in range(25):
        frame = av.VideoFrame(64, 64, "gray")
        frame.planes[0].update(bytes([index * 4]) * (64 * 64))
        frame.pts = 250 + index
        frame.time_base = Fraction(1, 25)
        for packet in video.encode(frame.reformat(format="yuv420p")):
            out.mux(packet)
    for packet in video.encode():
        out.mux(packet)

    silence = array("f", [0.0]) * 1024
    for pts in range(0, 47 * 1024, 1024):
        frame = av.AudioFrame(format="fltp", layout="mono", samples=1024)
        frame.sample_rate = SAMPLE_RATE
        frame.pts = pts
        frame.time_base = Fraction(1, SAMPLE_RATE)
        frame.planes[0].update(silence.tobytes())
        for packet in audio.encode(frame):
            out.mux(packet)
    for packet in audio.encode():
        out.mux(packet)
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


@pytest.fixture(scope="module")
def vfr_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("run_file_vfr") / "clip.mp4"
    _write_vfr_clip(path)
    return path


@pytest.fixture(scope="module")
def staggered_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("run_file_staggered") / "clip.mp4"
    _write_staggered_tracks(path)
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


def _decoded_audio_samples(path):
    with av.open(str(path)) as container:
        return sum(frame.samples for frame in container.decode(audio=0))


class TestFileSource:
    @pytest.mark.parametrize(("layout", "expected"), [
        (Layout.MLX_RGB_HWC, 1),
        (Layout.CV_RGBA_HALF, 1),
        (Layout.CV_BGRA, 2),
        (Layout.CV_NV12, 4),
    ])
    def test_decode_chunk_is_capped_by_layout_memory(self, layout, expected):
        from kinovsr.pipeline.run import _effective_decode_chunk_size

        assert _effective_decode_chunk_size(
            32, 3840, 2160, layout) == expected

    def test_decode_chunk_preserves_a_smaller_user_limit(self):
        from kinovsr.pipeline.run import _effective_decode_chunk_size

        assert _effective_decode_chunk_size(
            3, 640, 480, Layout.CV_RGBA_HALF) == 3

    def test_single_surface_larger_than_budget_still_makes_progress(self):
        from kinovsr.pipeline.run import _effective_decode_chunk_size

        assert _effective_decode_chunk_size(
            32, 7680, 4320, Layout.CV_RGBA_HALF) == 1

    def test_forced_color_budgets_yuv_and_rgbahalf_surfaces(self):
        from kinovsr.pipeline.run import _effective_decode_chunk_size

        assert _effective_decode_chunk_size(
            32, 1280, 720, Layout.MLX_RGB_HWC,
            forced_color=True) == 4

    @pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
    def test_invalid_decode_chunk_is_rejected(self, chunk_size):
        from kinovsr.pipeline.run import _effective_decode_chunk_size

        with pytest.raises(MediaError, match="positive integer"):
            _effective_decode_chunk_size(
                chunk_size, 1920, 1080, Layout.MLX_RGB_HWC)

    def test_effective_chunk_reaches_the_reader(self, tmp_path):
        captured = {}

        class Reader:
            @staticmethod
            def probe_video(_path):
                return 3840, 2160, 25.0, 10, None, None

            @staticmethod
            def probe_color(_path):
                return {
                    "primaries": None,
                    "transfer": None,
                    "matrix": None,
                    "full_range": False,
                    "tagged": False,
                }

            @staticmethod
            def iter_video_buffer_chunks(
                    _path, _format, chunk_size=8, *, start_frame=0,
                    end_frame=None):
                captured["chunk_size"] = chunk_size
                return iter(())

        source = FileSource(
            tmp_path / "synthetic.mov",
            layout=Layout.CV_BGRA,
            chunk_size=32,
            reader=Reader,
        )
        assert source.chunk_size == 2
        assert list(source.units()) == []
        assert captured["chunk_size"] == 2

    def test_forced_effective_chunk_reaches_the_reader(self, tmp_path):
        captured = {}

        class Reader:
            @staticmethod
            def probe_video(_path):
                return 1280, 720, 25.0, 10, None, None

            @staticmethod
            def probe_color(_path):
                return {
                    "primaries": None,
                    "transfer": None,
                    "matrix": None,
                    "full_range": False,
                    "tagged": False,
                }

            @staticmethod
            def iter_forced_color_chunks(
                    _path, _format, _matrix, _full_range, chunk_size=8,
                    *, start_frame=0, end_frame=None,
                    reinterpret_full_range=None):
                captured["chunk_size"] = chunk_size
                return iter(())

        source = FileSource(
            tmp_path / "synthetic.mov",
            layout=Layout.MLX_RGB_HWC,
            chunk_size=32,
            source_color="bt709",
            reader=Reader,
        )
        assert source.chunk_size == 4
        assert list(source.units()) == []
        assert captured["chunk_size"] == 4

    def test_auto_geometry_recomputes_its_rgbahalf_chunk(
            self, tmp_path, monkeypatch):
        captured = {}

        class Reader:
            @staticmethod
            def probe_video(_path):
                return 3840, 2160, 25.0, 10, None, None

            @staticmethod
            def probe_color(_path):
                return {
                    "primaries": None,
                    "transfer": None,
                    "matrix": None,
                    "full_range": False,
                    "tagged": False,
                }

            @staticmethod
            def iter_video_buffer_chunks(
                    _path, _format, chunk_size=8, *, start_frame=0,
                    end_frame=None):
                return iter(())

        from kinovsr.pipeline import auto_geometry

        def resolve(config, *, video, vr, pixel_aspect, chunk_size):
            captured["chunk_size"] = chunk_size
            return {"pipeline": []}

        monkeypatch.setattr(auto_geometry, "resolve_auto_geometry", resolve)
        result = run_file(
            {"pipeline": ["crop"],
             "crop": {"processor": "crop", "bars": "auto"}},
            video=tmp_path / "synthetic.mov",
            output=tmp_path / "unused.mp4",
            settings=SETTINGS,
            layout=Layout.CV_NV12,
            chunk_size=32,
            reader=Reader,
            skip_post_mp4=True,
        )
        assert result.frames_out == 0
        assert captured["chunk_size"] == 1

    @pytest.mark.parametrize("chunk_size", [
        0, -1, True, False, 1.0, 1.5, "8", None,
    ])
    def test_invalid_chunk_precedes_file_source_media_io(
            self, tmp_path, chunk_size):
        class Reader:
            @staticmethod
            def probe_video_timing(_path):
                raise AssertionError("invalid chunk size reached media probe")

        with pytest.raises(MediaError, match="positive integer"):
            FileSource(
                tmp_path / "unread.mov",
                chunk_size=chunk_size,
                reader=Reader,
            )

    def test_invalid_chunk_precedes_run_file_media_io(self, tmp_path):
        class Reader:
            @staticmethod
            def probe_video_timing(_path):
                raise AssertionError("invalid chunk size reached media probe")

        with pytest.raises(MediaError, match="positive integer"):
            run_file(
                {"pipeline": []},
                video=tmp_path / "unread.mov",
                output=tmp_path / "unused.mp4",
                settings=SETTINGS,
                chunk_size=0,
                reader=Reader,
            )

    def test_probe_reports_resolved_source_color(self, clip, caplog):
        with caplog.at_level(logging.INFO, logger="kinovsr.pipeline.run"):
            FileSource(clip)
        assert any(
            message.startswith("Source color:")
            for message in caplog.messages)

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

    def test_vfr_is_rejected_from_actual_sample_timestamps(self, vfr_clip):
        from kinovsr.media import ffmpeg_reader, video_reader

        native = video_reader.probe_video_timing(vfr_clip)
        fallback = ffmpeg_reader.probe_video_timing(vfr_clip)
        assert native.sample_count == fallback.sample_count == 5
        assert native.cadence is fallback.cadence is None
        assert native.source_tick == fallback.source_tick == Fraction(1, 15360)
        with pytest.raises(MediaError, match="variable frame rate"):
            FileSource(vfr_clip)

    def test_vfr_rejection_precedes_the_output_transaction(
            self, vfr_clip, tmp_path, monkeypatch):
        from kinovsr.pipeline import run as run_module

        def transaction_must_not_start(_self):
            raise AssertionError("output transaction started for VFR input")

        monkeypatch.setattr(
            run_module._OutputTransaction, "__enter__",
            transaction_must_not_start)
        output = tmp_path / "vfr.mp4"
        with pytest.raises(MediaError, match="variable frame rate"):
            run_file(
                {"pipeline": []}, video=vfr_clip, output=output,
                settings=SETTINGS)
        assert not output.exists()
        assert not list(tmp_path.glob("*.partial"))

    def test_staggered_audio_video_origins_are_rejected_before_transaction(
            self, staggered_clip, tmp_path, monkeypatch):
        from kinovsr.media import ffmpeg_reader, video_reader
        from kinovsr.pipeline import run as run_module

        native_video = video_reader.probe_video_timing(staggered_clip)
        native_audio = video_reader.probe_audio_timing(staggered_clip)
        fallback_video = ffmpeg_reader.probe_video_timing(staggered_clip)
        fallback_audio = ffmpeg_reader.probe_audio_timing(staggered_clip)
        assert native_video.first_pts == fallback_video.first_pts == 10
        assert native_audio is not None and fallback_audio is not None
        assert native_audio.first_pts == fallback_audio.first_pts == 0

        def transaction_must_not_start(_self):
            raise AssertionError(
                "output transaction started for staggered track origins")

        monkeypatch.setattr(
            run_module._OutputTransaction, "__enter__",
            transaction_must_not_start)
        output = tmp_path / "staggered.mp4"
        with pytest.raises(MediaError, match="staggered audio/video"):
            run_file(
                {"pipeline": []}, video=staggered_clip, output=output,
                settings=SETTINGS, audio=True)
        assert not output.exists()
        assert not list(tmp_path.glob("*.partial"))

    def test_audio_origin_tolerance_is_only_the_half_tick_error_envelope(self):
        from kinovsr.media.timing import AudioTiming, VideoTiming
        from kinovsr.pipeline.run import _validate_audio_origin

        video = VideoTiming(
            sample_count=25,
            cadence=Fraction(25),
            first_pts=Fraction(0),
            duration=Fraction(1),
            source_tick=Fraction(1, 25),
        )
        envelope = (Fraction(1, 25) + Fraction(1, 48000)) / 2

        class BoundaryReader:
            @staticmethod
            def probe_audio_timing(_path):
                return AudioTiming(
                    first_pts=envelope, source_tick=Fraction(1, 48000))

        class WholeTickReader:
            @staticmethod
            def probe_audio_timing(_path):
                return AudioTiming(
                    first_pts=Fraction(1, 25),
                    source_tick=Fraction(1, 48000))

        _validate_audio_origin(Path("coarse.avi"), BoundaryReader, video)
        with pytest.raises(MediaError, match="staggered audio/video"):
            _validate_audio_origin(Path("coarse.avi"), WholeTickReader, video)

    def test_audio_carry_trims_to_the_window(self, clip_with_audio):
        source = FileSource(clip_with_audio, start=4, max_frames=8)
        track = source.audio_track()
        expected = round(8 / FPS * SAMPLE_RATE)
        assert abs(track.n_samples - expected) <= 2
        assert track._source is None

    def test_unbounded_reader_audio_hook_is_a_typed_public_error(
            self, clip_with_audio, tmp_path):
        from kinovsr.media import video_reader

        calls = []

        class LegacyAdapter:
            def __getattr__(self, name):
                return getattr(video_reader, name)

            @staticmethod
            def read_audio_track(path):
                calls.append(path)
                raise AssertionError("unbounded decoder must not run")

        output = tmp_path / "legacy.mp4"
        with pytest.raises(
                MediaError, match="refusing unbounded read_audio_track"):
            run_file(
                {"pipeline": []},
                video=clip_with_audio,
                output=output,
                settings=SETTINGS,
                audio=True,
                reader=LegacyAdapter(),
            )

        assert calls == []
        assert not output.exists()
        assert not list(tmp_path.glob("*.partial"))

    def test_run_file_refuses_output_over_input(self, clip_with_audio,
                                                tmp_path):
        with pytest.raises(MediaError, match="destroy the source"):
            run_file({"pipeline": []}, video=clip_with_audio,
                     output=clip_with_audio, settings=SETTINGS)
        assert clip_with_audio.exists()


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


def test_audio_sidecar_written_beside_the_output(clip_with_audio, tmp_path):
    from kinovsr.media.audio import read_wav

    out = tmp_path / "out.mp4"
    run_file(
        {"pipeline": []}, video=clip_with_audio, output=out, settings=SETTINGS,
        audio=True, save_audio_sidecar=True)
    sidecar = out.resolve().with_name("out_audio.wav")
    assert sidecar.exists()
    # a real WAV the audio reader round-trips at the source rate
    rate, samples = read_wav(sidecar)
    assert rate == SAMPLE_RATE
    assert samples.shape[0] >= 1
    assert samples.shape[1] == N * SAMPLE_RATE // FPS


def test_windowed_audio_replays_to_sidecar_post_and_comparison(
        clip_with_audio, tmp_path):
    """Every consumer receives an independent cursor over identical samples."""
    from kinovsr.media.audio import read_wav

    out = tmp_path / "out.mp4"
    comparison = tmp_path / "comparison.mp4"
    result = run_file(
        {"pipeline": []},
        video=clip_with_audio,
        output=out,
        settings=SETTINGS,
        start=4,
        max_frames=8,
        audio=True,
        audio_codec="alac",
        save_audio_sidecar=True,
        comparison=comparison,
    )
    expected = 8 * SAMPLE_RATE // FPS
    rate, sidecar = read_wav(out.with_name("out_audio.wav"))

    assert result.frames_out == 8
    assert rate == SAMPLE_RATE and sidecar.shape[1] == expected
    assert _decoded_audio_samples(out) == expected
    assert _decoded_audio_samples(comparison) == expected


def test_aac_audio_window_preserves_decoded_onset_and_count(
        clip_with_audio, tmp_path):
    out = tmp_path / "aac.mp4"
    run_file(
        {"pipeline": []},
        video=clip_with_audio,
        output=out,
        settings=SETTINGS,
        start=4,
        max_frames=8,
        audio=True,
        audio_codec="aac",
    )

    expected = 8 * SAMPLE_RATE // FPS
    assert abs(_decoded_audio_samples(out) - expected) <= 1024
    with av.open(str(out)) as container:
        first = next(container.decode(audio=0))
        assert first.pts is not None
        assert Fraction(first.pts) * Fraction(first.time_base) == 0


class TestForcedColor:
    def test_forces_the_resolved_matrix_tag(self, clip):
        from kinovsr.processors import ColorMatrix

        auto = FileSource(clip).spec.frame.color_matrix
        forced = FileSource(clip, source_color="bt2020")
        # forcing overrides how the source is read (this untagged clip is
        # auto-guessed as something other than 2020) and re-tags the spec.
        assert forced.spec.frame.color_matrix is ColorMatrix.BT2020
        assert forced.spec.frame.color_matrix is not auto
        assert forced._force_read

    def test_forced_run_re_decodes_and_re_tags_the_output(self, clip, tmp_path):
        res = run_file(
            {"pipeline": []}, video=clip, output=tmp_path / "o.mp4",
            settings=SETTINGS, source_color="bt2020")
        assert res.frames_out == N
        with av.open(str(res.path)) as container:
            # BT.2020 primaries (H.273 value 9): the forced decode + tag ran
            assert container.streams.video[0].codec_context.color_primaries == 9

    def test_forced_color_rejected_on_a_native_layout(self, clip):
        # The re-decode is RGBAHalf-only; a native-CV source layout cannot
        # reinterpret code values.
        with pytest.raises(MediaError, match="MLX decode path"):
            FileSource(clip, layout=Layout.CV_RGBA_HALF, source_color="bt2020")


class TestEncodeChroma:
    # An RGB/MLX chain feeds pooled 4:2:2 YUV directly, so auto is 4:2:2;
    # forcing 420 must reach 4:2:0 even here (the harness parity gap).
    @pytest.mark.parametrize(("chroma", "pix_fmt"), [
        ("auto", "yuv422p10le"),
        ("420", "yuv420p10le"),
        ("422", "yuv422p10le"),
    ])
    def test_forces_the_hevc_chroma_profile(self, clip, tmp_path, chroma, pix_fmt):
        res = run_file(
            {"pipeline": []}, video=clip, output=tmp_path / f"o_{chroma}.mp4",
            settings=SETTINGS, encode_chroma=chroma)
        with av.open(str(res.path)) as container:
            assert container.streams.video[0].codec_context.pix_fmt == pix_fmt


class TestSaveFrames:
    def test_pre_and_post_dumps(self, clip, tmp_path):
        from kinovsr.media.images import load_image_rgb

        pre, post = tmp_path / "pre", tmp_path / "post"
        res = run_file(
            {"pipeline": []}, video=clip, output=tmp_path / "o.mp4",
            settings=SETTINGS, save_pre_frames=pre, save_post_frames=post)
        pre_pngs = sorted(pre.glob("frame_*.png"))
        post_pngs = sorted(post.glob("frame_*.png"))
        # a 1:1 chain: one PNG per source frame in and per output frame out
        assert len(pre_pngs) == res.frames_in == N
        assert len(post_pngs) == res.frames_out == N
        assert pre_pngs[0].name == "frame_00000.png"
        img = load_image_rgb(str(post_pngs[0]))    # a real, loadable RGB image
        assert img.shape == (H, W, 3)

    def test_off_by_default(self, clip, tmp_path):
        run_file({"pipeline": []}, video=clip, output=tmp_path / "o.mp4",
                 settings=SETTINGS)
        assert list(tmp_path.rglob("frame_*.png")) == []


class TestComparison:
    def test_writes_the_side_by_side(self, clip, tmp_path):
        res = run_file(
            {"pipeline": []}, video=clip, output=tmp_path / "o.mp4",
            settings=SETTINGS, comparison=tmp_path / "cmp.mp4")
        assert res.comparison_path == tmp_path / "cmp.mp4"
        with av.open(str(res.comparison_path)) as container:
            codec = container.streams.video[0].codec_context
            assert (codec.width, codec.height) == (2 * W, H)
            decoded = [f.to_ndarray(format="rgb24")
                       for f in container.decode(video=0)]
        assert len(decoded) == res.frames_out == N
        # An empty chain: pre (left) and post (right) are the same frame at
        # scale 1, so the halves differ only by encode noise.
        first = decoded[0].astype("f4")
        halves_diff = abs(first[:, :W] - first[:, W:]).mean()
        assert halves_diff < 8.0

    def test_carries_audio_like_the_post(self, clip_with_audio, tmp_path):
        # The harness fed the same audio kwargs to both writers; the tee's
        # sink carries the same (trimmed) track.
        res = run_file(
            {"pipeline": []}, video=clip_with_audio,
            output=tmp_path / "o.mp4", settings=SETTINGS, audio=True,
            comparison=tmp_path / "cmp.mp4")
        with av.open(str(res.comparison_path)) as container:
            assert container.streams.audio

    def test_pairs_backward_on_a_cadence_doubling_chain(self, clip, tmp_path):
        # 25 -> 50 fps: each source frame yields two outputs, both pairing
        # to the SAME retained source frame (harness fed one src_arr to
        # every frame-rate-converted output). One comparison frame per
        # output unit; the tee raises if pairing desyncs.
        config = {
            "pipeline": ["fps"],
            "fps": {"processor": "videotoolbox", "profile": "normal",
                    "target_fps": 50},
        }
        res = run_file(
            config, video=clip, output=tmp_path / "o.mp4",
            settings=SETTINGS, layout=Layout.CV_RGBA_HALF,
            comparison=tmp_path / "cmp.mp4")
        with av.open(str(res.comparison_path)) as container:
            codec = container.streams.video[0].codec_context
            assert (codec.width, codec.height) == (2 * W, H)
            frames = sum(1 for _ in container.decode(video=0))
        assert frames == res.frames_out
        assert res.frames_out > N   # the cadence really doubled

    def test_none_means_no_comparison(self, clip, tmp_path):
        res = run_file({"pipeline": []}, video=clip,
                       output=tmp_path / "o.mp4", settings=SETTINGS)
        assert res.comparison_path is None


@pytest.fixture(scope="module")
def cut_clip(tmp_path_factory):
    # Two flat scenes, hard cut at frame 8.
    path = tmp_path_factory.mktemp("cuts") / "cuts.mp4"
    out = av.open(str(path), "w")
    vs = out.add_stream("mpeg4", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    vs.options = {"g": "8", "bf": "0", "qscale": "2"}
    for i in range(16):
        frame = av.VideoFrame(W, H, "gray")
        frame.planes[0].update(bytes([30 if i < 8 else 220]) * (W * H))
        for pkt in vs.encode(frame.reformat(format="yuv420p")):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()
    return path


class TestCutLogAndSkipPost:
    def test_cut_log_records_source_indices(self, cut_clip, tmp_path):
        # Harness format: one detected-cut source index per line, file
        # truncated at run start.
        log = tmp_path / "cuts.txt"
        log.write_text("stale\n", encoding="utf-8")
        run_file({"pipeline": ["cd"], "cd": {"processor": "cut_detect"}},
                 video=cut_clip, output=tmp_path / "o.mp4",
                 settings=SETTINGS, cut_log=log, overwrite=True)
        assert log.read_text(encoding="utf-8") == "8\n"

    def test_skip_post_mp4_processes_without_writing(self, cut_clip, tmp_path):
        post = tmp_path / "png"
        res = run_file(
            {"pipeline": []}, video=cut_clip, output=tmp_path / "o.mp4",
            settings=SETTINGS, skip_post_mp4=True, save_post_frames=post)
        assert res.path is None
        assert not (tmp_path / "o.mp4").exists()
        # the run still processed: dumps and counts are real
        assert res.frames_out == 16
        assert len(list(post.glob("frame_*.png"))) == 16


class TestRunDiagnostics:
    def test_noise_map_report_and_debug_png(self, clip, tmp_path, caplog):
        # A map-conditioned denoiser reports its estimated sigma at end of
        # run (the harness's [noise-map] block, family-owned), and
        # noise_map_debug dumps the map beside the post output.
        import logging

        from kinovsr.processors.bsvd import default_weights_path

        if not default_weights_path().exists():
            pytest.skip("bsvd weights not available")
        cfg = {"pipeline": ["dn"],
               "dn": {"processor": "bsvd", "strength": 0.05,
                      "noise_map": "auto"}}
        with caplog.at_level(logging.INFO, logger="kinovsr.pipeline.run"):
            run_file(cfg, video=clip, output=tmp_path / "o.mp4",
                     settings=SETTINGS, end=10, noise_map_debug=True)
        text = caplog.text
        assert "[noise-map] estimated sigma:" in text
        assert "[noise-map] effective conditioning:" in text
        assert (tmp_path / "o_noisemap.png").stat().st_size > 0

    def test_no_debug_flag_writes_no_png(self, clip, tmp_path):
        from kinovsr.processors.bsvd import default_weights_path

        if not default_weights_path().exists():
            pytest.skip("bsvd weights not available")
        cfg = {"pipeline": ["dn"],
               "dn": {"processor": "bsvd", "strength": 0.05,
                      "noise_map": "auto"}}
        run_file(cfg, video=clip, output=tmp_path / "o.mp4",
                 settings=SETTINGS, end=10)
        assert not (tmp_path / "o_noisemap.png").exists()


class TestGopAlign:
    """--snap-start / --gop-align parity: keyframe windowing on the typed
    endpoints. The clip fixture encodes g=8, so keyframes sit at 0, 8, 16."""

    def test_context_frames_ride_negative_pts(self, clip):
        src = FileSource(clip, start=12, end=20, context_frames=4)
        units = list(src.units())
        assert len(units) == 12                      # 4 context + 8 window
        assert units[0].pts < 0
        assert units[4].pts == 0                     # window start anchors 0
        assert src.frame_count == 8                  # context is not output

    def test_context_frames_must_fit_before_start(self, clip):
        with pytest.raises(MediaError, match="context_frames"):
            FileSource(clip, start=2, end=8, context_frames=4)

    def test_corrected_source_clock_reaches_gop_and_trim_readers(
            self, tmp_path):
        from kinovsr.media.timing import VideoTiming

        exact_timing = VideoTiming(
            sample_count=40,
            cadence=Fraction(30),
            first_pts=Fraction(1, 2),
            duration=Fraction(4, 3),
            source_tick=Fraction(1, 30000),
        )
        captured = {}

        class _Reader:
            @staticmethod
            def probe_video_timing(_path):
                return exact_timing

            @staticmethod
            def probe_video(_path):
                # Deliberately wrong legacy nominal/count. The exact timing
                # scan must own every downstream frame-index conversion.
                return 64, 64, 15.0, 20, None, None

            @staticmethod
            def probe_color(_path):
                return {
                    "primaries": None, "transfer": None, "matrix": None,
                    "full_range": False, "tagged": False,
                }

            @staticmethod
            def keyframe_display_indices(_path, *, timing=None):
                captured["keyframe_timing"] = timing
                return [0]

            @staticmethod
            def iter_video_buffer_chunks(
                    _path, _format, chunk_size=8, *, start_frame=0,
                    end_frame=None, timing=None):
                captured["decode"] = (start_frame, end_frame, timing)
                stop = exact_timing.sample_count if end_frame is None else end_frame
                for begin in range(start_frame, stop, chunk_size):
                    yield [object() for _ in range(
                        begin, min(stop, begin + chunk_size))]

        video = tmp_path / "synthetic.mov"
        video.write_bytes(b"reader-owned fixture")
        result = run_file(
            {"pipeline": []}, video=video, output=tmp_path / "unused.mp4",
            settings=SETTINGS, reader=_Reader, layout=Layout.CV_BGRA,
            start=10, end=20, gop_align=True, skip_post_mp4=True)

        assert result.frames_in == result.frames_out == 10
        assert captured["keyframe_timing"] is exact_timing
        assert captured["decode"] == (0, 20, exact_timing)

    @pytest.mark.parametrize(("minimum", "maximum"), [
        (0, 0),
        (-1, 16),
        (16, 0),
        (32, 16),
    ])
    def test_invalid_window_bounds_fail_before_output(
            self, clip, tmp_path, minimum, maximum):
        output = tmp_path / f"invalid_{minimum}_{maximum}.mp4"
        with pytest.raises(MediaError, match="invalid GOP window bounds"):
            run_file(
                {"pipeline": []}, video=clip, output=output,
                settings=SETTINGS, gop_align=True,
                gop_min_window=minimum, gop_max_window=maximum)
        assert not output.exists()

    def test_gop_align_drops_context_outputs(self, clip, tmp_path):
        # start mid-GOP: the enclosing keyframe (8) extends the read, the
        # [8, 12) context is processed but never written.
        res = run_file(
            {"pipeline": []}, video=clip, output=tmp_path / "gop.mp4",
            settings=SETTINGS, start=12, end=20, gop_align=True)
        plain = run_file(
            {"pipeline": []}, video=clip, output=tmp_path / "plain.mp4",
            settings=SETTINGS, start=12, end=20)
        assert res.frames_out == plain.frames_out == 8
        # identical content: the context changed nothing on an empty chain
        with av.open(str(res.path)) as a, av.open(str(plain.path)) as b:
            first_gop = next(a.decode(video=0)).to_ndarray(format="rgb24")
            first_plain = next(b.decode(video=0)).to_ndarray(format="rgb24")
        diff = abs(first_gop.astype("f4") - first_plain.astype("f4")).mean()
        assert diff < 2.0

    @pytest.mark.parametrize(("target_fps", "expected_frames"), [
        (40, 13),
        (50, 16),
    ])
    def test_gop_context_cadence_change_rebases_the_public_grid(
            self, clip, tmp_path, target_fps, expected_frames):
        config = {
            "pipeline": ["fps"],
            "fps": {"processor": "videotoolbox", "profile": "normal",
                    "target_fps": target_fps},
        }
        result = run_file(
            config, video=clip,
            output=tmp_path / f"gop_{target_fps}.mp4",
            settings=SETTINGS, layout=Layout.CV_RGBA_HALF,
            start=12, end=20, gop_align=True)
        skipped = run_file(
            config, video=clip,
            output=tmp_path / f"skip_{target_fps}.mp4",
            settings=SETTINGS, layout=Layout.CV_RGBA_HALF,
            start=12, end=20, gop_align=True, skip_post_mp4=True)

        assert result.frames_out == skipped.frames_out == expected_frames
        assert result.output_spec.timeline == skipped.output_spec.timeline
        with av.open(str(result.path)) as container:
            stream = container.streams.video[0]
            times = sorted(
                Fraction(frame.pts) * Fraction(stream.time_base)
                for frame in container.decode(video=0))
        assert times == [Fraction(i, target_fps)
                         for i in range(expected_frames)]
        # The first retained FRC sample must select the requested in-point,
        # not a frame from the GOP-only warmup prefix. The synthetic source
        # changes monotonically, so frame 12 is the closest source image.
        with av.open(str(clip)) as container:
            source_frames = [
                frame.to_ndarray(format="rgb24")
                for frame in container.decode(video=0)
            ]
        with av.open(str(result.path)) as container:
            first = next(container.decode(video=0)).to_ndarray(format="rgb24")
        diffs = [
            abs(first.astype("f4") - source_frames[index].astype("f4")).mean()
            for index in range(10, 15)
        ]
        assert diffs.index(min(diffs)) == 2

    def test_gop_nonintegral_cadence_keeps_audio_on_the_rebased_clip(
            self, clip_with_audio, tmp_path):
        config = {
            "pipeline": ["fps"],
            "fps": {"processor": "videotoolbox", "profile": "normal",
                    "target_fps": 40},
        }
        result = run_file(
            config, video=clip_with_audio, output=tmp_path / "gop_audio.mp4",
            settings=SETTINGS, layout=Layout.CV_RGBA_HALF,
            start=12, end=20, gop_align=True, audio=True)
        video_s, audio_s, frames, _ = _stream_seconds(result.path)
        assert result.frames_out == frames == 13
        assert abs(video_s - 8 / 25) < 0.008
        assert audio_s is not None
        assert abs(audio_s - video_s) < 0.03
        with av.open(str(result.path)) as container:
            assert container.streams.video[0].start_time == 0
            assert container.streams.audio[0].start_time == 0

    def test_one_frame_gop_window_keeps_the_true_target_phase_and_duration(
            self, clip_with_audio, tmp_path):
        config = {
            "pipeline": ["fps"],
            "fps": {"processor": "videotoolbox", "profile": "normal",
                    "target_fps": 40},
        }
        result = run_file(
            config, video=clip_with_audio, output=tmp_path / "one_frame.mp4",
            settings=SETTINGS, layout=Layout.CV_RGBA_HALF,
            start=4, end=5, gop_align=True, audio=True)
        plain = run_file(
            config, video=clip_with_audio, output=tmp_path / "one_plain.mp4",
            settings=SETTINGS, layout=Layout.CV_RGBA_HALF,
            start=4, end=5)

        with av.open(str(result.path)) as container:
            video = container.streams.video[0]
            audio = container.streams.audio[0]
            frames = list(container.decode(video=0))
            times = sorted(
                Fraction(frame.pts) * Fraction(video.time_base)
                for frame in frames)
            video_duration = Fraction(video.duration) * Fraction(video.time_base)
            audio_duration = Fraction(audio.duration) * Fraction(audio.time_base)
            first_gop = frames[0].to_ndarray(format="rgb24")
        with av.open(str(plain.path)) as container:
            first_plain = next(container.decode(video=0)).to_ndarray(
                format="rgb24")
        assert result.frames_out == 2
        assert times == [Fraction(0), Fraction(1, 40)]
        assert video_duration == Fraction(1, 25)
        assert audio_duration == video_duration
        assert (first_gop == first_plain).all()

    def test_gop_context_survives_two_cadence_changes(self, clip, tmp_path):
        config = {
            "pipeline": ["fps_40", "fps_50"],
            "fps_40": {
                "processor": "videotoolbox", "profile": "normal",
                "target_fps": 40,
            },
            "fps_50": {
                "processor": "videotoolbox", "profile": "normal",
                "target_fps": 50,
            },
        }
        result = run_file(
            config, video=clip, output=tmp_path / "chained_frc.mp4",
            settings=SETTINGS, layout=Layout.CV_RGBA_HALF,
            start=4, end=5, gop_align=True)

        with av.open(str(result.path)) as container:
            video = container.streams.video[0]
            times = sorted(
                Fraction(frame.pts) * Fraction(video.time_base)
                for frame in container.decode(video=0))
            duration = Fraction(video.duration) * Fraction(video.time_base)
        assert result.frames_out == 2
        assert times == [Fraction(0), Fraction(1, 50)]
        assert duration == Fraction(1, 25)

    def test_snap_start_moves_to_the_nearest_keyframe(self, clip, tmp_path):
        # start=11 snaps to keyframe 8 -> the window becomes [8, 20).
        res = run_file(
            {"pipeline": []}, video=clip, output=tmp_path / "snap.mp4",
            settings=SETTINGS, start=11, end=20, snap_start=True)
        assert res.frames_out == 12

    def test_schedule_windows_a_recurrent_stage(self, clip, tmp_path):
        # The one-schedule-drives-all contract, end to end on real weights:
        # gop-align resets the recurrent net per keyframe window, so its
        # output DIFFERS from continuous-stream mode while the frame count
        # and timeline stay identical. Runs through the public
        # process_video_file (which caps + clears the MLX cache, keeping
        # this net-loading test from starving later writer sessions in the
        # same process) and so also covers the api-level gop threading.
        from kinovsr.api import process_video_file
        from kinovsr.processors.bsvd import default_weights_path

        if not default_weights_path().exists():
            pytest.skip("bsvd weights not available")
        cfg = {"pipeline": ["dn"],
               "dn": {"processor": "bsvd", "strength": 0.05}}
        gop = process_video_file(
            cfg, video=clip, output=tmp_path / "gop.mp4", settings=SETTINGS,
            start=12, end=20, gop_align=True, gop_min_window=4,
            gop_max_window=16)
        cont = process_video_file(
            cfg, video=clip, output=tmp_path / "cont.mp4", settings=SETTINGS,
            start=12, end=20)
        assert gop.frames_out == cont.frames_out == 8

        def frames(path):
            with av.open(str(path)) as container:
                return [f.to_ndarray(format="rgb24")
                        for f in container.decode(video=0)]
        a, b = frames(gop.post_path), frames(cont.post_path)
        total = sum(
            abs(x.astype("f4") - y.astype("f4")).mean()
            for x, y in zip(a, b, strict=True))
        assert total > 0.5   # windowed state != continuous state


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


def test_interpolation_noninteger_ratio_stays_in_sync(
        clip_with_audio, tmp_path):
    """A non-integer cadence rewrite (25 -> 40 fps) whose regenerated grid
    overshoots the source window by ~15ms on the tail: the final unit is
    clamped so the output duration equals the source and muxed audio stays
    in sync (finding #2). The sibling 2x test cannot catch this - an exact
    ratio never overshoots, so its output already lands on the boundary."""
    config = {
        "pipeline": ["fps"],
        "fps": {"processor": "videotoolbox", "profile": "normal",
                "target_fps": 40},
    }
    result = run_file(
        config, video=clip_with_audio, output=tmp_path / "interp40.mp4",
        settings=SETTINGS, layout=Layout.CV_RGBA_HALF, audio=True)
    source_seconds = N / FPS   # 0.96s
    video_s, audio_s, _, _ = _stream_seconds(result.path)
    # Tight bound: without the clamp the tail rounds ~15ms past the source.
    assert abs(video_s - source_seconds) < 0.008
    assert audio_s is not None
    assert abs(audio_s - video_s) < 0.03


def test_cap_equal_to_natural_count_still_syncs_audio(
        clip_with_audio, tmp_path):
    """max_output_frames set to the non-integer rewrite's natural output
    count (25->40 fps emits 39) must still clamp the final frame - the
    earlier hit_cap skip left this case drifting (re-review #2)."""
    config = {
        "pipeline": ["fps"],
        "fps": {"processor": "videotoolbox", "profile": "normal",
                "target_fps": 40},
    }
    result = run_file(
        config, video=clip_with_audio, output=tmp_path / "capnat.mp4",
        settings=SETTINGS, layout=Layout.CV_RGBA_HALF, audio=True,
        max_output_frames=39)
    source_seconds = N / FPS   # 0.96s
    video_s, audio_s, frames, _ = _stream_seconds(result.path)
    assert frames == 39
    assert abs(video_s - source_seconds) < 0.008
    assert audio_s is not None
    assert abs(audio_s - video_s) < 0.03


def test_max_output_frames_caps_the_interpolated_stream(clip, tmp_path):
    """--max-frames semantics: the cap counts OUTPUT frames, so a
    cadence-doubling chain stops at the cap instead of emitting
    2x the capped input."""
    config = {
        "pipeline": ["fps"],
        "fps": {"processor": "videotoolbox", "profile": "normal",
                "target_fps": 50},
    }
    result = run_file(
        {"pipeline": []} | config, video=clip, output=tmp_path / "cap.mp4",
        settings=SETTINGS, layout=Layout.CV_RGBA_HALF,
        max_output_frames=10)
    assert result.frames_out == 10
    _, _, frames, _ = _stream_seconds(result.path)
    assert frames == 10


def test_time_form_cap_resolves_against_the_output_cadence(clip, tmp_path):
    """A seconds cap means OUTPUT duration: 0.2s of 50fps output is 10
    frames, not 0.2s of the 25fps input (which would be 5)."""
    config = {
        "pipeline": ["fps"],
        "fps": {"processor": "videotoolbox", "profile": "normal",
                "target_fps": 50},
    }
    result = run_file(
        config, video=clip, output=tmp_path / "tcap.mp4",
        settings=SETTINGS, layout=Layout.CV_RGBA_HALF,
        max_output_seconds=0.2)
    assert result.frames_out == 10


def test_capped_run_trims_the_audio_carry(clip_with_audio, tmp_path):
    """Capped video must not ship beside full-window audio."""
    result = run_file(
        {"pipeline": []}, video=clip_with_audio,
        output=tmp_path / "acap.mp4", settings=SETTINGS,
        audio=True, max_output_frames=10)
    assert result.frames_out == 10
    video_s, audio_s, frames, _ = _stream_seconds(result.path)
    assert frames == 10
    assert audio_s is not None
    assert abs(audio_s - video_s) < 0.06   # AAC priming skew only


def test_zero_output_cap_is_rejected(clip, tmp_path):
    with pytest.raises(MediaError, match="at least one frame"):
        run_file({"pipeline": []}, video=clip,
                 output=tmp_path / "zero.mp4", settings=SETTINGS,
                 max_output_frames=0)
    assert not (tmp_path / "zero.mp4").exists()


def test_cut_log_alias_is_rejected_before_source_mutation(clip, tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(clip.read_bytes())
    original = source.read_bytes()
    output = tmp_path / "out.mp4"

    with pytest.raises(MediaError, match="destroy the source"):
        run_file(
            {"pipeline": []},
            video=source,
            output=output,
            settings=SETTINGS,
            cut_log=source,
        )

    assert source.read_bytes() == original
    assert not output.exists()
    assert not list(tmp_path.glob(".*.partial"))


def test_comparison_cannot_alias_post_output(clip, tmp_path):
    output = tmp_path / "out.mp4"
    with pytest.raises(MediaError, match="artifact paths alias"):
        run_file(
            {"pipeline": []},
            video=clip,
            output=output,
            settings=SETTINGS,
            comparison=output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".*.partial"))


def test_second_writer_publish_failure_leaves_no_singleton(
        clip, tmp_path, monkeypatch):
    from kinovsr.pipeline.run import _OutputTransaction

    output = tmp_path / "out.mp4"
    comparison = tmp_path / "comparison.mp4"
    original_replace = _OutputTransaction._replace

    def fail_comparison(source, destination):
        if destination == comparison:
            raise OSError("injected comparison publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(
        _OutputTransaction, "_replace", staticmethod(fail_comparison))
    with pytest.raises(MediaError, match="comparison publication failure"):
        run_file(
            {"pipeline": []},
            video=clip,
            output=output,
            settings=SETTINGS,
            comparison=comparison,
        )

    assert not output.exists()
    assert not comparison.exists()
    assert not list(tmp_path.glob(".*.partial"))


def test_failed_run_preserves_existing_output(clip, tmp_path):
    """Finding #3 atomicity: a run that fails after the writer opened (the
    weights load at the first pull, past open-time validation) must leave a
    pre-existing output untouched and drop its partial temp file."""
    out = tmp_path / "keep.mp4"
    run_file({"pipeline": []}, video=clip, output=out, settings=SETTINGS)
    original = out.read_bytes()
    assert original

    # realesrgan with a bogus explicit weights path passes open (explicit
    # path + scale) but fails when the checkpoint loads at the first pull.
    config = {"pipeline": ["up"],
              "up": {"processor": "realesrgan",
                     "weights": str(tmp_path / "nope.safetensors"),
                     "scale": 4}}
    with pytest.raises(Exception):  # noqa: B017 - loader error, wrapped
        run_file(config, video=clip, output=out, settings=SETTINGS,
                 max_frames=4, overwrite=True)
    assert out.read_bytes() == original            # original intact
    assert not list(tmp_path.glob(".keep.mp4.*"))  # no partial temp left


def test_rename_failure_leaves_no_orphan_temp(clip, tmp_path):
    """A publish that can't land (the output path is an existing directory,
    so the atomic rename fails at finish) must still clean up the partial
    temp instead of orphaning it (re-review #3)."""
    outdir = tmp_path / "out.mp4"
    outdir.mkdir()
    with pytest.raises(Exception):  # noqa: B017 - IsADirectoryError/OSError
        run_file({"pipeline": []}, video=clip, output=outdir,
                 settings=SETTINGS)
    assert outdir.is_dir()                         # the directory is untouched
    assert not list(tmp_path.glob(".out.mp4.*"))   # no partial temp left


def test_discard_cancels_and_unlinks_even_if_cancel_is_interrupted(tmp_path):
    """discard() must cancel rather than finish and remain BaseException-safe."""
    from kinovsr.pipeline.run import FileSink

    temp = tmp_path / ".out.mp4.partial"
    temp.write_bytes(b"partial encode")
    sink = FileSink.__new__(FileSink)   # bypass the heavy native __init__
    sink._published = False
    sink._discarded = False
    sink._temp_path = temp

    class _InterruptingWriter:
        def cancel(self):
            raise KeyboardInterrupt("during-discard")

    sink.writer = _InterruptingWriter()
    sink.discard()                       # must not raise
    assert not temp.exists()             # temp cleaned despite the interrupt


_FBCNN_WEIGHTS = Path(
    "kinovsr/processors/fbcnn/weights/fbcnn_color.safetensors")
_NAFNET_WEIGHTS = Path(
    "kinovsr/processors/nafnet/weights/nafnet_gopro_width32.safetensors")


@pytest.mark.skipif(not _FBCNN_WEIGHTS.is_file(),
                    reason="fbcnn weights not installed")
def test_fbcnn_runs_through_the_typed_pipeline(clip, tmp_path):
    """The per-frame driver adapter: fbcnn speaks denoise(), and the
    factory must wrap it - the first pumped frame proves the protocol."""
    config = {
        "pipeline": ["db"],
        "db": {"processor": "fbcnn", "quality": "35", "strength": 0.5},
    }
    result = run_file(
        config, video=clip, output=tmp_path / "fbcnn.mp4",
        settings=SETTINGS, max_frames=3)
    assert result.frames_out == 3


@pytest.mark.skipif(not _NAFNET_WEIGHTS.is_file(),
                    reason="nafnet weights not installed")
def test_nafnet_runs_through_the_typed_pipeline(clip, tmp_path):
    config = {
        "pipeline": ["rs"],
        "rs": {"processor": "nafnet", "capability": "deblur",
               "profile": "gopro32", "strength": 0.5},
    }
    result = run_file(
        config, video=clip, output=tmp_path / "nafnet.mp4",
        settings=SETTINGS, max_frames=3)
    assert result.frames_out == 3


def test_ntsc_cadence_writes_the_exact_rational_grid(tmp_path):
    """The writer's index grid quantizes 30000/1001 to a fixed 801-tick
    frame duration (drifting ~0.2 ticks/frame); the sink must stamp each
    unit's own validated ticks. Constructed spec, no probe fuzz."""
    import mlx.core as mx

    from kinovsr.pipeline import FileSink
    from kinovsr.processors import (
        FrameUnit,
        Geometry,
        StreamSpec,
        TimelineSpec,
        frame_spec_for_matrix,
    )

    cadence = Fraction(30000, 1001)
    time_base = Fraction(1, 24000)
    spec = StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(W, H)),
        timeline=TimelineSpec(time_base=time_base, cadence=cadence))
    sink = FileSink(tmp_path / "ntsc.mp4", spec)
    assert sink._direct_mlx_encode
    assert sink._pool is None

    def ticks(i: int) -> int:
        return round(i / cadence / time_base)

    n = 24
    frame = mx.full((H, W, 3), 0.5, dtype=mx.float32)
    for i in range(n):
        sink.append(FrameUnit(payload=frame, pts=ticks(i),
                              duration=ticks(i + 1) - ticks(i)))
    sink.finalize()
    path = sink._temp_path

    half_tick = Fraction(1, 48000)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        times = sorted(
            Fraction(frame.pts) * Fraction(stream.time_base)
            for frame in container.decode(video=0))
    assert len(times) == n
    for i, t in enumerate(times):
        expected = Fraction(i) / cadence
        # the sink writes round(i/cadence*24000)/24000; allow that
        # rounding but not the index-grid quantization drift
        assert abs(t - expected) <= half_tick, (i, float(t), float(expected))


def test_source_less_full_range_mlx_sink_keeps_range_metadata(tmp_path):
    from dataclasses import replace

    import mlx.core as mx

    from kinovsr.pipeline import FileSink
    from kinovsr.processors import (
        ColorPrimaries,
        FrameUnit,
        Geometry,
        StreamSpec,
        TimelineSpec,
        TransferFunction,
        frame_spec_for_matrix,
    )

    spec = StreamSpec(
        frame=replace(
            frame_spec_for_matrix(
                "bt709", full_range=True, geometry=Geometry(W, H)),
            color_primaries=ColorPrimaries.BT2020,
            transfer_function=TransferFunction.BT2020,
        ),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))
    sink = FileSink(tmp_path / "full.mp4", spec)
    assert sink._direct_mlx_encode
    frame = mx.full((H, W, 3), 0.5, dtype=mx.float32)
    for i in range(3):
        sink.append(FrameUnit(
            payload=frame, pts=i * 960, duration=960))
    sink.finalize()
    path = sink._temp_path

    with av.open(str(path)) as container:
        codec = container.streams.video[0].codec_context
        assert int(codec.color_range) == 2
        assert int(codec.color_primaries) == 9
        assert int(codec.color_trc) == 1
        assert int(codec.colorspace) == 1


def test_source_less_mlx_sink_rejects_non_rgb_payload(tmp_path):
    import mlx.core as mx

    from kinovsr.pipeline import FileSink
    from kinovsr.processors import (
        FrameUnit,
        Geometry,
        StreamSpec,
        TimelineSpec,
        frame_spec_for_matrix,
    )

    spec = StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(W, H)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))
    sink = FileSink(tmp_path / "invalid.mp4", spec)
    try:
        with pytest.raises(MediaError, match=r"shape .* RGB spec"):
            sink.append(FrameUnit(
                payload=mx.zeros((H, W, 4)), pts=0, duration=960))
    finally:
        sink.discard()


def test_odd_output_dimension_is_rejected(tmp_path):
    from kinovsr.pipeline.run import FileSink
    from kinovsr.processors import (
        Geometry,
        StreamSpec,
        TimelineSpec,
        frame_spec_for_matrix,
    )

    # 4:2:0 subsamples both axes, so an odd height (or width) has no even luma
    # grid. The harness silently evened both dimensions; the sink now rejects
    # loudly (parity bug C7 - the guard checked only width before).
    odd = StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(100, 99)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))
    with pytest.raises(MediaError, match="odd dimension"):
        FileSink(tmp_path / "out.mp4", odd)


def test_sidecar_without_audio_is_rejected_before_any_io(tmp_path):
    with pytest.raises(MediaError, match="save_audio_sidecar requires audio"):
        run_file(
            {"pipeline": []},
            video=tmp_path / "missing.mov",
            output=tmp_path / "out.mp4",
            settings=SETTINGS,
            save_audio_sidecar=True,
        )
