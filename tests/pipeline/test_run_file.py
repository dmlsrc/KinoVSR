"""File endpoints: probe-to-spec, unit grids, mux policy, and the
file-to-file runs that close M3's deferred acceptance.

The interpolation case is the audio-synchronization file proof: duration
preservation was proven in-memory in M3; here the same chain runs
against a real container with a real audio track and the output file's
video and audio timelines must agree.
"""

import math
from pathlib import Path

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

    def test_audio_carry_trims_to_the_window(self, clip_with_audio):
        source = FileSource(clip_with_audio, start=4, max_frames=8)
        track = source.audio_track()
        expected = round(8 / FPS * SAMPLE_RATE)
        assert abs(track.n_samples - expected) <= 2

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
    # An RGB/MLX chain (no upscale) reaches the encoder as RGBAHalf, so auto
    # is 4:2:2; forcing 420 must reach 4:2:0 even here (the harness parity gap).
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
                 max_frames=4)
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


def test_discard_unlinks_temp_even_if_finalize_is_interrupted(tmp_path):
    """discard() must remove its partial temp even if writer finalization
    raises a BaseException (interrupt), and must not itself raise - so it
    cannot mask the original failure the caller re-raises (re-review #3)."""
    from kinovsr.pipeline.run import FileSink

    temp = tmp_path / ".out.mp4.partial"
    temp.write_bytes(b"partial encode")
    sink = FileSink.__new__(FileSink)   # bypass the heavy native __init__
    sink._published = False
    sink._temp_path = temp

    class _InterruptingWriter:
        def finish(self):
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

    def ticks(i: int) -> int:
        return round(i / cadence / time_base)

    n = 24
    frame = mx.full((H, W, 3), 0.5, dtype=mx.float32)
    for i in range(n):
        sink.append(FrameUnit(payload=frame, pts=ticks(i),
                              duration=ticks(i + 1) - ticks(i)))
    path = sink.finish()

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
