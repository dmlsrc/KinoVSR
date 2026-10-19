"""AVWriter must not leak hardware-encoder session slots across writers.

AVFoundation autoreleases session-holding objects during a writer's
construction, appends, and finish. A long-lived host process without a
draining run loop (the API use case, this very test process) used to leak
them, and after ~32 writers the encoder fell back to a software capability
set without the 4:2:2 profile - AVAssetWriterInput creation then failed
with "video codec type hvc1 only allows ... HEVC_Main10_AutoLevel ...".
AVWriter now drains a pool per phase; this exercises well past the old cap.
"""

import contextlib
import io
import time

import pytest

from kinovsr.media import pixel_buffers as _pb
from kinovsr.native.frameworks import CoreMedia, Foundation, av
from kinovsr.native.writer import HEVC_PROFILE_MAIN422_10, AVWriter

pytestmark = pytest.mark.integration


def test_forty_writers_do_not_exhaust_encoder_sessions(tmp_path):
    quiet = io.StringIO()  # 40 setup lines are noise
    with contextlib.redirect_stdout(quiet):
        for i in range(40):
            writer = AVWriter(
                tmp_path / f"w{i}.mp4",
                width=160,
                height=128,
                fps=25.0,
                source_pixel_format=_pb.PIX_RGBAHALF,
                profile=HEVC_PROFILE_MAIN422_10,
                quality=0.65,
                label="probe",
            )
            buffer = _pb.make_pixel_buffer_from_attrs(
                160, 128, {"PixelFormatType": _pb.PIX_RGBAHALF, "IOSurfaceProperties": {}}
            )
            writer.append(buffer, pts_ticks=0, duration_ticks=960)
            writer.finish()
    assert (tmp_path / "w39.mp4").stat().st_size > 0


def _samples_on_disk(path):
    """Video samples AVFoundation reads from ``path`` now; 0 when it finds no video track.

    Passthrough reading also vends marker buffers that hold no sample, so
    samples are counted, not buffers.
    """
    asset = av.AVURLAsset.URLAssetWithURL_options_(
        Foundation.NSURL.fileURLWithPath_(str(path)), None
    )
    tracks = asset.tracksWithMediaType_(av.AVMediaTypeVideo)
    if not tracks:
        return 0
    reader, _ = av.AVAssetReader.alloc().initWithAsset_error_(asset, None)
    output = av.AVAssetReaderTrackOutput.assetReaderTrackOutputWithTrack_outputSettings_(
        tracks[0], None
    )
    reader.addOutput_(output)
    reader.startReading()
    count = 0
    while (sample_buffer := output.copyNextSampleBuffer()) is not None:
        count += CoreMedia.CMSampleBufferGetNumSamples(sample_buffer)
    return count


def test_a_fragmented_file_is_readable_before_it_is_closed(tmp_path):
    # A plain MPEG-4 has no index until it is closed, so a crash leaves it
    # unplayable; movie fragments carry the index along as the file grows.
    frames = 75
    buffer = _pb.make_pixel_buffer_from_attrs(
        160, 128, {"PixelFormatType": _pb.PIX_RGBAHALF, "IOSurfaceProperties": {}}
    )
    writers = {}
    for name, interval in (("fragmented", 0.5), ("plain", None)):
        writer = AVWriter(
            tmp_path / f"{name}.mp4",
            width=160,
            height=128,
            fps=25.0,
            source_pixel_format=_pb.PIX_RGBAHALF,
            profile=HEVC_PROFILE_MAIN422_10,
            quality=0.65,
            label="probe",
            fragment_interval=interval,
        )
        for index in range(frames):
            writer.append(buffer, pts_ticks=960 * index, duration_ticks=960)
        writers[name] = writer
    # The encoder runs behind the appends; give its fragments a bounded time.
    deadline = time.monotonic() + 10
    while (readable := _samples_on_disk(tmp_path / "fragmented.mp4")) == 0:
        assert time.monotonic() < deadline, "no fragment became readable"
        time.sleep(0.05)
    assert readable > 0
    assert _samples_on_disk(tmp_path / "plain.mp4") == 0
    for name, writer in writers.items():
        writer.finish()
        assert _samples_on_disk(tmp_path / f"{name}.mp4") == frames


def _probe_writer(path, **kwargs):
    return AVWriter(
        path,
        width=160,
        height=128,
        fps=25.0,
        source_pixel_format=_pb.PIX_RGBAHALF,
        profile=HEVC_PROFILE_MAIN422_10,
        quality=0.65,
        label="probe",
        fragment_interval=0.5,
        **kwargs,
    )


def _probe_buffer():
    return _pb.make_pixel_buffer_from_attrs(
        160, 128, {"PixelFormatType": _pb.PIX_RGBAHALF, "IOSurfaceProperties": {}}
    )


def test_cancel_keeps_a_file_with_frames(tmp_path):
    # AVFoundation's cancelWriting deletes the output file; frames written
    # before a failure are the run's work and must survive it.
    path = tmp_path / "out.partial.mp4"
    writer = _probe_writer(path)
    buffer = _probe_buffer()
    for index in range(75):
        writer.append(buffer, pts_ticks=960 * index, duration_ticks=960)
    deadline = time.monotonic() + 10
    while _samples_on_disk(path) == 0:
        assert time.monotonic() < deadline, "no fragment became readable"
        time.sleep(0.05)
    writer.cancel()
    assert writer._wait_for_cancel_cleanup(timeout=10)
    assert _samples_on_disk(path) > 0
    assert sorted(item.name for item in tmp_path.iterdir()) == ["out.partial.mp4"]


def test_cancel_deletes_a_file_without_frames(tmp_path):
    path = tmp_path / "out.partial.mp4"
    writer = _probe_writer(path)
    writer.cancel()
    assert writer._wait_for_cancel_cleanup(timeout=10)
    assert not list(tmp_path.iterdir())


def test_early_finish_stops_the_audio_with_the_video(tmp_path):
    # A normal finish drains the whole source track; an early one, for a run
    # that stopped, ends the audio where the video ends.
    import mlx.core as mx

    from kinovsr.media.audio import AudioTrack

    rate = 48_000
    track = AudioTrack(mx.zeros((2, rate * 30), dtype=mx.float32), rate)
    path = tmp_path / "out.partial.mp4"
    writer = _probe_writer(path, audio_track=track)
    buffer = _probe_buffer()
    for index in range(50):  # 2 s of video against 30 s of audio
        writer.append(buffer, pts_ticks=960 * index, duration_ticks=960)
    writer.finish(early=True)
    assert writer._audio_progress[0] < track.n_samples
    asset = av.AVURLAsset.URLAssetWithURL_options_(
        Foundation.NSURL.fileURLWithPath_(str(path)), None
    )
    audio = asset.tracksWithMediaType_(av.AVMediaTypeAudio)[0]
    video = asset.tracksWithMediaType_(av.AVMediaTypeVideo)[0]
    video_seconds = CoreMedia.CMTimeGetSeconds(video.timeRange().duration)
    assert abs(video_seconds - 2.0) < 1e-6
    assert CoreMedia.CMTimeGetSeconds(audio.timeRange().duration) <= video_seconds + 1e-3
    assert _samples_on_disk(path) == 50
