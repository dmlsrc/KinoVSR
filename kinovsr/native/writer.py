"""AVAssetWriter wrapper: HEVC video + optional ALAC/AAC audio, no ffmpeg.

The writer takes a stream of CVPixelBuffers (typically straight from
VsrSession's adaptor pool - zero-copy from VSR output to encoder) and
encodes them as HEVC Main10 4:2:0 or Main42210 4:2:2 10-bit BT.709 at the
target fps. Audio (if attached) is pulled by AVAssetWriter on a dedicated
dispatch queue via requestMediaDataWhenReadyOnQueue:, so the audio encode
doesn't stall the video append loop.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from pathlib import Path
from typing import Any

from kinovsr.media import pixel_buffers as _pb
from kinovsr.media import yuv as _yuv
from kinovsr.media.audio import AudioTrack, audio_writer_settings

from .frameworks import (
    CoreMedia,
    Foundation,
    Quartz,
    autorelease_pool,
    av,
    libdispatch,
)

_log = logging.getLogger(__name__)

# HEVC profile identifiers (Apple-stable strings; not exposed as PyObjC consts)
HEVC_PROFILE_MAIN10 = "HEVC_Main10_AutoLevel"          # 4:2:0 10-bit
HEVC_PROFILE_MAIN422_10 = "HEVC_Main42210_AutoLevel"   # 4:2:2 10-bit (Range Extensions)


def hevc_video_settings(
    width: int, height: int, quality: float, profile: str,
    color_props: dict | None = None,
    pixel_aspect: tuple[int, int] | None = None,
) -> dict:
    """AVAssetWriterInput output settings for HEVC at the given size + profile.

    ``color_props`` is an AVVideoColorPropertiesKey dict (primaries/transfer/
    matrix) tagging the output to match the source; defaults to BT.709.
    ``pixel_aspect`` preserves anamorphic source display geometry while the
    encoded raster stays in model/VSR pixel coordinates.
    """
    settings = {
        av.AVVideoCodecKey: av.AVVideoCodecTypeHEVC,
        av.AVVideoWidthKey: width,
        av.AVVideoHeightKey: height,
        av.AVVideoColorPropertiesKey: color_props or {
            av.AVVideoColorPrimariesKey: av.AVVideoColorPrimaries_ITU_R_709_2,
            av.AVVideoTransferFunctionKey: av.AVVideoTransferFunction_ITU_R_709_2,
            av.AVVideoYCbCrMatrixKey: av.AVVideoYCbCrMatrix_ITU_R_709_2,
        },
        av.AVVideoCompressionPropertiesKey: {
            av.AVVideoProfileLevelKey: profile,
            av.AVVideoQualityKey: quality,
        },
    }
    if pixel_aspect is not None:
        h, v = pixel_aspect
        settings[av.AVVideoPixelAspectRatioKey] = {
            av.AVVideoPixelAspectRatioHorizontalSpacingKey: int(h),
            av.AVVideoPixelAspectRatioVerticalSpacingKey: int(v),
        }
    return settings


def _color_label(color_props: dict | None) -> str:
    """Short color name for the setup log, from an AVVideoColorProperties dict."""
    if not color_props:
        return "BT.709"
    prim = str(color_props.get(av.AVVideoColorPrimariesKey, ""))
    if "2020" in prim:
        return "BT.2020"
    if "SMPTE_C" in prim:
        return "BT.601"
    if "P3" in prim:
        return "P3"
    return "BT.709"


class AVWriter:
    """AVAssetWriter wrapping a HEVC video input + optional audio input.

    Construction kicks off `startWriting` + `startSessionAtSourceTime`. If
    `audio_track` is supplied, an audio AVAssetWriterInput is added and a
    GCD callback is scheduled to pull samples from the track as the encoder
    consumes them.

    Per-frame API:
        writer.append(pb)           # pixel buffer in the configured source format
    Finalize:
        writer.finish()             # waits for audio drain + finishWriting
    """

    READY_TIMEOUT_S = 30.0
    AUDIO_TIMEOUT_S = 120.0
    FINISH_TIMEOUT_S = 120.0

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        *,
        source_pixel_format: int,
        profile: str = HEVC_PROFILE_MAIN10,
        quality: float = 0.65,
        label: str = "video",
        audio_track: AudioTrack | None = None,
        audio_codec: str = "alac",
        transform: Any = None,
        source_attrs: dict | None = None,
        color_props: dict | None = None,
        pixel_aspect: tuple[int, int] | None = None,
        cv_color: tuple | None = None,
        full_range: bool = False,
    ):
        self._state_lock = threading.RLock()
        self._state = "constructing"
        self._failure: BaseException | None = None
        self._finish_done = threading.Event()
        self._native_cancelled = False
        # Every AVFoundation-touching phase of a writer's life (construction,
        # append, finish) leaves autoreleased objects behind that hold the
        # hardware-encoder session. A long-lived host process without a
        # draining run loop (the API use case, pytest, the typed pipeline)
        # never releases them, and after ~32 writers the encoder falls back
        # to a software capability set without the 4:2:2 profile - writer
        # creation then fails. Each phase drains its own pool.
        try:
            with autorelease_pool():
                self._construct(
                    output_path, width, height, fps,
                    source_pixel_format=source_pixel_format, profile=profile,
                    quality=quality, label=label, audio_track=audio_track,
                    audio_codec=audio_codec, transform=transform,
                    source_attrs=source_attrs, color_props=color_props,
                    pixel_aspect=pixel_aspect, cv_color=cv_color,
                    full_range=full_range)
            with self._state_lock:
                if self._failure is not None:
                    raise self._failure
                self._state = "writing"
        except BaseException as exc:
            self._record_failure(exc)
            self.cancel()
            raise

    def _construct(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        *,
        source_pixel_format: int,
        profile: str,
        quality: float,
        label: str,
        audio_track: AudioTrack | None,
        audio_codec: str,
        transform: Any,
        source_attrs: dict | None,
        color_props: dict | None,
        pixel_aspect: tuple[int, int] | None,
        cv_color: tuple | None,
        full_range: bool,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        url = Foundation.NSURL.fileURLWithPath_(str(output_path))
        writer, err = av.AVAssetWriter.alloc().initWithURL_fileType_error_(
            url, av.AVFileTypeMPEG4, None,
        )
        if writer is None:
            raise RuntimeError(f"AVAssetWriter init failed for {output_path}: {err}")

        # Video input + pixel buffer adaptor ---------------------------------
        video_input = av.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
            av.AVMediaTypeVideo,
            hevc_video_settings(width, height, quality, profile, color_props, pixel_aspect),
        )
        video_input.setExpectsMediaDataInRealTime_(False)
        # Pin the track timescale to the product's tick base. The default
        # movie scale (600) quantizes CMTime stamps: integer rates divide
        # 600 and survive, but NTSC-family ticks (e.g. 801/24000) snap to
        # 20/600 and re-introduce exactly the index-grid drift the
        # explicit-PTS path exists to avoid.
        video_input.setMediaTimeScale_(_pb.VIDEO_TIME_SCALE)
        # Carry the source track's rotation/flip as output metadata. The pixels
        # stay in stored orientation through VSR and the encoder; the container
        # transform makes players display them upright - lossless, no rotate.
        if transform is not None:
            video_input.setTransform_(transform)

        # Feed the encoder YUV we converted ourselves (see yuv.py) rather than
        # RGBAHalf: AVAssetWriter's internal RGB->YUV is colorspace-metadata-
        # dependent and color-shifts uploaded buffers. cv_color present => YUV feed,
        # and `append` converts the RGBAHalf input to 10-bit 4:2:2 with that matrix.
        # Only the RGBAHalf producers (balanced/image + the learned/upload upscalers)
        # need this; NV12 (fast) output is already YUV and feeds through untouched.
        self._yuv_feed = cv_color is not None and source_pixel_format == _pb.PIX_RGBAHALF
        self._yuv_matrix = cv_color[2] if cv_color is not None else None
        self._yuv_full = full_range
        adaptor_format = _yuv.pixel_format(full_range) if self._yuv_feed else source_pixel_format
        src_attrs = {
            Quartz.kCVPixelBufferPixelFormatTypeKey: adaptor_format,
            Quartz.kCVPixelBufferWidthKey: width,
            Quartz.kCVPixelBufferHeightKey: height,
            Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
        }
        # Carry over the extended-pixel padding the producer (VSR / temporal
        # session) requires of the buffers it writes into - its dst attrs. When
        # this pool feeds VSR's output directly (use_dst_pool, zero-copy), an
        # unpadded buffer makes VTFrameProcessor fail -19730 for output
        # geometries that need a padded destination (e.g. a 1088x816 output
        # wants 16 extended bottom rows). The encoder still reads the clean
        # width x height region, so the padding is transparent to it.
        if source_attrs is not None:
            for k in (
                Quartz.kCVPixelBufferExtendedPixelsLeftKey,
                Quartz.kCVPixelBufferExtendedPixelsRightKey,
                Quartz.kCVPixelBufferExtendedPixelsTopKey,
                Quartz.kCVPixelBufferExtendedPixelsBottomKey,
            ):
                if k in source_attrs:
                    src_attrs[k] = source_attrs[k]
        adaptor = av.AVAssetWriterInputPixelBufferAdaptor.assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes_(
            video_input, src_attrs,
        )
        if not writer.canAddInput_(video_input):
            raise RuntimeError(f"AVAssetWriter cannot add video input for {output_path}")
        writer.addInput_(video_input)

        # Optional audio input -----------------------------------------------
        audio_input = None
        if audio_track is not None:
            audio_input = av.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
                av.AVMediaTypeAudio,
                audio_writer_settings(audio_codec, audio_track.sample_rate, audio_track.channels),
            )
            audio_input.setExpectsMediaDataInRealTime_(False)
            if not writer.canAddInput_(audio_input):
                raise RuntimeError(
                    f"AVAssetWriter cannot add audio input ({audio_codec}) for {output_path}"
                )
            writer.addInput_(audio_input)

        # Start the writer ---------------------------------------------------
        writer.setMovieTimeScale_(_pb.VIDEO_TIME_SCALE)
        if not writer.startWriting():
            raise RuntimeError(f"AVAssetWriter.startWriting failed: {writer.error()}")
        writer.startSessionAtSourceTime_(CoreMedia.CMTimeMake(0, _pb.VIDEO_TIME_SCALE))

        self.writer = writer
        self.video_input = video_input
        self.audio_input = audio_input
        self.adaptor = adaptor
        self.fps = float(fps)
        self.label = label
        self.path = output_path
        self.frame_count = 0
        self._explicit_end_ticks: int | None = None
        self.audio_track = audio_track

        audio_desc = f", audio={audio_codec}" if audio_input is not None else ""
        _log.info(
            "[%s] AVAssetWriter -> %s (HEVC %s %s q=%s%s)",
            label,
            output_path,
            profile,
            _color_label(color_props),
            quality,
            audio_desc,
        )

        # Audio pump (GCD pull pattern) --------------------------------------
        self._audio_done = threading.Event()
        self._audio_progress = [0]
        if audio_track is not None:
            self._audio_queue = libdispatch.dispatch_queue_create(
                f"kinovsr.audio.{label}".encode(), None,
            )
            n_samples = audio_track.n_samples
            chunk_frames = max(4096, audio_track.sample_rate // 4)  # ~250 ms

            self.audio_input.requestMediaDataWhenReadyOnQueue_usingBlock_(
                self._audio_queue,
                lambda: self._pump_audio(n_samples, chunk_frames),
            )
        else:
            self._audio_done.set()
            self._audio_queue = None

    # ------------------------------------------------------------------------
    # Internal: wait-with-status-check
    # ------------------------------------------------------------------------

    def _record_failure(self, exc: BaseException) -> None:
        """Retain the first causal failure for the synchronous caller."""
        with self._state_lock:
            if self._failure is None:
                self._failure = exc
            if self._state not in ("finished", "cancelled"):
                self._state = "failed"

    def _raise_if_failed(self) -> None:
        with self._state_lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def _accepting_callbacks(self) -> bool:
        with self._state_lock:
            return self._state in ("constructing", "writing", "finishing")

    def _pump_audio(self, n_samples: int, chunk_frames: int) -> None:
        """Drain ready audio without letting callback failures disappear.

        GCD cannot propagate a Python exception to ``finish``. Store the first
        one, signal the waiter, and let the synchronous writer boundary raise
        it with its original position and native status.
        """
        try:
            while (self._accepting_callbacks()
                   and self.audio_input.isReadyForMoreMediaData()):
                pos = self._audio_progress[0]
                if pos >= n_samples:
                    self.audio_input.markAsFinished()
                    return
                end = min(pos + chunk_frames, n_samples)
                sb = self.audio_track.make_sample_buffer(pos, end)
                if sb is None or not self.audio_input.appendSampleBuffer_(sb):
                    raise RuntimeError(
                        f"[{self.label}] audio appendSampleBuffer failed at "
                        f"{pos}: status={self.writer.status()} "
                        f"error={self.writer.error()}")
                self._audio_progress[0] = end
        except BaseException as exc:  # callback failures cross via shared state
            self._record_failure(exc)
        finally:
            # A callback can be invoked again while more data becomes ready.
            # Signal only on completion, failure, or cancellation.
            if (self._audio_progress[0] >= n_samples
                    or not self._accepting_callbacks()
                    or self._failure is not None):
                self._audio_done.set()

    def _check_native_status(self, what: str) -> None:
        self._raise_if_failed()
        status = self.writer.status()
        if status in (3, 4):  # AVAssetWriterStatusFailed/Cancelled
            exc = RuntimeError(
                f"[{self.label}] writer entered status={status} during "
                f"{what}: {self.writer.error()}")
            self._record_failure(exc)
            raise exc

    def _wait_for_ready(self, input_obj: Any, what: str) -> None:
        """Block until input_obj.isReadyForMoreMediaData(). Bail with a clean
        error if the writer enters Failed/Cancelled, or after 30 s of no
        progress (so a stuck writer surfaces as a visible failure, not a hang).
        """
        deadline = time.monotonic() + self.READY_TIMEOUT_S
        while not input_obj.isReadyForMoreMediaData():
            self._check_native_status(f"waiting on {what}")
            time.sleep(0.001)
            if time.monotonic() >= deadline:
                exc = RuntimeError(
                    f"[{self.label}] {what} input never became ready "
                    f"(waited {self.READY_TIMEOUT_S:g}s, "
                    f"status={self.writer.status()})")
                self._record_failure(exc)
                raise exc
        self._check_native_status(f"waiting on {what}")

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def append(self, pb: Any, *, pts_ticks: int | None = None,
               duration_ticks: int | None = None) -> None:
        """Append one video frame.

        Default: the next index-grid PTS (frame_count/fps). A caller
        that owns an exact timeline (the typed pipeline's endpoints)
        passes ``pts_ticks``/``duration_ticks`` in VIDEO_TIME_SCALE
        instead - the index grid quantizes NTSC-family rates to a fixed
        per-frame duration, which drifts ~3.6 s/hour at 59.94 fps
        against the exact rational grid."""
        with self._state_lock:
            if self._state != "writing":
                self._raise_if_failed()
                raise RuntimeError(
                    f"[{self.label}] cannot append while writer is "
                    f"{self._state}")
        try:
            with autorelease_pool():
                self._wait_for_ready(self.video_input, "video")
                if self._yuv_feed:
                    rgb = _pb.read_rgbahalf_rgb(pb)
                    ybuf = _pb.pool_create_buffer(self.adaptor.pixelBufferPool())
                    if ybuf is None:
                        raise RuntimeError(
                            f"[{self.label}] YUV pool buffer allocation failed")
                    _yuv.rgb_to_yuv422_10(
                        rgb, ybuf, self._yuv_matrix, self._yuv_full)
                    pb = ybuf
                if pts_ticks is None:
                    pts = _pb.frame_pts(self.frame_count, self.fps)
                else:
                    pts = CoreMedia.CMTimeMake(
                        int(pts_ticks), _pb.VIDEO_TIME_SCALE)
                    if duration_ticks is not None:
                        self._explicit_end_ticks = (
                            int(pts_ticks) + int(duration_ticks))
                if not self.adaptor.appendPixelBuffer_withPresentationTime_(
                        pb, pts):
                    raise RuntimeError(
                        f"[{self.label}] appendPixelBuffer failed at frame "
                        f"{self.frame_count}: status={self.writer.status()} "
                        f"error={self.writer.error()}")
                self.frame_count += 1
        except Exception as exc:
            self._record_failure(exc)
            raise

    def finish(self) -> None:
        """Finalize once, with bounded audio and native completion waits."""
        failure_to_raise: BaseException | None = None
        with self._state_lock:
            if self._state == "finished":
                return
            if self._failure is not None:
                failure_to_raise = self._failure
                wait_for_other = False
            elif self._state == "finishing":
                wait_for_other = True
            else:
                if self._state == "cancelled":
                    raise RuntimeError(f"[{self.label}] writer was cancelled")
                self._state = "finishing"
                wait_for_other = False

        if failure_to_raise is not None:
            self.cancel()
            raise failure_to_raise

        if wait_for_other:
            if not self._finish_done.wait(timeout=self.FINISH_TIMEOUT_S):
                exc = RuntimeError(
                    f"[{self.label}] concurrent finish did not complete within "
                    f"{self.FINISH_TIMEOUT_S:g}s")
                self._record_failure(exc)
                self.cancel()
                raise exc
            with self._state_lock:
                if self._state == "finished":
                    return
            self._raise_if_failed()
            raise RuntimeError(f"[{self.label}] writer was cancelled")

        try:
            with autorelease_pool():
                self.video_input.markAsFinished()
                if (self.audio_input is not None
                        and not self._audio_done.wait(
                            timeout=self.AUDIO_TIMEOUT_S)):
                    raise RuntimeError(
                        f"[{self.label}] audio pump didn't finish within "
                        f"{self.AUDIO_TIMEOUT_S:g}s (progress="
                        f"{self._audio_progress[0]}/"
                        f"{self.audio_track.n_samples})")
                self._check_native_status("audio drain")
                if self._explicit_end_ticks is not None:
                    end = CoreMedia.CMTimeMake(
                        self._explicit_end_ticks, _pb.VIDEO_TIME_SCALE)
                else:
                    end = _pb.frame_pts(self.frame_count, self.fps)
                self.writer.endSessionAtSourceTime_(end)
                native_done = threading.Event()
                with self._state_lock:
                    self._native_finish_done = native_done
                self.writer.finishWritingWithCompletionHandler_(
                    lambda: native_done.set())
                if not native_done.wait(timeout=self.FINISH_TIMEOUT_S):
                    raise RuntimeError(
                        f"[{self.label}] AVAssetWriter finish callback did not "
                        f"arrive within {self.FINISH_TIMEOUT_S:g}s "
                        f"(status={self.writer.status()})")
                self._check_native_status("finish")
                if self.writer.status() != 2:
                    raise RuntimeError(
                        f"[{self.label}] AVAssetWriter finished with status "
                        f"{self.writer.status()}: {self.writer.error()}")
            with self._state_lock:
                if self._state == "cancelled":
                    raise RuntimeError(f"[{self.label}] writer was cancelled")
                self._state = "finished"
        except BaseException as exc:
            self._record_failure(exc)
            self.cancel()
            raise
        finally:
            with self._state_lock:
                self._native_finish_done = None
            self._finish_done.set()

    def cancel(self) -> None:
        """Stop callbacks and cancel native writing; safe to call repeatedly."""
        with self._state_lock:
            if self._state in ("finished", "cancelled"):
                return
            self._state = "cancelled"
            do_native_cancel = not self._native_cancelled
            self._native_cancelled = True

        audio_input = getattr(self, "audio_input", None)
        if audio_input is not None:
            with contextlib.suppress(BaseException):
                audio_input.stopRequestingMediaData()
        audio_done = getattr(self, "_audio_done", None)
        if audio_done is not None:
            audio_done.set()
        writer = getattr(self, "writer", None)
        if do_native_cancel and writer is not None:
            with contextlib.suppress(BaseException):
                writer.cancelWriting()
        native_finish_done = getattr(self, "_native_finish_done", None)
        if native_finish_done is not None:
            native_finish_done.set()
        self._finish_done.set()
