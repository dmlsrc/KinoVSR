"""AVAssetWriter wrapper: HEVC video + optional ALAC/AAC audio, no ffmpeg.

The writer takes CVPixelBuffers (typically straight from VsrSession's adaptor
pool) or typed MLX RGB frames. MLX input converts directly into the adaptor's
pooled YUV surface; it does not stage through RGBAHalf. Frames encode as HEVC
Main10 4:2:0 or Main42210 4:2:2 10-bit with source-derived color tags. Audio
(if attached) is pulled by AVAssetWriter on a dedicated dispatch queue via
requestMediaDataWhenReadyOnQueue:, so the audio encode doesn't stall the video
append loop.
"""

from __future__ import annotations

import logging
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from kinovsr.media import pixel_buffers as _pb
from kinovsr.media import yuv as _yuv
from kinovsr.media.audio import AudioTrack, audio_writer_settings
from kinovsr.media.timing import rational_cadence

from .frameworks import (
    CoreMedia,
    Foundation,
    Quartz,
    autorelease_pool,
    av,
    libdispatch,
)

_log = logging.getLogger(__name__)

_MAX_CLEANUP_LAUNCH_ROUNDS = 2

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


class _AsyncAttempt:
    """One immutable-result background cleanup generation."""

    __slots__ = (
        "claim_lock",
        "claimed",
        "done",
        "failure",
        "fallback_scheduled",
        "ambiguous_launch",
        "launch_failed",
        "launch_rounds",
        "runner_lock",
        "thread",
        "thread_start_returned",
        "threads",
    )

    def __init__(self) -> None:
        self.claim_lock = threading.Lock()
        self.claimed = False
        self.done = threading.Event()
        self.failure: BaseException | None = None
        self.fallback_scheduled = False
        self.ambiguous_launch = False
        self.launch_failed = False
        self.launch_rounds: dict[tuple[int, int], int] = {}
        self.runner_lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.thread_start_returned = False
        self.threads: list[threading.Thread] = []


def _merge_attempt_failure(
    generation: _AsyncAttempt,
    failure: BaseException,
    what: str,
) -> None:
    if generation.failure is None:
        generation.failure = failure
    elif generation.failure is not failure:
        generation.failure.add_note(f"additional {what}: {failure!r}")


def _publish_attempt(
    generation: _AsyncAttempt,
    wrapper_failure: BaseException | None,
) -> None:
    """Freeze a cleanup result after all native pool teardown has run."""
    if wrapper_failure is not None:
        _merge_attempt_failure(
            generation,
            wrapper_failure,
            "worker-wrapper failure",
        )
    try:
        generation.done.set()
    except BaseException as publication_failure:
        _merge_attempt_failure(
            generation,
            publication_failure,
            "completion-publication failure",
        )
        # The result remains immutable. A second best-effort signal handles
        # one-shot interruption; a waiter can recover after the owner exits if
        # publication itself remains unavailable.
        try:
            generation.done.set()
        except BaseException as retry_failure:
            generation.failure.add_note(
                f"completion publication retry failed: {retry_failure!r}")


def _run_claimed_cleanup(
    generation: _AsyncAttempt,
    operation: Any,
    args: tuple[Any, ...],
    claim_failure: BaseException | None,
) -> None:
    """Run an owned cleanup and publish only after its native pool drains."""
    wrapper_failure = claim_failure
    try:
        try:
            pool = autorelease_pool()
            pool.__enter__()
        except BaseException as entry_failure:
            # Infrastructure failure happened before the operation. Keep it
            # observable, but make one fresh-pool attempt so a failed
            # constructor does not abandon resources with no owner.
            if wrapper_failure is None:
                wrapper_failure = entry_failure
            else:
                wrapper_failure.add_note(
                    f"autorelease-pool entry failed: {entry_failure!r}")
            try:
                retry_pool = autorelease_pool()
                retry_pool.__enter__()
            except BaseException as retry_entry_failure:
                wrapper_failure.add_note(
                    f"second autorelease-pool entry failed: "
                    f"{retry_entry_failure!r}")
                # No asynchronous mechanism can manufacture a functioning
                # Objective-C pool. Run cleanup once without one as the
                # last-resort ownership path, while still publishing the pool
                # failure to the synchronous observer.
                try:
                    operation(*args)
                except BaseException as unpooled_failure:
                    wrapper_failure.add_note(
                        f"unpooled cleanup fallback failed: "
                        f"{unpooled_failure!r}")
            else:
                try:
                    operation(*args)
                except BaseException as retry_failure:
                    wrapper_failure.add_note(
                        f"cleanup after pool-entry failure failed: "
                        f"{retry_failure!r}")
                finally:
                    try:
                        retry_pool.__exit__(None, None, None)
                    except BaseException as retry_exit_failure:
                        wrapper_failure.add_note(
                            f"retry autorelease-pool exit failed: "
                            f"{retry_exit_failure!r}")
        else:
            try:
                operation(*args)
            except BaseException as operation_failure:
                if wrapper_failure is None:
                    wrapper_failure = operation_failure
                else:
                    wrapper_failure.add_note(
                        f"cleanup operation failed: {operation_failure!r}")
            finally:
                try:
                    pool.__exit__(None, None, None)
                except BaseException as exit_failure:
                    if wrapper_failure is None:
                        wrapper_failure = exit_failure
                    else:
                        wrapper_failure.add_note(
                            f"autorelease-pool exit failed: {exit_failure!r}")
    except BaseException as boundary_failure:
        if wrapper_failure is None:
            wrapper_failure = boundary_failure
        elif wrapper_failure is not boundary_failure:
            wrapper_failure.add_note(
                f"additional worker-boundary failure: {boundary_failure!r}")
    finally:
        _publish_attempt(generation, wrapper_failure)


def _run_with_autorelease_pool(
    generation: _AsyncAttempt,
    operation: Any,
    *args: Any,
) -> None:
    """Claim one generation and run cleanup from the claim guard itself."""
    # If Thread.start is interrupted in its launch-before-metadata window, a
    # libdispatch fallback races the possible Python thread through this lock.
    # Exactly one candidate claims the generation and touches native state.
    with generation.runner_lock:
        owns_generation = False
        claim_failure: BaseException | None = None
        try:
            try:
                with generation.claim_lock:
                    if generation.done.is_set() or generation.claimed:
                        return
                    owns_generation = True
                    generation.claimed = True
            except BaseException as failure:
                if generation.done.is_set():
                    return
                if owns_generation:
                    generation.claimed = True
                elif not generation.claimed:
                    owns_generation = True
                    generation.claimed = True
                if not owns_generation:
                    raise
                claim_failure = failure
        finally:
            # Keep owned execution in the claim-stage publication guard so
            # exceptions raised by lock, pool, operation, and publication
            # calls cannot bypass cleanup. CPython cannot make arbitrary
            # externally injected exceptions between bytecodes transactional;
            # daemon workers are therefore not an async-exception boundary.
            if owns_generation:
                _run_claimed_cleanup(
                    generation,
                    operation,
                    args,
                    claim_failure,
                )


def _recover_dead_attempt(generation: _AsyncAttempt, what: str) -> bool:
    """Freeze an ownerless generation so waiters can retry instead of hang."""
    if generation.done.is_set():
        return True
    if not generation.runner_lock.acquire(blocking=False):
        return False
    try:
        with generation.claim_lock:
            if generation.done.is_set():
                return True
            if not generation.claimed:
                owner_alive = False
                for owner in generation.threads:
                    try:
                        if owner.is_alive():
                            owner_alive = True
                            break
                    except RuntimeError:
                        continue
                if owner_alive:
                    return False
                # A successful Thread.start with a now-dead owner cannot have
                # a future candidate. After an ambiguous start exception, a
                # confirmed libdispatch candidate may merely be queued; never
                # steal its claim based on elapsed wall time.
                if generation.ambiguous_launch or generation.launch_failed:
                    return False
                if (not generation.thread_start_returned
                        and generation.fallback_scheduled):
                    return False
                generation.claimed = True
            if generation.failure is None:
                generation.failure = RuntimeError(
                    f"{what} exited without publishing completion")
            generation.done.set()
            return True
    finally:
        generation.runner_lock.release()


def _wait_for_attempt(
    generation: _AsyncAttempt,
    timeout: float | None,
    what: str,
) -> bool:
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if generation.done.is_set() or _recover_dead_attempt(generation, what):
            return True
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            wait = min(AVWriter.STATUS_POLL_S, remaining)
        else:
            wait = AVWriter.STATUS_POLL_S
        generation.done.wait(timeout=wait)


def _start_cleanup_attempt(
    generation: _AsyncAttempt,
    *,
    name: str,
    operation: Any,
    args: tuple[Any, ...],
) -> None:
    """Start via Python, racing an ambiguous launch failure through GCD."""
    generation.launch_failed = False
    executor_signature = (
        id(threading.Thread),
        id(libdispatch.dispatch_async),
    )
    generation.launch_rounds[executor_signature] = (
        generation.launch_rounds.get(executor_signature, 0) + 1
    )
    # A failed or metadata-hidden Thread object does not own the actual hidden
    # target and need not be retained forever. Keep only records whose Python
    # owner is observably live; generation ambiguity protects invisible owners.
    live_owners = []
    for existing_owner in generation.threads:
        try:
            if existing_owner.is_alive():
                live_owners.append(existing_owner)
        except RuntimeError:
            continue
    generation.threads = live_owners

    target_args = (generation, operation, *args)
    owner = threading.Thread(
        target=_run_with_autorelease_pool,
        args=target_args,
        name=name,
        daemon=True,
    )
    generation.thread = owner
    generation.threads.append(owner)
    try:
        owner.start()
        generation.thread_start_returned = True
        generation.ambiguous_launch = False
    except BaseException as start_failure:
        round_ambiguous = not isinstance(start_failure, Exception)
        # Thread.start can be interrupted after the OS launch but before
        # ident/is_alive metadata is visible. Always enqueue a second candidate;
        # runner_lock + claimed guarantee that at most one performs cleanup.
        try:
            queue = libdispatch.dispatch_get_global_queue(0, 0)
            libdispatch.dispatch_async(
                queue,
                lambda: _run_with_autorelease_pool(*target_args),
            )
            generation.fallback_scheduled = True
            generation.ambiguous_launch = round_ambiguous
        except BaseException as dispatch_failure:
            round_ambiguous = (
                round_ambiguous
                or not isinstance(dispatch_failure, Exception)
            )
            start_failure.add_note(
                f"libdispatch fallback failed: {dispatch_failure!r}")
            # One more Python candidate handles a genuine pre-launch failure.
            # If the first start was actually hidden in its metadata window,
            # the generation claim still guarantees exactly one native owner.
            retry_owner = threading.Thread(
                target=_run_with_autorelease_pool,
                args=target_args,
                name=f"{name}-recovery",
                daemon=True,
            )
            generation.threads.append(retry_owner)
            try:
                retry_owner.start()
            except BaseException as retry_start_failure:
                round_ambiguous = (
                    round_ambiguous
                    or not isinstance(retry_start_failure, Exception)
                )
                start_failure.add_note(
                    f"recovery Thread.start failed: "
                    f"{retry_start_failure!r}")
                generation.launch_failed = True
                generation.ambiguous_launch = round_ambiguous
            else:
                generation.thread = retry_owner
                generation.thread_start_returned = True
                generation.ambiguous_launch = round_ambiguous
        raise


def _can_retry_cleanup_launch(generation: _AsyncAttempt) -> bool:
    """Return whether a later caller should retry an unowned launch."""
    if (generation.done.is_set()
            or generation.claimed
            or not generation.launch_failed
            or generation.fallback_scheduled):
        return False
    executor_signature = (
        id(threading.Thread),
        id(libdispatch.dispatch_async),
    )
    if (generation.ambiguous_launch
            and generation.launch_rounds.get(executor_signature, 0)
            >= _MAX_CLEANUP_LAUNCH_ROUNDS):
        return False
    for owner in generation.threads:
        try:
            if owner.is_alive():
                return False
        except RuntimeError:
            continue
    return True


def _exception_chain_contains(
    root: BaseException,
    target: BaseException,
) -> bool:
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is target:
            return True
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


def _raise_primary(primary: BaseException, later: BaseException) -> None:
    """Deliver the first failure without replacing or cycling its root cause."""
    if primary is later:
        raise primary
    existing_cause = primary.__cause__
    if (existing_cause is not None
            or _exception_chain_contains(later, primary)):
        primary.add_note(f"later writer boundary failure: {later!r}")
        existing_context = primary.__context__
        try:
            raise primary from existing_cause
        except BaseException:
            # Raising while ``later`` is actively handled would otherwise set
            # primary.__context__ = later and form a mixed context/cause cycle.
            primary.__context__ = existing_context
            raise
    raise primary from later


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
    STATUS_POLL_S = 0.05

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: Fraction | int | float | str,
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
        cadence = rational_cadence(fps)
        self._state_lock = threading.RLock()
        self._audio_pump_lock = threading.Lock()
        self._video_append_lock = threading.Lock()
        self._native_mutations: set[Any] = set()
        self._state = "constructing"
        self._failure: BaseException | None = None
        self._finish_done = threading.Event()
        self._native_cancelled = False
        self._native_cancel_in_progress = False
        self._native_finished = False
        self._native_finish_done: threading.Event | None = None
        self._cancel_attempt: _AsyncAttempt | None = None
        self._audio_callbacks_inflight = 0
        self._audio_callbacks_done = threading.Event()
        self._audio_callbacks_done.set()
        self._audio_track_close_pending = False
        self._audio_track_closed = False
        self._audio_track_closing = False
        self._audio_close_attempt: _AsyncAttempt | None = None
        self._audio_done = threading.Event()
        self._audio_progress = [0]
        self._audio_complete = False
        self.writer: Any = None
        self.audio_input: Any = None
        self.audio_track: AudioTrack | None = None
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
                    output_path, width, height, cadence,
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
            with self._state_lock:
                primary = self._failure
            assert primary is not None
            self._cancel_for_failure(primary)
            _raise_primary(primary, exc)

    def _construct(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: Fraction,
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
        if audio_track is not None:
            # File-backed tracks own a mutable decoder cursor. A post writer
            # and comparison writer can pump concurrently, so each writer
            # must receive an independent lazy cursor. In-memory tracks are
            # immutable and return themselves here.
            audio_track = audio_track.fork()
        # Own the fork immediately so every later construction failure can
        # close it through cancel(), including startWriting failures.
        self.audio_track = audio_track
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        url = Foundation.NSURL.fileURLWithPath_(str(output_path))
        writer, err = av.AVAssetWriter.alloc().initWithURL_fileType_error_(
            url, av.AVFileTypeMPEG4, None,
        )
        if writer is None:
            raise RuntimeError(f"AVAssetWriter init failed for {output_path}: {err}")
        # Register native ownership before any later setup or start operation.
        # If startWriting succeeds and session start raises, cancel() can still
        # reach and release this writer.
        self.writer = writer

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
        # and append converts RGBAHalf or direct MLX RGB to 10-bit 4:2:2 with
        # that matrix. NV12 (fast) output is already YUV and feeds through.
        self._yuv_feed = cv_color is not None and source_pixel_format == _pb.PIX_RGBAHALF
        self.accepts_mlx_rgb = self._yuv_feed
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
        self.audio_input = audio_input

        # Start the writer ---------------------------------------------------
        writer.setMovieTimeScale_(_pb.VIDEO_TIME_SCALE)
        if not writer.startWriting():
            raise RuntimeError(f"AVAssetWriter.startWriting failed: {writer.error()}")
        writer.startSessionAtSourceTime_(CoreMedia.CMTimeMake(0, _pb.VIDEO_TIME_SCALE))

        self.video_input = video_input
        self.adaptor = adaptor
        self.adaptor_pixel_format = int(adaptor_format)
        self.adaptor_width = int(width)
        self.adaptor_height = int(height)
        self.cadence = fps
        self.fps = float(self.cadence)
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
        self._audio_done.clear()
        self._audio_progress[0] = 0
        self._audio_complete = False
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

    def _run_native_mutation(
        self,
        operation: Any,
        *,
        states: tuple[str, ...],
    ) -> tuple[bool, Any]:
        """Run one writer mutation under asynchronous-cancel ownership.

        Admission is atomic with logical cancellation. Once admitted, the call
        is allowed to begin and retire even if ``cancel()`` returns meanwhile.
        The cancellation coordinator waits for every token to retire before it
        calls AVAssetWriter.cancelWriting(), as required by AVFoundation.
        """
        token = threading.Lock()
        # The C-backed lock is the durable ownership record. Even if an async
        # exception interrupts the Python set-removal code, unwinding this
        # context releases the token and a later coordinator can reclaim it.
        with token:
            try:
                with self._state_lock:
                    if self._state not in states:
                        return False, None
                    self._native_mutations.add(token)
                return True, operation()
            finally:
                with self._state_lock:
                    self._native_mutations.discard(token)

    def _wait_for_native_mutations(self) -> None:
        """Wait for and reclaim every mutation admitted before cancellation."""
        while True:
            with self._state_lock:
                tokens = tuple(self._native_mutations)
            if not tokens:
                return
            for token in tokens:
                # Context-manager unwinding makes the coordinator's acquire
                # itself BaseException-safe. Stale unlocked tokens left by an
                # interrupted mutator are reclaimed here without blocking.
                with token:
                    pass
            with self._state_lock:
                self._native_mutations.difference_update(tokens)

    def _status_context(self) -> tuple[Any, Any]:
        writer = self.writer
        if writer is None:
            with self._state_lock:
                return self._state, None
        return writer.status(), writer.error()

    def _begin_audio_callback(self) -> bool:
        """Linearize callback admission against cancellation."""
        with self._state_lock:
            if self._state not in ("constructing", "writing", "finishing"):
                return False
            self._audio_callbacks_inflight += 1
            self._audio_callbacks_done.clear()
            return True

    def _record_cleanup_failure(self, exc: BaseException, what: str) -> None:
        with self._state_lock:
            if self._failure is None:
                self._failure = exc
                if self._state not in ("finished", "cancelled"):
                    self._state = "failed"
            else:
                self._failure.add_note(f"{what}: {exc!r}")

    def _audio_close_worker_main(
        self,
        track: AudioTrack,
        generation: _AsyncAttempt,
    ) -> None:
        closed = False
        failure: BaseException | None = None
        try:
            for attempt in range(2):
                try:
                    track.close()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    else:
                        failure.add_note(
                            f"audio close retry failed: {exc!r}")
                    if attempt == 0:
                        continue
                    break
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                    else:
                        failure.add_note(
                            f"audio close retry interrupted: {exc!r}")
                    break
                else:
                    closed = True
                    failure = None
                    break
        except BaseException as exc:
            failure = exc
        finally:
            if failure is not None:
                try:
                    self._record_cleanup_failure(
                        failure, "audio source close failed")
                except BaseException as record_failure:
                    failure.add_note(
                        f"recording audio close failure failed: "
                        f"{record_failure!r}")
            with self._state_lock:
                self._audio_track_closing = False
                self._audio_track_closed = closed
                self._audio_track_close_pending = not closed
                if failure is not None:
                    _merge_attempt_failure(
                        generation, failure, "audio close failure")

    def _ensure_audio_close_worker(
        self,
        *,
        retry_failed: bool = False,
    ) -> _AsyncAttempt | None:
        with self._state_lock:
            track = self.audio_track
            if track is None or self._audio_track_closed:
                self._audio_track_closed = True
                self._audio_track_close_pending = False
                return None
            if self._audio_callbacks_inflight:
                self._audio_track_close_pending = True
                return None

            generation = self._audio_close_attempt
            if generation is not None and not generation.done.is_set():
                if not _recover_dead_attempt(
                    generation, "audio close worker"):
                    if _can_retry_cleanup_launch(generation):
                        _start_cleanup_attempt(
                            generation,
                            name=(
                                f"kinovsr-close-"
                                f"{getattr(self, 'label', 'writer')}"
                            ),
                            operation=self._audio_close_worker_main,
                            args=(track, generation),
                        )
                    return generation
                self._audio_track_closing = False
                self._audio_track_close_pending = True
            if (generation is not None
                    and generation.done.is_set()
                    and generation.failure is not None
                    and not retry_failed):
                return generation

            generation = _AsyncAttempt()
            self._audio_track_closing = True
            self._audio_track_close_pending = False
            self._audio_close_attempt = generation
            _start_cleanup_attempt(
                generation,
                name=f"kinovsr-close-{getattr(self, 'label', 'writer')}",
                operation=self._audio_close_worker_main,
                args=(track, generation),
            )
            return generation

    def _wait_for_audio_close(
        self,
        timeout: float | None,
        *,
        retry_failed: bool = False,
    ) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._state_lock:
                closed = self._audio_track_closed
                generation = self._audio_close_attempt if closed else None
            if generation is None:
                if closed:
                    return True
                generation = self._ensure_audio_close_worker(
                    retry_failed=retry_failed)
            retry_failed = False
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
            else:
                remaining = None
            if generation is None:
                if not self._audio_callbacks_done.wait(timeout=remaining):
                    return False
                continue
            if not _wait_for_attempt(
                generation, remaining, "audio close worker"):
                return False
            if generation.failure is not None:
                raise generation.failure
            if closed:
                return True

    def _cancel_for_failure(
        self,
        primary: BaseException,
    ) -> None:
        """Request cleanup without replacing ``primary``'s causal chain."""
        try:
            self.cancel()
        except BaseException as cleanup:
            primary.add_note(f"writer cleanup request failed: {cleanup!r}")
            # Construction failures leave no caller-owned writer through which
            # a pre-launch Thread.start interruption could be retried. Re-run
            # only worker admission; an owner that did launch is detected and
            # shared, so this cannot create competing cleanup calls.
            for what, ensure in (
                (
                    "audio close worker recovery",
                    lambda: self._ensure_audio_close_worker(
                        retry_failed=True),
                ),
                ("cancel worker recovery", self._ensure_cancel_worker),
            ):
                try:
                    ensure()
                except BaseException as recovery:
                    primary.add_note(f"{what} failed: {recovery!r}")

    def _raise_after_cancel(self, exc: BaseException) -> None:
        """Poison the writer, clean up, and deliver the first causal error."""
        self._record_failure(exc)
        with self._state_lock:
            primary = self._failure
        assert primary is not None
        self._cancel_for_failure(primary)
        _raise_primary(primary, exc)

    def _end_audio_callback(self, n_samples: int) -> None:
        with self._state_lock:
            if self._audio_callbacks_inflight > 0:
                self._audio_callbacks_inflight -= 1
            callbacks_done = self._audio_callbacks_inflight == 0
            if callbacks_done:
                self._audio_callbacks_done.set()
            should_close = (
                callbacks_done and self._audio_track_close_pending
            )
            should_signal = (
                self._audio_complete
                or self._audio_progress[0] >= n_samples
                or self._state not in ("constructing", "writing", "finishing")
                or self._failure is not None
            )
        if should_close:
            try:
                self._ensure_audio_close_worker()
            except BaseException as exc:  # callback cannot raise to its caller
                self._record_cleanup_failure(
                    exc, "audio close worker start failed")
        if should_signal:
            self._audio_done.set()

    def _pump_audio(self, n_samples: int, chunk_frames: int) -> None:
        """Drain ready audio without letting callback failures disappear.

        GCD cannot propagate a Python exception to ``finish``. Store the first
        one, signal the waiter, and let the synchronous writer boundary raise
        it with its original position and native status.
        """
        if not self._begin_audio_callback():
            self._audio_done.set()
            return
        try:
            # GCD may schedule the readiness block more than once. Only one
            # callback owns the decoder cursor and progress counter at a time.
            # Cancel never takes this lock.
            with self._audio_pump_lock:
                while True:
                    with self._state_lock:
                        if (self._state not in (
                                "constructing", "writing", "finishing",
                            ) or self._audio_complete):
                            return
                        audio_input = self.audio_input
                    if audio_input is None:
                        return
                    if not audio_input.isReadyForMoreMediaData():
                        return
                    with self._state_lock:
                        if (self._state not in (
                                "constructing", "writing", "finishing",
                            ) or self._audio_complete):
                            return
                        pos = self._audio_progress[0]
                        end = min(pos + chunk_frames, n_samples)
                        track = self.audio_track
                    if track is None:
                        raise RuntimeError(
                            f"[{self.label}] audio callback has no source track")

                    # The callback owns the track until its finally block.
                    # Cancel can return while a decoder read is blocked and
                    # defers track close to the last admitted callback.
                    sb = (
                        None if pos >= n_samples
                        else track.make_sample_buffer(pos, end)
                    )
                    if sb is None:
                        called, _ = self._run_native_mutation(
                            audio_input.markAsFinished,
                            states=("constructing", "writing", "finishing"),
                        )
                        if not called:
                            return
                        with self._state_lock:
                            if self._state in (
                                "constructing", "writing", "finishing",
                            ):
                                self._audio_complete = True
                        return

                    called, accepted = self._run_native_mutation(
                        lambda audio_input=audio_input, sb=sb: (
                            audio_input.appendSampleBuffer_(sb)
                        ),
                        states=("constructing", "writing", "finishing"),
                    )
                    if not called:
                        return
                    with self._state_lock:
                        state = self._state
                        failure = self._failure
                    if state not in (
                        "constructing", "writing", "finishing",
                    ):
                        return
                    if failure is not None:
                        raise failure
                    if not accepted:
                        status, error = self._status_context()
                        raise RuntimeError(
                            f"[{self.label}] audio appendSampleBuffer failed "
                            f"at {pos}: status={status} error={error}")
                    appended = int(CoreMedia.CMSampleBufferGetNumSamples(sb))
                    if appended <= 0 or appended > end - pos:
                        raise RuntimeError(
                            f"[{self.label}] audio source returned invalid sample "
                            f"count {appended} for [{pos}, {end})")
                    with self._state_lock:
                        if self._state not in (
                            "constructing", "writing", "finishing",
                        ):
                            return
                        self._audio_progress[0] = pos + appended
                        complete = self._audio_progress[0] >= n_samples
                    if complete:
                        called, _ = self._run_native_mutation(
                            audio_input.markAsFinished,
                            states=("constructing", "writing", "finishing"),
                        )
                        if not called:
                            return
                        with self._state_lock:
                            if self._state in (
                                "constructing", "writing", "finishing",
                            ):
                                self._audio_complete = True
                        return
        except BaseException as exc:  # callback failures cross via shared state
            self._record_failure(exc)
        finally:
            self._end_audio_callback(n_samples)

    def _check_native_status(self, what: str) -> None:
        self._raise_if_failed()
        with self._state_lock:
            state = self._state
        if state == "cancelled":
            raise RuntimeError(
                f"[{self.label}] writer was cancelled during {what}")
        status, error = self._status_context()
        self._raise_if_failed()
        with self._state_lock:
            state = self._state
        if state == "cancelled":
            raise RuntimeError(
                f"[{self.label}] writer was cancelled during {what}")
        if status in (3, 4):  # AVAssetWriterStatusFailed/Cancelled
            exc = RuntimeError(
                f"[{self.label}] writer entered status={status} during "
                f"{what}: {error}")
            self._record_failure(exc)
            raise exc

    def _wait_for_event(
        self,
        event: threading.Event,
        *,
        timeout: float,
        what: str,
    ) -> bool:
        """Wait to one deadline while continuously checking native status."""
        deadline = time.monotonic() + timeout
        while True:
            self._check_native_status(what)
            if event.is_set():
                self._check_native_status(what)
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            event.wait(timeout=min(self.STATUS_POLL_S, remaining))

    def _wait_for_ready(self, input_obj: Any, what: str) -> None:
        """Block until input_obj.isReadyForMoreMediaData(). Bail with a clean
        error if the writer enters Failed/Cancelled, or after 30 s of no
        progress (so a stuck writer surfaces as a visible failure, not a hang).
        """
        deadline = time.monotonic() + self.READY_TIMEOUT_S
        while True:
            with self._state_lock:
                active = self._state in ("writing", "finishing")
            if not active:
                self._raise_if_failed()
                with self._state_lock:
                    state = self._state
                raise RuntimeError(
                    f"[{self.label}] cannot wait on {what} while writer is "
                    f"{state}")
            ready = input_obj.isReadyForMoreMediaData()
            self._raise_if_failed()
            with self._state_lock:
                state = self._state
            if state not in ("writing", "finishing"):
                raise RuntimeError(
                    f"[{self.label}] cannot wait on {what} while writer is "
                    f"{state}")
            if ready:
                break
            self._check_native_status(f"waiting on {what}")
            time.sleep(0.001)
            if time.monotonic() >= deadline:
                status, _error = self._status_context()
                exc = RuntimeError(
                    f"[{self.label}] {what} input never became ready "
                    f"(waited {self.READY_TIMEOUT_S:g}s, "
                    f"status={status})")
                self._record_failure(exc)
                raise exc
        self._check_native_status(f"waiting on {what}")

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def _rgb_to_yuv_buffer(self, rgb: Any) -> Any:
        ybuf = _pb.pool_create_buffer(self.adaptor.pixelBufferPool())
        if ybuf is None:
            raise RuntimeError(
                f"[{self.label}] YUV pool buffer allocation failed")
        _yuv.rgb_to_yuv422_10(
            rgb, ybuf, self._yuv_matrix, self._yuv_full)
        return ybuf

    def _append_payload(
        self,
        payload: Any,
        *,
        direct_mlx_rgb: bool,
        pts_ticks: int | None,
        duration_ticks: int | None,
    ) -> None:
        """Append one prepared buffer or one direct MLX RGB frame.

        Default: the next index-grid PTS (frame_count/fps). A caller
        that owns an exact timeline (the typed pipeline's endpoints)
        passes ``pts_ticks``/``duration_ticks`` in VIDEO_TIME_SCALE
        instead - the index grid quantizes NTSC-family rates to a fixed
        per-frame duration, which drifts ~3.6 s/hour at 59.94 fps
        against the exact rational grid."""
        try:
            # Serializing appends keeps frame_count and the native input in the
            # same order. Finish takes this lock before marking the input done;
            # cancel deliberately does not, so a blocked conversion or native
            # append cannot prevent it from cancelling the writer.
            with self._video_append_lock, autorelease_pool():
                with self._state_lock:
                    if self._state != "writing":
                        self._raise_if_failed()
                        raise RuntimeError(
                            f"[{self.label}] cannot append while writer is "
                            f"{self._state}")
                    frame_index = self.frame_count
                self._wait_for_ready(self.video_input, "video")
                if direct_mlx_rgb:
                    pb = self._rgb_to_yuv_buffer(payload)
                elif self._yuv_feed:
                    pb = self._rgb_to_yuv_buffer(
                        _pb.read_rgbahalf_rgb(payload))
                else:
                    pb = payload
                with self._state_lock:
                    # A finisher may have claimed the writer after this append
                    # was admitted. It waits on _video_append_lock, so this
                    # already-admitted frame still precedes markAsFinished.
                    if self._state not in ("writing", "finishing"):
                        self._raise_if_failed()
                        raise RuntimeError(
                            f"[{self.label}] cannot append while writer is "
                            f"{self._state}")
                if pts_ticks is None:
                    pts = _pb.frame_pts(frame_index, self.cadence)
                else:
                    pts = CoreMedia.CMTimeMake(
                        int(pts_ticks), _pb.VIDEO_TIME_SCALE)
                called, accepted = self._run_native_mutation(
                    lambda: self.adaptor.appendPixelBuffer_withPresentationTime_(
                        pb, pts,
                    ),
                    states=("writing", "finishing"),
                )
                if not called:
                    with self._state_lock:
                        state = self._state
                    self._raise_if_failed()
                    raise RuntimeError(
                        f"[{self.label}] cannot append while writer is {state}")
                with self._state_lock:
                    state = self._state
                    failure = self._failure
                    if accepted and state in ("writing", "finishing"):
                        if pts_ticks is not None and duration_ticks is not None:
                            self._explicit_end_ticks = (
                                int(pts_ticks) + int(duration_ticks))
                        self.frame_count = frame_index + 1
                        return
                if failure is not None:
                    raise failure
                if state not in ("writing", "finishing"):
                    raise RuntimeError(
                        f"[{self.label}] cannot append while writer is {state}")
                status, error = self._status_context()
                raise RuntimeError(
                    f"[{self.label}] appendPixelBuffer failed at frame "
                    f"{frame_index}: status={status} error={error}")
        except BaseException as exc:
            self._record_failure(exc)
            with self._state_lock:
                primary = self._failure
            assert primary is not None
            _raise_primary(primary, exc)

    def append(self, pb: Any, *, pts_ticks: int | None = None,
               duration_ticks: int | None = None) -> None:
        """Append a CVPixelBuffer, converting RGBAHalf to YUV when needed."""
        self._append_payload(
            pb,
            direct_mlx_rgb=False,
            pts_ticks=pts_ticks,
            duration_ticks=duration_ticks,
        )

    def prepare_mlx_rgb(self, rgb: Any) -> Any:
        """Apply the explicit MLX rounding boundary before native ownership."""
        if not self.accepts_mlx_rgb:
            raise RuntimeError(
                f"[{self.label}] direct MLX RGB input requires the YUV feed")
        import mlx.core as mx

        # Preserve the corrected legacy path's numeric boundary: FileSink
        # wrote fp16 RGBAHalf and read it back as float32 before YUV
        # quantization. Keep that rounding in the lazy MLX graph while
        # removing both CoreVideo copies and their synchronization points.
        return rgb[..., :3].astype(mx.float16).astype(mx.float32)

    def append_prepared_mlx_rgb(
        self,
        equivalent: Any,
        *,
        pts_ticks: int | None = None,
        duration_ticks: int | None = None,
    ) -> None:
        """Append MLX RGB after caller-visible conversion has succeeded."""
        if not self.accepts_mlx_rgb:
            raise RuntimeError(
                f"[{self.label}] direct MLX RGB input requires the YUV feed")
        self._append_payload(
            equivalent,
            direct_mlx_rgb=True,
            pts_ticks=pts_ticks,
            duration_ticks=duration_ticks,
        )

    def append_mlx_rgb(
        self,
        rgb: Any,
        *,
        pts_ticks: int | None = None,
        duration_ticks: int | None = None,
    ) -> None:
        """Convert typed MLX RGB directly into the pooled encoder YUV feed."""
        equivalent = self.prepare_mlx_rgb(rgb)
        self.append_prepared_mlx_rgb(
            equivalent,
            pts_ticks=pts_ticks,
            duration_ticks=duration_ticks,
        )

    def finish(self) -> None:
        """Finalize once, with bounded audio and native completion waits."""
        failure_to_raise: BaseException | None = None
        with self._state_lock:
            if self._state == "finished":
                return
            if self._failure is not None:
                failure_to_raise = self._failure
                wait_for_other = False
            elif self._state in ("finishing", "closing"):
                wait_for_other = True
            else:
                if self._state == "cancelled":
                    raise RuntimeError(f"[{self.label}] writer was cancelled")
                self._state = "finishing"
                wait_for_other = False

        if failure_to_raise is not None:
            self._raise_after_cancel(failure_to_raise)

        if wait_for_other:
            try:
                if not self._wait_for_event(
                    self._finish_done,
                    timeout=self.FINISH_TIMEOUT_S,
                    what="concurrent finish",
                ):
                    raise RuntimeError(
                        f"[{self.label}] concurrent finish did not complete "
                        f"within {self.FINISH_TIMEOUT_S:g}s")
                with self._state_lock:
                    if self._state == "finished":
                        return
                self._raise_if_failed()
                raise RuntimeError(f"[{self.label}] writer was cancelled")
            except BaseException as exc:
                self._raise_after_cancel(exc)

        try:
            with autorelease_pool():
                # Drain every append admitted before finish, then mark the
                # video input outside the state lock. Cancel remains prompt if
                # either operation is stuck in AVFoundation.
                with self._video_append_lock:
                    called, _ = self._run_native_mutation(
                        self.video_input.markAsFinished,
                        states=("finishing",),
                    )
                    if not called:
                        self._raise_if_failed()
                        raise RuntimeError(
                            f"[{self.label}] writer was cancelled")
                self._check_native_status("video input finish")
                if (self.audio_input is not None
                        and not self._wait_for_event(
                            self._audio_done,
                            timeout=self.AUDIO_TIMEOUT_S,
                            what="audio drain",
                        )):
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
                    end = _pb.frame_pts(self.frame_count, self.cadence)
                native_done = threading.Event()
                with self._state_lock:
                    if self._state != "finishing":
                        self._raise_if_failed()
                        raise RuntimeError(
                            f"[{self.label}] writer was cancelled")
                    self._native_finish_done = native_done
                called, _ = self._run_native_mutation(
                    lambda: self.writer.endSessionAtSourceTime_(end),
                    states=("finishing",),
                )
                if not called:
                    self._raise_if_failed()
                    raise RuntimeError(f"[{self.label}] writer was cancelled")
                called, _ = self._run_native_mutation(
                    lambda: self.writer.finishWritingWithCompletionHandler_(
                        native_done.set,
                    ),
                    states=("finishing",),
                )
                if not called:
                    self._raise_if_failed()
                    raise RuntimeError(f"[{self.label}] writer was cancelled")
                if not self._wait_for_event(
                    native_done,
                    timeout=self.FINISH_TIMEOUT_S,
                    what="native finish callback",
                ):
                    status, _error = self._status_context()
                    raise RuntimeError(
                        f"[{self.label}] AVAssetWriter finish callback did not "
                        f"arrive within {self.FINISH_TIMEOUT_S:g}s "
                        f"(status={status})")
                self._check_native_status("finish")
                status, error = self._status_context()
                if status != 2:
                    raise RuntimeError(
                        f"[{self.label}] AVAssetWriter finished with status "
                        f"{status}: {error}")
            with self._state_lock:
                if self._state != "finishing":
                    raise RuntimeError(f"[{self.label}] writer was cancelled")
                self._native_finished = True
                self._state = "closing"
            if not self._wait_for_audio_close(self.FINISH_TIMEOUT_S):
                raise RuntimeError(
                    f"[{self.label}] audio source close did not complete "
                    f"within {self.FINISH_TIMEOUT_S:g}s")
            with self._state_lock:
                if self._state != "closing":
                    raise RuntimeError(f"[{self.label}] writer was cancelled")
                self._state = "finished"
        except BaseException as exc:
            self._raise_after_cancel(exc)
        finally:
            with self._state_lock:
                self._native_finish_done = None
            self._finish_done.set()

    def _cancel_worker_main(self, generation: _AsyncAttempt) -> None:
        """Retire admitted mutations, then perform blocking cleanup off-thread."""
        failure: BaseException | None = None
        native_failure: BaseException | None = None
        try:
            # AVFoundation explicitly forbids cancelWriting concurrently with
            # either append API. Cancellation admission closed the mutation
            # set, so waiting here cannot race a new append.
            self._wait_for_native_mutations()
            with self._state_lock:
                writer = self.writer
                native_finished = self._native_finished
                native_cancelled = self._native_cancelled
            if writer is not None and not native_finished and not native_cancelled:
                with self._state_lock:
                    self._native_cancel_in_progress = True
                cancel_failure: BaseException | None = None
                cancelled = False
                try:
                    for attempt in range(2):
                        try:
                            writer.cancelWriting()
                        except Exception as exc:
                            if cancel_failure is None:
                                cancel_failure = exc
                            else:
                                cancel_failure.add_note(
                                    f"native cancel retry failed: {exc!r}")
                            if attempt == 0:
                                continue
                            break
                        except BaseException as exc:
                            if cancel_failure is None:
                                cancel_failure = exc
                            else:
                                cancel_failure.add_note(
                                    f"native cancel retry interrupted: {exc!r}")
                            break
                        else:
                            cancelled = True
                            cancel_failure = None
                            break
                finally:
                    with self._state_lock:
                        self._native_cancel_in_progress = False
                        if cancelled:
                            self._native_cancelled = True
                if cancel_failure is not None:
                    failure = cancel_failure
                    native_failure = cancel_failure

            # Track close cannot race a callback-owned decoder cursor. The
            # independent close worker may already have completed while native
            # cancellation was blocked; the coordinator only joins its result.
            self._audio_callbacks_done.wait()
            try:
                self._wait_for_audio_close(None)
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        f"additional audio close failure: {exc!r}")
        except BaseException as exc:
            if failure is None:
                failure = exc
            elif failure is not exc:
                failure.add_note(
                    f"additional cancellation coordinator failure: {exc!r}")
            native_failure = failure
        finally:
            if native_failure is not None:
                try:
                    self._record_cleanup_failure(
                        native_failure, "native writer cancellation failed")
                except BaseException as record_failure:
                    native_failure.add_note(
                        f"recording cancellation failure failed: "
                        f"{record_failure!r}")
            with self._state_lock:
                if failure is not None:
                    _merge_attempt_failure(
                        generation, failure, "cancel coordinator failure")

    def _ensure_cancel_worker(self) -> _AsyncAttempt | None:
        with self._state_lock:
            generation = self._cancel_attempt
            if (generation is not None
                    and not generation.done.is_set()
                    and not _recover_dead_attempt(
                        generation, "cancel worker")):
                if _can_retry_cleanup_launch(generation):
                    _start_cleanup_attempt(
                        generation,
                        name=(
                            f"kinovsr-cancel-"
                            f"{getattr(self, 'label', 'writer')}"
                        ),
                        operation=self._cancel_worker_main,
                        args=(generation,),
                    )
                return generation
            cleanup_complete = (
                (self.writer is None
                 or self._native_finished
                 or self._native_cancelled)
                and (self.audio_track is None or self._audio_track_closed)
            )
            if cleanup_complete:
                self._cancel_attempt = None
                return None
            generation = _AsyncAttempt()
            self._cancel_attempt = generation
            _start_cleanup_attempt(
                generation,
                name=f"kinovsr-cancel-{getattr(self, 'label', 'writer')}",
                operation=self._cancel_worker_main,
                args=(generation,),
            )
            return generation

    def _wait_for_cancel_cleanup(self, timeout: float | None) -> bool:
        with self._state_lock:
            generation = self._cancel_attempt
        if generation is None:
            return True
        if not _wait_for_attempt(generation, timeout, "cancel worker"):
            return False
        if generation.failure is not None:
            raise generation.failure
        return True

    def cancel(self) -> None:
        """Request cancellation promptly; blocking cleanup runs off-thread.

        A mutation admitted before this method linearizes may still begin after
        it returns. Its token keeps native ownership alive, and the coordinator
        waits for it to retire before calling ``cancelWriting``. This preserves
        AVFoundation's no-concurrent-append contract while keeping discard and
        repeated cancellation bounded even if native cancellation blocks.
        """
        with self._state_lock:
            finished = self._state == "finished"
            if not finished:
                self._state = "cancelled"
                if self.audio_track is not None:
                    self._audio_track_close_pending = True
            native_finish_done = self._native_finish_done

        if finished:
            return

        # Wake synchronous waiters before any framework cleanup. They poll the
        # cancelled state and do not depend on a native completion callback.
        self._audio_done.set()
        if native_finish_done is not None:
            native_finish_done.set()
        self._finish_done.set()

        # Native cancellation and decoder closure own independent resources.
        # Start both before waiting in either worker so a stuck framework
        # cancel cannot retain an otherwise idle audio decoder.
        start_failure: BaseException | None = None
        try:
            self._ensure_audio_close_worker(retry_failed=True)
        except BaseException as exc:
            start_failure = exc
        try:
            self._ensure_cancel_worker()
        except BaseException as exc:
            if start_failure is None:
                start_failure = exc
            else:
                start_failure.add_note(
                    f"additional cancel worker start failure: {exc!r}")
        if start_failure is not None:
            raise start_failure
