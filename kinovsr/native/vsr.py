"""VideoToolbox Super Resolution (spatial upscale) session wrapper.

`VsrSession` wraps VTSuperResolutionScalerConfiguration (HQ, scale=4) or
VTLowLatencySuperResolutionScalerConfiguration (LL, scale=2) plus its
VTFrameProcessor and the source/dst CVPixelBufferPools. The caller hands
in a frame (uint8 RGB or fp16 RGBA) and gets back a destination buffer
ready to feed straight into AVAssetWriter.
"""

import logging
import os
import sys
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from typing import Protocol, cast

import mlx.core as mx

from kinovsr.media import pixel_buffers as _pb
from kinovsr.settings import default_settings

from .frameworks import Quartz, autorelease_pool, vt
from .vision_flow import (
    FLOW_16H,
    VisionFlowToVtConverter,
    advertised_flow_destination_geometry,
    generate_vision_flow,
)

_log = logging.getLogger(__name__)
_NATIVE_STDERR_LOCK = threading.RLock()


class _DownloadableConfiguration(Protocol):
    def configurationModelStatus(self) -> object: ...

    def downloadConfigurationModelWithCompletionHandler_(
        self,
        callback: Callable[[object | None], None],
    ) -> None: ...

    def configurationModelPercentageAvailable(self) -> float: ...


class _ProcessorFrame(Protocol):
    def buffer(self) -> object: ...

    def presentationTimeStamp(self) -> object: ...


def _duplicate_stderr() -> int:
    return os.dup(2)


def _open_devnull() -> int:
    return os.open(os.devnull, os.O_WRONLY)


def _redirect_stderr(source: int) -> None:
    os.dup2(source, 2)


def _close_fd(fd: int) -> None:
    os.close(fd)


@contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    """Swallow OS-level stderr (fd 2) for the duration of the block.

    VideoToolbox compiles the super-resolution Metal pipeline when the frame
    processor session starts and logs 'Resolved compile flags ...
    SpatialSplitGenericDAG' straight to fd 2 via NSLog - bypassing Python's
    sys.stderr, so contextlib.redirect_stderr can't catch it. This redirects the
    file descriptor itself for the brief compile. VideoToolbox reports real
    failures through API return values (the ok/err tuple), not stderr, so
    nothing important is hidden. Set KINOVSR_VERBOSE=1 to keep the native logs.
    """
    if default_settings().verbose:
        yield
        return
    # fd 2 is process-global. Serialize the complete save/redirect/restore
    # interval so two session starts cannot save and later restore each
    # other's temporary /dev/null state. RLock keeps nested construction in
    # one thread safe as well.
    with _NATIVE_STDERR_LOCK:
        sys.stderr.flush()
        saved_fd = _duplicate_stderr()
        try:
            devnull_fd = _open_devnull()
        except BaseException:
            # EMFILE near the descriptor limit is the one realistic failure
            # here; the descriptor just duplicated must not leak with it.
            with suppress(OSError):
                _close_fd(saved_fd)
            raise

        primary: BaseException | None = None
        cleanup_failures: list[tuple[str, BaseException]] = []

        def _cleanup(label: str, fn: Callable[[int], None], fd: int) -> None:
            try:
                fn(fd)
            except BaseException as exc:  # broad: collected below
                cleanup_failures.append((label, exc))

        try:
            _redirect_stderr(devnull_fd)
            try:
                yield
            finally:
                # Restore fd 2 before closing either owned descriptor.
                _cleanup("restore stderr", _redirect_stderr, saved_fd)
        except BaseException as exc:
            primary = exc
        finally:
            _cleanup("close /dev/null", _close_fd, devnull_fd)
            _cleanup("close saved stderr", _close_fd, saved_fd)

        # First failure wins: a body error (the native failure being
        # silenced around) outranks cleanup errors, which ride as notes.
        if primary is None and cleanup_failures:
            _, primary = cleanup_failures[0]
            cleanup_failures = cleanup_failures[1:]
        if primary is not None:
            for label, failure in cleanup_failures:
                if failure is primary:
                    continue
                with suppress(BaseException):
                    primary.add_note(f"{label} also failed: {type(failure).__name__}: {failure}")
            raise primary


def scale_for_mode(mode: str) -> int:
    """Map a VSR spatial mode to its forced scale factor.

    VideoToolbox couples the spatial-mode choice to the scale: LowLatency
    is 2x-only, the HQ classes are 4x-only.  Centralized here so call sites
    don't reinvent the mapping.
    """
    if mode == "fast":
        return 2
    if mode in (
        "balanced",
        "image",
        "basicvsrpp",
        "realbasicvsr",
        "realesrgan",
        "safmn",
        "esc",
        "realviformer",
        "realplksr",
        "toflow",
    ):
        return 4
    if mode == "metalfx":
        # Config-driven (2/3/4); the harness overrides from --metalfx-scale.
        return 2
    raise ValueError(f"unknown VSR spatial-mode: {mode!r}")


# The HighQuality (balanced/image) scaler exposes NO dimension-query API -
# unlike LowLatency's minimumDimensions/maximumDimensions - so these practical
# input caps are determined empirically (config init fails with "Invalid input
# height/width" above them). The cap is per-dimension, not total pixels, and at
# 4x it bounds output to 7680x4320 (8K). Re-probe if a future OS raises it.
HQ_MAX_INPUT_W = 1920
HQ_MAX_INPUT_H = 1080

# The caller/native handoff needs two destination surfaces. balanced also
# threads one previous source while VideoToolbox retains its older sequential
# reference, so it needs three source slots; stateless modes reuse one.
DST_POOL_ALLOCATION_LIMIT = 2
TEMPORAL_SRC_POOL_ALLOCATION_LIMIT = 3
STATELESS_SRC_POOL_ALLOCATION_LIMIT = 1
EXPLICIT_FLOW_PAIR_COUNT = 2
EXPLICIT_FLOW_SOURCE_POOL_LIMIT = 2

# Evidence boundary, not a broad quality claim: Vision High and the
# backward-only policy were selected on one 150-frame 640x480 clip. On macOS
# 26.5.2, balanced VSR decoded byte-identically with full bidirectional High
# flow and with an all-zero forward field. Reverify after an OS update because
# a future VSR implementation may begin consuming the forward field.
VISION_VSR_ACCURACY = "high"


def _validate_combination(width: int, height: int, scale: int, mode: str) -> None:
    """Check the (input size, scale, mode) combo is something VT supports.

    VSR's HQ and LL classes each only support specific scale factors (and LL
    additionally restricts input size to <= 960x960). Failing fast here gives
    a clear error message instead of an opaque init/startSession failure.
    """
    if mode == "fast":
        cls = vt.VTLowLatencySuperResolutionScalerConfiguration
        if not cls.isSupported():
            raise SystemExit("LowLatency VSR not supported on this device.")
        ok = list(cls.supportedScaleFactorsForFrameWidth_frameHeight_(width, height))
        if not ok:
            mn = cls.minimumDimensions()
            mx = cls.maximumDimensions()
            raise SystemExit(
                f"--upscale fast does not support {width}x{height} input. "
                f"Allowed: {mn.width}x{mn.height} to {mx.width}x{mx.height}."
            )
        if float(scale) not in [float(s) for s in ok]:
            raise SystemExit(
                f"--upscale fast at {width}x{height} supports scale={ok}, requested scale={scale}."
            )
    else:
        cls = vt.VTSuperResolutionScalerConfiguration
        if not cls.isSupported():
            raise SystemExit("High-quality VSR not supported on this device.")
        ok = [int(s) for s in cls.supportedScaleFactors()]
        if scale not in ok:
            raise SystemExit(
                f"--upscale {mode} supports scale={ok}, requested scale={scale}. "
                f"Use --upscale fast for 2x."
            )
        # The HQ scaler has no dimension-query API; check the empirical caps so
        # an oversized input fails with a clear message (and before the model
        # download wait) instead of an opaque "config init returned nil".
        if width > HQ_MAX_INPUT_W or height > HQ_MAX_INPUT_H:
            fits_fast = width <= 960 and height <= 960
            hint = (
                "Use --upscale fast for a 2x upscale (input must be <= 960x960)."
                if fits_fast
                else f"This input is larger than any VSR mode supports; downscale it to "
                f"<= {HQ_MAX_INPUT_W}x{HQ_MAX_INPUT_H} (balanced/image) or <= 960x960 (fast) first."
            )
            raise SystemExit(
                f"--upscale {mode} (4x) does not support {width}x{height} input "
                f"(max {HQ_MAX_INPUT_W}x{HQ_MAX_INPUT_H}; a 4x output would exceed 8K). {hint}"
            )


def _wait_for_model_download(config: _DownloadableConfiguration) -> None:
    """Block until HQ VSR's downloadable model is ready, printing progress."""
    status = config.configurationModelStatus()
    if status == vt.VTSuperResolutionScalerConfigurationModelStatusReady:
        return
    _log.info("VSR model not ready (status=%s); requesting download", status)
    done = threading.Event()
    err_box: list[object | None] = [None]

    def completion(error: object | None) -> None:
        err_box[0] = error
        done.set()

    config.downloadConfigurationModelWithCompletionHandler_(completion)
    last_reported = -1
    while not done.is_set():
        pct = int(config.configurationModelPercentageAvailable() * 100)
        if pct // 5 != last_reported // 5:
            _log.info("VSR model download: %s%%", pct)
            last_reported = pct
        done.wait(timeout=0.5)
    if err_box[0] is not None:
        raise RuntimeError(f"VSR model download failed: {err_box[0]}")
    _log.info("VSR model download complete")


class VsrSession:
    """Per-frame VSR processor with prev-frame chain for temporal coherence.

    Spatial modes:
      "fast"      VTLowLatencySuperResolutionScalerConfiguration. scale=2,
                  input <= 960x960. NV12 source. Per-frame, no temporal context.
      "balanced"  VTSuperResolutionScalerConfiguration InputType=Video.
                  scale=4. RGBAHalf source. Uses prev source + prev output to
                  inform the per-frame upscale.  Default for video; slightly
                  crisper motion edges at the cost of slightly more
                  frame-to-frame variation than image mode. The scaler's
                  internal flow is the default; explicit Vision revision 1 flow
                  is selectable.
      "image"     VTSuperResolutionScalerConfiguration InputType=Image. scale=4.
                  RGBAHalf source. Per-frame deterministic upscale, no
                  prev-frame feedback.  Apple documents this as for stills,
                  but on real video it produces measurably lower temporal
                  second-difference than balanced - a legitimate alternative
                  if you prefer the smoother / less-edge-boosted trade-off.

    The previous-frame state can be reset at hard cuts via
    `reset_temporal_context()` - useful for input that may contain edits.
    """

    def __init__(
        self,
        in_w: int,
        in_h: int,
        mode: str,
        fps: float = 24.0,
        *,
        explicit_flow: bool = False,
        flow_backend: str = "internal",
        source_matrix: object | None = None,
    ):
        if mode not in ("fast", "balanced", "image"):
            raise ValueError(f"VsrSession only supports VideoToolbox modes, got {mode!r}")
        if flow_backend not in ("internal", "vision"):
            raise ValueError(
                f"VSR flow backend must be one of ['internal', 'vision'], got {flow_backend!r}"
            )
        if explicit_flow and mode != "balanced":
            raise ValueError("explicit optical flow is only valid for balanced VSR")
        if flow_backend == "vision" and not explicit_flow:
            raise ValueError("the Vision VSR flow backend requires explicit_flow=True")
        if flow_backend == "internal" and explicit_flow:
            raise ValueError("the internal VSR flow backend requires explicit_flow=False")
        scale = scale_for_mode(mode)
        _validate_combination(in_w, in_h, scale, mode)
        self.in_w, self.in_h = in_w, in_h
        self.scale = scale
        self.out_w, self.out_h = in_w * scale, in_h * scale
        self.mode = mode
        self.fps = float(fps)
        self._source_matrix = source_matrix
        self._flow_backend = flow_backend
        self._temporal_video = mode == "balanced"
        self._explicit_flow = bool(explicit_flow and self._temporal_video)
        self._flow_pairs: tuple[tuple[object, object], ...] | None = None
        self._flow_zero_pair: tuple[object, object] | None = None
        self._vision_flow_converter: VisionFlowToVtConverter | None = None
        self._vision_flow_destinations: tuple[object, ...] | None = None
        self._flow_executor: ThreadPoolExecutor | None = None
        self._flow_src_pool: object | None = None
        self._flow_future: Future[None] | None = None
        self._flow_pending_frame: _ProcessorFrame | None = None
        self._flow_pending_index: int | None = None
        self._flow_pending_slot: int | None = None
        self._vsr_needs_random = True

        if mode == "fast":
            self.config = vt.VTLowLatencySuperResolutionScalerConfiguration.alloc().initWithFrameWidth_frameHeight_scaleFactor_(
                in_w, in_h, float(scale)
            )
            if self.config is None:
                raise RuntimeError("LowLatency VSR config init returned nil")
        else:
            input_type = (
                vt.VTSuperResolutionScalerConfigurationInputTypeVideo
                if self._temporal_video
                else vt.VTSuperResolutionScalerConfigurationInputTypeImage
            )
            cls = vt.VTSuperResolutionScalerConfiguration
            self.config = cls.alloc().initWithFrameWidth_frameHeight_scaleFactor_inputType_usePrecomputedFlow_qualityPrioritization_revision_(
                in_w,
                in_h,
                scale,
                input_type,
                self._explicit_flow,
                vt.VTSuperResolutionScalerConfigurationQualityPrioritizationNormal,
                cls.defaultRevision(),
            )
            if self.config is None:
                raise RuntimeError(
                    f"High-quality VSR config init returned nil for {in_w}x{in_h} "
                    f"input at {scale}x. The HQ scaler accepts up to "
                    f"{HQ_MAX_INPUT_W}x{HQ_MAX_INPUT_H}; check the input dimensions."
                )
            _wait_for_model_download(self.config)

        self.processor = vt.VTFrameProcessor.alloc().init()
        # startSession compiles the VSR Metal pipeline, which NSLogs compile
        # chatter to fd 2; suppress just that call (errors come back via `err`).
        with _suppress_native_stderr():
            ok, err = self.processor.startSessionWithConfiguration_error_(self.config, None)
        if not ok:
            raise RuntimeError(
                f"VTFrameProcessor.startSessionWithConfiguration_error_ failed: {err}"
            )

        self.src_attrs = dict(self.config.sourcePixelBufferAttributes() or {})
        self.dst_attrs = dict(self.config.destinationPixelBufferAttributes() or {})
        _log.info(
            "VSR session ready (mode=%s, %sx%s -> %sx%s, src fmt %#x, dst fmt %#x)",
            mode,
            in_w,
            in_h,
            self.out_w,
            self.out_h,
            _pb.resolve_pixel_format(self.src_attrs),
            _pb.resolve_pixel_format(self.dst_attrs),
        )

        self._prev_src_frame: _ProcessorFrame | None = None
        self._prev_dst_frame: object | None = None

        # Lazily-created pixel-transfer session, used by
        # upscale_buffer_to_buffer to normalize externally-decoded buffers.
        self._xfer: object | None = None

        # Src pool: one surface for stateless modes; balanced needs current,
        # previous, and VT's older sequential reference during handoff.
        self._src_pool_allocation_limit = (
            TEMPORAL_SRC_POOL_ALLOCATION_LIMIT
            if self._temporal_video
            else STATELESS_SRC_POOL_ALLOCATION_LIMIT
        )
        self._src_pool = _pb.make_bounded_pool_from_attrs(
            self.src_attrs, self._src_pool_allocation_limit
        )
        if self._src_pool is None:
            try:
                self.processor.endSession()
            except Exception:  # cleanup must not mask the construction error
                _log.exception("failed to end VSR after source-pool failure")
            finally:
                self.processor = None
            raise RuntimeError(
                "VSR source CVPixelBufferPool creation failed; "
                "bounded source allocation is required"
            )
        # Dst pool: session-owned by default so typed/host pipelines reuse
        # IOSurfaces instead of allocating a CVPixelBuffer for every output.
        # A compatible file writer may replace it through use_dst_pool().
        self._dst_pool = _pb.make_bounded_pool_from_attrs(self.dst_attrs, DST_POOL_ALLOCATION_LIMIT)
        self._owns_dst_pool = True
        if self._dst_pool is None:
            try:
                self.processor.endSession()
            except Exception:  # cleanup must not mask the construction error
                _log.exception("failed to end VSR after destination-pool failure")
            finally:
                self.processor = None
                _pb.flush_pool(self._src_pool)
                self._src_pool = None
            raise RuntimeError(
                "VSR destination CVPixelBufferPool creation failed; "
                "bounded output allocation is required"
            )
        if self._explicit_flow:
            try:
                self._start_explicit_flow()
            except BaseException:
                with suppress(BaseException):
                    self.close()
                raise

    def use_dst_pool(self, pool: object) -> None:
        """Wire the writer's adaptor pixelBufferPool() as VSR's dst source -
        zero-copy from VSR output straight into the encoder's queue.
        """
        if pool is None:
            raise ValueError("destination pool must not be None")
        if self._owns_dst_pool and self._dst_pool is not pool:
            _pb.flush_pool(self._dst_pool)
        self._dst_pool = pool
        self._owns_dst_pool = False

    def reset_temporal_context(self) -> None:
        """Drop the previous-frame chain. Call at scene cuts on --video input."""
        if self._flow_pending_frame is not None:
            raise RuntimeError("finish_pending_upscale() must drain explicit flow before reset")
        self._prev_src_frame = None
        self._prev_dst_frame = None
        self._vsr_needs_random = True
        if self._flow_pairs is not None:
            # A precomputed-flow VSR submission requires a non-null flow object
            # even when there is no previous frame. Keep slot zero genuinely
            # empty after a cut rather than reusing the preceding shot's field.
            zero_pair = self._flow_zero_pair
            if zero_pair is None:
                raise RuntimeError("Vision zero-flow buffers are unavailable")
            pairs = list(self._flow_pairs)
            pairs[0] = zero_pair
            self._flow_pairs = tuple(pairs)

    def flush_pools(self) -> None:
        """Release excess cached buffers in every session-owned pool.

        Pool caching is what makes hot-path buffer allocation fast, but at
        steady state the cache should be ~3 buffers. Periodic flushing
        reclaims peak-watermark allocations that the workload no longer
        needs after an early decode or processing burst.
        """
        _pb.flush_pool(self._src_pool)
        flow_src_pool = getattr(self, "_flow_src_pool", None)
        if flow_src_pool is not None:
            _pb.flush_pool(flow_src_pool)
        if self._owns_dst_pool:
            _pb.flush_pool(self._dst_pool)

    def close(self) -> None:
        processor, self.processor = self.processor, None
        flow_executor = getattr(self, "_flow_executor", None)
        vision_flow_converter = getattr(
            self,
            "_vision_flow_converter",
            None,
        )
        self._flow_executor = None
        try:
            if flow_executor is not None:
                flow_executor.shutdown(wait=True)
        finally:
            try:
                self._vision_flow_converter = None
                if vision_flow_converter is not None:
                    vision_flow_converter.close()
            finally:
                try:
                    if processor is not None:
                        processor.endSession()
                finally:
                    self._prev_src_frame = None
                    self._prev_dst_frame = None
                    self._flow_future = None
                    self._flow_pending_frame = None
                    self._flow_pending_index = None
                    self._flow_pending_slot = None
                    self._flow_pairs = None
                    self._flow_zero_pair = None
                    self._vision_flow_destinations = None
                    self.config = None
                    xfer, self._xfer = self._xfer, None
                    try:
                        if xfer is not None:
                            vt.VTPixelTransferSessionInvalidate(xfer)
                    finally:
                        self.flush_pools()
                        self._src_pool = None
                        self._flow_src_pool = None
                        self._dst_pool = None
                        self._owns_dst_pool = False

    # ------------------------------------------------------------------------
    # Internal: buffer factories
    # ------------------------------------------------------------------------

    def _make_src_buffer(self) -> object:
        if self._src_pool is None:
            raise RuntimeError("VSR source pool is unavailable")
        return _pb.pool_create_buffer_bounded(self._src_pool, self._src_pool_allocation_limit)

    def _make_dst_buffer(self) -> object:
        if self._dst_pool is None:
            raise RuntimeError("VSR destination pool is unavailable")
        if self._owns_dst_pool:
            return _pb.pool_create_buffer_bounded(self._dst_pool, DST_POOL_ALLOCATION_LIMIT)
        pb = _pb.pool_create_buffer(self._dst_pool)
        if pb is None:
            raise RuntimeError("external VSR destination pool acquisition failed")
        return pb

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def submit_upscale_to_buffer(
        self,
        frame: mx.array,
        frame_index: int,
    ) -> object | None:
        """Submit an MLX/numpy frame, possibly returning one delayed output.

        Ordinary sessions are synchronous. An ``explicit_flow=True`` balanced
        session overlaps the selected optical-flow backend for frame N+1 with
        VSR for frame N, so its second and later submissions have one frame of
        bounded latency. Call :meth:`finish_pending_upscale` at a drain or cut.
        """
        src_pb = self._upload_src_buffer(frame)
        if self._explicit_flow:
            return self._submit_explicit(src_pb, frame_index)
        return self._process(src_pb, frame_index)

    def prepare_upscale_source(self, frame: mx.array) -> object:
        """Upload one MLX frame into a ready, pool-owned source buffer.

        The streaming executor calls this on its MLX bridge lane. The returned
        CVPixelBuffer is a durable cross-affinity value; the caller retains it
        until :meth:`submit_prepared_upscale` returns.
        """
        return self._upload_src_buffer(frame)

    def submit_prepared_upscale(
        self,
        src_pb: object,
        frame_index: int,
    ) -> object | None:
        """Submit a source produced by :meth:`prepare_upscale_source`."""
        if self._explicit_flow:
            return self._submit_explicit(src_pb, frame_index)
        return self._process(src_pb, frame_index)

    def submit_upscale_buffer_to_buffer(
        self,
        src_pb: object,
        frame_index: int,
    ) -> object | None:
        """Submit a native source buffer with the same latency contract."""
        clean = self._clean_src_buffer(src_pb)
        if self._explicit_flow:
            return self._submit_explicit(clean, frame_index)
        return self._process(clean, frame_index)

    def finish_pending_upscale(self) -> object | None:
        """Finish and return the final delayed explicit-flow output, if any."""
        if not self._explicit_flow or self._flow_pending_frame is None:
            return None
        future = self._flow_future
        if future is None:
            raise RuntimeError("explicit-flow pending frame has no flow future")
        future.result()
        slot = self._flow_pending_slot
        frame_index = self._flow_pending_index
        if slot is None or frame_index is None:
            raise RuntimeError("explicit-flow pending state is incomplete")
        # Match the ordinary submit path's native lifetime. Without this
        # scoped pool, the temporary VTFrameProcessorFrame around the drained
        # output survives the hard-cut reset and can pin a bounded destination
        # surface until the process-wide autorelease pool eventually drains.
        with autorelease_pool():
            output = self._process_precomputed_frame(
                self._flow_pending_frame,
                frame_index,
                slot,
            )
        self._flow_future = None
        self._flow_pending_frame = None
        self._flow_pending_index = None
        self._flow_pending_slot = None
        return output

    def upscale_to_buffer(self, frame: mx.array, frame_index: int) -> object:
        """Upscale one frame from an MLX/numpy array. Returns the dst
        CVPixelBuffer (RGBAHalf for HQ, NV12 for LL) ready to append to AVWriter.

        The array is uploaded into a pooled source buffer in VSR's source
        format (uint8 RGB and fp16 RGBA inputs are both accepted; see
        `pixel_buffers.upload_frame_to_buffer`). For frames that already exist
        as a CVPixelBuffer in the source format - e.g. straight from a native
        decoder - use `upscale_buffer_to_buffer` to skip the upload entirely.
        """
        src_pb = self._upload_src_buffer(frame)
        return self._process(src_pb, frame_index)

    def upscale_buffer_to_buffer(self, src_pb: object, frame_index: int) -> object:
        """Upscale one frame whose source CVPixelBuffer comes from an external
        decoder already configured for the session's source format.

        The buffer is normalized into a clean VSR pool buffer via
        VTPixelTransferSession before processing, rather than fed raw. A raw
        decoder buffer can carry IOSurface/attribute quirks - e.g. for input
        whose coded size is padded to a macroblock multiple (a 544x408 clip is
        coded at 544x416) - that the VSR processor rejects with -19730, even
        though the identical pixels in a VSR pool buffer upscale fine.
        The resolved source matrix is normalized here for balanced VSR; output
        colorimetry remains owned by the encoder tag.
        """
        clean = self._clean_src_buffer(src_pb)
        return self._process(clean, frame_index)

    def _tag_source_matrix(self, src_pb: object) -> None:
        """Apply the resolved matrix attachment required by balanced VSR."""
        if self._source_matrix is None:
            return
        Quartz.CVBufferSetAttachment(
            src_pb,
            Quartz.kCVImageBufferYCbCrMatrixKey,
            self._source_matrix,
            Quartz.kCVAttachmentMode_ShouldPropagate,
        )

    def _upload_src_buffer(self, frame: mx.array) -> object:
        src_pb = self._make_src_buffer()
        _pb.upload_frame_to_buffer(frame, src_pb)
        self._tag_source_matrix(src_pb)
        return src_pb

    def _clean_src_buffer(self, src_pb: object) -> object:
        """Copy `src_pb` into a clean VSR-source pool buffer via a lazily-created
        VTPixelTransferSession, so the VSR processor accepts it (the -19730 fix; see
        upscale_buffer_to_buffer).

        Then strip the TransferFunction attachment the transfer propagated from the
        source. The VSR scaler HONORS that tag: it linearizes the input through the
        (709) EOTF but never re-encodes, darkening the output by ~2x (measured:
        709-EOTF(0.39)=0.166, a hard black crush on tagged SD clips).

        The balanced Video model separately honors the YCbCr matrix attachment even
        on RGBAHalf input. Apply the pipeline's resolved source matrix after the
        transfer so tagged and untagged sources use the same modeled interpretation.
        Fast/Image pass no matrix override and retain their existing behavior.
        Output colorimetry comes from the encoder tag, not these input attachments.
        """
        if self._xfer is None:
            err, xfer = vt.VTPixelTransferSessionCreate(None, None)
            if err != 0 or xfer is None:
                raise RuntimeError(f"VTPixelTransferSessionCreate failed: {err}")
            self._xfer = xfer
        clean = self._make_src_buffer()
        err = vt.VTPixelTransferSessionTransferImage(self._xfer, src_pb, clean)
        if err != 0:
            raise RuntimeError(f"VTPixelTransferSessionTransferImage failed: {err}")
        Quartz.CVBufferRemoveAttachment(clean, Quartz.kCVImageBufferTransferFunctionKey)
        self._tag_source_matrix(clean)
        return clean

    @staticmethod
    def _zero_flow_buffer(buffer: object) -> None:
        Quartz.CVPixelBufferLockBaseAddress(buffer, 0)
        try:
            row_bytes = int(Quartz.CVPixelBufferGetBytesPerRow(buffer))
            height = int(Quartz.CVPixelBufferGetHeight(buffer))
            view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(row_bytes * height)
            view[:] = bytes(len(view))
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(buffer, 0)

    @classmethod
    def _zero_flow_pair(cls, pair: tuple[object, object]) -> None:
        cls._zero_flow_buffer(pair[0])
        cls._zero_flow_buffer(pair[1])

    def _start_explicit_flow(self) -> None:
        """Start the explicit Vision flow used by balanced VSR."""
        self._start_vision_flow()
        self._flow_src_pool = _pb.make_bounded_pool_from_attrs(
            self.src_attrs,
            EXPLICIT_FLOW_SOURCE_POOL_LIMIT,
        )
        if self._flow_src_pool is None:
            raise RuntimeError("explicit-flow source-isolation pool creation failed")

    def _start_vision_flow(self) -> None:
        """Prepare full-estimate Vision flow in VT SR's advertised geometry.

        Vision emits a source-sized field in source-pixel units. VT SR accepts
        that IOSurface but interprets it in its smaller advertised coordinate
        system, which over-warps the result. Keep Vision's better-conditioned
        full-resolution estimate, then resample it and scale its vectors into
        the raw VT grid on Metal. Portrait fields are also rotated
        counterclockwise into VT's landscape coordinates.

        The advertised grid is a quarter of the network canvas VT selects for
        the source size. A source that fits that canvas is placed unscaled at
        its top-left, so the field maps one grid cell per 4x4 source pixels
        (measured on macOS 27 against ground truth: the stretched mapping
        scored at or below no flow at such sizes). A larger source is resized
        onto the canvas, so it keeps the stretched mapping.

        On macOS 26.5.2, decoded output was identical with a zero forward field,
        so every slot reuses one immutable zero surface and retains only
        Vision's converted current-to-previous result. This remains an
        OS-dependent measured behavior, not a public API guarantee.
        """

        # The scaler publishes no flow-buffer attributes of its own; the
        # optical-flow configuration defines the explicit-flow contract. Only
        # its advertised attributes are read; no flow session is started.
        cls = vt.VTOpticalFlowConfiguration
        config = cls.alloc().initWithFrameWidth_frameHeight_qualityPrioritization_revision_(
            self.in_w,
            self.in_h,
            vt.VTOpticalFlowConfigurationQualityPrioritizationQuality,
            cls.defaultRevision(),
        )
        if config is None:
            raise RuntimeError(
                f"VT flow geometry configuration returned nil for Vision {self.in_w}x{self.in_h}"
            )
        attrs = dict(config.destinationPixelBufferAttributes() or {})
        flow_w, flow_h = advertised_flow_destination_geometry(attrs)
        if _pb.resolve_pixel_format(attrs) != FLOW_16H:
            raise RuntimeError(
                "VT flow geometry configuration did not advertise TwoComponent16Half"
            )
        attrs[Quartz.kCVPixelBufferMetalCompatibilityKey] = True
        zero_pair = (
            _pb.make_pixel_buffer_from_attrs(flow_w, flow_h, attrs),
            _pb.make_pixel_buffer_from_attrs(flow_w, flow_h, attrs),
        )
        destinations = tuple(
            _pb.make_pixel_buffer_from_attrs(flow_w, flow_h, attrs)
            for _ in range(EXPLICIT_FLOW_PAIR_COUNT)
        )
        self._zero_flow_pair(zero_pair)
        self._flow_zero_pair = zero_pair
        self._vision_flow_destinations = destinations
        self._flow_pairs = tuple(zero_pair for _ in range(EXPLICIT_FLOW_PAIR_COUNT))
        portrait = self.in_h > self.in_w
        oriented_w, oriented_h = (self.in_h, self.in_w) if portrait else (self.in_w, self.in_h)
        # The canvas is 4x the grid up to flooring (a 270-row canvas has a
        # 67-row grid), hence the 3-pixel allowance.
        unscaled = oriented_w <= 4 * flow_w + 3 and oriented_h <= 4 * flow_h + 3
        self._vision_flow_converter = VisionFlowToVtConverter(
            self.in_w,
            self.in_h,
            flow_w,
            flow_h,
            rotate_counterclockwise=portrait,
            unscaled=unscaled,
        )
        self._flow_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vsr-vision-flow",
        )
        _log.info(
            "VSR explicit flow ready (Vision revision 1 %s, "
            "current-to-previous only, full estimate %sx%s -> VT grid %sx%s "
            "%s on Metal%s, one-frame overlap)",
            VISION_VSR_ACCURACY.title(),
            self.in_w,
            self.in_h,
            flow_w,
            flow_h,
            "unscaled" if unscaled else "stretched",
            ", portrait CCW" if portrait else "",
        )

    def _flow_object(self, slot: int) -> object:
        pairs = self._flow_pairs
        if pairs is None:
            raise RuntimeError("explicit optical-flow buffers are unavailable")
        forward, backward = pairs[slot]
        optical_flow = vt.VTFrameProcessorOpticalFlow.alloc().initWithForwardFlow_backwardFlow_(
            forward, backward
        )
        if optical_flow is None:
            raise RuntimeError("VTFrameProcessorOpticalFlow rejected explicit flow buffers")
        return optical_flow

    def _isolate_flow_source_frame(self, source_frame: _ProcessorFrame) -> _ProcessorFrame:
        """Snapshot one source into flow-only storage before overlap.

        During the one-frame pipeline, VSR consumes frame N while the selected
        explicit backend computes N -> N+1. Passing frame N's IOSurface to both
        native sessions concurrently can intermittently corrupt the VSR result
        even though neither Python call writes it. A bounded copy keeps the
        pixels and public attachments identical while giving flow a distinct
        IOSurface and lifetime.
        """
        pool = self._flow_src_pool
        if pool is None:
            raise RuntimeError("explicit-flow source-isolation pool is unavailable")
        isolated = _pb.pool_create_buffer_bounded(
            pool,
            EXPLICIT_FLOW_SOURCE_POOL_LIMIT,
        )
        _pb.copy_pixel_buffer_into(source_frame.buffer(), isolated)
        frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            isolated,
            source_frame.presentationTimeStamp(),
        )
        if frame is None:
            raise RuntimeError("isolated explicit-flow source frame init returned nil")
        return cast(_ProcessorFrame, frame)

    def _run_explicit_flow(
        self,
        previous_frame: _ProcessorFrame,
        current_frame: _ProcessorFrame,
        slot: int,
    ) -> None:

        with autorelease_pool():
            zero_pair = self._flow_zero_pair
            if zero_pair is None:
                raise RuntimeError("Vision zero-flow buffers are unavailable")
            backward = generate_vision_flow(
                current_frame.buffer(),
                previous_frame.buffer(),
                accuracy=VISION_VSR_ACCURACY,
            )
            converter = self._vision_flow_converter
            destinations = self._vision_flow_destinations
            if converter is None or destinations is None:
                raise RuntimeError("Vision flow conversion resources are unavailable")
            converted = destinations[slot]
            converter.convert(backward, converted)
            pairs = self._flow_pairs
            if pairs is None:
                raise RuntimeError("explicit optical-flow buffers are unavailable")
            updated = list(pairs)
            updated[slot] = (zero_pair[0], converted)
            self._flow_pairs = tuple(updated)

    def _start_flow_future(
        self,
        previous_frame: _ProcessorFrame,
        current_frame: _ProcessorFrame,
        slot: int,
    ) -> Future[None]:
        executor = self._flow_executor
        if executor is None:
            raise RuntimeError("explicit optical-flow executor is unavailable")
        isolated_previous = self._isolate_flow_source_frame(previous_frame)
        return executor.submit(
            self._run_explicit_flow,
            isolated_previous,
            current_frame,
            slot,
        )

    def _process_precomputed_frame(
        self,
        source_frame: _ProcessorFrame,
        frame_index: int,
        flow_slot: int,
    ) -> object:
        dst_pb = self._make_dst_buffer()
        pts = _pb.frame_pts(frame_index, self.fps)
        dst_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            dst_pb, pts
        )
        submission_mode = (
            vt.VTSuperResolutionScalerParametersSubmissionModeRandom
            if self._vsr_needs_random
            else vt.VTSuperResolutionScalerParametersSubmissionModeSequential
        )
        params = vt.VTSuperResolutionScalerParameters.alloc().initWithSourceFrame_previousFrame_previousOutputFrame_opticalFlow_submissionMode_destinationFrame_(
            source_frame,
            self._prev_src_frame,
            self._prev_dst_frame,
            self._flow_object(flow_slot),
            submission_mode,
            dst_frame,
        )
        ok, err = self.processor.processWithParameters_error_(params, None)
        if not ok:
            raise RuntimeError(f"precomputed-flow VSR failed at frame {frame_index}: {err}")
        self._vsr_needs_random = False
        self._prev_src_frame = source_frame
        self._prev_dst_frame = dst_frame
        return dst_pb

    def _submit_explicit(self, src_pb: object, frame_index: int) -> object | None:
        with autorelease_pool():
            pts = _pb.frame_pts(frame_index, self.fps)
            source_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                src_pb, pts
            )
            if self._prev_src_frame is None:
                return self._process_precomputed_frame(
                    source_frame,
                    frame_index,
                    0,
                )

            if self._flow_pending_frame is None:
                slot = 0
                self._flow_future = self._start_flow_future(
                    self._prev_src_frame,
                    source_frame,
                    slot,
                )
                self._flow_pending_frame = source_frame
                self._flow_pending_index = frame_index
                self._flow_pending_slot = slot
                return None

            future = self._flow_future
            if future is None:
                raise RuntimeError("explicit-flow pending frame has no flow future")
            future.result()
            completed_frame = self._flow_pending_frame
            completed_index = self._flow_pending_index
            completed_slot = self._flow_pending_slot
            if completed_index is None or completed_slot is None:
                raise RuntimeError("explicit-flow pending state is incomplete")

            next_slot = 1 - completed_slot
            next_future = self._start_flow_future(
                completed_frame,
                source_frame,
                next_slot,
            )
            self._flow_future = next_future
            self._flow_pending_frame = source_frame
            self._flow_pending_index = frame_index
            self._flow_pending_slot = next_slot
            return self._process_precomputed_frame(
                completed_frame,
                completed_index,
                completed_slot,
            )

    def _process(self, src_pb: object, frame_index: int) -> object:
        """Run VSR on a ready source CVPixelBuffer; return the dst buffer.

        Shared tail of both upscale entry points: allocates the dst buffer,
        wraps src/dst as VTFrameProcessorFrames, builds the mode-appropriate
        parameters (threading the prev-frame chain for balanced), and advances
        that chain. The prev VTFrameProcessorFrame retains its CVPixelBuffer,
        so an externally-supplied src buffer stays valid across the one
        iteration balanced mode references it.
        """
        if self._explicit_flow:
            raise RuntimeError(
                "explicit-flow sessions require submit_upscale_to_buffer() "
                "or submit_upscale_buffer_to_buffer()"
            )
        with autorelease_pool():
            dst_pb = self._make_dst_buffer()
            pts = _pb.frame_pts(frame_index, self.fps)
            src_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                src_pb, pts
            )
            dst_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                dst_pb, pts
            )

            if self.mode == "fast":
                params = vt.VTLowLatencySuperResolutionScalerParameters.alloc().initWithSourceFrame_destinationFrame_(
                    src_frame, dst_frame
                )
            else:
                use_temporal = self._temporal_video
                params = vt.VTSuperResolutionScalerParameters.alloc().initWithSourceFrame_previousFrame_previousOutputFrame_opticalFlow_submissionMode_destinationFrame_(
                    src_frame,
                    self._prev_src_frame if use_temporal else None,
                    self._prev_dst_frame if use_temporal else None,
                    None,
                    vt.VTSuperResolutionScalerParametersSubmissionModeSequential,
                    dst_frame,
                )

            ok, err = self.processor.processWithParameters_error_(params, None)
            if not ok:
                raise RuntimeError(
                    f"VSR processWithParameters failed at frame {frame_index}: {err}"
                )
            if self._temporal_video:
                self._prev_src_frame = src_frame
                self._prev_dst_frame = dst_frame
            else:
                self.reset_temporal_context()
                del src_frame, dst_frame
            del params
        return dst_pb
