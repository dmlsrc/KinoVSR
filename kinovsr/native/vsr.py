"""VideoToolbox Super Resolution (spatial upscale) session wrapper.

`VsrSession` wraps VTSuperResolutionScalerConfiguration (HQ, scale=4) or
VTLowLatencySuperResolutionScalerConfiguration (LL, scale=2) plus its
VTFrameProcessor and the source/dst CVPixelBufferPools. The caller hands
in a frame (uint8 RGB or fp16 RGBA) and gets back a destination buffer
ready to feed straight into AVAssetWriter.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from typing import Any

from kinovsr.media import pixel_buffers as _pb
from kinovsr.settings import default_settings

from .frameworks import Quartz, autorelease_pool, vt
from .optical_flow import (
    flow_destination_geometry,
    mark_flow_pair_pending,
    require_flow_pair_written,
    source_sized_flow_is_reliable,
)

_log = logging.getLogger(__name__)
_NATIVE_STDERR_LOCK = threading.RLock()


def _duplicate_stderr() -> int:
    return os.dup(2)


def _open_devnull() -> int:
    return os.open(os.devnull, os.O_WRONLY)


def _redirect_stderr(source: int) -> None:
    os.dup2(source, 2)


def _close_fd(fd: int) -> None:
    os.close(fd)


@contextmanager
def _suppress_native_stderr():
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

        def _cleanup(label: str, fn: Any, fd: int) -> None:
            try:
                fn(fd)
            except BaseException as exc:  # noqa: BLE001 - collected below
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
                    primary.add_note(
                        f"{label} also failed: "
                        f"{type(failure).__name__}: {failure}")
            raise primary


def scale_for_mode(mode: str) -> int:
    """Map a VSR spatial mode to its forced scale factor.

    VideoToolbox couples the spatial-mode choice to the scale: LowLatency
    is 2x-only, the HQ classes are 4x-only.  Centralized here so call sites
    don't reinvent the mapping.
    """
    if mode == "fast":
        return 2
    if mode in ("balanced", "image", "basicvsrpp", "realbasicvsr", "realesrgan", "safmn", "esc", "realviformer", "realplksr", "toflow"):
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
                f"--upscale fast at {width}x{height} supports scale={ok}, "
                f"requested scale={scale}."
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
                if fits_fast else
                f"This input is larger than any VSR mode supports; downscale it to "
                f"<= {HQ_MAX_INPUT_W}x{HQ_MAX_INPUT_H} (balanced/image) or <= 960x960 (fast) first."
            )
            raise SystemExit(
                f"--upscale {mode} (4x) does not support {width}x{height} input "
                f"(max {HQ_MAX_INPUT_W}x{HQ_MAX_INPUT_H}; a 4x output would exceed 8K). {hint}"
            )


def _wait_for_model_download(config: Any) -> None:
    """Block until HQ VSR's downloadable model is ready, printing progress."""
    status = config.configurationModelStatus()
    if status == vt.VTSuperResolutionScalerConfigurationModelStatusReady:
        return
    _log.info("VSR model not ready (status=%s); requesting download", status)
    done = threading.Event()
    err_box: list[Any] = [None]

    def completion(error):
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
                  frame-to-frame variation than image mode. When explicit
                  public flow is requested below its reliable 128x128 writer
                  boundary, the session uses deterministic Image input instead
                  of silently feeding invalid flow or reverting to the
                  nondeterministic internal Video-flow path.
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
    ):
        if mode not in ("fast", "balanced", "image"):
            raise ValueError(f"VsrSession only supports VideoToolbox modes, got {mode!r}")
        if explicit_flow and mode != "balanced":
            raise ValueError("explicit optical flow is only valid for balanced VSR")
        scale = scale_for_mode(mode)
        _validate_combination(in_w, in_h, scale, mode)
        self.in_w, self.in_h = in_w, in_h
        self.scale = scale
        self.out_w, self.out_h = in_w * scale, in_h * scale
        self.mode = mode
        self.fps = float(fps)
        explicit_geometry_ok = source_sized_flow_is_reliable(in_w, in_h)
        self._image_fallback = bool(
            explicit_flow
            and mode == "balanced"
            and not explicit_geometry_ok
        )
        self._temporal_video = mode == "balanced" and not self._image_fallback
        self._explicit_flow = bool(
            explicit_flow
            and self._temporal_video
            and explicit_geometry_ok
        )
        if self._image_fallback:
            _log.info(
                "VSR balanced request at %sx%s uses deterministic Image input: "
                "VT can silently leave source-sized fields below 128 pixels "
                "untouched, and the internal Video-flow path is "
                "nondeterministic",
                in_w,
                in_h,
            )
        self._flow_config: Any = None
        self._flow_processor: Any = None
        self._flow_pairs: tuple[tuple[Any, Any], ...] | None = None
        self._flow_executor: ThreadPoolExecutor | None = None
        self._flow_future: Future[None] | None = None
        self._flow_pending_frame: Any = None
        self._flow_pending_index: int | None = None
        self._flow_pending_slot: int | None = None
        self._flow_needs_random = True
        self._vsr_needs_random = True

        if mode == "fast":
            self.config = vt.VTLowLatencySuperResolutionScalerConfiguration.alloc(
            ).initWithFrameWidth_frameHeight_scaleFactor_(in_w, in_h, float(scale))
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
                in_w, in_h, scale, input_type, self._explicit_flow,
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
            "VSR session ready (mode=%s, %sx%s -> %sx%s, src fmt %#x, "
            "dst fmt %#x)",
            mode,
            in_w,
            in_h,
            self.out_w,
            self.out_h,
            _pb.resolve_pixel_format(self.src_attrs),
            _pb.resolve_pixel_format(self.dst_attrs),
        )

        self._prev_src_frame: Any = None
        self._prev_dst_frame: Any = None

        # Lazily-created pixel-transfer session, used by
        # upscale_buffer_to_buffer to normalize externally-decoded buffers.
        self._xfer: Any = None

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
        self._dst_pool = _pb.make_bounded_pool_from_attrs(
            self.dst_attrs, DST_POOL_ALLOCATION_LIMIT
        )
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

    def use_dst_pool(self, pool: Any) -> None:
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
            raise RuntimeError(
                "finish_pending_upscale() must drain explicit flow before reset"
            )
        self._prev_src_frame = None
        self._prev_dst_frame = None
        self._flow_needs_random = True
        self._vsr_needs_random = True
        if self._flow_pairs is not None:
            # A precomputed-flow VSR submission requires a non-null flow object
            # even when there is no previous frame. Keep slot zero genuinely
            # empty after a cut rather than reusing the preceding shot's field.
            self._zero_flow_pair(self._flow_pairs[0])

    def flush_pools(self) -> None:
        """Release excess cached buffers in every session-owned pool.

        Pool caching is what makes hot-path buffer allocation fast, but at
        steady state the cache should be ~3 buffers. Periodic flushing
        reclaims peak-watermark allocations that the workload no longer
        needs after an early decode or processing burst.
        """
        _pb.flush_pool(self._src_pool)
        if self._owns_dst_pool:
            _pb.flush_pool(self._dst_pool)

    def close(self) -> None:
        processor, self.processor = self.processor, None
        flow_processor = getattr(self, "_flow_processor", None)
        flow_executor = getattr(self, "_flow_executor", None)
        self._flow_processor = None
        self._flow_executor = None
        try:
            if flow_executor is not None:
                flow_executor.shutdown(wait=True)
        finally:
            try:
                if flow_processor is not None:
                    flow_processor.endSession()
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
                    self._flow_config = None
                    self.config = None
                    xfer, self._xfer = self._xfer, None
                    try:
                        if xfer is not None:
                            vt.VTPixelTransferSessionInvalidate(xfer)
                    finally:
                        self.flush_pools()
                        self._src_pool = None
                        self._dst_pool = None
                        self._owns_dst_pool = False

    # ------------------------------------------------------------------------
    # Internal: buffer factories
    # ------------------------------------------------------------------------

    def _make_src_buffer(self) -> Any:
        if self._src_pool is None:
            raise RuntimeError("VSR source pool is unavailable")
        return _pb.pool_create_buffer_bounded(
            self._src_pool, self._src_pool_allocation_limit
        )

    def _make_dst_buffer(self) -> Any:
        if self._dst_pool is None:
            raise RuntimeError("VSR destination pool is unavailable")
        if self._owns_dst_pool:
            return _pb.pool_create_buffer_bounded(
                self._dst_pool, DST_POOL_ALLOCATION_LIMIT
            )
        pb = _pb.pool_create_buffer(self._dst_pool)
        if pb is None:
            raise RuntimeError("external VSR destination pool acquisition failed")
        return pb

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def submit_upscale_to_buffer(
        self,
        frame: Any,
        frame_index: int,
    ) -> Any | None:
        """Submit an MLX/numpy frame, possibly returning one delayed output.

        Ordinary sessions are synchronous. An ``explicit_flow=True`` balanced
        session overlaps public Quality optical flow for frame N+1 with VSR for
        frame N, so its second and later submissions have one frame of bounded
        latency. Call :meth:`finish_pending_upscale` at a drain or cut.
        """
        src_pb = self._make_src_buffer()
        _pb.upload_frame_to_buffer(frame, src_pb)
        if self._explicit_flow:
            return self._submit_explicit(src_pb, frame_index)
        return self._process(src_pb, frame_index)

    def submit_upscale_buffer_to_buffer(
        self,
        src_pb: Any,
        frame_index: int,
    ) -> Any | None:
        """Submit a native source buffer with the same latency contract."""
        clean = self._clean_src_buffer(src_pb)
        if self._explicit_flow:
            return self._submit_explicit(clean, frame_index)
        return self._process(clean, frame_index)

    def finish_pending_upscale(self) -> Any | None:
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

    def upscale_to_buffer(self, frame: Any, frame_index: int) -> Any:
        """Upscale one frame from an MLX/numpy array. Returns the dst
        CVPixelBuffer (RGBAHalf for HQ, NV12 for LL) ready to append to AVWriter.

        The array is uploaded into a pooled source buffer in VSR's source
        format (uint8 RGB and fp16 RGBA inputs are both accepted; see
        `pixel_buffers.upload_frame_to_buffer`). For frames that already exist
        as a CVPixelBuffer in the source format - e.g. straight from a native
        decoder - use `upscale_buffer_to_buffer` to skip the upload entirely.
        """
        src_pb = self._make_src_buffer()
        _pb.upload_frame_to_buffer(frame, src_pb)
        return self._process(src_pb, frame_index)

    def upscale_buffer_to_buffer(self, src_pb: Any, frame_index: int) -> Any:
        """Upscale one frame whose source CVPixelBuffer comes from an external
        decoder already configured for the session's source format.

        The buffer is normalized into a clean VSR pool buffer via
        VTPixelTransferSession before processing, rather than fed raw. A raw
        decoder buffer can carry IOSurface/attribute quirks - e.g. for input
        whose coded size is padded to a macroblock multiple (a 544x408 clip is
        coded at 544x416) - that the VSR processor rejects with -19730, even
        though the identical pixels in a VSR pool buffer upscale fine.
        Colorimetry is owned by the decode (see video_reader) and the encoder
        tag, not normalized here.
        """
        clean = self._clean_src_buffer(src_pb)
        return self._process(clean, frame_index)

    def _clean_src_buffer(self, src_pb: Any) -> Any:
        """Copy `src_pb` into a clean VSR-source pool buffer via a lazily-created
        VTPixelTransferSession, so the VSR processor accepts it (the -19730 fix; see
        upscale_buffer_to_buffer).

        Then strip the TransferFunction attachment the transfer propagated from the
        source. The VSR scaler HONORS that tag: it linearizes the input through the
        (709) EOTF but never re-encodes, darkening the output by ~2x (measured:
        709-EOTF(0.39)=0.166, a hard black crush on tagged SD clips). The MLX upload
        path (upscale_to_buffer) feeds an untagged buffer and is unaffected, which
        is why fastdvdnet stays correct -- this makes the native buffer-to-buffer path
        match it. Only the gamma tag is removed: the YCbCr matrix / range are left
        intact for the NV12 (fast) path's YUV interpretation, and output
        colorimetry comes from the encoder tag, not these scaler-input attachments.
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
        return clean

    @staticmethod
    def _zero_flow_buffer(buffer: Any) -> None:
        Quartz.CVPixelBufferLockBaseAddress(buffer, 0)
        try:
            row_bytes = int(Quartz.CVPixelBufferGetBytesPerRow(buffer))
            height = int(Quartz.CVPixelBufferGetHeight(buffer))
            view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(
                row_bytes * height
            )
            view[:] = bytes(len(view))
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(buffer, 0)

    @classmethod
    def _zero_flow_pair(cls, pair: tuple[Any, Any]) -> None:
        cls._zero_flow_buffer(pair[0])
        cls._zero_flow_buffer(pair[1])

    def _start_explicit_flow(self) -> None:
        """Start the public Quality-flow processor used by balanced VSR.

        VT advertises quarter-resolution destination attributes, but on the
        current macOS implementation those buffers complete successfully while
        remaining zero. Full-resolution buffers in VT's rotation-normalized
        geometry receive the real vector field and are accepted by
        precomputed-flow VSR. Keep that measured geometry explicit here;
        silently returning to the advertised shape is a quality regression.
        """
        cls = vt.VTOpticalFlowConfiguration
        if not cls.isSupported():
            raise RuntimeError("VTOpticalFlow is not supported on this device")
        config = cls.alloc(
        ).initWithFrameWidth_frameHeight_qualityPrioritization_revision_(
            self.in_w,
            self.in_h,
            vt.VTOpticalFlowConfigurationQualityPrioritizationQuality,
            cls.defaultRevision(),
        )
        if config is None:
            raise RuntimeError(
                f"VTOpticalFlow config init returned nil for "
                f"{self.in_w}x{self.in_h}"
            )
        processor = vt.VTFrameProcessor.alloc().init()
        with _suppress_native_stderr():
            ok, err = processor.startSessionWithConfiguration_error_(config, None)
        if not ok:
            raise RuntimeError(f"VTOpticalFlow startSession failed: {err}")
        self._flow_config = config
        self._flow_processor = processor

        attrs = dict(config.destinationPixelBufferAttributes() or {})
        flow_w, flow_h = flow_destination_geometry(self.in_w, self.in_h)
        pairs = []
        for _ in range(EXPLICIT_FLOW_PAIR_COUNT):
            pairs.append(
                (
                    _pb.make_pixel_buffer_from_attrs(flow_w, flow_h, attrs),
                    _pb.make_pixel_buffer_from_attrs(flow_w, flow_h, attrs),
                )
            )
        self._flow_pairs = tuple(pairs)
        self._zero_flow_pair(self._flow_pairs[0])
        self._flow_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vsr-explicit-flow",
        )
        _log.info(
            "VSR explicit flow ready (Quality, full-resolution %sx%s, "
            "one-frame overlap)",
            flow_w,
            flow_h,
        )

    def _flow_object(self, slot: int) -> Any:
        pairs = self._flow_pairs
        if pairs is None:
            raise RuntimeError("explicit optical-flow buffers are unavailable")
        forward, backward = pairs[slot]
        return vt.VTFrameProcessorOpticalFlow.alloc(
        ).initWithForwardFlow_backwardFlow_(forward, backward)

    def _run_explicit_flow(
        self,
        previous_frame: Any,
        current_frame: Any,
        slot: int,
        submission_mode: int,
        frame_index: int,
    ) -> None:
        processor = self._flow_processor
        if processor is None:
            raise RuntimeError("explicit optical-flow processor is unavailable")
        with autorelease_pool():
            pairs = self._flow_pairs
            if pairs is None:
                raise RuntimeError("explicit optical-flow buffers are unavailable")
            pair = pairs[slot]
            mark_flow_pair_pending(pair)
            optical_flow = self._flow_object(slot)
            params = vt.VTOpticalFlowParameters.alloc(
            ).initWithSourceFrame_nextFrame_submissionMode_destinationOpticalFlow_(
                previous_frame,
                current_frame,
                submission_mode,
                optical_flow,
            )
            ok, err = processor.processWithParameters_error_(params, None)
            if not ok:
                raise RuntimeError(
                    f"VTOpticalFlow process failed at frame "
                    f"{frame_index}: {err}"
                )
            require_flow_pair_written(
                pair,
                context=f"balanced VSR frame {frame_index}",
            )

    def _start_flow_future(
        self,
        previous_frame: Any,
        current_frame: Any,
        slot: int,
        frame_index: int,
    ) -> Future[None]:
        executor = self._flow_executor
        if executor is None:
            raise RuntimeError("explicit optical-flow executor is unavailable")
        submission_mode = (
            vt.VTOpticalFlowParametersSubmissionModeRandom
            if self._flow_needs_random
            else vt.VTOpticalFlowParametersSubmissionModeSequential
        )
        self._flow_needs_random = False
        return executor.submit(
            self._run_explicit_flow,
            previous_frame,
            current_frame,
            slot,
            submission_mode,
            frame_index,
        )

    def _process_precomputed_frame(
        self,
        source_frame: Any,
        frame_index: int,
        flow_slot: int,
    ) -> Any:
        dst_pb = self._make_dst_buffer()
        pts = _pb.frame_pts(frame_index, self.fps)
        dst_frame = vt.VTFrameProcessorFrame.alloc(
        ).initWithBuffer_presentationTimeStamp_(dst_pb, pts)
        submission_mode = (
            vt.VTSuperResolutionScalerParametersSubmissionModeRandom
            if self._vsr_needs_random
            else vt.VTSuperResolutionScalerParametersSubmissionModeSequential
        )
        params = vt.VTSuperResolutionScalerParameters.alloc(
        ).initWithSourceFrame_previousFrame_previousOutputFrame_opticalFlow_submissionMode_destinationFrame_(
            source_frame,
            self._prev_src_frame,
            self._prev_dst_frame,
            self._flow_object(flow_slot),
            submission_mode,
            dst_frame,
        )
        ok, err = self.processor.processWithParameters_error_(params, None)
        if not ok:
            raise RuntimeError(
                f"precomputed-flow VSR failed at frame {frame_index}: {err}"
            )
        self._vsr_needs_random = False
        self._prev_src_frame = source_frame
        self._prev_dst_frame = dst_frame
        return dst_pb

    def _submit_explicit(self, src_pb: Any, frame_index: int) -> Any | None:
        with autorelease_pool():
            pts = _pb.frame_pts(frame_index, self.fps)
            source_frame = vt.VTFrameProcessorFrame.alloc(
            ).initWithBuffer_presentationTimeStamp_(src_pb, pts)
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
                    frame_index,
                )
                self._flow_pending_frame = source_frame
                self._flow_pending_index = frame_index
                self._flow_pending_slot = slot
                return None

            future = self._flow_future
            if future is None:
                raise RuntimeError(
                    "explicit-flow pending frame has no flow future"
                )
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
                frame_index,
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

    def _process(self, src_pb: Any, frame_index: int) -> Any:
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
            src_frame = vt.VTFrameProcessorFrame.alloc(
            ).initWithBuffer_presentationTimeStamp_(src_pb, pts)
            dst_frame = vt.VTFrameProcessorFrame.alloc(
            ).initWithBuffer_presentationTimeStamp_(dst_pb, pts)

            if self.mode == "fast":
                params = vt.VTLowLatencySuperResolutionScalerParameters.alloc(
                ).initWithSourceFrame_destinationFrame_(src_frame, dst_frame)
            else:
                use_temporal = self._temporal_video
                params = vt.VTSuperResolutionScalerParameters.alloc(
                ).initWithSourceFrame_previousFrame_previousOutputFrame_opticalFlow_submissionMode_destinationFrame_(
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
