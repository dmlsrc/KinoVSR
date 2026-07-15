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
import signal
import sys
import threading
from contextlib import contextmanager, suppress
from typing import Any

from kinovsr.media import pixel_buffers as _pb
from kinovsr.settings import default_settings

from .frameworks import Quartz, autorelease_pool, vt

_log = logging.getLogger(__name__)
_NATIVE_STDERR_LOCK = threading.RLock()
_SYNCHRONOUS_SIGNAL_NAMES = (
    "SIGBUS",
    "SIGEMT",
    "SIGFPE",
    "SIGILL",
    "SIGSEGV",
    "SIGSYS",
    "SIGTRAP",
)
_FD_ACQUISITION_SIGNALS = signal.valid_signals()
_FD_ACQUISITION_SIGNALS.discard(signal.SIGKILL)
_FD_ACQUISITION_SIGNALS.discard(signal.SIGSTOP)
for _signal_name in _SYNCHRONOUS_SIGNAL_NAMES:
    _signal_number = getattr(signal, _signal_name, None)
    if _signal_number is not None:
        _FD_ACQUISITION_SIGNALS.discard(_signal_number)
_FD_ACQUISITION_SIGNALS = frozenset(_FD_ACQUISITION_SIGNALS)


def _duplicate_stderr() -> int:
    return os.dup(2)


def _open_devnull() -> int:
    return os.open(os.devnull, os.O_WRONLY)


def _redirect_stderr(source: int) -> None:
    os.dup2(source, 2)


def _close_fd(fd: int) -> None:
    os.close(fd)


@contextmanager
def _defer_fd_acquisition_signals():
    """Defer asynchronous signal handlers until an fd owner is recorded."""
    previous = signal.pthread_sigmask(
        signal.SIG_BLOCK, _FD_ACQUISITION_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


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
        saved_fd: int | None = None
        devnull_fd: int | None = None
        failures: list[tuple[str, BaseException]] = []
        try:
            try:
                # A Python signal handler can run after dup/open returns but
                # before the assignment records ownership. Keep asynchronous
                # signals masked through that publication; the surrounding
                # finally is already armed before either acquisition starts.
                with _defer_fd_acquisition_signals():
                    saved_fd = _duplicate_stderr()
            except BaseException as exc:
                failures.append(("duplicate stderr", exc))
            if saved_fd is not None and not failures:
                try:
                    with _defer_fd_acquisition_signals():
                        devnull_fd = _open_devnull()
                except BaseException as exc:
                    failures.append(("open /dev/null", exc))
            if devnull_fd is not None and not failures:
                try:
                    _redirect_stderr(devnull_fd)
                except BaseException as exc:
                    failures.append(("redirect stderr", exc))
                else:
                    try:
                        yield
                    except BaseException as exc:
                        failures.append(("suppressed body", exc))

                # Once /dev/null was acquired, redirect may have completed
                # before reporting an exception. Always restore fd 2 before
                # either owned descriptor is closed.
                try:
                    _redirect_stderr(saved_fd)
                except BaseException as exc:
                    failures.append(("restore stderr", exc))
        finally:
            try:
                if devnull_fd is not None:
                    owned_devnull, devnull_fd = devnull_fd, None
                    try:
                        _close_fd(owned_devnull)
                    except BaseException as exc:
                        failures.append(("close /dev/null", exc))
            finally:
                if saved_fd is not None:
                    owned_saved, saved_fd = saved_fd, None
                    try:
                        _close_fd(owned_saved)
                    except BaseException as exc:
                        failures.append(("close saved stderr", exc))

        if failures:
            _, primary = failures[0]
            for label, failure in failures[1:]:
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
                  frame-to-frame variation than image mode.
      "image"     VTSuperResolutionScalerConfiguration InputType=Image. scale=4.
                  RGBAHalf source. Per-frame deterministic upscale, no
                  prev-frame feedback.  Apple documents this as for stills,
                  but on real video it produces measurably lower temporal
                  second-difference than balanced - a legitimate alternative
                  if you prefer the smoother / less-edge-boosted trade-off.

    The previous-frame state can be reset at hard cuts via
    `reset_temporal_context()` - useful for input that may contain edits.
    """

    def __init__(self, in_w: int, in_h: int, mode: str, fps: float = 24.0):
        if mode not in ("fast", "balanced", "image"):
            raise ValueError(f"VsrSession only supports VideoToolbox modes, got {mode!r}")
        scale = scale_for_mode(mode)
        _validate_combination(in_w, in_h, scale, mode)
        self.in_w, self.in_h = in_w, in_h
        self.scale = scale
        self.out_w, self.out_h = in_w * scale, in_h * scale
        self.mode = mode
        self.fps = float(fps)

        if mode == "fast":
            self.config = vt.VTLowLatencySuperResolutionScalerConfiguration.alloc(
            ).initWithFrameWidth_frameHeight_scaleFactor_(in_w, in_h, float(scale))
            if self.config is None:
                raise RuntimeError("LowLatency VSR config init returned nil")
        else:
            input_type = (
                vt.VTSuperResolutionScalerConfigurationInputTypeVideo
                if mode == "balanced"
                else vt.VTSuperResolutionScalerConfigurationInputTypeImage
            )
            cls = vt.VTSuperResolutionScalerConfiguration
            self.config = cls.alloc().initWithFrameWidth_frameHeight_scaleFactor_inputType_usePrecomputedFlow_qualityPrioritization_revision_(
                in_w, in_h, scale, input_type, False,
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
            if mode == "balanced"
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
        self._prev_src_frame = None
        self._prev_dst_frame = None

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
        try:
            if processor is not None:
                processor.endSession()
        finally:
            self.reset_temporal_context()
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

    def _process(self, src_pb: Any, frame_index: int) -> Any:
        """Run VSR on a ready source CVPixelBuffer; return the dst buffer.

        Shared tail of both upscale entry points: allocates the dst buffer,
        wraps src/dst as VTFrameProcessorFrames, builds the mode-appropriate
        parameters (threading the prev-frame chain for balanced), and advances
        that chain. The prev VTFrameProcessorFrame retains its CVPixelBuffer,
        so an externally-supplied src buffer stays valid across the one
        iteration balanced mode references it.
        """
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
                use_temporal = self.mode == "balanced"
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
            if self.mode == "balanced":
                self._prev_src_frame = src_frame
                self._prev_dst_frame = dst_frame
            else:
                self.reset_temporal_context()
                del src_frame, dst_frame
            del params
        return dst_pb
