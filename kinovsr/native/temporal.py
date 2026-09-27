"""VideoToolbox Frame Rate Conversion (temporal upscaler) session wrapper.

Wraps VTFrameRateConversionConfiguration + VTFrameRateConversionParameters
to convert between arbitrary source and target frame rates. Unlike VSR,
the configuration takes only the frame dimensions + quality; the rate
conversion ratio is driven entirely by per-pair interpolation phases.

Per source frame pair (frame_N at PTS_N, frame_N+1 at PTS_N+1), we
compute the set of target output PTSes that fall in [PTS_N, PTS_N+1)
and their phases (where phase = (target_pts - PTS_N) / (PTS_N+1 - PTS_N)
in [0, 1)). VT's API takes a phase array and a matching destinationFrames
array, so a single call produces all interpolated frames for that pair.

Cleanly handles arbitrary float fps both sides:
  15 -> 30   exact 2x; phases always [0.5].
  24 -> 60   2.5x; phases cycle [0, 0.4, 0.8], [0.2, 0.6], ...
  24 -> 24   identity; phase array per pair is [0.0] = source pass-through
             (caller should detect this and skip the stage entirely).

Some frame sizes need a different canvas. On macOS 27, FRC runs 720x480,
352x288, 480x480, 640x272, portrait 480x720, and other sizes on a network
whose synthesis stage misreads its flow, so moving content comes out close to
the 50/50 blend of the two source frames. The same frames edge-padded (last
column and row repeated) into a configuration that works, then cropped back,
interpolate correctly. ``VtfrcSession`` decides this per frame size with a
synthetic moving-square check, so sizes FRC handles keep their direct path and
the workaround switches itself off wherever FRC is right.
"""

import logging
import math
import threading
from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import Final, Literal, Protocol, cast

import mlx.core as mx

from kinovsr.media import pixel_buffers as _pb
from kinovsr.media.timing import rational_cadence

from .canvas_copy import CanvasCopier
from .frameworks import CoreMedia, Quartz, autorelease_pool, vt

_log = logging.getLogger(__name__)

# One source pair can legitimately fan out to many target frames. Process the
# destinations in fixed-size requests and reserve the measured three-surface
# scheduler/processor/caller handoff high-water needed during final/cut drains.
DESTINATION_BATCH_SIZE = 4
DST_POOL_ALLOCATION_LIMIT = DESTINATION_BATCH_SIZE + 3
# FRC keeps the previous source frame between calls, so the session copies every
# source into its own pool (onto the padded canvas where one is needed). A
# borrowed upstream buffer, such as VideoToolbox super resolution's bounded
# output, then returns to its producer when the call ends. The pool holds the
# buffered frame and the incoming one, like the MLX upload pool.
SRC_POOL_ALLOCATION_LIMIT = 2
# Canvas-sized destinations live only until they are cropped into outputs.
CANVAS_DST_POOL_ALLOCATION_LIMIT = DESTINATION_BATCH_SIZE

# FRC's network configurations (landscape; portrait frames run on the same
# networks rotated). A frame edge-padded to exactly one of these runs on that
# network with no further padding inside FRC.
FRC_CANVASES = (
    (320, 180),
    (320, 240),
    (480, 270),
    (640, 360),
    (640, 480),
    (960, 540),
    (1280, 720),
    (1440, 1080),
    (1920, 1080),
)
AUTO_CANVAS: Final = "auto"

# On macOS 27.0, FRC renders the check's moving square 13-24 dB better than the
# 50/50 blend of its two source frames at sizes that work, and 0.6-1.5 dB better
# at sizes whose network fails.
MOTION_CHECK_MARGIN_DB = 6.0
_CHECK_STEP = 4  # pixels the square moves between the two source frames
_CHECK_MIN_SIDE = 32
_CANVAS_DECISIONS: dict[tuple[int, int, str, int], tuple[int, int] | None] = {}
_CANVAS_LOCK = threading.Lock()
_UNDECIDED = object()


class _FrcConfiguration(Protocol):
    def sourcePixelBufferAttributes(self) -> Mapping[str, object] | None: ...

    def destinationPixelBufferAttributes(self) -> Mapping[str, object] | None: ...


class _FrameProcessor(Protocol):
    def processWithParameters_error_(
        self, parameters: object, error: None
    ) -> tuple[bool, object]: ...

    def endSession(self) -> None: ...


def canvas_candidates(width: int, height: int) -> list[tuple[int, int]]:
    """FRC configurations that contain a ``width`` x ``height`` frame, best first.

    Smallest area first, the frame's own orientation first among equals. The
    frame's own size is not a candidate.
    """
    landscape = width >= height
    sizes = {*FRC_CANVASES, *((h, w) for w, h in FRC_CANVASES)}
    fits = [(w, h) for w, h in sizes if w >= width and h >= height and (w, h) != (width, height)]
    return sorted(fits, key=lambda size: (size[0] * size[1], (size[0] >= size[1]) != landscape))


def _motion_check_frames(
    width: int, height: int
) -> tuple[list[mx.array], tuple[int, int, int, int]]:
    """The check's first source, true midpoint, and second source frames.

    A square of 4x4-pixel random blocks, a third of the short side, sits
    ``_CHECK_STEP`` pixels lower in the second source than in the first, over a
    smooth static background. Frames are (H, W, 3) float32 RGB; the region
    (top, bottom, left, right) covers the square's sweep plus 8 pixels.
    """
    y, x = mx.meshgrid(
        mx.arange(height, dtype=mx.float32), mx.arange(width, dtype=mx.float32), indexing="ij"
    )
    background = mx.stack(
        [
            0.50 + 0.20 * mx.sin(x * 0.0647) * mx.cos(y * 0.0885),
            0.45 + 0.20 * mx.sin((x + y) * 0.0480),
            0.40 + 0.15 * mx.cos(x * 0.1185 - y * 0.0706),
        ],
        axis=-1,
    )
    side = max(16, min(width, height) // 3)
    cells = mx.random.uniform(shape=(side // 4 + 1, side // 4 + 1, 3), key=mx.random.key(3))
    texture = mx.repeat(mx.repeat(cells, 4, axis=0), 4, axis=1)[:side, :side]
    left = width // 2 - side // 2
    top = height // 2 - side // 2 - _CHECK_STEP // 2
    frames = []
    for shift in (0, _CHECK_STEP // 2, _CHECK_STEP):
        row = top + shift
        band = mx.concatenate(
            [
                background[row : row + side, :left],
                texture,
                background[row : row + side, left + side :],
            ],
            axis=1,
        )
        frames.append(mx.concatenate([background[:row], band, background[row + side :]], axis=0))
    region = (
        max(0, top - 8),
        min(height, top + _CHECK_STEP + side + 8),
        max(0, left - 8),
        min(width, left + side + 8),
    )
    return frames, region


def _luma_psnr(a: mx.array, b: mx.array, region: tuple[int, int, int, int]) -> float:
    top, bottom, left, right = region
    difference = (a[top:bottom, left:right] - b[top:bottom, left:right]) * mx.array(
        [0.2126, 0.7152, 0.0722]
    )
    luma = mx.sum(difference, axis=-1)
    return float(10 * mx.log10(1.0 / mx.maximum(mx.mean(luma * luma), 1e-12)))


def _rgbahalf_buffer(frame: mx.array) -> object:
    """An IOSurface RGBAHalf buffer holding an (H, W, 3) float RGB frame."""
    height, width = int(frame.shape[0]), int(frame.shape[1])
    buffer = _pb.make_pixel_buffer_from_attrs(
        width,
        height,
        {
            "PixelFormatType": _pb.PIX_RGBAHALF,
            "Width": width,
            "Height": height,
            "IOSurfaceProperties": {},
            "MetalCompatibility": True,
        },
    )
    alpha = mx.ones((height, width, 1), dtype=mx.float32)
    _pb.write_fp16_rgba(mx.concatenate([frame, alpha], axis=-1).astype(mx.float16), buffer)
    return buffer


def motion_check_margin(
    width: int, height: int, mode: str, canvas: tuple[int, int] | None
) -> float:
    """The motion-check margin, in dB, of ``width`` x ``height`` frames on ``canvas``.

    None checks FRC at the frame size; see ``VtfrcSession._motion_check``.
    """
    session = VtfrcSession(width, height, 2, 2, mode=mode, canvas=canvas)
    try:
        return session._motion_check()
    finally:
        session.close()


class VtfrcSession:
    """Per-pair temporal interpolator with arbitrary source/target fps.

    Construction takes frame dimensions, source fps, target fps, and a
    mode setting (Normal or Quality prioritization). The session buffers
    one source frame at a time and emits all output frames that fall in
    the gap when the next source frame arrives.

    Usage:
        session = VtfrcSession(width, height, source_fps=24, target_fps=60)
        session.use_dst_pool(av_writer.adaptor.pixelBufferPool())
        for src_idx, src_pb in enumerate(source_buffers):
            for dst_pb in session.feed(src_pb, src_idx):
                av_writer.append(dst_pb)
        for dst_pb in session.drain():
            av_writer.append(dst_pb)

    `feed()` yields zero or more interpolated frames per source frame. The
    first source frame just buffers (yields nothing); subsequent frames
    trigger interpolation between the buffered prev and the incoming curr.
    `drain()` emits the last source frame if it falls on a target PTS.

    ``canvas`` selects the FRC configuration. None runs FRC at the frame size;
    a (width, height) edge-pads every source frame into that size and crops
    every output back, so callers see frame-sized buffers either way. Every
    source the session keeps is its own copy; callers' buffers are never held
    past a call.
    ``AUTO_CANVAS`` (the default) uses the configuration the motion check
    picks for this frame size (``_start_checked``).
    """

    # mode enum (rate-conversion quality prioritization)
    MODE_NORMAL = "normal"
    MODE_HIGH = "high"

    # Native resources: None until started and after close. The padded-canvas
    # state stays None on the direct path.
    config: _FrcConfiguration | None = None
    processor: _FrameProcessor | None = None
    # FRC keeps the source pixel format of its first request (-19730 for any
    # other), so the session tracks the format its processor has seen.
    _processor_format: int | None = None
    _dst_pool: object | None = None
    _owns_dst_pool = False
    _copier: CanvasCopier | None = None
    _src_pool: object | None = None
    _src_format: int | None = None
    canvas: tuple[int, int] | None = None
    _canvas_dst_pool: object | None = None

    def __init__(
        self,
        in_w: int,
        in_h: int,
        source_fps: Fraction | int | float,
        target_fps: Fraction | int | float,
        *,
        mode: str = MODE_NORMAL,
        canvas: tuple[int, int] | Literal["auto"] | None = AUTO_CANVAS,
    ):
        self.source_cadence = rational_cadence(source_fps)
        self.target_cadence = rational_cadence(target_fps)
        if not vt.VTFrameRateConversionConfiguration.isSupported():
            raise SystemExit("VTFrameRateConversionConfiguration not supported on this device.")

        self.in_w, self.in_h = in_w, in_h
        self.source_fps = float(self.source_cadence)
        self.target_fps = float(self.target_cadence)
        self.mode = mode

        # Per-pair state ----------------------------------------------------
        # Source frames are tracked by their exact presentation TIME; the
        # target grid stays integer-indexed (target frame M at M/target_fps),
        # so a pair (prev_t, curr_t) emits every M with M/target_fps in
        # [prev_t, curr_t). Uniform sources reach this through the index
        # shim (frame N at N/source_fps), which reproduces the historical
        # integer arithmetic exactly; non-uniform sources feed real times
        # and the per-destination interpolationPhase array carries the
        # arbitrary spacing natively.
        self._prev_src_pb: object | None = None
        self._prev_time: Fraction | None = None
        self._next_target_index: int | None = None
        # Apple requires Random after a jump or skip so the processor clears
        # its internal interpolation cache. The first request has no preceding
        # references; later contiguous pairs clear this flag after succeeding.
        self._submission_needs_random = True

        try:
            if canvas == AUTO_CANVAS:
                self._start_checked()
            else:
                if canvas is not None:
                    canvas = (int(canvas[0]), int(canvas[1]))
                    if canvas[0] < in_w or canvas[1] < in_h:
                        raise ValueError(
                            f"FRC canvas {canvas[0]}x{canvas[1]} cannot hold {in_w}x{in_h} frames"
                        )
                self._start(None if canvas == (in_w, in_h) else canvas)
        except BaseException:
            try:
                self._stop()
            except Exception:
                _log.exception("FRC session cleanup failed after a setup failure")
            raise
        _log.info(
            "Temporal session ready (%.3ffps -> %.3ffps @ %sx%s%s, mode=%s, "
            "src fmt %#x, dst fmt %#x)",
            source_fps,
            target_fps,
            in_w,
            in_h,
            "" if self.canvas is None else " on a {}x{} canvas".format(*self.canvas),
            mode,
            _pb.resolve_pixel_format(self.src_attrs),
            _pb.resolve_pixel_format(self.dst_attrs),
        )

    def _start(self, canvas: tuple[int, int] | None) -> None:
        """Start FRC on ``canvas`` (None: the frame size) with its bounded pools."""
        self.canvas = canvas
        frame_w, frame_h = canvas or (self.in_w, self.in_h)
        q = (
            vt.VTFrameRateConversionConfigurationQualityPrioritizationQuality
            if self.mode == self.MODE_HIGH
            else vt.VTFrameRateConversionConfigurationQualityPrioritizationNormal
        )
        cls = vt.VTFrameRateConversionConfiguration
        self.config = cls.alloc().initWithFrameWidth_frameHeight_usePrecomputedFlow_qualityPrioritization_revision_(
            frame_w,
            frame_h,
            False,
            q,
            cls.defaultRevision(),
        )
        if self.config is None:
            raise RuntimeError("VTFrameRateConversionConfiguration init returned nil")

        self._start_processor()
        self.src_attrs = dict(self.config.sourcePixelBufferAttributes() or {})
        self.dst_attrs = dict(self.config.destinationPixelBufferAttributes() or {})
        self._copier = CanvasCopier()
        if canvas is not None:
            self._start_canvas()
        self._dst_pool = _pb.make_bounded_pool_from_attrs(self.dst_attrs, DST_POOL_ALLOCATION_LIMIT)
        self._owns_dst_pool = True
        if self._dst_pool is None:
            raise RuntimeError(
                "FRC destination CVPixelBufferPool creation failed; "
                "bounded output allocation is required"
            )

    def _start_processor(self) -> None:
        processor = vt.VTFrameProcessor.alloc().init()
        ok, err = processor.startSessionWithConfiguration_error_(self.config, None)
        if not ok:
            raise RuntimeError(f"VTFrameProcessor (rate conversion) startSession failed: {err}")
        self.processor = processor
        self._processor_format = None
        self._submission_needs_random = True

    def _start_checked(self) -> None:
        """Start on the configuration the motion check picks for this frame size.

        The first session for a frame size, mode, and FRC revision in a process
        checks FRC at the frame size and, where it fails, each of
        ``canvas_candidates`` in turn. The session keeps the first configuration
        that passes, already warmed up, and later sessions reuse the decision.
        When nothing passes, the session warns and runs at the frame size; that
        outcome is not cached, so the next session checks and warns again.
        """
        width, height = self.in_w, self.in_h
        revision = int(vt.VTFrameRateConversionConfiguration.defaultRevision())
        key = (width, height, self.mode, revision)
        with _CANVAS_LOCK:
            decision = _CANVAS_DECISIONS.get(key, _UNDECIDED)
            if decision is not _UNDECIDED:
                self._start(cast(tuple[int, int] | None, decision))  # a stored decision
                return
            self._start(None)
            if min(width, height) < _CHECK_MIN_SIDE:
                _CANVAS_DECISIONS[key] = None
                return
            native = self._motion_check()
            if native >= MOTION_CHECK_MARGIN_DB:
                _log.debug(
                    "FRC motion check at %dx%d: %.1f dB over the frame blend", width, height, native
                )
                _CANVAS_DECISIONS[key] = None
                return
            for canvas in canvas_candidates(width, height):
                self._stop()
                try:
                    self._start(canvas)
                    margin = self._motion_check()
                except RuntimeError as exc:
                    _log.debug("FRC canvas %dx%d unavailable: %s", *canvas, exc)
                    continue
                if margin >= MOTION_CHECK_MARGIN_DB:
                    _log.info(
                        "VideoToolbox frame-rate conversion renders motion wrong at %dx%d "
                        "(%.1f dB over the frame blend); interpolating it edge-padded into "
                        "%dx%d (%.1f dB)",
                        width,
                        height,
                        native,
                        *canvas,
                        margin,
                    )
                    _CANVAS_DECISIONS[key] = canvas
                    return
            _log.warning(
                "VideoToolbox frame-rate conversion renders motion wrong at %dx%d (%.1f dB "
                "over the frame blend) and no padded configuration passed its check; "
                "interpolating at the frame size, where moving edges may look doubled",
                width,
                height,
                native,
            )
            self._stop()
            self._start(None)

    def _motion_check(self) -> float:
        """By how many dB this session's midpoint of a synthetic pair beats the blend.

        The pair from ``_motion_check_frames`` takes the production path
        (padding, one FRC request at phase 0.5, cropping) and is scored by luma
        PSNR against the true midpoint over the moving square. The session stays
        as new: its next pair is still submitted Random.
        """
        frames, region = _motion_check_frames(self.in_w, self.in_h)
        sources = [
            self._own_source(_rgbahalf_buffer(frames[0])),
            self._own_source(_rgbahalf_buffer(frames[2])),
        ]
        try:
            outputs = list(
                self._process_destination_batches(
                    sources[0],
                    Fraction(0),
                    sources[1],
                    Fraction(1),
                    [0],
                    [0.5],
                    "FRC motion check failed",
                )
            )
        finally:
            self._submission_needs_random = True
        self._processor_format = _pb.PIX_RGBAHALF
        rendered = _pb.read_rgbahalf_rgb(outputs[0])
        blend = (frames[0] + frames[2]) / 2
        return _luma_psnr(rendered, frames[1], region) - _luma_psnr(blend, frames[1], region)

    def use_dst_pool(self, pool: object) -> None:
        """Wire AVWriter's adaptor pool for zero-copy output."""
        if pool is None:
            raise ValueError("destination pool must not be None")
        if self._owns_dst_pool and self._dst_pool is not pool:
            _pb.flush_pool(self._dst_pool)
        self._dst_pool = pool
        self._owns_dst_pool = False

    def flush_pools(self) -> None:
        """Release excess buffers from the session-owned pools."""
        if self._owns_dst_pool:
            _pb.flush_pool(self._dst_pool)
        for pool in (self._src_pool, self._canvas_dst_pool):
            if pool is not None:
                _pb.flush_pool(pool)

    def _stop_processor(self) -> None:
        processor, self.processor = self.processor, None
        if processor is not None:
            processor.endSession()

    def _stop(self) -> None:
        """End the FRC session and release its pools."""
        try:
            self._stop_processor()
        finally:
            self.config = None
            self.flush_pools()
            self._dst_pool = None
            self._owns_dst_pool = False
            self._release_copies()

    def close(self) -> None:
        self._prev_src_pb = None
        self._next_target_index = None
        self._submission_needs_random = True
        self._stop()

    # ------------------------------------------------------------------------
    # Internal buffer factory
    # ------------------------------------------------------------------------

    def _make_dst_buffer(self) -> object:
        if self._dst_pool is None:
            raise RuntimeError("FRC destination pool is unavailable")
        if self._owns_dst_pool:
            return _pb.pool_create_buffer_bounded(self._dst_pool, DST_POOL_ALLOCATION_LIMIT)
        pb = _pb.pool_create_buffer(self._dst_pool)
        if pb is None:
            raise RuntimeError("external FRC destination pool acquisition failed")
        return pb

    # ------------------------------------------------------------------------
    # Owned sources and the padded canvas (see _start_checked)
    # ------------------------------------------------------------------------

    def _start_canvas(self) -> None:
        """Pool canvas-sized FRC destinations; advertise frame-sized buffers."""
        canvas_dst_attrs = {**self.dst_attrs, "MetalCompatibility": True}
        self._canvas_dst_pool = _pb.make_bounded_pool_from_attrs(
            canvas_dst_attrs, CANVAS_DST_POOL_ALLOCATION_LIMIT
        )
        if self._canvas_dst_pool is None:
            raise RuntimeError("FRC canvas destination CVPixelBufferPool creation failed")
        frame = {"Width": self.in_w, "Height": self.in_h}
        self.src_attrs = {**self.src_attrs, **frame}
        self.dst_attrs = {**canvas_dst_attrs, **frame}

    def _release_copies(self) -> None:
        """Drop the source and canvas pools (``flush_pools`` flushed them) and the copier."""
        self._src_pool = self._canvas_dst_pool = None
        self._src_format = None
        copier, self._copier = self._copier, None
        if copier is not None:
            copier.close()

    def _own_source(self, src_pb: object) -> object:
        """Copy a frame-sized source into the session's own pool.

        FRC keeps the previous source between calls, so it must not keep the
        caller's buffer. On a padded canvas the copy repeats the last column and
        row. The direct path keeps a source whose format the copier cannot
        handle as given.
        """
        width = int(Quartz.CVPixelBufferGetWidth(src_pb))
        height = int(Quartz.CVPixelBufferGetHeight(src_pb))
        if (width, height) != (self.in_w, self.in_h):
            raise ValueError(
                f"FRC source is {width}x{height}; this session interpolates "
                f"{self.in_w}x{self.in_h} frames"
            )
        pixel_format = int(Quartz.CVPixelBufferGetPixelFormatType(src_pb))
        if self._copier is None or not self._copier.supports(pixel_format):
            if self.canvas is None:
                return src_pb
            raise RuntimeError(f"FRC cannot pad pixel format {pixel_format:#x}")
        if self._src_pool is None or pixel_format != self._src_format:
            if self._src_pool is not None:
                _pb.flush_pool(self._src_pool)
            pool_w, pool_h = self.canvas or (self.in_w, self.in_h)
            self._src_pool = _pb.make_bounded_pool_from_attrs(
                {
                    "PixelFormatType": pixel_format,
                    "Width": pool_w,
                    "Height": pool_h,
                    "IOSurfaceProperties": {},
                    "MetalCompatibility": True,
                },
                SRC_POOL_ALLOCATION_LIMIT,
            )
            self._src_format = pixel_format
            if self._src_pool is None:
                raise RuntimeError("FRC source CVPixelBufferPool creation failed")
        owned = _pb.pool_create_buffer_bounded(self._src_pool, SRC_POOL_ALLOCATION_LIMIT)
        return self._copier.copy(src_pb, owned)

    def _make_frc_dst_buffer(self) -> object:
        """A destination for FRC itself: canvas-sized when padding."""
        if self.canvas is None:
            return self._make_dst_buffer()
        if self._canvas_dst_pool is None:
            raise RuntimeError("FRC canvas destination pool is unavailable")
        return _pb.pool_create_buffer_bounded(
            self._canvas_dst_pool, CANVAS_DST_POOL_ALLOCATION_LIMIT
        )

    def _crop(self, canvas_pb: object) -> object:
        assert self._copier is not None
        return self._copier.copy(canvas_pb, self._make_dst_buffer())

    # ------------------------------------------------------------------------
    # Phase / target-index math (pure; unit-tested without a VT session)
    # ------------------------------------------------------------------------

    def _targets_between(
        self,
        prev_time: Fraction,
        curr_time: Fraction,
    ) -> list[int]:
        """Target frame indices M with M/target_fps in [prev_time, curr_time)."""
        lower = math.ceil(prev_time * self.target_cadence)
        upper = math.ceil(curr_time * self.target_cadence)
        start = lower if self._next_target_index is None else max(lower, self._next_target_index)
        if upper - start > 10_000:
            raise RuntimeError(
                f"pathological frame spacing emits {upper - start} targets "
                f"for the source pair at {float(prev_time):.6g}s"
            )
        return list(range(start, upper))

    def _phases_between(
        self,
        target_indices: list[int],
        prev_time: Fraction,
        curr_time: Fraction,
    ) -> list[float]:
        """For each target index M, phase = (M/target - prev) / (curr - prev)
        clamped to [0, 1). Phase 0 = the prev source frame, phase 1 = next.
        The per-destination phase array is what lets arbitrary (non-uniform)
        source spacing ride the same VT call as the uniform grid.
        """
        phases = []
        denom = curr_time - prev_time
        if denom <= 0:
            raise RuntimeError(
                f"source times must be strictly increasing; got "
                f"{float(prev_time):.6g}s then {float(curr_time):.6g}s"
            )
        for m in target_indices:
            phase = float((Fraction(m) / self.target_cadence - prev_time) / denom)
            # Clamp to [0, 1) for robustness against float drift.
            if phase < 0.0:
                phase = 0.0
            elif phase >= 1.0:
                phase = 1.0 - 1e-9
            phases.append(phase)
        return phases

    def _time_pts(self, value: Fraction) -> object:
        """A source time as CMTime on the product tick base (frame identity
        for the processor; equals frame_pts(N, cadence) on the index shim)."""
        return CoreMedia.CMTimeMake(round(value * _pb.VIDEO_TIME_SCALE), _pb.VIDEO_TIME_SCALE)

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def reset_temporal_context(self) -> None:
        """Mark the next submitted pair as unrelated to prior references.

        The caller must drain the preceding segment first so no buffered pair
        is discarded. The next request then uses Random exactly once; later
        contiguous pairs return to Sequential.
        """
        if self._prev_src_pb is not None:
            raise RuntimeError("drain() must clear the buffered FRC pair before reset")
        self._submission_needs_random = True

    def _process_destination_batches(
        self,
        source_pb: object,
        source_time: Fraction,
        next_pb: object,
        next_time: Fraction,
        target_indices: list[int],
        phases: list[float],
        error_label: str,
    ) -> Iterator[object]:
        """Process one source pair without materializing all destinations.

        The first request establishes the sequential references. Later chunks
        explicitly reuse them, which preserves one-call FRC numerics while
        bounding live destination surfaces independently of the rate ratio.
        """
        batch_size = DESTINATION_BATCH_SIZE

        source_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            source_pb, self._time_pts(source_time)
        )
        next_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            next_pb, self._time_pts(next_time)
        )

        for offset in range(0, len(target_indices), batch_size):
            batch_indices = target_indices[offset : offset + batch_size]
            batch_phases = phases[offset : offset + batch_size]
            with autorelease_pool():
                dest_buffers = [self._make_frc_dst_buffer() for _ in batch_indices]
                dest_frames = [
                    vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                        buffer, _pb.frame_pts(index, self.target_cadence)
                    )
                    for index, buffer in zip(batch_indices, dest_buffers, strict=True)
                ]
                submission_mode = (
                    (
                        vt.VTFrameRateConversionParametersSubmissionModeRandom
                        if self._submission_needs_random
                        else vt.VTFrameRateConversionParametersSubmissionModeSequential
                    )
                    if offset == 0
                    else vt.VTFrameRateConversionParametersSubmissionModeSequentialReferencesUnchanged
                )
                params = vt.VTFrameRateConversionParameters.alloc().initWithSourceFrame_nextFrame_opticalFlow_interpolationPhase_submissionMode_destinationFrames_(
                    source_frame,
                    next_frame,
                    None,
                    batch_phases,
                    submission_mode,
                    dest_frames,
                )
                ok, err = cast(_FrameProcessor, self.processor).processWithParameters_error_(
                    params, None
                )  # started in __init__; None only after close() or a failed restart
                del params, dest_frames
            if not ok:
                raise RuntimeError(f"{error_label}: {err}")
            if offset == 0:
                self._submission_needs_random = False
            if self.canvas is not None:
                dest_buffers = [self._crop(buffer) for buffer in dest_buffers]

            while dest_buffers:
                yield dest_buffers.pop(0)

        del source_frame, next_frame

    def feed(self, src_pb: object, src_index: int) -> Iterator[object]:
        """Index shim: feed source frame N at its uniform-grid time
        N/source_fps. Exact Fraction arithmetic makes this reproduce the
        historical integer mapping bit-identically."""
        yield from self.feed_at(src_pb, Fraction(src_index) / self.source_cadence)

    def feed_at(self, src_pb: object, src_time: Fraction | int | float) -> Iterator[object]:
        """Feed one source frame at its exact presentation time. Yields the
        interpolated destination buffers whose PTSes fall in
        [prev_time, src_time).

        For the very first source frame, this is empty (no pair yet). For
        subsequent frames we compute the target indices in the prev->curr
        gap, process them in bounded destination batches, and yield each
        output buffer. Times must be strictly increasing; their spacing is
        free (a carried non-uniform timeline feeds real stamps here).
        """
        src_time = Fraction(src_time)
        if self._prev_src_pb is not None:
            assert self._prev_time is not None
            if src_time <= self._prev_time:
                # Checked BEFORE the empty-target fast path (and before the
                # copy): a duplicate stamp used to swap the buffered frame out
                # silently (the tied pair emits no targets), dropping a source
                # frame with no refusal and no ledger entry.
                raise RuntimeError(
                    f"source times must be strictly increasing; got "
                    f"{float(self._prev_time):.6g}s then {float(src_time):.6g}s"
                )
        else:
            pixel_format = int(Quartz.CVPixelBufferGetPixelFormatType(src_pb))
            if self._processor_format not in (None, pixel_format):
                # A new segment in another format (after the motion check's
                # RGBAHalf pair, or at a cut): same configuration, new processor.
                self._stop_processor()
                self._start_processor()
            self._processor_format = pixel_format
        src_pb = self._own_source(src_pb)
        if self._prev_src_pb is None:
            self._prev_src_pb = src_pb
            self._prev_time = src_time
            self._next_target_index = math.ceil(src_time * self.target_cadence)
            return

        assert self._prev_time is not None
        target_indices = self._targets_between(self._prev_time, src_time)
        if not target_indices:
            # Identity / downsample case: no output frame falls in this gap.
            # The next submitted pair will have skipped references relative
            # to the processor's cache and must clear it with Random.
            self._submission_needs_random = True
            self._prev_src_pb = src_pb
            self._prev_time = src_time
            return

        phases = self._phases_between(target_indices, self._prev_time, src_time)
        yield from self._process_destination_batches(
            self._prev_src_pb,
            self._prev_time,
            src_pb,
            src_time,
            target_indices,
            phases,
            f"VTFC processWithParameters failed at source pair "
            f"{float(self._prev_time):.6g}s->{float(src_time):.6g}s",
        )
        self._next_target_index = target_indices[-1] + 1
        self._prev_src_pb = src_pb
        self._prev_time = src_time

    def drain(self, hold: Fraction | int | float | None = None) -> Iterator[object]:
        """After all source frames have been fed, yield the target frames of the
        FINAL source period -- feed() only emits a pair's outputs when the next
        source frame arrives, so the last source frame's targets (its phase-0
        passthrough and any same-period interpolations) have no pair to ride and
        would otherwise be dropped, ending the output one source period early.

        With no next frame to interpolate toward, hold the last frame: run the
        processor with the buffered frame as both source and next -- interpolating
        a frame with itself reproduces it exactly at any phase -- producing one
        held output per remaining target index in [last_time, last_time + hold).
        ``hold`` defaults to one uniform source period; a carried timeline
        passes its final frame's real duration.
        """
        if self._prev_src_pb is None:
            return
        assert self._prev_time is not None
        hold_span = Fraction(hold) if hold is not None else 1 / self.source_cadence
        if hold_span <= 0:
            self._prev_src_pb = None
            self._submission_needs_random = True
            return
        end_time = self._prev_time + hold_span
        target_indices = self._targets_between(self._prev_time, end_time)
        if not target_indices:
            self._prev_src_pb = None
            self._submission_needs_random = True
            return
        phases = self._phases_between(target_indices, self._prev_time, end_time)
        yield from self._process_destination_batches(
            self._prev_src_pb,
            self._prev_time,
            self._prev_src_pb,
            end_time,
            target_indices,
            phases,
            f"VTFRC drain failed for final source period at {float(self._prev_time):.6g}s",
        )
        self._next_target_index = target_indices[-1] + 1
        self._prev_src_pb = None
        self._submission_needs_random = True
