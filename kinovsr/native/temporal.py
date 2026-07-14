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
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from fractions import Fraction
from typing import Any

from kinovsr.media import pixel_buffers as _pb
from kinovsr.media.timing import rational_cadence

from .frameworks import autorelease_pool, vt

_log = logging.getLogger(__name__)

# One source pair can legitimately fan out to many target frames. Process the
# destinations in fixed-size requests and reserve the measured three-surface
# scheduler/processor/caller handoff high-water needed during final/cut drains.
DESTINATION_BATCH_SIZE = 4
DST_POOL_ALLOCATION_LIMIT = DESTINATION_BATCH_SIZE + 3


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
    """

    # mode enum (rate-conversion quality prioritization)
    MODE_NORMAL = "normal"
    MODE_HIGH = "high"

    def __init__(
        self,
        in_w: int,
        in_h: int,
        source_fps: Fraction | int | float,
        target_fps: Fraction | int | float,
        *,
        mode: str = MODE_NORMAL,
    ):
        self.source_cadence = rational_cadence(source_fps)
        self.target_cadence = rational_cadence(target_fps)
        if not vt.VTFrameRateConversionConfiguration.isSupported():
            raise SystemExit("VTFrameRateConversionConfiguration not supported on this device.")

        self.in_w, self.in_h = in_w, in_h
        self.source_fps = float(self.source_cadence)
        self.target_fps = float(self.target_cadence)
        self.mode = mode

        q = (
            vt.VTFrameRateConversionConfigurationQualityPrioritizationQuality
            if mode == self.MODE_HIGH
            else vt.VTFrameRateConversionConfigurationQualityPrioritizationNormal
        )
        cls = vt.VTFrameRateConversionConfiguration
        self.config = cls.alloc().initWithFrameWidth_frameHeight_usePrecomputedFlow_qualityPrioritization_revision_(
            in_w,
            in_h,
            False,
            q,
            cls.defaultRevision(),
        )
        if self.config is None:
            raise RuntimeError("VTFrameRateConversionConfiguration init returned nil")

        self.processor = vt.VTFrameProcessor.alloc().init()
        ok, err = self.processor.startSessionWithConfiguration_error_(self.config, None)
        if not ok:
            raise RuntimeError(f"VTFrameProcessor (rate conversion) startSession failed: {err}")

        self.src_attrs = dict(self.config.sourcePixelBufferAttributes() or {})
        self.dst_attrs = dict(self.config.destinationPixelBufferAttributes() or {})
        _log.info(
            "Temporal session ready (%.3ffps -> %.3ffps @ %sx%s, mode=%s, "
            "src fmt %#x, dst fmt %#x)",
            source_fps,
            target_fps,
            in_w,
            in_h,
            mode,
            _pb.resolve_pixel_format(self.src_attrs),
            _pb.resolve_pixel_format(self.dst_attrs),
        )

        self._dst_pool = _pb.make_bounded_pool_from_attrs(self.dst_attrs, DST_POOL_ALLOCATION_LIMIT)
        self._owns_dst_pool = True
        if self._dst_pool is None:
            try:
                self.processor.endSession()
            except Exception:
                _log.exception("FRC session cleanup failed after destination pool creation failure")
            finally:
                self.processor = None
            raise RuntimeError(
                "FRC destination CVPixelBufferPool creation failed; "
                "bounded output allocation is required"
            )

        # Per-pair state ----------------------------------------------------
        # We track source frames by their target-fps frame index so we can
        # discover output frames that fall in the source-pair's interval by
        # iterating contiguous integer target indices.
        # source frame N is at time N / source_fps.
        # target frame M is at time M / target_fps.
        # M / target_fps in [N/source_fps, (N+1)/source_fps) means
        #   M in [N * (target/source), (N+1) * (target/source))
        self._prev_src_pb: Any = None
        self._prev_src_index: int = -1  # source frame index of buffered prev
        self._next_target_index: int | None = None

    def use_dst_pool(self, pool: Any) -> None:
        """Wire AVWriter's adaptor pool for zero-copy output."""
        if pool is None:
            raise ValueError("destination pool must not be None")
        if self._owns_dst_pool and self._dst_pool is not pool:
            _pb.flush_pool(self._dst_pool)
        self._dst_pool = pool
        self._owns_dst_pool = False

    def flush_pools(self) -> None:
        """Release excess buffers from the session-owned destination pool."""
        if self._owns_dst_pool:
            _pb.flush_pool(self._dst_pool)

    def close(self) -> None:
        processor, self.processor = self.processor, None
        try:
            if processor is not None:
                processor.endSession()
        finally:
            self._prev_src_pb = None
            self._next_target_index = None
            self.config = None
            self.flush_pools()
            self._dst_pool = None
            self._owns_dst_pool = False

    # ------------------------------------------------------------------------
    # Internal buffer factory
    # ------------------------------------------------------------------------

    def _make_dst_buffer(self) -> Any:
        if self._dst_pool is None:
            raise RuntimeError("FRC destination pool is unavailable")
        if self._owns_dst_pool:
            return _pb.pool_create_buffer_bounded(self._dst_pool, DST_POOL_ALLOCATION_LIMIT)
        pb = _pb.pool_create_buffer(self._dst_pool)
        if pb is None:
            raise RuntimeError("external FRC destination pool acquisition failed")
        return pb

    # ------------------------------------------------------------------------
    # Phase / target-index math
    # ------------------------------------------------------------------------

    def _target_indices_in_pair(self, src_index: int) -> list[int]:
        """Target frame indices M such that M's PTS falls in
        [src_index / source_fps, (src_index + 1) / source_fps).
        """
        lower = math.ceil(Fraction(src_index) * self.target_cadence / self.source_cadence)
        upper = math.ceil(Fraction(src_index + 1) * self.target_cadence / self.source_cadence)
        start = lower if self._next_target_index is None else max(lower, self._next_target_index)
        if upper - start > 10_000:
            raise RuntimeError(
                f"pathological frame-rate ratio emits {upper - start} "
                f"targets for source frame {src_index}"
            )
        return list(range(start, upper))

    def _phases_for_targets(self, target_indices: list[int], src_index: int) -> list[float]:
        """For each target index M, return phase = (M/target - src/source) /
        (1/source) clamped to [0, 1). Phase 0 = source frame, phase 1 = next.
        """
        phases = []
        src_time = Fraction(src_index) / self.source_cadence
        denom = 1 / self.source_cadence
        for m in target_indices:
            phase = float((Fraction(m) / self.target_cadence - src_time) / denom)
            # Clamp to [0, 1) for robustness against float drift.
            if phase < 0.0:
                phase = 0.0
            elif phase >= 1.0:
                phase = 1.0 - 1e-9
            phases.append(phase)
        return phases

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def _process_destination_batches(
        self,
        source_pb: Any,
        source_index: int,
        next_pb: Any,
        next_index: int,
        target_indices: list[int],
        phases: list[float],
        error_label: str,
    ) -> Iterator[Any]:
        """Process one source pair without materializing all destinations.

        The first request establishes the sequential references. Later chunks
        explicitly reuse them, which preserves one-call FRC numerics while
        bounding live destination surfaces independently of the rate ratio.
        """
        batch_size = DESTINATION_BATCH_SIZE

        source_pts = _pb.frame_pts(source_index, self.source_cadence)
        next_pts = _pb.frame_pts(next_index, self.source_cadence)
        source_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            source_pb, source_pts
        )
        next_frame = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
            next_pb, next_pts
        )

        for offset in range(0, len(target_indices), batch_size):
            batch_indices = target_indices[offset : offset + batch_size]
            batch_phases = phases[offset : offset + batch_size]
            with autorelease_pool():
                dest_buffers = [self._make_dst_buffer() for _ in batch_indices]
                dest_frames = [
                    vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                        buffer, _pb.frame_pts(index, self.target_cadence)
                    )
                    for index, buffer in zip(batch_indices, dest_buffers, strict=True)
                ]
                submission_mode = (
                    vt.VTFrameRateConversionParametersSubmissionModeSequential
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
                ok, err = self.processor.processWithParameters_error_(params, None)
                del params, dest_frames
            if not ok:
                raise RuntimeError(f"{error_label}: {err}")

            while dest_buffers:
                yield dest_buffers.pop(0)

        del source_frame, next_frame

    def feed(self, src_pb: Any, src_index: int) -> Iterator[Any]:
        """Feed one source frame. Yields the interpolated destination buffers
        whose PTSes fall in [prev_src_pts, this_src_pts).

        For the very first source frame, this is empty (no pair yet). For
        subsequent frames we compute the target indices in the prev->curr
        gap, process them in bounded destination batches, and yield each
        output buffer.
        """
        if self._prev_src_pb is None:
            self._prev_src_pb = src_pb
            self._prev_src_index = src_index
            self._next_target_index = math.ceil(
                Fraction(src_index) * self.target_cadence / self.source_cadence
            )
            return

        target_indices = self._target_indices_in_pair(self._prev_src_index)
        if not target_indices:
            # Identity / downsample case: no output frame falls in this gap.
            self._prev_src_pb = src_pb
            self._prev_src_index = src_index
            return

        phases = self._phases_for_targets(target_indices, self._prev_src_index)
        yield from self._process_destination_batches(
            self._prev_src_pb,
            self._prev_src_index,
            src_pb,
            src_index,
            target_indices,
            phases,
            f"VTFC processWithParameters failed at source pair {self._prev_src_index}->{src_index}",
        )
        self._next_target_index = target_indices[-1] + 1
        self._prev_src_pb = src_pb
        self._prev_src_index = src_index

    def drain(self) -> Iterator[Any]:
        """After all source frames have been fed, yield the target frames of the
        FINAL source period -- feed() only emits a pair's outputs when the next
        source frame arrives, so the last source frame's targets (its phase-0
        passthrough and any same-period interpolations) have no pair to ride and
        would otherwise be dropped, ending the output one source period early.

        With no next frame to interpolate toward, hold the last frame: run the
        processor with the buffered frame as both source and next -- interpolating
        a frame with itself reproduces it exactly at any phase -- producing one
        held output per remaining target index in [last_pts, last_pts + 1/source).
        """
        if self._prev_src_pb is None:
            return
        target_indices = self._target_indices_in_pair(self._prev_src_index)
        if not target_indices:
            self._prev_src_pb = None
            return
        phases = self._phases_for_targets(target_indices, self._prev_src_index)
        yield from self._process_destination_batches(
            self._prev_src_pb,
            self._prev_src_index,
            self._prev_src_pb,
            self._prev_src_index + 1,
            target_indices,
            phases,
            f"VTFRC drain failed for final source period at frame {self._prev_src_index}",
        )
        self._next_target_index = target_indices[-1] + 1
        self._prev_src_pb = None
