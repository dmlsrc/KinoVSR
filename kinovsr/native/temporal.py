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

from .frameworks import CoreMedia, autorelease_pool, vt

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
        # Source frames are tracked by their exact presentation TIME; the
        # target grid stays integer-indexed (target frame M at M/target_fps),
        # so a pair (prev_t, curr_t) emits every M with M/target_fps in
        # [prev_t, curr_t). Uniform sources reach this through the index
        # shim (frame N at N/source_fps), which reproduces the historical
        # integer arithmetic exactly; non-uniform sources feed real times
        # and the per-destination interpolationPhase array carries the
        # arbitrary spacing natively.
        self._prev_src_pb: Any = None
        self._prev_time: Fraction | None = None
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
    # Phase / target-index math (pure; unit-tested without a VT session)
    # ------------------------------------------------------------------------

    def _targets_between(
        self, prev_time: Fraction, curr_time: Fraction,
    ) -> list[int]:
        """Target frame indices M with M/target_fps in [prev_time, curr_time)."""
        lower = math.ceil(prev_time * self.target_cadence)
        upper = math.ceil(curr_time * self.target_cadence)
        start = (lower if self._next_target_index is None
                 else max(lower, self._next_target_index))
        if upper - start > 10_000:
            raise RuntimeError(
                f"pathological frame spacing emits {upper - start} targets "
                f"for the source pair at {float(prev_time):.6g}s"
            )
        return list(range(start, upper))

    def _phases_between(
        self, target_indices: list[int],
        prev_time: Fraction, curr_time: Fraction,
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
                f"{float(prev_time):.6g}s then {float(curr_time):.6g}s")
        for m in target_indices:
            phase = float((Fraction(m) / self.target_cadence - prev_time)
                          / denom)
            # Clamp to [0, 1) for robustness against float drift.
            if phase < 0.0:
                phase = 0.0
            elif phase >= 1.0:
                phase = 1.0 - 1e-9
            phases.append(phase)
        return phases

    def _time_pts(self, value: Fraction) -> Any:
        """A source time as CMTime on the product tick base (frame identity
        for the processor; equals frame_pts(N, cadence) on the index shim)."""
        return CoreMedia.CMTimeMake(
            round(value * _pb.VIDEO_TIME_SCALE), _pb.VIDEO_TIME_SCALE)

    # ------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------

    def _process_destination_batches(
        self,
        source_pb: Any,
        source_time: Fraction,
        next_pb: Any,
        next_time: Fraction,
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
        """Index shim: feed source frame N at its uniform-grid time
        N/source_fps. Exact Fraction arithmetic makes this reproduce the
        historical integer mapping bit-identically."""
        yield from self.feed_at(
            src_pb, Fraction(src_index) / self.source_cadence)

    def feed_at(self, src_pb: Any, src_time: Fraction | int | float) -> Iterator[Any]:
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
        if self._prev_src_pb is None:
            self._prev_src_pb = src_pb
            self._prev_time = src_time
            self._next_target_index = math.ceil(
                src_time * self.target_cadence)
            return

        assert self._prev_time is not None
        if src_time <= self._prev_time:
            # Checked BEFORE the empty-target fast path: a duplicate
            # stamp used to swap the buffered frame out silently (the
            # tied pair emits no targets), dropping a source frame with
            # no refusal and no ledger entry.
            raise RuntimeError(
                f"source times must be strictly increasing; got "
                f"{float(self._prev_time):.6g}s then {float(src_time):.6g}s")
        target_indices = self._targets_between(self._prev_time, src_time)
        if not target_indices:
            # Identity / downsample case: no output frame falls in this gap.
            self._prev_src_pb = src_pb
            self._prev_time = src_time
            return

        phases = self._phases_between(
            target_indices, self._prev_time, src_time)
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

    def drain(self, hold: Fraction | int | float | None = None) -> Iterator[Any]:
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
        hold_span = (Fraction(hold) if hold is not None
                     else 1 / self.source_cadence)
        if hold_span <= 0:
            self._prev_src_pb = None
            return
        end_time = self._prev_time + hold_span
        target_indices = self._targets_between(self._prev_time, end_time)
        if not target_indices:
            self._prev_src_pb = None
            return
        phases = self._phases_between(
            target_indices, self._prev_time, end_time)
        yield from self._process_destination_batches(
            self._prev_src_pb,
            self._prev_time,
            self._prev_src_pb,
            end_time,
            target_indices,
            phases,
            f"VTFRC drain failed for final source period at "
            f"{float(self._prev_time):.6g}s",
        )
        self._next_target_index = target_indices[-1] + 1
        self._prev_src_pb = None
