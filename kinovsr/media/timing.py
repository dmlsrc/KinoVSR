"""Exact source-cadence and integer-grid helpers.

File readers use :func:`analyze_sample_table` to build a display-ordered
per-sample timing table and classify how the observed presentation
timestamps relate to a constant grid. The file pipeline currently
publishes CFR output only, so its accept/reject contract keys off the
legacy :class:`VideoTiming` view (:meth:`SampleTable.timing`); the table
itself additionally distinguishes a gapped constant grid from genuinely
variable timing and carries per-sample sync flags and coded sizes, the
groundwork for carrying non-uniform sources and GOP-aware windowing.

The same rational grid helper owns writer timestamps. Rounding each
complete ``frame_index / cadence`` value avoids the unbounded drift
caused by rounding one frame duration and multiplying it by the index.
"""

from __future__ import annotations

import bisect
import dataclasses
import enum
import math
from collections import Counter
from collections.abc import Iterable
from fractions import Fraction


class TimingVerdict(enum.Enum):
    """How observed display timestamps relate to a constant grid."""

    EXACT_CFR = "exact_cfr"      # every interval identical
    CFR = "cfr"                  # bounded source-tick quantization of one grid
    GAPPED_CFR = "gapped_cfr"    # one grid with missing slots (dropped frames)
    VFR = "vfr"                  # genuinely variable


@dataclasses.dataclass(frozen=True, slots=True)
class SampleTiming:
    """One display sample's metadata-only timing record.

    ``is_sync`` is ``None`` when the reader did not report sync flags
    (pure timestamp inputs); readers that walk coded samples report the
    real flag. ``coded_size`` is the coded payload size in bytes when
    the reader knows it.
    """

    pts: Fraction
    duration: Fraction | None
    is_sync: bool | None = None
    coded_size: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class VideoTiming:
    """Observed display-sample timing for one video track.

    ``cadence`` is ``None`` when the presentation timestamps are variable.
    ``duration`` spans the first presentation timestamp through the final
    sample duration; it is diagnostic for VFR and exact ``count / cadence``
    for accepted CFR input.
    """

    sample_count: int
    cadence: Fraction | None
    first_pts: Fraction
    duration: Fraction
    source_tick: Fraction


@dataclasses.dataclass(frozen=True, slots=True)
class AudioTiming:
    """Presentation origin and clock precision for one audio track."""

    first_pts: Fraction
    source_tick: Fraction


@dataclasses.dataclass(frozen=True, slots=True)
class SampleTable:
    """Display-ordered per-sample timing with its grid classification.

    ``cadence`` preserves the legacy CFR-accept contract: it is set only
    for the EXACT_CFR/CFR verdicts (a grid the file pipeline may publish
    as-is). ``grid_cadence`` additionally carries the underlying grid of
    a GAPPED_CFR source; it is ``None`` only for VFR. ``duration`` is
    exact ``count / cadence`` for accepted CFR input and the observed
    first-pts-through-final-sample span otherwise.
    """

    samples: tuple[SampleTiming, ...]
    verdict: TimingVerdict
    cadence: Fraction | None
    grid_cadence: Fraction | None
    first_pts: Fraction
    duration: Fraction
    source_tick: Fraction
    # Display-order table positions whose samples are sync samples.
    # A stored field, not a property: frame_gop() bisects it once per
    # frame, and recomputing it per call made per-frame GOP metadata
    # quadratic in clip length (a 90-minute clip appeared hung).
    keyframe_indices: tuple[int, ...] = ()
    # Mid-file timestamp resets the reader unwrapped into this table's
    # monotonic clock. Nonzero disables coarse decode seeks: a seek that
    # lands inside a later epoch would start the decode-side unwrapper
    # without the history that maps raw stamps onto this clock.
    epoch_resets: int = 0

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def frame_gop(self, index: int) -> tuple[int, int] | None:
        """(gop_ordinal, gop_length) for one display position.

        ``gop_ordinal`` is the frame's distance from its enclosing sync
        sample and ``gop_length`` that GOP's span (sync to next sync, in
        samples) - ABSOLUTE source properties, independent of any window.
        Samples before the first sync sample count from position 0 (an
        open leading GOP). ``None`` when the reader reported no sync
        flags at all.
        """
        if not 0 <= index < len(self.samples):
            raise IndexError(f"sample index {index} out of range")
        keys = self.keyframe_indices
        if not keys:
            return None
        position = bisect.bisect_right(keys, index) - 1
        start = keys[position] if position >= 0 else 0
        end = (keys[position + 1] if position + 1 < len(keys)
               else len(self.samples))
        return index - start, end - start

    def timing(self) -> VideoTiming:
        """The legacy CFR-or-variable view consumed by the file pipeline."""
        return VideoTiming(
            sample_count=self.sample_count,
            cadence=self.cadence,
            first_pts=self.first_pts,
            duration=self.duration,
            source_tick=self.source_tick,
        )


# Backward jumps beyond this many seconds are container joins / encoder
# clock resets, not decode reordering: B-frame dips span a reorder depth
# of a few frame intervals, while joined transport streams and camera
# segment resets jump back by minutes or to zero.
EPOCH_RESET_SECONDS = 10


class EpochUnwrapper:
    """Streaming raw-to-monotonic PTS mapping for mid-file clock resets.

    Concatenated transport streams and camera joins reset PTS mid-file.
    Globally sorting such stamps interleaves the segments: the later
    segment's small stamps sort FIRST, so every carried timestamp, sync
    flag, and coded size lands on the wrong decoded frame. Fed stamps in
    stream order, this maps each raw PTS onto one declared monotonic
    clock: a backward jump larger than :data:`EPOCH_RESET_SECONDS` starts
    a new epoch placed one typical frame interval after the previous
    epoch's end. Bounded decode reordering passes through untouched, and
    because reset detection depends only on the large jump - which
    survives B-frame reordering - the metadata walk (demux order) and the
    decode path (display order within each epoch) compute identical
    mappings, so window bisects stay aligned. Segments are assumed not to
    interleave across a join (falsifiable: a B-frame straddling a
    concatenation boundary would desync the two mappings).
    """

    def __init__(self) -> None:
        self._offset = Fraction(0)
        self._prev_raw: Fraction | None = None
        self._max_adjusted: Fraction | None = None
        self._last_delta: Fraction | None = None
        self.resets = 0

    def push(self, pts: Fraction) -> Fraction:
        pts = Fraction(pts)
        if self._prev_raw is not None:
            if self._prev_raw - pts > EPOCH_RESET_SECONDS:
                gap = (self._last_delta
                       if self._last_delta and self._last_delta > 0
                       else Fraction(1, 25))
                assert self._max_adjusted is not None
                self._offset = self._max_adjusted + gap - pts
                self._last_delta = None
                self.resets += 1
            elif pts > self._prev_raw:
                self._last_delta = pts - self._prev_raw
        self._prev_raw = pts
        adjusted = pts + self._offset
        if self._max_adjusted is None or adjusted > self._max_adjusted:
            self._max_adjusted = adjusted
        return adjusted


def rational_cadence(
    value: Fraction | int | float | str,
    *,
    max_denominator: int = 1_000_000,
) -> Fraction:
    """Return a positive exact cadence from a public or probed value."""
    if isinstance(value, bool):
        raise ValueError("cadence must be a positive number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("cadence must be finite")
    try:
        cadence = (value if isinstance(value, Fraction)
                   else Fraction(str(value)).limit_denominator(max_denominator))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid cadence {value!r}") from exc
    if cadence <= 0:
        raise ValueError(f"cadence must be positive, got {value!r}")
    return cadence


def grid_ticks(
    frame_index: int,
    cadence: Fraction | int | float | str,
    time_base: Fraction,
) -> int:
    """Round one complete rational frame position into integer ticks."""
    exact_cadence = rational_cadence(cadence)
    return round(Fraction(frame_index) / exact_cadence / time_base)


def _quantization_tolerance(value: Fraction, source_tick: Fraction) -> Fraction:
    # An interval that is exactly representable on the source clock has no
    # quantization uncertainty. Giving it a whole tick would hide a dropped
    # frame when the clock itself is as coarse as the cadence (for example,
    # 25 fps timestamps on a 1/25 time base).
    return (Fraction()
            if (value / source_tick).denominator == 1
            else source_tick)


def _uniform_cadence(
    ordered: list[SampleTiming],
    deltas: list[Fraction],
    nominal: Fraction | None,
    source_tick: Fraction,
) -> Fraction | None:
    """The single constant grid the samples sit on, or ``None``.

    A CFR container may quantize adjacent intervals by one source tick
    (notably NTSC rates); that bounded alternation is accepted, while
    duplicated or materially different intervals are not.
    """
    first_pts = ordered[0].pts
    cadence: Fraction | None = None

    if not deltas:
        only_duration = ordered[0].duration
        if nominal is not None:
            cadence = nominal
        elif only_duration is not None:
            cadence = 1 / only_duration
    elif all(delta > 0 for delta in deltas):
        span = ordered[-1].pts - first_pts
        mean_delta = span / len(deltas)

        def fits(candidate: Fraction) -> bool:
            interval = 1 / candidate
            return (
                abs(span - len(deltas) * interval)
                <= _quantization_tolerance(
                    len(deltas) * interval, source_tick)
                and all(abs(delta - interval)
                        <= _quantization_tolerance(interval, source_tick)
                        for delta in deltas)
            )

        if all(abs(delta - mean_delta) <= source_tick
               for delta in deltas):
            observed = 1 / mean_delta
            if nominal is not None and fits(nominal):
                cadence = nominal
            elif all(delta == mean_delta for delta in deltas):
                # A uniform sequence is exact on the source clock even when
                # it has only a few ticks per frame. No quantization pattern
                # needs to be inferred, so absent or stale nominal metadata
                # cannot make it ambiguous.
                cadence = observed
            elif mean_delta >= 4 * source_tick:
                # A container can round nominal metadata to 24/30/60 while
                # its long-run clock is 24000/1001, 30000/1001, or
                # 60000/1001. Prefer that broadcast family only when its
                # cumulative span also fits its source-clock uncertainty.
                # Do not infer a different cadence from a clock with fewer
                # than four ticks per frame: there is too little resolution
                # to distinguish quantization from a dropped-frame pattern.
                broadcast = Fraction(round(observed * 1001), 1001)
                cadence = broadcast if fits(broadcast) else observed

    if cadence is not None:
        interval = 1 / cadence
        duration_tolerance = _quantization_tolerance(interval, source_tick)
        last_index = len(ordered) - 1
        for index, sample in enumerate(ordered):
            duration = sample.duration
            if duration is None:
                continue
            if abs(duration - interval) <= duration_tolerance:
                continue
            if index == last_index and duration < interval:
                # A truncated FINAL duration is a container-end artifact
                # (duration-preserving writers clamp it; editors cut
                # mid-interval). It cannot lie about mid-stream display
                # timing, so it does not demote the clock. A LONG tail
                # still does - it extends playback (see the
                # silent-truncation pin in the tests).
                continue
            cadence = None
            break

    return cadence


def _gapped_grid(
    deltas: list[Fraction],
    nominal: Fraction | None,
    source_tick: Fraction,
) -> Fraction | None:
    """The constant grid a monotonic gapped sequence sits on, or ``None``.

    Every interval must be an exact positive integer multiple of one base
    interval (within the same per-interval source-tick tolerance the CFR
    accept uses), and at least one multiple must exceed one - otherwise
    the sequence either is CFR (already accepted upstream) or carries
    quantization the CFR rules rejected, which gapped rules must not
    quietly re-admit.
    """
    if not deltas or any(delta <= 0 for delta in deltas):
        return None

    candidates: list[Fraction] = []

    def add(interval: Fraction) -> None:
        if interval > 0 and interval not in candidates:
            candidates.append(interval)

    if nominal is not None:
        add(1 / nominal)
    counts = Counter(deltas)
    top = max(counts.values())
    modal = min(delta for delta, count in counts.items() if count == top)
    for base in (modal, min(deltas)):
        add(base)
        broadcast = Fraction(round((1 / base) * 1001), 1001)
        if broadcast > 0:
            add(1 / broadcast)

    for interval in candidates:
        multiples: list[int] = []
        for delta in deltas:
            multiple = round(delta / interval)
            if multiple < 1 or (
                    abs(delta - multiple * interval)
                    > _quantization_tolerance(
                        multiple * interval, source_tick)):
                multiples.clear()
                break
            multiples.append(multiple)
        if multiples and any(multiple > 1 for multiple in multiples):
            return 1 / interval
    return None


def analyze_sample_table(
    samples: Iterable[SampleTiming],
    *,
    nominal_cadence: Fraction | int | float | str | None,
    source_tick: Fraction,
    epoch_resets: int = 0,
) -> SampleTable:
    """Classify exact display timing records as one grid, gapped, or VFR.

    Records may arrive in decode order, so classification is performed
    after sorting by presentation time. Nonpositive durations are treated
    as absent. The legacy CFR accept (``cadence``) is unchanged from
    :func:`analyze_sample_timing`'s original contract; the verdict and
    ``grid_cadence`` add the gapped/variable distinction on top.
    """
    if source_tick <= 0:
        raise ValueError("source_tick must be positive")
    ordered = sorted(
        (dataclasses.replace(
            sample,
            pts=Fraction(sample.pts),
            duration=(Fraction(sample.duration)
                      if sample.duration is not None and sample.duration > 0
                      else None))
         for sample in samples),
        key=lambda sample: sample.pts,
    )
    if not ordered:
        raise ValueError("video track contains no display samples")

    first_pts = ordered[0].pts
    count = len(ordered)
    deltas = [ordered[i + 1].pts - ordered[i].pts
              for i in range(count - 1)]

    nominal = None
    if nominal_cadence is not None:
        try:
            nominal = rational_cadence(
                nominal_cadence, max_denominator=1001)
        except ValueError:
            nominal = None

    cadence = _uniform_cadence(ordered, deltas, nominal, source_tick)
    grid_cadence = cadence
    if cadence is not None:
        verdict = (TimingVerdict.EXACT_CFR
                   if all(delta == deltas[0] for delta in deltas)
                   else TimingVerdict.CFR)
    else:
        grid_cadence = _gapped_grid(deltas, nominal, source_tick)
        verdict = (TimingVerdict.GAPPED_CFR if grid_cadence is not None
                   else TimingVerdict.VFR)

    if cadence is not None:
        duration = Fraction(count) / cadence
    else:
        last = ordered[-1]
        tail = (last.duration if last.duration is not None
                else deltas[-1] if deltas else Fraction())
        duration = last.pts - first_pts + tail

    return SampleTable(
        samples=tuple(ordered),
        verdict=verdict,
        cadence=cadence,
        grid_cadence=grid_cadence,
        first_pts=first_pts,
        duration=duration,
        source_tick=source_tick,
        keyframe_indices=tuple(
            index for index, sample in enumerate(ordered)
            if sample.is_sync),
        epoch_resets=int(epoch_resets),
    )


def analyze_sample_timing(
    samples: Iterable[tuple[Fraction, Fraction | None]],
    *,
    nominal_cadence: Fraction | int | float | str | None,
    source_tick: Fraction,
) -> VideoTiming:
    """Classify exact display PTS values as CFR or VFR.

    The legacy pair-based entry point: builds the sample table from bare
    ``(pts, duration)`` pairs and returns its CFR-or-variable view.
    """
    table = analyze_sample_table(
        (SampleTiming(pts=pts, duration=duration)
         for pts, duration in samples),
        nominal_cadence=nominal_cadence,
        source_tick=source_tick,
    )
    return table.timing()


__all__ = [
    "AudioTiming",
    "SampleTable",
    "SampleTiming",
    "TimingVerdict",
    "VideoTiming",
    "EPOCH_RESET_SECONDS",
    "EpochUnwrapper",
    "analyze_sample_table",
    "analyze_sample_timing",
    "grid_ticks",
    "rational_cadence",
]
