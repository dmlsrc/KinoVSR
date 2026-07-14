"""Exact source-cadence and integer-grid helpers.

File readers use :func:`analyze_sample_timing` to decide whether the observed
presentation timestamps form one constant cadence. The file pipeline is CFR
only for now, so a variable result is rejected before output reservation.

The same rational grid helper owns writer timestamps. Rounding each complete
``frame_index / cadence`` value avoids the unbounded drift caused by rounding
one frame duration and multiplying it by the index.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from fractions import Fraction


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


def analyze_sample_timing(
    samples: Iterable[tuple[Fraction, Fraction | None]],
    *,
    nominal_cadence: Fraction | int | float | str | None,
    source_tick: Fraction,
) -> VideoTiming:
    """Classify exact display PTS values as CFR or VFR.

    Presentation timestamps may arrive in decode order, so classification is
    performed after sorting. A CFR container may quantize adjacent intervals
    by one source tick (notably NTSC rates); that bounded alternation is
    accepted, while duplicated or materially different intervals are VFR.
    """
    if source_tick <= 0:
        raise ValueError("source_tick must be positive")
    ordered = sorted(
        ((Fraction(pts), duration) for pts, duration in samples),
        key=lambda sample: sample[0],
    )
    if not ordered:
        raise ValueError("video track contains no display samples")

    first_pts = ordered[0][0]
    count = len(ordered)
    deltas = [ordered[i + 1][0] - ordered[i][0]
              for i in range(count - 1)]
    cadence: Fraction | None = None

    nominal = None
    if nominal_cadence is not None:
        try:
            nominal = rational_cadence(
                nominal_cadence, max_denominator=1001)
        except ValueError:
            nominal = None

    if not deltas:
        only_duration = ordered[0][1]
        if nominal is not None:
            cadence = nominal
        elif only_duration is not None and only_duration > 0:
            cadence = 1 / Fraction(only_duration)
    elif all(delta > 0 for delta in deltas):
        span = ordered[-1][0] - first_pts
        mean_delta = span / len(deltas)

        def quantization_tolerance(value: Fraction) -> Fraction:
            # An interval that is exactly representable on the source clock
            # has no quantization uncertainty. Giving it a whole tick would
            # hide a dropped frame when the clock itself is as coarse as the
            # cadence (for example, 25 fps timestamps on a 1/25 time base).
            return (Fraction()
                    if (value / source_tick).denominator == 1
                    else source_tick)

        def fits(candidate: Fraction) -> bool:
            interval = 1 / candidate
            return (
                abs(span - len(deltas) * interval)
                <= quantization_tolerance(len(deltas) * interval)
                and all(abs(delta - interval)
                        <= quantization_tolerance(interval)
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
        durations = [Fraction(duration) for _, duration in ordered
                     if duration is not None and duration > 0]
        duration_tolerance = (
            Fraction()
            if (interval / source_tick).denominator == 1
            else source_tick)
        if any(abs(duration - interval) > duration_tolerance
               for duration in durations):
            cadence = None

    last_pts, last_duration = ordered[-1]
    if cadence is not None:
        duration = Fraction(count) / cadence
    else:
        tail = (Fraction(last_duration)
                if last_duration is not None and last_duration > 0
                else deltas[-1] if deltas else Fraction())
        duration = last_pts - first_pts + tail

    return VideoTiming(
        sample_count=count,
        cadence=cadence,
        first_pts=first_pts,
        duration=duration,
        source_tick=source_tick,
    )


__all__ = [
    "AudioTiming",
    "VideoTiming",
    "analyze_sample_timing",
    "grid_ticks",
    "rational_cadence",
]
