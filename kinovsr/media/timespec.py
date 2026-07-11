"""Time/frame position specs: the user-facing forms, resolved to frames.

The CLI's --start/--end/--max-frames accept frames, seconds, or clock
strings; every consumer resolves them against a concrete fps through
these two helpers (moved verbatim from the inherited harness).
"""

from __future__ import annotations


def parse_time_or_frames(spec: str, fps: float) -> int:
    """Convert a position/duration spec to a frame count at `fps`.

    Accepted forms (case-insensitive):
        "120"         bare integer  -> 120 frames
        "120f"        explicit f    -> 120 frames
        "5s", "2.5s"  seconds       -> round(seconds * fps) frames
        "1.5"         bare decimal  -> seconds (a fractional frame is meaningless)
        "1:30"        mm:ss         -> seconds -> frames
        "1:02:03"     hh:mm:ss      -> seconds -> frames
        "0:04.5"      mm:ss.frac    -> seconds -> frames

    The bare-integer-is-frames / bare-decimal-is-seconds split keeps existing
    integer `--max-frames N` invocations meaning frames, while letting any
    time be given as seconds or a clock string. Returns a non-negative int.
    """
    s = str(spec).strip().lower()
    if not s:
        raise ValueError("empty time/frame spec")
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            hh, (mm, ss) = "0", parts
        elif len(parts) == 3:
            hh, mm, ss = parts
        else:
            raise ValueError(f"bad time spec {spec!r} (use mm:ss or hh:mm:ss)")
        seconds = int(hh) * 3600 + int(mm) * 60 + float(ss)
        frames = round(seconds * fps)
    elif s.endswith("f"):
        frames = int(s[:-1])
    elif s.endswith("s"):
        frames = round(float(s[:-1]) * fps)
    elif "." in s:
        frames = round(float(s) * fps)
    else:
        frames = int(s)
    if frames < 0:
        raise ValueError(f"time/frame spec {spec!r} is negative")
    return int(frames)


def resolve_trim(
    start_spec: str | None, end_spec: str | None, fps: float, total_frames: int,
) -> tuple[int, int | None]:
    """Resolve --start/--end specs to a half-open frame window [start, end).

    `end` is None for an open-ended window. Clamps end to the input length and
    rejects an empty or out-of-range window with a clean SystemExit.
    """
    try:
        start_frame = parse_time_or_frames(start_spec, fps) if start_spec else 0
        end_frame = parse_time_or_frames(end_spec, fps) if end_spec else None
    except ValueError as e:
        raise SystemExit(f"bad --start/--end value: {e}") from None
    if end_frame is not None and end_frame <= start_frame:
        raise SystemExit(
            f"--end ({end_frame}f) must be greater than --start ({start_frame}f)"
        )
    if total_frames and start_frame >= total_frames:
        raise SystemExit(
            f"--start ({start_frame}f) is at or past the input length "
            f"({total_frames} frames)"
        )
    if total_frames and end_frame is not None:
        end_frame = min(end_frame, total_frames)
    return start_frame, end_frame
