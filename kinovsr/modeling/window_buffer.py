"""Shared fixed-sliding and GOP-aligned recurrent window buffering."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

GOP_SPLIT_TRIM = 2
_MISSING = object()


class WindowBuffer:
    """Own one recurrent stream's frames, tokens, ranges, and retention.

    The ordinary constructor preserves fixed-sliding behavior. ``gop()``
    selects reactive sync-aligned closes. ``run_window`` receives one buffered
    processing range and the local half-open range whose results are final.
    """

    def __init__(self, window: int, trim: int, run_window: Any):
        self._window = window
        self._trim = trim
        self._run_window = run_window
        self._minimum: int | None = None
        self.reset()

    @classmethod
    def gop(cls, minimum: int, maximum: int, run_window: Any) -> WindowBuffer:
        machine = cls(maximum, GOP_SPLIT_TRIM, run_window)
        machine._minimum = minimum
        return machine

    @property
    def is_gop(self) -> bool:
        return self._minimum is not None

    def reset(self) -> None:
        self._frames: list[Any] = []
        self._tokens: list[Any] = []
        self._base = 0
        self._emit = 0
        self._anchor = 0
        self._last_source: object | int | None = _MISSING

    @property
    def _total(self) -> int:
        return self._base + len(self._frames)

    def feed(self, frame: Any, token: Any = None) -> Iterable:
        if self.is_gop:
            return self._feed_gop(frame, token)
        self._frames.append(frame)
        self._tokens.append(token)
        out: list = []
        while self._total >= max(0, self._emit - self._trim) + self._window:
            start = max(0, self._emit - self._trim)
            end = start + self._window
            out.extend(self._run(start, end, end - self._trim))
        keep = max(0, min(
            self._emit - self._trim, self._total - self._window))
        self._discard(keep)
        return out

    def flush(self) -> Iterable:
        if self.is_gop:
            return self._flush_gop()
        out: list = []
        if self._emit < self._total:
            start = max(0, min(
                self._emit - self._trim, self._total - self._window))
            out.extend(self._run(start, self._total, self._total))
        self.reset()
        return out

    def _feed_gop(self, frame: Any, token: Any) -> Iterable:
        position = self._total
        self._frames.append(frame)
        self._tokens.append(token)

        source = getattr(token, "source", None)
        source_index = getattr(source, "index", None)
        first_source_slot = source_index != self._last_source
        self._last_source = source_index
        eligible = (
            bool(getattr(source, "is_sync", False))
            and first_source_slot
            and position - self._anchor >= self._minimum
        )

        while self._emit < self._total:
            forced = self._emit + self._window
            if eligible and position <= forced:
                yield from self._run(
                    self._gop_start(), position + 1, position,
                    anchored=True)
                break
            if self._total <= forced:
                break
            if eligible:
                yield from self._run(
                    self._gop_start(), position, forced, anchored=False)
                continue
            if self._total >= forced + self._trim:
                yield from self._run(
                    self._gop_start(), forced + self._trim, forced,
                    anchored=False)
                continue
            break

    def _flush_gop(self) -> Iterable:
        total = self._total
        while total - self._emit > self._window:
            forced = self._emit + self._window
            yield from self._run(
                self._gop_start(), min(total, forced + self._trim), forced,
                anchored=False)
        if self._emit < total:
            yield from self._run(
                self._gop_start(), total, total, anchored=False)
        self.reset()

    def _gop_start(self) -> int:
        if self._emit == self._anchor:
            return self._emit
        return max(self._anchor, self._emit - self._trim)

    def _run(self, start: int, end: int, emit_end: int, *,
             anchored: bool = False) -> Iterable:
        begin = start - self._base
        finish = end - self._base
        result = self._run_window(
            self._frames[begin:finish], self._tokens[begin:finish],
            self._emit - start, emit_end - start,
        )
        # Fixed mode has always completed the whole window before returning
        # from feed()/flush(), but need not retain discarded output frames.
        if not self.is_gop:
            result = list(result)
        yield from result
        self._emit = emit_end
        if anchored:
            self._anchor = emit_end
        if self.is_gop:
            self._discard(self._gop_start())

    def _discard(self, keep_from: int) -> None:
        drop = keep_from - self._base
        if drop > 0:
            del self._frames[:drop]
            del self._tokens[:drop]
            self._base = keep_from


__all__ = ["GOP_SPLIT_TRIM", "WindowBuffer"]
