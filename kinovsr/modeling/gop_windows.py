"""Reactive GOP-policy window buffering shared by recurrent families."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

GOP_SPLIT_TRIM = 2
_MISSING = object()


class GopWindows:
    """Buffer a stream into sync-anchored or bounded recurrent windows.

    ``run_window`` receives the buffered frames and tokens plus the local
    half-open emit range. It may return a list or stream results lazily.
    """

    def __init__(self, min_window: int, max_window: int, run_window: Any):
        self._min = min_window
        self._max = max_window
        self._run_window = run_window
        self.reset()

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
            and position - self._anchor >= self._min
        )

        while self._emit < self._total:
            forced = self._emit + self._max
            if eligible and position <= forced:
                yield from self._run(position + 1, position, anchored=True)
                break
            if self._total <= forced:
                break
            if eligible:
                # The max split was committed before this sync arrived. Keep
                # only the trailing frames preceding the sync as context, then
                # let the same sync close the short remainder on the next pass.
                yield from self._run(position, forced, anchored=False)
                continue
            if self._total >= forced + GOP_SPLIT_TRIM:
                yield from self._run(
                    forced + GOP_SPLIT_TRIM, forced, anchored=False)
                continue
            break

    def flush(self) -> Iterable:
        total = self._total
        while total - self._emit > self._max:
            forced = self._emit + self._max
            yield from self._run(
                min(total, forced + GOP_SPLIT_TRIM),
                forced,
                anchored=False,
            )
        if self._emit < total:
            yield from self._run(total, total, anchored=False)

    def _proc_start(self) -> int:
        if self._emit == self._anchor:
            return self._emit
        return max(self._anchor, self._emit - GOP_SPLIT_TRIM)

    def _run(self, proc_end: int, emit_end: int, *, anchored: bool) -> Iterable:
        proc_start = self._proc_start()
        emit_start = self._emit
        begin = proc_start - self._base
        end = proc_end - self._base
        frames = self._frames[begin:end]
        tokens = self._tokens[begin:end]
        yield from self._run_window(
            frames,
            tokens,
            emit_start - proc_start,
            emit_end - proc_start,
        )
        self._emit = emit_end
        if anchored:
            self._anchor = emit_end
        keep_from = self._proc_start()
        drop = keep_from - self._base
        if drop > 0:
            del self._frames[:drop]
            del self._tokens[:drop]
            self._base = keep_from


__all__ = ["GOP_SPLIT_TRIM", "GopWindows"]
