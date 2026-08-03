"""Bounded unit-and-byte channels for the streaming graph."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from .leases import Envelope


class ChannelClosed(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EndOfStream:
    pass


EOS = EndOfStream()
type ChannelMessage = Envelope | EndOfStream


class BoundedChannel:
    """FIFO bounded independently by message count and payload bytes."""

    def __init__(
        self,
        *,
        max_units: int,
        max_bytes: int,
        name: str,
        observer=None,
    ) -> None:
        if max_units <= 0:
            raise ValueError("channel max_units must be positive")
        if max_bytes <= 0:
            raise ValueError("channel max_bytes must be positive")
        self.name = name
        self.max_units = int(max_units)
        self.max_bytes = int(max_bytes)
        self._items: deque[ChannelMessage] = deque()
        self._bytes = 0
        self._closed = False
        self._condition = threading.Condition()
        self._observer = observer
        self.high_water_units = 0
        self.high_water_bytes = 0
        self.put_count = 0
        self.get_count = 0
        self.put_wait_count = 0
        self.put_wait_seconds = 0.0

    def _observe(self, action: str, units: int, payload_bytes: int) -> None:
        if self._observer is not None:
            self._observer(action, self.name, units, payload_bytes)

    @staticmethod
    def _charge(message: ChannelMessage) -> int:
        return message.estimated_bytes if isinstance(message, Envelope) else 0

    def put(self, message: ChannelMessage) -> None:
        charge = self._charge(message)
        if charge > self.max_bytes:
            raise ValueError(
                f"message charge {charge} exceeds {self.name} byte bound "
                f"{self.max_bytes}"
            )
        waited_from: float | None = None
        with self._condition:
            while not self._closed and (
                len(self._items) >= self.max_units
                or self._bytes + charge > self.max_bytes
            ):
                if waited_from is None:
                    waited_from = time.perf_counter()
                self._condition.wait()
            if self._closed:
                raise ChannelClosed(f"channel {self.name} is closed")
            if waited_from is not None:
                self.put_wait_count += 1
                self.put_wait_seconds += time.perf_counter() - waited_from
            self._items.append(message)
            self._bytes += charge
            self.put_count += 1
            self.high_water_units = max(self.high_water_units, len(self._items))
            self.high_water_bytes = max(self.high_water_bytes, self._bytes)
            units, payload_bytes = len(self._items), self._bytes
            self._condition.notify_all()
        self._observe("put", units, payload_bytes)

    def get(self) -> ChannelMessage:
        with self._condition:
            while not self._items and not self._closed:
                self._condition.wait()
            if not self._items:
                raise ChannelClosed(f"channel {self.name} is closed")
            message = self._items.popleft()
            self._bytes -= self._charge(message)
            self.get_count += 1
            units, payload_bytes = len(self._items), self._bytes
            self._condition.notify_all()
        self._observe("get", units, payload_bytes)
        return message

    def close(self) -> None:
        with self._condition:
            self._closed = True
            units, payload_bytes = len(self._items), self._bytes
            self._condition.notify_all()
        self._observe("close", units, payload_bytes)

    def drain(self) -> None:
        """Release all payloads left behind by cancellation or failure."""
        with self._condition:
            items, self._items = self._items, deque()
            self._bytes = 0
            self._condition.notify_all()
        self._observe("drain", 0, 0)
        for message in items:
            if isinstance(message, Envelope):
                message.release()

    @property
    def current_units(self) -> int:
        with self._condition:
            return len(self._items)

    @property
    def current_bytes(self) -> int:
        with self._condition:
            return self._bytes

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed


__all__ = [
    "BoundedChannel",
    "ChannelClosed",
    "ChannelMessage",
    "EOS",
    "EndOfStream",
]
