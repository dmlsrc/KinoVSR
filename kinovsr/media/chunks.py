"""Shared validation for bounded decoded-frame chunks."""

from __future__ import annotations

from typing import Any

DECODE_MEMORY_BUDGET = 64 * 1024 * 1024


def validate_decode_chunk_size(value: Any) -> int:
    """Require a positive plain integer without lossy coercion."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"chunk_size must be a positive integer, got {value!r}")
    return value


def budgeted_decode_chunk_size(
    requested: int,
    width: int,
    height: int,
    bytes_per_pixel: int,
) -> int:
    """Apply the approximate retained-output-surface byte budget."""
    requested = validate_decode_chunk_size(requested)
    frame_bytes = max(
        1, int(width) * int(height) * int(bytes_per_pixel))
    budget_frames = max(1, DECODE_MEMORY_BUDGET // frame_bytes)
    return min(requested, budget_frames)


__all__ = [
    "DECODE_MEMORY_BUDGET",
    "budgeted_decode_chunk_size",
    "validate_decode_chunk_size",
]
