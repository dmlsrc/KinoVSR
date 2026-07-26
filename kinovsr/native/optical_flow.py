"""Shared safeguards for public VideoToolbox optical-flow destinations."""

from __future__ import annotations

import struct
from typing import Any

from .frameworks import Quartz

# macOS 26.5.2's public VTOpticalFlow implementation can return success
# without touching a destination whose width or height is below 128.  The
# configuration may advertise a smaller destination, so callers must enforce
# this measured writer boundary themselves.
VT_FLOW_MIN_DESTINATION_DIMENSION = 128

# Two distinct quiet-NaN payloads.  A valid flow field is finite, and VT
# overwrites every valid component on a successful write.  Marking one vector
# lets a caller distinguish a real all-zero field from the API's silent
# no-write behavior without scanning or clearing the whole IOSurface.
_PENDING_WRITE_MARKER = struct.pack("=HH", 0x7E01, 0x7E02)


def flow_destination_geometry(width: int, height: int) -> tuple[int, int]:
    """Return a writable destination in VT's rotation-normalized geometry.

    Portrait configurations expose and write their flow in a landscape
    coordinate system: destination width corresponds to source height and
    destination height to source width.  Keeping that orientation avoids the
    anisotropic vector scaling seen when a portrait-shaped destination is
    forced on the writer.
    """
    width = int(width)
    height = int(height)
    if width < 1 or height < 1:
        raise ValueError("optical-flow geometry must be positive")
    if height > width:
        width, height = height, width
    return (
        max(width, VT_FLOW_MIN_DESTINATION_DIMENSION),
        max(height, VT_FLOW_MIN_DESTINATION_DIMENSION),
    )


def source_sized_flow_is_reliable(width: int, height: int) -> bool:
    """Whether a source-sized destination clears VT's writer boundary."""
    return (
        int(width) >= VT_FLOW_MIN_DESTINATION_DIMENSION
        and int(height) >= VT_FLOW_MIN_DESTINATION_DIMENSION
    )


def mark_flow_buffer_pending(buffer: Any) -> None:
    """Mark one destination vector before a VTOpticalFlow submission."""
    Quartz.CVPixelBufferLockBaseAddress(buffer, 0)
    try:
        base = Quartz.CVPixelBufferGetBaseAddress(buffer)
        view = base.as_buffer(len(_PENDING_WRITE_MARKER))
        view[:] = _PENDING_WRITE_MARKER
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 0)


def flow_buffer_was_written(buffer: Any) -> bool:
    """Return false when VT left the pre-submission marker untouched."""
    Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
    try:
        base = Quartz.CVPixelBufferGetBaseAddress(buffer)
        view = base.as_buffer(len(_PENDING_WRITE_MARKER))
        return bytes(view) != _PENDING_WRITE_MARKER
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)


def mark_flow_pair_pending(pair: tuple[Any, Any]) -> None:
    """Mark the forward and backward destinations for one submission."""
    mark_flow_buffer_pending(pair[0])
    mark_flow_buffer_pending(pair[1])


def require_flow_pair_written(
    pair: tuple[Any, Any],
    *,
    context: str,
) -> None:
    """Raise when VT reported success but retained either pending marker."""
    missing = []
    if not flow_buffer_was_written(pair[0]):
        missing.append("forward")
    if not flow_buffer_was_written(pair[1]):
        missing.append("backward")
    if missing:
        width = int(Quartz.CVPixelBufferGetWidth(pair[0]))
        height = int(Quartz.CVPixelBufferGetHeight(pair[0]))
        raise RuntimeError(
            "VTOpticalFlow returned success without writing "
            f"{'/'.join(missing)} flow for {width}x{height} destination"
            f" ({context})"
        )
