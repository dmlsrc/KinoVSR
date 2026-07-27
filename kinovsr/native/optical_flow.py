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


def advertised_flow_destination_geometry(
    advertised: Any,
) -> tuple[int, int]:
    """Return the unmodified destination geometry advertised by VT.

    This is the coordinate contract consumed by VT SR, not necessarily a
    writable destination for the public VT optical-flow producer. In
    particular, do not apply ``VT_FLOW_MIN_DESTINATION_DIMENSION`` here:
    Vision writes its own full-resolution field and a Metal conversion writes
    this destination, so the public VT producer's measured 128-pixel floor is
    irrelevant.
    """
    attrs = dict(advertised or {})
    try:
        width = int(attrs.get(Quartz.kCVPixelBufferWidthKey, 0) or 0)
        height = int(attrs.get(Quartz.kCVPixelBufferHeightKey, 0) or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width < 1 or height < 1:
        raise RuntimeError(
            "VT optical-flow configuration advertised invalid destination "
            f"geometry {width}x{height}"
        )
    return width, height


def select_flow_destination_geometry(
    advertised: Any,
    width: int,
    height: int,
) -> tuple[int, int, bool]:
    """Pick the flow destination, preferring the configuration's own shape.

    Returns ``(w, h, used_advertised)``.

    Flow vectors are expressed in destination-buffer coordinates, so the
    destination shape is not a free choice: it scales the field the SR scaler
    consumes.  Measured on an 854x480 clip, where the advertised destination is
    240x135, forcing the full source size instead raised edge temporal
    instability from 4.89 to 8.57 while leaving spatial detail unchanged
    (gradient energy 8.20 versus 8.23), so the forced shape was over-warping
    rather than adding information.  Accuracy is equal either way: a known
    +3 source-pixel shift reads +0.811 into 240x135 against an expected +0.843,
    and +2.857 into 854x480 against an expected +3.000 - the same ~0.95 ratio.

    ``VT_FLOW_MIN_DESTINATION_DIMENSION`` records that VT returns success without
    writing a destination below 128 in either dimension, and a configuration can
    advertise exactly that: 640x360 advertises 160x90 and 640x480 advertises
    160x120.  When that happens, REPAIR the advertisement minimally by raising
    only the offending dimension rather than replacing it with the full source
    size.  Measured on a 640x360 clip whose advertisement is unusable, edge
    temporal instability was 4.28 at the repaired 160x128, against 4.75 at an
    aspect-preserving 228x128, 4.85 at the canonical 240x135, 4.62 for
    non-temporal Image mode, and 7.96 at the forced full 640x360 - so the
    minimal repair is both the best measured option and the only one that beats
    Image.  Spatial detail is within about 2% of Image across all three coarse
    shapes, so this is not a blur win.

    The source-sized shape remains the last resort for a configuration that
    advertises nothing usable at all.
    """
    fallback_w, fallback_h = flow_destination_geometry(width, height)
    attrs = dict(advertised or {})
    try:
        adv_w = int(attrs.get(Quartz.kCVPixelBufferWidthKey, 0) or 0)
        adv_h = int(attrs.get(Quartz.kCVPixelBufferHeightKey, 0) or 0)
    except (TypeError, ValueError):
        return fallback_w, fallback_h, False
    if adv_w < 1 or adv_h < 1:
        return fallback_w, fallback_h, False
    if (
        adv_w >= VT_FLOW_MIN_DESTINATION_DIMENSION
        and adv_h >= VT_FLOW_MIN_DESTINATION_DIMENSION
    ):
        return adv_w, adv_h, True
    return (
        max(adv_w, VT_FLOW_MIN_DESTINATION_DIMENSION),
        max(adv_h, VT_FLOW_MIN_DESTINATION_DIMENSION),
        False,
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
