"""Validated Vision optical-flow buffers for native and MLX consumers.

Vision revision 1 at Medium accuracy is the measured native flow variant used
elsewhere in KinoVSR. This adapter asks Vision for IOSurface-backed
TwoComponent16Half buffers so VideoToolbox can consume them directly: no MLX
readback, format conversion, or buffer copy sits between the two frameworks.
"""

from __future__ import annotations

from typing import Any

import Vision

from .frameworks import Quartz

# kCVPixelFormatType_TwoComponent16Half ("2C0h"). This is the native public
# VT optical-flow format and the format accepted by VTFrameProcessorOpticalFlow.
FLOW_16H = 0x32433068
# kCVPixelFormatType_TwoComponent32Float ("2C0f"). MLX consumers request this
# format to preserve precision through their direct readback.
FLOW_32F = 0x32433066


def _validate_flow_buffer(
    buffer: Any,
    *,
    width: int,
    height: int,
    pixel_format: int,
) -> None:
    actual_width = int(Quartz.CVPixelBufferGetWidth(buffer))
    actual_height = int(Quartz.CVPixelBufferGetHeight(buffer))
    if (actual_width, actual_height) != (width, height):
        raise RuntimeError(
            "Vision optical flow returned "
            f"{actual_width}x{actual_height} for a {width}x{height} request"
        )
    actual_pixel_format = int(Quartz.CVPixelBufferGetPixelFormatType(buffer))
    if actual_pixel_format != pixel_format:
        raise RuntimeError(
            "Vision optical flow returned pixel format "
            f"{actual_pixel_format:#x}, expected {pixel_format:#x}"
        )
    if Quartz.CVPixelBufferGetIOSurface(buffer) is None:
        raise RuntimeError("Vision optical flow returned a non-IOSurface buffer")


def generate_vision_flow(
    from_buffer: Any,
    to_buffer: Any,
    *,
    pixel_format: int = FLOW_16H,
) -> Any:
    """Return source-geometry flow from ``from_buffer`` to ``to_buffer``.

    The returned CVPixelBuffer is retained by its PyObjC wrapper and can be
    handed directly to ``VTFrameProcessorOpticalFlow``.
    """
    width = int(Quartz.CVPixelBufferGetWidth(from_buffer))
    height = int(Quartz.CVPixelBufferGetHeight(from_buffer))
    target_width = int(Quartz.CVPixelBufferGetWidth(to_buffer))
    target_height = int(Quartz.CVPixelBufferGetHeight(to_buffer))
    if (target_width, target_height) != (width, height):
        raise ValueError(
            "Vision optical-flow source and target geometry differ: "
            f"{width}x{height} vs {target_width}x{target_height}"
        )

    request = Vision.VNGenerateOpticalFlowRequest.alloc(
    ).initWithTargetedCVPixelBuffer_options_(to_buffer, {})
    request.setRevision_(Vision.VNGenerateOpticalFlowRequestRevision1)
    request.setComputationAccuracy_(
        Vision.VNGenerateOpticalFlowRequestComputationAccuracyMedium
    )
    request.setOutputPixelFormat_(pixel_format)
    handler = Vision.VNImageRequestHandler.alloc(
    ).initWithCVPixelBuffer_options_(from_buffer, {})
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision optical flow failed: {error}")
    results = request.results()
    if not results:
        raise RuntimeError("Vision optical flow returned no observation")
    output = results[0].pixelBuffer()
    if output is None:
        raise RuntimeError("Vision optical flow observation has no pixel buffer")
    _validate_flow_buffer(
        output,
        width=width,
        height=height,
        pixel_format=pixel_format,
    )
    return output


def generate_bidirectional_vision_flow(
    previous_buffer: Any,
    current_buffer: Any,
) -> tuple[Any, Any]:
    """Return previous->current and current->previous Vision flow buffers."""
    return (
        generate_vision_flow(previous_buffer, current_buffer),
        generate_vision_flow(current_buffer, previous_buffer),
    )


__all__ = [
    "FLOW_16H",
    "FLOW_32F",
    "generate_bidirectional_vision_flow",
    "generate_vision_flow",
]
