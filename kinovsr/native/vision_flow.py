"""Validated Vision optical-flow buffers for native and MLX consumers.

Callers select the measured policy for their consumer. Shared MLX flow keeps
revision 1 Medium, while VideoToolbox SR uses revision 1 High. This adapter can
ask Vision for IOSurface-backed TwoComponent16Half buffers so VideoToolbox
consumes them without an MLX or CPU readback. VT SR's explicit-flow consumer
uses a smaller, rotation-normalized coordinate system, so
``VisionFlowToVtConverter`` performs the required field resampling and vector
unit conversion directly between IOSurfaces on Metal.
"""

from __future__ import annotations

import threading
from typing import Any

import Vision

from .frameworks import Quartz

# kCVPixelFormatType_TwoComponent16Half ("2C0h"). This is the native public
# VT optical-flow format and the format accepted by VTFrameProcessorOpticalFlow.
FLOW_16H = 0x32433068
# kCVPixelFormatType_TwoComponent32Float ("2C0f"). MLX consumers request this
# format to preserve precision through their direct readback.
FLOW_32F = 0x32433066

_ACCURACY = {
    "medium": Vision.VNGenerateOpticalFlowRequestComputationAccuracyMedium,
    "high": Vision.VNGenerateOpticalFlowRequestComputationAccuracyHigh,
}

_FLOW_CONVERSION_METAL = r"""
#include <metal_stdlib>
using namespace metal;

inline float2 landscape_value(
    texture2d<half, access::read> source,
    uint x,
    uint y)
{
    return float2(source.read(uint2(x, y)).xy);
}

inline float2 portrait_ccw_value(
    texture2d<half, access::read> source,
    uint x,
    uint y)
{
    const uint source_x = source.get_width() - 1 - y;
    const uint source_y = x;
    const float2 value = float2(source.read(uint2(source_x, source_y)).xy);
    return float2(value.y, -value.x);
}

template <typename ReadValue>
inline float2 resample_flow(
    texture2d<half, access::read> source,
    uint2 oriented_size,
    texture2d<half, access::write> destination,
    uint2 gid,
    ReadValue read_value)
{
    const uint2 destination_size = uint2(
        destination.get_width(), destination.get_height());
    const float2 source_scale =
        float2(oriented_size) / float2(destination_size);
    float2 value;

    if (any(destination_size > oriented_size)) {
        const float2 maximum = float2(oriented_size - 1);
        const float2 position = clamp(
            (float2(gid) + 0.5f) * source_scale - 0.5f,
            float2(0.0f),
            maximum);
        const uint2 low = uint2(floor(position));
        const uint2 high = min(low + 1, oriented_size - 1);
        const float2 fraction = position - float2(low);
        const float2 top = mix(
            read_value(source, low.x, low.y),
            read_value(source, high.x, low.y),
            fraction.x);
        const float2 bottom = mix(
            read_value(source, low.x, high.y),
            read_value(source, high.x, high.y),
            fraction.x);
        value = mix(top, bottom, fraction.y);
    } else {
        const float2 start = float2(gid) * source_scale;
        const float2 end = float2(gid + 1) * source_scale;
        const uint2 first = uint2(floor(start));
        const uint2 stop = min(oriented_size, uint2(ceil(end)));
        float2 sum = float2(0.0f);
        float weight_sum = 0.0f;
        for (uint y = first.y; y < stop.y; ++y) {
            const float weight_y = max(
                0.0f,
                min(end.y, float(y + 1)) - max(start.y, float(y)));
            for (uint x = first.x; x < stop.x; ++x) {
                const float weight_x = max(
                    0.0f,
                    min(end.x, float(x + 1)) - max(start.x, float(x)));
                const float weight = weight_x * weight_y;
                sum += read_value(source, x, y) * weight;
                weight_sum += weight;
            }
        }
        value = sum / weight_sum;
    }

    return value * float2(destination_size) / float2(oriented_size);
}

kernel void flow_resample_landscape(
    texture2d<half, access::read> source [[texture(0)]],
    texture2d<half, access::write> destination [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint2 destination_size = uint2(
        destination.get_width(), destination.get_height());
    if (any(gid >= destination_size)) return;
    const uint2 oriented_size = uint2(
        source.get_width(), source.get_height());
    const float2 value = resample_flow(
        source, oriented_size, destination, gid, landscape_value);
    destination.write(
        half4(half2(value), half(0.0h), half(1.0h)), gid);
}

kernel void flow_resample_portrait_ccw(
    texture2d<half, access::read> source [[texture(0)]],
    texture2d<half, access::write> destination [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint2 destination_size = uint2(
        destination.get_width(), destination.get_height());
    if (any(gid >= destination_size)) return;
    const uint2 oriented_size = uint2(
        source.get_height(), source.get_width());
    const float2 value = resample_flow(
        source, oriented_size, destination, gid, portrait_ccw_value);
    destination.write(
        half4(half2(value), half(0.0h), half(1.0h)), gid);
}
"""


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


class VisionFlowToVtConverter:
    """Convert a full-resolution Vision field into VT SR flow coordinates.

    Vision revision 1 reports vectors in the source buffer's pixel units.
    VT SR instead interprets explicit flow in the raw geometry advertised by
    ``VTOpticalFlowConfiguration.destinationPixelBufferAttributes``. The
    converter area-filters reductions (bilinear for the rare enlargement),
    rescales both vector components into destination-grid units, and rotates
    portrait fields counterclockwise into VT's landscape coordinates.

    Conversion is synchronous and owns one command queue. Callers may reuse an
    instance, but the lock deliberately prevents concurrent writes through the
    same cache and queue.
    """

    def __init__(
        self,
        source_width: int,
        source_height: int,
        destination_width: int,
        destination_height: int,
        *,
        rotate_counterclockwise: bool,
    ) -> None:
        dimensions = (
            source_width,
            source_height,
            destination_width,
            destination_height,
        )
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("flow conversion dimensions must be positive")

        import Metal

        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("Vision flow conversion found no Metal device")
        library, error = device.newLibraryWithSource_options_error_(
            _FLOW_CONVERSION_METAL,
            None,
            None,
        )
        if library is None:
            raise RuntimeError(
                f"Vision flow conversion Metal library failed: {error}"
            )
        function_name = (
            "flow_resample_portrait_ccw"
            if rotate_counterclockwise
            else "flow_resample_landscape"
        )
        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise RuntimeError(
                f"Vision flow conversion kernel {function_name!r} is unavailable"
            )
        pipeline, error = device.newComputePipelineStateWithFunction_error_(
            function,
            None,
        )
        if pipeline is None:
            raise RuntimeError(
                f"Vision flow conversion pipeline failed: {error}"
            )
        queue = device.newCommandQueue()
        if queue is None:
            raise RuntimeError(
                "Vision flow conversion command queue creation failed"
            )
        status, texture_cache = Quartz.CVMetalTextureCacheCreate(
            None,
            None,
            device,
            None,
            None,
        )
        if status != 0 or texture_cache is None:
            raise RuntimeError(
                "Vision flow conversion texture-cache creation failed: "
                f"status={status}"
            )

        self.source_width = int(source_width)
        self.source_height = int(source_height)
        self.destination_width = int(destination_width)
        self.destination_height = int(destination_height)
        self.rotate_counterclockwise = bool(rotate_counterclockwise)
        self._metal = Metal
        self._device = device
        self._library = library
        self._pipeline = pipeline
        self._queue = queue
        self._texture_cache = texture_cache
        self._lock = threading.Lock()

    def _texture(
        self,
        buffer: Any,
        width: int,
        height: int,
    ) -> tuple[Any, Any]:
        Metal = self._metal
        status, reference = Quartz.CVMetalTextureCacheCreateTextureFromImage(
            None,
            self._texture_cache,
            buffer,
            None,
            Metal.MTLPixelFormatRG16Float,
            width,
            height,
            0,
            None,
        )
        if status != 0 or reference is None:
            raise RuntimeError(
                "Vision flow conversion could not wrap an IOSurface: "
                f"status={status}"
            )
        texture = Quartz.CVMetalTextureGetTexture(reference)
        if texture is None:
            raise RuntimeError(
                "Vision flow conversion produced no Metal texture"
            )
        return reference, texture

    def convert(self, source: Any, destination: Any) -> None:
        """Write ``source`` into a caller-owned VT-geometry destination."""
        _validate_flow_buffer(
            source,
            width=self.source_width,
            height=self.source_height,
            pixel_format=FLOW_16H,
        )
        _validate_flow_buffer(
            destination,
            width=self.destination_width,
            height=self.destination_height,
            pixel_format=FLOW_16H,
        )

        with self._lock:
            if self._queue is None or self._texture_cache is None:
                raise RuntimeError("Vision flow converter is closed")
            texture_references = []
            reference = None
            source_texture = destination_texture = None
            command = encoder = None
            try:
                reference, source_texture = self._texture(
                    source,
                    self.source_width,
                    self.source_height,
                )
                texture_references.append(reference)
                reference, destination_texture = self._texture(
                    destination,
                    self.destination_width,
                    self.destination_height,
                )
                texture_references.append(reference)
                command = self._queue.commandBuffer()
                if command is None:
                    raise RuntimeError(
                        "Vision flow conversion command-buffer creation failed"
                    )
                encoder = command.computeCommandEncoder()
                if encoder is None:
                    raise RuntimeError(
                        "Vision flow conversion compute encoder creation failed"
                    )
                encoder.setComputePipelineState_(self._pipeline)
                encoder.setTexture_atIndex_(source_texture, 0)
                encoder.setTexture_atIndex_(destination_texture, 1)

                thread_width = int(self._pipeline.threadExecutionWidth())
                max_threads = int(
                    self._pipeline.maxTotalThreadsPerThreadgroup()
                )
                thread_height = max(1, min(16, max_threads // thread_width))
                Metal = self._metal
                encoder.dispatchThreads_threadsPerThreadgroup_(
                    Metal.MTLSizeMake(
                        self.destination_width,
                        self.destination_height,
                        1,
                    ),
                    Metal.MTLSizeMake(thread_width, thread_height, 1),
                )
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                error = command.error()
                if error is not None:
                    raise RuntimeError(
                        f"Vision flow conversion Metal encode failed: {error}"
                    )
            finally:
                encoder = command = None
                source_texture = destination_texture = None
                reference = None
                texture_references.clear()
                if self._texture_cache is not None:
                    Quartz.CVMetalTextureCacheFlush(self._texture_cache, 0)

    def close(self) -> None:
        """Release cached Metal wrappers after the flow executor has drained."""
        with self._lock:
            if self._texture_cache is not None:
                Quartz.CVMetalTextureCacheFlush(self._texture_cache, 0)
            self._texture_cache = None
            self._queue = None
            self._pipeline = None
            self._library = None
            self._device = None


def generate_vision_flow(
    from_buffer: Any,
    to_buffer: Any,
    *,
    pixel_format: int = FLOW_16H,
    accuracy: str = "medium",
) -> Any:
    """Return source-geometry flow from ``from_buffer`` to ``to_buffer``.

    The returned CVPixelBuffer is retained by its PyObjC wrapper and can be
    handed directly to ``VTFrameProcessorOpticalFlow``.
    """
    try:
        computation_accuracy = _ACCURACY[accuracy]
    except KeyError:
        raise ValueError(
            "Vision flow accuracy must be one of "
            f"{sorted(_ACCURACY)}, got {accuracy!r}"
        ) from None

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
    request.setComputationAccuracy_(computation_accuracy)
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
    *,
    accuracy: str = "medium",
) -> tuple[Any, Any]:
    """Return previous->current and current->previous Vision flow buffers."""
    return (
        generate_vision_flow(
            previous_buffer,
            current_buffer,
            accuracy=accuracy,
        ),
        generate_vision_flow(
            current_buffer,
            previous_buffer,
            accuracy=accuracy,
        ),
    )


__all__ = [
    "FLOW_16H",
    "FLOW_32F",
    "VisionFlowToVtConverter",
    "generate_bidirectional_vision_flow",
    "generate_vision_flow",
]
