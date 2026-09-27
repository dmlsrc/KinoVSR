"""Exact Metal copies between CVPixelBuffers of different sizes.

A native session sometimes needs its frames on a canvas of another size.
``CanvasCopier.copy`` writes destination pixel (x, y) from source pixel
(min(x, width - 1), min(y, height - 1)), plane by plane in the buffer's own
pixel format: a larger destination receives the source at its top-left with
the last column and row repeated (edge padding), and a smaller one receives the
source's top-left (a crop). Pixel values pass through unchanged.
"""

import threading

import Metal

from kinovsr.media import pixel_buffers as _pb

from .frameworks import Quartz

# kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ("420f").
PIX_NV12_FULL_RANGE = int.from_bytes(b"420f", "big")

_COPY_METAL = r"""
#include <metal_stdlib>
using namespace metal;

kernel void copy_top_left_clamped(
    texture2d<float, access::read> source [[texture(0)]],
    texture2d<float, access::write> destination [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (any(gid >= uint2(destination.get_width(), destination.get_height()))) return;
    const uint2 last = uint2(source.get_width(), source.get_height()) - 1;
    destination.write(source.read(min(gid, last)), gid);
}
"""


class CanvasCopier:
    """Copy a buffer's top-left into another buffer of the same pixel format.

    Supports BGRA, RGBAHalf, and 8-bit NV12 (video and full range). Copies are
    synchronous and propagate the source's attachments, so color tags survive.
    One command queue is owned per instance; a lock serializes callers.
    """

    def __init__(self) -> None:

        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("canvas copy found no Metal device")
        library, error = device.newLibraryWithSource_options_error_(_COPY_METAL, None, None)
        if library is None:
            raise RuntimeError(f"canvas copy Metal library failed: {error}")
        function = library.newFunctionWithName_("copy_top_left_clamped")
        if function is None:
            raise RuntimeError("canvas copy kernel is unavailable")
        pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None:
            raise RuntimeError(f"canvas copy pipeline failed: {error}")
        queue = device.newCommandQueue()
        if queue is None:
            raise RuntimeError("canvas copy command queue creation failed")
        status, texture_cache = Quartz.CVMetalTextureCacheCreate(None, None, device, None, None)
        if status != 0 or texture_cache is None:
            raise RuntimeError(f"canvas copy texture-cache creation failed: status={status}")

        # One Metal texture format per plane; unorm and half values round-trip
        # exactly through the kernel's float reads and writes.
        nv12 = (Metal.MTLPixelFormatR8Unorm, Metal.MTLPixelFormatRG8Unorm)
        self._plane_formats = {
            _pb.PIX_BGRA: (Metal.MTLPixelFormatBGRA8Unorm,),
            _pb.PIX_RGBAHALF: (Metal.MTLPixelFormatRGBA16Float,),
            _pb.PIX_NV12: nv12,
            PIX_NV12_FULL_RANGE: nv12,
        }
        self._metal = Metal
        self._pipeline = pipeline
        self._queue = queue
        self._texture_cache = texture_cache
        self._lock = threading.Lock()

    def supports(self, pixel_format: int) -> bool:
        return int(pixel_format) in self._plane_formats

    def _texture(
        self, buffer: object, plane: int, texture_format: int
    ) -> tuple[object, object, int, int]:
        if Quartz.CVPixelBufferIsPlanar(buffer):
            width = int(Quartz.CVPixelBufferGetWidthOfPlane(buffer, plane))
            height = int(Quartz.CVPixelBufferGetHeightOfPlane(buffer, plane))
        else:
            width = int(Quartz.CVPixelBufferGetWidth(buffer))
            height = int(Quartz.CVPixelBufferGetHeight(buffer))
        status, reference = Quartz.CVMetalTextureCacheCreateTextureFromImage(
            None, self._texture_cache, buffer, None, texture_format, width, height, plane, None
        )
        if status != 0 or reference is None:
            raise RuntimeError(f"canvas copy could not wrap an IOSurface plane: status={status}")
        texture = Quartz.CVMetalTextureGetTexture(reference)
        if texture is None:
            raise RuntimeError("canvas copy produced no Metal texture")
        return reference, texture, width, height

    def copy(self, source: object, destination: object) -> object:
        """Write ``source``'s top-left, edge-extended as needed, into ``destination``."""
        pixel_format = int(Quartz.CVPixelBufferGetPixelFormatType(source))
        destination_format = int(Quartz.CVPixelBufferGetPixelFormatType(destination))
        if destination_format != pixel_format:
            raise ValueError(
                "canvas copy requires one pixel format; "
                f"got {pixel_format:#x} -> {destination_format:#x}"
            )
        planes = self._plane_formats.get(pixel_format)
        if planes is None:
            raise RuntimeError(f"canvas copy does not support pixel format {pixel_format:#x}")

        with self._lock:
            if self._queue is None or self._texture_cache is None:
                raise RuntimeError("canvas copier is closed")
            references = []
            command = encoder = None
            try:
                command = self._queue.commandBuffer()
                if command is None:
                    raise RuntimeError("canvas copy command-buffer creation failed")
                encoder = command.computeCommandEncoder()
                if encoder is None:
                    raise RuntimeError("canvas copy compute encoder creation failed")
                encoder.setComputePipelineState_(self._pipeline)
                thread_width = int(self._pipeline.threadExecutionWidth())
                thread_height = max(
                    1, min(16, int(self._pipeline.maxTotalThreadsPerThreadgroup()) // thread_width)
                )
                for plane, texture_format in enumerate(planes):
                    source_reference, source_texture, _, _ = self._texture(
                        source, plane, texture_format
                    )
                    references.append(source_reference)
                    destination_reference, destination_texture, width, height = self._texture(
                        destination, plane, texture_format
                    )
                    references.append(destination_reference)
                    encoder.setTexture_atIndex_(source_texture, 0)
                    encoder.setTexture_atIndex_(destination_texture, 1)
                    encoder.dispatchThreads_threadsPerThreadgroup_(
                        self._metal.MTLSizeMake(width, height, 1),
                        self._metal.MTLSizeMake(thread_width, thread_height, 1),
                    )
                encoder.endEncoding()
                command.commit()
                command.waitUntilCompleted()
                error = command.error()
                if error is not None:
                    raise RuntimeError(f"canvas copy Metal encode failed: {error}")
            finally:
                encoder = command = None
                references.clear()
                if self._texture_cache is not None:
                    Quartz.CVMetalTextureCacheFlush(self._texture_cache, 0)
        Quartz.CVBufferPropagateAttachments(source, destination)
        return destination

    def close(self) -> None:
        """Release the cached Metal wrappers and the command queue."""
        with self._lock:
            if self._texture_cache is not None:
                Quartz.CVMetalTextureCacheFlush(self._texture_cache, 0)
            self._texture_cache = None
            self._queue = None
            self._pipeline = None
