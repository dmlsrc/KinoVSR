"""Pool compatibility and the MLX-to-FRC upload bridge."""

from __future__ import annotations

from typing import Any

from kinovsr.media import pixel_buffers as pb
from kinovsr.native.frameworks import Quartz

_PADDING_KEYS = (
    Quartz.kCVPixelBufferExtendedPixelsLeftKey,
    Quartz.kCVPixelBufferExtendedPixelsRightKey,
    Quartz.kCVPixelBufferExtendedPixelsTopKey,
    Quartz.kCVPixelBufferExtendedPixelsBottomKey,
)
UPLOAD_POOL_ALLOCATION_LIMIT = 2


def _pool_descriptor(
    pool: Any,
) -> tuple[int, int, int, tuple[int, int, int, int]] | None:
    """Probe the actual surfaces an offered pool creates."""
    probe = pb.pool_create_buffer(pool)
    if probe is None:
        return None
    try:
        pixel_format = int(Quartz.CVPixelBufferGetPixelFormatType(probe))
        width = int(Quartz.CVPixelBufferGetWidth(probe))
        height = int(Quartz.CVPixelBufferGetHeight(probe))
        padding = tuple(
            int(value)
            for value in Quartz.CVPixelBufferGetExtendedPixels(probe, None, None, None, None)
        )
        return pixel_format, width, height, padding
    finally:
        del probe


def apply_output_pool(
    session: Any,
    binding: tuple[Any, int, int, int] | None,
    width: int,
    height: int,
) -> None:
    """Use an offered writer pool only for a verified destination match."""
    if binding is None:
        return
    pool, pixel_format, pool_width, pool_height = binding
    expected_format = pb.resolve_pixel_format(session.dst_attrs)
    if (pool_width, pool_height) != (width, height) or pixel_format != expected_format:
        return
    descriptor = _pool_descriptor(pool)
    if descriptor is None:
        return
    actual_format, actual_width, actual_height, actual_padding = descriptor
    required_padding = tuple(int(session.dst_attrs.get(key, 0)) for key in _PADDING_KEYS)
    if (actual_format, actual_width, actual_height) != (expected_format, width, height) or any(
        actual < required for actual, required in zip(actual_padding, required_padding, strict=True)
    ):
        return
    session.use_dst_pool(pool)


class MlxUploadPool:
    """Own the reusable RGBAHalf source pool for an MLX-fed FRC stage."""

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.attrs = {
            "PixelFormatType": pb.PIX_RGBAHALF,
            "Width": self.width,
            "Height": self.height,
            "IOSurfaceProperties": {},
            "MetalCompatibility": True,
        }
        self.pool = pb.make_bounded_pool_from_attrs(self.attrs, UPLOAD_POOL_ALLOCATION_LIMIT)
        if self.pool is None:
            raise RuntimeError(
                "FRC upload CVPixelBufferPool creation failed; "
                "bounded source allocation is required"
            )

    def upload(self, rgb: Any) -> Any:
        import mlx.core as mx

        rgba = mx.concatenate(
            [rgb[..., :3].astype(mx.float16), mx.ones((*rgb.shape[:2], 1), mx.float16)], axis=-1
        )
        if self.pool is None:
            raise RuntimeError("FRC upload pool is unavailable")
        buffer = pb.pool_create_buffer_bounded(self.pool, UPLOAD_POOL_ALLOCATION_LIMIT)
        pb.upload_frame_to_buffer(rgba, buffer)
        return buffer

    def close(self) -> None:
        pool, self.pool = self.pool, None
        if pool is not None:
            pb.flush_pool(pool)
