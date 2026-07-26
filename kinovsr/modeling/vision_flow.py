"""Vision optical flow, revision 1: the measured-best native flow backend.

``VNGenerateOpticalFlowRequest`` REVISION 1 at Medium accuracy, pinned
explicitly: revision 2 (the system default, ML-based) is slower, pays a
long cold initialization per accuracy tier, and undertracks both
synthetic and real motion, so it must never be allowed to apply.

Convention (empirically validated, and re-checked at session start by the
mc self-test): with the request handler holding the FROM frame and the
request targeting the TO frame, content at FROM pixel ``p`` lands at TO
pixel ``p + flow[p]``; units are buffer pixels. ``compute(a, b)``
therefore returns flow with ``a[p] ~= b[p + flow[p]]`` -- the same
orientation as ``spynet_flow(a, b)``, so callers negate it identically.

Placement: the request runs off the MLX GPU queue (measured ~8 ms per
pair including MLX readback when idle, ~2x that under a saturated MLX
pipeline), so it does not serialize against learned stages the way an
MLX flow net does.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from kinovsr.media import pixel_buffers as _pb
from kinovsr.native.frameworks import Quartz, autorelease_pool
from kinovsr.native.vision_flow import FLOW_32F, generate_vision_flow


class VisionFlowEngine:
    """One geometry's Vision flow engine; owns two RGBAHalf source buffers.

    Not thread-safe: the two source buffers are reused per ``compute``
    call, matching the one-borrower contract of the flow-services
    manager.
    """

    def __init__(self, width: int, height: int) -> None:
        self.w, self.h = int(width), int(height)
        if self.w < 1 or self.h < 1:
            raise ValueError("Vision flow geometry must be positive")
        attrs = {
            "PixelFormatType": _pb.PIX_RGBAHALF,
            "Width": self.w, "Height": self.h,
            "IOSurfaceProperties": {},
        }
        self._from_buf = _pb.make_pixel_buffer_from_attrs(self.w, self.h, attrs)
        self._to_buf = _pb.make_pixel_buffer_from_attrs(self.w, self.h, attrs)

    def _upload(self, rgb_f32: Any, buf: Any) -> None:
        rgba = mx.concatenate(
            [rgb_f32.astype(mx.float16),
             mx.ones((self.h, self.w, 1), mx.float16)],
            axis=-1,
        )
        _pb.write_fp16_rgba(rgba, buf)

    def compute(self, from_rgb: Any, to_rgb: Any) -> Any:
        """(H,W,2) fp32 buffer-px flow: ``from_rgb[p] ~= to_rgb[p+flow[p]]``."""
        if self._from_buf is None:
            raise RuntimeError("Vision flow engine is closed")
        with autorelease_pool():
            self._upload(from_rgb, self._from_buf)
            self._upload(to_rgb, self._to_buf)
            fb = generate_vision_flow(
                self._from_buf,
                self._to_buf,
                pixel_format=FLOW_32F,
                accuracy="medium",
            )
            Quartz.CVPixelBufferLockBaseAddress(fb, 1)
            try:
                bpr = Quartz.CVPixelBufferGetBytesPerRow(fb)
                base = Quartz.CVPixelBufferGetBaseAddress(fb)
                raw = mx.array(memoryview(base.as_buffer(self.h * bpr)))
                flow = (raw.view(mx.float32)
                        .reshape(self.h, bpr // 4)[:, : self.w * 2]
                        .reshape(self.h, self.w, 2))
                mx.eval(flow)
            finally:
                Quartz.CVPixelBufferUnlockBaseAddress(fb, 1)
        return flow

    def close(self) -> None:
        self._from_buf = None
        self._to_buf = None


__all__ = ["VisionFlowEngine"]
