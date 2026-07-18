"""VideoToolbox motion-estimation flow: block matching on the media engine.

``VTMotionEstimationSession`` (macOS 26+) exposes the video encoder's
motion estimator. This engine pins the one configuration measured to be
warp-grade: 4x4 search blocks with multipass ("true motion") search,
which adds quarter-pel refinement and coherent fields (single-pass is
integer-only encoder-style vectors with ragged fields). The work runs on
the media engine, off both the MLX GPU queue and the CPU hot path:
engine latency does not change under full MLX GPU saturation, where
Vision r1 pays about +45%.

Convention (empirically verified, and re-checked at session start by the
mc self-test): ``compute(a, b)`` returns dense (H,W,2) fp32 buffer-px
flow with ``a[p] ~= b[p + flow[p]]`` -- the same orientation as
``spynet_flow(a, b)`` and ``VisionFlowEngine.compute``, so callers
negate it identically. The session is fed (reference=b, current=a) and
already emits this orientation; vectors pass through unnegated.

Densification is NEAREST (each pixel inherits its block's vector,
piecewise constant): block fields are discontinuous at block granularity
and linear seam interpolation invents vectors that belong to neither
neighbor (measured 4-6 dB warp-PSNR worse on REDS). Inputs are collapsed
to Rec.709 luma and fed as OneComponent8, one of the session's native
source formats (feeding RGB works but adds a hidden conversion).

Scope note from the mc ship gate: this engine wins warp-PSNR against
every other backend (+1.6 to +3.3 dB over vision on REDS truth-scored
warps) yet loses 0.2-0.5 dB to vision on end-to-end mc denoise output --
photometric block matching is the matcher's own objective, and for
temporal noise averaging it both quantizes sub-pixel alignment per block
and partially matches the very noise the blend should cancel. Treat it
as the cheap, contention-immune correspondence engine, not a quality
upgrade for averaging.

Two known behaviors callers must own: flat regions read exactly zero
(good for gating, no rate-distortion garbage), and across a scene cut
the field explodes to huge incoherent vectors -- reset temporal
consumers at cuts rather than warping across them.
"""

from __future__ import annotations

import threading
from typing import Any

import mlx.core as mx

from kinovsr.native.frameworks import Quartz, autorelease_pool, vt

_BLOCK = 4
_L008 = 0x4C303038  # kCVPixelFormatType_OneComponent8
_FLOW_16H = 0x32433068  # kCVPixelFormatType_TwoComponent16Half
_REC709_LUMA = (0.2126, 0.7152, 0.0722)


class VtmeFlowEngine:
    """One geometry's motion-estimation engine; owns two luma buffers.

    Not thread-safe: the two source buffers are reused per ``compute``
    call, matching the one-borrower contract of the flow-services
    manager.
    """

    def __init__(self, width: int, height: int) -> None:
        self.w, self.h = int(width), int(height)
        if self.w < 1 or self.h < 1:
            raise ValueError("motion-estimation geometry must be positive")
        if not hasattr(vt, "VTMotionEstimationSessionCreate"):
            raise SystemExit(
                "VTMotionEstimationSession is unavailable; "
                "--mc-flow vtme requires macOS 26 or newer.")
        options = {
            vt.kVTMotionEstimationSessionCreationOption_MotionVectorSize:
                _BLOCK,
            vt.kVTMotionEstimationSessionCreationOption_UseMultiPassSearch:
                True,
        }
        err, session = vt.VTMotionEstimationSessionCreate(
            None, options, self.w, self.h, None)
        if err != 0 or session is None:
            raise RuntimeError(
                f"VTMotionEstimationSessionCreate failed for "
                f"{self.w}x{self.h}: {err}")
        self._session = session
        self._luma = mx.array(_REC709_LUMA, dtype=mx.float32)
        try:
            self._from_buf = self._make_l008()
            self._to_buf = self._make_l008()
        except BaseException:
            vt.VTMotionEstimationSessionInvalidate(session)
            self._session = None
            raise

    def _make_l008(self) -> Any:
        attrs = {
            Quartz.kCVPixelBufferPixelFormatTypeKey: _L008,
            Quartz.kCVPixelBufferWidthKey: self.w,
            Quartz.kCVPixelBufferHeightKey: self.h,
            Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
        }
        err, buf = Quartz.CVPixelBufferCreate(
            None, self.w, self.h, _L008, attrs, None)
        if err != 0 or buf is None:
            raise RuntimeError(
                f"CVPixelBufferCreate(L008 {self.w}x{self.h}) failed: {err}")
        return buf

    def _upload(self, rgb_f32: Any, buf: Any) -> None:
        luma = mx.clip(
            (rgb_f32[..., :3].astype(mx.float32) @ self._luma) * 255.0 + 0.5,
            0, 255).astype(mx.uint8)
        Quartz.CVPixelBufferLockBaseAddress(buf, 0)
        try:
            bpr = int(Quartz.CVPixelBufferGetBytesPerRow(buf))
            if bpr != self.w:
                luma = mx.pad(luma, ((0, 0), (0, bpr - self.w)))
            mx.eval(luma)
            base = Quartz.CVPixelBufferGetBaseAddress(buf)
            dst = memoryview(base.as_buffer(self.h * bpr)).cast("B")
            dst[:] = memoryview(mx.contiguous(luma)).cast("B")
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(buf, 0)

    def _estimate(self) -> Any:
        result: dict[str, Any] = {}
        done = threading.Event()

        def handler(status: int, info: int, extra: Any, vectors: Any) -> None:
            result["status"] = int(status)
            result["buf"] = vectors
            done.set()

        status = vt.VTMotionEstimationSessionEstimateMotionVectors(
            self._session, self._to_buf, self._from_buf, 0, None, handler)
        if status != 0:
            raise RuntimeError(f"VTMotionEstimation submit failed: {status}")
        vt.VTMotionEstimationSessionCompleteFrames(self._session)
        if not done.wait(5.0):
            raise RuntimeError("VTMotionEstimation timed out")
        if result["status"] != 0 or result["buf"] is None:
            raise RuntimeError(
                f"VTMotionEstimation failed: {result['status']}")
        return result["buf"]

    def _read_dense(self, mv_buf: Any) -> Any:
        bw = int(Quartz.CVPixelBufferGetWidth(mv_buf))
        bh = int(Quartz.CVPixelBufferGetHeight(mv_buf))
        fmt = int(Quartz.CVPixelBufferGetPixelFormatType(mv_buf))
        if fmt != _FLOW_16H:
            raise RuntimeError(
                f"VTMotionEstimation returned pixel format {fmt:#x}, "
                f"expected TwoComponent16Half")
        if bw * _BLOCK < self.w or bh * _BLOCK < self.h:
            raise RuntimeError(
                f"VTMotionEstimation returned {bw}x{bh} blocks for a "
                f"{self.w}x{self.h} request")
        Quartz.CVPixelBufferLockBaseAddress(mv_buf, 1)
        try:
            bpr = Quartz.CVPixelBufferGetBytesPerRow(mv_buf)
            base = Quartz.CVPixelBufferGetBaseAddress(mv_buf)
            raw = mx.array(memoryview(base.as_buffer(bh * bpr)))
            blocks = (raw.view(mx.float16)
                      .reshape(bh, bpr // 2)[:, : bw * 2]
                      .reshape(bh, bw, 2).astype(mx.float32))
            dense = mx.repeat(mx.repeat(blocks, _BLOCK, axis=0),
                              _BLOCK, axis=1)[: self.h, : self.w]
            mx.eval(dense)
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(mv_buf, 1)
        return dense

    def compute(self, from_rgb: Any, to_rgb: Any) -> Any:
        """(H,W,2) fp32 buffer-px flow: ``from_rgb[p] ~= to_rgb[p+flow[p]]``."""
        if self._from_buf is None:
            raise RuntimeError("motion-estimation engine is closed")
        with autorelease_pool():
            self._upload(from_rgb, self._from_buf)
            self._upload(to_rgb, self._to_buf)
            return self._read_dense(self._estimate())

    def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            vt.VTMotionEstimationSessionInvalidate(session)
        self._from_buf = None
        self._to_buf = None


__all__ = ["VtmeFlowEngine"]
