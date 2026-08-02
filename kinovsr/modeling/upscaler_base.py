"""Shared driver scaffolding for the learned upscaler wrappers.

`to_rgb_batch` normalizes a fed frame to a batched fp32 RGB array. `WindowedUpscaler`
is the sliding-window feed()/flush() driver shared by the clip-recurrent nets
(BasicVSR++, RealBasicVSR); the per-frame RealESRGAN wrapper uses only
`to_rgb_batch`, since each frame upscales independently.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import mlx.core as mx

from .gop_windows import GopWindows


def to_rgb_batch(rgb: Any) -> Any:
    """Frame -> (1,H,W,3) fp32: add a batch axis if missing, drop alpha, cast f32,
    and CLIP to [0,1]. Decoded RGBAHalf carries legal YUV->RGB overshoot (measured
    -0.14..1.25 at saturated color edges) and every learned upscaler is trained on
    clipped RGB; feeding the overshoot drives the nets outside their input domain
    -- measured 56x the confetti-speck area on one GAN checkpoint. Same rule as
    the preprocessor entry points (see nafnet.restorer.model_rgb)."""
    a = rgb if rgb.ndim == 4 else rgb[None]
    return mx.clip(a[..., :3].astype(mx.float32), 0.0, 1.0)


class WindowedUpscaler:
    """Sliding-window feed()/flush() driver for a clip-recurrent upscaler net.

    A bidirectional / second-order recurrent net can't upscale a frame in
    isolation, so we buffer a window of `window` LR frames, emit its stable
    interior, and trim `trim` warm-up frames at each window join (the
    propagation's transient edge). Memory stays bounded to ~`window` buffered LR
    frames regardless of clip length.

    Subclasses load their weights in __init__ (then call super().__init__ with the
    resolved window/trim) and implement `_upscale_window(frames)`, yielding one
    upscaled (1,sH,sW,3) per input frame. feed(rgb, token) buffers a frame and
    returns the (upscaled_rgb, token) pairs that are now final; flush() drains the
    tail. Frame order and token pairing are preserved.
    """

    SCALE = 4

    def __init__(
        self,
        window: int,
        trim: int,
        *,
        vt_flow_geometries: int = 0,
    ):
        self._T = int(trim)
        self._W = int(window)
        self._gop: GopWindows | None = None
        self._vt_flow_services: Any = None
        if vt_flow_geometries:
            from kinovsr.modeling.vt_flow import VtFlowServices

            self._vt_flow_services = VtFlowServices(vt_flow_geometries)
        self.reset()

    def close(self) -> None:
        """Release buffered frames and this driver's native flow services."""
        self.reset()
        services, self._vt_flow_services = self._vt_flow_services, None
        if services is not None:
            services.close()

    def reset(self) -> None:
        self._frames: list = []  # sliding LR buffer (1,H,W,3)
        self._tokens: list = []
        self._base = 0  # global index of _frames[0]
        self._emitted = 0  # global index of the next frame to emit
        if self._gop is not None:
            self._gop.reset()

    def set_gop_policy(self, policy: Any) -> None:
        self._gop = (
            None
            if policy is None
            else GopWindows(
                policy.min_window, policy.max_window, self._run_gop_window)
        )

    def feed(self, rgb: Any, token: Any = None) -> Iterable:
        frame = to_rgb_batch(rgb)
        if self._gop is not None:
            return self._gop.feed(frame, token)
        self._frames.append(frame)
        self._tokens.append(token)
        out: list = []
        while (self._base + len(self._frames)) >= max(0, self._emitted - self._T) + self._W:
            ws = max(0, self._emitted - self._T)
            out.extend(self._run(ws, ws + self._W, last=False))
        # Retain enough for the next interior window (back to emitted-T) AND a
        # full-width flush window (back to total-W); drop only what neither needs.
        total = self._base + len(self._frames)
        keep = max(0, min(self._emitted - self._T, total - self._W)) - self._base
        if keep > 0:
            self._frames = self._frames[keep:]
            self._tokens = self._tokens[keep:]
            self._base += keep
        return out

    def flush(self) -> Iterable:
        if self._gop is not None:
            return self._flush_gop()
        total = self._base + len(self._frames)
        if self._emitted >= total:
            self.reset()
            return []
        ws = max(0, min(self._emitted - self._T, total - self._W))
        out = self._run(ws, total, last=True)
        self.reset()
        return out

    def _flush_gop(self) -> Iterable:
        yield from self._gop.flush()
        self.reset()

    def _run_gop_window(
        self, frames: list, tokens: list, emit_start: int, emit_end: int,
    ) -> Iterable:
        # Conditioning consumers read the source identities parallel to this
        # one window without owning any boundary or buffer bookkeeping.
        self._window_tokens = tokens
        produced = 0
        for produced, frame in enumerate(self._upscale_window(frames), start=1):
            index = produced - 1
            if emit_start <= index < emit_end:
                yield frame[0], tokens[index]
        if produced != len(frames):
            raise RuntimeError(
                f"window returned {produced} outputs for {len(frames)} frames")

    def _run(self, ws: int, we: int, last: bool) -> list:
        # Both window nets (BasicVSR++ _upsample, RealBasicVSR _basicvsr) mx.eval each
        # output frame as it is produced, so the frames arrive materialized -- no extra
        # sync barrier here.
        sr = list(self._upscale_window(
            self._frames[ws - self._base : we - self._base]))
        end = we if last else we - self._T
        out = [(sr[g - ws][0], self._tokens[g - self._base]) for g in range(self._emitted, end)]
        self._emitted = end
        return out

    def _upscale_window(self, frames: list) -> Iterable:
        raise NotImplementedError
