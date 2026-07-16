"""Shared driver scaffolding for the learned upscaler wrappers.

`to_rgb_batch` normalizes a fed frame to a batched fp32 RGB array. `WindowedUpscaler`
is the sliding-window feed()/flush() driver shared by the clip-recurrent nets
(BasicVSR++, RealBasicVSR); the per-frame RealESRGAN wrapper uses only
`to_rgb_batch`, since each frame upscales independently.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def plan_gop_windows(
    keyframes: list[int], n_frames: int, min_window: int, max_window: int
) -> list[tuple[int, int, int, int]]:
    """Plan GOP-aligned recurrent windows from source keyframe positions.

    Returns a list of (proc_start, proc_end, emit_start, emit_end) specs. `proc` is
    the frame range fed to the recurrent net; `emit` is the range it outputs. The
    emit ranges tile [0, n_frames) exactly, in order (each frame emitted once);
    proc ranges may overlap by the anchor/trim region.

    Each window starts and ENDS on a keyframe (the closing keyframe is included in
    proc so the backward recurrence pass also cold-starts on a clean frame, and it
    is emitted by the next window as its start) -- so both directions anchor on
    keyframes and no trim is needed at GOP boundaries. A window spans as many whole
    GOPs as it takes to reach `min_window` frames. A span longer than `max_window`
    (pathological long-GOP) is split into <=max_window sub-windows with a small trim
    at the internal, non-keyframe splits only. A clip with no usable keyframe run
    (e.g. a single-keyframe open GOP) falls back to fixed max_window+trim tiling.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, int):
        raise ValueError("n_frames must be an integer")
    if n_frames < 0:
        raise ValueError("n_frames must be >= 0")
    for name, value in (("min_window", min_window), ("max_window", max_window)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
    if min_window > max_window:
        raise ValueError("min_window must be <= max_window")
    if any(isinstance(k, bool) or not isinstance(k, int) for k in keyframes):
        raise ValueError("keyframes must contain integers")
    if n_frames == 0:
        return []

    kf = sorted({k for k in keyframes if 0 <= k < n_frames})
    if not kf or kf[0] != 0:
        kf = [0, *kf]
    trim = 2
    out: list[tuple[int, int, int, int]] = []
    pos = 0
    # Keyframes and output position are monotonic. Keep one forward cursor
    # instead of rescanning kf from index zero for every window (quadratic on
    # dense all-I sources).
    kf_cursor = 1
    while pos < n_frames:
        target = pos + min_window
        while kf_cursor < len(kf) and kf[kf_cursor] < target:
            kf_cursor += 1
        close = kf[kf_cursor] if kf_cursor < len(kf) else n_frames
        if close - pos <= max_window:
            # one keyframe-anchored window; include the closing keyframe in proc
            proc_end = min(close + 1, n_frames) if close < n_frames else n_frames
            out.append((pos, proc_end, pos, close))
        else:
            sub = pos
            while sub < close:
                sub_end = min(sub + max_window, close)
                p_start = sub if sub == pos else max(pos, sub - trim)
                if sub_end == close:
                    p_end = min(close + 1, n_frames) if close < n_frames else n_frames
                else:
                    p_end = min(close, sub_end + trim)
                out.append((p_start, p_end, sub, sub_end))
                sub = sub_end
        previous = pos
        pos = close
        if pos <= previous:
            raise RuntimeError("GOP planner did not advance; invalid window bounds")
    return out


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
    resolved window/trim) and implement `_upscale_window(frames) -> list`, one
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
        self._schedule: list | None = None
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
        self._sched_i = 0  # next schedule window to run (schedule mode)

    def set_schedule(self, schedule: list | None) -> None:
        """Switch to GOP-aligned windowing: a list of (proc_start, proc_end,
        emit_start, emit_end) specs (see plan_gop_windows) whose emit ranges tile
        the clip. Each window's proc range is fed to _upscale_window and its emit
        range output. None (the default) keeps the fixed window/trim sliding. Call
        once before feeding; the schedule persists across reset()."""
        self._schedule = list(schedule) if schedule else None
        self._sched_i = 0

    def feed(self, rgb: Any, token: Any = None) -> list:
        self._frames.append(to_rgb_batch(rgb))
        self._tokens.append(token)
        if self._schedule is not None:
            return self._feed_scheduled(final=False)
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

    def flush(self) -> list:
        if self._schedule is not None:
            out = self._feed_scheduled(final=True)
            self.reset()
            return out
        total = self._base + len(self._frames)
        if self._emitted >= total:
            self.reset()
            return []
        ws = max(0, min(self._emitted - self._T, total - self._W))
        out = self._run(ws, total, last=True)
        self.reset()
        return out

    def _feed_scheduled(self, final: bool) -> list:
        """Run every scheduled window whose proc range is fully buffered (or, when
        `final`, clamp the tail window to whatever frames arrived), emit its emit
        range, and free frames no unrun window still needs."""
        out: list = []
        total = self._base + len(self._frames)
        while self._sched_i < len(self._schedule):
            p0, p1, e0, e1 = self._schedule[self._sched_i]
            if not final and total < p1:
                break
            pe, ee = (min(p1, total), min(e1, total)) if final else (p1, e1)
            if p0 < pe and e0 < ee:
                # The window's tokens, parallel to the frames passed below:
                # families that key conditioning to raw-stream identity
                # (e.g. the pulse gain's sync veto) read them from here
                # without changing the _upscale_window contract.
                self._window_tokens = self._tokens[
                    p0 - self._base : pe - self._base]
                sr = self._upscale_window(self._frames[p0 - self._base : pe - self._base])
                out.extend((sr[g - p0][0], self._tokens[g - self._base]) for g in range(e0, ee))
            self._sched_i += 1
        keep_from = (
            self._schedule[self._sched_i][0]
            if self._sched_i < len(self._schedule)
            else self._base + len(self._frames)
        )
        drop = keep_from - self._base
        if drop > 0:
            self._frames = self._frames[drop:]
            self._tokens = self._tokens[drop:]
            self._base += drop
        return out

    def _run(self, ws: int, we: int, last: bool) -> list:
        # Both window nets (BasicVSR++ _upsample, RealBasicVSR _basicvsr) mx.eval each
        # output frame as it is produced, so the frames arrive materialized -- no extra
        # sync barrier here.
        sr = self._upscale_window(self._frames[ws - self._base : we - self._base])
        end = we if last else we - self._T
        out = [(sr[g - ws][0], self._tokens[g - self._base]) for g in range(self._emitted, end)]
        self._emitted = end
        return out

    def _upscale_window(self, frames: list) -> list:
        raise NotImplementedError
