"""Motion-compensated temporal denoise on optical flow.

Recursive/causal: keeps previous denoised frames, warps them into
alignment (VTOpticalFlow on CPU/AMX by default, the shared SpyNet
for accuracy at GPU cost), and blends where the photometric gate
verifies the warp - so static regions integrate over time and
moving edges do not ghost. MLX-array in / out, zero output delay.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mlx.core as mx

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.media import pixel_buffers as _pb
from kinovsr.modeling.vsr_blocks import compiled_spynet_flow
from kinovsr.modeling.vt_flow import _append_cleanup_context
from kinovsr.native.frameworks import Foundation, Quartz, autorelease_pool, vt
from kinovsr.native.vsr import _suppress_native_stderr
from kinovsr.processors.capabilities import (
    Capability,
    CapabilitySpec,
    TemporalMode,
)
from kinovsr.processors.conditioning import (
    NOISE_MAP_KEYS,
    NoiseMapConfig,
    build_conditioning,
    parse_noise_map,
)
from kinovsr.processors.feed_driver import (
    LUMA_CHROMA_KEYS,
    FeedFlushProcessor,
    parse_luma_chroma,
)
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Domain,
    DType,
    Layout,
    StreamConstraint,
    StreamSpec,
)
from kinovsr.settings import Settings, default_settings

# Max optical-flow calls in flight at once. Each flow is an ANE/IOKit dispatch
# (fixed ~17 ms, latency-bound) - a couple overlap well, but the kernel-side
# dispatch cost (System CPU) climbs with every concurrent flow. 2 measured as
# the sweet spot: it captures most of the overlap (~1.6x for two) while keeping
# System CPU low; beyond it the per-flow wall-time gain shrinks fast, and past
# ~5 it oversubscribes and spills to CPU. A large --mc-window still computes all
# its references, just no more than this many flows concurrently.
_MAX_CONCURRENT_FLOWS = 2


def _grid(h: int, w: int) -> tuple[Any, Any]:
    ys, xs = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    return ys.astype(mx.float32), xs.astype(mx.float32)


def warp(
    img: Any,
    flow: Any,
    grid: tuple[Any, Any] | None = None,
) -> Any:
    """Backward-warp an (H,W,C) f32 image by an (H,W,2) px flow field.

    out[p] = bilinear_sample(img, p + flow[p]). Used to pull a reference frame
    into alignment with the current one. Out-of-bounds samples clamp to edge.
    """
    h, w, c = img.shape
    ys, xs = grid if grid is not None else _grid(h, w)
    sx = mx.clip(xs + flow[..., 0], 0, w - 1)
    sy = mx.clip(ys + flow[..., 1], 0, h - 1)
    x0 = mx.floor(sx).astype(mx.int32)
    y0 = mx.floor(sy).astype(mx.int32)
    x1 = mx.clip(x0 + 1, 0, w - 1)
    y1 = mx.clip(y0 + 1, 0, h - 1)
    wx = (sx - x0.astype(mx.float32))[..., None]
    wy = (sy - y0.astype(mx.float32))[..., None]
    flat = img.reshape(h * w, c)

    def g(yy: Any, xx: Any) -> Any:
        return flat[(yy * w + xx).reshape(-1)].reshape(h, w, c)

    top = g(y0, x0) * (1 - wx) + g(y0, x1) * wx
    bot = g(y1, x0) * (1 - wx) + g(y1, x1) * wx
    return top * (1 - wy) + bot * wy


def _box_mean(x: Any, k: int) -> Any:
    """KxK box mean of an (H,W,C) array, same size out, via a depthwise grouped
    conv2d (no transposes: each channel convolved with its own box kernel)."""
    c = x.shape[2]
    ker = mx.full((c, k, k, 1), 1.0 / (k * k), dtype=x.dtype)
    return mx.conv2d(x[None], ker, stride=1, padding=k // 2, groups=c)[0]


class McTemporalDenoiser:
    """Motion-compensated temporal denoise via VTOpticalFlow, with optional
    anti-ghosting refinements that compose:

    - window=0 (default): recursive/IIR - blends the current frame with the
      previous *output*, warped into alignment. Strongest noise reduction but
      ghosts have a long (recursive) lifetime.
    - window=N>=1: causal FIR - averages the current frame with the last N
      *input* frames, each warped into alignment. Bounded ghost lifetime
      (a bad warp ages out in <= N frames) at the cost of N flow computes/frame.
      Causal (past frames only); no lookahead.

    Optional gates, each multiplied into the per-reference blend weight:
    - clamp:      neighborhood color clamping (TAA variance-clip): clamp the
      warped reference into mean +/- gamma*std of the current frame's local
      window, so history that disagrees with the local appearance can't ghost.
    - occlusion:  forward-backward flow consistency - reject history where the
      forward and backward flow don't round-trip (occlusion / bad flow).
    - confidence: down-weight where the flow magnitude is large (fast motion).

    Interface: (H,W,3) float32 RGB in [0,1] in/out. Apply reset() at scene cuts.
    """

    # Converts a per-channel AWGN sigma (the noise-map estimator's units) to the
    # expected scale of mc's residual statistic: resid = mean_c |curr - warped|,
    # and for two noise-carrying frames E[|N(0, sqrt(2) sigma)|] = sqrt(4/pi) sigma.
    RESID_FROM_SIGMA = 1.1283791670955126

    MAP_WARMUP = 9  # frames observed before estimating a spatial noise map

    def __init__(
        self,
        width: int,
        height: int,
        strength: float = 0.5,
        window: int = 0,
        clamp: bool = False,
        occlusion: bool = False,
        confidence: bool = False,
        sigma: float = 0.06,
        self_test: bool = True,
        noise_map: Any = None,
        map_refresh: int = 64,
        pulse: Any = None,
        map_floor: float = 0.0,
        gate: str = "smooth",
        flow: str = "vt",
        flow_weights: Any = None,
    ):
        self.w, self.h = int(width), int(height)
        # flow: the motion engine. "vt" (default) = VTOpticalFlow Quality on
        # CPU/AMX (~17 ms fixed, zero GPU contention, mediocre accuracy with
        # smooth errors). "spynet" = the stock BasicSR SpyNet via the shared
        # MLX implementation: +3-5 dB warp-PSNR in every motion regime
        # (static/moving/fast 40.7/29.6/20.1 vs VT 36.9/25.0/16.1), so more
        # pixels pass the photometric gate and get denoised -- at the cost
        # of MLX GPU time. Flow errors only lower the ceiling either way:
        # the residual gate audits every warp before it blends.
        self.flow_source = str(flow)
        self.strength = float(strength)  # max blend weight toward a reference
        self.window = max(0, int(window))
        self.clamp = bool(clamp)
        self.occlusion = bool(occlusion)
        self.confidence = bool(confidence)
        # Tunables (sensible fixed defaults; strength is the user knob).
        # sigma is the residual-rejection scale of the photometric match gate
        # exp(-(resid/sigma)^2): larger = tolerate a bigger current-vs-history
        # difference before throttling the blend, so noise (which inflates that
        # residual) stops gating its own removal -> stronger denoise, more ghosting.
        self.sigma = float(sigma)  # residual rejection scale (luma, [0,1])
        # gate: what a reference's residual is measured AGAINST.
        # "smooth" (default): the residual anchor is a 3x3 box mean of the
        # current frame, with the gate width recalibrated so the mean
        # tolerance matches "curr". The anchor's own noise then stops
        # randomly opening/closing the gate per pixel (less self-gating)
        # while the anchor still needs no correspondence, so ghost
        # rejection is untouched. "curr": legacy, residual vs the raw
        # current frame. ("median" -- gating against the warped-consensus
        # median -- was tried and REFUTED on ground truth: flow-warp
        # errors are correlated across references, so in occlusion regions
        # the median IS the ghost and gating against it admits it, -6 dB
        # on a motion fixture. Do not retry.)
        self.gate = str(gate)
        # E|curr - warped| for two sigma-noisy frames is sqrt(2)*sqrt(2/pi)
        # *sigma; with a box-9 anchor it is sqrt(1+1/9)*sqrt(2/pi)*sigma,
        # a factor 0.745 -- fold it in so the sigma knob keeps its meaning.
        self._resid_scale = 0.745 if self.gate == "smooth" else 1.0
        # optional NoiseMapTracker / PulseGain: replace the scalar sigma with a
        # per-pixel plane (estimated from the footage, scaled to residual units)
        # and scale it per frame for GOP-phase noise pulsing. mc's sigma has an
        # exact analytic role, so unlike the learned nets there is no training
        # distribution to respect -- the gate simply gets the measured scale.
        self._tracker = noise_map
        self._pulse = pulse
        self._map_refresh = max(0, int(map_refresh))
        # user sigma floor under the map (static grain does not flicker, so the
        # temporal estimate reads low on it; the floor keeps a base gate width)
        self._map_floor = max(0.0, float(map_floor))
        self._sigma_plane: Any = None  # (H,W,1) residual-units plane, or None
        self._recent: list[Any] = []  # rolling frames for estimate/refresh
        self._since_refresh = 0
        self._gain = 1.0  # current per-frame pulse gain
        self.last_noise_map: Any = None  # fp32 (H,W,1) sigma actually used (debug)
        self._pulse_log: list[float] = []
        self.clamp_k = 5  # neighborhood window for color clamping
        self.clamp_gamma = 1.25  # box half-width in std units
        self.occ_tau = 1.5  # FB-consistency tolerance (pixels)
        self.conf_scale = 10.0  # flow magnitude (px) at which confidence ~1/e
        # gate-openness run stat: mean realized blend weight / strength =
        # the fraction of the possible temporal denoise the flow actually
        # unlocked (flow-limited clips read low)
        self._w_sum = 0.0
        self._w_n = 0
        self._warp_grid = _grid(self.h, self.w)
        self._src_attrs: Any = None
        self._dst_attrs: Any = None
        self._workers: list[dict[str, Any]] = []
        self._curr_buf: Any = None
        self._pool: ThreadPoolExecutor | None = None
        self._prev: Any = None
        self._hist: list[Any] = []
        if self.flow_source == "spynet":
            path = flow_weights or default_settings().spynet_weights
            if not path:
                # The stock checkpoint ships as the shared modeling
                # component (5.5 MB); the family package carries none.
                from pathlib import Path as _P

                import kinovsr.modeling as _modeling

                path = (
                    _P(_modeling.__file__).parent
                    / "spynet"
                    / "weights"
                    / "spynet_stock_20210409.safetensors"
                )
            self._spynet_p = dict(mx.load(str(path)))
            return
        cls = vt.VTOpticalFlowConfiguration
        if not cls.isSupported():
            raise SystemExit("VTOpticalFlow is not supported on this device.")
        # One flow worker (session + buffers) per window reference, so the
        # references' flows run concurrently. VTOpticalFlow is fixed-overhead /
        # latency-bound (~17 ms at any resolution) and releases the GIL during
        # the call, so N parallel sessions overlap (~1.6x for 2) rather than
        # serialize - the only real lever for the window's cost.
        try:
            for _ in range(max(1, self.window)):
                self._workers.append(self._make_worker(cls))
            # Every flow reads the same current frame, so upload it once.
            self._curr_buf = _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._src_attrs)
            self._pool = (
                ThreadPoolExecutor(max_workers=min(self.window, _MAX_CONCURRENT_FLOWS))
                if self.window > 1
                else None
            )
            if self_test:
                self._self_test_flow()
        except BaseException as active:
            try:
                self.close()
            except BaseException as cleanup:
                _append_cleanup_context(active, cleanup)
            raise

    def _make_worker(self, cls: Any) -> dict:
        cfg = cls.alloc().initWithFrameWidth_frameHeight_qualityPrioritization_revision_(
            self.w,
            self.h,
            vt.VTOpticalFlowConfigurationQualityPrioritizationQuality,
            cls.defaultRevision(),
        )
        if cfg is None:
            raise RuntimeError(f"VTOpticalFlow config init returned nil for {self.w}x{self.h}")
        proc = vt.VTFrameProcessor.alloc().init()
        with _suppress_native_stderr():
            ok, err = proc.startSessionWithConfiguration_error_(cfg, None)
        if not ok:
            raise RuntimeError(f"VTOpticalFlow startSession failed: {err}")
        try:
            self._src_attrs = dict(cfg.sourcePixelBufferAttributes() or {})
            self._dst_attrs = dict(cfg.destinationPixelBufferAttributes() or {})
            return {
                "proc": proc,
                "ref": _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._src_attrs),
                "fwd": _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._dst_attrs),
                "bwd": _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._dst_attrs),
            }
        except BaseException as active:
            try:
                proc.endSession()
            except BaseException as cleanup:
                _append_cleanup_context(active, cleanup)
            raise

    def _self_test_frames(self, shift: int) -> tuple[Any, Any]:
        ys, xs = mx.meshgrid(mx.arange(self.h), mx.arange(self.w), indexing="ij")
        xi = xs.astype(mx.int32)
        yi = ys.astype(mx.int32)
        noise = ((xi * 37 + yi * 17 + (xi // 13) * 29 + (yi // 11) * 31) % 256).astype(
            mx.float32
        ) / 255.0
        blocks = (((xi // 8 + yi // 8) % 2).astype(mx.float32) - 0.5) * 0.25
        base = 0.5 + (noise - 0.5) * 0.65 + blocks
        xs = xs.astype(mx.float32)
        ys = ys.astype(mx.float32)
        ref = mx.clip(
            mx.stack(
                [
                    base,
                    0.5 + (noise - 0.5) * 0.45 + 0.2 * mx.sin(xs * 0.31),
                    0.5 + (noise - 0.5) * 0.35 + 0.2 * mx.cos(ys * 0.27),
                ],
                axis=-1,
            ),
            0.0,
            1.0,
        ).astype(mx.float32)
        left = mx.broadcast_to(ref[:, :1], (self.h, shift, 3))
        curr = mx.concatenate([left, ref[:, : self.w - shift]], axis=1)
        mx.eval(ref, curr)
        return ref, curr

    def _self_test_flow(self) -> None:
        """Catch VTOpticalFlow silent-zero and bad-priority failures up front."""
        if self.w < 16 or self.h < 16:
            raise RuntimeError(
                f"VTOpticalFlow self-test requires at least 16x16; got {self.w}x{self.h}"
            )
        shift = 3 if self.w >= 32 else 1
        ref, curr = self._self_test_frames(shift)
        fwd, _ = self._compute_flows(curr, [ref])[0]
        y0, y1 = self.h // 4, self.h - self.h // 4
        x0, x1 = self.w // 4, self.w - self.w // 4
        crop = fwd[y0:y1, x0:x1] if y1 > y0 and x1 > x0 else fwd
        mean_x = mx.mean(crop[..., 0])
        mean_y = mx.mean(crop[..., 1])
        max_abs = mx.max(mx.abs(fwd))
        mx.eval(mean_x, mean_y, max_abs)
        mean_x_f = float(mean_x)
        mean_y_f = float(mean_y)
        max_abs_f = float(max_abs)
        expected = float(shift)
        if max_abs_f < 0.25:
            raise RuntimeError(
                "VTOpticalFlow self-test returned all-zero/near-zero flow for "
                f"{self.w}x{self.h}; --denoise mc is unsafe on this clip/device."
            )
        if (
            mean_x_f < expected * 0.65
            or mean_x_f > expected * 1.35
            or abs(mean_y_f) > max(0.75, expected * 0.5)
        ):
            raise RuntimeError(
                "VTOpticalFlow self-test failed for "
                f"{self.w}x{self.h}: expected about +{expected:.1f}px horizontal "
                f"flow, got mean=({mean_x_f:.3f}, {mean_y_f:.3f}), "
                f"max_abs={max_abs_f:.3f}."
            )

    def reset(self) -> None:
        """Drop temporal history (call at scene cuts). The estimated noise map is
        kept (the encoder's noise character persists across cuts); the pulse diff
        chain restarts."""
        self._prev = None
        self._hist = []
        if self._pulse is not None:
            self._pulse.reset()

    def _condition(self, rgb_f32: Any) -> None:
        """Per-frame conditioning upkeep: estimate/refresh the sigma plane from
        recent frames and update the pulse gain."""
        if self._pulse is not None:
            self._gain = self._pulse.update(rgb_f32)
            self._pulse_log.append(self._gain)
        if self._tracker is None:
            return
        self._recent.append(rgb_f32)
        if len(self._recent) > self.MAP_WARMUP:
            self._recent.pop(0)
        due = self._sigma_plane is None and len(self._recent) >= self.MAP_WARMUP
        if not due and self._sigma_plane is not None and self._map_refresh > 0:
            self._since_refresh += 1
            due = self._since_refresh >= self._map_refresh and len(self._recent) >= 2
        if due:
            sig = self._tracker.update(self._recent)
            if sig is not None:
                if self._map_floor > 0.0:
                    sig = mx.maximum(sig, self._map_floor)
                self.last_noise_map = sig
                self._sigma_plane = sig.astype(mx.float32) * self.RESID_FROM_SIGMA
            self._since_refresh = 0

    def close(self) -> None:
        failures: list[BaseException] = []
        pool, self._pool = self._pool, None
        if pool is not None:
            try:
                pool.shutdown(wait=True)
            except BaseException as exc:
                failures.append(exc)
        for wk in self._workers:
            proc, wk["proc"] = wk["proc"], None
            if proc is not None:
                try:
                    proc.endSession()
                except BaseException as exc:
                    failures.append(exc)
            wk["ref"] = None
            wk["fwd"] = None
            wk["bwd"] = None
        self._workers = []
        self._curr_buf = None
        self._src_attrs = None
        self._dst_attrs = None
        self._prev = None
        self._hist = []
        self._warp_grid = None
        if failures:
            for cleanup in failures[1:]:
                _append_cleanup_context(failures[0], cleanup)
            raise failures[0]

    def _upload(self, rgb_f32: Any, buf: Any) -> None:
        h, w = int(rgb_f32.shape[0]), int(rgb_f32.shape[1])
        rgba = mx.concatenate(
            [rgb_f32.astype(mx.float16), mx.ones((h, w, 1), mx.float16)],
            axis=-1,
        )
        _pb.write_fp16_rgba(rgba, buf)

    def _read_flow(self, pb: Any) -> Any:
        Quartz.CVPixelBufferLockBaseAddress(pb, 1)
        try:
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
            base = Quartz.CVPixelBufferGetBaseAddress(pb)
            raw = mx.array(memoryview(base.as_buffer(self.h * bpr)))
            flow = (
                raw.view(mx.float16)
                .reshape(self.h, bpr // 2)[:, : self.w * 2]
                .reshape(
                    self.h,
                    self.w,
                    2,
                )
                .astype(mx.float32)
            )
            mx.eval(flow)
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
        return flow

    def _compute_flows(self, curr: Any, refs: list[Any]) -> list[tuple[Any, Any]]:
        """Optical flow of each reference -> current. VT path: references run
        on separate sessions concurrently (the GIL is released during the VT
        call, so they overlap). SpyNet path: the shared MLX implementation;
        spynet_flow(cur, ref) equals -forward in this convention, so the
        downstream warp/occlusion/confidence math is untouched. Returns
        [(forwardFlow, backwardFlow_or_None), ...] as (H,W,2) px MLX arrays.
        """
        if self.flow_source == "spynet":
            out = []
            cur_b = curr[None]
            for ref in refs:
                fwd = -compiled_spynet_flow(self._spynet_p, cur_b, ref[None])[0]
                bwd = None
                if self.occlusion:
                    bwd = -compiled_spynet_flow(self._spynet_p, ref[None], cur_b)[0]
                mx.eval(fwd)
                out.append((fwd, bwd))
            return out
        self._upload(curr, self._curr_buf)  # once, shared by all flows
        jobs = []
        for j, ref in enumerate(refs):
            wk = self._workers[j]
            self._upload(ref, wk["ref"])
            sf = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                wk["ref"],
                _pb.frame_pts(0, 24.0),
            )
            nf = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                self._curr_buf,
                _pb.frame_pts(1, 24.0),
            )
            fo = vt.VTFrameProcessorOpticalFlow.alloc().initWithForwardFlow_backwardFlow_(
                wk["fwd"],
                wk["bwd"],
            )
            pa = vt.VTOpticalFlowParameters.alloc().initWithSourceFrame_nextFrame_submissionMode_destinationOpticalFlow_(
                sf,
                nf,
                vt.VTOpticalFlowParametersSubmissionModeRandom,
                fo,
            )
            jobs.append((wk["proc"], pa))

        errs: list[Any] = [None] * len(jobs)

        def run(j: int) -> None:
            with autorelease_pool():
                ok, err = jobs[j][0].processWithParameters_error_(jobs[j][1], None)
                if not ok:
                    errs[j] = err

        if self._pool is None or len(jobs) == 1:
            for j in range(len(jobs)):
                run(j)
        else:
            list(self._pool.map(run, range(len(jobs))))  # <= _MAX_CONCURRENT_FLOWS in flight
        for e in errs:
            if e is not None:
                raise RuntimeError(f"VTOpticalFlow process failed: {e}")

        out = []
        for j in range(len(refs)):
            wk = self._workers[j]
            out.append(
                (
                    self._read_flow(wk["fwd"]),
                    self._read_flow(wk["bwd"]) if self.occlusion else None,
                )
            )
        return out

    def _weight(self, anchor: Any, warped: Any, fwd: Any, bwd: Any) -> Any:
        """Per-pixel blend weight (H,W,1) toward `warped`, combining the enabled
        gates: residual match (vs the anchor -- current frame or window
        median), FB-consistency occlusion, motion confidence."""
        resid = mx.mean(mx.abs(anchor - warped), axis=-1, keepdims=True)
        sigma = self.sigma if self._sigma_plane is None else self._sigma_plane
        if self._gain != 1.0:
            sigma = sigma * self._gain
        if self._resid_scale != 1.0:
            sigma = sigma * self._resid_scale
        w = self.strength * mx.exp(-((resid / sigma) ** 2))
        if self.occlusion:
            # Round-trip: curr pixel p -> ref at p+bwd[p], then fwd should return
            # it; |bwd + fwd(at p+bwd)| ~ 0 when consistent, large at occlusion.
            fwd_at = warp(fwd, bwd, self._warp_grid)
            fb = mx.sqrt(mx.sum((bwd + fwd_at) ** 2, axis=-1, keepdims=True) + 1e-8)
            w = w * mx.exp(-((fb / self.occ_tau) ** 2))
        if self.confidence:
            mag = mx.sqrt(mx.sum(fwd**2, axis=-1, keepdims=True) + 1e-8)
            w = w * mx.exp(-((mag / self.conf_scale) ** 2))
        self._w_sum += float(mx.mean(w))
        self._w_n += 1
        return w

    @property
    def gate_openness(self) -> float:
        """Mean realized blend weight / strength over the run: how much of
        the possible temporal denoise the flow unlocked (flow-limited
        footage reads low; raising strength cannot fix a low value, a
        better flow engine can)."""
        if not self._w_n or self.strength <= 0:
            return 0.0
        return (self._w_sum / self._w_n) / self.strength

    def denoise(self, rgb_f32: Any) -> Any:
        rgb_f32 = mx.clip(rgb_f32[..., :3].astype(mx.float32), 0.0, 1.0)
        if self._tracker is not None or self._pulse is not None:
            self._condition(rgb_f32)
        refs = (
            ([self._prev] if self._prev is not None else [])
            if self.window == 0
            else list(self._hist)
        )
        if not refs:
            self._remember(rgb_f32, rgb_f32)
            return rgb_f32
        lo = hi = None
        if self.clamp:
            mean = _box_mean(rgb_f32, self.clamp_k)
            var = mx.maximum(_box_mean(rgb_f32 * rgb_f32, self.clamp_k) - mean * mean, 0.0)
            std = mx.sqrt(var)
            lo, hi = mean - self.clamp_gamma * std, mean + self.clamp_gamma * std
        flows = self._compute_flows(rgb_f32, refs)  # references run concurrently
        warpeds = []
        for ref, (fwd, _bwd) in zip(refs, flows, strict=True):
            warped = warp(ref, -fwd, self._warp_grid)
            if self.clamp:
                warped = mx.clip(warped, lo, hi)
            warpeds.append(warped)
        anchor = _box_mean(rgb_f32, 3) if self.gate == "smooth" else rgb_f32
        acc = rgb_f32  # current frame, weight 1
        wsum = mx.ones((self.h, self.w, 1))
        for warped, (fwd, bwd) in zip(warpeds, flows, strict=True):
            w = self._weight(anchor, warped, fwd, bwd)
            acc = acc + w * warped
            wsum = wsum + w
        out = mx.clip(acc / wsum, 0.0, 1.0)
        mx.eval(out)
        self._remember(rgb_f32, out)
        return out

    def _remember(self, curr: Any, out: Any) -> None:
        if self.window == 0:
            self._prev = out  # recursive: keep output
        else:
            self._hist.append(curr)  # FIR: keep input frames
            if len(self._hist) > self.window:
                self._hist.pop(0)


# ===========================================================================
# Processor family: a causal motion-compensated temporal denoiser
# ===========================================================================

_FLOW_ENGINES = ("vt", "spynet")
_GATES = ("smooth", "curr")


@dataclasses.dataclass(frozen=True, slots=True)
class McStageConfig:
    strength: float
    window: int
    sigma: float
    gate: str
    clamp: bool
    occlusion: bool
    confidence: bool
    flow: str
    flow_weights: str | None
    noise_map: NoiseMapConfig
    luma_strength: float = 1.0
    chroma_strength: float = 1.0


def _passthrough(spec: StreamSpec, config: object) -> StreamSpec:
    return spec


class _McDriver:
    """feed()/flush() shape over the recursive engine.

    The engine binds to a geometry at construction (its flow sessions
    are size-specific), so the driver creates it on the first frame -
    the same lazy pattern the metalfx driver uses.
    """

    def __init__(self, config: McStageConfig) -> None:
        self._config = config
        self._engine: McTemporalDenoiser | None = None

    def _make_engine(self, height: int, width: int) -> McTemporalDenoiser:
        config = self._config
        tracker, pulse = build_conditioning(config.noise_map)
        return McTemporalDenoiser(
            width,
            height,
            strength=config.strength,
            window=config.window,
            clamp=config.clamp,
            occlusion=config.occlusion,
            confidence=config.confidence,
            sigma=config.sigma,
            gate=config.gate,
            flow=config.flow,
            flow_weights=config.flow_weights,
            noise_map=tracker,
            map_refresh=config.noise_map.refresh,
            pulse=pulse,
            map_floor=config.noise_map.floor,
        )

    def feed(self, rgb: Any, token: Any = None) -> list:
        if self._engine is None:
            self._engine = self._make_engine(int(rgb.shape[0]), int(rgb.shape[1]))
        return [(self._engine.denoise(rgb), token)]

    def flush(self) -> list:
        return []

    def reset(self) -> None:
        if self._engine is not None:
            self._engine.reset()

    def run_diagnostics(self) -> list:
        from kinovsr.processors.conditioning import noise_map_diagnostics

        engine = self._engine
        if engine is None:
            return []
        lines = noise_map_diagnostics(engine)
        if engine.gate_openness > 0:
            lines.append(
                f"[denoise] mc gate openness: "
                f"{engine.gate_openness * 100:.1f}% of the strength ceiling "
                f"realized (flow={engine.flow_source}; low = flow-limited, "
                f"the lever is a better flow, not more strength)"
            )
        return lines

    def debug_images(self) -> dict:
        from kinovsr.processors.conditioning import noise_map_debug_image

        return noise_map_debug_image(self._engine) if self._engine else {}

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()


class McFactory:
    name = "mc"

    capabilities = {
        Capability.DENOISE: CapabilitySpec(
            capability=Capability.DENOISE,
            profiles=(),
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32,),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            produces=_passthrough,
            temporal_mode=TemporalMode.CAUSAL,
            temporal_radius=1,
            stateful=True,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> McStageConfig:
        reject_unknown_keys(
            raw,
            (
                "strength",
                "window",
                "sigma",
                "gate",
                "clamp",
                "occlusion",
                "confidence",
                "flow",
                "flow_weights",
                *LUMA_CHROMA_KEYS,
                *NOISE_MAP_KEYS,
            ),
        )
        strength = typed_value(raw, "strength", float, 0.5)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        window = typed_value(raw, "window", int, 0)
        if window < 0:
            raise ValueError("window must be >= 0")
        sigma = typed_value(raw, "sigma", float, 0.06)
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        gate = typed_value(raw, "gate", str, "smooth")
        if gate not in _GATES:
            raise ValueError(f"gate must be one of {_GATES}")
        flow = typed_value(raw, "flow", str, "vt")
        if flow not in _FLOW_ENGINES:
            raise ValueError(f"flow must be one of {_FLOW_ENGINES}")
        luma_strength, chroma_strength = parse_luma_chroma(raw)
        return McStageConfig(
            strength=strength,
            window=window,
            sigma=sigma,
            gate=gate,
            clamp=typed_value(raw, "clamp", bool, False),
            occlusion=typed_value(raw, "occlusion", bool, False),
            confidence=typed_value(raw, "confidence", bool, False),
            flow=flow,
            flow_weights=(typed_value(raw, "flow_weights", str) or settings.spynet_weights),
            noise_map=parse_noise_map(raw),
            luma_strength=luma_strength,
            chroma_strength=chroma_strength,
        )

    def build(self, config: McStageConfig, *, context: PipelineContext) -> FeedFlushProcessor:
        return FeedFlushProcessor(
            lambda: _McDriver(config),
            luma_strength=config.luma_strength,
            chroma_strength=config.chroma_strength,
        )


FACTORY = McFactory()
