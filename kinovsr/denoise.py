"""Pre-upscale denoisers for the VSR harness.

Two options, both run at native resolution BEFORE super-resolution (the correct
order: SR synthesizes/amplifies high-frequency detail, so it bakes in noise it's
fed - clean first):

- SpatialDenoiser: per-frame CoreImage CINoiseReduction. No temporal state, cheap.
- McTemporalDenoiser: motion-compensated temporal denoise built on VideoToolbox
  optical flow (VTOpticalFlow, GPU, supported on M1 unlike the AVE-based
  VTTemporalNoiseFilter). Recursive/causal: it keeps the previous denoised
  frame, computes optical flow to it, warps it into alignment, and blends - more
  where the warp matches (static regions), less where it doesn't (occlusion /
  fast motion), so moving edges don't ghost.

Interface is MLX-array in / MLX-array out: (H,W,3) float32 RGB in [0, 1].
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mlx.core as mx

from . import pixel_buffers as _pb
from ._compat import Foundation, Quartz, autorelease_pool, require_pyobjc, vt
from .vsr import _suppress_native_stderr

# Max optical-flow calls in flight at once. Each flow is an ANE/IOKit dispatch
# (fixed ~17 ms, latency-bound) - a couple overlap well, but the kernel-side
# dispatch cost (System CPU) climbs with every concurrent flow. 2 measured as
# the sweet spot: it captures most of the overlap (~1.6x for two) while keeping
# System CPU low; beyond it the per-flow wall-time gain shrinks fast, and past
# ~5 it oversubscribes and spills to CPU. A large --mc-window still computes all
# its references, just no more than this many flows concurrently.
_MAX_CONCURRENT_FLOWS = 2

# Cache the sampling grid per resolution; it's constant across frames.
_GRID: dict[tuple[int, int], tuple[Any, Any]] = {}


def _grid(h: int, w: int) -> tuple[Any, Any]:
    g = _GRID.get((h, w))
    if g is None:
        ys, xs = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
        g = (ys.astype(mx.float32), xs.astype(mx.float32))
        _GRID[(h, w)] = g
    return g


def warp(img: Any, flow: Any) -> Any:
    """Backward-warp an (H,W,C) f32 image by an (H,W,2) px flow field.

    out[p] = bilinear_sample(img, p + flow[p]). Used to pull a reference frame
    into alignment with the current one. Out-of-bounds samples clamp to edge.
    """
    h, w, c = img.shape
    ys, xs = _grid(h, w)
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


class SpatialDenoiser:
    """Per-frame CoreImage CINoiseReduction. Spatial only; no temporal state."""

    def __init__(self, strength: float = 0.5):
        require_pyobjc()
        # CINoiseReduction's inputNoiseLevel is ~0.0-0.1 in practice; map strength
        # onto a gentle range so strength=0.5 is a moderate clean.
        self.noise_level = 0.01 + 0.04 * float(strength)
        self.sharpness = 0.4

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass

    def denoise(self, rgb_f32: Any) -> Any:
        # fp16 in / fp16 out: feed CoreImage a half-float CIImage and render back
        # to half-float (kCIFormatRGBAh), so no 8-bit quantization round trip.
        rgb_f32 = mx.clip(rgb_f32[..., :3].astype(mx.float32), 0.0, 1.0)
        h, w = int(rgb_f32.shape[0]), int(rgb_f32.shape[1])
        rgba = mx.concatenate(
            [rgb_f32.astype(mx.float16), mx.ones((h, w, 1), dtype=mx.float16)], axis=-1,
        )
        src = memoryview(mx.contiguous(rgba)).cast("B")
        data = Foundation.NSData.dataWithBytes_length_(src, len(src))
        ci = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
            data, w * 8, (w, h), Quartz.kCIFormatRGBAh, _pb.srgb_colorspace(),
        )
        filt = Quartz.CIFilter.filterWithName_("CINoiseReduction")
        filt.setValue_forKey_(ci, "inputImage")
        filt.setValue_forKey_(float(self.noise_level), "inputNoiseLevel")
        filt.setValue_forKey_(float(self.sharpness), "inputSharpness")
        out = filt.valueForKey_("outputImage")
        buf = bytearray(w * h * 8)
        _pb.ci_context().render_toBitmap_rowBytes_bounds_format_colorSpace_(
            out, buf, w * 8, ((0, 0), (w, h)), Quartz.kCIFormatRGBAh, _pb.srgb_colorspace(),
        )
        rgba_out = mx.array(memoryview(buf)).view(mx.float16).reshape(h, w, 4)
        return mx.contiguous(rgba_out[..., :3]).astype(mx.float32)


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

    MAP_WARMUP = 9   # frames observed before estimating a spatial noise map

    def __init__(
        self, width: int, height: int, strength: float = 0.5,
        window: int = 0, clamp: bool = False, occlusion: bool = False,
        confidence: bool = False, sigma: float = 0.06, self_test: bool = True,
        noise_map: Any = None, map_refresh: int = 64, pulse: Any = None,
    ):
        require_pyobjc()
        self.w, self.h = int(width), int(height)
        self.strength = float(strength)   # max blend weight toward a reference
        self.window = max(0, int(window))
        self.clamp = bool(clamp)
        self.occlusion = bool(occlusion)
        self.confidence = bool(confidence)
        # Tunables (sensible fixed defaults; strength is the user knob).
        # sigma is the residual-rejection scale of the photometric match gate
        # exp(-(resid/sigma)^2): larger = tolerate a bigger current-vs-history
        # difference before throttling the blend, so noise (which inflates that
        # residual) stops gating its own removal -> stronger denoise, more ghosting.
        self.sigma = float(sigma)   # residual rejection scale (luma, [0,1])
        # optional NoiseMapTracker / PulseGain: replace the scalar sigma with a
        # per-pixel plane (estimated from the footage, scaled to residual units)
        # and scale it per frame for GOP-phase noise pulsing. mc's sigma has an
        # exact analytic role, so unlike the learned nets there is no training
        # distribution to respect -- the gate simply gets the measured scale.
        self._tracker = noise_map
        self._pulse = pulse
        self._map_refresh = max(0, int(map_refresh))
        self._sigma_plane: Any = None    # (H,W,1) residual-units plane, or None
        self._recent: list[Any] = []     # rolling frames for estimate/refresh
        self._since_refresh = 0
        self._gain = 1.0                 # current per-frame pulse gain
        self.last_noise_map: Any = None  # fp32 (H,W,1) sigma actually used (debug)
        self._pulse_log: list[float] = []
        self.clamp_k = 5         # neighborhood window for color clamping
        self.clamp_gamma = 1.25  # box half-width in std units
        self.occ_tau = 1.5       # FB-consistency tolerance (pixels)
        self.conf_scale = 10.0   # flow magnitude (px) at which confidence ~1/e
        cls = vt.VTOpticalFlowConfiguration
        if not cls.isSupported():
            raise SystemExit("VTOpticalFlow is not supported on this device.")
        # One flow worker (session + buffers) per window reference, so the
        # references' flows run concurrently. VTOpticalFlow is fixed-overhead /
        # latency-bound (~17 ms at any resolution) and releases the GIL during
        # the call, so N parallel sessions overlap (~1.6x for 2) rather than
        # serialize - the only real lever for the window's cost.
        self._src_attrs: Any = None
        self._dst_attrs: Any = None
        self._workers = [self._make_worker(cls) for _ in range(max(1, self.window))]
        # Single shared "current" buffer: every flow reads the same current frame,
        # so we upload it once per frame instead of once per reference.
        self._curr_buf = _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._src_attrs)
        # Bounded thread pool so concurrent flows never oversubscribe (capped
        # well below the window for large windows). None in recursive mode.
        self._pool = (
            ThreadPoolExecutor(max_workers=min(self.window, _MAX_CONCURRENT_FLOWS))
            if self.window > 1 else None
        )
        self._prev: Any = None       # previous OUTPUT frame (recursive mode)
        self._hist: list[Any] = []   # last N INPUT frames, oldest first (FIR mode)
        self._idx = 0
        if self_test:
            try:
                self._self_test_flow()
            except Exception:
                self.close()
                raise

    def _make_worker(self, cls: Any) -> dict:
        cfg = cls.alloc().initWithFrameWidth_frameHeight_qualityPrioritization_revision_(
            self.w, self.h,
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
        self._src_attrs = dict(cfg.sourcePixelBufferAttributes() or {})
        self._dst_attrs = dict(cfg.destinationPixelBufferAttributes() or {})
        return {
            "proc": proc,
            # Per-worker source (the reference) + flow outputs. The "next" frame
            # (current) is a single shared buffer (see _curr_buf) - the flow only
            # reads it, so concurrent reads are fine and we upload it once/frame.
            "ref": _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._src_attrs),
            "fwd": _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._dst_attrs),
            "bwd": _pb.make_pixel_buffer_from_attrs(self.w, self.h, self._dst_attrs),
        }

    def _self_test_frames(self, shift: int) -> tuple[Any, Any]:
        ys, xs = mx.meshgrid(mx.arange(self.h), mx.arange(self.w), indexing="ij")
        xi = xs.astype(mx.int32)
        yi = ys.astype(mx.int32)
        noise = (
            (xi * 37 + yi * 17 + (xi // 13) * 29 + (yi // 11) * 31) % 256
        ).astype(mx.float32) / 255.0
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
        due = (self._sigma_plane is None and len(self._recent) >= self.MAP_WARMUP)
        if not due and self._sigma_plane is not None and self._map_refresh > 0:
            self._since_refresh += 1
            due = self._since_refresh >= self._map_refresh and len(self._recent) >= 2
        if due:
            sig = self._tracker.update(self._recent)
            if sig is not None:
                self.last_noise_map = sig
                self._sigma_plane = sig.astype(mx.float32) * self.RESID_FROM_SIGMA
            self._since_refresh = 0

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for wk in self._workers:
            if wk["proc"] is not None:
                wk["proc"].endSession()
                wk["proc"] = None

    def _upload(self, rgb_f32: Any, buf: Any) -> None:
        h, w = int(rgb_f32.shape[0]), int(rgb_f32.shape[1])
        rgba = mx.concatenate(
            [rgb_f32.astype(mx.float16), mx.ones((h, w, 1), mx.float16)], axis=-1,
        )
        _pb.write_fp16_rgba(rgba, buf)

    def _read_flow(self, pb: Any) -> Any:
        Quartz.CVPixelBufferLockBaseAddress(pb, 1)
        try:
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
            base = Quartz.CVPixelBufferGetBaseAddress(pb)
            raw = mx.array(memoryview(base.as_buffer(self.h * bpr)))
            flow = raw.view(mx.float16).reshape(self.h, bpr // 2)[:, : self.w * 2].reshape(
                self.h, self.w, 2,
            ).astype(mx.float32)
            mx.eval(flow)
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
        return flow

    def _compute_flows(self, curr: Any, refs: list[Any]) -> list[tuple[Any, Any]]:
        """Optical flow of each reference -> current. The references run on
        separate sessions concurrently (the GIL is released during the VT call,
        so they overlap). Uploads/reads are MLX and stay on the main thread;
        only the processWithParameters calls are threaded. Returns
        [(forwardFlow, backwardFlow_or_None), ...] as (H,W,2) px MLX arrays.
        """
        self._upload(curr, self._curr_buf)              # once, shared by all flows
        jobs = []
        for j, ref in enumerate(refs):
            wk = self._workers[j]
            self._upload(ref, wk["ref"])
            sf = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                wk["ref"], _pb.frame_pts(0, 24.0),
            )
            nf = vt.VTFrameProcessorFrame.alloc().initWithBuffer_presentationTimeStamp_(
                self._curr_buf, _pb.frame_pts(1, 24.0),
            )
            fo = vt.VTFrameProcessorOpticalFlow.alloc().initWithForwardFlow_backwardFlow_(
                wk["fwd"], wk["bwd"],
            )
            pa = vt.VTOpticalFlowParameters.alloc().initWithSourceFrame_nextFrame_submissionMode_destinationOpticalFlow_(
                sf, nf, vt.VTOpticalFlowParametersSubmissionModeRandom, fo,
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
            list(self._pool.map(run, range(len(jobs))))   # <= _MAX_CONCURRENT_FLOWS in flight
        for e in errs:
            if e is not None:
                raise RuntimeError(f"VTOpticalFlow process failed: {e}")

        out = []
        for j in range(len(refs)):
            wk = self._workers[j]
            out.append((
                self._read_flow(wk["fwd"]),
                self._read_flow(wk["bwd"]) if self.occlusion else None,
            ))
        return out

    def _weight(self, curr: Any, warped: Any, fwd: Any, bwd: Any) -> Any:
        """Per-pixel blend weight (H,W,1) toward `warped`, combining the enabled
        gates: residual match, FB-consistency occlusion, motion confidence."""
        resid = mx.mean(mx.abs(curr - warped), axis=-1, keepdims=True)
        sigma = self.sigma if self._sigma_plane is None else self._sigma_plane
        if self._gain != 1.0:
            sigma = sigma * self._gain
        w = self.strength * mx.exp(-((resid / sigma) ** 2))
        if self.occlusion:
            # Round-trip: curr pixel p -> ref at p+bwd[p], then fwd should return
            # it; |bwd + fwd(at p+bwd)| ~ 0 when consistent, large at occlusion.
            fwd_at = warp(fwd, bwd)
            fb = mx.sqrt(mx.sum((bwd + fwd_at) ** 2, axis=-1, keepdims=True) + 1e-8)
            w = w * mx.exp(-((fb / self.occ_tau) ** 2))
        if self.confidence:
            mag = mx.sqrt(mx.sum(fwd ** 2, axis=-1, keepdims=True) + 1e-8)
            w = w * mx.exp(-((mag / self.conf_scale) ** 2))
        return w

    def denoise(self, rgb_f32: Any) -> Any:
        rgb_f32 = mx.clip(rgb_f32[..., :3].astype(mx.float32), 0.0, 1.0)
        if self._tracker is not None or self._pulse is not None:
            self._condition(rgb_f32)
        refs = ([self._prev] if self._prev is not None else []) if self.window == 0 \
            else list(self._hist)
        if not refs:
            self._remember(rgb_f32, rgb_f32)
            return rgb_f32
        lo = hi = None
        if self.clamp:
            mean = _box_mean(rgb_f32, self.clamp_k)
            var = mx.maximum(_box_mean(rgb_f32 * rgb_f32, self.clamp_k) - mean * mean, 0.0)
            std = mx.sqrt(var)
            lo, hi = mean - self.clamp_gamma * std, mean + self.clamp_gamma * std
        flows = self._compute_flows(rgb_f32, refs)      # references run concurrently
        acc = rgb_f32                                   # current frame, weight 1
        wsum = mx.ones((self.h, self.w, 1))
        for ref, (fwd, bwd) in zip(refs, flows, strict=True):
            warped = warp(ref, -fwd)
            if self.clamp:
                warped = mx.clip(warped, lo, hi)
            w = self._weight(rgb_f32, warped, fwd, bwd)
            acc = acc + w * warped
            wsum = wsum + w
        out = mx.clip(acc / wsum, 0.0, 1.0)
        mx.eval(out)
        self._remember(rgb_f32, out)
        return out

    def _remember(self, curr: Any, out: Any) -> None:
        if self.window == 0:
            self._prev = out                            # recursive: keep output
        else:
            self._hist.append(curr)                     # FIR: keep input frames
            if len(self._hist) > self.window:
                self._hist.pop(0)
        self._idx += 1


def luma_chroma_blend(orig: Any, new: Any, a_luma: float, a_chroma: float,
                      kr: float = 0.299, kb: float = 0.114) -> Any:
    """Recombine `orig` and `new` (both (H,W,3) RGB in [0,1]) with separate blend
    strengths for luma and chroma: the output luma is lerp(orig, new, a_luma) and the
    chroma is lerp(orig, new, a_chroma). a=1 takes the new (denoised) value, a=0 keeps the
    original; a_luma=a_chroma=1 returns `new` exactly. (kr, kb) are the ITU-R luma
    coefficients (default BT.601); pass the source matrix's so the split matches the clip's
    color space -- though they only affect the result when a_luma != a_chroma (otherwise
    the YCbCr basis cancels out of the lerp). Computed in float32 -- the chroma-scale
    divisions coarsen in fp16."""
    kg = 1.0 - kr - kb
    cb_s, cr_s = 2.0 * (1.0 - kb), 2.0 * (1.0 - kr)
    o = orig.astype(mx.float32)
    n = new.astype(mx.float32)

    def _yc(x):
        y = kr * x[..., 0:1] + kg * x[..., 1:2] + kb * x[..., 2:3]
        return y, (x[..., 2:3] - y) / cb_s, (x[..., 0:1] - y) / cr_s     # y, cb, cr

    yo, cbo, cro = _yc(o)
    yn, cbn, crn = _yc(n)
    y = yo + a_luma * (yn - yo)
    cb = cbo + a_chroma * (cbn - cbo)
    cr = cro + a_chroma * (crn - cro)
    r = y + cr_s * cr
    b = y + cb_s * cb
    g = (y - kr * r - kb * b) / kg
    return mx.clip(mx.concatenate([r, g, b], axis=-1), 0.0, 1.0)


class LumaChromaDenoiser:
    """Wrap any harness denoiser to apply separate luma/chroma blend strengths between
    its input and output -- e.g. denoise chroma hard (a_chroma=1) while keeping luma
    texture (a_luma<1), the split a single joint RGB sigma cannot do. The base still
    denoises RGB jointly; this only re-weights its effect per channel group on the way
    out.

    Threads the input frame through the base's token so delay-line denoisers (FastDVDnet)
    still pair each delayed output with its own input; per-frame denoisers (spatial / mc)
    blend in step. Presents the feed/flush interface either way."""

    def __init__(self, base: Any, luma_strength: float = 1.0, chroma_strength: float = 1.0,
                 kr: float = 0.299, kb: float = 0.114):
        self._base = base
        self._al = float(luma_strength)
        self._ac = float(chroma_strength)
        self._kr = float(kr)       # ITU-R luma coefficients of the source matrix
        self._kb = float(kb)

    def reset(self) -> None:
        self._base.reset()

    def close(self) -> None:
        if hasattr(self._base, "close"):
            self._base.close()

    def set_schedule(self, schedule: Any) -> None:
        if hasattr(self._base, "set_schedule"):
            self._base.set_schedule(schedule)

    def _blend(self, orig: Any, den: Any) -> Any:
        return luma_chroma_blend(orig, den, self._al, self._ac, self._kr, self._kb)

    def feed(self, rgb: Any, token: Any = None) -> list:
        if hasattr(self._base, "feed"):
            return [(self._blend(o, d), t)
                    for d, (o, t) in self._base.feed(rgb, token=(rgb, token))]
        return [(self._blend(rgb, self._base.denoise(rgb)), token)]

    def flush(self) -> list:
        if hasattr(self._base, "flush"):
            return [(self._blend(o, d), t) for d, (o, t) in self._base.flush()]
        return []
