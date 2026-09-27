"""Motion-compensated temporal denoise on optical flow.

Recursive/causal: keeps previous denoised frames, warps them into
alignment (Vision optical flow revision 1 by default; the media-engine
block matcher or the shared SpyNet as measured alternatives), and blends
where the photometric gate verifies the warp - so static regions integrate over time and moving edges do
not ghost. MLX-array in / out, zero output delay.
"""

import dataclasses
from collections.abc import Mapping
from pathlib import Path as _P
from typing import ClassVar, cast

import mlx.core as mx

import kinovsr.modeling as _modeling
from kinovsr.analysis.noise.track import NoiseMapTracker, PulseGain, source_since_sync
from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.modeling.flow_check import self_test_flow
from kinovsr.modeling.vision_flow import VisionFlowEngine
from kinovsr.modeling.vsr_blocks import compiled_spynet_flow
from kinovsr.modeling.vtme_flow import VtmeFlowEngine
from kinovsr.processors.capabilities import (
    Capability,
    CapabilitySpec,
    TemporalMode,
)
from kinovsr.processors.conditioning import (
    NOISE_MAP_KEYS,
    NoiseMapConfig,
    build_conditioning,
    noise_map_debug_image,
    noise_map_diagnostics,
    parse_noise_map,
)
from kinovsr.processors.errors import append_cleanup_context
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


def _grid(h: int, w: int) -> tuple[mx.array, mx.array]:
    ys, xs = mx.meshgrid(mx.arange(h), mx.arange(w), indexing="ij")
    return ys.astype(mx.float32), xs.astype(mx.float32)


def warp(
    img: mx.array,
    flow: mx.array,
    grid: tuple[mx.array, mx.array] | None = None,
) -> mx.array:
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

    def g(yy: mx.array, xx: mx.array) -> mx.array:
        return flat[(yy * w + xx).reshape(-1)].reshape(h, w, c)

    top = g(y0, x0) * (1 - wx) + g(y0, x1) * wx
    bot = g(y1, x0) * (1 - wx) + g(y1, x1) * wx
    return top * (1 - wy) + bot * wy


def _box_mean(x: mx.array, k: int) -> mx.array:
    """KxK box mean of an (H,W,C) array, same size out, via a depthwise grouped
    conv2d (no transposes: each channel convolved with its own box kernel)."""
    c = x.shape[2]
    ker = mx.full((c, k, k, 1), 1.0 / (k * k), dtype=x.dtype)
    return mx.conv2d(x[None], ker, stride=1, padding=k // 2, groups=c)[0]


class McTemporalDenoiser:
    """Motion-compensated temporal denoise on optical flow, with optional
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
        noise_map: NoiseMapTracker | None = None,
        map_refresh: int = 64,
        pulse: PulseGain | None = None,
        map_floor: float = 0.0,
        gate: str = "smooth",
        flow: str = "vision",
        flow_weights: str | _P | None = None,
    ):
        self.w, self.h = int(width), int(height)
        # flow: the motion engine. "vision" (default) = Vision optical flow
        # revision 1: the denoise-quality pick (best e2e mc output on every
        # measured clip). "spynet" = the stock BasicSR SpyNet via the shared
        # MLX implementation (warp-PSNR static/moving/fast 40.7/29.6/20.1 on
        # the measured real pairs) -- at the cost of MLX GPU time. "vtme" = VTMotionEstimationSession block matching on the media
        # engine (macOS 26+): the speed/isolation pick -- cheapest accurate
        # engine at SD and fully immune to MLX GPU saturation, at a
        # measured 0.2-0.5 dB e2e cost vs vision (its 4x4-block-constant
        # field aligns sub-pixel detail slightly worse, and photometric
        # matching partially matches the noise it should average away --
        # winning warp-PSNR does not mean winning denoise). Flow errors
        # only lower the ceiling either way: the residual gate audits
        # every warp before it blends.
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
        self._sigma_plane: mx.array | None = None  # (H,W,1) residual-units plane, or None
        self._recent: list[mx.array] = []  # rolling frames for estimate/refresh
        self._since_refresh = 0
        self._gain = 1.0  # current per-frame pulse gain
        self.last_noise_map: mx.array | None = None  # fp32 (H,W,1) sigma actually used (debug)
        self._pulse_log: list[float] = []
        self.clamp_k = 5  # neighborhood window for color clamping
        self.clamp_gamma = 1.25  # box half-width in std units
        self.occ_tau = 1.5  # FB-consistency tolerance (pixels)
        self.conf_scale = 10.0  # flow magnitude (px) at which confidence ~1/e
        # gate-openness run stat: mean realized blend weight / strength =
        # the fraction of the possible temporal denoise the flow actually
        # unlocked (flow-limited clips read low). Accumulated as a RESIDENT
        # MLX scalar and read once at end of run: a per-frame float() here
        # forces an extra mid-frame host/device sync (doc 06 forbids that
        # for diagnostics).
        self._w_sum: mx.array | None = None
        self._w_n = 0
        self._warp_grid: tuple[mx.array, mx.array] | None = _grid(self.h, self.w)
        self._vision: VisionFlowEngine | None = None
        self._vtme: VtmeFlowEngine | None = None
        self._prev: mx.array | None = None
        self._hist: list[mx.array] = []
        if self.flow_source == "spynet":
            path = flow_weights or default_settings().mc_flow_weights
            if not path:
                # The stock checkpoint ships as the shared modeling
                # component (5.5 MB); the family package carries none.

                path = (
                    _P(_modeling.__file__).parent
                    / "spynet"
                    / "weights"
                    / "spynet_stock_20210409.safetensors"
                )
            self._spynet_p = dict(cast(dict[str, mx.array], mx.load(str(path))))
            return
        if self.flow_source == "vision":
            # Vision revision 1 (Medium accuracy), pinned; see
            # kinovsr.modeling.vision_flow for the convention and placement.

            try:
                self._vision = VisionFlowEngine(self.w, self.h)
                if self_test:
                    self._self_test_flow()
            except BaseException as active:
                try:
                    self.close()
                except BaseException as cleanup:
                    append_cleanup_context(active, cleanup)
                raise
            return
        if self.flow_source == "vtme":
            # Media-engine block matching (4x4 multipass pinned); see
            # kinovsr.modeling.vtme_flow for the convention and placement.

            try:
                self._vtme = VtmeFlowEngine(self.w, self.h)
                if self_test:
                    self._self_test_flow()
            except BaseException as active:
                try:
                    self.close()
                except BaseException as cleanup:
                    append_cleanup_context(active, cleanup)
                raise
            return
        raise ValueError(f"mc flow engine must be one of {_FLOW_ENGINES}; got {self.flow_source!r}")

    def _self_test_flow(self) -> None:
        """Catch silent-zero and wrong-sign flow up front."""
        self_test_flow(
            lambda curr, ref: self._compute_flows(curr, [ref])[0][0],
            self.w,
            self.h,
            name="VTMotionEstimation" if self.flow_source == "vtme" else "Vision optical flow",
            consumer="--denoise mc",
        )

    def reset(self) -> None:
        """Drop temporal history (call at scene cuts). The estimated noise map is
        kept (the encoder's noise character persists across cuts); the pulse diff
        chain restarts."""
        self._prev = None
        self._hist = []
        if self._pulse is not None:
            self._pulse.reset()

    def _condition(self, rgb_f32: mx.array, since_sync: int | None = None) -> None:
        """Per-frame conditioning upkeep: estimate/refresh the sigma plane from
        recent frames and update the pulse gain."""
        if self._pulse is not None:
            self._gain = self._pulse.update(rgb_f32, since_sync=since_sync)
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
        vision, self._vision = getattr(self, "_vision", None), None
        if vision is not None:
            try:
                vision.close()
            except BaseException as exc:
                failures.append(exc)
        vtme, self._vtme = getattr(self, "_vtme", None), None
        if vtme is not None:
            try:
                vtme.close()
            except BaseException as exc:
                failures.append(exc)
        self._prev = None
        self._hist = []
        self._warp_grid = None
        if failures:
            for cleanup in failures[1:]:
                append_cleanup_context(failures[0], cleanup)
            raise failures[0]

    def _compute_flows(
        self, curr: mx.array, refs: list[mx.array]
    ) -> list[tuple[mx.array, mx.array | None]]:
        """Optical flow of each reference -> current. SpyNet path: the shared
        MLX implementation; spynet_flow(cur, ref) equals -forward in this
        convention, so the downstream warp/occlusion/confidence math is
        untouched. Returns [(forwardFlow, backwardFlow_or_None), ...] as
        (H,W,2) px MLX arrays.
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
        # Vision and vtme: compute(curr, ref) shares spynet_flow(curr, ref)'s
        # orientation, so the same negation lands in the warp convention.
        eng = self._vision if self.flow_source == "vision" else self._vtme
        out = []
        for ref in refs:
            assert eng is not None
            fwd = -eng.compute(curr, ref)
            bwd = -eng.compute(ref, curr) if self.occlusion else None
            mx.eval(fwd)
            out.append((fwd, bwd))
        return out

    def _weight(
        self, anchor: mx.array, warped: mx.array, fwd: mx.array, bwd: mx.array | None
    ) -> mx.array:
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
            assert bwd is not None
            fwd_at = warp(fwd, bwd, self._warp_grid)
            fb = mx.sqrt(mx.sum((bwd + fwd_at) ** 2, axis=-1, keepdims=True) + 1e-8)
            w = w * mx.exp(-((fb / self.occ_tau) ** 2))
        if self.confidence:
            mag = mx.sqrt(mx.sum(fwd**2, axis=-1, keepdims=True) + 1e-8)
            w = w * mx.exp(-((mag / self.conf_scale) ** 2))
        wm = mx.mean(w)
        self._w_sum = wm if self._w_sum is None else self._w_sum + wm
        self._w_n += 1
        return w

    @property
    def gate_openness(self) -> float:
        """Mean realized blend weight / strength over the run: how much of
        the possible temporal denoise the flow unlocked (flow-limited
        footage reads low; raising strength cannot fix a low value, a
        better flow engine can)."""
        if not self._w_n or self._w_sum is None or self.strength <= 0:
            return 0.0
        return (float(self._w_sum) / self._w_n) / self.strength

    def denoise(self, rgb_f32: mx.array, since_sync: int | None = None) -> mx.array:
        rgb_f32 = mx.clip(rgb_f32[..., :3].astype(mx.float32), 0.0, 1.0)
        if self._tracker is not None or self._pulse is not None:
            self._condition(rgb_f32, since_sync=since_sync)
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
        # One sync per frame: the gate-openness accumulator rides along
        # instead of forcing its own mid-frame evaluation per reference.
        if self._w_sum is None:
            mx.eval(out)
        else:
            mx.eval(out, self._w_sum)
        self._remember(rgb_f32, out)
        return out

    def _remember(self, curr: mx.array, out: mx.array) -> None:
        if self.window == 0:
            self._prev = out  # recursive: keep output
        else:
            self._hist.append(curr)  # FIR: keep input frames
            if len(self._hist) > self.window:
                self._hist.pop(0)


# ===========================================================================
# Processor family: a causal motion-compensated temporal denoiser
# ===========================================================================

_FLOW_ENGINES = ("vision", "spynet", "vtme")
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


def _passthrough(spec: StreamSpec, config: object) -> StreamSpec:  # noqa: ARG001 - hook signature
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

    def feed(self, rgb: mx.array, token: object = None) -> list[tuple[mx.array, object]]:

        if self._engine is None:
            self._engine = self._make_engine(int(rgb.shape[0]), int(rgb.shape[1]))
        return [(self._engine.denoise(rgb, since_sync=source_since_sync(token)), token)]

    def flush(self) -> list[tuple[mx.array, object]]:
        return []

    def reset(self) -> None:
        if self._engine is not None:
            self._engine.reset()

    def run_diagnostics(self) -> list[str]:

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

    def debug_images(self) -> dict[str, mx.array]:

        return noise_map_debug_image(self._engine) if self._engine else {}

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()


class McFactory:
    name = "mc"

    capabilities: ClassVar[dict[Capability, CapabilitySpec]] = {
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
        raw: Mapping[str, object],
        *,
        capability: Capability,  # noqa: ARG002 - protocol signature
        profile: str | None,  # noqa: ARG002 - protocol signature
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
        flow = typed_value(raw, "flow", str, "vision")
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
            flow_weights=(typed_value(raw, "flow_weights", str) or settings.mc_flow_weights),
            noise_map=parse_noise_map(raw),
            luma_strength=luma_strength,
            chroma_strength=chroma_strength,
        )

    def build(self, config: McStageConfig, *, context: PipelineContext) -> FeedFlushProcessor:  # noqa: ARG002 - protocol signature
        return FeedFlushProcessor(
            lambda: _McDriver(config),
            luma_strength=config.luma_strength,
            chroma_strength=config.chroma_strength,
        )


FACTORY = McFactory()
