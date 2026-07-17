"""Streaming driver for the MLX RealViformer upscaler.

RealViformer's recurrence is CAUSAL (forward-only), so unlike the bidirectional
VSR nets there is no window/trim buffering: each frame upscales as it arrives,
carrying the propagated features (and the previous frame, for the flow) across
calls. reset() drops the temporal state -- the harness calls it at hard cuts.

`window` chunks the recurrence: the state resets cold every `window` frames,
matching the reference inference's 100-frame interval processing. Unbounded
recurrence (window=0) runs the net into state depths the released tool never
reaches; on long near-static shots the texture lock then engraves hallucinated
detail (notched ridges on locked highlights, ripple contours on smooth
gradients) that keeps accumulating for hundreds of frames.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from kinovsr.modeling.upscaler_base import to_rgb_batch
from kinovsr.modeling.vsr_blocks import box3, compiled_spynet_flow, flow_warp

try:
    from . import net
except ImportError:   # running directly as a script
    import net


class RealViformerUpscaler:
    """Streaming feed()/flush() driver for the causal RealViformer."""

    def __init__(
        self, weights: Any = None, dtype: Any = mx.float16, compile: bool = True,
        window: int = 100, flow_mode: str = "spynet",
        history_strength: float = 1.0, history_gate: str = "off",
        history_cleanup: float = 0.25, history_gate_drop: float = 0.85,
        history_risk_decay: float = 0.80, history_static_cap: float = 0.0,
    ):
        if flow_mode not in ("spynet", "zero", "vt", "vision"):
            raise ValueError(
                f"RealViformer flow_mode must be 'spynet', 'zero', 'vt', or "
                f"'vision'; got {flow_mode!r}"
            )
        if history_gate not in ("off", "improve", "holistic"):
            raise ValueError(
                f"RealViformer history_gate must be 'off', 'improve', or 'holistic'; got {history_gate!r}"
            )
        if history_strength < 0.0:
            raise ValueError(
                f"RealViformer history_strength must be >= 0; got {history_strength!r}"
            )
        if not 0.0 <= history_cleanup <= 1.0:
            raise ValueError(
                f"RealViformer history_cleanup must be in [0, 1]; got {history_cleanup!r}"
            )
        if not 0.0 <= history_gate_drop <= 1.0:
            raise ValueError(
                f"RealViformer history_gate_drop must be in [0, 1]; got {history_gate_drop!r}"
            )
        if not 0.0 <= history_risk_decay < 1.0:
            raise ValueError(
                f"RealViformer history_risk_decay must be in [0, 1); got {history_risk_decay!r}"
            )
        if not 0.0 <= history_static_cap <= 1.0:
            raise ValueError(
                f"RealViformer history_static_cap must be in [0, 1]; got {history_static_cap!r}"
            )
        self._p = net.load_params(weights, dtype=dtype)
        self._cfg = net._config(self._p)
        self.scale = 4
        self._window = max(0, int(window))
        self._flow_mode = flow_mode
        self._history_strength = float(history_strength)
        self._history_gate = history_gate
        self._history_cleanup = float(history_cleanup)
        self._history_gate_drop = float(history_gate_drop)
        self._history_risk_decay = float(history_risk_decay)
        self._history_static_cap = float(history_static_cap)
        self._vt_flow: Any = None
        self._first, self._next = net.make_steps(self._p, self._cfg, compile=compile)
        self.reset()

    def reset(self) -> None:
        self._prev: Any = None       # previous input frame (for the flow)
        self._feat: Any = None       # propagated features
        self._risk: Any = None       # propagated suspicion map for holistic history
        self._depth = 0              # frames since the last cold start

    def close(self) -> None:
        if self._vt_flow is not None:
            self._vt_flow.close()
            self._vt_flow = None

    @staticmethod
    def _pad4(x: Any) -> tuple[Any, int, int]:
        """Reflect-pad top/left to multiples of 4, matching reference inference.

        RealViformer's U-Net downsamples twice, so arbitrary input sizes need
        padding to a multiple of 4. The reference pads left/top with torch
        ``reflect`` and later crops that scaled offset from the output.
        """
        _, h, w, _ = x.shape
        ph, pw = (-h) % 4, (-w) % 4
        if ph:
            if h <= ph:
                raise ValueError(f"RealViformer reflect padding needs height > {ph}; got {h}")
            yidx = mx.arange(ph, 0, -1)
            x = mx.concatenate([mx.take(x, yidx, axis=1), x], axis=1)
        if pw:
            if w <= pw:
                raise ValueError(f"RealViformer reflect padding needs width > {pw}; got {w}")
            xidx = mx.arange(pw, 0, -1)
            x = mx.concatenate([mx.take(x, xidx, axis=2), x], axis=2)
        return x, ph, pw

    def _vt_current_to_prev_flow(self, curr: Any, prev: Any, dtype: Any) -> Any:
        """VTOpticalFlow in the convention RealViformer's warp expects.

        McTemporalDenoiser's helper returns source -> next flow. For recurrent SR
        we need current -> previous so flow_warp(previous_features, flow) samples
        the previous feature map at p + flow[p].
        """
        if self._vt_flow is None:
            from kinovsr.processors.mc import McTemporalDenoiser

            _, h, w, _ = curr.shape
            self._vt_flow = McTemporalDenoiser(
                w, h, strength=0.0, window=1, self_test=True,
            )
        flow, _ = self._vt_flow._compute_flows(
            prev[0].astype(mx.float32), [curr[0].astype(mx.float32)],
        )[0]
        return flow[None].astype(dtype)

    def _history_gate_map(self, curr: Any, prev: Any, flow: Any, dtype: Any) -> Any:
        gate = mx.full((*curr.shape[:3], 1), self._history_strength, dtype=dtype)
        if self._history_gate == "off" or self._history_strength <= 0.0:
            return gate

        curr32 = curr.astype(mx.float32)
        prev32 = prev.astype(mx.float32)
        flow32 = flow.astype(mx.float32)
        warped_prev = flow_warp(prev32, flow32, "border")
        resid_warp = mx.mean(mx.abs(curr32 - warped_prev), axis=-1, keepdims=True)
        resid_zero = mx.mean(mx.abs(curr32 - prev32), axis=-1, keepdims=True)

        # Conservative alignment: history is useful only where the flow warp
        # measurably improves the source-frame residual. Near-static regions with
        # no clear improvement get little history, which prevents self-etching.
        improve = mx.clip((resid_zero - resid_warp) / 0.004, 0.0, 1.0)
        match = mx.exp(-((resid_warp / 0.035) ** 2))
        return (gate.astype(mx.float32) * improve * match).astype(dtype)

    def _holistic_history_policy(
        self, curr: Any, prev: Any, flow: Any, warped: Any, dtype: Any,
    ) -> tuple[Any, Any]:
        """Training-free HSA-lite: risk gates history and gently cleans it.

        Signals are computed from input frames + flow only. The warped hidden
        state is filtered after risk is decided, so cleanup cannot feed back into
        the detector inside the same frame.
        """
        gate0 = mx.full((*curr.shape[:3], 1), self._history_strength, dtype=dtype)
        if self._history_strength <= 0.0:
            return warped, gate0

        curr32 = curr.astype(mx.float32)
        prev32 = prev.astype(mx.float32)
        flow32 = flow.astype(mx.float32)
        warped_prev = flow_warp(prev32, flow32, "border")
        resid_warp = mx.mean(mx.abs(curr32 - warped_prev), axis=-1, keepdims=True)
        resid_zero = mx.mean(mx.abs(curr32 - prev32), axis=-1, keepdims=True)

        rel_benefit = mx.clip(
            (resid_zero - resid_warp) / mx.maximum(resid_zero, 2.0 / 255.0),
            0.0, 1.0,
        )
        benefit_conf = mx.clip((rel_benefit - 0.05) / 0.30, 0.0, 1.0)
        match_conf = mx.exp(-((resid_warp / 0.035) ** 2))
        confidence = benefit_conf * match_conf
        if self._history_static_cap > 0.0:
            static_conf = self._history_static_cap * mx.exp(-((resid_zero / 0.010) ** 2))
            confidence = mx.maximum(confidence, static_conf)

        risk = 1.0 - mx.clip(confidence, 0.0, 1.0)
        if self._risk is not None:
            risk = mx.maximum(risk, self._history_risk_decay * flow_warp(self._risk, flow32, "border"))
        risk = box3(mx.clip(risk, 0.0, 1.0))
        self._risk = risk.astype(mx.float32)

        if self._history_cleanup > 0.0:
            clean_strength = (self._history_cleanup * risk).astype(mx.float32)
            blurred = box3(warped.astype(mx.float32))
            warped = (
                warped.astype(mx.float32) * (1.0 - clean_strength)
                + blurred * clean_strength
            ).astype(warped.dtype)

        gate = gate0.astype(mx.float32) * mx.clip(1.0 - self._history_gate_drop * risk, 0.0, 1.0)
        return warped, gate.astype(dtype)

    def _history_policy(self, curr: Any, prev: Any, flow: Any, warped: Any, dtype: Any) -> tuple[Any, Any]:
        if self._history_gate == "holistic":
            return self._holistic_history_policy(curr, prev, flow, warped, dtype)
        return warped, self._history_gate_map(curr, prev, flow, dtype)

    def feed(self, rgb: Any, token: Any = None) -> list:
        # Reference inference reads 8-bit images as RGB [0, 1]. RGBAHalf decode can
        # carry small legal/transfer overshoots, so clamp before the learned model.
        x = mx.clip(to_rgb_batch(rgb), 0.0, 1.0)
        h, w = x.shape[1], x.shape[2]
        xp, pad_top, pad_left = self._pad4(x)
        if self._window and self._depth >= self._window:
            self.reset()
        if self._prev is None:
            sr, feat = self._first(xp)
        else:
            dt = self._feat.dtype
            if self._flow_mode == "zero":
                flow = mx.zeros((*xp.shape[:3], 2), dtype=dt)
            elif self._flow_mode == "vt":
                flow = self._vt_current_to_prev_flow(xp, self._prev, dt)
            else:
                flow = compiled_spynet_flow(self._p, xp.astype(dt), self._prev.astype(dt))
            warped = flow_warp(self._feat, flow, "zeros")
            warped, history_gate = self._history_policy(xp, self._prev, flow, warped, dt)
            sr, feat = self._next(xp, warped, history_gate)
        sy = pad_top * 4
        sx = pad_left * 4
        sr = sr[:, sy:sy + h * 4, sx:sx + w * 4, :]
        mx.eval(sr, feat)
        self._prev, self._feat = xp, feat
        self._depth += 1
        return [(sr[0], token)]

    def flush(self) -> list:
        self.reset()
        return []
