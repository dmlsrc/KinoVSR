"""Per-frame NAFNet restorer, RGB in / RGB out.

NAFNet (deblur / denoise / video variants) is a single-image residual net, so this is
a stateless per-frame stage -- restore each frame independently. Pair it after
deblock + denoise as a light detail/deblur pass. `strength` scales the predicted
residual (1.0 = full, <1 = light); since it is single-image, a strong strength can
flicker on video, so keep it light there. `pool_mode="auto"` follows the official
per-variant configs: NAFNetLocal TLSC/TLC for deblur checkpoints, plain NAFNet for
SIDD denoise. `guard_mode="auto"` protects the GoPro deblur variants from their
known out-of-domain periodic-lattice resonance. Exposes .denoise() as the harness
per-frame contract (same as FbcnnDeblocker) -- the name is the interface, not a
claim about the task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from ..compile_cache import cached as _cached
from ..vsr_blocks import box3
from . import net

_LOCAL_POOL_VARIANTS = {"gopro", "gopro32", "reds"}
_GLOBAL_POOL_VARIANTS = {"sidd", "sidd32"}
_CONTROL_GUARD_VARIANTS = {"gopro", "gopro32"}
_DEFAULT_GUARD = 0.12
_DEFAULT_FAST_FRACTION = 0.85
_CONTROL_SOURCE_COMPILE_CACHE: dict = {}
_FAST_GUARD_COMPILE_CACHE: dict = {}
_RESIDUAL_GUARD_COMPILE_CACHE: dict = {}


def model_rgb(rgb: Any) -> Any:
    """NAFNet input tensor, clipped to the training domain."""
    a = rgb if rgb.ndim == 4 else rgb[None]
    return mx.clip(a[..., :3].astype(mx.float32), 0.0, 1.0)


def _variant_token(weights: Any = None, variant: str | None = None) -> str:
    if variant:
        return variant.lower()
    if weights is None or weights == "":
        return net._DEFAULT_VARIANT
    return Path(str(weights).lower()).stem


def resolve_pool_mode(weights: Any = None, pool_mode: str = "auto", variant: str | None = None) -> str:
    """Resolve auto/local/global SCA pooling to the reference mode for a variant."""
    if pool_mode in {"local", "global"}:
        return pool_mode
    if pool_mode != "auto":
        raise ValueError(f"unknown NAFNet pool_mode {pool_mode!r}")

    token = _variant_token(weights, variant)
    if token in _LOCAL_POOL_VARIANTS or any(v in token for v in _LOCAL_POOL_VARIANTS):
        return "local"
    if token in _GLOBAL_POOL_VARIANTS or any(v in token for v in _GLOBAL_POOL_VARIANTS):
        return "global"
    raise ValueError(
        "cannot infer NAFNet pool_mode from weights; pass pool_mode='local' for "
        "NAFNetLocal/TLC deblur checkpoints or pool_mode='global' for plain NAFNet"
    )


def resolve_guard_mode(weights: Any = None, guard_mode: str = "auto", variant: str | None = None) -> str:
    """Resolve auto/off/residual/control guard behavior for a selected variant."""
    if guard_mode in {"off", "residual", "control", "control-source", "fast", "reject"}:
        return guard_mode
    if guard_mode != "auto":
        raise ValueError(f"unknown NAFNet guard_mode {guard_mode!r}")

    token = _variant_token(weights, variant)
    if token in _CONTROL_GUARD_VARIANTS or any(v in token for v in _CONTROL_GUARD_VARIANTS):
        return "reject"
    return "off"


def _blur_y3(x: Any) -> Any:
    """Replicate-padded 3x1 vertical blur, NHWC."""
    h = x.shape[1]
    xp = mx.concatenate([x[:, :1], x, x[:, -1:]], axis=1)
    return ((xp[:, :h] + xp[:, 1:h + 1] + xp[:, 2:h + 2]) / 3.0).astype(x.dtype)


def _luma(rgb: Any) -> Any:
    rgb32 = rgb.astype(mx.float32)
    return (
        rgb32[..., :1] * 0.299
        + rgb32[..., 1:2] * 0.587
        + rgb32[..., 2:3] * 0.114
    )


def luma_control_input(rgb: Any, amount: float = 1.0) -> Any:
    """Input copy with only vertical luma row energy gently smoothed.

    Chroma is preserved as RGB-minus-luma; only the luma plane is nudged toward a
    3x1 vertical blur. This is a control input for the network, not the output.
    """
    y = _luma(rgb)
    delta = (_blur_y3(y) - y) * float(amount)
    return mx.clip(rgb.astype(mx.float32) + delta, 0.0, 1.0)


def _local_mag(x: Any) -> Any:
    mag = mx.mean(mx.abs(x.astype(mx.float32)), axis=-1, keepdims=True)
    return box3(box3(mag))


def _smoothstep(x: Any, lo: float, hi: float) -> Any:
    span = max(float(hi) - float(lo), 1e-8)
    t = mx.clip((x.astype(mx.float32) - float(lo)) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def residual_guard_map(residual: Any, guard: float = _DEFAULT_GUARD) -> Any:
    """Per-pixel soft-knee attenuation map for an out-of-domain residual.

    The map is 1.0 for healthy local residual magnitudes, then quadratically
    collapses above `guard`. This is the old output-side damage limiter; the
    control guard below uses the same local magnitude as one risk signal.
    """
    if guard <= 0.0:
        return mx.ones((*residual.shape[:3], 1), dtype=residual.dtype)
    mag = _local_mag(residual)
    return mx.minimum(1.0, (float(guard) / mx.maximum(mag, 1e-8)) ** 2)


def guard_probe_map(residual: Any, guard: float = _DEFAULT_GUARD) -> Any:
    """Cheap risk probe from the normal residual alone.

    This runs before the expensive luma-control second pass. It combines a
    magnitude signal with a residual high-pass signal so low-amplitude grids can
    still trip the guard.
    """
    if guard <= 0.0:
        return mx.zeros((*residual.shape[:3], 1), dtype=residual.dtype)
    mag = _local_mag(residual)
    highpass = residual.astype(mx.float32) - box3(residual.astype(mx.float32))
    structure = _local_mag(highpass)
    risk_mag = _smoothstep(mag, guard * 0.5, guard)
    risk_structure = _smoothstep(structure, guard * 0.2, guard * 0.65)
    risk = mx.maximum(risk_mag, risk_structure)
    return mx.clip(box3(risk), 0.0, 1.0).astype(residual.dtype)


def control_risk_map(
    residual: Any,
    control_residual: Any,
    guard: float = _DEFAULT_GUARD,
) -> Any:
    """Blend risk for replacing normal residual with control residual.

    `residual` catches obvious amplitude explosions. The disagreement between
    normal and luma-control residual catches the lower-amplitude lattice body
    that can sit below a pure output-amplitude threshold.
    """
    if guard <= 0.0:
        return mx.zeros((*residual.shape[:3], 1), dtype=residual.dtype)
    mag = _local_mag(residual)
    delta = _local_mag(residual - control_residual)
    risk_mag = _smoothstep(mag, guard * 0.5, guard)
    risk_delta = _smoothstep(delta, guard * 0.25, guard * 0.85)
    return mx.clip(box3(mx.maximum(risk_mag, risk_delta)), 0.0, 1.0).astype(residual.dtype)


def _make_control_source_forward(
    p: dict,
    cfg: tuple,
    strength: float,
    pool_mode: str,
    compile: bool,
) -> Callable[[Any], Any]:
    def run(x):
        control = luma_control_input(x)
        control_out = net.nafnet(control, p, cfg=cfg, strength=strength, pool_mode=pool_mode)
        control_residual = control_out - control
        return mx.clip(x + control_residual, 0.0, 1.0)

    if not compile:
        return run
    return _cached(
        _CONTROL_SOURCE_COMPILE_CACHE,
        (id(p), float(strength), cfg, pool_mode),
        lambda: mx.compile(run),
    )


def _make_fast_guard_forward(
    p: dict,
    cfg: tuple,
    strength: float,
    pool_mode: str,
    guard: float,
    compile: bool,
) -> Callable[[Any], Any]:
    def run(x):
        out = net.nafnet(x, p, cfg=cfg, strength=strength, pool_mode=pool_mode)
        residual = out - x
        risk = guard_probe_map(residual, guard)
        safe_residual = residual * (1.0 - risk.astype(residual.dtype))
        return mx.clip(x + safe_residual, 0.0, 1.0)

    if not compile:
        return run
    return _cached(
        _FAST_GUARD_COMPILE_CACHE,
        (id(p), float(strength), cfg, pool_mode, float(guard)),
        lambda: mx.compile(run),
    )


def _make_residual_guard_forward(
    p: dict,
    cfg: tuple,
    strength: float,
    pool_mode: str,
    guard: float,
    compile: bool,
) -> Callable[[Any], Any]:
    def run(x):
        out = net.nafnet(x, p, cfg=cfg, strength=strength, pool_mode=pool_mode)
        residual = out - x
        knee = residual_guard_map(residual, guard)
        return mx.clip(x + residual * knee.astype(residual.dtype), 0.0, 1.0)

    if not compile:
        return run
    return _cached(
        _RESIDUAL_GUARD_COMPILE_CACHE,
        (id(p), float(strength), cfg, pool_mode, float(guard)),
        lambda: mx.compile(run),
    )


class NafnetRestorer:
    """Stateless per-frame RGB NAFNet restorer (deblur / denoise / video variant)."""

    def __init__(self, weights: Any = None, strength: float = 1.0,
                 pool_mode: str = "auto", variant: str | None = None,
                 guard_mode: str = "auto", residual_guard: float = _DEFAULT_GUARD,
                 guard_fast_fraction: float = _DEFAULT_FAST_FRACTION,
                 progress_message: Callable[[str], None] | None = None,
                 compile: bool = True, dtype: Any = mx.float32):   # fp16 overflows -- see net.load_params
        pool_mode = resolve_pool_mode(weights, pool_mode, variant=variant)
        guard_mode = resolve_guard_mode(weights, guard_mode, variant=variant)
        self._p = net.load_params(weights, dtype=dtype)
        self._cfg = net._config(self._p)
        self._strength = float(strength)
        self._pool_mode = pool_mode
        self._guard_mode = guard_mode
        self._guard = float(residual_guard)
        self._guard_fast_fraction = float(guard_fast_fraction)
        self._control_source_locked = False
        self._reject_locked = False
        self._guarded_frames = 0
        self._progress_message = progress_message
        self._fwd = net.make_forward(
            self._p, self._strength, self._cfg,
            compile=compile, pool_mode=pool_mode,
        )
        self._control_source_fwd = _make_control_source_forward(
            self._p, self._cfg, self._strength, self._pool_mode, compile,
        )
        self._fast_guard_fwd = _make_fast_guard_forward(
            self._p, self._cfg, self._strength, self._pool_mode, self._guard, compile,
        )
        self._residual_guard_fwd = _make_residual_guard_forward(
            self._p, self._cfg, self._strength, self._pool_mode, self._guard, compile,
        )

    def set_progress_message(self, progress_message: Callable[[str], None] | None) -> None:
        self._progress_message = progress_message

    def reset(self) -> None:
        self._control_source_locked = False
        self._reject_locked = False
        self._guarded_frames = 0

    def close(self) -> None:
        pass

    def denoise(self, rgb_f32: Any) -> Any:
        """Restore one RGB frame (H,W,3) in [0,1]; returns (H,W,3)."""
        inp = model_rgb(rgb_f32)
        if self._guard > 0.0 and self._guard_mode == "control-source":
            out = self._control_source_fwd(inp)
        elif self._guard > 0.0 and self._guard_mode == "fast":
            out = self._fast_guard_fwd(inp)
        elif self._guard > 0.0 and self._guard_mode == "residual":
            out = self._residual_guard_fwd(inp)
        elif self._guard > 0.0 and self._guard_mode == "reject":
            if self._reject_locked:
                out = inp
            else:
                out = self._apply_reject_guard(inp, self._fwd(inp))
        elif self._guard > 0.0 and self._guard_mode == "control":
            if self._control_source_locked:
                out = self._control_source_fwd(inp)
            else:
                out = self._apply_control_guard(inp, self._fwd(inp))
                out = mx.clip(out, 0.0, 1.0)
        else:
            out = mx.clip(self._fwd(inp), 0.0, 1.0)
        mx.eval(out)
        return out[0]

    def _apply_residual_guard(self, inp: Any, out: Any) -> Any:
        residual = out - inp
        local_peak = float(mx.max(_local_mag(residual)))
        if local_peak <= self._guard:
            return out
        knee = residual_guard_map(residual, self._guard)
        self._notice_once(
            "residual",
            local_peak,
            float(mx.mean((knee < 0.999).astype(mx.float32))),
        )
        return inp + residual * knee.astype(residual.dtype)

    def _apply_reject_guard(self, inp: Any, out: Any) -> Any:
        residual = out - inp
        local_peak = float(mx.max(_local_mag(residual)))
        risk = guard_probe_map(residual, self._guard)
        risk_frac = float(mx.mean((risk > 0.001).astype(mx.float32)))
        if local_peak <= self._guard and risk_frac < 0.01:
            return out
        self._reject_locked = True
        self._notice_once("reject", local_peak, risk_frac)
        return inp

    def _apply_control_guard(self, inp: Any, out: Any) -> Any:
        residual = out - inp
        local_peak = float(mx.max(_local_mag(residual)))
        if local_peak <= self._guard:
            return out

        risk_probe = guard_probe_map(residual, self._guard)
        risk_frac = float(mx.mean((risk_probe > 0.001).astype(mx.float32)))
        if 0.0 < self._guard_fast_fraction <= 1.0 and risk_frac >= self._guard_fast_fraction:
            self._control_source_locked = True
            self._notice_once("control-source", local_peak, risk_frac)
            return self._control_source_fwd(inp)

        control = luma_control_input(inp)
        control_out = self._fwd(control)
        control_residual = control_out - control
        risk = control_risk_map(residual, control_residual, self._guard)
        self._notice_once(
            "control",
            local_peak,
            risk_frac,
        )
        mixed = residual * (1.0 - risk) + control_residual * risk
        return inp + mixed.astype(residual.dtype)

    def _notice_once(self, mode: str, peak: float, frac: float) -> None:
        self._guarded_frames += 1
        if self._guarded_frames != 1:
            return
        if mode == "control":
            action = "two-pass luma control"
        elif mode == "control-source":
            action = "stable control-source residual"
        elif mode == "fast":
            action = "single-pass attenuation"
        elif mode == "reject":
            action = "rejecting NAFNet residual for this shot"
        else:
            action = "residual attenuation"
        message = (
            f"[nafnet] {mode} guard: risk {100 * frac:.1f}%, "
            f"peak {peak:.3f}>{self._guard:.3f}; {action}."
        )
        if self._progress_message is not None:
            self._progress_message(message)
        else:
            print(message)
