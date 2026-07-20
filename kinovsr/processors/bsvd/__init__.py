"""BSVD video denoiser, ported to MLX.

Architecture ported from BSVD (C. Qi et al., "Real-time Streaming Video
Denoising with Bidirectional Buffers", ACM MM 2022). The forward pass is a clean
MLX reimplementation: NHWC tensors, plain convs, ReLU6, pixel shuffle, and the
reference bidirectional buffer streaming schedule.

The public RGB checkpoint is non-blind: each input frame is RGB plus a constant
noise map. The loader also supports blind RGB-only checkpoints by inferring the
first conv's input channels from the weights.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_VARIANTS = {
    "c64": "bsvd_64.safetensors",
    "c32": "bsvd_32.safetensors",
}


def default_weights_path(variant: str = "c64") -> Path:
    """Local BSVD weights for the given variant.

    The weights are not bundled. See weights/README.md for source hashes and the
    safe conversion command.
    """
    return _WEIGHTS_DIR / _VARIANTS[variant]


def _strength_to_sigma(strength: float) -> float:
    """Map a [0,1] denoise strength to BSVD's AWGN sigma.

    The unblind training configs use noise_ival [5, 55], expressed as sigma_255.
    """
    s = max(0.0, min(1.0, float(strength)))
    return (5.0 + 50.0 * s) / 255.0


def _conv(
    w: dict[str, Any], prefix: str, subkey: str, dtype: Any, stride: int = 1
) -> tuple[Any, Any, int]:
    """Load one torch-layout conv weight as MLX conv2d's OHWI weight + bias."""
    key = f"{prefix}.{subkey}"
    W = w[f"{key}.weight"].astype(mx.float32)
    b = w.get(f"{key}.bias")
    return mx.transpose(W, (0, 2, 3, 1)).astype(dtype), (
        None if b is None else b.astype(dtype)
    ), stride


def _cv(x: Any, conv: tuple[Any, Any, int]) -> Any:
    weight, bias, stride = conv
    y = mx.conv2d(x, weight, stride=stride, padding=1)
    return y if bias is None else y + bias


def _pad_inc_gate(
    inc0: tuple[Any, Any, int], inc3: tuple[Any, Any, int]
) -> tuple[tuple[Any, Any, int], tuple[Any, Any, int]]:
    """Zero-pad the inc block's intermediate channels up to a multiple of 16.

    The c32 checkpoint uses interm_ch=30, which misses MLX's fast conv gate and
    makes the full-resolution inc block ~2.4x slower than an aligned width. Zero-
    padding inc0's output channels and inc3's input channels preserves the output
    exactly -- the extra out-channels are zero after bias + ReLU6, and the extra
    in-channels meet zero weights -- while letting inc run on an aligned width.
    A no-op when interm is already a multiple of 16 (e.g. c64's 64).
    """
    w0, b0, s0 = inc0
    interm = int(w0.shape[0])
    if interm % 16 == 0:
        return inc0, inc3
    pad = (-interm) % 16
    zeros0 = mx.zeros((pad,) + tuple(w0.shape[1:]), dtype=w0.dtype)
    w0 = mx.concatenate([w0, zeros0], axis=0)
    if b0 is not None:
        b0 = mx.concatenate([b0, mx.zeros((pad,), dtype=b0.dtype)], axis=0)
    w1, b1, s1 = inc3
    zeros1 = mx.zeros(tuple(w1.shape[:3]) + (pad,), dtype=w1.dtype)
    w1 = mx.concatenate([w1, zeros1], axis=3)
    return (w0, b0, s0), (w1, b1, s1)


def _relu6(x: Any) -> Any:
    return mx.clip(x, 0.0, 6.0)


def _cv_relu6_s1_kernel(x: Any, weight: Any, bias: Any) -> Any:
    return mx.clip(mx.conv2d(x, weight, stride=1, padding=1) + bias, 0.0, 6.0)


def _inc_kernel(x: Any, w0: Any, b0: Any, w1: Any, b1: Any) -> Any:
    x = mx.clip(mx.conv2d(x, w0, stride=1, padding=1) + b0, 0.0, 6.0)
    return mx.clip(mx.conv2d(x, w1, stride=1, padding=1) + b1, 0.0, 6.0)


def _out_kernel(x: Any, w0: Any, b0: Any, w1: Any, b1: Any) -> Any:
    x = mx.clip(mx.conv2d(x, w0, stride=1, padding=1) + b0, 0.0, 6.0)
    return mx.conv2d(x, w1, stride=1, padding=1) + b1


# Compiled lazily on first use, NOT at import: a module-level mx.compile
# initializes Metal as an import side effect, and a native fault there
# (seen once under memory pressure) aborts the whole process during test
# collection before any test runs. These kernels take weights as
# arguments, so one compiled trace per kernel serves every checkpoint.
_KERNEL_COMPILE_CACHE: dict = {}


def _kernel(name: str, make: Any) -> Any:
    from kinovsr.modeling.compile_cache import cached

    return cached(_KERNEL_COMPILE_CACHE, name, lambda: mx.compile(make))


def _cv_relu6_s1(x: Any, conv: tuple[Any, Any, int]) -> Any:
    weight, bias, stride = conv
    if stride == 1 and bias is not None:
        return _kernel("cv_relu6_s1", _cv_relu6_s1_kernel)(x, weight, bias)
    return _relu6(_cv(x, conv))


def _two_conv_ready(a: tuple[Any, Any, int], b: tuple[Any, Any, int]) -> bool:
    return a[1] is not None and b[1] is not None and a[2] == 1 and b[2] == 1


def _pixelshuffle2(x: Any) -> Any:
    """(N,H,W,4C) -> (N,2H,2W,C), matching torch PixelShuffle(2)."""
    n, h, w, c4 = x.shape
    c = c4 // 4
    x = x.reshape(n, h, w, c, 2, 2)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * 2, w * 2, c)


def _reflect_pad_to4(x: Any) -> tuple[Any, int, int]:
    """Reflect-pad NHWC x on bottom/right so H and W are multiples of 4."""
    _, h, w, _ = x.shape
    ph, pw = (-h) % 4, (-w) % 4
    if ph:
        x = mx.concatenate([x, x[:, h - 1 - ph:h - 1, :, :][:, ::-1, :, :]], axis=1)
    if pw:
        x = mx.concatenate([x, x[:, :, w - 1 - pw:w - 1, :][:, :, ::-1, :]], axis=2)
    return x, ph, pw


class _BiBufferConv:
    """Reference BiBufferConv in NHWC form.

    A call with a real tensor shifts one eighth of the future feature channels
    into the current conv input, carries one eighth from the previous center, and
    uses the current center's remaining channels. A call with None drains the
    right side with zeros, exactly like the upstream streaming end stage.
    """

    def __init__(self, conv: tuple[Any, Any, int], relu: bool = False):
        self._conv = conv
        self._relu = relu
        self.reset()

    def reset(self) -> None:
        self._left_fold_2fold: Any = None
        self._center: Any = None
        self._shape: tuple[int, int, int, int] | None = None
        self._zero_right: Any = None
        self._fold = 0

    def __call__(self, input_right: Any | None) -> Any | None:
        if input_right is not None:
            n, h, w, c = input_right.shape
            self._shape = (int(n), int(h), int(w), int(c))
            self._fold = int(c) // 8

        if self._center is None:
            self._center = input_right
            if input_right is not None and self._left_fold_2fold is None:
                n, h, w, _c = self._shape
                self._left_fold_2fold = mx.zeros((n, h, w, self._fold), dtype=input_right.dtype)
            return None

        if input_right is None:
            if self._shape is None:
                return None
            n, h, w, _c = self._shape
            if (
                self._zero_right is None
                or self._zero_right.shape != (n, h, w, self._fold)
                or self._zero_right.dtype != self._center.dtype
            ):
                self._zero_right = mx.zeros((n, h, w, self._fold), dtype=self._center.dtype)
            right = self._zero_right
        else:
            right = input_right[..., : self._fold]

        x = mx.concatenate(
            [right, self._left_fold_2fold, self._center[..., 2 * self._fold:]], axis=-1
        )
        out = _cv_relu6_s1(x, self._conv) if self._relu else _cv(x, self._conv)
        self._left_fold_2fold = self._center[..., self._fold: 2 * self._fold]
        self._center = input_right
        return out


class _MemCvBlock:
    def __init__(self, c1: tuple[Any, Any, int], c2: tuple[Any, Any, int]):
        self.c1 = _BiBufferConv(c1, relu=True)
        self.c2 = _BiBufferConv(c2, relu=True)

    def reset(self) -> None:
        self.c1.reset()
        self.c2.reset()

    def __call__(self, x: Any | None) -> Any | None:
        x = self.c1(x)
        return self.c2(x)


class _MemSkip:
    def __init__(self):
        self._items: list[Any] = []

    def reset(self) -> None:
        self._items = []

    def push(self, x: Any | None) -> None:
        if x is not None:
            self._items.insert(0, x)

    def pop(self, trigger: Any | None) -> Any | None:
        if trigger is None:
            return None
        if not self._items:
            return None
        return self._items.pop()


class _DenBlock:
    def __init__(self, p: dict[str, Any]):
        self.p = p
        self.down0 = _MemCvBlock(p["d0c1"], p["d0c2"])
        self.down1 = _MemCvBlock(p["d1c1"], p["d1c2"])
        self.up2 = _MemCvBlock(p["u2c1"], p["u2c2"])
        self.up1 = _MemCvBlock(p["u1c1"], p["u1c2"])
        self.skip1 = _MemSkip()
        self.skip2 = _MemSkip()
        self.skip3 = _MemSkip()

    def reset(self) -> None:
        self.down0.reset()
        self.down1.reset()
        self.up2.reset()
        self.up1.reset()
        self.skip1.reset()
        self.skip2.reset()
        self.skip3.reset()

    @staticmethod
    def _none_add(a: Any | None, b: Any | None) -> Any | None:
        return None if a is None or b is None else a + b

    @staticmethod
    def _none_minus(skip_rgb: Any | None, pred: Any | None) -> Any | None:
        if skip_rgb is None or pred is None:
            return None
        head = skip_rgb[..., :3] - pred[..., :3]
        return head if pred.shape[-1] == 3 else mx.concatenate([head, pred[..., 3:]], axis=-1)

    def _inc(self, x: Any | None) -> Any | None:
        if x is None:
            return None
        if _two_conv_ready(self.p["inc0"], self.p["inc3"]):
            w0, b0, _ = self.p["inc0"]
            w1, b1, _ = self.p["inc3"]
            return _kernel("inc", _inc_kernel)(x, w0, b0, w1, b1)
        return _relu6(_cv(_relu6(_cv(x, self.p["inc0"])), self.p["inc3"]))

    def _down(self, x: Any | None, conv0: tuple[Any, Any, int], mem: _MemCvBlock) -> Any | None:
        if x is not None:
            x = _relu6(_cv(x, conv0))
        return mem(x)

    def _up(self, x: Any | None, conv: tuple[Any, Any, int], mem: _MemCvBlock) -> Any | None:
        x = mem(x)
        if x is None:
            return None
        return _pixelshuffle2(_cv(x, conv))

    def _out(self, x: Any | None) -> Any | None:
        if x is None:
            return None
        if _two_conv_ready(self.p["out0"], self.p["out3"]):
            w0, b0, _ = self.p["out0"]
            w1, b1, _ = self.p["out3"]
            return _kernel("out", _out_kernel)(x, w0, b0, w1, b1)
        return _cv(_relu6(_cv(x, self.p["out0"])), self.p["out3"])

    def __call__(self, x: Any | None) -> Any | None:
        self.skip1.push(None if x is None else x[..., :3])
        x0 = self._inc(x)
        self.skip2.push(x0)
        x1 = self._down(x0, self.p["d0"], self.down0)
        self.skip3.push(x1)
        x2 = self._down(x1, self.p["d1"], self.down1)
        x2 = self._up(x2, self.p["u2"], self.up2)
        x1 = self._up(self._none_add(x2, self.skip3.pop(x2)), self.p["u1"], self.up1)
        x = self._out(self._none_add(x1, self.skip2.pop(x1)))
        return self._none_minus(self.skip1.pop(x), x)


def _block_prefix(w: dict[str, Any], index: int) -> str:
    for p in (f"base_model.nets_list.{index}", f"module.base_model.nets_list.{index}"):
        if f"{p}.inc.convblock.0.weight" in w:
            return p
    raise KeyError(f"could not find BSVD nets_list.{index} weights")


def _load_block(w: dict[str, Any], index: int, dtype: Any) -> dict[str, Any]:
    prefix = _block_prefix(w, index)
    p = {
        "inc0": _conv(w, prefix, "inc.convblock.0", dtype),
        "inc3": _conv(w, prefix, "inc.convblock.3", dtype),
        "d0": _conv(w, prefix, "downc0.convblock.0", dtype, stride=2),
        "d0c1": _conv(w, prefix, "downc0.convblock.3.c1.net", dtype),
        "d0c2": _conv(w, prefix, "downc0.convblock.3.c2.net", dtype),
        "d1": _conv(w, prefix, "downc1.convblock.0", dtype, stride=2),
        "d1c1": _conv(w, prefix, "downc1.convblock.3.c1.net", dtype),
        "d1c2": _conv(w, prefix, "downc1.convblock.3.c2.net", dtype),
        "u2c1": _conv(w, prefix, "upc2.convblock.0.c1.net", dtype),
        "u2c2": _conv(w, prefix, "upc2.convblock.0.c2.net", dtype),
        "u2": _conv(w, prefix, "upc2.convblock.1", dtype),
        "u1c1": _conv(w, prefix, "upc1.convblock.0.c1.net", dtype),
        "u1c2": _conv(w, prefix, "upc1.convblock.0.c2.net", dtype),
        "u1": _conv(w, prefix, "upc1.convblock.1", dtype),
        "out0": _conv(w, prefix, "outc.convblock.0", dtype),
        "out3": _conv(w, prefix, "outc.convblock.3", dtype),
    }
    p["inc0"], p["inc3"] = _pad_inc_gate(p["inc0"], p["inc3"])
    return p


def load_bsvd(path: str | Path, dtype: Any = mx.float16) -> tuple[dict[str, Any], int]:
    """Load BSVD weights and infer whether the first block is RGB or RGB+sigma."""
    wp = Path(path)
    if wp.suffix in {".pth", ".pt"}:
        raise ValueError(
            f"BSVD weights must be .safetensors, got {wp}. Convert with "
            "kinovsr weights convert --param-key params first."
        )
    w = mx.load(str(wp))
    p0 = _block_prefix(w, 0)
    input_channels = int(w[f"{p0}.inc.convblock.0.weight"].shape[1])
    if input_channels not in (3, 4):
        raise ValueError(
            f"BSVD first conv expects {input_channels} channels; expected 3 or 4."
        )
    return {"temp1": _load_block(w, 0, dtype), "temp2": _load_block(w, 1, dtype)}, input_channels


class BSVD:
    """Stateful BSVD network. Call step(frame_or_none) once per stream item."""

    SHIFT_NUM = 16

    def __init__(self, weights_path: str | Path, dtype: Any = mx.float16):
        self.dtype = dtype
        self.params, self.input_channels = load_bsvd(weights_path, dtype=dtype)
        self.temp1 = _DenBlock(self.params["temp1"])
        self.temp2 = _DenBlock(self.params["temp2"])

    def reset(self) -> None:
        self.temp1.reset()
        self.temp2.reset()

    def step(self, x: Any | None) -> Any | None:
        return self.temp2(self.temp1(x))


class BsvdDenoiser:
    """Streaming BSVD denoiser for harness preprocess chains.

    feed(frame, token) and flush() return lists of (denoised_frame, token). The
    network has a 16-step bidirectional-buffer delay; the first 16 intermediate
    outputs are discarded, then outputs are paired with the oldest input token.
    """

    MAP_WARMUP = 9   # frames buffered before estimating a spatial noise map

    def __init__(
        self,
        weights_path: str | Path | None = None,
        strength: float = 0.5,
        variant: str = "c64",
        dtype: Any = mx.float16,
        noise_map: Any | None = None,
        map_refresh: int = 64,
        pulse: Any | None = None,
        map_floor: float = 0.0,
        backend: str = "mlx",
    ):
        if backend not in ("mlx", "ane"):
            raise ValueError(f"unknown BSVD backend {backend!r}; expected 'mlx' or 'ane'")
        wp = Path(weights_path) if weights_path else default_weights_path(variant)
        if not wp.is_file():
            raise FileNotFoundError(
                f"BSVD weights not found at {wp}. They are not bundled; convert the "
                "source .pth with kinovsr weights convert or pass --bsvd-weights."
            )
        if backend == "ane":
            # Explicitly requested; construction and first-step failures
            # raise rather than silently running a different backend.
            from .ane import AneBSVD

            self.net: Any = AneBSVD(wp, dtype=dtype)
        else:
            self.net = BSVD(wp, dtype=dtype)
        self.sigma = _strength_to_sigma(strength)
        # optional NoiseMapTracker: replaces the constant sigma plane with a
        # per-pixel estimate (sigma units, same scale as the constant).
        self._tracker = noise_map
        # optional PulseGain: per-frame scalar on the sigma plane tracking
        # GOP-phase noise pulsing (I-frame grain refresh).
        self._pulse = pulse
        if (self._tracker is not None or self._pulse is not None) \
                and self.net.input_channels != 4:
            raise ValueError(
                "--noise-map / --noise-map-pulse need a non-blind (4-channel) "
                "BSVD checkpoint; this one is blind RGB-only."
            )
        # streaming-mode map refresh cadence (frames); 0 disables. Scheduled
        # (gop-aligned) mode re-estimates per window instead and ignores this.
        self._map_refresh = max(0, int(map_refresh))
        # user sigma floor under the map: the temporal estimator only measures
        # FLICKERING noise, so static grain / structured junk reads near zero
        # even though conditioning the net higher visibly cleans it. The floor
        # guarantees a base denoise level (the manual dial's role) with the
        # map's spatial and pulse adaptation applying above it.
        self._map_floor = max(0.0, float(map_floor))
        self.last_noise_map: Any = None   # fp32 (H,W,1) actually used (debug)
        # diagnostics: gains across the whole run (survives the end-of-stream reset)
        self._pulse_log: list[float] = []
        self._schedule: list | None = None
        self._reset_state()

    def _reset_conditioning(self, clear_debug: bool = False) -> None:
        if self._tracker is not None and hasattr(self._tracker, "reset"):
            self._tracker.reset()
        if clear_debug:
            self.last_noise_map = None

    def _reset_state(self) -> None:
        self.net.reset()
        self._hw: tuple[int, int] | None = None
        self._nm: Any = None
        self._padded_hw: tuple[int, int] | None = None
        self._tokens: list[Any] = []
        self._frames: list[Any] = []
        self._frame_tokens: list[Any] = []
        self._warm: list = []    # (frame3ch, token, gain) held until the map is estimated
        self._recent: list = []  # rolling last frames for the streaming map refresh
        self._since_refresh = 0
        if self._pulse is not None:
            self._pulse.reset()
        self._base = 0
        self._sched_i = 0
        self._received = 0
        self._emitted = 0
        self._steps = 0

    def reset(self) -> None:
        self._reset_state()
        self._reset_conditioning(clear_debug=True)

    def set_schedule(self, schedule: list | None) -> None:
        """Use GOP-aligned windows instead of one continuous stream.

        The schedule is a list of (proc_start, proc_end, emit_start, emit_end)
        specs whose emit ranges tile the stream. Default BSVD remains the
        reference-like continuous stream; schedule mode resets BSVD per proc
        window and emits only that window's requested output range.
        """
        self._schedule = list(schedule) if schedule else None
        self._sched_i = 0

    def run_diagnostics(self) -> list:
        from kinovsr.processors.conditioning import noise_map_diagnostics

        return noise_map_diagnostics(self)

    def debug_images(self) -> dict:
        from kinovsr.processors.conditioning import noise_map_debug_image

        return noise_map_debug_image(self)

    def close(self) -> None:
        net, self.net = self.net, None
        try:
            close = getattr(net, "close", None)
            if callable(close):
                close()
        finally:
            # Release delayed frames/tokens and conditioning tensors even if
            # the backend reports an in-flight prediction failure while it
            # shuts down.  FeedFlushProcessor has already captured diagnostics
            # before calling this lifecycle edge.
            self._nm = None
            self._tokens = []
            self._frames = []
            self._frame_tokens = []
            self._warm = []
            self._recent = []

    def _prepare(self, frame: Any) -> Any:
        """Clip + pad one frame to a 3-channel net-dtype tensor (no noise map yet;
        the map channel is concatenated at step time so a spatial estimate made
        from the first frames can apply to those same frames)."""
        frame = mx.clip(frame[..., :3].astype(mx.float32), 0.0, 1.0)
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if self._hw is None:
            self._hw = (h, w)
        elif self._hw != (h, w):
            raise ValueError(f"BSVD stream changed resolution from {self._hw} to {(h, w)}")
        return _reflect_pad_to4(frame[None].astype(self.net.dtype))[0]

    def _plane_from_map(self, sig_map: Any) -> Any:
        """(H,W,1) sigma map -> (1,hp,wp,1) net-dtype plane (reflect-padded)."""
        self.last_noise_map = sig_map.astype(mx.float32)
        return _reflect_pad_to4(sig_map[None].astype(self.net.dtype))[0]

    def _ensure_nm(self, x: Any) -> None:
        """Make sure self._nm exists for this padded size (constant-sigma path)."""
        _, hp, wp, _ = x.shape
        if self._nm is None or self._padded_hw != (int(hp), int(wp)):
            self._nm = mx.full((1, hp, wp, 1), float(self.sigma), dtype=self.net.dtype)
            self._padded_hw = (int(hp), int(wp))

    # the unblind checkpoints were trained with noise_ival [5, 55]: sigma
    # conditioning outside that dial is out of distribution (below the floor the
    # net no-ops, above the ceiling it over-smooths), so the plane is clamped
    # into it after the map gain and pulse gain -- the same range the manual
    # --denoise-strength dial spans.
    SIGMA_MIN = 5.0 / 255.0
    SIGMA_MAX = 55.0 / 255.0

    def _with_nm(self, x: Any, nm: Any | None = None, gain: float = 1.0) -> Any:
        if self.net.input_channels != 4:
            return x
        if nm is None:
            if self._nm is None:
                self._ensure_nm(x)
            nm = self._nm
        if gain != 1.0:
            nm = nm * gain
        if self._tracker is not None or self._pulse is not None:
            nm = mx.clip(nm, max(self.SIGMA_MIN, self._map_floor), self.SIGMA_MAX)
        return mx.concatenate([x, nm], axis=-1)

    def _pulse_gain(self, x3: Any, new_segment: bool = False,
                    since_sync: int | None = None) -> float:
        """Per-frame pulse gain from the cropped frame (1.0 when pulse is off)."""
        if self._pulse is None:
            return 1.0
        g = self._pulse.update(self._crop(x3), new_segment=new_segment,
                               since_sync=since_sync)
        self._pulse_log.append(g)
        return g

    def _crop(self, x: Any) -> Any:
        h, w = self._hw
        return x[:, :h, :w, :]

    def _estimate_from(self, frames3: list) -> None:
        """Estimate the map from padded 3ch frames; fall back to the constant
        sigma when the tracker cannot estimate (too few frames)."""
        sig = self._tracker.update([self._crop(f) for f in frames3])
        if sig is None:
            h, w = self._hw
            sig = mx.full((h, w, 1), float(self.sigma), dtype=mx.float32)
        plane = self._plane_from_map(sig)
        self._nm = plane
        self._padded_hw = (int(plane.shape[1]), int(plane.shape[2]))

    def _drain_warm(self) -> list:
        out: list = []
        for x, tok, gain in self._warm:
            out += self._step(x, token=tok, real=True, gain=gain)
        self._warm = []
        return out

    def _emit(self, out: Any, token: Any) -> tuple[Any, Any]:
        if self._hw is None:
            raise RuntimeError("BSVD emitted before any input frame")
        h, w = self._hw
        out = mx.clip(out, 0.0, 1.0)[0, :h, :w, :3].astype(mx.float32)
        mx.eval(out)
        return out, token

    def _step(self, x: Any | None, token: Any = None, real: bool = False,
              gain: float = 1.0) -> list:
        if real:
            self._tokens.append(token)
            self._received += 1
        out = self.net.step(None if x is None else self._with_nm(x, gain=gain))
        self._steps += 1
        if self._steps <= self.net.SHIFT_NUM or out is None or self._emitted >= self._received:
            return []
        tok = self._tokens.pop(0)
        self._emitted += 1
        return [self._emit(out, tok)]

    def feed(self, frame: Any, token: Any = None) -> list:
        from kinovsr.analysis.noise.track import source_since_sync

        x = self._prepare(frame)
        if self._schedule is not None:
            self._frames.append(x)
            self._frame_tokens.append(token)
            return self._feed_scheduled(final=False)
        gain = self._pulse_gain(x, since_sync=source_since_sync(token))
        if self._tracker is not None and self._nm is None:
            # hold the first frames, estimate the spatial map from them, then
            # drain them through the net with that map attached.
            self._warm.append((x, token, gain))
            if len(self._warm) >= self.MAP_WARMUP:
                self._estimate_from([f for f, _, _ in self._warm])
                self._recent = [f for f, _, _ in self._warm]
                return self._drain_warm()
            return []
        if self._tracker is not None and self._map_refresh > 0:
            # periodic streaming refresh: re-estimate from a rolling buffer of
            # recent frames; the tracker's EMA keeps the transition gradual.
            self._recent.append(x)
            if len(self._recent) > self.MAP_WARMUP:
                self._recent.pop(0)
            self._since_refresh += 1
            if self._since_refresh >= self._map_refresh and len(self._recent) >= 2:
                self._estimate_from(self._recent)
                self._since_refresh = 0
        return self._step(x, token=token, real=True, gain=gain)

    def flush(self) -> list:
        if self._schedule is not None:
            out = self._feed_scheduled(final=True)
            self._reset_state()
            self._reset_conditioning(clear_debug=False)
            return out
        out = []
        if self._warm:
            # short stream ended before the map warmup filled: estimate from
            # whatever arrived (the tracker falls back to constant below 2 frames)
            self._estimate_from([f for f, _, _ in self._warm])
            out += self._drain_warm()
        guard = self.net.SHIFT_NUM + self._received + 2
        while self._emitted < self._received:
            before = self._emitted
            out += self._step(None)
            guard -= 1
            if guard <= 0 and self._emitted == before:
                raise RuntimeError("BSVD flush did not produce enough delayed frames")
        self._reset_state()
        self._reset_conditioning(clear_debug=False)
        return out

    def _feed_scheduled(self, final: bool) -> list:
        out: list = []
        total = self._base + len(self._frames)
        while self._sched_i < len(self._schedule):
            p0, p1, e0, e1 = self._schedule[self._sched_i]
            if not final and total < p1:
                break
            pe, ee = (min(p1, total), min(e1, total)) if final else (p1, e1)
            if p0 < pe and e0 < ee:
                rel_p0 = p0 - self._base
                rel_pe = pe - self._base
                local_frames = self._frames[rel_p0:rel_pe]
                local_tokens = self._frame_tokens[rel_p0:rel_pe]
                out.extend(self._run_window(local_frames, local_tokens, e0 - p0, ee - p0))
            self._sched_i += 1
        keep_from = (self._schedule[self._sched_i][0] if self._sched_i < len(self._schedule)
                     else self._base + len(self._frames))
        drop = keep_from - self._base
        if drop > 0:
            self._frames = self._frames[drop:]
            self._frame_tokens = self._frame_tokens[drop:]
            self._base += drop
        return out

    def _run_window(self, frames: list, tokens: list, emit_start: int, emit_end: int) -> list:
        self.net.reset()
        nm = None
        if self._tracker is not None:
            # per-window estimate, EMA-blended across windows by the tracker so
            # the conditioning does not pump at gop-aligned window boundaries.
            sig = self._tracker.update([self._crop(f) for f in frames])
            if sig is not None:
                nm = self._plane_from_map(sig)
        from kinovsr.analysis.noise.track import source_since_sync

        conditioned = []
        for i, x in enumerate(frames):
            # window starts break temporal adjacency (proc ranges overlap), so
            # the pulse tracker restarts its diff chain at each window.
            gain = self._pulse_gain(
                x, new_segment=(i == 0),
                since_sync=source_since_sync(
                    tokens[i] if i < len(tokens) else None))
            conditioned.append(self._with_nm(x, nm, gain=gain))

        run_window = getattr(self.net, "run_window", None)
        if callable(run_window) and len(conditioned) >= 16:
            window_outputs = run_window(conditioned)
            out = [self._emit(window_outputs[index], tokens[index])
                   for index in range(emit_start, emit_end)]
            self.net.reset()
            return out

        out = []
        for i, x in enumerate(conditioned):
            y = self.net.step(x)
            idx = i - self.net.SHIFT_NUM
            if emit_start <= idx < emit_end:
                if y is None:
                    raise RuntimeError("BSVD scheduled window emitted an empty frame")
                out.append(self._emit(y, tokens[idx]))
        for i in range(self.net.SHIFT_NUM):
            y = self.net.step(None)
            idx = len(frames) + i - self.net.SHIFT_NUM
            if emit_start <= idx < emit_end:
                if y is None:
                    raise RuntimeError("BSVD scheduled window emitted an empty frame")
                out.append(self._emit(y, tokens[idx]))
        self.net.reset()
        return out


__all__ = ["BSVD", "BsvdDenoiser", "default_weights_path", "load_bsvd"]
