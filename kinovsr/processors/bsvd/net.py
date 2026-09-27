"""The BSVD network, ported to MLX.

Architecture from BSVD (C. Qi et al., "Real-time Streaming Video Denoising
with Bidirectional Buffers", ACM MM 2022): NHWC tensors, plain convs, ReLU6,
pixel shuffle, and the reference bidirectional buffer streaming schedule. The
checkpoint loader, the blocks, and the MLX forward live here; the family's
denoiser and its Neural Engine and MPSGraph backends import them.
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinovsr.modeling.compile_cache import cached

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
    w: dict[str, mx.array], prefix: str, subkey: str, dtype: mx.Dtype, stride: int = 1
) -> tuple[mx.array, mx.array | None, int]:
    """Load one torch-layout conv weight as MLX conv2d's OHWI weight + bias."""
    key = f"{prefix}.{subkey}"
    W = w[f"{key}.weight"].astype(mx.float32)
    b = w.get(f"{key}.bias")
    return (
        mx.transpose(W, (0, 2, 3, 1)).astype(dtype),
        (None if b is None else b.astype(dtype)),
        stride,
    )


def _cv(x: mx.array, conv: tuple[mx.array, mx.array | None, int]) -> mx.array:
    weight, bias, stride = conv
    y = mx.conv2d(x, weight, stride=stride, padding=1)
    return y if bias is None else y + bias


def _pad_inc_gate(
    inc0: tuple[mx.array, mx.array | None, int], inc3: tuple[mx.array, mx.array | None, int]
) -> tuple[tuple[mx.array, mx.array | None, int], tuple[mx.array, mx.array | None, int]]:
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
    zeros0 = mx.zeros((pad, *tuple(w0.shape[1:])), dtype=w0.dtype)
    w0 = mx.concatenate([w0, zeros0], axis=0)
    if b0 is not None:
        b0 = mx.concatenate([b0, mx.zeros((pad,), dtype=b0.dtype)], axis=0)
    w1, b1, s1 = inc3
    zeros1 = mx.zeros((*tuple(w1.shape[:3]), pad), dtype=w1.dtype)
    w1 = mx.concatenate([w1, zeros1], axis=3)
    return (w0, b0, s0), (w1, b1, s1)


def _relu6(x: mx.array) -> mx.array:
    return mx.clip(x, 0.0, 6.0)


def _cv_relu6_s1_kernel(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    return mx.clip(mx.conv2d(x, weight, stride=1, padding=1) + bias, 0.0, 6.0)


def _inc_kernel(x: mx.array, w0: mx.array, b0: mx.array, w1: mx.array, b1: mx.array) -> mx.array:
    x = mx.clip(mx.conv2d(x, w0, stride=1, padding=1) + b0, 0.0, 6.0)
    return mx.clip(mx.conv2d(x, w1, stride=1, padding=1) + b1, 0.0, 6.0)


def _out_kernel(x: mx.array, w0: mx.array, b0: mx.array, w1: mx.array, b1: mx.array) -> mx.array:
    x = mx.clip(mx.conv2d(x, w0, stride=1, padding=1) + b0, 0.0, 6.0)
    return mx.conv2d(x, w1, stride=1, padding=1) + b1


# Compiled lazily on first use, NOT at import: a module-level mx.compile
# initializes Metal as an import side effect, and a native fault there
# (seen once under memory pressure) aborts the whole process during test
# collection before any test runs. These kernels take weights as
# arguments, so one compiled trace per kernel serves every checkpoint.
_KERNEL_COMPILE_CACHE: dict[str, object] = {}


def _kernel[**P](name: str, make: Callable[P, mx.array]) -> Callable[P, mx.array]:

    return cast(  # each name is always compiled from the same kernel function
        Callable[P, mx.array], cached(_KERNEL_COMPILE_CACHE, name, lambda: mx.compile(make))
    )


def _cv_relu6_s1(x: mx.array, conv: tuple[mx.array, mx.array | None, int]) -> mx.array:
    weight, bias, stride = conv
    if stride == 1 and bias is not None:
        return _kernel("cv_relu6_s1", _cv_relu6_s1_kernel)(x, weight, bias)
    return _relu6(_cv(x, conv))


def _two_conv_ready(
    a: tuple[mx.array, mx.array | None, int], b: tuple[mx.array, mx.array | None, int]
) -> bool:
    return a[1] is not None and b[1] is not None and a[2] == 1 and b[2] == 1


def _pixelshuffle2(x: mx.array) -> mx.array:
    """(N,H,W,4C) -> (N,2H,2W,C), matching torch PixelShuffle(2)."""
    n, h, w, c4 = x.shape
    c = c4 // 4
    x = x.reshape(n, h, w, c, 2, 2)
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(n, h * 2, w * 2, c)


def _reflect_pad_to4(x: mx.array) -> tuple[mx.array, int, int]:
    """Reflect-pad NHWC x on bottom/right so H and W are multiples of 4."""
    _, h, w, _ = x.shape
    ph, pw = (-h) % 4, (-w) % 4
    if ph:
        x = mx.concatenate([x, x[:, h - 1 - ph : h - 1, :, :][:, ::-1, :, :]], axis=1)
    if pw:
        x = mx.concatenate([x, x[:, :, w - 1 - pw : w - 1, :][:, :, ::-1, :]], axis=2)
    return x, ph, pw


def _pad_width_reflect(x: mx.array, target: int) -> mx.array:
    """Reflect-pad NHWC ``x`` on the right to ``target`` columns."""
    width = int(x.shape[2])
    pad = target - width
    if pad == 0:
        return x
    mirror = x[:, :, width - 1 - pad : width - 1, :][:, :, ::-1, :]
    return mx.concatenate([x, mirror], axis=2)


class _BiBufferConv:
    """Reference BiBufferConv in NHWC form.

    A call with a real tensor shifts one eighth of the future feature channels
    into the current conv input, carries one eighth from the previous center, and
    uses the current center's remaining channels. A call with None drains the
    right side with zeros, exactly like the upstream streaming end stage.
    """

    def __init__(self, conv: tuple[mx.array, mx.array | None, int], relu: bool = False):
        self._conv = conv
        self._relu = relu
        self._zero: mx.array | None = None
        self.reset()

    def reset(self) -> None:
        self._left_fold_2fold: mx.array | None = None
        self._center: mx.array | None = None
        self._shape: tuple[int, int, int, int] | None = None
        self._fold = 0

    def __call__(self, input_right: mx.array | None) -> mx.array | None:
        if input_right is not None:
            n, h, w, c = input_right.shape
            self._shape = (int(n), int(h), int(w), int(c))
            self._fold = int(c) // 8

        if self._center is None:
            self._center = input_right
            if input_right is not None and self._left_fold_2fold is None:
                assert self._shape is not None
                n, h, w, _c = self._shape
                zero_shape = (n, h, w, self._fold)
                if (
                    self._zero is None
                    or self._zero.shape != zero_shape
                    or self._zero.dtype != input_right.dtype
                ):
                    self._zero = mx.zeros(zero_shape, dtype=input_right.dtype)
                self._left_fold_2fold = self._zero
            return None

        if input_right is None:
            if self._shape is None:
                return None
            right = self._zero
        else:
            right = input_right[..., : self._fold]

        assert right is not None
        assert self._left_fold_2fold is not None
        x = mx.concatenate(
            [right, self._left_fold_2fold, self._center[..., 2 * self._fold :]], axis=-1
        )
        out = _cv_relu6_s1(x, self._conv) if self._relu else _cv(x, self._conv)
        self._left_fold_2fold = self._center[..., self._fold : 2 * self._fold]
        self._center = input_right
        return out


class _MemCvBlock:
    def __init__(
        self, c1: tuple[mx.array, mx.array | None, int], c2: tuple[mx.array, mx.array | None, int]
    ):
        self.c1 = _BiBufferConv(c1, relu=True)
        self.c2 = _BiBufferConv(c2, relu=True)

    def reset(self) -> None:
        self.c1.reset()
        self.c2.reset()

    def __call__(self, x: mx.array | None) -> mx.array | None:
        x = self.c1(x)
        return self.c2(x)


class _MemSkip:
    def __init__(self) -> None:
        self._items: list[mx.array] = []

    def reset(self) -> None:
        self._items = []

    def push(self, x: mx.array | None) -> None:
        if x is not None:
            self._items.insert(0, x)

    def pop(self, trigger: mx.array | None) -> mx.array | None:
        if trigger is None:
            return None
        if not self._items:
            return None
        return self._items.pop()


class _DenBlock:
    def __init__(self, p: dict[str, tuple[mx.array, mx.array | None, int]]):
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
    def _none_add(a: mx.array | None, b: mx.array | None) -> mx.array | None:
        return None if a is None or b is None else a + b

    @staticmethod
    def _none_minus(skip_rgb: mx.array | None, pred: mx.array | None) -> mx.array | None:
        if skip_rgb is None or pred is None:
            return None
        head = skip_rgb[..., :3] - pred[..., :3]
        return head if pred.shape[-1] == 3 else mx.concatenate([head, pred[..., 3:]], axis=-1)

    def _inc(self, x: mx.array | None) -> mx.array | None:
        if x is None:
            return None
        if _two_conv_ready(self.p["inc0"], self.p["inc3"]):
            w0, b0, _ = self.p["inc0"]
            w1, b1, _ = self.p["inc3"]
            assert b0 is not None  # _two_conv_ready checked both biases
            assert b1 is not None  # _two_conv_ready checked both biases
            return _kernel("inc", _inc_kernel)(x, w0, b0, w1, b1)
        return _relu6(_cv(_relu6(_cv(x, self.p["inc0"])), self.p["inc3"]))

    def _down(
        self, x: mx.array | None, conv0: tuple[mx.array, mx.array | None, int], mem: _MemCvBlock
    ) -> mx.array | None:
        if x is not None:
            x = _relu6(_cv(x, conv0))
        return mem(x)

    def _up(
        self, x: mx.array | None, conv: tuple[mx.array, mx.array | None, int], mem: _MemCvBlock
    ) -> mx.array | None:
        x = mem(x)
        if x is None:
            return None
        return _pixelshuffle2(_cv(x, conv))

    def _out(self, x: mx.array | None) -> mx.array | None:
        if x is None:
            return None
        if _two_conv_ready(self.p["out0"], self.p["out3"]):
            w0, b0, _ = self.p["out0"]
            w1, b1, _ = self.p["out3"]
            assert b0 is not None  # _two_conv_ready checked both biases
            assert b1 is not None  # _two_conv_ready checked both biases
            return _kernel("out", _out_kernel)(x, w0, b0, w1, b1)
        return _cv(_relu6(_cv(x, self.p["out0"])), self.p["out3"])

    def __call__(self, x: mx.array | None) -> mx.array | None:
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


def _block_prefix(w: dict[str, mx.array], index: int) -> str:
    for p in (f"base_model.nets_list.{index}", f"module.base_model.nets_list.{index}"):
        if f"{p}.inc.convblock.0.weight" in w:
            return p
    raise KeyError(f"could not find BSVD nets_list.{index} weights")


def _load_block(
    w: dict[str, mx.array], index: int, dtype: mx.Dtype
) -> dict[str, tuple[mx.array, mx.array | None, int]]:
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


def load_bsvd(
    path: str | Path, dtype: mx.Dtype = mx.float16
) -> tuple[dict[str, dict[str, tuple[mx.array, mx.array | None, int]]], int]:
    """Load BSVD weights and infer whether the first block is RGB or RGB+sigma."""
    wp = Path(path)
    if wp.suffix in {".pth", ".pt"}:
        raise ValueError(
            f"BSVD weights must be .safetensors, got {wp}. Convert with "
            "kinovsr weights convert --param-key params first."
        )
    w = cast(dict[str, mx.array], mx.load(str(wp)))
    p0 = _block_prefix(w, 0)
    input_channels = int(w[f"{p0}.inc.convblock.0.weight"].shape[1])
    if input_channels not in (3, 4):
        raise ValueError(f"BSVD first conv expects {input_channels} channels; expected 3 or 4.")
    return {"temp1": _load_block(w, 0, dtype), "temp2": _load_block(w, 1, dtype)}, input_channels


class BSVD:
    """Stateful BSVD network. Call step(frame_or_none) once per stream item."""

    SHIFT_NUM = 16

    def __init__(self, weights_path: str | Path, dtype: mx.Dtype = mx.float16):
        self.dtype = dtype
        self.params, self.input_channels = load_bsvd(weights_path, dtype=dtype)
        self.temp1 = _DenBlock(self.params["temp1"])
        self.temp2 = _DenBlock(self.params["temp2"])

    def reset(self) -> None:
        self.temp1.reset()
        self.temp2.reset()

    def step(self, x: mx.array | None) -> mx.array | None:
        return self.temp2(self.temp1(x))
