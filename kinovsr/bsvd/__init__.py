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


def _relu6(x: Any) -> Any:
    return mx.clip(x, 0.0, 6.0)


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

    def __init__(self, conv: tuple[Any, Any, int]):
        self._conv = conv
        self.reset()

    def reset(self) -> None:
        self._left_fold_2fold: Any = None
        self._center: Any = None
        self._shape: tuple[int, int, int, int] | None = None
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
            right = mx.zeros((n, h, w, self._fold), dtype=self._center.dtype)
        else:
            right = input_right[..., : self._fold]

        x = mx.concatenate(
            [right, self._left_fold_2fold, self._center[..., 2 * self._fold:]], axis=-1
        )
        out = _cv(x, self._conv)
        self._left_fold_2fold = self._center[..., self._fold: 2 * self._fold]
        self._center = input_right
        return out


class _MemCvBlock:
    def __init__(self, c1: tuple[Any, Any, int], c2: tuple[Any, Any, int]):
        self.c1 = _BiBufferConv(c1)
        self.c2 = _BiBufferConv(c2)

    def reset(self) -> None:
        self.c1.reset()
        self.c2.reset()

    def __call__(self, x: Any | None) -> Any | None:
        x = self.c1(x)
        if x is not None:
            x = _relu6(x)
        x = self.c2(x)
        if x is not None:
            x = _relu6(x)
        return x


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
    return p


def load_bsvd(path: str | Path, dtype: Any = mx.float16) -> tuple[dict[str, Any], int]:
    """Load BSVD weights and infer whether the first block is RGB or RGB+sigma."""
    wp = Path(path)
    if wp.suffix in {".pth", ".pt"}:
        raise ValueError(
            f"BSVD weights must be .safetensors, got {wp}. Convert with "
            "scripts/pth_to_safetensors.py --param-key params first."
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

    def __init__(
        self,
        weights_path: str | Path | None = None,
        strength: float = 0.5,
        variant: str = "c64",
        dtype: Any = mx.float16,
    ):
        wp = Path(weights_path) if weights_path else default_weights_path(variant)
        if not wp.is_file():
            raise FileNotFoundError(
                f"BSVD weights not found at {wp}. They are not bundled; convert the "
                "source .pth with scripts/pth_to_safetensors.py or pass --bsvd-weights."
            )
        self.net = BSVD(wp, dtype=dtype)
        self.sigma = _strength_to_sigma(strength)
        self._schedule: list | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.net.reset()
        self._hw: tuple[int, int] | None = None
        self._tokens: list[Any] = []
        self._frames: list[Any] = []
        self._frame_tokens: list[Any] = []
        self._base = 0
        self._sched_i = 0
        self._received = 0
        self._emitted = 0
        self._steps = 0

    def reset(self) -> None:
        self._reset_state()

    def set_schedule(self, schedule: list | None) -> None:
        """Use GOP-aligned windows instead of one continuous stream.

        The schedule is a list of (proc_start, proc_end, emit_start, emit_end)
        specs whose emit ranges tile the stream. Default BSVD remains the
        reference-like continuous stream; schedule mode resets BSVD per proc
        window and emits only that window's requested output range.
        """
        self._schedule = list(schedule) if schedule else None
        self._sched_i = 0

    def close(self) -> None:
        pass

    def _prepare(self, frame: Any) -> Any:
        frame = mx.clip(frame[..., :3].astype(mx.float32), 0.0, 1.0)
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if self._hw is None:
            self._hw = (h, w)
        elif self._hw != (h, w):
            raise ValueError(f"BSVD stream changed resolution from {self._hw} to {(h, w)}")
        x = _reflect_pad_to4(frame[None].astype(self.net.dtype))[0]
        if self.net.input_channels == 4:
            _, hp, wp, _ = x.shape
            nm = mx.full((1, hp, wp, 1), float(self.sigma), dtype=self.net.dtype)
            x = mx.concatenate([x, nm], axis=-1)
        return x

    def _emit(self, out: Any, token: Any) -> tuple[Any, Any]:
        if self._hw is None:
            raise RuntimeError("BSVD emitted before any input frame")
        h, w = self._hw
        out = mx.clip(out, 0.0, 1.0)[0, :h, :w, :3].astype(mx.float32)
        mx.eval(out)
        return out, token

    def _step(self, x: Any | None, token: Any = None, real: bool = False) -> list:
        if real:
            self._tokens.append(token)
            self._received += 1
        out = self.net.step(x)
        self._steps += 1
        if self._steps <= self.net.SHIFT_NUM or out is None or self._emitted >= self._received:
            return []
        tok = self._tokens.pop(0)
        self._emitted += 1
        return [self._emit(out, tok)]

    def feed(self, frame: Any, token: Any = None) -> list:
        x = self._prepare(frame)
        if self._schedule is not None:
            self._frames.append(x)
            self._frame_tokens.append(token)
            return self._feed_scheduled(final=False)
        return self._step(x, token=token, real=True)

    def flush(self) -> list:
        if self._schedule is not None:
            out = self._feed_scheduled(final=True)
            self._reset_state()
            return out
        out = []
        guard = self.net.SHIFT_NUM + self._received + 2
        while self._emitted < self._received:
            before = self._emitted
            out += self._step(None)
            guard -= 1
            if guard <= 0 and self._emitted == before:
                raise RuntimeError("BSVD flush did not produce enough delayed frames")
        self._reset_state()
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
        out_seq = [self.net.step(x) for x in frames]
        for _ in range(self.net.SHIFT_NUM):
            out_seq.append(self.net.step(None))
        clipped = out_seq[self.net.SHIFT_NUM:self.net.SHIFT_NUM + len(frames)]
        out = []
        for y, tok in zip(clipped[emit_start:emit_end], tokens[emit_start:emit_end], strict=True):
            if y is None:
                raise RuntimeError("BSVD scheduled window emitted an empty frame")
            out.append(self._emit(y, tok))
        self.net.reset()
        return out


__all__ = ["BSVD", "BsvdDenoiser", "default_weights_path", "load_bsvd"]
