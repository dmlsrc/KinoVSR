"""Probe which input traits trigger NAFNet artifact blow-ups.

This is intentionally a one-frame diagnostic, not a production filter. It decodes a
video frame, applies small controlled perturbations, runs the local MLX NAFNet port,
and prints raw residual plus deep SCA amplification stats. A perturbation that makes
the residual collapse is a strong clue about the trigger in that frame.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx  # noqa: E402

from kinovsr._optional import require_numpy  # noqa: E402
from kinovsr.processors.nafnet import net  # noqa: E402
from kinovsr.processors.nafnet.restorer import model_rgb, resolve_pool_mode  # noqa: E402
from kinovsr.ui.console import get_console


def _print(*parts: object) -> None:
    get_console().print(*parts, markup=False, highlight=False)

if TYPE_CHECKING:
    import numpy as np

    # Runtime evaluation of this alias crashed the original script on
    # import (np is None until _load_numpy runs); annotations are lazy
    # under future-annotations, so the alias lives here.
    Transform = tuple[str, Callable[[np.ndarray], np.ndarray]]

np: Any = None


def _load_numpy() -> Any:
    global np
    if np is None:
        np = require_numpy("kinovsr probe nafnet")
    return np


def _dtype(name: str) -> Any:
    if name == "float32":
        return mx.float32
    if name == "float16":
        return mx.float16
    if name == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"unknown dtype {name!r}")


def _parse_frames(spec: str) -> list[int]:
    frames: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            frames.append(int(part))
            continue
        bits = [b.strip() for b in part.split(":")]
        if len(bits) not in {2, 3}:
            raise ValueError(f"bad frame range {part!r}; use start:end[:step]")
        start = int(bits[0])
        end = int(bits[1])
        step = int(bits[2]) if len(bits) == 3 and bits[2] else 1
        if step <= 0:
            raise ValueError("frame range step must be positive")
        frames.extend(range(start, end + 1, step))
    if not frames:
        raise ValueError("empty frame list")
    return sorted(set(frames))


def _read_frames(video: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    try:
        import av
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit("PyAV is required for this probe; run it with the Kino/diffusers venv") from exc

    wanted_set = set(wanted)
    frames: dict[int, np.ndarray] = {}
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in wanted_set:
                rgb = frame.to_rgb().to_ndarray().astype(np.float32) / 255.0
                frames[i] = rgb
                if len(frames) == len(wanted_set):
                    break
    missing = sorted(wanted_set - set(frames))
    if missing:
        raise SystemExit(f"video ended before frame(s): {missing}")
    return frames


def _box_blur(img: np.ndarray, ky: int, kx: int) -> np.ndarray:
    h, w, _ = img.shape
    py, px = ky // 2, kx // 2
    padded = np.pad(img, ((py, py), (px, px), (0, 0)), mode="edge")
    acc = np.zeros_like(img, dtype=np.float32)
    for y in range(ky):
        for x in range(kx):
            acc += padded[y:y + h, x:x + w, :]
    return acc / float(ky * kx)


def _luma601(img: np.ndarray) -> np.ndarray:
    return (img[..., :1] * 0.299 + img[..., 1:2] * 0.587 + img[..., 2:3] * 0.114).astype(np.float32)


def _luma_only(img: np.ndarray) -> np.ndarray:
    return np.repeat(_luma601(img), 3, axis=-1)


def _blur_luma(img: np.ndarray, ky: int, kx: int) -> np.ndarray:
    y = _luma601(img)
    chroma = img - y
    out = _box_blur(y, ky, kx) + chroma
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _blur_chroma(img: np.ndarray, ky: int, kx: int) -> np.ndarray:
    y = _luma601(img)
    chroma = _box_blur(img - y, ky, kx)
    out = y + chroma
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _block_smooth(img: np.ndarray, block: int, amount: float) -> np.ndarray:
    out = img.copy()
    h, w, _ = out.shape
    for y0 in range(0, h, block):
        y1 = min(y0 + block, h)
        for x0 in range(0, w, block):
            x1 = min(x0 + block, w)
            tile = out[y0:y1, x0:x1, :]
            mean = np.mean(tile, axis=(0, 1), keepdims=True)
            tile[...] = tile * (1.0 - amount) + mean * amount
    return out


def _shadow_floor(img: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(img, floor).astype(np.float32)


def _transforms() -> list[Transform]:
    return [
        ("identity", lambda x: x.copy()),
        ("blur_y3_scanline", lambda x: _box_blur(x, 3, 1)),
        ("blur_y5_scanline", lambda x: _box_blur(x, 5, 1)),
        ("blur_x3_edges", lambda x: _box_blur(x, 1, 3)),
        ("blur_3x3", lambda x: _box_blur(x, 3, 3)),
        ("blur_5x5", lambda x: _box_blur(x, 5, 5)),
        ("luma_blur_y3", lambda x: _blur_luma(x, 3, 1)),
        ("chroma_blur_5x5", lambda x: _blur_chroma(x, 5, 5)),
        ("luma_only", _luma_only),
        ("block8_smooth_25", lambda x: _block_smooth(x, 8, 0.25)),
        ("block8_smooth_50", lambda x: _block_smooth(x, 8, 0.50)),
        ("shadow_floor_4_255", lambda x: _shadow_floor(x, 4.0 / 255.0)),
        ("shadow_floor_8_255", lambda x: _shadow_floor(x, 8.0 / 255.0)),
    ]


def _stats(a: Any) -> dict[str, float]:
    mx.eval(a)
    arr = np.abs(np.array(a, dtype=np.float32))
    return {
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def _trace_block(
    x: Any,
    p: dict,
    prefix: str,
    cfg: tuple,
    pool_mode: str,
    tlsc_train_hw: tuple[int, int],
) -> tuple[Any, dict[str, dict[str, float]]]:
    stats: dict[str, dict[str, float]] = {}

    def record(name: str, value: Any) -> Any:
        stats[name] = _stats(value)
        return value

    inp = record("input", x)
    y = record("norm1", net._layernorm(x, p[f"{prefix}.norm1.weight"], p[f"{prefix}.norm1.bias"]))
    y = record("conv1", net._conv(y, p, f"{prefix}.conv1"))
    y = record("dwconv", net._depthwise3x3(y, p, f"{prefix}.conv2"))
    y = record("sg1", net._simplegate(y))
    if pool_mode == "local":
        pooled = net._local_avg_pool2d(y, net._tlsc_kernel(f"{prefix}.sca.1", cfg, tlsc_train_hw))
    elif pool_mode == "global":
        pooled = mx.mean(y, axis=(1, 2), keepdims=True)
    else:
        raise ValueError(f"unknown NAFNet pool_mode {pool_mode!r}")
    record("pool", pooled)
    scale = record("sca_conv", net._conv(pooled, p, f"{prefix}.sca.1"))
    y = record("sca_mul", y * scale)
    y = record("conv3", net._conv(y, p, f"{prefix}.conv3"))
    x = record("after_beta", inp + y * p[f"{prefix}.beta"])
    z = record("norm2", net._layernorm(x, p[f"{prefix}.norm2.weight"], p[f"{prefix}.norm2.bias"]))
    z = record("conv4", net._conv(z, p, f"{prefix}.conv4"))
    z = record("sg2", net._simplegate(z))
    z = record("conv5", net._conv(z, p, f"{prefix}.conv5"))
    out = record("block_out", x + z * p[f"{prefix}.gamma"])
    return out, stats


def _forward_trace(
    inp: Any,
    p: dict,
    cfg: tuple,
    strength: float,
    pool_mode: str,
    trace_prefix: str,
    tlsc_train_hw: tuple[int, int],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    _, enc_nums, mid_num, dec_nums = cfg
    n_stages = len(enc_nums)
    h, w = inp.shape[1], inp.shape[2]
    dt = p["intro.weight"].dtype
    x_pad = net._pad(inp.astype(dt), 2 ** n_stages)

    block_stats: dict[str, dict[str, float]] | None = None

    def block(x: Any, prefix: str) -> Any:
        nonlocal block_stats
        if prefix == trace_prefix:
            x, block_stats = _trace_block(x, p, prefix, cfg, pool_mode, tlsc_train_hw)
            return x
        return net._naf_block(x, p, prefix, cfg, pool_mode, tlsc_train_hw)

    x = net._conv(x_pad, p, "intro", pad=1)
    encs = []
    for i in range(n_stages):
        for b in range(enc_nums[i]):
            x = block(x, f"encoders.{i}.{b}")
        encs.append(x)
        x = net._conv(x, p, f"downs.{i}", stride=2)
    for b in range(mid_num):
        x = block(x, f"middle_blks.{b}")
    for i in range(n_stages):
        x = mx.conv2d(x, p[f"ups.{i}.0.weight"], padding=0)
        x = net._pixel_shuffle(x, 2)
        x = x + encs[n_stages - 1 - i]
        for b in range(dec_nums[i]):
            x = block(x, f"decoders.{i}.{b}")
    residual = net._conv(x, p, "ending", pad=1)
    out = x_pad + strength * residual
    out = out[:, :h, :w, :]
    raw_delta = out - inp.astype(dt)
    if block_stats is None:
        raise RuntimeError(f"trace prefix {trace_prefix!r} was not visited")
    return block_stats, _stats(raw_delta)


def _probe_frame(
    frame: np.ndarray,
    p: dict,
    cfg: tuple,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, fn in _transforms():
        t0 = time.perf_counter()
        changed = np.clip(fn(frame).astype(np.float32), 0.0, 1.0)
        x = model_rgb(mx.array(changed[None, ...]))
        block_stats, residual = _forward_trace(
            x, p, cfg, args.strength, args.pool_mode,
            args.trace_block, tuple(args.tlsc_train_hw),
        )
        elapsed = time.perf_counter() - t0
        results.append({
            "transform": name,
            "residual": residual,
            "trace": block_stats,
            "seconds": elapsed,
        })
    return results


def _print_table(frame_no: int, shape: tuple[int, int, int], rows: list[dict[str, Any]], sort: str) -> None:
    if sort == "p99":
        rows = sorted(rows, key=lambda r: r["residual"]["p99"])
    ident = next(r for r in rows if r["transform"] == "identity")
    base_p99 = max(ident["residual"]["p99"], 1e-12)
    _print(f"\nframe {frame_no}  {shape[1]}x{shape[0]}")
    _print(
        "transform            "
        "res_mean  res_p99  res_max  ratio  "
        "sg1_p99  sca_p99  scamul_p99  conv3_p99  block_p99"
    )
    _print("-" * 105)
    for row in rows:
        res = row["residual"]
        tr = row["trace"]
        ratio = res["p99"] / base_p99
        _print(
            f"{row['transform']:<20} "
            f"{res['mean']:8.4f} {res['p99']:8.4f} {res['max']:8.4f} {ratio:6.2f} "
            f"{tr['sg1']['p99']:8.2f} {tr['sca_conv']['p99']:8.2f} "
            f"{tr['sca_mul']['p99']:11.2f} {tr['conv3']['p99']:10.2f} "
            f"{tr['block_out']['p99']:9.2f}"
        )


def run_probe_nafnet(argv: list[str] | None = None) -> int:
    _load_numpy()
    ap = argparse.ArgumentParser(
        prog="kinovsr probe nafnet", description=__doc__)
    ap.add_argument("--video", required=True, type=Path, help="source video to probe")
    ap.add_argument("--frames", default="0", help="frame list/ranges, e.g. 180 or 60,180,330 or 0:300:30")
    ap.add_argument("--nafnet", choices=["gopro", "gopro32", "sidd", "sidd32", "reds"], default="gopro32")
    ap.add_argument("--nafnet-weights", type=Path, default=None, help="optional explicit safetensors path")
    ap.add_argument("--pool", choices=["auto", "local", "global"], default="auto", help="SCA pooling mode")
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    ap.add_argument("--strength", type=float, default=1.0, help="residual scale, matching --nafnet-strength")
    ap.add_argument("--trace-block", default="encoders.3.12", help="NAFBlock prefix to instrument")
    ap.add_argument("--tlsc-train-hw", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    ap.add_argument("--sort", choices=["input", "p99"], default="input", help="table order")
    ap.add_argument("--json", type=Path, default=None, help="optional JSON report path")
    args = ap.parse_args(argv)

    try:
        frames = _parse_frames(args.frames)
    except ValueError as exc:
        raise SystemExit(f"bad --frames: {exc}") from exc
    weights = args.nafnet_weights if args.nafnet_weights is not None else args.nafnet
    args.pool_mode = resolve_pool_mode(weights, args.pool, variant=args.nafnet)

    _print(
        f"loading NAFNet weights={weights} dtype={args.dtype} "
        f"pool={args.pool_mode} strength={args.strength}"
    )
    p = net.load_params(weights, dtype=_dtype(args.dtype))
    cfg = net._config(p)
    _print(f"config width={cfg[0]} enc={cfg[1]} middle={cfg[2]} dec={cfg[3]} trace={args.trace_block}")

    decoded = _read_frames(args.video, frames)
    report: dict[str, Any] = {
        "video": str(args.video),
        "nafnet": args.nafnet,
        "weights": str(weights),
        "pool": args.pool_mode,
        "dtype": args.dtype,
        "strength": args.strength,
        "trace_block": args.trace_block,
        "frames": {},
    }
    for frame_no in frames:
        rows = _probe_frame(decoded[frame_no], p, cfg, args)
        _print_table(frame_no, decoded[frame_no].shape, rows, args.sort)
        report["frames"][str(frame_no)] = rows

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _print(f"\nwrote {args.json}")
