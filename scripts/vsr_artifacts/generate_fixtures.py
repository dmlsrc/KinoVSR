#!/usr/bin/env python3
"""Generate repeatable VSR/deblock/noise-map stress clips from input videos.

The point is not photographic realism. The point is to create a small,
deterministic corpus that exercises failure modes we have seen in the wild:
scanlines that jump between frames, block-grid compression structure, pulsed
GOP noise, mosquito flicker around edges, and a mixed analog-plus-H.264 case.

Examples:
    scripts/vsr_artifacts/generate_fixtures.py \\
        --config scripts/vsr_artifacts/vsr_artifacts.local.toml

    scripts/vsr_artifacts/generate_fixtures.py \\
        --sources path/to/source.y4m \\
        --modes jumping_scanlines,mixed_analog_h264 \\
        --seconds 3 --output-dir path/to/generated-fixtures --overwrite
"""
from __future__ import annotations

import argparse
import copy
import json
import zlib
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from config import config_get, config_section, listify, load_config, resolve_path


MODE_ORDER = [
    "reencode_only",
    "mild_high_iso",
    "block_grid",
    "jumping_scanlines",
    "gop_pulse_noise",
    "mosquito_edges",
    "mixed_analog_h264",
]

MODE_NOTES = {
    "reencode_only": "source clip re-encoded with the configured codec/CRF/GOP and no synthetic artifact",
    "mild_high_iso": "luma-dependent gaussian sensor noise plus weak row drift",
    "block_grid": "8x8-ish quantization, block bias, and explicit grid edges",
    "jumping_scanlines": "horizontal line darkening/noise with frame-to-frame phase jumps",
    "gop_pulse_noise": "noise bursts at the start of each GOP-sized interval",
    "mosquito_edges": "sparse high-frequency flicker near detected edges",
    "mixed_analog_h264": "high ISO noise, jumping scanlines, GOP pulses, edge flicker, and block structure",
}

DEFAULTS = {
    "seconds": 5.0,
    "modes": "all",
    "seed": 20260705,
    "codec": "libx264",
    "crf": 34,
    "gop": 30,
    "preset": "veryfast",
}

DEFAULT_MODE_PARAMS: dict[str, dict] = {
    "mild_high_iso": {
        "sigma": 0.026,
        "row_sigma": 0.004,
    },
    "block_grid": {
        "block": 8,
        "qstep": 1.0 / 28.0,
        "bias_sigma": 0.010,
        "grid_amp": 0.014,
    },
    "jumping_scanlines": {
        "period": 3,
        "line_width": 1,
        "amp": 0.038,
        "row_jitter": 0.010,
        "dropout": 0.0,
    },
    "gop_pulse_noise": {
        "pulse_len": 5,
        "peak_sigma": 0.060,
        "floor_sigma": 0.005,
    },
    "mosquito_edges": {
        "amp": 0.045,
        "density": 0.22,
    },
    "mixed_analog_h264": {
        "high_iso_sigma": 0.030,
        "high_iso_row_sigma": 0.004,
        "scanline_period": 3,
        "scanline_line_width": 1,
        "scanline_amp": 0.032,
        "scanline_row_jitter": 0.008,
        "scanline_dropout": 0.0,
        "pulse_peak_sigma": 0.030,
        "pulse_floor_sigma": 0.002,
        "mosquito_amp": 0.032,
        "mosquito_density": 0.16,
        "include_block_grid": True,
        "block_qstep": 1.0 / 32.0,
        "block_bias_sigma": 0.007,
        "block_grid_amp": 0.010,
    },
}


def _fps(stream) -> float:
    rate = stream.average_rate or getattr(stream, "base_rate", None) or getattr(stream, "guessed_rate", None)
    return float(rate) if rate else 30.0


def _rate_fraction(fps: float) -> Fraction:
    return Fraction(fps).limit_denominator(100000)


def _mode_seed(seed: int, source: Path, mode: str) -> int:
    payload = f"{seed}:{source.name}:{mode}".encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


def _luma(frame: np.ndarray) -> np.ndarray:
    return 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]


def _repeat_blocks(values: np.ndarray, height: int, width: int, block: int) -> np.ndarray:
    return np.repeat(np.repeat(values, block, axis=0), block, axis=1)[:height, :width]


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def _add_high_iso(
    frame: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.026,
    row_sigma: float = 0.004,
) -> np.ndarray:
    y = _luma(frame)[..., None]
    shadow_weight = 0.65 + 0.9 * (1.0 - y)
    channel_weight = np.array([1.0, 0.85, 0.75], dtype=np.float32)
    noise = rng.normal(0.0, sigma, frame.shape).astype(np.float32)
    row = rng.normal(0.0, row_sigma, (frame.shape[0], 1, 1)).astype(np.float32)
    return _clip(frame + noise * shadow_weight * channel_weight + row)


def _add_block_grid(
    frame: np.ndarray,
    rng: np.random.Generator,
    block: int = 8,
    qstep: float = 1.0 / 28.0,
    bias_sigma: float = 0.010,
    grid_amp: float = 0.014,
) -> np.ndarray:
    h, w, _ = frame.shape
    quant = np.round(frame / qstep) * qstep
    bh = (h + block - 1) // block
    bw = (w + block - 1) // block
    bias = rng.normal(0.0, bias_sigma, (bh, bw, 1)).astype(np.float32)
    bias = _repeat_blocks(bias, h, w, block)
    grid = np.zeros((h, w, 1), dtype=np.float32)
    grid[block - 1 :: block, :, :] -= grid_amp
    grid[:, block - 1 :: block, :] += grid_amp * 0.6
    return _clip(quant + bias + grid)


def _add_jumping_scanlines(
    frame: np.ndarray,
    frame_index: int,
    rng: np.random.Generator,
    period: int = 3,
    line_width: int = 1,
    amp: float = 0.038,
    row_jitter: float = 0.010,
    dropout: float = 0.0,
) -> np.ndarray:
    h = frame.shape[0]
    phase = int((frame_index + rng.integers(0, period * 2)) % period)
    rows = np.arange(h)
    lines_1d = ((rows + phase) % max(1, period) < max(1, line_width))
    if dropout > 0.0:
        lines_1d &= rng.random(h) >= float(dropout)
    lines = lines_1d.astype(np.float32)[:, None, None]
    row_noise = rng.normal(0.0, row_jitter, (h, 1, 1)).astype(np.float32)
    strength = amp * (0.75 + 0.5 * rng.random())
    return _clip(frame - lines * strength + lines * row_noise)


def _add_gop_pulse_noise(
    frame: np.ndarray,
    frame_index: int,
    rng: np.random.Generator,
    gop: int,
    pulse_len: int = 5,
    peak_sigma: float = 0.060,
    floor_sigma: float = 0.005,
) -> np.ndarray:
    phase = frame_index % max(1, gop)
    if phase >= pulse_len:
        sigma = floor_sigma
    else:
        sigma = floor_sigma + peak_sigma * (1.0 - phase / max(1, pulse_len))
    return _add_high_iso(frame, rng, sigma=sigma, row_sigma=0.0015)


def _add_mosquito_edges(
    frame: np.ndarray,
    rng: np.random.Generator,
    amp: float = 0.045,
    density: float = 0.22,
) -> np.ndarray:
    y = _luma(frame)
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:] = np.abs(y[:, 1:] - y[:, :-1])
    gy[1:, :] = np.abs(y[1:, :] - y[:-1, :])
    edge = gx + gy
    threshold = float(np.quantile(edge, 0.82)) if np.any(edge) else 1.0
    mask = _dilate_mask(edge >= threshold)[..., None]
    sparse = (rng.random(frame.shape[:2] + (1,)) < density).astype(np.float32)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=frame.shape)
    chroma = np.array([1.0, 0.85, 0.85], dtype=np.float32)
    return _clip(frame + mask.astype(np.float32) * sparse * sign * amp * chroma)


def _apply_mode(
    mode: str,
    frame: np.ndarray,
    frame_index: int,
    rng: np.random.Generator,
    gop: int,
    params: dict | None = None,
) -> np.ndarray:
    params = params or {}
    if mode == "reencode_only":
        return frame
    if mode == "mild_high_iso":
        return _add_high_iso(frame, rng, **params)
    if mode == "block_grid":
        return _add_block_grid(frame, rng, **params)
    if mode == "jumping_scanlines":
        return _add_jumping_scanlines(frame, frame_index, rng, **params)
    if mode == "gop_pulse_noise":
        return _add_gop_pulse_noise(frame, frame_index, rng, gop=gop, **params)
    if mode == "mosquito_edges":
        return _add_mosquito_edges(frame, rng, **params)
    if mode == "mixed_analog_h264":
        out = _add_high_iso(
            frame,
            rng,
            sigma=float(params.get("high_iso_sigma", 0.030)),
            row_sigma=float(params.get("high_iso_row_sigma", 0.004)),
        )
        out = _add_jumping_scanlines(
            out,
            frame_index,
            rng,
            period=int(params.get("scanline_period", 3)),
            line_width=int(params.get("scanline_line_width", 1)),
            amp=float(params.get("scanline_amp", 0.032)),
            row_jitter=float(params.get("scanline_row_jitter", 0.008)),
            dropout=float(params.get("scanline_dropout", 0.0)),
        )
        out = _add_gop_pulse_noise(
            out,
            frame_index,
            rng,
            gop=gop,
            peak_sigma=float(params.get("pulse_peak_sigma", 0.030)),
            floor_sigma=float(params.get("pulse_floor_sigma", 0.002)),
        )
        out = _add_mosquito_edges(
            out,
            rng,
            amp=float(params.get("mosquito_amp", 0.032)),
            density=float(params.get("mosquito_density", 0.16)),
        )
        if bool(params.get("include_block_grid", True)):
            out = _add_block_grid(
                out,
                rng,
                qstep=float(params.get("block_qstep", 1.0 / 32.0)),
                bias_sigma=float(params.get("block_bias_sigma", 0.007)),
                grid_amp=float(params.get("block_grid_amp", 0.010)),
            )
        return out
    raise ValueError(f"unknown mode: {mode}")


def _read_frames(source: Path, seconds: float) -> tuple[list[np.ndarray], float]:
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        fps = _fps(stream)
        max_frames = max(1, int(round(seconds * fps)))
        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24").astype(np.float32) * (1.0 / 255.0)
            h_even = rgb.shape[0] & ~1
            w_even = rgb.shape[1] & ~1
            frames.append(rgb[:h_even, :w_even])
            if len(frames) >= max_frames:
                break
    if not frames:
        raise RuntimeError(f"no video frames decoded from {source}")
    return frames, fps


def _open_output(
    path: Path,
    codec: str,
    fps: float,
    width: int,
    height: int,
    crf: int,
    gop: int,
    preset: str,
    pix_fmt: str,
):
    out = av.open(str(path), "w")
    stream = out.add_stream(codec, rate=_rate_fraction(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = pix_fmt
    stream.codec_context.gop_size = gop
    stream.options = {
        "crf": str(crf),
        "preset": preset,
        "g": str(gop),
        "keyint": str(gop),
        "sc_threshold": "0",
    }
    return out, stream


def _write_clip(
    path: Path,
    source_frames: list[np.ndarray],
    fps: float,
    mode: str,
    seed: int,
    codec: str,
    crf: int,
    gop: int,
    preset: str,
    pix_fmt: str,
    mode_params: dict,
) -> dict:
    h, w = source_frames[0].shape[:2]
    rng = np.random.default_rng(seed)
    out, stream = _open_output(path, codec, fps, w, h, crf, gop, preset, pix_fmt)
    try:
        for idx, src in enumerate(source_frames):
            rgb = _apply_mode(mode, src, idx, rng, gop, mode_params)
            frame = av.VideoFrame.from_ndarray((rgb * 255.0 + 0.5).astype(np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()
    return {
        "path": str(path),
        "mode": mode,
        "mode_note": MODE_NOTES[mode],
        "seed": seed,
        "codec": codec,
        "crf": crf,
        "gop": gop,
        "pix_fmt": pix_fmt,
        "mode_params": mode_params,
        "fps": fps,
        "width": w,
        "height": h,
        "frames": len(source_frames),
    }


def _parse_modes(value: str) -> list[str]:
    if isinstance(value, (list, tuple)):
        modes = [str(m).strip() for m in value if str(m).strip()]
        bad = sorted(set(modes) - set(MODE_ORDER))
        if bad:
            raise SystemExit(f"unknown --modes entries: {', '.join(bad)}")
        return modes
    if value == "all":
        return MODE_ORDER[:]
    modes = [m.strip() for m in value.split(",") if m.strip()]
    bad = sorted(set(modes) - set(MODE_ORDER))
    if bad:
        raise SystemExit(f"unknown --modes entries: {', '.join(bad)}")
    return modes


def _resolve_sources(sources: list[str | Path], base_dir: Path | None = None) -> list[Path]:
    if not sources:
        raise SystemExit(
            "no input sources configured; pass --sources or set fixtures.sources "
            "in scripts/vsr_artifacts/vsr_artifacts.local.toml"
        )
    selected = [resolve_path(p, base_dir) for p in sources]
    existing = [p for p in selected if p.exists()]
    missing = [p for p in selected if not p.exists()]
    for path in missing:
        print(f"[fixtures] missing source, skipped: {path}", flush=True)
    if not existing:
        raise SystemExit("no input sources found")
    return existing


def _normalise_config(args: argparse.Namespace) -> argparse.Namespace:
    config, config_path = load_config(args.config)
    section = config_section(config, "fixtures")
    config_base = config_path.parent if config_path is not None else None
    merged = argparse.Namespace()
    merged.config = config_path
    merged.config_base = config_base
    merged.sources = listify(config_get(section, args, "sources", []))
    merged.sources_base = None if args.sources is not None else config_base
    merged.output_dir = config_get(section, args, "output_dir")
    if merged.output_dir is None:
        raise SystemExit("missing output directory; pass --output-dir or set fixtures.output_dir in config")
    output_base = None if args.output_dir is not None else config_base
    merged.output_dir = resolve_path(merged.output_dir, output_base)
    for key, default in DEFAULTS.items():
        setattr(merged, key, config_get(section, args, key, default))
    merged.seconds = float(merged.seconds)
    merged.seed = int(merged.seed)
    merged.crf = int(merged.crf)
    merged.gop = int(merged.gop)
    merged.codec = str(merged.codec)
    merged.preset = str(merged.preset)
    merged.pix_fmt = str(config_get(section, args, "pix_fmt", section.get("pix_fmt", "yuv420p")))
    merged.mode_params = copy.deepcopy(DEFAULT_MODE_PARAMS)
    for mode, values in config_section(section, "mode_params").items():
        if mode not in MODE_ORDER:
            raise SystemExit(f"unknown fixtures.mode_params entry: {mode}")
        if not isinstance(values, dict):
            raise SystemExit(f"fixtures.mode_params.{mode} must be a table/object")
        merged.mode_params.setdefault(mode, {}).update(values)
    merged.encode_overrides = {}
    for mode, values in config_section(section, "encode").items():
        if mode not in MODE_ORDER:
            raise SystemExit(f"unknown fixtures.encode entry: {mode}")
        if not isinstance(values, dict):
            raise SystemExit(f"fixtures.encode.{mode} must be a table/object")
        merged.encode_overrides[mode] = values
    merged.overwrite = bool(args.overwrite if args.overwrite is not None else section.get("overwrite", False))
    return merged


def _write_readme(output_dir: Path, manifest: dict) -> None:
    lines = [
        "# VSR Artifact Fixtures",
        "",
        "Synthetic clips generated from local DAVIS/Xiph references.",
        "They are deterministic stress cases for VSR, deblock, and noise-map tests.",
        "",
        f"- seconds: {manifest['seconds']}",
        f"- codec: {manifest['codec']}",
        f"- crf: {manifest['crf']}",
        f"- gop: {manifest['gop']}",
        f"- pix_fmt: {manifest['pix_fmt']}",
        f"- seed: {manifest['seed']}",
        "",
        "Modes:",
    ]
    for mode in MODE_ORDER:
        lines.append(f"- {mode}: {MODE_NOTES[mode]}")
    lines.extend(["", "Files are listed in `manifest.json`.", ""])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_fixtures(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = _parse_modes(args.modes)
    sources = _resolve_sources(args.sources, args.sources_base)
    manifest = {
        "config": str(args.config) if args.config is not None else None,
        "seconds": args.seconds,
        "codec": args.codec,
        "crf": args.crf,
        "gop": args.gop,
        "preset": args.preset,
        "pix_fmt": args.pix_fmt,
        "seed": args.seed,
        "modes": modes,
        "mode_params": {mode: args.mode_params.get(mode, {}) for mode in modes},
        "encode_overrides": args.encode_overrides,
        "sources": [],
        "outputs": [],
    }

    for source in sources:
        frames, fps = _read_frames(source, args.seconds)
        source_entry = {
            "path": str(source),
            "fps": fps,
            "width": int(frames[0].shape[1]),
            "height": int(frames[0].shape[0]),
            "frames": len(frames),
        }
        manifest["sources"].append(source_entry)
        for mode in modes:
            enc = args.encode_overrides.get(mode, {})
            codec = str(enc.get("codec", args.codec))
            crf = int(enc.get("crf", args.crf))
            gop = int(enc.get("gop", args.gop))
            preset = str(enc.get("preset", args.preset))
            pix_fmt = str(enc.get("pix_fmt", args.pix_fmt))
            mode_seed = _mode_seed(args.seed, source, mode)
            out_name = f"{source.stem}__{mode}__s{args.seed}_crf{crf}.mp4"
            out_path = output_dir / out_name
            if out_path.exists() and not args.overwrite:
                print(f"[fixtures] exists, skipped: {out_path}", flush=True)
                continue
            print(f"[fixtures] {source.name} -> {out_path.name}", flush=True)
            manifest["outputs"].append(
                {
                    "source": str(source),
                    **_write_clip(
                        out_path,
                        frames,
                        fps,
                        mode,
                        mode_seed,
                        codec,
                        crf,
                        gop,
                        preset,
                        pix_fmt,
                        args.mode_params.get(mode, {}),
                    ),
                }
            )

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_readme(output_dir, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="TOML/JSON config; defaults to scripts/vsr_artifacts/vsr_artifacts.local.{toml,json} if present")
    parser.add_argument("--sources", nargs="*", help="source videos; overrides config sources")
    parser.add_argument("--output-dir", help="directory for generated MP4s and manifest.json")
    parser.add_argument("--seconds", type=float, help="seconds to decode from each source")
    parser.add_argument("--modes", help=f"comma list or 'all'; choices: {', '.join(MODE_ORDER)}")
    parser.add_argument("--seed", type=int, help="base RNG seed")
    parser.add_argument("--codec", help="PyAV/FFmpeg encoder name")
    parser.add_argument("--crf", type=int, help="H.264 CRF; larger means more compression")
    parser.add_argument("--gop", type=int, help="GOP/keyframe interval used by encoder and pulse mode")
    parser.add_argument("--preset", help="x264 preset")
    parser.add_argument("--pix-fmt", help="encoder pixel format, e.g. yuv420p or yuv444p")
    parser.add_argument("--overwrite", action="store_true", default=None, help="regenerate clips that already exist")
    return parser.parse_args()


def main() -> int:
    manifest = build_fixtures(_normalise_config(parse_args()))
    print(f"[fixtures] wrote {len(manifest['outputs'])} clips", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
