#!/usr/bin/env python3
"""Generate repeatable VSR/deblock/noise-map stress clips from input videos.

The point is not photographic realism. The point is to create a small,
deterministic corpus that exercises failure modes we have seen in the wild:
scanlines that jump between frames, block-grid compression structure, pulsed
GOP noise, mosquito flicker around edges, and a mixed analog-plus-H.264 case.

Examples:
    scripts/dev/vsr_artifacts/generate_fixtures.py \\
        --config scripts/dev/vsr_artifacts/vsr_artifacts.local.toml

    scripts/dev/vsr_artifacts/generate_fixtures.py \\
        --sources path/to/source.y4m \\
        --modes jumping_scanlines,mixed_analog_h264 \\
        --seconds 3 --output-dir path/to/generated-fixtures --overwrite
"""
from __future__ import annotations

import argparse
import copy
import json
import zlib
from contextlib import suppress
from fractions import Fraction
from pathlib import Path

import av
import cv2
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
    "mosquito_edges": "per-frame DCT re-quantization residuals masked into thin edge halos",
    "mixed_analog_h264": "high ISO noise, jumping scanlines, GOP pulses, mosquito halos, and block structure",
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
        # jpeg_cycle: per-frame JPEG re-quantization of a lightly dithered
        # base.  Both real-world flicker mechanisms are present: source noise
        # flips DCT coefficients across quantization boundaries, and the
        # quantizer itself breathes frame to frame (cycling quality).  Intra
        # 8x8 DCT with no in-loop deblocking keeps genuine ringing, unlike an
        # x264 proxy whose deadzone/skip freeze the residual on static shots.
        "method": "jpeg_cycle",
        "jpeg_quality_lo": 28,
        "jpeg_quality_hi": 45,
        "pre_dither_sigma": 0.012,
        # codec_residual (method="codec_residual"): residual of a low-quality
        # inter-codec proxy encode.  Kept as an alternate flavor.
        "proxy_codec": "libx264",
        "proxy_crf": 44,
        "proxy_preset": "ultrafast",
        "proxy_pix_fmt": "yuv420p",
        "residual_gain": 1.45,
        "residual_highpass_radius": 2,
        "residual_lowpass_reject": 0.85,
        "edge_quantile": 0.88,
        "halo_radius": 4,
        "core_radius": 0,
        "texture_quantile": 0.65,
        "texture_radius": 4,
        "texture_max": 0.55,
        "texture_softness": 0.18,
        "mask_blur_radius": 1,
        "keep_proxy": False,
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
        "mosquito_method": "jpeg_cycle",
        "mosquito_jpeg_quality_lo": 30,
        "mosquito_jpeg_quality_hi": 44,
        # the analog base is already noisy, which is mechanism one; no extra
        # dither needed before the proxy re-quantization.
        "mosquito_pre_dither_sigma": 0.0,
        "mosquito_proxy_codec": "libx264",
        "mosquito_proxy_crf": 42,
        "mosquito_proxy_preset": "ultrafast",
        "mosquito_proxy_pix_fmt": "yuv420p",
        "mosquito_residual_gain": 1.20,
        "mosquito_residual_highpass_radius": 2,
        "mosquito_residual_lowpass_reject": 0.85,
        "mosquito_edge_quantile": 0.88,
        "mosquito_halo_radius": 4,
        "mosquito_core_radius": 0,
        "mosquito_texture_quantile": 0.65,
        "mosquito_texture_radius": 4,
        "mosquito_texture_max": 0.55,
        "mosquito_texture_softness": 0.18,
        "mosquito_mask_blur_radius": 1,
        "mosquito_keep_proxy": False,
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
    payload = f"{seed}:{source.name}:{mode}".encode()
    return zlib.crc32(payload) & 0xFFFFFFFF


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


def _luma(frame: np.ndarray) -> np.ndarray:
    return 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]


def _repeat_blocks(values: np.ndarray, height: int, width: int, block: int) -> np.ndarray:
    return np.repeat(np.repeat(values, block, axis=0), block, axis=1)[:height, :width]


def _dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    iterations = max(0, int(iterations))
    if iterations == 0:
        return mask.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)


def _box_blur2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Small OpenCV box blur for local edge-density estimates."""
    radius = max(0, int(radius))
    if radius == 0:
        return x
    k = 2 * radius + 1
    return cv2.blur(x.astype(np.float32, copy=False), (k, k), borderType=cv2.BORDER_REPLICATE)


def _box_blur_rgb(x: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return x
    k = 2 * radius + 1
    return cv2.blur(x.astype(np.float32, copy=False), (k, k), borderType=cv2.BORDER_REPLICATE)


def _edge_magnitude(frame: np.ndarray) -> np.ndarray:
    y = _luma(frame).astype(np.float32, copy=False)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)


def _edge_halo_mask(
    frame: np.ndarray,
    edge_quantile: float = 0.88,
    halo_radius: int = 4,
    core_radius: int = 0,
    texture_quantile: float = 0.65,
    texture_radius: int = 4,
    texture_max: float = 0.55,
    texture_softness: float = 0.18,
    mask_blur_radius: int = 1,
) -> np.ndarray:
    edge = _edge_magnitude(frame)
    if not np.any(edge):
        return np.zeros(edge.shape, dtype=np.float32)
    edge_threshold = float(np.quantile(edge, float(edge_quantile)))
    texture_threshold = float(np.quantile(edge, float(texture_quantile)))
    core = edge >= edge_threshold
    texture = _box_blur2d((edge >= texture_threshold).astype(np.float32), texture_radius)
    structural = core & (texture <= float(texture_max))
    halo = _dilate_mask(structural, halo_radius)
    if core_radius >= 0:
        halo &= ~_dilate_mask(structural, core_radius)
    softness = max(float(texture_softness), 1e-6)
    texture_gate = np.clip((float(texture_max) + softness - texture) / softness, 0.0, 1.0)
    edge_strength = _box_blur2d(edge, max(1, min(int(halo_radius), 4)))
    edge_scale = np.clip(edge_strength / (np.quantile(edge_strength, 0.97) + 1e-6), 0.0, 1.0)
    mask = halo.astype(np.float32) * texture_gate * (0.50 + 0.75 * edge_scale)
    if mask_blur_radius > 0:
        mask = _box_blur2d(mask, mask_blur_radius)
    return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)


def _highpass_residual(
    residual: np.ndarray,
    radius: int = 2,
    lowpass_reject: float = 0.85,
) -> np.ndarray:
    if radius <= 0 or lowpass_reject <= 0.0:
        return residual
    return residual - _box_blur_rgb(residual, radius) * float(lowpass_reject)


def _mosquito_method(params: dict, mode: str) -> str:
    method = str(params.get("method", "jpeg_cycle")).strip().lower().replace("-", "_")
    if method not in {"jpeg_cycle", "codec_residual"}:
        raise ValueError(
            f"{mode} supports method='jpeg_cycle' or 'codec_residual', got {method!r}"
        )
    return method


def _strip_mosquito_prefix(params: dict) -> dict:
    out = {}
    for key, value in params.items():
        if key.startswith("mosquito_"):
            out[key.removeprefix("mosquito_")] = value
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
    if mode in {"mosquito_edges", "mixed_analog_h264"}:
        raise RuntimeError(f"{mode} needs the full frame sequence")
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


def _encode_frames(
    path: Path,
    frames: list[np.ndarray],
    fps: float,
    codec: str,
    crf: int,
    gop: int,
    preset: str,
    pix_fmt: str,
) -> None:
    h, w = frames[0].shape[:2]
    out, stream = _open_output(path, codec, fps, w, h, crf, gop, preset, pix_fmt)
    try:
        for rgb in frames:
            frame = av.VideoFrame.from_ndarray((rgb * 255.0 + 0.5).astype(np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()


def _decode_clip_frames(path: Path, expected_frames: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24").astype(np.float32) * (1.0 / 255.0))
            if len(frames) >= expected_frames:
                break
    if len(frames) < expected_frames:
        raise RuntimeError(f"decoded {len(frames)} frames from {path}, expected {expected_frames}")
    return frames[:expected_frames]


def _jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor((rgb * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)


def _jpeg_cycle_mosquito_frames(
    base_frames: list[np.ndarray],
    rng: np.random.Generator,
    params: dict,
) -> tuple[list[np.ndarray], dict]:
    """Mosquito noise via per-frame JPEG re-quantization.

    Each frame runs through an independent JPEG round trip (MJPEG semantics:
    intra 8x8 DCT, no skip blocks, no in-loop deblocking) at a per-frame
    quality drawn from [jpeg_quality_lo, jpeg_quality_hi].  A small gaussian
    dither is added before the round trip.  Together these reproduce the two
    mechanisms that make real mosquito noise flicker: source noise flipping
    quantized coefficients across boundaries, and the quantizer changing frame
    to frame.  The high-passed residual against the clean base is then masked
    into edge halos, so flat regions stay clean and the fixture stays
    mosquito-only.
    """
    qlo = int(params.get("jpeg_quality_lo", 28))
    qhi = int(params.get("jpeg_quality_hi", 45))
    if qhi < qlo:
        qlo, qhi = qhi, qlo
    dither_sigma = float(params.get("pre_dither_sigma", 0.012))
    residual_gain = float(params.get("residual_gain", 1.45))
    highpass_radius = int(params.get("residual_highpass_radius", 2))
    lowpass_reject = float(params.get("residual_lowpass_reject", 0.85))
    info = {
        "mosquito_method": "jpeg_cycle",
        "jpeg_quality_lo": qlo,
        "jpeg_quality_hi": qhi,
        "pre_dither_sigma": dither_sigma,
        "residual_gain": residual_gain,
        "residual_highpass_radius": highpass_radius,
        "residual_lowpass_reject": lowpass_reject,
    }
    rendered: list[np.ndarray] = []
    for base in base_frames:
        x = base
        if dither_sigma > 0.0:
            x = _clip(base + rng.normal(0.0, dither_sigma, base.shape).astype(np.float32))
        q = qlo if qhi == qlo else int(rng.integers(qlo, qhi + 1))
        proxy = _jpeg_roundtrip(x, q)
        if proxy.shape != base.shape:
            proxy = proxy[: base.shape[0], : base.shape[1], :]
        residual = _highpass_residual(proxy - base, highpass_radius, lowpass_reject)
        mask = _edge_halo_mask(
            base,
            edge_quantile=float(params.get("edge_quantile", 0.88)),
            halo_radius=int(params.get("halo_radius", 4)),
            core_radius=int(params.get("core_radius", 0)),
            texture_quantile=float(params.get("texture_quantile", 0.65)),
            texture_radius=int(params.get("texture_radius", 4)),
            texture_max=float(params.get("texture_max", 0.55)),
            texture_softness=float(params.get("texture_softness", 0.18)),
            mask_blur_radius=int(params.get("mask_blur_radius", 1)),
        )
        rendered.append(_clip(base + residual * residual_gain * mask[..., None]))
    return rendered, info


def _mosquito_frames(
    base_frames: list[np.ndarray],
    fps: float,
    gop: int,
    seed: int,
    rng: np.random.Generator,
    output_path: Path,
    params: dict,
    mode: str,
) -> tuple[list[np.ndarray], dict]:
    method = _mosquito_method(params, mode)
    if method == "jpeg_cycle":
        return _jpeg_cycle_mosquito_frames(base_frames, rng, params)
    return _codec_residual_mosquito_frames(base_frames, fps, gop, seed, output_path, params)


def _codec_residual_mosquito_frames(
    base_frames: list[np.ndarray],
    fps: float,
    gop: int,
    seed: int,
    output_path: Path,
    params: dict,
) -> tuple[list[np.ndarray], dict]:
    proxy_codec = str(params.get("proxy_codec", "libx264"))
    proxy_crf = int(params.get("proxy_crf", 44))
    proxy_gop = int(params.get("proxy_gop", gop))
    proxy_preset = str(params.get("proxy_preset", "ultrafast"))
    proxy_pix_fmt = str(params.get("proxy_pix_fmt", "yuv420p"))
    residual_gain = float(params.get("residual_gain", 1.45))
    highpass_radius = int(params.get("residual_highpass_radius", 2))
    lowpass_reject = float(params.get("residual_lowpass_reject", 0.85))
    keep_proxy = bool(params.get("keep_proxy", False))
    scratch_dir = output_path.parent / ".codec_residual_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = scratch_dir / f"{output_path.stem}__proxy_s{seed}_crf{proxy_crf}.mp4"
    info = {
        "mosquito_method": "codec_residual",
        "proxy_codec": proxy_codec,
        "proxy_crf": proxy_crf,
        "proxy_gop": proxy_gop,
        "proxy_preset": proxy_preset,
        "proxy_pix_fmt": proxy_pix_fmt,
        "residual_gain": residual_gain,
        "residual_highpass_radius": highpass_radius,
        "residual_lowpass_reject": lowpass_reject,
    }
    try:
        _encode_frames(proxy_path, base_frames, fps, proxy_codec, proxy_crf, proxy_gop, proxy_preset, proxy_pix_fmt)
        proxy_frames = _decode_clip_frames(proxy_path, len(base_frames))
    finally:
        if keep_proxy:
            info["proxy_path"] = str(proxy_path)
        else:
            proxy_path.unlink(missing_ok=True)
            with suppress(OSError):
                scratch_dir.rmdir()

    rendered: list[np.ndarray] = []
    for base, proxy in zip(base_frames, proxy_frames, strict=True):
        if proxy.shape != base.shape:
            proxy = proxy[: base.shape[0], : base.shape[1], :]
        residual = _highpass_residual(proxy - base, highpass_radius, lowpass_reject)
        mask = _edge_halo_mask(
            base,
            edge_quantile=float(params.get("edge_quantile", 0.88)),
            halo_radius=int(params.get("halo_radius", 4)),
            core_radius=int(params.get("core_radius", 0)),
            texture_quantile=float(params.get("texture_quantile", 0.65)),
            texture_radius=int(params.get("texture_radius", 4)),
            texture_max=float(params.get("texture_max", 0.55)),
            texture_softness=float(params.get("texture_softness", 0.18)),
            mask_blur_radius=int(params.get("mask_blur_radius", 1)),
        )
        rendered.append(_clip(base + residual * residual_gain * mask[..., None]))
    return rendered, info


def _mixed_analog_base_frames(
    source_frames: list[np.ndarray],
    rng: np.random.Generator,
    gop: int,
    params: dict,
) -> list[np.ndarray]:
    rendered: list[np.ndarray] = []
    for idx, frame in enumerate(source_frames):
        out = _add_high_iso(
            frame,
            rng,
            sigma=float(params.get("high_iso_sigma", 0.030)),
            row_sigma=float(params.get("high_iso_row_sigma", 0.004)),
        )
        out = _add_jumping_scanlines(
            out,
            idx,
            rng,
            period=int(params.get("scanline_period", 3)),
            line_width=int(params.get("scanline_line_width", 1)),
            amp=float(params.get("scanline_amp", 0.032)),
            row_jitter=float(params.get("scanline_row_jitter", 0.008)),
            dropout=float(params.get("scanline_dropout", 0.0)),
        )
        out = _add_gop_pulse_noise(
            out,
            idx,
            rng,
            gop=gop,
            peak_sigma=float(params.get("pulse_peak_sigma", 0.030)),
            floor_sigma=float(params.get("pulse_floor_sigma", 0.002)),
        )
        rendered.append(out)
    return rendered


def _apply_block_grid_frames(
    frames: list[np.ndarray],
    rng: np.random.Generator,
    params: dict,
) -> list[np.ndarray]:
    return [
        _add_block_grid(
            frame,
            rng,
            qstep=float(params.get("block_qstep", 1.0 / 32.0)),
            bias_sigma=float(params.get("block_bias_sigma", 0.007)),
            grid_amp=float(params.get("block_grid_amp", 0.010)),
        )
        for frame in frames
    ]


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
    extra: dict = {}
    if mode == "mosquito_edges":
        rendered, extra = _mosquito_frames(source_frames, fps, gop, seed, rng, path, mode_params, mode)
    elif mode == "mixed_analog_h264":
        rendered = _mixed_analog_base_frames(source_frames, rng, gop, mode_params)
        mosquito_params = _strip_mosquito_prefix(mode_params)
        rendered, extra = _mosquito_frames(rendered, fps, gop, seed, rng, path, mosquito_params, mode)
        if bool(mode_params.get("include_block_grid", True)):
            rendered = _apply_block_grid_frames(rendered, rng, mode_params)
    else:
        rendered = [_apply_mode(mode, src, idx, rng, gop, mode_params) for idx, src in enumerate(source_frames)]
    _encode_frames(path, rendered, fps, codec, crf, gop, preset, pix_fmt)
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
        **extra,
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
            "in scripts/dev/vsr_artifacts/vsr_artifacts.local.toml"
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
    parser.add_argument("--config", type=Path, help="TOML/JSON config; defaults to scripts/dev/vsr_artifacts/vsr_artifacts.local.{toml,json} if present")
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
