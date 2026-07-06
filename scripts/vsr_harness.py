#!/usr/bin/env python3
"""VAE-decode (or read MP4) and pump frames through VideoToolbox VSR +
optional temporal frame-rate conversion. Writes the upscaled MP4 directly
via AVAssetWriter - no ffmpeg, no PNG round-trip, no disk WAV by default.

Usage
-----
    # Latent path: VAE-decode the NPZ sidecar, then VSR it.
    scripts/vsr_harness.py --latent run.npz --weights $LTX_DEFAULT_WEIGHTS_PATH \
        --output-dir outputs/vsr/run1

    # Same, with audio muxed and frame rate doubled to 48 fps.
    scripts/vsr_harness.py --latent run.npz --weights ... \
        --output-dir outputs/vsr/run1 --audio --target-fps 48

    # Video path: skip VAE; VSR an existing clip. Add --audio to carry the
    # source file's audio track through to the upscaled MP4.
    scripts/vsr_harness.py --video clip.mp4 \
        --output-dir outputs/vsr/run2 --spatial-mode balanced --audio

    # Process only the middle: upscale the [5s, 8s) window of a long clip.
    # --start/--end (and --max-frames) accept frames (120), seconds (5s / 1.5),
    # or a clock string (0:05, 1:02:03). --video seeks natively to the window.
    scripts/vsr_harness.py --video clip.mp4 \
        --output-dir outputs/vsr/run3 --start 0:05 --end 0:08 --audio

Spatial modes (scale is implied by the mode)
--------------------------------------------
    fast      VTLowLatencySuperResolutionScalerConfiguration.  Scale 2x.
              Input must fit between 96x96 and 960x960.  Per-frame,
              no temporal context.
    balanced  VTSuperResolutionScalerConfiguration, InputType=Video.
              Scale 4x.  Downloadable model (auto-fetched on first use).
              Uses previous source + previous output frames to inform the
              upscale.  Default for video.  Tends to be slightly crisper
              on motion at the cost of slightly higher frame-to-frame
              variation than image mode.
    image     VTSuperResolutionScalerConfiguration, InputType=Image.  Scale 4x.
              Per-frame deterministic upscale, no prev-frame feedback.
              Apple documents this as for stills, but on real video it's a
              legitimate alternative - slightly softer per-frame detail
              than balanced but measurably smoother frame-to-frame (lower
              temporal second-difference).  Use scripts/compare_video_shimmer.py
              to A/B the two modes on your own content.
    basicvsrpp MLX BasicVSR++ 4x learned SR, recurrent sliding windows.
    realbasicvsr
              MLX RealBasicVSR 4x learned SR. Cleans the LR clip first, then
              runs BasicVSR propagation in sliding windows.

Temporal modes (only relevant when --target-fps is set)
-------------------------------------------------------
    normal    Default.  Fast and adequate for ~2x rate-up.
    high      VTFrameRateConversion's QualityPrioritizationQuality - more
              compute per interpolated frame, cleaner motion.

The VAE decoder defaults track LTX_2_MLX/generate.py's happy path
(native backend + zero spatial padding) via the encode_modes_harness
helpers. Chunks are cast to fp16 RGBA inside MLX so the full bf16
precision is preserved through to VSR's RGBAHalf source format -
quantization happens at the destination (either CIContext rendering
to NV12 for LL, or AVAssetWriter encoding to HEVC for HQ).

Known limitation for `--video` on edited footage
------------------------------------------------
`--spatial-mode balanced` chains previous-frame state through VSR for
temporal coherence. Across a hard cut that's the wrong context and can
produce ghosting around the cut frame. LTX latents are single-shot
generations so this is moot for `--latent`. For edited MP4s, enable
`--cut-detect` to reset the chain at hard cuts.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from LTX_2_MLX.progress import StackedPhaseBars
from LTX_2_MLX.videotoolbox import (
    AudioTrack,
    AVWriter,
    CutDetector,
    VsrSession,
    VtfrcSession,
    autorelease_pool,
    require_pyobjc,
)
from LTX_2_MLX.videotoolbox import color as _color
from LTX_2_MLX.videotoolbox import pixel_buffers as _pb
from LTX_2_MLX.videotoolbox import video_reader as _vr
from LTX_2_MLX.videotoolbox import yuv as _yuv
from LTX_2_MLX.videotoolbox.bsvd import BsvdDenoiser
from LTX_2_MLX.videotoolbox.comparison import render_comparison
from LTX_2_MLX.videotoolbox.denoise import LumaChromaDenoiser, McTemporalDenoiser, SpatialDenoiser
from LTX_2_MLX.videotoolbox.edge_sanitize import (
    compute_aspect_crop,
    detect_bars,
    detect_junk_edges,
    parse_edges_spec,
)
from LTX_2_MLX.videotoolbox.edge_sanitize import (
    crop_rgb as _crop_rgb,
)
from LTX_2_MLX.videotoolbox.edge_sanitize import (
    restore_borders as _restore_borders,
)
from LTX_2_MLX.videotoolbox.edge_sanitize import (
    sanitize_rgb as _sanitize_rgb,
)
from LTX_2_MLX.videotoolbox.fastdvdnet import FastDvdDenoiser
from LTX_2_MLX.videotoolbox.images import save_image
from LTX_2_MLX.videotoolbox.vsr import NativePassthrough
from LTX_2_MLX.videotoolbox.vsr_blocks import make_lanczos_plan, resample_width
from LTX_2_MLX.videotoolbox.writer import (
    HEVC_PROFILE_MAIN10,
    HEVC_PROFILE_MAIN422_10,
)

NATIVE_FPS = 24.0


def parse_mlx_dtype_name(name: str) -> Any:
    try:
        return {"float16": mx.float16, "float32": mx.float32}[name]
    except KeyError:
        raise ValueError(f"unknown MLX dtype {name!r}") from None


def parse_time_or_frames(spec: str, fps: float) -> int:
    """Convert a position/duration spec to a frame count at `fps`.

    Accepted forms (case-insensitive):
        "120"         bare integer  -> 120 frames
        "120f"        explicit f    -> 120 frames
        "5s", "2.5s"  seconds       -> round(seconds * fps) frames
        "1.5"         bare decimal  -> seconds (a fractional frame is meaningless)
        "1:30"        mm:ss         -> seconds -> frames
        "1:02:03"     hh:mm:ss      -> seconds -> frames
        "0:04.5"      mm:ss.frac    -> seconds -> frames

    The bare-integer-is-frames / bare-decimal-is-seconds split keeps existing
    integer `--max-frames N` invocations meaning frames, while letting any
    time be given as seconds or a clock string. Returns a non-negative int.
    """
    s = str(spec).strip().lower()
    if not s:
        raise ValueError("empty time/frame spec")
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            hh, (mm, ss) = "0", parts
        elif len(parts) == 3:
            hh, mm, ss = parts
        else:
            raise ValueError(f"bad time spec {spec!r} (use mm:ss or hh:mm:ss)")
        seconds = int(hh) * 3600 + int(mm) * 60 + float(ss)
        frames = round(seconds * fps)
    elif s.endswith("f"):
        frames = int(s[:-1])
    elif s.endswith("s"):
        frames = round(float(s[:-1]) * fps)
    elif "." in s:
        frames = round(float(s) * fps)
    else:
        frames = int(s)
    if frames < 0:
        raise ValueError(f"time/frame spec {spec!r} is negative")
    return int(frames)


def resolve_trim(
    start_spec: str | None, end_spec: str | None, fps: float, total_frames: int,
) -> tuple[int, int | None]:
    """Resolve --start/--end specs to a half-open frame window [start, end).

    `end` is None for an open-ended window. Clamps end to the input length and
    rejects an empty or out-of-range window with a clean SystemExit.
    """
    try:
        start_frame = parse_time_or_frames(start_spec, fps) if start_spec else 0
        end_frame = parse_time_or_frames(end_spec, fps) if end_spec else None
    except ValueError as e:
        raise SystemExit(f"bad --start/--end value: {e}") from None
    if end_frame is not None and end_frame <= start_frame:
        raise SystemExit(
            f"--end ({end_frame}f) must be greater than --start ({start_frame}f)"
        )
    if total_frames and start_frame >= total_frames:
        raise SystemExit(
            f"--start ({start_frame}f) is at or past the input length "
            f"({total_frames} frames)"
        )
    if total_frames and end_frame is not None:
        end_frame = min(end_frame, total_frames)
    return start_frame, end_frame


# ---------------------------------------------------------------------------
# MLX-side chunk conversion. Lives in the harness (not the videotoolbox
# package) because it depends on LTX_2_MLX.model.video_vae internals.
# ---------------------------------------------------------------------------

# A/B toggle (env): VSR_CHUNK_AS_ARRAY=1 returns one big ndarray per chunk
# (300 MiB resident until chunk-end). Default 0 returns a list of per-frame
# ndarrays so each frame's ~1.2 MiB can be freed as the inner loop progresses.
#
# Measured A/B (721-frame latent, bare `time`, no instrumentation):
#   --vae-tiling auto  list   wall 142.2s  VAE 109.3s  5.07 fps
#   --vae-tiling auto  array  wall 142.3s  VAE 109.3s  5.07 fps
#   --vae-tiling single list  wall 151.1s  VAE  58.3s  4.77 fps
#   --vae-tiling single array wall 165.5s  VAE  70.2s  4.36 fps
# Tiled mode: list/array indistinguishable - chunks are small enough that
# list-vs-array allocation overhead is in the noise. Single-shot: list is
# ~10% faster wall and ~17% faster through the VAE itself. So this env var
# is NOT a no-op - it controls a real perf difference for `--vae-tiling off`.
# List stays the default because it's the faster path everywhere AND lets
# the inner loop drop per-frame memory as it goes.
import os as _os

_CHUNK_AS_ARRAY = _os.environ.get("VSR_CHUNK_AS_ARRAY", "0") == "1"


def chunk_to_rgba_fp16(chunk: Any, mx_mod: Any):
    """(B,3,T,H,W) bf16 in [-1,1] -> list[(H,W,4) fp16] per-frame arrays.

    Direct path for VSR's RGBAHalf source format and for CIImage's
    kCIFormatRGBAh upload to NV12. Skips uint8 quantization that
    chunk_to_uint8 would impose, so the VAE's full bf16 precision
    survives into VSR.

    Returns a list of independently-allocated per-frame ndarrays rather than
    one big (T,H,W,4) array. The downstream inner loop can then null out
    `chunk[i]` once a frame is consumed, freeing that frame's ~1.2 MB back
    to the OS - so the resident chunk memory tapers as we work through it
    instead of sitting at full size until chunk-end. Allocator overhead is
    one mmap per frame (cheap; macOS mmaps allocations >= 16 KiB directly).
    """
    B, C, T, H, W = chunk.shape
    rescaled = mx_mod.clip((chunk + 1.0) * 0.5, 0.0, 1.0).astype(mx_mod.float16)
    alpha = mx_mod.ones((B, 1, T, H, W), dtype=mx_mod.float16)
    rgba = mx_mod.concatenate([rescaled, alpha], axis=1)
    transposed = mx_mod.transpose(rgba, (0, 2, 3, 4, 1))  # (B, T, H, W, 4)
    mx_mod.eval(transposed)
    if _CHUNK_AS_ARRAY:
        arr = mx_mod.contiguous(transposed)
        result: Any = arr[0] if arr.ndim == 5 else arr
    else:
        # List of per-frame mx arrays so each frame's memory can be freed
        # independently by the main loop. mx.contiguous gives each frame its own
        # buffer, so dropping a frame lets MLX release it without pinning the
        # whole chunk's Metal state across the loop.
        result = [mx_mod.contiguous(transposed[0, t]) for t in range(T)]
    # Drop refs to all MLX intermediates AND force the cache to release.
    # Without clear_cache here, the rescaled/alpha/rgba/transposed Metal
    # buffers (which can be GiB-scale for single-shot decodes) sit in MLX's
    # cache for the entire downstream inner loop - only released when the
    # generator resumes after the loop drains. The numpy result is already
    # an independent Python-owned copy, so MLX state is safe to drop now.
    del rescaled, alpha, rgba, transposed
    try:
        mx_mod.clear_cache()
    except Exception:
        pass
    return result


def make_video_decoder_default(
    weights_path: str, compute_dtype: Any,
):
    """generate.py's happy-path defaults via encode_modes_harness."""
    from scripts.encode_modes_harness import make_video_decoder
    return make_video_decoder(weights_path, compute_dtype)


def latent_dims(latent: Any) -> tuple[int, int, int]:
    _, _, latent_frames, latent_height, latent_width = latent.shape
    n_frames = 1 + (latent_frames - 1) * 8
    height = latent_height * 32
    width = latent_width * 32
    return n_frames, height, width


def plan_vae_tiling(latent: Any) -> tuple[Any, int, str]:
    """Decide the tiling cfg + chunk count up front.

    Returns (cfg, n_chunks, human_description). `cfg` is the TilingConfig
    (or None for single-shot decode). Pure CPU/dim arithmetic - no GPU
    work - so it's cheap to call before any tqdm bar starts (which is
    what avoids clobbering the bar with VAE tiling status mid-stream).
    """
    from LTX_2_MLX.model.video_vae.tiling import TilingConfig

    n_frames, height, width = latent_dims(latent)
    cfg = TilingConfig.auto(
        height=height, width=width, num_frames=n_frames,
    )
    if cfg is None:
        return cfg, 1, f"off (single-shot decode of {n_frames} frames)"

    sp = cfg.spatial_config
    tp = cfg.temporal_config
    spatial_desc = (
        f"spatial tile={sp.tile_size_in_pixels} overlap={sp.tile_overlap_in_pixels}"
        if sp else "no spatial"
    )
    temporal_desc = (
        f"temporal tile={tp.chunk_size_in_frames} overlap={tp.chunk_overlap_in_frames}"
        if tp else "no temporal"
    )
    if tp is not None:
        tile = tp.chunk_size_in_frames
        overlap = tp.chunk_overlap_in_frames
        step = max(1, tile - overlap)
        n_chunks = max(1, -(-(n_frames - overlap) // step))
    else:
        n_chunks = 1
    return cfg, n_chunks, f"{spatial_desc}, {temporal_desc}"


def iter_latent_chunks(
    latent: Any,
    decoder: Any,
    *,
    cfg: Any,
    mx_mod: Any,
    output_format: str = "uint8_rgb",
    single_pass: bool = False,
) -> Iterator[mx.array]:
    """Yield decoded chunks. output_format selects the conversion:
       "uint8_rgb"  -> (T,H,W,3) uint8  (for LowLatency VSR / NV12 source)
       "fp16_rgba"  -> (T,H,W,4) fp16   (for HighQuality VSR / RGBAHalf source)
    """
    from LTX_2_MLX.model.video_vae.tiling import decode_single_pass, decode_streaming
    from scripts.encode_modes_harness import chunk_to_uint8

    convert = chunk_to_rgba_fp16 if output_format == "fp16_rgba" else chunk_to_uint8

    if single_pass:
        # --vae-tiling single: one whole-clip decode. decode_single_pass logs whether
        # the frame count crosses the int32 boundary (frames past it decode white).
        out = convert(decode_single_pass(latent, decoder), mx_mod)
        try:
            mx_mod.clear_cache()
        except Exception:
            pass
        gc.collect()
        yield out
        return

    # decode_streaming handles cfg=None (no spatial tiling + default temporal
    # chunking), so this streams chunk-by-chunk for every case -- no whole-video
    # accumulate. convert() quantizes each chunk at the destination format.
    for chunk in decode_streaming(latent, decoder, cfg, show_progress=False):
        out = convert(chunk, mx_mod)
        # convert() clears the cache; `chunk` is the only MLX tensor still
        # live, so drop it + clear before yielding to the downstream loop.
        del chunk
        try:
            mx_mod.clear_cache()
        except Exception:
            pass
        gc.collect()
        yield out
        del out


# ---------------------------------------------------------------------------
# Audio decode (latent only)
# ---------------------------------------------------------------------------

def _decode_audio_track(audio_latent: Any, weights: str, compute_dtype: Any) -> AudioTrack:
    """Decode the audio latent through the audio VAE + vocoder into an
    in-memory AudioTrack. No disk WAV unless the caller asks for a sidecar.
    """
    import mlx.core as mx

    from scripts.decode_latent_debug import decode_audio_latent, make_audio_decoder_and_vocoder

    print("Decoding audio latent (audio VAE + vocoder)...")
    audio_decoder, vocoder, sample_rate = make_audio_decoder_and_vocoder(weights, compute_dtype)
    waveform = decode_audio_latent(audio_latent, audio_decoder, vocoder, mx, onset_mode="auto")
    arr = waveform
    if arr.ndim == 3:
        arr = arr[0]
    track = AudioTrack(arr, sample_rate=int(sample_rate))
    print(f"  audio: {track.channels}ch, {track.sample_rate} Hz, {track.n_samples} samples")
    del waveform, audio_decoder, vocoder
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass
    return track


def _read_audio_track_from_video(mp4_path: Path) -> AudioTrack | None:
    """Read the audio track of an MP4/MOV into an in-memory AudioTrack.

    Uses AVFoundation's AVAudioFile (via videotoolbox.audio.read_wav), which
    decodes the container's audio stream straight into a (channels, frames)
    float32 MLX array - no ffmpeg, no disk WAV. Returns None if the file has
    no audio track.
    """
    from LTX_2_MLX.videotoolbox.audio import read_wav

    if hasattr(_vr, "read_audio_track"):
        # ffmpeg compatibility reader: decode audio via the same backend that
        # reads the video (the native audio path cannot open these containers).
        print(f"[setup] reading audio track from {mp4_path} (ffmpeg)")
        try:
            return _vr.read_audio_track(mp4_path)
        except Exception as e:
            print(f"[setup] audio decode failed ({type(e).__name__}); continuing without audio")
            return None

    print(f"[setup] reading audio track from {mp4_path}")
    try:
        sample_rate, samples = read_wav(mp4_path)
    except Exception as e:
        # No audio track (or an unsupported audio format) - carry on silent.
        print(f"[setup] no usable audio track ({type(e).__name__}: {e}); output will be silent")
        return None
    track = AudioTrack(samples, sample_rate=int(sample_rate))
    print(f"  audio: {track.channels}ch, {track.sample_rate} Hz, {track.n_samples} samples")
    return track


# ---------------------------------------------------------------------------
# HEVC profile selection
# ---------------------------------------------------------------------------

def _pick_hevc_profile(spatial_mode: str, encode_chroma: str) -> str:
    """auto picks 4:2:2 for HQ modes (RGBAHalf preserves chroma), 4:2:0 for fast (LL/NV12)."""
    if encode_chroma == "420":
        return HEVC_PROFILE_MAIN10
    if encode_chroma == "422":
        return HEVC_PROFILE_MAIN422_10
    # balanced/image/none/learned upscalers carry RGB (4:4:4 chroma) to the encoder -> 4:2:2.
    return (HEVC_PROFILE_MAIN422_10
            if spatial_mode in ("balanced", "image", "none", "basicvsrpp", "realbasicvsr",
                                "realesrgan", "safmn", "esc", "realviformer", "realplksr")
            else HEVC_PROFILE_MAIN10)


_PP_STAGE_NAMES = ("restore", "deblock", "denoise", "nafnet")


def _pp_order(spec: str) -> list:
    """argparse type for --preprocess-order: a comma-separated permutation/subset of the
    preprocess stage names (restore, deblock, denoise, nafnet)."""
    names = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [n for n in names if n not in _PP_STAGE_NAMES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown preprocess stage(s) {bad}; valid: {list(_PP_STAGE_NAMES)}")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"duplicate stage in {names}")
    return names


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    from LTX_2_MLX.generate import sanitize_output_prefix
    stem = f"{sanitize_output_prefix(args.output_prefix)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pre_dir = out_root / f"{stem}_pre"
    post_dir = out_root / f"{stem}_post"
    if args.save_pre_frames:
        pre_dir.mkdir(parents=True, exist_ok=True)
    if args.save_post_frames:
        post_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output stem: {stem}")

    audio_track: AudioTrack | None = None
    gop_schedule: Any = None   # GOP-aligned window plan for recurrent stages (--gop-align)
    _gop_head_skip = 0         # gop-align context frames whose outputs are dropped
    src_transform: Any = None  # source rotation/flip (set on the --video path)
    src_pixel_aspect: tuple[int, int] | None = None
    # Input frame window [loop_win_start, loop_win_end). The --video reader
    # trims at decode time (efficient seek), so for that path these stay
    # (0, None) and the trim happens upstream; the --latent path enforces the
    # window in the main loop instead.
    loop_win_start, loop_win_end = 0, None
    win_start, win_end = 0, None  # resolved input window (both paths set these)
    # Output color tags: the video path fills these from the source container;
    # the latent path has no source container, so use the SDR BT.709/video-range
    # default explicitly. Passing cv_color also enables the writer's deterministic
    # RGBAHalf->YUV conversion for latent/uploaded-buffer producers.
    _resolved_color = _color.resolve({"full_range": False}, "bt709")
    color_props: dict | None = _color.av_color_properties(_resolved_color)
    cv_color: tuple | None = _color.cv_triple(_resolved_color)
    output_full_range = _resolved_color[3]

    # ---- Input source ------------------------------------------------------
    if args.latent:
        from scripts.decode_latent_debug import load_latents, parse_dtype

        print(f"[setup] VAE-decoding latent: {args.latent}")
        t = time.perf_counter()
        latent, audio_latent = load_latents(args.latent, mx, "auto", stage=args.latent_stage)
        compute_dtype = parse_dtype(mx, args.vae_dtype)
        print(
            f"[setup] load_latents done in {time.perf_counter() - t:.2f}s "
            f"(video_latent={tuple(latent.shape)}, "
            f"audio_latent={'yes' if audio_latent is not None else 'no'})"
        )

        # Audio decode runs serially - threading it against VAE chunk 1 was
        # tried and made total setup slower (MLX serializes work across
        # threads on the single Metal scheduler).
        if audio_latent is not None and args.audio:
            t = time.perf_counter()
            audio_track = _decode_audio_track(audio_latent, args.weights, compute_dtype)
            print(f"[setup] audio decode in {time.perf_counter() - t:.2f}s")
        # The audio latent is consumed at this point; free it (and any other
        # post-audio state) so the Metal heap is clean before VAE decode.
        del audio_latent
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

        t = time.perf_counter()
        decoder = make_video_decoder_default(
            args.weights, compute_dtype,
        )
        print(f"[setup] video VAE loaded in {time.perf_counter() - t:.2f}s")
        total_frames, in_h, in_w = latent_dims(latent)
        source_fps = args.source_fps
        src_w, src_h = in_w, in_h
        crop_box = None
        _edge_samples = None
        square_resample: tuple | None = None
        square_apply = None
        if args.crop_bars or args.crop_aspect:
            print("[crop] --crop-bars/--crop-aspect need --video; disabled")

        if args.snap_start or args.gop_align:
            print("[warn] --snap-start/--gop-align apply to --video only; ignored for --latent")
        # --start/--end trim the decoded frames. The VAE still decodes the whole
        # latent (it is temporally tiled, not seekable), so the window is
        # enforced in the main loop rather than at the reader.
        win_start, win_end = resolve_trim(args.start, args.end, source_fps, total_frames)
        _win_end_abs = win_end if win_end is not None else total_frames
        if win_start or win_end is not None:
            loop_win_start, loop_win_end = win_start, win_end
            total_frames = _win_end_abs - win_start
        if source_fps > 0:
            print(f"[setup] range: start {win_start} ({win_start / source_fps:.3f}s) -> "
                  f"end {_win_end_abs} ({_win_end_abs / source_fps:.3f}s), "
                  f"{total_frames} frames @ {source_fps:.3f} fps")
        else:
            print(f"[setup] range: frames [{win_start}, {_win_end_abs}), {total_frames} frames")

        if args.vae_tiling == "single":
            vae_cfg, n_vae_chunks, vae_tiling_desc = None, 1, "single (one decode)"
        else:
            vae_cfg, n_vae_chunks, vae_tiling_desc = plan_vae_tiling(latent)
        print(
            f"VAE tiling: {vae_tiling_desc} "
            f"({n_vae_chunks} chunk{'s' if n_vae_chunks != 1 else ''})"
        )
        # Always carry fp16 RGBA from MLX through to VSR - quantization
        # happens at the destination format, not earlier. For LL this means
        # CIContext quantizes once at NV12 render time (in YUV space) rather
        # than twice (in RGB then YUV). For HQ this preserves full bf16
        # precision into RGBAHalf.
        chunks = iter_latent_chunks(
            latent, decoder,
            cfg=vae_cfg, mx_mod=mx,
            output_format="fp16_rgba",
            single_pass=args.vae_tiling == "single",
        )
    else:
        from LTX_2_MLX.videotoolbox.vsr import source_format_for_mode

        # ---- reader selection: native (AVFoundation) unless forced or refused.
        # The ffmpeg compatibility reader mirrors the native surface for
        # containers/codecs AVFoundation cannot open (MKV, VP9, ...).
        global _vr
        if args.reader == "ffmpeg":
            from LTX_2_MLX.videotoolbox import ffmpeg_reader
            _vr = ffmpeg_reader
            print("[reader] ffmpeg compatibility reader (forced)")
        elif args.reader == "auto":
            try:
                _vr.probe_video(Path(args.video))
            except Exception as e:
                from LTX_2_MLX.videotoolbox import ffmpeg_reader
                _vr = ffmpeg_reader
                print(f"[reader] native reader cannot open this file "
                      f"({type(e).__name__}); using the ffmpeg compatibility reader")

        print(f"Reading video: {args.video}")
        in_w, in_h, source_fps, total_frames, src_transform, src_pixel_aspect = _vr.probe_video(
            Path(args.video),
        )
        _src_color = _vr.probe_color(Path(args.video))
        _resolved_color = _color.resolve(_src_color, args.source_color, args.source_range)
        color_props = _color.av_color_properties(_resolved_color)
        cv_color = _color.cv_triple(_resolved_color)
        output_full_range = _resolved_color[3]
        _origin = ("tagged" if _src_color["tagged"]
                   else "untagged, VT guessed" if _src_color.get("guessed")
                   else "untagged")
        print(f"Source color: {_origin} -> output {_color.describe(_resolved_color)}")
        if args.source_color != "auto":
            print(f"  (forcing the source to be DECODED as {args.source_color}, "
                  "overriding VideoToolbox's resolution guess)")
        if args.source_range != "auto":
            _container_range = "full" if _src_color["full_range"] else "video"
            if args.source_range == _container_range:
                print(f"  (--source-range {args.source_range} matches the container "
                      "flag; no reinterpretation needed)")
            else:
                print(f"  (reinterpreting the source's code values as {args.source_range} "
                      f"range, overriding the container's {_container_range}-range flag)")

        # ---- Bar crop + border sampling (before any dims are consumed) -----
        src_w, src_h = in_w, in_h
        if (src_w % 2) or (src_h % 2):
            print(f"[warn] source has ODD dimensions {src_w}x{src_h}: 4:2:0 "
                  "paths (--spatial-mode fast NV12 input; Main10/H.264 encodes "
                  "at 1x) may misalign chroma or fail downstream. Consider "
                  "--crop-bars 0,1,0,1-style trims to even them.")
        crop_box = None
        _edge_samples = None
        if args.crop_bars or args.sanitize_edges == "auto":
            _edge_samples, s_idx = [], 0
            want = set(range(0, 24, 4))
            for s_chunk in _vr.iter_video_buffer_chunks(
                    Path(args.video), _pb.PIX_RGBAHALF, chunk_size=8):
                for s_buf in s_chunk:
                    if s_idx in want:
                        _edge_samples.append(mx.clip(
                            _pb.read_pixel_buffer_rgb(s_buf).astype(mx.float32) / 255.0,
                            0, 1))
                    s_idx += 1
                if s_idx > max(want):
                    break
        if args.crop_bars:
            if args.crop_bars == "auto":
                bars = detect_bars(_edge_samples)
            else:
                bars = list(parse_edges_spec(args.crop_bars))
                # Keep the active area even (4:2:0 chroma / NV12 paths): bump
                # the bottom/right trim into the content by one px if needed.
                bumped = []
                if (in_h - bars[0] - bars[1]) % 2:
                    bars[1] += 1
                    bumped.append("bottom")
                if (in_w - bars[2] - bars[3]) % 2:
                    bars[3] += 1
                    bumped.append("right")
                if bumped:
                    print(f"[crop] bumped {'/'.join(bumped)} by 1 px so the "
                          "active area keeps even dimensions")
                bars = tuple(bars)
            if any(bars):
                if bars[0] + bars[1] >= in_h or bars[2] + bars[3] >= in_w:
                    raise SystemExit(f"--crop-bars {bars} leaves no active area")
                crop_box = bars
                in_h -= bars[0] + bars[1]
                in_w -= bars[2] + bars[3]
                print(f"[crop] bars: top={bars[0]} bottom={bars[1]} left={bars[2]} "
                      f"right={bars[3]} px -> active {in_w}x{in_h}")
                if _edge_samples:
                    _edge_samples = [_crop_rgb(s, bars) for s in _edge_samples]
            elif args.crop_bars == "auto":
                print("[crop] auto: no bars detected")

        # Junk-edge TRIM: fold detected junk lines into the crop instead of
        # filling them, BEFORE the aspect window is computed, so the aspect
        # math runs on the clean picture.
        if args.sanitize_edges and args.sanitize_edges_fill == "trim":
            if args.sanitize_edges == "auto":
                trim_edges, notices = detect_junk_edges(_edge_samples or [])
                for note in notices:
                    print(f"[sanitize] {note}")
            else:
                trim_edges = parse_edges_spec(args.sanitize_edges)
            if any(trim_edges):
                te = list(trim_edges)
                bumped = []
                if (in_h - te[0] - te[1]) % 2:
                    te[1] += 1
                    bumped.append("bottom")
                if (in_w - te[2] - te[3]) % 2:
                    te[3] += 1
                    bumped.append("right")
                if te[0] + te[1] >= in_h or te[2] + te[3] >= in_w:
                    raise SystemExit(
                        f"--sanitize-edges trim {tuple(te)} leaves no active area")
                if bumped:
                    print(f"[sanitize] trim bumped {'/'.join(bumped)} by 1 px so "
                          "the active area keeps even dimensions")
                te = tuple(te)
                base = crop_box or (0, 0, 0, 0)
                crop_box = tuple(b + a for b, a in zip(base, te, strict=True))
                in_h -= te[0] + te[1]
                in_w -= te[2] + te[3]
                print(f"[sanitize] trim: top={te[0]} bottom={te[1]} left={te[2]} "
                      f"right={te[3]} px cropped off -> active {in_w}x{in_h}")
                if _edge_samples:
                    _edge_samples = [_crop_rgb(s, te) for s in _edge_samples]
            else:
                print("[sanitize] trim: no junk edges detected")
        if args.crop_aspect:
            try:
                ar_w, ar_h = (int(p) for p in args.crop_aspect.split(":"))
            except ValueError:
                raise SystemExit(f"--crop-aspect must be W:H, got {args.crop_aspect!r}") from None
            try:
                dx, dy = (int(p) for p in args.crop_offset.split(","))
            except ValueError:
                raise SystemExit(f"--crop-offset must be dx,dy, got {args.crop_offset!r}") from None
            eff_w, eff_h = ar_w, ar_h
            if src_pixel_aspect and src_pixel_aspect[0] != src_pixel_aspect[1]:
                # The requested ratio is a DISPLAY aspect; on anamorphic
                # sources the storage-pixel target must fold the PAR in
                # (display = storage x PAR), or a "16:9" crop of 128:117-wide
                # pixels would display at ~1.95:1.
                eff_w = ar_w * src_pixel_aspect[1]
                eff_h = ar_h * src_pixel_aspect[0]
                print(f"[crop] aspect {ar_w}:{ar_h} at source pixel aspect "
                      f"{src_pixel_aspect[0]}:{src_pixel_aspect[1]} -> "
                      f"storage target {eff_w}:{eff_h}")
            asp = compute_aspect_crop(in_w, in_h, eff_w, eff_h, dx, dy,
                                      anchor=args.crop_anchor)
            if any(asp):
                base = crop_box or (0, 0, 0, 0)
                crop_box = tuple(b + a for b, a in zip(base, asp, strict=True))
                in_h -= asp[0] + asp[1]
                in_w -= asp[2] + asp[3]
                off = f" offset {dx:+d},{dy:+d}" if (dx or dy) else ""
                print(f"[crop] aspect {ar_w}:{ar_h} anchor {args.crop_anchor}{off}: "
                      f"window at x={crop_box[2]} y={crop_box[0]} "
                      f"-> active {in_w}x{in_h}")
                if _edge_samples:
                    _edge_samples = [_crop_rgb(s, asp) for s in _edge_samples]

        # ---- Square-pixel resample (anamorphic sources) --------------------
        # Horizontal-only bilinear at SOURCE resolution: the cheapest point,
        # and the upscaler re-synthesizes the mild resample softness. Output
        # is then tagged 1:1. Runs AFTER the crops (which are PAR-aware).
        square_resample: tuple | None = None
        square_apply = None
        if args.square_pixels and src_pixel_aspect and src_pixel_aspect[0] != src_pixel_aspect[1]:
            sq_w = int(round(in_w * src_pixel_aspect[0] / src_pixel_aspect[1]))
            sq_w -= sq_w % 2
            if sq_w != in_w and sq_w >= 2:
                square_resample = make_lanczos_plan(in_w, sq_w)
                square_apply = mx.compile(
                    lambda t: resample_width(t, square_resample))
                square_ratio = sq_w / in_w
                print(f"[square-pixels] pixel aspect {src_pixel_aspect[0]}:"
                      f"{src_pixel_aspect[1]} -> width {in_w} -> {sq_w} "
                      "(Lanczos-3 at source resolution); output tagged 1:1")
                in_w = sq_w
            src_pixel_aspect = None
        elif args.square_pixels:
            square_resample = None   # already square; no-op

        # Decode straight into VSR's source format (NV12 for fast, RGBAHalf for
        # balanced/image) and feed the buffers directly to VSR - no RGB
        # intermediate, no MLX round-trip, no re-quantization. Size the decode
        # chunk to a ~64 MiB budget so peak resident decoded frames stay bounded
        # regardless of resolution (1 frame for 4K RGBAHalf, more for small SD).
        vsr_src_fmt = (
            _pb.PIX_RGBAHALF if args.spatial_mode == "none"
            else source_format_for_mode(args.spatial_mode)
        )
        bytes_per_px = 8 if vsr_src_fmt == _pb.PIX_RGBAHALF else 2
        frame_bytes = max(1, src_w * src_h * bytes_per_px)   # decode happens at source size
        buf_chunk = max(1, min(args.video_chunk_size, (64 * 1024 * 1024) // frame_bytes))
        # --start/--end trim the input. The reader seeks to the window so the
        # head of a long clip is never decoded (frame-exact, see video_reader).
        win_start, win_end = resolve_trim(args.start, args.end, source_fps, total_frames)
        _orig_total = total_frames       # full-clip frame count (before trim)

        def _tc(fr: int) -> str:         # frame index -> "N (S.SSSs)"
            return f"{fr} ({fr / source_fps:.3f}s)" if source_fps > 0 else f"{fr}"

        # Keyframes are needed by --snap-start and/or --gop-align; detect once.
        _kf_all = (_vr.keyframe_display_indices(Path(args.video))
                   if (args.snap_start or args.gop_align) else None)
        if args.snap_start and _kf_all:
            _snap = min(_kf_all, key=lambda k: abs(k - win_start))
            if _snap != win_start:
                print(f"[snap] --start {_tc(win_start)} snapped to nearest keyframe "
                      f"{_tc(_snap)} (output begins there, not at the requested frame)")
                win_start = _snap
        _win_end_abs = win_end if win_end is not None else _orig_total
        total_frames = _win_end_abs - win_start
        # Always echo the effective range (frames + timecodes, ms precision).
        print(f"[setup] range: start {_tc(win_start)} -> end {_tc(_win_end_abs)}, "
              f"{total_frames} frames @ {source_fps:.3f} fps")

        # GOP-aligned windowing: plan windows whose boundaries land on keyframes
        # (both recurrence directions cold-start on a clean frame -> no trim needed).
        # One schedule drives every recurrent stage (they preserve frame positions).
        _read_start = win_start          # gop-align may extend the read back to a keyframe
        _gop_head_skip = 0               # context frames read before --start, output-dropped
        if args.gop_align:
            from LTX_2_MLX.videotoolbox.upscaler_base import plan_gop_windows
            _win_e = _win_end_abs
            # Anchor the first window on the keyframe enclosing --start: read from
            # it and feed [kf, start) as recurrence context (processed, not output),
            # so the forward pass cold-starts on a clean I-frame even on an
            # arbitrary start.
            _encl = [k for k in _kf_all if k <= win_start]
            _read_start = max(_encl) if _encl else win_start
            _gop_head_skip = win_start - _read_start
            _n_sched = _win_e - _read_start
            _kf = sorted({k - _read_start for k in _kf_all if _read_start <= k < _win_e})
            gop_schedule = plan_gop_windows(_kf, _n_sched, args.gop_min_window, args.gop_max_window)
            # ---- diagnostics: everything the scheduler saw + computed ----
            _kf_abs = [k for k in _kf_all if _read_start <= k < _win_e]
            # Anchor the first window on the keyframe enclosing --start: read from
            # it and feed [kf, start) as recurrence context (processed, not output),
            # so the forward pass cold-starts on a clean I-frame even on an
            # arbitrary start.
            _encl = [k for k in _kf_all if k <= win_start]
            _read_start = max(_encl) if _encl else win_start
            _gop_head_skip = win_start - _read_start
            _n_sched = _win_e - _read_start
            _kf = sorted({k - _read_start for k in _kf_all if _read_start <= k < _win_e})
            gop_schedule = plan_gop_windows(_kf, _n_sched, args.gop_min_window, args.gop_max_window)
            # ---- diagnostics: everything the scheduler saw + computed ----
            _kf_abs = [k for k in _kf_all if _read_start <= k < _win_e]
            _gaps = [_kf[i] - _kf[i - 1] for i in range(1, len(_kf))]
            _uniq = sorted(set(_gaps))
            _cadence = ("single-keyframe (no cadence -> fixed max-window tiling)" if not _gaps
                        else f"constant {_uniq[0]} frames" if len(_uniq) == 1
                        else f"variable: min {min(_gaps)} / median {sorted(_gaps)[len(_gaps) // 2]} / max {max(_gaps)}")
            _proc = sum(p1 - p0 for p0, p1, *_ in gop_schedule)
            _emit_n = sum(e1 - e0 for *_, e0, e1 in gop_schedule)
            print(f"[gop-align] source: {len(_kf_abs)} keyframes in frames "
                  f"[{_read_start}, {_win_e}); GOP cadence {_cadence}")
            print("[gop-align]   keyframe frames: "
                  + (str(_kf_abs) if len(_kf_abs) <= 24
                     else f"{_kf_abs[:8]} ... {_kf_abs[-3:]}  ({len(_kf_abs)} total)"))
            if _gop_head_skip:
                print(f"[gop-align]   WARNING: --start {win_start} is mid-GOP; reading from "
                      f"keyframe {_read_start} and feeding [{_read_start}, {win_start}) "
                      f"({_gop_head_skip} frames) as recurrence context (processed, NOT output)")
            print(f"[gop-align] planned {len(gop_schedule)} windows (min {args.gop_min_window} "
                  f"/ max {args.gop_max_window} frames), keyframe-anchored both ends, trim 0:")
            _show = (gop_schedule if len(gop_schedule) <= 12
                     else [*gop_schedule[:6], None, *gop_schedule[-3:]])
            for _w in _show:
                if _w is None:
                    print(f"[gop-align]   ... ({len(gop_schedule) - 9} more) ...")
                    continue
                _p0, _p1, _e0, _e1 = _w
                print(f"[gop-align]   proc[{_p0}:{_p1}] emit[{_e0}:{_e1}]  "
                      f"({_p1 - _p0} processed -> {_e1 - _e0} output)")
            _ovh = (_proc / _emit_n - 1.0) * 100 if _emit_n else 0.0
            print(f"[gop-align] {_proc} frames processed for {_emit_n} output "
                  f"({_ovh:.1f}% re-processing overhead vs trim's ~2x)")
        _force_read = args.source_color != "auto" or args.source_range != "auto"
        if args.source_range != "auto" and vsr_src_fmt != _pb.PIX_RGBAHALF:
            # The NV12 fast path feeds container-range-typed YUV straight into
            # VSR and the encoder; retyping it would make the encode session
            # rescale the values. No reinterpretation is possible there.
            raise SystemExit(
                "--source-range is not supported with --spatial-mode fast (the "
                "NV12 path passes YUV through without a range conversion point). "
                "Use --spatial-mode none/balanced/image or an MLX upscaler.")
        if _force_read and vsr_src_fmt == _pb.PIX_RGBAHALF:
            # Force the READ: decode raw YUV in the CONTAINER's range format
            # (pass-through code values), re-interpret with the chosen matrix
            # and range, overriding the container tag / VideoToolbox's
            # resolution-based guess. (NV12 'fast' keeps the default decode;
            # the LowLatency scaler consumes YUV directly.)
            chunks = _vr.iter_forced_color_chunks(
                Path(args.video), vsr_src_fmt, cv_color[2], _src_color["full_range"],
                chunk_size=buf_chunk, start_frame=_read_start, end_frame=win_end,
                reinterpret_full_range=output_full_range,
            )
        else:
            chunks = _vr.iter_video_buffer_chunks(
                Path(args.video), vsr_src_fmt, chunk_size=buf_chunk,
                start_frame=_read_start, end_frame=win_end,
            )
        n_vae_chunks = None  # no VAE on --video path

        # Carry the source file's audio through to the output MP4 (native
        # AVFoundation read; no ffmpeg). Latents decode audio from a latent
        # instead - see _decode_audio_track above. Trim + sidecar happen below,
        # uniformly for both paths.
        if args.audio:
            audio_track = _read_audio_track_from_video(Path(args.video))

    # ---- Audio trim + sidecar (uniform for both paths) ---------------------
    # When --start/--end trim the video, trim the audio to the same window so
    # the muxed track stays in sync (otherwise a short clip carries full-length
    # audio). The sidecar, if requested, reflects the trimmed audio.
    if audio_track is not None and (win_start or win_end is not None):
        a_start = win_start / source_fps
        a_end = (win_end / source_fps) if win_end is not None else None
        audio_track = audio_track.trimmed(a_start, a_end)
        print(
            f"[setup] audio trimmed to [{a_start:.3f}s, "
            f"{'end' if a_end is None else f'{a_end:.3f}s'})"
        )
    if audio_track is not None and args.save_audio_sidecar:
        sidecar = out_root / f"{stem}_audio.wav"
        audio_track.save_wav(sidecar)
        print(f"[setup] audio sidecar: {sidecar}")

    # ---- Output geometry + encoder settings --------------------------------
    from LTX_2_MLX.videotoolbox.vsr import scale_for_mode
    spatial_scale = 1 if args.spatial_mode == "none" else scale_for_mode(args.spatial_mode)
    if args.spatial_mode == "realesrgan":
        # realesrgan covers 2x (x2plus) as well as 4x models; read the real scale from the
        # checkpoint instead of assuming 4x, so output dims + encoder match the frames.
        from LTX_2_MLX.videotoolbox.realesrgan import net as _rnet
        spatial_scale = _rnet.scale_of(_rnet.load_params(
            _rnet.resolve_weights(args.realesrgan_weights or os.environ.get("REALESRGAN_WEIGHTS"))))
    elif args.spatial_mode == "safmn":
        from LTX_2_MLX.videotoolbox.safmn import net as _snet
        spatial_scale = _snet._config(_snet.load_params(args.safmn_weights))[3]
    elif args.spatial_mode == "esc":
        from LTX_2_MLX.videotoolbox.esc import net as _enet
        spatial_scale = _enet._config(_enet.load_params(args.esc_weights))[6]
    elif args.spatial_mode == "realplksr":
        # realplksr covers 2x (public2x) and 4x (nomos4x); read the scale from the
        # checkpoint so output dims + encoder match the frames.
        from LTX_2_MLX.videotoolbox.realplksr import net as _pnet
        spatial_scale = _pnet._config(_pnet.load_params(args.realplksr_weights))[4]
    out_w, out_h = in_w * spatial_scale, in_h * spatial_scale
    profile = _pick_hevc_profile(args.spatial_mode, args.encode_chroma)
    target_fps = args.target_fps if args.target_fps is not None else source_fps
    do_temporal = abs(target_fps - source_fps) > 1e-6
    # --max-frames caps OUTPUT frames; a time spec here is output duration at
    # the target fps. Parsed now that target_fps is known.
    try:
        max_frames = (
            parse_time_or_frames(args.max_frames, target_fps)
            if args.max_frames is not None else None
        )
    except ValueError as e:
        raise SystemExit(f"bad --max-frames value: {e}") from None

    print(
        f"Source: {in_w}x{in_h}, "
        f"total frames: {total_frames or 'unknown'}, "
        f"fps: {source_fps:.3f}"
    )
    if src_pixel_aspect is not None:
        print(f"Source pixel aspect: {src_pixel_aspect[0]}:{src_pixel_aspect[1]}")
    print(
        f"Target: {out_w}x{out_h} (spatial {spatial_scale}x), "
        f"fps: {target_fps:.3f}"
        f"{' (temporal upscale)' if do_temporal else ''}, "
        f"spatial-mode={args.spatial_mode}"
    )
    print(
        f"Encoder: HEVC profile={profile} q={args.encode_quality} "
        f"audio={args.audio_codec if audio_track else 'none'}"
    )

    # ---- Sessions + writers ------------------------------------------------
    # Defer constructing the VSR session, VtfrcSession, and AVWriters until
    # the *first* VAE chunk has materialized.  These hold Metal resources
    # - the HQ VSR model in particular pins ~100MB of Metal heap - so
    # creating them up front would compete with chunk-1 VAE decode for
    # the same unified-memory pool.  Lazy init via _build_post_pipeline.
    session: VsrSession | None = None
    vtfrc: VtfrcSession | None = None
    post_writer: AVWriter | None = None
    comparison_writer: AVWriter | None = None
    denoiser: Any = None  # SpatialDenoiser / McTemporalDenoiser / learned denoiser when --denoise set
    upscaler: Any = None  # learned MLX upscaler when --spatial-mode basicvsrpp/realbasicvsr
    # deblocker/nafnet pre-set (like denoiser/upscaler) so a _build_post_pipeline failure
    # -- e.g. a missing weight -- cannot UnboundLocalError in the cleanup finally.
    deblocker: Any = None  # STDF / FBCNN when --deblock set
    nafnet: Any = None     # NAFNet restorer when --nafnet set
    restorer: Any = None   # BasicVSR++ 1x recurrent restoration when --restore set

    def _build_post_pipeline() -> tuple[
        VsrSession, VtfrcSession | None, AVWriter | None, AVWriter | None, Any, Any, Any, Any
    ]:  # session, vtfrc, post_writer, comparison_writer, deblocker, denoiser, upscaler, nafnet
        """Materialize VSR + temporal + writer sessions just-in-time.

        Called on the first chunk so chunk-1 VAE has the Metal heap to
        itself.  Returns (session, vtfrc, post_writer, comparison_writer);
        the dst-pool wiring for zero-copy is set up before returning.
        """
        s: Any
        if args.spatial_mode == "none":
            s = NativePassthrough(in_w, in_h, fps=source_fps)
        elif args.spatial_mode in ("basicvsrpp", "realbasicvsr", "realesrgan", "safmn", "esc", "realviformer", "realplksr"):
            # Learned MLX upscalers do the upscale in the loop; the session is a
            # passthrough at the output dims that just packs the already-upscaled
            # frame for the encoder.
            s = NativePassthrough(out_w, out_h, fps=source_fps, label=f"{args.spatial_mode} packer")
        else:
            s = VsrSession(in_w, in_h, mode=args.spatial_mode, fps=source_fps)
        v: VtfrcSession | None = None
        if do_temporal:
            v = VtfrcSession(
                out_w, out_h,
                source_fps=source_fps, target_fps=target_fps,
                mode=args.temporal_mode,
            )
        audio_kwargs: dict[str, Any] = {}
        if audio_track is not None:
            audio_kwargs = {"audio_track": audio_track, "audio_codec": args.audio_codec}

        pw: AVWriter | None = None
        if not args.skip_post_mp4:
            # The producer feeding this writer's pool is the temporal session
            # if present, else VSR. Pass its dst attrs so the pool's buffers
            # carry the extended-pixel padding that producer requires (else
            # VTFrameProcessor rejects them with -19730 for some geometries).
            producer_attrs = v.dst_attrs if v is not None else s.dst_attrs
            pw = AVWriter(
                out_root / f"{stem}_post.mp4",
                width=out_w, height=out_h, fps=target_fps,
                source_pixel_format=_pb.resolve_pixel_format(producer_attrs),
                profile=profile,
                quality=args.encode_quality,
                label="post",
                transform=src_transform,
                source_attrs=producer_attrs,
                color_props=color_props,
                pixel_aspect=src_pixel_aspect,
                cv_color=cv_color,
                full_range=output_full_range,
                **audio_kwargs,
            )
            # The writer feeds the encoder 10-bit YUV it converts itself (yuv.py),
            # so its adaptor pool is YUV; the producer keeps its own RGBAHalf dst
            # pool (no use_dst_pool zero-copy -- the RGB->YUV conversion is the copy).

        cw: AVWriter | None = None
        if args.comparison:
            cw = AVWriter(
                out_root / f"{stem}_comparison.mp4",
                width=2 * out_w, height=out_h, fps=target_fps,
                source_pixel_format=_pb.PIX_BGRA,
                profile=profile,
                quality=args.encode_quality,
                label="comparison",
                color_props=color_props,
                pixel_aspect=src_pixel_aspect,
                **audio_kwargs,
            )

        # --noise-map auto: one tracker shared conceptually (one per denoiser in
        # practice); estimates a per-pixel sigma map from the footage. Only the
        # map-conditioned denoisers can consume it. --noise-map-pulse adds a
        # per-frame gain (GOP-phase noise pulsing) on the same conditioning.
        nm_tracker: Any = None
        nm_pulse: Any = None
        _map_capable = (args.denoise in ("fastdvd", "bsvd", "mc")
                        or (args.denoise == "pvdd" and "level" in args.pvdd_variant))
        if args.noise_map == "auto":
            if _map_capable:
                from LTX_2_MLX.videotoolbox.noise_map import NoiseMapTracker
                nm_tracker = NoiseMapTracker(gain=args.noise_map_gain)
                print(f"[noise-map] auto: per-pixel sigma estimated from the footage "
                      f"(gain {args.noise_map_gain:g})")
            elif args.denoise != "off":
                why = ("blind PVDD variant (use --pvdd-variant pvdd_level)"
                       if args.denoise == "pvdd" else f"--denoise {args.denoise} has no map input")
                print(f"[noise-map] auto ignored: {why}")
        if args.noise_map_pulse:
            if _map_capable:
                from LTX_2_MLX.videotoolbox.noise_map import PulseGain
                nm_pulse = PulseGain()
                print("[noise-map] pulse: per-frame gain tracking GOP-phase noise "
                      "(I-frame grain refresh)")
            elif args.denoise != "off":
                print("[noise-map] pulse ignored: needs a map-conditioned denoiser "
                      "(fastdvd, bsvd, mc, or a pvdd level variant)")

        den: Any = None
        if args.denoise == "spatial":
            den = SpatialDenoiser(strength=args.denoise_strength)
        elif args.denoise == "mc":
            den = McTemporalDenoiser(
                in_w, in_h, strength=args.denoise_strength,
                window=args.mc_window, clamp=args.mc_clamp,
                occlusion=args.mc_occlusion, confidence=args.mc_confidence,
                sigma=args.mc_sigma,
                noise_map=nm_tracker,
                map_refresh=args.noise_map_refresh,
                pulse=nm_pulse,
            )
        elif args.denoise == "fastdvd":
            # Weights ship with the package; --fastdvd-variant picks one, or
            # --fastdvd-weights / $FASTDVD_WEIGHTS overrides the path entirely.
            den = FastDvdDenoiser(
                args.fastdvd_weights or os.environ.get("FASTDVD_WEIGHTS"),
                variant=args.fastdvd_variant,
                strength=args.denoise_strength,
                noise_map=nm_tracker,
                map_refresh=args.noise_map_refresh,
                pulse=nm_pulse,
            )
        elif args.denoise == "bsvd":
            # BSVD weights are not bundled; --bsvd-variant picks a local token,
            # or --bsvd-weights / $BSVD_WEIGHTS overrides the path entirely.
            den = BsvdDenoiser(
                args.bsvd_weights or os.environ.get("BSVD_WEIGHTS"),
                variant=args.bsvd_variant,
                strength=args.denoise_strength,
                dtype=parse_mlx_dtype_name(args.bsvd_dtype),
                noise_map=nm_tracker,
                map_refresh=args.noise_map_refresh,
                pulse=nm_pulse,
            )
        elif args.denoise == "pvdd":
            # PVDD weights are not bundled; --pvdd-variant picks a local token, or
            # --pvdd-weights / $PVDD_WEIGHTS overrides the path. The `level`
            # variants take a noise-variance dial (--pvdd-noise-preset/-variance);
            # blind variants ignore it. --denoise-strength does not apply.
            from LTX_2_MLX.videotoolbox.pvdd.upscaler import PvddDenoiser
            from LTX_2_MLX.videotoolbox.pvdd import LEVEL_PRESETS
            nv = args.pvdd_noise_variance
            if nv is None and args.pvdd_noise_preset != "off":
                nv = LEVEL_PRESETS[args.pvdd_noise_preset]
            den = PvddDenoiser(
                args.pvdd_weights or os.environ.get("PVDD_WEIGHTS"),
                variant=args.pvdd_variant,
                window=args.pvdd_window, trim=args.pvdd_trim,
                noise_variance=nv,
                dtype=parse_mlx_dtype_name(args.pvdd_dtype),
                noise_map=nm_tracker,
                pulse=nm_pulse,
            )

        if den is not None and (args.denoise_luma_strength != 1.0
                                or args.denoise_chroma_strength != 1.0):
            kr, kb = _yuv.coef_for_matrix(_resolved_color[2])    # match the source color matrix
            den = LumaChromaDenoiser(den, args.denoise_luma_strength,
                                     args.denoise_chroma_strength, kr=kr, kb=kb)

        # --deblock-map auto: a per-pixel blockiness mask (grid-phase detected,
        # boundary-vs-interior contrast) gating the deblocker's correction, so
        # blocked flats get the full correction and detailed/clean areas keep
        # their texture.
        blk_tracker: Any = None
        if args.deblock_map == "auto":
            if args.deblock in ("stdf", "fbcnn"):
                from LTX_2_MLX.videotoolbox.noise_map import (
                    NoiseMapTracker, estimate_blockiness_map)
                blk_tracker = NoiseMapTracker(gain=args.deblock_map_gain, min_frames=1,
                                              estimator=estimate_blockiness_map)
                print(f"[deblock-map] auto: per-pixel blockiness mask on the "
                      f"correction (gain {args.deblock_map_gain:g})")
            elif args.deblock != "off":
                print(f"[deblock-map] auto ignored: --deblock {args.deblock} unsupported")

        deb: Any = None
        if args.deblock == "stdf":
            from LTX_2_MLX.videotoolbox.stdf.deblocker import StdfDeblocker
            kr, kb = _yuv.coef_for_matrix(_resolved_color[2])    # match the source color matrix
            deb = StdfDeblocker(args.deblock_weights or os.environ.get("STDF_WEIGHTS"),
                                strength=args.deblock_strength, kr=kr, kb=kb,
                                blockiness_map=blk_tracker)
        elif args.deblock == "fbcnn":
            from LTX_2_MLX.videotoolbox.fbcnn import FbcnnDeblocker
            deb = FbcnnDeblocker(args.deblock_weights or os.environ.get("FBCNN_WEIGHTS"),
                                 quality=args.fbcnn_quality, strength=args.fbcnn_strength,
                                 blockiness_map=blk_tracker)

        up: Any = None
        if args.spatial_mode == "basicvsrpp":
            from LTX_2_MLX.videotoolbox.basicvsrpp.upscaler import BasicVsrUpscaler
            # weights spec: --basicvsrpp-weights (token or path) > env > --variant token
            up = BasicVsrUpscaler(
                args.basicvsrpp_weights or os.environ.get("BASICVSRPP_WEIGHTS")
                or args.basicvsrpp_variant,
                window=args.basicvsrpp_window, trim=args.basicvsrpp_trim,
                flow_mode=args.basicvsrpp_flow_mode,
                history_strength=args.basicvsrpp_history_strength,
                history_gate=args.basicvsrpp_history_gate,
                ensemble=args.basicvsrpp_ensemble,
            )
        elif args.spatial_mode == "realbasicvsr":
            from LTX_2_MLX.videotoolbox.realbasicvsr.upscaler import RealBasicVsrUpscaler
            up = RealBasicVsrUpscaler(
                args.realbasicvsr_weights or os.environ.get("REALBASICVSR_WEIGHTS"),
                window=args.realbasicvsr_window,
                trim=args.realbasicvsr_trim,
                dynamic_refine_thres=args.realbasicvsr_dynamic_refine_thres,
                clean_iters=args.realbasicvsr_clean_iters,
                residual_strength=args.realbasicvsr_residual_strength,
                flow_consistency=args.realbasicvsr_flow_consistency,
                flow_mode=args.realbasicvsr_flow_mode,
                history_strength=args.realbasicvsr_history_strength,
                history_gate=args.realbasicvsr_history_gate,
            )
        elif args.spatial_mode == "realesrgan":
            from LTX_2_MLX.videotoolbox.realesrgan.upscaler import RealEsrganUpscaler
            up = RealEsrganUpscaler(
                args.realesrgan_weights or os.environ.get("REALESRGAN_WEIGHTS"),
                denoise_strength=args.realesrgan_denoise,
            )
        elif args.spatial_mode == "safmn":
            from LTX_2_MLX.videotoolbox.safmn import SafmnUpscaler
            up = SafmnUpscaler(args.safmn_weights, safm_up=args.safmn_safm_up,
                               pool_clamp=args.safmn_pool_clamp)
        elif args.spatial_mode == "esc":
            from LTX_2_MLX.videotoolbox.esc import EscUpscaler
            up = EscUpscaler(args.esc_weights)
        elif args.spatial_mode == "realplksr":
            from LTX_2_MLX.videotoolbox.realplksr import RealPlksrUpscaler
            up = RealPlksrUpscaler(args.realplksr_weights,
                                   dtype=parse_mlx_dtype_name(args.realplksr_dtype))
        elif args.spatial_mode == "realviformer":
            from LTX_2_MLX.videotoolbox.realviformer import RealViformerUpscaler
            up = RealViformerUpscaler(
                args.realviformer_weights, window=args.realviformer_window,
                dtype=parse_mlx_dtype_name(args.realviformer_dtype),
                flow_mode=args.realviformer_flow_mode,
                history_strength=args.realviformer_history_strength,
                history_gate=args.realviformer_history_gate,
                history_cleanup=args.realviformer_history_cleanup,
                history_gate_drop=args.realviformer_history_gate_drop,
                history_risk_decay=args.realviformer_history_risk_decay,
                history_static_cap=args.realviformer_history_static_cap,
            )

        naf: Any = None
        if args.nafnet != "off":
            from LTX_2_MLX.videotoolbox.nafnet import NafnetRestorer
            naf = NafnetRestorer(
                args.nafnet_weights or os.environ.get("NAFNET_WEIGHTS") or args.nafnet,
                strength=args.nafnet_strength,
                pool_mode=args.nafnet_pool,
                variant=args.nafnet,
                guard_mode=args.nafnet_guard,
                residual_guard=args.nafnet_guard_threshold,
                guard_fast_fraction=args.nafnet_guard_fast_fraction,
                guard_lockout_frames=args.nafnet_guard_lockout,
                guard_ramp_frames=args.nafnet_guard_ramp,
                guard_fall_frames=args.nafnet_guard_fall,
            )

        res: Any = None
        if args.restore != "off":
            from LTX_2_MLX.videotoolbox.basicvsrpp.restorer import BasicVsrRestorer
            # --restore accepts a comma-separated list that CHAINS restorers in one
            # pass (e.g. decompress_track1,denoise = temporal deblock then temporal
            # denoise). --restore-weights (a single path) only applies when one
            # variant is given; multi-variant chains use the bundled token files.
            variants = [v.strip() for v in args.restore.split(",") if v.strip()]
            try:
                strengths = [float(s) for s in str(args.restore_strength).split(",")]
            except ValueError:
                raise SystemExit(
                    f"--restore-strength must be float(s); got {args.restore_strength!r}") from None
            if len(strengths) == 1:
                strengths = strengths * len(variants)
            elif len(strengths) != len(variants):
                raise SystemExit(
                    f"--restore-strength: give 1 value or one per --restore stage "
                    f"({len(variants)}), got {len(strengths)}")
            wenv = os.environ.get("BASICVSRPP_RESTORE_WEIGHTS")
            res = []
            for variant, strength in zip(variants, strengths, strict=True):
                spec = variant
                if len(variants) == 1:
                    spec = args.restore_weights or wenv or variant
                res.append(BasicVsrRestorer(
                    spec, window=args.restore_window, trim=args.restore_trim,
                    strength=strength, flow_mode=args.restore_flow_mode,
                    ensemble=args.restore_ensemble))

        return s, v, pw, cw, deb, den, up, naf, res

    # ---- Synthetic-border sanitizer ----------------------------------------
    # Detect junk edge rows/cols (letterbox lines, capture garbage) and
    # replicate-fill them before ANY processor sees the frame; the learned
    # stages are trained on photographic content and hallucinate around
    # synthetic edges. Frame dims are untouched, so aspect and pixel-aspect
    # handling are unaffected.
    sanitize_edges: tuple | None = None
    if args.sanitize_edges and args.sanitize_edges_fill == "trim":
        if not args.video:
            print("[sanitize] trim mode needs --video; disabled")
        # otherwise handled in the crop pre-pass: junk folded into crop_box.
    elif args.sanitize_edges:
        if args.sanitize_edges == "auto":
            if _edge_samples is None:
                print("[sanitize] auto detection needs --video; disabled "
                      "(pass explicit T,B,L,R to force)")
            else:
                edges, notices = detect_junk_edges(_edge_samples)
                for note in notices:
                    print(f"[sanitize] {note}")
                if any(edges):
                    sanitize_edges = edges
                    print(f"[sanitize] junk edges detected: top={edges[0]} "
                          f"bottom={edges[1]} left={edges[2]} right={edges[3]} px "
                          "-- sanitized before processing")
                else:
                    print("[sanitize] auto: no junk edges detected")
        else:
            sanitize_edges = parse_edges_spec(args.sanitize_edges)
            print(f"[sanitize] manual edges: top={sanitize_edges[0]} "
                  f"bottom={sanitize_edges[1]} left={sanitize_edges[2]} "
                  f"right={sanitize_edges[3]} px")
    _edge_samples = None   # detection done; release the sampled frames

    # Output-side policy: a replicate fill that reaches the screen turns a
    # quiet static-dark border into moving light content. 1 px fills are
    # imperceptible (and remove the junk); wider bands get the ORIGINAL
    # border composited back over the processed frame.
    sanitize_restore: tuple | None = None
    if sanitize_edges is not None:
        restore = (0, 0, 0, 0) if args.sanitize_edges_fill == "extend" else sanitize_edges
        if any(restore) and square_resample is not None:
            # The restore composite runs on the resampled grid: scale the
            # left/right band widths by the resample ratio (feather hides
            # the sub-pixel rounding).
            restore = (restore[0], restore[1],
                       int(round(restore[2] * square_ratio)),
                       int(round(restore[3] * square_ratio)))
        if any(restore):
            sanitize_restore = restore
        names = ("top", "bottom", "left", "right")
        policy = ", ".join(
            f"{n}={d}px {'restore' if rr else 'extend'}"
            for n, d, rr in zip(names, sanitize_edges, restore, strict=True) if d
        )
        feather = (f", feather {args.sanitize_edges_feather}px"
                   if sanitize_restore is not None else "")
        print(f"[sanitize] fill policy: {policy}{feather}")

    # ---- Cut detector ------------------------------------------------------
    cut_detector: CutDetector | None = None
    cut_log = None
    if args.cut_detect != "off":
        cut_detector = CutDetector(args.cut_detect, args.cut_threshold)
        if args.cut_log:
            cut_log = open(args.cut_log, "w")
        print(
            f"Cut detector: mode={args.cut_detect} threshold={args.cut_threshold}"
            + (f", log={args.cut_log}" if args.cut_log else "")
        )

    # ---- Progress bars (stacked, deferred-start, median-window rate) -------
    # PhaseBar's clock starts at the first update() and the displayed pace
    # is the median of the last-N inter-tick intervals; see videotoolbox/
    # progress.py for the rationale. Plain tqdm gave both stacked bars the
    # same wall-clock elapsed (= total run time) and inflated the VSR rate
    # as the VAE-chunk-1 warmup amortized over a growing frame count.
    target_frame_total = total_frames
    if target_frame_total and do_temporal:
        target_frame_total = int(round(total_frames * (target_fps / source_fps)))
    pbar_total = (
        min(target_frame_total, max_frames)
        if (target_frame_total and max_frames is not None) else
        (max_frames or target_frame_total or None)
    )
    bars = StackedPhaseBars()
    vae_pbar = (
        bars.add(total=n_vae_chunks, desc="VAE chunks", unit="chunk")
        if n_vae_chunks is not None else None
    )
    out_pbar = bars.add(
        total=pbar_total,
        desc="OUT frames" if do_temporal else "VSR frames",
        unit="frame",
    )

    # ---- Pipeline loop -----------------------------------------------------
    processed = 0          # source frames upscaled
    appended = 0           # output frames written (= processed when no temporal)
    in_idx = 0             # input frame index (counts skipped pre-window frames)
    window_done = False    # set once the input window [loop_win_*) is exhausted
    t_total = time.perf_counter()

    def _emit(den_rgb: Any, src_frame: Any, src_arr: Any) -> None:
        """Upscale one frame (denoised ``den_rgb``, or raw ``src_frame`` when
        ``den_rgb`` is None) and write it out, with sidecars / FRC / comparison.
        Advances the processed and appended counters. Used for every output frame
        so the no-denoise, per-frame (spatial/mc), and lookahead (fastdvd) paths
        share one emit path."""
        nonlocal processed, appended, _gop_head_skip
        # gop-align enclosing-keyframe anchor: the frames read before --start were
        # fed to the recurrent stages as context only; drop their outputs here,
        # after every stage has consumed them.
        if _gop_head_skip > 0:
            _gop_head_skip -= 1
            return
        if den_rgb is not None:
            if sanitize_restore is not None and src_arr is not None:
                den_rgb = _restore_borders(den_rgb, src_arr, sanitize_restore,
                                           feather=args.sanitize_edges_feather)
            den_rgba = mx.concatenate(
                [den_rgb.astype(mx.float16),
                 mx.ones((den_rgb.shape[0], den_rgb.shape[1], 1), mx.float16)],
                axis=-1,
            )
            vsr_pb = session.upscale_to_buffer(den_rgba, processed)
        elif frames_are_buffers:
            vsr_pb = session.upscale_buffer_to_buffer(src_frame, processed)
        else:
            vsr_pb = session.upscale_to_buffer(src_frame, processed)

        if args.save_pre_frames:
            if src_arr.dtype != mx.uint8:
                pre_rgb_u8 = mx.clip(src_arr[..., :3] * 255.0, 0, 255).astype(mx.uint8)
            else:
                pre_rgb_u8 = src_arr if src_arr.shape[-1] == 3 else src_arr[..., :3]
            save_image(pre_rgb_u8, pre_dir / f"frame_{processed:05d}.png")
        if args.save_post_frames:
            save_image(
                _pb.read_pixel_buffer_rgb(vsr_pb), post_dir / f"frame_{processed:05d}.png",
            )

        out_iter = iter([vsr_pb]) if vtfrc is None else vtfrc.feed(vsr_pb, processed)
        for out_pb in out_iter:
            if max_frames is not None and appended >= max_frames:
                break
            if post_writer is not None:
                post_writer.append(out_pb)
            if comparison_writer is not None:
                comp_pb = _pb.make_bgra_buffer(comparison_writer.adaptor, 2 * out_w, out_h)
                render_comparison(src_arr, out_pb, spatial_scale, comp_pb)
                comparison_writer.append(comp_pb)
                del comp_pb
            del out_pb
            appended += 1
            out_pbar.update(1)
        del vsr_pb, out_iter
        processed += 1

    def _emit_scaled(den_rgb: Any, src_frame: Any, src_arr: Any) -> None:
        """For learned MLX upscalers, route the (denoised or raw) LR frame
        through the windowed recurrent upscaler and emit each 4x frame it
        releases; otherwise emit straight through. Preserves denoise -> upscale
        ordering and keeps one emit path for both the per-frame and flush cases."""
        if upscaler is None:
            _emit(den_rgb, src_frame, src_arr)
            return
        if den_rgb is not None:
            lr = den_rgb
        elif frames_are_buffers:
            lr = _pb.read_buffer_rgb_f32(src_frame)
        else:
            lr = src_frame[..., :3].astype(mx.float32)
        for up_rgb, (u_sf, u_sa) in upscaler.feed(lr, token=(None, src_arr)):
            _emit(up_rgb, u_sf, u_sa)

    def _pp_stages() -> list:
        """Enabled preprocessors in order. Default: deblock (compression) then denoise
        (analog) then a NAFNet detail/deblur pass; --denoise-first swaps the first two;
        --preprocess-order sets the full explicit order (any enabled stage omitted from it
        is appended in the default order). Only enabled (non-None) stages run."""
        by_name = {"restore": restorer, "deblock": deblocker, "denoise": denoiser, "nafnet": nafnet}
        if args.preprocess_order:
            order = list(args.preprocess_order)
        else:
            order = ["denoise", "deblock"] if args.denoise_first else ["deblock", "denoise"]
            # BasicVSR++ restoration is temporal; run it FIRST by default so it
            # establishes frame-to-frame consistency (killing GOP-periodic
            # compression flicker + sensor noise) before any per-frame stage.
            order = ["restore", *order]
        for name in _PP_STAGE_NAMES:                  # append any enabled stage not listed
            if name not in order:
                order.append(name)
        stages: list = []
        for n in order:
            s = by_name[n]
            if s is None:
                continue
            stages.extend(s if isinstance(s, list) else [s])   # "restore" may be a chain
        return stages

    def _stage_feed(stage: Any, rgb: Any, tok: Any) -> list:
        """Push one frame through a preprocessor -> [(rgb, tok), ...]. feed/flush delay
        lines (deblock, fastdvd) buffer; per-frame denoisers (spatial, mc) emit in step."""
        if hasattr(stage, "feed"):
            return stage.feed(rgb, token=tok)
        return [(stage.denoise(rgb), tok)]

    def _preprocess(base_rgb: Any, src_arr: Any) -> list:
        """Chain the enabled preprocessors in order; returns [(rgb, (None, src_arr)),
        ...] for _emit_scaled. Any stage may buffer now; tails drain via _preprocess_flush."""
        items = [(base_rgb, (None, src_arr))]
        for stage in _pp_stages():
            items = [r for rgb, tok in items for r in _stage_feed(stage, rgb, tok)]
        return items

    def _preprocess_flush() -> list:
        """Drain each stage's lookahead tail in order, feeding it through the stages
        downstream of it so a flushed tail still passes through later preprocessors."""
        stages = _pp_stages()
        out: list = []
        for i, stage in enumerate(stages):
            if not hasattr(stage, "flush"):
                continue
            for rgb, tok in stage.flush():
                items = [(rgb, tok)]
                for ds in stages[i + 1:]:
                    items = [r for rr, tt in items for r in _stage_feed(ds, rr, tt)]
                out += items
        return out

    try:
        for chunk in chunks:
            # Both paths yield a list, freed per-frame as consumed: the latent
            # path a list of MLX arrays (fp16 RGBA), the --video path a list of
            # decoded CVPixelBuffers in VSR's source format (fed to VSR direct).
            chunk_len = len(chunk)
            frames_are_buffers = not isinstance(chunk[0], mx.array)
            if frames_are_buffers:
                t_w, t_h = _pb.buffer_dims(chunk[0])
            else:
                t_h, t_w = int(chunk[0].shape[0]), int(chunk[0].shape[1])
            if (t_w, t_h) != (src_w, src_h):
                raise RuntimeError(
                    f"chunk dims {t_w}x{t_h} don't match the source {src_w}x{src_h}"
                )
            if vae_pbar is not None:
                vae_pbar.update(1)
            # Lazy init of the post-VAE pipeline.  Doing this *after* chunk 1
            # materialized keeps the VSR HQ model + AVWriter pixel pool out of
            # the Metal heap during chunk-1 VAE decode (which is the most
            # memory-contended step of the run).
            #
            # _build_post_pipeline() prints status from VsrSession / AVWriter
            # constructors; route those through bars.write() so they appear
            # above the live progress bars instead of stomping mid-line.
            if session is None:
                import contextlib
                import io as _io
                _buf = _io.StringIO()
                with contextlib.redirect_stdout(_buf):
                    session, vtfrc, post_writer, comparison_writer, deblocker, denoiser, upscaler, nafnet, restorer = _build_post_pipeline()
                msg = _buf.getvalue().rstrip("\n")
                if msg:
                    bars.write(msg)
                if nafnet is not None and hasattr(nafnet, "set_progress_message"):
                    nafnet.set_progress_message(bars.write)
                # Drive every recurrent stage from the one GOP-aligned schedule.
                # Per-frame stages lack the method and are skipped.
                if gop_schedule is not None:
                    for _st in [*(restorer or []),
                                *([denoiser] if denoiser is not None else []),
                                *([upscaler] if upscaler is not None else [])]:
                        if hasattr(_st, "set_schedule"):
                            _st.set_schedule(gop_schedule)
            for i in range(chunk_len):
                if max_frames is not None and appended >= max_frames:
                    break
                # Input-frame window (latent path; --video already trimmed at
                # the reader, where these bounds are 0/None). Skip frames before
                # the window, and stop once past it.
                if in_idx < loop_win_start:
                    chunk[i] = None
                    in_idx += 1
                    continue
                if loop_win_end is not None and in_idx >= loop_win_end:
                    window_done = True
                    break
                # Wrap the per-frame body in a fresh ObjC autorelease pool so
                # transient autoreleased objects (NSData, CIImage, CIImage
                # affine-translated, CIImage composited, CIContext render
                # intermediates, ...) drain at the end of each iteration
                # instead of piling up on the process top-level pool until
                # the interpreter exits. Without this the RSS climbs
                # unboundedly during long runs even though Python refcounts
                # are tracking correctly - PyObjC just doesn't drain
                # autoreleased ObjC objects on Python GC.
                with autorelease_pool():
                    src_frame = chunk[i]

                    # The upscale itself never needs an MLX array on the buffer
                    # path - the decoded buffer feeds VSR directly. Only
                    # materialize a uint8 RGB array when a feature consumes the
                    # source pixels: cut detection, --save-pre-frames, or the
                    # --comparison composite. For the latent path src_frame is
                    # already the array.
                    src_arr = None
                    if (
                        cut_detector is not None
                        or args.save_pre_frames
                        or comparison_writer is not None
                        or sanitize_restore is not None
                    ):
                        src_arr = (
                            _pb.read_pixel_buffer_rgb(src_frame)
                            if frames_are_buffers else src_frame
                        )
                        if crop_box is not None:
                            src_arr = _crop_rgb(src_arr, crop_box)
                        if square_apply is not None:
                            sa = (src_arr.astype(mx.float32) / 255.0
                                  if src_arr.dtype == mx.uint8
                                  else src_arr[..., :3].astype(mx.float32))
                            src_arr = square_apply(sa)

                    if cut_detector is not None and cut_detector.is_cut(src_arr):
                        # Flush the lookahead deblock + denoiser's buffered (pre-cut)
                        # frames before resetting so no window bridges the cut.
                        if deblocker is not None or (denoiser is not None and hasattr(denoiser, "flush")):
                            for d_rgb, (d_sf, d_sa) in _preprocess_flush():
                                _emit_scaled(d_rgb, d_sf, d_sa)
                        if upscaler is not None:
                            for up_rgb, (u_sf, u_sa) in upscaler.flush():
                                _emit(up_rgb, u_sf, u_sa)
                        session.reset_temporal_context()
                        if deblocker is not None:
                            deblocker.reset()
                        if denoiser is not None:
                            denoiser.reset()
                        if nafnet is not None:
                            nafnet.reset()
                        if cut_log is not None:
                            cut_log.write(f"{processed}\n")
                            cut_log.flush()

                    # Produce this input's output frame(s). No denoise -> the raw
                    # frame goes straight to VSR; spatial/mc denoise one frame in
                    # step; fastdvd buffers and emits centered-window frames once
                    # their two future neighbours have arrived (feed() may return
                    # nothing now; the tail drains after the loop). f32 RGB [0,1].
                    if (deblocker is not None or denoiser is not None
                            or nafnet is not None or restorer is not None
                            or sanitize_edges is not None
                            or crop_box is not None or square_apply is not None):
                        # fp16-preserving for RGBAHalf (balanced/image/none); 8-bit
                        # CoreImage fallback for NV12 (fast). Deblock (compression)
                        # runs before denoise (analog), both ahead of the upscaler.
                        if frames_are_buffers:
                            base_rgb = _pb.read_buffer_rgb_f32(src_frame)
                        else:
                            base_rgb = src_frame[..., :3].astype(mx.float32)
                        if crop_box is not None:
                            base_rgb = _crop_rgb(base_rgb, crop_box)
                        if sanitize_edges is not None:
                            base_rgb = _sanitize_rgb(base_rgb, sanitize_edges)
                        if square_apply is not None:
                            base_rgb = square_apply(base_rgb)
                        ready = _preprocess(base_rgb, src_arr)
                    else:
                        ready = [(None, (src_frame, src_arr))]
                    for d_rgb, (d_sf, d_sa) in ready:
                        _emit_scaled(d_rgb, d_sf, d_sa)

                    in_idx += 1
                    # Drop this frame's reference so its memory (the MLX array,
                    # or the decoded CVPixelBuffer on the --video path) can be
                    # freed now instead of staying resident until the outer
                    # `del chunk` at chunk-end.
                    chunk[i] = None
                    del src_frame, src_arr
                # autorelease pool drains here; PyObjC objects created in
                # this iteration are released back to the system.

                # Periodic janitorial work: CIContext caches grow with
                # render calls, and CVPixelBufferPools accumulate cached
                # buffers that the workload no longer needs.
                if processed % 64 == 0:
                    _pb.clear_ci_caches()
                    session.flush_pools()

            if window_done or (max_frames is not None and appended >= max_frames):
                break
            del chunk
            gc.collect()
        # Drain any frames a lookahead denoiser still holds (the centered-window
        # tail, with the standard reflected window at the clip's end).
        if (deblocker is not None or restorer is not None
                or (denoiser is not None and hasattr(denoiser, "flush"))) and session is not None:
            for d_rgb, (d_sf, d_sa) in _preprocess_flush():
                if max_frames is not None and appended >= max_frames:
                    break
                with autorelease_pool():
                    _emit_scaled(d_rgb, d_sf, d_sa)
        # Drain the learned upscaler's final window (the clip-end frames).
        if upscaler is not None and session is not None:
            for up_rgb, (u_sf, u_sa) in upscaler.flush():
                if max_frames is not None and appended >= max_frames:
                    break
                with autorelease_pool():
                    _emit(up_rgb, u_sf, u_sa)
        # Drain the frame-rate converter's tail: the final source period's target
        # frames, which feed() can never emit (no next source frame arrives). Held
        # copies of the last frame; no comparison composite (there is no distinct
        # source frame to composite against).
        if vtfrc is not None:
            for out_pb in vtfrc.drain():
                if max_frames is not None and appended >= max_frames:
                    break
                if post_writer is not None:
                    post_writer.append(out_pb)
                del out_pb
                appended += 1
                out_pbar.update(1)
    finally:
        bars.close()
        if vtfrc is not None:
            vtfrc.close()
        if session is not None:
            session.close()
        if denoiser is not None:
            denoiser.close()
        if deblocker is not None and args.deblock_map == "auto":
            _bm = getattr(deblocker, "last_blockiness_map", None)
            if _bm is not None:
                _s = mx.sort(_bm.reshape(-1))
                _n = _s.shape[0]
                print(f"[deblock-map] blockiness mask: median {float(_s[_n // 2]):.3f}  "
                      f"p95 {float(_s[int(0.95 * (_n - 1))]):.3f}  max {float(_s[-1]):.3f}  "
                      f"({float(mx.mean((_bm > 0.5).astype(mx.float32))) * 100:.0f}% of frame > 0.5)")
                if args.noise_map_debug and post_writer is not None:
                    from LTX_2_MLX.videotoolbox.images import save_image
                    _vp = Path(post_writer.path)
                    _png = _vp.with_name(_vp.stem + "_blockmap.png")
                    _u8 = (mx.clip(_bm[:, :, 0], 0, 1) * 255).astype(mx.uint8)
                    save_image(mx.stack([_u8, _u8, _u8], axis=-1), _png)
                    print(f"[deblock-map] mask written: {_png}")
        if denoiser is not None and args.noise_map_pulse:
            _pl = getattr(denoiser, "_pulse_log", None)
            if _pl is None:
                _pl = getattr(getattr(denoiser, "_base", None), "_pulse_log", None)
            if _pl:
                _ps = sorted(_pl)
                print(f"[noise-map] pulse gain over {len(_ps)} frames: "
                      f"min {_ps[0]:.2f}  median {_ps[len(_ps) // 2]:.2f}  "
                      f"max {_ps[-1]:.2f}  ({sum(1 for g in _ps if g > 1.2)} frames > 1.2)")
        if denoiser is not None and args.noise_map == "auto":
            # surface the estimated map (unwrap the luma/chroma splitter if present)
            _nm_src = getattr(denoiser, "last_noise_map", None)
            if _nm_src is None:
                _nm_src = getattr(getattr(denoiser, "_base", None), "last_noise_map", None)
            if _nm_src is not None:
                _s = mx.sort(_nm_src.reshape(-1))
                _n = _s.shape[0]
                print(f"[noise-map] estimated sigma: min {float(_s[0]):.4f}  "
                      f"median {float(_s[_n // 2]):.4f}  p95 {float(_s[int(0.95 * (_n - 1))]):.4f}  "
                      f"max {float(_s[-1]):.4f}")
                if args.noise_map_debug and post_writer is not None:
                    from LTX_2_MLX.videotoolbox.images import save_image
                    _vp = Path(post_writer.path)
                    _png = _vp.with_name(_vp.stem + "_noisemap.png")
                    _u8 = (mx.clip(_nm_src[:, :, 0] / 0.15, 0, 1) * 255).astype(mx.uint8)
                    save_image(mx.stack([_u8, _u8, _u8], axis=-1), _png)
                    print(f"[noise-map] map written: {_png}")
        if upscaler is not None and hasattr(upscaler, "close"):
            upscaler.close()
        if nafnet is not None:
            nafnet.close()
        for writer in (post_writer, comparison_writer):
            if writer is not None:
                writer.finish()
        if cut_log is not None:
            cut_log.close()

    elapsed = time.perf_counter() - t_total
    rate = appended / elapsed if elapsed > 0 else 0
    print(f"Processed {processed} source frames, wrote {appended} output frames "
          f"in {elapsed:.2f}s ({rate:.2f} fps out)")
    if post_writer is not None:
        print(f"Post: {post_writer.path}")
    if comparison_writer is not None:
        print(f"Comparison: {comparison_writer.path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--latent", help="--save-latents NPZ sidecar (VAE-decoded first).")
    src.add_argument("--video", help="Already-decoded video file (mp4/mov/...).")

    parser.add_argument(
        "--latent-stage",
        choices=["final", "stage1", "stage2"],
        default="final",
        help=(
            "Which latent to decode from a distilled-two-stage sidecar. "
            "'final' (default) = final_video_latent = stage 2 (the upscaled+refined result). "
            "'stage1' = pre-upscaler half-resolution latent. "
            "'stage2' = explicit stage 2 (same content as 'final' on distilled two-stage)."
        ),
    )
    parser.add_argument("--weights", help="LTX-2 .safetensors path (required with --latent).")
    parser.add_argument(
        "--vae-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16",
    )
    parser.add_argument(
        "--vae-tiling", choices=["auto", "single"], default="auto",
        help=(
            "auto (default) lets TilingConfig.auto_native_conv3d size to RAM + the "
            "int32 conv3d boundary (one decode if it fits, else bounded tiles). single "
            "forces one decode; frames past the int32 boundary decode white."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--output-prefix", default="vsr",
        help="Filename prefix for the timestamped outputs (matches generate.py).",
    )
    parser.add_argument(
        "--source-fps", type=float, default=NATIVE_FPS,
        help=(
            f"Source frame rate for --latent (latents don't carry an fps; "
            f"default {NATIVE_FPS} matches generate.py). Ignored for --video - "
            f"the input file's r_frame_rate is honored instead. Pair with "
            f"--target-fps to drive temporal frame-rate conversion."
        ),
    )
    parser.add_argument(
        "--target-fps", type=float, default=None,
        help=(
            "Target output fps. Defaults to the source fps (no temporal upscale). "
            "Setting a different value routes VSR output through "
            "VTFrameRateConversionConfiguration, motion-interpolating to the "
            "target rate. Arbitrary float values supported; 24->60, 15->30, "
            "30->24 (downsample), etc. The CMTime base is 24000 so common rates "
            "land bit-exact."
        ),
    )
    parser.add_argument(
        "--temporal-mode", choices=["normal", "high"], default="normal",
        help=(
            "VTFrameRateConversion mode. Only active when --target-fps is set. "
            "normal (default) = fast and adequate for 2x rate-up; "
            "high = QualityPrioritizationQuality, more compute for cleaner motion."
        ),
    )
    parser.add_argument(
        "--spatial-mode",
        choices=["fast", "balanced", "image", "none", "basicvsrpp", "realbasicvsr",
                 "realesrgan", "safmn", "esc", "realviformer", "realplksr"],
        default="balanced",
        help=(
            "VSR spatial mode.  Scale factor is implied by the mode (fast=2x, "
            "balanced=4x, image=4x, none=1x, basicvsrpp=4x, realbasicvsr=4x, "
            "realesrgan=4x, safmn=4x, esc=4x, realviformer=4x, realplksr=2x/4x). "
            "realplksr = MLX RealPLKSR per-frame SR (Phhofm checkpoints; scale from "
            "the checkpoint -- nomos4x is 4x, public2x is 2x). Single-image, no "
            "pooled modulation gate so it cannot produce SAFMN's block lattice; "
            "choose the checkpoint with --realplksr-weights. "
            "realesrgan = MLX Real-ESRGAN / ESRGAN RRDBNet 4x per-frame SR "
            "(single-image: no temporal propagation, so no flow ghosting; choose "
            "the checkpoint with --realesrgan-weights). "
            "basicvsrpp = MLX BasicVSR++ 4x super-resolution (recurrent, learned); "
            "realbasicvsr = MLX RealBasicVSR 4x real-world video SR with an "
            "iterative cleaning stage before BasicVSR propagation; "
            "runs on the GPU via MLX instead of VideoToolbox, processed in sliding "
            "windows (see the model-specific --*-window/--*-trim flags). "
            "Slower than the VT modes but uses learned SR checkpoints. "
            "none = no super-resolution; output stays at native resolution. Use "
            "it with --denoise to denoise-only, or alone for a plain transcode. "
            "balanced (default) = HQ Video mode; uses prev source + prev output "
            "frames to inform the upscale.  Tends to produce crisper motion edges "
            "at the cost of slightly more frame-to-frame variation in detail. "
            "fast = VTLowLatency 2x scaler.  Per-frame, no temporal context.  "
            "Input must be 96x96 to 960x960. "
            "image = HQ Image mode.  Per-frame deterministic upscale with no "
            "prev-frame feedback.  Slightly softer per-frame detail than balanced, "
            "but measurably smoother frame-to-frame (lower temporal second-difference). "
            "Apple documents this as for stills; on video it's a legitimate "
            "alternative to balanced if you prefer the smoother trade-off."
        ),
    )
    parser.add_argument(
        "--encode-quality", type=float, default=0.65,
        help="AVVideoQualityKey (0..1) for the HEVC encoder. 0.65 matches the default tier.",
    )
    parser.add_argument(
        "--source-color", choices=["auto", "bt709", "bt601", "bt2020"], default="auto",
        help=(
            "How the source is interpreted. auto (default) trusts the container's "
            "color tags, or VideoToolbox's resolution guess when untagged (SD "
            "width -> BT.601, HD -> BT.709). bt601/bt709/bt2020 FORCE the source to "
            "be decoded as that matrix -- the fix for untagged/mistagged clips VT "
            "guesses wrong. For balanced/image it decodes raw YUV and re-interprets "
            "it (not just an output tag); fast/NV12 keeps VideoToolbox's decode and "
            "only re-tags. The output is tagged to match."
        ),
    )
    parser.add_argument(
        "--source-range", choices=["auto", "video", "full"], default="auto",
        help=(
            "How the source's YUV code values map to RGB. auto (default) trusts "
            "the container's range flag (untagged = video/limited range, the "
            "standard assumption -- Y 16-235). video/full FORCE the interpretation "
            "for mis-flagged sources, e.g. full-range screen recordings or "
            "phone/webcam MP4s an encoder left untagged: read as limited they get "
            "shadows crushed to black and highlights clipped BEFORE any model runs "
            "(and the learned upscalers respond badly to the clipped regions). The "
            "reinterpretation is exact -- same decoded code values, different "
            "YUV->RGB scaling -- and the output is tagged to match. Not available "
            "with --spatial-mode fast (its YUV passes through untouched)."
        ),
    )
    parser.add_argument(
        "--encode-chroma", choices=["auto", "420", "422"], default="auto",
        help=(
            "HEVC profile chroma subsampling. auto = 4:2:2 (Main42210) for "
            "balanced/image modes, 4:2:0 (Main10) for fast. 420 forces Main10 for "
            "generate.py-tier parity."
        ),
    )
    parser.add_argument(
        "--audio", action="store_true",
        help=(
            "Mux audio into both MP4s. For --latent the audio is decoded from "
            "final_audio_latent (audio VAE + vocoder); for --video the source "
            "file's audio track is read natively (AVFoundation) and carried "
            "through. A --video input with no audio track stays silent."
        ),
    )
    parser.add_argument(
        "--audio-codec", choices=["alac", "aac"], default="alac",
        help="Audio codec for muxed audio (alac=lossless, aac=256kbps).",
    )
    parser.add_argument(
        "--save-audio-sidecar", action="store_true",
        help="Also write the muxed audio as <stem>_audio.wav next to the MP4s.",
    )
    parser.add_argument(
        "--restore", default="off", metavar="off|VARIANT",
        help=(
            "Pre-upscale TEMPORAL restoration (BasicVSR++, recurrent). Unlike the "
            "per-frame deblock/denoise stages, this is bidirectional + second-order "
            "over a frame window, so it enforces frame-to-frame consistency -- the fix "
            "for GOP-periodic compression flicker (artifacts that appear/disappear "
            "between keyframes) and temporally-varying sensor noise that per-frame "
            "models pulse on. Runs FIRST by default (before deblock/denoise/nafnet) so "
            "it stabilizes the sequence before any per-frame stage. Accepts a "
            "COMMA-SEPARATED list to chain restorers in one pass, in order -- e.g. "
            "'decompress_track1,denoise' runs temporal deblock then temporal denoise. "
            "Variants: off "
            "(default); decompress_track1 (NTIRE'21 compressed-video enhancement, the "
            "H.264/HEVC deblock choice) / decompress_track2 / decompress_track3; "
            "denoise (temporal video denoise -- softens far less than a per-frame "
            "denoiser); deblur_dvd / deblur_gopro (real / synthetic motion deblur). "
            "Weights downloaded, not bundled -- see basicvsrpp/weights/README.md. "
            "Domain note: trained on HEVC/synthetic degradations, so temporal-right "
            "but domain-approximate for 2010-era H.264."
        ),
    )
    parser.add_argument(
        "--restore-weights", default=None, metavar="VARIANT|PATH",
        help="Weights for --restore: a restoration variant token or a .safetensors path "
             "(or $BASICVSRPP_RESTORE_WEIGHTS). Overrides the --restore token's default file.",
    )
    parser.add_argument(
        "--restore-strength", type=str, default="1.0", metavar="S[,S...]",
        help="Blend each restored frame with the original (1.0 = full restore, default; "
             "0.0 = passthrough). A single value applies to every chained --restore stage; "
             "a comma-separated list sets per-stage strengths in order and must match the "
             "number of --restore variants -- e.g. --restore decompress_track1,denoise "
             "--restore-strength 1.0,0.5 = full deblock then a gentle denoise.",
    )
    parser.add_argument(
        "--restore-window", type=int, default=14, metavar="N",
        help="Sliding-window length (frames) for --restore recurrence (default 14). "
             "Longer = more temporal context (better consistency) but more memory/compute.",
    )
    parser.add_argument(
        "--restore-trim", type=int, default=2, metavar="N",
        help="Warm-up frames trimmed at each --restore window join (default 2); the "
             "propagation's transient edge. Interior frames match the full-clip result.",
    )
    parser.add_argument(
        "--restore-flow-mode", choices=["spynet", "zero", "vt"], default="spynet",
        help="Optical-flow source for --restore alignment: spynet (default, the trained "
             "flow net), zero (no motion), or vt (VideoToolbox flow).",
    )
    parser.add_argument(
        "--snap-start", action="store_true",
        help=(
            "Snap --start to the nearest source keyframe, so the output actually "
            "BEGINS on a clean I-frame (a clean editorial cut, a pristine first frame, "
            "and no context-decode overhead). This MOVES the start -- you get a "
            "slightly different range than requested -- so it is warned and the "
            "effective range is echoed. Contrast --gop-align's anchor, which keeps "
            "your exact --start and reads the pre-start frames as context. --video only."
        ),
    )
    parser.add_argument(
        "--gop-align", action="store_true",
        help=(
            "Align recurrent windows (--restore and the basicvsrpp/realbasicvsr/"
            "realviformer upscalers) to the source's GOP: detect keyframes natively "
            "(no ffprobe) and window keyframe-to-keyframe so BOTH recurrence "
            "directions cold-start on a clean I-frame -- no warm-up trim needed. "
            "Overhead is ~1 re-processed frame per GOP (a few percent) instead of "
            "trim's ~2x, and it removes window-boundary flicker on GOP-structured "
            "video. Falls back to fixed max-window tiling when the source has no "
            "keyframe cadence. --video only. Overrides the per-stage --*-window/-trim."
        ),
    )
    parser.add_argument(
        "--gop-min-window", type=int, default=16, metavar="N",
        help="Minimum recurrent window (frames) under --gop-align (default 16). A "
             "window spans whole GOPs until it reaches this, so short GOPs are merged "
             "for enough temporal context.",
    )
    parser.add_argument(
        "--gop-max-window", type=int, default=96, metavar="N",
        help="Maximum recurrent window (frames) under --gop-align (default 96). A GOP "
             "longer than this is split into sub-windows (with a small trim at the "
             "internal splits) to bound memory.",
    )
    parser.add_argument(
        "--restore-ensemble", action="store_true",
        help="Run --restore through the reference's 8-way geometric self-ensemble "
             "(the SpatialTemporalEnsemble the NTIRE decompress/ntire-vsr configs apply "
             "at inference, dropped as dead config in the mmagic re-port): restore under "
             "8 flip/rotate variants and average. Cancels orientation-specific "
             "hallucinated texture -- measured ~2.5x less flat-region 'alligator skin' "
             "and ~1.7x less temporal crawl on the aggressive decompress_track2 -- at 8x "
             "the compute. Off by default; worth it when a variant hallucinates on smooth "
             "surfaces and you still want its aggression. Applies to every chained "
             "--restore stage.",
    )
    parser.add_argument(
        "--deblock", choices=["off", "stdf", "fbcnn"], default="off",
        help=(
            "Pre-upscale compression-artifact deblock, applied before denoise + VSR "
            "(deblock before SR amplifies the blocking). off (default); stdf = STDF "
            "deformable spatio-temporal fusion (HEVC-trained, luma-only 7-frame window, "
            "weights bundled); fbcnn = FBCNN flexible blind JPEG-artifact removal "
            "(single-image RGB, ~72M params, weights downloaded not bundled -- see "
            "videotoolbox/fbcnn/weights/README.md). Routes frames through the MLX path."
        ),
    )
    parser.add_argument(
        "--deblock-weights", default=None, metavar="VARIANT|PATH",
        help=(
            "Weights for --deblock. stdf: a bundled token (mfqev2 = HEVC multi-QP, the "
            "default; vimeo90k = All-Intra QP37) or a path (or $STDF_WEIGHTS). fbcnn: a "
            "fbcnn_color.safetensors path (or $FBCNN_WEIGHTS); not bundled."
        ),
    )
    parser.add_argument(
        "--deblock-strength", type=float, default=1.0, metavar="S",
        help=(
            "Scale the STDF deblock residual (1.0 = full, default; lower keeps more "
            "fine texture at the cost of less deblocking -- try 0.5-0.7 on faces)."
        ),
    )
    parser.add_argument(
        "--fbcnn-quality", type=float, default=None, metavar="QF",
        help=(
            "Assumed JPEG quality factor (1-100, lower = more compressed = stronger "
            "removal) for --deblock fbcnn. Default None = blind per-frame estimate, but "
            "the JPEG-trained estimator reads loop-filtered H.264/HEVC as near-lossless "
            "(~QF 96) and barely acts -- on compressed video PIN a value: 25-50, going "
            "lower for heavier compression. A fixed value also avoids the shot-to-shot "
            "flicker blind can show (single-image net), and skips the (then-unused) QF "
            "predictor, ~1.1x faster."
        ),
    )
    parser.add_argument(
        "--fbcnn-strength", type=float, default=1.0, metavar="A",
        help=(
            "Linear dry/wet blend of FBCNN's correction for --deblock fbcnn: out = "
            "(1-A)*input + A*fbcnn(input). 1.0 = full (default); <1 keeps more original "
            "texture (and faint residual artifacts) uniformly; >1 over-drives (can ring). "
            "A QF-independent strength dial, complementary to --fbcnn-quality."
        ),
    )
    parser.add_argument(
        "--nafnet", choices=["off", "gopro", "gopro32", "sidd", "sidd32", "reds"], default="off",
        help=(
            "NAFNet restoration pass, run LAST in the preprocess chain (a light "
            "detail/deblur residual after deblock + denoise). off (default); "
            "gopro/gopro32 = motion deblur (width 64/32); sidd/sidd32 = real-noise "
            "denoise; reds = video restore. Single-image RGB net; weights are downloaded, "
            "not bundled (see videotoolbox/nafnet/weights/README.md)."
        ),
    )
    parser.add_argument(
        "--nafnet-weights", default=None, metavar="PATH",
        help="NAFNet .safetensors path override for --nafnet (or $NAFNET_WEIGHTS); not bundled.",
    )
    parser.add_argument(
        "--nafnet-strength", type=float, default=1.0, metavar="A",
        help=(
            "Scale NAFNet's residual for --nafnet: out = input + A*residual. 1.0 = full "
            "(default); lower keeps it a LIGHT pass -- recommended on video, since it is a "
            "single-image net and a strong pass can flicker. >1 over-drives."
        ),
    )
    parser.add_argument(
        "--nafnet-pool", choices=["auto", "local", "global"], default="auto",
        help=(
            "NAFNet SCA pooling mode. auto (default) follows the reference config for the "
            "selected variant: gopro/gopro32/reds use NAFNetLocal TLSC/TLC, sidd/sidd32 use "
            "plain global-pool NAFNet. local forces TLSC; global disables TLSC."
        ),
    )
    parser.add_argument(
        "--nafnet-guard",
        choices=["auto", "off", "residual", "control", "control-source", "fast", "reject"],
        default="auto",
        help=(
            "Guard against out-of-domain NAFNet residual blow-ups. auto (default) uses "
            "reject for gopro/gopro32 and off for other variants. reject emits "
            "passthrough for frames whose residual explosion covers a visible area, "
            "locks the net out after two consecutive such frames (or one catastrophic "
            "one), and re-probes on a --nafnet-guard-lockout cadence so a scene "
            "change recovers the stage. residual applies a local output-side soft knee; "
            "control reruns localized risky frames on a vertically luma-smoothed "
            "control input, but locks into control-source if risk is frame-wide. "
            "control-source predicts the residual from a stable luma-control input "
            "and adds it to the original; fast always uses single-pass residual "
            "attenuation."
        ),
    )
    parser.add_argument(
        "--nafnet-guard-threshold", type=float, default=0.12, metavar="T",
        help=(
            "Local residual magnitude threshold for --nafnet-guard (default 0.12). "
            "Healthy frames skip the guard bit-exactly; lower catches more lattice "
            "but can replace more legitimate deblur residual."
        ),
    )
    parser.add_argument(
        "--nafnet-guard-fast-fraction", type=float, default=0.85, metavar="F",
        help=(
            "Fraction of frame-risk coverage where control guard switches to a "
            "stable control-source residual instead of per-region blending "
            "(default 0.85). Set <=0 or >1 to force the slower two-pass regional path."
        ),
    )
    parser.add_argument(
        "--nafnet-guard-lockout", type=int, default=48, metavar="N",
        help=(
            "Reject-guard lockout period: frames to hold passthrough between "
            "re-probes of the net once the reject guard has locked (default 48, "
            "about 1.6-2s). 0 = never re-probe; stay locked for the rest of the "
            "clip. Locked frames skip the net entirely, so long lockouts also "
            "run faster."
        ),
    )
    parser.add_argument(
        "--nafnet-guard-ramp", type=int, default=12, metavar="N",
        help=(
            "Reject-guard transition smoothing: restoration strength eases in "
            "over N clean frames (smoothstep, so fades start and stop without a "
            "visible temporal edge) and eases OUT on moderate trips instead of "
            "cutting to passthrough (default 12; fall length from "
            "--nafnet-guard-fall). Catastrophic or large-area explosions still "
            "cut instantly. 0 = hard switching both ways."
        ),
    )
    parser.add_argument(
        "--nafnet-guard-fall", type=int, default=None, metavar="N",
        help=(
            "Frames to fade restoration OUT on a moderate reject-guard trip "
            "(the fade emits the knee-damped residual). Default: derived from "
            "--nafnet-guard-ramp as ramp/4, min 2. 0 = hard cut on trips while "
            "keeping the eased fade-in. Longer = softer off-switch but more "
            "frames carrying damped trip residual."
        ),
    )
    parser.add_argument(
        "--denoise-first", action="store_true",
        help=(
            "Run --denoise before --deblock (default is deblock then denoise). The "
            "default suits captured-then-encoded footage: undo the last degradation "
            "(compression) first, and a denoiser's white-noise assumption is broken by "
            "structured blocking. Use --denoise-first only when noise was added AFTER "
            "compression (regrained master, analog/transmission noise)."
        ),
    )
    parser.add_argument(
        "--sanitize-edges", default=None, metavar="auto|T,B,L,R",
        help=(
            "Detect and clean synthetic border junk (letterbox lines, capture "
            "garbage rows) BEFORE any processor sees the frame: the affected "
            "edge rows/cols are overwritten with the adjacent interior line "
            "(replicate fill), because learned restorers are trained on "
            "photographic content and hallucinate texture around synthetic "
            "edges. Frame dimensions and pixel-aspect are untouched, so the "
            "output geometry is identical. auto (needs --video) samples early "
            "frames and only trims edges that are anomalous in every sample, "
            "capped at 8 px per edge; thick constant bars (letterbox-class) "
            "are reported but never filled -- crop those instead. T,B,L,R "
            "forces explicit per-edge pixel counts. Default: off."
        ),
    )
    parser.add_argument(
        "--sanitize-edges-fill", choices=["restore", "extend", "trim"],
        default="restore",
        help=(
            "What happens where junk edges were detected. restore (default) "
            "= the nets see a replicate-extended frame and the ORIGINAL "
            "border is composited back over the processed output, feathered "
            "into the content (--sanitize-edges-feather): the border stays "
            "exactly as quiet/static/dark as the source. extend = keep the "
            "replicated content in the output, removing the junk -- but "
            "replicated content MOVES with the interior, visible shimmer "
            "where the eye expects a static border. trim = CROP the junk "
            "lines off entirely (folded into the crop before --crop-aspect "
            "runs, so the aspect window is computed on the clean picture; "
            "bottom/right bumped 1 px if needed to keep even dimensions)."
        ),
    )
    parser.add_argument(
        "--sanitize-edges-feather", type=int, default=2, metavar="N",
        help=(
            "Crossfade width (source px) from a restored border band into the "
            "processed content (default 2). Softens the seam between the "
            "authentic soft border and the crisply processed interior. "
            "0 = hard splice."
        ),
    )
    parser.add_argument(
        "--crop-bars", default=None, metavar="auto|T,B,L,R",
        help=(
            "Crop constant letterbox/pillarbox bars off BEFORE processing and "
            "output only the active picture (e.g. 16:9 letterboxed in 4:3 "
            "becomes true 16:9 out; 9:16 pillarboxed in 16:9 becomes true "
            "9:16). auto detects bars that are constant-extreme in every "
            "sampled frame, up to 45 percent per edge, and rounds so the "
            "active area keeps even dimensions; T,B,L,R forces explicit "
            "counts. The pixel aspect is unchanged -- the display aspect "
            "becomes the content's true aspect, which is the point. Composes "
            "with --sanitize-edges (junk detection runs on the cropped "
            "picture). Needs --video. Default: off."
        ),
    )
    parser.add_argument(
        "--crop-aspect", default=None, metavar="W:H",
        help=(
            "Crop the picture to the largest even-dimension window with this "
            "DISPLAY aspect (e.g. 16:9 on a 4:3 source, 1:1, 9:16 for a "
            "portrait extract). On anamorphic sources the pixel aspect is "
            "folded into the target automatically, so 16:9 means 16:9 on "
            "screen, not in storage pixels. Even-integer windows approximate "
            "most ratios; the closest fit is chosen. Applies AFTER "
            "--crop-bars, so a letterboxed source can be bar-cropped and then "
            "reframed in one run. Place with --crop-anchor, shift with "
            "--crop-offset. Needs --video."
        ),
    )
    parser.add_argument(
        "--crop-anchor",
        choices=["top-left", "top", "top-right", "left", "center", "right",
                 "bottom-left", "bottom", "bottom-right"],
        default="center",
        help=(
            "Where to place the --crop-aspect window (default center). "
            "E.g. 16:9 from a 4:3 source anchored at 'bottom' keeps the "
            "lower two-thirds; 'top-right' pins the window to that corner. "
            "--crop-offset nudges from the anchor."
        ),
    )
    parser.add_argument(
        "--square-pixels", action="store_true",
        help=(
            "Resample anamorphic sources to square pixels (1:1 pixel aspect) "
            "before processing: a horizontal-only resample at SOURCE "
            "resolution (Lanczos-3, GPU-resident precomputed plan) -- the "
            "cheapest point, and the upscaler re-synthesizes the mild resample "
            "softness -- with the output tagged 1:1 for "
            "PAR-ignorant players and toolchains. Also a mild DISPLAY-domain "
            "sharpness win (~7 percent measured): the anamorphic stretch must "
            "happen somewhere, and pre-SR (here, then re-synthesized by the "
            "net) beats post-SR in the player, which dilutes rendered detail. "
            "Default behavior (off) passes the source pixel aspect through "
            "losslessly instead. No-op on square-pixel sources."
        ),
    )
    parser.add_argument(
        "--crop-offset", default="0,0", metavar="DX,DY",
        help=(
            "Pixel offset of the --crop-aspect window from its anchor "
            "(right/down positive, clamped so the window stays inside the "
            "frame). Default 0,0."
        ),
    )
    parser.add_argument(
        "--preprocess-order", type=_pp_order, default=None, metavar="A,B,C",
        help=(
            "Explicit order for the preprocess stages, comma-separated from "
            "{deblock, denoise, nafnet}, e.g. 'nafnet,denoise,deblock'. Only enabled "
            "stages run; any enabled stage you omit is appended in the default order "
            "(deblock, denoise, nafnet). Overrides --denoise-first when set."
        ),
    )
    parser.add_argument(
        "--denoise", choices=["off", "spatial", "mc", "fastdvd", "bsvd", "pvdd"], default="off",
        help=(
            "Pre-upscale denoise, applied at native resolution before VSR (the "
            "correct order - SR amplifies noise). off (default); spatial = "
            "per-frame CoreImage CINoiseReduction (cheap, no temporal state); "
            "mc = motion-compensated temporal denoise via VideoToolbox optical "
            "flow (Quality tier plus startup self-test; recursive, GPU; averages "
            "static regions over time without ghosting moving edges); fastdvd = "
            "FastDVDnet CNN denoiser (MLX, learned; causal 5-frame window, "
            "strongest denoise, weights bundled); bsvd = BSVD bidirectional-buffer "
            "streaming denoiser (MLX, learned; 16-frame delay, weights local); "
            "pvdd = PVDD real-world denoiser (MLX, learned; bidirectional window "
            "attention, real-noise-trained -- unlike the AWGN-trained fastdvd/bsvd; "
            "--pvdd-variant picks pvdd/crvd/davis, weights local). "
            "Enabling denoise routes frames through the MLX upload path instead of "
            "the zero-copy direct feed."
        ),
    )
    parser.add_argument(
        "--denoise-strength", type=float, default=0.5,
        help=(
            "Denoise strength 0..1 (default 0.5). For mc, the max temporal blend "
            "toward motion-compensated history; for spatial, the noise level; for "
            "fastdvd/bsvd, the noise sigma (mapped onto sigma_255 in [5, 55]). "
            "With --noise-map auto the estimated map REPLACES this constant "
            "(strength then only serves as the fallback when estimation is "
            "impossible); scale the estimate with --noise-map-gain instead."
        ),
    )
    parser.add_argument(
        "--denoise-luma-strength", type=float, default=1.0, metavar="A",
        help=(
            "Luma half of --denoise: blend strength for the luma channel between the "
            "input and the denoiser output. 1.0 = full denoise (default); lower keeps "
            "original luma texture. Split from --denoise-chroma-strength via a BT.601 "
            "recombine (works with any --denoise backend), so you can preserve luma "
            "detail while still cleaning chroma. >1 over-drives."
        ),
    )
    parser.add_argument(
        "--denoise-chroma-strength", type=float, default=1.0, metavar="A",
        help=(
            "Chroma half of --denoise: blend strength for the chroma channels. 1.0 = "
            "full (default). The standard split is a low --denoise-luma-strength with "
            "this at 1.0 (aggressive chroma NR, gentle luma -- the eye barely sees chroma "
            "detail). <1 keeps original chroma noise."
        ),
    )
    parser.add_argument(
        "--fastdvd-weights", default=None, metavar="PATH",
        help=(
            "Override FastDVDnet weights (.safetensors) for --denoise fastdvd. "
            "Optional - defaults to the bundled --fastdvd-variant weights (or "
            "$FASTDVD_WEIGHTS). Convert a .pth with scripts/pth_to_safetensors.py."
        ),
    )
    parser.add_argument(
        "--fastdvd-variant", choices=["clipped", "standard"], default="clipped",
        help=(
            "Which bundled FastDVDnet model for --denoise fastdvd. clipped "
            "(default) is trained with clipped noise and stays clean on real "
            "footage at moderate strength; standard is the plain-AWGN model and "
            "shows a faint pixel-shuffle grid above ~0.1 strength on clean content. "
            "Ignored when --fastdvd-weights is given."
        ),
    )
    parser.add_argument(
        "--bsvd-weights", default=None, metavar="PATH",
        help=(
            "Override BSVD weights (.safetensors) for --denoise bsvd. Optional - "
            "defaults to the local --bsvd-variant weights (or $BSVD_WEIGHTS). "
            "Convert a .pth with scripts/pth_to_safetensors.py --param-key params."
        ),
    )
    parser.add_argument(
        "--bsvd-variant", choices=["c64", "c32"], default="c64",
        help=(
            "Which local BSVD model token for --denoise bsvd. c64 matches the "
            "public unblind test config; c32 is a smaller mirror checkpoint with "
            "weaker provenance. Ignored when --bsvd-weights is given."
        ),
    )
    parser.add_argument(
        "--bsvd-dtype", choices=["float16", "float32"], default="float16",
        help="MLX dtype for --denoise bsvd (default float16; use float32 for parity probes).",
    )
    parser.add_argument(
        "--pvdd-variant",
        choices=["pvdd", "crvd", "davis", "pvdd_level", "pvdd_raw", "pvdd_raw_level"],
        default="pvdd",
        help=(
            "Which local PVDD model for --denoise pvdd. pvdd (default) = real-world "
            "sRGB blind; crvd = real high-ISO sensor noise; davis = synthetic-AWGN "
            "sibling (baseline, behaves like fastdvd); pvdd_level = noise-level dial "
            "(non-blind, see --pvdd-noise-*); pvdd_raw / pvdd_raw_level = packed "
            "Bayer (need a raw pipeline, not sRGB video). Ignored when --pvdd-weights "
            "is given."
        ),
    )
    parser.add_argument(
        "--pvdd-weights", default=None, metavar="PATH",
        help=(
            "Override PVDD weights (.safetensors) for --denoise pvdd. Optional - "
            "defaults to the local --pvdd-variant weights (or $PVDD_WEIGHTS). Not "
            "bundled; convert a .pth with scripts/pth_to_safetensors.py (see "
            "videotoolbox/pvdd/weights/README.md)."
        ),
    )
    parser.add_argument(
        "--pvdd-window", type=int, default=10, metavar="N",
        help=(
            "Sliding-window length (frames) for --denoise pvdd's bidirectional "
            "recurrence (default 10). Larger windows give more temporal context at "
            "higher cost; use --pvdd-trim to overlap windows."
        ),
    )
    parser.add_argument(
        "--pvdd-trim", type=int, default=0, metavar="N",
        help=(
            "Overlap frames trimmed from each --denoise pvdd window edge (default 0 = "
            "reference-like non-overlapping chunks). Must be < window/2."
        ),
    )
    parser.add_argument(
        "--pvdd-noise-preset", choices=["off", "S", "M", "L"], default="M",
        help=(
            "Noise-level preset for the pvdd_level variants (non-blind). S/M/L map to "
            "the reference noise-variance levels (0.00069 / 0.0022 / 0.0055); M is "
            "default. off disables (needs --pvdd-noise-variance). Ignored by blind "
            "variants."
        ),
    )
    parser.add_argument(
        "--pvdd-noise-variance", type=float, default=None, metavar="V",
        help=(
            "Explicit noise-variance value for the pvdd_level variants, overriding "
            "--pvdd-noise-preset. This is variance (sigma^2), not sigma. Ignored by "
            "blind variants."
        ),
    )
    parser.add_argument(
        "--pvdd-dtype", choices=["float16", "float32"], default="float16",
        help="MLX dtype for --denoise pvdd (default float16; use float32 for parity probes).",
    )
    parser.add_argument(
        "--noise-map", choices=["constant", "auto"], default="constant",
        help=(
            "Noise conditioning for map-driven denoisers (fastdvd, bsvd, pvdd level "
            "variants). constant (default) = one sigma everywhere (from "
            "--denoise-strength / --pvdd-noise-*). auto = estimate a per-pixel sigma "
            "map from the footage itself (temporal frame-difference statistics: "
            "texture-safe, motion-capped, smooth), so noisy shadows get denoised "
            "harder than clean lit areas. For mc, the map replaces --mc-sigma as "
            "the per-pixel residual-rejection scale. Ignored with a warning by "
            "denoisers that have no map input (spatial, blind pvdd variants)."
        ),
    )
    parser.add_argument(
        "--noise-map-gain", type=float, default=1.0, metavar="G",
        help=(
            "Multiplier on the auto-estimated noise map (default 1.0). >1 denoises "
            "harder everywhere while keeping the spatial shape; <1 is gentler. Only "
            "used with --noise-map auto."
        ),
    )
    parser.add_argument(
        "--noise-map-debug", action="store_true",
        help=(
            "With --noise-map auto: after the run, write the estimated sigma map as "
            "<output>_noisemap.png (grayscale, sigma 0..0.15 -> black..white) and "
            "print its stats, so the estimate can be eyeballed."
        ),
    )
    parser.add_argument(
        "--noise-map-refresh", type=int, default=64, metavar="N",
        help=(
            "With --noise-map auto in plain streaming mode (fastdvd/bsvd without "
            "--gop-align): re-estimate the map from the last frames every N input "
            "frames, EMA-blended so it adapts without pumping (default 64; 0 = "
            "estimate once from the first frames and hold). Windowed modes "
            "(--gop-align, pvdd) re-estimate per window and ignore this."
        ),
    )
    parser.add_argument(
        "--reader", choices=["auto", "native", "ffmpeg"], default="auto",
        help=(
            "Video reader backend. auto (default) = the native AVFoundation reader "
            "(zero-copy, full precision), falling back to the ffmpeg compatibility "
            "reader when the container/codec is refused (MKV, VP9, AVI-era "
            "material). ffmpeg = force the compatibility reader (needs PyAV: "
            "install the 'ffmpeg' extra). native = never fall back. The ffmpeg "
            "reader mirrors color tags, keyframe windows (--gop-align), trims, and "
            "audio; its frame indices are self-consistent but may differ from the "
            "native reader's by an edit-list offset on the same file."
        ),
    )
    parser.add_argument(
        "--deblock-map", choices=["constant", "auto"], default="constant",
        help=(
            "Spatial gating for --deblock stdf/fbcnn. constant (default) = the "
            "correction applies everywhere at --deblock-strength. auto = estimate a "
            "per-pixel blockiness mask from the footage (coding-grid phase detection "
            "+ boundary-vs-interior gradient contrast; texture and 1D content edges "
            "reject) and gate the correction by it -- blocked flats get the full "
            "deblock, detailed/clean areas keep their texture."
        ),
    )
    parser.add_argument(
        "--deblock-map-gain", type=float, default=1.0, metavar="G",
        help=(
            "Multiplier on the auto blockiness mask (default 1.0). >1 saturates the "
            "mask sooner (more area fully deblocked); <1 is more conservative. Only "
            "used with --deblock-map auto."
        ),
    )
    parser.add_argument(
        "--noise-map-pulse", action="store_true",
        help=(
            "Per-frame gain on the noise conditioning that tracks GOP-phase noise "
            "pulsing: old encoders re-code the grain at every I-frame, so temporal "
            "noise is elevated right after keyframes and suppressed once P/B "
            "prediction settles. Each frame's global sigma is measured against the "
            "running settled level and the sigma plane is scaled by the ratio "
            "(clamped 0.6..1.8), so I-frame grain refreshes get proportionally "
            "stronger denoising. Works with --noise-map constant or auto; "
            "map-conditioned denoisers only (fastdvd, bsvd, mc, pvdd level "
            "variants)."
        ),
    )
    parser.add_argument(
        "--basicvsrpp-variant",
        choices=["reds4", "vimeo90k_bi", "vimeo90k_bd", "ntire_vsr"], default="vimeo90k_bd",
        help=(
            "Which bundled BasicVSR++ 4x checkpoint for --spatial-mode basicvsrpp. "
            "All are bicubic/blur-degradation SR models. ntire_vsr = the big "
            "c128n25 model (sharpest, 175MB, more memory); vimeo90k_bd (default) = "
            "blur-downsample, the best of the small c64n7 models on native footage; "
            "vimeo90k_bi / reds4 = bicubic, softer on non-bicubic input. Ignored "
            "when --basicvsrpp-weights is given."
        ),
    )
    parser.add_argument(
        "--basicvsrpp-weights", default=None, metavar="VARIANT|PATH",
        help=(
            "BasicVSR++ weights for --spatial-mode basicvsrpp: a bundled variant token "
            "(reds4/vimeo90k_bi/vimeo90k_bd/ntire_vsr) or a .safetensors path. Overrides "
            "--basicvsrpp-variant (or $BASICVSRPP_WEIGHTS)."
        ),
    )
    parser.add_argument(
        "--basicvsrpp-window", type=int, default=14, metavar="N",
        help=(
            "Sliding-window length (frames) for --spatial-mode basicvsrpp. The "
            "recurrent net is run per window; larger = closer to whole-clip quality "
            "but more memory + compute (default 14). Clips shorter than this are "
            "processed whole."
        ),
    )
    parser.add_argument(
        "--basicvsrpp-trim", type=int, default=2, metavar="N",
        help=(
            "Warm-up frames trimmed at each window join for --spatial-mode "
            "basicvsrpp (default 2). Trimmed frames are re-emitted by the "
            "neighbouring window with fuller propagation context."
        ),
    )
    parser.add_argument(
        "--basicvsrpp-ensemble", action="store_true",
        help=(
            "Run --spatial-mode basicvsrpp through the reference's 8-way geometric "
            "self-ensemble (average the 8 flip/rotate variants). The ntire_vsr "
            "checkpoint's config declares this SpatialTemporalEnsemble at inference "
            "(the reds4/vimeo SR models do not); the mmagic re-port dropped it as dead "
            "config. Reference-faithful for ntire_vsr and a mild artifact-reducer, at "
            "8x the compute. Off by default."
        ),
    )
    parser.add_argument(
        "--basicvsrpp-flow-mode", choices=["spynet", "zero", "vt"], default="spynet",
        help=(
            "Optical-flow source for --spatial-mode basicvsrpp (default spynet). "
            "zero disables motion compensation while keeping recurrent propagation, "
            "as a control for flow-painted temporal artifacts. vt uses "
            "VTOpticalFlow (Quality tier, self-tested at startup; smooth fields, "
            "no content-shaped flow noise)."
        ),
    )
    parser.add_argument(
        "--basicvsrpp-history-strength", type=float, default=1.0, metavar="S",
        help=(
            "Scale BasicVSR++'s aligned propagation features (default 1.0 = "
            "reference strength). 0 disables temporal propagation entirely."
        ),
    )
    parser.add_argument(
        "--basicvsrpp-history-gate", choices=["off", "improve"], default="off",
        help=(
            "Per-pixel history admission for BasicVSR++'s propagation (default "
            "off = reference behavior). improve admits aligned history only "
            "where the flow warp measurably improves the photometric residual "
            "against the current frame; second-order sources gate with the "
            "better of their two flows. Mitigates propagation ghosting on "
            "content optical flow cannot explain."
        ),
    )
    parser.add_argument(
        "--realbasicvsr-weights", default=None, metavar="VARIANT|PATH",
        help=(
            "RealBasicVSR weights for --spatial-mode realbasicvsr: the bundled variant "
            "token 'x4' (default) or a .safetensors path (or $REALBASICVSR_WEIGHTS). "
            "Convert a .pth with scripts/pth_to_safetensors.py --only-prefix "
            "generator_ema. --strip-prefix generator_ema."
        ),
    )
    parser.add_argument(
        "--realbasicvsr-window", type=int, default=14, metavar="N",
        help=(
            "Sliding-window length (frames) for --spatial-mode realbasicvsr "
            "(default 14). Larger = more temporal context, but more memory + compute."
        ),
    )
    parser.add_argument(
        "--realbasicvsr-trim", type=int, default=0, metavar="N",
        help=(
            "Warm-up frames trimmed at each window join for --spatial-mode "
            "realbasicvsr (default 0, matching the reference max_seq_len "
            "non-overlap chunking). Values >0 re-run overlapping windows and "
            "discard boundary frames, which costs more compute."
        ),
    )
    parser.add_argument(
        "--realbasicvsr-dynamic-refine-thres", type=float, default=5.0, metavar="V",
        help=(
            "RealBasicVSR cleaning stop threshold in 0..255 units (default 5, "
            "the GAN test-time setting; 255 forces one cleaning pass)."
        ),
    )
    parser.add_argument(
        "--realbasicvsr-clean-iters", type=int, default=3, metavar="N",
        help="Maximum RealBasicVSR cleaning passes before propagation (default 3).",
    )
    parser.add_argument(
        "--realbasicvsr-residual-strength", type=float, default=1.0, metavar="V",
        help=(
            "Scale the learned RealBasicVSR residual before adding it to the 4x "
            "bilinear base (default 1.0). Try 0.6-0.85 to reduce GAN/pixel-shuffle "
            "lattice artifacts on moving objects while retaining most sharpening."
        ),
    )
    parser.add_argument(
        "--realbasicvsr-flow-consistency", type=float, default=0.0, metavar="S",
        help=(
            "Forward-backward flow-consistency masking strength in 0..1 for "
            "--spatial-mode realbasicvsr (default 0 = off, reference behavior). "
            "Down-weights the recurrent feature where the optical-flow round-trip "
            "fails -- occlusions and fast-moving regions (panning backgrounds, "
            "objects/people passing, body edges) -- cutting propagation ghosting "
            "there while keeping detail on well-aligned regions. Note: it does "
            "NOT fix ghosting on a stable subject with consistent-but-wrong flow "
            "(a selfie face's specular highlights); use --realbasicvsr-window 1 "
            "for that. Try 0.7-1.0."
        ),
    )
    parser.add_argument(
        "--realbasicvsr-flow-mode", choices=["spynet", "zero", "vt"], default="spynet",
        help=(
            "Optical-flow source for --spatial-mode realbasicvsr (default spynet). "
            "zero disables motion compensation while keeping recurrent propagation, "
            "as a control for flow-painted temporal artifacts. vt uses "
            "VTOpticalFlow (Quality tier, self-tested at startup; smooth fields, "
            "no content-shaped flow noise)."
        ),
    )
    parser.add_argument(
        "--realbasicvsr-history-strength", type=float, default=1.0, metavar="S",
        help=(
            "Scale RealBasicVSR's propagated features before each trunk (default "
            "1.0 = reference strength). 0 disables temporal propagation entirely "
            "(per-frame within the cleaning + upsampler)."
        ),
    )
    parser.add_argument(
        "--realbasicvsr-history-gate", choices=["off", "improve"], default="off",
        help=(
            "Per-pixel history admission for RealBasicVSR's propagation (default "
            "off = reference behavior). improve admits warped history only where "
            "the flow warp measurably improves the photometric residual against "
            "the current cleaned frame -- the mitigation for the window>1 "
            "propagation smear (dark ghosting trails around faces/movers); "
            "regions the flow cannot explain fall back to the trained "
            "window-start (zero history) behavior."
        ),
    )
    parser.add_argument(
        "--realesrgan-weights", default=None, metavar="VARIANT|PATH",
        help=(
            "RRDBNet/SRVGG weights for --spatial-mode realesrgan: a variant token or a "
            ".safetensors path (or $REALESRGAN_WEIGHTS). Tokens: general (default; SRVGG, "
            "fast/gentle), x4plus (RRDBNet crisp/GAN, ~20x slower), realesrnet / bsrnet "
            "(MSE, faithful/soft), bsrgan, x2plus (2x output), anime / animevideo (anime), "
            "esrgan (original ESRGAN). Only general is bundled; the rest download + convert "
            "(see videotoolbox/realesrgan/weights/README.md)."
        ),
    )
    parser.add_argument(
        "--realesrgan-denoise", type=float, default=1.0, metavar="S",
        help=(
            "Denoise dial (dni) for realesr-general-x4v3 only, 0..1 (default "
            "1.0 = pure general). Blends s*general + (1-s)*wdn; per Real-ESRGAN, "
            "higher = stronger denoise (smoother), lower keeps more real-world "
            "texture/grain. Needs the realesr_general_wdn_x4v3 companion weight."
        ),
    )
    parser.add_argument(
        "--realviformer-weights", default=None, metavar="VARIANT|PATH",
        help=(
            "RealViformer weights for --spatial-mode realviformer: the x4 token or a "
            ".safetensors path (or $REALVIFORMER_WEIGHTS). A causal recurrent real-world "
            "4x video upscaler (channel-attention transformer); streams frame by frame "
            "with temporal state, reset at cuts. Not bundled; see "
            "videotoolbox/realviformer/weights/README.md."
        ),
    )
    parser.add_argument(
        "--realviformer-dtype", choices=["float16", "float32"], default="float16",
        help=(
            "Compute/weight dtype for --spatial-mode realviformer. float16 is the "
            "default MLX fast path; float32 is closer to the PyTorch reference and "
            "useful when isolating recurrent-flow artifacts."
        ),
    )
    parser.add_argument(
        "--realviformer-window", type=int, default=100, metavar="N",
        help=(
            "Recurrence chunk length (frames) for --spatial-mode realviformer: the "
            "temporal state resets cold every N frames, matching the reference "
            "inference's 100-frame chunking (default 100). The reset caps the "
            "texture-lock etching that unbounded recurrence engraves on long static "
            "shots, at the cost of a texture refresh at each chunk join (the "
            "reference tool has the same one). 0 = never reset: unbounded streaming "
            "recurrence, deeper temporal lock than the released tool ever runs."
        ),
    )
    parser.add_argument(
        "--realviformer-flow-mode", choices=["spynet", "zero", "vt"], default="spynet",
        help=(
            "Optical-flow source for --spatial-mode realviformer (default spynet). "
            "zero disables motion compensation while keeping recurrent state, "
            "as a control for flow-painted etching; vt uses VideoToolbox "
            "VTOpticalFlow in the current-to-previous warp convention."
        ),
    )
    parser.add_argument(
        "--realviformer-history-strength", type=float, default=1.0, metavar="S",
        help=(
            "Scale RealViformer's normalized propagated-history branch before the "
            "merge FFN (default 1.0 = reference strength). 0 disables temporal "
            "history while still exercising the recurrent code path; values below "
            "1 soften temporal etching."
        ),
    )
    parser.add_argument(
        "--realviformer-history-gate", choices=["off", "improve", "holistic"], default="off",
        help=(
            "History confidence gate for --spatial-mode realviformer. off = "
            "reference merge. improve = pass history only where the flow-warped "
            "previous RGB frame improves the current-vs-previous residual; this "
            "drops ambiguous near-static/compressed regions that otherwise etch. "
            "holistic = opt-in HSA-lite risk policy: relative warp benefit + match "
            "confidence + risk memory, with optional hidden-state cleanup."
        ),
    )
    parser.add_argument(
        "--realviformer-history-cleanup", type=float, default=0.25, metavar="S",
        help=(
            "For --realviformer-history-gate holistic, max blend toward a 3x3 "
            "box-blurred warped hidden state in risky regions (default 0.25)."
        ),
    )
    parser.add_argument(
        "--realviformer-history-gate-drop", type=float, default=0.85, metavar="S",
        help=(
            "For --realviformer-history-gate holistic, max fraction of temporal "
            "history gate removed in risky regions (default 0.85)."
        ),
    )
    parser.add_argument(
        "--realviformer-history-risk-decay", type=float, default=0.80, metavar="S",
        help=(
            "For --realviformer-history-gate holistic, decay for the flow-warped "
            "risk memory between frames (default 0.80; must be <1)."
        ),
    )
    parser.add_argument(
        "--realviformer-history-static-cap", type=float, default=0.0, metavar="S",
        help=(
            "For --realviformer-history-gate holistic, capped confidence admitted "
            "for perfectly static regions (default 0.0 until long-static etch "
            "tests prove a nonzero cap is safe)."
        ),
    )
    parser.add_argument(
        "--esc-weights", default=None, metavar="VARIANT|PATH",
        help=(
            "ESC-Real weights for --spatial-mode esc: a variant token or a .safetensors "
            "path (or $ESC_WEIGHTS). Tokens: gan (default; perceptual, Real-ESRGAN-style "
            "degradation training) and mse (fidelity twin). Neither is bundled; see "
            "videotoolbox/esc/weights/README.md."
        ),
    )
    parser.add_argument(
        "--realplksr-weights", default=None, metavar="VARIANT|PATH",
        help=(
            "RealPLKSR weights for --spatial-mode realplksr: a variant token or a "
            ".safetensors path (or $REALPLKSR_WEIGHTS). Tokens: public2x (default; "
            "2x LayerNorm+DySample, real-world photo/JPEG, Apache-2.0), public2x-nn "
            "(same trained without noise -- for cleaner sources), nomos4x "
            "(4xNomosWebPhoto, 4x GroupNorm+PixelShuffle photo restoration, CC-BY). "
            "Scale (2x/4x) is read from the checkpoint. Single-image, no pooled gate "
            "so no SAFMN block lattice; no denoising prior beyond its training, so "
            "prefer it on decent sources or pair with --denoise. None are bundled; "
            "see videotoolbox/realplksr/weights/README.md."
        ),
    )
    parser.add_argument(
        "--realplksr-dtype", choices=["float16", "float32"], default="float16",
        help=(
            "Compute/storage dtype for --spatial-mode realplksr. float16 (default) "
            "runs the convs in half precision with fp32 precision islands in the norm "
            "reductions and Mish (visually lossless, ~72 dB vs fp32, and it is these "
            "islands that make the fp16-flagged GroupNorm 4x checkpoint safe). float32 "
            "forces a full single-precision run (slower, more memory)."
        ),
    )
    parser.add_argument(
        "--safmn-weights", default=None, metavar="VARIANT|PATH",
        help=(
            "SAFMN weights for --spatial-mode safmn: a variant token or a .safetensors "
            "path (or $SAFMN_WEIGHTS). Tokens: light (default; light_SAFMN++, tiny "
            "fidelity 4x trained on compressed content), real (SAFMN_L_Real_LSDIR, "
            "real-world perceptual 4x trained with the Real-ESRGAN degradation -- "
            "video-appropriate, ~3x faster than the RRDBNet class), real2x (same family, "
            "2x output); purescale / purescale2x / purescale2x-sharp (PureScale 2.0, "
            "SAFMN-L retrained with a fixed SAFM branch that eliminates the known "
            "transient block-lattice artifact of the stock models; sharp adds deblur. "
            "For DECENT-quality sources: it has no denoising prior, so noisy or "
            "compressed video gets its noise rendered as etched, flickering "
            "texture -- use real with --safmn-pool-clamp there. CAUTION: "
            "PureScale weights are CC BY-NC-SA "
            "4.0 -- NON-COMMERCIAL use only). None are bundled; see "
            "videotoolbox/safmn/weights/README.md."
        ),
    )
    parser.add_argument(
        "--safmn-safm-up", choices=["auto", "nearest", "bicubic"], default="auto",
        help=(
            "SAFM upsampler inside the SAFMN blocks. auto (default) = the mode each "
            "checkpoint was trained with (nearest for the stock models, bicubic for "
            "the purescale retrains, verified against the reference). Forcing the "
            "other mode is a safe shape-only override. On the stock real models, "
            "bicubic is a CREATIVE dial: the trained nearest-up gate is blocky and "
            "flattens micro-texture within its blocks, and a smooth gate frees the "
            "GAN's texture synthesis -- measured ~30 percent more output "
            "high-frequency energy, visibly grainier/sharper surfaces (judge on "
            "video: hallucinated micro-texture can shimmer temporally). It also "
            "rounds lattice blocks into soft blobs without fixing the underlying "
            "hot-pixel winner. The POOLING statistic "
            "is trained in and always follows the checkpoint -- swapping it "
            "corrupts the output."
        ),
    )
    parser.add_argument(
        "--safmn-pool-clamp", type=float, default=0.0, metavar="K",
        help=(
            "Winsorize SAFM pooled features to mean +/- K sigma per channel; a "
            "stock-weights (real/real2x) mitigation for the transient "
            "block-lattice artifact (default 0 = off). DIRECTION: K is the "
            "allowed width in sigmas, so LOWER K = STRONGER suppression -- "
            "raising K weakens it. Calibrated levels: 4 = visually free on "
            "normal content, lattice greatly reduced but findable frame by "
            "frame; 3 = lattice imperceptible, cost still negligible (the "
            "recommended setting); 2.5 = boundary, specular-rich content "
            "starts dulling; 2 = visibly muted highlights and flattened "
            "texture sparkle. Failure is graceful at any K (it can only "
            "under-modulate, never corrupt). Frame-boundary pooled cells are "
            "exempt (one cell = 2/4/8 px per level): synthetic border "
            "rows/bars saturate the features into a quiet self-limiting "
            "response, and clamping them re-engages texture hallucination "
            "and makes borders bloom. The purescale variants fix the "
            "artifact at the root and do not need this."
        ),
    )
    parser.add_argument(
        "--mc-window", type=int, default=0, metavar="N",
        help=(
            "mc temporal structure (mutually exclusive). 0 (default) = recursive "
            "IIR (blends the previous output; strongest denoise, longest ghosts). "
            "N>=1 = causal FIR over the last N input frames (bounded ghost "
            "lifetime, ~N optical-flow computes per frame)."
        ),
    )
    parser.add_argument(
        "--mc-sigma", type=float, default=0.06, metavar="S",
        help=(
            "mc residual-rejection scale (luma, 0..1; default 0.06 ~= 15/255). The "
            "blend gate is exp(-(current-vs-history residual / sigma)^2); noise "
            "inflates that residual, so at the default it throttles its own removal "
            "even at --denoise-strength 1. RAISE it (e.g. 0.10-0.15) to denoise "
            "harder when strength alone plateaus -- the real 'make it stronger' knob "
            "for noisy footage. Cost: it also tolerates motion mismatch, so higher = "
            "more ghosting/smearing on fast motion. With --noise-map auto this "
            "scalar is REPLACED by the estimated per-pixel map (in residual units); "
            "scale that with --noise-map-gain instead."
        ),
    )
    parser.add_argument(
        "--mc-clamp", action="store_true",
        help="mc: clamp warped history to the current frame's local color box "
             "(TAA variance-clip). The strongest single anti-ghost. Combinable.",
    )
    parser.add_argument(
        "--mc-occlusion", action="store_true",
        help="mc: reject history via forward-backward flow consistency "
             "(occlusion / bad-flow detection). Combinable.",
    )
    parser.add_argument(
        "--mc-confidence", action="store_true",
        help="mc: down-weight history where flow magnitude is large (fast "
             "motion). Combinable.",
    )
    parser.add_argument(
        "--cut-detect", choices=["off", "simple", "hist"], default="off",
        help=(
            "Reset VSR's prev-frame chain at hard cuts. off = never reset "
            "(correct for single-shot LTX latents). Only meaningful for "
            "edited --video input under --spatial-mode balanced (which "
            "chains prev-frame state); a no-op under fast/image modes."
        ),
    )
    parser.add_argument("--cut-threshold", type=float, default=0.25)
    parser.add_argument(
        "--cut-log", default=None,
        help="Write detected cut frame indices to this file (one per line).",
    )
    parser.add_argument(
        "--video-chunk-size", type=int, default=32,
        help=(
            "Upper bound on decoded frames held in flight for --video input. "
            "The actual chunk is further capped to a ~64 MiB memory budget "
            "based on resolution, so peak resident decode memory stays bounded "
            "(often 1 frame at 4K, this many at SD)."
        ),
    )
    parser.add_argument(
        "--start", default=None,
        help=(
            "Trim the input to start at this position (process the middle of a "
            "clip). Accepts frames or time: bare integer = frames (e.g. 120), "
            "Nf = frames (120f), Ns / decimal = seconds (5s, 1.5), or a clock "
            "string mm:ss / hh:mm:ss (0:05, 1:02:03). --video seeks here "
            "natively (the head is not decoded); --latent windows the decode."
        ),
    )
    parser.add_argument(
        "--end", default=None,
        help=(
            "Trim the input to stop before this position (exclusive), same "
            "frames-or-time forms as --start. Output is a fresh clip starting "
            "at PTS 0 spanning [--start, --end)."
        ),
    )
    parser.add_argument(
        "--max-frames", default=None,
        help=(
            "Cap the number of OUTPUT frames. Same frames-or-time forms as "
            "--start (a time here is output duration, measured at the target "
            "fps). Composes with --start/--end, which trim the input."
        ),
    )
    parser.add_argument("--save-pre-frames", action="store_true")
    parser.add_argument("--save-post-frames", action="store_true")
    parser.add_argument(
        "--skip-post-mp4", action="store_true",
        help="Skip writing the upscaled _post.mp4 (e.g. when you only want frame dumps).",
    )
    parser.add_argument(
        "--comparison", action="store_true",
        help="Also write a side-by-side <stem>_comparison.mp4 "
             "(NEAREST-upscaled pre vs VSR post).",
    )
    parser.add_argument(
        "--mlx-cache-limit-gb", type=float, default=1.0,
        help="Cap MLX's buffer cache (GB) so per-frame allocation churn does not "
             "grow into swap; 0 disables. Default 1.0, matching generate.py.",
    )
    args = parser.parse_args()

    if args.latent and not args.weights:
        parser.error("--latent requires --weights")
    if args.spatial_mode == "realbasicvsr":
        if args.realbasicvsr_window < 1:
            parser.error("--realbasicvsr-window must be >= 1")
        if args.realbasicvsr_trim < 0:
            parser.error("--realbasicvsr-trim must be >= 0")
        if args.realbasicvsr_trim and args.realbasicvsr_window <= 2 * args.realbasicvsr_trim:
            parser.error(
                "--realbasicvsr-window must be greater than 2*--realbasicvsr-trim; "
                "use --realbasicvsr-trim 0 for reference-like chunks"
            )
    if args.spatial_mode == "realviformer" and args.realviformer_window < 0:
        parser.error("--realviformer-window must be >= 0")
    if args.spatial_mode == "realviformer" and args.realviformer_history_strength < 0:
        parser.error("--realviformer-history-strength must be >= 0")
    if args.spatial_mode == "realviformer":
        if not 0.0 <= args.realviformer_history_cleanup <= 1.0:
            parser.error("--realviformer-history-cleanup must be in [0, 1]")
        if not 0.0 <= args.realviformer_history_gate_drop <= 1.0:
            parser.error("--realviformer-history-gate-drop must be in [0, 1]")
        if not 0.0 <= args.realviformer_history_risk_decay < 1.0:
            parser.error("--realviformer-history-risk-decay must be in [0, 1)")
        if not 0.0 <= args.realviformer_history_static_cap <= 1.0:
            parser.error("--realviformer-history-static-cap must be in [0, 1]")

    if args.mlx_cache_limit_gb and args.mlx_cache_limit_gb > 0:
        mx.set_cache_limit(int(args.mlx_cache_limit_gb * (1000 ** 3)))
        mx.clear_cache()
        print(f"MLX cache limit: {args.mlx_cache_limit_gb:g} GB")

    require_pyobjc()
    run(args)


if __name__ == "__main__":
    main()
