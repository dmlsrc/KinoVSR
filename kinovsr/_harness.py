"""Inherited file-to-file orchestration (transitional).

This is the extracted harness runtime: probing, crop/sanitize pre-passes,
stage construction, the frame pump, writers, and reporting, behind
``run(args, settings=...)``. The M2 facade
(:func:`kinovsr.api.process_video_file`) wraps it; M3/M4 dismantle it into
``pipeline/`` modules and processor families per the planning milestones.
Do not add new behavior here; add it to package modules instead.

``args`` is the canonical-options namespace assembled by
:mod:`kinovsr.cli.config`; ``settings`` already folds env vars, TOML
``[settings]``, and CLI overrides, so weight fallbacks read from it
directly.
"""

from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx

from kinovsr import (
    AudioTrack,
    AVWriter,
    CutDetector,
    VsrSession,
    VtfrcSession,
    autorelease_pool,
)
from kinovsr.api import VideoProcessResult
from kinovsr.comparison import render_comparison
from kinovsr.denoise import LumaChromaDenoiser, McTemporalDenoiser, SpatialDenoiser
from kinovsr.edge_sanitize import (
    compute_aspect_crop,
    detect_bars,
    detect_junk_edges,
    parse_edges_spec,
)
from kinovsr.edge_sanitize import (
    crop_rgb as _crop_rgb,
)
from kinovsr.edge_sanitize import (
    restore_borders as _restore_borders,
)
from kinovsr.edge_sanitize import (
    sanitize_rgb as _sanitize_rgb,
)
from kinovsr.media import color as _color
from kinovsr.media import pixel_buffers as _pb
from kinovsr.media import video_reader as _native_vr
from kinovsr.media import yuv as _yuv
from kinovsr.modeling.vsr_blocks import make_lanczos_plan, resample_width
from kinovsr.native.vsr import NativePassthrough
from kinovsr.native.writer import (
    HEVC_PROFILE_MAIN10,
    HEVC_PROFILE_MAIN422_10,
)
from kinovsr.processors.bsvd import BsvdDenoiser
from kinovsr.processors.fastdvdnet import FastDvdDenoiser
from kinovsr.processors.toflow import TOFlowDenoiser
from kinovsr.progress import StackedPhaseBars
from kinovsr.settings import Settings


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


def _read_audio_track_from_video(mp4_path: Path, vr: Any) -> AudioTrack | None:
    """Read the audio track of an MP4/MOV into an in-memory AudioTrack.

    Uses AVFoundation's AVAudioFile (via videotoolbox.audio.read_wav), which
    decodes the container's audio stream straight into a (channels, frames)
    float32 MLX array - no ffmpeg, no disk WAV. Returns None if the file has
    no audio track.
    """
    from kinovsr.media.audio import read_wav

    if hasattr(vr, "read_audio_track"):
        # ffmpeg compatibility reader: decode audio via the same backend that
        # reads the video (the native audio path cannot open these containers).
        print(f"[setup] reading audio track from {mp4_path} (ffmpeg)")
        try:
            return vr.read_audio_track(mp4_path)
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
                                "realesrgan", "safmn", "esc", "realviformer", "realplksr",
                                "toflow", "metalfx")
            else HEVC_PROFILE_MAIN10)


# Preprocess slots in default execution order (the CLI's pp_order type in
# kinovsr.cli.options validates --preprocess-order against the same names).
_PP_STAGE_NAMES = ("restore", "deflicker", "deblock", "denoise", "nafnet")


def sanitize_output_prefix(prefix: str | None) -> str:
    """Keep generated filenames shell-friendly while preserving readable prefixes."""
    prefix = (prefix or "kinovsr").strip()
    if not prefix:
        prefix = "kinovsr"
    sanitized = []
    for char in prefix:
        if char.isalnum() or char in ("-", "_", "."):
            sanitized.append(char)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("._") or "kinovsr"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace, *, settings: Settings) -> VideoProcessResult:
    stem = f"{sanitize_output_prefix(args.output_prefix)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = Path(args.output_dir) if args.output_dir else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)
    pre_dir = (out_root / f"{stem}_pre") if out_root is not None else None
    post_dir = (out_root / f"{stem}_post") if out_root is not None else None
    if args.save_pre_frames and pre_dir is not None:
        pre_dir.mkdir(parents=True, exist_ok=True)
    if args.save_post_frames and post_dir is not None:
        post_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] output stem: {stem}")

    audio_track: AudioTrack | None = None
    gop_schedule: Any = None   # GOP-aligned window plan for recurrent stages (--gop-align)
    _gop_head_skip = 0         # gop-align context frames whose outputs are dropped
    src_transform: Any = None  # source rotation/flip from the input container
    src_pixel_aspect: tuple[int, int] | None = None
    win_start, win_end = 0, None  # resolved input window
    # Output color tags are filled from the source container after probing.
    _resolved_color = _color.resolve({"full_range": False}, "bt709")
    color_props: dict | None = _color.av_color_properties(_resolved_color)
    cv_color: tuple | None = _color.cv_triple(_resolved_color)
    output_full_range = _resolved_color[3]

    # ---- Input source ------------------------------------------------------
    from kinovsr.native.vsr import source_format_for_mode

    # ---- reader selection: native (AVFoundation) unless forced or refused.
    # The ffmpeg compatibility reader mirrors the native surface for
    # containers/codecs AVFoundation cannot open (MKV, VP9, ...). The choice
    # is strictly per-run: rebinding a module global here leaked one run's
    # ffmpeg fallback into every later facade call in the same process.
    vr = _native_vr
    if args.reader == "ffmpeg":
        from kinovsr.media import ffmpeg_reader
        vr = ffmpeg_reader
        print("[reader] ffmpeg compatibility reader (forced)")
    elif args.reader == "auto":
        try:
            vr.probe_video(Path(args.video))
        except Exception as e:
            from kinovsr.media import ffmpeg_reader
            vr = ffmpeg_reader
            print(f"[reader] native reader cannot open this file "
                  f"({type(e).__name__}); using the ffmpeg compatibility reader")

    print(f"Reading video: {args.video}")
    in_w, in_h, source_fps, total_frames, src_transform, src_pixel_aspect = vr.probe_video(
        Path(args.video),
    )
    _src_color = vr.probe_color(Path(args.video))
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
        for s_chunk in vr.iter_video_buffer_chunks(
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
        _pb.PIX_RGBAHALF if args.upscale == "none"
        else source_format_for_mode(args.upscale)
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
    _kf_all = (vr.keyframe_display_indices(Path(args.video))
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
        from kinovsr.modeling.upscaler_base import plan_gop_windows
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
        chunks = vr.iter_forced_color_chunks(
            Path(args.video), vsr_src_fmt, cv_color[2], _src_color["full_range"],
            chunk_size=buf_chunk, start_frame=_read_start, end_frame=win_end,
            reinterpret_full_range=output_full_range,
        )
    else:
        chunks = vr.iter_video_buffer_chunks(
            Path(args.video), vsr_src_fmt, chunk_size=buf_chunk,
            start_frame=_read_start, end_frame=win_end,
        )
    # Carry the source file's audio through to the output MP4.
    if args.audio:
        audio_track = _read_audio_track_from_video(Path(args.video), vr)

    # ---- Audio trim + sidecar ----------------------------------------------
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
        if out_root is None:
            raise SystemExit("--save-audio-sidecar requires --output-dir")
        sidecar = out_root / f"{stem}_audio.wav"
        audio_track.save_wav(sidecar)
        print(f"[setup] audio sidecar: {sidecar}")

    # ---- Output geometry + encoder settings --------------------------------
    # Weight specs resolve once: an explicit path (CLI flag or legacy env
    # var, both already folded into settings) beats the profile token, and
    # each family resolves its own default when both are unset. Scale
    # detection below must use the same spec as stage construction.
    realesrgan_spec = settings.realesrgan_weights or args.realesrgan_profile
    safmn_spec = settings.safmn_weights or args.safmn_profile
    esc_spec = settings.esc_weights or args.esc_profile
    realplksr_spec = settings.realplksr_weights or args.realplksr_profile
    from kinovsr.native.vsr import scale_for_mode
    spatial_scale = 1 if args.upscale == "none" else scale_for_mode(args.upscale)
    if args.upscale == "realesrgan":
        # realesrgan covers 2x (x2plus) as well as 4x models; read the real scale from the
        # checkpoint instead of assuming 4x, so output dims + encoder match the frames.
        from kinovsr.processors.realesrgan import net as _rnet
        spatial_scale = _rnet.scale_of(_rnet.load_params(
            _rnet.resolve_weights(realesrgan_spec)))
    elif args.upscale == "safmn":
        from kinovsr.processors.safmn import net as _snet
        spatial_scale = _snet._config(_snet.load_params(safmn_spec))[3]
    elif args.upscale == "esc":
        from kinovsr.processors.esc import net as _enet
        spatial_scale = _enet._config(_enet.load_params(esc_spec))[6]
    elif args.upscale == "realplksr":
        # realplksr covers 2x (public2x) and 4x (nomos4x); read the scale from the
        # checkpoint so output dims + encoder match the frames.
        from kinovsr.processors.realplksr import net as _pnet
        spatial_scale = _pnet._config(_pnet.load_params(realplksr_spec))[4]
    elif args.upscale == "metalfx":
        spatial_scale = args.metalfx_scale
    out_w, out_h = in_w * spatial_scale, in_h * spatial_scale
    profile = _pick_hevc_profile(args.upscale, args.encode_chroma)
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

    if getattr(args, "probe_noise", False) and args.video:
        from kinovsr.noise_map import (
            analyze_noise,
            classify_noise_analysis,
            detect_grid_period,
            estimate_blockiness_map,
        )
        from kinovsr.quant_comb import estimate_qf
        _pw_end = win_end if win_end is not None else total_frames
        _span = max(1, _pw_end - win_start)
        _starts = sorted({win_start + int(f * max(0, _span - 12)) for f in (0.1, 0.5, 0.9)})
        print(f"[probe] noise analysis: {len(_starts)} windows of 12 frames in "
              f"[{win_start}, {_pw_end})")
        try:
            _kfl = vr.keyframe_display_indices(Path(args.video)) or [0]
        except Exception:
            _kfl = [0]
        _all_frames: list = []
        _mid_frames: list = []
        _all_labels: list = []
        _mc_sigs: list = []
        for ws in _starts:
            _fr = []
            for _chk in vr.iter_video_buffer_chunks(
                    Path(args.video), _pb.PIX_RGBAHALF, chunk_size=6,
                    start_frame=ws, end_frame=min(ws + 12, _pw_end)):
                _fr += [mx.clip(_pb.read_buffer_rgb_f32(b), 0, 1) for b in _chk]
            if len(_fr) < 3:
                continue
            r = analyze_noise(_fr)
            diag = classify_noise_analysis(r)
            _all_frames.extend(_fr)
            if ws == _starts[len(_starts) // 2]:
                _mid_frames = list(_fr)
            _all_labels.extend(diag["labels"])
            _mc_sigs.append(float(r.get("mc_sigma", 0.0)))
            tr = r.pop("frame_trace")

            def _kfd(i, _ws=ws):
                pri = [k for k in _kfl if k <= _ws + i + 1]
                return _ws + i + 1 - (max(pri) if pri else 0)
            print(f"[probe] window @ frame {ws}:")
            print("  sigma (med/p90 per block): " + "  ".join(
                f"{k} {r[k][0]:.4f}/{r[k][1]:.4f}"
                for k in ("q50", "q75", "q90", "q95", "rms", "tail5", "tail1", "max")))
            print(f"  flicker: density {r['flicker_density'][0]:.3f}/{r['flicker_density'][1]:.3f}"
                  f"   amplitude {r['flicker_amplitude'][0]:.4f}/{r['flicker_amplitude'][1]:.4f}"
                  f"   (fraction of pixels moving; how hard those move)")
            print(f"  structure: lag2/lag1 {r.get('lag2_over_lag1', 0):.2f}   "
                  f"edge/flat {r['edge_over_flat']:.2f}   luma-corr {r['luma_corr']:+.2f}   "
                  f"static-frac {r['static_fraction']:.2f}   "
                  f"static-spatial-hf {r['static_spatial_hf']:.4f}   "
                  f"row-period {r.get('row_periodicity', 0.0):.2f}@"
                  f"{r.get('row_period_px', 0.0):.0f}px")
            print(f"  noise floor: mc sigma {r.get('mc_sigma', 0.0):.4f} "
                  f"lag {r.get('mc_lag21', 1.0):.2f}   |   "
                  f"flat sigma {r.get('flat_sigma', 0.0):.4f} "
                  f"lag {r.get('flat_lag21', 1.0):.2f} "
                  f"diff-corr {r.get('flat_diff_corr', 0.0):.2f}   "
                  f"(mc = the map's default floor: aligned-residual noise on "
                  f"all pixels; lag ~1 + sigma >= 0.03 = dense real noise)")
            print(f"  spatial floor: sigma {r.get('spatial_sigma', 0.0):.4f}   "
                  f"static grain ~{r.get('static_grain_sigma', 0.0):.4f}   "
                  f"static-banding {r.get('static_row_periodicity', 0.0):.2f}@"
                  f"{r.get('static_row_period_px', 0.0):.0f}px   "
                  f"interlace {r.get('row_interlace', 0.0):.1f}x")
            print(f"  channels: R {r['sigma_R']:.4f}  G {r['sigma_G']:.4f}  B {r['sigma_B']:.4f}")
            print(f"  verdict: {', '.join(diag['labels'])}  risk={diag['risk']}")
            for _msg in diag["warnings"][:2]:
                print(f"    warning: {_msg}")
            for _msg in diag["suggestions"][:2]:
                print(f"    try: {_msg}")
            _pk = sorted(range(len(tr)), key=lambda i: -tr[i])[:3]
            print(f"  frame trace: med {sorted(tr)[len(tr) // 2]:.4f}  max {max(tr):.4f}"
                  "   top flashes: " + ", ".join(
                      f"diff{i + 1} {tr[i]:.4f} (kf+{_kfd(i)})" for i in _pk))
        # ---- tool guidance: name the tool the measurements support --------
        print("[probe] tool guidance:")
        _comb = (estimate_qf(_all_frames) if _all_frames
                 else {"qf": None, "confidence": 0.0})
        if _comb["qf"] is not None:
            print(f"  JPEG ancestry: QF ~{_comb['qf']} "
                  f"(confidence {_comb['confidence']:g}) -> --deblock fbcnn "
                  f"(auto QF tracks it per tile)")
        else:
            print("  no JPEG-family comb (native H.264/HEVC, or combs killed "
                  "by a later re-encode)")
        _bp95 = None
        if _mid_frames:
            _gp = detect_grid_period(_mid_frames)
            _pnon8 = [r[0] for r in (_gp.get("px"), _gp.get("py"))
                      if r is not None and abs(r[0] - 8.0) > 0.3]
            if _pnon8:
                _pm = sum(_pnon8) / len(_pnon8)
                print(f"  grid period ~{_pm:.1f} px (~{_pm / 8.0:.2f}x resize of "
                      f"an 8-grid): footage was compressed then RESIZED; the "
                      f"blockiness map tracks it (period=auto)")
            _blk = estimate_blockiness_map(_mid_frames)
            if _blk is not None:
                _bs = mx.sort(_blk.reshape(-1))
                _bp95 = float(_bs[int(0.95 * (int(_bs.shape[0]) - 1))])
                # clean modern re-encodes read p95 ~0.3 from their own light
                # grid; recommending stdf there costs quality (measured on the
                # corpus controls), so "little" extends past that baseline
                if _bp95 >= 0.6:
                    _bmsg = ("strong coding grid -> --deblock stdf "
                             "--deblock-map auto (strength 0.3-0.4)")
                elif _bp95 >= 0.4:
                    _bmsg = ("mild coding grid -> --deblock stdf "
                             "--deblock-map auto (strength 0.15-0.25)")
                else:
                    _bmsg = ("little grid evidence (clean-encode baseline): "
                             "nothing grid-locked to deblock")
                print(f"  blockiness p95 {_bp95:.2f}: {_bmsg}")
        _mc_med = sorted(_mc_sigs)[len(_mc_sigs) // 2] if _mc_sigs else 0.0
        if "dense sensor noise" in _all_labels:
            print(f"  dense noise (mc floor ~{_mc_med:.3f}) -> --denoise bsvd "
                  f"--noise-map auto")
        elif "sparse edge flicker" in _all_labels:
            print("  sparse edge flicker -> compression cleanup first "
                  "(deblock / fbcnn); plain denoise adds little")
        elif "dense motion, low noise floor" in _all_labels:
            print(f"  clean motion (mc floor ~{_mc_med:.3f}) -> skip or keep "
                  f"denoise minimal; heavy denoise eats texture")
        elif "static/structured grain" in _all_labels:
            print("  static grain: temporal denoisers cannot remove it -> "
                  "spatial cleanup or a small --noise-map-floor")
        if "interlace/field residue" in _all_labels:
            print("  interlace residue -> deinterlace upstream before any "
                  "denoise/deblock (temporal nets smear combing)")
        if "static row banding" in _all_labels:
            print("  static row banding -> spatial-only artifact; temporal "
                  "tools will not touch it")
        _bpp = None
        try:
            _sizes = vr.coded_frame_sizes(Path(args.video))
        except Exception:
            _sizes = []
        if _sizes:
            _seg = [s for s in _sizes[win_start:_pw_end] if s > 0]
            if _seg:
                _bpp = 8.0 * (sum(_seg) / len(_seg)) / float(in_w * in_h)
                _bnote = ("starved encode, damage certain" if _bpp < 0.08
                          else "lean encode" if _bpp < 0.2 else "generous encode")
                print(f"  bpp {_bpp:.3f} over the probed range: {_bnote} "
                      f"(last generation ONLY: high bpp proves nothing after "
                      f"a re-encode)")
        if (_comb["qf"] is None and _bp95 is not None and _bp95 < 0.4
                and _bpp is not None and _bpp < 0.12):
            print("  low bpp with no measurable grid or comb (mush): auto "
                  "dials have no anchor here -> manual call (bsvd for "
                  "shimmer, nafnet for structure)")
        print("[probe] done -- no processing performed")
        return VideoProcessResult()

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
        f"upscale={args.upscale}"
    )
    print(
        f"Encoder: HEVC profile={profile} q={args.encode_quality} "
        f"audio={args.audio_codec if audio_track else 'none'}"
    )

    # ---- Sessions + writers ------------------------------------------------
    # Defer constructing the VSR session, VtfrcSession, and AVWriters until
    # the first source frame is ready.  These hold Metal resources - the HQ VSR
    # model in particular pins ~100MB of Metal heap - so lazy init keeps startup
    # lighter and lets source decoding own the early memory peak.
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
    deflicker_stage: Any = None  # static-state deflicker when --deflicker on

    def _den_members(d: Any) -> list:
        """The denoise slot as a list: None, a single denoiser, or a chain."""
        return [] if d is None else (d if isinstance(d, list) else [d])

    def _build_post_pipeline() -> tuple[
        VsrSession, VtfrcSession | None, AVWriter | None, AVWriter | None,
        Any, Any, Any, Any, Any, Any
    ]:  # session, vtfrc, post_writer, comparison_writer, deblocker, denoiser, upscaler, nafnet, restorer, deflicker
        """Materialize VSR + temporal + writer sessions just-in-time.

        Called on the first source frame so decoder startup gets the early
        Metal heap to itself. Returns (session, vtfrc, post_writer,
        comparison_writer); the dst-pool wiring for zero-copy is set up before
        returning.
        """
        s: Any
        if args.upscale == "none":
            s = NativePassthrough(in_w, in_h, fps=source_fps)
        elif args.upscale in ("basicvsrpp", "realbasicvsr", "realesrgan", "safmn", "esc", "realviformer", "realplksr", "toflow", "metalfx"):
            # Learned MLX upscalers do the upscale in the loop; the session is a
            # passthrough at the output dims that just packs the already-upscaled
            # frame for the encoder.
            s = NativePassthrough(out_w, out_h, fps=source_fps, label=f"{args.upscale} packer")
        else:
            s = VsrSession(in_w, in_h, mode=args.upscale, fps=source_fps)
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
        _den_names = ([] if args.denoise == "off" else
                      [s.strip() for s in str(args.denoise).split(",")
                       if s.strip() and s.strip() != "off"])
        _den_known = {"spatial", "mc", "fastdvdnet", "bsvd", "pvdd", "toflow"}
        _den_bad = [n for n in _den_names if n not in _den_known]
        if _den_bad:
            raise SystemExit(f"--denoise: unknown stage(s) {_den_bad}; "
                             f"choose from {sorted(_den_known)} (comma-chain to "
                             f"run several in order, e.g. mc,bsvd)")
        _den_str = [s.strip() for s in str(args.denoise_strength).split(",")]
        if len(_den_str) == 1:
            # one value distributes over the whole chain; an empty chain
            # takes no strengths at all (so the strict zip below holds)
            _den_str = _den_str * len(_den_names)
        if _den_names and len(_den_str) != len(_den_names):
            raise SystemExit("--denoise-strength: give one value or exactly one "
                             "per chained --denoise stage")
        _den_strengths = [float(s) for s in _den_str] if _den_names else []

        def _stage_map_capable(n: str) -> bool:
            return (n in ("fastdvdnet", "bsvd", "mc")
                    or (n == "pvdd" and "level" in args.pvdd_profile))

        _map_capable = any(_stage_map_capable(n) for n in _den_names)
        if args.noise_map == "auto":
            if _map_capable:
                _floor_note = (f", floor {args.noise_map_floor:g}"
                               if args.noise_map_floor > 0 else "")
                print(f"[noise-map] auto: per-pixel sigma estimated from the footage "
                      f"(gain {args.noise_map_gain:g}{_floor_note}, "
                      f"floor-mode {args.noise_map_floor_mode})")
            elif args.denoise != "off":
                why = ("blind PVDD variant (use --pvdd-variant pvdd_level)"
                       if "pvdd" in _den_names else f"--denoise {args.denoise} has no map input")
                print(f"[noise-map] auto ignored: {why}")
        if args.noise_map_pulse:
            if _map_capable:
                print("[noise-map] pulse: per-frame gain tracking GOP-phase noise "
                      "(I-frame grain refresh)")
            elif args.denoise != "off":
                print("[noise-map] pulse ignored: needs a map-conditioned denoiser "
                      "(fastdvd, bsvd, mc, or a pvdd level variant)")

        def _make_tracker(n: str) -> tuple:
            # one tracker/pulse pair PER map-capable stage: the trackers keep
            # rolling frame state, so sharing one across chained denoisers
            # would double-feed it
            tr = pu = None
            if _stage_map_capable(n):
                if args.noise_map == "auto":
                    from kinovsr.noise_map import NoiseMapTracker
                    tr = NoiseMapTracker(gain=args.noise_map_gain,
                                         motion_cap=args.noise_map_motion_cap,
                                         masking=args.noise_map_masking,
                                         pulse_robust=args.noise_map_pulse,
                                         floor_mode=args.noise_map_floor_mode)
                if args.noise_map_pulse:
                    from kinovsr.noise_map import PulseGain
                    pu = PulseGain()
            return tr, pu

        _denoise_strength_flags = {
            "spatial": args.spatial_strength, "mc": args.mc_strength,
            "fastdvdnet": args.fastdvdnet_strength, "bsvd": args.bsvd_strength,
            "toflow": args.toflow_strength, "pvdd": args.pvdd_strength,
        }

        def _make_denoiser(name: str, stg: float) -> Any:
            nonlocal nm_tracker, nm_pulse
            # A family-level --<family>-strength flag overrides the
            # positional --denoise-strength value for that family.
            override = _denoise_strength_flags.get(name)
            if override is not None:
                stg = override
            tr, pu = _make_tracker(name)
            if tr is not None:
                nm_tracker = tr          # last one wins for the debug report
            if pu is not None:
                nm_pulse = pu
            if name == "spatial":
                return SpatialDenoiser(strength=stg)
            if name == "mc":
                return McTemporalDenoiser(
                    in_w, in_h, strength=stg,
                    window=args.mc_window, clamp=args.mc_clamp,
                    occlusion=args.mc_occlusion, confidence=args.mc_confidence,
                    sigma=args.mc_sigma, gate=args.mc_gate,
                    flow=args.mc_flow, flow_weights=settings.spynet_weights,
                    noise_map=tr,
                    map_refresh=args.noise_map_refresh,
                    pulse=pu,
                    map_floor=args.noise_map_floor,
                )
            if name == "fastdvdnet":
                # Weights ship with the package; --fastdvdnet-profile picks one,
                # or --fastdvdnet-weights / $FASTDVD_WEIGHTS overrides the path.
                return FastDvdDenoiser(
                    settings.fastdvdnet_weights,
                    variant=args.fastdvdnet_profile,
                    strength=stg,
                    noise_map=tr,
                    map_refresh=args.noise_map_refresh,
                    pulse=pu,
                    map_floor=args.noise_map_floor,
                )
            if name == "bsvd":
                # BSVD weights are not bundled; --bsvd-variant picks a local token,
                # or --bsvd-weights / $BSVD_WEIGHTS overrides the path entirely.
                return BsvdDenoiser(
                    settings.bsvd_weights,
                    variant=args.bsvd_profile,
                    strength=stg,
                    dtype=parse_mlx_dtype_name(args.bsvd_dtype),
                    noise_map=tr,
                    map_refresh=args.noise_map_refresh,
                    pulse=pu,
                    map_floor=args.noise_map_floor,
                )
            if name == "toflow":
                # TOFlow's Torch7 checkpoints convert into a safetensors + JSON
                # pair. It is blind (no noise-map input); strength is an output
                # dry/wet blend to keep the old model usable on real footage.
                return TOFlowDenoiser(
                    settings.toflow_weights,
                    variant=args.toflow_profile,
                    flow_scale=args.toflow_flow_scale,
                    passes=args.toflow_passes,
                    graph=settings.toflow_graph,
                    strength=stg,
                    dtype=parse_mlx_dtype_name(args.toflow_dtype),
                )
            raise AssertionError(name)

        den: Any = None
        _dens: list = []
        for _dn, _ds in zip(_den_names, _den_strengths, strict=False):
            if _dn != "pvdd":
                _dens.append(_make_denoiser(_dn, _ds))
                continue
            _dens.append(None)           # placeholder, built below
        if "pvdd" in _den_names:
            # PVDD weights are not bundled; --pvdd-variant picks a local token, or
            # --pvdd-weights / $PVDD_WEIGHTS overrides the path. The `level`
            # variants take a noise-variance dial (--pvdd-noise-preset/-variance);
            # blind variants ignore it. --denoise-strength does not apply.
            from kinovsr.processors.pvdd import LEVEL_PRESETS
            from kinovsr.processors.pvdd.upscaler import PvddDenoiser
            nv = args.pvdd_noise_variance
            if nv is None and args.pvdd_noise_preset != "off":
                nv = LEVEL_PRESETS[args.pvdd_noise_preset]
            _ptr, _ppu = _make_tracker("pvdd")
            if _ptr is not None:
                nm_tracker = _ptr
            if _ppu is not None:
                nm_pulse = _ppu
            _dens[_den_names.index("pvdd")] = PvddDenoiser(
                settings.pvdd_weights,
                variant=args.pvdd_profile,
                window=args.pvdd_window, trim=args.pvdd_trim,
                noise_variance=nv,
                dtype=parse_mlx_dtype_name(args.pvdd_dtype),
                noise_map=_ptr,
                pulse=_ppu,
            )

        if _dens and (args.denoise_luma_strength != 1.0
                      or args.denoise_chroma_strength != 1.0):
            kr, kb = _yuv.coef_for_matrix(_resolved_color[2])    # match the source color matrix
            _dens = [LumaChromaDenoiser(_d, args.denoise_luma_strength,
                                        args.denoise_chroma_strength, kr=kr, kb=kb)
                     for _d in _dens]
        if len(_den_names) > 1:
            print(f"[denoise] chain: {' -> '.join(_den_names)} "
                  f"(strengths {', '.join(f'{s:g}' for s in _den_strengths)})")
        den = (_dens if len(_dens) > 1 else (_dens[0] if _dens else None))

        # --deblock-map auto: a per-pixel blockiness mask (grid-phase detected,
        # boundary-vs-interior contrast) gating the deblocker's correction, so
        # blocked flats get the full correction and detailed/clean areas keep
        # their texture.
        _deb_names = ([] if args.deblock == "off" else
                      [s.strip() for s in str(args.deblock).split(",")
                       if s.strip() and s.strip() != "off"])
        _deb_known = {"stdf", "fbcnn", "toflow"}
        _deb_bad = [n for n in _deb_names if n not in _deb_known]
        if _deb_bad:
            raise SystemExit(f"--deblock: unknown stage(s) {_deb_bad}; choose "
                             f"from {sorted(_deb_known)} (comma-chain to run "
                             f"several in order, e.g. toflow,stdf)")
        _deb_str = [s.strip() for s in str(args.deblock_strength).split(",")]
        if len(_deb_str) == 1:
            # one value distributes over the whole chain; an empty chain
            # takes no strengths at all (so the strict zip below holds --
            # the inherited max(1, ...) form crashed every --deblock-less
            # run once the lint pass made this zip strict)
            _deb_str = _deb_str * len(_deb_names)
        if _deb_names and len(_deb_str) != len(_deb_names):
            raise SystemExit("--deblock-strength: give one value or exactly "
                             "one per chained --deblock stage")
        _deb_strengths = [float(s) for s in _deb_str] if _deb_names else []

        blk_tracker: Any = None
        _deb_mask_capable = [n for n in _deb_names if n in ("stdf", "fbcnn")]
        if args.deblock_map == "auto":
            if _deb_mask_capable:
                print(f"[deblock-map] auto: per-pixel blockiness mask on the "
                      f"correction (gain {args.deblock_map_gain:g})")
            elif args.deblock != "off":
                print(f"[deblock-map] auto ignored: --deblock {args.deblock} unsupported")

        def _make_blk_tracker() -> Any:
            # one tracker per mask-capable stage: it keeps rolling frame
            # state, so sharing across chained deblockers would double-feed
            nonlocal blk_tracker
            if args.deblock_map != "auto":
                return None
            from kinovsr.noise_map import (
                NoiseMapTracker,
                estimate_blockiness_map,
            )
            tr = NoiseMapTracker(gain=args.deblock_map_gain, min_frames=1,
                                 estimator=estimate_blockiness_map)
            blk_tracker = tr             # last one wins for the debug report
            return tr

        defl: Any = None
        if args.deflicker == "on":
            from kinovsr.deflicker import StaticStateDeflicker
            defl = StaticStateDeflicker(window=args.deflicker_window,
                                        band=args.deflicker_band,
                                        strength=args.deflicker_strength,
                                        frac=args.deflicker_frac,
                                        max_fix=args.deflicker_max_fix,
                                        jitter=(args.deflicker_jitter == "on"))
            _jit = ", jitter-compensated" if args.deflicker_jitter == "on" else ""
            print(f"[deflicker] static-state integration: window +/-"
                  f"{args.deflicker_window}, band {args.deflicker_band:g}, "
                  f"frac {args.deflicker_frac:g}, max-fix "
                  f"{args.deflicker_max_fix:g}{_jit} (verified-static only; "
                  f"untouched pixels pass through bit-identical)")

        _deblock_strength_flags = {
            "stdf": args.stdf_strength, "toflow": args.toflow_strength,
            "fbcnn": args.fbcnn_strength,
        }

        def _make_deblocker(name: str, stg: float) -> Any:
            # A family-level --<family>-strength flag overrides the
            # positional --deblock-strength value for that family.
            override = _deblock_strength_flags.get(name)
            if override is not None:
                stg = override
            if name == "stdf":
                from kinovsr.processors.stdf.deblocker import StdfDeblocker
                kr, kb = _yuv.coef_for_matrix(_resolved_color[2])    # match the source color matrix
                return StdfDeblocker(settings.stdf_weights or args.stdf_profile,
                                     strength=stg, kr=kr, kb=kb,
                                     blockiness_map=_make_blk_tracker())
            if name == "toflow":
                # the TOFlow deblock checkpoint in the deblock SLOT (the same
                # processor is also reachable as --denoise toflow
                # --toflow-variant deblock; this placement lets it chain with
                # a real denoiser in the natural order)
                return TOFlowDenoiser(
                    settings.toflow_weights,
                    variant="deblock",
                    flow_scale=args.toflow_flow_scale,
                    passes=args.toflow_passes,
                    graph=settings.toflow_graph,
                    strength=stg,
                    dtype=parse_mlx_dtype_name(args.toflow_dtype),
                )
            if name == "fbcnn":
                from kinovsr.processors.fbcnn import FbcnnDeblocker
                _q = args.fbcnn_quality.strip().lower()
                if _q == "auto":
                    _quality: Any = "auto"
                elif _q in ("blind", "none"):
                    _quality = None
                else:
                    try:
                        _quality = float(_q)
                    except ValueError:
                        raise SystemExit(
                            f"bad --fbcnn-quality {args.fbcnn_quality!r}: expected 'auto', "
                            f"'blind', or a number 1-100") from None
                fb = FbcnnDeblocker(settings.fbcnn_weights,
                                    quality=_quality, strength=stg,
                                    blockiness_map=_make_blk_tracker(),
                                    quality_fallback=args.fbcnn_quality_fallback)
                if _quality == "auto":
                    print(f"[deblock] fbcnn auto QF: per-tile comb measurement "
                          f"(128px tiles, fallback {args.fbcnn_quality_fallback:g})")
                return fb
            raise AssertionError(name)

        _debs = [_make_deblocker(n, s) for n, s in zip(_deb_names, _deb_strengths, strict=True)]
        if len(_deb_names) > 1:
            print(f"[deblock] chain: {' -> '.join(_deb_names)} "
                  f"(strengths {', '.join(f'{s:g}' for s in _deb_strengths)})")
        deb: Any = (_debs if len(_debs) > 1 else (_debs[0] if _debs else None))

        up: Any = None
        if args.upscale == "basicvsrpp":
            from kinovsr.processors.basicvsrpp.upscaler import BasicVsrUpscaler
            # weights spec: --basicvsrpp-weights (token or path) > env > --variant token
            up = BasicVsrUpscaler(
                settings.basicvsrpp_weights or args.basicvsrpp_profile,
                window=args.basicvsrpp_window, trim=args.basicvsrpp_trim,
                flow_mode=args.basicvsrpp_flow,
                history_strength=args.basicvsrpp_history_strength,
                history_gate=args.basicvsrpp_history_gate,
                ensemble=args.basicvsrpp_ensemble,
            )
        elif args.upscale == "realbasicvsr":
            from kinovsr.processors.realbasicvsr.upscaler import RealBasicVsrUpscaler
            up = RealBasicVsrUpscaler(
                settings.realbasicvsr_weights or args.realbasicvsr_profile,
                window=args.realbasicvsr_window,
                trim=args.realbasicvsr_trim,
                dynamic_refine_thres=args.realbasicvsr_clean_threshold,
                clean_iters=args.realbasicvsr_clean_iters,
                residual_strength=args.realbasicvsr_residual_strength,
                flow_consistency=args.realbasicvsr_flow_consistency,
                flow_mode=args.realbasicvsr_flow,
                history_strength=args.realbasicvsr_history_strength,
                history_gate=args.realbasicvsr_history_gate,
            )
        elif args.upscale == "realesrgan":
            from kinovsr.processors.realesrgan.upscaler import RealEsrganUpscaler
            up = RealEsrganUpscaler(
                realesrgan_spec,
                denoise_strength=args.realesrgan_denoise_strength,
            )
        elif args.upscale == "safmn":
            from kinovsr.processors.safmn import SafmnUpscaler
            up = SafmnUpscaler(safmn_spec, safm_up=args.safmn_safm_up,
                               pool_clamp=args.safmn_pool_clamp)
        elif args.upscale == "esc":
            from kinovsr.processors.esc import EscUpscaler
            up = EscUpscaler(esc_spec)
        elif args.upscale == "realplksr":
            from kinovsr.processors.realplksr import RealPlksrUpscaler
            up = RealPlksrUpscaler(realplksr_spec,
                                   dtype=parse_mlx_dtype_name(args.realplksr_dtype))
        elif args.upscale == "realviformer":
            from kinovsr.processors.realviformer import RealViformerUpscaler
            up = RealViformerUpscaler(
                settings.realviformer_weights or args.realviformer_profile,
                window=args.realviformer_window,
                dtype=parse_mlx_dtype_name(args.realviformer_dtype),
                flow_mode=args.realviformer_flow,
                history_strength=args.realviformer_history_strength,
                history_gate=args.realviformer_history_gate,
                history_cleanup=args.realviformer_history_cleanup,
                history_gate_drop=args.realviformer_history_gate_drop,
                history_risk_decay=args.realviformer_history_risk_decay,
                history_static_cap=args.realviformer_history_static_cap,
            )
        elif args.upscale == "toflow":
            from kinovsr.processors.toflow import TOFlowSrUpscaler
            up = TOFlowSrUpscaler(
                settings.toflow_sr_weights,
                graph=settings.toflow_sr_graph,
                dtype=parse_mlx_dtype_name(args.toflow_sr_dtype),
            )
        elif args.upscale == "metalfx":
            from kinovsr.processors.metalfx import MetalFxSpatialUpscaler
            up = MetalFxSpatialUpscaler(scale=args.metalfx_scale)

        naf: Any = None
        if args.nafnet != "off":
            from kinovsr.processors.nafnet import NafnetRestorer
            naf = NafnetRestorer(
                settings.nafnet_weights or args.nafnet,
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
            from kinovsr.processors.basicvsrpp.restorer import BasicVsrRestorer
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
            res = []
            for variant, strength in zip(variants, strengths, strict=True):
                spec = variant
                if len(variants) == 1:
                    spec = settings.basicvsrpp_restore_weights or variant
                res.append(BasicVsrRestorer(
                    spec, window=args.restore_window, trim=args.restore_trim,
                    strength=strength, flow_mode=args.restore_flow,
                    ensemble=args.restore_ensemble))

        return s, v, pw, cw, deb, den, up, naf, res, defl

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
    cut_log_path: Path | None = None
    if args.cut_detect != "off":
        cut_detector = CutDetector(args.cut_detect, args.cut_threshold)
        if args.cut_log:
            cut_log_path = Path(args.cut_log)
            cut_log_path.write_text("", encoding="utf-8")
        print(
            f"Cut detector: mode={args.cut_detect} threshold={args.cut_threshold}"
            + (f", log={args.cut_log}" if args.cut_log else "")
        )

    # ---- Progress bars (stacked, deferred-start, median-window rate) -------
    # PhaseBar's clock starts at the first update() and the displayed pace
    # is the median of the last-N inter-tick intervals; see kinovsr.progress
    # for the rationale.
    target_frame_total = total_frames
    if target_frame_total and do_temporal:
        target_frame_total = int(round(total_frames * (target_fps / source_fps)))
    pbar_total = (
        min(target_frame_total, max_frames)
        if (target_frame_total and max_frames is not None) else
        (max_frames or target_frame_total or None)
    )
    bars = StackedPhaseBars()
    out_pbar = bars.add(
        total=pbar_total,
        desc="OUT frames" if do_temporal else "VSR frames",
        unit="frame",
    )

    # ---- Pipeline loop -----------------------------------------------------
    processed = 0          # source frames upscaled
    appended = 0           # output frames written (= processed when no temporal)
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
        by_name = {"restore": restorer, "deflicker": deflicker_stage, "deblock": deblocker,
                   "denoise": denoiser, "nafnet": nafnet}   # denoise may be a chain (list)
        if args.preprocess_order:
            order = list(args.preprocess_order)
        else:
            order = ["denoise", "deblock"] if args.denoise_first else ["deblock", "denoise"]
            # BasicVSR++ restoration is temporal; run it FIRST by default so it
            # establishes frame-to-frame consistency (killing GOP-periodic
            # compression flicker + sensor noise) before any per-frame stage.
            # deflicker runs before deblock/denoise: stabilizing the codec
            # state flicker first lets the broadband stages run gentler.
            order = ["restore", "deflicker", *order]
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
            # Chunks are lists of decoded CVPixelBuffers in VSR's source format.
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
            # Lazy init of the processing pipeline after the first chunk keeps
            # the VSR HQ model + AVWriter pixel pool out of memory until frames
            # are actually available. _build_post_pipeline() prints status from VsrSession / AVWriter
            # constructors; route those through bars.write() so they appear
            # above the live progress bars instead of stomping mid-line.
            if session is None:
                import contextlib
                import io as _io
                _buf = _io.StringIO()
                with contextlib.redirect_stdout(_buf):
                    session, vtfrc, post_writer, comparison_writer, deblocker, denoiser, upscaler, nafnet, restorer, deflicker_stage = _build_post_pipeline()
                msg = _buf.getvalue().rstrip("\n")
                if msg:
                    bars.write(msg)
                if nafnet is not None and hasattr(nafnet, "set_progress_message"):
                    nafnet.set_progress_message(bars.write)
                # Drive every recurrent stage from the one GOP-aligned schedule.
                # Per-frame stages lack the method and are skipped.
                if gop_schedule is not None:
                    for _st in [*(restorer or []),
                                *_den_members(denoiser),
                                *([upscaler] if upscaler is not None else [])]:
                        if hasattr(_st, "set_schedule"):
                            _st.set_schedule(gop_schedule)
            for i in range(chunk_len):
                if max_frames is not None and appended >= max_frames:
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
                    # --comparison composite.
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
                        # Flush buffered pre-cut frames before resetting so no
                        # lookahead stage's window bridges the cut.
                        if (deblocker is not None or deflicker_stage is not None
                                or restorer is not None
                                or any(hasattr(_d, "flush") for _d in _den_members(denoiser))):
                            for d_rgb, (d_sf, d_sa) in _preprocess_flush():
                                _emit_scaled(d_rgb, d_sf, d_sa)
                        if upscaler is not None:
                            for up_rgb, (u_sf, u_sa) in upscaler.flush():
                                _emit(up_rgb, u_sf, u_sa)
                        session.reset_temporal_context()
                        for _d in _den_members(deblocker):
                            _d.reset()
                        for _d in _den_members(denoiser):
                            _d.reset()
                        if deflicker_stage is not None:
                            deflicker_stage.reset()
                        if nafnet is not None:
                            nafnet.reset()
                        if cut_log_path is not None:
                            with cut_log_path.open("a", encoding="utf-8") as cut_log:
                                cut_log.write(f"{processed}\n")

                    # Produce this input's output frame(s). No denoise -> the raw
                    # frame goes straight to VSR; spatial/mc denoise one frame in
                    # step; fastdvd buffers and emits centered-window frames once
                    # their two future neighbours have arrived (feed() may return
                    # nothing now; the tail drains after the loop). f32 RGB [0,1].
                    if (deblocker is not None or denoiser is not None
                            or deflicker_stage is not None
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

                    # Drop this frame's reference so its memory (the MLX array,
                    # or the decoded CVPixelBuffer) can be freed now instead of
                    # staying resident until the outer `del chunk` at chunk-end.
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

            if max_frames is not None and appended >= max_frames:
                break
            del chunk
            gc.collect()
        # Drain any frames a lookahead preprocessor still holds.
        if (deblocker is not None or restorer is not None
                or deflicker_stage is not None
                or any(hasattr(_d, "flush") for _d in _den_members(denoiser))) and session is not None:
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
        for _d in _den_members(denoiser):
            _d.close()
        if deflicker_stage is not None:
            _dst = deflicker_stage.stats()
            _jit = (f", compensated jitter avg {_dst['jitter_px']:.2f}px"
                    if _dst["jitter_px"] else "")
            print(f"[deflicker] run avg: static-verified "
                  f"{_dst['verified'] * 100:.1f}% of pixels, oscillatory "
                  f"{_dst['oscillatory'] * 100:.1f}%, fired "
                  f"{_dst['fired'] * 100:.1f}%, applied "
                  f"{_dst['applied'] * 1000:.2f}e-3 luma{_jit} "
                  f"(verification is the scope gate: bounded by "
                  f"camera/subject motion, not band/window)")
        if deblocker is not None:
            _qi = getattr(deblocker, "last_qf_info", None)
            if _qi is not None:
                _qg = getattr(deblocker, "_qf_grid", None)
                _qmed = ""
                if _qg is not None:
                    _qs = mx.sort(_qg.reshape(-1))
                    _qmed = f"  tile QF median {float(_qs[int(_qs.shape[0]) // 2]):.0f}"
                _mode = _qi.get("mode", "measured")
                _note = {"measured": "per-tile measured",
                         "gentle": "sparse combs -> mostly gentle",
                         "fallback": "no JPEG evidence -> fallback QF"}[_mode]
                print(f"[deblock] fbcnn auto QF: global {_qi['global']['qf']} "
                      f"(conf {_qi['global']['confidence']:g})  "
                      f"comb coverage {_qi['coverage'] * 100:.0f}%{_qmed}  "
                      f"[{_note}]")
        _deb_dbg = [d for d in _den_members(deblocker)
                    if getattr(d, "last_blockiness_map", None) is not None]
        if _deb_dbg and args.deblock_map == "auto":
            _bm = _deb_dbg[0].last_blockiness_map
            if _bm is None:
                print("[deblock-map] no mask was estimated (no frames deblocked?)")
            else:
                _s = mx.sort(_bm.reshape(-1))
                _n = _s.shape[0]
                print(f"[deblock-map] blockiness mask: median {float(_s[_n // 2]):.3f}  "
                      f"p95 {float(_s[int(0.95 * (_n - 1))]):.3f}  max {float(_s[-1]):.3f}  "
                      f"({float(mx.mean((_bm > 0.5).astype(mx.float32))) * 100:.0f}% of frame > 0.5)")
                if args.noise_map_debug and post_writer is not None:
                    from kinovsr.media.images import save_image
                    _vp = Path(post_writer.path)
                    _png = _vp.with_name(_vp.stem + "_blockmap.png")
                    _u8 = (mx.clip(_bm[:, :, 0], 0, 1) * 255).astype(mx.uint8)
                    save_image(mx.stack([_u8, _u8, _u8], axis=-1), _png)
                    print(f"[deblock-map] mask written: {_png}")
        for _d in _den_members(denoiser):
            _mc = _d if hasattr(_d, "gate_openness") else getattr(_d, "_base", None)
            if _mc is not None and hasattr(_mc, "gate_openness") and _mc.gate_openness > 0:
                print(f"[denoise] mc gate openness: {_mc.gate_openness * 100:.1f}% "
                      f"of the strength ceiling realized (flow={_mc.flow_source}; "
                      f"low = flow-limited, the lever is a better flow, not "
                      f"more strength)")
        _den_dbg = _den_members(denoiser)
        if _den_dbg and args.noise_map_pulse:
            _pl = None
            for _d in _den_dbg:
                _pl = (getattr(_d, "_pulse_log", None)
                       or getattr(getattr(_d, "_base", None), "_pulse_log", None))
                if _pl:
                    break
            if _pl:
                _ps = sorted(_pl)
                print(f"[noise-map] pulse gain over {len(_ps)} frames: "
                      f"min {_ps[0]:.2f}  median {_ps[len(_ps) // 2]:.2f}  "
                      f"max {_ps[-1]:.2f}  ({sum(1 for g in _ps if g > 1.2)} frames > 1.2)")
        if _den_dbg and args.noise_map == "auto":
            # surface the estimated map (unwrap the luma/chroma splitter if
            # present; in a chain, report the first member that has one)
            _nm_src = None
            _nm_holder = None
            for _d in _den_dbg:
                _nm_src = getattr(_d, "last_noise_map", None)
                _nm_holder = _d
                if _nm_src is None:
                    _nm_src = getattr(getattr(_d, "_base", None), "last_noise_map", None)
                    _nm_holder = getattr(_d, "_base", None) if _nm_src is not None else _nm_holder
                if _nm_src is not None:
                    break
            if _nm_src is not None:
                _s = mx.sort(_nm_src.reshape(-1))
                _n = _s.shape[0]
                print(f"[noise-map] estimated sigma: min {float(_s[0]):.4f}  "
                      f"median {float(_s[_n // 2]):.4f}  p95 {float(_s[int(0.95 * (_n - 1))]):.4f}  "
                      f"max {float(_s[-1]):.4f}")
                # what the net actually receives: the estimate clamped into the
                # consumer's conditioning bounds (trained range and/or user floor)
                _dsrc = _nm_holder if hasattr(_nm_holder, "_map_floor") else getattr(_nm_holder, "_base", None)
                _lo = max(float(getattr(_dsrc, "SIGMA_MIN", 0.0) or 0.0),
                          float(getattr(_dsrc, "_map_floor", 0.0) or 0.0))
                _hi = float(getattr(_dsrc, "SIGMA_MAX", 0.0) or 0.0)
                if _lo > 0.0 or _hi > 0.0:
                    _e = mx.sort(mx.clip(_nm_src, _lo, _hi if _hi > 0 else 1.0).reshape(-1))
                    print(f"[noise-map] effective conditioning: min {float(_e[0]):.4f}  "
                          f"median {float(_e[_n // 2]):.4f}  max {float(_e[-1]):.4f}  "
                          f"(floor {_lo:.4f}{f', ceil {_hi:.4f}' if _hi > 0 else ''})")
                if args.noise_map_debug and post_writer is not None:
                    from kinovsr.media.images import save_image
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

    elapsed = time.perf_counter() - t_total
    rate = appended / elapsed if elapsed > 0 else 0
    print(f"Processed {processed} source frames, wrote {appended} output frames "
          f"in {elapsed:.2f}s ({rate:.2f} fps out)")
    if post_writer is not None:
        print(f"Post: {post_writer.path}")
    if comparison_writer is not None:
        print(f"Comparison: {comparison_writer.path}")
    return VideoProcessResult(
        post_path=Path(post_writer.path) if post_writer is not None else None,
        comparison_path=(Path(comparison_writer.path)
                         if comparison_writer is not None else None),
        frames_out=appended,
        elapsed_s=elapsed,
    )
