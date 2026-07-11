"""Compare two videos by temporal second-difference (TSD) - a frame-to-frame
"shimmer" / non-smoothness metric.

Why TSD?

  The naive frame-to-frame difference |frame[i] - frame[i-1]| picks up both
  real motion and shimmer in one number, so it can't tell them apart.

  TSD = |frame[i] - (frame[i-1] + frame[i+1]) / 2|

  cancels smooth (linear) motion - real motion is approximately linear
  frame-to-frame at small scales, so the midpoint average predicts frame[i]
  well - and leaves only non-smooth temporal variation: shimmer, flicker,
  cuts, encoding artifacts.  Useful for A/B-ing two videos that show the
  same source content but went through different processing (VSR modes,
  encoder settings, temporal upscaling, etc.).

Streams frames through ffmpeg at full source resolution as gray8 luma -
keeps a 3-frame ring buffer in memory, computes per-frame TSD plus an
8x8 spatial grid of per-patch shimmer, never materializes more than ~30 MB
of luma at any time regardless of video resolution.

Usage:
    scripts/compare_video_shimmer.py <video-a> <video-b>
    scripts/compare_video_shimmer.py a.mp4 b.mp4 --label-a image --label-b balanced
    scripts/compare_video_shimmer.py a.mp4 b.mp4 --grid 16  # finer spatial grid
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

from kinovsr.ui.console import get_console


def _print(*parts: object) -> None:
    get_console().print(*parts, markup=False, highlight=False)


def _percentile(values: mx.array, q: float) -> float:
    """Linear-interpolated percentile of a 1-D array (numpy semantics)."""
    ordered = mx.sort(values.reshape(-1))
    n = ordered.shape[0]
    if n == 1:
        return float(ordered[0])
    pos = q / 100.0 * (n - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, n - 1)
    return float(ordered[lo]) * (1.0 - frac) + float(ordered[hi]) * frac


def probe_dimensions(path: str) -> tuple[int, int, float, int]:
    """Get (width, height, fps, total_frames) from a video via ffprobe.

    Returns total_frames=0 if it's not in the metadata (some containers
    don't store it; we'll just count what ffmpeg streams to us).
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "default=noprint_wrappers=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    fields = dict(line.split("=", 1) for line in out if "=" in line)
    w = int(fields["width"])
    h = int(fields["height"])
    num, den = fields.get("r_frame_rate", "24/1").split("/")
    fps = float(num) / float(den) if float(den) != 0 else 24.0
    try:
        nframes = int(fields.get("nb_frames", "0"))
    except ValueError:
        nframes = 0
    return w, h, fps, nframes


def stream_tsd(path: str, w: int, h: int, grid: int) -> tuple[mx.array, mx.array]:
    """Stream the video as gray8 luma via ffmpeg; return per-frame TSD and
    per-patch (grid x grid) mean TSD.

    Memory: 3 frames * w * h bytes (e.g. 3 * 4096 * 2304 = ~28 MB at 4K).
    """
    frame_bytes = w * h
    patch_h = h // grid
    patch_w = w // grid
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", path,
        "-vf", "format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None

    ring: list[mx.array] = []
    tsd_per_frame: list[float] = []
    tsd_per_patch = mx.zeros((grid, grid), dtype=mx.float32)
    n_pairs = 0

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = mx.array(memoryview(raw)).reshape(h, w).astype(mx.int16)
            ring.append(frame)
            if len(ring) > 3:
                ring.pop(0)
            if len(ring) == 3:
                tsd = mx.abs(ring[1] - (ring[0] + ring[2]) // 2)
                tsd_per_frame.append(float(tsd.astype(mx.float32).mean()))
                # Average over each grid cell.  Trim trailing pixels that
                # don't fit a full row/column of patches.
                trimmed = tsd[: patch_h * grid, : patch_w * grid]
                patches = trimmed.astype(mx.float32).reshape(
                    grid, patch_h, grid, patch_w).mean(axis=(1, 3))
                tsd_per_patch = tsd_per_patch + patches
                mx.eval(tsd_per_patch)   # keep the lazy graph per-frame flat
                n_pairs += 1
    finally:
        proc.wait()
        if proc.returncode not in (0, None):
            sys.stderr.write(f"ffmpeg exited with {proc.returncode} on {path}\n")

    tsd_per_patch = tsd_per_patch / max(n_pairs, 1)
    return mx.array(tsd_per_frame), tsd_per_patch


def _patch_glyph(value: float) -> str:
    """Compact 2-char glyph for the per-patch ascii heatmap."""
    if value >= 0.10:
        return "##"
    if value >= 0.05:
        return "++"
    if value >= 0.02:
        return "+ "
    if value <= -0.10:
        return "@@"
    if value <= -0.05:
        return "--"
    if value <= -0.02:
        return "- "
    return ". "


def run_shimmer(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="kinovsr metrics shimmer", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video_a", help="First video to analyze.")
    p.add_argument("video_b", help="Second video to analyze.")
    p.add_argument("--label-a", default=None, help="Label for video A (defaults to filename stem).")
    p.add_argument("--label-b", default=None, help="Label for video B (defaults to filename stem).")
    p.add_argument(
        "--grid", type=int, default=8,
        help="Spatial grid resolution for the per-patch heatmap (default 8 = 8x8 grid).",
    )
    p.add_argument(
        "--per-second", action="store_true",
        help="Also print mean TSD bucketed by output-clip second.",
    )
    args = p.parse_args(argv)

    label_a = args.label_a or Path(args.video_a).stem
    label_b = args.label_b or Path(args.video_b).stem

    wa, ha, fps_a, _ = probe_dimensions(args.video_a)
    wb, hb, fps_b, _ = probe_dimensions(args.video_b)
    if (wa, ha) != (wb, hb):
        p.error(
            f"video dimensions differ: {args.video_a} is {wa}x{ha}, "
            f"{args.video_b} is {wb}x{hb}. TSD comparison requires matched resolutions."
        )
    fps = fps_a  # for per-second bucketing
    w, h = wa, ha

    _print(f"Resolution:  {w}x{h}")
    _print(f"Grid:        {args.grid}x{args.grid}  ({h // args.grid}x{w // args.grid} pixels per patch)")
    _print(f"FPS:         {fps:.3f}")
    _print()

    _print(f"Streaming {label_a}: {args.video_a}")
    tsd_a, patches_a = stream_tsd(args.video_a, w, h, args.grid)
    _print(f"  {len(tsd_a)} TSD samples")

    _print(f"Streaming {label_b}: {args.video_b}")
    tsd_b, patches_b = stream_tsd(args.video_b, w, h, args.grid)
    _print(f"  {len(tsd_b)} TSD samples")

    n = min(len(tsd_a), len(tsd_b))
    if n < len(tsd_a) or n < len(tsd_b):
        _print(f"  (truncating both series to {n} frame pairs for comparison)")
    tsd_a = tsd_a[:n]
    tsd_b = tsd_b[:n]
    diff = tsd_a - tsd_b

    _print()
    _print("=== Temporal second-difference (TSD) per frame ===")
    _print(f"{'video':<20s}{'mean':>10s}{'median':>10s}{'std':>10s}{'p95':>10s}{'max':>10s}")
    for label, series in ((label_a, tsd_a), (label_b, tsd_b)):
        _print(
            f"{label:<20s}"
            f"{float(series.mean()):>10.4f}{_percentile(series, 50):>10.4f}"
            f"{float(mx.std(series)):>10.4f}"
            f"{_percentile(series, 95):>10.4f}{float(series.max()):>10.4f}"
        )

    _print()
    _print(f"=== Shimmer signal: TSD({label_a}) - TSD({label_b}) ===")
    _print(f"  mean:    {float(diff.mean()):+.4f}")
    _print(f"  median:  {_percentile(diff, 50):+.4f}")
    _print(f"  p5:      {_percentile(diff, 5):+.4f}")
    _print(f"  p95:     {_percentile(diff, 95):+.4f}")
    _print(f"  max:     {float(diff.max()):+.4f}  (frame pair {int(mx.argmax(diff)) + 1})")
    _print(f"  min:     {float(diff.min()):+.4f}  (frame pair {int(mx.argmin(diff)) + 1})")
    a_higher = float(mx.mean(diff > 0)) * 100
    _print(f"  fraction of frame pairs where {label_a} > {label_b}: {a_higher:.1f}%")

    if args.per_second:
        _print()
        _print("=== Per-second mean TSD ===")
        step = max(1, int(round(fps)))
        _print(f"  {'sec':>4s}  {label_a:>16s}  {label_b:>16s}  {'diff':>8s}")
        for s in range((n + step - 1) // step):
            seg_a = tsd_a[s * step:(s + 1) * step]
            seg_b = tsd_b[s * step:(s + 1) * step]
            if seg_a.size == 0:
                continue
            sa, sb = float(seg_a.mean()), float(seg_b.mean())
            _print(f"  {s:>4d}  {sa:>16.4f}  {sb:>16.4f}  {sa - sb:>+8.4f}")

    diff_patches = patches_a - patches_b
    _print()
    _print(f"=== Per-patch shimmer ({label_a} - {label_b}, {args.grid}x{args.grid} grid) ===")
    _print("  (rows top->bottom, cols left->right)")
    for r in range(args.grid):
        row = "  "
        for c in range(args.grid):
            row += _patch_glyph(float(diff_patches[r, c])) + " "
        _print(row)
    _print(
        "  legend: ##>=0.10  ++>=0.05  + >=0.02  . ~0  "
        "- <=-0.02  --<=-0.05  @@<=-0.10"
    )
    flat_max = int(mx.argmax(diff_patches))
    flat_min = int(mx.argmin(diff_patches))
    grid_w = diff_patches.shape[1]
    _print(f"  max:  {float(diff_patches.max()):+.4f} at row {flat_max // grid_w}, col {flat_max % grid_w}")
    _print(f"  min:  {float(diff_patches.min()):+.4f} at row {flat_min // grid_w}, col {flat_min % grid_w}")


_SUBCOMMANDS = ("shimmer", "perceptual", "niqe", "faces", "dover", "musiq")


def run_metrics_command(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        _print(f"usage: kinovsr metrics {{{'|'.join(_SUBCOMMANDS)}}} ...")
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name == "shimmer":
        return run_shimmer(rest) or 0
    if name == "perceptual":
        from kinovsr.eval.perceptual_metrics import run_perceptual

        return run_perceptual(rest) or 0
    if name == "niqe":
        from kinovsr.eval.niqe import run_niqe

        return run_niqe(rest) or 0
    if name == "faces":
        from kinovsr.eval.face_yunet_metrics import run_faces

        return run_faces(rest) or 0
    if name == "dover":
        from kinovsr.eval.dover_score import run_dover

        return run_dover(rest) or 0
    if name == "musiq":
        from kinovsr.eval.musiq_score import run_musiq

        return run_musiq(rest) or 0
    _print(f"unknown metrics subcommand {name!r} "
           f"(available: {', '.join(_SUBCOMMANDS)})")
    return 2

