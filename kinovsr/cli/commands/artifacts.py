"""Compare two matched VSR videos with a DeSRA-style artifact map.

The intended use is recurrent VSR debugging:

    scripts/compare_vsr_artifacts.py --reference window1.mp4 --candidate recurrent.mp4 \
        --output-dir /tmp/vsr_artifacts --heatmap-video --overlay-video

`reference` should be the safer render, for example window=1/history-free.
`candidate` should be the artifact-prone render, for example a recurrent run.

The score is the SSIM contrast term used by DeSRA's released detector: local
standard deviations are compared in an 11x11 Gaussian window, then normalized
so ordinary high-frequency regions are not automatically flagged.  Low contrast
means one render invented or destroyed local texture energy.  This script
reports risk = 1 - contrast, plus an optional flat-region weighted risk that
uses reference texture as a lightweight stand-in for DeSRA's semantic weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
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


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int | None


@dataclass
class FrameMetrics:
    index: int
    raw_mean: float
    raw_p95: float
    raw_p99: float
    raw_area: float
    weighted_mean: float
    weighted_p95: float
    weighted_p99: float
    weighted_area: float
    ref_texture_mean: float


class RawVideoStream:
    """Selected-frame raw RGB reader backed by ffmpeg."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        start_frame: int,
        stride: int,
        max_width: int,
    ) -> None:
        self.path = path
        self.width, self.height = scaled_dimensions(width, height, max_width)
        self._frame_bytes = self.width * self.height * 3
        self._start_frame = start_frame
        self._stride = stride
        self._next_index = 0

        vf_parts = []
        if (self.width, self.height) != (width, height):
            vf_parts.append(f"scale={self.width}:{self.height}:flags=lanczos")
        vf_parts.append("format=rgb24")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            ",".join(vf_parts),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if self._proc.stdout is None:
            raise RuntimeError("ffmpeg did not expose stdout")

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._proc.stdout is not None:
            self._proc.stdout.close()
        if self._proc.stderr is not None:
            self._proc.stderr.close()

    def read_selected(self) -> tuple[int, mx.array] | None:
        while True:
            raw = self._proc.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                self._proc.wait()
                if self._proc.returncode not in (0, None):
                    err = ""
                    if self._proc.stderr is not None:
                        err = self._proc.stderr.read().decode("utf-8", "replace").strip()
                    suffix = f": {err}" if err else ""
                    raise RuntimeError(f"ffmpeg exited with {self._proc.returncode} on {self.path}{suffix}")
                return None

            frame_index = self._next_index
            self._next_index += 1
            if frame_index < self._start_frame:
                continue
            if (frame_index - self._start_frame) % self._stride:
                continue

            frame = mx.array(memoryview(raw)).reshape(
                self.height, self.width, 3)
            return frame_index, frame


class RawVideoWriter:
    """Small raw RGB -> H.264 writer for diagnostic maps."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 18) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        self.path = path
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg did not expose stdin")

    def write(self, frame: mx.array) -> None:
        self._proc.stdin.write(bytes(memoryview(mx.contiguous(frame))))

    def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg exited with {self._proc.returncode} while writing {self.path}")


def run_command(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(path: Path) -> VideoInfo:
    data = json.loads(
        run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-print_format",
                "json",
                str(path),
            ]
        )
    )
    stream = data["streams"][0]
    fps = parse_rate(stream.get("avg_frame_rate")) or parse_rate(stream.get("r_frame_rate")) or 24.0
    frames = stream.get("nb_frames")
    frame_count: int | None = None
    if frames is not None and frames != "N/A":
        frame_count = int(frames)
    elif stream.get("duration") is not None:
        frame_count = max(1, int(round(float(stream["duration"]) * fps)))
    return VideoInfo(int(stream["width"]), int(stream["height"]), fps, frame_count)


def scaled_dimensions(width: int, height: int, max_width: int) -> tuple[int, int]:
    if max_width <= 0 or width <= max_width:
        return width, height
    scaled_w = max_width - (max_width % 2)
    scaled_h = max(2, int(round(height * scaled_w / width)))
    if scaled_h % 2:
        scaled_h += 1
    return scaled_w, scaled_h


def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> mx.array:
    if size <= 0 or size % 2 == 0:
        raise ValueError("kernel size must be a positive odd integer")
    radius = size // 2
    xs = mx.arange(-radius, radius + 1, dtype=mx.float32)
    kernel = mx.exp(-(xs * xs) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).astype(mx.float32)


def _edge_pad_axis(image: mx.array, radius: int, axis: int) -> mx.array:
    if axis == 1:
        left = mx.broadcast_to(
            image[:, :1], (image.shape[0], radius, image.shape[2]))
        right = mx.broadcast_to(
            image[:, -1:], (image.shape[0], radius, image.shape[2]))
        return mx.concatenate([left, image, right], axis=1)
    top = mx.broadcast_to(image[:1], (radius, *image.shape[1:]))
    bottom = mx.broadcast_to(image[-1:], (radius, *image.shape[1:]))
    return mx.concatenate([top, image, bottom], axis=0)


def blur_rgb(image: mx.array, kernel: mx.array) -> mx.array:
    radius = int(kernel.shape[0] // 2)
    weights = [float(w) for w in kernel]
    padded_x = _edge_pad_axis(image, radius, axis=1)
    tmp = mx.zeros(image.shape, dtype=mx.float32)
    for i, weight in enumerate(weights):
        tmp = tmp + weight * padded_x[:, i:i + image.shape[1], :]

    padded_y = _edge_pad_axis(tmp, radius, axis=0)
    out = mx.zeros(tmp.shape, dtype=mx.float32)
    for i, weight in enumerate(weights):
        out = out + weight * padded_y[i:i + image.shape[0], :, :]
    return out


def local_std_rgb(image: mx.array, kernel: mx.array) -> mx.array:
    image = image.astype(mx.float32)
    mean = blur_rgb(image, kernel)
    mean_sq = blur_rgb(image * image, kernel)
    var = mean_sq - mean * mean
    return mx.sqrt(mx.maximum(var, 0.0))


def desra_contrast(
    reference_rgb: mx.array,
    candidate_rgb: mx.array,
    kernel: mx.array,
    constant: float = 0.03 * 0.03,
) -> tuple[mx.array, mx.array]:
    """Return (contrast, reference_texture) for RGB float frames in [0, 1]."""
    sigma_ref = local_std_rgb(reference_rgb, kernel)
    sigma_cand = local_std_rgb(candidate_rgb, kernel)
    numerator = 2.0 * sigma_ref * sigma_cand + constant
    denominator = sigma_ref * sigma_ref + sigma_cand * sigma_cand + constant
    contrast_rgb = mx.clip(numerator / mx.maximum(denominator, 1e-12), 0.0, 1.0)
    return contrast_rgb.mean(axis=2), sigma_ref.mean(axis=2)


def flat_weight_from_texture(
    texture: mx.array,
    busy_floor: float = 0.25,
    low_pct: float = 20.0,
    high_pct: float = 85.0,
) -> mx.array:
    """Softly emphasize flat/reference-smooth regions.

    This is the segmentation-free analog of DeSRA's semantic tolerance: busy
    regions are allowed more generated texture, while flat regions are not.
    """
    lo = _percentile(texture, low_pct)
    hi = _percentile(texture, high_pct)
    if hi <= lo + 1e-6:
        return mx.ones(texture.shape, dtype=mx.float32)
    busy = mx.clip((texture - lo) / (hi - lo), 0.0, 1.0)
    return (busy_floor + (1.0 - busy_floor) * (1.0 - busy)).astype(mx.float32)


def frame_metrics(
    frame_index: int,
    contrast: mx.array,
    texture: mx.array,
    threshold: float,
    busy_floor: float,
) -> tuple[FrameMetrics, mx.array, mx.array]:
    raw_risk = mx.clip(1.0 - contrast, 0.0, 1.0)
    weighted_risk = raw_risk * flat_weight_from_texture(texture, busy_floor=busy_floor)
    metrics = FrameMetrics(
        index=frame_index,
        raw_mean=float(raw_risk.mean()),
        raw_p95=_percentile(raw_risk, 95.0),
        raw_p99=_percentile(raw_risk, 99.0),
        raw_area=float(mx.mean(raw_risk >= threshold)),
        weighted_mean=float(weighted_risk.mean()),
        weighted_p95=_percentile(weighted_risk, 95.0),
        weighted_p99=_percentile(weighted_risk, 99.0),
        weighted_area=float(mx.mean(weighted_risk >= threshold)),
        ref_texture_mean=float(texture.mean()),
    )
    return metrics, raw_risk, weighted_risk


def risk_heatmap(risk: mx.array, threshold: float) -> mx.array:
    v = mx.clip(risk / max(threshold, 1e-6), 0.0, 1.0)
    r = mx.clip(v * 255.0, 0.0, 255.0)
    g = mx.clip((v ** 2) * 160.0, 0.0, 255.0)
    b = mx.clip((1.0 - v) * 50.0, 0.0, 255.0)
    return mx.stack([r, g, b], axis=-1).astype(mx.uint8)


def risk_overlay(candidate_rgb_u8: mx.array, risk: mx.array, threshold: float) -> mx.array:
    alpha = mx.clip(risk / max(threshold, 1e-6), 0.0, 1.0)[..., None] * 0.65
    red = mx.broadcast_to(
        mx.array([255.0, 0.0, 0.0], dtype=mx.float32), candidate_rgb_u8.shape)
    base = candidate_rgb_u8.astype(mx.float32)
    return mx.clip(base * (1.0 - alpha) + red * alpha, 0.0, 255.0).astype(mx.uint8)


def write_outputs(output_dir: Path, rows: list[FrameMetrics], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "artifact_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(FrameMetrics.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    (output_dir / "artifact_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize(rows: list[FrameMetrics], args: argparse.Namespace, info: VideoInfo) -> dict:
    if not rows:
        raise RuntimeError("no frames were compared")

    def series(name: str) -> mx.array:
        return mx.array([getattr(row, name) for row in rows], dtype=mx.float32)

    raw_mean = series("raw_mean")
    raw_area = series("raw_area")
    weighted_mean = series("weighted_mean")
    weighted_area = series("weighted_area")
    worst_raw = rows[int(mx.argmax(raw_area))]
    worst_weighted = rows[int(mx.argmax(weighted_area))]
    return {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "frames_compared": len(rows),
        "resolution": {"width": info.width, "height": info.height},
        "fps": info.fps,
        "threshold": args.threshold,
        "risk": {
            "raw_mean": float(raw_mean.mean()),
            "raw_p95_frame_mean": _percentile(raw_mean, 95.0),
            "raw_area_mean": float(raw_area.mean()),
            "raw_area_p95": _percentile(raw_area, 95.0),
            "weighted_mean": float(weighted_mean.mean()),
            "weighted_p95_frame_mean": _percentile(weighted_mean, 95.0),
            "weighted_area_mean": float(weighted_area.mean()),
            "weighted_area_p95": _percentile(weighted_area, 95.0),
        },
        "worst_frames": {
            "raw_area": worst_raw.__dict__,
            "weighted_area": worst_weighted.__dict__,
        },
    }


def compare(args: argparse.Namespace) -> dict:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")

    ref_info = probe_video(args.reference)
    cand_info = probe_video(args.candidate)
    if (ref_info.width, ref_info.height) != (cand_info.width, cand_info.height):
        raise ValueError(
            f"video dimensions differ: reference {ref_info.width}x{ref_info.height}, "
            f"candidate {cand_info.width}x{cand_info.height}"
        )

    metric_w, metric_h = scaled_dimensions(ref_info.width, ref_info.height, args.max_width)
    metric_info = VideoInfo(metric_w, metric_h, ref_info.fps, ref_info.frames)
    kernel = gaussian_kernel(args.window_size, args.gaussian_sigma)

    ref_stream = RawVideoStream(
        args.reference, ref_info.width, ref_info.height,
        args.start_frame, args.stride, args.max_width,
    )
    cand_stream = RawVideoStream(
        args.candidate, cand_info.width, cand_info.height,
        args.start_frame, args.stride, args.max_width,
    )

    heatmap_writer = None
    overlay_writer = None
    if args.output_dir is not None and args.heatmap_video:
        heatmap_writer = RawVideoWriter(args.output_dir / "artifact_heatmap.mp4", metric_w, metric_h, ref_info.fps)
    if args.output_dir is not None and args.overlay_video:
        overlay_writer = RawVideoWriter(args.output_dir / "artifact_overlay.mp4", metric_w, metric_h, ref_info.fps)

    rows: list[FrameMetrics] = []
    try:
        while args.max_frames <= 0 or len(rows) < args.max_frames:
            ref_item = ref_stream.read_selected()
            cand_item = cand_stream.read_selected()
            if ref_item is None or cand_item is None:
                break
            ref_idx, ref_u8 = ref_item
            cand_idx, cand_u8 = cand_item
            if ref_idx != cand_idx:
                raise RuntimeError(f"frame selection drifted: reference {ref_idx}, candidate {cand_idx}")

            ref = ref_u8.astype(mx.float32) / 255.0
            cand = cand_u8.astype(mx.float32) / 255.0
            contrast, texture = desra_contrast(ref, cand, kernel)
            row, raw_risk, weighted_risk = frame_metrics(
                ref_idx, contrast, texture, args.threshold, args.busy_floor,
            )
            rows.append(row)

            map_risk = weighted_risk if args.map_metric == "weighted" else raw_risk
            if heatmap_writer is not None:
                heatmap_writer.write(risk_heatmap(map_risk, args.threshold))
            if overlay_writer is not None:
                overlay_writer.write(risk_overlay(cand_u8, map_risk, args.threshold))

            if args.progress and len(rows) % args.progress == 0:
                _print(
                    f"[{len(rows):5d}] frame={ref_idx:6d} "
                    f"raw_mean={row.raw_mean:.5f} raw_area={row.raw_area:.5f} "
                    f"weighted_area={row.weighted_area:.5f}",
                )
    finally:
        ref_stream.close()
        cand_stream.close()
        if heatmap_writer is not None:
            heatmap_writer.close()
        if overlay_writer is not None:
            overlay_writer.close()

    summary = summarize(rows, args, metric_info)
    if args.output_dir is not None:
        write_outputs(args.output_dir, rows, summary)
    return summary


def print_summary(summary: dict) -> None:
    _print("=== VSR Artifact Risk ===")
    _print(f"reference: {summary['reference']}")
    _print(f"candidate: {summary['candidate']}")
    _print(
        f"frames:    {summary['frames_compared']} at "
        f"{summary['resolution']['width']}x{summary['resolution']['height']}"
    )
    _print(f"threshold: {summary['threshold']:.3f}")
    _print()
    risk = summary["risk"]
    _print("metric                 mean       p95")
    _print(f"raw frame risk     {risk['raw_mean']:.6f}  {risk['raw_p95_frame_mean']:.6f}")
    _print(f"raw area fraction  {risk['raw_area_mean']:.6f}  {risk['raw_area_p95']:.6f}")
    _print(f"flat-weighted risk {risk['weighted_mean']:.6f}  {risk['weighted_p95_frame_mean']:.6f}")
    _print(f"weighted area      {risk['weighted_area_mean']:.6f}  {risk['weighted_area_p95']:.6f}")
    _print()
    worst = summary["worst_frames"]
    _print(
        "worst raw-area frame:      "
        f"{worst['raw_area']['index']} area={worst['raw_area']['raw_area']:.6f} "
        f"p99={worst['raw_area']['raw_p99']:.6f}"
    )
    _print(
        "worst weighted-area frame: "
        f"{worst['weighted_area']['index']} area={worst['weighted_area']['weighted_area']:.6f} "
        f"p99={worst['weighted_area']['weighted_p99']:.6f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinovsr artifacts compare", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reference", required=True, type=Path, help="Safer reference video.")
    parser.add_argument("--candidate", required=True, type=Path, help="Artifact-prone candidate video.")
    parser.add_argument("--output-dir", type=Path, help="Directory for CSV/JSON and optional videos.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many selected frames.")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame to compare.")
    parser.add_argument("--stride", type=int, default=1, help="Compare every Nth frame.")
    parser.add_argument(
        "--max-width", type=int, default=0,
        help="Downscale both videos to this width for the metric and maps; 0 keeps full resolution.",
    )
    parser.add_argument("--window-size", type=int, default=11, help="Odd local-variance window size.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.5, help="Gaussian sigma for local variance.")
    parser.add_argument(
        "--threshold", type=float, default=0.30,
        help="Risk threshold for area fractions. DeSRA C<0.7 corresponds to risk>0.3.",
    )
    parser.add_argument(
        "--busy-floor", type=float, default=0.25,
        help="Minimum weighting retained in busy reference-texture regions.",
    )
    parser.add_argument(
        "--map-metric", choices=["raw", "weighted"], default="weighted",
        help="Risk map used for optional videos.",
    )
    parser.add_argument("--heatmap-video", action="store_true", help="Write artifact_heatmap.mp4.")
    parser.add_argument("--overlay-video", action="store_true", help="Write artifact_overlay.mp4.")
    parser.add_argument("--progress", type=int, default=25, help="Print progress every N frames; 0 disables.")
    return parser


def run_compare(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if args.start_frame < 0:
        parser.error("--start-frame must be >= 0")
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if args.max_width and args.max_width < 2:
        parser.error("--max-width must be 0 or >= 2")
    if not (0.0 < args.threshold < 1.0):
        parser.error("--threshold must be in (0, 1)")
    if not (0.0 <= args.busy_floor <= 1.0):
        parser.error("--busy-floor must be in [0, 1]")
    if (args.heatmap_video or args.overlay_video) and args.output_dir is None:
        parser.error("--heatmap-video/--overlay-video require --output-dir")

    summary = compare(args)
    print_summary(summary)


_SUBCOMMANDS = ("compare",)


def run_artifacts_command(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        _print(f"usage: kinovsr artifacts {{{'|'.join(_SUBCOMMANDS)}}} ...")
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name == "compare":
        return run_compare(rest) or 0
    _print(f"unknown artifacts subcommand {name!r} "
           f"(available: {', '.join(_SUBCOMMANDS)})")
    return 2

