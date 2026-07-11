#!/usr/bin/env python3
"""Run config-driven denoiser sweeps; score faces and perceptual metrics.

This wraps `scripts/vsr_harness.py` for repeatable local experiments. Checked-in
logic is generic; machine-specific clips, output roots, and artifact paths live
in `scripts/dev/vsr_eval_sweep/vsr_eval.local.toml`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kinovsr.eval.config import (
    as_str_list,
    config_get,
    config_section,
    default_shared_temp,
    listify,
    load_config,
    resolve_path,
)

# scripts/dev/vsr_eval_sweep/ -> repo root is three levels up.
REPO = Path(__file__).resolve().parents[3]

DEFAULT_BASE_FLAGS = [
    "--reader", "ffmpeg",
    "--spatial-mode", "none",
    "--start", "0",
    "--end", "5s",
]

DEFAULT_VARIANTS = [
    {"name": "none", "flags": []},
    {
        "name": "bsvd_f05_l65",
        "flags": [
            "--denoise", "bsvd",
            "--noise-map", "auto",
            "--noise-map-gain", "2.0",
            "--noise-map-motion-cap", "loose",
            "--noise-map-masking", "1",
            "--noise-map-floor", "0.05",
            "--noise-map-pulse",
            "--denoise-luma-strength", "0.65",
            "--denoise-chroma-strength", "1.0",
        ],
    },
    {
        "name": "bsvd_f07_l50",
        "flags": [
            "--denoise", "bsvd",
            "--noise-map", "auto",
            "--noise-map-gain", "2.0",
            "--noise-map-motion-cap", "loose",
            "--noise-map-masking", "1",
            "--noise-map-floor", "0.07",
            "--noise-map-pulse",
            "--denoise-luma-strength", "0.50",
            "--denoise-chroma-strength", "1.0",
        ],
    },
    {
        "name": "fastdvd_f05_l45",
        "flags": [
            "--denoise", "fastdvd",
            "--noise-map", "auto",
            "--noise-map-gain", "1.5",
            "--noise-map-motion-cap", "loose",
            "--noise-map-masking", "1",
            "--noise-map-floor", "0.05",
            "--noise-map-pulse",
            "--denoise-luma-strength", "0.45",
            "--denoise-chroma-strength", "1.0",
        ],
    },
]


def resolve_executable(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return path


def _parse_percent(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_run_log(text: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    posts = re.findall(r"Post: (.+)", text)
    if posts:
        row["post"] = posts[-1].strip()
    if (gate := _parse_percent(r"mc gate openness: ([0-9.]+)%", text)) is not None:
        row["mc_gate_openness_pct"] = gate
    if (static := _parse_percent(r"static-verified ([0-9.]+)%", text)) is not None:
        row["deflicker_static_verified_pct"] = static
    m = re.search(r"effective conditioning: min ([0-9.]+)\s+median ([0-9.]+)\s+max ([0-9.]+)", text)
    if m:
        row["effective_conditioning_min"] = float(m.group(1))
        row["effective_conditioning_median"] = float(m.group(2))
        row["effective_conditioning_max"] = float(m.group(3))
    return row


def _selected(rows: list[dict[str, Any]], selected: str | None, kind: str) -> list[dict[str, Any]]:
    if not selected:
        return rows
    names = {name.strip() for name in selected.split(",") if name.strip()}
    out = [row for row in rows if str(row.get("label") or row.get("name")) in names]
    missing = names - {str(row.get("label") or row.get("name")) for row in out}
    if missing:
        raise SystemExit(f"unknown --{kind}: {', '.join(sorted(missing))}")
    return out


def _normalise_config(args: argparse.Namespace) -> argparse.Namespace:
    config, config_path = load_config(args.config)
    sweep = config_section(config, "sweep")
    face_eval = config_section(config, "face_eval")
    perceptual = config_section(config, "perceptual")
    config_base = config_path.parent if config_path is not None else None

    merged = argparse.Namespace()
    merged.config = config_path
    merged.config_base = config_base
    merged.python = config_get(sweep, args, "python", sys.executable)
    merged.python = resolve_executable(merged.python, None if args.python is not None else config_base)
    merged.output_root = config_get(sweep, args, "output_root", None)
    if merged.output_root is None:
        merged.output_root = default_shared_temp() / f"vsr_denoise_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        merged.output_root = resolve_path(merged.output_root, None if args.output_root is not None else config_base)

    merged.base_flags = as_str_list(sweep.get("base_flags"), DEFAULT_BASE_FLAGS)
    merged.baseline = str(config_get(sweep, args, "baseline", "none"))
    merged.skip_existing = bool(config_get(sweep, args, "skip_existing", False))
    merged.dry_run = bool(args.dry_run)
    merged.evaluate_faces = bool(face_eval.get("enabled", sweep.get("evaluate_faces", True)))
    merged.face_max_frames = int(face_eval.get("max_frames", 150))
    merged.face_threshold = float(face_eval.get("threshold", 0.25))
    merged.face_min_side = int(face_eval.get("min_side", 20))
    merged.face_min_track_frames = int(face_eval.get("min_track_frames", 1))
    merged.face_model = face_eval.get("model")
    if merged.face_model is not None:
        merged.face_model = resolve_path(merged.face_model, config_base)
    merged.evaluate_perceptual = bool(perceptual.get("enabled", True))
    merged.perceptual_max_frames = int(perceptual.get("max_frames", 400))
    merged.perceptual_musiq_every = int(perceptual.get("musiq_every", 10))
    merged.perceptual_metrics = str(
        perceptual.get("metrics", "musiq,dover,niqe,flicker,vmaf"))

    clip_rows = listify(sweep.get("clips"))
    if not clip_rows:
        raise SystemExit("no sweep clips configured; add [[sweep.clips]] entries to local config")
    if not all(isinstance(row, dict) for row in clip_rows):
        raise SystemExit("sweep.clips must be a list of tables/objects")
    variants = listify(sweep.get("variants")) or DEFAULT_VARIANTS
    if not all(isinstance(row, dict) for row in variants):
        raise SystemExit("sweep.variants must be a list of tables/objects")
    merged.clips = _selected(clip_rows, args.clips, "clips")
    merged.variants = _selected(variants, args.variants, "variants")
    for clip in merged.clips:
        if "video" not in clip:
            raise SystemExit(f"sweep clip is missing video: {clip}")
        clip["video"] = str(resolve_path(clip["video"], config_base))
        clip["label"] = str(clip.get("label") or Path(clip["video"]).stem)
    for variant in merged.variants:
        variant["name"] = str(variant.get("name") or "variant")
        variant["flags"] = as_str_list(variant.get("flags"), [])
    return merged


def _latest_post(out_dir: Path) -> Path | None:
    posts = sorted(out_dir.glob("*_post.mp4"), key=lambda p: p.stat().st_mtime)
    return posts[-1] if posts else None


def run_harness(args: argparse.Namespace, clip: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    label = str(clip["label"])
    name = str(variant["name"])
    out_dir = args.output_root / label / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / label / f"{name}.log"

    if args.skip_existing and (post := _latest_post(out_dir)) is not None:
        return {
            "clip": label,
            "variant": name,
            "seconds": 0.0,
            "returncode": 0,
            "log": str(log_path) if log_path.exists() else None,
            "post": str(post),
            "skipped_existing": True,
        }

    cmd = [
        str(args.python),
        "scripts/vsr_harness.py",
        *args.base_flags,
        "--video", str(clip["video"]),
        "--output-dir", str(out_dir),
        "--output-prefix", f"{label}_{name}",
        *variant["flags"],
    ]
    row: dict[str, Any] = {
        "clip": label,
        "variant": name,
        "cmd": cmd,
        "log": str(log_path),
    }
    print(f"[run] {label}:{name}", flush=True)
    if args.dry_run:
        print(" ".join(cmd), flush=True)
        row.update({"returncode": 0, "seconds": 0.0, "dry_run": True})
        return row

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env={**os.environ, "MLX_CACHE_LIMIT": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    row["seconds"] = time.time() - t0
    row["returncode"] = proc.returncode
    log_path.write_text(proc.stdout, encoding="utf-8")
    row.update(parse_run_log(proc.stdout))
    if proc.returncode != 0:
        row["error"] = f"vsr_harness returned {proc.returncode}"
    print(f"[done] {label}:{name} rc={proc.returncode} {row['seconds']:.1f}s", flush=True)
    return row


def evaluate_clip_faces(args: argparse.Namespace, clip: dict[str, Any], manifest: Path) -> list[dict[str, Any]]:
    out_dir = args.output_root / str(clip["label"]) / "face_eval"
    cmd = [
        str(args.python),
        "-m", "kinovsr.cli.main", "metrics", "faces",
        "--variants-json", str(manifest),
        "--baseline", args.baseline,
        "--out-dir", str(out_dir),
        "--max-frames", str(args.face_max_frames),
        "--threshold", str(args.face_threshold),
        "--min-side", str(args.face_min_side),
        "--min-track-frames", str(args.face_min_track_frames),
    ]
    if args.face_model is not None:
        cmd.extend(["--model", str(args.face_model)])
    print(f"[face-eval] {clip['label']}", flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)
    rows = json.loads((out_dir / "face_yunet_metrics.json").read_text(encoding="utf-8"))
    for row in rows:
        row["clip"] = clip["label"]
    return rows


def evaluate_clip_perceptual(args: argparse.Namespace, clip: dict[str, Any], manifest: Path) -> list[dict[str, Any]]:
    out_dir = args.output_root / str(clip["label"]) / "perceptual_eval"
    cmd = [
        str(args.python),
        "-m", "kinovsr.cli.main", "metrics", "perceptual",
        "--variants-json", str(manifest),
        "--source", str(clip["video"]),
        "--out-dir", str(out_dir),
        "--max-frames", str(args.perceptual_max_frames),
        "--musiq-every", str(args.perceptual_musiq_every),
        "--metrics", args.perceptual_metrics,
    ]
    print(f"[perceptual] {clip['label']}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO, check=False)
    if proc.returncode != 0:
        print(f"[perceptual] {clip['label']} failed rc={proc.returncode}; continuing", flush=True)
        return []
    rows = json.loads((out_dir / "perceptual_metrics.json").read_text(encoding="utf-8"))
    for row in rows:
        row["clip"] = clip["label"]
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "config": str(args.config) if args.config is not None else None,
        "output_root": str(args.output_root),
        "base_flags": args.base_flags,
        "runs": [],
        "face_metrics": [],
        "perceptual_metrics": [],
    }
    for clip in args.clips:
        clip_dir = args.output_root / str(clip["label"])
        clip_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, str] = {}
        for variant in args.variants:
            run = run_harness(args, clip, variant)
            summary["runs"].append(run)
            if post := run.get("post"):
                manifest[str(variant["name"])] = str(post)
            (args.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        manifest_path = clip_dir / "variants.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if args.evaluate_faces and args.baseline in manifest and not args.dry_run:
            summary["face_metrics"].extend(evaluate_clip_faces(args, clip, manifest_path))
            (args.output_root / "face_metrics.json").write_text(json.dumps(summary["face_metrics"], indent=2) + "\n", encoding="utf-8")
            write_csv(summary["face_metrics"], args.output_root / "face_metrics.csv")
        if args.evaluate_perceptual and manifest and not args.dry_run:
            summary["perceptual_metrics"].extend(evaluate_clip_perceptual(args, clip, manifest_path))
            (args.output_root / "perceptual_metrics.json").write_text(json.dumps(summary["perceptual_metrics"], indent=2) + "\n", encoding="utf-8")
            write_csv(summary["perceptual_metrics"], args.output_root / "perceptual_metrics.csv")
    (args.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(summary["runs"], args.output_root / "run_summary.csv")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="TOML/JSON config; without it, discovery checks the working directory then scripts/dev/vsr_eval_sweep/ for vsr_eval.local.{toml,json}")
    parser.add_argument("--clips", help="comma-separated subset of sweep.clips labels")
    parser.add_argument("--variants", help="comma-separated subset of sweep.variants names")
    parser.add_argument("--output-root", help="output directory for run logs/videos/summaries")
    parser.add_argument("--python", help="Python executable for subprocesses; defaults to current interpreter")
    parser.add_argument("--baseline", help="baseline variant name for face metrics")
    parser.add_argument("--skip-existing", action="store_true", default=None, help="reuse existing *_post.mp4 files in each variant output dir")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return parser.parse_args()


def main() -> int:
    summary = run_sweep(_normalise_config(parse_args()))
    print(f"[done] {Path(summary['output_root']) / 'run_summary.json'}", flush=True)
    if summary.get("face_metrics"):
        print(f"[done] {Path(summary['output_root']) / 'face_metrics.csv'}", flush=True)
    if summary.get("perceptual_metrics"):
        print(f"[done] {Path(summary['output_root']) / 'perceptual_metrics.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
