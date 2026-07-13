#!/usr/bin/env python3
"""Analyze VSR artifact fixtures without running a denoiser/deblocker model.

This is the fast first pass for noise-map and deblock-map logic. It decodes
configured fixture clips, runs the diagnostic probe, estimates sigma maps under
one or more motion-cap modes, estimates the blockiness map, and writes JSON/CSV
summaries. Machine-specific paths belong in
`scripts/dev/vsr_artifacts/vsr_artifacts.local.toml`.

Examples:
    scripts/dev/vsr_artifacts/analyze_maps.py

    scripts/dev/vsr_artifacts/analyze_maps.py \\
        --config scripts/dev/vsr_artifacts/vsr_artifacts.local.toml \\
        --videos walking__mild_high_iso__s20260705_crf34.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import av
import mlx.core as mx
import numpy as np
from config import (
    config_get,
    config_section,
    default_shared_temp,
    listify,
    load_config,
    resolve_path,
)

_log = logging.getLogger("kinovsr.dev.vsr_artifacts.analyze_maps")

# scripts/dev/vsr_artifacts/ -> repo root is three levels up.
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kinovsr.analysis.noise import (  # noqa: E402
    analyze_noise,
    classify_noise_analysis,
    estimate_blockiness_map,
    estimate_sigma_map,
)

DEFAULT_WINDOWS = [0.1, 0.5, 0.9]
DEFAULT_MOTION_CAPS = ["strict", "loose", "off"]


def _as_float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default[:]
    if isinstance(value, str):
        return [float(v) for v in value.split(",") if v.strip()]
    return [float(v) for v in value]


def _as_str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default[:]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mode_from_name(path: Path) -> str:
    parts = path.stem.split("__")
    return parts[1] if len(parts) >= 2 else "unknown"


def _source_from_name(path: Path) -> str:
    return path.stem.split("__")[0]


def _read_frames(path: Path, seconds: float) -> tuple[list[Any], float]:
    frames: list[Any] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 30.0)
        max_frames = max(3, int(round(seconds * fps)))
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24").astype(np.float32) * (1.0 / 255.0)
            frames.append(mx.array(rgb))
            if len(frames) >= max_frames:
                break
    if len(frames) < 3:
        raise RuntimeError(f"not enough frames decoded from {path}")
    return frames, fps


def _sample_windows(frames: list[Any], window_frames: int, fractions: list[float]) -> list[dict[str, Any]]:
    n = max(3, int(window_frames))
    if len(frames) <= n:
        return [{"start": 0, "frames": frames}]
    span = len(frames) - n
    starts = sorted({round(max(0.0, min(1.0, f)) * span) for f in fractions})
    return [{"start": s, "frames": frames[s:s + n]} for s in starts]


def _stats_map(m: Any) -> dict[str, float]:
    flat = mx.sort(m.reshape(-1))
    n = int(flat.shape[0])
    return {
        "min": float(flat[0]),
        "median": float(flat[n // 2]),
        "p90": float(flat[int(0.90 * (n - 1))]),
        "p95": float(flat[int(0.95 * (n - 1))]),
        "max": float(flat[-1]),
        "gt_0p02": float(mx.mean((m > 0.02).astype(mx.float32))),
        "gt_0p05": float(mx.mean((m > 0.05).astype(mx.float32))),
        "gt_0p5": float(mx.mean((m > 0.5).astype(mx.float32))),
    }


def _median_map_stats(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    return {key: _median([r[key] for r in rows]) for key in rows[0]}


def _summarize_probe(windows: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    risks = {"low": 0, "medium": 1, "high": 2}
    risk = "low"
    warnings: list[str] = []
    suggestions: list[str] = []
    for win in windows:
        stats = analyze_noise(win["frames"])
        diag = classify_noise_analysis(stats)
        for label in diag["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1
        if risks[diag["risk"]] > risks[risk]:
            risk = diag["risk"]
        for msg in diag["warnings"]:
            if msg not in warnings:
                warnings.append(msg)
        for msg in diag["suggestions"]:
            if msg not in suggestions:
                suggestions.append(msg)
        rows.append({"start": win["start"], "stats": stats, "diagnosis": diag})

    out: dict[str, Any] = {
        "windows": rows,
        "labels": sorted(label_counts, key=lambda k: (-label_counts[k], k)),
        "label_counts": label_counts,
        "risk": risk,
        "warnings": warnings,
        "suggestions": suggestions,
    }
    for key in (
        "lag2_over_lag1",
        "edge_over_flat",
        "luma_corr",
        "static_fraction",
        "static_spatial_hf",
        "row_coherence",
        "row_periodicity",
        "row_period_px",
        "sigma_R",
        "sigma_G",
        "sigma_B",
    ):
        out[key] = _median([float(r["stats"][key]) for r in rows if key in r["stats"]])
    for key in ("flicker_density", "flicker_amplitude", "tail5", "tail1", "max", "rms"):
        out[key + "_med"] = _median([float(r["stats"][key][0]) for r in rows if key in r["stats"]])
        out[key + "_p90"] = _median([float(r["stats"][key][1]) for r in rows if key in r["stats"]])
    traces = [float(v) for r in rows for v in r["stats"].get("frame_trace", [])]
    out["trace_med"] = _median(traces)
    out["trace_max"] = max(traces, default=0.0)
    return out


def _map_assessment(probe: dict[str, Any], sigma: dict[str, Any], blockiness: dict[str, Any] | None) -> dict[str, Any]:
    flags: list[str] = []
    labels = set(probe.get("labels", []))
    strict = sigma.get("strict")
    if strict is not None and "motion/noise ambiguous" in labels and strict["p95"] > 0.10:
        flags.append("strict sigma map may over-condition dense textured motion")
    if strict is not None and "source-wide motion contamination" in labels and strict["p95"] > 0.12:
        flags.append("strict sigma map may still be motion-contaminated")
    if strict is not None and "sparse edge flicker" in labels and strict["p95"] < 0.02:
        flags.append("sigma map likely under-conditions sparse edge flicker")
    if strict is not None and "row-coherent scanline flicker" in labels and strict["p95"] > 0.06:
        flags.append("row-periodic scanlines may need reduced noise-map gain")
    if "static/structured grain" in labels:
        flags.append("temporal sigma map is blind to some static structure; floor/manual strength matters")
    if blockiness is not None and blockiness["p95"] > 0.90:
        flags.append("blockiness map saturates; baseline not available yet")
    return {"flags": flags, "risk": "high" if flags else probe.get("risk", "low")}


def _apply_reencode_baselines(rows: list[dict[str, Any]]) -> None:
    baselines = {
        row["source"]: row
        for row in rows
        if row["mode"] == "reencode_only"
    }
    for row in rows:
        base = baselines.get(row["source"])
        if base is None:
            continue
        sig = row["sigma"].get("strict") or {}
        bsig = base["sigma"].get("strict") or {}
        block = row["blockiness"] or {}
        bblock = base["blockiness"] or {}
        baseline = {
            "source": base["path"],
            "sigma_strict_p95": bsig.get("p95"),
            "sigma_strict_p95_delta": (
                sig.get("p95") - bsig.get("p95")
                if sig.get("p95") is not None and bsig.get("p95") is not None else None
            ),
            "blockiness_p95": bblock.get("p95"),
            "blockiness_p95_delta": (
                block.get("p95") - bblock.get("p95")
                if block.get("p95") is not None and bblock.get("p95") is not None else None
            ),
        }
        row["baseline"] = baseline
        if row["mode"] == "reencode_only":
            flags = [
                flag for flag in row["assessment"]["flags"]
                if flag != "blockiness map saturates; baseline not available yet"
            ]
            if bblock.get("p95") is not None and bblock["p95"] > 0.90:
                flags.append("reencode baseline blockiness saturates")
            row["assessment"]["flags"] = flags
            if flags:
                row["assessment"]["risk"] = "high"
            continue

        flags = [
            flag for flag in row["assessment"]["flags"]
            if flag != "blockiness map saturates; baseline not available yet"
        ]
        block_p95 = block.get("p95")
        block_delta = baseline["blockiness_p95_delta"]
        base_block_p95 = baseline["blockiness_p95"]
        if block_p95 is not None and block_p95 > 0.90:
            if base_block_p95 is not None and base_block_p95 > 0.90 and block_delta is not None and block_delta < 0.08:
                flags.append("blockiness saturation is already present in reencode baseline")
            elif block_delta is not None and block_delta > 0.15:
                flags.append("blockiness rises well above reencode baseline")
            else:
                flags.append("blockiness map saturates relative to uncertain baseline")
        row["assessment"]["flags"] = flags
        if flags:
            row["assessment"]["risk"] = "high"


def _summarize_clip(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    frames, fps = _read_frames(path, args.seconds)
    windows = _sample_windows(frames, args.window_frames, args.windows)
    probe = _summarize_probe(windows)
    sigma: dict[str, Any] = {}
    for cap in args.motion_caps:
        maps = [
            estimate_sigma_map(
                win["frames"],
                motion_cap=cap,
                masking=args.masking,
                pulse_robust=args.pulse_robust,
            )
            for win in windows
        ]
        sigma[cap] = _median_map_stats([_stats_map(m) for m in maps if m is not None])
    block_maps = [estimate_blockiness_map(win["frames"]) for win in windows]
    blockiness = _median_map_stats([_stats_map(m) for m in block_maps if m is not None])
    return {
        "path": str(path),
        "source": _source_from_name(path),
        "mode": _mode_from_name(path),
        "fps": fps,
        "frames_decoded": len(frames),
        "window_frames": args.window_frames,
        "probe": probe,
        "sigma": sigma,
        "blockiness": blockiness,
        "assessment": _map_assessment(probe, sigma, blockiness),
    }


def _load_manifest_videos(manifest: Path) -> list[Path]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    outputs = data.get("outputs")
    if not isinstance(outputs, list):
        raise SystemExit(f"manifest has no outputs list: {manifest}")
    return [Path(str(row["path"])) for row in outputs if isinstance(row, dict) and row.get("path")]


def _resolve_videos(values: list[Any], fixture_dir: Path | None, config_base: Path | None) -> list[Path]:
    videos: list[Path] = []
    base = fixture_dir if fixture_dir is not None else config_base
    for value in values:
        raw = str(value)
        if any(ch in raw for ch in "*?[]"):
            pattern = resolve_path(raw, base)
            parent = pattern.parent
            matches = sorted(parent.glob(pattern.name))
            videos.extend(matches)
        else:
            videos.append(resolve_path(raw, base))
    missing = [p for p in videos if not p.exists()]
    if missing:
        raise SystemExit("missing analysis videos:\n" + "\n".join(str(p) for p in missing))
    return videos


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    out: dict[str, Any] = {}
    for name, items in groups.items():
        out[name] = {
            "count": len(items),
            "risk_counts": {
                risk: sum(1 for item in items if item["assessment"]["risk"] == risk)
                for risk in ("low", "medium", "high")
            },
            "sigma_strict_p95_median": _median([
                item["sigma"]["strict"]["p95"]
                for item in items
                if item["sigma"].get("strict") is not None
            ]),
            "blockiness_p95_median": _median([
                item["blockiness"]["p95"]
                for item in items
                if item["blockiness"] is not None
            ]),
            "blockiness_p95_delta_median": _median([
                item.get("baseline", {}).get("blockiness_p95_delta")
                for item in items
                if item.get("baseline", {}).get("blockiness_p95_delta") is not None
                and item["mode"] != "reencode_only"
            ]),
        }
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "source", "mode", "risk", "labels", "flags",
        "baseline_sigma_strict_p95", "delta_sigma_strict_p95",
        "baseline_block_p95", "delta_block_p95",
        "probe_trace_med", "probe_trace_max",
        "probe_flicker_density_med", "probe_flicker_density_p90",
        "probe_flicker_amp_med", "probe_flicker_amp_p90",
        "probe_lag2_over_lag1", "probe_edge_over_flat",
        "probe_static_fraction", "probe_static_spatial_hf",
        "sigma_strict_median", "sigma_strict_p95", "sigma_strict_max",
        "sigma_loose_median", "sigma_loose_p95", "sigma_loose_max",
        "sigma_off_median", "sigma_off_p95", "sigma_off_max",
        "block_median", "block_p95", "block_max", "block_gt_0p5",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            p = row["probe"]
            s = row["sigma"]
            b = row["blockiness"] or {}
            base = row.get("baseline", {})
            writer.writerow({
                "source": row["source"],
                "mode": row["mode"],
                "risk": row["assessment"]["risk"],
                "labels": "; ".join(p["labels"]),
                "flags": "; ".join(row["assessment"]["flags"]),
                "baseline_sigma_strict_p95": base.get("sigma_strict_p95", ""),
                "delta_sigma_strict_p95": base.get("sigma_strict_p95_delta", ""),
                "baseline_block_p95": base.get("blockiness_p95", ""),
                "delta_block_p95": base.get("blockiness_p95_delta", ""),
                "probe_trace_med": p["trace_med"],
                "probe_trace_max": p["trace_max"],
                "probe_flicker_density_med": p["flicker_density_med"],
                "probe_flicker_density_p90": p["flicker_density_p90"],
                "probe_flicker_amp_med": p["flicker_amplitude_med"],
                "probe_flicker_amp_p90": p["flicker_amplitude_p90"],
                "probe_lag2_over_lag1": p["lag2_over_lag1"],
                "probe_edge_over_flat": p["edge_over_flat"],
                "probe_static_fraction": p["static_fraction"],
                "probe_static_spatial_hf": p["static_spatial_hf"],
                "sigma_strict_median": (s.get("strict") or {}).get("median", ""),
                "sigma_strict_p95": (s.get("strict") or {}).get("p95", ""),
                "sigma_strict_max": (s.get("strict") or {}).get("max", ""),
                "sigma_loose_median": (s.get("loose") or {}).get("median", ""),
                "sigma_loose_p95": (s.get("loose") or {}).get("p95", ""),
                "sigma_loose_max": (s.get("loose") or {}).get("max", ""),
                "sigma_off_median": (s.get("off") or {}).get("median", ""),
                "sigma_off_p95": (s.get("off") or {}).get("p95", ""),
                "sigma_off_max": (s.get("off") or {}).get("max", ""),
                "block_median": b.get("median", ""),
                "block_p95": b.get("p95", ""),
                "block_max": b.get("max", ""),
                "block_gt_0p5": b.get("gt_0p5", ""),
            })


def _normalise_config(args: argparse.Namespace) -> argparse.Namespace:
    config, config_path = load_config(args.config)
    fixtures = config_section(config, "fixtures")
    tests = config_section(config, "tests")
    analysis = config_section(config, "analysis")
    config_base = config_path.parent if config_path is not None else None

    merged = argparse.Namespace()
    merged.config = config_path
    merged.config_base = config_base
    merged.fixture_dir = config_get(analysis, args, "fixture_dir", tests.get("fixture_dir", fixtures.get("output_dir")))
    fixture_base = None if args.fixture_dir is not None else config_base
    merged.fixture_dir = resolve_path(merged.fixture_dir, fixture_base) if merged.fixture_dir else None
    merged.manifest = config_get(analysis, args, "manifest", None)
    if merged.manifest is None and merged.fixture_dir is not None:
        merged.manifest = merged.fixture_dir / "manifest.json"
    elif merged.manifest is not None:
        manifest_base = None if args.manifest is not None else config_base
        merged.manifest = resolve_path(merged.manifest, manifest_base)
    merged.output_root = config_get(analysis, args, "output_root", None)
    if merged.output_root is None:
        merged.output_root = default_shared_temp() / f"vsr_artifact_map_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        output_base = None if args.output_root is not None else config_base
        merged.output_root = resolve_path(merged.output_root, output_base)
    merged.seconds = float(config_get(analysis, args, "seconds", 2.0))
    merged.window_frames = int(config_get(analysis, args, "window_frames", 12))
    merged.masking = float(config_get(analysis, args, "masking", 1.0))
    merged.pulse_robust = bool(config_get(analysis, args, "pulse_robust", False))
    merged.windows = _as_float_list(config_get(analysis, args, "windows", DEFAULT_WINDOWS), DEFAULT_WINDOWS)
    merged.motion_caps = _as_str_list(config_get(analysis, args, "motion_caps", DEFAULT_MOTION_CAPS), DEFAULT_MOTION_CAPS)
    bad_caps = sorted(set(merged.motion_caps) - {"strict", "loose", "off"})
    if bad_caps:
        raise SystemExit(f"unknown motion cap(s): {', '.join(bad_caps)}")
    videos_cfg = listify(config_get(analysis, args, "videos", []))
    if videos_cfg:
        merged.videos = _resolve_videos(videos_cfg, merged.fixture_dir, config_base)
    elif merged.manifest is not None and merged.manifest.exists():
        merged.videos = _load_manifest_videos(merged.manifest)
    else:
        raise SystemExit("no analysis videos configured and no fixture manifest found")
    merged.limit = int(config_get(analysis, args, "limit", 0) or 0)
    if merged.limit > 0:
        merged.videos = merged.videos[:merged.limit]
    return merged


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summary = {
        "config": str(args.config) if args.config is not None else None,
        "fixture_dir": str(args.fixture_dir) if args.fixture_dir is not None else None,
        "manifest": str(args.manifest) if args.manifest is not None else None,
        "output_root": str(args.output_root),
        "seconds": args.seconds,
        "window_frames": args.window_frames,
        "windows": args.windows,
        "motion_caps": args.motion_caps,
        "masking": args.masking,
        "pulse_robust": args.pulse_robust,
        "clips": rows,
    }
    for idx, path in enumerate(args.videos, 1):
        _log.info("analysis %02d/%02d %s", idx, len(args.videos), path.name)
        rows.append(_summarize_clip(path, args))
        (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _apply_reencode_baselines(rows)
    summary["by_source"] = _group_summary(rows, "source")
    summary["by_mode"] = _group_summary(rows, "mode")
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(rows, args.output_root / "summary.csv")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="TOML/JSON config; defaults to scripts/dev/vsr_artifacts/vsr_artifacts.local.{toml,json} if present")
    parser.add_argument("--fixture-dir", help="directory containing generated fixtures")
    parser.add_argument("--manifest", help="fixture manifest; defaults to fixture_dir/manifest.json")
    parser.add_argument("--videos", nargs="*", help="specific videos or globs, relative to fixture_dir when set")
    parser.add_argument("--output-root", help="root directory for JSON/CSV summaries")
    parser.add_argument("--seconds", type=float, help="seconds to decode from each clip")
    parser.add_argument("--window-frames", type=int, help="frames per probe/estimator window")
    parser.add_argument("--windows", help="comma-separated fractional window starts, e.g. 0.1,0.5,0.9")
    parser.add_argument("--motion-caps", help="comma-separated subset: strict,loose,off")
    parser.add_argument("--masking", type=float, help="noise-map masking value for estimated maps")
    parser.add_argument("--pulse-robust", action="store_true", default=None,
                        help="winsorize whole-frame pulse spikes in estimated maps, matching --noise-map-pulse base-map behavior")
    parser.add_argument("--limit", type=int, help="debug: analyze only the first N videos")
    return parser.parse_args()


def main() -> int:
    from kinovsr.ui.logging import configure_logging

    configure_logging()
    summary = run_analysis(_normalise_config(parse_args()))
    _log.info("wrote %s", Path(summary["output_root"]) / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
