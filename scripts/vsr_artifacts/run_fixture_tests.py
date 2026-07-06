#!/usr/bin/env python3
"""Run manual-vs-auto VSR artifact fixture probes from a local TOML/JSON config.

The checked-in logic is generic. Machine-specific source paths, generated
fixture filenames, and output roots belong in
`scripts/vsr_artifacts/vsr_artifacts.local.toml`.

Examples:
    scripts/vsr_artifacts/run_fixture_tests.py \\
        --config scripts/vsr_artifacts/vsr_artifacts.local.toml

    scripts/vsr_artifacts/run_fixture_tests.py \\
        --cases scanlines,mixed --output-root "$SHARED_TEMP_DIR/vsr_tests"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import av
import numpy as np
from config import (
    config_get,
    config_section,
    default_shared_temp,
    listify,
    load_config,
    resolve_path,
)

REPO = Path(__file__).resolve().parents[2]

DEFAULT_BASE_FLAGS = [
    "--spatial-mode", "none",
    "--gop-align",
    "--snap-start",
    "--start", "0",
    "--end", "2s",
]

DEFAULT_MANUAL_FLAGS = [
    "--denoise", "bsvd",
    "--denoise-strength", "0.05",
    "--deblock", "stdf",
    "--deblock-strength", "0.25",
]

DEFAULT_AUTO_FLAGS = [
    "--denoise", "bsvd",
    "--noise-map", "auto",
    "--noise-map-pulse",
    "--noise-map-motion-cap", "strict",
    "--noise-map-masking", "1",
    "--noise-map-debug",
    "--deblock", "stdf",
    "--deblock-map", "auto",
]


def _as_str_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default[:]
    if isinstance(value, str):
        return value.split()
    return [str(v) for v in value]


def parse_noise_sigma(text: str) -> dict[str, float] | None:
    m = re.search(
        r"\[noise-map\] estimated sigma: min ([0-9.]+)\s+median ([0-9.]+)\s+p95 ([0-9.]+)\s+max ([0-9.]+)",
        text,
    )
    if not m:
        return None
    return {k: float(v) for k, v in zip(("min", "median", "p95", "max"), m.groups(), strict=True)}


def parse_effective(text: str) -> dict[str, float] | None:
    m = re.search(
        r"\[noise-map\] effective conditioning: min ([0-9.]+)\s+median ([0-9.]+)\s+max ([0-9.]+)",
        text,
    )
    if not m:
        return None
    return {k: float(v) for k, v in zip(("min", "median", "max"), m.groups(), strict=True)}


def parse_pulse(text: str) -> dict[str, float | int] | None:
    m = re.search(
        r"\[noise-map\] pulse gain over ([0-9]+) frames: min ([0-9.]+)\s+median ([0-9.]+)\s+max ([0-9.]+)\s+\(([0-9]+) frames > 1.2\)",
        text,
    )
    if not m:
        return None
    return {
        "frames": int(m.group(1)),
        "min": float(m.group(2)),
        "median": float(m.group(3)),
        "max": float(m.group(4)),
        "frames_gt_1p2": int(m.group(5)),
    }


def parse_blockiness(text: str) -> dict[str, float] | None:
    m = re.search(
        r"\[deblock-map\] blockiness mask: median ([0-9.]+)\s+p95 ([0-9.]+)\s+max ([0-9.]+)\s+\(([0-9.]+)% of frame > 0.5\)",
        text,
    )
    if not m:
        return None
    return {
        "median": float(m.group(1)),
        "p95": float(m.group(2)),
        "max": float(m.group(3)),
        "percent_gt_0p5": float(m.group(4)),
    }


def summarize_probe(text: str) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        if line.startswith("[probe] window @ frame "):
            if current:
                windows.append(current)
            current = {"frame": int(re.search(r"frame ([0-9]+)", line).group(1))}
        elif current is not None and "flicker: density" in line:
            m = re.search(r"density ([0-9.]+)/([0-9.]+)\s+amplitude ([0-9.]+)/([0-9.]+)", line)
            if m:
                current["flicker_density_med"] = float(m.group(1))
                current["flicker_density_p90"] = float(m.group(2))
                current["flicker_amp_med"] = float(m.group(3))
                current["flicker_amp_p90"] = float(m.group(4))
        elif current is not None and "structure:" in line:
            m = re.search(
                r"lag2/lag1 ([0-9.]+)\s+edge/flat ([0-9.]+)\s+luma-corr ([+-]?[0-9.]+)\s+static-frac ([0-9.]+)\s+static-spatial-hf ([0-9.]+)",
                line,
            )
            if m:
                current["lag2_over_lag1"] = float(m.group(1))
                current["edge_over_flat"] = float(m.group(2))
                current["luma_corr"] = float(m.group(3))
                current["static_fraction"] = float(m.group(4))
                current["static_spatial_hf"] = float(m.group(5))
            rm = re.search(r"row-period ([0-9.]+)@([0-9.]+)px", line)
            if rm:
                current["row_periodicity"] = float(rm.group(1))
                current["row_period_px"] = float(rm.group(2))
        elif current is not None and "verdict:" in line:
            m = re.search(r"verdict: (.+?)\s+risk=([a-z]+)", line)
            if m:
                current["labels"] = [part.strip() for part in m.group(1).split(",") if part.strip()]
                current["risk"] = m.group(2)
        elif current is not None and "warning:" in line:
            current.setdefault("warnings", []).append(line.split("warning:", 1)[1].strip())
        elif current is not None and "try:" in line:
            current.setdefault("suggestions", []).append(line.split("try:", 1)[1].strip())
        elif current is not None and "frame trace:" in line:
            m = re.search(r"med ([0-9.]+)\s+max ([0-9.]+)", line)
            if m:
                current["trace_med"] = float(m.group(1))
                current["trace_max"] = float(m.group(2))
    if current:
        windows.append(current)
    return {"windows": windows}


def run(name: str, cmd: list[str], log_path: Path, python: Path) -> dict[str, Any]:
    print(f"[run] {name}", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [str(python), *cmd],
        cwd=REPO,
        env={**os.environ, "MLX_CACHE_LIMIT": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    dt = time.time() - t0
    log_path.write_text(proc.stdout, encoding="utf-8")
    posts = re.findall(r"Post: (.+)", proc.stdout)
    return {
        "name": name,
        "cmd": [str(python), *cmd],
        "log": str(log_path),
        "returncode": proc.returncode,
        "seconds": dt,
        "post": posts[-1].strip() if posts else None,
        "noise_sigma": parse_noise_sigma(proc.stdout),
        "effective_conditioning": parse_effective(proc.stdout),
        "pulse": parse_pulse(proc.stdout),
        "blockiness": parse_blockiness(proc.stdout),
    }


def parse_start_frame(text: str) -> int:
    m = re.search(r"\[setup\] range: start ([0-9]+)", text)
    return int(m.group(1)) if m else 0


def _source_seed_from_fixture(path: Path) -> tuple[str, str | None] | None:
    parts = path.stem.split("__")
    if len(parts) < 3:
        return None
    seed_match = re.search(r"(?:^|_)s([0-9]+)(?:_|$)", parts[2])
    return parts[0], seed_match.group(1) if seed_match else None


def _infer_clean_reference(video: Path, fixture_dir: Path | None) -> Path | None:
    parsed = _source_seed_from_fixture(video)
    base = fixture_dir or video.parent
    if parsed is None or base is None:
        return None
    source, seed = parsed
    patterns = []
    if seed:
        patterns.append(f"{source}__reencode_only__s{seed}_*.mp4")
    patterns.append(f"{source}__reencode_only__*.mp4")
    for pattern in patterns:
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def _read_luma_frames(path: Path, skip: int = 0, max_frames: int = 60) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx < skip:
                continue
            rgb = frame.to_ndarray(format="rgb24").astype(np.float32) * (1.0 / 255.0)
            frames.append(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
            if len(frames) >= max_frames:
                break
    return frames


def compare_to_clean(clean: Path, candidate: Path, clean_skip: int, max_frames: int) -> dict[str, float | int | str]:
    ref = _read_luma_frames(clean, skip=clean_skip, max_frames=max_frames)
    cand = _read_luma_frames(candidate, max_frames=max_frames)
    n = min(len(ref), len(cand))
    if n == 0:
        return {"frames": 0, "error": "no overlapping frames"}
    maes: list[float] = []
    p95s: list[float] = []
    mses: list[float] = []
    for a, b in zip(ref[:n], cand[:n], strict=True):
        if a.shape != b.shape:
            return {
                "frames": n,
                "error": f"shape mismatch clean {a.shape} vs candidate {b.shape}",
            }
        d = np.abs(a - b)
        maes.append(float(np.mean(d)))
        p95s.append(float(np.quantile(d, 0.95)))
        mses.append(float(np.mean((a - b) ** 2)))
    mse = float(sum(mses) / len(mses))
    return {
        "frames": n,
        "clean_reference": str(clean),
        "clean_skip_frames": int(clean_skip),
        "mean_abs_luma": float(sum(maes) / len(maes)),
        "p95_abs_luma": float(sum(p95s) / len(p95s)),
        "worst_frame_mean_abs_luma": float(max(maes)),
        "psnr_luma": 99.0 if mse <= 0 else float(-10.0 * math.log10(mse)),
    }


def _case_video(case: dict[str, Any], fixture_dir: Path | None, config_base: Path | None) -> Path:
    value = case.get("video")
    if value is None:
        raise SystemExit(f"test case is missing video: {case!r}")
    base = fixture_dir if fixture_dir is not None else config_base
    return resolve_path(value, base)


def _normalise_config(args: argparse.Namespace) -> argparse.Namespace:
    config, config_path = load_config(args.config)
    fixtures = config_section(config, "fixtures")
    tests = config_section(config, "tests")
    config_base = config_path.parent if config_path is not None else None

    merged = argparse.Namespace()
    merged.config = config_path
    merged.config_base = config_base
    merged.python = resolve_path(
        config_get(tests, args, "python", sys.executable),
        None if args.python is not None else config_base,
    )
    merged.fixture_dir = config_get(tests, args, "fixture_dir", fixtures.get("output_dir"))
    fixture_base = None if args.fixture_dir is not None else config_base
    merged.fixture_dir = resolve_path(merged.fixture_dir, fixture_base) if merged.fixture_dir else None
    merged.output_root = config_get(tests, args, "output_root", None)
    if merged.output_root is None:
        merged.output_root = default_shared_temp() / f"vsr_fixture_tests_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        output_base = None if args.output_root is not None else config_base
        merged.output_root = resolve_path(merged.output_root, output_base)
    merged.base_flags = _as_str_list(tests.get("base_flags"), DEFAULT_BASE_FLAGS)
    merged.manual_flags = _as_str_list(tests.get("manual_flags"), DEFAULT_MANUAL_FLAGS)
    merged.auto_flags = _as_str_list(tests.get("auto_flags"), DEFAULT_AUTO_FLAGS)
    merged.compare_max_frames = int(tests.get("compare_max_frames", 60))
    merged.progress = str(tests.get("compare_progress", 0))
    merged.clean_compare = bool(tests.get("clean_compare", True))
    cases = listify(tests.get("cases", []))
    if args.cases:
        selected = {name.strip() for name in args.cases.split(",") if name.strip()}
        cases = [c for c in cases if isinstance(c, dict) and str(c.get("label")) in selected]
        missing = selected - {str(c.get("label")) for c in cases if isinstance(c, dict)}
        if missing:
            raise SystemExit(f"unknown --cases labels: {', '.join(sorted(missing))}")
    if not cases:
        raise SystemExit("no test cases configured; add [[tests.cases]] entries to local config")
    if not all(isinstance(c, dict) for c in cases):
        raise SystemExit("tests.cases must be a list of tables/objects")
    merged.cases = cases
    return merged


def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "config": str(args.config) if args.config is not None else None,
        "out_root": str(args.output_root),
        "fixture_dir": str(args.fixture_dir) if args.fixture_dir is not None else None,
        "cases": [],
    }
    for case_cfg in args.cases:
        label = str(case_cfg.get("label") or Path(str(case_cfg.get("video"))).stem)
        video = _case_video(case_cfg, args.fixture_dir, args.config_base)
        if not video.exists():
            raise SystemExit(f"missing fixture for {label}: {video}")
        case_dir = args.output_root / label
        case_dir.mkdir(parents=True, exist_ok=True)
        case: dict[str, Any] = {"label": label, "video": str(video), "runs": {}}

        probe_cmd = [
            "scripts/vsr_harness.py",
            *args.base_flags,
            "--video", str(video),
            "--probe-noise",
        ]
        probe = run(f"{label}:probe", probe_cmd, case_dir / "probe.log", args.python)
        case["probe"] = probe
        case["probe_summary"] = summarize_probe(Path(probe["log"]).read_text(encoding="utf-8"))

        for mode, flags in (("manual", args.manual_flags), ("auto", args.auto_flags)):
            run_dir = case_dir / mode
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "scripts/vsr_harness.py",
                *args.base_flags,
                "--video", str(video),
                "--output-dir", str(run_dir),
                "--output-prefix", f"{label}_{mode}",
                *flags,
            ]
            case["runs"][mode] = run(f"{label}:{mode}", cmd, case_dir / f"{mode}.log", args.python)

        if args.clean_compare:
            clean_ref = _infer_clean_reference(video, args.fixture_dir)
            if clean_ref is not None and clean_ref.exists():
                case["clean_reference"] = str(clean_ref)
                for mode in ("manual", "auto"):
                    run_info = case["runs"][mode]
                    post = run_info.get("post")
                    if post and Path(post).exists():
                        log_text = Path(run_info["log"]).read_text(encoding="utf-8")
                        run_info["quality_vs_clean"] = compare_to_clean(
                            clean_ref,
                            Path(post),
                            parse_start_frame(log_text),
                            args.compare_max_frames,
                        )

        man = case["runs"]["manual"]["post"]
        aut = case["runs"]["auto"]["post"]
        if man and aut and Path(man).exists() and Path(aut).exists():
            cmp_dir = case_dir / "compare_manual_auto"
            cmp_cmd = [
                "scripts/compare_vsr_artifacts.py",
                "--reference", man,
                "--candidate", aut,
                "--output-dir", str(cmp_dir),
                "--max-frames", str(args.compare_max_frames),
                "--progress", args.progress,
            ]
            cmp_run = run(f"{label}:compare_manual_auto", cmp_cmd, case_dir / "compare_manual_auto.log", args.python)
            summary_path = cmp_dir / "artifact_summary.json"
            if summary_path.exists():
                cmp_run["artifact_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            case["compare_manual_auto"] = cmp_run

        summary["cases"].append(case)
        (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="TOML/JSON config; defaults to scripts/vsr_artifacts/vsr_artifacts.local.{toml,json} if present")
    parser.add_argument("--cases", help="comma-separated subset of tests.cases labels")
    parser.add_argument("--output-root", help="root directory for logs/videos/summaries")
    parser.add_argument("--python", help="Python executable for subprocesses; defaults to the current interpreter")
    parser.add_argument("--fixture-dir", help="directory containing generated fixture videos")
    return parser.parse_args()


def main() -> int:
    summary = run_cases(_normalise_config(parse_args()))
    print(f"[done] {Path(summary['out_root']) / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
