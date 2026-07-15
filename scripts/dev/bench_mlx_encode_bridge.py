"""Benchmark the legacy RGBAHalf encode bridge against direct MLX RGB input.

Each sample runs in a fresh process. Warmup includes compile and pool setup;
only the following measured frames enter steady-state timing. The two paths
must produce bit-identical active YUV planes before timing is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

MIN_RUNS = 3
MIN_WARMUP = 30
MIN_MEASURED = 120
DEFAULT_RUNS = 4
DEFAULT_WARMUP = 30
DEFAULT_MEASURED = 120
DEFAULT_GEOMETRIES = "1920x1080,3840x2160"


def _parse_geometry(value: str) -> tuple[int, int]:
    try:
        width_raw, height_raw = value.lower().split("x", 1)
        width, height = int(width_raw), int(height_raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"geometry must be WIDTHxHEIGHT, got {value!r}") from exc
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError(
            f"geometry must have positive even dimensions, got {value!r}"
        )
    return width, height


def _attrs(pixel_format: int, width: int, height: int) -> dict[str, Any]:
    return {
        "PixelFormatType": pixel_format,
        "Width": width,
        "Height": height,
        "IOSurfaceProperties": {},
        "MetalCompatibility": True,
    }


def _pull(pool: Any, label: str) -> Any:
    from kinovsr.media import pixel_buffers as pb

    buffer = pb.pool_create_buffer(pool)
    if buffer is None:
        raise RuntimeError(f"{label} pool allocation failed")
    return buffer


def _active_yuv_digest(buffer: Any, width: int) -> str:
    from kinovsr.native.frameworks import Quartz

    digest = hashlib.sha256()
    Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
    try:
        for plane in range(Quartz.CVPixelBufferGetPlaneCount(buffer)):
            rows = Quartz.CVPixelBufferGetHeightOfPlane(buffer, plane)
            bpr = Quartz.CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
            base = Quartz.CVPixelBufferGetBaseAddressOfPlane(buffer, plane)
            raw = base.as_buffer(rows * bpr)
            row_bytes = width * 2
            for row in range(rows):
                digest.update(raw[row * bpr : row * bpr + row_bytes])
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)
    return digest.hexdigest()


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def _command_value(command: list[str]) -> str:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    return proc.stdout.strip() or "unknown"


def _runtime_conditions() -> dict[str, Any]:
    from kinovsr.native.frameworks import Foundation

    process = Foundation.NSProcessInfo.processInfo()
    thermal = {
        0: "nominal",
        1: "fair",
        2: "serious",
        3: "critical",
    }.get(int(process.thermalState()), "unknown")
    return {
        "power_source": _command_value(["pmset", "-g", "batt"]).splitlines()[0],
        "low_power_mode": bool(process.isLowPowerModeEnabled()),
        "thermal_state": thermal,
    }


def _require_conditions(
    actual: dict[str, Any],
    boundary: str,
    *,
    expected: dict[str, Any] | None = None,
) -> None:
    errors = []
    if actual["thermal_state"] != "nominal":
        errors.append(f"thermal_state={actual['thermal_state']!r}")
    if actual["low_power_mode"]:
        errors.append("low_power_mode=true")
    if "AC Power" not in actual["power_source"]:
        errors.append(f"power_source={actual['power_source']!r}")
    if expected is not None and actual != expected:
        errors.append(f"changed from {expected!r}")
    if errors:
        raise RuntimeError(f"benchmark conditions invalid at {boundary}: " + "; ".join(errors))


def _revision_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=False
    )
    status_proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if commit_proc.returncode or status_proc.returncode:
        raise RuntimeError("cannot read product revision from git")
    status = status_proc.stdout
    digest = hashlib.sha256()
    diff_proc = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if diff_proc.returncode:
        raise RuntimeError("cannot hash the product working-tree diff")
    digest.update(diff_proc.stdout)
    digest.update(status)
    for record in status.split(b"\0"):
        if not record.startswith(b"?? "):
            continue
        path = root / record[3:].decode(errors="surrogateescape")
        if path.is_file():
            digest.update(record[3:])
            digest.update(path.read_bytes())
    return {
        "commit": commit_proc.stdout.decode("ascii").strip(),
        "dirty": bool(status),
        "diff_sha256": digest.hexdigest() if status else None,
    }


def _require_revision(expected: dict[str, Any], boundary: str) -> dict[str, Any]:
    actual = _revision_state()
    if actual != expected:
        raise RuntimeError(
            f"product revision changed at {boundary}: start={expected!r}, current={actual!r}"
        )
    return actual


def _worker(
    path: str,
    width: int,
    height: int,
    warmup: int,
    measured: int,
) -> dict[str, Any]:
    conditions_start = _runtime_conditions()
    _require_conditions(conditions_start, f"{path} worker start")
    import mlx.core as mx

    from kinovsr.media import pixel_buffers as pb
    from kinovsr.media import yuv
    from kinovsr.native.frameworks import Quartz, autorelease_pool

    mx.random.seed(17)
    frame = mx.random.uniform(shape=(height, width, 3)).astype(mx.float32)
    mx.eval(frame)
    rgba_pool = pb.make_pool_from_attrs(_attrs(pb.PIX_RGBAHALF, width, height))
    yuv_pool = pb.make_pool_from_attrs(_attrs(yuv.pixel_format(False), width, height))
    if rgba_pool is None or yuv_pool is None:
        raise RuntimeError("benchmark requires working RGBAHalf and YUV pools")
    acquisitions = 0

    def convert() -> Any:
        nonlocal acquisitions
        with autorelease_pool():
            if path == "legacy":
                rgba = _pull(rgba_pool, "RGBAHalf")
                acquisitions += 1
                alpha = mx.ones((height, width, 1), dtype=mx.float16)
                pb.write_fp16_rgba(
                    mx.concatenate([frame[..., :3].astype(mx.float16), alpha], axis=-1), rgba
                )
                rgb = pb.read_rgbahalf_rgb(rgba)
            else:
                rgb = frame[..., :3].astype(mx.float16).astype(mx.float32)
            dst = _pull(yuv_pool, "YUV")
            acquisitions += 1
            yuv.rgb_to_yuv422_10(rgb, dst, Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2, False)
            return dst

    output = None
    for _ in range(warmup):
        output = convert()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    acquisitions = 0
    frame_ms = []
    started = time.perf_counter()
    for _ in range(measured):
        frame_started = time.perf_counter()
        output = convert()
        frame_ms.append((time.perf_counter() - frame_started) * 1000.0)
    steady_ms = (time.perf_counter() - started) * 1000.0
    if output is None:
        raise RuntimeError("benchmark emitted no output")
    conditions_end = _runtime_conditions()
    _require_conditions(
        conditions_end,
        f"{path} worker end",
        expected=conditions_start,
    )
    return {
        "path": path,
        "width": width,
        "height": height,
        "warmup_frames": warmup,
        "measured_frames": measured,
        "steady_total_ms": steady_ms,
        "steady_ms_per_frame": steady_ms / measured,
        "frame_ms": frame_ms,
        "pool_acquisitions": acquisitions,
        "peak_rss_mib": _peak_rss_mib(),
        # Hard requirement: _summarize does float(...) on this, and a None
        # here would surface as a TypeError long after the run.
        "peak_mlx_mib": float(mx.get_peak_memory()) / (1024.0 * 1024.0),
        "active_yuv_sha256": _active_yuv_digest(output, width),
        "conditions_start": conditions_start,
        "conditions_end": conditions_end,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _run_worker(
    path: str,
    geometry: tuple[int, int],
    warmup: int,
    measured: int,
) -> dict[str, Any]:
    width, height = geometry
    with tempfile.TemporaryDirectory(prefix="kinovsr-bridge-bench-") as tmp:
        result_path = Path(tmp) / "result.json"
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker-path",
                path,
                "--worker-geometry",
                f"{width}x{height}",
                "--worker-output",
                str(result_path),
                "--warmup",
                str(warmup),
                "--measured",
                str(measured),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{path} worker failed for {width}x{height}:\n{proc.stderr[-4000:]}")
        if not result_path.is_file():
            raise RuntimeError(
                f"{path} worker returned no result for {width}x{height}:\n"
                f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
            )
        return json.loads(result_path.read_text())


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    frame_samples = [float(value) for run in runs for value in run["frame_ms"]]
    per_run = [float(run["steady_ms_per_frame"]) for run in runs]
    return {
        "median_ms_per_frame": statistics.median(per_run),
        "p95_frame_ms": _percentile(frame_samples, 0.95),
        "per_run_ms_per_frame": per_run,
        "peak_rss_mib": max(float(run["peak_rss_mib"]) for run in runs),
        "peak_mlx_mib": max(float(run["peak_mlx_mib"]) for run in runs),
        "pool_acquisitions_per_frame": (runs[0]["pool_acquisitions"] / runs[0]["measured_frames"]),
        "active_yuv_sha256": runs[0]["active_yuv_sha256"],
        "conditions": [
            {"start": run["conditions_start"], "end": run["conditions_end"]} for run in runs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--measured", type=int, default=DEFAULT_MEASURED)
    parser.add_argument("--geometries", default=DEFAULT_GEOMETRIES)
    parser.add_argument("--output")
    parser.add_argument("--worker-path", choices=("legacy", "direct"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-geometry", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_path:
        if not args.worker_output:
            parser.error("worker mode requires --worker-output")
        geometry = _parse_geometry(args.worker_geometry)
        result = _worker(args.worker_path, *geometry, args.warmup, args.measured)
        Path(args.worker_output).write_text(json.dumps(result, sort_keys=True) + "\n")
        return 0
    if args.runs < MIN_RUNS or args.warmup < MIN_WARMUP or args.measured < MIN_MEASURED:
        parser.error(
            f"require runs >= {MIN_RUNS}, warmup >= {MIN_WARMUP}, and measured >= {MIN_MEASURED}"
        )
    geometries = [
        _parse_geometry(value.strip()) for value in args.geometries.split(",") if value.strip()
    ]
    if not geometries:
        parser.error("select at least one geometry")

    conditions_start = _runtime_conditions()
    _require_conditions(conditions_start, "parent start")
    revision_start = _revision_state()

    report: dict[str, Any] = {
        "schema": 1,
        "machine": {
            "platform": platform.platform(),
            "hardware_model": _command_value(["sysctl", "-n", "hw.model"]),
            "chip": _command_value(["sysctl", "-n", "machdep.cpu.brand_string"]),
        },
        "runtime": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
        },
        "product_revision_start": revision_start,
        "conditions_start": conditions_start,
        "protocol": {
            "fresh_process_per_run": True,
            "runs": args.runs,
            "warmup_frames": args.warmup,
            "measured_frames": args.measured,
            "matrix": "bt709",
            "range": "video",
            "input_dtype": "float32",
            "compatibility_boundary": "float32 -> float16 -> float32",
        },
        "geometries": {},
    }
    for geometry in geometries:
        width, height = geometry
        by_path = {"legacy": [], "direct": []}
        run_order = []
        for run_index in range(args.runs):
            order = ("legacy", "direct") if run_index % 2 == 0 else ("direct", "legacy")
            run_order.extend(order)
            for path in order:
                by_path[path].append(_run_worker(path, geometry, args.warmup, args.measured))
                _require_revision(
                    revision_start,
                    f"{width}x{height} run {run_index + 1} {path}",
                )
        digests = {run["active_yuv_sha256"] for runs in by_path.values() for run in runs}
        if len(digests) != 1:
            raise RuntimeError(f"{width}x{height} active YUV differs across paths or runs")
        legacy = _summarize(by_path["legacy"])
        direct = _summarize(by_path["direct"])
        avoided_bytes = width * height * 20
        report["geometries"][f"{width}x{height}"] = {
            "legacy": legacy,
            "direct": direct,
            "speedup": (legacy["median_ms_per_frame"] / direct["median_ms_per_frame"]),
            "median_ms_saved_per_frame": (
                legacy["median_ms_per_frame"] - direct["median_ms_per_frame"]
            ),
            "rgba_pool_acquisitions_avoided_per_frame": 1,
            "minimum_traffic_avoided_bytes_per_frame": avoided_bytes,
            "minimum_traffic_avoided_gib_per_second_at_30fps": (avoided_bytes * 30 / (1024.0**3)),
            "active_yuv_match": True,
            "worker_execution_order": run_order,
        }
    report["product_revision_end"] = _require_revision(revision_start, "benchmark end")
    conditions_end = _runtime_conditions()
    report["conditions_end"] = conditions_end
    _require_conditions(
        conditions_end,
        "parent end",
        expected=conditions_start,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger(__name__).info("%s", rendered.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
