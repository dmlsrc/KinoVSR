"""Measure bounded VideoToolbox output pools against fresh allocation.

Every sample runs in a fresh process. Warmup includes native setup; the
following outputs measure steady state. The pooled and fresh paths must emit
the same final active-pixel digest. Underlying IOSurface IDs, not Python proxy
or pool-acquire counts, prove whether expensive surfaces plateau.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from bench_mlx_encode_bridge import (
    _command_value,
    _peak_rss_mib,
    _percentile,
    _require_conditions,
    _require_revision,
    _revision_state,
    _runtime_conditions,
)

MIN_RUNS = 3
MIN_WARMUP = 30
MIN_MEASURED = 120
DEFAULT_RUNS = 3
DEFAULT_WARMUP = 30
DEFAULT_MEASURED = 120
WORKLOADS = ("vsr", "frc")
PATHS = ("fresh", "pooled")


def _attrs(pixel_format: int, width: int, height: int) -> dict[str, Any]:
    return {
        "PixelFormatType": pixel_format,
        "Width": width,
        "Height": height,
        "IOSurfaceProperties": {},
        "MetalCompatibility": True,
    }


def _surface_id(buffer: Any) -> int:
    from kinovsr.native.frameworks import Quartz

    surface = Quartz.CVPixelBufferGetIOSurface(buffer)
    if surface is None:
        raise RuntimeError("benchmark output is not IOSurface-backed")
    return int(surface.surfaceID())


def _active_digest(buffer: Any) -> str:
    from kinovsr.native.frameworks import Quartz

    digest = hashlib.sha256()
    width = int(Quartz.CVPixelBufferGetWidth(buffer))
    height = int(Quartz.CVPixelBufferGetHeight(buffer))
    Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
    try:
        if Quartz.CVPixelBufferIsPlanar(buffer):
            for plane in range(Quartz.CVPixelBufferGetPlaneCount(buffer)):
                rows = Quartz.CVPixelBufferGetHeightOfPlane(buffer, plane)
                bpr = Quartz.CVPixelBufferGetBytesPerRowOfPlane(buffer, plane)
                row_bytes = Quartz.CVPixelBufferGetWidthOfPlane(buffer, plane)
                base = Quartz.CVPixelBufferGetBaseAddressOfPlane(buffer, plane)
                raw = base.as_buffer(rows * bpr)
                for row in range(rows):
                    digest.update(raw[row * bpr : row * bpr + row_bytes])
        else:
            bpr = Quartz.CVPixelBufferGetBytesPerRow(buffer)
            row_bytes = width * 8
            base = Quartz.CVPixelBufferGetBaseAddress(buffer)
            raw = base.as_buffer(height * bpr)
            for row in range(height):
                digest.update(raw[row * bpr : row * bpr + row_bytes])
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)
    return digest.hexdigest()


def _install_fresh_allocation_counter():
    from kinovsr.media import pixel_buffers as pb

    original = pb.make_pixel_buffer_from_attrs
    fresh_allocations = 0

    def counted(*args, **kwargs):
        nonlocal fresh_allocations
        fresh_allocations += 1
        return original(*args, **kwargs)

    pb.make_pixel_buffer_from_attrs = counted

    return lambda: fresh_allocations


def _install_fresh_destination(session: Any, width: int, height: int) -> None:
    from kinovsr.media import pixel_buffers as pb

    session._make_dst_buffer = lambda: pb.make_pixel_buffer_from_attrs(
        width, height, session.dst_attrs
    )


def _vsr_worker(path: str, warmup: int, measured: int) -> dict[str, Any]:
    import mlx.core as mx

    from kinovsr.native.vsr import VsrSession

    width = height = 256
    mx.random.seed(29)
    rgb_a = mx.random.uniform(shape=(height, width, 3)).astype(mx.float16)
    rgb_b = mx.random.uniform(shape=(height, width, 3)).astype(mx.float16)
    alpha = mx.ones((height, width, 1), dtype=mx.float16)
    inputs = (
        mx.concatenate([rgb_a, alpha], axis=-1),
        mx.concatenate([rgb_b, alpha], axis=-1),
    )
    mx.eval(*inputs)
    session = VsrSession(width, height, mode="balanced", fps=24.0)
    fresh_count = _install_fresh_allocation_counter()
    if path == "fresh":
        _install_fresh_destination(session, session.out_w, session.out_h)
    surface_ids: set[int] = set()
    frame_ms = []
    digest = None
    try:
        for index in range(warmup):
            output = session.upscale_to_buffer(inputs[index % 2], index)
            del output
        fresh_before = fresh_count()
        started = time.perf_counter()
        for offset in range(measured):
            index = warmup + offset
            frame_started = time.perf_counter()
            output = session.upscale_to_buffer(inputs[index % 2], index)
            frame_ms.append((time.perf_counter() - frame_started) * 1000.0)
            surface_ids.add(_surface_id(output))
            if offset == measured - 1:
                digest = _active_digest(output)
            del output
        steady_ms = (time.perf_counter() - started) * 1000.0
        fresh_measured = fresh_count() - fresh_before
    finally:
        session.close()
    return {
        "workload": "vsr",
        "path": path,
        "geometry": "256x256->1024x1024",
        "mode": "balanced",
        "warmup_outputs": warmup,
        "measured_outputs": measured,
        "steady_total_ms": steady_ms,
        "frame_ms": frame_ms,
        "unique_destination_iosurfaces": len(surface_ids),
        "destination_surface_ids": sorted(surface_ids),
        "destination_pool_cap": 2 if path == "pooled" else None,
        "fresh_destination_allocations": fresh_measured,
        "final_active_sha256": digest,
        "peak_rss_mib": _peak_rss_mib(),
    }


def _frc_source(seed: int):
    import numpy as np

    from kinovsr.media import pixel_buffers as pb

    width, height = 128, 96
    y, x = np.mgrid[0:height, 0:width]
    rgba = np.stack(
        (
            ((x + seed * 3) % width) / (width - 1),
            ((y + seed * 5) % height) / (height - 1),
            ((x + y + seed * 7) % (width + height)) / (width + height - 1),
            np.ones_like(x),
        ),
        axis=-1,
    ).astype(np.float16)
    buffer = pb.make_pixel_buffer_from_attrs(width, height, _attrs(pb.PIX_RGBAHALF, width, height))
    pb.write_fp16_rgba(rgba, buffer)
    return buffer


def _frc_worker(path: str, warmup: int, measured: int) -> dict[str, Any]:
    from kinovsr.native.temporal import VtfrcSession

    outputs_per_input = 10
    if warmup % outputs_per_input or measured % outputs_per_input:
        raise RuntimeError("FRC warmup and measured outputs must be multiples of 10")
    sources = (_frc_source(1), _frc_source(2))
    session = VtfrcSession(128, 96, 24, 240, mode="normal")
    fresh_count = _install_fresh_allocation_counter()
    if path == "fresh":
        _install_fresh_destination(session, 128, 96)
    surface_ids: set[int] = set()
    frame_ms = []
    digest = None
    source_index = 0
    assert list(session.feed(sources[0], source_index)) == []
    source_index += 1
    try:
        for _ in range(warmup // outputs_per_input):
            for output in session.feed(sources[source_index % 2], source_index):
                del output
            source_index += 1
        fresh_before = fresh_count()
        started = time.perf_counter()
        measured_count = 0
        for _ in range(measured // outputs_per_input):
            group_started = time.perf_counter()
            group = session.feed(sources[source_index % 2], source_index)
            group_ids = []
            for output in group:
                group_ids.append(_surface_id(output))
                measured_count += 1
                if measured_count == measured:
                    digest = _active_digest(output)
                del output
            group_ms = (time.perf_counter() - group_started) * 1000.0 / len(group_ids)
            frame_ms.extend([group_ms] * len(group_ids))
            surface_ids.update(group_ids)
            source_index += 1
        steady_ms = (time.perf_counter() - started) * 1000.0
        fresh_measured = fresh_count() - fresh_before
        fresh_before_drain = fresh_count()
        drain_ids = []
        for output in session.drain():
            drain_ids.append(_surface_id(output))
            del output
        surface_ids.update(drain_ids)
        fresh_drain = fresh_count() - fresh_before_drain
    finally:
        session.close()
    if measured_count != measured:
        raise RuntimeError(f"FRC emitted {measured_count} measured outputs, expected {measured}")
    return {
        "workload": "frc",
        "path": path,
        "geometry": "128x96",
        "mode": "normal 24->240",
        "warmup_outputs": warmup,
        "measured_outputs": measured,
        "steady_total_ms": steady_ms,
        "frame_ms": frame_ms,
        "unique_destination_iosurfaces": len(surface_ids),
        "destination_surface_ids": sorted(surface_ids),
        "destination_pool_cap": 7 if path == "pooled" else None,
        "fresh_destination_allocations": fresh_measured,
        "drain_fresh_destination_allocations": fresh_drain,
        "drain_outputs": len(drain_ids),
        "final_active_sha256": digest,
        "peak_rss_mib": _peak_rss_mib(),
    }


def _worker(workload: str, path: str, warmup: int, measured: int) -> dict[str, Any]:
    conditions_start = _runtime_conditions()
    _require_conditions(conditions_start, f"{workload}/{path} worker start")
    result = (
        _vsr_worker(path, warmup, measured)
        if workload == "vsr"
        else _frc_worker(path, warmup, measured)
    )
    conditions_end = _runtime_conditions()
    _require_conditions(
        conditions_end,
        f"{workload}/{path} worker end",
        expected=conditions_start,
    )
    result["conditions_start"] = conditions_start
    result["conditions_end"] = conditions_end
    return result


def _run_worker(workload: str, path: str, warmup: int, measured: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="kinovsr-vt-pool-bench-") as tmp:
        result_path = Path(tmp) / "result.json"
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker-workload",
                workload,
                "--worker-path",
                path,
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
            raise RuntimeError(f"{workload}/{path} worker failed:\n{proc.stderr[-5000:]}")
        if not result_path.is_file():
            raise RuntimeError(
                f"{workload}/{path} worker returned no result:\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        return json.loads(result_path.read_text())


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_run = [run["steady_total_ms"] / run["measured_outputs"] for run in runs]
    frame_samples = [float(value) for run in runs for value in run["frame_ms"]]
    measured_fresh = [int(run["fresh_destination_allocations"]) for run in runs]
    drain_fresh = [int(run.get("drain_fresh_destination_allocations", 0)) for run in runs]
    return {
        "median_ms_per_output": statistics.median(per_run),
        "p95_ms_per_output": _percentile(frame_samples, 0.95),
        "per_run_ms_per_output": per_run,
        "peak_rss_mib": max(float(run["peak_rss_mib"]) for run in runs),
        "max_unique_destination_iosurfaces": max(
            int(run["unique_destination_iosurfaces"]) for run in runs
        ),
        "fresh_destination_allocations_per_run": measured_fresh,
        "drain_fresh_destination_allocations_per_run": drain_fresh,
        "total_fresh_destination_allocations_per_run": [
            measured + drain for measured, drain in zip(measured_fresh, drain_fresh, strict=True)
        ],
        "final_active_sha256": runs[0]["final_active_sha256"],
        "conditions": [
            {"start": run["conditions_start"], "end": run["conditions_end"]} for run in runs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--measured", type=int, default=DEFAULT_MEASURED)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--output")
    parser.add_argument("--worker-workload", choices=WORKLOADS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-path", choices=PATHS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_workload:
        if not args.worker_path or not args.worker_output:
            parser.error("worker mode requires path and output")
        result = _worker(args.worker_workload, args.worker_path, args.warmup, args.measured)
        Path(args.worker_output).write_text(json.dumps(result, sort_keys=True) + "\n")
        return 0
    if args.runs < MIN_RUNS or args.warmup < MIN_WARMUP or args.measured < MIN_MEASURED:
        parser.error(
            f"require runs >= {MIN_RUNS}, warmup >= {MIN_WARMUP}, and measured >= {MIN_MEASURED}"
        )
    workloads = tuple(value.strip() for value in args.workloads.split(",") if value.strip())
    if not workloads or any(value not in WORKLOADS for value in workloads):
        parser.error(f"workloads must be selected from {WORKLOADS}")

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
            "alternating_path_order": True,
            "runs": args.runs,
            "warmup_outputs": args.warmup,
            "measured_outputs": args.measured,
            "identity": "IOSurface.surfaceID",
            "retained_host_outputs": False,
        },
        "workloads": {},
    }
    for workload in workloads:
        by_path = {path: [] for path in PATHS}
        execution_order = []
        for run_index in range(args.runs):
            order = PATHS if run_index % 2 == 0 else tuple(reversed(PATHS))
            for path in order:
                execution_order.append(path)
                by_path[path].append(_run_worker(workload, path, args.warmup, args.measured))
                _require_revision(
                    revision_start,
                    f"{workload} run {run_index + 1} {path}",
                )
        digests = {
            run["final_active_sha256"] for path_runs in by_path.values() for run in path_runs
        }
        if len(digests) != 1:
            raise RuntimeError(f"{workload} active pixels differ across paths or runs")
        fresh = _summarize(by_path["fresh"])
        pooled = _summarize(by_path["pooled"])
        if any(pooled["total_fresh_destination_allocations_per_run"]):
            raise RuntimeError(f"{workload} pooled path allocated a fresh destination")
        cap = int(by_path["pooled"][0]["destination_pool_cap"])
        if pooled["max_unique_destination_iosurfaces"] > cap:
            raise RuntimeError(
                f"{workload} used {pooled['max_unique_destination_iosurfaces']} "
                f"destination surfaces with cap {cap}"
            )
        report["workloads"][workload] = {
            "fresh": fresh,
            "pooled": pooled,
            "destination_pool_cap": cap,
            "allocation_plateau_verified": True,
            "active_pixels_match": True,
            "execution_order": execution_order,
            "median_speedup": (fresh["median_ms_per_output"] / pooled["median_ms_per_output"]),
        }
    report["product_revision_end"] = _require_revision(revision_start, "benchmark end")
    conditions_end = _runtime_conditions()
    report["conditions_end"] = conditions_end
    _require_conditions(conditions_end, "parent end", expected=conditions_start)
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
