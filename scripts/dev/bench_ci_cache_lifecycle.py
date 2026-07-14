#!/usr/bin/env python3
"""Measure Core Image conversion cleanup, RSS plateau, and steady latency.

Run this script in a fresh process. The default workload drives the typed host
session through the Core Image spatial stage. ``--path helpers`` instead reuses
one IOSurface-backed NV12 destination for RGB upload and readback. Samples use
the process's own Mach VM region tags, so Core Image and IOSurface growth are
measured without privileged process inspection.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import importlib.metadata
import json
import logging
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

CLEANUP_INTERVAL = 64
KERN_INVALID_ADDRESS = 1
VM_MEMORY_IOKIT = 21
VM_MEMORY_COREIMAGE = 68
VM_MEMORY_IOSURFACE = 88
VM_TAGS = {
    VM_MEMORY_IOKIT: "iokit",
    VM_MEMORY_COREIMAGE: "coreimage",
    VM_MEMORY_IOSURFACE: "iosurface",
}


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _current_rss_mib() -> float:
    try:
        raw = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        )
        return int(raw.strip()) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return _peak_rss_mib()


class _VmRegionSubmapInfo64(ctypes.Structure):
    _fields_ = [
        ("protection", ctypes.c_int32),
        ("max_protection", ctypes.c_int32),
        ("inheritance", ctypes.c_int32),
        ("offset", ctypes.c_uint64),
        ("user_tag", ctypes.c_uint32),
        ("pages_resident", ctypes.c_uint32),
        ("pages_shared_now_private", ctypes.c_uint32),
        ("pages_swapped_out", ctypes.c_uint32),
        ("pages_dirtied", ctypes.c_uint32),
        ("ref_count", ctypes.c_uint32),
        ("shadow_depth", ctypes.c_uint16),
        ("external_pager", ctypes.c_uint8),
        ("share_mode", ctypes.c_uint8),
        ("is_submap", ctypes.c_int32),
        ("behavior", ctypes.c_int32),
        ("object_id", ctypes.c_uint32),
        ("user_wired_count", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
        ("pages_reusable", ctypes.c_uint32),
        ("object_id_full", ctypes.c_uint64),
    ]


def _walk_vm_tags(
    region: Callable[..., int],
    task_self: Callable[[], int],
    *,
    page_size: int,
) -> dict[str, dict[str, float]]:
    """Walk Mach regions, accepting only the documented end sentinel."""
    address = ctypes.c_uint64(0)
    size = ctypes.c_uint64(0)
    depth = ctypes.c_uint32(0)
    totals = {
        name: {"regions": 0.0, "virtual_mib": 0.0, "resident_mib": 0.0}
        for name in VM_TAGS.values()
    }
    for _ in range(200_000):
        info = _VmRegionSubmapInfo64()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
        status = region(
            task_self(),
            ctypes.byref(address),
            ctypes.byref(size),
            ctypes.byref(depth),
            ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(count),
        )
        if status == KERN_INVALID_ADDRESS:
            return totals
        if status != 0:
            raise RuntimeError(
                "mach_vm_region_recurse failed before end-of-map: "
                f"status={status}"
            )
        if info.is_submap:
            depth.value += 1
            continue
        name = VM_TAGS.get(int(info.user_tag))
        if name is not None:
            totals[name]["regions"] += 1.0
            totals[name]["virtual_mib"] += size.value / (1024.0 * 1024.0)
            totals[name]["resident_mib"] += (
                info.pages_resident * page_size / (1024.0 * 1024.0)
            )
        if size.value == 0:
            return totals
        address.value += size.value
    raise RuntimeError("Mach VM region walk exceeded its safety bound")


def _vm_tag_usage() -> dict[str, dict[str, float]] | None:
    """Read this process's Core Image/IOSurface Mach VM region tags."""
    if platform.system() != "Darwin":
        return None
    library = ctypes.CDLL(None)
    task_self = library.mach_task_self
    task_self.restype = ctypes.c_uint32
    region = library.mach_vm_region_recurse
    region.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    region.restype = ctypes.c_int32
    return _walk_vm_tags(
        region,
        task_self,
        page_size=int(os.sysconf("SC_PAGE_SIZE")),
    )


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _revision_state(module_path: Path) -> dict[str, Any]:
    repo = module_path.parents[2]
    revision = _git(repo, "rev-parse", "HEAD")
    dirty = bool(_git(repo, "status", "--porcelain"))
    return {"revision": revision, "dirty": dirty}


def _surface_id(buffer: Any) -> int:
    from kinovsr.native.frameworks import Quartz

    surface = Quartz.CVPixelBufferGetIOSurface(buffer)
    if surface is None:
        raise RuntimeError("conversion destination is not IOSurface-backed")
    return int(surface.surfaceID())


def _rss_summary(samples: list[dict[str, float]], tail_count: int) -> dict[str, float]:
    tail = samples[-min(len(samples), max(2, tail_count)):]
    rss = [sample["rss_mib"] for sample in tail]
    renders = [sample["renders"] for sample in tail]
    if len(tail) > 1 and len(set(renders)) > 1:
        slope = statistics.linear_regression(renders, rss).slope * 1_000.0
    else:
        slope = 0.0
    return {
        "tail_samples": len(tail),
        "tail_render_span": renders[-1] - renders[0],
        "tail_rss_first_mib": rss[0],
        "tail_rss_last_mib": rss[-1],
        "tail_rss_span_mib": max(rss) - min(rss),
        "tail_rss_growth_mib": max(0.0, rss[-1] - rss[0]),
        "tail_rss_slope_mib_per_1000_renders": slope,
    }


def _vm_tail_summary(
    samples: list[dict[str, Any]],
    tail_count: int,
) -> dict[str, Any] | None:
    if not samples or samples[0].get("vm_tags") is None:
        return None
    tail = samples[-min(len(samples), max(2, tail_count)):]
    summary: dict[str, Any] = {}
    for tag in VM_TAGS.values():
        resident = [sample["vm_tags"][tag]["resident_mib"] for sample in tail]
        virtual = [sample["vm_tags"][tag]["virtual_mib"] for sample in tail]
        regions = [sample["vm_tags"][tag]["regions"] for sample in tail]
        summary[tag] = {
            "resident_first_mib": resident[0],
            "resident_last_mib": resident[-1],
            "resident_span_mib": max(resident) - min(resident),
            "resident_growth_mib": max(0.0, resident[-1] - resident[0]),
            "virtual_first_mib": virtual[0],
            "virtual_last_mib": virtual[-1],
            "virtual_span_mib": max(virtual) - min(virtual),
            "virtual_growth_mib": max(0.0, virtual[-1] - virtual[0]),
            "region_first": regions[0],
            "region_last": regions[-1],
            "region_growth": max(0.0, regions[-1] - regions[0]),
        }
    return summary


def _expected_clears(total_renders: int) -> int:
    return (total_renders + CLEANUP_INTERVAL - 1) // CLEANUP_INTERVAL


def _hardware_model() -> str:
    if platform.system() != "Darwin":
        return platform.machine()
    try:
        model = subprocess.check_output(
            ["sysctl", "-n", "hw.model"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("could not identify Apple hardware model") from exc
    if not model:
        raise RuntimeError("Apple hardware model query returned an empty value")
    return model


def _environment_state() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "hardware_model": _hardware_model(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": importlib.metadata.version("mlx"),
    }


def _validate_baseline(
    baseline: dict[str, Any],
    *,
    protocol: dict[str, Any],
    environment: dict[str, str],
) -> None:
    """Reject latency comparisons whose timing domains are not equivalent."""
    if baseline.get("schema") != 1:
        raise RuntimeError(
            f"baseline schema mismatch: {baseline.get('schema')!r} != 1"
        )
    baseline_environment = baseline.get("environment")
    if baseline_environment != environment:
        raise RuntimeError(
            "baseline environment mismatch: "
            f"{baseline_environment!r} != {environment!r}"
        )
    baseline_protocol = baseline.get("protocol", {})
    for key, value in protocol.items():
        if key in {
            "managed_lifecycle_available",
            "cleanup_interval_renders",
        }:
            continue
        if baseline_protocol.get(key) != value:
            raise RuntimeError(
                f"baseline protocol mismatch for {key}: "
                f"{baseline_protocol.get(key)!r} != {value!r}"
            )

    current_policy = (
        protocol["managed_lifecycle_available"],
        protocol["cleanup_interval_renders"],
    )
    baseline_policy = (
        baseline_protocol.get("managed_lifecycle_available"),
        baseline_protocol.get("cleanup_interval_renders"),
    )
    # The historical PERF-08 baseline is intentionally unmanaged. Otherwise
    # cleanup policy must match exactly; arbitrary policy drift is not a valid
    # latency comparison.
    allowed_policies = {current_policy}
    if current_policy == (True, CLEANUP_INTERVAL):
        allowed_policies.add((False, None))
    if baseline_policy not in allowed_policies:
        raise RuntimeError(
            "baseline cleanup policy mismatch: "
            f"{baseline_policy!r} not in {sorted(allowed_policies)!r}"
        )


def _protocol_sufficient(
    *,
    iterations: int,
    sample_every: int,
    requested_tail_samples: int,
    renders_per_iteration: int,
    sample_count: int,
    tail_render_span: float,
) -> bool:
    return (
        requested_tail_samples >= 4
        and sample_count >= requested_tail_samples
        and iterations % sample_every == 0
        and iterations * renders_per_iteration >= 4_096
        and tail_render_span >= 2 * CLEANUP_INTERVAL
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    if args.product_root is not None:
        product_root = args.product_root.resolve()
        if not (product_root / "kinovsr").is_dir():
            raise RuntimeError(
                f"--product-root has no kinovsr package: {product_root}"
            )
        # Import MLX before inserting a temporary implementation root. On the
        # supported macOS runtime, setting PYTHONPATH before process launch can
        # change Metal device discovery; an in-process path override selects
        # the historical product tree without perturbing device startup.
        sys.path.insert(0, str(product_root))

    from kinovsr.media import pixel_buffers as pb

    clear_calls = 0
    original_clear = getattr(pb, "_clear_ci_context", None)
    managed = callable(getattr(pb, "ci_cache_owner", None))
    if original_clear is not None:
        def counted_clear() -> None:
            nonlocal clear_calls
            clear_calls += 1
            original_clear()

        pb._clear_ci_context = counted_clear
    samples: list[dict[str, Any]] = []
    surface_ids: list[int] = []
    run = None
    try:
        with contextlib.ExitStack() as stack:
            values = mx.arange(
                args.width * args.height * 3,
                dtype=mx.uint32,
            )
            if args.path == "helpers":
                from kinovsr.native.frameworks import Quartz

                renders_per_iteration = 2
                frame = (values % 251).astype(mx.uint8).reshape(
                    args.height,
                    args.width,
                    3,
                )
                attrs = {
                    Quartz.kCVPixelBufferPixelFormatTypeKey: pb.PIX_NV12,
                    Quartz.kCVPixelBufferWidthKey: args.width,
                    Quartz.kCVPixelBufferHeightKey: args.height,
                    Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
                    Quartz.kCVPixelBufferMetalCompatibilityKey: True,
                }
                destination = pb.make_pixel_buffer_from_attrs(
                    args.width,
                    args.height,
                    attrs,
                )
                surface_ids.append(_surface_id(destination))
                stack.enter_context(
                    pb.ci_cache_owner() if managed else contextlib.nullcontext()
                )

                def convert_once() -> None:
                    pb.upload_frame_to_buffer(frame, destination)
                    rgb = pb.read_pixel_buffer_rgb(destination)
                    mx.eval(rgb)

            else:
                from fractions import Fraction

                from kinovsr.pipeline import open_pipeline
                from kinovsr.processors import (
                    FrameUnit,
                    Geometry,
                    StreamSpec,
                    TimelineSpec,
                    frame_spec_for_matrix,
                )
                from kinovsr.settings import Settings

                renders_per_iteration = 1
                frame = ((values % 251).astype(mx.float32) / 250.0).reshape(
                    args.height,
                    args.width,
                    3,
                )
                input_spec = StreamSpec(
                    frame=frame_spec_for_matrix(
                        "bt709",
                        full_range=True,
                        geometry=Geometry(args.width, args.height),
                    ),
                    timeline=TimelineSpec(
                        time_base=Fraction(1, 24_000),
                        cadence=Fraction(25),
                    ),
                )
                total = args.warmup + args.iterations

                def units():
                    for index in range(total):
                        yield FrameUnit(
                            payload=frame,
                            pts=index * 960,
                            duration=960,
                        )

                session = stack.enter_context(open_pipeline(
                    {
                        "pipeline": ["spatial"],
                        "spatial": {
                            "processor": "spatial",
                            "strength": 0.5,
                        },
                    },
                    input_spec,
                    settings=Settings(),
                ))
                run = stack.enter_context(session.process(
                    units(),
                    retain_outputs=False,
                ))

                def convert_once() -> None:
                    unit = next(run)
                    mx.eval(unit.payload)

            mx.eval(frame)
            for _ in range(args.warmup):
                convert_once()
            measured_started = time.perf_counter()
            batch_started = measured_started
            batch_iterations = 0
            for index in range(1, args.iterations + 1):
                convert_once()
                batch_iterations += 1
                if index % args.sample_every == 0 or index == args.iterations:
                    batch_elapsed = time.perf_counter() - batch_started
                    samples.append({
                        "iterations": float(index),
                        "renders": float(index * renders_per_iteration),
                        "rss_mib": _current_rss_mib(),
                        "peak_rss_mib": _peak_rss_mib(),
                        "vm_tags": _vm_tag_usage(),
                        "batch_ms_per_render": (
                            batch_elapsed * 1_000.0
                            / (batch_iterations * renders_per_iteration)
                        ),
                    })
                    batch_started = time.perf_counter()
                    batch_iterations = 0
            measured_seconds = time.perf_counter() - measured_started
            if run is not None:
                try:
                    next(run)
                except StopIteration:
                    pass
                else:
                    raise RuntimeError("typed host workload emitted extra units")
    finally:
        if original_clear is not None:
            pb._clear_ci_context = original_clear

    total_renders = (args.warmup + args.iterations) * renders_per_iteration
    summary = _rss_summary(samples, args.tail_samples)
    vm_summary = _vm_tail_summary(samples, args.tail_samples)
    summary.update({
        "measured_seconds": measured_seconds,
        "median_batch_ms_per_render": statistics.median(
            sample["batch_ms_per_render"] for sample in samples
        ),
        "final_rss_mib": _current_rss_mib(),
        "peak_rss_mib": _peak_rss_mib(),
        "clear_calls": clear_calls if original_clear is not None else None,
        "expected_clear_calls": _expected_clears(total_renders) if managed else None,
        "destination_iosurface_ids": surface_ids,
        "vm_tail": vm_summary,
    })
    cleanup_ok = (
        not managed
        or summary["clear_calls"] == summary["expected_clear_calls"]
    )
    protocol_ok = _protocol_sufficient(
        iterations=args.iterations,
        sample_every=args.sample_every,
        requested_tail_samples=args.tail_samples,
        renders_per_iteration=renders_per_iteration,
        sample_count=len(samples),
        tail_render_span=summary["tail_render_span"],
    )
    rss_ok = (
        summary["tail_rss_growth_mib"] <= args.max_tail_growth_mib
        and summary["tail_rss_span_mib"] <= args.max_tail_span_mib
        and summary["tail_rss_slope_mib_per_1000_renders"]
        <= args.max_tail_slope_mib_per_1000_renders
    )
    vm_ok = vm_summary is not None and all(
        vm_summary[tag]["resident_growth_mib"] <= args.max_vm_growth_mib
        and vm_summary[tag]["virtual_growth_mib"] <= args.max_vm_virtual_growth_mib
        and vm_summary[tag]["region_growth"] <= args.max_vm_region_growth
        for tag in ("iokit", "coreimage", "iosurface")
    )
    latency = None
    latency_ok = True
    environment = _environment_state()
    if args.baseline_report is not None:
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        expected_protocol = {
            "fresh_process_required": True,
            "path": args.path,
            "geometry": [args.width, args.height],
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "renders_per_iteration": renders_per_iteration,
            "sample_every_iterations": args.sample_every,
            "managed_lifecycle_available": managed,
            "cleanup_interval_renders": (
                CLEANUP_INTERVAL if managed else None
            ),
        }
        # Cleanup availability is allowed to differ only for the intentional
        # unmanaged pre-PERF-08 baseline; all timing-shaping fields and the
        # complete machine/runtime environment must match.
        comparison_protocol = {
            key: value
            for key, value in expected_protocol.items()
            if key not in {
                "managed_lifecycle_available",
                "cleanup_interval_renders",
            }
        }
        _validate_baseline(
            baseline,
            protocol={
                **comparison_protocol,
                "managed_lifecycle_available": managed,
                "cleanup_interval_renders": (
                    CLEANUP_INTERVAL if managed else None
                ),
            },
            environment=environment,
        )
        baseline_ms = float(baseline["summary"]["median_batch_ms_per_render"])
        current_ms = float(summary["median_batch_ms_per_render"])
        regression_pct = (current_ms / baseline_ms - 1.0) * 100.0
        latency_ok = regression_pct <= args.max_latency_regression_pct
        latency = {
            "baseline_ms_per_render": baseline_ms,
            "current_ms_per_render": current_ms,
            "regression_pct": regression_pct,
            "max_regression_pct": args.max_latency_regression_pct,
            "pass": latency_ok,
        }
    plateau_ok = protocol_ok and rss_ok and vm_ok
    return {
        "schema": 1,
        "revision": _revision_state(Path(pb.__file__).resolve()),
        "environment": environment,
        "protocol": {
            "fresh_process_required": True,
            "path": args.path,
            "geometry": [args.width, args.height],
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "renders_per_iteration": renders_per_iteration,
            "sample_every_iterations": args.sample_every,
            "cleanup_interval_renders": CLEANUP_INTERVAL if managed else None,
            "managed_lifecycle_available": managed,
        },
        "samples": samples,
        "summary": summary,
        "latency_comparison": latency,
        "gates": {
            "cleanup_count": cleanup_ok,
            "protocol_sufficient": protocol_ok,
            "rss_plateau": plateau_ok,
            "rss_bounds": rss_ok,
            "vm_tag_plateau": vm_ok,
            "latency_margin": latency_ok if latency is not None else None,
            "pass": cleanup_ok and plateau_ok and latency_ok,
            "max_tail_growth_mib": args.max_tail_growth_mib,
            "max_tail_span_mib": args.max_tail_span_mib,
            "max_tail_slope_mib_per_1000_renders": (
                args.max_tail_slope_mib_per_1000_renders
            ),
            "max_vm_growth_mib": args.max_vm_growth_mib,
            "max_vm_virtual_growth_mib": args.max_vm_virtual_growth_mib,
            "max_vm_region_growth": args.max_vm_region_growth,
            "min_tail_render_span": 2 * CLEANUP_INTERVAL,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=("host", "helpers"), default="host")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=4_096)
    parser.add_argument("--sample-every", type=int, default=128)
    parser.add_argument("--tail-samples", type=int, default=16)
    parser.add_argument("--max-tail-growth-mib", type=float, default=4.0)
    parser.add_argument("--max-tail-span-mib", type=float, default=8.0)
    parser.add_argument(
        "--max-tail-slope-mib-per-1000-renders",
        type=float,
        default=1.0,
    )
    parser.add_argument("--max-vm-growth-mib", type=float, default=2.0)
    parser.add_argument("--max-vm-virtual-growth-mib", type=float, default=4.0)
    parser.add_argument("--max-vm-region-growth", type=float, default=2.0)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument(
        "--product-root",
        type=Path,
        help="load KinoVSR from this tree (for a detached baseline)",
    )
    parser.add_argument("--max-latency-regression-pct", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-gates", action="store_true")
    parser.add_argument("--assert-latency", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("width", "height", "iterations", "sample_every", "tail_samples"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    report = _run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    logging.getLogger(__name__).info("%s", rendered)
    if args.assert_gates and not report["gates"]["pass"]:
        return 1
    if args.assert_latency:
        comparison = report["latency_comparison"]
        if comparison is None or not comparison["pass"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
