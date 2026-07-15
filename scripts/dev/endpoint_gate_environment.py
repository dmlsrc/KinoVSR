"""Environment, workload, and provenance signatures for endpoint gates."""

from __future__ import annotations

import hashlib
import platform
import resource
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from endpoint_gate_protocol import (
    QUALITY_MAX_ABS_RGB10,
    QUALITY_MIN_PSNR_DB,
    SINK_HOLDBACK_FRAMES,
    TIMING_SCHEMA,
    _gate_definition,
    _jsonable,
    _total_frames,
)
from endpoint_gate_quality import _track_metadata


def _power_source() -> str:
    proc = subprocess.run(
        ["pmset", "-g", "batt"],
        capture_output=True,
        text=True,
        check=False,
    )
    first = proc.stdout.splitlines()[0] if proc.stdout.splitlines() else ""
    marker = "Now drawing from '"
    if marker in first:
        return first.split(marker, 1)[1].split("'", 1)[0]
    return first.strip() or "unknown"


def _runtime_conditions() -> dict[str, Any]:
    from kinovsr.native.frameworks import Foundation

    process = Foundation.NSProcessInfo.processInfo()
    thermal_value = int(process.thermalState())
    thermal_names = {
        0: "nominal",
        1: "fair",
        2: "serious",
        3: "critical",
    }
    thermal = thermal_names.get(thermal_value, f"unknown_{thermal_value}")
    return {
        "power_source": _power_source(),
        "low_power_mode": bool(process.isLowPowerModeEnabled()),
        "thermal_state": thermal,
    }


def _comparison_conditions() -> dict[str, Any]:
    current = _runtime_conditions()
    if current["thermal_state"] != "nominal":
        raise RuntimeError(
            f"benchmark requires a nominal thermal state; got {current['thermal_state']!r}"
        )
    return {
        "power_source": current["power_source"],
        "low_power_mode": current["low_power_mode"],
        "thermal_state": current["thermal_state"],
        "thermal_policy": "nominal_only",
    }


def _assert_run_conditions(
    runs: Sequence[dict[str, Any]],
    expected: Mapping[str, Any],
) -> None:
    mismatches: list[str] = []
    for run_index, run in enumerate(runs):
        for boundary in ("conditions_start", "conditions_end"):
            actual = run[boundary]
            comparable = {
                "power_source": actual["power_source"],
                "low_power_mode": actual["low_power_mode"],
                "thermal_state": actual["thermal_state"],
            }
            for key in ("power_source", "low_power_mode", "thermal_state"):
                if comparable[key] != expected[key]:
                    mismatches.append(
                        f"runs[{run_index}].{boundary}.{key}: "
                        f"expected={expected[key]!r}, actual={comparable[key]!r}"
                    )
    if mismatches:
        raise RuntimeError("benchmark runtime conditions changed: " + "; ".join(mismatches))


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sysctl(name: str) -> str:
    proc = subprocess.run(
        ["sysctl", "-n", name],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() or "unknown"


def _command_value(command: Sequence[str]) -> str:
    proc = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() or "unknown"


def _checkpoint_signature(stage: Any) -> list[dict[str, Any]]:
    path: Path | None = None
    if stage.family == "bsvd":
        from kinovsr.processors.bsvd import default_weights_path

        path = (
            Path(stage.config.weights_path)
            if stage.config.weights_path
            else default_weights_path(stage.config.variant)
        )
    elif stage.family == "realplksr":
        from kinovsr.processors.realplksr.net import resolve_weights

        path = resolve_weights(stage.config.weights_spec)
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"resolved checkpoint does not exist: {path}")
    return [
        {
            "basename": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    ]


def _resolved_workload(
    chain: str,
    clip: Path,
    total_frames: int,
) -> dict[str, Any]:
    from kinovsr.pipeline.builder import resolve_pipeline
    from kinovsr.pipeline.run import FileSource
    from kinovsr.processors.specs import Layout
    from kinovsr.settings import Settings

    definition = _gate_definition(chain)
    settings = Settings()
    source = FileSource(
        clip,
        layout=Layout(definition["input_layout"]),
        max_frames=total_frames,
        chunk_size=definition["endpoint_args"]["chunk_size"],
        source_color=definition["endpoint_args"]["source_color"],
        source_range=definition["endpoint_args"]["source_range"],
    )
    plan = resolve_pipeline(
        definition["config"],
        input_spec=source.spec,
        settings=settings,
    )
    stages = []
    self_buffered_delay = 0
    for stage in plan.stages:
        capability = stage.capability_spec
        if capability.temporal_mode.value == "centered":
            self_buffered_delay += capability.temporal_radius
        stages.append(
            {
                "name": stage.name,
                "position": stage.position,
                "family": stage.family,
                "capability": stage.capability.value,
                "profile": stage.profile,
                "config": _jsonable(stage.config),
                "capability_contract": {
                    "temporal_mode": capability.temporal_mode.value,
                    "temporal_radius": capability.temporal_radius,
                    "stateful": capability.stateful,
                    "is_tap": capability.is_tap,
                    "emits_boundaries": _jsonable(capability.emits_boundaries),
                    "requires_boundaries": _jsonable(capability.requires_boundaries),
                },
                "input_spec": _jsonable(stage.input_spec),
                "output_spec": _jsonable(stage.output_spec),
                "checkpoints": _checkpoint_signature(stage),
            }
        )
    return {
        "definition": definition,
        "resolved": {
            "input_spec": _jsonable(plan.input_spec),
            "output_spec": _jsonable(plan.output_spec),
            "stages": stages,
        },
        "measurement_contract": {
            "self_buffered_delay_frames": self_buffered_delay,
            "sink_holdback_frames": SINK_HOLDBACK_FRAMES,
            "required_tail_frames": self_buffered_delay + SINK_HOLDBACK_FRAMES,
        },
    }


def _validate_workload_tail(
    tail_frames: int,
    workloads: Mapping[str, Mapping[str, Any]],
) -> None:
    invalid = []
    for chain, workload in sorted(workloads.items()):
        required = int(workload["measurement_contract"]["required_tail_frames"])
        if tail_frames < required:
            invalid.append(
                f"{chain}.tail_frames must be >= resolved delay + sink holdback "
                f"({required}), got {tail_frames}"
            )
    if invalid:
        raise ValueError("; ".join(invalid))


def _comparison_signature(
    *,
    clip: Path,
    runs: int,
    warmup_frames: int,
    measured_frames: int,
    tail_frames: int,
) -> dict[str, Any]:
    import av
    import mlx.core as mx

    from kinovsr.api import resolve_mlx_cache_limit_gb
    from kinovsr.settings import Settings

    settings = Settings()
    return {
        "machine": {
            "hardware_model": _sysctl("hw.model"),
            "chip": _sysctl("machdep.cpu.brand_string"),
            "architecture": platform.machine(),
        },
        "os": {
            "name": "macOS",
            "version": platform.mac_ver()[0],
            "build": _command_value(("sw_vers", "-buildVersion")),
            "darwin_release": platform.release(),
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_build": list(platform.python_build()),
            "python_compiler": platform.python_compiler(),
            "mlx_version": mx.__version__,
            "pyav_version": av.__version__,
            "libav_versions": {
                name: list(version) for name, version in sorted(av.library_versions.items())
            },
        },
        "protocol": {
            "schema": TIMING_SCHEMA,
            "fresh_process_per_run": True,
            "runs": runs,
            "warmup_frames": warmup_frames,
            "measured_frames": measured_frames,
            "tail_frames": tail_frames,
            "sink_holdback_frames": SINK_HOLDBACK_FRAMES,
            "total_frames": _total_frames(
                warmup_frames,
                measured_frames,
                tail_frames,
            ),
            "timing_boundary": "after synchronous FileSink.append",
            "steady_interval": "after warmup output through measured output",
            "instrumentation_host_device_sync": (
                "none; CPU clock only, decoded quality probe runs after timing"
            ),
            "quality_policy": {
                "method": "decoded-rgb10-full-v4",
                "decoder": "PyAV/libav rgb48le to RGBAHalf",
                "compression": "zlib",
                "max_abs_rgb10": QUALITY_MAX_ABS_RGB10,
                "min_psnr_db": QUALITY_MIN_PSNR_DB,
            },
        },
        "clip": {
            "sha256": _sha256(clip),
            "track": _track_metadata(clip),
        },
        "cache_compile_policy": {
            "settings_source": "Settings constructor defaults",
            "mlx_cache_limit_gb": resolve_mlx_cache_limit_gb(settings),
            "clear_mlx_cache_at_endpoint_start": True,
            "fresh_process_per_run": True,
            "model_compile": True,
            "system_compilation_cache": "retained",
        },
        "power_thermal": _comparison_conditions(),
    }


def _revision_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    status_proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if commit_proc.returncode != 0 or status_proc.returncode != 0:
        raise RuntimeError("cannot read product revision from git")
    commit = commit_proc.stdout.decode("ascii").strip()
    status = status_proc.stdout
    dirty = bool(status)
    diff_sha256 = None
    if dirty:
        diff_proc = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if diff_proc.returncode != 0:
            raise RuntimeError("cannot hash the product working-tree diff")
        digest = hashlib.sha256()
        digest.update(diff_proc.stdout)
        digest.update(status)
        for record in status.split(b"\0"):
            if not record.startswith(b"?? "):
                continue
            path = root / record[3:].decode(errors="surrogateescape")
            if path.is_file():
                digest.update(record[3:])
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
        diff_sha256 = digest.hexdigest()
    return {"commit": commit, "dirty": dirty, "diff_sha256": diff_sha256}
