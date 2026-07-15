"""Pure protocol values and validation for the endpoint performance gate."""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Mapping, Sequence
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

BASELINE_SCHEMA = 3
BASELINE_KIND = "kinovsr_endpoint_performance_baseline"
WORKLOAD_SCHEMA = 1
TIMING_SCHEMA = 1
BASELINE_FILENAME = "kinovsr_endpoint_baseline.json"
RECORD_ONLY_EXIT = 3

MIN_RUNS = 3
MIN_WARMUP_FRAMES = 30
MIN_MEASURED_FRAMES = 120
MIN_TAIL_FRAMES = 17
DEFAULT_RUNS = 4
DEFAULT_WARMUP_FRAMES = 30
DEFAULT_MEASURED_FRAMES = 120
DEFAULT_TAIL_FRAMES = 32
SINK_HOLDBACK_FRAMES = 1
QUALITY_MAX_ABS_RGB10 = 2
QUALITY_MIN_PSNR_DB = 60.0

# (label, fraction): the allowed regression is max(fraction * baseline
# median, 3 sigma of the baseline's own recorded run medians).  The old
# fixed ms/frame floors (2.0 / 5.0) predate the steady-state protocol and
# exceeded the entire passthrough steady baseline, so the gate could bless
# a ~2x plumbing regression; the noise term is measured, not asserted.
GATES = {
    "pass": ("file passthrough", 0.10),
    "learned": ("bsvd + realplksr 2x", 0.03),
}

_MISSING = object()


def _jsonable(value: Any) -> Any:
    """Turn resolved contracts into stable, JSON-safe signature values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    raise TypeError(f"cannot fingerprint {type(value).__name__}")


def _gate_definition(chain: str) -> dict[str, Any]:
    if chain == "pass":
        config: dict[str, Any] = {"pipeline": []}
        layout = "cv_rgba_half"
    elif chain == "learned":
        config = {
            "pipeline": ["den", "up"],
            "den": {
                "processor": "bsvd",
                "profile": "c64",
                "strength": 0.5,
                "dtype": "float16",
            },
            "up": {
                "processor": "realplksr",
                "profile": "public2x",
                "dtype": "float16",
            },
        }
        layout = "mlx_rgb_hwc"
    else:
        raise ValueError(f"unknown gate {chain!r}")

    return {
        "schema": WORKLOAD_SCHEMA,
        "public_api": "kinovsr.api.process_video_file",
        "chain": chain,
        "config": config,
        "input_layout": layout,
        "endpoint_args": {
            "audio": False,
            "chunk_size": 8,
            "encode_chroma": "auto",
            "quality": 0.65,
            "source_color": "auto",
            "source_range": "auto",
        },
        "output_encoder": {
            "codec": "hevc_videotoolbox",
            "container": "mp4",
            "profile": "HEVC_Main42210_AutoLevel",
            "chroma": "4:2:2",
            "selection": "auto resolved from output layout",
            "quality": 0.65,
            "audio": False,
        },
    }


def _total_frames(warmup_frames: int, measured_frames: int, tail_frames: int) -> int:
    return warmup_frames + measured_frames + tail_frames


def _validate_protocol(
    runs: int,
    warmup_frames: int,
    measured_frames: int,
    tail_frames: int,
) -> None:
    minimums = {
        "runs": (runs, MIN_RUNS),
        "warmup_frames": (warmup_frames, MIN_WARMUP_FRAMES),
        "measured_frames": (measured_frames, MIN_MEASURED_FRAMES),
        "tail_frames": (tail_frames, MIN_TAIL_FRAMES),
    }
    invalid = [
        f"{field} must be >= {minimum}, got {value}"
        for field, (value, minimum) in minimums.items()
        if value < minimum
    ]
    if invalid:
        raise ValueError("; ".join(invalid))


def _require_clip_frames(sample_count: int, required: int) -> None:
    if sample_count < required:
        raise ValueError(
            f"clip.sample_count must be >= {required} to reach warmup, measured, "
            f"and tail boundaries; got {sample_count}"
        )


def _assert_frame_counts(
    chain: str,
    requested: int,
    frames_in: int,
    frames_out: int,
) -> None:
    if frames_in != requested or frames_out != requested:
        raise RuntimeError(
            f"worker {chain} requested {requested} frames but processed "
            f"frames_in={frames_in}, frames_out={frames_out}"
        )


@dataclasses.dataclass
class _OutputTiming:
    """Output-boundary recorder; timestamps are injected for deterministic tests."""

    warmup_frames: int
    measured_frames: int
    count: int = 0
    first_append_s: float | None = None
    warmup_end_s: float | None = None
    measured_end_s: float | None = None

    def record_append(self, timestamp_s: float) -> None:
        self.count += 1
        if self.count == 1:
            self.first_append_s = timestamp_s
        if self.count == self.warmup_frames:
            self.warmup_end_s = timestamp_s
        if self.count == self.warmup_frames + self.measured_frames:
            self.measured_end_s = timestamp_s

    def metrics(
        self,
        *,
        start_s: float,
        end_s: float,
        expected_frames: int,
    ) -> dict[str, float | int]:
        if self.count != expected_frames:
            raise RuntimeError(
                f"output timing observed {self.count} frames, expected {expected_frames}"
            )
        missing = [
            name
            for name, value in (
                ("first_append", self.first_append_s),
                ("warmup_end", self.warmup_end_s),
                ("measured_end", self.measured_end_s),
            )
            if value is None
        ]
        if missing:
            raise RuntimeError(f"output timing did not reach boundaries: {missing}")
        assert self.first_append_s is not None
        assert self.warmup_end_s is not None
        assert self.measured_end_s is not None
        if not (
            start_s <= self.first_append_s <= self.warmup_end_s <= self.measured_end_s <= end_s
        ):
            raise RuntimeError("output timing boundaries are not monotonic")

        steady_s = self.measured_end_s - self.warmup_end_s
        return {
            "frames_written": self.count,
            "setup_compile_ms": (self.first_append_s - start_s) * 1000.0,
            "warmup_ms": (self.warmup_end_s - self.first_append_s) * 1000.0,
            "steady_total_ms": steady_s * 1000.0,
            "steady_ms_per_frame": steady_s * 1000.0 / self.measured_frames,
            "tail_finalize_ms": (end_s - self.measured_end_s) * 1000.0,
            "total_ms": (end_s - start_s) * 1000.0,
        }


@dataclasses.dataclass
class _SourceTiming:
    count: int = 0
    exhausted_s: float | None = None

    def record_unit(self) -> None:
        self.count += 1

    def record_exhausted(self, timestamp_s: float) -> None:
        self.exhausted_s = timestamp_s

    def measured_headroom_ms(
        self,
        output: _OutputTiming,
        *,
        expected_frames: int,
    ) -> float:
        if self.count != expected_frames:
            raise RuntimeError(
                f"source timing observed {self.count} frames, expected {expected_frames}"
            )
        if self.exhausted_s is None:
            raise RuntimeError("source timing never observed natural exhaustion")
        if output.measured_end_s is None:
            raise RuntimeError("output timing never reached the measured boundary")
        headroom_ms = (self.exhausted_s - output.measured_end_s) * 1000.0
        if headroom_ms <= 0:
            raise RuntimeError(
                "measured output reached the sink only after source exhaustion; "
                "increase tail_frames so EOF flush is excluded"
            )
        return headroom_ms


_MEDIAN_FIELDS = (
    "setup_compile_ms",
    "warmup_ms",
    "steady_total_ms",
    "steady_ms_per_frame",
    "tail_finalize_ms",
    "total_ms",
    "peak_rss_mib",
    "peak_mlx_mib",
    "measured_before_source_exhaustion_ms",
)


def _summarize_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("cannot summarize zero runs")
    medians: dict[str, float | None] = {}
    for field in _MEDIAN_FIELDS:
        values = [float(run[field]) for run in runs if run.get(field) is not None]
        medians[field] = statistics.median(values) if values else None
    reports = []
    for index, run in enumerate(runs, start=1):
        report = {key: value for key, value in run.items() if key != "path"}
        report["run"] = index
        reports.append(report)
    return {"median": medians, "runs": reports}


def _display(value: Any) -> str:
    return "<missing>" if value is _MISSING else repr(value)


def _signature_mismatches(
    baseline: Any,
    current: Any,
    *,
    prefix: str = "fingerprint",
) -> list[str]:
    """Return every mismatch with its exact nested field path."""
    if isinstance(baseline, Mapping) and isinstance(current, Mapping):
        mismatches: list[str] = []
        for key in sorted(set(baseline) | set(current), key=str):
            left = baseline.get(key, _MISSING)
            right = current.get(key, _MISSING)
            path = f"{prefix}.{key}"
            if left is _MISSING or right is _MISSING:
                mismatches.append(f"{path}: baseline={_display(left)}, current={_display(right)}")
            else:
                mismatches.extend(_signature_mismatches(left, right, prefix=path))
        return mismatches
    if isinstance(baseline, list) and isinstance(current, list):
        mismatches = []
        if len(baseline) != len(current):
            mismatches.append(f"{prefix}.length: baseline={len(baseline)}, current={len(current)}")
        for index, (left, right) in enumerate(zip(baseline, current, strict=False)):
            mismatches.extend(_signature_mismatches(left, right, prefix=f"{prefix}[{index}]"))
        return mismatches
    if baseline != current:
        return [f"{prefix}: baseline={baseline!r}, current={current!r}"]
    return []


def _result_exit_code(*, recording: bool, failures: Sequence[str]) -> int:
    if recording:
        return RECORD_ONLY_EXIT
    return 1 if failures else 0
