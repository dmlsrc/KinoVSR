"""Steady-state performance gate for the public file-processing endpoint.

One file, one job: compare the endpoint's steady-state cost against a
machine-, environment-, workload-, and clip-specific baseline recorded by
this same script, and refuse to compare when anything semantic differs.

    bench_endpoint_gates.py --clip CLIP --record-baseline   # exit 3
    bench_endpoint_gates.py --clip CLIP                     # exit 0/1
    bench_endpoint_gates.py --clip CLIP --gates pass        # subset

Exit codes: 0 pass, 1 gate failure, 2 cannot run/compare, 3 recorded.

Methodology (the parts that keep MLX/VideoToolbox numbers honest):

- every sample is one public ``process_video_file`` call in a fresh
  process; timing is CPU-clock at the synchronous ``FileSink.append``
  boundary;
- each run splits into setup/warmup, a steady measured window, and an
  unmeasured tail; only the steady median is gated;
- the worker proves the measured boundary precedes natural source
  exhaustion, so EOF flush can never fake throughput;
- the tail must cover the resolved chain's self-buffered delay plus the
  sink holdback;
- runs are rejected on power/thermal drift and recorded only in a
  nominal thermal state;
- the allowed regression is max(fraction * baseline, 3 sigma of the
  baseline's own recorded run spread) - measured noise, not an asserted
  constant;
- output behavior must match: track metadata exactly, five decoded
  RGB10 frames within max-abs/PSNR tolerances (exact hashes flap on
  encoder nondeterminism; the tolerances do not).

The baseline lives at $SHARED_TEMP_DIR/trace_analysis/ by default;
re-recording preserves the previous file as a timestamped sibling and
logs the per-gate deltas it is about to accept.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import dataclasses
import hashlib
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
import zlib
from collections.abc import Mapping, Sequence
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any

BASELINE_SCHEMA = 4
BASELINE_KIND = "kinovsr_endpoint_performance_baseline"
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
# median, 3 sigma of the baseline's own recorded run medians).
GATES = {
    "pass": ("file passthrough", 0.10),
    "learned": ("bsvd + realplksr 2x", 0.03),
}

_MISSING = object()
_log = logging.getLogger("kinovsr.dev.bench_endpoint_gates")
_result_log = logging.getLogger("kinovsr.dev.bench_endpoint_gates.result")


# --------------------------------------------------------------- helpers

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
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
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
                mismatches.append(
                    f"{path}: baseline={_display(left)}, current={_display(right)}")
            else:
                mismatches.extend(_signature_mismatches(left, right, prefix=path))
        return mismatches
    if isinstance(baseline, list) and isinstance(current, list):
        mismatches = []
        if len(baseline) != len(current):
            mismatches.append(
                f"{prefix}.length: baseline={len(baseline)}, current={len(current)}")
        for index, (left, right) in enumerate(zip(baseline, current, strict=False)):
            mismatches.extend(
                _signature_mismatches(left, right, prefix=f"{prefix}[{index}]"))
        return mismatches
    if baseline != current:
        return [f"{prefix}: baseline={baseline!r}, current={current!r}"]
    return []


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sysctl(name: str) -> str:
    proc = subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, check=False)
    return proc.stdout.strip() or "unknown"


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return value / divisor


# ------------------------------------------------------ gate definitions

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
            f"clip.sample_count must be >= {required} to reach warmup, "
            f"measured, and tail boundaries; got {sample_count}")


def _assert_frame_counts(
    chain: str,
    requested: int,
    frames_in: int,
    frames_out: int,
) -> None:
    if frames_in != requested or frames_out != requested:
        raise RuntimeError(
            f"worker {chain} requested {requested} frames but processed "
            f"frames_in={frames_in}, frames_out={frames_out}")


# ------------------------------------------------------ run conditions

def _power_source() -> str:
    proc = subprocess.run(
        ["pmset", "-g", "batt"], capture_output=True, text=True, check=False)
    first = proc.stdout.splitlines()[0] if proc.stdout.splitlines() else ""
    marker = "Now drawing from '"
    if marker in first:
        return first.split(marker, 1)[1].split("'", 1)[0]
    return first.strip() or "unknown"


def _runtime_conditions() -> dict[str, Any]:
    from kinovsr.native.frameworks import Foundation

    process = Foundation.NSProcessInfo.processInfo()
    thermal_value = int(process.thermalState())
    thermal = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}.get(
        thermal_value, f"unknown_{thermal_value}")
    return {
        "power_source": _power_source(),
        "low_power_mode": bool(process.isLowPowerModeEnabled()),
        "thermal_state": thermal,
    }


def _comparison_conditions() -> dict[str, Any]:
    current = _runtime_conditions()
    if current["thermal_state"] != "nominal":
        raise RuntimeError(
            f"benchmark requires a nominal thermal state; got "
            f"{current['thermal_state']!r}")
    return {**current, "thermal_policy": "nominal_only"}


def _assert_run_conditions(
    runs: Sequence[dict[str, Any]],
    expected: Mapping[str, Any],
) -> None:
    mismatches: list[str] = []
    for run_index, run in enumerate(runs):
        for boundary in ("conditions_start", "conditions_end"):
            actual = run[boundary]
            for key in ("power_source", "low_power_mode", "thermal_state"):
                if actual[key] != expected[key]:
                    mismatches.append(
                        f"runs[{run_index}].{boundary}.{key}: "
                        f"expected={expected[key]!r}, actual={actual[key]!r}")
    if mismatches:
        raise RuntimeError(
            "benchmark runtime conditions changed: " + "; ".join(mismatches))


# ------------------------------------------------------------- timing

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
                f"output timing observed {self.count} frames, expected "
                f"{expected_frames}")
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
        if not (start_s <= self.first_append_s <= self.warmup_end_s
                <= self.measured_end_s <= end_s):
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
                f"source timing observed {self.count} frames, expected "
                f"{expected_frames}")
        if self.exhausted_s is None:
            raise RuntimeError("source timing never observed natural exhaustion")
        if output.measured_end_s is None:
            raise RuntimeError("output timing never reached the measured boundary")
        headroom_ms = (self.exhausted_s - output.measured_end_s) * 1000.0
        if headroom_ms <= 0:
            raise RuntimeError(
                "measured output reached the sink only after source exhaustion; "
                "increase tail_frames so EOF flush is excluded")
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


# --------------------------------------------------------- output probe

def _fraction(value: Fraction | None) -> str | None:
    return None if value is None else f"{value.numerator}/{value.denominator}"


def _hevc_details(format_description: Any) -> dict[str, Any] | None:
    from kinovsr.native.frameworks import CoreMedia

    extensions = CoreMedia.CMFormatDescriptionGetExtensions(
        format_description) or {}
    atoms = extensions.get("SampleDescriptionExtensionAtoms") or {}
    configuration = atoms.get("hvcC")
    if configuration is None:
        return None
    configuration = bytes(configuration)
    if len(configuration) < 19:
        return None
    chroma_idc = configuration[16] & 0x03
    return {
        "profile_idc": configuration[1] & 0x1F,
        "chroma": {0: "monochrome", 1: "4:2:0", 2: "4:2:2", 3: "4:4:4"}[chroma_idc],
        "bit_depth_luma": 8 + (configuration[17] & 0x07),
        "bit_depth_chroma": 8 + (configuration[18] & 0x07),
    }


def _track_metadata(path: Path | str) -> dict[str, Any]:
    from kinovsr.media.video_reader import (
        _first_video_track,
        _video_codec_fourcc,
        probe_color,
        probe_video,
        probe_video_timing,
    )
    from kinovsr.native.frameworks import Foundation, av

    media_path = Path(path)
    url = Foundation.NSURL.fileURLWithPath_(str(media_path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    format_description = track.formatDescriptions()[0]
    width, height, nominal_fps, _, transform, pixel_aspect = probe_video(media_path)
    timing = probe_video_timing(media_path)
    return {
        "width": width,
        "height": height,
        "nominal_fps": nominal_fps,
        "sample_count": timing.sample_count,
        "cadence": _fraction(timing.cadence),
        "first_pts": _fraction(timing.first_pts),
        "duration": _fraction(timing.duration),
        "source_tick": _fraction(timing.source_tick),
        "codec_fourcc": _video_codec_fourcc(track),
        "hevc": _hevc_details(format_description),
        "pixel_aspect": (
            None if pixel_aspect is None
            else f"{pixel_aspect[0]}/{pixel_aspect[1]}"),
        "transform": {
            key: float(getattr(transform, key))
            for key in ("a", "b", "c", "d", "tx", "ty")
        },
        "color": _jsonable(probe_color(media_path)),
    }


def _quality_indices(
    *,
    warmup_frames: int,
    measured_frames: int,
    total_frames: int,
) -> list[int]:
    return sorted({
        0,
        warmup_frames - 1,
        warmup_frames,
        warmup_frames + measured_frames - 1,
        total_frames - 1,
    })


def _encode_rgb10_frame(frame: Any) -> str:
    import numpy as np

    raw = np.ascontiguousarray(frame, dtype="<u2").tobytes()
    return base64.b64encode(zlib.compress(raw, level=6)).decode("ascii")


def _decode_rgb10_frame(sample: Mapping[str, Any]) -> Any:
    import numpy as np

    shape = sample.get("shape")
    if (not isinstance(shape, list) or len(shape) != 3
            or any(not isinstance(v, int) or v < 1 for v in shape)
            or shape[2] != 3):
        raise ValueError(f"invalid RGB10 frame shape {shape!r}")
    encoded = sample.get("rgb10_zlib_b64")
    if not isinstance(encoded, str):
        raise ValueError("RGB10 frame payload must be base64 text")
    try:
        raw = zlib.decompress(base64.b64decode(encoded, validate=True))
    except (binascii.Error, ValueError, zlib.error) as exc:
        raise ValueError("RGB10 frame payload is not valid zlib/base64") from exc
    expected_bytes = math.prod(shape) * 2
    if len(raw) != expected_bytes:
        raise ValueError(
            f"RGB10 frame has {len(raw)} bytes, expected {expected_bytes} "
            f"for {shape}")
    return np.frombuffer(raw, dtype="<u2").reshape(shape)


def _output_probe(
    path: Path,
    *,
    warmup_frames: int,
    measured_frames: int,
    total_frames: int,
) -> dict[str, Any]:
    """Track metadata plus five decoded full RGB10 frames for comparison."""
    import numpy as np

    from kinovsr.media import ffmpeg_reader
    from kinovsr.media import pixel_buffers as pb

    indices = _quality_indices(
        warmup_frames=warmup_frames,
        measured_frames=measured_frames,
        total_frames=total_frames,
    )
    digest = hashlib.sha256()
    samples = []
    for index in indices:
        chunks = ffmpeg_reader.iter_video_buffer_chunks(
            path, pb.PIX_RGBAHALF, chunk_size=1,
            start_frame=index, end_frame=index + 1)
        try:
            buffer = next(iter(chunks))[0]
        except (StopIteration, IndexError) as exc:
            raise RuntimeError(
                f"cannot decode quality-probe frame {index}") from exc
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
        rgb = np.asarray(pb.read_buffer_rgb_f32(buffer), dtype=np.float32)
        quantized = np.rint(np.clip(rgb, 0.0, 1.0) * 1023.0).astype("<u2")
        digest.update(index.to_bytes(8, "little", signed=False))
        digest.update(np.ascontiguousarray(quantized).tobytes())
        samples.append({
            "index": index,
            "shape": list(quantized.shape),
            "rgb10_zlib_b64": _encode_rgb10_frame(quantized),
        })
    return {
        "track": _track_metadata(path),
        "quality": {
            "indices": indices,
            "diagnostic_full_frame_sha256": digest.hexdigest(),
            "samples": samples,
        },
    }


def _compare_quality(
    baseline: Any,
    current: Any,
) -> tuple[list[str], dict[str, Any]]:
    if not isinstance(baseline, Mapping) or not isinstance(current, Mapping):
        return ["output_behavior.quality: baseline and current must be objects"], {}
    mismatches = _signature_mismatches(
        baseline.get("indices"), current.get("indices"),
        prefix="output_behavior.quality.indices")
    baseline_samples = baseline.get("samples")
    current_samples = current.get("samples")
    if (not isinstance(baseline_samples, list)
            or not isinstance(current_samples, list)):
        mismatches.append("output_behavior.quality.samples: must be lists")
        return mismatches, {}
    if len(baseline_samples) != len(current_samples):
        mismatches.append(
            "output_behavior.quality.samples.length: "
            f"baseline={len(baseline_samples)}, current={len(current_samples)}")

    import numpy as np

    comparisons = []
    for position, (left, right) in enumerate(
            zip(baseline_samples, current_samples, strict=False)):
        prefix = f"output_behavior.quality.samples[{position}]"
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            mismatches.append(f"{prefix}: baseline and current must be objects")
            continue
        structural = _signature_mismatches(
            {key: left.get(key, _MISSING) for key in ("index", "shape")},
            {key: right.get(key, _MISSING) for key in ("index", "shape")},
            prefix=prefix)
        if structural:
            mismatches.extend(structural)
            continue
        try:
            left_frame = _decode_rgb10_frame(left)
            right_frame = _decode_rgb10_frame(right)
        except ValueError as exc:
            mismatches.append(f"{prefix}: {exc}")
            continue
        difference = np.abs(
            left_frame.astype(np.int32) - right_frame.astype(np.int32))
        max_abs = int(difference.max(initial=0))
        mse = float(np.mean(np.square(difference, dtype=np.float64)))
        psnr_db = None if mse == 0.0 else 20.0 * math.log10(1023.0 / math.sqrt(mse))
        sample_pass = max_abs <= QUALITY_MAX_ABS_RGB10 and (
            psnr_db is None or psnr_db >= QUALITY_MIN_PSNR_DB)
        comparisons.append({
            "index": left["index"],
            "max_abs_rgb10": max_abs,
            "psnr_db": None if psnr_db is None else round(psnr_db, 4),
            "pass": sample_pass,
        })
        if not sample_pass:
            rendered = "exact" if psnr_db is None else f"{psnr_db:.4f} dB"
            mismatches.append(
                f"{prefix}: max_abs_rgb10={max_abs} (limit "
                f"{QUALITY_MAX_ABS_RGB10}), psnr={rendered} "
                f"(minimum {QUALITY_MIN_PSNR_DB:.1f} dB)")

    metrics = {
        "policy": {
            "max_abs_rgb10": QUALITY_MAX_ABS_RGB10,
            "min_psnr_db": QUALITY_MIN_PSNR_DB,
        },
        "diagnostic_full_frame_hash_match": (
            baseline.get("diagnostic_full_frame_sha256")
            == current.get("diagnostic_full_frame_sha256")),
        "samples": comparisons,
    }
    return mismatches, metrics


def _compare_output_behavior(
    baseline: Any,
    current: Any,
) -> tuple[list[str], dict[str, Any]]:
    if not isinstance(baseline, Mapping) or not isinstance(current, Mapping):
        return ["output_behavior: baseline and current must be objects"], {}
    mismatches = _signature_mismatches(
        {key: value for key, value in baseline.items() if key != "quality"},
        {key: value for key, value in current.items() if key != "quality"},
        prefix="output_behavior")
    quality_mismatches, quality_metrics = _compare_quality(
        baseline.get("quality"), current.get("quality"))
    mismatches.extend(quality_mismatches)
    return mismatches, quality_metrics


def _compare_behavior_runs(
    reference: Mapping[str, Any],
    output_behaviors: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not output_behaviors:
        raise ValueError("cannot compare zero per-run output behaviors")
    all_mismatches = []
    run_results = []
    for index, output_behavior in enumerate(output_behaviors, start=1):
        mismatches, quality_comparison = _compare_output_behavior(
            reference, output_behavior)
        run_results.append({
            "run": index,
            "pass": not mismatches,
            "mismatches": mismatches,
            "quality_comparison": quality_comparison,
        })
        all_mismatches.extend(
            f"runs[{index - 1}].{mismatch}" for mismatch in mismatches)
    return all_mismatches, run_results


# ----------------------------------------------------------- signatures

def _checkpoint_signature(stage: Any) -> list[dict[str, Any]]:
    path: Path | None = None
    if stage.family == "bsvd":
        from kinovsr.processors.bsvd import default_weights_path

        path = (Path(stage.config.weights_path) if stage.config.weights_path
                else default_weights_path(stage.config.variant))
    elif stage.family == "realplksr":
        from kinovsr.processors.realplksr.net import resolve_weights

        path = resolve_weights(stage.config.weights_spec)
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"resolved checkpoint does not exist: {path}")
    return [{
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }]


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
    source = FileSource(
        clip,
        layout=Layout(definition["input_layout"]),
        max_frames=total_frames,
        chunk_size=definition["endpoint_args"]["chunk_size"],
        source_color=definition["endpoint_args"]["source_color"],
        source_range=definition["endpoint_args"]["source_range"],
    )
    plan = resolve_pipeline(
        definition["config"], input_spec=source.spec, settings=Settings())
    stages = []
    self_buffered_delay = 0
    for stage in plan.stages:
        capability = stage.capability_spec
        if capability.temporal_mode.value == "centered":
            self_buffered_delay += capability.temporal_radius
        stages.append({
            "name": stage.name,
            "family": stage.family,
            "capability": stage.capability.value,
            "profile": stage.profile,
            "config": _jsonable(stage.config),
            "checkpoints": _checkpoint_signature(stage),
        })
    return {
        "definition": definition,
        "stages": stages,
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
                f"{chain}.tail_frames must be >= resolved delay + sink "
                f"holdback ({required}), got {tail_frames}")
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
    """The compared identity: semantic fields only.

    Every key here forces a re-record when it changes, so each must earn
    its place: hardware model, macOS version, the three runtime versions
    that own the numbers (python, mlx, pyav), the numeric protocol, the
    clip identity, the cache policy, and the power/thermal envelope.
    """
    import av
    import mlx.core as mx

    from kinovsr.api import resolve_mlx_cache_limit_gb
    from kinovsr.settings import Settings

    return {
        "machine": {
            "hardware_model": _sysctl("hw.model"),
            "chip": _sysctl("machdep.cpu.brand_string"),
            "architecture": platform.machine(),
        },
        "os": {"macos_version": platform.mac_ver()[0]},
        "runtime": {
            "python_version": platform.python_version(),
            "mlx_version": mx.__version__,
            "pyav_version": av.__version__,
        },
        "protocol": {
            "schema": BASELINE_SCHEMA,
            "fresh_process_per_run": True,
            "runs": runs,
            "warmup_frames": warmup_frames,
            "measured_frames": measured_frames,
            "tail_frames": tail_frames,
            "sink_holdback_frames": SINK_HOLDBACK_FRAMES,
            "total_frames": _total_frames(
                warmup_frames, measured_frames, tail_frames),
            "quality": {
                "max_abs_rgb10": QUALITY_MAX_ABS_RGB10,
                "min_psnr_db": QUALITY_MIN_PSNR_DB,
            },
        },
        "clip": {
            "sha256": _sha256(clip),
            "track": _track_metadata(clip),
        },
        "cache_compile_policy": {
            "mlx_cache_limit_gb": resolve_mlx_cache_limit_gb(Settings()),
            "clear_mlx_cache_at_endpoint_start": True,
            "model_compile": True,
        },
        "power_thermal": _comparison_conditions(),
    }


def _revision_state() -> dict[str, Any]:
    """Recorded provenance (never compared): commit plus a dirty flag."""
    root = Path(__file__).resolve().parents[2]
    commit_proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=False)
    status_proc = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root,
        capture_output=True, check=False)
    if commit_proc.returncode != 0 or status_proc.returncode != 0:
        raise RuntimeError("cannot read product revision from git")
    return {
        "commit": commit_proc.stdout.decode("ascii").strip(),
        "dirty": bool(status_proc.stdout.strip()),
    }


# ------------------------------------------------------------- baseline

def _default_baseline_path() -> Path:
    from kinovsr.settings import Settings

    return (Settings.from_env().shared_temp_dir / "trace_analysis"
            / BASELINE_FILENAME)


def _validate_baseline(
    baseline: Any,
    *,
    fingerprint: dict[str, Any],
    workloads: Mapping[str, dict[str, Any]],
    selected: set[str],
) -> None:
    if not isinstance(baseline, Mapping):
        raise ValueError("baseline must be a JSON object")
    mismatches: list[str] = []
    if baseline.get("kind") != BASELINE_KIND:
        mismatches.append(
            f"kind: baseline={baseline.get('kind')!r}, current={BASELINE_KIND!r}")
    if baseline.get("schema") != BASELINE_SCHEMA:
        mismatches.append(
            f"schema: baseline={baseline.get('schema')!r}, "
            f"current={BASELINE_SCHEMA!r}")
    mismatches.extend(_signature_mismatches(
        baseline.get("fingerprint", _MISSING), fingerprint,
        prefix="fingerprint"))
    gates = baseline.get("gates")
    if not isinstance(gates, Mapping):
        mismatches.append("gates: baseline must contain an object")
        gates = {}
    for chain in sorted(selected):
        entry = gates.get(chain)
        if not isinstance(entry, Mapping):
            mismatches.append(f"gates.{chain}: missing baseline gate")
            continue
        for field in ("workload", "measurement", "output_behavior"):
            if field not in entry:
                mismatches.append(f"gates.{chain}.{field}: missing baseline field")
        if "workload" in entry:
            mismatches.extend(_signature_mismatches(
                entry["workload"], workloads[chain],
                prefix=f"gates.{chain}.workload"))
        # The recorded file is this script's own output; shape-check only
        # the gated number and the run count.
        measurement = entry.get("measurement")
        if isinstance(measurement, Mapping):
            runs = measurement.get("runs")
            expected_runs = fingerprint["protocol"]["runs"]
            if not isinstance(runs, list) or len(runs) != expected_runs:
                mismatches.append(
                    f"gates.{chain}.measurement.runs: expected "
                    f"{expected_runs} runs")
            gated = (measurement.get("median") or {}).get("steady_ms_per_frame")
            if (not isinstance(gated, (int, float)) or isinstance(gated, bool)
                    or not math.isfinite(gated) or gated <= 0):
                mismatches.append(
                    f"gates.{chain}.measurement.median.steady_ms_per_frame: "
                    f"must be a finite positive number, got {gated!r}")
    if mismatches:
        raise ValueError(
            "baseline does not match this run: " + "; ".join(mismatches))


def _baseline_noise_ms(baseline_measurement: Mapping[str, Any]) -> float:
    """Measured run-to-run spread of the recorded baseline, as a margin floor.

    3 sigma of the baseline's own steady per-run medians; 0 with fewer
    than three usable runs (the fractional term still applies).
    """
    runs = baseline_measurement.get("runs") or []
    values = [
        float(run["steady_ms_per_frame"])
        for run in runs
        if run.get("steady_ms_per_frame") is not None
    ]
    if len(values) < 3:
        return 0.0
    return 3.0 * statistics.stdev(values)


def _evaluate_gate(
    measurement: dict[str, Any],
    output_behaviors: list[dict[str, Any]],
    baseline: Mapping[str, Any],
    *,
    fraction: float,
) -> dict[str, Any]:
    if len(output_behaviors) != len(measurement["runs"]):
        raise ValueError(
            f"got {len(output_behaviors)} output probes for "
            f"{len(measurement['runs'])} timing runs")
    baseline_ms = float(baseline["measurement"]["median"]["steady_ms_per_frame"])
    current_ms = float(measurement["median"]["steady_ms_per_frame"])
    margin = max(
        _baseline_noise_ms(baseline["measurement"]), fraction * baseline_ms)
    delta = current_ms - baseline_ms
    timing_pass = delta <= margin
    behavior_mismatches, behavior_runs = _compare_behavior_runs(
        baseline["output_behavior"], output_behaviors)
    behavior_pass = not behavior_mismatches
    return {
        "baseline_steady_ms_per_frame": round(baseline_ms, 3),
        "current_steady_ms_per_frame": round(current_ms, 3),
        "delta_ms": round(delta, 3),
        "delta_pct": round(100.0 * delta / baseline_ms, 2),
        "allowed_margin_ms": round(margin, 3),
        "timing_pass": timing_pass,
        "behavior_pass": behavior_pass,
        "behavior_mismatches": behavior_mismatches,
        "quality_comparison": behavior_runs[-1]["quality_comparison"],
        "behavior_runs": behavior_runs,
        "pass": timing_pass and behavior_pass,
    }


# --------------------------------------------------------------- runner

def _worker(args: argparse.Namespace) -> None:
    import mlx.core as mx

    if not hasattr(mx, "get_peak_memory"):
        raise RuntimeError(
            "this MLX runtime lacks mx.get_peak_memory; the endpoint gate "
            "requires it to record peak_mlx_mib")

    from kinovsr.api import process_video_file
    from kinovsr.pipeline.run import FileSink, FileSource
    from kinovsr.processors.specs import Layout
    from kinovsr.settings import Settings

    definition = _gate_definition(args.chain)
    total_frames = _total_frames(
        args.warmup_frames, args.measured_frames, args.tail_frames)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = _OutputTiming(args.warmup_frames, args.measured_frames)
    source_timer = _SourceTiming()

    original_append = FileSink.append
    original_units = FileSource.units

    def timed_append(sink: FileSink, unit: Any) -> None:
        original_append(sink, unit)
        timer.record_append(time.perf_counter())

    def timed_units(source: FileSource) -> Any:
        for unit in original_units(source):
            source_timer.record_unit()
            yield unit
        source_timer.record_exhausted(time.perf_counter())

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    conditions_start = _runtime_conditions()
    FileSink.append = timed_append
    FileSource.units = timed_units
    started = time.perf_counter()
    try:
        endpoint = definition["endpoint_args"]
        result = process_video_file(
            definition["config"],
            video=args.clip,
            output=out_dir / "current.mp4",
            settings=Settings(),
            layout=Layout(definition["input_layout"]),
            max_frames=total_frames,
            audio=endpoint["audio"],
            quality=endpoint["quality"],
            chunk_size=endpoint["chunk_size"],
            source_color=endpoint["source_color"],
            source_range=endpoint["source_range"],
        )
    finally:
        FileSink.append = original_append
        FileSource.units = original_units
    finished = time.perf_counter()
    conditions_end = _runtime_conditions()

    _assert_frame_counts(
        args.chain, total_frames, result.frames_in, result.frames_out)
    metrics = timer.metrics(
        start_s=started, end_s=finished, expected_frames=total_frames)
    metrics.update({
        "frames_in": result.frames_in,
        "peak_rss_mib": _peak_rss_mib(),
        "peak_mlx_mib": float(mx.get_peak_memory()) / (1024.0 * 1024.0),
        "measured_before_source_exhaustion_ms": (
            source_timer.measured_headroom_ms(
                timer, expected_frames=total_frames)),
        "conditions_start": conditions_start,
        "conditions_end": conditions_end,
        "path": str(result.post_path),
    })
    _result_log.info("RESULT %s", json.dumps(metrics, sort_keys=True))


def _run_once(
    clip: str,
    chain: str,
    warmup_frames: int,
    measured_frames: int,
    tail_frames: int,
    keep_dir: Path | None = None,
) -> dict[str, Any]:
    total_frames = _total_frames(warmup_frames, measured_frames, tail_frames)
    wall_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        out = keep_dir if keep_dir is not None else Path(scratch)
        proc = subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()),
                "--worker-chain", chain,
                "--clip", clip,
                "--warmup-frames", str(warmup_frames),
                "--measured-frames", str(measured_frames),
                "--tail-frames", str(tail_frames),
                "--out", str(out),
            ],
            capture_output=True, text=True, timeout=1800, check=False)
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                if proc.returncode != 0:
                    break
                result = json.loads(line.removeprefix("RESULT "))
                _log.info(
                    "%s %s frames: steady %.3f ms/frame, total %.1fs (wall %.1fs)",
                    chain, total_frames, result["steady_ms_per_frame"],
                    result["total_ms"] / 1000.0,
                    time.perf_counter() - wall_start)
                return result
        raise RuntimeError(
            f"worker {chain}/{total_frames} exited {proc.returncode} without "
            f"a result:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")


def _measure(
    clip: str,
    chain: str,
    warmup_frames: int,
    measured_frames: int,
    tail_frames: int,
    runs: int,
    keep_dir: Path,
    expected_conditions: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    if chain not in GATES:
        raise ValueError(f"unknown gate {chain!r}")
    samples: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for index in range(runs):
        result = _run_once(
            clip, chain, warmup_frames, measured_frames, tail_frames,
            keep_dir=keep_dir / f"run-{index + 1:02d}")
        samples.append(result)
        outputs.append(Path(result["path"]))
    _assert_run_conditions(samples, expected_conditions)
    return _summarize_runs(samples), outputs


# ----------------------------------------------------------------- main

def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _result_exit_code(*, recording: bool, failures: Sequence[str]) -> int:
    if recording:
        return RECORD_ONLY_EXIT
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES)
    parser.add_argument(
        "--measured-frames", type=int, default=DEFAULT_MEASURED_FRAMES)
    parser.add_argument("--tail-frames", type=int, default=DEFAULT_TAIL_FRAMES)
    parser.add_argument(
        "--baseline",
        help=("baseline JSON path; defaults to "
              "$SHARED_TEMP_DIR/trace_analysis/" + BASELINE_FILENAME))
    parser.add_argument(
        "--record-baseline", action="store_true",
        help="record a non-gating baseline; successful recording exits 3")
    parser.add_argument(
        "--report",
        help="directory for the current-run endpoint_gates_report.json")
    parser.add_argument("--gates", default="pass,learned",
                        help="comma list of gates")
    parser.add_argument("--worker-chain", choices=tuple(GATES),
                        help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        _validate_protocol(
            args.runs, args.warmup_frames, args.measured_frames,
            args.tail_frames)
    except ValueError as exc:
        parser.error(str(exc))

    if args.worker_chain:
        from kinovsr.ui.logging import configure_machine_output

        configure_machine_output(_result_log.name)
        _worker(argparse.Namespace(
            chain=args.worker_chain,
            clip=args.clip,
            warmup_frames=args.warmup_frames,
            measured_frames=args.measured_frames,
            tail_frames=args.tail_frames,
            out=args.out,
        ))
        return 0

    selected = {value.strip() for value in args.gates.split(",") if value.strip()}
    unknown = sorted(selected - set(GATES))
    if unknown:
        parser.error(f"unknown gates: {unknown}")
    if not selected:
        parser.error("select at least one gate")
    if args.record_baseline and selected != set(GATES):
        parser.error("recording a baseline requires both pass and learned gates")

    from kinovsr.ui.logging import configure_logging

    configure_logging()
    clip = Path(args.clip)
    total_frames = _total_frames(
        args.warmup_frames, args.measured_frames, args.tail_frames)
    try:
        fingerprint = _comparison_signature(
            clip=clip, runs=args.runs,
            warmup_frames=args.warmup_frames,
            measured_frames=args.measured_frames,
            tail_frames=args.tail_frames)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _log.error("cannot fingerprint benchmark environment: %s", exc)
        return 2
    try:
        _require_clip_frames(
            fingerprint["clip"]["track"]["sample_count"], total_frames)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        workloads = {
            chain: _resolved_workload(chain, clip, total_frames)
            for chain in sorted(selected)
        }
        _validate_workload_tail(args.tail_frames, workloads)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _log.error("cannot resolve benchmark workloads: %s", exc)
        return 2
    try:
        revision = _revision_state()
    except RuntimeError as exc:
        _log.error("cannot record product revision: %s", exc)
        return 2

    baseline_path = Path(args.baseline) if args.baseline else _default_baseline_path()
    baseline: dict[str, Any] | None = None
    if not args.record_baseline:
        try:
            baseline = json.loads(baseline_path.read_text())
            _validate_baseline(
                baseline, fingerprint=fingerprint, workloads=workloads,
                selected=selected)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _log.error(
                "cannot use endpoint baseline %s: %s; record it with "
                "--record-baseline", baseline_path, exc)
            return 2

    _log.info(
        "product revisions: baseline=%s current=%s",
        None if baseline is None else baseline.get("product_revision"),
        revision)
    report: dict[str, Any] = {
        "schema": BASELINE_SCHEMA,
        "mode": "record" if args.record_baseline else "gate",
        "pass": None,
        "clip_path": str(clip),
        "baseline_path": str(baseline_path),
        "current_product_revision": revision,
        "baseline_product_revision": (
            None if baseline is None else baseline.get("product_revision")),
        "fingerprint": fingerprint,
        "workloads": dict(workloads),
        "gates": {},
    }
    recorded: dict[str, Any] = {
        "schema": BASELINE_SCHEMA,
        "kind": BASELINE_KIND,
        "product_revision": revision,
        "fingerprint": fingerprint,
        "gates": {},
    }
    failures: list[str] = []
    expected_conditions = fingerprint["power_thermal"]

    for chain, (label, fraction) in GATES.items():
        if chain not in selected:
            continue
        try:
            with tempfile.TemporaryDirectory(prefix=f"gate_{chain}_") as keep_raw:
                measurement, outputs = _measure(
                    str(clip), chain,
                    args.warmup_frames, args.measured_frames,
                    args.tail_frames, args.runs,
                    keep_dir=Path(keep_raw),
                    expected_conditions=expected_conditions)
                output_behaviors = [
                    _output_probe(
                        output,
                        warmup_frames=args.warmup_frames,
                        measured_frames=args.measured_frames,
                        total_frames=total_frames)
                    for output in outputs
                ]
        except RuntimeError as exc:
            # Worker crashes, run-count mismatches, and thermal/power drift
            # are "cannot run" (exit 2), not a gate FAIL (exit 1).
            _log.error("cannot run %s gate: %s", chain, exc)
            return 2

        current_ms = float(measurement["median"]["steady_ms_per_frame"])
        if args.record_baseline:
            canonical = output_behaviors[-1]
            behavior_mismatches, _ = _compare_behavior_runs(
                canonical, output_behaviors)
            if behavior_mismatches:
                _log.error(
                    "cannot record inconsistent per-run output behavior: %s",
                    "; ".join(behavior_mismatches))
                return 2
            entry = {
                "label": label,
                "workload": workloads[chain],
                "measurement": measurement,
                "output_behavior": canonical,
            }
            recorded["gates"][chain] = entry
            report["gates"][chain] = entry
            _log.info(
                "[RECORDED, NOT EVALUATED] %s: %.3f ms/frame", label, current_ms)
            continue

        assert baseline is not None
        entry = _evaluate_gate(
            measurement, output_behaviors, baseline["gates"][chain],
            fraction=fraction)
        entry["label"] = label
        report["gates"][chain] = entry
        status = "PASS" if entry["pass"] else "FAIL"
        if not entry["pass"]:
            failures.append(chain)
        log = _log.info if entry["pass"] else _log.error
        log(
            "[%s] %s: baseline %.3f ms/frame, current %.3f, delta %+.3f "
            "(margin %.3f); output=%s",
            status, label,
            entry["baseline_steady_ms_per_frame"],
            entry["current_steady_ms_per_frame"],
            entry["delta_ms"], entry["allowed_margin_ms"],
            "MATCH" if entry["behavior_pass"] else "MISMATCH")

    if args.record_baseline:
        if baseline_path.exists():
            # Re-recording must not silently absorb a regression: keep the
            # old file and log the per-gate deltas being accepted.
            stamp = time.strftime("%Y%m%dT%H%M%S")
            backup_path = baseline_path.with_name(
                f"{baseline_path.stem}.pre-{stamp}{baseline_path.suffix}")
            try:
                previous = json.loads(baseline_path.read_text())
            except (OSError, ValueError):
                previous = None
            baseline_path.rename(backup_path)
            _log.info("previous baseline kept at %s", backup_path)
            if isinstance(previous, dict):
                for chain, entry in recorded["gates"].items():
                    old_entry = (previous.get("gates") or {}).get(chain)
                    if not isinstance(old_entry, dict):
                        continue
                    try:
                        old_ms = float(old_entry["measurement"]["median"]
                                       ["steady_ms_per_frame"])
                        new_ms = float(entry["measurement"]["median"]
                                       ["steady_ms_per_frame"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    _log.info(
                        "re-record %s: %.3f -> %.3f ms/frame (%+.1f%%)",
                        chain, old_ms, new_ms,
                        100.0 * (new_ms - old_ms) / old_ms if old_ms else 0.0)
        _write_json(baseline_path, recorded)
        report["status"] = "recorded_not_evaluated"
        _log.info("baseline recorded but not evaluated: %s", baseline_path)
    else:
        report["pass"] = not failures
        report["status"] = "pass" if not failures else "fail"
    if args.report:
        report_path = Path(args.report) / "endpoint_gates_report.json"
        _write_json(report_path, report)
        _log.info("report: %s", report_path)
    return _result_exit_code(recording=args.record_baseline, failures=failures)


if __name__ == "__main__":
    sys.exit(main())
