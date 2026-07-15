"""Steady-state performance gates for the public file-processing endpoint.

The post-M6 product has one public file orchestration path, so the live gate
compares that path with a machine-, environment-, workload-, and clip-specific
baseline recorded by this script. A comparison is valid only when the complete
signature matches; product revisions are separate provenance because a revision
change is the thing this gate evaluates.

Each sample is one public ``process_video_file`` call in a fresh process. It has
at least 30 warmup outputs, 120 measured outputs, and an unmeasured tail. The
tail is checked against the resolved self-buffered temporal delay plus the file
sink holdback, and the worker proves the measured boundary precedes natural
source exhaustion. Setup/compile, warmup, steady state, tail/finalization, RSS,
and MLX peaks are separate results. Only median steady-state time is gated.

Output metadata must match exactly. Five selected full decoded RGB10 frames are
retained in compressed form and compared with bounded max-error and PSNR
tolerances; their exact hashes remain diagnostic. Baseline recording is not a
passing gate: a successful recording returns exit status 3.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import endpoint_gate_environment as _environment  # noqa: F401
from endpoint_gate_baseline import _validate_baseline
from endpoint_gate_environment import (
    _assert_run_conditions as _assert_run_conditions,
)
from endpoint_gate_environment import (
    _comparison_conditions as _comparison_conditions,
)
from endpoint_gate_environment import (
    _comparison_signature,
    _resolved_workload,
    _revision_state,
    _validate_workload_tail,
)
from endpoint_gate_protocol import (
    BASELINE_FILENAME,
    BASELINE_KIND,
    BASELINE_SCHEMA,
    DEFAULT_MEASURED_FRAMES,
    DEFAULT_RUNS,
    DEFAULT_TAIL_FRAMES,
    DEFAULT_WARMUP_FRAMES,
    GATES,
    _require_clip_frames,
    _result_exit_code,
    _total_frames,
    _validate_protocol,
)
from endpoint_gate_protocol import (
    RECORD_ONLY_EXIT as RECORD_ONLY_EXIT,
)
from endpoint_gate_protocol import (
    _assert_frame_counts as _assert_frame_counts,
)
from endpoint_gate_protocol import _gate_definition as _gate_definition
from endpoint_gate_protocol import _OutputTiming as _OutputTiming
from endpoint_gate_protocol import _SourceTiming as _SourceTiming
from endpoint_gate_protocol import _summarize_runs as _summarize_runs
from endpoint_gate_quality import (
    _compare_output_behavior,
    _output_probe,
)
from endpoint_gate_quality import (
    _encode_rgb10_frame as _encode_rgb10_frame,
)
from endpoint_gate_quality import (
    _quality_indices as _quality_indices,
)
from endpoint_gate_runner import _measure, _worker

_log = logging.getLogger("kinovsr.dev.bench_endpoint_gates")
_result_log = logging.getLogger("kinovsr.dev.bench_endpoint_gates.result")


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
            reference,
            output_behavior,
        )
        run_results.append(
            {
                "run": index,
                "pass": not mismatches,
                "mismatches": mismatches,
                "quality_comparison": quality_comparison,
            }
        )
        all_mismatches.extend(f"runs[{index - 1}].{mismatch}" for mismatch in mismatches)
    return all_mismatches, run_results


def _baseline_noise_ms(baseline_measurement: Mapping[str, Any]) -> float:
    """Measured run-to-run spread of the recorded baseline, as a margin floor.

    3 sigma of the baseline's own steady per-run medians.  Falls back to 0
    when the baseline predates per-run retention or has fewer than three
    usable runs (the fractional term still applies).
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
            f"got {len(output_behaviors)} output probes for {len(measurement['runs'])} timing runs"
        )
    baseline_ms = float(baseline["measurement"]["median"]["steady_ms_per_frame"])
    current_ms = float(measurement["median"]["steady_ms_per_frame"])
    margin = max(_baseline_noise_ms(baseline["measurement"]), fraction * baseline_ms)
    delta = current_ms - baseline_ms
    timing_pass = delta <= margin
    behavior_mismatches, behavior_runs = _compare_behavior_runs(
        baseline["output_behavior"],
        output_behaviors,
    )
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
        "baseline_measurement": baseline["measurement"],
        "current_measurement": measurement,
        "baseline_output_behavior": baseline["output_behavior"],
        "current_output_behavior": output_behaviors[-1],
        "pass": timing_pass and behavior_pass,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _report_header(
    *,
    recording: bool,
    clip: Path,
    baseline_path: Path,
    current_revision: dict[str, Any],
    baseline: Mapping[str, Any] | None,
    fingerprint: dict[str, Any],
    workloads: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": BASELINE_SCHEMA,
        "mode": "record" if recording else "gate",
        "status": "recording" if recording else "running",
        "pass": None,
        "clip_path": str(clip),
        "baseline_path": str(baseline_path),
        "current_product_revision": current_revision,
        "baseline_product_revision": (
            None if baseline is None else baseline.get("product_revision")
        ),
        "fingerprint": fingerprint,
        "workloads": dict(workloads),
        "gates": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--measured-frames", type=int, default=DEFAULT_MEASURED_FRAMES)
    parser.add_argument("--tail-frames", type=int, default=DEFAULT_TAIL_FRAMES)
    parser.add_argument(
        "--baseline",
        help=(
            "baseline JSON path; defaults to "
            "$SHARED_TEMP_DIR/trace_analysis/kinovsr_endpoint_baseline.json"
        ),
    )
    parser.add_argument(
        "--record-baseline",
        action="store_true",
        help="record a non-gating baseline; successful recording exits 3",
    )
    parser.add_argument(
        "--report",
        help="directory for the current-run endpoint_gates_report.json",
    )
    parser.add_argument("--gates", default="pass,learned", help="comma list of gates")
    parser.add_argument("--worker-chain", choices=tuple(GATES), help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        _validate_protocol(
            args.runs,
            args.warmup_frames,
            args.measured_frames,
            args.tail_frames,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.worker_chain:
        from kinovsr.ui.logging import configure_machine_output

        configure_machine_output(_result_log.name)
        _worker(
            argparse.Namespace(
                chain=args.worker_chain,
                clip=args.clip,
                warmup_frames=args.warmup_frames,
                measured_frames=args.measured_frames,
                tail_frames=args.tail_frames,
                out=args.out,
            )
        )
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
        args.warmup_frames,
        args.measured_frames,
        args.tail_frames,
    )
    try:
        fingerprint = _comparison_signature(
            clip=clip,
            runs=args.runs,
            warmup_frames=args.warmup_frames,
            measured_frames=args.measured_frames,
            tail_frames=args.tail_frames,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _log.error("cannot fingerprint benchmark environment: %s", exc)
        return 2
    try:
        _require_clip_frames(fingerprint["clip"]["track"]["sample_count"], total_frames)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        workloads = {
            chain: _resolved_workload(chain, clip, total_frames) for chain in sorted(selected)
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
                baseline,
                fingerprint=fingerprint,
                workloads=workloads,
                selected=selected,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _log.error(
                "cannot use endpoint baseline %s: %s; record it with --record-baseline",
                baseline_path,
                exc,
            )
            return 2

    _log.info(
        "product revisions: baseline=%s current=%s",
        None if baseline is None else baseline.get("product_revision"),
        revision,
    )
    report = _report_header(
        recording=args.record_baseline,
        clip=clip,
        baseline_path=baseline_path,
        current_revision=revision,
        baseline=baseline,
        fingerprint=fingerprint,
        workloads=workloads,
    )
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
                    str(clip),
                    chain,
                    args.warmup_frames,
                    args.measured_frames,
                    args.tail_frames,
                    args.runs,
                    keep_dir=Path(keep_raw),
                    expected_conditions=expected_conditions,
                )
                output_behaviors = [
                    _output_probe(
                        output,
                        warmup_frames=args.warmup_frames,
                        measured_frames=args.measured_frames,
                        total_frames=total_frames,
                    )
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
                canonical,
                output_behaviors,
            )
            if behavior_mismatches:
                _log.error(
                    "cannot record inconsistent per-run output behavior: %s",
                    "; ".join(behavior_mismatches),
                )
                return 2
            entry = {
                "label": label,
                "workload": workloads[chain],
                "measurement": measurement,
                "output_behavior": canonical,
                "output_behavior_run_count": len(output_behaviors),
            }
            recorded["gates"][chain] = entry
            report["gates"][chain] = entry
            _log.info("[RECORDED, NOT EVALUATED] %s: %.3f ms/frame", label, current_ms)
            continue

        assert baseline is not None
        entry = _evaluate_gate(
            measurement,
            output_behaviors,
            baseline["gates"][chain],
            fraction=fraction,
        )
        entry["label"] = label
        report["gates"][chain] = entry
        status = "PASS" if entry["pass"] else "FAIL"
        if not entry["pass"]:
            failures.append(chain)
        log = _log.info if entry["pass"] else _log.error
        log(
            "[%s] %s: baseline %.3f ms/frame, current %.3f, delta %+.3f (margin %.3f); output=%s",
            status,
            label,
            entry["baseline_steady_ms_per_frame"],
            entry["current_steady_ms_per_frame"],
            entry["delta_ms"],
            entry["allowed_margin_ms"],
            "MATCH" if entry["behavior_pass"] else "MISMATCH",
        )

    if args.record_baseline:
        if baseline_path.exists():
            # Re-recording must not silently absorb a regression into the new
            # baseline: keep the old file and log the per-gate deltas so the
            # maintainer sees what the ratchet is about to accept.
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


def _default_baseline_path() -> Path:
    from kinovsr.settings import Settings

    return Settings.from_env().shared_temp_dir / "trace_analysis" / BASELINE_FILENAME


if __name__ == "__main__":
    sys.exit(main())
