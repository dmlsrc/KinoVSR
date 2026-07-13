"""Post-M6 performance gates for the public file-processing endpoint.

M3 originally compared the typed file endpoints with the inherited harness.
M6 deleted that second orchestration path, so a same-checkout A/B comparison is
no longer possible or useful. This gate now measures the one public typed path
against a machine- and clip-specific baseline recorded by this script:

- passthrough: current path slower by at most max(2 ms/frame, 10% of baseline)
- learned chain: current path slower by at most max(5 ms/frame, 3% of baseline)
  (bsvd temporal denoise + realplksr 2x spatial SR, the M3 provers)

The source hash, source probe, output probe, chip, run count, and measured frame
count must match the baseline before timings are compared. Raw baselines and
reports belong under $SHARED_TEMP_DIR rather than in the repository.

Record the baseline once after an intentional path change:

    python scripts/dev/bench_endpoint_gates.py --clip PATH --record-baseline

Run the gate afterward:

    python scripts/dev/bench_endpoint_gates.py --clip PATH

Use --baseline PATH to override the default baseline at
$SHARED_TEMP_DIR/trace_analysis/kinovsr_endpoint_baseline.json.

Method: fresh process per run; each sample is end-to-end elapsed time divided by
the written frame count. This intentionally includes setup, probe, compile,
decode, processing, and encode because it gates the public endpoint as a whole.
The final output is retained only long enough to verify its frame count,
geometry, and cadence against the recorded baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_log = logging.getLogger("kinovsr.dev.bench_endpoint_gates")
_result_log = logging.getLogger("kinovsr.dev.bench_endpoint_gates.result")

BASELINE_SCHEMA = 2
BASELINE_FILENAME = "kinovsr_endpoint_baseline.json"
GATES = {
    "pass": ("file passthrough", 2.0, 0.10),
    "learned": ("bsvd + realplksr 2x", 5.0, 0.03),
}


# ---------------------------------------------------------------------------
# Worker mode: one measured run in a fresh process
# ---------------------------------------------------------------------------

def _worker(args: argparse.Namespace) -> None:
    from kinovsr.api import process_video_file
    from kinovsr.processors.specs import Layout
    from kinovsr.settings import Settings

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.chain == "pass":
        config: dict = {"pipeline": []}
        layout = Layout.CV_RGBA_HALF
    else:
        config = {
            "pipeline": ["den", "up"],
            "den": {
                "processor": "bsvd",
                "profile": "c64",
                "strength": 0.5,
            },
            "up": {"processor": "realplksr", "profile": "public2x"},
        }
        layout = Layout.MLX_RGB_HWC

    t0 = time.perf_counter()
    result = process_video_file(
        config,
        video=args.clip,
        output=out_dir / "current.mp4",
        settings=Settings(),
        layout=layout,
        max_frames=args.frames,
    )
    elapsed = time.perf_counter() - t0
    _result_log.info(
        "RESULT %s",
        json.dumps(
            {
                "elapsed": elapsed,
                "frames": result.frames_out,
                "path": str(result.post_path),
            }
        ),
    )


# ---------------------------------------------------------------------------
# Parent mode: orchestrate fresh-process runs and evaluate the gates
# ---------------------------------------------------------------------------

def _run_once(
    clip: str,
    chain: str,
    frames: int,
    keep_dir: Path | None = None,
) -> dict:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        out = keep_dir if keep_dir is not None else Path(scratch)
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                "--worker-chain",
                chain,
                "--clip",
                clip,
                "--frames",
                str(frames),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                result = json.loads(line.removeprefix("RESULT "))
                _log.info(
                    "%s frames=%s: %.1fs (wall %.1fs)",
                    chain,
                    frames,
                    result["elapsed"],
                    time.perf_counter() - t0,
                )
                return result
        raise RuntimeError(
            f"worker {chain}/{frames} produced no result:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )


def _measure_ms(
    clip: str,
    chain: str,
    frames: int,
    runs: int,
    keep_dir: Path,
) -> tuple[float, list[float], Path]:
    """Return median end-to-end cost and retain the final output for probing."""
    samples = []
    kept: Path | None = None
    for index in range(runs):
        last = index == runs - 1
        result = _run_once(
            clip,
            chain,
            frames,
            keep_dir=keep_dir if last else None,
        )
        samples.append(result["elapsed"] * 1000.0 / result["frames"])
        if last:
            kept = Path(result["path"])
    assert kept is not None
    return statistics.median(samples), samples, kept


def _probe(path: Path | str) -> list[int | float]:
    from kinovsr.media.video_reader import probe_video

    width, height, fps, frames, _, _ = probe_video(path)
    return [width, height, round(fps, 3), frames]


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chip_name() -> str:
    result = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or platform.machine()


def _default_baseline_path() -> Path:
    from kinovsr.settings import Settings

    return Settings.from_env().shared_temp_dir / "trace_analysis" / BASELINE_FILENAME


def _validate_baseline(
    baseline: dict,
    *,
    chip: str,
    clip_sha256: str,
    clip_probe: list[int | float],
    runs: int,
    frames: int,
    selected: set[str],
) -> None:
    if baseline.get("schema") != BASELINE_SCHEMA:
        raise ValueError(
            f"baseline schema must be {BASELINE_SCHEMA}, got {baseline.get('schema')!r}"
        )
    expected = {
        "chip": chip,
        "clip_sha256": clip_sha256,
        "clip_probe": clip_probe,
        "runs": runs,
        "frames": frames,
    }
    mismatches = [
        f"{key}: baseline={baseline.get(key)!r}, current={value!r}"
        for key, value in expected.items()
        if baseline.get(key) != value
    ]
    missing = sorted(selected - set(baseline.get("gates", {})))
    if missing:
        mismatches.append(f"missing gates: {missing}")
    if mismatches:
        raise ValueError("baseline does not match this run: " + "; ".join(mismatches))


def _evaluate_gate(
    measured_ms: float,
    samples: list[float],
    output_probe: list[int | float],
    baseline: dict,
    *,
    floor_ms: float,
    fraction: float,
) -> dict:
    baseline_ms = float(baseline["baseline_ms_per_frame"])
    margin = max(floor_ms, fraction * baseline_ms)
    delta = measured_ms - baseline_ms
    timing_pass = delta <= margin
    behavior_pass = output_probe == baseline["output_probe"]
    return {
        "baseline_ms_per_frame": round(baseline_ms, 3),
        "current_ms_per_frame": round(measured_ms, 3),
        "delta_ms": round(delta, 3),
        "delta_pct": round(100.0 * delta / baseline_ms, 2) if baseline_ms else None,
        "allowed_margin_ms": round(margin, 3),
        "current_runs": [round(value, 3) for value in samples],
        "timing_pass": timing_pass,
        "baseline_output_probe": baseline["output_probe"],
        "current_output_probe": output_probe,
        "behavior_pass": behavior_pass,
        "pass": timing_pass and behavior_pass,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--frames", type=int, default=36)
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
        help="record the selected machine/clip as the new baseline instead of gating",
    )
    parser.add_argument(
        "--report",
        help="directory for the current-run endpoint_gates_report.json",
    )
    parser.add_argument(
        "--gates",
        default="pass,learned",
        help="comma list of gates to run",
    )
    parser.add_argument("--worker-chain", choices=tuple(GATES))
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.worker_chain:
        from kinovsr.ui.logging import configure_machine_output

        configure_machine_output(_result_log.name)
        _worker(
            argparse.Namespace(
                chain=args.worker_chain,
                clip=args.clip,
                frames=args.frames,
                out=args.out,
            )
        )
        return 0

    if args.runs < 1 or args.frames < 1:
        parser.error("require runs >= 1 and frames >= 1")

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

    import mlx.core as mx

    baseline_path = Path(args.baseline) if args.baseline else _default_baseline_path()
    chip = _chip_name()
    clip_sha256 = _sha256(args.clip)
    clip_probe = _probe(args.clip)
    baseline: dict | None = None
    if not args.record_baseline:
        try:
            baseline = json.loads(baseline_path.read_text())
            _validate_baseline(
                baseline,
                chip=chip,
                clip_sha256=clip_sha256,
                clip_probe=clip_probe,
                runs=args.runs,
                frames=args.frames,
                selected=selected,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _log.error(
                "cannot use endpoint baseline %s: %s; record it with --record-baseline",
                baseline_path,
                exc,
            )
            return 2

    report: dict = {
        "schema": BASELINE_SCHEMA,
        "mode": "record" if args.record_baseline else "gate",
        "machine": platform.platform(),
        "chip": chip,
        "python": sys.version.split()[0],
        "mlx": mx.__version__,
        "clip": args.clip,
        "clip_sha256": clip_sha256,
        "clip_probe": clip_probe,
        "runs": args.runs,
        "frames": args.frames,
        "baseline": str(baseline_path),
        "gates": {},
    }
    recorded: dict = {
        "schema": BASELINE_SCHEMA,
        "chip": chip,
        "python": sys.version.split()[0],
        "mlx": mx.__version__,
        "clip_sha256": clip_sha256,
        "clip_probe": clip_probe,
        "runs": args.runs,
        "frames": args.frames,
        "gates": {},
    }
    failures = []
    for chain, (label, floor_ms, fraction) in GATES.items():
        if chain not in selected:
            continue
        with tempfile.TemporaryDirectory(prefix=f"gate_{chain}_") as keep_raw:
            measured_ms, samples, output = _measure_ms(
                args.clip,
                chain,
                args.frames,
                args.runs,
                keep_dir=Path(keep_raw),
            )
            output_probe = _probe(output)

        if args.record_baseline:
            entry = {
                "label": label,
                "baseline_ms_per_frame": round(measured_ms, 3),
                "runs": [round(value, 3) for value in samples],
                "output_probe": output_probe,
            }
            recorded["gates"][chain] = entry
            report["gates"][chain] = entry
            _log.info(
                "[RECORDED] %s: %.2f ms/frame; output=%s",
                label,
                measured_ms,
                output_probe,
            )
            continue

        assert baseline is not None
        entry = _evaluate_gate(
            measured_ms,
            samples,
            output_probe,
            baseline["gates"][chain],
            floor_ms=floor_ms,
            fraction=fraction,
        )
        entry["label"] = label
        report["gates"][chain] = entry
        status = "PASS" if entry["pass"] else "FAIL"
        if not entry["pass"]:
            failures.append(chain)
        log = _log.info if entry["pass"] else _log.error
        log(
            "[%s] %s: baseline %.2f ms/frame, current %.2f, "
            "delta %+.2f (margin %.2f); output=%s",
            status,
            label,
            entry["baseline_ms_per_frame"],
            entry["current_ms_per_frame"],
            entry["delta_ms"],
            entry["allowed_margin_ms"],
            "MATCH" if entry["behavior_pass"] else "MISMATCH",
        )

    if args.record_baseline:
        _write_json(baseline_path, recorded)
        _log.info("baseline: %s", baseline_path)
    if args.report:
        report_path = Path(args.report) / "endpoint_gates_report.json"
        _write_json(report_path, report)
        _log.info("report: %s", report_path)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
