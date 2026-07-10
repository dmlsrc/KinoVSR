#!/usr/bin/env python3
"""Scheduler overhead microbenchmark (planning 06 M3 gate).

In-memory source and sink, no MLX work: measures pure framework cost per
stage per frame across no-op chains of 1, 4, and 8 stages. Gate values
live in MEDIAN_GATE_MS / P95_GATE_MS (ratcheted per planning 06);
scaling in stage count should stay roughly linear.

Parent mode spawns fresh child processes (one per run) and aggregates;
results land under $SHARED_TEMP_DIR/trace_analysis/.

    python scripts/dev/bench_scheduler.py            # full gate protocol
    python scripts/dev/bench_scheduler.py --single 4 # one in-process run
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

WARMUP = 30
FRAMES = 120
RUNS = 3
STAGE_COUNTS = (1, 4, 8)
# Ratcheted from the 0.25/0.50 pre-implementation budget after the first
# real run (planning 06): measured ~0.0007 median worst case on M1 Max.
MEDIAN_GATE_MS = 0.01
P95_GATE_MS = 0.02


def _chain(n_stages: int):
    from kinovsr.pipeline.builder import ResolvedStage
    from kinovsr.processors import (
        Capability,
        CapabilitySpec,
        StreamConstraint,
        StreamSpec,
        TimelineSpec,
        preserve_stream,
    )
    from kinovsr.processors.specs import Geometry, frame_spec_for_matrix

    class Noop:
        def prepare(self, input_spec, context):
            pass

        def process(self, unit, context):
            yield unit

        def reset(self, boundary, context):
            pass

        def flush(self, context):
            return ()

        def close(self, context):
            pass

    spec = StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(640, 480)),
        timeline=TimelineSpec(time_base=Fraction(1, 24000),
                              cadence=Fraction(25)))
    cap = CapabilitySpec(
        capability=Capability.PREPROCESS, profiles=(),
        accepts=StreamConstraint(), produces=preserve_stream)
    return tuple(
        (ResolvedStage(
            name=f"noop{i}", position=i, family="noop", factory=None,
            capability=Capability.PREPROCESS, capability_spec=cap,
            profile=None, config=None, input_spec=spec, output_spec=spec),
         Noop())
        for i in range(n_stages))


def run_single(n_stages: int) -> dict:
    from kinovsr.pipeline import run_chain
    from kinovsr.processors import FrameUnit, PipelineContext
    from kinovsr.settings import Settings

    context = PipelineContext(settings=Settings())
    payload = object()
    units = (FrameUnit(payload=payload, pts=i, duration=960)
             for i in range(WARMUP + FRAMES))
    stream = run_chain(_chain(n_stages), units, context)

    for _ in range(WARMUP):
        next(stream)

    samples_ns = []
    for _ in range(FRAMES):
        t0 = time.perf_counter_ns()
        next(stream)
        samples_ns.append(time.perf_counter_ns() - t0)
    stream.close()

    per_stage_ms = sorted(ns / n_stages / 1e6 for ns in samples_ns)
    return {
        "stages": n_stages,
        "frames": FRAMES,
        "warmup": WARMUP,
        "median_ms_per_stage_frame": statistics.median(per_stage_ms),
        "p95_ms_per_stage_frame": per_stage_ms[int(0.95 * (FRAMES - 1))],
        "total_ms_per_frame": statistics.median(
            ns / 1e6 for ns in samples_ns),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=int, default=None,
                        help="run one measurement in-process and print JSON")
    args = parser.parse_args()

    if args.single is not None:
        print(json.dumps(run_single(args.single)))
        return 0

    report = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "protocol": {"warmup": WARMUP, "frames": FRAMES, "runs": RUNS},
        "gates_ms": {"median": MEDIAN_GATE_MS, "p95": P95_GATE_MS},
        "results": {},
    }
    ok = True
    for n_stages in STAGE_COUNTS:
        runs = []
        for _ in range(RUNS):
            out = subprocess.run(
                [sys.executable, __file__, "--single", str(n_stages)],
                capture_output=True, text=True, check=True)
            runs.append(json.loads(out.stdout))
        medians = [r["median_ms_per_stage_frame"] for r in runs]
        p95s = [r["p95_ms_per_stage_frame"] for r in runs]
        median = statistics.median(medians)
        p95 = statistics.median(p95s)
        passed = median <= MEDIAN_GATE_MS and p95 <= P95_GATE_MS
        ok = ok and passed
        report["results"][str(n_stages)] = {
            "runs": runs,
            "median_ms_per_stage_frame": median,
            "p95_ms_per_stage_frame": p95,
            "pass": passed,
        }
        print(f"stages={n_stages}: median {median:.4f} ms/stage/frame, "
              f"p95 {p95:.4f} (gate {MEDIAN_GATE_MS}/{P95_GATE_MS}) "
              f"{'PASS' if passed else 'FAIL'}")

    base = os.environ.get("SHARED_TEMP_DIR") or os.environ.get("TMPDIR") or "/tmp"
    out_dir = Path(base) / "trace_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"kinovsr_scheduler_bench_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=1))
    print(f"report: {out_path}")
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
