#!/usr/bin/env python3
"""Scheduler overhead microbenchmark (planning 06 runtime gate).

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
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_log = logging.getLogger("kinovsr.dev.bench_scheduler")
_result_log = logging.getLogger("kinovsr.dev.bench_scheduler.result")

WARMUP = 30
FRAMES = 120
RUNS = 3
STAGE_COUNTS = (1, 4, 8)
# M8's independent stage actors deliberately replace the generator-chain
# timings that supported the old 0.01/0.02 M3 ratchet. The original
# 0.25/0.50 budget is now the worst-case physical-lane transit gate; adjacent
# same-affinity stages fuse and do not pay one transit per logical stage.
MEDIAN_GATE_MS = 0.25
P95_GATE_MS = 0.50


def _chain(n_stages: int):
    from kinovsr.pipeline.builder import ResolvedStage
    from kinovsr.processors import (
        Capability,
        CapabilitySpec,
        Layout,
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
            "bt709",
            full_range=False,
            geometry=Geometry(640, 480),
            layout=Layout.CV_BGRA,
        ),
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
        from kinovsr.ui.logging import configure_machine_output

        configure_machine_output(_result_log.name)
        _result_log.info("%s", json.dumps(run_single(args.single)))
        return 0

    from kinovsr.ui.logging import configure_logging

    configure_logging()

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
        log = _log.info if passed else _log.error
        log(
            "stages=%s: median %.4f ms/stage/frame, p95 %.4f "
            "(gate %s/%s) %s",
            n_stages,
            median,
            p95,
            MEDIAN_GATE_MS,
            P95_GATE_MS,
            "PASS" if passed else "FAIL",
        )

    base = os.environ.get("SHARED_TEMP_DIR") or os.environ.get("TMPDIR") or "/tmp"
    out_dir = Path(base) / "trace_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"kinovsr_scheduler_bench_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=1))
    _log.info("report: %s", out_path)
    if ok:
        _log.info("GATE: PASS")
    else:
        _log.error("GATE: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
