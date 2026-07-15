"""Fresh-process execution and measurement for endpoint performance gates."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import endpoint_gate_environment as _environment
from endpoint_gate_environment import _assert_run_conditions
from endpoint_gate_protocol import (
    GATES,
    _assert_frame_counts,
    _gate_definition,
    _OutputTiming,
    _SourceTiming,
    _summarize_runs,
    _total_frames,
)

_ENTRYPOINT = Path(__file__).with_name("bench_endpoint_gates.py")
_log = logging.getLogger("kinovsr.dev.bench_endpoint_gates")
_result_log = logging.getLogger("kinovsr.dev.bench_endpoint_gates.result")


def _worker(args: argparse.Namespace) -> None:
    import mlx.core as mx

    if not hasattr(mx, "get_peak_memory"):
        # Fail fast with one clear message instead of eight per-field
        # validation errors after a full measurement run.
        raise RuntimeError(
            "this MLX runtime lacks mx.get_peak_memory; the endpoint gate "
            "requires it to record peak_mlx_mib")

    from kinovsr.api import process_video_file
    from kinovsr.pipeline.run import FileSink, FileSource
    from kinovsr.processors.specs import Layout
    from kinovsr.settings import Settings

    definition = _gate_definition(args.chain)
    total_frames = _total_frames(
        args.warmup_frames,
        args.measured_frames,
        args.tail_frames,
    )
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
    conditions_start = _environment._runtime_conditions()
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
            encode_chroma=endpoint["encode_chroma"],
        )
    finally:
        FileSink.append = original_append
        FileSource.units = original_units
    finished = time.perf_counter()
    conditions_end = _environment._runtime_conditions()

    _assert_frame_counts(
        args.chain,
        total_frames,
        result.frames_in,
        result.frames_out,
    )
    metrics = timer.metrics(
        start_s=started,
        end_s=finished,
        expected_frames=total_frames,
    )
    metrics.update(
        {
            "frames_in": result.frames_in,
            "peak_rss_mib": _environment._peak_rss_mib(),
            "peak_mlx_mib": float(mx.get_peak_memory()) / (1024.0 * 1024.0),
            "measured_before_source_exhaustion_ms": (
                source_timer.measured_headroom_ms(
                    timer,
                    expected_frames=total_frames,
                )
            ),
            "conditions_start": conditions_start,
            "conditions_end": conditions_end,
            "path": str(result.post_path),
        }
    )
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
                sys.executable,
                str(_ENTRYPOINT),
                "--worker-chain",
                chain,
                "--clip",
                clip,
                "--warmup-frames",
                str(warmup_frames),
                "--measured-frames",
                str(measured_frames),
                "--tail-frames",
                str(tail_frames),
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
                if proc.returncode != 0:
                    break
                result = json.loads(line.removeprefix("RESULT "))
                _log.info(
                    "%s %s frames: steady %.3f ms/frame, total %.1fs (wall %.1fs)",
                    chain,
                    total_frames,
                    result["steady_ms_per_frame"],
                    result["total_ms"] / 1000.0,
                    time.perf_counter() - wall_start,
                )
                return result
        raise RuntimeError(
            f"worker {chain}/{total_frames} exited {proc.returncode} without a result:\n"
            f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
        )


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
            clip,
            chain,
            warmup_frames,
            measured_frames,
            tail_frames,
            keep_dir=keep_dir / f"run-{index + 1:02d}",
        )
        samples.append(result)
        outputs.append(Path(result["path"]))
    _assert_run_conditions(samples, expected_conditions)
    if len(outputs) != runs:
        raise RuntimeError(f"benchmark retained {len(outputs)} outputs for {runs} runs")
    return _summarize_runs(samples), outputs
