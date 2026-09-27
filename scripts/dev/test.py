#!/usr/bin/env python3
"""Run KinoVSR's quick developer lane, the complete suite, or the complete suite in
concurrent lanes.

Extra arguments are passed through to pytest for quick and full, for example:

    python scripts/dev/test.py quick -x tests/media/test_timing.py

concurrent runs the complete suite as three pytest processes at once, then the solo
tests alone:

    neural-engine  Core ML, MPSGraph and Neural Engine models, and the learned
                   families whose SpyNet runs on the Neural Engine by default.
                   Many processes mapping Neural Engine models at once can exhaust
                   its address space and stall it, so these share one process.
    videotoolbox   VideoToolbox frame-rate conversion and super resolution, and
                   whole-file runs. Those networks run on the Neural Engine too.
    other          every other test file.
    solo           tests marked solo, once the lanes finish. They compare
                   VideoToolbox output bit for bit across two runs, and load from
                   other processes changes VideoToolbox's arithmetic.

Each lane logs to its own file under $SHARED_TEMP_DIR (else $TMPDIR)/kinovsr-tests.
"""

import argparse
import contextlib
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LANES = {
    "quick": ("-m", "not integration and not slow and not requires_weights"),
    "full": (),
}
# Test paths per concurrent lane; every other test file runs in "other".
NEURAL_ENGINE = (
    "tests/modeling/test_spynet_ane.py",
    "tests/modeling/test_vision_flow_integration.py",
    "tests/modeling/test_vsr_blocks.py",
    "tests/native/test_anecir.py",
    "tests/native/test_anemil_builder.py",
    "tests/native/test_anemil_runtime.py",
    "tests/native/test_mpsgraph.py",
    "tests/native/test_mpsgraph_state.py",
    "tests/pipeline/test_streaming.py",
    "tests/processors/basicvsrpp",
    "tests/processors/bsvd/test_ane.py",
    "tests/processors/bsvd/test_ane_direct.py",
    "tests/processors/bsvd/test_mpsgraph.py",
    "tests/processors/mc/test_factory.py",
    "tests/processors/realbasicvsr",
    "tests/processors/realviformer",
    "tests/processors/test_factory_sweep.py",
    "tests/processors/test_mlx_conv_padding.py",
    "tests/processors/test_vision_flow_ownership.py",
)
VIDEOTOOLBOX = (
    "tests/api/test_process_video_file.py",
    "tests/native/test_pool_ownership.py",
    "tests/native/test_temporal.py",
    "tests/native/test_temporal_batching.py",
    "tests/native/test_temporal_canvas.py",
    "tests/native/test_vsr.py",
    "tests/native/test_vsr_flow_integration.py",
    "tests/native/test_vsr_pool_profiles.py",
    "tests/pipeline/test_run_file.py",
    "tests/pipeline/test_session.py",
    "tests/processors/sanitize_edges/test_factory.py",
    "tests/processors/videotoolbox",
)
# The complete suite takes under two minutes serially; a lane still running after
# this long has stalled.
TIMEOUT_S = 900
SUMMARY = re.compile(r"\b(?:passed|failed|error|errors|skipped|no tests ran)\b.* in [\d.]+s")
_log = logging.getLogger("kinovsr.dev.test")


def _log_dir() -> Path:
    base = Path(os.environ.get("SHARED_TEMP_DIR") or tempfile.gettempdir()) / "kinovsr-tests"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=base))


def _start(name: str, argv: list[str], log_dir: Path) -> subprocess.Popen:
    with (log_dir / f"{name}.log").open("w") as log:
        return subprocess.Popen(
            argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )


def _signal(procs: dict[str, subprocess.Popen], sig: int) -> None:
    for proc in procs.values():
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, sig)


def _report(name: str, proc: subprocess.Popen, seconds: float, log_dir: Path) -> bool:
    """Log a lane's pytest summary (and its failures); True when it passed."""
    lines = (log_dir / f"{name}.log").read_text(errors="replace").splitlines()
    summary = next((line.strip("= ") for line in reversed(lines) if SUMMARY.search(line)), None)
    # Exit status 5 means the lane selected no tests, which is not a failure.
    passed = proc.returncode in (0, 5)
    status = summary or f"exit status {proc.returncode}"
    _log.info("  %-14s %s (done at %.0f s)", name, status, seconds)
    if not passed:
        for line in lines:
            if line.startswith(("FAILED ", "ERROR ")):
                _log.error("    %s", line)
    return passed


def _wait(
    procs: dict[str, subprocess.Popen], start: float, log_dir: Path, results: dict[str, bool]
) -> str | None:
    """Wait for every process; None when all ended, else why they were stopped."""
    pending = dict(procs)
    try:
        while pending:
            for name, proc in list(pending.items()):
                if proc.poll() is not None:
                    del pending[name]
                    results[name] = _report(name, proc, time.monotonic() - start, log_dir)
            if pending and time.monotonic() - start > TIMEOUT_S:
                _signal(pending, signal.SIGKILL)
                return f"timed out after {TIMEOUT_S} s; stopped {', '.join(pending)}"
            time.sleep(0.2)
    except KeyboardInterrupt:
        _signal(pending, signal.SIGINT)
        deadline = time.monotonic() + 5
        while any(proc.poll() is None for proc in pending.values()):
            if time.monotonic() > deadline:
                _signal(pending, signal.SIGKILL)
                break
            time.sleep(0.1)
        return "interrupted"
    return None


def run_concurrent() -> int:
    listed = (*NEURAL_ENGINE, *VIDEOTOOLBOX)
    missing = [path for path in listed if not (REPO / path).exists()]
    if missing:
        raise SystemExit(f"concurrent lane paths no longer exist: {', '.join(missing)}")
    pytest_argv = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]
    lanes = {
        "neural-engine": [*pytest_argv, "-m", "not solo", *NEURAL_ENGINE],
        "videotoolbox": [*pytest_argv, "-m", "not solo", *VIDEOTOOLBOX],
        "other": [
            *pytest_argv,
            "-m",
            "not solo",
            "tests",
            *(f"--ignore={path}" for path in listed),
        ],
    }
    log_dir = _log_dir()
    _log.info("concurrent: logs in %s", log_dir)
    start = time.monotonic()
    results: dict[str, bool] = {}
    procs = {name: _start(name, argv, log_dir) for name, argv in lanes.items()}
    stopped = _wait(procs, start, log_dir, results)
    if stopped is None:
        solo = {"solo": _start("solo", [*pytest_argv, "-m", "solo", "tests"], log_dir)}
        stopped = _wait(solo, start, log_dir, results)
    elapsed = time.monotonic() - start
    if stopped is not None:
        _log.error("concurrent: %s (%.0f s)", stopped, elapsed)
        return 130 if stopped == "interrupted" else 1
    failed = [name for name, passed in results.items() if not passed]
    if failed:
        _log.error("concurrent: failed in %s (%.0f s)", ", ".join(failed), elapsed)
        return 1
    _log.info("concurrent: passed (%.0f s)", elapsed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "lane",
        choices=(*LANES, "concurrent"),
        default="quick",
        nargs="?",
        help="test selection (default: quick)",
    )
    args, pytest_args = parser.parse_known_args()
    if args.lane == "concurrent":
        if pytest_args:
            parser.error("concurrent runs the complete suite and takes no pytest arguments")
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return run_concurrent()
    os.chdir(REPO)
    return pytest.main([*LANES[args.lane], *pytest_args])


if __name__ == "__main__":
    raise SystemExit(main())
