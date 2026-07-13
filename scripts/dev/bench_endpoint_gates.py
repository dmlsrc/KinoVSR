"""M3 performance gates: file passthrough and a representative learned chain.

Compares the typed pipeline's file endpoints (kinovsr.pipeline.run_file)
against the harness control (kinovsr.api.process_video_file) on a real
clip, per planning 06-perf-baseline.md:

- passthrough:   new path slower by at most max(2 ms/frame, 10% of control)
- learned chain: new path slower by at most max(5 ms/frame, 3% of control)
  (bsvd temporal denoise + realplksr 2x spatial SR, the M3 provers)

Method: fresh process per run; steady-state ms/frame is the slope between
a short and a long run of the same clip so startup, probe, and compile
cost cancel in the intercept. Behavior must match too: frame count,
geometry, container duration, and a PSNR-between-outputs probe computed
from the final slope run's kept outputs (no extra runs).

Usage:
    python scripts/dev/bench_endpoint_gates.py --clip PATH [--runs 2]
        [--short 12] [--long 36] [--report DIR]

The clip should be a CFR H.264/HEVC file of at least --long frames. The
gate margins carry absolute floors (2 / 5 ms per frame), so a small
geometry keeps the protocol fast without changing which code paths run;
record the clip geometry with the result.
"""

from __future__ import annotations

import argparse
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

GATES = {
    "pass": ("file passthrough", 2.0, 0.10),
    "learned": ("bsvd + realplksr 2x", 5.0, 0.03),
}


# ---------------------------------------------------------------------------
# Worker mode: one measured run in a fresh process
# ---------------------------------------------------------------------------

def _worker(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.path == "control":
        from kinovsr.api import VideoFileConfig, process_video_file
        from kinovsr.cli.args import build_parser, validate_args
        from kinovsr.cli.config import assemble
        from kinovsr.settings import Settings

        argv = [
            "--video", args.clip, "--output-dir", str(out_dir),
            "--max-frames", str(args.frames),
            "--mlx-cache-limit-gb", "1",
            "--output-prefix", "control",
        ]
        if args.chain == "pass":
            argv += ["--upscale", "none"]
        else:
            argv += ["--upscale", "realplksr", "--denoise", "bsvd",
                     "--bsvd-strength", "0.5"]
        parser = build_parser()
        parsed = parser.parse_args(argv)
        validate_args(parser, parsed)
        invocation = assemble(parsed, base=Settings())
        t0 = time.perf_counter()
        result = process_video_file(VideoFileConfig(
            settings=invocation.settings, options=invocation.options))
        elapsed = time.perf_counter() - t0
        _result_log.info(
            "RESULT %s",
            json.dumps({
                "elapsed": elapsed,
                "frames": result.frames_out,
                "path": str(result.post_path),
            }),
        )
        return

    import mlx.core as mx

    from kinovsr.pipeline import run_file
    from kinovsr.processors.specs import Layout
    from kinovsr.settings import Settings

    mx.set_cache_limit(10 ** 9)
    if args.chain == "pass":
        config: dict = {"pipeline": []}
        layout = Layout.CV_RGBA_HALF
    else:
        config = {
            "pipeline": ["den", "up"],
            "den": {"processor": "bsvd", "profile": "c64",
                    "strength": 0.5},
            "up": {"processor": "realplksr", "profile": "public2x"},
        }
        layout = Layout.MLX_RGB_HWC
    t0 = time.perf_counter()
    result = run_file(
        config, video=args.clip, output=out_dir / "new.mp4",
        settings=Settings(), layout=layout, max_frames=args.frames)
    elapsed = time.perf_counter() - t0
    _result_log.info(
        "RESULT %s",
        json.dumps({
            "elapsed": elapsed,
            "frames": result.frames_out,
            "path": str(result.path),
        }),
    )


# ---------------------------------------------------------------------------
# Parent mode: orchestrate fresh-process runs and evaluate the gates
# ---------------------------------------------------------------------------

def _run_once(clip: str, path: str, chain: str, frames: int,
              keep_dir: Path | None = None) -> dict:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as scratch:
        out = keep_dir if keep_dir is not None else Path(scratch)
        proc = subprocess.run(
            [sys.executable, __file__, "--worker-path", path,
             "--worker-chain", chain, "--clip", clip,
             "--frames", str(frames), "--out", str(out)],
            capture_output=True, text=True, timeout=1800, check=False)
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
                _log.info(
                    "%s/%s frames=%s: %.1fs (wall %.1fs)",
                    chain,
                    path,
                    frames,
                    result["elapsed"],
                    time.perf_counter() - t0,
                )
                return result
        raise RuntimeError(
            f"worker {path}/{chain}/{frames} produced no result:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def _steady_ms(clip: str, path: str, chain: str, short: int, long: int,
               runs: int, keep_dir: Path) -> tuple[float, list[float], Path]:
    """Median slope over `runs`; the final long run keeps its output for
    the behavior comparison, so no extra measurement runs are needed."""
    slopes = []
    kept: Path | None = None
    for i in range(runs):
        last = i == runs - 1
        t_short = _run_once(clip, path, chain, short)["elapsed"]
        long_result = _run_once(clip, path, chain, long,
                                keep_dir=keep_dir if last else None)
        slopes.append((long_result["elapsed"] - t_short) * 1000.0
                      / (long - short))
        if last:
            kept = Path(long_result["path"])
    return statistics.median(slopes), slopes, kept


def _compare_outputs(control_mp4: Path, new_mp4: Path) -> dict:
    """Frame count, geometry, duration, and PSNR between the two outputs."""
    import mlx.core as mx

    from kinovsr.media import pixel_buffers as _pb
    from kinovsr.media import video_reader as vr

    info = {}
    probes = {}
    for name, path in (("control", control_mp4), ("new", new_mp4)):
        w, h, fps, n, _, _ = vr.probe_video(path)
        probes[name] = (w, h, round(fps, 3), n)
    info["geometry_match"] = probes["control"] == probes["new"]
    info["control_probe"] = probes["control"]
    info["new_probe"] = probes["new"]

    total, count = 0.0, 0
    readers = [
        iter(f for chunk in vr.iter_video_buffer_chunks(
            p, _pb.PIX_RGBAHALF, chunk_size=4) for f in chunk)
        for p in (control_mp4, new_mp4)]
    for a, b in zip(*readers, strict=True):
        fa = mx.clip(_pb.read_buffer_rgb_f32(a), 0, 1)
        fb = mx.clip(_pb.read_buffer_rgb_f32(b), 0, 1)
        diff = fa - fb
        mse = float(mx.mean(diff * diff))
        total += 10.0 * (0.0 - float(mx.log10(mx.array(max(mse, 1e-12)))))
        count += 1
    info["psnr_between_outputs_db"] = round(total / max(count, 1), 2)
    info["frames_compared"] = count
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--short", type=int, default=12)
    parser.add_argument("--long", type=int, default=36)
    parser.add_argument("--report", default=None,
                        help="directory for the JSON report")
    parser.add_argument("--gates", default="pass,learned",
                        help="comma list of gates to run")
    parser.add_argument("--worker-path", choices=("control", "new"))
    parser.add_argument("--worker-chain", choices=("pass", "learned"))
    parser.add_argument("--frames", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.worker_path:
        from kinovsr.ui.logging import configure_machine_output

        configure_machine_output(_result_log.name)
        ns = argparse.Namespace(
            path=args.worker_path, chain=args.worker_chain,
            clip=args.clip, frames=args.frames, out=args.out)
        _worker(ns)
        return 0

    from kinovsr.ui.logging import configure_logging

    configure_logging()

    import mlx.core as mx

    report: dict = {
        "machine": platform.platform(),
        "python": sys.version.split()[0],
        "mlx": mx.__version__,
        "clip": args.clip,
        "short_frames": args.short, "long_frames": args.long,
        "runs": args.runs,
        "gates": {},
    }
    failures = []
    selected = {g.strip() for g in args.gates.split(",") if g.strip()}
    for chain, (label, floor_ms, fraction) in GATES.items():
        if chain not in selected:
            continue
        keep = Path(tempfile.mkdtemp(prefix=f"gate_{chain}_"))
        control_ms, control_raw, control_out = _steady_ms(
            args.clip, "control", chain, args.short, args.long, args.runs,
            keep_dir=keep)
        new_ms, new_raw, new_out = _steady_ms(
            args.clip, "new", chain, args.short, args.long, args.runs,
            keep_dir=keep)
        margin = max(floor_ms, fraction * control_ms)
        delta = new_ms - control_ms
        ok = delta <= margin
        behavior = _compare_outputs(control_out, new_out)

        entry = {
            "label": label,
            "control_ms_per_frame": round(control_ms, 3),
            "new_ms_per_frame": round(new_ms, 3),
            "delta_ms": round(delta, 3),
            "delta_pct": round(100.0 * delta / control_ms, 2)
            if control_ms else None,
            "allowed_margin_ms": round(margin, 3),
            "control_slopes": [round(s, 3) for s in control_raw],
            "new_slopes": [round(s, 3) for s in new_raw],
            "timing_pass": ok,
            "behavior": behavior,
        }
        report["gates"][chain] = entry
        status = "PASS" if ok and behavior["geometry_match"] else "FAIL"
        if status == "FAIL":
            failures.append(chain)
        log = _log.info if status == "PASS" else _log.error
        log(
            "[%s] %s: control %.2f ms/frame, new %.2f ms/frame, "
            "delta %+.2f (margin %.2f); psnr-between %s dB",
            status,
            label,
            control_ms,
            new_ms,
            delta,
            margin,
            behavior["psnr_between_outputs_db"],
        )

    if args.report:
        out = Path(args.report)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "endpoint_gates_report.json"
        path.write_text(json.dumps(report, indent=2))
        _log.info("report: %s", path)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
