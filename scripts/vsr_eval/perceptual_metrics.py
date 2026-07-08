#!/usr/bin/env python3
"""Perceptual metric table for a manifest of processed VSR variants.

Scores every variant video (and optionally the source) with:

- ``musiq``       MUSIQ (koniq): per-frame human-opinion metric, mean over
                  sampled frames; higher is better; blind to temporal wins.
- ``dover_*``     DOVER-Mobile tech/aes/fused: temporal-aware opinion
                  metric (32-frame clips through the net); higher is
                  better; fused is 0-1.
- ``niqe``        NIQE: over-smoothing TRIPWIRE only (blur scores worst).
                  Codec junk reads as natural texture, so never rank or
                  tune deblockers by it.
- ``flicker_e3``  mean |frame-to-frame luma diff| x1000 on pixels the
                  SOURCE says are static; lower is stabler. Needs
                  ``--source``.
- ``vmaf_src``    VMAF against the source: fidelity anchor (how far the
                  variant strayed), not a quality score. Needs
                  ``--source`` and an ffmpeg build with libvmaf.

Flicker and VMAF assume every variant starts at source frame 0 (true for
sweep runs; pass pre-trimmed sources otherwise). Metrics whose weights or
tools are unavailable print a note to stderr and leave the column null.

Usage:
  perceptual_metrics.py --variants-json manifest.json --source in.mp4 \
      --out-dir eval/ [--max-frames 400] [--musiq-every 10] \
      [--metrics musiq,dover,niqe,flicker,vmaf]

The manifest is a JSON object mapping variant name to video path (the
same format run_denoise_sweep.py writes as variants.json).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from niqe import _luma_of, score_lumas  # noqa: E402
from LTX_2_MLX.videotoolbox.dover import DoverMobile  # noqa: E402
from LTX_2_MLX.videotoolbox.musiq import Musiq  # noqa: E402

ALL_METRICS = ("musiq", "dover", "niqe", "flicker", "vmaf")
STATIC_THRESHOLD = 0.01          # source luma |diff| below this = static
EDGE_SKIP = (3, 2)               # frames dropped at clip head/tail


def _note(msg: str) -> None:
    print(f"[perceptual] {msg}", file=sys.stderr, flush=True)


def read_frames(path: Path, max_frames: int) -> mx.array:
    import av
    out = []
    with av.open(str(path)) as c:
        for f in c.decode(c.streams.video[0]):
            r = f.reformat(format="rgb24")
            plane = r.planes[0]
            h, w, stride = r.height, r.width, plane.line_size
            raw = mx.array(memoryview(plane)).reshape(h, stride)[:, : w * 3]
            out.append(raw.reshape(h, w, 3))
            if max_frames and len(out) >= max_frames:
                break
    return mx.stack(out)


def _lumas01(frames: mx.array) -> list:
    return [_luma_of(frames[t].astype(mx.float32) / 255.0)
            for t in range(frames.shape[0])]


def static_masks(src_lumas: list) -> list:
    """masks[k] flags pixels static across source frames (k, k+1)."""
    return [mx.abs(src_lumas[t] - src_lumas[t - 1]) < STATIC_THRESHOLD
            for t in range(1, len(src_lumas))]


def flicker_e3(cand_lumas: list, masks: list) -> float | None:
    head, tail = EDGE_SKIP
    n = min(len(cand_lumas), len(masks) + 1)
    vals = []
    for t in range(head, n - tail):
        m = masks[t - 1].astype(mx.float32)
        denom = float(m.sum())
        if denom == 0.0:
            continue
        d = mx.abs(cand_lumas[t] - cand_lumas[t - 1])
        vals.append(float((d * m).sum()) / denom)
    if not vals:
        return None
    return sum(vals) / len(vals) * 1000.0


def vmaf_src(cand: Path, ref: Path, log_path: Path) -> float | None:
    if shutil.which("ffmpeg") is None:
        _note("ffmpeg not found; vmaf column skipped")
        return None
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(cand), "-i", str(ref),
           "-lavfi",
           f"[0:v][1:v]libvmaf=log_fmt=json:log_path={log_path}:n_threads=8",
           "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not log_path.exists():
        _note(f"libvmaf failed for {cand.name}; vmaf column skipped")
        return None
    data = json.loads(log_path.read_text(encoding="utf-8"))
    return float(data["pooled_metrics"]["vmaf"]["mean"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants-json", type=Path, required=True,
                    help="JSON mapping variant name -> video path")
    ap.add_argument("--source", type=Path,
                    help="source clip: adds a 'source' row and enables "
                         "flicker + vmaf columns")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--musiq-every", type=int, default=10,
                    help="score every Nth frame with MUSIQ")
    ap.add_argument("--metrics", default=",".join(ALL_METRICS),
                    help=f"comma subset of {','.join(ALL_METRICS)}")
    args = ap.parse_args()

    want = {m.strip() for m in args.metrics.split(",") if m.strip()}
    if unknown := want - set(ALL_METRICS):
        raise SystemExit(f"unknown metrics: {', '.join(sorted(unknown))}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    musiq = dover = None
    if "musiq" in want:
        try:
            musiq = Musiq()
        except FileNotFoundError as e:
            _note(f"musiq column skipped: {e}")
    if "dover" in want:
        try:
            dover = DoverMobile()
        except FileNotFoundError as e:
            _note(f"dover columns skipped: {e}")

    videos: dict[str, Path] = {
        name: Path(p)
        for name, p in json.loads(
            args.variants_json.read_text(encoding="utf-8")).items()
    }
    masks = None
    if args.source is not None:
        src_frames = read_frames(args.source, args.max_frames)
        if "flicker" in want:
            masks = static_masks(_lumas01(src_frames))
        videos = {"source": args.source, **videos}
    elif want & {"flicker", "vmaf"}:
        _note("no --source; flicker and vmaf columns skipped")

    rows = []
    for name, vp in videos.items():
        row: dict = {"variant": name, "video": str(vp)}
        frames = (src_frames if name == "source"
                  else read_frames(vp, args.max_frames))
        if musiq is not None:
            sampled = [frames[t].astype(mx.float32) / 255.0
                       for t in range(0, frames.shape[0], args.musiq_every)]
            scores = musiq.score_frames(sampled)
            row["musiq"] = sum(scores) / len(scores)
        if dover is not None:
            s = dover.score(frames)
            row["dover_tech"] = s["tech"]
            row["dover_aes"] = s["aes"]
            row["dover_fused"] = s["fused"]
        if "niqe" in want or masks is not None:
            lumas = _lumas01(frames)
            if "niqe" in want:
                row["niqe"] = score_lumas(lumas[::3])
            if masks is not None:
                row["flicker_e3"] = flicker_e3(lumas, masks)
        if "vmaf" in want and args.source is not None:
            row["vmaf_src"] = (100.0 if name == "source" else vmaf_src(
                vp, args.source, args.out_dir / f"vmaf_{name}.json"))
        rows.append(row)
        print(json.dumps(row), flush=True)

    (args.out_dir / "perceptual_metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    keys = sorted({k for row in rows for k in row})
    with (args.out_dir / "perceptual_metrics.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    def sort_key(r):
        v = r.get("dover_fused", r.get("musiq"))
        return -(v if isinstance(v, (int, float)) else float("-inf"))

    cols = [("musiq", "{:.2f}"), ("dover_fused", "{:.4f}"),
            ("niqe", "{:.2f}"), ("flicker_e3", "{:.2f}"),
            ("vmaf_src", "{:.1f}")]
    present = [(k, fmt) for k, fmt in cols if any(k in r for r in rows)]
    print(f"\n{'variant':20s}" + "".join(f"{k:>12s}" for k, _ in present))
    for r in sorted(rows, key=sort_key):
        line = f"{r['variant']:20s}"
        for k, fmt in present:
            v = r.get(k)
            cell = fmt.format(v) if isinstance(v, (int, float)) else "-"
            line += cell.rjust(12)
        print(line)
    print("\nhigher better: musiq, dover_fused; lower better: niqe "
          "(tripwire only), flicker_e3; vmaf_src = fidelity anchor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
