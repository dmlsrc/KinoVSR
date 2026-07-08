#!/usr/bin/env python3
"""DOVER-Mobile video scorer -- human-opinion-trained *video* quality.

Prints technical (fragments), aesthetic (resize), and fused scores per
video; higher is better, fused is in (0, 1).  Unlike per-frame metrics
(MUSIQ, NIQE) the clips pass through the network 32 frames at a time,
so temporal artifacts -- flicker, pumping, warping -- move the score.

Usage: dover_score.py <video> [...] [--max-frames N] [--weights PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from LTX_2_MLX.videotoolbox.dover import DoverMobile  # noqa: E402


def _read_video(path: Path, max_frames: int) -> mx.array:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap decoded frames (0 = whole video)")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model = DoverMobile(weights=args.weights)
    results = {}
    for vp in args.videos:
        s = model.score(_read_video(vp, args.max_frames))
        results[str(vp)] = s
        if not args.json:
            print(f"tech {s['tech']:8.4f}  aes {s['aes']:8.4f}  "
                  f"fused {s['fused']:6.4f}  {vp}")
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
