#!/usr/bin/env python3
"""MUSIQ (koniq) video/image scorer -- human-opinion-trained quality.

Higher is better (roughly 0-100). Per-frame spatial metric: pair it with a
temporal-stability measure for video A/Bs. Unlike NSS blind metrics (NIQE),
MUSIQ was trained on human ratings of realistically distorted photos and
correctly ranks deblocked output above its blocky input (verified against
a ground-truth-validated deblocker on this workspace's fixtures).

Usage: musiq_score.py <video-or-image> [...] [--every N] [--weights PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kinovsr.musiq import Musiq  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _read_video_frames(path: Path, every: int) -> list:
    import av
    out = []
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i % every:
                continue
            r = f.reformat(format="rgb24")
            plane = r.planes[0]
            h, w, stride = r.height, r.width, plane.line_size
            raw = mx.array(memoryview(plane)).reshape(h, stride)[:, : w * 3]
            out.append(raw.reshape(h, w, 3).astype(mx.float32) / 255.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--every", type=int, default=5, help="score every Nth frame")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    m = Musiq(args.weights)
    rows = {}
    for inp in args.inputs:
        p = Path(inp)
        if p.suffix.lower() in IMAGE_EXTS:
            from kinovsr.media.images import load_image_rgb
            frames = [load_image_rgb(p).astype(mx.float32) / 255.0]
        else:
            frames = _read_video_frames(p, args.every)
        scores = m.score_frames(frames)
        rows[inp] = sum(scores) / len(scores)
        print(f"{rows[inp]:7.2f}  {inp}")
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
