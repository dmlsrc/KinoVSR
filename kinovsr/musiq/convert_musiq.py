#!/usr/bin/env python3
"""Convert a released MUSIQ checkpoint (.pth) to the MLX safetensors layout.

Net-specific on purpose -- the general scripts/pth_to_safetensors.py would
produce a checkpoint that loads but scores garbage, because MUSIQ's
conversion is not a plain re-serialization:

- conv weights: weight standardization folded in (they are fixed at
  inference: w = (w - mean) / (unbiased_std + 1e-5), matching the reference
  StdConv), then OIHW -> OHWI for MLX conv2d.
- everything else passes through as fp32.

Loading uses torch.load(weights_only=True) (restricted unpickler). See
weights/README.md for the source URL and sha256.

Usage: convert_musiq.py <musiq_koniq_ckpt.pth> [out.safetensors]
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import torch

CONV_KEYS = {
    "conv_root.weight",
    "block1.conv1.weight",
    "block1.conv2.weight",
    "block1.conv3.weight",
    "block1.conv_proj.weight",
}


def main() -> int:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent / "weights" / "musiq_koniq.safetensors")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        t = v.float()
        if k in CONV_KEYS:
            t = t - t.mean((1, 2, 3), keepdim=True)
            t = t / (t.std((1, 2, 3), keepdim=True) + 1e-5)
            t = t.permute(0, 2, 3, 1).contiguous()
        out[k] = mx.array(t.numpy())
    dst.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst), out)
    print(f"{len(out)} tensors -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
