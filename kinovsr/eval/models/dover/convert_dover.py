#!/usr/bin/env python3
"""Convert the DOVER-Mobile checkpoint to MLX safetensors.

Layout changes (why the generic pth_to_safetensors.py is not enough):

- Conv3d weights go OIDHW -> ODHWI for ``mx.conv3d``'s channels-last
  layout (stem and the three downsample convs).
- Depthwise conv weights (C, 1, Lt, 7, 7) become (Lt, C, 7, 7, 1)
  stacks of 2D grouped kernels.  Temporal extents alternate 1 and 3
  across blocks (the reference's inflation pattern), and the port runs
  them as a temporal shift-and-sum of per-frame ``mx.conv2d(groups=C)``
  calls -- MLX has no grouped conv3d.
- Linear / 1x1x1-conv weights are stored pre-transposed as (in, out) so
  the forward pass is a plain ``x @ w + b``.
- GRN gamma/beta are flattened to (C,).
- The two timm classifier heads (``*_backbone.head.*``, 1000-way
  ImageNet leftovers) are dropped; DOVER never runs them.

Requires torch (conversion only; the runtime does not).

Usage: convert_dover.py <DOVER-Mobile.pth> [out.safetensors]
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import torch

from kinovsr._optional import require_numpy


def main() -> int:
    require_numpy("kinovsr/eval/models/dover/convert_dover.py")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).parent / "weights" / "dover_mobile.safetensors")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    out = {}
    skipped = []
    for k, t in sd.items():
        t = t.float()
        if ".head." in k and "_head." not in k:
            skipped.append(k)          # timm classifier leftovers
            continue
        if k.endswith("dwconv.weight"):
            t = t[:, 0].permute(1, 0, 2, 3)[..., None]      # (Lt,C,7,7,1)
        elif k.endswith(("downsample_layers.0.0.weight",
                         "1.1.weight", "2.1.weight", "3.1.weight")) \
                and t.ndim == 5:
            t = t.permute(0, 2, 3, 4, 1)                    # OIDHW -> ODHWI
        elif k.endswith(("pwconv1.weight", "pwconv2.weight")):
            t = t.T                                         # (in, out)
        elif k.endswith(("fc_hid.weight", "fc_last.weight")):
            t = t.reshape(t.shape[0], t.shape[1]).T         # (in, out)
        elif k.endswith(("grn.gamma", "grn.beta")):
            t = t.reshape(-1)                               # (C,)
        out[k] = mx.array(t.contiguous().numpy())
    dst.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst), out)
    print(f"wrote {dst} ({len(out)} tensors; skipped {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
