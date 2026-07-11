#!/usr/bin/env python3
"""Convert the stock BasicSR SpyNet .pth into the bundled safetensors.

The runtime form (see the weights README): every key gains the
``spynet.`` prefix, 4D conv weights transpose from torch OIHW to MLX
OHWI, and the ImageNet normalization constants are embedded as
``spynet.mean`` / ``spynet.std`` shaped (1, 1, 1, 3) so the runtime
needs no external preprocessing constants. Torch-free: the checkpoint
loads through the restricted unpickler in
:mod:`kinovsr.cli.commands.weights_convert`.

    python kinovsr/modeling/spynet/convert_spynet.py spynet_20210409-c6c1bd09.pth
"""

from __future__ import annotations

import sys
from pathlib import Path


def _state_dict(node):
    if isinstance(node, dict):
        keys = [k for k in node if isinstance(k, str)]
        if keys and any("." in k for k in keys):
            return node
        for value in node.values():
            found = _state_dict(value)
            if found is not None:
                return found
    if isinstance(node, (list, tuple)):
        for value in node:
            found = _state_dict(value)
            if found is not None:
                return found
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: convert_spynet.py <spynet .pth> [output.safetensors]",
              file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent / "weights"
        / "spynet_stock_20210409.safetensors")

    import mlx.core as mx

    from kinovsr.cli.commands.weights_convert import _load_pth_tree

    tree = _load_pth_tree(src)
    if isinstance(tree, tuple):
        tree, _demoted = tree
    state = _state_dict(tree)
    if state is None:
        print("error: no state dict found in the checkpoint", file=sys.stderr)
        return 2

    out: dict = {}
    for key, value in state.items():
        tensor = (value.materialize() if hasattr(value, "materialize")
                  else mx.array(value))
        if tensor.ndim == 4:                       # torch OIHW -> MLX OHWI
            tensor = mx.contiguous(tensor.transpose(0, 2, 3, 1))
        out[f"spynet.{key}"] = tensor
    out["spynet.mean"] = mx.array([0.485, 0.456, 0.406],
                                  dtype=mx.float32).reshape(1, 1, 1, 3)
    out["spynet.std"] = mx.array([0.229, 0.224, 0.225],
                                 dtype=mx.float32).reshape(1, 1, 1, 3)

    dst.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst), out)
    check = mx.load(str(dst))
    print(f"wrote {dst} ({len(check)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
