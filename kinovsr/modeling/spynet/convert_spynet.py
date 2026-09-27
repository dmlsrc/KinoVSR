#!/usr/bin/env python3
"""Convert the stock BasicSR SpyNet .pth into the bundled safetensors.

The runtime form (see the weights README): every key gains the
``spynet.`` prefix, 4D conv weights transpose from torch OIHW to MLX
OHWI, and the ImageNet normalization constants are embedded as
``spynet.mean`` / ``spynet.std`` shaped (1, 1, 1, 3) so the runtime
needs no external preprocessing constants. Torch-free: the checkpoint loads
through :mod:`kinovsr.modeling.torch_checkpoint`.

    python kinovsr/modeling/spynet/convert_spynet.py spynet_20210409-c6c1bd09.pth
"""

import logging
import sys
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinovsr.modeling.torch_checkpoint import load_restricted_checkpoint
from kinovsr.modeling.weights import safetensors_output_path

_log = logging.getLogger(__name__)


def _state_dict(node: object) -> dict[object, object] | None:
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
        _log.error("usage: convert_spynet.py <spynet .pth> [output.safetensors]")
        return 2
    src = Path(sys.argv[1])
    try:
        dst = (
            safetensors_output_path(sys.argv[2])
            if len(sys.argv) > 2
            else (Path(__file__).resolve().parent / "weights" / "spynet_stock_20210409.safetensors")
        )
    except ValueError as exc:
        _log.error("%s", exc)
        return 2

    checkpoint = load_restricted_checkpoint(src)
    state = _state_dict(checkpoint.tree)
    if state is None:
        _log.error("no state dict found in the checkpoint")
        return 2

    out: dict[str, mx.array] = {}
    for key, value in state.items():
        tensor = (
            value.materialize()
            if hasattr(value, "materialize")
            else mx.array(cast(mx.array, value))  # resolved checkpoint tensors are mx.array
        )
        if tensor.ndim == 4:  # torch OIHW -> MLX OHWI
            tensor = mx.contiguous(tensor.transpose(0, 2, 3, 1))
        out[f"spynet.{key}"] = tensor
    out["spynet.mean"] = mx.array([0.485, 0.456, 0.406], dtype=mx.float32).reshape(1, 1, 1, 3)
    out["spynet.std"] = mx.array([0.229, 0.224, 0.225], dtype=mx.float32).reshape(1, 1, 1, 3)

    dst.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst), out)
    check = mx.load(str(dst))
    _log.info(f"wrote {dst} ({len(check)} tensors)")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
