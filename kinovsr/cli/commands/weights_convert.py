"""Convert tensor-only PyTorch checkpoints to MLX safetensors.

The installed converter never imports torch. It statically scans pickle globals
against the same exact allowlist used by a restricted unpickler, validates
archive/storage/tensor metadata against explicit resource limits, reconstructs
the tensors with MLX, and verifies the saved artifact through ``mlx.core.load``.

    kinovsr weights convert model.pth
    kinovsr weights convert model.pth -o weights.safetensors
    kinovsr weights convert ckpt.pth --param-key params_ema
"""

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinovsr.modeling.pickle_scan import (
    PickleScanError,
    scan_checkpoint_globals,
    suspicious_globals,
)
from kinovsr.modeling.torch_checkpoint import (
    CheckpointFormatError,
    load_restricted_checkpoint,
)
from kinovsr.modeling.weights import safetensors_output_path

_log = logging.getLogger(__name__)


def _is_tensor(value: object) -> bool:

    return isinstance(value, mx.array)


def _resolve_source(spec: str) -> Path | None:
    """Resolve a literal checkpoint or one unique source-collection match."""
    literal = Path(spec)
    if literal.is_file():
        return literal
    root = Path("weights-src")
    direct = root / spec
    if direct.is_file():
        _log.info("source resolved from the source collection: %s", direct)
        return direct
    if root.is_dir() and "/" not in spec:
        matches = sorted(path for path in root.rglob(spec) if path.is_file())
        if len(matches) == 1:
            _log.info("source resolved from the source collection: %s", matches[0])
            return matches[0]
        if matches:
            raise SystemExit(
                f"{spec!r} is ambiguous in weights-src/: "
                + ", ".join(str(match) for match in matches)
            )
    return None


def _select_state_dict(
    obj: object,
    *,
    param_key: str | None,
) -> object:
    if param_key:
        if not (isinstance(obj, dict) and param_key in obj and hasattr(obj[param_key], "items")):
            have = list(obj) if isinstance(obj, dict) else type(obj).__name__
            _log.error("--param-key %r not in checkpoint (has: %s)", param_key, have)
            return None
        _log.info("using explicit nested checkpoint key %r", param_key)
        return obj[param_key]

    state = obj
    # A tree node with ``items`` is one of the reader's dict mappings.
    if hasattr(state, "items") and any(
        _is_tensor(value) for value in cast(Mapping[object, object], state).values()
    ):
        return state
    if (
        isinstance(obj, dict)
        and "params" in obj
        and "params_ema" in obj
        and hasattr(obj["params"], "items")
        and hasattr(obj["params_ema"], "items")
    ):
        _log.error(
            "checkpoint carries BOTH 'params' and 'params_ema'; pass "
            "--param-key params or --param-key params_ema to match the "
            "model's reference inference"
        )
        return None
    recognized = [
        key
        for key in ("state_dict", "model", "net", "weights", "params", "params_ema")
        if isinstance(obj, dict) and key in obj and hasattr(obj[key], "items")
    ]
    if len(recognized) > 1:
        _log.error(
            "checkpoint carries multiple recognized parameter mappings %s; "
            "select one with --param-key",
            recognized,
        )
        return None
    if recognized:
        key = recognized[0]
        _log.info("using nested checkpoint key %r", key)
        return cast(dict[object, object], obj)[key]  # recognized keys imply a dict
    return state


def run_convert(argv: list[str] | None = None) -> int:
    """Run the installed torch-free converter."""
    parser = argparse.ArgumentParser(
        prog="kinovsr weights convert",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to the .pth / .pt checkpoint.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .safetensors (default: input with .safetensors).",
    )
    parser.add_argument(
        "--strip-prefix",
        default="module.",
        help="Key prefix to strip when present (default: module.; pass '' to keep).",
    )
    parser.add_argument(
        "--only-prefix",
        default="",
        help="Keep only tensor keys beginning with this prefix.",
    )
    parser.add_argument(
        "--keep-fp64",
        action="store_true",
        help="Refused: MLX has no float64, so float64 always demotes to float32.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Proceed past static-scan findings for a trusted file; the restricted "
            "unpickler still refuses non-allowlisted globals."
        ),
    )
    parser.add_argument(
        "--param-key",
        default=None,
        help=(
            "Nested checkpoint mapping to extract. Required when inference intent "
            "is ambiguous, such as params versus params_ema."
        ),
    )
    args = parser.parse_args(argv)

    source = _resolve_source(args.input)
    if source is None:
        parser.error(f"no such file: {args.input} (also looked under weights-src/)")
    if args.keep_fp64:
        _log.error(
            "--keep-fp64 cannot be honored: MLX has no float64 dtype; "
            "float64 always demotes to float32"
        )
        return 2
    if args.output:
        try:
            output = safetensors_output_path(args.output)
        except ValueError as exc:
            _log.error("-o %s", exc)
            return 2
    elif source.parts[:1] == ("weights-src",):
        _log.error(
            "state -o when converting from weights-src/: the default output "
            "would land inside the source collection"
        )
        return 2
    else:
        output = source.with_suffix(".safetensors")

    try:
        references = scan_checkpoint_globals(source)
    except PickleScanError as exc:
        _log.error("cannot statically scan checkpoint: %s", exc)
        return 1
    _log.info("pickle scan found %s global reference(s)", len(references))
    for reference in sorted(references):
        _log.debug("pickle global: %s", reference)
    flagged = suspicious_globals(references)
    if flagged:
        _log.warning("pickle globals outside tensor-rebuild allowlist: %s", flagged)
        if not args.force:
            _log.error(
                "refusing to load; rerun with --force only for a trusted file "
                "(the restricted unpickler still rejects non-allowlisted globals)"
            )
            return 2
    else:
        _log.info("pickle scan clean: exact tensor-rebuild/container allowlist")

    try:
        checkpoint = load_restricted_checkpoint(
            source,
            allow_suspicious=args.force,
        )
    except CheckpointFormatError as exc:
        _log.error(
            "restricted checkpoint load refused: %s; extract a plain tensor state dict first",
            exc,
        )
        return 1
    if checkpoint.demoted_fp64:
        _log.warning("float64 storage demoted to float32 (MLX has no float64)")

    state = _select_state_dict(checkpoint.tree, param_key=args.param_key)
    if state is None:
        return 1
    if not hasattr(state, "items"):
        _log.error("checkpoint does not contain a parameter mapping")
        return 1

    tensors: dict[str, mx.array] = {}
    dropped: list[object] = []
    stripped = 0
    filtered = 0
    for key, value in state.items():
        if not _is_tensor(value):
            dropped.append(key)
            continue
        if not isinstance(key, str):
            _log.error("tensor key %r is not a string", key)
            return 1
        if args.only_prefix and not key.startswith(args.only_prefix):
            filtered += 1
            continue
        normalized = (
            key[len(args.strip_prefix) :]
            if args.strip_prefix and key.startswith(args.strip_prefix)
            else key
        )
        stripped += normalized != key
        tensors[normalized] = mx.contiguous(value)
    if not tensors:
        _log.error("no tensors found in the checkpoint")
        return 1
    mx.eval(list(tensors.values()))
    parameters = sum(tensor.size for tensor in tensors.values())
    dtype_names = sorted({str(tensor.dtype).split(".")[-1] for tensor in tensors.values()})
    _log.info(
        "converted %s tensors, %.3fM params, dtypes=%s",
        len(tensors),
        parameters / 1e6,
        dtype_names,
    )
    if stripped:
        _log.info("stripped prefix %r from %s keys", args.strip_prefix, stripped)
    if filtered:
        _log.info("filtered out %s tensor keys outside %r", filtered, args.only_prefix)
    if dropped:
        _log.warning(
            "dropped %s non-tensor entries: %s%s",
            len(dropped),
            dropped[:6],
            "..." if len(dropped) > 6 else "",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output), tensors)
    loaded = cast(dict[str, mx.array], mx.load(str(output)))
    sample = next(iter(loaded.items()))
    _log.info(
        "mlx.core.load verified %s arrays (for example %s %s %s); match=%s",
        len(loaded),
        sample[0],
        tuple(sample[1].shape),
        sample[1].dtype,
        len(loaded) == len(tensors),
    )
    _log.info("wrote %s", output)
    return 0


__all__ = ["run_convert"]
