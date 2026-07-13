"""Convert a PyTorch .pth/.pt checkpoint to safetensors, directly MLX-loadable.

Torch-free and safe by construction - nothing in the checkpoint is
executed, and no PyTorch installation is involved:

1. A static pickle scan (pure pickletools, no unpickling) lists every global
   the file would invoke and refuses anything outside the tensor-rebuild /
   container allowlist (no os/subprocess/eval/exec/import machinery).
2. A restricted unpickler with a hard find_class allowlist rebuilds tensors
   straight from the zip archive's raw storage blobs into MLX arrays. The
   zip-format checkpoint (torch >= 1.6) is one pickle of metadata plus flat
   per-storage byte files; dtype, shape, offset, and strides are enough to
   reconstruct every tensor without torch. Anything outside the allowlist
   raises instead of running.

Then it makes the weights MLX-friendly: strips DataParallel 'module.'
prefixes, demotes float64 -> float32 (MLX has no float64), drops non-tensor
entries, saves through mx.save_safetensors, and verifies the output loads
with mlx.core.load().

    kinovsr weights convert model.pth                  # -> model.safetensors
    kinovsr weights convert model.pth -o weights.safetensors
    kinovsr weights convert ckpt.pth --only-prefix generator_ema. --strip-prefix generator_ema.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import pickletools
import zipfile
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


# Pickle opcodes that push the module/name strings a STACK_GLOBAL then consumes.
_STR_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "SHORT_BINSTRING", "BINSTRING",
}
# Modules a plain weights checkpoint legitimately references.
_SAFE_MODULES = {"torch", "collections", "numpy"}
# Substrings that mark a global as code-execution capable.
_DANGER = (
    "subprocess", "posix", "eval", "exec", "system", "popen", "__import__",
    "importlib", "runpy", "socket", "shutil", "compile", "getattr", "pty",
    "builtins", "os.", "commands", "webbrowser",
)


def _pickle_globals(data: bytes) -> set[str]:
    """Collect every GLOBAL / STACK_GLOBAL the (possibly concatenated) pickles
    reference, without unpickling. The legacy torch format is several pickles
    back to back followed by raw storage bytes, so walk pickle-by-pickle and
    stop when genops hits the non-pickle tail."""
    refs: set[str] = set()
    off, npk = 0, 0
    while off < len(data) and npk < 16:
        strs: list = []
        stop = None
        try:
            for op, arg, _pos in pickletools.genops(data[off:]):
                if op.name in _STR_OPS:
                    strs.append(arg)
                elif op.name == "GLOBAL":
                    refs.add(str(arg).replace("\n", " "))
                elif op.name == "STACK_GLOBAL" and len(strs) >= 2:
                    refs.add(f"{strs[-2]} {strs[-1]}")
                if op.name == "STOP":
                    stop = _pos
                    break
        except Exception:
            break
        if stop is None:
            break
        off += stop + 1
        npk += 1
    return refs


def _suspicious(refs: set[str]) -> list[str]:
    bad = []
    for r in refs:
        rl = r.lower()
        module = r.split()[0] if " " in r else r          # "torch._utils name" -> "torch._utils"
        top = module.split(".")[0]                          # -> "torch"
        if any(d in rl for d in _DANGER) or top not in _SAFE_MODULES:
            bad.append(r)
    return bad


# ---------------------------------------------------------------------------
# Torch-free checkpoint reading: zip-format .pth -> dict tree of mx.array
# ---------------------------------------------------------------------------

# torch.<X>Storage token -> (mlx dtype or "float64", itemsize)
_STORAGE_DTYPES = {
    "FloatStorage": ("float32", 4),
    "HalfStorage": ("float16", 2),
    "BFloat16Storage": ("bfloat16", 2),
    "DoubleStorage": ("float64", 8),
    "LongStorage": ("int64", 8),
    "IntStorage": ("int32", 4),
    "ShortStorage": ("int16", 2),
    "CharStorage": ("int8", 1),
    "ByteStorage": ("uint8", 1),
    "BoolStorage": ("bool_", 1),
}


class _CheckpointFormatError(RuntimeError):
    pass


_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C


def _storage_from_bytes(token: str, buf: bytes) -> tuple[Any, bool]:
    """Raw storage bytes -> 1-D mx array in the storage dtype.

    Returns (array, demoted): float64 demotes to float32 at this
    boundary because MLX has no float64 (the array-module decode is slow
    but float64 checkpoints are rare and small)."""
    import mlx.core as mx

    if token not in _STORAGE_DTYPES:
        raise _CheckpointFormatError(f"unsupported storage type torch.{token}")
    dtype_name, _itemsize = _STORAGE_DTYPES[token]
    if dtype_name == "float64":
        import array as _array

        return mx.array(list(_array.array("d", buf)), dtype=mx.float32), True
    raw = mx.array(memoryview(buf))
    if dtype_name == "uint8":
        return raw, False
    return raw.view(getattr(mx, dtype_name)), False


def _materialize(flat: Any, offset: int, shape: tuple[int, ...],
                 strides: tuple[int, ...]) -> Any:
    """Rebuild one tensor from its flat storage by offset/shape/strides."""
    import mlx.core as mx

    count = 1
    for s in shape:
        count *= s
    if count == 0:
        return mx.zeros(shape, dtype=flat.dtype)
    expected = []
    acc = 1
    for s in reversed(shape):
        expected.append(acc)
        acc *= s
    if strides == tuple(reversed(expected)) or count == 1:
        return flat[offset:offset + count].reshape(shape)
    # Non-contiguous save (transposed/sliced view): gather by the
    # declared strides so the rebuilt tensor matches torch exactly.
    index = mx.array(offset, dtype=mx.int64)
    for axis, (extent, step) in enumerate(zip(shape, strides, strict=True)):
        axis_shape = [1] * len(shape)
        axis_shape[axis] = extent
        index = index + (mx.arange(extent, dtype=mx.int64)
                         * step).reshape(axis_shape)
    return mx.take(flat, index.reshape(-1)).reshape(shape)


class _LazyTensor:
    """Tensor spec captured during the main unpickle; materialized once
    the storage bytes are available (the legacy format streams them
    AFTER the object pickle)."""

    __slots__ = ("token", "key", "offset", "shape", "strides")

    def __init__(self, token: str, key: str, offset: int,
                 shape: tuple[int, ...], strides: tuple[int, ...]) -> None:
        self.token = token
        self.key = key
        self.offset = offset
        self.shape = shape
        self.strides = strides


def _make_unpickler(handle: Any) -> Any:
    """A pickle.Unpickler locked to the tensor-rebuild allowlist. Tensors
    come back as _LazyTensor specs; persistent ids as
    ("storage", token, key) with the pid's numel preserved when present
    (the legacy tail is read against it)."""

    def rebuild_lazy(storage_ref: tuple, offset: Any, size: Any,
                     stride: Any, *rest: Any) -> _LazyTensor:
        _, token, key = storage_ref[:3]
        return _LazyTensor(
            token, key, int(offset),
            tuple(int(s) for s in size), tuple(int(s) for s in stride))

    class _Restricted(pickle.Unpickler):
        storage_meta: dict[str, tuple[str, int | None]] = {}

        def find_class(self, module: str, name: str) -> Any:
            if (module, name) == ("collections", "OrderedDict"):
                import collections

                return collections.OrderedDict
            if module == "torch._utils" and name in (
                    "_rebuild_tensor_v2", "_rebuild_tensor"):
                return rebuild_lazy
            if module == "torch" and name in _STORAGE_DTYPES:
                return name
            if (module, name) == ("torch", "Size"):
                return tuple
            raise pickle.UnpicklingError(
                f"{module}.{name} is outside the tensor-rebuild allowlist")

        def persistent_load(self, pid: Any) -> Any:
            if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
                raise pickle.UnpicklingError(
                    f"unsupported persistent id {pid!r}")
            token, key = pid[1], str(pid[2])
            numel = int(pid[4]) if len(pid) > 4 else None
            self.storage_meta[key] = (token, numel)
            return ("storage", token, key)

    unpickler = _Restricted(handle)
    unpickler.storage_meta = {}
    return unpickler


def _resolve_lazy(node: Any, storages: dict[str, Any]) -> Any:
    if isinstance(node, _LazyTensor):
        return _materialize(storages[node.key], node.offset,
                            node.shape, node.strides)
    if isinstance(node, dict):
        return type(node)(
            (k, _resolve_lazy(v, storages)) for k, v in node.items())
    if isinstance(node, (list, tuple)):
        return type(node)(_resolve_lazy(v, storages) for v in node)
    return node


def _load_zip_tree(src: Path) -> tuple[Any, bool]:
    archive = zipfile.ZipFile(src)
    pickles = [n for n in archive.namelist()
               if n == "data.pkl" or n.endswith("/data.pkl")]
    if not pickles:
        raise _CheckpointFormatError("zip archive carries no data.pkl")
    pickle_name = pickles[0]
    prefix = pickle_name[: -len("data.pkl")]
    with archive.open(pickle_name) as handle:
        unpickler = _make_unpickler(handle)
        tree = unpickler.load()
    storages: dict[str, Any] = {}
    demoted = False
    for key, (token, _numel) in unpickler.storage_meta.items():
        arr, was_demoted = _storage_from_bytes(
            token, archive.read(f"{prefix}data/{key}"))
        storages[key] = arr
        demoted = demoted or was_demoted
    return _resolve_lazy(tree, storages), demoted


def _load_legacy_tree(src: Path) -> tuple[Any, bool]:
    """The pre-1.6 stream: magic, protocol, sys_info, and object pickles
    back to back, then a storage-key list, then each storage as an int64
    numel followed by its raw bytes."""
    import struct

    with src.open("rb") as handle:
        magic = _make_unpickler(handle).load()
        if magic != _LEGACY_MAGIC:
            raise _CheckpointFormatError(
                "neither a zip-format nor a legacy torch checkpoint "
                "(bad magic)")
        _protocol = _make_unpickler(handle).load()
        _sys_info = _make_unpickler(handle).load()
        unpickler = _make_unpickler(handle)
        tree = unpickler.load()
        keys = _make_unpickler(handle).load()
        storages: dict[str, Any] = {}
        demoted = False
        for key in keys:
            key = str(key)
            token, pid_numel = unpickler.storage_meta[key]
            _dtype_name, itemsize = _STORAGE_DTYPES[token]
            (numel,) = struct.unpack("<q", handle.read(8))
            if pid_numel is not None and numel != pid_numel:
                raise _CheckpointFormatError(
                    f"storage {key}: tail carries {numel} elements, the "
                    f"pickle declared {pid_numel}")
            arr, was_demoted = _storage_from_bytes(
                token, handle.read(numel * itemsize))
            storages[key] = arr
            demoted = demoted or was_demoted
    return _resolve_lazy(tree, storages), demoted


def _load_pth_tree(src: Path) -> tuple[Any, bool]:
    """Restricted-unpickle a torch checkpoint (zip or legacy format).

    Returns (tree, demoted_fp64): the unpickled object tree with every
    tensor materialized as an mx.array, and whether any float64 storage
    was demoted to float32 on the way in (MLX has no float64).
    """
    if zipfile.is_zipfile(src):
        return _load_zip_tree(src)
    return _load_legacy_tree(src)


def _is_tensor(value: Any) -> bool:
    import mlx.core as mx

    return isinstance(value, mx.array)


def _resolve_source(spec: str) -> Path | None:
    """Resolve the input checkpoint, looking in weights-src/ first.

    Order: the literal path; then ``weights-src/<spec>``; then a unique
    ``weights-src/**/<basename>`` match. weights-src/ is the repo-root
    folder of collected upstream sources (see its README), resolved
    relative to the working directory. An ambiguous basename lists the
    candidates instead of guessing.
    """
    literal = Path(spec)
    if literal.is_file():
        return literal
    root = Path("weights-src")
    direct = root / spec
    if direct.is_file():
        _log.info("source resolved from the source collection: %s", direct)
        return direct
    if root.is_dir() and "/" not in spec:
        matches = sorted(p for p in root.rglob(spec) if p.is_file())
        if len(matches) == 1:
            _log.info(
                "source resolved from the source collection: %s", matches[0]
            )
            return matches[0]
        if matches:
            raise SystemExit(
                f"{spec!r} is ambiguous in weights-src/: "
                + ", ".join(str(m) for m in matches))
    return None


def run_convert(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="kinovsr weights convert",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="Path to the .pth / .pt checkpoint.")
    ap.add_argument("-o", "--output", help="Output .safetensors (default: input with .safetensors).")
    ap.add_argument(
        "--strip-prefix", default="module.",
        help="Key prefix to strip from every weight (default 'module.'; '' to keep).",
    )
    ap.add_argument(
        "--only-prefix", default="",
        help="Keep only tensor keys with this prefix before applying --strip-prefix.",
    )
    ap.add_argument(
        "--keep-fp64", action="store_true",
        help="Refused: MLX has no float64, so float64 always demotes to float32.",
    )
    ap.add_argument("--force", action="store_true", help="Convert even if the static scan flags non-tensor globals.")
    ap.add_argument(
        "--param-key", default=None,
        help="Nested checkpoint dict to extract (e.g. 'params' or 'params_ema'). "
             "Checkpoints often carry BOTH; pick the one the model's reference "
             "inference loads -- they are different weights. If a checkpoint has "
             "both params and params_ema, the converter refuses to guess.",
    )
    args = ap.parse_args(argv)

    src = _resolve_source(args.input)
    if src is None:
        ap.error(
            f"no such file: {args.input} (also looked under weights-src/)")
    if args.keep_fp64:
        _log.error(
            "--keep-fp64 cannot be honored: MLX has no float64 dtype and the "
            "output exists to be MLX-loadable; float64 demotes to float32"
        )
        return 2
    if args.output:
        out = Path(args.output)
    elif src.parts[:1] == ("weights-src",):
        _log.error(
            "state -o when converting from weights-src/: the default output "
            "would land inside the source collection"
        )
        return 2
    else:
        out = src.with_suffix(".safetensors")

    # ---- 1. static safety scan (no execution) ------------------------------
    refs = _pickle_globals(src.read_bytes())
    _log.info("pickle scan found %s global reference(s)", len(refs))
    for r in sorted(refs):
        _log.debug("pickle global: %s", r)
    bad = _suspicious(refs)
    if bad:
        _log.warning("pickle globals outside tensor-rebuild allowlist: %s", bad)
        if not args.force:
            _log.error(
                "refusing to load; rerun with --force only if you trust this "
                "file (the restricted unpickler still rejects non-allowlisted "
                "globals)"
            )
            return 2
    else:
        _log.info("pickle scan clean: only tensor-rebuild/container globals")

    # ---- 2. safe load (restricted unpickler, torch-free) -------------------
    import mlx.core as mx

    try:
        obj, demoted = _load_pth_tree(src)
    except (_CheckpointFormatError, pickle.UnpicklingError, KeyError,
            zipfile.BadZipFile) as e:
        _log.error(
            "restricted checkpoint load refused: %s; the checkpoint carries "
            "constructs outside plain tensor state dicts (optimizer state, "
            "custom classes, or legacy format); extract just the weights first",
            e,
        )
        return 1
    if demoted:
        _log.warning("float64 storage demoted to float32 (MLX has no float64)")

    # ---- 3. find the state_dict (handle common nesting) --------------------
    sd = obj
    if args.param_key:
        if not (isinstance(obj, dict) and args.param_key in obj
                and hasattr(obj[args.param_key], "items")):
            have = list(obj.keys()) if isinstance(obj, dict) else type(obj).__name__
            _log.error(
                "--param-key %r not in checkpoint (has: %s)", args.param_key, have
            )
            return 1
        sd = obj[args.param_key]
        _log.info("using explicit nested checkpoint key %r", args.param_key)
    elif not (hasattr(sd, "items") and any(_is_tensor(v) for v in sd.values())):
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
            return 1
        for key in ("state_dict", "model", "net", "weights", "params", "params_ema"):
            if isinstance(obj, dict) and key in obj and hasattr(obj[key], "items"):
                sd = obj[key]
                _log.info("using nested checkpoint key %r", key)
                break

    # ---- 4. make MLX-friendly: strip prefix, drop non-tensors --------------
    prefix = args.strip_prefix
    only_prefix = args.only_prefix
    tensors, dropped, stripped, filtered = {}, [], 0, 0
    for k, v in sd.items():
        if not _is_tensor(v):
            dropped.append(k)
            continue
        if only_prefix and not k.startswith(only_prefix):
            filtered += 1
            continue
        nk = k[len(prefix):] if prefix and k.startswith(prefix) else k
        stripped += (nk != k)
        tensors[nk] = mx.contiguous(v)
    if not tensors:
        _log.error("no tensors found in the checkpoint")
        return 1
    mx.eval(list(tensors.values()))
    n_params = sum(t.size for t in tensors.values())
    dtype_names = sorted({str(t.dtype).split(".")[-1] for t in tensors.values()})
    _log.info(
        "converted %s tensors, %.3fM params, dtypes=%s",
        len(tensors),
        n_params / 1e6,
        dtype_names,
    )
    if stripped:
        _log.info("stripped prefix %r from %s keys", prefix, stripped)
    if filtered:
        _log.info(
            "filtered out %s tensor keys outside %r", filtered, only_prefix
        )
    if dropped:
        _log.warning(
            "dropped %s non-tensor entries: %s%s",
            len(dropped),
            dropped[:6],
            "..." if len(dropped) > 6 else "",
        )

    # ---- 5. save + verify it loads back ------------------------------------
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), tensors)
    loaded = mx.load(str(out))
    ok = len(loaded) == len(tensors)
    sample = next(iter(loaded.items()))
    _log.info(
        "mlx.core.load verified %s arrays (for example %s %s %s); match=%s",
        len(loaded),
        sample[0],
        tuple(sample[1].shape),
        sample[1].dtype,
        ok,
    )
    _log.info("wrote %s", out)
    return 0
