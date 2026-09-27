#!/usr/bin/env python3
"""Convert TOFlow Torch7 `.t7` checkpoints into MLX safetensors + graph JSON.

This does not execute Lua/Torch code. It uses a pure Python Torch7 serializer
reader to deserialize tensors and module metadata, keeps only the inference
weights/statistics used by the released TOFlow module subset, and writes:

  - `toflow_<variant>.safetensors`: MLX-loadable tensors
  - `toflow_<variant>.json`: static Torch7 module tree for the MLX interpreter

Example:

    python kinovsr/processors/toflow/convert_t7_to_safetensors.py \\
      "$TOFLOW_REF/models/denoise.t7" \\
      -o kinovsr/processors/toflow/weights/toflow_denoise.safetensors
"""

import argparse
import hashlib
import json
import logging
import re
import struct
from pathlib import Path
from typing import BinaryIO, NotRequired, TypedDict, cast

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from kinovsr.modeling.weights import safetensors_output_path

_log = logging.getLogger(__name__)

_PARAMS_BY_TYPE = {
    "nn.Mul": ("weight",),
    "nn.SpatialConvolution": ("weight", "bias"),
    "nn.SpatialBatchNormalization": ("weight", "bias", "running_mean", "running_var"),
}

_ATTRS_BY_TYPE = {
    "nn.AddConstant": ("constant_scalar",),
    "nn.SelectTable": ("index",),
    "nn.JoinTable": ("dimension", "nInputDims"),
    "nn.Replicate": ("nfeatures", "dim", "ndim"),
    "nn.ShuffleTable": ("idx",),
    "nn.SplitTable": ("dimension", "nInputDims"),
    "nn.Sum": ("dimension", "nInputDims", "sizeAverage"),
    "nn.SpatialConvolution": (
        "nInputPlane",
        "nOutputPlane",
        "kH",
        "kW",
        "dH",
        "dW",
        "padH",
        "padW",
        "groups",
    ),
    "nn.SpatialBatchNormalization": ("eps", "affine"),
    "nn.SpatialAveragePooling": (
        "kH",
        "kW",
        "dH",
        "dW",
        "padH",
        "padW",
        "ceil_mode",
        "count_include_pad",
    ),
    "nn.SpatialUpSamplingBilinear": ("scale_factor",),
    "nn.SpatialUpSamplingNearest": ("scale_factor",),
    "nn.MulConstant": ("constant_scalar",),
    "nn.WarpFlowNew": ("squeezed",),
}

_SUPPORTED_TYPES = {
    "nn.AddConstant",
    "nn.CAddTable",
    "nn.CDivTable",
    "nn.CMulTable",
    "nn.ConcatTable",
    "nn.Identity",
    "nn.JoinTable",
    "nn.Mul",
    "nn.MulConstant",
    "nn.ParallelTable",
    "nn.ReLU",
    "nn.Replicate",
    "nn.SelectTable",
    "nn.Sequential",
    "nn.ShuffleTable",
    "nn.SpatialAveragePooling",
    "nn.SpatialBatchNormalization",
    "nn.SpatialConvolution",
    "nn.SpatialUpSamplingBilinear",
    "nn.SpatialUpSamplingNearest",
    "nn.SplitTable",
    "nn.Sum",
    "nn.WarpFlowNew",
}


_T7_NIL = 0
_T7_NUMBER = 1
_T7_STRING = 2
_T7_TABLE = 3
_T7_TORCH = 4
_T7_BOOLEAN = 5
_T7_FUNCTION = 6
_T7_LEGACY_RECUR_FUNCTION = 7
_T7_RECUR_FUNCTION = 8


def _tensor_dtypes() -> dict[bytes, type[np.generic[int | float]]]:
    return {
        b"torch.ByteTensor": np.uint8,
        b"torch.CharTensor": np.int8,
        b"torch.ShortTensor": np.int16,
        b"torch.IntTensor": np.int32,
        b"torch.LongTensor": np.int64,
        b"torch.FloatTensor": np.float32,
        b"torch.DoubleTensor": np.float64,
        b"torch.CudaTensor": np.float32,
        b"torch.CudaByteTensor": np.uint8,
        b"torch.CudaCharTensor": np.int8,
        b"torch.CudaShortTensor": np.int16,
        b"torch.CudaIntTensor": np.int32,
        b"torch.CudaDoubleTensor": np.float64,
    }


def _storage_dtypes() -> dict[bytes, type[np.generic[int | float]]]:
    return {
        b"torch.ByteStorage": np.uint8,
        b"torch.CharStorage": np.int8,
        b"torch.ShortStorage": np.int16,
        b"torch.IntStorage": np.int32,
        b"torch.LongStorage": np.int64,
        b"torch.FloatStorage": np.float32,
        b"torch.DoubleStorage": np.float64,
        b"torch.CudaStorage": np.float32,
        b"torch.CudaByteStorage": np.uint8,
        b"torch.CudaCharStorage": np.int8,
        b"torch.CudaShortStorage": np.int16,
        b"torch.CudaIntStorage": np.int32,
        b"torch.CudaDoubleStorage": np.float64,
    }


# A deserialized Torch7 value: nil, boolean, number, string (bytes), a table (a
# list when its keys are exactly 1..n, else a dict), a tensor or storage, a
# Torch object, or an inert Lua function.
type _T7Array = npt.NDArray[np.generic[int | float]]
type _T7Value = (
    bool
    | int
    | float
    | bytes
    | list[_T7Value]
    | dict[object, _T7Value]
    | _T7Array
    | _TorchObject
    | _LuaFunction
    | None
)


class _LuaFunction:
    """Inert serialized Lua function. Stored only so object references stay valid."""

    def __init__(self, dumped: bytes, upvalues: _T7Value):
        self.dumped = dumped
        self.upvalues = upvalues


class _TorchObject:
    """Minimal Torch7 object wrapper: class name + deserialized attribute table."""

    def __init__(self, typename: bytes, attrs: dict[object, _T7Value] | None = None):
        self._typename = typename
        self._obj = attrs or {}

    def torch_typename(self) -> bytes:
        return self._typename

    def __getitem__(self, key: str | bytes) -> object:
        return _dict_get(self._obj, key)

    def __getattr__(self, key: str) -> object:
        val = _dict_get(self._obj, key, _MISSING)
        if val is _MISSING:
            raise AttributeError(key)
        return val


_MISSING = object()


def _dict_get(d: dict[object, _T7Value], key: str | bytes, default: object = None) -> object:
    if key in d:
        return d[key]
    if isinstance(key, str):
        return d.get(key.encode("utf-8"), default)
    try:
        return d.get(key.decode("utf-8"), default)
    except UnicodeDecodeError:
        return default


class _T7Reader:
    """Small Torch7 binary reader for TOFlow checkpoint conversion.

    Torch7 saves object references, Lua tables, Torch classes, storages and tensor
    views. TOFlow's files were written on a 64-bit Torch build, so `long` fields
    are eight bytes by default; `--long-size 4` is available for old checkpoints.
    """

    def __init__(self, fileobj: BinaryIO, long_size: int = 8):
        if long_size not in (4, 8):
            raise ValueError("long_size must be 4 or 8")
        self.f = fileobj
        self.long_size = long_size
        self.objects: dict[int, _T7Value] = {}

    def _read(self, fmt: str) -> tuple[int | float, ...]:
        data = self.f.read(struct.calcsize(fmt))
        if len(data) != struct.calcsize(fmt):
            raise EOFError("unexpected end of Torch7 file")
        return struct.unpack(fmt, data)

    # Explicit little-endian standard sizes: struct's native "l" is eight
    # bytes on 64-bit macOS, which silently turned --long-size 4 into 8.
    def _read_int(self) -> int:
        return int(self._read("<i")[0])

    def _read_long(self) -> int:
        return int(self._read("<q" if self.long_size == 8 else "<i")[0])

    def _read_double(self) -> float:
        return float(self._read("<d")[0])

    def _read_string(self) -> bytes:
        size = self._read_int()
        data = self.f.read(size)
        if len(data) != size:
            raise EOFError("unexpected end of Torch7 string")
        return data

    def _read_long_array(self, n: int) -> list[int]:
        return [self._read_long() for _ in range(n)]

    def _read_storage(self, dtype: type[np.generic[int | float]]) -> _T7Array:
        size = self._read_long()
        arr = np.fromfile(self.f, dtype=dtype, count=size)
        if arr.size != size:
            raise EOFError("unexpected end of Torch7 storage")
        return arr

    _MAX_TENSOR_DIMS = 8

    def _read_tensor(self, dtype: type[np.generic[int | float]]) -> _T7Array:
        ndim = self._read_int()
        if ndim < 0 or ndim > self._MAX_TENSOR_DIMS:
            raise ValueError(
                f"tensor record has implausible ndim {ndim} (limit {self._MAX_TENSOR_DIMS})"
            )
        size = self._read_long_array(ndim)
        stride = self._read_long_array(ndim)
        storage_offset = self._read_long() - 1
        storage = cast(_T7Array | None, self.read_obj())  # a storage record or nil
        if storage is None or ndim == 0 or not size:
            return np.empty((0,), dtype=dtype)
        # size/stride/offset come straight from the file; a lying record must
        # not drive as_strided (and the tobytes/ascontiguousarray that
        # materialize the view) outside the storage buffer.
        if any(s < 0 for s in size) or any(s < 0 for s in stride):
            raise ValueError(
                f"tensor record has negative size/stride: size={size}, stride={stride}"
            )
        if storage_offset < 0:
            raise ValueError(f"tensor record has negative storage offset {storage_offset}")
        if any(s == 0 for s in size):
            return np.zeros(tuple(int(s) for s in size), dtype=dtype)
        last = storage_offset + sum((s - 1) * st for s, st in zip(size, stride, strict=True))
        if last >= storage.size:
            raise ValueError(
                f"tensor record reaches element {last} of a "
                f"{storage.size}-element storage (offset {storage_offset}, "
                f"size={size}, stride={stride})"
            )
        byte_stride = tuple(int(s) * storage.dtype.itemsize for s in stride)
        return np.lib.stride_tricks.as_strided(
            storage[int(storage_offset) :],
            shape=tuple(int(s) for s in size),
            strides=byte_stride,
        )

    def _read_table(self, index: int) -> list[_T7Value] | dict[object, _T7Value]:
        size = self._read_int()
        d: dict[object, _T7Value] = {}
        self.objects[index] = d
        natural = True
        key_sum = 0
        for _ in range(size):
            key = self.read_obj()
            val = self.read_obj()
            d[key] = val
            if isinstance(key, int) and key > 0:
                key_sum += key
            else:
                natural = False
        n = len(d)
        if natural and n * (n + 1) == 2 * key_sum:
            out = [d[i + 1] for i in range(n)]
            self.objects[index] = out
            return out
        return d

    def _read_torch_object(self, index: int) -> _T7Array | _TorchObject:
        version = self._read_string()
        typename = self._read_string() if version.startswith(b"V ") else version

        storage_dtypes = _storage_dtypes()
        tensor_dtypes = _tensor_dtypes()
        if typename in storage_dtypes:
            self.objects[index] = None
            obj = self._read_storage(storage_dtypes[typename])
            self.objects[index] = obj
            return obj
        if typename in tensor_dtypes:
            self.objects[index] = None
            obj = self._read_tensor(tensor_dtypes[typename])
            self.objects[index] = obj
            return obj

        torch_obj = _TorchObject(typename)
        self.objects[index] = torch_obj
        attrs = self.read_obj()
        if not isinstance(attrs, dict):
            raise TypeError(f"Torch object {typename!r} attributes are not a table")
        torch_obj._obj = attrs
        return torch_obj

    def read_obj(self) -> _T7Value:
        typeidx = self._read_int()
        if typeidx == _T7_NIL:
            return None
        if typeidx == _T7_NUMBER:
            x = self._read_double()
            return int(x) if x.is_integer() else x
        if typeidx == _T7_BOOLEAN:
            return self._read_int() == 1
        if typeidx == _T7_STRING:
            return self._read_string()

        if typeidx not in {
            _T7_TABLE,
            _T7_TORCH,
            _T7_FUNCTION,
            _T7_LEGACY_RECUR_FUNCTION,
            _T7_RECUR_FUNCTION,
        }:
            raise TypeError(f"unknown Torch7 type index {typeidx}")

        index = self._read_int()
        if index in self.objects:
            return self.objects[index]
        if typeidx in {_T7_FUNCTION, _T7_LEGACY_RECUR_FUNCTION, _T7_RECUR_FUNCTION}:
            size = self._read_int()
            dumped = self.f.read(size)
            if len(dumped) != size:
                raise EOFError("unexpected end of Torch7 Lua function")
            obj = _LuaFunction(dumped, self.read_obj())
            self.objects[index] = obj
            return obj
        if typeidx == _T7_TORCH:
            return self._read_torch_object(index)
        return self._read_table(index)


def _load_t7(path: Path, long_size: int = 8) -> _T7Value:
    with path.open("rb") as f:
        return _T7Reader(f, long_size=long_size).read_obj()


def _typename(obj: _TorchObject) -> str:
    name = obj.torch_typename()
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def _raw_attrs(obj: _T7Value) -> dict[object, _T7Value]:
    return getattr(obj, "_obj", {})


def _get(obj: _T7Value, key: str, default: _T7Value = None) -> _T7Value:
    d = _raw_attrs(obj)
    if key in d:
        return d[key]
    return d.get(key.encode("utf-8"), default)


def _jsonable(v: object) -> object:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, dict):
        return {_jsonable(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _clean_key(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("._") or "tensor"


def _tensor_for_mlx(module_type: str, param_name: str, arr: _T7Array) -> _T7Array:
    out = np.asarray(arr)
    if module_type == "nn.SpatialConvolution" and param_name == "weight":
        # Torch7 SpatialConvolution stores OIHW. MLX conv2d wants OHWI.
        out = np.transpose(out, (0, 2, 3, 1))
    if np.issubdtype(out.dtype, np.floating) and out.dtype != np.float32:
        out = out.astype(np.float32)
    return np.ascontiguousarray(out)


def _store_tensor(
    tensors: dict[str, mx.array],
    seen: dict[tuple[str, tuple[int, ...], str], str],
    key_hint: str,
    arr: _T7Array,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(str(tuple(arr.shape)).encode("ascii"))
    digest.update(arr.tobytes())
    sig = (arr.dtype.str, tuple(arr.shape), digest.hexdigest())
    if sig in seen:
        return seen[sig]
    key = f"{len(tensors):04d}_{_clean_key(key_hint)}"
    tensors[key] = mx.array(arr)
    seen[sig] = key
    return key


# One module of the graph JSON; the MLX interpreter reads the same shape.
class _GraphNode(TypedDict):
    type: str
    attrs: NotRequired[dict[str, object]]
    params: NotRequired[dict[str, str]]
    modules: NotRequired[list[_GraphNode]]


def _module_tree(
    obj: _TorchObject,
    path: str,
    tensors: dict[str, mx.array],
    seen: dict[tuple[str, tuple[int, ...], str], str],
) -> _GraphNode:
    typ = _typename(obj)
    if typ not in _SUPPORTED_TYPES:
        raise ValueError(f"unsupported TOFlow module {typ!r} at {path}")

    node: _GraphNode = {"type": typ}
    attrs: dict[str, object] = {}
    for key in _ATTRS_BY_TYPE.get(typ, ()):
        val = _get(obj, key, None)
        if val is not None:
            attrs[key] = _jsonable(val)
    if attrs:
        node["attrs"] = attrs

    params: dict[str, str] = {}
    for pname in _PARAMS_BY_TYPE.get(typ, ()):
        val = _get(obj, pname, None)
        if isinstance(val, np.ndarray) and val.size:
            arr = _tensor_for_mlx(typ, pname, val)
            params[pname] = _store_tensor(tensors, seen, f"{path}.{pname}", arr)
    if params:
        node["params"] = params

    modules = _get(obj, "modules", None)
    if modules:
        node["modules"] = [
            _module_tree(child, f"{path}.modules[{i}]", tensors, seen)
            for i, child in enumerate(cast(list[_TorchObject], modules), 1)  # array of modules
        ]
    return node


def _count_types(node: _GraphNode, counts: dict[str, int]) -> None:
    counts[node["type"]] = counts.get(node["type"], 0) + 1
    for child in node.get("modules", ()):
        _count_types(child, counts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="TOFlow Torch7 .t7 checkpoint")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .safetensors path (default: input stem + .safetensors)",
    )
    ap.add_argument(
        "--graph",
        type=Path,
        help="Output graph .json path (default: output with .json suffix)",
    )
    ap.add_argument(
        "--variant",
        default=None,
        help="Variant metadata token (default: input stem)",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip MLX reload verification after writing.",
    )
    ap.add_argument(
        "--long-size",
        type=int,
        choices=[4, 8],
        default=8,
        help="Torch7 C long size in bytes (default 8, matching TOFlow's released files).",
    )
    args = ap.parse_args()

    src = args.input.expanduser()
    if not src.is_file():
        ap.error(f"no such file: {src}")
    try:
        out = (
            safetensors_output_path(args.output.expanduser())
            if args.output
            else src.with_suffix(".safetensors")
        )
    except ValueError as exc:
        ap.error(str(exc))
    graph_out = args.graph.expanduser() if args.graph else out.with_suffix(".json")
    variant = args.variant or src.stem

    data = src.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    _log.info(f"[load] {src.name}: {len(data):,} bytes sha256={sha}")
    root = cast(_TorchObject, _load_t7(src, long_size=args.long_size))  # a saved nn module
    if _typename(root) != "nn.Sequential":
        raise SystemExit(f"error: expected nn.Sequential root, got {_typename(root)}")

    tensors: dict[str, mx.array] = {}
    seen: dict[tuple[str, tuple[int, ...], str], str] = {}
    tree = _module_tree(root, "root", tensors, seen)
    counts: dict[str, int] = {}
    _count_types(tree, counts)

    graph = {
        "format": "toflow-mlx-graph-v1",
        "variant": variant,
        "source": {
            "filename": src.name,
            "size_bytes": len(data),
            "sha256": sha,
        },
        "layout": {
            "runtime": "NHWC",
            "conv_weight": "OHWI",
            "flow": "NHW2 pixel offsets, channel order x/y",
        },
        "module_type_counts": dict(sorted(counts.items())),
        "root": tree,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), tensors)
    graph_out.parent.mkdir(parents=True, exist_ok=True)
    graph_out.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    n_params = sum(int(np.prod(t.shape)) for t in tensors.values())
    _log.info(
        f"[convert] modules={sum(counts.values())} tensors={len(tensors)} "
        f"params={n_params / 1e6:.3f}M unique"
    )
    _log.info(f"[write] weights: {out}")
    _log.info(f"[write] graph:   {graph_out}")
    if not args.no_verify:
        w = cast(dict[str, mx.array], mx.load(str(out)))
        ok = len(w) == len(tensors)
        sample_key = next(iter(w))
        _log.info(
            f"[verify] mlx.core.load OK: {len(w)} arrays "
            f"(e.g. {sample_key} {tuple(w[sample_key].shape)} {w[sample_key].dtype}); "
            f"match={ok}"
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
