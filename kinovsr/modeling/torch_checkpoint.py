"""Restricted, torch-free PyTorch tensor-checkpoint reconstruction."""

import array
import collections
import io
import pickle
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

import mlx.core as mx

from kinovsr.media.buffer import mlx_array_from_buffer

from .pickle_scan import (
    _STORAGE_TOKENS,
    PickleScanError,
    _is_allowed_global,
    scan_checkpoint_globals,
    suspicious_globals,
)


class CheckpointFormatError(RuntimeError):
    """A checkpoint is malformed or outside the tensor-only contract."""


@dataclass(frozen=True)
class CheckpointReadLimits:
    """Resource limits applied before checkpoint-controlled MLX allocations."""

    max_pickle_bytes: int = 64 * 1024 * 1024
    max_storage_bytes: int = 64 * 1024**3
    max_total_storage_bytes: int = 64 * 1024**3
    max_tensor_bytes: int = 64 * 1024**3
    max_total_tensor_bytes: int = 96 * 1024**3
    max_archive_members: int = 200_000
    max_storages: int = 200_000
    max_tensors: int = 1_000_000
    max_tree_nodes: int = 2_000_000
    max_tree_depth: int = 256
    max_rank: int = 64


_DEFAULT_LIMITS = CheckpointReadLimits()
_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
# torch storage token -> (MLX dtype name or float64 sentinel, source item size)
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

if frozenset(_STORAGE_DTYPES) != _STORAGE_TOKENS:
    raise RuntimeError("torch checkpoint scanner and reader storage tokens differ")


def _require_plain_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise pickle.UnpicklingError(f"{label} must be an integer")
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise pickle.UnpicklingError(f"{label} is outside the int64 range")
    return value


@dataclass(frozen=True)
class _StorageRef:
    token: str
    key: str
    numel: int


class _LazyTensor:
    __slots__ = ("offset", "shape", "storage", "strides")

    def __init__(
        self,
        storage: _StorageRef,
        offset: int,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
    ) -> None:
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.strides = strides


class _RestrictedUnpickler(Protocol):
    storage_meta: dict[str, _StorageRef]

    def load(self) -> object: ...


def _make_unpickler(handle: BinaryIO, limits: CheckpointReadLimits) -> _RestrictedUnpickler:
    def rebuild_lazy(
        storage_ref: object,
        offset: object,
        size: object,
        stride: object,
        *_rest: object,
    ) -> _LazyTensor:
        if not isinstance(storage_ref, _StorageRef):
            raise pickle.UnpicklingError("tensor rebuild received an invalid storage")
        if not isinstance(size, (list, tuple)) or not isinstance(stride, (list, tuple)):
            raise pickle.UnpicklingError("tensor size and stride must be sequences")
        if len(size) != len(stride):
            raise pickle.UnpicklingError("tensor size and stride ranks differ")
        if len(size) > limits.max_rank:
            raise pickle.UnpicklingError(f"tensor rank {len(size)} exceeds limit {limits.max_rank}")
        shape = tuple(
            _require_plain_int(value, f"tensor extent {index}") for index, value in enumerate(size)
        )
        strides = tuple(
            _require_plain_int(value, f"tensor stride {index}")
            for index, value in enumerate(stride)
        )
        return _LazyTensor(
            storage_ref,
            _require_plain_int(offset, "tensor storage offset"),
            shape,
            strides,
        )

    class _Restricted(pickle.Unpickler):
        storage_meta: dict[str, _StorageRef]

        def find_class(self, module: str, name: str) -> object:
            if not _is_allowed_global(module, name):
                raise pickle.UnpicklingError(
                    f"{module}.{name} is outside the tensor-rebuild allowlist"
                )
            if (module, name) == ("collections", "OrderedDict"):
                return collections.OrderedDict
            if module == "torch._utils" and name in {
                "_rebuild_tensor",
                "_rebuild_tensor_v2",
            }:
                return rebuild_lazy
            if module == "torch" and name in _STORAGE_DTYPES:
                return name
            if (module, name) == ("torch", "Size"):
                return tuple
            raise pickle.UnpicklingError(f"unsupported allowlisted global {module}.{name}")

        def persistent_load(self, pid: object) -> object:
            if not (isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage"):
                raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
            token = pid[1]
            if not isinstance(token, str) or token not in _STORAGE_DTYPES:
                raise pickle.UnpicklingError(f"unsupported storage token {token!r}")
            raw_key = pid[2]
            if isinstance(raw_key, bool) or not isinstance(raw_key, (str, int)):
                raise pickle.UnpicklingError("storage key must be a string or integer")
            key = str(raw_key)
            if not key or len(key) > 256 or any(char in key for char in "/\\\0"):
                raise pickle.UnpicklingError(f"invalid storage key {key!r}")
            numel = _require_plain_int(pid[4], f"storage {key!r} element count")
            if numel < 0:
                raise pickle.UnpicklingError(f"storage {key!r} has a negative element count")
            _dtype_name, itemsize = _STORAGE_DTYPES[token]
            if numel > limits.max_storage_bytes // itemsize:
                raise pickle.UnpicklingError(
                    f"storage {key!r} exceeds the {limits.max_storage_bytes}-byte limit"
                )
            incoming = _StorageRef(token, key, numel)
            existing = self.storage_meta.get(key)
            if existing is not None and existing != incoming:
                raise pickle.UnpicklingError(f"storage key {key!r} has conflicting declarations")
            if existing is None and len(self.storage_meta) >= limits.max_storages:
                raise pickle.UnpicklingError(
                    f"checkpoint exceeds the {limits.max_storages}-storage limit"
                )
            self.storage_meta[key] = incoming
            return existing or incoming

    unpickler = _Restricted(handle)
    unpickler.storage_meta = {}
    return unpickler


def _load_pickle(
    handle: BinaryIO,
    limits: CheckpointReadLimits,
) -> tuple[object, dict[str, _StorageRef]]:
    unpickler = _make_unpickler(handle, limits)
    try:
        tree = unpickler.load()
    except (
        AttributeError,
        EOFError,
        IndexError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as exc:
        raise CheckpointFormatError(f"restricted checkpoint load refused: {exc}") from exc
    return tree, unpickler.storage_meta


def _storage_from_bytes(token: str, raw: bytes) -> tuple[mx.array, bool]:

    dtype_name, _itemsize = _STORAGE_DTYPES[token]
    if dtype_name == "float64":
        doubles = array.array("d")
        doubles.frombytes(raw)
        return mlx_array_from_buffer(doubles, dtype=mx.float32), True
    flat = mlx_array_from_buffer(memoryview(raw))
    return (
        flat if dtype_name == "uint8" else flat.view(getattr(mx, dtype_name)),
        False,
    )


def _read_exact(handle: BinaryIO, size: int, label: str) -> bytes:
    data: bytes = handle.read(size)
    if len(data) != size:
        raise CheckpointFormatError(
            f"{label} was truncated: expected {size} bytes, got {len(data)}"
        )
    return data


def _read_zip_tree(
    source: Path,
    limits: CheckpointReadLimits,
) -> tuple[object, dict[str, mx.array], bool]:
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if len(infos) > limits.max_archive_members:
            raise CheckpointFormatError(
                f"checkpoint has {len(infos)} members; limit is {limits.max_archive_members}"
            )
        members: dict[str, list[zipfile.ZipInfo]] = {}
        for info in infos:
            members.setdefault(info.filename, []).append(info)
        pickles = [
            info
            for info in infos
            if info.filename == "data.pkl" or info.filename.endswith("/data.pkl")
        ]
        if len(pickles) != 1:
            detail = "no" if not pickles else "multiple"
            raise CheckpointFormatError(f"checkpoint carries {detail} data.pkl members")
        pickle_info = pickles[0]
        if pickle_info.flag_bits & 0x1:
            raise CheckpointFormatError("encrypted checkpoint metadata is unsupported")
        if pickle_info.file_size > limits.max_pickle_bytes:
            raise CheckpointFormatError(
                f"checkpoint pickle exceeds {limits.max_pickle_bytes} bytes"
            )
        pickle_data = archive.read(pickle_info)
        if len(pickle_data) != pickle_info.file_size:
            raise CheckpointFormatError("checkpoint pickle member was truncated")
        tree, storage_meta = _load_pickle(io.BytesIO(pickle_data), limits)
        prefix = pickle_info.filename[: -len("data.pkl")]
        storages: dict[str, mx.array] = {}
        total_bytes = 0
        demoted = False
        for key, storage in storage_meta.items():
            name = f"{prefix}data/{key}"
            matches = members.get(name, [])
            if len(matches) != 1:
                detail = "missing" if not matches else "duplicated"
                raise CheckpointFormatError(f"storage member {name!r} is {detail}")
            info = matches[0]
            if info.flag_bits & 0x1:
                raise CheckpointFormatError(f"encrypted storage member {name!r} is unsupported")
            if info.compress_type != zipfile.ZIP_STORED:
                raise CheckpointFormatError(f"storage member {name!r} must be uncompressed")
            _dtype_name, itemsize = _STORAGE_DTYPES[storage.token]
            expected = storage.numel * itemsize
            if info.file_size != expected or info.compress_size != expected:
                raise CheckpointFormatError(
                    f"storage member {name!r} has {info.file_size} bytes; "
                    f"metadata declares {expected}"
                )
            total_bytes += expected
            if total_bytes > limits.max_total_storage_bytes:
                raise CheckpointFormatError(
                    "checkpoint storages exceed the cumulative "
                    f"{limits.max_total_storage_bytes}-byte limit"
                )
            raw = archive.read(info)
            if len(raw) != expected:
                raise CheckpointFormatError(f"storage member {name!r} was truncated")
            try:
                storages[key], was_demoted = _storage_from_bytes(storage.token, raw)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise CheckpointFormatError(
                    f"cannot materialize storage {name!r} as {storage.token}: {exc}"
                ) from exc
            demoted = demoted or was_demoted
    return tree, storages, demoted


def _read_legacy_tree(
    source: Path,
    limits: CheckpointReadLimits,
) -> tuple[object, dict[str, mx.array], bool]:
    with source.open("rb") as handle:
        magic, _unused = _load_pickle(handle, limits)
        if magic != _LEGACY_MAGIC:
            raise CheckpointFormatError(
                "neither a zip-format nor a legacy torch checkpoint (bad magic)"
            )
        _protocol, _unused = _load_pickle(handle, limits)
        _sys_info, _unused = _load_pickle(handle, limits)
        tree, storage_meta = _load_pickle(handle, limits)
        keys, key_meta = _load_pickle(handle, limits)
        if key_meta:
            raise CheckpointFormatError("legacy storage-key pickle declares tensors")
        if not isinstance(keys, (list, tuple)):
            raise CheckpointFormatError("legacy storage keys must be a sequence")
        normalized_keys = [str(key) for key in keys]
        if len(normalized_keys) != len(set(normalized_keys)):
            raise CheckpointFormatError("legacy storage key list contains duplicates")
        if set(normalized_keys) != set(storage_meta):
            raise CheckpointFormatError(
                "legacy storage key list does not match tensor declarations"
            )
        storages: dict[str, mx.array] = {}
        total_bytes = 0
        demoted = False
        for key in normalized_keys:
            storage = storage_meta[key]
            (numel,) = struct.unpack(
                "<q",
                _read_exact(handle, 8, f"legacy storage {key!r} element count"),
            )
            if numel < 0 or numel != storage.numel:
                raise CheckpointFormatError(
                    f"storage {key!r}: tail carries {numel} elements, metadata "
                    f"declares {storage.numel}"
                )
            _dtype_name, itemsize = _STORAGE_DTYPES[storage.token]
            expected = numel * itemsize
            total_bytes += expected
            if total_bytes > limits.max_total_storage_bytes:
                raise CheckpointFormatError(
                    "checkpoint storages exceed the cumulative "
                    f"{limits.max_total_storage_bytes}-byte limit"
                )
            raw = _read_exact(handle, expected, f"legacy storage {key!r}")
            try:
                storages[key], was_demoted = _storage_from_bytes(storage.token, raw)
            except (RuntimeError, TypeError, ValueError) as exc:
                raise CheckpointFormatError(
                    f"cannot materialize legacy storage {key!r} as {storage.token}: {exc}"
                ) from exc
            demoted = demoted or was_demoted
    return tree, storages, demoted


@dataclass
class _ResolveBudget:
    nodes: int = 0
    tensors: int = 0
    tensor_bytes: int = 0


def _materialize(
    tensor: _LazyTensor,
    flat: mx.array,
    limits: CheckpointReadLimits,
) -> tuple[mx.array, int]:

    _dtype_name, source_itemsize = _STORAGE_DTYPES[tensor.storage.token]
    logical_itemsize = 4 if tensor.storage.token == "DoubleStorage" else source_itemsize
    max_elements = limits.max_tensor_bytes // logical_itemsize
    count = 1
    for axis, extent in enumerate(tensor.shape):
        if extent < 0:
            raise CheckpointFormatError(f"tensor extent {axis} is negative: {extent}")
        if extent > max_elements or (count and extent > max_elements // count):
            raise CheckpointFormatError(
                f"tensor shape {tensor.shape} exceeds the "
                f"{limits.max_tensor_bytes}-byte per-tensor limit"
            )
        count *= extent
    logical_bytes = count * logical_itemsize
    if count == 0:
        if not 0 <= tensor.offset <= tensor.storage.numel:
            raise CheckpointFormatError(
                f"empty tensor offset is outside storage {tensor.storage.key!r}"
            )
        return mx.zeros(tensor.shape, dtype=flat.dtype), logical_bytes
    minimum = tensor.offset
    maximum = tensor.offset
    for extent, stride in zip(tensor.shape, tensor.strides, strict=True):
        delta = (extent - 1) * stride
        minimum += min(delta, 0)
        maximum += max(delta, 0)
    if not (_INT64_MIN <= minimum <= _INT64_MAX and _INT64_MIN <= maximum <= _INT64_MAX):
        raise CheckpointFormatError("tensor index range exceeds int64")
    available = min(tensor.storage.numel, flat.size)
    if minimum < 0 or maximum >= available:
        raise CheckpointFormatError(
            f"tensor index range [{minimum}, {maximum}] is outside storage "
            f"{tensor.storage.key!r} with {available} elements"
        )
    contiguous = []
    accumulated = 1
    for extent in reversed(tensor.shape):
        contiguous.append(accumulated)
        accumulated *= extent
    if tensor.strides == tuple(reversed(contiguous)) or count == 1:
        value = flat[tensor.offset : tensor.offset + count].reshape(tensor.shape)
    else:
        index = mx.array(tensor.offset, dtype=mx.int64)
        for axis, (extent, step) in enumerate(zip(tensor.shape, tensor.strides, strict=True)):
            axis_shape = [1] * len(tensor.shape)
            axis_shape[axis] = extent
            index = index + (mx.arange(extent, dtype=mx.int64) * step).reshape(axis_shape)
        value = mx.take(flat, index.reshape(-1)).reshape(tensor.shape)
    return value, logical_bytes


def _resolve_tree(
    node: object,
    storages: dict[str, mx.array],
    limits: CheckpointReadLimits,
    budget: _ResolveBudget,
    active: set[int],
    depth: int = 0,
) -> object:
    budget.nodes += 1
    if budget.nodes > limits.max_tree_nodes:
        raise CheckpointFormatError("checkpoint object tree exceeds its node limit")
    if depth > limits.max_tree_depth:
        raise CheckpointFormatError(f"checkpoint object tree exceeds depth {limits.max_tree_depth}")
    if isinstance(node, _LazyTensor):
        budget.tensors += 1
        if budget.tensors > limits.max_tensors:
            raise CheckpointFormatError("checkpoint exceeds its tensor-count limit")
        try:
            flat = storages[node.storage.key]
        except KeyError as exc:
            raise CheckpointFormatError(
                f"tensor references missing storage {node.storage.key!r}"
            ) from exc
        value, logical_bytes = _materialize(node, flat, limits)
        budget.tensor_bytes += logical_bytes
        if budget.tensor_bytes > limits.max_total_tensor_bytes:
            raise CheckpointFormatError(
                "checkpoint tensors exceed the cumulative logical-byte limit"
            )
        return value
    if not isinstance(node, (dict, list, tuple)):
        return node
    identity = id(node)
    if identity in active:
        raise CheckpointFormatError("cyclic checkpoint object trees are unsupported")
    active.add(identity)
    try:
        if isinstance(node, dict):
            return type(node)(
                (
                    key,
                    _resolve_tree(
                        value,
                        storages,
                        limits,
                        budget,
                        active,
                        depth + 1,
                    ),
                )
                for key, value in node.items()
            )
        resolved = [
            _resolve_tree(value, storages, limits, budget, active, depth + 1) for value in node
        ]
        return tuple(resolved) if isinstance(node, tuple) else resolved
    finally:
        active.remove(identity)


@dataclass(frozen=True)
class RestrictedCheckpoint:
    """A reconstructed tensor tree plus its safety and dtype receipts."""

    tree: object
    demoted_fp64: bool
    flagged_globals: tuple[str, ...]


def load_restricted_checkpoint(
    source: Path | str,
    *,
    allow_suspicious: bool = False,
    limits: CheckpointReadLimits | None = None,
) -> RestrictedCheckpoint:
    """Scan and reconstruct one zip-format or legacy torch checkpoint."""
    path = Path(source)
    policy = limits or _DEFAULT_LIMITS
    try:
        references = scan_checkpoint_globals(
            path,
            max_pickle_bytes=policy.max_pickle_bytes,
        )
        flagged = suspicious_globals(references)
        if flagged and not allow_suspicious:
            raise CheckpointFormatError(
                f"checkpoint references globals outside the tensor-rebuild allowlist: {flagged}"
            )
        if zipfile.is_zipfile(path):
            tree, storages, demoted = _read_zip_tree(path, policy)
        else:
            tree, storages, demoted = _read_legacy_tree(path, policy)
        try:
            resolved = _resolve_tree(tree, storages, policy, _ResolveBudget(), set())
        except RecursionError as exc:
            raise CheckpointFormatError(
                "checkpoint object tree exceeds the safe recursion depth"
            ) from exc
    except CheckpointFormatError:
        raise
    except PickleScanError as exc:
        raise CheckpointFormatError(f"cannot statically scan checkpoint: {exc}") from exc
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise CheckpointFormatError(f"cannot read checkpoint {path}: {exc}") from exc
    return RestrictedCheckpoint(
        tree=resolved,
        demoted_fp64=demoted,
        flagged_globals=tuple(flagged),
    )


__all__ = [
    "CheckpointFormatError",
    "CheckpointReadLimits",
    "RestrictedCheckpoint",
    "load_restricted_checkpoint",
]
