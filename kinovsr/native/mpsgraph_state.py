"""Persistent-state, multi-procedure MPSGraph executables on the ANE.

MPSGraph's graph API has no public state operation, but the placed ANE
dialect does.  This module keeps that SPI behind a model-neutral contract:
callers describe named state tensors and ordinary MPSGraph programs; the
compiler moves their compute into ``mpsx.ane`` procedures while retaining
MPSGraph's canonical variable read/write sequence around each procedure. All
procedures then specialize into one ANE model and share the same Metal-backed
live state. The older placed ``anec.state`` product path remains available for
explicit ANECIR experiments.

The lowering deliberately operates on names, contracts, and def-use links.
It does not depend on operation numbers emitted by MPSGraph.  Family code
owns graph topology; this module owns state ABI construction, product
mapping, cache publication, and runtime buffers.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import logging
import mmap
import os
import platform
import plistlib
import re
import shutil
import struct
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

import mlx.core as mx

from . import mpsgraph as mg

_CACHE_FORMAT = 3
_MULTIPROCEDURE_CACHE_FORMAT = 5
_RESOURCE_HEADER = b"{-#\n  dialect_resources: {\n    mps: {\n"
_RESOURCE_FOOTER = b"\n    }\n  }\n#-}\n"
_MODULE_MARKER = b"module attributes "
_PLAIN_MODULE_MARKER = b"module {"
_WRITE_GUARD_ATTEMPTS = 4
_log = logging.getLogger("kinovsr.mpsgraph.state")
_operation_pointer_type: Any | None = None


def _write_probe_offsets(length: int, element: int) -> tuple[int, ...]:
    """Three aligned samples that prove a result backing was overwritten."""
    if length < element or length % element:
        raise ValueError(
            f"result backing has {length} bytes for {element}-byte elements")
    middle = ((length // 2) // element) * element
    return tuple(sorted({0, middle, length - element}))


def _run_with_write_guard(
    entry: str,
    element: int,
    nonce: int,
    probes: Sequence[tuple[str, Any, tuple[int, ...]]],
    dispatch: Callable[[], None],
) -> None:
    """Retry only MPSGraph's proven all-results-untouched failure.

    A mapped ANE product can return successfully while leaving every result
    buffer untouched during lazy program activation. Poisoning three words in
    each owned result distinguishes that failure from a healthy dispatch at
    negligible cost. Partial writes are ambiguous and therefore fail loudly;
    only an all-results-untouched call is safe to retry.
    """
    if not probes:
        dispatch()
        return
    for attempt in range(_WRITE_GUARD_ATTEMPTS):
        payload = (
            struct.pack("<H", 0x7E01 + ((nonce + attempt) & 0xFF))
            if element == 2
            else struct.pack(
                "<I", 0x7FC00001 + ((nonce + attempt) & 0xFFFF))
        )
        for _name, view, offsets in probes:
            for offset in offsets:
                view[offset:offset + element] = payload
        dispatch()

        untouched = []
        partial = []
        for name, view, offsets in probes:
            marked = [
                bytes(view[offset:offset + element]) == payload
                for offset in offsets
            ]
            if all(marked):
                untouched.append(name)
            elif any(marked):
                partial.append(name)
        if partial or (untouched and len(untouched) != len(probes)):
            names = sorted(set(partial) | set(untouched))
            raise RuntimeError(
                f"{entry}: MPSGraph returned incomplete result writes for "
                f"{names}; refusing stale output")
        if not untouched:
            if attempt:
                _log.warning(
                    "MPSGraph entry %s recovered after %d silent no-write "
                    "dispatches", entry, attempt)
            return
    raise RuntimeError(
        f"{entry}: MPSGraph returned without writing any result on "
        f"{_WRITE_GUARD_ATTEMPTS} consecutive dispatches")


class _ModuleOp(ctypes.Structure):
    _fields_ = [("operation", ctypes.c_void_p)]


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("filename", ctypes.c_char_p),
        ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p),
        ("symbol_address", ctypes.c_void_p),
    ]


@dataclass(frozen=True)
class StateTensorSpec:
    """One logical recurrent tensor and its ANE-safe physical backing."""

    name: str
    logical_shape: tuple[int, ...]
    storage_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        logical = tuple(int(value) for value in self.logical_shape)
        storage = tuple(int(value) for value in self.storage_shape)
        object.__setattr__(self, "logical_shape", logical)
        object.__setattr__(self, "storage_shape", storage)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise ValueError(f"invalid MPSGraph state name {self.name!r}")
        if len(logical) != 4 or any(value < 1 for value in logical):
            raise ValueError(
                f"MPSGraph state needs a positive rank-four shape: {logical}")
        if len(storage) != 4 or any(value < 1 for value in storage):
            raise ValueError(
                f"MPSGraph state storage must be positive rank four: {storage}")
        if any(value > 255 for value in storage):
            raise ValueError(
                f"MPSGraph state storage exceeds the ANE dimension limit: "
                f"{storage}")
        if prod(logical) != prod(storage):
            raise ValueError(
                f"MPSGraph state storage changes element count: "
                f"{logical} -> {storage}")

    @classmethod
    def create(
        cls, name: str, logical_shape: Sequence[int]
    ) -> StateTensorSpec:
        logical = tuple(int(value) for value in logical_shape)
        return cls(name, logical, safe_storage_shape(logical))


@dataclass
class Program:
    """An ordinary graph plus the results that update named state tensors."""

    name: str
    builder: mg.GraphBuilder
    targets: list[tuple[str, Any, tuple[int, ...]]]
    state_results: Mapping[str, str]
    dynamic: set[str]


@dataclass(frozen=True)
class _EntryContract:
    name: str
    function: str
    region: str
    order: tuple[tuple[str, tuple[int, ...]], ...]
    targets: tuple[tuple[str, tuple[int, ...]], ...]
    state_results: tuple[tuple[str, str], ...]
    ane_input_order: tuple[str, ...]
    ane_output_order: tuple[str, ...]
    dynamic: frozenset[str]
    product: str

    @property
    def state_result_names(self) -> frozenset[str]:
        return frozenset(name for name, _state in self.state_results)

    @property
    def runtime_targets(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        removed = self.state_result_names
        return tuple(item for item in self.targets if item[0] not in removed)


def safe_storage_shape(shape: Sequence[int], *, limit: int = 255) -> tuple[int, ...]:
    """Fold oversized axes into axis zero while preserving four-D layout.

    ANE ring buffers reject physical dimensions above 255.  A reshape is
    sufficient because state is opaque storage: peel integer factors from
    C/H/W into N until every physical dimension fits.  The BSVD production
    shapes become the measured-safe ``2x144x240x160`` and
    ``2x144x120x160`` forms.
    """
    values = [int(value) for value in shape]
    if len(values) != 4 or any(value < 1 for value in values):
        raise ValueError(f"MPSGraph state needs a positive rank-four shape: {shape}")
    if limit < 2:
        raise ValueError("MPSGraph state dimension limit must be at least two")
    leading = values[0]
    for axis in range(1, 4):
        while values[axis] > limit:
            factor = next(
                (candidate for candidate in range(2, values[axis] + 1)
                 if values[axis] % candidate == 0),
                None,
            )
            if factor is None:
                raise ValueError(
                    f"cannot factor state axis {values[axis]} below {limit}")
            leading *= factor
            values[axis] //= factor
    values[0] = leading
    if any(value > limit for value in values):
        raise ValueError(
            f"state shape {tuple(shape)} has no four-D storage below {limit}: "
            f"{tuple(values)}")
    if prod(values) != prod(int(value) for value in shape):
        raise AssertionError("state storage reshape changed element count")
    return tuple(values)


def state_placeholders(
    builder: mg.GraphBuilder, states: Sequence[StateTensorSpec]
) -> dict[str, Any]:
    """Create physical state feeds and return their logical graph views."""
    return {
        state.name: builder.reshape(
            builder.placeholder(state.storage_shape, state.name),
            state.logical_shape,
            state.name + ".logical",
        )
        for state in states
    }


def state_result(
    builder: mg.GraphBuilder, state: StateTensorSpec, value: Any
) -> tuple[str, Any, tuple[int, ...]]:
    """Expose a logical update through the state's physical result shape."""
    name = state.name + ".next"
    return (
        name,
        builder.reshape(value, state.storage_shape, name + ".storage"),
        state.storage_shape,
    )


class TensorBinding:
    """A reusable entry-specific tensor-data view over shared Metal storage."""

    __slots__ = ("_owner", "_data", "_buffer", "_view", "shape")

    def __init__(
        self,
        owner: StatefulEntry,
        data: Any,
        buffer: Any,
        view: Any,
        shape: tuple[int, ...],
    ):
        self._owner = owner
        self._data = data
        self._buffer = buffer
        self._view = view
        self.shape = shape

    def write(self, value: Any) -> None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            payload = value
        else:
            actual = tuple(int(item) for item in value.shape)
            if actual != self.shape:
                raise ValueError(
                    f"binding expects shape {self.shape}, got {actual}")
            payload = memoryview(
                mx.contiguous(value.astype(self._owner.mx_dtype))).cast("B")
        if len(payload) != len(self._view):
            raise ValueError(
                f"binding expects {len(self._view)} bytes, got {len(payload)}")
        self._view[:] = payload

    def array(self) -> Any:
        return mx.array(bytes(self._view)).view(
            self._owner.mx_dtype).reshape(self.shape)


class StatefulEntry:
    """One named entry in a shared persistent-state executable."""

    def __init__(
        self,
        owner: StatefulExecutable,
        contract: _EntryContract,
        state_data: Mapping[str, Any],
    ):
        self._owner = owner
        self._contract = contract
        self._exe = owner._exe
        self._queue = owner._queue
        self.dtype = owner.dtype
        self.mx_dtype = owner.mx_dtype
        self._element = owner._element
        self._parity = 0
        self._buffers: list[Any] = []
        self._data_buffers: dict[int, Any] = {}

        state_names = set(state_data)
        self._feed_shapes = dict(contract.order)
        self._target_shapes = dict(contract.runtime_targets)
        dynamic = set(contract.dynamic) - state_names - contract.state_result_names
        unknown = dynamic - (set(self._feed_shapes) | set(self._target_shapes))
        if unknown:
            raise ValueError(
                f"{contract.name}: unknown dynamic tensors {sorted(unknown)}")

        self._writes: list[list[tuple[str, Any]]] = [[], []]
        self._feed_data: list[list[Any]] = [[], []]
        self._dynamic_feeds: dict[str, int] = {}
        for name, shape in contract.order:
            if name in state_data:
                data = state_data[name]
                self._feed_data[0].append(data)
                self._feed_data[1].append(data)
            elif name in dynamic:
                self._dynamic_feeds[name] = len(self._feed_data[0])
                self._feed_data[0].append(None)
                self._feed_data[1].append(None)
            else:
                pair = (self._backing(shape), self._backing(shape))
                for parity in (0, 1):
                    data, _buffer, view = pair[parity]
                    self._feed_data[parity].append(data)
                    self._writes[parity].append((name, view))

        self._result_data: list[list[Any]] = [[], []]
        self._dynamic_results: dict[str, int] = {}
        self._reads: dict[str, tuple[Any, tuple[int, ...]]] = {}
        for name, shape in contract.runtime_targets:
            if name in dynamic:
                self._dynamic_results[name] = len(self._result_data[0])
                self._result_data[0].append(None)
                self._result_data[1].append(None)
            else:
                data, _buffer, view = self._backing(shape)
                self._result_data[0].append(data)
                self._result_data[1].append(data)
                self._reads[name] = (view, shape)

        fw = mg._fw()
        Execution = fw["objc"].lookUpClass(
            "MPSGraphExecutableExecutionDescriptor")
        self._execution = Execution.alloc().init()
        self._execution.setEntryFunctionName_(self._contract.function)
        # The dispatch worker's completion is the pipeline's dependency edge.
        # Without this, runWithMTLCommandQueue only enqueues work; callers can
        # reuse shared state/results before ANE completion and the write guard
        # races the device under GPU contention.
        self._execution.setWaitUntilCompleted_(True)
        if self._owner._entry_map is not None:
            self._execution.setPerEntryPointToSymbolAndFileNameMap_(
                self._owner._entry_map)
        self._dispatch_nonce = 0
        self._write_probes = tuple(
            (
                name,
                view,
                _write_probe_offsets(len(view), self._element),
            )
            for name, (view, _shape) in self._reads.items()
        )

    @property
    def feed_names(self) -> tuple[str, ...]:
        return tuple(name for name, _shape in self._contract.order)

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(name for name, _shape in self._contract.runtime_targets)

    def _backing(self, shape: tuple[int, ...]) -> tuple[Any, Any, Any]:
        size = self._element * prod(shape)
        buffer = self._owner._metal.newBufferWithLength_options_(size, 0)
        if buffer is None:
            raise MemoryError(
                f"Metal could not allocate {size} bytes for MPSGraph entry")
        view = buffer.contents().as_buffer(size)
        data = mg._fw()["MPSGraphTensorData"].alloc(
        ).initWithMTLBuffer_shape_dataType_(buffer, list(shape), self.dtype)
        self._buffers.append(buffer)
        self._data_buffers[id(data)] = buffer
        return data, buffer, view

    def bind(
        self, name: str, *, shared: TensorBinding | None = None
    ) -> TensorBinding:
        shape = self._feed_shapes.get(name, self._target_shapes.get(name))
        if shape is None:
            raise KeyError(f"unknown entry tensor '{name}'")
        if name not in self._dynamic_feeds and name not in self._dynamic_results:
            raise ValueError(f"entry tensor '{name}' is not dynamic")
        if shared is None:
            data, buffer, view = self._backing(shape)
        else:
            if not isinstance(shared, TensorBinding):
                raise TypeError("shared backing must be a stateful TensorBinding")
            if shared.shape != shape or shared._owner.dtype != self.dtype:
                raise ValueError(
                    f"shared binding for '{name}' is incompatible with {shape}")
            buffer, view = shared._buffer, shared._view
            self._buffers.append(buffer)
            data = mg._fw()["MPSGraphTensorData"].alloc(
            ).initWithMTLBuffer_shape_dataType_(
                buffer, list(shape), self.dtype)
            self._data_buffers[id(data)] = buffer
        return TensorBinding(self, data, buffer, view, shape)

    def write_feeds(self, values: Mapping[str, Any]) -> None:
        for name, view in self._writes[self._parity]:
            value = values[name]
            actual = tuple(int(item) for item in value.shape)
            expected = self._feed_shapes[name]
            if actual != expected:
                raise ValueError(
                    f"feed '{name}' expects {expected}, got {actual}")
            view[:] = memoryview(
                mx.contiguous(value.astype(self.mx_dtype))).cast("B")

    def begin_dispatch(
        self, bindings: Mapping[str, TensorBinding] | None = None
    ):
        bindings = bindings or {}
        required = set(self._dynamic_feeds) | set(self._dynamic_results)
        missing = required - set(bindings)
        extra = set(bindings) - required
        if missing:
            raise ValueError(
                f"missing dynamic tensor bindings: {sorted(missing)}")
        if extra:
            raise ValueError(
                f"unexpected dynamic tensor bindings: {sorted(extra)}")
        parity = self._parity
        self._parity = 1 - parity
        feeds = list(self._feed_data[parity])
        results = list(self._result_data[parity])
        for name, index in self._dynamic_feeds.items():
            binding = bindings[name]
            self._check_binding(name, binding, self._feed_shapes[name])
            feeds[index] = binding._data
        for name, index in self._dynamic_results.items():
            binding = bindings[name]
            self._check_binding(name, binding, self._target_shapes[name])
            results[index] = binding._data
        exe = self._exe
        queue = self._queue
        execution = self._execution
        self._dispatch_nonce += 1
        nonce = self._dispatch_nonce
        probes = self._write_probes
        entry = self._contract.name
        element = self._element
        pool = mg._fw()["objc"].autorelease_pool

        def job() -> None:
            with pool():
                def dispatch() -> None:
                    exe.runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor_(
                        queue, feeds, results, execution)

                _run_with_write_guard(
                    entry, element, nonce, probes, dispatch)

        return job

    def _check_binding(
        self, name: str, binding: TensorBinding, shape: tuple[int, ...]
    ) -> None:
        if not isinstance(binding, TensorBinding) or binding._owner is not self:
            raise ValueError(f"binding for '{name}' belongs to another entry")
        if binding.shape != shape:
            raise ValueError(
                f"binding for '{name}' expects {shape}, got {binding.shape}")

    def read(self, wanted: set[str] | None = None) -> dict[str, Any]:
        output = {}
        for name, (view, shape) in self._reads.items():
            if wanted is not None and name not in wanted:
                continue
            output[name] = mx.array(bytes(view)).view(
                self.mx_dtype).reshape(shape)
        return output

    def reset(self) -> None:
        self._parity = 0


class StatefulExecutable:
    """One executable, one ANE identity, and persistent state across entries."""

    def __init__(
        self,
        executable: Any,
        metal: Any,
        dtype: int,
        states: Sequence[StateTensorSpec],
        contracts: Sequence[_EntryContract],
        entry_map: Any | None,
    ):
        self._exe = executable
        self._metal = metal
        self._queue = metal.newCommandQueue()
        self.dtype = dtype
        self.mx_dtype = mg._MX_OF[dtype]
        self._element = 2 if dtype == mg.FLOAT16 else 4
        self._entry_map = entry_map
        self._states = tuple(states)
        self._prepare_lock = threading.Lock()
        self._prepared = False
        self._state_buffers: list[Any] = []
        self._state_views: list[Any] = []
        state_data = {}
        TensorData = mg._fw()["MPSGraphTensorData"]
        for state in states:
            size = self._element * prod(state.storage_shape)
            buffer = metal.newBufferWithLength_options_(size, 0)
            if buffer is None:
                raise MemoryError(
                    f"Metal could not allocate {size} bytes for state {state.name}")
            view = buffer.contents().as_buffer(size)
            view[:] = bytes(size)
            self._state_buffers.append(buffer)
            self._state_views.append(view)
            state_data[state.name] = TensorData.alloc(
            ).initWithMTLBuffer_shape_dataType_(
                buffer, list(state.storage_shape), dtype)
        self._entries = {
            contract.name: StatefulEntry(self, contract, state_data)
            for contract in contracts
        }
        self._closed = False

    @property
    def state_specs(self) -> tuple[StateTensorSpec, ...]:
        return self._states

    def entry(self, name: str) -> StatefulEntry:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"unknown MPSGraph state entry '{name}'") from exc

    def prepare(self) -> None:
        """Finish any runtime-specific ANE attachment exactly once.

        Mapped-product executables need an eager attachment call; grouped
        executables were already specialized as one model while loading.
        The method remains idempotent for prepare-edge callers.
        """
        with self._prepare_lock:
            if self._closed:
                raise RuntimeError("MPSGraph state executable is closed")
            if self._prepared:
                return
            if self._entry_map is None:
                # Multiprocedure executables are specialized as one model at
                # load time; there is no per-entry product map to attach.
                self._prepared = True
                return
            fw = mg._fw()
            device = fw["MPSGraphDevice"].deviceWithMTLDevice_(self._metal)
            compilation = fw["MPSGraphCompilationDescriptor"].alloc().init()
            compilation.setOptimizationLevel_(0)
            compilation.setPreferredDevice_(mg.DEVICE_GPU)
            compilation.setWaitForCompilationCompletion_(True)
            self._exe.applyEntryPointToSymbolAndFileNameMap_device_compilationDescriptor_(
                self._entry_map, device, compilation)
            self._prepared = True

    def reset(self) -> None:
        for view in self._state_views:
            view[:] = bytes(len(view))
        for entry in self._entries.values():
            entry.reset()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._entries.clear()
        self._state_views.clear()
        self._state_buffers.clear()
        self._states = ()


def _split_top_level(value: str) -> list[str]:
    """Split a comma list while respecting MLIR's nested delimiters."""
    parts = []
    start = 0
    stack: list[str] = []
    pairs = {"<": ">", "(": ")", "[": "]", "{": "}"}
    quote = False
    escaped = False
    for index, character in enumerate(value):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quote = False
            continue
        if character == '"':
            quote = True
        elif character in pairs:
            stack.append(pairs[character])
        elif stack and character == stack[-1]:
            stack.pop()
        elif character == "," and not stack:
            parts.append(value[start:index].strip())
            start = index + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _balanced_end(value: str, start: int) -> int:
    opening = value[start]
    closing = {"(": ")", "[": "]", "<": ">", "{": "}"}.get(opening)
    if closing is None:
        raise ValueError(f"not a balanced delimiter at {start}: {opening!r}")
    stack = [closing]
    quote = False
    escaped = False
    for index in range(start + 1, len(value)):
        character = value[index]
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quote = False
            continue
        if character == '"':
            quote = True
        elif character in "([<{":
            stack.append({"(": ")", "[": "]", "<": ">", "{": "}"}[character])
        elif stack and character == stack[-1]:
            stack.pop()
            if not stack:
                return index
    raise ValueError(f"unterminated MLIR delimiter at {start}")


def _replace_span(value: str, start: int, end: int, replacement: str) -> str:
    return value[:start] + replacement + value[end:]


def _signature_spans(line: str, marker: str) -> tuple[tuple[int, int], tuple[int, int]]:
    marker_at = line.index(marker) + len(marker)
    args_start = line.index("(", marker_at)
    args_end = _balanced_end(line, args_start)
    arrow = line.index("->", args_end)
    results_start = line.index("(", arrow)
    results_end = _balanced_end(line, results_start)
    return (args_start + 1, args_end), (results_start + 1, results_end)


def _operation_spans(line: str, operation: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    operation_at = line.index(operation)
    operands_start = line.index("(", operation_at + len(operation))
    operands_end = _balanced_end(line, operands_start)
    type_marker = line.index(" : (", operands_end)
    inputs_start = type_marker + 3
    inputs_end = _balanced_end(line, inputs_start)
    arrow = line.index("->", inputs_end)
    outputs_start = line.index("(", arrow)
    outputs_end = _balanced_end(line, outputs_start)
    return (
        (operands_start + 1, operands_end),
        (inputs_start + 1, inputs_end),
        (outputs_start + 1, outputs_end),
    )


def _state_attrs(indices: Sequence[int]) -> str:
    return "mps.stateInputIndices = array<i64: " + ", ".join(
        str(index) for index in indices) + ">"


def _ring_attributes(shape: tuple[int, ...]) -> str:
    values = ", ".join(str(value) for value in shape)
    return (
        "{is_dynamic_offsets = dense<0> : tensor<4xui8>, "
        "offsets = dense<0> : tensor<4xui64>, "
        f"slice_size = dense<[{values}]> : tensor<4xui64>}}"
    )


def _parse_return(line: str) -> tuple[list[str], list[str], tuple[int, int], tuple[int, int]]:
    stripped = line.lstrip()
    if not stripped.startswith("return "):
        raise ValueError("not an MLIR return")
    value_start = line.index("return ") + len("return ")
    type_marker = line.index(" : ", value_start)
    values = _split_top_level(line[value_start:type_marker])
    types_start = type_marker + 3
    types = _split_top_level(line[types_start:].strip())
    return values, types, (value_start, type_marker), (types_start, len(line.rstrip("\n")))


def _tensor_declaration(
    declaration: str, *, function: str
) -> tuple[str, str]:
    """Return the SSA name and bare tensor type from a function argument."""
    formal, separator, remainder = declaration.partition(":")
    formal = formal.strip()
    if not separator or not re.fullmatch(r"%[A-Za-z0-9_.$-]+", formal):
        raise RuntimeError(
            f"{function}: cannot parse function argument {declaration!r}")
    remainder = remainder.lstrip()
    if not remainder.startswith("tensor<"):
        raise RuntimeError(
            f"{function}: function argument is not a tensor: {declaration!r}")
    angle = remainder.index("<")
    end = _balanced_end(remainder, angle)
    return formal, remainder[:end + 1]


def _bare_tensor_type(value: str, *, function: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("tensor<"):
        raise RuntimeError(
            f"{function}: result is not a tensor type: {value!r}")
    end = _balanced_end(stripped, stripped.index("<"))
    trailing = stripped[end + 1:].strip()
    if trailing:
        raise RuntimeError(
            f"{function}: unsupported result type suffix {trailing!r}")
    return stripped[:end + 1]


def _memref_type(tensor_type: str) -> str:
    if not tensor_type.startswith("tensor<"):
        raise ValueError(f"not a tensor type: {tensor_type!r}")
    return "memref<" + tensor_type[len("tensor<"):]


def _formatted_result_types(types: Sequence[str]) -> str:
    if len(types) == 1:
        return types[0]
    return "(" + ", ".join(types) + ")"


def _transform_multiprocedure_body(
    body: str,
    *,
    function: str,
    family: str,
    order: Sequence[tuple[str, tuple[int, ...]]],
    targets: Sequence[tuple[str, tuple[int, ...]]],
    states: Mapping[str, StateTensorSpec],
    state_results: Mapping[str, str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Move one high-level graph behind an ``mpsx.ane`` procedure.

    ``mpsx.ane`` is MPSGraph's multiprocedure compilation boundary. State
    variables deliberately remain around that boundary in the same
    read-region-slice-update-assign order emitted for Core ML ``MLState``.
    Keeping variable operations out of the ANE procedure avoids adding their
    full state tensors to its internal pressure analysis.
    """
    marker = f"func.func @{function}"
    if body.count(marker) != 1 or body.count("func.func @") != 1:
        raise RuntimeError(
            f"{function}: expected exactly one captured graph function")
    if "mpsx.ane" in body or "%kst_" in body:
        raise RuntimeError(f"{function}: captured graph uses reserved MPSX names")

    function_at = body.index(marker)
    args_start = body.index("(", function_at + len(marker))
    args_end = _balanced_end(body, args_start)
    declarations = _split_top_level(body[args_start + 1:args_end])
    parsed_args = [
        _tensor_declaration(value, function=function)
        for value in declarations
    ]
    if len(parsed_args) != len(order):
        raise RuntimeError(
            f"{function}: feed contract has {len(order)} values but IR has "
            f"{len(parsed_args)}")

    cursor = args_end + 1
    while cursor < len(body) and body[cursor].isspace():
        cursor += 1
    if not body.startswith("->", cursor):
        raise RuntimeError(f"{function}: captured result signature is missing")
    cursor += 2
    while cursor < len(body) and body[cursor].isspace():
        cursor += 1
    if cursor >= len(body):
        raise RuntimeError(f"{function}: captured result signature is truncated")
    if body[cursor] == "(":
        cursor = _balanced_end(body, cursor) + 1
    elif body.startswith("tensor<", cursor):
        cursor = _balanced_end(body, body.index("<", cursor)) + 1
    else:
        raise RuntimeError(f"{function}: captured result is not a tensor")
    while cursor < len(body) and body[cursor].isspace():
        cursor += 1
    if body.startswith("attributes", cursor):
        attributes = body.index("{", cursor + len("attributes"))
        cursor = _balanced_end(body, attributes) + 1
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
    if cursor >= len(body) or body[cursor] != "{":
        raise RuntimeError(f"{function}: captured function body is missing")
    function_end = _balanced_end(body, cursor)
    original_body = body[cursor + 1:function_end]

    returns = list(re.finditer(
        r"(?m)^[ \t]*return(?:[ \t].*)?$", original_body))
    if len(returns) != 1:
        raise RuntimeError(
            f"{function}: expected one captured return, got {len(returns)}")
    return_match = returns[0]
    if original_body[return_match.end():].strip():
        raise RuntimeError(f"{function}: operations follow the captured return")
    return_line = original_body[return_match.start():return_match.end()]
    return_values, raw_return_types, _values, _types = _parse_return(
        return_line + "\n")
    return_types = [
        _bare_tensor_type(value, function=function)
        for value in raw_return_types
    ]
    if len(return_values) != len(targets) or len(return_types) != len(targets):
        raise RuntimeError(
            f"{function}: target contract has {len(targets)} values but IR has "
            f"{len(return_values)}")

    feed_names = [name for name, _shape in order]
    if len(set(feed_names)) != len(feed_names):
        raise RuntimeError(f"{function}: feed contract names are not unique")
    missing_states = set(states) - set(feed_names)
    if missing_states:
        raise RuntimeError(
            f"{function}: missing state feeds {sorted(missing_states)}")
    target_names = [name for name, _shape in targets]
    unknown_results = set(state_results) - set(target_names)
    if unknown_results:
        raise RuntimeError(
            f"{function}: missing state results {sorted(unknown_results)}")
    result_positions = {
        index: state_results[name]
        for index, name in enumerate(target_names)
        if name in state_results
    }
    visible_positions = [
        index for index in range(len(targets))
        if index not in result_positions
    ]
    if not visible_positions:
        raise RuntimeError(
            f"{function}: mpsx.ane needs at least one visible result")

    input_memrefs = [
        _memref_type(value_type) for _formal, value_type in parsed_args]
    state_positions = {
        name: index for index, name in enumerate(feed_names) if name in states
    }
    input_lines: list[str] = []
    for index, ((formal, tensor_type), memref_type) in enumerate(
        zip(parsed_args, input_memrefs, strict=True)
    ):
        block = f"%kst_input_{index}"
        input_lines.append(
            f'    {formal} = "placement.ane_io_cast"({block}) : '
            f'({memref_type}) -> {tensor_type}')

    operations = original_body[:return_match.start()]
    operations = operations.strip("\n")

    for position, state_name in sorted(result_positions.items()):
        if state_name not in state_positions:
            raise RuntimeError(
                f"{function}: state result refers to unknown feed {state_name!r}")
        input_type = parsed_args[state_positions[state_name]][1]
        if return_types[position] != input_type:
            raise RuntimeError(
                f"{function}: state result {target_names[position]!r} has type "
                f"{return_types[position]}, expected {input_type}")

    visible_types = [return_types[index] for index in visible_positions]
    result_memrefs = [_memref_type(value) for value in return_types]
    output_lines = [
        f'    %kst_output_{index} = "placement.ane_io_cast"('
        f'{return_values[index]}) : ({return_types[index]}) -> '
        f'{result_memrefs[index]}'
        for index in range(len(return_values))
    ]
    output_values = [
        f"%kst_output_{index}" for index in range(len(return_values))]
    output_lines.append(
        f'    "mpsx.region_return"({", ".join(output_values)}) : '
        f' ({", ".join(result_memrefs)}) -> ()')

    symbol = f"{function}_ANE_region_0_0"
    block_args = ", ".join(
        f"%kst_input_{index}: {value_type}"
        for index, value_type in enumerate(input_memrefs)
    )
    region_lines = [
        '  "mpsx.ane"() ({',
        f"  ^bb0({block_args}):",
        *input_lines,
    ]
    if operations:
        region_lines.append(operations)
    region_lines.extend(output_lines)
    region_lines.append(
        f'  }}) {{ane_family = "{family}", function_type = '
        f'({", ".join(input_memrefs)}) -> '
        f'{_formatted_result_types(result_memrefs)}, '
        f'sym_name = "{symbol}"}} : () -> ()'
    )

    outer_lines = [
        '    %kst_one = "mps.constant"() '
        '<{value = dense<1> : tensor<4xsi32>}> : '
        '() -> tensor<4xsi32>',
        '    %kst_zero = "mps.constant"() '
        '<{value = dense<0> : tensor<4xsi32>}> : '
        '() -> tensor<4xsi32>',
    ]
    state_variables: dict[str, str] = {}
    state_reads: dict[str, str] = {}
    for index, ((formal, tensor_type), memref_type) in enumerate(
        zip(parsed_args, input_memrefs, strict=True)
    ):
        feed_name = feed_names[index]
        source = formal
        if feed_name in states:
            variable = f"%kst_state_var_{index}"
            read = f"%kst_state_read_{index}"
            outer_lines.extend((
                f'    {variable} = "mps.variable_from_tensor"({formal}) : '
                f'({tensor_type}) -> {tensor_type}',
                f'    {read} = "mps.read_variable"({variable}) : '
                f'({tensor_type}) -> {tensor_type}',
            ))
            state_variables[feed_name] = variable
            state_reads[feed_name] = read
            source = read
        outer_lines.append(
            f'    %kst_outer_input_{index} = "placement.tensor_to_memref"('
            f'{source}) : ({tensor_type}) -> {memref_type}')
    call_count = len(return_values)
    call_lhs = "%kst_call" if call_count == 1 else f"%kst_call:{call_count}"
    outer_operands = ", ".join(
        f"%kst_outer_input_{index}" for index in range(len(parsed_args)))
    outer_lines.append(
        f'    {call_lhs} = "placement.region_call"({outer_operands}) '
        f'{{callee = @{symbol}, mps.regionSHA = "KINO_MPSX_{function}", '
        'region_type = #placement.region_type<ANE>} : '
        f' ({", ".join(input_memrefs)}) -> '
        f'{_formatted_result_types(result_memrefs)}')
    outer_results = []
    for index, (memref_type, tensor_type) in enumerate(
        zip(result_memrefs, return_types, strict=True)):
        source = "%kst_call" if call_count == 1 else f"%kst_call#{index}"
        result = f"%kst_result_{index}"
        outer_results.append(result)
        outer_lines.append(
            f'    {result} = "placement.memref_to_tensor"({source}) : '
            f' ({memref_type}) -> {tensor_type}')
    for position, state_name in sorted(result_positions.items()):
        input_type = parsed_args[state_positions[state_name]][1]
        updated = f"%kst_state_update_{position}"
        outer_lines.extend((
            f'    {updated} = "mps.strided_slice_update"('
            f'{state_reads[state_name]}, {outer_results[position]}, '
            '%kst_zero, %kst_zero, %kst_one) '
            '<{begin_mask = 14 : ui32, end_mask = 15 : ui32, '
            'shrink_axis_mask = 0 : ui32}> : '
            f'({input_type}, {input_type}, tensor<4xsi32>, '
            f'tensor<4xsi32>, tensor<4xsi32>) -> {input_type}',
            f'    "mps.assign_variable"({state_variables[state_name]}, '
            f'{updated}) : ({input_type}, {input_type}) -> ()',
        ))
    visible_results = [outer_results[index] for index in visible_positions]
    outer_lines.append(
        f'    return {", ".join(visible_results)} : '
        f'{", ".join(visible_types)}')
    state_indices = [
        index for index, name in enumerate(feed_names) if name in states]
    outer_header = (
        f"  func.func @{function}({', '.join(declarations)}) -> "
        f"{_formatted_result_types(visible_types)} attributes "
        f"{{{_state_attrs(state_indices)}}} {{"
    )
    replacement = "\n".join((
        *region_lines,
        outer_header,
        *outer_lines,
        "  }",
    ))
    transformed = body[:function_at] + replacement + body[function_end + 1:]
    return (
        transformed,
        tuple(feed_names),
        tuple(target_names[index] for index in visible_positions),
    )


def _transform_placed_body(
    body: str,
    *,
    function: str,
    order: Sequence[tuple[str, tuple[int, ...]]],
    targets: Sequence[tuple[str, tuple[int, ...]]],
    states: Mapping[str, StateTensorSpec],
    state_results: Mapping[str, str],
    ane_input_order: list[str] | None = None,
    ane_output_order: list[str] | None = None,
) -> str:
    """Lower contracted ports in one placed module body to ``anec.state``."""
    lines = body.splitlines(keepends=True)
    function_line = next(
        (index for index, line in enumerate(lines)
         if line.startswith(f"  func.func @{function}(")),
        None,
    )
    region_line = next(
        (index for index, line in enumerate(lines)
         if line.startswith("  anec.A14 @")),
        None,
    )
    if function_line is None or region_line is None:
        raise RuntimeError(f"{function}: placed function or ANE region is missing")

    state_input_indices = [
        index for index, (name, _shape) in enumerate(order) if name in states
    ]
    feed_names = {name for name, _shape in order}
    if feed_names & set(states) != set(states):
        missing = set(states) - feed_names
        raise RuntimeError(f"{function}: missing state feeds {sorted(missing)}")
    result_positions = {
        index: state_results[name]
        for index, (name, _shape) in enumerate(targets)
        if name in state_results
    }
    target_names = {name for name, _shape in targets}
    if set(state_results) - target_names:
        missing = set(state_results) - target_names
        raise RuntimeError(f"{function}: missing state results {sorted(missing)}")

    # Find the outer return and map contracted result positions onto the
    # placement.region_call result indices that supply them.
    return_line = next(
        (index for index in range(len(lines) - 1, function_line, -1)
         if lines[index].lstrip().startswith("return ")),
        None,
    )
    if return_line is None:
        raise RuntimeError(f"{function}: outer return is missing")
    return_values, return_types, value_span, type_span = _parse_return(
        lines[return_line])
    if len(return_values) != len(targets) or len(return_types) != len(targets):
        raise RuntimeError(
            f"{function}: target contract has {len(targets)} values but IR has "
            f"{len(return_values)}")

    state_return_values = {
        return_values[position]: state_name
        for position, state_name in result_positions.items()
    }
    target_for_return_value = {
        value: targets[position][0]
        for position, value in enumerate(return_values)
    }
    call_output_for_state: dict[int, str] = {}
    call_output_names: dict[int, str] = {}
    conversion_lines: set[int] = set()
    conversion_pattern = re.compile(
        r'^\s*(%[A-Za-z0-9_.$-]+) = "placement\.memref_to_tensor"'
        r'\((%[A-Za-z0-9_.$-]+)#([0-9]+)\)')
    call_base = None
    for index in range(function_line + 1, len(lines)):
        match = conversion_pattern.match(lines[index])
        if match is None or match.group(1) not in target_for_return_value:
            continue
        base = match.group(2)
        if call_base is None:
            call_base = base
        elif call_base != base:
            raise RuntimeError(f"{function}: state results use multiple region calls")
        position = int(match.group(3))
        call_output_names[position] = target_for_return_value[match.group(1)]
        if match.group(1) in state_return_values:
            call_output_for_state[position] = state_return_values[
                match.group(1)]
            conversion_lines.add(index)
    if len(call_output_for_state) != len(result_positions):
        raise RuntimeError(
            f"{function}: resolved {len(call_output_for_state)} of "
            f"{len(result_positions)} state result links")

    call_line = next(
        (index for index in range(function_line + 1, len(lines))
         if '"placement.region_call"' in lines[index]),
        None,
    )
    if call_line is None:
        raise RuntimeError(f"{function}: placement region call is missing")
    lhs = re.match(r"^(\s*)(%[A-Za-z0-9_.$-]+)(?::([0-9]+))? =", lines[call_line])
    if lhs is None:
        raise RuntimeError(f"{function}: cannot parse region call result")
    actual_call_base = lhs.group(2)
    if call_base is not None and call_base != actual_call_base:
        raise RuntimeError(f"{function}: region call result link changed")
    call_base = actual_call_base
    operand_span, call_input_span, call_output_span = _operation_spans(
        lines[call_line], '"placement.region_call"')
    call_operands = _split_top_level(lines[call_line][slice(*operand_span)])
    call_input_types = _split_top_level(
        lines[call_line][slice(*call_input_span)])
    call_outputs = _split_top_level(lines[call_line][slice(*call_output_span)])
    if len(call_operands) != len(call_input_types):
        raise RuntimeError(f"{function}: region call input count changed")
    if any(position >= len(call_outputs) for position in call_output_for_state):
        raise RuntimeError(f"{function}: state region result is out of range")
    if set(call_output_names) != set(range(len(call_outputs))):
        raise RuntimeError(
            f"{function}: resolved {len(call_output_names)} of "
            f"{len(call_outputs)} ANE result links"
        )

    # Map each outer state argument through tensor_to_memref and the call
    # operand list to its ANE region argument position. MPSGraph preserves
    # internal SSA names such as ``%arg16`` after pruning earlier feeds, so
    # function position and the numeric suffix of its formal are not the
    # same contract.
    function_arg_span, _function_results = _signature_spans(
        lines[function_line], "func.func")
    function_args = _split_top_level(
        lines[function_line][slice(*function_arg_span)])
    function_formals = []
    function_types = []
    for declaration in function_args:
        formal, separator, value_type = declaration.partition(":")
        formal = formal.strip()
        if not separator or not re.fullmatch(r"%[A-Za-z0-9_.$-]+", formal):
            raise RuntimeError(
                f"{function}: cannot parse function argument {declaration!r}")
        function_formals.append(formal)
        match = re.match(r"\s*(tensor<[^>]+>)", value_type)
        if match is None:
            raise RuntimeError(
                f"{function}: state argument is not a tensor: {declaration!r}")
        function_types.append(match.group(1))
    if len(function_formals) != len(order):
        raise RuntimeError(
            f"{function}: feed contract has {len(order)} values but IR has "
            f"{len(function_formals)}")

    to_memref: dict[str, str] = {}
    feed_conversion = re.compile(
        r'^\s*(%[A-Za-z0-9_.$-]+) = "placement\.tensor_to_memref"'
        r'\((%[A-Za-z0-9_.$-]+)\)')
    for index in range(function_line + 1, call_line):
        match = feed_conversion.match(lines[index])
        if match:
            to_memref[match.group(2)] = match.group(1)

    region_arg_span, _region_results = _signature_spans(
        lines[region_line], "anec.A14")
    region_args = _split_top_level(
        lines[region_line][slice(*region_arg_span)])
    region_formals = []
    region_types = []
    for declaration in region_args:
        formal, separator, value_type = declaration.partition(":")
        formal = formal.strip()
        if not separator or not re.fullmatch(r"%[A-Za-z0-9_.$-]+", formal):
            raise RuntimeError(
                f"{function}: cannot parse ANE argument {declaration!r}")
        region_formals.append(formal)
        region_types.append(value_type.strip())

    region_arg_for_feed: dict[int, str] = {}
    for outer_index, (feed_name, _shape) in enumerate(order):
        converted = to_memref.get(function_formals[outer_index])
        if converted is None or converted not in call_operands:
            if feed_name not in states:
                raise RuntimeError(
                    f"{function}: ordinary feed {feed_name!r} is absent from "
                    "the ANE region call"
                )
            continue
        positions = [
            position for position, operand in enumerate(call_operands)
            if operand == converted
        ]
        if len(positions) != 1 or positions[0] in region_arg_for_feed:
            raise RuntimeError(
                f"{function}: feed {feed_name!r} has an ambiguous ANE argument"
            )
        region_arg_for_feed[positions[0]] = feed_name

    injected_conversions = []
    for outer_index in state_input_indices:
        state_name = order[outer_index][0]
        if state_name in region_arg_for_feed.values():
            continue
        formal = function_formals[outer_index]
        tensor_type = function_types[outer_index]
        memref_type = "memref<" + tensor_type[len("tensor<"):]
        converted = f"%kst_{state_name}_memref"
        region_formal = f"%kstarg_{state_name}"
        injected_conversions.append(
            f'    {converted} = "placement.tensor_to_memref"({formal}) : '
            f'({tensor_type}) -> {memref_type}\n')
        call_operands.append(converted)
        call_input_types.append(memref_type)
        region_arg_for_feed[len(region_formals)] = state_name
        region_formals.append(region_formal)
        region_types.append(memref_type)
        region_args.append(f"{region_formal}: {memref_type}")

    if injected_conversions:
        lines[region_line] = _replace_span(
            lines[region_line], region_arg_span[0], region_arg_span[1],
            ", ".join(region_args),
        )
        call = lines[call_line]
        for span, replacement in sorted(
            ((operand_span, ", ".join(call_operands)),
             (call_input_span, ", ".join(call_input_types))),
            reverse=True,
        ):
            call = _replace_span(call, span[0], span[1], replacement)
        lines[call_line - 1] += "".join(injected_conversions)
        lines[call_line] = call
    if (
        set(region_arg_for_feed.values()) != feed_names
        or set(region_arg_for_feed) != set(range(len(region_formals)))
    ):
        raise RuntimeError(
            f"{function}: ANE feed mapping is incomplete or ambiguous"
        )
    region_arg_for_state = {
        position: name
        for position, name in region_arg_for_feed.items()
        if name in states
    }
    if ane_input_order is not None:
        ane_input_order.extend(
            region_arg_for_feed[position]
            for wanted_state in (False, True)
            for position in range(len(region_formals))
            if (region_arg_for_feed[position] in states) is wanted_state
        )
    region_formal_for_state = {
        region_formals[position]: state_name
        for position, state_name in region_arg_for_state.items()
    }

    # Replace region input views with persistent readers while retaining the
    # original result SSA name consumed by the graph.
    input_view_pattern = re.compile(
        r'^(\s*)(%[A-Za-z0-9_.$-]+) = "anec\.input_view"'
        r'\((%[A-Za-z0-9_.$-]+)\).* : '
        r'\((memref<.*>)\) -> (memref<.*>)\s*$')
    replaced_inputs: set[str] = set()
    for index in range(region_line + 1, function_line):
        match = input_view_pattern.match(lines[index].rstrip("\n"))
        if match is None:
            continue
        formal = match.group(3)
        state_name = region_formal_for_state.get(formal)
        if state_name is None:
            continue
        state = states[state_name]
        indent, input_type = match.group(1), match.group(4)
        live = f"%kst_{state_name}_live"
        read = f"%kst_{state_name}_read"
        definition = ""
        if state_name not in replaced_inputs:
            definition = (
                f'{indent}{live} = "anec.state"({formal}) : '
                f'({input_type}) -> {input_type}\n'
                f'{indent}{read} = "anec.ring_buffer_reader"({live}) '
                f'{_ring_attributes(state.storage_shape)} : '
                f'({input_type}) -> {input_type}\n')
        operand = re.compile(
            re.escape(formal) + r"(?![A-Za-z0-9_.$-])")
        lines[index] = definition + operand.sub(read, lines[index])
        replaced_inputs.add(state_name)
    direct_states = set(states) - replaced_inputs
    if direct_states:
        prefixes = []
        for region_arg, state_name in region_arg_for_state.items():
            if state_name not in direct_states:
                continue
            state = states[state_name]
            value_type = region_types[region_arg]
            formal = region_formals[region_arg]
            live = f"%kst_{state_name}_live"
            read = f"%kst_{state_name}_read"
            argument = re.compile(
                re.escape(formal) + r"(?![A-Za-z0-9_.$-])")
            for index in range(region_line + 1, function_line):
                lines[index] = argument.sub(read, lines[index])
            prefixes.extend((
                f'    {live} = "anec.state"({formal}) : '
                f'({value_type}) -> {value_type}\n',
                f'    {read} = "anec.ring_buffer_reader"({live}) '
                f'{_ring_attributes(state.storage_shape)} : '
                f'({value_type}) -> {value_type}\n',
            ))
            replaced_inputs.add(state_name)
        lines[region_line + 1] = "".join(prefixes) + lines[region_line + 1]
    if replaced_inputs != set(states):
        candidates = [
            line.strip()
            for line in lines[region_line + 1:function_line]
            if "input_view" in line or "(%arg" in line
        ][:8]
        raise RuntimeError(
            f"{function}: state readers changed; missing "
            f"{sorted(set(states) - replaced_inputs)}; candidates={candidates}")

    # Convert state result operands into writers and remove them from the ANE
    # return signature.  Results may appear in any compiler-chosen order.
    region_return = next(
        (index for index in range(region_line + 1, function_line)
         if '"anec.region_return"' in lines[index]),
        None,
    )
    if region_return is None:
        raise RuntimeError(f"{function}: ANE region return is missing")
    return_operand_span, return_input_span, _empty_output = _operation_spans(
        lines[region_return], '"anec.region_return"')
    region_values = _split_top_level(
        lines[region_return][slice(*return_operand_span)])
    region_types = _split_top_level(
        lines[region_return][slice(*return_input_span)])
    if len(region_values) != len(call_outputs) or len(region_types) != len(call_outputs):
        raise RuntimeError(f"{function}: ANE return and call result counts differ")
    writers = []
    for position, state_name in sorted(call_output_for_state.items()):
        state = states[state_name]
        value = region_values[position]
        value_type = region_types[position]
        live = f"%kst_{state_name}_live"
        writers.append(
            f'    "anec.ring_buffer_writer"({live}, {value}) '
            f'{_ring_attributes(state.storage_shape)} : '
            f'({value_type}, {value_type}) -> ()\n')
    keep_call_outputs = [
        value for index, value in enumerate(call_outputs)
        if index not in call_output_for_state]
    keep_region_values = [
        value for index, value in enumerate(region_values)
        if index not in call_output_for_state]
    keep_region_types = [
        value for index, value in enumerate(region_types)
        if index not in call_output_for_state]
    if ane_output_order is not None:
        ane_output_order.extend(
            call_output_names[index]
            for index in range(len(call_outputs))
            if index not in call_output_for_state
        )

    line = lines[region_return]
    for span, replacement in sorted(
        ((return_operand_span, ", ".join(keep_region_values)),
         (return_input_span, ", ".join(keep_region_types))),
        reverse=True,
    ):
        line = _replace_span(line, span[0], span[1], replacement)
    lines[region_return] = "".join(writers) + line

    # Remove the same result types from the ANE region declaration.
    _region_args, region_result_span = _signature_spans(
        lines[region_line], "anec.A14")
    region_result_types = _split_top_level(
        lines[region_line][slice(*region_result_span)])
    if len(region_result_types) != len(call_outputs):
        raise RuntimeError(f"{function}: ANE region signature result count changed")
    lines[region_line] = _replace_span(
        lines[region_line], region_result_span[0], region_result_span[1],
        ", ".join(
            value for index, value in enumerate(region_result_types)
            if index not in call_output_for_state),
    )

    # Rewrite the placement call count/types, then renumber every surviving
    # use of its tuple results.
    removed_positions = set(call_output_for_state)
    new_count = len(keep_call_outputs)
    call = lines[call_line]
    if lhs.group(3) is None:
        if len(call_outputs) != 1:
            raise RuntimeError(f"{function}: unnumbered multi-result region call")
    else:
        count_start, count_end = lhs.span(3)
        call = _replace_span(call, count_start, count_end, str(new_count))
    # Recompute spans after changing the count width.
    _operands, _inputs, outputs = _operation_spans(
        call, '"placement.region_call"')
    call = _replace_span(
        call, outputs[0], outputs[1], ", ".join(keep_call_outputs))
    lines[call_line] = call
    shifts = {
        old: old - sum(position < old for position in removed_positions)
        for old in range(len(call_outputs)) if old not in removed_positions
    }
    result_use = re.compile(re.escape(call_base) + r"#([0-9]+)")
    for index in range(call_line + 1, len(lines)):
        if index in conversion_lines:
            lines[index] = ""
            continue

        def replace_use(match: re.Match[str]) -> str:
            old = int(match.group(1))
            if old in removed_positions:
                raise RuntimeError(
                    f"{function}: removed state result {old} still has a use")
            return f"{call_base}#{shifts[old]}"

        lines[index] = result_use.sub(replace_use, lines[index])

    # Remove state values/types from the public function return and signature.
    keep_target_positions = [
        index for index in range(len(targets)) if index not in result_positions]
    outer_return = lines[return_line]
    # The earlier result renumbering can change the return value text, so parse
    # its current spans again.
    current_values, current_types, current_value_span, current_type_span = (
        _parse_return(outer_return))
    outer_return = _replace_span(
        outer_return, current_type_span[0], current_type_span[1],
        ", ".join(current_types[index] for index in keep_target_positions))
    current_values, _current_types, current_value_span, _current_type_span = (
        _parse_return(outer_return))
    outer_return = _replace_span(
        outer_return, current_value_span[0], current_value_span[1],
        ", ".join(current_values[index] for index in keep_target_positions))
    lines[return_line] = outer_return

    _function_args, function_result_span = _signature_spans(
        lines[function_line], "func.func")
    function_result_types = _split_top_level(
        lines[function_line][slice(*function_result_span)])
    if len(function_result_types) != len(targets):
        raise RuntimeError(f"{function}: public result signature changed")
    function_text = _replace_span(
        lines[function_line], function_result_span[0], function_result_span[1],
        ", ".join(function_result_types[index] for index in keep_target_positions),
    )
    attrs = _state_attrs(state_input_indices)
    attribute_marker = "attributes {"
    if attribute_marker not in function_text:
        raise RuntimeError(f"{function}: public function attributes are missing")
    function_text = function_text.replace(
        attribute_marker, attribute_marker + attrs + ", ", 1)
    lines[function_line] = function_text

    result = "".join(lines)
    result = re.sub(
        r'mps\.aneRegionsSHA = "[^"]+"',
        f'mps.aneRegionsSHA = "KINO_STATE_{function}"',
        result,
    )
    return re.sub(
        r'mps\.regionSHA = "[^"]+"',
        f'mps.regionSHA = "KINO_STATE_{function}"',
        result,
    )


@dataclass(frozen=True)
class _ModuleBounds:
    preamble_end: int
    module_header_start: int
    module_header_end: int
    body_start: int
    body_end: int
    resources_start: int
    resources_end: int


@dataclass(frozen=True)
class _Resource:
    path: Path
    name: str
    payload_start: int
    payload_end: int
    digest: str


def _module_bounds(path: Path) -> _ModuleBounds:
    with (
        path.open("rb") as source,
        mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as view,
    ):
        marker = view.find(b"{-#")
        if (
            marker >= 0
            and view[marker:marker + len(_RESOURCE_HEADER)] != _RESOURCE_HEADER
        ):
            raise RuntimeError(f"{path}: unexpected MPSGraph resources")
        starts = [
            value for value in (
                view.find(_MODULE_MARKER),
                view.find(_PLAIN_MODULE_MARKER),
            )
            if value >= 0
        ]
        if not starts:
            raise RuntimeError(f"{path}: module is missing")
        module_start = min(starts)
        body_start = view.find(b" {\n", module_start)
        if body_start < 0 or (marker >= 0 and body_start > marker):
            raise RuntimeError(f"{path}: module body is missing")
        body_start += 3
        body_end = (marker if marker >= 0 else len(view)) - 1
        while body_end >= body_start and chr(view[body_end]).isspace():
            body_end -= 1
        if view[body_end:body_end + 1] != b"}":
            raise RuntimeError(f"{path}: module close is missing")
        if marker >= 0:
            footer = view.rfind(_RESOURCE_FOOTER)
            if footer < 0:
                raise RuntimeError(f"{path}: resource footer is missing")
            resources_start = marker + len(_RESOURCE_HEADER)
            resources_end = footer
        else:
            resources_start = resources_end = -1
        return _ModuleBounds(
            preamble_end=module_start,
            module_header_start=module_start,
            module_header_end=body_start,
            body_start=body_start,
            body_end=body_end,
            resources_start=resources_start,
            resources_end=resources_end,
        )


def _copy_range(source: Any, target: Any, start: int, end: int) -> None:
    source.seek(start)
    remaining = end - start
    while remaining:
        payload = source.read(min(8 * 1024 * 1024, remaining))
        if not payload:
            raise RuntimeError("unexpected EOF while copying MPSGraph module")
        target.write(payload)
        remaining -= len(payload)


def _write_transformed_module(
    source_path: Path,
    output_path: Path,
    *,
    function: str,
    order: Sequence[tuple[str, tuple[int, ...]]],
    targets: Sequence[tuple[str, tuple[int, ...]]],
    states: Mapping[str, StateTensorSpec],
    state_results: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    bounds = _module_bounds(source_path)
    with source_path.open("rb") as source:
        source.seek(bounds.body_start)
        body = source.read(bounds.body_end - bounds.body_start).decode("utf-8")
    ane_input_order: list[str] = []
    ane_output_order: list[str] = []
    transformed = _transform_placed_body(
        body,
        function=function,
        order=order,
        targets=targets,
        states=states,
        state_results=state_results,
        ane_input_order=ane_input_order,
        ane_output_order=ane_output_order,
    ).encode("utf-8")
    with output_path.open("wb") as output, source_path.open("rb") as source:
        _copy_range(source, output, 0, bounds.body_start)
        output.write(transformed)
        output.write(b"}\n")
        if bounds.resources_start >= 0:
            _copy_range(
                source,
                output,
                bounds.body_end + 1,
                source_path.stat().st_size,
            )
    return tuple(ane_input_order), tuple(ane_output_order)


def _write_multiprocedure_module(
    raw_path: Path,
    placed_path: Path,
    output_path: Path,
    *,
    function: str,
    order: Sequence[tuple[str, tuple[int, ...]]],
    targets: Sequence[tuple[str, tuple[int, ...]]],
    states: Mapping[str, StateTensorSpec],
    state_results: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Wrap raw graph IR using the placed module's runtime attributes."""
    raw_bounds = _module_bounds(raw_path)
    placed_bounds = _module_bounds(placed_path)
    with raw_path.open("rb") as raw:
        raw.seek(raw_bounds.body_start)
        raw_body = raw.read(
            raw_bounds.body_end - raw_bounds.body_start).decode("utf-8")
    with placed_path.open("rb") as placed:
        placed.seek(placed_bounds.body_start)
        placed_body = placed.read(
            placed_bounds.body_end - placed_bounds.body_start
        ).decode("utf-8")
        placed.seek(placed_bounds.module_header_start)
        header = placed.read(
            placed_bounds.module_header_end
            - placed_bounds.module_header_start)
    families = set(re.findall(
        r"\banec\.([A-Za-z0-9_]+)\s+@", placed_body))
    if len(families) != 1:
        raise RuntimeError(
            f"{function}: expected one placed ANE family, got "
            f"{sorted(families)}")
    transformed, ane_inputs, ane_outputs = _transform_multiprocedure_body(
        raw_body,
        function=function,
        family=next(iter(families)),
        order=order,
        targets=targets,
        states=states,
        state_results=state_results,
    )
    with output_path.open("wb") as output, raw_path.open("rb") as raw:
        _copy_range(raw, output, 0, raw_bounds.module_header_start)
        output.write(header)
        output.write(transformed.encode("utf-8"))
        output.write(b"}\n")
        _copy_range(
            raw,
            output,
            raw_bounds.body_end + 1,
            raw_path.stat().st_size,
        )
    return ane_inputs, ane_outputs


def _resources(path: Path, bounds: _ModuleBounds) -> list[_Resource]:
    if bounds.resources_start < 0:
        return []
    records = []
    with (
        path.open("rb") as source,
        mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as view,
    ):
        position = bounds.resources_start
        while position < bounds.resources_end:
            while (
                position < bounds.resources_end
                and chr(view[position]).isspace()
            ):
                position += 1
            if position >= bounds.resources_end:
                break
            line_end = view.find(b"\n", position, bounds.resources_end)
            if line_end < 0:
                line_end = bounds.resources_end
            colon = view.find(b': "', position, line_end)
            if colon < 0:
                raise RuntimeError(f"{path}: malformed dialect resource")
            name = bytes(view[position:colon]).decode("ascii").strip()
            payload_start = colon + 3
            payload_end = line_end
            if (
                payload_end > payload_start
                and view[payload_end - 1:payload_end] == b","
            ):
                payload_end -= 1
            if (
                payload_end <= payload_start
                or view[payload_end - 1:payload_end] != b'"'
            ):
                raise RuntimeError(
                    f"{path}: unterminated dialect resource {name}")
            payload_end -= 1
            digest = hashlib.sha256()
            cursor = payload_start
            while cursor < payload_end:
                next_cursor = min(cursor + 8 * 1024 * 1024, payload_end)
                digest.update(view[cursor:next_cursor])
                cursor = next_cursor
            records.append(_Resource(
                path=path,
                name=name,
                payload_start=payload_start,
                payload_end=payload_end,
                digest=digest.hexdigest(),
            ))
            position = line_end + 1
    return records


def _renamed_preamble_and_body(
    path: Path,
    bounds: _ModuleBounds,
    entry: str,
    resources: Mapping[str, str],
) -> tuple[str, str]:
    with path.open("rb") as source:
        preamble = source.read(bounds.preamble_end).decode("utf-8")
        source.seek(bounds.body_start)
        body = source.read(bounds.body_end - bounds.body_start).decode("utf-8")
    aliases = re.findall(r"(?m)^(#[A-Za-z0-9_.$-]+)\s*=", preamble)
    alias_map = {
        alias: f"#{entry}_{alias[1:]}" for alias in aliases
    }
    replacements = dict(alias_map)
    replacements.update(resources)
    if replacements:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_.$-])(?:"
            + "|".join(
                re.escape(name)
                for name in sorted(replacements, key=len, reverse=True))
            + r")(?![A-Za-z0-9_.$-])")
        preamble = pattern.sub(lambda match: replacements[match.group(0)], preamble)
        body = pattern.sub(lambda match: replacements[match.group(0)], body)
    return preamble, body


def _merge_modules(sources: Mapping[str, Path], output_path: Path) -> None:
    """Merge entries and content-deduplicate their large constant resources."""
    bounds = {name: _module_bounds(path) for name, path in sources.items()}
    per_source_resources = {
        name: _resources(sources[name], bounds[name]) for name in sources
    }
    canonical: dict[tuple[str, int], tuple[str, _Resource]] = {}
    resource_maps: dict[str, dict[str, str]] = {}
    for entry, records in per_source_resources.items():
        mapping = {}
        for record in records:
            key = (record.digest, record.payload_end - record.payload_start)
            if key not in canonical:
                canonical_name = "kst_" + record.digest[:24]
                canonical[key] = (canonical_name, record)
            mapping[record.name] = canonical[key][0]
        resource_maps[entry] = mapping

    renamed = {
        entry: _renamed_preamble_and_body(
            sources[entry], bounds[entry], entry, resource_maps[entry])
        for entry in sources
    }
    first = next(iter(sources))
    with sources[first].open("rb") as source:
        source.seek(bounds[first].module_header_start)
        header = source.read(
            bounds[first].module_header_end - bounds[first].module_header_start)
    identity = hashlib.sha256(b"KINO_STATE_COMBINED_V2\0")
    neutral_header = re.sub(
        rb'mps\.aneRegionsSHA = "[^"]+"',
        b'mps.aneRegionsSHA = ""',
        header,
    )
    identity.update(struct.pack("<Q", len(neutral_header)))
    identity.update(neutral_header)
    for entry in sources:
        for value in (entry, *renamed[entry]):
            payload = value.encode("utf-8")
            identity.update(struct.pack("<Q", len(payload)))
            identity.update(payload)
    for canonical_name, record in canonical.values():
        for value in (
            canonical_name,
            record.digest,
            str(record.payload_end - record.payload_start),
        ):
            payload = value.encode("ascii")
            identity.update(struct.pack("<Q", len(payload)))
            identity.update(payload)
    combined_identity = (
        "KINO_STATE_COMBINED_" + identity.hexdigest().upper())
    header = re.sub(
        rb'mps\.aneRegionsSHA = "[^"]+"',
        f'mps.aneRegionsSHA = "{combined_identity}"'.encode("ascii"),
        header,
    )
    with output_path.open("wb") as output:
        emitted_preamble: set[str] = set()
        for entry in sources:
            preamble, _body = renamed[entry]
            for line in preamble.splitlines(keepends=True):
                if line not in emitted_preamble:
                    output.write(line.encode("utf-8"))
                    emitted_preamble.add(line)
        output.write(header)
        for entry in sources:
            output.write(renamed[entry][1].encode("utf-8"))
            output.write(b"\n")
        output.write(b"}\n")
        records = list(canonical.values())
        if records:
            output.write(_RESOURCE_HEADER)
            for index, (name, record) in enumerate(records):
                output.write(b"      " + name.encode("ascii") + b': "')
                with record.path.open("rb") as source:
                    _copy_range(
                        source, output, record.payload_start, record.payload_end)
                output.write(b'"')
                if index + 1 < len(records):
                    output.write(b",")
                output.write(b"\n")
            output.write(_RESOURCE_FOOTER.lstrip(b"\n"))


def _namespace_resources(source: str, entry: str) -> str:
    names = set(re.findall(
        r"#mps\.buffer_tensor<([A-Za-z0-9_.$-]+)>", source))
    if not names:
        return source
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.$-])(?:"
        + "|".join(
            re.escape(name) for name in sorted(names, key=len, reverse=True))
        + r")(?![A-Za-z0-9_.$-])")
    return pattern.sub(lambda match: f"{entry}_{match.group(0)}", source)


def _source_descriptor(fw: Mapping[str, Any]) -> tuple[Any, Any]:
    compilation = fw["MPSGraphCompilationDescriptor"].alloc().init()
    compilation.setOptimizationLevel_(0)
    compilation.setPreferredDevice_(mg.DEVICE_GPU)
    compilation.setWaitForCompilationCompletion_(True)
    descriptor = fw["objc"].lookUpClass(
        "MPSGraphExecutableDescriptor").alloc().init()
    descriptor.setCompilationDescriptor_(compilation)
    return compilation, descriptor


def _ane_source_descriptor(
    fw: Mapping[str, Any],
    *,
    ane_fw_to_fw_signal: bool,
    ane_late_latch: bool,
) -> tuple[Any, Any]:
    compilation = fw["MPSGraphCompilationDescriptor"].alloc().init()
    compilation.setOptimizationLevel_(1)
    compilation.setPreferredDevice_(mg.DEVICE_ANE)
    compilation.setWaitForCompilationCompletion_(True)
    compilation.setEnableMLIRDiagnostics_(True)
    if ane_fw_to_fw_signal:
        compilation.setEnableANEFWToFWSignal_(True)
    if ane_late_latch:
        compilation.setEnableANELateLatch_(True)
    descriptor = fw["objc"].lookUpClass(
        "MPSGraphExecutableDescriptor").alloc().init()
    descriptor.setCompilationDescriptor_(compilation)
    return compilation, descriptor


def _capture_program(
    program: Program,
    *,
    fw: Mapping[str, Any],
    graph_device: Any,
) -> tuple[Any, list[tuple[str, tuple[int, ...]]], str]:
    compilation, executable_descriptor = _source_descriptor(fw)
    shaped = {
        tensor: fw["MPSGraphShapedType"].alloc().initWithShape_dataType_(
            list(shape), program.builder.dtype)
        for tensor, shape, _name in program.builder.feeds
    }
    raw = program.builder.graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor_(
        graph_device,
        shaped,
        [tensor for _name, tensor, _shape in program.targets],
        None,
        compilation,
    )
    by_id = {
        id(tensor): (name, tuple(int(item) for item in shape))
        for tensor, shape, name in program.builder.feeds
    }
    order = [by_id[id(tensor)] for tensor in raw.feedTensors()]
    source = str(raw.getIR())
    old = "func.func @main"
    if source.count(old) != 1:
        raise RuntimeError(
            f"{program.name}: expected one MPSGraph main function")
    source = source.replace(old, f"func.func @{program.name}", 1)
    source = _namespace_resources(source, program.name)
    executable = fw["MPSGraphExecutable"].alloc(
    ).initWithMLIRSource_executableDescriptor_(source, executable_descriptor)
    functions = [str(value) for value in executable.functionNames()]
    if functions != [program.name]:
        raise RuntimeError(
            f"{program.name}: captured functions changed to {functions}")
    return executable, order, source


def _symbol_name(address: int) -> str:
    runtime = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    runtime.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    runtime.dladdr.restype = ctypes.c_int
    info = _DlInfo()
    if not runtime.dladdr(ctypes.c_void_p(address), ctypes.byref(info)):
        return ""
    return info.symbol.decode(errors="replace") if info.symbol else ""


def _resolve_specialized_module(
    executable: Any,
    entry_point: Any,
    *,
    fw: Mapping[str, Any],
    graph_device: Any,
    compilation: Any,
) -> tuple[Any, Any]:
    """Resolve MPSGraph's resource-backed module reference to a ModuleOp.

    The private specialization API returns an inline SmallVector whose second
    field is an ``InMemoryModuleRef``.  Its virtual ``get`` method is the
    resource-aware step omitted by ``optimizedBytecode``.  Check the exact
    runtime symbol before calling it so a changed framework ABI fails closed.
    """
    vector = executable.specializedModuleWithDevice_shapedEntryPoints_compilationDescriptor_error_(
        graph_device, [entry_point], compilation, None)
    if not isinstance(vector, tuple) or len(vector) != 4:
        raise RuntimeError(
            f"MPSGraph returned an unexpected specialization vector: {vector!r}")
    begin, size, capacity, signed_inline = vector
    inline = bytes(value & 0xFF for value in signed_inline)
    if not begin or size != 1 or capacity < 1 or len(inline) < 16:
        raise RuntimeError(
            "MPSGraph specialization vector layout changed: "
            f"begin={begin!r}, size={size!r}, capacity={capacity!r}, "
            f"inline={len(inline)}")
    module_ref = struct.unpack_from("<Q", inline, 8)[0]
    if not module_ref or module_ref % ctypes.sizeof(ctypes.c_void_p):
        raise RuntimeError(
            f"MPSGraph returned an invalid module reference: {module_ref:#x}")

    vtable = ctypes.c_void_p.from_address(module_ref).value
    if not vtable:
        raise RuntimeError("MPSGraph module reference has no virtual table")
    get_address = ctypes.c_void_p.from_address(
        vtable + 3 * ctypes.sizeof(ctypes.c_void_p)).value
    if not get_address:
        raise RuntimeError("MPSGraph module reference has no get method")
    symbol = _symbol_name(get_address)
    if "InMemoryModuleRef3get" not in symbol:
        raise RuntimeError(
            "MPSGraph module-reference ABI changed: "
            f"expected InMemoryModuleRef::get, got {symbol or hex(get_address)}")

    get_module = ctypes.CFUNCTYPE(
        _ModuleOp, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(
            get_address)
    error = ctypes.c_void_p()
    module = get_module(module_ref, ctypes.byref(error))
    if error.value:
        error_object = fw["objc"].objc_object(c_void_p=error.value)
        raise RuntimeError(
            f"MPSGraph could not resolve its specialized module: {error_object}")
    if not module.operation:
        raise RuntimeError("MPSGraph resolved a null specialized module")

    global _operation_pointer_type
    if _operation_pointer_type is None:
        objc = fw["objc"]
        name = "KinovsrMLIROperationPointer"
        _operation_pointer_type = getattr(objc, name, None)
        if _operation_pointer_type is None:
            _operation_pointer_type = objc.createOpaquePointerType(
                name, b"^{Operation}", "MLIR operation pointer")
    return module, _operation_pointer_type(c_void_p=module.operation)


def _extract_placed_source(
    executable: Any,
    entry_point: Any,
    *,
    fw: Mapping[str, Any],
    graph_device: Any,
    compilation: Any,
) -> str:
    module, operation_pointer = _resolve_specialized_module(
        executable,
        entry_point,
        fw=fw,
        graph_device=graph_device,
        compilation=compilation,
    )
    descriptor = fw["objc"].lookUpClass(
        "MPSGraphExecutableDescriptor").alloc().init()
    descriptor.setCompilationDescriptor_(compilation)
    wrapped = fw["MPSGraphExecutable"].alloc(
    ).initWithSpecializedMLIRModule_device_shapedEntryPoint_compilationDescriptor_executableDescriptor_(
        (operation_pointer,),
        graph_device,
        entry_point,
        compilation,
        descriptor,
    )
    if wrapped is None:
        raise RuntimeError("MPSGraph could not wrap its specialized module")
    source = str(wrapped.getIR())
    # Keep the ctypes structure alive through the initializer call.  The
    # resulting executable owns the referenced module independently.
    del module
    return source


def _load_mlir_source(path: Path, fw: Mapping[str, Any]) -> Any:
    NSString = fw["objc"].lookUpClass("NSString")
    loaded = NSString.stringWithContentsOfFile_encoding_error_(
        str(path), 4, None)
    source = loaded[0] if isinstance(loaded, tuple) else loaded
    error = loaded[1] if isinstance(loaded, tuple) else None
    if source is None:
        raise RuntimeError(f"could not load MPSGraph module {path}: {error}")
    return source


def _compile_product(
    module_path: Path,
    output_root: Path,
    *,
    function: str,
    region: str,
    input_count: int,
    output_count: int,
    state_count: int,
    fw: Mapping[str, Any],
) -> Path:
    _compilation, descriptor = _source_descriptor(fw)
    source = _load_mlir_source(module_path, fw)
    executable = fw["MPSGraphExecutable"].alloc(
    ).initWithMLIRSource_executableDescriptor_(source, descriptor)
    executable.setOptions_(0)
    functions = [str(value) for value in executable.functionNames()]
    if functions != [function]:
        raise RuntimeError(
            f"{function}: transformed functions changed to {functions}")
    positions = list(
        executable.getStateInputPositionsWithEntryFunctionName_(function))
    if not positions:
        raise RuntimeError(f"{function}: transformed state ABI is missing")
    output_root.mkdir(parents=True, exist_ok=False)
    archive = Path(str(executable.getMutableWeightsFilePath())).parent
    generated_product = archive / f"{region}.plist"
    generated_weights = archive / f"{region}.weights"
    generated_options = archive / f"compiler_options_{region}.plist"
    if generated_product.is_file() and generated_weights.is_file():
        destination = output_root / "compiled"
        destination.mkdir()
        shutil.copy2(generated_product, destination / generated_product.name)
        shutil.copy2(generated_weights, destination / generated_weights.name)
        if generated_options.is_file():
            shutil.copy2(generated_options, destination / generated_options.name)
        product = destination / generated_product.name
    else:
        executable.setValue_forKey_(str(output_root), "dumpCompiledProductsPath")
        executable.dumpCompiledProducts()
        products = list(output_root.glob(f"mpsgraph-*/{region}.plist"))
        if len(products) != 1:
            raise RuntimeError(
                f"{function}: expected one compiled ANE product, got {products}")
        product = products[0]
    if not product.with_suffix(".weights").is_file():
        raise RuntimeError(f"{function}: compiled ANE weights are missing")
    compiler_options = product.parent / f"compiler_options_{region}.plist"
    if not compiler_options.is_file():
        raise RuntimeError(f"{function}: ANE compiler options are missing")
    with product.open("rb") as handle:
        record = plistlib.load(handle)
    if record.get("Networks") != [region] or not isinstance(
        network := record.get(region), dict
    ):
        raise RuntimeError(f"{function}: compiled ANE network identity changed")
    realized = {
        "inputs": len(network.get("Inputs", ())),
        "outputs": len(network.get("Outputs", ())),
        "states": len(network.get("States", ())),
    }
    expected = {
        "inputs": input_count,
        "outputs": output_count,
        "states": state_count,
    }
    if realized != expected:
        raise RuntimeError(
            f"{function}: compiled ANE ports changed: expected {expected}, "
            f"got {realized}")
    return product


def _retarget_compiler_options(
    product: Path, *, region: str, published_product: Path
) -> None:
    """Point copied MPSGraph options at the durable published netplist."""
    path = product.parent / f"compiler_options_{region}.plist"
    with path.open("rb") as handle:
        options = plistlib.load(handle)
    if not isinstance(options, dict) or not options:
        raise RuntimeError(f"{region}: malformed ANE compiler options")
    for architecture, record in options.items():
        if not isinstance(record, dict):
            raise RuntimeError(
                f"{region}: malformed options for architecture {architecture!r}"
            )
        record["NetworkPlistName"] = region
        record["NetworkPlistPath"] = str(published_product)
    with path.open("wb") as handle:
        plistlib.dump(options, handle, fmt=plistlib.FMT_BINARY)


def _system_cache_key() -> str:
    value = f"{platform.mac_ver()[0]}-{platform.machine()}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def stateful_cache_ready(cache_directory: str | Path) -> bool:
    """Whether a durable stateful cache has every published product file."""
    root = Path(cache_directory) / _system_cache_key()
    contract = root / "contract.json"
    module = root / "model.mlir"
    if not contract.is_file() or not module.is_file():
        return False
    try:
        _dtype, _states, entries = _parse_contract(
            json.loads(contract.read_text()))
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    return all(
        (product := root / entry.product).is_file()
        and product.with_suffix(".weights").is_file()
        and (
            product.parent / f"compiler_options_{entry.region}.plist"
        ).is_file()
        for entry in entries
    )


def multiprocedure_cache_ready(cache_directory: str | Path) -> bool:
    """Whether a grouped cache has its source module and ABI contract."""
    root = Path(cache_directory) / _system_cache_key()
    contract = root / "contract.json"
    module = root / "model.mlir"
    if not contract.is_file() or not module.is_file():
        return False
    try:
        _dtype, _states, entries = _parse_contract(
            json.loads(contract.read_text()),
            expected_format=_MULTIPROCEDURE_CACHE_FORMAT,
        )
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    return bool(entries) and all(not entry.product for entry in entries)


def _contract_json(
    *,
    dtype: int,
    states: Sequence[StateTensorSpec],
    entries: Sequence[_EntryContract],
    format_version: int = _CACHE_FORMAT,
) -> dict[str, Any]:
    return {
        "format": format_version,
        "dtype": dtype,
        "system": _system_cache_key(),
        "states": [
            {
                "name": state.name,
                "logical_shape": list(state.logical_shape),
                "storage_shape": list(state.storage_shape),
            }
            for state in states
        ],
        "entries": [
            {
                "name": entry.name,
                "function": entry.function,
                "region": entry.region,
                "order": [[name, list(shape)] for name, shape in entry.order],
                "targets": [
                    [name, list(shape)] for name, shape in entry.targets],
                "state_results": [list(item) for item in entry.state_results],
                "ane_input_order": list(entry.ane_input_order),
                "ane_output_order": list(entry.ane_output_order),
                "dynamic": sorted(entry.dynamic),
                "product": entry.product,
            }
            for entry in entries
        ],
    }


def _parse_contract(
    record: Mapping[str, Any], *, expected_format: int = _CACHE_FORMAT
) -> tuple[
    int, tuple[StateTensorSpec, ...], tuple[_EntryContract, ...]
]:
    if record.get("format") != expected_format:
        raise RuntimeError("MPSGraph state cache format changed")
    if record.get("system") != _system_cache_key():
        raise RuntimeError("MPSGraph state cache belongs to another OS runtime")
    states = tuple(
        StateTensorSpec(
            str(item["name"]),
            tuple(int(value) for value in item["logical_shape"]),
            tuple(int(value) for value in item["storage_shape"]),
        )
        for item in record["states"]
    )
    entries = tuple(
        _EntryContract(
            name=str(item["name"]),
            function=str(item["function"]),
            region=str(item["region"]),
            order=tuple(
                (str(name), tuple(int(value) for value in shape))
                for name, shape in item["order"]),
            targets=tuple(
                (str(name), tuple(int(value) for value in shape))
                for name, shape in item["targets"]),
            state_results=tuple(
                (str(name), str(state))
                for name, state in item["state_results"]),
            ane_input_order=tuple(
                str(name) for name in item["ane_input_order"]
            ),
            ane_output_order=tuple(
                str(name) for name in item["ane_output_order"]
            ),
            dynamic=frozenset(str(name) for name in item["dynamic"]),
            product=str(item["product"]),
        )
        for item in record["entries"]
    )
    for entry in entries:
        feed_names = [name for name, _shape in entry.order]
        target_names = [name for name, _shape in entry.runtime_targets]
        if (
            len(entry.ane_input_order) != len(feed_names)
            or len(set(entry.ane_input_order)) != len(feed_names)
            or set(entry.ane_input_order) != set(feed_names)
        ):
            raise RuntimeError(
                f"{entry.name}: cached ANE input order changed"
            )
        if (
            len(entry.ane_output_order) != len(target_names)
            or len(set(entry.ane_output_order)) != len(target_names)
            or set(entry.ane_output_order) != set(target_names)
        ):
            raise RuntimeError(
                f"{entry.name}: cached ANE output order changed"
            )
    return int(record["dtype"]), states, entries


def _validate_program(
    program: Program,
    states: Mapping[str, StateTensorSpec],
) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", program.name):
        raise ValueError(f"invalid MPSGraph entry name {program.name!r}")
    feed_items = [
        (name, tuple(int(value) for value in shape))
        for _tensor, shape, name in program.builder.feeds
    ]
    feeds = dict(feed_items)
    if len(feeds) != len(feed_items):
        raise ValueError(f"{program.name}: feed names must be unique")
    for name, state in states.items():
        if feeds.get(name) != state.storage_shape:
            raise ValueError(
                f"{program.name}: state feed {name!r} must have storage shape "
                f"{state.storage_shape}, got {feeds.get(name)}")
    target_items = [
        (name, tuple(int(value) for value in shape))
        for name, _tensor, shape in program.targets
    ]
    targets = dict(target_items)
    if len(targets) != len(target_items):
        raise ValueError(f"{program.name}: target names must be unique")
    if len(set(program.state_results.values())) != len(program.state_results):
        raise ValueError(
            f"{program.name}: each state may have at most one result")
    unknown_dynamic = set(program.dynamic) - (set(feeds) | set(targets))
    if unknown_dynamic:
        raise ValueError(
            f"{program.name}: unknown dynamic tensors "
            f"{sorted(unknown_dynamic)}")
    targets = {
        name: tuple(int(value) for value in shape)
        for name, _tensor, shape in program.targets
    }
    for result_name, state_name in program.state_results.items():
        if state_name not in states:
            raise ValueError(
                f"{program.name}: unknown state result {state_name!r}")
        if targets.get(result_name) != states[state_name].storage_shape:
            raise ValueError(
                f"{program.name}: state result {result_name!r} must have "
                f"shape {states[state_name].storage_shape}")


def _build_cache(
    cache_directory: Path,
    factories: Mapping[str, Callable[[], Program]],
    states: Sequence[StateTensorSpec],
    *,
    dtype: int,
    ane_fw_to_fw_signal: bool,
    ane_late_latch: bool,
) -> None:
    fw = mg._fw()
    metal = fw["Metal"].MTLCreateSystemDefaultDevice()
    graph_device = fw["MPSGraphDevice"].deviceWithMTLDevice_(metal)
    parent = cache_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{cache_directory.name}.partial-", dir=parent))
    build = staging / "build"
    build.mkdir()
    state_by_name = {state.name: state for state in states}
    captured: dict[str, dict[str, Any]] = {}
    try:
        for name, factory in factories.items():
            program = factory()
            if program.name != name:
                raise ValueError(
                    f"MPSGraph program factory {name!r} returned {program.name!r}")
            if program.builder.dtype != dtype:
                raise ValueError(f"{name}: MPSGraph state dtype changed")
            _validate_program(program, state_by_name)
            executable, order, source = _capture_program(
                program, fw=fw, graph_device=graph_device)
            raw_package = build / f"{name}.mpsgraphpackage"
            serialization = fw[
                "MPSGraphExecutableSerializationDescriptor"].alloc().init()
            serialization.setAppend_(False)
            executable.serializeToMPSGraphPackageAtURL_descriptor_(
                fw["NSURL"].fileURLWithPath_(str(raw_package)), serialization)
            captured[name] = {
                "package": raw_package,
                "order": order,
                "targets": [
                    (target_name, tuple(int(value) for value in shape))
                    for target_name, _tensor, shape in program.targets
                ],
                "state_results": dict(program.state_results),
                "dynamic": set(program.dynamic),
            }
            del executable, source, program
            gc.collect()

        compilation, _descriptor = _ane_source_descriptor(
            fw,
            ane_fw_to_fw_signal=ane_fw_to_fw_signal,
            ane_late_latch=ane_late_latch,
        )
        transformed_paths = {}
        entry_contracts = []
        for name, contract in captured.items():
            ordinary = fw["MPSGraphExecutable"].alloc(
            ).initWithMPSGraphPackageAtURL_compilationDescriptor_(
                fw["NSURL"].fileURLWithPath_(str(contract["package"])),
                compilation,
            )
            input_types = [
                fw["MPSGraphShapedType"].alloc().initWithShape_dataType_(
                    list(shape), dtype)
                for _feed_name, shape in contract["order"]
            ]
            entry_point = fw["objc"].lookUpClass(
                "MPSGraphExecutableShapedEntryPoint").alloc(
            ).initWithEntryFunctionName_inputTypes_(name, input_types)
            placed_path = build / f"{name}.placed.mlir"
            transformed_path = build / f"{name}.state.mlir"
            placed_path.write_text(_extract_placed_source(
                ordinary,
                entry_point,
                fw=fw,
                graph_device=graph_device,
                compilation=compilation,
            ))
            contract = captured[name]
            function = name + "_0"
            region = function + "_ane_region_0_0"
            ane_input_order, ane_output_order = _write_transformed_module(
                placed_path,
                transformed_path,
                function=function,
                order=contract["order"],
                targets=contract["targets"],
                states=state_by_name,
                state_results=contract["state_results"],
            )
            product = _compile_product(
                transformed_path,
                staging / "products" / name,
                function=function,
                region=region,
                input_count=sum(
                    feed_name not in state_by_name
                    for feed_name, _shape in contract["order"]),
                output_count=(
                    len(contract["targets"])
                    - len(contract["state_results"])),
                state_count=len(state_by_name),
                fw=fw,
            )
            _retarget_compiler_options(
                product,
                region=region,
                published_product=(
                    cache_directory / product.relative_to(staging)
                ),
            )
            entry_contracts.append(_EntryContract(
                name=name,
                function=function,
                region=region,
                order=tuple(contract["order"]),
                targets=tuple(contract["targets"]),
                state_results=tuple(contract["state_results"].items()),
                ane_input_order=ane_input_order,
                ane_output_order=ane_output_order,
                dynamic=frozenset(contract["dynamic"]),
                product=str(product.relative_to(staging)),
            ))
            transformed_paths[name] = transformed_path
            placed_path.unlink()
            del ordinary, entry_point
            gc.collect()

        _merge_modules(transformed_paths, staging / "model.mlir")
        (staging / "contract.json").write_text(json.dumps(
            _contract_json(
                dtype=dtype, states=states, entries=entry_contracts),
            indent=2,
        ))
        shutil.rmtree(build)
        try:
            staging.replace(cache_directory)
        except OSError:
            if not cache_directory.is_dir():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _build_multiprocedure_cache(
    cache_directory: Path,
    factories: Mapping[str, Callable[[], Program]],
    states: Sequence[StateTensorSpec],
    *,
    dtype: int,
    ane_fw_to_fw_signal: bool,
    ane_late_latch: bool,
) -> None:
    """Publish high-level inline-state procedures for one grouped model."""
    fw = mg._fw()
    metal = fw["Metal"].MTLCreateSystemDefaultDevice()
    graph_device = fw["MPSGraphDevice"].deviceWithMTLDevice_(metal)
    compilation, _descriptor = _ane_source_descriptor(
        fw,
        ane_fw_to_fw_signal=ane_fw_to_fw_signal,
        ane_late_latch=ane_late_latch,
    )
    parent = cache_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{cache_directory.name}.partial-", dir=parent))
    build = staging / "build"
    build.mkdir()
    state_by_name = {state.name: state for state in states}
    transformed_paths: dict[str, Path] = {}
    entry_contracts = []
    try:
        EntryPoint = fw["objc"].lookUpClass(
            "MPSGraphExecutableShapedEntryPoint")
        for name, factory in factories.items():
            program = factory()
            if program.name != name:
                raise ValueError(
                    f"MPSGraph program factory {name!r} returned "
                    f"{program.name!r}")
            if program.builder.dtype != dtype:
                raise ValueError(f"{name}: MPSGraph state dtype changed")
            _validate_program(program, state_by_name)
            executable, order, source = _capture_program(
                program, fw=fw, graph_device=graph_device)
            raw_path = build / f"{name}.raw.mlir"
            placed_path = build / f"{name}.placed.mlir"
            transformed_path = build / f"{name}.mpsx.mlir"
            raw_path.write_text(source)
            input_types = [
                fw["MPSGraphShapedType"].alloc().initWithShape_dataType_(
                    list(shape), dtype)
                for _feed_name, shape in order
            ]
            entry_point = EntryPoint.alloc(
            ).initWithEntryFunctionName_inputTypes_(name, input_types)
            placed_path.write_text(_extract_placed_source(
                executable,
                entry_point,
                fw=fw,
                graph_device=graph_device,
                compilation=compilation,
            ))
            targets = [
                (target_name, tuple(int(value) for value in shape))
                for target_name, _tensor, shape in program.targets
            ]
            state_results = dict(program.state_results)
            ane_input_order, ane_output_order = _write_multiprocedure_module(
                raw_path,
                placed_path,
                transformed_path,
                function=name,
                order=order,
                targets=targets,
                states=state_by_name,
                state_results=state_results,
            )
            entry_contracts.append(_EntryContract(
                name=name,
                function=name,
                region=f"{name}_ANE_region_0_0",
                order=tuple(order),
                targets=tuple(targets),
                state_results=tuple(state_results.items()),
                ane_input_order=ane_input_order,
                ane_output_order=ane_output_order,
                dynamic=frozenset(program.dynamic),
                product="",
            ))
            transformed_paths[name] = transformed_path
            raw_path.unlink()
            placed_path.unlink()
            del executable, entry_point, source, program
            gc.collect()

        _merge_modules(transformed_paths, staging / "model.mlir")
        (staging / "contract.json").write_text(json.dumps(
            _contract_json(
                dtype=dtype,
                states=states,
                entries=entry_contracts,
                format_version=_MULTIPROCEDURE_CACHE_FORMAT,
            ),
            indent=2,
        ))
        shutil.rmtree(build)
        try:
            staging.replace(cache_directory)
        except OSError:
            if not cache_directory.is_dir():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _read_cache_contract(
    cache_directory: Path,
    *,
    expected_entries: Sequence[str],
    expected_states: Sequence[StateTensorSpec],
    expected_dtype: int,
) -> tuple[_EntryContract, ...] | None:
    contract_path = cache_directory / "contract.json"
    module_path = cache_directory / "model.mlir"
    if not contract_path.is_file() or not module_path.is_file():
        return None
    dtype, states, entries = _parse_contract(json.loads(contract_path.read_text()))
    if dtype != expected_dtype or states != tuple(expected_states):
        raise RuntimeError(f"MPSGraph state cache contract changed at {cache_directory}")
    if [entry.name for entry in entries] != list(expected_entries):
        raise RuntimeError(f"MPSGraph state cache entries changed at {cache_directory}")
    for entry in entries:
        product = cache_directory / entry.product
        compiler_options = (
            product.parent / f"compiler_options_{entry.region}.plist"
        )
        if not (
            product.is_file()
            and product.with_suffix(".weights").is_file()
            and compiler_options.is_file()
        ):
            raise RuntimeError(
                f"{entry.function}: cached ANE product is incomplete"
            )
    return entries


def _read_multiprocedure_contract(
    cache_directory: Path,
    *,
    expected_entries: Sequence[str],
    expected_states: Sequence[StateTensorSpec],
    expected_dtype: int,
) -> tuple[_EntryContract, ...] | None:
    contract_path = cache_directory / "contract.json"
    module_path = cache_directory / "model.mlir"
    if not contract_path.is_file() or not module_path.is_file():
        return None
    dtype, states, entries = _parse_contract(
        json.loads(contract_path.read_text()),
        expected_format=_MULTIPROCEDURE_CACHE_FORMAT,
    )
    if dtype != expected_dtype or states != tuple(expected_states):
        raise RuntimeError(
            f"MPSGraph multiprocedure cache contract changed at "
            f"{cache_directory}")
    if [entry.name for entry in entries] != list(expected_entries):
        raise RuntimeError(
            f"MPSGraph multiprocedure entries changed at {cache_directory}")
    if any(entry.function != entry.name or entry.product for entry in entries):
        raise RuntimeError(
            f"MPSGraph multiprocedure cache ABI changed at {cache_directory}")
    return entries


def _load_cache(
    cache_directory: Path,
    *,
    expected_entries: Sequence[str],
    expected_states: Sequence[StateTensorSpec],
    expected_dtype: int,
) -> StatefulExecutable | None:
    entries = _read_cache_contract(
        cache_directory,
        expected_entries=expected_entries,
        expected_states=expected_states,
        expected_dtype=expected_dtype,
    )
    if entries is None:
        return None

    module_path = cache_directory / "model.mlir"
    dtype = expected_dtype
    states = tuple(expected_states)
    fw = mg._fw()
    _compilation, descriptor = _source_descriptor(fw)
    source = _load_mlir_source(module_path, fw)
    executable = fw["MPSGraphExecutable"].alloc(
    ).initWithMLIRSource_executableDescriptor_(source, descriptor)
    executable.setOptions_(0)
    functions = [str(value) for value in executable.functionNames()]
    if set(functions) != {entry.function for entry in entries}:
        raise RuntimeError(
            f"MPSGraph state cache functions changed at {cache_directory}")
    metal = fw["Metal"].MTLCreateSystemDefaultDevice()
    archive_directory = Path(str(executable.getMutableWeightsFilePath())).parent
    if not (
        archive_directory.name.startswith(f"mpsgraph-{os.getpid()}-")
        and archive_directory.parent.name
        == "com.apple.MetalPerformanceShadersGraph"
    ):
        raise RuntimeError(
            f"MPSGraph returned an unexpected product directory: "
            f"{archive_directory}")
    # Importing already-placed IR eagerly emits products into the framework's
    # process-private archive. Remove only those exact derived files before
    # installing the durable cache map; otherwise MPSGraph rejects the map as
    # an attempt to overwrite its own just-created product.
    for entry in entries:
        for suffix in (".plist", ".weights"):
            generated = archive_directory / f"{entry.region}{suffix}"
            if generated.exists():
                generated.unlink()
    EntryPoint = fw["MPSGraphExecutableEntryPoint"]
    entry_points = {}
    per_entry_map = {}
    state_names = {state.name for state in states}
    for entry in entries:
        positions = list(
            executable.getStateInputPositionsWithEntryFunctionName_(
                entry.function))
        expected_positions = [
            index for index, (name, _shape) in enumerate(entry.order)
            if name in state_names
        ]
        if positions != expected_positions:
            raise RuntimeError(
                f"{entry.function}: cached state positions changed to {positions}")
        input_types = [
            fw["MPSGraphShapedType"].alloc().initWithShape_dataType_(
                list(shape), dtype)
            for _name, shape in entry.order
        ]
        point = EntryPoint.alloc().initWithEntryFunctionName_inputTypes_(
            entry.function, input_types)
        product = cache_directory / entry.product
        entry_points[entry.name] = point
        per_entry_map[point] = {
            entry.region: os.path.relpath(product, archive_directory)}
    MapClass = fw["objc"].lookUpClass(
        "MPSGraphExecutableEntryPointToSymbolAndFileNameMap")
    entry_map = MapClass.alloc().initWithPerEntryPointMap_(per_entry_map)
    result = StatefulExecutable(
        executable, metal, dtype, states, entries, entry_map)
    result._entry_points = entry_points
    return result


def _load_multiprocedure_cache(
    cache_directory: Path,
    *,
    expected_entries: Sequence[str],
    expected_states: Sequence[StateTensorSpec],
    expected_dtype: int,
    ane_fw_to_fw_signal: bool,
    ane_late_latch: bool,
) -> StatefulExecutable | None:
    entries = _read_multiprocedure_contract(
        cache_directory,
        expected_entries=expected_entries,
        expected_states=expected_states,
        expected_dtype=expected_dtype,
    )
    if entries is None:
        return None

    fw = mg._fw()
    compilation, descriptor = _ane_source_descriptor(
        fw,
        ane_fw_to_fw_signal=ane_fw_to_fw_signal,
        ane_late_latch=ane_late_latch,
    )
    source = _load_mlir_source(cache_directory / "model.mlir", fw)
    executable = fw["MPSGraphExecutable"].alloc(
    ).initWithMLIRSource_executableDescriptor_(source, descriptor)
    if executable is None:
        raise RuntimeError("MPSGraph could not import the multiprocedure module")
    executable.setOptions_(0)
    functions = [str(value) for value in executable.functionNames()]
    if set(functions) != {entry.function for entry in entries}:
        raise RuntimeError(
            f"MPSGraph multiprocedure functions changed at {cache_directory}")

    metal = fw["Metal"].MTLCreateSystemDefaultDevice()
    graph_device = fw["MPSGraphDevice"].deviceWithMTLDevice_(metal)
    archive = Path(str(executable.getMutableWeightsFilePath())).parent
    if not (
        archive.name.startswith(f"mpsgraph-{os.getpid()}-")
        and archive.parent.name == "com.apple.MetalPerformanceShadersGraph"
    ):
        raise RuntimeError(
            f"MPSGraph returned an unexpected product directory: {archive}")

    # Import compiles one provisional bytecode product.  Remove only that
    # exact product and its derived options before registering every shaped
    # entry at once; the all-entry specialization is what records all
    # procedures under a single ANE model identity.
    provisional = sorted(archive.glob("*.bc.mlir"))
    if len(provisional) != 1:
        raise RuntimeError(
            f"MPSGraph emitted {len(provisional)} provisional grouped "
            f"products at {archive}")
    base = provisional[0].name.removesuffix(".bc.mlir")
    provisional_options = archive / f"compiler_options_{base}.plist"
    if not provisional_options.is_file():
        raise RuntimeError(
            f"MPSGraph omitted grouped compiler options for {base}")
    provisional[0].unlink()
    provisional_options.unlink()

    EntryPoint = fw["objc"].lookUpClass(
        "MPSGraphExecutableShapedEntryPoint")
    points = []
    state_names = {state.name for state in expected_states}
    for entry in entries:
        input_types = [
            fw["MPSGraphShapedType"].alloc().initWithShape_dataType_(
                list(shape), expected_dtype)
            for _name, shape in entry.order
        ]
        point = EntryPoint.alloc().initWithEntryFunctionName_inputTypes_(
            entry.function, input_types)
        points.append(point)
        positions = list(
            executable.getStateInputPositionsWithEntryFunctionName_(
                entry.function))
        expected_positions = [
            index for index, (name, _shape) in enumerate(entry.order)
            if name in state_names
        ]
        if positions != expected_positions:
            raise RuntimeError(
                f"{entry.function}: grouped state positions changed to "
                f"{positions}")
    executable.specializeWithDevice_shapedEntryPoints_compilationDescriptor_error_(
        graph_device, points, compilation, None)
    grouped_products = sorted(archive.glob("*.bc.mlir"))
    if len(grouped_products) != 1:
        raise RuntimeError(
            "MPSGraph did not specialize the procedures as one grouped "
            f"product: {grouped_products}")

    result = StatefulExecutable(
        executable,
        metal,
        expected_dtype,
        expected_states,
        entries,
        None,
    )
    result._entry_points = tuple(points)
    result._grouped_product = grouped_products[0]
    return result


def _stateful_request(
    factories: Mapping[str, Callable[[], Program]],
    states: Sequence[StateTensorSpec],
) -> tuple[list[str], tuple[StateTensorSpec, ...]]:
    if not factories:
        raise ValueError("MPSGraph state executable needs at least one entry")
    names = list(factories)
    if len(set(names)) != len(names):
        raise ValueError("MPSGraph state entry names must be unique")
    state_tuple = tuple(states)
    if (
        not state_tuple
        or len({state.name for state in state_tuple}) != len(state_tuple)
    ):
        raise ValueError("MPSGraph state names must be nonempty and unique")
    return names, state_tuple


def _ensure_stateful_cache(
    factories: Mapping[str, Callable[[], Program]],
    states: Sequence[StateTensorSpec],
    *,
    dtype: int,
    cache_directory: str | Path,
    ane_fw_to_fw_signal: bool,
    ane_late_latch: bool,
) -> tuple[Path, tuple[StateTensorSpec, ...], tuple[_EntryContract, ...]]:
    names, state_tuple = _stateful_request(factories, states)
    resolved = Path(cache_directory) / _system_cache_key()
    entries = _read_cache_contract(
        resolved,
        expected_entries=names,
        expected_states=state_tuple,
        expected_dtype=dtype,
    )
    if entries is None:
        _build_cache(
            resolved,
            factories,
            state_tuple,
            dtype=dtype,
            ane_fw_to_fw_signal=ane_fw_to_fw_signal,
            ane_late_latch=ane_late_latch,
        )
        entries = _read_cache_contract(
            resolved,
            expected_entries=names,
            expected_states=state_tuple,
            expected_dtype=dtype,
        )
    if entries is None:
        raise RuntimeError(
            f"MPSGraph state cache was not published at {resolved}"
        )
    return resolved, state_tuple, entries


def _ensure_multiprocedure_cache(
    factories: Mapping[str, Callable[[], Program]],
    states: Sequence[StateTensorSpec],
    *,
    dtype: int,
    cache_directory: str | Path,
    ane_fw_to_fw_signal: bool,
    ane_late_latch: bool,
) -> tuple[Path, tuple[StateTensorSpec, ...], tuple[_EntryContract, ...]]:
    names, state_tuple = _stateful_request(factories, states)
    resolved = Path(cache_directory) / _system_cache_key()
    entries = _read_multiprocedure_contract(
        resolved,
        expected_entries=names,
        expected_states=state_tuple,
        expected_dtype=dtype,
    )
    if entries is None:
        _build_multiprocedure_cache(
            resolved,
            factories,
            state_tuple,
            dtype=dtype,
            ane_fw_to_fw_signal=ane_fw_to_fw_signal,
            ane_late_latch=ane_late_latch,
        )
        entries = _read_multiprocedure_contract(
            resolved,
            expected_entries=names,
            expected_states=state_tuple,
            expected_dtype=dtype,
        )
    if entries is None:
        raise RuntimeError(
            f"MPSGraph multiprocedure cache was not published at {resolved}")
    return resolved, state_tuple, entries


def compile_stateful(
    factories: Mapping[str, Callable[[], Program]],
    states: Sequence[StateTensorSpec],
    *,
    dtype: int = mg.FLOAT16,
    cache_directory: str | Path,
    ane_fw_to_fw_signal: bool = False,
    ane_late_latch: bool = False,
) -> StatefulExecutable:
    """Build or load one persistent-state executable for named programs.

    Factories are intentionally lazy: a warm cache load does not construct
    large source graphs merely to rediscover the saved I/O contract.
    The caller's versioned cache path is the topology/weights identity; this
    layer adds an OS-runtime component because the lowered dialect is private.
    """
    resolved, state_tuple, entries = _ensure_stateful_cache(
        factories,
        states,
        dtype=dtype,
        cache_directory=cache_directory,
        ane_fw_to_fw_signal=ane_fw_to_fw_signal,
        ane_late_latch=ane_late_latch,
    )
    loaded = _load_cache(
        resolved,
        expected_entries=[entry.name for entry in entries],
        expected_states=state_tuple,
        expected_dtype=dtype,
    )
    if loaded is None:
        raise RuntimeError(f"MPSGraph state cache was not published at {resolved}")
    return loaded


def compile_multiprocedure(
    factories: Mapping[str, Callable[[], Program]],
    states: Sequence[StateTensorSpec],
    *,
    dtype: int = mg.FLOAT16,
    cache_directory: str | Path,
    ane_fw_to_fw_signal: bool = False,
    ane_late_latch: bool = False,
) -> StatefulExecutable:
    """Build or load named procedures sharing one persistent ANE model.

    Each factory remains an ordinary high-level MPSGraph. This layer moves it
    behind an ``mpsx.ane`` procedure, wraps contracted state results in the
    canonical outer variable update sequence, and specializes every entry
    together. Warm loads reuse the transformed source and never rebuild the
    family graph.
    """
    resolved, state_tuple, entries = _ensure_multiprocedure_cache(
        factories,
        states,
        dtype=dtype,
        cache_directory=cache_directory,
        ane_fw_to_fw_signal=ane_fw_to_fw_signal,
        ane_late_latch=ane_late_latch,
    )
    loaded = _load_multiprocedure_cache(
        resolved,
        expected_entries=[entry.name for entry in entries],
        expected_states=state_tuple,
        expected_dtype=dtype,
        ane_fw_to_fw_signal=ane_fw_to_fw_signal,
        ane_late_latch=ane_late_latch,
    )
    if loaded is None:
        raise RuntimeError(
            f"MPSGraph multiprocedure cache was not published at {resolved}")
    return loaded


def compile_stateful_direct(
    factories: Mapping[str, Callable[[], Program]],
    states: Sequence[StateTensorSpec],
    *,
    dtype: int = mg.FLOAT16,
    cache_directory: str | Path,
    ane_fw_to_fw_signal: bool = False,
    ane_late_latch: bool = False,
) -> Any:
    """Build/load stateful products and execute them with explicit ANE life.

    The compilation and durable contract are identical to
    :func:`compile_stateful`; only runtime ownership changes.  Consecutive
    evaluations retain the active program and its binding maps; a semantic
    entry change performs the explicit unload/load transition.  All entries
    bind the same state IOSurfaces.
    """
    resolved, state_tuple, entries = _ensure_stateful_cache(
        factories,
        states,
        dtype=dtype,
        cache_directory=cache_directory,
        ane_fw_to_fw_signal=ane_fw_to_fw_signal,
        ane_late_latch=ane_late_latch,
    )
    from . import anecir

    return anecir.StatefulExecutable(
        resolved, dtype, state_tuple, entries
    )


__all__ = [
    "Program",
    "StateTensorSpec",
    "StatefulEntry",
    "StatefulExecutable",
    "TensorBinding",
    "compile_multiprocedure",
    "compile_stateful",
    "compile_stateful_direct",
    "multiprocedure_cache_ready",
    "safe_storage_shape",
    "stateful_cache_ready",
    "state_placeholders",
    "state_result",
]
