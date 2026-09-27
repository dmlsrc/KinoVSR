"""Reusable mlprogram authoring for ANE processors, protobuf-backed.

`Graph` emits MIL operations into a protobuf Block with shape tracking, a
weights blob file, and a curated op library whose spellings encode what the
BSVD port learned the hard way (each rule cost a debugging session; see the
op docstrings):

* state writes must be a fully-attributed `slice_update` over that state's
  own `read_state`, lowered through `write_state` (:meth:`Graph.update_state`);
* channel slices use the torch-frontend mask spelling (:meth:`Graph.slice_channels`);
* `conv_transpose` carries an explicit `output_shape`
  (:meth:`Graph.conv_transpose2d`).

Anything not covered composes through the raw :meth:`Graph.op` escape hatch
plus the const helpers. Weights come in as MLX arrays and leave as fp16
bytes; nothing here touches numpy or coremltools.

Serialization is deterministic (`SerializeToString(deterministic=True)`), so
identical emissions produce identical packages byte for byte.
"""

import json
import struct
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast, overload

import mlx.core as mx

from . import schema

_BLOB_SENTINEL = 0xDEADBEEF
_BLOB_DTYPE_FLOAT16 = 1
_BLOB_FILENAME = "@model_path/weights/weight.bin"

# fp16 consts above this many bytes go to the weights blob file; below it
# they are inlined. Purely a size/IO tradeoff, not load-bearing.
BLOB_THRESHOLD = 4096

# One named input binding: a single value or several (concat's values).
type OpInput = str | list[str] | tuple[str, ...]
# (name, dims) per model input or state, in feature order.
type FeatureShapes = Sequence[tuple[str, Sequence[int]]]


# The MIL / Core ML message fields this module writes. The message classes are
# built at run time from the vendored descriptors (see schema), so each new
# message is bound to these Protocols where it is created.
class _Repeated[T](Protocol):
    def add(self) -> T: ...


class _RepeatedScalars[T](Protocol):
    def append(self, value: T, /) -> None: ...

    def extend(self, values: Iterable[T], /) -> None: ...


class _Map[T](Protocol):
    def __getitem__(self, key: str, /) -> T: ...


class _ConstantDimension(Protocol):
    size: int


class _Dimension(Protocol):
    constant: _ConstantDimension


class _TensorType(Protocol):
    dataType: int
    rank: int
    dimensions: _Repeated[_Dimension]


class _StateType(Protocol):
    wrappedType: _ValueType


class _ValueType(Protocol):
    tensorType: _TensorType
    stateType: _StateType


class _NamedValueType(Protocol):
    name: str
    type: _ValueType


class _Binding(Protocol):
    name: str


class _Argument(Protocol):
    arguments: _Repeated[_Binding]


class _RepeatedInts(Protocol):
    values: _RepeatedScalars[int]


class _RepeatedBools(Protocol):
    values: _RepeatedScalars[bool]


class _RepeatedStrings(Protocol):
    values: _RepeatedScalars[str]


class _RepeatedBytes(Protocol):
    values: bytes


class _TensorValue(Protocol):
    ints: _RepeatedInts
    bools: _RepeatedBools
    strings: _RepeatedStrings
    bytes: _RepeatedBytes


class _ImmediateValue(Protocol):
    tensor: _TensorValue


class _BlobFileValue(Protocol):
    fileName: str
    offset: int


class _Value(Protocol):
    type: _ValueType
    immediateValue: _ImmediateValue
    blobFileValue: _BlobFileValue


class _Operation(Protocol):
    type: str
    attributes: _Map[_Value]
    inputs: _Map[_Argument]
    outputs: _Repeated[_NamedValueType]


class _Block(Protocol):
    operations: _Repeated[_Operation]
    outputs: _RepeatedScalars[str]

    def CopyFrom(self, other: _Block, /) -> None: ...


class _Function(Protocol):
    inputs: _Repeated[_NamedValueType]
    opset: str
    block_specializations: _Map[_Block]


class _Program(Protocol):
    version: int
    functions: _Map[_Function]


class _ArrayFeatureType(Protocol):
    shape: _RepeatedScalars[int]
    dataType: int


class _StateFeatureType(Protocol):
    arrayType: _ArrayFeatureType


class _FeatureType(Protocol):
    multiArrayType: _ArrayFeatureType
    stateType: _StateFeatureType


class _FeatureDescription(Protocol):
    name: str
    type: _FeatureType


class _Metadata(Protocol):
    shortDescription: str


class _FunctionDescription(Protocol):
    name: str
    input: _Repeated[_FeatureDescription]
    output: _Repeated[_FeatureDescription]
    state: _Repeated[_FeatureDescription]


class _ModelDescription(Protocol):
    input: _Repeated[_FeatureDescription]
    output: _Repeated[_FeatureDescription]
    state: _Repeated[_FeatureDescription]
    metadata: _Metadata
    defaultFunctionName: str
    functions: _Repeated[_FunctionDescription]


class _Model(Protocol):
    specificationVersion: int
    description: _ModelDescription
    mlProgram: _Program

    def SerializeToString(self, *, deterministic: bool = False) -> bytes: ...


class BlobFile:
    """The weights/weight.bin blob-storage format.

    Byte-reversed from a known-good package: 64-byte header {u32 count,
    u32 version=2}; per blob a 64-byte-aligned metadata record
    {u32 0xDEADBEEF, u32 dtype (fp16=1), u64 sizeInBytes, u64 dataOffset}
    followed by 64-aligned raw data. BlobFileValue.offset points at the
    metadata record.
    """

    def __init__(self) -> None:
        self._entries: list[bytes] = []
        self._offsets: dict[bytes, int] = {}
        self._cursor = 64

    @staticmethod
    def _align(value: int) -> int:
        return (value + 63) & ~63

    def add_fp16(self, raw: bytes) -> int:
        # Unrolled and multi-function graphs reuse the same convolution
        # weights many times.  Blob references are immutable, so one exact
        # byte payload can safely serve every const operation that names it.
        if raw in self._offsets:
            return self._offsets[raw]
        meta_offset = self._align(self._cursor)
        data_offset = meta_offset + 64
        meta = struct.pack(
            "<IIQQ", _BLOB_SENTINEL, _BLOB_DTYPE_FLOAT16, len(raw), data_offset
        ).ljust(64, b"\x00")
        padded = raw.ljust(self._align(len(raw)), b"\x00")
        self._entries.append(b"\x00" * (meta_offset - self._cursor) + meta + padded)
        self._cursor = data_offset + len(padded)
        self._offsets[raw] = meta_offset
        return meta_offset

    def serialize(self) -> bytes:
        header = struct.pack("<II", len(self._entries), 2).ljust(64, b"\x00")
        return header + b"".join(self._entries)


def _set_tensor_type(message: _TensorType, dtype: int, dims: Sequence[int]) -> None:
    message.dataType = dtype
    if dims:
        message.rank = len(dims)
        for size in dims:
            message.dimensions.add().constant.size = int(size)


def _fp16_bytes(array: mx.array) -> tuple[bytes, tuple[int, ...]]:
    dense = mx.contiguous(array.astype(mx.float16))
    mx.eval(dense)
    return bytes(memoryview(dense)), tuple(int(s) for s in dense.shape)


class Graph:
    """Sequential MIL emitter with shape tracking and a weights blob."""

    def __init__(self, blob: BlobFile | None = None) -> None:
        self._block = cast(_Block, schema.Block())
        self.blob = blob if blob is not None else BlobFile()
        self.shape: dict[str, tuple[int, ...]] = {}
        self._seq = 0

    # ------------------------------------------------------------- plumbing

    def n(self, tag: str) -> str:
        """Unique op name; the counter makes emissions reproducible."""
        self._seq += 1
        return f"{tag}_{self._seq}"

    def register_input(self, name: str, dims: Sequence[int]) -> str:
        self.shape[name] = tuple(int(d) for d in dims)
        return name

    def _operation(self, op_type: str, name: str) -> _Operation:
        operation = self._block.operations.add()
        operation.type = op_type
        attr = operation.attributes["name"]
        attr.type.tensorType.dataType = schema.STRING
        attr.immediateValue.tensor.strings.values.append(name)
        return operation

    @overload
    def op(
        self,
        op_type: str,
        inputs: Mapping[str, OpInput],
        out_name: str,
        out_dims: Sequence[int],
        name: str | None = None,
        dtype: int | None = None,
    ) -> str: ...

    @overload
    def op(
        self,
        op_type: str,
        inputs: Mapping[str, OpInput],
        out_name: None,
        out_dims: None = None,
        *,
        name: str,
        dtype: int | None = None,
    ) -> None: ...

    def op(
        self,
        op_type: str,
        inputs: Mapping[str, OpInput],
        out_name: str | None,
        out_dims: Sequence[int] | None = None,
        name: str | None = None,
        dtype: int | None = None,
    ) -> str | None:
        """Raw escape hatch: named bindings in, one output (or none).

        The output is fp16 unless `dtype` names another schema dtype -
        the float32 translation-island casts are the one current user."""
        op_name = name or out_name
        assert op_name is not None  # the overloads require a name or an output
        operation = self._operation(op_type, op_name)
        for key, ref in inputs.items():
            refs = ref if isinstance(ref, (list, tuple)) else [ref]
            argument = operation.inputs[key]
            for r in refs:
                argument.arguments.add().name = r
        if out_name is not None:
            assert out_dims is not None  # the overloads pair an output with its dims
            named = operation.outputs.add()
            named.name = out_name
            _set_tensor_type(
                named.type.tensorType, schema.FLOAT16 if dtype is None else dtype, out_dims
            )
            self.shape[out_name] = tuple(int(d) for d in out_dims)
        return out_name

    # ---------------------------------------------------------------- consts

    def _const(
        self, name: str, dtype: int, dims: Sequence[int], fill: Callable[[_Value], None]
    ) -> str:
        operation = self._operation("const", name)
        named = operation.outputs.add()
        named.name = name
        _set_tensor_type(named.type.tensorType, dtype, dims)
        value = operation.attributes["val"]
        _set_tensor_type(value.type.tensorType, dtype, dims)
        fill(value)
        self.shape[name] = tuple(int(d) for d in dims)
        return name

    def const_i32(self, name: str, values: Iterable[int], dims: Sequence[int] | None = None) -> str:
        values = list(values)
        dims = (len(values),) if dims is None else dims

        def fill(value: _Value) -> None:
            value.immediateValue.tensor.ints.values.extend(int(v) for v in values)

        return self._const(name, schema.INT32, dims, fill)

    def const_i32_scalar(self, name: str, value: int) -> str:
        return self.const_i32(name, [value], dims=())

    def const_bool(
        self, name: str, values: Iterable[bool], dims: Sequence[int] | None = None
    ) -> str:
        values = list(values)
        dims = (len(values),) if dims is None else dims

        def fill(value: _Value) -> None:
            value.immediateValue.tensor.bools.values.extend(bool(v) for v in values)

        return self._const(name, schema.BOOL, dims, fill)

    def const_bool_scalar(self, name: str, value: bool) -> str:
        return self.const_bool(name, [value], dims=())

    def const_str(self, name: str, text: str) -> str:
        def fill(value: _Value) -> None:
            value.immediateValue.tensor.strings.values.append(text)

        return self._const(name, schema.STRING, (), fill)

    def fp16_const(self, name: str, array: mx.array) -> str:
        """fp16 tensor const from an MLX array; large ones go to the blob."""
        raw, dims = _fp16_bytes(array)
        if len(raw) > BLOB_THRESHOLD:
            offset = self.blob.add_fp16(raw)

            def fill(value: _Value) -> None:
                value.blobFileValue.fileName = _BLOB_FILENAME
                value.blobFileValue.offset = offset
        else:

            def fill(value: _Value) -> None:
                value.immediateValue.tensor.bytes.values = raw

        return self._const(name, schema.FLOAT16, dims, fill)

    # ------------------------------------------------- curated op spellings

    def relu6(self, x: str, name: str) -> str:
        return self.op("relu6", {"x": x}, name, self.shape[x])

    def relu(self, x: str, name: str) -> str:
        return self.op("relu", {"x": x}, name, self.shape[x])

    def upsample_bilinear2x(self, x: str, tag: str, align_corners: bool = True) -> str:
        """2x bilinear upsample in the reference serialization: int32 scalar
        scale factors plus a bool align_corners const (no half_pixel_centers,
        matching what the coremltools frontend emits for this op)."""
        opname = self.n(tag)
        inputs = {
            "x": x,
            "scale_factor_height": self.const_i32_scalar(f"{opname}_scale_factor_height_0", 2),
            "scale_factor_width": self.const_i32_scalar(f"{opname}_scale_factor_width_0", 2),
            "align_corners": self.const_bool_scalar(f"{opname}_align_corners_0", align_corners),
        }
        xs = self.shape[x]
        return self.op("upsample_bilinear", inputs, opname, (xs[0], xs[1], xs[2] * 2, xs[3] * 2))

    def binary(self, op_type: str, x: str, y: str, tag: str, name: str | None = None) -> str:
        """x op y with the output taking x's shape (y broadcasts into x)."""
        return self.op(op_type, {"x": x, "y": y}, name or self.n(tag), self.shape[x])

    def pixel_shuffle2x(self, x: str, tag: str) -> str:
        """Native 2x pixel shuffle: (1, C, H, W) -> (1, C/4, 2H, 2W).

        The single-op native shuffle miscomputes on the ANE when the
        INPUT has more than 256 channels (measured 2026-07-20: 512->128
        is badly wrong, 256->64 is bit-exact), so callers must slice
        wider tensors into <=256-channel groups and concat the shuffled
        groups - the channel->pixel mapping is block-diagonal, so the
        grouped form is exactly equivalent.
        """
        xs = self.shape[x]
        if xs[1] > 256:
            raise ValueError(
                f"pixel_shuffle2x input has {xs[1]} channels; the native "
                f"op is numerically wrong on the ANE above 256 - slice "
                f"into groups and concat instead"
            )
        opname = self.n(tag)
        factor = self.const_i32_scalar(f"{opname}_factor_0", 2)
        return self.op(
            "pixel_shuffle",
            {"x": x, "upscale_factor": factor},
            opname,
            (xs[0], xs[1] // 4, xs[2] * 2, xs[3] * 2),
        )

    def concat_channels(self, values: Sequence[str], tag: str, name: str | None = None) -> str:
        opname = name or self.n(tag)
        axis = self.const_i32_scalar(f"{opname}_axis_0", 1)
        inter = self.const_bool_scalar(f"{opname}_interleave_0", False)
        first = self.shape[values[0]]
        channels = sum(self.shape[v][1] for v in values)
        return self.op(
            "concat",
            {"values": list(values), "axis": axis, "interleave": inter},
            opname,
            (first[0], channels, first[2], first[3]),
        )

    def slice_channels(
        self, x: str, start: int, count: int, tag: str, name: str | None = None
    ) -> str:
        """Channel slice in the torch-frontend mask spelling.

        Dims that run to their end are expressed through end_mask with a
        placeholder end value. The ANE plan builder has treated equivalent
        slice spellings differently in this graph family; this is the one
        verified at production scale.
        """
        shape = self.shape[x]
        to_end = start + count == shape[1]
        opname = name or self.n(tag)
        begin = self.const_i32(f"{opname}_begin_0", [0, start, 0, 0])
        end = self.const_i32(
            f"{opname}_end_0", [1, 1 if to_end else start + count, shape[2], shape[3]]
        )
        mask = self.const_bool(f"{opname}_end_mask_0", [True, to_end, True, True])
        return self.op(
            "slice_by_index",
            {"x": x, "begin": begin, "end": end, "end_mask": mask},
            opname,
            (1, count, shape[2], shape[3]),
        )

    def conv2d(
        self,
        x: str,
        weight: mx.array,
        bias: mx.array | None,
        tag: str,
        stride: int = 1,
        pad: int = 1,
        relu6: bool = False,
        relu6_name: str | None = None,
    ) -> str:
        """3x3-family convolution; weight is an MLX OIHW array."""
        opname = self.n(tag)
        inputs = {"x": x, "weight": self.fp16_const(f"{opname}_weight_0", weight)}
        if bias is not None:
            inputs["bias"] = self.fp16_const(f"{opname}_bias_0", bias)
        inputs["strides"] = self.const_i32(f"{opname}_strides_0", [stride, stride])
        inputs["pad"] = self.const_i32(f"{opname}_pad_0", [pad] * 4)
        inputs["pad_type"] = self.const_str(f"{opname}_pad_type_0", "custom")
        inputs["dilations"] = self.const_i32(f"{opname}_dilations_0", [1, 1])
        inputs["groups"] = self.const_i32_scalar(f"{opname}_groups_0", 1)
        xs = self.shape[x]
        kh, kw = int(weight.shape[2]), int(weight.shape[3])
        out = (
            1,
            int(weight.shape[0]),
            (xs[2] + 2 * pad - kh) // stride + 1,
            (xs[3] + 2 * pad - kw) // stride + 1,
        )
        y = self.op("conv", inputs, opname, out)
        if relu6:
            y = self.relu6(y, relu6_name or self.n(f"{tag}_relu6"))
        return y

    def conv_transpose2d(
        self, x: str, weight: mx.array, tag: str, stride: int = 2, pad: int = 2
    ) -> str:
        """Transposed convolution; weight is an MLX [Cin, Cout, K, K] array.

        Emits the explicit `output_shape` const the reference converter's
        backend pass adds - spelling fidelity is load-bearing on the ANE.
        """
        opname = self.n(tag)
        xs = self.shape[x]
        kh, kw = int(weight.shape[2]), int(weight.shape[3])
        out = (
            1,
            int(weight.shape[1]),
            (xs[2] - 1) * stride - 2 * pad + kh,
            (xs[3] - 1) * stride - 2 * pad + kw,
        )
        inputs = {
            "x": x,
            "weight": self.fp16_const(f"{opname}_weight_0", weight),
            "strides": self.const_i32(f"{opname}_strides_0", [stride, stride]),
            "pad": self.const_i32(f"{opname}_pad_0", [pad] * 4),
            "pad_type": self.const_str(f"{opname}_pad_type_0", "custom"),
            "dilations": self.const_i32(f"{opname}_dilations_0", [1, 1]),
            "groups": self.const_i32_scalar(f"{opname}_groups_0", 1),
            "output_shape": self.const_i32(f"{opname}_output_shape_0", list(out)),
        }
        return self.op("conv_transpose", inputs, opname, out)

    def read_state(self, state_name: str) -> str:
        return self.op(
            "read_state",
            {"input": state_name},
            self.n(f"read_{state_name}"),
            self.shape[state_name],
        )

    def update_state(self, state_name: str, read_name: str, value_name: str) -> None:
        """The canonical MLState write: fully-attributed slice_update over
        this state's own read, lowered through write_state.

        Two rules, each worth an E5RT -14 if violated (verified on the BSVD
        graph, 16 states): the written value must be an in-place-style
        update whose base is the state's own read (a value merely derived
        from the read is rejected), and the slice_update must carry its
        complete attribute set - omitting stride/squeeze_mask compiles with
        one state and fails at sixteen, so a small probe does not clear it.
        """
        opname = self.n(f"{state_name}_assign")
        inputs = {
            "x": read_name,
            "update": value_name,
            "begin": self.const_i32(f"{opname}_begin_0", [0, 0, 0, 0]),
            "end": self.const_i32(f"{opname}_end_0", [0, 0, 0, 0]),
            "stride": self.const_i32(f"{opname}_stride_0", [1, 1, 1, 1]),
            "begin_mask": self.const_bool(f"{opname}_begin_mask_0", [False, True, True, True]),
            "end_mask": self.const_bool(f"{opname}_end_mask_0", [True, True, True, True]),
            "squeeze_mask": self.const_bool(
                f"{opname}_squeeze_mask_0", [False, False, False, False]
            ),
        }
        assign = self.op("slice_update", inputs, opname, self.shape[read_name])
        self.op(
            "write_state",
            {"data": assign, "input": state_name},
            None,
            name=self.n(f"{state_name}_write"),
        )

    # -------------------------------------------------------------- assembly

    def finish(
        self,
        inputs: FeatureShapes,
        states: FeatureShapes,
        output_names: Sequence[str],
        short_description: str,
        opset: str = "CoreML9",
        spec_version: int = 10,
    ) -> bytes:
        """Serialize the model. inputs/states are (name, dims) in feature
        order; output shapes come from the emitted ops.

        The defaults target macOS 26 (opset CoreML9 / spec version 10 -
        coremltools's iOS26; the compiled dialect prints ios19), matching
        this project's deployment floor. Measured behaviorally identical
        to CoreML8 for this op vocabulary - placement, outputs, and timing
        unchanged, verified bit-exact against the CoreML8-built references.
        Pass opset="CoreML8", spec_version=9 to emit the iOS18 form (the
        minimum MLState needs) if an older floor ever matters; packages at
        spec 10 refuse to load on pre-Tahoe systems.
        """
        model = cast(_Model, schema.Model())
        model.specificationVersion = spec_version

        desc = model.description
        for name, dims in inputs:
            feature = desc.input.add()
            feature.name = name
            feature.type.multiArrayType.shape.extend(int(d) for d in dims)
            feature.type.multiArrayType.dataType = schema.FEATURE_FLOAT16
        for name in output_names:
            feature = desc.output.add()
            feature.name = name
            feature.type.multiArrayType.shape.extend(self.shape[name])
            feature.type.multiArrayType.dataType = schema.FEATURE_FLOAT16
        for name, dims in states:
            feature = desc.state.add()
            feature.name = name
            array = feature.type.stateType.arrayType
            array.shape.extend(int(d) for d in dims)
            array.dataType = schema.FEATURE_FLOAT16
        desc.metadata.shortDescription = short_description

        prog = model.mlProgram
        prog.version = 1
        fn = prog.functions["main"]
        for name, dims in inputs:
            named = fn.inputs.add()
            named.name = name
            _set_tensor_type(named.type.tensorType, schema.FLOAT16, dims)
        for name, dims in states:
            named = fn.inputs.add()
            named.name = name
            _set_tensor_type(named.type.stateType.wrappedType.tensorType, schema.FLOAT16, dims)
        fn.opset = opset
        target = fn.block_specializations[opset]
        target.CopyFrom(self._block)
        target.outputs.extend(output_names)
        data: bytes = model.SerializeToString(deterministic=True)
        return data


def finish_functions(
    functions: Iterable[tuple[str, Graph, FeatureShapes, FeatureShapes, Sequence[str]]],
    short_description: str,
    default_function: str | None = None,
    opset: str = "CoreML9",
    spec_version: int = 10,
) -> bytes:
    """Serialize several graphs as functions of one ML Program asset.

    Each entry is ``(name, graph, inputs, states, output_names)`` and all
    graphs must have been constructed with the same :class:`BlobFile`.
    Core ML selects a function when loading the asset; sharing one package
    keeps repeated weights on disk exactly once.
    """
    functions = list(functions)
    if not functions:
        raise ValueError("at least one function is required")
    names = [name for name, *_rest in functions]
    if len(set(names)) != len(names):
        raise ValueError("function names must be unique")
    default_function = default_function or names[0]
    if default_function not in names:
        raise ValueError(f"unknown default function {default_function!r}")
    shared_blob = functions[0][1].blob
    if any(graph.blob is not shared_blob for _name, graph, *_rest in functions):
        raise ValueError("all functions must share one BlobFile")

    model = cast(_Model, schema.Model())
    model.specificationVersion = spec_version
    desc = model.description
    desc.defaultFunctionName = default_function
    desc.metadata.shortDescription = short_description
    program = model.mlProgram
    program.version = 1

    for name, graph, inputs, states, output_names in functions:
        function_desc = desc.functions.add()
        function_desc.name = name
        for input_name, dims in inputs:
            feature = function_desc.input.add()
            feature.name = input_name
            feature.type.multiArrayType.shape.extend(int(d) for d in dims)
            feature.type.multiArrayType.dataType = schema.FEATURE_FLOAT16
        for output_name in output_names:
            feature = function_desc.output.add()
            feature.name = output_name
            feature.type.multiArrayType.shape.extend(graph.shape[output_name])
            feature.type.multiArrayType.dataType = schema.FEATURE_FLOAT16
        for state_name, dims in states:
            feature = function_desc.state.add()
            feature.name = state_name
            array = feature.type.stateType.arrayType
            array.shape.extend(int(d) for d in dims)
            array.dataType = schema.FEATURE_FLOAT16

        function = program.functions[name]
        for input_name, dims in inputs:
            named = function.inputs.add()
            named.name = input_name
            _set_tensor_type(named.type.tensorType, schema.FLOAT16, dims)
        for state_name, dims in states:
            named = function.inputs.add()
            named.name = state_name
            _set_tensor_type(named.type.stateType.wrappedType.tensorType, schema.FLOAT16, dims)
        function.opset = opset
        target = function.block_specializations[opset]
        target.CopyFrom(graph._block)
        target.outputs.extend(output_names)

    data: bytes = model.SerializeToString(deterministic=True)
    return data


def write_package(directory: Path, model_bytes: bytes, blob: BlobFile) -> None:
    """Write a complete .mlpackage bundle at `directory`."""
    data = directory / "Data" / "com.apple.CoreML"
    (data / "weights").mkdir(parents=True, exist_ok=True)
    (data / "model.mlmodel").write_bytes(model_bytes)
    (data / "weights" / "weight.bin").write_bytes(blob.serialize())
    model_id = str(uuid.uuid4()).upper()
    weights_id = str(uuid.uuid4()).upper()
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            weights_id: {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Weights",
                "name": "weights",
                "path": "com.apple.CoreML/weights",
            },
            model_id: {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel",
            },
        },
        "rootModelIdentifier": model_id,
    }
    (directory / "Manifest.json").write_text(json.dumps(manifest, indent=4))
