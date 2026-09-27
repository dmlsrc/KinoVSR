"""Persistent-state MPSGraph programs executed directly on the ANE.

MPSGraph's public graph API has variables and read/assign operations, but it
does not document an ``MLState``-equivalent contract for persistent ANE state.
On macOS 27 MPSGraph places ANE work as an ``mpsx.ane`` procedure of ordinary
``mps`` operations and hands the Neural Engine compiler that procedure as MLIR
bytecode. This module wraps each caller program in one such procedure with its
state variables inside it, so the compiler realizes them as live ANE state,
and publishes the bytecode product with the semantic I/O contract that
:mod:`kinovsr.native.anecir` executes.

The wrapping works on the captured graph's own function, whose ports are the
caller's feeds and targets in order, so the product's ports follow that order
rather than compiler-chosen placement, pruning, or SSA numbering. MPSGraph's
specialization is consulted only for the module attributes and ANE family its
importer requires. Family code owns graph topology; this module owns state ABI
construction, product publication, and the cache contract.
"""

import ctypes
import gc
import json
import mmap
import platform
import plistlib
import re
import shutil
import struct
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import BinaryIO, Protocol, TypedDict, cast

from . import anecir
from . import mpsgraph as mg

_CACHE_FORMAT = 4
_RESOURCE_HEADER = b"{-#\n  dialect_resources: {\n    mps: {\n"
_RESOURCE_FOOTER = b"\n    }\n  }\n#-}\n"
_MODULE_MARKER = b"module attributes "
_PLAIN_MODULE_MARKER = b"module {"
_PROCEDURE_MARKER = "  mpsx.ane @"
_operation_pointer_type: _OperationPointerType | None = None


# Structural views of ObjC objects this module receives untyped (looked up by
# class name, or handed to a constructor); each lists only the selectors called.
class _OperationPointerType(Protocol):
    def __call__(self, *, c_void_p: int) -> object: ...


class _ExecutableDescriptor(Protocol):
    def init(self) -> _ExecutableDescriptor: ...
    def setCompilationDescriptor_(self, descriptor: object) -> None: ...


class _NSStringClass(Protocol):
    def stringWithContentsOfFile_encoding_error_(
        self, path: str, encoding: int, error: None
    ) -> object: ...


# JSON spelling of one (name, shape) pair: [name, [dims...]].
type _PortJson = list[str | list[int]]


class _StateJson(TypedDict):
    name: str
    logical_shape: list[int]
    storage_shape: list[int]


class _EntryJson(TypedDict):
    name: str
    function: str
    region: str
    order: list[_PortJson]
    targets: list[_PortJson]
    state_results: list[list[str]]
    ane_input_order: list[str]
    ane_output_order: list[str]
    dynamic: list[str]
    product: str


class _StateContract(TypedDict):
    format: int
    dtype: int
    system: str
    states: list[_StateJson]
    entries: list[_EntryJson]


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
            raise ValueError(f"MPSGraph state needs a positive rank-four shape: {logical}")
        if len(storage) != 4 or any(value < 1 for value in storage):
            raise ValueError(f"MPSGraph state storage must be positive rank four: {storage}")
        if any(value > 255 for value in storage):
            raise ValueError(f"MPSGraph state storage exceeds the ANE dimension limit: {storage}")
        if prod(logical) != prod(storage):
            raise ValueError(
                f"MPSGraph state storage changes element count: {logical} -> {storage}"
            )

    @classmethod
    def create(cls, name: str, logical_shape: Sequence[int]) -> StateTensorSpec:
        logical = tuple(int(value) for value in logical_shape)
        return cls(name, logical, safe_storage_shape(logical))


@dataclass
class Program:
    """An ordinary graph plus the results that update named state tensors."""

    name: str
    builder: mg.GraphBuilder
    targets: list[tuple[str, object, tuple[int, ...]]]
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
                (
                    candidate
                    for candidate in range(2, values[axis] + 1)
                    if values[axis] % candidate == 0
                ),
                None,
            )
            if factor is None:
                raise ValueError(f"cannot factor state axis {values[axis]} below {limit}")
            leading *= factor
            values[axis] //= factor
    values[0] = leading
    if any(value > limit for value in values):
        raise ValueError(
            f"state shape {tuple(shape)} has no four-D storage below {limit}: {tuple(values)}"
        )
    if prod(values) != prod(int(value) for value in shape):
        raise AssertionError("state storage reshape changed element count")
    return tuple(values)


def state_placeholders(
    builder: mg.GraphBuilder, states: Sequence[StateTensorSpec]
) -> dict[str, object]:
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
    builder: mg.GraphBuilder, state: StateTensorSpec, value: object
) -> tuple[str, object, tuple[int, ...]]:
    """Expose a logical update through the state's physical result shape."""
    name = state.name + ".next"
    return (
        name,
        builder.reshape(value, state.storage_shape, name + ".storage"),
        state.storage_shape,
    )


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


def _state_attrs(indices: Sequence[int]) -> str:
    return "mps.stateInputIndices = array<i64: " + ", ".join(str(index) for index in indices) + ">"


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


def _tensor_declaration(declaration: str, *, function: str) -> tuple[str, str]:
    """Return the SSA name and bare tensor type from a function argument."""
    formal, separator, remainder = declaration.partition(":")
    formal = formal.strip()
    if not separator or not re.fullmatch(r"%[A-Za-z0-9_.$-]+", formal):
        raise RuntimeError(f"{function}: cannot parse function argument {declaration!r}")
    remainder = remainder.lstrip()
    if not remainder.startswith("tensor<"):
        raise RuntimeError(f"{function}: function argument is not a tensor: {declaration!r}")
    angle = remainder.index("<")
    end = _balanced_end(remainder, angle)
    return formal, remainder[: end + 1]


def _bare_tensor_type(value: str, *, function: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("tensor<"):
        raise RuntimeError(f"{function}: result is not a tensor type: {value!r}")
    end = _balanced_end(stripped, stripped.index("<"))
    trailing = stripped[end + 1 :].strip()
    if trailing:
        raise RuntimeError(f"{function}: unsupported result type suffix {trailing!r}")
    return stripped[: end + 1]


def _memref_type(tensor_type: str) -> str:
    if not tensor_type.startswith("tensor<"):
        raise ValueError(f"not a tensor type: {tensor_type!r}")
    return "memref<" + tensor_type[len("tensor<") :]


def _formatted_result_types(types: Sequence[str]) -> str:
    if len(types) == 1:
        return types[0]
    return "(" + ", ".join(types) + ")"


def _inline_state_procedure(
    body: str,
    *,
    function: str,
    family: str,
    identity: str,
    order: Sequence[tuple[str, tuple[int, ...]]],
    targets: Sequence[tuple[str, tuple[int, ...]]],
    states: Mapping[str, StateTensorSpec],
    state_results: Mapping[str, str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Wrap one captured graph as an ``mpsx.ane`` procedure with inline state.

    Each state feed becomes a variable read at the top of the procedure and
    each state result a full-range update assigned back to that variable, so
    the ANE compiler realizes the variables as live state ports. The
    procedure's arguments are the graph's feeds in order and its results the
    non-state targets in order; the compiled product names them
    ``__arg{i}`` and ``__out:{k}``.
    """
    marker = f"func.func @{function}"
    if body.count(marker) != 1 or body.count("func.func @") != 1:
        raise RuntimeError(f"{function}: expected exactly one captured graph function")
    if "mpsx.ane" in body or "%kst_" in body:
        raise RuntimeError(f"{function}: captured graph uses reserved MPSX names")

    function_at = body.index(marker)
    args_start = body.index("(", function_at + len(marker))
    args_end = _balanced_end(body, args_start)
    declarations = _split_top_level(body[args_start + 1 : args_end])
    parsed_args = [_tensor_declaration(value, function=function) for value in declarations]
    if len(parsed_args) != len(order):
        raise RuntimeError(
            f"{function}: feed contract has {len(order)} values but IR has {len(parsed_args)}"
        )

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
    original_body = body[cursor + 1 : function_end]

    returns = list(re.finditer(r"(?m)^[ \t]*return(?:[ \t].*)?$", original_body))
    if len(returns) != 1:
        raise RuntimeError(f"{function}: expected one captured return, got {len(returns)}")
    return_match = returns[0]
    if original_body[return_match.end() :].strip():
        raise RuntimeError(f"{function}: operations follow the captured return")
    return_line = original_body[return_match.start() : return_match.end()]
    return_values, raw_return_types, _values, _types = _parse_return(return_line + "\n")
    return_types = [_bare_tensor_type(value, function=function) for value in raw_return_types]
    if len(return_values) != len(targets) or len(return_types) != len(targets):
        raise RuntimeError(
            f"{function}: target contract has {len(targets)} values but IR has {len(return_values)}"
        )

    feed_names = [name for name, _shape in order]
    if len(set(feed_names)) != len(feed_names):
        raise RuntimeError(f"{function}: feed contract names are not unique")
    missing_states = set(states) - set(feed_names)
    if missing_states:
        raise RuntimeError(f"{function}: missing state feeds {sorted(missing_states)}")
    target_names = [name for name, _shape in targets]
    unknown_results = set(state_results) - set(target_names)
    if unknown_results:
        raise RuntimeError(f"{function}: missing state results {sorted(unknown_results)}")
    result_positions = {
        index: state_results[name]
        for index, name in enumerate(target_names)
        if name in state_results
    }
    visible_positions = [index for index in range(len(targets)) if index not in result_positions]
    if not visible_positions:
        raise RuntimeError(f"{function}: mpsx.ane needs at least one visible result")
    state_positions = {name: index for index, name in enumerate(feed_names) if name in states}
    for position, state_name in sorted(result_positions.items()):
        if state_name not in state_positions:
            raise RuntimeError(f"{function}: state result refers to unknown feed {state_name!r}")
        input_type = parsed_args[state_positions[state_name]][1]
        if return_types[position] != input_type:
            raise RuntimeError(
                f"{function}: state result {target_names[position]!r} has type "
                f"{return_types[position]}, expected {input_type}"
            )

    input_memrefs = [_memref_type(value_type) for _formal, value_type in parsed_args]
    block_args = ", ".join(
        f"%kst_input_{index}: {memref}" for index, memref in enumerate(input_memrefs)
    )
    region_lines = ['  "mpsx.ane"() ({', f"  ^bb0({block_args}):"]
    for index, ((formal, tensor_type), memref_type) in enumerate(
        zip(parsed_args, input_memrefs, strict=True)
    ):
        cast_op = f'"placement.ane_io_cast"(%kst_input_{index}) : ({memref_type}) -> {tensor_type}'
        if feed_names[index] not in states:
            region_lines.append(f"    {formal} = {cast_op}")
            continue
        variable = f"%kst_state_var_{index}"
        region_lines.extend(
            (
                f"    %kst_state_input_{index} = {cast_op}",
                f'    {variable} = "mps.variable_from_tensor"(%kst_state_input_{index}) : '
                f"({tensor_type}) -> {tensor_type}",
                f'    {formal} = "mps.read_variable"({variable}) : '
                f"({tensor_type}) -> {tensor_type}",
            )
        )

    operations = original_body[: return_match.start()].strip("\n")
    if operations:
        region_lines.append(operations)
    region_lines.extend(
        (
            '    %kst_one = "mps.constant"() '
            "<{value = dense<1> : tensor<4xsi32>}> : () -> tensor<4xsi32>",
            '    %kst_zero = "mps.constant"() '
            "<{value = dense<0> : tensor<4xsi32>}> : () -> tensor<4xsi32>",
        )
    )
    for position, state_name in sorted(result_positions.items()):
        index = state_positions[state_name]
        formal, tensor_type = parsed_args[index]
        updated = f"%kst_state_update_{position}"
        region_lines.extend(
            (
                f'    {updated} = "mps.strided_slice_update"('
                f"{formal}, {return_values[position]}, %kst_zero, %kst_zero, %kst_one) "
                "<{begin_mask = 14 : ui32, end_mask = 15 : ui32, "
                "shrink_axis_mask = 0 : ui32}> : "
                f"({tensor_type}, {tensor_type}, tensor<4xsi32>, "
                f"tensor<4xsi32>, tensor<4xsi32>) -> {tensor_type}",
                f'    "mps.assign_variable"(%kst_state_var_{index}, {updated}) : '
                f"({tensor_type}, {tensor_type}) -> ()",
            )
        )

    visible_types = [return_types[index] for index in visible_positions]
    visible_memrefs = [_memref_type(value) for value in visible_types]
    outputs = []
    for slot, position in enumerate(visible_positions):
        output = f"%kst_output_{slot}"
        outputs.append(output)
        region_lines.append(
            f'    {output} = "placement.ane_io_cast"({return_values[position]}) : '
            f"({return_types[position]}) -> {visible_memrefs[slot]}"
        )
    region_lines.append(
        f'    "mpsx.region_return"({", ".join(outputs)}) : ({", ".join(visible_memrefs)}) -> ()'
    )
    symbol = f"{function}_ANE_region_0_0"
    region_lines.append(
        f'  }}) {{ane_family = "{family}", function_type = '
        f"({', '.join(input_memrefs)}) -> "
        f"{_formatted_result_types(visible_memrefs)}, "
        f'sym_name = "{symbol}"}} : () -> ()'
    )

    outer_lines = [
        f'    %kst_outer_input_{index} = "placement.tensor_to_memref"({formal}) : '
        f"({tensor_type}) -> {memref_type}"
        for index, ((formal, tensor_type), memref_type) in enumerate(
            zip(parsed_args, input_memrefs, strict=True)
        )
    ]
    count = len(visible_positions)
    call_lhs = "%kst_call" if count == 1 else f"%kst_call:{count}"
    operands = ", ".join(f"%kst_outer_input_{index}" for index in range(len(parsed_args)))
    outer_lines.append(
        f'    {call_lhs} = "placement.region_call"({operands}) '
        f'{{callee = @{symbol}, mps.regionSHA = "{identity}", '
        "region_type = #placement.region_type<ANE>} : "
        f"({', '.join(input_memrefs)}) -> {_formatted_result_types(visible_memrefs)}"
    )
    results = []
    for slot, (memref_type, tensor_type) in enumerate(
        zip(visible_memrefs, visible_types, strict=True)
    ):
        source = "%kst_call" if count == 1 else f"%kst_call#{slot}"
        result = f"%kst_result_{slot}"
        results.append(result)
        outer_lines.append(
            f'    {result} = "placement.memref_to_tensor"({source}) : '
            f"({memref_type}) -> {tensor_type}"
        )
    outer_lines.append(f"    return {', '.join(results)} : {', '.join(visible_types)}")
    outer_header = (
        f"  func.func @{function}({', '.join(declarations)}) -> "
        f"{_formatted_result_types(visible_types)} attributes "
        f"{{{_state_attrs(sorted(state_positions.values()))}}} {{"
    )
    replacement = "\n".join((*region_lines, outer_header, *outer_lines, "  }"))
    transformed = body[:function_at] + replacement + body[function_end + 1 :]
    return (
        transformed,
        tuple(feed_names),
        tuple(target_names[index] for index in visible_positions),
    )


class PlacedFormUnsupported(RuntimeError):
    """MPSGraph did not place the program as one whole-graph ``mpsx.ane``
    procedure, which is the form the direct stateful entry is built from
    (macOS 26 lowers placed regions to ``anec`` instead)."""


@dataclass(frozen=True)
class _ModuleBounds:
    preamble_end: int
    module_header_start: int
    module_header_end: int
    body_start: int
    body_end: int
    resources_start: int
    resources_end: int


def _module_bounds(path: Path) -> _ModuleBounds:
    with (
        path.open("rb") as source,
        mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as view,
    ):
        marker = view.find(b"{-#")
        if marker >= 0 and view[marker : marker + len(_RESOURCE_HEADER)] != _RESOURCE_HEADER:
            raise RuntimeError(f"{path}: unexpected MPSGraph resources")
        starts = [
            value
            for value in (
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
        if view[body_end : body_end + 1] != b"}":
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


def _copy_range(source: BinaryIO, target: BinaryIO, start: int, end: int) -> None:
    source.seek(start)
    remaining = end - start
    while remaining:
        payload = source.read(min(8 * 1024 * 1024, remaining))
        if not payload:
            raise RuntimeError("unexpected EOF while copying MPSGraph module")
        target.write(payload)
        remaining -= len(payload)


def _placed_family(body: str, *, function: str) -> str:
    """The ANE family MPSGraph gave its single whole-graph procedure."""
    lines = body.splitlines()
    procedures = [line for line in lines if line.startswith(_PROCEDURE_MARKER)]
    if len(procedures) != 1:
        raise PlacedFormUnsupported(
            f"{function}: MPSGraph placed {len(procedures)} mpsx.ane procedures; "
            "the direct entry needs exactly one"
        )
    functions = [line for line in lines if line.startswith("  func.func @")]
    if len(functions) != 1 or "mps.fullyPlacedOnANE" not in functions[0]:
        raise PlacedFormUnsupported(
            f"{function}: MPSGraph did not place the whole graph on the ANE"
        )
    families: list[str] = re.findall(r'\bane_family = "([A-Za-z0-9_]+)"', procedures[0])
    if len(families) != 1:
        raise PlacedFormUnsupported(f"{function}: the placed procedure names no ANE family")
    return families[0]


def _write_procedure_module(
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
    """Wrap raw graph IR as one procedure under the placed module attributes.

    MPSGraph's importer needs the module attributes and ANE family its own
    placement records; the procedure body comes from the raw graph. The
    module's region list and the call name the procedure by one identity.
    """
    raw_bounds = _module_bounds(raw_path)
    placed_bounds = _module_bounds(placed_path)
    with raw_path.open("rb") as raw:
        raw.seek(raw_bounds.body_start)
        raw_body = raw.read(raw_bounds.body_end - raw_bounds.body_start).decode("utf-8")
    with placed_path.open("rb") as placed:
        placed.seek(placed_bounds.module_header_start)
        header = placed.read(
            placed_bounds.module_header_end - placed_bounds.module_header_start
        ).decode("utf-8")
        placed.seek(placed_bounds.body_start)
        placed_body = placed.read(placed_bounds.body_end - placed_bounds.body_start).decode("utf-8")
    family = _placed_family(placed_body, function=function)
    regions = re.findall(r'mps\.aneRegionsSHA = "[^"]*"', header)
    if len(regions) != 1 or "mps.aneArch = " not in header:
        raise PlacedFormUnsupported(f"{function}: the placed module lacks its ANE attributes")
    identity = f"KINO_STATE_{function}"
    header = header.replace(regions[0], f'mps.aneRegionsSHA = "{identity}"')
    transformed, ane_inputs, ane_outputs = _inline_state_procedure(
        raw_body,
        function=function,
        family=family,
        identity=identity,
        order=order,
        targets=targets,
        states=states,
        state_results=state_results,
    )
    with output_path.open("wb") as output, raw_path.open("rb") as raw:
        _copy_range(raw, output, 0, raw_bounds.module_header_start)
        output.write(header.encode("utf-8"))
        output.write(transformed.encode("utf-8"))
        output.write(b"}\n")
        _copy_range(
            raw,
            output,
            raw_bounds.body_end + 1,
            raw_path.stat().st_size,
        )
    return ane_inputs, ane_outputs


def _namespace_resources(source: str, entry: str) -> str:
    names = set(re.findall(r"#mps\.buffer_tensor<([A-Za-z0-9_.$-]+)>", source))
    if not names:
        return source
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.$-])(?:"
        + "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
        + r")(?![A-Za-z0-9_.$-])"
    )
    return pattern.sub(lambda match: f"{entry}_{match.group(0)}", source)


def _source_descriptor(
    fw: mg._Framework,
) -> tuple[mg._CompilationDescriptor, _ExecutableDescriptor]:
    compilation = fw["MPSGraphCompilationDescriptor"].alloc().init()
    compilation.setOptimizationLevel_(0)
    compilation.setPreferredDevice_(mg.DEVICE_GPU)
    compilation.setWaitForCompilationCompletion_(True)
    descriptor = (
        cast(
            mg._Class[_ExecutableDescriptor],  # lookUpClass returns the named class
            fw["objc"].lookUpClass("MPSGraphExecutableDescriptor"),
        )
        .alloc()
        .init()
    )
    descriptor.setCompilationDescriptor_(compilation)
    return compilation, descriptor


def _ane_source_descriptor(
    fw: mg._Framework,
    *,
    ane_fw_to_fw_signal: bool,
    ane_late_latch: bool,
) -> tuple[mg._CompilationDescriptor, _ExecutableDescriptor]:
    compilation = fw["MPSGraphCompilationDescriptor"].alloc().init()
    compilation.setOptimizationLevel_(1)
    compilation.setPreferredDevice_(mg.DEVICE_ANE)
    compilation.setWaitForCompilationCompletion_(True)
    compilation.setEnableMLIRDiagnostics_(True)
    if ane_fw_to_fw_signal:
        compilation.setEnableANEFWToFWSignal_(True)
    if ane_late_latch:
        compilation.setEnableANELateLatch_(True)
    descriptor = (
        cast(
            mg._Class[_ExecutableDescriptor],  # lookUpClass returns the named class
            fw["objc"].lookUpClass("MPSGraphExecutableDescriptor"),
        )
        .alloc()
        .init()
    )
    descriptor.setCompilationDescriptor_(compilation)
    return compilation, descriptor


def _capture_program(
    program: Program,
    *,
    fw: mg._Framework,
    graph_device: object,
) -> tuple[mg._Executable, list[tuple[str, tuple[int, ...]]], str]:
    compilation, executable_descriptor = _source_descriptor(fw)
    shaped = {
        tensor: fw["MPSGraphShapedType"]
        .alloc()
        .initWithShape_dataType_(list(shape), program.builder.dtype)
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
        raise RuntimeError(f"{program.name}: expected one MPSGraph main function")
    source = source.replace(old, f"func.func @{program.name}", 1)
    source = _namespace_resources(source, program.name)
    executable = (
        fw["MPSGraphExecutable"]
        .alloc()
        .initWithMLIRSource_executableDescriptor_(source, executable_descriptor)
    )
    if executable is None:
        raise RuntimeError(f"{program.name}: MPSGraph could not import the captured module")
    functions = [str(value) for value in executable.functionNames()]
    if functions != [program.name]:
        raise RuntimeError(f"{program.name}: captured functions changed to {functions}")
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
    executable: mg._Executable,
    entry_point: object,
    *,
    fw: mg._Framework,
    graph_device: object,
    compilation: object,
) -> tuple[_ModuleOp, object]:
    """Resolve MPSGraph's resource-backed module reference to a ModuleOp.

    The private specialization API returns an inline SmallVector whose second
    field is an ``InMemoryModuleRef``.  Its virtual ``get`` method is the
    resource-aware step omitted by ``optimizedBytecode``.  Check the exact
    runtime symbol before calling it so a changed framework ABI fails closed.
    """
    vector = executable.specializedModuleWithDevice_shapedEntryPoints_compilationDescriptor_error_(
        graph_device, [entry_point], compilation, None
    )
    if not isinstance(vector, tuple) or len(vector) != 4:
        raise RuntimeError(f"MPSGraph returned an unexpected specialization vector: {vector!r}")
    begin, size, capacity, signed_inline = vector
    inline = bytes(value & 0xFF for value in signed_inline)
    if not begin or size != 1 or capacity < 1 or len(inline) < 16:
        raise RuntimeError(
            "MPSGraph specialization vector layout changed: "
            f"begin={begin!r}, size={size!r}, capacity={capacity!r}, "
            f"inline={len(inline)}"
        )
    module_ref = struct.unpack_from("<Q", inline, 8)[0]
    if not module_ref or module_ref % ctypes.sizeof(ctypes.c_void_p):
        raise RuntimeError(f"MPSGraph returned an invalid module reference: {module_ref:#x}")

    vtable = ctypes.c_void_p.from_address(module_ref).value
    if not vtable:
        raise RuntimeError("MPSGraph module reference has no virtual table")
    get_address = ctypes.c_void_p.from_address(vtable + 3 * ctypes.sizeof(ctypes.c_void_p)).value
    if not get_address:
        raise RuntimeError("MPSGraph module reference has no get method")
    symbol = _symbol_name(get_address)
    if "InMemoryModuleRef3get" not in symbol:
        raise RuntimeError(
            "MPSGraph module-reference ABI changed: "
            f"expected InMemoryModuleRef::get, got {symbol or hex(get_address)}"
        )

    get_module = ctypes.CFUNCTYPE(_ModuleOp, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))(
        get_address
    )
    error = ctypes.c_void_p()
    module = get_module(module_ref, ctypes.byref(error))
    if error.value:
        error_object = fw["objc"].objc_object(c_void_p=error.value)
        raise RuntimeError(f"MPSGraph could not resolve its specialized module: {error_object}")
    if not module.operation:
        raise RuntimeError("MPSGraph resolved a null specialized module")

    global _operation_pointer_type
    if _operation_pointer_type is None:
        objc = fw["objc"]
        name = "KinovsrMLIROperationPointer"
        _operation_pointer_type = getattr(objc, name, None)
        if _operation_pointer_type is None:
            _operation_pointer_type = cast(
                _OperationPointerType,  # a pointer type is called with c_void_p=
                objc.createOpaquePointerType(name, b"^{Operation}", "MLIR operation pointer"),
            )
    return module, _operation_pointer_type(c_void_p=module.operation)


def _extract_placed_source(
    executable: mg._Executable,
    entry_point: object,
    *,
    fw: mg._Framework,
    graph_device: object,
    compilation: object,
) -> str:
    module, operation_pointer = _resolve_specialized_module(
        executable,
        entry_point,
        fw=fw,
        graph_device=graph_device,
        compilation=compilation,
    )
    descriptor = (
        cast(
            mg._Class[_ExecutableDescriptor],  # lookUpClass returns the named class
            fw["objc"].lookUpClass("MPSGraphExecutableDescriptor"),
        )
        .alloc()
        .init()
    )
    descriptor.setCompilationDescriptor_(compilation)
    wrapped = (
        fw["MPSGraphExecutable"]
        .alloc()
        .initWithSpecializedMLIRModule_device_shapedEntryPoint_compilationDescriptor_executableDescriptor_(
            (operation_pointer,),
            graph_device,
            entry_point,
            compilation,
            descriptor,
        )
    )
    if wrapped is None:
        raise RuntimeError("MPSGraph could not wrap its specialized module")
    source = str(wrapped.getIR())
    # Keep the ctypes structure alive through the initializer call.  The
    # resulting executable owns the referenced module independently.
    del module
    return source


def _load_mlir_source(path: Path, fw: mg._Framework) -> object:
    NSString = cast(_NSStringClass, fw["objc"].lookUpClass("NSString"))  # the NSString class
    loaded = NSString.stringWithContentsOfFile_encoding_error_(str(path), 4, None)
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
    fw: mg._Framework,
) -> Path:
    """Import one procedure module and keep the ANE product MPSGraph emits.

    Import writes the procedure's MLIR bytecode and its ANE compiler options
    into the executable's process-private archive; the ANE compiles that
    bytecode when the product is first loaded.
    """
    _compilation, descriptor = _source_descriptor(fw)
    source = _load_mlir_source(module_path, fw)
    executable = (
        fw["MPSGraphExecutable"]
        .alloc()
        .initWithMLIRSource_executableDescriptor_(source, descriptor)
    )
    if executable is None:
        raise RuntimeError(f"{function}: MPSGraph could not import the procedure module")
    executable.setOptions_(0)
    functions = [str(value) for value in executable.functionNames()]
    if functions != [function]:
        raise RuntimeError(f"{function}: procedure functions changed to {functions}")
    archive = Path(str(executable.getMutableWeightsFilePath())).parent
    product = archive / f"{region}.bc.mlir"
    compiler_options = archive / f"compiler_options_{region}.plist"
    if not product.is_file() or not compiler_options.is_file():
        raise RuntimeError(f"{function}: MPSGraph did not emit the ANE product for {region}")
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(product, output_root / product.name)
    shutil.copy2(compiler_options, output_root / compiler_options.name)
    return output_root / product.name


def _retarget_compiler_options(product: Path, *, region: str, published_product: Path) -> None:
    """Point copied MPSGraph options at the durable published netplist."""
    path = product.parent / f"compiler_options_{region}.plist"
    with path.open("rb") as handle:
        options = plistlib.load(handle)
    if not isinstance(options, dict) or not options:
        raise RuntimeError(f"{region}: malformed ANE compiler options")
    for architecture, record in options.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"{region}: malformed options for architecture {architecture!r}")
        record["NetworkPlistName"] = region
        record["NetworkPlistPath"] = str(published_product)
    with path.open("wb") as handle:
        plistlib.dump(options, handle, fmt=plistlib.FMT_BINARY)


def _system_cache_key() -> str:
    value = f"{platform.mac_ver()[0]}-{platform.machine()}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _product_files(root: Path, entry: _EntryContract) -> tuple[Path, Path]:
    product = root / entry.product
    return product, product.parent / f"compiler_options_{entry.region}.plist"


def stateful_cache_ready(cache_directory: str | Path) -> bool:
    """Whether a durable stateful cache has every published product file."""
    root = Path(cache_directory) / _system_cache_key()
    contract = root / "contract.json"
    if not contract.is_file():
        return False
    try:
        _dtype, _states, entries = _parse_contract(json.loads(contract.read_text()))
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    return all(path.is_file() for entry in entries for path in _product_files(root, entry))


def _contract_json(
    *,
    dtype: int,
    states: Sequence[StateTensorSpec],
    entries: Sequence[_EntryContract],
) -> _StateContract:
    return {
        "format": _CACHE_FORMAT,
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
                "targets": [[name, list(shape)] for name, shape in entry.targets],
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
    record: _StateContract,
) -> tuple[int, tuple[StateTensorSpec, ...], tuple[_EntryContract, ...]]:
    if record.get("format") != _CACHE_FORMAT:
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
                (str(name), tuple(int(value) for value in shape)) for name, shape in item["order"]
            ),
            targets=tuple(
                (str(name), tuple(int(value) for value in shape)) for name, shape in item["targets"]
            ),
            state_results=tuple((str(name), str(state)) for name, state in item["state_results"]),
            ane_input_order=tuple(str(name) for name in item["ane_input_order"]),
            ane_output_order=tuple(str(name) for name in item["ane_output_order"]),
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
            raise RuntimeError(f"{entry.name}: cached ANE input order changed")
        if (
            len(entry.ane_output_order) != len(target_names)
            or len(set(entry.ane_output_order)) != len(target_names)
            or set(entry.ane_output_order) != set(target_names)
        ):
            raise RuntimeError(f"{entry.name}: cached ANE output order changed")
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
                f"{state.storage_shape}, got {feeds.get(name)}"
            )
    target_items = [
        (name, tuple(int(value) for value in shape)) for name, _tensor, shape in program.targets
    ]
    targets = dict(target_items)
    if len(targets) != len(target_items):
        raise ValueError(f"{program.name}: target names must be unique")
    if len(set(program.state_results.values())) != len(program.state_results):
        raise ValueError(f"{program.name}: each state may have at most one result")
    unknown_dynamic = set(program.dynamic) - (set(feeds) | set(targets))
    if unknown_dynamic:
        raise ValueError(f"{program.name}: unknown dynamic tensors {sorted(unknown_dynamic)}")
    targets = {
        name: tuple(int(value) for value in shape) for name, _tensor, shape in program.targets
    }
    for result_name, state_name in program.state_results.items():
        if state_name not in states:
            raise ValueError(f"{program.name}: unknown state result {state_name!r}")
        if targets.get(result_name) != states[state_name].storage_shape:
            raise ValueError(
                f"{program.name}: state result {result_name!r} must have "
                f"shape {states[state_name].storage_shape}"
            )


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
    compilation, _descriptor = _ane_source_descriptor(
        fw,
        ane_fw_to_fw_signal=ane_fw_to_fw_signal,
        ane_late_latch=ane_late_latch,
    )
    EntryPoint = cast(
        mg._Class[mg._EntryPoint],  # lookUpClass returns the named class
        fw["objc"].lookUpClass("MPSGraphExecutableShapedEntryPoint"),
    )
    parent = cache_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cache_directory.name}.partial-", dir=parent))
    build = staging / "build"
    build.mkdir()
    state_by_name = {state.name: state for state in states}
    entry_contracts = []
    try:
        for name, factory in factories.items():
            # A caller's thread may never drain a pool (a script's or test's
            # main thread). Drained per entry, this entry's executables are
            # freed after their last use, and MPSGraph removes their temp
            # archives.
            with fw["objc"].autorelease_pool():
                program = factory()
                if program.name != name:
                    raise ValueError(f"MPSGraph program factory {name!r} returned {program.name!r}")
                if program.builder.dtype != dtype:
                    raise ValueError(f"{name}: MPSGraph state dtype changed")
                _validate_program(program, state_by_name)
                executable, order, source = _capture_program(
                    program, fw=fw, graph_device=graph_device
                )
                raw_path = build / f"{name}.raw.mlir"
                placed_path = build / f"{name}.placed.mlir"
                module_path = build / f"{name}.mlir"
                raw_path.write_text(source)
                input_types = [
                    fw["MPSGraphShapedType"].alloc().initWithShape_dataType_(list(shape), dtype)
                    for _feed_name, shape in order
                ]
                entry_point = EntryPoint.alloc().initWithEntryFunctionName_inputTypes_(
                    name, input_types
                )
                placed_path.write_text(
                    _extract_placed_source(
                        executable,
                        entry_point,
                        fw=fw,
                        graph_device=graph_device,
                        compilation=compilation,
                    )
                )
                targets = [
                    (target_name, tuple(int(value) for value in shape))
                    for target_name, _tensor, shape in program.targets
                ]
                state_results = dict(program.state_results)
                ane_input_order, ane_output_order = _write_procedure_module(
                    raw_path,
                    placed_path,
                    module_path,
                    function=name,
                    order=order,
                    targets=targets,
                    states=state_by_name,
                    state_results=state_results,
                )
                region = f"{name}_ANE_region_0_0"
                product = _compile_product(
                    module_path,
                    staging / "products" / name,
                    function=name,
                    region=region,
                    fw=fw,
                )
                _retarget_compiler_options(
                    product,
                    region=region,
                    published_product=(cache_directory / product.relative_to(staging)),
                )
                entry_contracts.append(
                    _EntryContract(
                        name=name,
                        function=name,
                        region=region,
                        order=tuple(order),
                        targets=tuple(targets),
                        state_results=tuple(state_results.items()),
                        ane_input_order=ane_input_order,
                        ane_output_order=ane_output_order,
                        dynamic=frozenset(program.dynamic),
                        product=str(product.relative_to(staging)),
                    )
                )
                for path in (raw_path, placed_path, module_path):
                    path.unlink()
                del executable, entry_point, source, program
                gc.collect()

        (staging / "contract.json").write_text(
            json.dumps(
                _contract_json(dtype=dtype, states=states, entries=entry_contracts),
                indent=2,
            )
        )
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
    if not contract_path.is_file():
        return None
    dtype, states, entries = _parse_contract(json.loads(contract_path.read_text()))
    if dtype != expected_dtype or states != tuple(expected_states):
        raise RuntimeError(f"MPSGraph state cache contract changed at {cache_directory}")
    if [entry.name for entry in entries] != list(expected_entries):
        raise RuntimeError(f"MPSGraph state cache entries changed at {cache_directory}")
    for entry in entries:
        if not all(path.is_file() for path in _product_files(cache_directory, entry)):
            raise RuntimeError(f"{entry.name}: cached ANE product is incomplete")
    return entries


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
    if not state_tuple or len({state.name for state in state_tuple}) != len(state_tuple):
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
        raise RuntimeError(f"MPSGraph state cache was not published at {resolved}")
    return resolved, state_tuple, entries


def compile_stateful_direct(
    factories: Mapping[str, Callable[[], Program]],
    states: Sequence[StateTensorSpec],
    *,
    dtype: int = mg.FLOAT16,
    cache_directory: str | Path,
    ane_fw_to_fw_signal: bool = False,
    ane_late_latch: bool = False,
) -> anecir.StatefulExecutable:
    """Build or load stateful procedures and execute them with explicit ANE life.

    Factories are intentionally lazy: a warm cache load does not construct
    large source graphs merely to rediscover the saved I/O contract. The
    caller's versioned cache path is the topology/weights identity; this layer
    adds an OS-runtime component because the procedure form is private.
    Consecutive evaluations retain the active program and its binding maps; a
    semantic entry change performs the explicit unload/load transition. All
    entries bind the same state IOSurfaces.
    """
    resolved, state_tuple, entries = _ensure_stateful_cache(
        factories,
        states,
        dtype=dtype,
        cache_directory=cache_directory,
        ane_fw_to_fw_signal=ane_fw_to_fw_signal,
        ane_late_latch=ane_late_latch,
    )

    return anecir.StatefulExecutable(resolved, dtype, state_tuple, entries)


__all__ = [
    "PlacedFormUnsupported",
    "Program",
    "StateTensorSpec",
    "compile_stateful_direct",
    "safe_storage_shape",
    "state_placeholders",
    "state_result",
    "stateful_cache_ready",
]
