"""Model-neutral contracts for persistent MPSGraph ANE state."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from kinovsr.native import mpsgraph_state as state

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("logical", "storage"),
    [
        ((1, 144, 240, 320), (2, 144, 240, 160)),
        ((1, 288, 120, 160), (2, 144, 120, 160)),
        ((1, 8, 32, 32), (1, 8, 32, 32)),
    ],
)
def test_safe_storage_shape_preserves_elements_below_ane_limits(
    logical, storage
):
    spec = state.StateTensorSpec.create("memory", logical)
    assert spec.logical_shape == logical
    assert spec.storage_shape == storage


def test_state_spec_rejects_unsafe_or_incompatible_contracts():
    with pytest.raises(ValueError, match="state name"):
        state.StateTensorSpec.create("not.a.port", (1, 8, 32, 32))
    with pytest.raises(ValueError, match="dimension limit"):
        state.StateTensorSpec("memory", (1, 8, 32, 32), (1, 8, 32, 256))
    with pytest.raises(ValueError, match="element count"):
        state.StateTensorSpec("memory", (1, 8, 32, 32), (1, 8, 16, 32))


@pytest.mark.parametrize("input_view", [False, True])
def test_structural_lowering_tracks_ports_without_ssa_number_assumptions(
    input_view
):
    tensor = "tensor<1x8x32x32xf16>"
    memref = "memref<1x8x32x32xf16>"
    if input_view:
        state_read = (
            f'    %state_view = "anec.input_view"(%ane3) '
            f'{{offset = 0 : i64}} : ({memref}) -> {memref}\n'
            f'    %state_view_unused = "anec.input_view"(%ane3) '
            f'{{offset = 0 : i64}} : ({memref}) -> {memref}\n'
        )
        state_value = "%state_view"
    else:
        state_read = ""
        state_value = "%ane3"
    body = (
        f"  anec.A14 @entry_0_ane_region_0_0(%ane7: {memref}, "
        f"%ane3: {memref}) -> ({memref}, {memref}) {{\n"
        f"{state_read}"
        f'    %sum = "anec.scaled_elementwise"(%ane7, {state_value}) '
        f'{{mode = "add"}} : ({memref}, {memref}) -> {memref}\n'
        f'    "anec.region_return"(%sum, %sum) : '
        f'({memref}, {memref}) -> ()\n'
        "  }\n"
        f"  func.func @entry_0(%feed99: {tensor}, %state41: {tensor}) -> "
        f"({tensor}, {tensor}) attributes {{mps.disablePreAllocate, "
        "mps.fullyPlacedOnANE} {\n"
        f'    %to_value = "placement.tensor_to_memref"(%feed99) : '
        f'({tensor}) -> {memref}\n'
        f'    %to_state = "placement.tensor_to_memref"(%state41) : '
        f'({tensor}) -> {memref}\n'
        f'    %call782:2 = "placement.region_call"(%to_value, %to_state) '
        "{callee = @entry_0_ane_region_0_0, mps.regionSHA = \"old\", "
        "region_type = #placement.region_type<ANE>} : "
        f"({memref}, {memref}) -> ({memref}, {memref})\n"
        f'    %state_tensor = "placement.memref_to_tensor"(%call782#0) : '
        f'({memref}) -> {tensor}\n'
        f'    %visible = "placement.memref_to_tensor"(%call782#1) : '
        f'({memref}) -> {tensor}\n'
        f"    return %state_tensor, %visible : {tensor}, {tensor}\n"
        "  }\n"
    )
    spec = state.StateTensorSpec.create("memory", (1, 8, 32, 32))
    ane_input_order = []
    ane_output_order = []
    lowered = state._transform_placed_body(
        body,
        function="entry_0",
        order=(("value", spec.storage_shape), ("memory", spec.storage_shape)),
        targets=(("memory.next", spec.storage_shape),
                 ("visible", spec.logical_shape)),
        states={"memory": spec},
        state_results={"memory.next": "memory"},
        ane_input_order=ane_input_order,
        ane_output_order=ane_output_order,
    )

    assert '"anec.state"(%ane3)' in lowered
    assert lowered.count('"anec.state"(%ane3)') == 1
    assert lowered.count('"anec.ring_buffer_reader"') == 1
    assert lowered.count('"anec.input_view"(%kst_memory_read)') == (
        2 if input_view else 0)
    assert '"anec.ring_buffer_writer"' in lowered
    assert "mps.stateInputIndices = array<i64: 1>" in lowered
    assert "%call782:1" in lowered
    assert "%call782#0" in lowered
    assert "%call782#1" not in lowered
    assert "%state_tensor" not in lowered
    assert "KINO_STATE_entry_0" in lowered
    assert ane_input_order == ["value", "memory"]
    assert ane_output_order == ["visible"]


def test_structural_lowering_records_pruned_state_reinsertion_order():
    tensor = "tensor<1x8x32x32xf16>"
    memref = "memref<1x8x32x32xf16>"
    body = (
        f"  anec.A14 @entry_0_ane_region_0_0(%ane_value: {memref}, "
        f"%ane_second: {memref}) -> ({memref}, {memref}, {memref}) {{\n"
        f'    %sum = "anec.scaled_elementwise"(%ane_value, %ane_second) '
        f'{{mode = "add"}} : ({memref}, {memref}) -> {memref}\n'
        f'    "anec.region_return"(%sum, %sum, %sum) : '
        f'({memref}, {memref}, {memref}) -> ()\n'
        "  }\n"
        f"  func.func @entry_0(%feed: {tensor}, %first: {tensor}, "
        f"%second: {tensor}) -> ({tensor}, {tensor}, {tensor}) "
        "attributes {mps.disablePreAllocate, mps.fullyPlacedOnANE} {\n"
        f'    %to_value = "placement.tensor_to_memref"(%feed) : '
        f'({tensor}) -> {memref}\n'
        f'    %to_second = "placement.tensor_to_memref"(%second) : '
        f'({tensor}) -> {memref}\n'
        f'    %call:3 = "placement.region_call"(%to_value, %to_second) '
        "{callee = @entry_0_ane_region_0_0, mps.regionSHA = \"old\", "
        "region_type = #placement.region_type<ANE>} : "
        f"({memref}, {memref}) -> ({memref}, {memref}, {memref})\n"
        f'    %first_result = "placement.memref_to_tensor"(%call#0) : '
        f'({memref}) -> {tensor}\n'
        f'    %second_result = "placement.memref_to_tensor"(%call#1) : '
        f'({memref}) -> {tensor}\n'
        f'    %visible = "placement.memref_to_tensor"(%call#2) : '
        f'({memref}) -> {tensor}\n'
        f"    return %first_result, %second_result, %visible : "
        f"{tensor}, {tensor}, {tensor}\n"
        "  }\n"
    )
    first = state.StateTensorSpec.create("first", (1, 8, 32, 32))
    second = state.StateTensorSpec.create("second", (1, 8, 32, 32))
    ane_input_order = []
    state._transform_placed_body(
        body,
        function="entry_0",
        order=(
            ("value", first.storage_shape),
            ("first", first.storage_shape),
            ("second", second.storage_shape),
        ),
        targets=(
            ("first.next", first.storage_shape),
            ("second.next", second.storage_shape),
            ("visible", first.logical_shape),
        ),
        states={"first": first, "second": second},
        state_results={"first.next": "first", "second.next": "second"},
        ane_input_order=ane_input_order,
    )

    assert ane_input_order == ["value", "second", "first"]


def test_module_merge_deduplicates_resources_and_namespaces_aliases(tmp_path):
    payload = "0x0011223344556677"

    def source(entry: str, resource: str) -> str:
        return (
            "#map = affine_map<(d0) -> (d0)>\n"
            'module attributes {mps.aneRegionsSHA = "old"} {\n'
            f"  func.func @{entry}_0() -> () {{\n"
            f'    %constant = "mps.constant"() '
            f'<{{value = #mps.buffer_tensor<{resource}> : tensor<1xf16>}}> '
            " : () -> memref<1xf16, #map>\n"
            "    return\n"
            "  }\n"
            "}\n"
            "{-#\n"
            "  dialect_resources: {\n"
            "    mps: {\n"
            f'      {resource}: "{payload}"\n'
            "    }\n"
            "  }\n"
            "#-}\n"
        )

    first = tmp_path / "first.mlir"
    second = tmp_path / "second.mlir"
    merged = tmp_path / "merged.mlir"
    first.write_text(source("first", "first_blob"))
    second.write_text(source("second", "second_blob"))

    state._merge_modules({"first": first, "second": second}, merged)
    result = merged.read_text()

    assert result.count(payload) == 1
    assert result.count("#mps.buffer_tensor<kst_") == 2
    assert "#first_map" in result
    assert "#second_map" in result
    assert "@first_0" in result
    assert "@second_0" in result


def test_program_validation_rejects_ambiguous_or_unknown_ports():
    spec = state.StateTensorSpec.create("memory", (1, 8, 32, 32))
    shape = spec.storage_shape
    builder = SimpleNamespace(
        feeds=[(object(), shape, "memory"), (object(), shape, "value")],
        dtype=0,
    )
    program = state.Program(
        "entry",
        builder,
        [("visible", object(), shape)],
        {},
        {"missing"},
    )
    with pytest.raises(ValueError, match="unknown dynamic"):
        state._validate_program(program, {"memory": spec})

    builder.feeds.append((object(), shape, "value"))
    program.dynamic.clear()
    with pytest.raises(ValueError, match="feed names must be unique"):
        state._validate_program(program, {"memory": spec})


def test_stateful_cache_ready_requires_every_published_product(tmp_path):
    root = tmp_path / state._system_cache_key()
    root.mkdir()
    (root / "model.mlir").write_text("module {}\n")
    product = root / "products" / "entry.plist"
    product.parent.mkdir()
    product.touch()
    (root / "contract.json").write_text(json.dumps({
        "format": state._CACHE_FORMAT,
        "dtype": 0,
        "system": state._system_cache_key(),
        "states": [],
        "entries": [{
            "name": "entry",
            "function": "entry_0",
            "region": "entry_0_ane_region_0_0",
            "order": [],
            "targets": [],
            "state_results": [],
            "ane_input_order": [],
            "ane_output_order": [],
            "dynamic": [],
            "product": "products/entry.plist",
        }],
    }))

    assert not state.stateful_cache_ready(tmp_path)
    product.with_suffix(".weights").touch()
    assert not state.stateful_cache_ready(tmp_path)
    (product.parent / "compiler_options_entry_0_ane_region_0_0.plist").touch()
    assert state.stateful_cache_ready(tmp_path)


def test_prepare_attaches_cached_products_only_once(monkeypatch):
    calls = []

    class Device:
        @staticmethod
        def deviceWithMTLDevice_(metal):
            return ("device", metal)

    class Compilation:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return self

        def setOptimizationLevel_(self, value):
            self.optimization = value

        def setPreferredDevice_(self, value):
            self.device = value

        def setWaitForCompilationCompletion_(self, value):
            self.wait = value

    class Executable:
        def applyEntryPointToSymbolAndFileNameMap_device_compilationDescriptor_(
            self, entry_map, device, compilation
        ):
            calls.append((entry_map, device, compilation))

    monkeypatch.setattr(state.mg, "_fw", lambda: {
        "MPSGraphDevice": Device,
        "MPSGraphCompilationDescriptor": Compilation,
    })
    owner = SimpleNamespace(
        _prepare_lock=threading.Lock(),
        _closed=False,
        _prepared=False,
        _metal=object(),
        _exe=Executable(),
        _entry_map=object(),
    )

    state.StatefulExecutable.prepare(owner)
    state.StatefulExecutable.prepare(owner)

    assert owner._prepared
    assert len(calls) == 1


def _write_probes(*buffers: bytearray):
    return tuple(
        (
            f"out_{index}",
            memoryview(buffer),
            state._write_probe_offsets(len(buffer), 2),
        )
        for index, buffer in enumerate(buffers)
    )


def test_write_guard_recovers_only_an_all_results_untouched_dispatch():
    buffers = (bytearray(32), bytearray(64))
    calls = 0

    def dispatch():
        nonlocal calls
        calls += 1
        if calls == 4:
            for buffer in buffers:
                buffer[:] = bytes(len(buffer))

    state._run_with_write_guard(
        "entry", 2, 7, _write_probes(*buffers), dispatch)

    assert calls == 4


def test_write_guard_refuses_partial_results_without_retrying():
    first, second = bytearray(32), bytearray(64)
    calls = 0

    def dispatch():
        nonlocal calls
        calls += 1
        first[:] = bytes(len(first))

    with pytest.raises(RuntimeError, match="incomplete result writes"):
        state._run_with_write_guard(
            "entry", 2, 3, _write_probes(first, second), dispatch)

    assert calls == 1


def test_write_guard_bounds_an_unrecoverable_no_write():
    calls = 0

    def dispatch():
        nonlocal calls
        calls += 1

    with pytest.raises(RuntimeError, match="4 consecutive dispatches"):
        state._run_with_write_guard(
            "entry", 2, 11, _write_probes(bytearray(32)), dispatch)

    assert calls == 4
