"""Model-neutral contracts for persistent MPSGraph ANE state."""

import json
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
def test_safe_storage_shape_preserves_elements_below_ane_limits(logical, storage):
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


_TENSOR = "tensor<1x8x32x32xf16>"
_MEMREF = "memref<1x8x32x32xf16>"
_RESOURCES = (
    "{-#\n  dialect_resources: {\n    mps: {\n"
    '      entry_blob: "0x0011223344556677"\n'
    "    }\n  }\n#-}\n"
)


def _raw_entry() -> str:
    """A captured graph: two states updated by a sum and a product."""
    t = _TENSOR
    return (
        f"  func.func @entry(%memory: {t}, %value: {t}, %second: {t}) -> ({t}, {t}, {t}) {{\n"
        f'    %sum = "mps.add"(%memory, %value) : ({t}, {t}) -> {t}\n'
        f'    %product = "mps.multiply"(%second, %value) : ({t}, {t}) -> {t}\n'
        f"    return %sum, %sum, %product : {t}, {t}, {t}\n"
        "  }\n"
    )


def _ports():
    memory = state.StateTensorSpec.create("memory", (1, 8, 32, 32))
    second = state.StateTensorSpec.create("second", (1, 8, 32, 32))
    shape = memory.storage_shape
    return {
        "order": (("memory", shape), ("value", shape), ("second", shape)),
        "targets": (("memory.next", shape), ("visible", shape), ("second.next", shape)),
        "states": {"memory": memory, "second": second},
        "state_results": {"memory.next": "memory", "second.next": "second"},
    }


def _procedure(family: str = "A14", name: str = "entry_0_ANE_region_0_0") -> str:
    return (
        f"  mpsx.ane @{name}(%arg0: {_MEMREF}) -> ({_MEMREF}) "
        f'attributes {{ane_family = "{family}", output_ordering = array<i64: 0>}} {{\n'
        f'    "mpsx.region_return"(%arg0) : ({_MEMREF}) -> ()\n'
        "  }\n"
    )


_ANE_ATTRIBUTES = 'mps.aneArch = "h13c", mps.aneRegionsSHA = "E0_F1", mps.useANELLIR = false'


def _placed(procedures: str, *, on_ane: bool = True, attributes: str = _ANE_ATTRIBUTES) -> str:
    placement = (
        "mps.disablePreAllocate, mps.fullyPlacedOnANE" if on_ane else "mps.disablePreAllocate"
    )
    return (
        "#map = affine_map<(d0) -> (d0)>\n"
        f"module attributes {{{attributes}}} {{\n"
        f"{procedures}"
        f"  func.func @entry_0(%arg0: {_TENSOR}) -> {_TENSOR} attributes {{{placement}}} {{\n"
        f"    return %arg0 : {_TENSOR}\n"
        "  }\n"
        "}\n"
    )


def test_procedure_updates_state_inside_the_region():
    lowered, inputs, outputs = state._inline_state_procedure(
        _raw_entry(), function="entry", family="A14", identity="KINO_STATE_entry", **_ports()
    )

    assert inputs == ("memory", "value", "second")
    assert outputs == ("visible",)
    region, outer = lowered.split("  func.func @entry", 1)
    for operation in (
        "variable_from_tensor",
        "read_variable",
        "strided_slice_update",
        "assign_variable",
    ):
        assert region.count(f'"mps.{operation}"') == 2
        assert f'"mps.{operation}"' not in outer
    assert '"mps.strided_slice_update"(%memory, %sum, ' in region
    assert '"mps.strided_slice_update"(%second, %product, ' in region
    assert '"mpsx.region_return"(%kst_output_0) :' in region
    assert "%kst_output_1" not in region
    assert 'ane_family = "A14"' in region
    assert 'sym_name = "entry_ANE_region_0_0"' in region
    assert "mps.stateInputIndices = array<i64: 0, 2>" in outer
    assert '%kst_call = "placement.region_call"' in outer
    assert "callee = @entry_ANE_region_0_0" in outer
    assert 'mps.regionSHA = "KINO_STATE_entry"' in outer
    assert "anec." not in lowered


def test_procedure_module_takes_attributes_and_family_from_the_placement(tmp_path):
    raw = tmp_path / "entry.raw.mlir"
    placed = tmp_path / "entry.placed.mlir"
    module = tmp_path / "entry.mlir"
    raw.write_text("module {\n" + _raw_entry() + "}\n" + _RESOURCES)
    placed.write_text(_placed(_procedure(family="A15")))

    ports = state._write_procedure_module(raw, placed, module, function="entry", **_ports())

    text = module.read_text()
    assert ports == (("memory", "value", "second"), ("visible",))
    assert text.startswith(
        'module attributes {mps.aneArch = "h13c", mps.aneRegionsSHA = "KINO_STATE_entry", '
    )
    assert 'mps.regionSHA = "KINO_STATE_entry"' in text
    assert "E0_F1" not in text
    assert 'ane_family = "A15"' in text
    assert text.count("0x0011223344556677") == 1
    assert text.endswith(_RESOURCES)


@pytest.mark.parametrize(
    ("procedures", "on_ane", "attributes", "reason"),
    [
        # macOS 26 lowers the placed region to anec instead.
        (
            f"  anec.A14 @entry_0_ane_region_0_0(%arg0: {_MEMREF}) -> ({_MEMREF}) {{\n  }}\n",
            True,
            _ANE_ATTRIBUTES,
            "0 mpsx.ane procedures",
        ),
        (
            _procedure() + _procedure(name="entry_0_ANE_region_0_1"),
            True,
            _ANE_ATTRIBUTES,
            "2 mpsx.ane procedures",
        ),
        (_procedure(), False, _ANE_ATTRIBUTES, "whole graph"),
        (_procedure(), True, "mps.useANELLIR = false", "ANE attributes"),
    ],
)
def test_a_placement_without_one_whole_graph_procedure_is_unsupported(
    tmp_path, procedures, on_ane, attributes, reason
):
    raw = tmp_path / "entry.raw.mlir"
    placed = tmp_path / "entry.placed.mlir"
    raw.write_text("module {\n" + _raw_entry() + "}\n")
    placed.write_text(_placed(procedures, on_ane=on_ane, attributes=attributes))

    with pytest.raises(state.PlacedFormUnsupported, match=reason):
        state._write_procedure_module(
            raw, placed, tmp_path / "entry.mlir", function="entry", **_ports()
        )


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
    product = root / "products" / "entry" / "entry_ANE_region_0_0.bc.mlir"
    product.parent.mkdir(parents=True)
    spec = state.StateTensorSpec.create("memory", (1, 8, 32, 32))
    entry = state._EntryContract(
        name="entry",
        function="entry",
        region="entry_ANE_region_0_0",
        order=(("memory", spec.storage_shape),),
        targets=(("visible", spec.storage_shape), ("memory.next", spec.storage_shape)),
        state_results=(("memory.next", "memory"),),
        ane_input_order=("memory",),
        ane_output_order=("visible",),
        dynamic=frozenset(),
        product="products/entry/entry_ANE_region_0_0.bc.mlir",
    )
    contract = state._contract_json(dtype=0, states=(spec,), entries=(entry,))
    (root / "contract.json").write_text(json.dumps(contract))

    assert not state.stateful_cache_ready(tmp_path)
    product.touch()
    assert not state.stateful_cache_ready(tmp_path)
    (product.parent / "compiler_options_entry_ANE_region_0_0.plist").touch()
    assert state.stateful_cache_ready(tmp_path)

    contract["format"] -= 1
    (root / "contract.json").write_text(json.dumps(contract))
    assert not state.stateful_cache_ready(tmp_path)
