"""Pure-protobuf tests for the first-party ML Program builder."""
from __future__ import annotations

import struct

import mlx.core as mx
import pytest

from kinovsr.native.anemil import builder, schema

pytestmark = pytest.mark.unit


def _function(blob, scale: float):
    graph = builder.Graph(blob)
    graph.register_input("x", (1, 1, 2, 2))
    value = graph.fp16_const(
        "scale", mx.full((1, 1, 1, 1), scale, dtype=mx.float16))
    graph.binary("mul", "x", value, "scaled", name="out")
    return graph, [("x", (1, 1, 2, 2))], [], ["out"]


def test_blob_file_reuses_identical_payloads():
    blob = builder.BlobFile()
    raw = bytes(range(64)) * 128

    first = blob.add_fp16(raw)
    second = blob.add_fp16(raw)
    serialized = blob.serialize()

    assert second == first
    assert struct.unpack_from("<I", serialized)[0] == 1


def test_multifunction_program_describes_and_serializes_each_function():
    blob = builder.BlobFile()
    functions = [
        ("fill", *_function(blob, 0.5)),
        ("drain", *_function(blob, 2.0)),
    ]

    raw = builder.finish_functions(
        functions, "phase test", default_function="fill")
    model = schema.Model()
    model.ParseFromString(raw)

    assert model.description.defaultFunctionName == "fill"
    assert [item.name for item in model.description.functions] == [
        "fill", "drain"]
    assert set(model.mlProgram.functions) == {"fill", "drain"}
    for item in model.description.functions:
        assert [feature.name for feature in item.input] == ["x"]
        assert [feature.name for feature in item.output] == ["out"]


def test_multifunction_program_requires_one_shared_blob():
    first = _function(builder.BlobFile(), 1.0)
    second = _function(builder.BlobFile(), 2.0)
    with pytest.raises(ValueError, match="share one BlobFile"):
        builder.finish_functions([
            ("first", *first), ("second", *second)], "invalid")
