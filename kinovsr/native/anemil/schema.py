"""Core ML protobuf message classes, built dynamically from vendored schemas.

Loads `coreml_spec.fds` (a FileDescriptorSet captured once from Apple's
public Core ML .proto files; capture provenance and hashes are recorded in
`coreml_spec.provenance.json`, and `scripts/dev/make_coreml_fds.py`
regenerates both) into a DescriptorPool and materializes message classes
with message_factory. This deliberately avoids
protoc-generated _pb2 modules: generated code is version-coupled to the
protobuf runtime (the 4->5->6 major transitions repeatedly broke pinned
gencode), while a descriptor set is plain data and the dynamic API is the
stable path.

Requires the `protobuf` package (>= 4.25). Its macOS wheels are abi3, so
one wheel covers every CPython >= 3.10 - no per-Python-version lag, which
is the property coremltools lacks and the reason anemil exists. The
message classes are built on first use.
"""

from collections.abc import Callable
from pathlib import Path

import google.protobuf
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

_FDS_PATH = Path(__file__).resolve().parent / "coreml_spec.fds"

# Message full names anemil needs. Everything else in the pool is the
# transitive closure Model.proto drags in (legacy model types); harmless.
_NAMES = {
    "Model": "CoreML.Specification.Model",
    "Program": "CoreML.Specification.MILSpec.Program",
    "Function": "CoreML.Specification.MILSpec.Function",
    "Block": "CoreML.Specification.MILSpec.Block",
    "Operation": "CoreML.Specification.MILSpec.Operation",
    "NamedValueType": "CoreML.Specification.MILSpec.NamedValueType",
    "ValueType": "CoreML.Specification.MILSpec.ValueType",
    "TensorType": "CoreML.Specification.MILSpec.TensorType",
    "StateType": "CoreML.Specification.MILSpec.StateType",
    "Dimension": "CoreML.Specification.MILSpec.Dimension",
    "Argument": "CoreML.Specification.MILSpec.Argument",
    "Value": "CoreML.Specification.MILSpec.Value",
    "TensorValue": "CoreML.Specification.MILSpec.TensorValue",
}

# MILSpec.DataType enum values (from the vendored schema; assert at load).
FLOAT16 = 10
FLOAT32 = 11
INT32 = 23
BOOL = 1
STRING = 2
# CoreML.Specification.ArrayFeatureType.ArrayDataType.FLOAT16
FEATURE_FLOAT16 = 65552

# Message class by short name. The classes are dynamic (no generated code),
# so a caller binds each new message to a Protocol of the fields it touches.
_classes: dict[str, Callable[[], object]] | None = None


def _load() -> dict[str, Callable[[], object]]:
    global _classes
    if _classes is not None:
        return _classes

    version = tuple(int(p) for p in google.protobuf.__version__.split(".")[:2])
    if version < (4, 25):
        raise RuntimeError(
            f"protobuf {google.protobuf.__version__} is too old for the "
            f"dynamic message API anemil uses; need >= 4.25"
        )

    fds = descriptor_pb2.FileDescriptorSet()
    fds.ParseFromString(_FDS_PATH.read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for file_proto in fds.file:  # dependency order, by construction
        pool.Add(file_proto)

    classes: dict[str, Callable[[], object]] = {}
    for short, full in _NAMES.items():
        classes[short] = message_factory.GetMessageClass(pool.FindMessageTypeByName(full))

    data_type = pool.FindEnumTypeByName("CoreML.Specification.MILSpec.DataType")
    for name, expected in (
        ("FLOAT16", FLOAT16),
        ("FLOAT32", FLOAT32),
        ("INT32", INT32),
        ("BOOL", BOOL),
        ("STRING", STRING),
    ):
        actual = data_type.values_by_name[name].number
        if actual != expected:
            raise RuntimeError(
                f"vendored schema disagrees with baked enum: "
                f"DataType.{name} is {actual}, expected {expected}"
            )
    feature = pool.FindEnumTypeByName("CoreML.Specification.ArrayFeatureType.ArrayDataType")
    if feature.values_by_name["FLOAT16"].number != FEATURE_FLOAT16:
        raise RuntimeError("ArrayDataType.FLOAT16 mismatch in vendored schema")

    _classes = classes
    return classes


def __getattr__(name: str) -> Callable[[], object]:
    classes = _load()
    if name in classes:
        return classes[name]
    raise AttributeError(name)
