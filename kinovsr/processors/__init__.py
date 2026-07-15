"""Typed processor contracts and the first-party catalog.

The M3 stream-pipeline surface: frozen spec values (:mod:`specs`),
timestamped units and boundaries (:mod:`units`, :mod:`boundaries`),
capability declarations (:mod:`capabilities`), the lifecycle protocol
(:mod:`protocol`), typed failures (:mod:`errors`), and the lazy family
catalog (:mod:`catalog`).

Importing this package imports no processor family and no MLX model code.
"""

from .boundaries import Boundary, BoundaryKind
from .capabilities import (
    Capability,
    CapabilitySpec,
    CompanionSpec,
    TemporalMode,
    preserve_stream,
)
from .catalog import (
    CatalogEntry,
    CliOptionContribution,
    UnknownFamilyError,
    available_families,
    catalog_entries,
    get_factory,
    register,
)
from .errors import (
    MediaError,
    PipelineError,
    PipelineRuntimeError,
    StageConfigError,
    StreamEdgeError,
    UnknownStageError,
)
from .protocol import (
    BracketFactory,
    PipelineContext,
    Processor,
    ProcessorFactory,
)
from .specs import (
    Cardinality,
    ColorMatrix,
    ColorPrimaries,
    ColorRange,
    Domain,
    DType,
    DurationPolicy,
    FieldViolation,
    FrameSpec,
    Geometry,
    Layout,
    StreamConstraint,
    StreamSpec,
    TimelineSpec,
    TimestampPolicy,
    TransferFunction,
    VariableCadence,
    coherence_violations,
    describe_spec,
    frame_spec_for_matrix,
)
from .units import FrameUnit

__all__ = [
    "Boundary",
    "BoundaryKind",
    "BracketFactory",
    "Capability",
    "CapabilitySpec",
    "CatalogEntry",
    "Cardinality",
    "CompanionSpec",
    "ColorMatrix",
    "ColorPrimaries",
    "ColorRange",
    "CliOptionContribution",
    "DType",
    "Domain",
    "DurationPolicy",
    "FieldViolation",
    "FrameSpec",
    "FrameUnit",
    "Geometry",
    "Layout",
    "MediaError",
    "PipelineContext",
    "PipelineError",
    "PipelineRuntimeError",
    "Processor",
    "ProcessorFactory",
    "StageConfigError",
    "StreamConstraint",
    "StreamEdgeError",
    "StreamSpec",
    "TemporalMode",
    "TimelineSpec",
    "TimestampPolicy",
    "TransferFunction",
    "UnknownFamilyError",
    "UnknownStageError",
    "VariableCadence",
    "available_families",
    "catalog_entries",
    "coherence_violations",
    "describe_spec",
    "frame_spec_for_matrix",
    "get_factory",
    "preserve_stream",
    "register",
]
