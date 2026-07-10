"""Typed processor contracts and the first-party catalog.

The M3 stream-pipeline surface: frozen spec values (:mod:`specs`),
timestamped units and boundaries (:mod:`units`, :mod:`boundaries`),
capability declarations (:mod:`capabilities`), the lifecycle protocol
(:mod:`protocol`), typed failures (:mod:`errors`), and the lazy family
catalog (:mod:`catalog`).

Importing this package imports no processor family and no MLX model code.
"""

from .boundaries import Boundary, BoundaryKind
from .capabilities import Capability, CapabilitySpec, TemporalMode, preserve_stream
from .catalog import UnknownFamilyError, available_families, get_factory, register
from .errors import (
    MediaError,
    PipelineCancelled,
    PipelineError,
    PipelineRuntimeError,
    StageConfigError,
    StreamEdgeError,
    UnknownStageError,
    WeightsError,
)
from .protocol import PipelineContext, Processor, ProcessorFactory
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
    "Capability",
    "CapabilitySpec",
    "Cardinality",
    "ColorMatrix",
    "ColorPrimaries",
    "ColorRange",
    "DType",
    "Domain",
    "DurationPolicy",
    "FieldViolation",
    "FrameSpec",
    "FrameUnit",
    "Geometry",
    "Layout",
    "MediaError",
    "PipelineCancelled",
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
    "WeightsError",
    "available_families",
    "coherence_violations",
    "describe_spec",
    "frame_spec_for_matrix",
    "get_factory",
    "preserve_stream",
    "register",
]
