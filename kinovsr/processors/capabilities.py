"""Capabilities: the semantic roles a processor family can perform.

A :class:`CapabilitySpec` declares, as data plus one pure transform, how a
stage behaves on a chain edge: what stream contract it accepts, what it
produces from a given input spec, and its temporal shape (radius, causal
or centered or whole-sequence, stateful or not). The builder consumes
these declarations to validate a whole chain before any frame moves.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .boundaries import BoundaryKind
from .specs import StreamConstraint, StreamSpec


class Capability(Enum):
    PREPROCESS = "preprocess"
    DENOISE = "denoise"
    DEBLOCK = "deblock"
    DEBLUR = "deblur"
    RESTORE = "restore"
    UPSCALE = "upscale"
    INTERPOLATE = "interpolate"
    METRIC = "metric"


class TemporalMode(Enum):
    # Stateless per-frame function of the current unit only.
    PER_FRAME = "per_frame"
    # Uses past units only; emits for each unit immediately (zero
    # output delay).
    CAUSAL = "causal"
    # Uses past and FUTURE units within temporal_radius. The future
    # context is SELF-BUFFERED: the stage consumes units in arrival
    # order and pays its lookahead as output delay (emit t once t+radius
    # arrived; flush drains the tail). It demands nothing of the source,
    # so live edges are legal inputs - bsvd's bidirectional buffer and
    # deflicker's +/-K integration window are this shape. A
    # scheduler-prefetch variant (future units delivered at process time
    # in exchange for a source-lookahead requirement) is deliberately
    # NOT modeled until a real stage needs it.
    CENTERED = "centered"
    # Propagates over whole chunks/sequences (bidirectional recurrents);
    # the scheduler hands it framework-owned windows.
    SEQUENCE = "sequence"


def preserve_stream(spec: StreamSpec, config: object = None) -> StreamSpec:
    """The identity ``produces`` transform: spatial processors normally
    preserve both frame and timeline contracts."""
    return spec


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    capability: Capability
    # Profile names this family advertises for the capability, in the
    # shared vocabulary (planning 07): the existing product tokens.
    profiles: tuple[str, ...]
    accepts: StreamConstraint
    # Pure transform: the output StreamSpec this stage produces from a
    # given (already accepted) input spec and the stage's parsed config
    # (scale and target cadence are config decisions). Interpolation
    # rewrites cadence/cardinality here, explicitly; upscalers rewrite
    # geometry. Must not perform I/O.
    produces: Callable[[StreamSpec, object], StreamSpec] = preserve_stream
    # How many neighbor frames the unit consumes on each side (0 for
    # per-frame). For CAUSAL modes this is the look-back depth; for
    # CENTERED it is the future reach, which equals the stage's output
    # delay (self-buffered lookahead).
    temporal_radius: int = 0
    temporal_mode: TemporalMode = TemporalMode.PER_FRAME
    stateful: bool = False
    # METRIC taps observe units and publish results without rewriting the
    # stream contract; the builder enforces produces == identity for them.
    is_tap: bool = False
    # Boundary kinds this stage ADDS to the stream (cut detectors emit
    # HARD_CUT). Provision accumulates along the chain; input endpoints
    # always provide STREAM_START.
    emits_boundaries: tuple[BoundaryKind, ...] = ()
    # Boundary kinds this stage is INCORRECT without (not merely improved
    # by). The builder rejects a chain where no upstream provider exists.
    requires_boundaries: tuple[BoundaryKind, ...] = ()


__all__ = [
    "Capability",
    "CapabilitySpec",
    "TemporalMode",
    "preserve_stream",
]
