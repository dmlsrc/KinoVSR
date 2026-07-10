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
    # Uses past units only (streaming recurrence); no lookahead needed.
    CAUSAL = "causal"
    # Uses past and FUTURE units within temporal_radius; requires a source
    # with lookahead (a file, a buffered stream), never a live edge.
    CENTERED = "centered"
    # Propagates over whole chunks/sequences (bidirectional recurrents);
    # the scheduler hands it framework-owned windows.
    SEQUENCE = "sequence"


def preserve_stream(spec: StreamSpec) -> StreamSpec:
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
    # given (already accepted) input spec. Interpolation rewrites
    # cadence/cardinality here, explicitly; upscalers rewrite geometry.
    produces: Callable[[StreamSpec], StreamSpec] = preserve_stream
    # How many neighbor frames the unit consumes on each side (0 for
    # per-frame). For CAUSAL modes this is the look-back depth; for
    # CENTERED it is the symmetric radius that lookahead must cover.
    temporal_radius: int = 0
    temporal_mode: TemporalMode = TemporalMode.PER_FRAME
    stateful: bool = False
    # METRIC taps observe units and publish results without rewriting the
    # stream contract; the builder enforces produces == identity for them.
    is_tap: bool = False


__all__ = [
    "Capability",
    "CapabilitySpec",
    "TemporalMode",
    "preserve_stream",
]
