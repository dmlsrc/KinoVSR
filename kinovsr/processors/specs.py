"""Typed stream contracts: what flows along a chain edge.

Image-tensor properties (:class:`FrameSpec`) and video timing
(:class:`TimelineSpec`) are separate contracts; an edge carries both as a
:class:`StreamSpec`. The builder threads a spec from the input endpoint
through every stage to the output endpoint and rejects mismatches before
any frame is processed.

Everything here is a frozen value or a pure function over values (the
pure-core convention): constraint checking returns violation data instead
of raising, so the builder can compose precise field-level errors naming
both sides of an edge.

The enums are grounded in the foundation's real surfaces:

- MLX chain currency is an HWC RGB float array in nominal [0, 1]
  (:attr:`Layout.MLX_RGB_HWC` with float32/float16 dtype);
- native payloads are CVPixelBuffers in the three formats the readers,
  VideoToolbox sessions, and writers actually exchange (BGRA8, 64RGBAHalf,
  NV12 biplanar video-range);
- color identity mirrors ``kinovsr.media.color.resolve``: matrix, primaries,
  transfer, and range, with the bt601/bt709/bt2020 token families.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any

# ===========================================================================
# Frame identity
# ===========================================================================


class Layout(Enum):
    """Payload memory layout on a chain edge."""

    MLX_RGB_HWC = "mlx_rgb_hwc"    # mx.array, shape (H, W, 3)
    CV_BGRA = "cv_bgra"            # CVPixelBuffer 'BGRA', 8-bit
    CV_RGBA_HALF = "cv_rgba_half"  # CVPixelBuffer 64RGBAHalf
    CV_NV12 = "cv_nv12"            # CVPixelBuffer 420v biplanar video-range


class DType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    UINT8 = "uint8"


# The dtype each CV layout implies; MLX layouts choose per stream.
_LAYOUT_FIXED_DTYPE = {
    Layout.CV_BGRA: DType.UINT8,
    Layout.CV_RGBA_HALF: DType.FLOAT16,
    Layout.CV_NV12: DType.UINT8,
}


class ColorRange(Enum):
    VIDEO = "video"
    FULL = "full"


class ColorMatrix(Enum):
    BT601 = "bt601"
    BT709 = "bt709"
    BT2020 = "bt2020"


# ITU-R (Kr, Kb) luma coefficients per stream color matrix. The single
# source of truth for families that split RGB into luma/chroma from the
# input StreamSpec (stdf's Y-only deblock, the denoise luma/chroma blend).
_LUMA_COEFFICIENTS: dict[ColorMatrix, tuple[float, float]] = {
    ColorMatrix.BT601: (0.299, 0.114),
    ColorMatrix.BT709: (0.2126, 0.0722),
    ColorMatrix.BT2020: (0.2627, 0.0593),
}


def luma_coefficients(color_matrix: ColorMatrix) -> tuple[float, float]:
    """ITU-R (Kr, Kb) luma coefficients for a stream color matrix."""
    return _LUMA_COEFFICIENTS[color_matrix]


class ColorPrimaries(Enum):
    SMPTE_C = "smpte_c"    # NTSC 601 primaries
    BT709 = "bt709"
    BT2020 = "bt2020"


class TransferFunction(Enum):
    BT709 = "bt709"
    BT2020 = "bt2020"


class Domain(Enum):
    """Value-domain contract of the payload samples.

    ``UNIT`` is the decoded-RGB reality: nominally [0, 1] but legally
    overshooting (BT.709 decode of super-white/blacker-than-black).
    ``UNIT_SANITIZED`` is a declared clamp - what strict learned inputs
    want. ``CODED`` is codec-native data (YUV code values) where unit
    semantics do not apply.
    """

    CODED = "coded"
    UNIT = "unit"
    UNIT_SANITIZED = "unit_sanitized"


@dataclass(frozen=True, slots=True)
class Geometry:
    width: int
    height: int
    # Pixel aspect ratio; 1 for square pixels. Display aspect is
    # width * pixel_aspect : height.
    pixel_aspect: Fraction = Fraction(1)

    def scaled(self, factor: int) -> Geometry:
        return Geometry(self.width * factor, self.height * factor,
                        self.pixel_aspect)


@dataclass(frozen=True, slots=True)
class FrameSpec:
    layout: Layout
    dtype: DType
    color_range: ColorRange
    color_matrix: ColorMatrix
    color_primaries: ColorPrimaries
    transfer_function: TransferFunction
    domain: Domain
    geometry: Geometry


# ===========================================================================
# Timeline identity
# ===========================================================================


class VariableCadence(Enum):
    """Sentinel for VFR sources; a concrete rounding policy is an open
    planning question and arrives with real VFR footage."""

    VFR = "vfr"


class TimestampPolicy(Enum):
    SOURCE = "source"            # PTS carried from the source timeline
    REGENERATED = "regenerated"  # PTS rewritten on a constant grid


class Cardinality(Enum):
    """Unit-count relation of this stream to the input endpoint's frames."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"    # e.g. interpolation emitted extra units
    MANY_TO_ONE = "many_to_one"    # e.g. decimation


class DurationPolicy(Enum):
    PRESERVED = "preserved"    # clip duration equals the source window's
    REWRITTEN = "rewritten"    # a stage deliberately changed clip duration


@dataclass(frozen=True, slots=True)
class TimelineSpec:
    # Ticks per second of pts/duration integers on FrameUnit. The product
    # encodes at CMTime base 24000 so common rates land bit-exact.
    time_base: Fraction
    cadence: Fraction | VariableCadence            # frames per second
    timestamp_policy: TimestampPolicy = TimestampPolicy.SOURCE
    cardinality: Cardinality = Cardinality.ONE_TO_ONE
    duration_policy: DurationPolicy = DurationPolicy.PRESERVED


@dataclass(frozen=True, slots=True)
class StreamSpec:
    frame: FrameSpec
    timeline: TimelineSpec
    seekable: bool = False
    lookahead_available: bool = False


# ===========================================================================
# Constraint checking - pure, data-returning
# ===========================================================================


@dataclass(frozen=True, slots=True)
class FieldViolation:
    """One field-level mismatch on an edge: the field, what the consumer
    accepts, and what the producer supplies."""

    field: str
    accepted: str
    actual: str


@dataclass(frozen=True, slots=True)
class StreamConstraint:
    """What a capability accepts on its input edge.

    ``None`` means "anything" for that field. Geometry bounds are
    inclusive and apply to both axes. The check is a pure function
    returning violations as data; the builder turns them into a
    :class:`~kinovsr.processors.errors.StreamEdgeError` naming both sides.
    """

    layouts: tuple[Layout, ...] | None = None
    dtypes: tuple[DType, ...] | None = None
    color_ranges: tuple[ColorRange, ...] | None = None
    domains: tuple[Domain, ...] | None = None
    min_side: int | None = None
    max_side: int | None = None
    # Chroma-subsampled encode targets need even dimensions.
    require_even_dims: bool = False
    requires_lookahead: bool = False
    requires_seekable: bool = False
    cadences: tuple[type, ...] | None = None   # e.g. (Fraction,) = CFR only

    def violations(self, spec: StreamSpec) -> tuple[FieldViolation, ...]:
        found: list[FieldViolation] = []

        def check(name: str, allowed: tuple | None, actual: Any) -> None:
            if allowed is not None and actual not in allowed:
                found.append(FieldViolation(
                    field=name,
                    accepted=" | ".join(a.value for a in allowed),
                    actual=actual.value))

        check("frame.layout", self.layouts, spec.frame.layout)
        check("frame.dtype", self.dtypes, spec.frame.dtype)
        check("frame.color_range", self.color_ranges, spec.frame.color_range)
        check("frame.domain", self.domains, spec.frame.domain)

        geo = spec.frame.geometry
        small, large = sorted((geo.width, geo.height))
        if self.min_side is not None and small < self.min_side:
            found.append(FieldViolation(
                "frame.geometry", f"min side >= {self.min_side}",
                f"{geo.width}x{geo.height}"))
        if self.max_side is not None and large > self.max_side:
            found.append(FieldViolation(
                "frame.geometry", f"max side <= {self.max_side}",
                f"{geo.width}x{geo.height}"))
        if self.require_even_dims and (geo.width % 2 or geo.height % 2):
            found.append(FieldViolation(
                "frame.geometry", "even width and height",
                f"{geo.width}x{geo.height}"))

        if self.requires_lookahead and not spec.lookahead_available:
            found.append(FieldViolation(
                "lookahead_available", "true", "false"))
        if self.requires_seekable and not spec.seekable:
            found.append(FieldViolation("seekable", "true", "false"))

        if self.cadences is not None and not isinstance(
                spec.timeline.cadence, self.cadences):
            accepted = " | ".join(t.__name__ for t in self.cadences)
            found.append(FieldViolation(
                "timeline.cadence", accepted,
                type(spec.timeline.cadence).__name__))

        return tuple(found)


def coherence_violations(spec: StreamSpec) -> tuple[FieldViolation, ...]:
    """Internal consistency of a single spec (not an edge check): CV
    layouts carry their fixed dtype; CODED domain only rides CV layouts."""
    found: list[FieldViolation] = []
    fixed = _LAYOUT_FIXED_DTYPE.get(spec.frame.layout)
    if fixed is not None and spec.frame.dtype is not fixed:
        found.append(FieldViolation(
            "frame.dtype", f"{fixed.value} (implied by {spec.frame.layout.value})",
            spec.frame.dtype.value))
    if (spec.frame.domain is Domain.CODED
            and spec.frame.layout is Layout.MLX_RGB_HWC):
        found.append(FieldViolation(
            "frame.domain", "unit | unit_sanitized (MLX RGB payloads)",
            spec.frame.domain.value))
    return tuple(found)


# ===========================================================================
# Foundation bridge - pure translation from the probe/color surfaces
# ===========================================================================

_MATRIX_TOKENS = {
    "601": ColorMatrix.BT601, "bt601": ColorMatrix.BT601,
    "709": ColorMatrix.BT709, "bt709": ColorMatrix.BT709,
    "2020": ColorMatrix.BT2020, "bt2020": ColorMatrix.BT2020,
}

_MATRIX_FAMILY = {
    ColorMatrix.BT601: (ColorPrimaries.SMPTE_C, TransferFunction.BT709),
    ColorMatrix.BT709: (ColorPrimaries.BT709, TransferFunction.BT709),
    ColorMatrix.BT2020: (ColorPrimaries.BT2020, TransferFunction.BT2020),
}


def frame_spec_for_matrix(
    matrix_token: str,
    *,
    full_range: bool,
    geometry: Geometry,
    layout: Layout = Layout.MLX_RGB_HWC,
    dtype: DType = DType.FLOAT32,
    domain: Domain = Domain.UNIT,
) -> FrameSpec:
    """Build a FrameSpec from the foundation's color-token vocabulary.

    ``matrix_token`` accepts the resolve()-style tokens (``bt709``/``709``,
    ...); primaries and transfer follow the matrix family, matching how
    ``kinovsr.media.color`` pairs them.
    """
    matrix = _MATRIX_TOKENS.get(str(matrix_token).lower())
    if matrix is None:
        raise ValueError(f"unknown color matrix token {matrix_token!r}")
    primaries, transfer = _MATRIX_FAMILY[matrix]
    return FrameSpec(
        layout=layout,
        dtype=_LAYOUT_FIXED_DTYPE.get(layout, dtype),
        color_range=ColorRange.FULL if full_range else ColorRange.VIDEO,
        color_matrix=matrix,
        color_primaries=primaries,
        transfer_function=transfer,
        domain=domain,
        geometry=geometry,
    )


def describe_spec(spec: StreamSpec) -> str:
    """One-line human rendering used by errors and diagnostics."""
    f, t = spec.frame, spec.timeline
    cadence = (t.cadence.value if isinstance(t.cadence, VariableCadence)
               else f"{float(t.cadence):g}fps")
    return (f"{f.layout.value}/{f.dtype.value} "
            f"{f.geometry.width}x{f.geometry.height} "
            f"{f.color_matrix.value}/{f.color_range.value} "
            f"{f.domain.value} @ {cadence}")


__all__ = [
    "Cardinality",
    "ColorMatrix",
    "ColorPrimaries",
    "ColorRange",
    "DType",
    "Domain",
    "DurationPolicy",
    "FieldViolation",
    "FrameSpec",
    "Geometry",
    "Layout",
    "StreamConstraint",
    "StreamSpec",
    "TimelineSpec",
    "TimestampPolicy",
    "TransferFunction",
    "VariableCadence",
    "coherence_violations",
    "describe_spec",
    "frame_spec_for_matrix",
    "luma_coefficients",
]
