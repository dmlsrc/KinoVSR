"""An edge-band replicate-fill preprocess family, with an optional restore.

``fill = "extend"``: the declared edge bands are overwritten with the
adjacent interior before any learned stage sees the frame, so nets trained
on photographic content never see synthetic border structure (junk capture
rows, stabilization edges). The replicated content reaches the output;
geometry is untouched.

``fill = "restore"``: the nets still see the extended frame, but the
ORIGINAL border is composited back over the processed output at output
geometry, feathered into the content - the border stays exactly as
quiet/static/dark as the source. This BRACKETS the chain: the extend
pre-pass captures each original frame, and a companion post-pass that the
builder appends at the chain end restores it (planning 05's companion
mechanism). The two halves share a PTS-keyed buffer, so a windowed stage's
delay does not desynchronize them; ``restore_borders`` nearest-upscales the
captured band to whatever geometry the chain produced.

The bands are declared, not detected - junk DETECTION is probe-time, like
crop's bar detection. ``trim`` (crop the junk off) is the crop family's
job, not a fill mode here.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.edge_sanitize import (
    parse_edges_spec,
    restore_borders,
    sanitize_rgb,
)
from kinovsr.processors.capabilities import (
    Capability,
    CapabilitySpec,
    CompanionSpec,
    preserve_stream,
)
from kinovsr.processors.feed_driver import FeedFlushProcessor
from kinovsr.processors.protocol import PipelineContext, Processor
from kinovsr.processors.specs import (
    Domain,
    DType,
    Layout,
    StreamConstraint,
    StreamSpec,
)
from kinovsr.settings import Settings

_FILLS = ("extend", "restore")

# The RGB contract both halves work in: MLX HWC, unit-domain, either dtype.
_RGB = StreamConstraint(
    layouts=(Layout.MLX_RGB_HWC,),
    dtypes=(DType.FLOAT32, DType.FLOAT16),
    domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
)


@dataclasses.dataclass(frozen=True, slots=True)
class SanitizeEdgesStageConfig:
    edges: tuple[int, int, int, int]    # (top, bottom, left, right)
    fill: str                            # "extend" | "restore"
    feather: int                         # restore seam crossfade (source px)


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    # Geometry is known at resolve time: bands that consume an entire
    # axis fail preflight instead of after processing begins.
    assert isinstance(config, SanitizeEdgesStageConfig)
    top, bottom, left, right = config.edges
    geometry = spec.frame.geometry
    if top + bottom >= geometry.height or left + right >= geometry.width:
        raise ValueError(
            f"edge bands {config.edges} leave no interior in a "
            f"{geometry.width}x{geometry.height} stream")
    return spec


def _companion(config: object) -> CompanionSpec | None:
    assert isinstance(config, SanitizeEdgesStageConfig)
    if config.fill != "restore":
        return None
    # The post-pass composites the original border at output geometry; it
    # rewrites pixels, not geometry, so it preserves the stream contract.
    return CompanionSpec(accepts=_RGB, produces=preserve_stream)


class _RestoreBuffer:
    """Original input frames held by PTS until the post-pass composites them.

    Bounded by the chain's output delay on a 1:1 timeline: preflight rejects
    cardinality changes, and spatial processing preserves PTS, so every
    captured frame is taken exactly once.
    """

    def __init__(self) -> None:
        self._frames: dict[int, Any] = {}

    def capture(self, pts: int, frame: Any) -> None:
        self._frames[pts] = frame

    def take(self, pts: int) -> Any | None:
        return self._frames.pop(pts, None)


class _ExtendDriver:
    """Replicate-fill the bands; in restore mode also capture the original
    frame by PTS so the post-pass can composite it back."""

    def __init__(self, edges: tuple[int, int, int, int],
                 buffer: _RestoreBuffer | None = None) -> None:
        self._edges = edges
        self._buffer = buffer

    def feed(self, rgb: Any, token: Any = None) -> list:
        if self._buffer is not None and token is not None:
            self._buffer.capture(token.pts, rgb)
        return [(sanitize_rgb(rgb, self._edges), token)]

    def flush(self) -> list:
        return []

    def reset(self) -> None:
        pass


class _RestoreDriver:
    """Composite each captured original border over the processed frame."""

    def __init__(self, edges: tuple[int, int, int, int], feather: int,
                 buffer: _RestoreBuffer) -> None:
        self._edges = edges
        self._feather = feather
        self._buffer = buffer

    def feed(self, rgb: Any, token: Any = None) -> list:
        original = self._buffer.take(token.pts) if token is not None else None
        if original is not None:
            rgb = restore_borders(rgb, original, self._edges,
                                  feather=self._feather)
        return [(rgb, token)]

    def flush(self) -> list:
        return []

    def reset(self) -> None:
        pass


class SanitizeEdgesFactory:
    name = "sanitize_edges"

    capabilities = {
        Capability.PREPROCESS: CapabilitySpec(
            capability=Capability.PREPROCESS,
            profiles=(),
            accepts=_RGB,
            produces=_produces,
            companion=_companion,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> SanitizeEdgesStageConfig:
        reject_unknown_keys(raw, ("edges", "fill", "feather"))
        edges_spec = typed_value(raw, "edges", str)
        if not edges_spec:
            raise ValueError(
                "state edges as T,B,L,R (junk detection is probe-time; "
                "a sanitize stage with no bands is config noise)")
        edges = parse_edges_spec(edges_spec)
        if edges == (0, 0, 0, 0):
            raise ValueError("edges declare no bands")
        fill = typed_value(raw, "fill", str, "extend")
        if fill not in _FILLS:
            raise ValueError(
                "fill must be 'extend' or 'restore' ('trim' is the crop "
                "family's job)")
        feather = typed_value(raw, "feather", int, 2)
        if feather < 0:
            raise ValueError("feather must be >= 0")
        return SanitizeEdgesStageConfig(edges=edges, fill=fill, feather=feather)

    def build(self, config: SanitizeEdgesStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        # Reached for extend only; restore declares a companion and routes
        # through build_bracket.
        return FeedFlushProcessor(lambda: _ExtendDriver(config.edges))

    def build_bracket(
        self, config: SanitizeEdgesStageConfig, *,
        context: PipelineContext,
    ) -> tuple[Processor, Processor]:
        buffer = _RestoreBuffer()
        pre = FeedFlushProcessor(lambda: _ExtendDriver(config.edges, buffer))
        post = FeedFlushProcessor(
            lambda: _RestoreDriver(config.edges, config.feather, buffer))
        return pre, post


FACTORY = SanitizeEdgesFactory()

__all__ = ["FACTORY", "SanitizeEdgesFactory", "SanitizeEdgesStageConfig"]
