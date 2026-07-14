"""VideoToolbox as a processor family: the native-session prover.

Two capabilities, both wrapping native VTFrameProcessor sessions:

- INTERPOLATE (VTFrameRateConversion): a native session with real
  lifecycle, one-to-many unit emission, a rewritten cadence with
  regenerated timestamps, and preserved clip duration - which is what
  keeps copied audio synchronized.
- UPSCALE (VTSuperResolutionScaler / VTLowLatencySuperResolutionScaler):
  the `--upscale fast|balanced|image` spatial modes. This is
  also the pipeline's MLX->CV bridge: it accepts an MLX RGB frame (the
  output of any MLX preprocessing chain) and produces a native CV buffer,
  while a head-stage upscale decodes directly into the session's native
  source format for a zero-copy CV->CV path.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any

from kinovsr.config.helpers import reject_unknown_keys
from kinovsr.media.timing import grid_ticks
from kinovsr.processors.boundaries import Boundary
from kinovsr.processors.capabilities import (
    Capability,
    CapabilitySpec,
    TemporalMode,
)
from kinovsr.processors.errors import MediaError
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Cardinality,
    Domain,
    DType,
    Layout,
    StreamConstraint,
    StreamSpec,
    TimestampPolicy,
)
from kinovsr.processors.units import FrameUnit
from kinovsr.settings import Settings

_PROFILES = ("normal", "high")

# Spatial modes = profiles. Each couples scale, output CV layout, dtype,
# domain, and the input size cap VideoToolbox enforces (native/vsr.py:
# HQ 1920x1080, LowLatency 960x960). balanced is temporal (prev-frame
# chain); fast/image are per-frame.
_UPSCALE_PROFILES = ("fast", "balanced", "image")
_UPSCALE_MODE = {
    #        scale, dst layout,          dtype,          domain,        max_w, max_h
    "fast":     (2, Layout.CV_NV12,      DType.UINT8,    Domain.CODED,   960,  960),
    "balanced": (4, Layout.CV_RGBA_HALF, DType.FLOAT16,  Domain.UNIT,   1920, 1080),
    "image":    (4, Layout.CV_RGBA_HALF, DType.FLOAT16,  Domain.UNIT,   1920, 1080),
}

# The CVPixelBuffer layout each mode's VSR session consumes as its source.
# A source already decoded in this layout feeds upscale_buffer_to_buffer
# zero-copy.
_UPSCALE_NATIVE_SRC = {
    "fast": Layout.CV_NV12,
    "balanced": Layout.CV_RGBA_HALF,
    "image": Layout.CV_RGBA_HALF,
}


@dataclasses.dataclass(frozen=True, slots=True)
class VtInterpolateConfig:
    target_fps: Fraction
    mode: str


def _parse_fps(value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError("target_fps must be a number")
    fps = Fraction(str(value))
    if fps <= 0:
        raise ValueError("target_fps must be positive")
    return fps


def _produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, VtInterpolateConfig)
    timeline = dataclasses.replace(
        spec.timeline,
        cadence=config.target_fps,
        timestamp_policy=TimestampPolicy.REGENERATED,
        cardinality=(Cardinality.ONE_TO_MANY
                     if config.target_fps > spec.timeline.cadence
                     else Cardinality.MANY_TO_ONE),
    )
    frame = spec.frame
    if frame.layout is Layout.MLX_RGB_HWC:
        # MLX frames upload into RGBAHalf buffers for the FRC session (a
        # learned upscale followed by --target-fps, the harness shape
        # where FRC ran on the writer-bound buffers); the output is the
        # session's native CV currency.
        frame = dataclasses.replace(
            frame, layout=Layout.CV_RGBA_HALF, dtype=DType.FLOAT16)
    return dataclasses.replace(spec, timeline=timeline, frame=frame)


class VtInterpolateProcessor:
    """Wrap VtfrcSession with grid-exact regenerated timestamps.

    Output unit ``m`` sits at ``m / target_fps``; PTS/duration are that
    instant expressed in the stream's integer time base, with durations
    computed as successive-grid differences so no drift accumulates.
    The session's target-index grid is monotonic across mid-stream
    drains, so a hard cut (scheduler: flush, then reset) keeps timestamps
    strictly increasing while interpolation never crosses the cut.

    Payload contract: source CVPixelBuffers must be IOSurface-backed
    (every real reader, decoder, and pool in this package produces those;
    a bare ``CVPixelBufferCreate`` buffer crashes inside VTFrameProcessor
    natively, not catchably).
    """

    def __init__(self, config: VtInterpolateConfig) -> None:
        self._config = config
        self._session: Any = None
        self._time_base: Fraction | None = None
        self._source_cadence: Fraction | None = None
        self._publication_origin_pts: int | None = None
        self._target_index: int | None = None
        self._input_end_pts: int | None = None
        self._pending: FrameUnit | None = None
        # A normal host stream keeps its first input PTS as the output origin.
        # Negative input PTS are the file endpoint's processing-only GOP
        # context; those indices stay negative on the target grid so frame 0
        # is generated at the requested public in-point, not relabeled later.
        self._origin: int | None = None

    def _grid_ticks(self, target_index: int) -> int:
        return grid_ticks(
            target_index, self._config.target_fps, self._time_base)

    def _emit(self, payload: Any) -> FrameUnit:
        assert self._target_index is not None
        m = self._target_index
        self._target_index += 1
        pts = self._grid_ticks(m)
        origin = self._origin or 0
        return FrameUnit(payload=payload, pts=origin + pts,
                         duration=self._grid_ticks(m + 1) - pts)

    def prepare(self, input_spec: StreamSpec,
                context: PipelineContext) -> None:
        from kinovsr.native.temporal import VtfrcSession

        cadence = input_spec.timeline.cadence
        assert isinstance(cadence, Fraction)
        geometry = input_spec.frame.geometry
        self._time_base = input_spec.timeline.time_base
        self._source_cadence = cadence
        self._publication_origin_pts = context.publication_origin_pts
        # An MLX input uploads each frame into an IOSurface-backed
        # RGBAHalf buffer for the session (the learned-upscale ->
        # --target-fps shape); CV inputs feed straight through.
        self._upload_pool = None
        if input_spec.frame.layout is Layout.MLX_RGB_HWC:
            from kinovsr.media import pixel_buffers as _pb

            self._pb = _pb
            self._upload_pool = _pb.make_pool_from_attrs({
                "PixelFormatType": _pb.PIX_RGBAHALF,
                "Width": geometry.width, "Height": geometry.height,
                "IOSurfaceProperties": {},
                "MetalCompatibility": True,
            })
        try:
            self._session = VtfrcSession(
                geometry.width, geometry.height,
                source_fps=cadence,
                target_fps=self._config.target_fps,
                mode=self._config.mode)
        except (RuntimeError, SystemExit) as exc:
            raise MediaError(
                f"VTFrameRateConversion session unavailable: {exc}") from exc

    def process(self, unit: FrameUnit,
                context: PipelineContext) -> Iterable[FrameUnit]:
        if self._origin is None:
            self._origin = (unit.pts if self._publication_origin_pts is None
                            else self._publication_origin_pts)
        assert self._time_base is not None
        assert self._source_cadence is not None
        index = round(
            Fraction(unit.pts - self._origin)
            * self._time_base * self._source_cadence)
        source_step = (
            grid_ticks(index + 1, self._source_cadence, self._time_base)
            - grid_ticks(index, self._source_cadence, self._time_base))
        self._input_end_pts = unit.pts + (
            unit.duration if unit.duration > 0 else source_step)
        if self._target_index is None:
            self._target_index = math.ceil(
                Fraction(index) * self._config.target_fps
                / self._source_cadence)
        payload = unit.payload
        if self._upload_pool is not None:
            import mlx.core as mx

            rgb = payload
            rgba = mx.concatenate(
                [rgb[..., :3].astype(mx.float16),
                 mx.ones((*rgb.shape[:2], 1), mx.float16)], axis=-1)
            buffer = self._pb.pool_create_buffer(self._upload_pool)
            self._pb.upload_frame_to_buffer(rgba, buffer)
            payload = buffer
        for produced in self._session.feed(payload, index):
            unit = self._emit(produced)
            if self._pending is not None:
                yield self._pending
            self._pending = unit

    def reset(self, boundary: Boundary,
              context: PipelineContext) -> None:
        # The scheduler drained the pre-boundary tail via flush(), which
        # clears the buffered pair inside the session; the target grid
        # deliberately keeps counting so PTS stays monotonic.
        pass

    def flush(self, context: PipelineContext) -> Iterable[FrameUnit]:
        if self._session is None:
            return
        for payload in self._session.drain():
            unit = self._emit(payload)
            if (self._input_end_pts is not None
                    and unit.pts >= self._input_end_pts):
                continue
            if self._pending is not None:
                yield self._pending
            self._pending = unit
        if self._pending is not None:
            pending, self._pending = self._pending, None
            if self._input_end_pts is not None:
                duration = min(
                    pending.duration, self._input_end_pts - pending.pts)
                if duration <= 0:
                    return
                pending = pending.retimed(pending.pts, duration)
            yield pending

    def close(self, context: PipelineContext) -> None:
        session, self._session = self._session, None
        self._pending = None
        self._input_end_pts = None
        if session is not None:
            session.close()


@dataclasses.dataclass(frozen=True, slots=True)
class VtUpscaleConfig:
    mode: str    # "fast" | "balanced" | "image"


def _upscale_produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, VtUpscaleConfig)
    scale, dst_layout, dst_dtype, dst_domain, max_w, max_h = (
        _UPSCALE_MODE[config.mode])
    g = spec.frame.geometry
    # The size cap is a VideoToolbox hard limit and mode-dependent, so it is
    # checked here (open time) rather than as a StreamConstraint bound.
    if g.width > max_w or g.height > max_h:
        raise ValueError(
            f"videotoolbox {config.mode} upscale accepts input up to "
            f"{max_w}x{max_h}; got {g.width}x{g.height}")
    # A CV source must be the mode's own session format (zero-copy path);
    # the layout<->mode pairing is mode-dependent, so it is checked here
    # like the size cap rather than as a StreamConstraint bound.
    native = _UPSCALE_NATIVE_SRC[config.mode]
    if spec.frame.layout is not Layout.MLX_RGB_HWC and (
            spec.frame.layout is not native):
        raise ValueError(
            f"videotoolbox {config.mode} upscale takes MLX frames or its "
            f"native {native.value} source; got {spec.frame.layout.value}")
    frame = dataclasses.replace(
        spec.frame, layout=dst_layout, dtype=dst_dtype, domain=dst_domain,
        geometry=g.scaled(scale))
    return dataclasses.replace(spec, frame=frame)


class VtUpscaleProcessor:
    """Wrap VsrSession as an MLX->CV spatial upscaler.

    Accepts the typed MLX RGB frame (float32/float16, HWC) and uploads it
    to the scaler as fp16 RGBA, deferring the 8/10-bit quantization into the
    scaler's output the same way the harness's ``den_rgba`` path does. The
    output is a native CV buffer at ``scale x`` (RGBAHalf for the HQ modes,
    NV12 for fast). balanced threads a prev-frame chain inside the session;
    a hard cut resets it (``reset_temporal_context``). Spatial: timeline and
    PTS are preserved, so audio and any restore companion stay in sync.
    """

    def __init__(self, config: VtUpscaleConfig) -> None:
        self._config = config
        self._session: Any = None
        self._mx: Any = None
        self._index = 0
        self._cv_input = False

    def prepare(self, input_spec: StreamSpec,
                context: PipelineContext) -> None:
        import mlx.core as mx

        from kinovsr.native.vsr import VsrSession

        self._mx = mx
        # A CV source is already in the session's source format (validated
        # at resolve): feed it zero-copy via upscale_buffer_to_buffer, the
        # harness's mainstream decode->VSR path. MLX frames upload.
        self._cv_input = input_spec.frame.layout is not Layout.MLX_RGB_HWC
        g = input_spec.frame.geometry
        try:
            self._session = VsrSession(
                g.width, g.height, mode=self._config.mode,
                fps=float(input_spec.timeline.cadence))
        except (RuntimeError, SystemExit, ValueError) as exc:
            raise MediaError(
                f"VideoToolbox VSR session unavailable: {exc}") from exc

    def process(self, unit: FrameUnit,
                context: PipelineContext) -> Iterable[FrameUnit]:
        if self._cv_input:
            out = self._session.upscale_buffer_to_buffer(
                unit.payload, self._index)
            self._index += 1
            yield unit.with_payload(out)
            return
        mx = self._mx
        rgb = unit.payload
        rgba = mx.concatenate(
            [rgb.astype(mx.float16),
             mx.ones((*rgb.shape[:2], 1), mx.float16)], axis=-1)
        out = self._session.upscale_to_buffer(rgba, self._index)
        self._index += 1
        yield unit.with_payload(out)

    def reset(self, boundary: Boundary,
              context: PipelineContext) -> None:
        if self._session is not None:
            self._session.reset_temporal_context()

    def flush(self, context: PipelineContext) -> Iterable[FrameUnit]:
        return ()

    def close(self, context: PipelineContext) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.close()


class VideoToolboxFactory:
    name = "videotoolbox"

    capabilities = {
        Capability.INTERPOLATE: CapabilitySpec(
            capability=Capability.INTERPOLATE,
            profiles=_PROFILES,
            accepts=StreamConstraint(
                layouts=(Layout.CV_BGRA, Layout.CV_RGBA_HALF,
                         Layout.CV_NV12, Layout.MLX_RGB_HWC),
                cadences=(Fraction,),      # CFR sources only
            ),
            produces=_produces,
            stateful=True,
        ),
        Capability.UPSCALE: CapabilitySpec(
            capability=Capability.UPSCALE,
            profiles=_UPSCALE_PROFILES,
            # MLX in (the bridge) or the mode's own CV source format
            # (zero-copy decode->VSR, the harness's mainstream path). The
            # mode<->CV-layout pairing is enforced in _upscale_produces at
            # resolve time, like the size cap.
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC, Layout.CV_RGBA_HALF,
                         Layout.CV_NV12),
                dtypes=(DType.FLOAT32, DType.FLOAT16, DType.UINT8),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED, Domain.CODED),
                cadences=(Fraction,),      # CFR sources only
            ),
            produces=_upscale_produces,
            # balanced threads one prev frame; declared for all three modes
            # so the scheduler resets the temporal chain on a hard cut.
            temporal_mode=TemporalMode.CAUSAL,
            temporal_radius=1,
            stateful=True,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, Any],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> VtInterpolateConfig | VtUpscaleConfig:
        if capability is Capability.UPSCALE:
            reject_unknown_keys(raw, ())
            mode = profile or "balanced"
            if mode not in _UPSCALE_PROFILES:
                raise ValueError(
                    f"videotoolbox upscale profile must be one of "
                    f"{list(_UPSCALE_PROFILES)}")
            return VtUpscaleConfig(mode=mode)
        reject_unknown_keys(raw, ("target_fps",))
        if "target_fps" not in raw:
            raise ValueError("target_fps is required for interpolation")
        return VtInterpolateConfig(
            target_fps=_parse_fps(raw["target_fps"]),
            mode=profile or "normal")

    def build(self, config: VtInterpolateConfig | VtUpscaleConfig, *,
              context: PipelineContext) -> VtInterpolateProcessor | VtUpscaleProcessor:
        if isinstance(config, VtUpscaleConfig):
            return VtUpscaleProcessor(config)
        return VtInterpolateProcessor(config)

    def preferred_source_layout(self, *, capability: Capability,
                                profile: str | None) -> Layout | None:
        """The source layout a HEAD stage of this capability decodes best
        from (the optional-hook shape of ``profile_defaults``). An upscale
        head wants its mode's own session format so the decode feeds the
        scaler zero-copy; every other case defers to the caller's default
        preference."""
        if capability is Capability.UPSCALE:
            return _UPSCALE_NATIVE_SRC[profile or "balanced"]
        if capability is Capability.INTERPOLATE:
            # The FRC session's currency; MLX is accepted mid-chain (a
            # learned upscale feeding --target-fps) but a HEAD interpolate
            # should decode straight into buffers, not upload per frame.
            return Layout.CV_RGBA_HALF
        return None


FACTORY = VideoToolboxFactory()

__all__ = [
    "FACTORY",
    "VideoToolboxFactory",
    "VtInterpolateConfig",
    "VtInterpolateProcessor",
    "VtUpscaleConfig",
    "VtUpscaleProcessor",
]
