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
from contextlib import suppress
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

from .pools import MlxUploadPool, apply_output_pool

_PROFILES = ("normal", "high")

# Spatial modes = profiles. Each couples scale, output CV layout, dtype,
# domain, and the input size cap VideoToolbox enforces (native/vsr.py:
# HQ 1920x1080, LowLatency 960x960). balanced is temporal (prev-frame
# chain); fast/image are per-frame.
_UPSCALE_PROFILES = ("fast", "balanced", "image")
_UPSCALE_MODE = {
    #        scale, dst layout,          dtype,          domain,        max_w, max_h
    "fast": (2, Layout.CV_NV12, DType.UINT8, Domain.CODED, 960, 960),
    "balanced": (4, Layout.CV_RGBA_HALF, DType.FLOAT16, Domain.UNIT, 1920, 1080),
    "image": (4, Layout.CV_RGBA_HALF, DType.FLOAT16, Domain.UNIT, 1920, 1080),
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
    cadence = spec.timeline.cadence
    # A carried non-uniform timeline interpolates by real source times
    # (the session's per-destination phase array carries arbitrary
    # spacing); its unit-count relation to the source is judged against
    # the nominal rate when one exists.
    reference = (cadence if isinstance(cadence, Fraction)
                 else spec.timeline.nominal_cadence)
    timeline = dataclasses.replace(
        spec.timeline,
        cadence=config.target_fps,
        timestamp_policy=TimestampPolicy.REGENERATED,
        cardinality=(
            Cardinality.ONE_TO_MANY
            if reference is None or config.target_fps > reference
            else Cardinality.MANY_TO_ONE
        ),
        nominal_cadence=None,
    )
    # VTFrameRateConversion always emits its destination currency,
    # RGBAHalf, regardless of whether the accepted source was BGRA, NV12,
    # RGBAHalf, or uploaded MLX RGB. The typed edge must describe the actual
    # payload so downstream writer-pool compatibility is decided correctly.
    frame = dataclasses.replace(
        spec.frame, layout=Layout.CV_RGBA_HALF, dtype=DType.FLOAT16, domain=Domain.UNIT
    )
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
        self._nominal: Fraction | None = None
        self._last_hold: Fraction | None = None
        self._publication_origin_pts: int | None = None
        self._target_index: int | None = None
        self._input_end_pts: int | None = None
        self._pending: FrameUnit | None = None
        self._upload_bridge: MlxUploadPool | None = None
        # A normal host stream keeps its first input PTS as the output origin.
        # Negative input PTS are the file endpoint's processing-only GOP
        # context; those indices stay negative on the target grid so frame 0
        # is generated at the requested public in-point, not relabeled later.
        self._origin: int | None = None
        self._output_pool_binding: tuple[Any, int, int, int] | None = None

    def _bind_output_pool(
        self,
        pool: Any,
        pixel_format: int,
        width: int,
        height: int,
    ) -> None:
        self._output_pool_binding = (pool, pixel_format, width, height)

    def _grid_ticks(self, target_index: int) -> int:
        return grid_ticks(target_index, self._config.target_fps, self._time_base)

    def _emit(self, payload: Any) -> FrameUnit:
        assert self._target_index is not None
        m = self._target_index
        self._target_index += 1
        pts = self._grid_ticks(m)
        origin = self._origin or 0
        return FrameUnit(payload=payload, pts=origin + pts, duration=self._grid_ticks(m + 1) - pts)

    def prepare(self, input_spec: StreamSpec, context: PipelineContext) -> None:
        from kinovsr.native.temporal import VtfrcSession

        cadence = input_spec.timeline.cadence
        uniform = isinstance(cadence, Fraction)
        geometry = input_spec.frame.geometry
        self._time_base = input_spec.timeline.time_base
        # Uniform sources keep the exact index grid (bit-identical to the
        # historical mapping); a carried timeline feeds real times and the
        # nominal rate serves only as the session's identity hint.
        self._source_cadence = cadence if uniform else None
        self._nominal = (cadence if uniform
                         else (input_spec.timeline.nominal_cadence
                               or Fraction(30)))
        self._publication_origin_pts = context.publication_origin_pts
        output_pool_binding, self._output_pool_binding = (
            self._output_pool_binding,
            None,
        )
        # An MLX input uploads each frame into an IOSurface-backed
        # RGBAHalf buffer for the session (the learned-upscale ->
        # --target-fps shape); CV inputs feed straight through.
        try:
            if input_spec.frame.layout is Layout.MLX_RGB_HWC:
                self._upload_bridge = MlxUploadPool(geometry.width, geometry.height)
            self._session = VtfrcSession(
                geometry.width,
                geometry.height,
                source_fps=self._nominal,
                target_fps=self._config.target_fps,
                mode=self._config.mode,
            )
            apply_output_pool(self._session, output_pool_binding, geometry.width, geometry.height)
        except (RuntimeError, SystemExit) as exc:
            session, self._session = self._session, None
            upload, self._upload_bridge = self._upload_bridge, None
            with suppress(Exception):
                if session is not None:
                    session.close()
            with suppress(Exception):
                if upload is not None:
                    upload.close()
            raise MediaError(f"VTFrameRateConversion session unavailable: {exc}") from exc

    def process(self, unit: FrameUnit, context: PipelineContext) -> Iterable[FrameUnit]:
        if self._origin is None:
            self._origin = (
                unit.pts if self._publication_origin_pts is None else self._publication_origin_pts
            )
        assert self._time_base is not None
        payload = unit.payload
        if self._upload_bridge is not None:
            payload = self._upload_bridge.upload(payload)
        if self._source_cadence is not None:
            index = round(Fraction(unit.pts - self._origin) * self._time_base * self._source_cadence)
            source_step = grid_ticks(index + 1, self._source_cadence, self._time_base) - grid_ticks(
                index, self._source_cadence, self._time_base
            )
            self._input_end_pts = unit.pts + (unit.duration if unit.duration > 0 else source_step)
            if self._target_index is None:
                self._target_index = math.ceil(
                    Fraction(index) * self._config.target_fps / self._source_cadence
                )
            produced_iter = self._session.feed(payload, index)
        else:
            # Carried non-uniform timeline: feed the frame's REAL time; the
            # session brackets target grid slots between real stamps.
            assert self._nominal is not None
            src_time = Fraction(unit.pts - self._origin) * self._time_base
            fallback = max(round(1 / self._nominal / self._time_base), 1)
            hold_ticks = unit.duration if unit.duration > 0 else fallback
            self._last_hold = Fraction(hold_ticks) * self._time_base
            self._input_end_pts = unit.pts + hold_ticks
            if self._target_index is None:
                self._target_index = math.ceil(
                    src_time * self._config.target_fps)
            produced_iter = self._session.feed_at(payload, src_time)
        for produced in produced_iter:
            out = self._emit(produced)
            if self._pending is not None:
                yield self._pending
            self._pending = out

    def reset(self, boundary: Boundary, context: PipelineContext) -> None:
        # The scheduler drained the pre-boundary tail via flush(), which
        # clears the buffered pair inside the session; the target grid
        # deliberately keeps counting so PTS stays monotonic.
        pass

    def flush(self, context: PipelineContext) -> Iterable[FrameUnit]:
        if self._session is None:
            return
        hold = self._last_hold if self._source_cadence is None else None
        for payload in self._session.drain(hold=hold):
            unit = self._emit(payload)
            if self._input_end_pts is not None and unit.pts >= self._input_end_pts:
                continue
            if self._pending is not None:
                yield self._pending
            self._pending = unit
        if self._pending is not None:
            pending, self._pending = self._pending, None
            if self._input_end_pts is not None:
                duration = min(pending.duration, self._input_end_pts - pending.pts)
                if duration <= 0:
                    return
                pending = pending.retimed(pending.pts, duration)
            yield pending

    def close(self, context: PipelineContext) -> None:
        session, self._session = self._session, None
        self._output_pool_binding = None
        self._pending = None
        self._input_end_pts = None
        try:
            if session is not None:
                session.close()
        finally:
            upload, self._upload_bridge = self._upload_bridge, None
            if upload is not None:
                upload.close()


@dataclasses.dataclass(frozen=True, slots=True)
class VtUpscaleConfig:
    mode: str  # "fast" | "balanced" | "image"
    flow: str = "vt"  # balanced only: "vt" | "vision"


def _upscale_produces(spec: StreamSpec, config: object) -> StreamSpec:
    assert isinstance(config, VtUpscaleConfig)
    scale, dst_layout, dst_dtype, dst_domain, max_w, max_h = _UPSCALE_MODE[config.mode]
    g = spec.frame.geometry
    # The size cap is a VideoToolbox hard limit and mode-dependent, so it is
    # checked here (open time) rather than as a StreamConstraint bound.
    if g.width > max_w or g.height > max_h:
        raise ValueError(
            f"videotoolbox {config.mode} upscale accepts input up to "
            f"{max_w}x{max_h}; got {g.width}x{g.height}"
        )
    # A CV source must be the mode's own session format (zero-copy path);
    # the layout<->mode pairing is mode-dependent, so it is checked here
    # like the size cap rather than as a StreamConstraint bound.
    native = _UPSCALE_NATIVE_SRC[config.mode]
    if spec.frame.layout is not Layout.MLX_RGB_HWC and (spec.frame.layout is not native):
        raise ValueError(
            f"videotoolbox {config.mode} upscale takes MLX frames or its "
            f"native {native.value} source; got {spec.frame.layout.value}"
        )
    frame = dataclasses.replace(
        spec.frame, layout=dst_layout, dtype=dst_dtype, domain=dst_domain, geometry=g.scaled(scale)
    )
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
        self._pending: FrameUnit | None = None
        self._output_pool_binding: tuple[Any, int, int, int] | None = None

    def _bind_output_pool(
        self,
        pool: Any,
        pixel_format: int,
        width: int,
        height: int,
    ) -> None:
        self._output_pool_binding = (pool, pixel_format, width, height)

    def prepare(self, input_spec: StreamSpec, context: PipelineContext) -> None:
        import mlx.core as mx

        from kinovsr.native.vsr import VsrSession

        self._mx = mx
        output_pool_binding, self._output_pool_binding = (
            self._output_pool_binding,
            None,
        )
        # A CV source is already in the session's source format (validated
        # at resolve): feed it zero-copy via upscale_buffer_to_buffer, the
        # harness's mainstream decode->VSR path. MLX frames upload.
        self._cv_input = input_spec.frame.layout is not Layout.MLX_RGB_HWC
        g = input_spec.frame.geometry
        # The session fps is VT's internal frame-identity grid, a rate hint
        # only - a carried non-uniform timeline supplies its nominal rate.
        cadence = input_spec.timeline.cadence
        rate_hint = (cadence if isinstance(cadence, Fraction)
                     else input_spec.timeline.nominal_cadence or Fraction(30))
        try:
            self._session = VsrSession(
                g.width, g.height, mode=self._config.mode,
                fps=float(rate_hint),
                # Balanced VSR's private internal flow is timing-sensitive.
                # Both explicit backends are overlapped one frame ahead.
                # Public VT Quality flow is the default; Vision is an opt-in
                # zero-copy alternative that also works below VT's measured
                # 128-pixel destination-writer boundary.
                explicit_flow=self._config.mode == "balanced",
                flow_backend=self._config.flow,
            )
            apply_output_pool(
                self._session, output_pool_binding, self._session.out_w, self._session.out_h
            )
        except (RuntimeError, SystemExit, ValueError) as exc:
            session, self._session = self._session, None
            with suppress(Exception):
                if session is not None:
                    session.close()
            raise MediaError(f"VideoToolbox VSR session unavailable: {exc}") from exc

    def process(self, unit: FrameUnit, context: PipelineContext) -> Iterable[FrameUnit]:
        if self._cv_input:
            out = self._session.submit_upscale_buffer_to_buffer(
                unit.payload,
                self._index,
            )
        else:
            mx = self._mx
            rgb = unit.payload
            rgba = mx.concatenate(
                [
                    rgb.astype(mx.float16),
                    mx.ones((*rgb.shape[:2], 1), mx.float16),
                ],
                axis=-1,
            )
            out = self._session.submit_upscale_to_buffer(rgba, self._index)
        self._index += 1
        if out is None:
            if self._pending is not None:
                raise RuntimeError(
                    "VideoToolbox VSR accepted a second delayed frame"
                )
            self._pending = unit
            return
        if self._pending is None:
            yield unit.with_payload(out)
            return
        pending, self._pending = self._pending, unit
        yield pending.with_payload(out)

    def reset(self, boundary: Boundary, context: PipelineContext) -> None:
        if self._session is not None:
            if self._pending is not None:
                raise RuntimeError(
                    "VideoToolbox VSR must be flushed before a hard-cut reset"
                )
            self._session.reset_temporal_context()

    def flush(self, context: PipelineContext) -> Iterable[FrameUnit]:
        if self._session is None:
            return
        out = self._session.finish_pending_upscale()
        if out is None:
            if self._pending is not None:
                raise RuntimeError(
                    "VideoToolbox VSR lost its delayed output while flushing"
                )
            return
        pending, self._pending = self._pending, None
        if pending is None:
            raise RuntimeError(
                "VideoToolbox VSR produced an output with no pending frame"
            )
        yield pending.with_payload(out)

    def close(self, context: PipelineContext) -> None:
        session, self._session = self._session, None
        self._output_pool_binding = None
        self._pending = None
        if session is not None:
            session.close()


class VideoToolboxFactory:
    name = "videotoolbox"

    capabilities = {
        Capability.INTERPOLATE: CapabilitySpec(
            capability=Capability.INTERPOLATE,
            profiles=_PROFILES,
            accepts=StreamConstraint(
                layouts=(Layout.CV_BGRA, Layout.CV_RGBA_HALF, Layout.CV_NV12, Layout.MLX_RGB_HWC),
                # No cadence bound: uniform sources ride the exact index
                # grid; carried non-uniform timelines feed real source
                # times, which the VT per-destination phase array supports
                # natively.
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
                layouts=(Layout.MLX_RGB_HWC, Layout.CV_RGBA_HALF, Layout.CV_NV12),
                dtypes=(DType.FLOAT32, DType.FLOAT16, DType.UINT8),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED, Domain.CODED),
                # No cadence bound: upscaling is per-frame (balanced threads
                # one prev frame ORDINALLY) and never consumes the clock, so
                # carried non-uniform timelines pass through. Only genuine
                # time-resampling (INTERPOLATE) requires a uniform grid.
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
            reject_unknown_keys(raw, ("flow",))
            mode = profile or "balanced"
            if mode not in _UPSCALE_PROFILES:
                raise ValueError(
                    f"videotoolbox upscale profile must be one of {list(_UPSCALE_PROFILES)}"
                )
            flow = str(raw.get("flow", "vt"))
            if flow not in ("vt", "vision"):
                raise ValueError(
                    "videotoolbox upscale flow must be one of ['vt', 'vision']"
                )
            if mode != "balanced" and "flow" in raw:
                raise ValueError(
                    "videotoolbox upscale flow is only valid for "
                    "the balanced profile"
                )
            return VtUpscaleConfig(mode=mode, flow=flow)
        reject_unknown_keys(raw, ("target_fps",))
        if "target_fps" not in raw:
            raise ValueError("target_fps is required for interpolation")
        return VtInterpolateConfig(
            target_fps=_parse_fps(raw["target_fps"]), mode=profile or "normal"
        )

    def build(
        self, config: VtInterpolateConfig | VtUpscaleConfig, *, context: PipelineContext
    ) -> VtInterpolateProcessor | VtUpscaleProcessor:
        if isinstance(config, VtUpscaleConfig):
            return VtUpscaleProcessor(config)
        return VtInterpolateProcessor(config)

    def preferred_source_layout(
        self, *, capability: Capability, profile: str | None
    ) -> Layout | None:
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
