"""Foundation endpoints and file-to-file orchestration.

The typed pipeline gets its ground here (M4 step 0): a probing input
endpoint that turns a video file into a concrete ``StreamSpec`` plus a
stream of ``FrameUnit``s, an output endpoint that consumes units into
the native AVAssetWriter with the audio carry/mux policy, and
``run_file`` connecting endpoint -> scheduler -> endpoint. Endpoints
are foundations, not processors (planning 05): they never appear in a
pipeline, and payload conversion between CVPixelBuffers and MLX arrays
happens only at these edges.

Timeline convention: the product-wide integer time base 1/24000
(``VIDEO_TIME_SCALE``), unit PTS on the cadence grid anchored at the
window start, durations as successive-grid differences so no drift
accumulates - the same grid the writer stamps, which the sink verifies
per unit.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

from kinovsr.processors.errors import MediaError, PipelineError
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Domain,
    DurationPolicy,
    Geometry,
    Layout,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.processors.units import FrameUnit
from kinovsr.settings import Settings

from .builder import BuildPlan, resolve_pipeline
from .scheduler import run_plan

# Endpoint-supported payload layouts and the CVPixelBuffer format each
# decodes through. MLX frames ride RGBAHalf so fp16 precision survives
# the decode (the learned-chain currency); CV layouts pass the decoded
# buffer through untouched (the native-session currency).
_DECODE_FORMATS = {
    Layout.MLX_RGB_HWC: "PIX_RGBAHALF",
    Layout.CV_RGBA_HALF: "PIX_RGBAHALF",
    Layout.CV_NV12: "PIX_NV12",
    Layout.CV_BGRA: "PIX_BGRA",
}


def _matrix_token(resolved: tuple) -> str:
    """Map the resolved CV matrix constant onto the spec vocabulary."""
    matrix = str(resolved[2])
    for token in ("2020", "709", "601"):
        if token in matrix:
            return token
    return "709"


def _cadence(fps: float) -> Fraction:
    """Snap a probed float fps to its exact broadcast rational (24000/1001
    family and integer rates land exactly)."""
    if fps <= 0:
        raise MediaError(f"source reports non-positive fps {fps!r}")
    return Fraction(fps).limit_denominator(1001)


class FileSource:
    """Input endpoint: probe a video file into a concrete ``StreamSpec``
    and iterate its decoded frames as ``FrameUnit``s.

    Windowing (``start``/``end``/``max_frames``) is reader-level per the
    architecture: the decode seeks near the window and trims frame-exact,
    so upstream frames are never decoded. Unit PTS are grid ticks
    relative to the window start (output files start at t=0, matching
    the writer's session clock).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        layout: Layout = Layout.MLX_RGB_HWC,
        start: int = 0,
        end: int | None = None,
        max_frames: int | None = None,
        chunk_size: int = 8,
        reader: Any = None,
    ) -> None:
        if layout not in _DECODE_FORMATS:
            supported = ", ".join(k.value for k in _DECODE_FORMATS)
            raise MediaError(
                f"input endpoint cannot produce layout {layout.value!r} "
                f"(supported: {supported})")
        from kinovsr import pixel_buffers as _pb
        from kinovsr import video_reader as _native_vr

        self._vr = reader if reader is not None else _native_vr
        self._pb = _pb
        self.path = Path(path)
        self.layout = layout
        self.chunk_size = int(chunk_size)

        width, height, fps, total, transform, pixel_aspect = (
            self._vr.probe_video(self.path))
        from kinovsr import color as _color

        src_color = self._vr.probe_color(self.path)
        self.resolved_color = _color.resolve(src_color, "auto", "auto")
        self.transform = transform
        self.pixel_aspect = pixel_aspect
        self.source_fps = fps

        if start < 0:
            raise MediaError(f"start must be >= 0, got {start}")
        self.start = start
        stop = total if end is None else min(end, total)
        if max_frames is not None:
            stop = min(stop, start + max_frames)
        if stop <= start:
            raise MediaError(
                f"empty frame window [{start}, {stop}) of {total}-frame "
                f"source {self.path.name}")
        self.end = stop

        cadence = _cadence(fps)
        geometry_kwargs = {}
        if pixel_aspect is not None:
            geometry_kwargs["pixel_aspect"] = Fraction(*pixel_aspect)
        domain = Domain.CODED if layout is Layout.CV_NV12 else Domain.UNIT
        self.spec = StreamSpec(
            frame=frame_spec_for_matrix(
                _matrix_token(self.resolved_color),
                full_range=bool(self.resolved_color[3]),
                geometry=Geometry(width, height, **geometry_kwargs),
                layout=layout,
                domain=domain,
            ),
            timeline=TimelineSpec(
                time_base=Fraction(1, _pb.VIDEO_TIME_SCALE),
                cadence=cadence,
            ),
            seekable=True,
            lookahead_available=True,
        )

    @property
    def frame_count(self) -> int:
        return self.end - self.start

    def _grid_ticks(self, index: int) -> int:
        timeline = self.spec.timeline
        return round(index / timeline.cadence / timeline.time_base)

    def units(self) -> Iterator[FrameUnit]:
        """Decode the window and yield timestamped units, one per frame."""
        decode_format = getattr(self._pb, _DECODE_FORMATS[self.layout])
        to_mlx = self.layout is Layout.MLX_RGB_HWC
        index = 0
        for chunk in self._vr.iter_video_buffer_chunks(
                self.path, decode_format, chunk_size=self.chunk_size,
                start_frame=self.start, end_frame=self.end):
            for buffer in chunk:
                payload = (self._pb.read_buffer_rgb_f32(buffer)
                           if to_mlx else buffer)
                pts = self._grid_ticks(index)
                yield FrameUnit(
                    payload=payload, pts=pts,
                    duration=self._grid_ticks(index + 1) - pts)
                index += 1

    def audio_track(self) -> Any:
        """Read the source's audio for carry, or None when it has none.

        Carry is only coherent when the video window starts at the clip
        head: the endpoint has no audio-trimming policy yet, and pairing
        offset video with unoffset audio would desynchronize the output.
        """
        if self.start != 0:
            raise MediaError(
                "audio carry with a nonzero start window is not supported "
                "by the file endpoints yet; run the full clip or drop audio")
        from kinovsr._harness import _read_audio_track_from_video

        return _read_audio_track_from_video(self.path, self._vr)


class FileSink:
    """Output endpoint: consume ``FrameUnit``s into the native writer.

    Audio carry/mux policy lives here (planning 04): a supplied audio
    track is muxed only when the chain preserved clip duration, which is
    exactly what keeps it synchronized. Each appended unit's PTS is
    verified against the writer's cadence grid so a mistimed chain fails
    loudly instead of writing a silently-drifting file.
    """

    def __init__(
        self,
        path: Path | str,
        output_spec: StreamSpec,
        *,
        source: FileSource | None = None,
        quality: float = 0.65,
        label: str = "pipeline",
        audio_track: Any = None,
        audio_codec: str = "alac",
    ) -> None:
        from kinovsr import pixel_buffers as _pb
        from kinovsr.writer import (
            HEVC_PROFILE_MAIN10,
            HEVC_PROFILE_MAIN422_10,
            AVWriter,
        )

        layout = output_spec.frame.layout
        if layout not in _DECODE_FORMATS:
            raise MediaError(
                f"output endpoint cannot encode layout {layout.value!r}")
        timeline = output_spec.timeline
        if not isinstance(timeline.cadence, Fraction):
            raise MediaError("output endpoint requires a CFR cadence")
        if audio_track is not None and (
                timeline.duration_policy is not DurationPolicy.PRESERVED):
            raise MediaError(
                "audio carry requires preserved clip duration; the chain "
                "declared duration_policy=rewritten")

        self._pb = _pb
        self.spec = output_spec
        geometry = output_spec.frame.geometry
        self._is_mlx = layout is Layout.MLX_RGB_HWC
        resolved = source.resolved_color if source is not None else None
        from kinovsr import color as _color

        writer_kwargs: dict[str, Any] = {}
        if resolved is not None:
            writer_kwargs["color_props"] = _color.av_color_properties(resolved)
            if layout in (Layout.MLX_RGB_HWC, Layout.CV_RGBA_HALF):
                writer_kwargs["cv_color"] = _color.cv_triple(resolved)
                writer_kwargs["full_range"] = bool(resolved[3])
        if source is not None:
            writer_kwargs["transform"] = source.transform
            if geometry.pixel_aspect != 1:
                writer_kwargs["pixel_aspect"] = (
                    geometry.pixel_aspect.numerator,
                    geometry.pixel_aspect.denominator)

        profile = (HEVC_PROFILE_MAIN10 if layout is Layout.CV_NV12
                   else HEVC_PROFILE_MAIN422_10)
        self.writer = AVWriter(
            Path(path),
            width=geometry.width, height=geometry.height,
            fps=float(timeline.cadence),
            source_pixel_format=getattr(_pb, _DECODE_FORMATS[layout]),
            profile=profile, quality=quality, label=label,
            audio_track=audio_track, audio_codec=audio_codec,
            **writer_kwargs)

        self._pool = None
        if self._is_mlx:
            self._pool = _pb.make_pool_from_attrs({
                "PixelFormatType": _pb.PIX_RGBAHALF,
                "Width": geometry.width, "Height": geometry.height,
                "IOSurfaceProperties": {},
                "MetalCompatibility": True,
            })

    def _grid_ticks(self, index: int) -> int:
        timeline = self.spec.timeline
        return round(index / timeline.cadence / timeline.time_base)

    def _mlx_to_buffer(self, frame: Any) -> Any:
        import mlx.core as mx

        geometry = self.spec.frame.geometry
        if (int(frame.shape[0]), int(frame.shape[1])) != (
                geometry.height, geometry.width):
            raise PipelineError(
                f"output frame is {frame.shape[1]}x{frame.shape[0]} but the "
                f"validated output spec says {geometry.width}x"
                f"{geometry.height}")
        pb = self._pb.pool_create_buffer(self._pool)
        if pb is None:
            pb = self._pb.make_pixel_buffer_from_attrs(
                geometry.width, geometry.height, {
                    "PixelFormatType": self._pb.PIX_RGBAHALF,
                    "IOSurfaceProperties": {},
                    "MetalCompatibility": True,
                })
        rgb = frame[..., :3].astype(mx.float16)
        alpha = mx.ones((geometry.height, geometry.width, 1),
                        dtype=mx.float16)
        self._pb.write_fp16_rgba(
            mx.contiguous(mx.concatenate([rgb, alpha], axis=-1)), pb)
        return pb

    def append(self, unit: FrameUnit) -> None:
        expected = self._grid_ticks(self.writer.frame_count)
        if abs(unit.pts - expected) > 1:
            raise PipelineError(
                f"unit {self.writer.frame_count} arrived at pts {unit.pts} "
                f"but the output cadence grid expects {expected}; the chain "
                f"broke its declared timeline")
        payload = (self._mlx_to_buffer(unit.payload)
                   if self._is_mlx else unit.payload)
        self.writer.append(payload)

    def finish(self) -> Path:
        self.writer.finish()
        return self.writer.path


@dataclasses.dataclass(frozen=True, slots=True)
class FileRunResult:
    path: Path
    frames_in: int
    frames_out: int
    output_spec: StreamSpec
    elapsed_s: float


def run_file(
    config: dict,
    *,
    video: Path | str,
    output: Path | str,
    settings: Settings,
    layout: Layout = Layout.MLX_RGB_HWC,
    start: int = 0,
    end: int | None = None,
    max_frames: int | None = None,
    audio: bool = False,
    audio_codec: str = "alac",
    quality: float = 0.65,
    chunk_size: int = 8,
    reader: Any = None,
) -> FileRunResult:
    """Run a composed pipeline config file-to-file through the endpoints.

    Resolution and preflight validation happen against the probed input
    spec before any frame decodes; the sink verifies the plan's declared
    output timeline as units arrive.
    """
    t0 = time.perf_counter()
    source = FileSource(
        video, layout=layout, start=start, end=end,
        max_frames=max_frames, chunk_size=chunk_size, reader=reader)
    plan: BuildPlan = resolve_pipeline(
        config, input_spec=source.spec, settings=settings)
    track = source.audio_track() if audio else None
    sink = FileSink(
        output, plan.output_spec, source=source, quality=quality,
        audio_track=track, audio_codec=audio_codec)
    context = PipelineContext(settings=settings)
    frames_out = 0
    run = run_plan(plan, source.units(), context)
    try:
        with run:
            for unit in run:
                sink.append(unit)
                frames_out += 1
    except BaseException:
        # Release writer resources without masking the chain's error; the
        # partial output is not a deliverable, so a finalize failure here
        # is noise.
        with contextlib.suppress(Exception):
            sink.finish()
        raise
    path = sink.finish()
    return FileRunResult(
        path=path, frames_in=source.frame_count, frames_out=frames_out,
        output_spec=plan.output_spec,
        elapsed_s=time.perf_counter() - t0)


__all__ = ["FileRunResult", "FileSink", "FileSource", "run_file"]
