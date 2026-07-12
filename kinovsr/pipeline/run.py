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
import os
import tempfile
import time
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

from kinovsr.processors.errors import MediaError, PipelineError
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
        source_color: str = "auto",
        source_range: str = "auto",
        reader: Any = None,
    ) -> None:
        if layout not in _DECODE_FORMATS:
            supported = ", ".join(k.value for k in _DECODE_FORMATS)
            raise MediaError(
                f"input endpoint cannot produce layout {layout.value!r} "
                f"(supported: {supported})")
        from kinovsr.media import pixel_buffers as _pb
        from kinovsr.media import video_reader as _native_vr

        self._vr = reader if reader is not None else _native_vr
        self._pb = _pb
        self.path = Path(path)
        self.layout = layout
        self.chunk_size = int(chunk_size)

        width, height, fps, total, transform, pixel_aspect = (
            self._vr.probe_video(self.path))
        from kinovsr.media import color as _color

        self._src_color = self._vr.probe_color(self.path)
        self.resolved_color = _color.resolve(
            self._src_color, source_color, source_range)
        # A forced matrix/range does not just re-tag the output: it re-reads
        # the raw YUV with the chosen matrix (the fix for untagged SD clips VT
        # mis-guesses). That native re-decode path is RGBAHalf-only.
        self._force_read = source_color != "auto" or source_range != "auto"
        if self._force_read:
            if layout is not Layout.MLX_RGB_HWC:
                raise MediaError(
                    "forced --source-color/--source-range needs the MLX decode "
                    "path (RGBAHalf); a native-CV chain cannot reinterpret the "
                    "source's code values")
            if not hasattr(self._vr, "iter_forced_color_chunks"):
                raise MediaError(
                    "forced --source-color/--source-range needs the native "
                    "reader; the ffmpeg reader cannot re-decode raw YUV")
        self.transform = transform
        self.pixel_aspect = pixel_aspect
        self.source_fps = fps

        if start < 0:
            raise MediaError(f"start must be >= 0, got {start}")
        self.start = start
        self._total = total
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
        if self._force_read:
            # Re-decode raw YUV, force the resolved matrix, reinterpret the
            # range: source's actual full_range in, resolved range out.
            chunks = self._vr.iter_forced_color_chunks(
                self.path, decode_format, self.resolved_color[2],
                self._src_color["full_range"], chunk_size=self.chunk_size,
                start_frame=self.start, end_frame=self.end,
                reinterpret_full_range=self.resolved_color[3])
        else:
            chunks = self._vr.iter_video_buffer_chunks(
                self.path, decode_format, chunk_size=self.chunk_size,
                start_frame=self.start, end_frame=self.end)
        index = 0
        for chunk in chunks:
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

        The carry covers exactly the video window: a windowed run trims
        the track to [start, end) in seconds, so shortened video never
        ships beside full-length audio.
        """
        from kinovsr.media.audio import read_audio_track_from_video

        track = read_audio_track_from_video(self.path, self._vr)
        if track is None or (self.start, self.end) == (0, self._total):
            return track
        return track.trimmed(self.start / self.source_fps,
                             self.end / self.source_fps)


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
        encode_chroma: str = "auto",
    ) -> None:
        from kinovsr.media import pixel_buffers as _pb
        from kinovsr.native.writer import (
            HEVC_PROFILE_MAIN10,
            HEVC_PROFILE_MAIN422_10,
            AVWriter,
        )

        layout = output_spec.frame.layout
        if layout not in _DECODE_FORMATS:
            raise MediaError(
                f"output endpoint cannot encode layout {layout.value!r}")
        geometry = output_spec.frame.geometry
        if geometry.width % 2 or geometry.height % 2:
            raise MediaError(
                f"output geometry {geometry.width}x{geometry.height} has an "
                f"odd dimension; the 4:2:2 and 4:2:0 encoder paths need even "
                f"luma width AND height (4:2:0 subsamples both) - crop or pad "
                f"the chain to an even geometry")
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
        from kinovsr.media import color as _color

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

        # Publish atomically: encode into a unique temp sibling and only
        # rename it over the requested output when finish() succeeds. A
        # failure mid-run (e.g. weights that fail to load at the first pull,
        # after the writer already opened) then leaves any pre-existing
        # output file untouched instead of destroying it.
        self._final_path = Path(path)
        self._final_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=self._final_path.parent,
            prefix=f".{self._final_path.name}.", suffix=".partial")
        os.close(fd)
        self._temp_path = Path(tmp)
        self._published = False

        # Everything below can fail (AVWriter construction, pool creation);
        # drop the just-created temp so a failed construction leaves nothing
        # behind (and never touches the requested output).
        try:
            # auto picks by output layout (NV12/fast is inherently 4:2:0, the
            # RGBAHalf/learned path preserves chroma -> 4:2:2); an explicit
            # 420/422 forces the profile, mirroring the harness's
            # _pick_hevc_profile.
            if encode_chroma == "420":
                profile = HEVC_PROFILE_MAIN10
            elif encode_chroma == "422":
                profile = HEVC_PROFILE_MAIN422_10
            else:
                profile = (HEVC_PROFILE_MAIN10 if layout is Layout.CV_NV12
                           else HEVC_PROFILE_MAIN422_10)
            self.writer = AVWriter(
                self._temp_path,
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
        except BaseException:
            with contextlib.suppress(Exception):
                self._temp_path.unlink()
            raise

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
        # The chain's timeline is the validated one: stamp the unit's own
        # ticks (the writer's index grid quantizes NTSC-family rates).
        self.writer.append(payload, pts_ticks=unit.pts,
                           duration_ticks=unit.duration or None)

    def finish(self) -> Path:
        """Finalize the encode and publish it atomically to the requested
        output path (Path.replace is atomic on the same filesystem). On ANY
        failure - a writer-finalization error, or a rename that cannot land
        (e.g. the output path is an existing directory, or a full disk) -
        the partial temp is removed and the requested output is left
        untouched."""
        try:
            self.writer.finish()
            self._temp_path.replace(self._final_path)
        except BaseException:
            with contextlib.suppress(Exception):
                self._temp_path.unlink()
            raise
        self._published = True
        return self._final_path

    def discard(self) -> None:
        """Abandon the run: release the writer and delete the partial temp
        file WITHOUT publishing, leaving any pre-existing output untouched.
        Safe to call after a failure at any point, and a no-op once finish()
        has already published.

        Cleanup is BaseException-safe and never raises: the temp is unlinked
        in a ``finally`` even if the writer finalization is interrupted
        (KeyboardInterrupt/SystemExit), and discard swallows that interrupt so
        it cannot mask the original processing failure the caller re-raises."""
        if self._published:
            return
        try:
            self.writer.finish()
        except BaseException:  # noqa: BLE001 - best-effort; must not mask the original
            pass
        finally:
            with contextlib.suppress(Exception):
                self._temp_path.unlink()


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
    reporter: Any = None,
    layout: Layout = Layout.MLX_RGB_HWC,
    start: int = 0,
    end: int | None = None,
    max_frames: int | None = None,
    max_output_frames: int | None = None,
    max_output_seconds: float | None = None,
    audio: bool = False,
    audio_codec: str = "alac",
    save_audio_sidecar: bool = False,
    quality: float = 0.65,
    chunk_size: int = 8,
    source_color: str = "auto",
    source_range: str = "auto",
    encode_chroma: str = "auto",
    reader: Any = None,
) -> FileRunResult:
    """Run a composed pipeline config file-to-file through the endpoints.

    The chain itself is the host session (:mod:`.session`): resolution
    and preflight validation happen against the probed input spec before
    any frame decodes, and the sink verifies the declared output
    timeline as units arrive. ``max_frames`` windows the INPUT;
    ``max_output_frames`` caps what the sink writes (the distinction
    matters for cadence-changing chains).
    """
    from .session import open_pipeline

    t0 = time.perf_counter()
    video_path = Path(video).resolve()
    output_path = Path(output).resolve()
    if video_path == output_path or (
            output_path.exists() and video_path.samefile(output_path)):
        raise MediaError(
            f"output {output_path} is the input file; the writer truncates "
            f"its target before the first decoded frame, which would "
            f"destroy the source")
    source = FileSource(
        video, layout=layout, start=start, end=end,
        max_frames=max_frames, chunk_size=chunk_size,
        source_color=source_color, source_range=source_range, reader=reader)
    session = open_pipeline(
        config, source.spec, settings=settings, reporter=reporter)
    # The output cap resolves against the OUTPUT cadence (a time-form cap
    # on a cadence-changing chain means output duration, not input).
    out_cadence = session.output_spec.timeline.cadence
    if max_output_seconds is not None:
        if max_output_frames is not None:
            raise MediaError(
                "state max_output_frames or max_output_seconds, not both")
        max_output_frames = round(max_output_seconds * out_cadence)
    if max_output_frames is not None and max_output_frames < 1:
        raise MediaError(
            f"the output cap must be at least one frame; got "
            f"{max_output_frames}")
    track = source.audio_track() if audio else None
    if track is not None and max_output_frames is not None:
        # Capped video must not ship beside longer audio: trim the carry
        # to the capped output duration (a cap past the natural end is a
        # no-op; trimmed() clamps).
        track = track.trimmed(0.0, max_output_frames / float(out_cadence))
    if save_audio_sidecar and track is not None:
        # A WAV sidecar of the (trimmed) carried track, beside the output.
        track.save_wav(output_path.with_name(f"{output_path.stem}_audio.wav"))
    sink = FileSink(
        output, session.output_spec, source=source, quality=quality,
        audio_track=track, audio_codec=audio_codec,
        encode_chroma=encode_chroma)
    # A duration-preserving chain must end exactly at the source-window
    # duration. Interpolation's regenerated grid can round the final unit's
    # end past that (its natural grid-interval duration overshoots the last
    # source frame), which would leave muxed audio drifting; clamp the last
    # emitted unit's duration so total output duration matches the source.
    preserve_duration = (session.output_spec.timeline.duration_policy
                         is DurationPolicy.PRESERVED)
    source_end_ticks = round(
        source.frame_count / source.source_fps
        / session.output_spec.timeline.time_base)
    frames_out = 0
    pending: FrameUnit | None = None
    try:
        # retain_outputs=False: the sink consumes each unit into the encoder
        # synchronously, so outputs need not be copied for retention. A
        # one-unit holdback lets the final frame be clamped before it is
        # written (the clamp needs to know it is the last).
        with session, session.process(
                source.units(), retain_outputs=False) as run:
            for unit in run:
                if pending is not None:
                    sink.append(pending)
                    frames_out += 1
                pending = unit
                if (max_output_frames is not None
                        and frames_out + 1 == max_output_frames):
                    # The staged unit is the capped final frame; stop pulling
                    # (cadence-changing stages mean output count != input).
                    break
            if pending is not None:
                if preserve_duration:
                    # End exactly at the source-window duration: interpolation's
                    # regenerated grid can round the final unit past it. Clamp
                    # to min(source end, natural end) - an interior or capped
                    # frame that already lands earlier is untouched, and only a
                    # true tail overshoot is trimmed. This also covers a cap set
                    # to the natural output count, where the final frame IS the
                    # overshoot (the earlier hit_cap skip missed that).
                    clamped_end = min(source_end_ticks,
                                      pending.pts + pending.duration)
                    pending = pending.retimed(
                        pending.pts, max(1, clamped_end - pending.pts))
                sink.append(pending)
                frames_out += 1
    except BaseException:
        # The partial output is not a deliverable; drop the temp file and
        # leave any pre-existing file at `output` untouched.
        sink.discard()
        raise
    path = sink.finish()
    return FileRunResult(
        path=path, frames_in=source.frame_count, frames_out=frames_out,
        output_spec=session.output_spec,
        elapsed_s=time.perf_counter() - t0)


__all__ = ["FileRunResult", "FileSink", "FileSource", "run_file"]
