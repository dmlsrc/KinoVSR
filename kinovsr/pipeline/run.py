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
    comparison_path: Path | None = None


class _ComparisonTee:
    """Side-by-side comparison output: NEAREST-upscaled source vs post.

    The typed analog of the harness's comparison writer. ``run_file`` owns
    both endpoints, so the tee retains decoded source frames (uint8 RGB,
    bounded by chain latency), pairs each emitted output unit with the
    latest source frame at-or-before its instant (interpolated outputs
    repeat the source frame, exactly as the harness fed the same ``src_arr``
    to every frame-rate-converted output), nearest-upscales the source half
    to the output geometry, and appends the composite to a second FileSink
    riding the same output timeline. The composite is built in pure MLX -
    integer-gather nearest resample + concat - which for the harness's
    integer-scale shapes is pixel-identical to its ``mx.repeat`` pre half.
    """

    def __init__(
        self,
        path: Path | str,
        output_spec: StreamSpec,
        source: FileSource,
        *,
        quality: float,
        audio_track: Any = None,
        audio_codec: str = "alac",
        encode_chroma: str = "auto",
    ) -> None:
        from collections import deque

        from kinovsr.processors.specs import DType

        out_geometry = output_spec.frame.geometry
        comp_frame = dataclasses.replace(
            output_spec.frame,
            layout=Layout.MLX_RGB_HWC,
            dtype=DType.FLOAT32,
            geometry=Geometry(out_geometry.width * 2, out_geometry.height,
                              out_geometry.pixel_aspect))
        self.sink = FileSink(
            path, dataclasses.replace(output_spec, frame=comp_frame),
            source=source, quality=quality, label="comparison",
            audio_track=audio_track, audio_codec=audio_codec,
            encode_chroma=encode_chroma)
        self._retained: Any = deque()   # (seconds, uint8 (H,W,3) mx.array)
        self._src_time_base = float(source.spec.timeline.time_base)
        self._out_time_base = float(output_spec.timeline.time_base)
        self._src_layout = source.spec.frame.layout
        self._out_layout = output_spec.frame.layout
        self._out_w, self._out_h = out_geometry.width, out_geometry.height
        src_geometry = source.spec.frame.geometry
        # Nearest-neighbor gather maps, source -> output geometry. For an
        # integer scale s this is j // s, identical to mx.repeat.
        import mlx.core as mx

        self._ix = mx.array(
            [(x * src_geometry.width) // self._out_w
             for x in range(self._out_w)], dtype=mx.int32)
        self._iy = mx.array(
            [(y * src_geometry.height) // self._out_h
             for y in range(self._out_h)], dtype=mx.int32)

    def _to_uint8_rgb(self, payload: Any, layout: Layout) -> Any:
        import mlx.core as mx

        from kinovsr.media import pixel_buffers as _pb

        if layout is Layout.MLX_RGB_HWC:
            rgb = (payload[..., :3] if payload.dtype == mx.uint8
                   else mx.clip(payload[..., :3] * 255.0, 0,
                                255).astype(mx.uint8))
        else:
            rgb = _pb.read_pixel_buffer_rgb(payload)
        rgb = mx.contiguous(rgb)
        mx.eval(rgb)   # materialize: free the decode graph, keep only uint8
        return rgb

    def tap(self, units: Any) -> Any:
        """Wrap the source iterator, retaining each decoded frame."""
        for unit in units:
            self._retained.append(
                (unit.pts * self._src_time_base,
                 self._to_uint8_rgb(unit.payload, self._src_layout)))
            yield unit

    def emit(self, unit: FrameUnit) -> None:
        """Composite ``unit`` against its paired source frame and append."""
        import mlx.core as mx

        out_seconds = unit.pts * self._out_time_base
        # Advance to the LATEST retained source frame at-or-before this
        # output instant; drop everything strictly earlier (paired frames
        # can repeat for cadence-upsampled outputs, so keep the pair).
        while (len(self._retained) >= 2
               and self._retained[1][0] <= out_seconds + 1e-9):
            self._retained.popleft()
        if not self._retained:
            raise PipelineError(
                "comparison tee has no retained source frame to pair with "
                "an emitted output unit; the chain emitted before consuming")
        pre = self._retained[0][1]
        pre_up = mx.take(mx.take(pre, self._iy, axis=0), self._ix, axis=1)
        pre_f = pre_up.astype(mx.float32) / 255.0
        if self._out_layout is Layout.MLX_RGB_HWC:
            post_f = unit.payload[..., :3].astype(mx.float32)
        else:
            from kinovsr.media import pixel_buffers as _pb

            post_f = _pb.read_pixel_buffer_rgb(
                unit.payload).astype(mx.float32) / 255.0
        self.sink.append(
            unit.with_payload(mx.concatenate([pre_f, post_f], axis=1)))


def _save_frame_png(payload: Any, layout: Layout, out_dir: Path,
                    index: int, _pb: Any) -> None:
    """Write one unit's RGB to ``out_dir/frame_NNNNN.png``.

    An MLX unit carries a float32 [0,1] (H,W,3|4) array (uint8 already
    display-ready); a native unit carries a CVPixelBuffer read straight to
    uint8 RGB - the same two forms the harness dumped.
    """
    import mlx.core as mx

    from kinovsr.media.images import save_image

    if layout is Layout.MLX_RGB_HWC:
        rgb = (payload[..., :3] if payload.dtype == mx.uint8
               else mx.clip(payload[..., :3] * 255.0, 0, 255).astype(mx.uint8))
    else:
        rgb = _pb.read_pixel_buffer_rgb(payload)
    save_image(rgb, out_dir / f"frame_{index:05d}.png")


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
    save_pre_frames: Path | str | None = None,
    save_post_frames: Path | str | None = None,
    comparison: Path | str | None = None,
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

    # Optional per-frame PNG dumps: pre = the SOURCE frames (before the chain),
    # post = the encoded output frames (after it). Debug taps, not chain
    # stages - the PRE tap wraps the source iterator so it dumps exactly the
    # frames the session pulls, and the POST tap dumps each emitted unit.
    from kinovsr.media import pixel_buffers as _pbmod
    pre_dir = Path(save_pre_frames) if save_pre_frames else None
    post_dir = Path(save_post_frames) if save_post_frames else None
    source_units = source.units()
    if pre_dir is not None:
        pre_dir.mkdir(parents=True, exist_ok=True)
        src_layout = source.spec.frame.layout

        def _pre_tapped(units: Any) -> Any:
            for i, unit in enumerate(units):
                _save_frame_png(unit.payload, src_layout, pre_dir, i, _pbmod)
                yield unit
        source_units = _pre_tapped(source_units)
    if post_dir is not None:
        post_dir.mkdir(parents=True, exist_ok=True)
    post_layout = session.output_spec.frame.layout
    post_index = 0

    # The side-by-side comparison output (harness --comparison parity):
    # retained source frames pair with emitted output units into a second
    # sink at 2*out_w. auto chroma follows the POST output's own pick so
    # both files carry the same profile, as the harness's shared `profile`
    # variable did.
    tee = None
    if comparison is not None:
        comp_chroma = encode_chroma
        if (comp_chroma == "auto"
                and session.output_spec.frame.layout is Layout.CV_NV12):
            comp_chroma = "420"
        tee = _ComparisonTee(
            comparison, session.output_spec, source, quality=quality,
            audio_track=track, audio_codec=audio_codec,
            encode_chroma=comp_chroma)
        source_units = tee.tap(source_units)

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
                source_units, retain_outputs=False) as run:
            for unit in run:
                if post_dir is not None:
                    _save_frame_png(unit.payload, post_layout, post_dir,
                                    post_index, _pbmod)
                    post_index += 1
                if pending is not None:
                    sink.append(pending)
                    if tee is not None:
                        tee.emit(pending)
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
                if tee is not None:
                    tee.emit(pending)
                frames_out += 1
    except BaseException:
        # The partial output is not a deliverable; drop the temp files and
        # leave any pre-existing files untouched.
        sink.discard()
        if tee is not None:
            tee.sink.discard()
        raise
    path = sink.finish()
    comparison_path = tee.sink.finish() if tee is not None else None
    return FileRunResult(
        path=path, frames_in=source.frame_count, frames_out=frames_out,
        output_spec=session.output_spec,
        elapsed_s=time.perf_counter() - t0,
        comparison_path=comparison_path)


__all__ = ["FileRunResult", "FileSink", "FileSource", "run_file"]
