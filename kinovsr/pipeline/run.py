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
import enum
import errno
import fcntl
import hashlib
import logging
import os
import shutil
import stat as stat_module
import tempfile
import threading
import time
import unicodedata
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

from kinovsr.media.chunks import (
    budgeted_decode_chunk_size,
    validate_decode_chunk_size,
)
from kinovsr.media.errors import is_native_operation_error
from kinovsr.media.filesystem import rename_exclusive
from kinovsr.media.timing import (
    AudioTiming,
    VideoTiming,
    grid_ticks,
    rational_cadence,
)
from kinovsr.processors.boundaries import BoundaryKind
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

_log = logging.getLogger(__name__)

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

_DECODE_BYTES_PER_PIXEL = {
    # MLX input decodes through one 64RGBAHalf CVPixelBuffer before upload.
    Layout.MLX_RGB_HWC: 8,
    Layout.CV_RGBA_HALF: 8,
    Layout.CV_BGRA: 4,
    # NV12 is nominally 1.5 bytes/pixel. Two is deliberately conservative
    # for row alignment and keeps the budget independent of plane stride.
    Layout.CV_NV12: 2,
}


@contextlib.contextmanager
def _media_operation(operation: str, path: Path | str) -> Iterator[None]:
    """Normalize an explicit native or filesystem operation.

    The scope at each call site must stay narrow: ``RuntimeError`` is the
    established failure type used by the native media adapters, but can also
    describe an internal invariant elsewhere. Programmer errors and process
    control exceptions deliberately pass through unchanged.
    """
    try:
        yield
    except PipelineError:
        raise
    except Exception as exc:
        if is_native_operation_error(exc):
            raise MediaError(f"{operation} {path}: {exc}") from exc
        raise


def _close_created_fd(fd: int) -> None:
    """Close a newly-created fd exactly once.

    A failed ``close`` has ambiguous ownership: the kernel may already have
    released the descriptor and another thread may reuse its number. Retrying
    could therefore close an unrelated resource, even when ``fstat`` reports
    the same inode through a newly-opened descriptor.
    """
    os.close(fd)


def _unlink_temporary(path: Path) -> None:
    """Make one best-effort removal without replacing an active failure."""
    try:
        path.unlink(missing_ok=True)
    except BaseException as exc:
        # An unlink may complete before reporting an interruption. A retry can
        # delete a same-name replacement, so an ambiguous private residue is
        # safer than a second destructive operation.
        _log.warning("could not remove temporary %s: %s", path, exc)


def _unlink_created_raw(raw: str, primary: BaseException) -> None:
    """Remove a raw mkstemp pathname without invoking Path conversion."""
    try:
        os.unlink(raw)  # noqa: PTH108 - Path conversion may be the failure
    except FileNotFoundError:
        pass
    except BaseException as cleanup_exc:
        primary.add_note(
            f"temporary acquisition cleanup also failed: {cleanup_exc!r}")


def _effective_decode_chunk_size(
    requested: int,
    width: int,
    height: int,
    layout: Layout,
    *,
    forced_color: bool = False,
) -> int:
    """Cap retained decode surfaces to the endpoint's 64 MiB budget."""
    requested = _validate_requested_chunk_size(requested)
    bytes_per_pixel = _DECODE_BYTES_PER_PIXEL[layout]
    if forced_color:
        # Forced reads retain the decoded 10-bit 4:2:2 YUV chunk while
        # building the RGBAHalf output chunk. A range reinterpretation can
        # also retain one retyped YUV surface, so use the conservative sum of
        # two four-byte YUV surfaces plus eight RGBAHalf bytes per pixel.
        bytes_per_pixel = 16
    return budgeted_decode_chunk_size(
        requested, width, height, bytes_per_pixel)


def _validate_requested_chunk_size(requested: Any) -> int:
    """Return a strict positive frame-count request or raise a typed error."""
    try:
        return validate_decode_chunk_size(requested)
    except ValueError as exc:
        raise MediaError(str(exc)) from exc


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
    try:
        return rational_cadence(fps, max_denominator=1001)
    except ValueError as exc:
        raise MediaError(f"source reports invalid fps {fps!r}") from exc


def _reader_module(reader: Any) -> Any:
    if reader is not None:
        return reader
    from kinovsr.media import video_reader

    return video_reader


def _probe_cfr_timing(path: Path, reader: Any) -> VideoTiming | None:
    """Return exact CFR metadata or reject VFR before output mutation.

    Custom reader adapters written before this contract may omit the timing
    probe; their existing explicit ``probe_video`` cadence remains supported.
    Both built-in readers implement this metadata-only scan.
    """
    probe = getattr(_reader_module(reader), "probe_video_timing", None)
    if probe is None:
        return None
    try:
        timing = probe(path)
    except MediaError:
        raise
    except Exception as exc:
        if is_native_operation_error(exc):
            raise MediaError(
                f"failed to inspect source timing for {path}: {exc}") from exc
        raise
    if timing.cadence is None:
        raise MediaError(
            f"{path.name}: variable frame rate (VFR) is not supported by the "
            f"file pipeline ({timing.sample_count} display samples over "
            f"{float(timing.duration):.6g}s); convert to CFR before processing")
    return timing


def _validate_audio_origin(
    path: Path,
    reader: Any,
    video_timing: VideoTiming | None,
) -> None:
    """Reject audio carry whose source clock cannot yet be preserved."""
    if video_timing is None:
        return
    probe = getattr(_reader_module(reader), "probe_audio_timing", None)
    if probe is None:
        return
    try:
        audio_timing: AudioTiming | None = probe(path)
    except MediaError:
        raise
    except Exception as exc:
        if is_native_operation_error(exc):
            raise MediaError(
                f"failed to inspect source audio timing for {path}: {exc}"
            ) from exc
        raise
    if audio_timing is None:
        return
    # Each reported origin can be rounded by at most half of its own clock
    # tick. A whole coarse video tick is a real frame-scale skew, not harmless
    # quantization (notably for AVI streams whose time base is 1/fps).
    tolerance = (
        video_timing.source_tick + audio_timing.source_tick) / 2
    if abs(audio_timing.first_pts - video_timing.first_pts) > tolerance:
        raise MediaError(
            f"{path.name}: staggered audio/video track origins are not "
            f"supported (video starts at {float(video_timing.first_pts):.6g}s, "
            f"audio at {float(audio_timing.first_pts):.6g}s); align or remux "
            f"the tracks before requesting audio carry")


class FileSource:
    """Input endpoint: probe a video file into a concrete ``StreamSpec``
    and iterate its decoded frames as ``FrameUnit``s.

    Windowing (``start``/``end``/``max_frames``) is reader-level per the
    architecture: the decode seeks near the window and trims frame-exact,
    so upstream frames are never decoded. Unit PTS are grid ticks
    relative to the window start (output files start at t=0, matching
    the writer's session clock). ``chunk_size`` is capped against an
    approximate 64 MiB retained output-surface budget after source geometry
    and decode layout are known; it does not describe total process RSS.
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
        context_frames: int = 0,
        reader: Any = None,
        timing: VideoTiming | None = None,
    ) -> None:
        if layout not in _DECODE_FORMATS:
            supported = ", ".join(k.value for k in _DECODE_FORMATS)
            raise MediaError(
                f"input endpoint cannot produce layout {layout.value!r} "
                f"(supported: {supported})")
        requested_chunk_size = _validate_requested_chunk_size(chunk_size)
        from kinovsr.media import pixel_buffers as _pb
        self._vr = _reader_module(reader)
        self._pb = _pb
        self.path = Path(path)
        self.layout = layout
        if timing is None:
            timing = _probe_cfr_timing(self.path, self._vr)

        with _media_operation("failed to probe video source", self.path):
            width, height, fps, total, transform, pixel_aspect = (
                self._vr.probe_video(self.path))
        cadence = timing.cadence if timing is not None else _cadence(fps)
        assert cadence is not None
        if timing is not None:
            total = timing.sample_count
        fps = float(cadence)
        self._timing = timing
        from kinovsr.media import color as _color

        with _media_operation("failed to probe source color", self.path):
            self._src_color = self._vr.probe_color(self.path)
        self.resolved_color = _color.resolve(
            self._src_color, source_color, source_range)
        origin = ("tagged" if self._src_color["tagged"]
                  else "untagged, reader guessed"
                  if self._src_color.get("guessed") else "untagged")
        _log.info(
            "Source color: %s -> output %s",
            origin, _color.describe(self.resolved_color))
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
        self.chunk_size = _effective_decode_chunk_size(
            requested_chunk_size,
            width,
            height,
            layout,
            forced_color=self._force_read,
        )
        if self.chunk_size != requested_chunk_size:
            _log.info(
                "Decode chunk capped from %s to %s frames for %sx%s %s",
                requested_chunk_size,
                self.chunk_size,
                width,
                height,
                layout.value,
            )
        self.transform = transform
        self.pixel_aspect = pixel_aspect
        self.source_fps = fps
        self.source_cadence = cadence

        if start < 0:
            raise MediaError(f"start must be >= 0, got {start}")
        self.start = start
        # gop-align context: frames BEFORE the window, decoded and fed as
        # recurrence warmup with NEGATIVE pts. They are processed, never
        # output - run_file drops negative-time outputs - so the output
        # timeline stays anchored at `start` and frame_count is unchanged.
        if context_frames < 0 or context_frames > start:
            raise MediaError(
                f"context_frames must be within [0, start={start}], got "
                f"{context_frames}")
        self.context_frames = int(context_frames)
        self._total = total
        stop = total if end is None else min(end, total)
        if max_frames is not None:
            stop = min(stop, start + max_frames)
        if stop <= start:
            raise MediaError(
                f"empty frame window [{start}, {stop}) of {total}-frame "
                f"source {self.path.name}")
        self.end = stop

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
        return grid_ticks(index, timeline.cadence, timeline.time_base)

    def units(self) -> Iterator[FrameUnit]:
        """Decode the window and yield timestamped units, one per frame.

        Context frames (gop-align) extend the read BEFORE the window and
        carry negative pts; the window's first frame stays at pts 0.
        """
        decode_format = getattr(self._pb, _DECODE_FORMATS[self.layout])
        to_mlx = self.layout is Layout.MLX_RGB_HWC
        read_start = self.start - self.context_frames
        timing_kwargs = ({"timing": self._timing}
                         if self._timing is not None else {})
        if self._force_read:
            # Re-decode raw YUV, force the resolved matrix, reinterpret the
            # range: source's actual full_range in, resolved range out.
            with _media_operation("failed to open video decode", self.path):
                chunks = self._vr.iter_forced_color_chunks(
                    self.path, decode_format, self.resolved_color[2],
                    self._src_color["full_range"], chunk_size=self.chunk_size,
                    start_frame=read_start, end_frame=self.end,
                    reinterpret_full_range=self.resolved_color[3],
                    **timing_kwargs)
        else:
            with _media_operation("failed to open video decode", self.path):
                chunks = self._vr.iter_video_buffer_chunks(
                    self.path, decode_format, chunk_size=self.chunk_size,
                    start_frame=read_start, end_frame=self.end,
                    **timing_kwargs)
        with _media_operation("failed to open video decode", self.path):
            chunk_iterator = iter(chunks)
        index = -self.context_frames
        while True:
            try:
                with _media_operation("failed to decode video source", self.path):
                    chunk = next(chunk_iterator)
            except StopIteration:
                break
            for buffer in chunk:
                if to_mlx:
                    with _media_operation(
                            "failed to read decoded pixel buffer", self.path):
                        payload = self._pb.read_buffer_rgb_f32(buffer)
                else:
                    payload = buffer
                pts = self._grid_ticks(index)
                yield FrameUnit(
                    payload=payload, pts=pts,
                    duration=self._grid_ticks(index + 1) - pts)
                index += 1

    def audio_track(self, *, max_duration: Fraction | None = None) -> Any:
        """Open the source's bounded audio window, or None when absent.

        Exact rational frame boundaries and an optional output-duration cap
        are resolved before either backend allocates or decodes PCM.
        """
        from kinovsr.media.audio import read_audio_track_from_video

        start_sec = Fraction(self.start) / self.source_cadence
        end_sec = Fraction(self.end) / self.source_cadence
        try:
            return read_audio_track_from_video(
                self.path,
                self._vr,
                start_sec=start_sec,
                end_sec=end_sec,
                max_duration_sec=max_duration,
            )
        except MediaError:
            raise
        except Exception as exc:
            if is_native_operation_error(exc):
                raise MediaError(
                    f"failed to open bounded audio from {self.path}: {exc}"
                ) from exc
            raise


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
        overwrite: bool = False,
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

        if resolved is None:
            resolved = _color.resolve_frame_spec(output_spec.frame)

        writer_kwargs: dict[str, Any] = {}
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
        self._overwrite = overwrite
        if self._final_path.exists() and not overwrite:
            raise MediaError(
                f"destination already exists: {self._final_path}; pass "
                f"overwrite=True to replace it")
        raw: str | None = None
        fd: int | None = None
        try:
            self._final_path.parent.mkdir(parents=True, exist_ok=True)
            fd, raw = tempfile.mkstemp(
                dir=self._final_path.parent,
                prefix=f".{self._final_path.name}.", suffix=".partial")
            owned_fd, fd = fd, None
            _close_created_fd(owned_fd)
            temp_path = Path(raw)
        except BaseException as exc:
            if fd is not None:
                try:
                    _close_created_fd(fd)
                except BaseException as cleanup_exc:
                    exc.add_note(
                        f"writer temporary descriptor cleanup also failed: "
                        f"{cleanup_exc!r}")
            if raw is not None:
                _unlink_created_raw(raw, exc)
            if isinstance(exc, OSError):
                raise MediaError(
                    f"failed to create writer temporary for "
                    f"{self._final_path}: {exc}") from exc
            raise
        self._temp_path = temp_path
        self._published = False
        self._finalized = False
        self._discarded = False
        self._transaction_managed = False

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
                fps=timeline.cadence,
                source_pixel_format=getattr(_pb, _DECODE_FORMATS[layout]),
                profile=profile, quality=quality, label=label,
                audio_track=audio_track, audio_codec=audio_codec,
                **writer_kwargs)

            self._direct_mlx_encode = (
                self._is_mlx
                and bool(getattr(self.writer, "accepts_mlx_rgb", False))
            )
            self._pool = None
            if self._is_mlx and not self._direct_mlx_encode:
                self._pool = _pb.make_pool_from_attrs({
                    "PixelFormatType": _pb.PIX_RGBAHALF,
                    "Width": geometry.width, "Height": geometry.height,
                    "IOSurfaceProperties": {},
                    "MetalCompatibility": True,
                })
        except BaseException as exc:
            writer = getattr(self, "writer", None)
            if writer is not None:
                with contextlib.suppress(BaseException):
                    writer.cancel()
            _unlink_temporary(self._temp_path)
            if isinstance(exc, PipelineError):
                raise
            if is_native_operation_error(exc, allow_value_error=True):
                raise MediaError(
                    f"writer setup failed for {self._final_path}: {exc}"
                ) from exc
            raise

    def _grid_ticks(self, index: int) -> int:
        timeline = self.spec.timeline
        return grid_ticks(index, timeline.cadence, timeline.time_base)

    def _validate_mlx_frame(self, frame: Any) -> None:
        geometry = self.spec.frame.geometry
        shape = getattr(frame, "shape", ())
        if (len(shape) != 3
                or int(shape[0]) != geometry.height
                or int(shape[1]) != geometry.width
                or int(shape[2]) != 3):
            raise PipelineError(
                f"output frame shape {tuple(shape)!r} does not match the "
                f"validated {geometry.width}x{geometry.height} RGB spec")

    def _mlx_to_buffer(self, frame: Any) -> Any:
        import mlx.core as mx

        geometry = self.spec.frame.geometry
        with _media_operation(
                "writer buffer allocation failed for", self._final_path):
            pb = self._pb.pool_create_buffer(self._pool)
        if pb is None:
            with _media_operation(
                    "writer buffer allocation failed for", self._final_path):
                pb = self._pb.make_pixel_buffer_from_attrs(
                    geometry.width, geometry.height, {
                        "PixelFormatType": self._pb.PIX_RGBAHALF,
                        "IOSurfaceProperties": {},
                        "MetalCompatibility": True,
                    })
        rgb = frame[..., :3].astype(mx.float16)
        alpha = mx.ones((geometry.height, geometry.width, 1),
                        dtype=mx.float16)
        rgba = mx.contiguous(mx.concatenate([rgb, alpha], axis=-1))
        with _media_operation(
                "writer buffer upload failed for", self._final_path):
            self._pb.write_fp16_rgba(rgba, pb)
        return pb

    def append(self, unit: FrameUnit) -> None:
        expected = self._grid_ticks(self.writer.frame_count)
        if abs(unit.pts - expected) > 1:
            raise PipelineError(
                f"unit {self.writer.frame_count} arrived at pts {unit.pts} "
                f"but the output cadence grid expects {expected}; the chain "
                f"broke its declared timeline")
        # The chain's timeline is the validated one: stamp the unit's own
        # ticks (the writer's index grid quantizes NTSC-family rates).
        if self._is_mlx:
            self._validate_mlx_frame(unit.payload)
            if self._direct_mlx_encode:
                try:
                    prepared = self.writer.prepare_mlx_rgb(unit.payload)
                except Exception as exc:
                    if is_native_operation_error(exc):
                        raise MediaError(
                            f"writer MLX preparation failed for "
                            f"{self._final_path}: {exc}") from exc
                    raise
                try:
                    self.writer.append_prepared_mlx_rgb(
                        prepared, pts_ticks=unit.pts,
                        duration_ticks=unit.duration or None)
                except Exception as exc:
                    if is_native_operation_error(
                            exc, allow_value_error=True):
                        raise MediaError(
                            f"writer append failed for {self._final_path}: "
                            f"{exc}") from exc
                    raise
                return
            payload = self._mlx_to_buffer(unit.payload)
        else:
            payload = unit.payload
        try:
            self.writer.append(
                payload, pts_ticks=unit.pts,
                duration_ticks=unit.duration or None)
        except Exception as exc:
            if is_native_operation_error(exc, allow_value_error=True):
                raise MediaError(
                    f"writer append failed for {self._final_path}: {exc}"
                ) from exc
            raise

    def finalize(self) -> None:
        """Finish encoding while the file is still transaction-private."""
        if self._finalized or self._published:
            return
        if self._discarded:
            raise MediaError(
                f"cannot finalize discarded output {self._final_path}")
        try:
            self.writer.finish()
        except BaseException as exc:
            self.discard()
            if isinstance(exc, PipelineError):
                raise
            if is_native_operation_error(exc, allow_value_error=True):
                raise MediaError(
                    f"writer finalization failed for {self._final_path}: "
                    f"{exc}") from exc
            raise
        self._finalized = True

    def _mark_published(self) -> None:
        self._published = True

    def finish(self) -> Path:
        """Finalize the encode and publish it atomically to the requested
        output path (Darwin exclusive rename closes the no-overwrite race). On ANY
        failure - a writer-finalization error, or a rename that cannot land
        (e.g. the output path is an existing directory, or a full disk) -
        the partial temp is removed and the requested output is left
        untouched."""
        self.finalize()
        try:
            if self._overwrite:
                self._temp_path.replace(self._final_path)
            else:
                rename_exclusive(self._temp_path, self._final_path)
        except BaseException as exc:
            _unlink_temporary(self._temp_path)
            if isinstance(exc, PipelineError):
                raise
            if isinstance(exc, (OSError, RuntimeError)):
                raise MediaError(
                    f"writer publication failed for {self._final_path}: {exc}"
                ) from exc
            raise
        self._mark_published()
        return self._final_path

    def discard(self) -> None:
        """Abandon the run: release the writer and delete the partial temp
        file WITHOUT publishing, leaving any pre-existing output untouched.
        Safe to call after a failure at any point, and a no-op once finish()
        has already published.

        Cleanup is BaseException-safe and never raises: native writing is
        cancelled (never finalized), and the temp is unlinked even if native
        cancellation is interrupted, so cleanup cannot mask the original
        processing failure the caller re-raises."""
        if self._published or self._discarded:
            return
        self._discarded = True
        try:
            self.writer.cancel()
        except BaseException:  # noqa: BLE001 - best-effort; must not mask the original
            pass
        finally:
            if not getattr(self, "_transaction_managed", False):
                _unlink_temporary(self._temp_path)


@dataclasses.dataclass(frozen=True, slots=True)
class _Artifact:
    label: str
    path: Path
    directory: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _ArtifactPlan:
    """Complete, normalized destructive path graph for one file run."""

    input_path: Path
    output_path: Path
    artifacts: tuple[_Artifact, ...]
    overwrite: bool

    @staticmethod
    def _resolved(path: Path | str, label: str) -> Path:
        try:
            return Path(path).expanduser().resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise MediaError(f"cannot resolve {label} path {path}: {exc}") from exc

    @staticmethod
    def _namespace_path(path: Path) -> Path:
        """Conservative APFS/HFS namespace identity for uncreated paths."""
        return Path(unicodedata.normalize("NFD", str(path)).casefold())

    @classmethod
    def build(
        cls,
        *,
        video: Path | str,
        output: Path | str,
        comparison: Path | str | None,
        cut_log: Path | str | None,
        save_audio_sidecar: bool,
        save_pre_frames: Path | str | None,
        save_post_frames: Path | str | None,
        skip_post_mp4: bool,
        noise_map_debug: bool,
        overwrite: bool,
    ) -> _ArtifactPlan:
        input_path = cls._resolved(video, "input")
        output_path = cls._resolved(output, "post output")
        artifacts: list[_Artifact] = []

        def add(label: str, path: Path | str, *, directory: bool = False) -> None:
            artifacts.append(_Artifact(
                label, cls._resolved(path, label), directory))

        if not skip_post_mp4:
            add("post output", output_path)
        if comparison is not None:
            add("comparison output", comparison)
        if save_audio_sidecar:
            add(
                "audio sidecar",
                output_path.with_name(f"{output_path.stem}_audio.wav"),
            )
        if cut_log is not None:
            add("cut log", cut_log)
        if save_pre_frames is not None:
            add("pre-frame directory", save_pre_frames, directory=True)
        if save_post_frames is not None:
            add("post-frame directory", save_post_frames, directory=True)
        if noise_map_debug and not skip_post_mp4:
            for suffix in ("noisemap", "blockmap"):
                add(
                    f"{suffix} debug image",
                    output_path.with_name(
                        f"{output_path.stem}_{suffix}.png"),
                )

        plan = cls(input_path, output_path, tuple(artifacts), overwrite)
        plan.validate()
        return plan

    def path(self, label: str) -> Path:
        for artifact in self.artifacts:
            if artifact.label == label:
                return artifact.path
        raise KeyError(label)

    def get(self, label: str) -> Path | None:
        with contextlib.suppress(KeyError):
            return self.path(label)
        return None

    @staticmethod
    def _same_existing_path(left: Path, right: Path) -> bool:
        if not left.exists() or not right.exists():
            return False
        try:
            return left.samefile(right)
        except OSError:
            return False

    @staticmethod
    def _validate_parent(path: Path, label: str) -> None:
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        if not parent.is_dir():
            raise MediaError(
                f"{label} parent is not a directory: {parent}")

    def validate(self) -> None:
        """Validate aliases, nesting, types, and overwrite policy read-only."""
        named = [("input", self.input_path), *(
            (artifact.label, artifact.path) for artifact in self.artifacts)]
        for index, (left_label, left) in enumerate(named):
            for right_label, right in named[index + 1:]:
                if (left == right
                        or self._namespace_path(left) == self._namespace_path(right)
                        or self._same_existing_path(left, right)):
                    source_warning = (
                        "; writing there would destroy the source"
                        if "input" in (left_label, right_label) else "")
                    raise MediaError(
                        f"artifact paths alias: {left_label} and "
                        f"{right_label} both identify {left}"
                        f"{source_warning}")

        for index, left in enumerate(self.artifacts):
            for right in self.artifacts[index + 1:]:
                left_namespace = self._namespace_path(left.path)
                right_namespace = self._namespace_path(right.path)
                if (left_namespace in right_namespace.parents
                        or right_namespace in left_namespace.parents):
                    raise MediaError(
                        f"artifact paths overlap: {left.label} ({left.path}) "
                        f"and {right.label} ({right.path})")
            if (left.directory
                    and self._namespace_path(left.path)
                    in self._namespace_path(self.input_path).parents):
                raise MediaError(
                    f"{left.label} contains the input file: {left.path}")

        for artifact in self.artifacts:
            self._validate_parent(artifact.path, artifact.label)
            if not artifact.path.exists():
                continue
            if artifact.directory != artifact.path.is_dir():
                expected = "directory" if artifact.directory else "file"
                raise MediaError(
                    f"{artifact.label} must be a {expected}: "
                    f"{artifact.path}")
            if not self.overwrite:
                raise MediaError(
                    f"destination already exists: {artifact.path}; pass "
                    f"overwrite=True to replace it")


_RESERVATION_GUARD = threading.RLock()
_RESERVED_NAMESPACES: set[Path] = set()


class _ArtifactReservation:
    """Cross-thread/process advisory reservation for a complete artifact set."""

    def __init__(self, plan: _ArtifactPlan, settings: Settings) -> None:
        self._paths = tuple(sorted(
            (artifact.path for artifact in plan.artifacts), key=str))
        self._namespaces = tuple(
            _ArtifactPlan._namespace_path(path) for path in self._paths)
        requests: dict[Path, int] = {}
        for path in self._namespaces:
            requests[path] = fcntl.LOCK_EX
            for parent in path.parents:
                requests.setdefault(parent, fcntl.LOCK_SH)
        self._lock_requests = tuple(sorted(requests.items(), key=lambda item: str(item[0])))
        try:
            self._lock_root = (
                settings.shared_temp_dir.expanduser().resolve(strict=False)
                / "kinovsr-artifact-locks")
        except (OSError, RuntimeError) as exc:
            raise MediaError(
                f"cannot resolve artifact lock directory "
                f"{settings.shared_temp_dir}: {exc}") from exc
        self._fds: list[int] = []
        self._held = False

    def acquire(self) -> None:
        if self._held:
            return
        with _RESERVATION_GUARD:
            conflict = {
                requested
                for requested in self._namespaces
                for reserved in _RESERVED_NAMESPACES
                if (requested == reserved
                    or requested in reserved.parents
                    or reserved in requested.parents)
            }
            if conflict:
                path = min(conflict, key=str)
                raise MediaError(
                    f"destination hierarchy is already reserved: {path}")
            try:
                self._lock_root.mkdir(parents=True, exist_ok=True)
                for path, mode in self._lock_requests:
                    digest = hashlib.sha256(
                        os.fsencode(str(path))).hexdigest()
                    lock_path = self._lock_root / f"{digest}.lock"
                    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                    try:
                        fcntl.flock(fd, mode | fcntl.LOCK_NB)
                    except OSError as exc:
                        os.close(fd)
                        if exc.errno in (errno.EACCES, errno.EAGAIN):
                            raise MediaError(
                                f"destination hierarchy is already reserved: "
                                f"{path}") from exc
                        raise
                    self._fds.append(fd)
            except BaseException as exc:
                self._release_fds()
                if isinstance(exc, OSError):
                    raise MediaError(
                        f"cannot reserve artifact destinations under "
                        f"{self._lock_root}: {exc}") from exc
                raise
            _RESERVED_NAMESPACES.update(self._namespaces)
            self._held = True

    def _release_fds(self) -> None:
        for fd in reversed(self._fds):
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)
        self._fds.clear()

    def release(self) -> None:
        if not self._held:
            return
        with _RESERVATION_GUARD:
            self._release_fds()
            _RESERVED_NAMESPACES.difference_update(self._namespaces)
            self._held = False


_PathIdentity = tuple[int, int, int]


class _PublicationState(enum.Enum):
    TEMP_OWNED = enum.auto()
    MOVING_TO_FINAL = enum.auto()
    FINAL_OWNED = enum.auto()
    ABSENT = enum.auto()
    AMBIGUOUS = enum.auto()
    CLEANUP_AMBIGUOUS = enum.auto()


class _BackupState(enum.Enum):
    NONE = enum.auto()
    SLOT_OWNED = enum.auto()
    MOVING_ORIGINAL = enum.auto()
    ORIGINAL_HELD = enum.auto()
    RESTORING = enum.auto()
    RESTORED = enum.auto()
    REMOVING = enum.auto()
    REMOVED = enum.auto()
    AMBIGUOUS = enum.auto()


@dataclasses.dataclass(slots=True)
class _TransactionEntry:
    artifact: _Artifact
    temp_path: Path
    sink: FileSink | None = None
    temp_identity: _PathIdentity | None = None
    publication_state: _PublicationState = _PublicationState.TEMP_OWNED
    backup_path: Path | None = None
    original_identity: _PathIdentity | None = None
    backup_slot_identity: _PathIdentity | None = None
    backup_state: _BackupState = _BackupState.NONE


class _OutputTransaction:
    """Own every temporary and publish the full artifact set all-or-none."""

    def __init__(self, plan: _ArtifactPlan, settings: Settings) -> None:
        self.plan = plan
        self._reservation = _ArtifactReservation(plan, settings)
        self._entries: dict[str, _TransactionEntry] = {}
        self._created_parents: list[Path] = []
        self._committed = False
        self._closed = False

    def __enter__(self) -> _OutputTransaction:
        self._reservation.acquire()
        try:
            # Close the validation/reservation race before any output temp is
            # created. Cooperative concurrent runs now hold the same locks.
            self.plan.validate()
        except BaseException:
            self._reservation.release()
            raise
        return self

    def _artifact(self, label: str) -> _Artifact:
        for artifact in self.plan.artifacts:
            if artifact.label == label:
                return artifact
        raise KeyError(label)

    @staticmethod
    def _identity_from_stat(result: os.stat_result) -> _PathIdentity:
        return (
            result.st_dev,
            result.st_ino,
            stat_module.S_IFMT(result.st_mode),
        )

    @classmethod
    def _path_identity(cls, path: Path | str) -> _PathIdentity:
        return cls._identity_from_stat(os.lstat(path))

    @classmethod
    def _inspect_identity(
        cls, path: Path | str,
    ) -> tuple[_PathIdentity | None, BaseException | None]:
        try:
            return cls._path_identity(path), None
        except FileNotFoundError:
            return None, None
        except BaseException as exc:
            return None, exc

    @classmethod
    def _require_owned_identity(
        cls, path: Path, *, label: str,
    ) -> _PathIdentity:
        try:
            return cls._path_identity(path)
        except OSError as exc:
            raise MediaError(f"cannot inspect {label} {path}: {exc}") from exc

    @staticmethod
    def _clear_backup(entry: _TransactionEntry, state: _BackupState) -> None:
        entry.backup_path = None
        entry.original_identity = None
        entry.backup_slot_identity = None
        entry.backup_state = state

    def _ensure_parent(self, path: Path) -> None:
        missing: list[Path] = []
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            missing.append(parent)
            parent = parent.parent
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MediaError(
                f"cannot create artifact parent {path.parent}: {exc}") from exc
        self._created_parents.extend(
            candidate for candidate in reversed(missing)
            if candidate.is_dir())

    def prepare_sink(self, label: str) -> Path:
        artifact = self._artifact(label)
        self._ensure_parent(artifact.path)
        return artifact.path

    def register_sink(self, label: str, sink: FileSink) -> None:
        if label in self._entries:
            raise RuntimeError(f"artifact already materialized: {label}")
        entry = _TransactionEntry(
            self._artifact(label), sink._temp_path, sink=sink)
        self._entries[label] = entry
        sink._transaction_managed = True
        entry.temp_identity = self._require_owned_identity(
            entry.temp_path, label=f"{label} temporary")

    def temp_file(self, label: str) -> Path:
        existing = self._entries.get(label)
        if existing is not None:
            return existing.temp_path
        artifact = self._artifact(label)
        if artifact.directory:
            raise RuntimeError(f"{label} is a directory artifact")
        self._ensure_parent(artifact.path)
        raw: str | None = None
        fd: int | None = None
        temp_identity: _PathIdentity | None = None
        try:
            fd, raw = tempfile.mkstemp(
                dir=artifact.path.parent,
                prefix=f".{artifact.path.name}.", suffix=".partial")
            owned_fd, fd = fd, None
            _close_created_fd(owned_fd)
            temp_path = Path(raw)
            temp_identity = self._require_owned_identity(
                temp_path, label=f"{label} temporary")
        except BaseException as exc:
            if fd is not None:
                try:
                    _close_created_fd(fd)
                except BaseException as cleanup_exc:
                    exc.add_note(
                        f"artifact temporary descriptor cleanup also failed: "
                        f"{cleanup_exc!r}")
            if raw is not None:
                _unlink_created_raw(raw, exc)
            if isinstance(exc, OSError):
                raise MediaError(
                    f"cannot create {artifact.label} temporary for "
                    f"{artifact.path}: {exc}") from exc
            raise
        self._entries[label] = _TransactionEntry(
            artifact,
            temp_path,
            temp_identity=temp_identity,
        )
        return temp_path

    def temp_directory(self, label: str) -> Path:
        existing = self._entries.get(label)
        if existing is not None:
            return existing.temp_path
        artifact = self._artifact(label)
        if not artifact.directory:
            raise RuntimeError(f"{label} is a file artifact")
        self._ensure_parent(artifact.path)
        raw: str | None = None
        temp_identity: _PathIdentity | None = None
        try:
            raw = tempfile.mkdtemp(
                dir=artifact.path.parent,
                prefix=f".{artifact.path.name}.", suffix=".partial")
            temp_path = Path(raw)
            temp_identity = self._require_owned_identity(
                temp_path, label=f"{label} temporary")
        except BaseException as exc:
            if raw is not None:
                with contextlib.suppress(BaseException):
                    shutil.rmtree(raw)
            if isinstance(exc, OSError):
                raise MediaError(
                    f"cannot create {artifact.label} temporary for "
                    f"{artifact.path}: {exc}") from exc
            raise
        self._entries[label] = _TransactionEntry(
            artifact,
            temp_path,
            temp_identity=temp_identity,
        )
        return temp_path

    def _ordered_entries(self) -> list[_TransactionEntry]:
        return [
            self._entries[artifact.label]
            for artifact in self.plan.artifacts
            if artifact.label in self._entries
        ]

    @staticmethod
    def _remove(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    @classmethod
    def _backup_slot(
        cls, artifact: _Artifact,
    ) -> tuple[Path, _PathIdentity | None]:
        """Reserve, then vacate, a unique rollback pathname.

        The raw name is made absent before ``Path`` conversion, so a conversion
        failure cannot orphan a slot. Destructive operations are never retried:
        an exception may arrive after the name was removed and reused.
        """
        if artifact.directory:
            raw = tempfile.mkdtemp(
                dir=artifact.path.parent,
                prefix=f".{artifact.path.name}.", suffix=".rollback")
            # The name must be absent before Path conversion so conversion
            # failure cannot orphan the directory slot.
            os.rmdir(raw)  # noqa: PTH106
            return Path(raw), None

        fd, raw = tempfile.mkstemp(
            dir=artifact.path.parent,
            prefix=f".{artifact.path.name}.", suffix=".rollback")
        unlink_failure: BaseException | None = None
        try:
            os.unlink(raw)  # noqa: PTH108 - Path conversion follows removal
        except BaseException as exc:
            unlink_failure = exc
        close_failure: BaseException | None = None
        try:
            _close_created_fd(fd)
        except BaseException as close_exc:
            close_failure = close_exc
        if unlink_failure is not None:
            if close_failure is not None:
                unlink_failure.add_note(
                    f"rollback-slot descriptor cleanup also failed: "
                    f"{close_failure!r}")
            raise unlink_failure
        if close_failure is not None:
            raise close_failure
        return Path(raw), None

    @staticmethod
    def _replace(source: Path, destination: Path) -> None:
        rename_exclusive(source, destination)

    def _remove_owned_path(
        self,
        path: Path,
        expected: _PathIdentity,
        *,
        operation: str,
    ) -> tuple[bool, list[str]]:
        """Remove a captured path once and reconcile without retrying.

        Even a matching inode cannot prove pathname-generation ownership: an
        actor can relink the same inode, or add external children to the same
        directory, after a removal completes. A second destructive call is
        therefore never safe.
        """
        try:
            self._remove(path)
        except BaseException as exc:
            errors = [f"{operation} {path}: {exc}"]
            identity, probe_exc = self._inspect_identity(path)
            if probe_exc is not None:
                errors.append(
                    f"inspect {path} after failed removal: {probe_exc}")
            elif identity is not None:
                ownership = "same identity" if identity == expected else "changed"
                errors.append(
                    f"did not retry {operation} {path}: pathname ownership "
                    f"is ambiguous ({ownership})")
            else:
                return True, errors
            return False, errors
        return True, []

    def _quarantine_and_remove(
        self,
        path: Path,
        expected: _PathIdentity,
        artifact: _Artifact,
        *,
        operation: str,
    ) -> tuple[bool, list[str]]:
        """Move a public entry aside, verify it, then remove it once."""
        errors: list[str] = []
        try:
            quarantine, _ = self._backup_slot(artifact)
        except BaseException as exc:
            return False, [f"acquire quarantine for {path}: {exc}"]
        try:
            self._replace(path, quarantine)
        except BaseException as exc:
            errors.append(f"quarantine {path}: {exc}")
            path_identity, path_exc = self._inspect_identity(path)
            quarantine_identity, quarantine_exc = self._inspect_identity(
                quarantine)
            if path_exc is not None:
                errors.append(f"inspect {path}: {path_exc}")
            if quarantine_exc is not None:
                errors.append(f"inspect {quarantine}: {quarantine_exc}")
            if path_exc is not None or quarantine_exc is not None:
                return False, errors
            if quarantine_identity == expected and path_identity != expected:
                removed, remove_errors = self._remove_owned_path(
                    quarantine,
                    expected,
                    operation=operation,
                )
                errors.extend(remove_errors)
                return removed, errors
            if quarantine_identity is not None and (
                    quarantine_identity != expected):
                try:
                    self._replace(quarantine, path)
                except BaseException as restore_exc:
                    errors.append(
                        f"restore foreign entry to {path}: {restore_exc}")
            return False, errors

        identity, probe_exc = self._inspect_identity(quarantine)
        if probe_exc is not None:
            errors.append(f"inspect quarantined {path}: {probe_exc}")
            return False, errors
        if identity != expected:
            errors.append(
                f"refused to {operation} {path}: identity changed before "
                f"quarantine")
            try:
                self._replace(quarantine, path)
            except BaseException as restore_exc:
                errors.append(
                    f"restore foreign entry to {path}: {restore_exc}")
            return False, errors

        removed, remove_errors = self._remove_owned_path(
            quarantine,
            expected,
            operation=operation,
        )
        errors.extend(remove_errors)
        return removed, errors

    def _reconcile_publication(
        self, entry: _TransactionEntry,
    ) -> list[str]:
        errors: list[str] = []
        expected = entry.temp_identity
        if expected is None:
            entry.publication_state = _PublicationState.AMBIGUOUS
            return [f"missing temporary identity for {entry.temp_path}"]

        temp_identity, temp_exc = self._inspect_identity(entry.temp_path)
        final_identity, final_exc = self._inspect_identity(entry.artifact.path)
        if temp_exc is not None:
            errors.append(f"inspect {entry.temp_path}: {temp_exc}")
        if final_exc is not None:
            errors.append(f"inspect {entry.artifact.path}: {final_exc}")
        if errors:
            entry.publication_state = _PublicationState.AMBIGUOUS
            return errors

        if temp_identity is None and final_identity == expected:
            entry.publication_state = _PublicationState.FINAL_OWNED
        elif temp_identity == expected and final_identity != expected:
            entry.publication_state = _PublicationState.TEMP_OWNED
        else:
            entry.publication_state = _PublicationState.AMBIGUOUS
            errors.append(
                f"publication ownership is ambiguous for "
                f"{entry.artifact.path}; no destination was removed")
        return errors

    def _verify_published(self, entry: _TransactionEntry) -> None:
        expected = entry.temp_identity
        if expected is None:
            entry.publication_state = _PublicationState.AMBIGUOUS
            raise MediaError(
                f"missing published identity for {entry.artifact.path}")

        temp_identity, temp_exc = self._inspect_identity(entry.temp_path)
        final_identity, final_exc = self._inspect_identity(entry.artifact.path)
        probe_failures = [
            ("temporary", temp_exc),
            ("destination", final_exc),
        ]
        probe_failures = [
            (label, failure)
            for label, failure in probe_failures
            if failure is not None
        ]
        if probe_failures:
            entry.publication_state = _PublicationState.AMBIGUOUS
            primary_label, primary = probe_failures[0]
            assert primary is not None
            for label, failure in probe_failures[1:]:
                primary.add_note(
                    f"{label} publication probe also failed: {failure!r}")
            primary.add_note(
                f"{primary_label} publication probe failed after rename")
            raise primary

        if temp_identity is None and final_identity == expected:
            entry.publication_state = _PublicationState.FINAL_OWNED
            return
        entry.publication_state = _PublicationState.AMBIGUOUS
        raise MediaError(
            f"published identity does not match completed temporary: "
            f"{entry.artifact.path}")

    def _reconcile_backup_move(
        self, entry: _TransactionEntry,
    ) -> list[str]:
        backup = entry.backup_path
        original = entry.original_identity
        if backup is None or original is None:
            self._clear_backup(entry, _BackupState.NONE)
            return []

        errors: list[str] = []
        final_identity, final_exc = self._inspect_identity(entry.artifact.path)
        backup_identity, backup_exc = self._inspect_identity(backup)
        if final_exc is not None:
            errors.append(f"inspect {entry.artifact.path}: {final_exc}")
        if backup_exc is not None:
            errors.append(f"inspect {backup}: {backup_exc}")
        if errors:
            entry.backup_state = _BackupState.AMBIGUOUS
            return errors

        if backup_identity == original and final_identity != original:
            entry.backup_state = _BackupState.ORIGINAL_HELD
            return []

        slot = entry.backup_slot_identity
        if final_identity == original and (
                backup_identity is None or backup_identity == slot):
            if backup_identity is None:
                self._clear_backup(entry, _BackupState.REMOVED)
                return []
            removed, remove_errors = self._remove_owned_path(
                backup,
                slot,
                operation="remove unused rollback slot",
            )
            errors.extend(remove_errors)
            if removed:
                self._clear_backup(entry, _BackupState.REMOVED)
            else:
                entry.backup_state = _BackupState.AMBIGUOUS
            return errors

        entry.backup_state = _BackupState.AMBIGUOUS
        errors.append(
            f"backup ownership is ambiguous for {entry.artifact.path}; "
            f"retained {backup}")
        return errors

    def _remove_published(self, entry: _TransactionEntry) -> list[str]:
        if entry.publication_state in {
            _PublicationState.MOVING_TO_FINAL,
            _PublicationState.AMBIGUOUS,
        }:
            errors = self._reconcile_publication(entry)
        else:
            errors = []
        if entry.publication_state is not _PublicationState.FINAL_OWNED:
            return errors
        if entry.temp_identity is None:
            entry.publication_state = _PublicationState.AMBIGUOUS
            errors.append(
                f"missing published identity for {entry.artifact.path}")
            return errors

        identity, probe_exc = self._inspect_identity(entry.artifact.path)
        if probe_exc is not None:
            entry.publication_state = _PublicationState.AMBIGUOUS
            errors.append(f"inspect {entry.artifact.path}: {probe_exc}")
            return errors
        if identity is None:
            entry.publication_state = _PublicationState.ABSENT
            return errors
        if identity != entry.temp_identity:
            entry.publication_state = _PublicationState.AMBIGUOUS
            errors.append(
                f"refused to remove {entry.artifact.path}: destination "
                f"identity changed")
            return errors

        removed, remove_errors = self._quarantine_and_remove(
            entry.artifact.path,
            entry.temp_identity,
            entry.artifact,
            operation="remove published artifact",
        )
        errors.extend(remove_errors)
        entry.publication_state = (
            _PublicationState.ABSENT
            if removed else _PublicationState.CLEANUP_AMBIGUOUS
        )
        return errors

    def _restore_backup(self, entry: _TransactionEntry) -> list[str]:
        errors: list[str] = []
        if entry.backup_state in {
            _BackupState.SLOT_OWNED,
            _BackupState.MOVING_ORIGINAL,
            _BackupState.AMBIGUOUS,
        }:
            errors.extend(self._reconcile_backup_move(entry))
        if entry.backup_state is not _BackupState.ORIGINAL_HELD:
            return errors

        backup = entry.backup_path
        original = entry.original_identity
        if backup is None or original is None:
            entry.backup_state = _BackupState.AMBIGUOUS
            errors.append(
                f"lost rollback metadata for {entry.artifact.path}")
            return errors

        final_identity, final_exc = self._inspect_identity(entry.artifact.path)
        backup_identity, backup_exc = self._inspect_identity(backup)
        if final_exc is not None:
            errors.append(f"inspect {entry.artifact.path}: {final_exc}")
        if backup_exc is not None:
            errors.append(f"inspect {backup}: {backup_exc}")
        if errors:
            entry.backup_state = _BackupState.AMBIGUOUS
            return errors

        if final_identity == original and backup_identity is None:
            self._clear_backup(entry, _BackupState.RESTORED)
            return errors
        if final_identity is not None:
            entry.backup_state = _BackupState.AMBIGUOUS
            errors.append(
                f"refused to restore {entry.artifact.path}: destination is "
                f"occupied; retained {backup}")
            return errors
        if backup_identity != original:
            entry.backup_state = _BackupState.AMBIGUOUS
            errors.append(
                f"refused to restore {entry.artifact.path}: rollback backup "
                f"identity changed; retained {backup}")
            return errors

        entry.backup_state = _BackupState.RESTORING
        try:
            self._replace(backup, entry.artifact.path)
        except BaseException as exc:
            errors.append(f"restore {entry.artifact.path}: {exc}")
            final_identity, final_exc = self._inspect_identity(
                entry.artifact.path)
            backup_identity, backup_exc = self._inspect_identity(backup)
            if final_exc is not None:
                errors.append(
                    f"inspect {entry.artifact.path} after restore: "
                    f"{final_exc}")
            if backup_exc is not None:
                errors.append(f"inspect {backup} after restore: {backup_exc}")
            if final_exc is not None or backup_exc is not None:
                entry.backup_state = _BackupState.AMBIGUOUS
            elif final_identity == original and backup_identity is None:
                self._clear_backup(entry, _BackupState.RESTORED)
            elif final_identity is None and backup_identity == original:
                entry.backup_state = _BackupState.ORIGINAL_HELD
            else:
                entry.backup_state = _BackupState.AMBIGUOUS
            return errors

        self._clear_backup(entry, _BackupState.RESTORED)
        return errors

    def _remove_temporary(self, entry: _TransactionEntry) -> list[str]:
        if entry.publication_state is not _PublicationState.TEMP_OWNED:
            return []
        if entry.temp_identity is None:
            entry.publication_state = _PublicationState.AMBIGUOUS
            return [f"missing temporary identity for {entry.temp_path}"]

        identity, probe_exc = self._inspect_identity(entry.temp_path)
        if probe_exc is not None:
            entry.publication_state = _PublicationState.AMBIGUOUS
            return [f"inspect {entry.temp_path}: {probe_exc}"]
        if identity is None:
            entry.publication_state = _PublicationState.ABSENT
            return []
        if identity != entry.temp_identity:
            entry.publication_state = _PublicationState.AMBIGUOUS
            return [
                f"refused to remove {entry.temp_path}: temporary identity "
                f"changed",
            ]

        removed, errors = self._remove_owned_path(
            entry.temp_path,
            entry.temp_identity,
            operation="remove temporary artifact",
        )
        entry.publication_state = (
            _PublicationState.ABSENT
            if removed else _PublicationState.CLEANUP_AMBIGUOUS
        )
        return errors

    def _rollback(self, entries: list[_TransactionEntry]) -> list[str]:
        errors: list[str] = []
        for entry in reversed(entries):
            errors.extend(self._remove_published(entry))
        for entry in reversed(entries):
            errors.extend(self._restore_backup(entry))
        for entry in reversed(entries):
            errors.extend(self._remove_temporary(entry))
        return errors

    def _cleanup_committed_backup(
        self, entry: _TransactionEntry,
    ) -> BaseException | None:
        """Quarantine, verify, then remove one displaced destination.

        Moving to another private, exclusive name before deletion prevents a
        same-name replacement at the rollback path from being deleted. The
        final removal operates only on the random transaction-private name.
        """
        backup = entry.backup_path
        original = entry.original_identity
        if backup is None or original is None:
            return None
        try:
            quarantine, _ = self._backup_slot(entry.artifact)
            self._replace(backup, quarantine)
        except BaseException as exc:
            entry.backup_state = _BackupState.AMBIGUOUS
            return exc

        identity, probe_exc = self._inspect_identity(quarantine)
        if probe_exc is not None:
            entry.backup_state = _BackupState.AMBIGUOUS
            return probe_exc
        if identity != original:
            failure = MediaError(
                f"rollback backup identity changed during cleanup: {backup}")
            try:
                self._replace(quarantine, backup)
            except BaseException as restore_exc:
                failure.add_note(
                    f"foreign rollback-path entry was retained at "
                    f"{quarantine}: {restore_exc!r}")
            entry.backup_state = _BackupState.AMBIGUOUS
            return failure

        entry.backup_state = _BackupState.REMOVING
        try:
            self._remove(quarantine)
        except BaseException as exc:
            entry.backup_state = _BackupState.AMBIGUOUS
            return exc
        self._clear_backup(entry, _BackupState.REMOVED)
        return None

    def commit(self) -> None:
        if self._committed:
            return
        entries = self._ordered_entries()
        try:
            # Native finish is bounded and still private. No destination is
            # visible until every writer has completed successfully.
            for entry in entries:
                if entry.sink is not None:
                    entry.sink.finalize()

            # Some owned materializers atomically replace the initially
            # reserved placeholder (for example ImageIO when writing PNG).
            # Seal the identity of each completed private artifact only after
            # all writers finish, before any requested destination is moved.
            for entry in entries:
                entry.temp_identity = self._require_owned_identity(
                    entry.temp_path,
                    label=f"completed {entry.artifact.label} temporary",
                )
                entry.publication_state = _PublicationState.TEMP_OWNED

            self.plan.validate()
            for entry in entries:
                final = entry.artifact.path
                try:
                    original_identity = self._path_identity(final)
                except FileNotFoundError:
                    continue
                if not self.plan.overwrite:
                    raise MediaError(
                        f"destination appeared before publication: {final}")
                backup, slot_identity = self._backup_slot(entry.artifact)
                entry.backup_path = backup
                entry.original_identity = original_identity
                entry.backup_slot_identity = slot_identity
                entry.backup_state = _BackupState.MOVING_ORIGINAL
                try:
                    self._replace(final, backup)
                except BaseException as exc:
                    reconciliation_errors = self._reconcile_backup_move(entry)
                    if reconciliation_errors:
                        exc.add_note(
                            "backup move reconciliation: "
                            + "; ".join(reconciliation_errors))
                    raise
                moved_identity = self._path_identity(backup)
                if moved_identity != original_identity:
                    failure = MediaError(
                        f"destination identity changed while creating "
                        f"rollback backup: {final}")
                    entry.backup_state = _BackupState.AMBIGUOUS
                    try:
                        self._replace(backup, final)
                    except BaseException as restore_exc:
                        failure.add_note(
                            f"foreign destination entry was retained at "
                            f"{backup}: {restore_exc!r}")
                    else:
                        self._clear_backup(entry, _BackupState.RESTORED)
                    raise failure
                entry.backup_state = _BackupState.ORIGINAL_HELD

            for entry in entries:
                entry.publication_state = _PublicationState.MOVING_TO_FINAL
                try:
                    self._replace(entry.temp_path, entry.artifact.path)
                except BaseException as exc:
                    reconciliation_errors = self._reconcile_publication(entry)
                    if reconciliation_errors:
                        exc.add_note(
                            "publication reconciliation: "
                            + "; ".join(reconciliation_errors))
                    raise
                self._verify_published(entry)
        except BaseException as exc:
            rollback_errors = self._rollback(entries)
            if rollback_errors:
                detail = "; ".join(rollback_errors)
                exc.add_note(f"artifact rollback also failed: {detail}")
                if isinstance(exc, PipelineError):
                    raise
                if isinstance(exc, OSError):
                    raise MediaError(
                        f"artifact publication failed ({exc}); rollback also "
                        f"failed: {detail}") from exc
                raise
            if isinstance(exc, PipelineError):
                raise
            if isinstance(exc, OSError):
                raise MediaError(
                    f"artifact publication failed: {exc}") from exc
            raise

        # Every destination has landed. This is the commit point: a later
        # interruption during backup cleanup must not roll back a complete set.
        self._committed = True
        cleanup_failures: list[BaseException] = []
        for entry in entries:
            if entry.sink is not None:
                entry.sink._mark_published()
            cleanup_failure = self._cleanup_committed_backup(entry)
            if cleanup_failure is not None:
                cleanup_failures.append(cleanup_failure)
                _log.warning(
                    "could not remove rollback backup %s: %s",
                    entry.backup_path, cleanup_failure)
        cleanup_failure = next(
            (failure for failure in cleanup_failures
             if not isinstance(failure, OSError)),
            None,
        )
        if cleanup_failure is not None:
            for failure in cleanup_failures:
                if failure is not cleanup_failure:
                    cleanup_failure.add_note(
                        f"another rollback backup cleanup failed: "
                        f"{failure!r}")
            raise cleanup_failure

    def discard(self) -> None:
        if self._closed or self._committed:
            return
        entries = self._ordered_entries()
        for entry in reversed(entries):
            if entry.sink is not None:
                entry.sink.discard()
        errors = self._rollback(entries)
        for error in errors:
            _log.error("artifact cleanup failed: %s", error)
        for parent in reversed(self._created_parents):
            with contextlib.suppress(BaseException):
                parent.rmdir()
        self._closed = True

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if not self._committed:
            self.discard()
        self._reservation.release()
        return False


@dataclasses.dataclass(frozen=True, slots=True)
class FileRunResult:
    path: Path | None
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
        overwrite: bool = False,
    ) -> None:
        from collections import deque

        from kinovsr.processors.specs import DType

        self._path = Path(path)
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
            encode_chroma=encode_chroma, overwrite=overwrite)
        try:
            self._retained: Any = deque()  # (seconds, uint8 (H,W,3) mx.array)
            self._src_time_base = float(source.spec.timeline.time_base)
            self._out_time_base = float(output_spec.timeline.time_base)
            self._src_layout = source.spec.frame.layout
            self._out_layout = output_spec.frame.layout
            self._out_w, self._out_h = out_geometry.width, out_geometry.height
            src_geometry = source.spec.frame.geometry
            # Nearest-neighbor gather maps, source -> output geometry. For an
            # integer scale s this is j // s, identical to mx.repeat.
            import mlx.core as mx

            with _media_operation(
                    "comparison map allocation failed for", self._path):
                self._ix = mx.array(
                    [(x * src_geometry.width) // self._out_w
                     for x in range(self._out_w)], dtype=mx.int32)
                self._iy = mx.array(
                    [(y * src_geometry.height) // self._out_h
                     for y in range(self._out_h)], dtype=mx.int32)
        except BaseException:
            self.sink.discard()
            raise

    def _to_uint8_rgb(self, payload: Any, layout: Layout) -> Any:
        import mlx.core as mx

        from kinovsr.media import pixel_buffers as _pb

        if layout is Layout.MLX_RGB_HWC:
            rgb = (payload[..., :3] if payload.dtype == mx.uint8
                   else mx.clip(payload[..., :3] * 255.0, 0,
                                255).astype(mx.uint8))
        else:
            with _media_operation(
                    "comparison source read failed for", self._path):
                rgb = _pb.read_pixel_buffer_rgb(payload)
        rgb = mx.contiguous(rgb)
        with _media_operation(
                "comparison materialization failed for", self._path):
            mx.eval(rgb)  # materialize: free decode graph, retain only uint8
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

            with _media_operation(
                    "comparison output read failed for", self._path):
                post_f = _pb.read_pixel_buffer_rgb(
                    unit.payload).astype(mx.float32) / 255.0
        payload = mx.concatenate([pre_f, post_f], axis=1)
        self.sink.append(unit.with_payload(payload))


def _save_frame_png(payload: Any, layout: Layout, out_dir: Path,
                    index: int, _pb: Any) -> None:
    """Write one unit's RGB to ``out_dir/frame_NNNNN.png``.

    An MLX unit carries a float32 [0,1] (H,W,3|4) array (uint8 already
    display-ready); a native unit carries a CVPixelBuffer read straight to
    uint8 RGB - the same two forms the harness dumped.
    """
    import mlx.core as mx

    from kinovsr.media.images import save_image

    frame_path = out_dir / f"frame_{index:05d}.png"
    if layout is Layout.MLX_RGB_HWC:
        rgb = (payload[..., :3] if payload.dtype == mx.uint8
               else mx.clip(payload[..., :3] * 255.0, 0,
                            255).astype(mx.uint8))
    else:
        with _media_operation("frame pixel-buffer read failed for", frame_path):
            rgb = _pb.read_pixel_buffer_rgb(payload)
    with _media_operation("frame image write failed for", frame_path):
        save_image(rgb, frame_path)


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
    snap_start: bool = False,
    gop_align: bool = False,
    gop_min_window: int = 16,
    gop_max_window: int = 96,
    cut_log: Path | str | None = None,
    skip_post_mp4: bool = False,
    noise_map_debug: bool = False,
    overwrite: bool = False,
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
    chunk_size = _validate_requested_chunk_size(chunk_size)
    t0 = time.perf_counter()
    plan = _ArtifactPlan.build(
        video=video,
        output=output,
        comparison=comparison,
        cut_log=cut_log,
        save_audio_sidecar=save_audio_sidecar,
        save_pre_frames=save_pre_frames,
        save_post_frames=save_post_frames,
        skip_post_mp4=skip_post_mp4,
        noise_map_debug=noise_map_debug,
        overwrite=overwrite,
    )
    # Inspect the complete source sample clock before entering the artifact
    # transaction. The file pipeline currently publishes CFR only; rejecting
    # VFR here prevents both silent retiming/frame loss and partial outputs.
    timing = _probe_cfr_timing(plan.input_path, reader)
    if audio:
        _validate_audio_origin(plan.input_path, reader, timing)
    from kinovsr.media.pixel_buffers import ci_cache_owner

    # The file owner outlives the inner session owner: auto-geometry may render
    # before the session exists, and comparison/final-frame work can render
    # after the chain has exhausted. Only this outer release performs the final
    # cache cleanup for the complete file operation.
    with ci_cache_owner(), _OutputTransaction(plan, settings) as transaction:
        return _run_file_reserved(
            config,
            plan=plan,
            transaction=transaction,
            settings=settings,
            t0=t0,
            reporter=reporter,
            layout=layout,
            start=start,
            end=end,
            max_frames=max_frames,
            max_output_frames=max_output_frames,
            max_output_seconds=max_output_seconds,
            audio=audio,
            audio_codec=audio_codec,
            quality=quality,
            chunk_size=chunk_size,
            source_color=source_color,
            source_range=source_range,
            encode_chroma=encode_chroma,
            snap_start=snap_start,
            gop_align=gop_align,
            gop_min_window=gop_min_window,
            gop_max_window=gop_max_window,
            reader=reader,
            timing=timing,
        )


def _run_file_reserved(
    config: dict,
    *,
    plan: _ArtifactPlan,
    transaction: _OutputTransaction,
    settings: Settings,
    t0: float,
    reporter: Any,
    layout: Layout,
    start: int,
    end: int | None,
    max_frames: int | None,
    max_output_frames: int | None,
    max_output_seconds: float | None,
    audio: bool,
    audio_codec: str,
    quality: float,
    chunk_size: int,
    source_color: str,
    source_range: str,
    encode_chroma: str,
    snap_start: bool,
    gop_align: bool,
    gop_min_window: int,
    gop_max_window: int,
    reader: Any,
    timing: VideoTiming | None,
) -> FileRunResult:
    """Execute while the complete output graph is reserved by ``transaction``."""
    from .session import open_pipeline

    video_path = plan.input_path
    # Keyframe-driven windowing (the harness's --snap-start / --gop-align).
    # snap-start MOVES the window start to the nearest keyframe; gop-align
    # KEEPS it, reads back to the enclosing keyframe as recurrence context
    # (processed, never output), and plans keyframe-anchored windows that
    # drive every schedule-capable stage through PipelineContext.windowing.
    windowing = None
    context_frames = 0
    if snap_start or gop_align:
        from kinovsr.media import video_reader as _native_vr

        vr = reader if reader is not None else _native_vr
        keyframe_kwargs = ({"timing": timing} if timing is not None else {})
        with _media_operation("keyframe inspection failed for", video_path):
            keyframes = vr.keyframe_display_indices(
                video_path, **keyframe_kwargs)
        if snap_start and keyframes:
            start = min(keyframes, key=lambda k: abs(k - start))
        if gop_align:
            from kinovsr.modeling.upscaler_base import plan_gop_windows

            if timing is not None:
                total = timing.sample_count
            else:
                with _media_operation(
                        "GOP video probe failed for", video_path):
                    _, _, _, total, _, _ = vr.probe_video(video_path)
            end_abs = total if end is None else min(end, total)
            enclosing = [k for k in keyframes if k <= start]
            read_start = max(enclosing) if enclosing else start
            context_frames = start - read_start
            kf_rel = sorted({k - read_start for k in keyframes
                             if read_start <= k < end_abs})
            try:
                windowing = plan_gop_windows(
                    kf_rel, end_abs - read_start,
                    gop_min_window, gop_max_window)
            except ValueError as exc:
                raise MediaError(f"invalid GOP window bounds: {exc}") from exc

    source = FileSource(
        video_path, layout=layout, start=start, end=end,
        max_frames=max_frames, chunk_size=chunk_size,
        source_color=source_color, source_range=source_range,
        context_frames=context_frames, reader=reader, timing=timing)
    # Probe-time auto geometry: rewrite bars="auto"/edges="auto" stage
    # tables into detected literal counts before the chain resolves
    # (sampling through the same reader the run decodes with).
    from .auto_geometry import resolve_auto_geometry, wants_auto_geometry

    if wants_auto_geometry(config):
        geometry = source.spec.frame.geometry
        auto_chunk_size = _effective_decode_chunk_size(
            chunk_size,
            geometry.width,
            geometry.height,
            Layout.CV_RGBA_HALF,
        )
        config = resolve_auto_geometry(
            config, video=video_path, vr=source._vr,
            pixel_aspect=geometry.pixel_aspect,
            chunk_size=auto_chunk_size)
    session = open_pipeline(
        config, source.spec, settings=settings, reporter=reporter,
        windowing=windowing,
        publication_origin_pts=0 if context_frames else None)
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
    audio_duration = (
        Fraction(max_output_frames) / out_cadence
        if max_output_frames is not None else None
    )
    track = (
        source.audio_track(max_duration=audio_duration)
        if audio else None
    )
    if plan.get("audio sidecar") is not None and track is not None:
        # A WAV sidecar of the (trimmed) carried track, beside the output.
        sidecar_temp = transaction.temp_file("audio sidecar")
        with _media_operation(
                "audio sidecar write failed for", plan.path("audio sidecar")):
            track.save_wav(sidecar_temp)
    # --skip-post-mp4 parity: process the chain (frame dumps, comparison,
    # sidecar still apply) without writing the post MP4.
    sink = None
    post_path = plan.get("post output")
    if post_path is not None:
        sink = FileSink(
            transaction.prepare_sink("post output"),
            session.output_spec,
            source=source,
            quality=quality,
            audio_track=track,
            audio_codec=audio_codec,
            encode_chroma=encode_chroma,
            overwrite=plan.overwrite,
        )
        transaction.register_sink("post output", sink)

    # --cut-log parity: detected cuts' source-frame indices, one per line,
    # truncated at run start like the harness. The cut_detect stage stamps
    # source_index on the HARD_CUT boundary and the scheduler carries it
    # downstream on the first post-cut unit.
    cut_log_path = None
    if plan.get("cut log") is not None:
        cut_log_path = transaction.temp_file("cut log")
        with _media_operation(
                "cut log initialization failed for", plan.path("cut log")):
            cut_log_path.write_text("", encoding="utf-8")

    # Optional per-frame PNG dumps: pre = the SOURCE frames (before the chain),
    # post = the encoded output frames (after it). Debug taps, not chain
    # stages - the PRE tap wraps the source iterator so it dumps exactly the
    # frames the session pulls, and the POST tap dumps each emitted unit.
    from kinovsr.media import pixel_buffers as _pbmod
    pre_dir = (transaction.temp_directory("pre-frame directory")
               if plan.get("pre-frame directory") is not None else None)
    post_dir = (transaction.temp_directory("post-frame directory")
                if plan.get("post-frame directory") is not None else None)
    source_units = source.units()
    if pre_dir is not None:
        src_layout = source.spec.frame.layout

        def _pre_tapped(units: Any) -> Any:
            for i, unit in enumerate(units):
                _save_frame_png(unit.payload, src_layout, pre_dir, i, _pbmod)
                yield unit
        source_units = _pre_tapped(source_units)
    post_layout = session.output_spec.frame.layout
    post_index = 0

    # The side-by-side comparison output (harness --comparison parity):
    # retained source frames pair with emitted output units into a second
    # sink at 2*out_w. auto chroma follows the POST output's own pick so
    # both files carry the same profile, as the harness's shared `profile`
    # variable did.
    tee = None
    comparison_path = plan.get("comparison output")
    if comparison_path is not None:
        comp_chroma = encode_chroma
        if (comp_chroma == "auto"
                and session.output_spec.frame.layout is Layout.CV_NV12):
            comp_chroma = "420"
        tee = _ComparisonTee(
            transaction.prepare_sink("comparison output"),
            session.output_spec, source, quality=quality,
            audio_track=track, audio_codec=audio_codec,
            encode_chroma=comp_chroma, overwrite=plan.overwrite)
        transaction.register_sink("comparison output", tee.sink)
        source_units = tee.tap(source_units)

    # A terminal native processor may write directly into the primary file
    # writer's adaptor pool, but only when its actual destination format and
    # geometry match. HQ/FRC RGBAHalf outputs deliberately reject the writer's
    # explicit-YUV pool and keep their processor-owned pool instead.
    pool_sink = sink if sink is not None else (tee.sink if tee is not None else None)
    if pool_sink is not None:
        writer = pool_sink.writer
        with _media_operation(
                "writer output-pool lookup failed for", pool_sink._final_path):
            pool = writer.adaptor.pixelBufferPool()
        if pool is not None:
            session._bind_terminal_output_pool(
                pool,
                writer.adaptor_pixel_format,
                writer.adaptor_width,
                writer.adaptor_height,
            )

    # A duration-preserving chain must end exactly at the source-window
    # duration. Interpolation's regenerated grid can round the final unit's
    # end past that (its natural grid-interval duration overshoots the last
    # source frame), which would leave muxed audio drifting; clamp the last
    # emitted unit's duration so total output duration matches the source.
    preserve_duration = (session.output_spec.timeline.duration_policy
                         is DurationPolicy.PRESERVED)
    source_end_ticks = grid_ticks(
        source.frame_count, source.source_cadence,
        session.output_spec.timeline.time_base)
    frames_out = 0
    pending: FrameUnit | None = None
    # retain_outputs=False: the sink consumes each unit into the encoder
    # synchronously, so outputs need not be copied for retention. A one-unit
    # holdback lets the final frame be clamped before it is written.
    with session, session.process(
            source_units, retain_outputs=False) as run:
        for unit in run:
            if cut_log_path is not None and unit.boundaries:
                with (
                    _media_operation(
                        "cut log append failed for", plan.path("cut log")),
                    cut_log_path.open("a", encoding="utf-8") as log,
                ):
                    for boundary in unit.boundaries:
                        if boundary.kind is BoundaryKind.HARD_CUT:
                            log.write(f"{boundary.source_index}\n")
            if unit.pts < 0:
                # gop-align context frames: recurrence warmup, never output.
                continue
            if post_dir is not None:
                _save_frame_png(unit.payload, post_layout, post_dir,
                                post_index, _pbmod)
                post_index += 1
            if pending is not None:
                if sink is not None:
                    sink.append(pending)
                if tee is not None:
                    tee.emit(pending)
                frames_out += 1
            pending = unit
            if (max_output_frames is not None
                    and frames_out + 1 == max_output_frames):
                break
        if pending is not None:
            if preserve_duration:
                clamped_end = min(
                    source_end_ticks, pending.pts + pending.duration)
                pending = pending.retimed(
                    pending.pts, max(1, clamped_end - pending.pts))
            if sink is not None:
                sink.append(pending)
            if tee is not None:
                tee.emit(pending)
            frames_out += 1
        # Stage state is still live here (drained, not yet closed).
        diagnostics = session.stage_diagnostics()
        debug_images = (
            session.stage_debug_images()
            if plan.get("noisemap debug image") is not None and sink is not None
            else {})

    for line in diagnostics:
        _log.info("%s", line)
    for suffix, image in debug_images.items():
        import mlx.core as mx

        from kinovsr.media.images import save_image

        label = f"{suffix} debug image"
        try:
            png = transaction.temp_file(label)
            final_png = plan.path(label)
        except KeyError as exc:
            raise MediaError(
                f"stage returned unplanned debug artifact {suffix!r}") from exc
        u8 = (mx.clip(image, 0, 1) * 255).astype(mx.uint8)
        debug_rgb = mx.stack([u8, u8, u8], axis=-1)
        with _media_operation("debug image write failed for", final_png):
            save_image(debug_rgb, png)
        _log.info("[noise-map-debug] %s written: %s", suffix, final_png)

    transaction.commit()
    return FileRunResult(
        path=post_path, frames_in=source.frame_count, frames_out=frames_out,
        output_spec=session.output_spec,
        elapsed_s=time.perf_counter() - t0,
        comparison_path=comparison_path)


__all__ = ["FileRunResult", "FileSink", "FileSource", "run_file"]
