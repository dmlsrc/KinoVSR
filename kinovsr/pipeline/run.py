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
import errno
import fcntl
import hashlib
import logging
import os
import shutil
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
        raise MediaError(f"failed to inspect source timing for {path}: {exc}") from exc
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
        raise MediaError(
            f"failed to inspect source audio timing for {path}: {exc}") from exc
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

        width, height, fps, total, transform, pixel_aspect = (
            self._vr.probe_video(self.path))
        cadence = timing.cadence if timing is not None else _cadence(fps)
        assert cadence is not None
        if timing is not None:
            total = timing.sample_count
        fps = float(cadence)
        self._timing = timing
        from kinovsr.media import color as _color

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
            chunks = self._vr.iter_forced_color_chunks(
                self.path, decode_format, self.resolved_color[2],
                self._src_color["full_range"], chunk_size=self.chunk_size,
                start_frame=read_start, end_frame=self.end,
                reinterpret_full_range=self.resolved_color[3],
                **timing_kwargs)
        else:
            chunks = self._vr.iter_video_buffer_chunks(
                self.path, decode_format, chunk_size=self.chunk_size,
                start_frame=read_start, end_frame=self.end,
                **timing_kwargs)
        index = -self.context_frames
        for chunk in chunks:
            for buffer in chunk:
                payload = (self._pb.read_buffer_rgb_f32(buffer)
                           if to_mlx else buffer)
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
        except RuntimeError as exc:
            raise MediaError(
                f"failed to open bounded audio from {self.path}: {exc}") from exc


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
        self._overwrite = overwrite
        if self._final_path.exists() and not overwrite:
            raise MediaError(
                f"destination already exists: {self._final_path}; pass "
                f"overwrite=True to replace it")
        self._final_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=self._final_path.parent,
            prefix=f".{self._final_path.name}.", suffix=".partial")
        os.close(fd)
        self._temp_path = Path(tmp)
        self._published = False
        self._finalized = False
        self._discarded = False

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

            self._pool = None
            if self._is_mlx:
                self._pool = _pb.make_pool_from_attrs({
                    "PixelFormatType": _pb.PIX_RGBAHALF,
                    "Width": geometry.width, "Height": geometry.height,
                    "IOSurfaceProperties": {},
                    "MetalCompatibility": True,
                })
        except BaseException:
            writer = getattr(self, "writer", None)
            if writer is not None:
                with contextlib.suppress(BaseException):
                    writer.cancel()
            with contextlib.suppress(Exception):
                self._temp_path.unlink()
            raise

    def _grid_ticks(self, index: int) -> int:
        timeline = self.spec.timeline
        return grid_ticks(index, timeline.cadence, timeline.time_base)

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
        try:
            self.writer.append(payload, pts_ticks=unit.pts,
                               duration_ticks=unit.duration or None)
        except Exception as exc:
            raise MediaError(
                f"writer append failed for {self._final_path}: {exc}") from exc

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
            if isinstance(exc, Exception):
                raise MediaError(
                    f"writer finalization failed for {self._final_path}: "
                    f"{exc}") from exc
            raise
        self._finalized = True

    def _mark_published(self) -> None:
        self._published = True

    def finish(self) -> Path:
        """Finalize the encode and publish it atomically to the requested
        output path (Path.replace is atomic on the same filesystem). On ANY
        failure - a writer-finalization error, or a rename that cannot land
        (e.g. the output path is an existing directory, or a full disk) -
        the partial temp is removed and the requested output is left
        untouched."""
        self.finalize()
        try:
            if self._final_path.exists() and not self._overwrite:
                raise MediaError(
                    f"destination appeared before publication: "
                    f"{self._final_path}")
            self._temp_path.replace(self._final_path)
        except BaseException:
            with contextlib.suppress(Exception):
                self._temp_path.unlink()
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
            with contextlib.suppress(Exception):
                self._temp_path.unlink()


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
        self._lock_root = (
            settings.shared_temp_dir.expanduser().resolve(strict=False)
            / "kinovsr-artifact-locks")
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


@dataclasses.dataclass(slots=True)
class _TransactionEntry:
    artifact: _Artifact
    temp_path: Path
    sink: FileSink | None = None
    backup_path: Path | None = None
    publishing: bool = False
    published: bool = False


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

    def _ensure_parent(self, path: Path) -> None:
        missing: list[Path] = []
        parent = path.parent
        while not parent.exists() and parent != parent.parent:
            missing.append(parent)
            parent = parent.parent
        path.parent.mkdir(parents=True, exist_ok=True)
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
        self._entries[label] = _TransactionEntry(
            self._artifact(label), sink._temp_path, sink=sink)

    def temp_file(self, label: str) -> Path:
        existing = self._entries.get(label)
        if existing is not None:
            return existing.temp_path
        artifact = self._artifact(label)
        if artifact.directory:
            raise RuntimeError(f"{label} is a directory artifact")
        self._ensure_parent(artifact.path)
        fd, raw = tempfile.mkstemp(
            dir=artifact.path.parent,
            prefix=f".{artifact.path.name}.", suffix=".partial")
        os.close(fd)
        temp_path = Path(raw)
        self._entries[label] = _TransactionEntry(artifact, temp_path)
        return temp_path

    def temp_directory(self, label: str) -> Path:
        existing = self._entries.get(label)
        if existing is not None:
            return existing.temp_path
        artifact = self._artifact(label)
        if not artifact.directory:
            raise RuntimeError(f"{label} is a file artifact")
        self._ensure_parent(artifact.path)
        temp_path = Path(tempfile.mkdtemp(
            dir=artifact.path.parent,
            prefix=f".{artifact.path.name}.", suffix=".partial"))
        self._entries[label] = _TransactionEntry(artifact, temp_path)
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

    @staticmethod
    def _backup_slot(artifact: _Artifact) -> Path:
        if artifact.directory:
            raw = tempfile.mkdtemp(
                dir=artifact.path.parent,
                prefix=f".{artifact.path.name}.", suffix=".rollback")
            backup = Path(raw)
            backup.rmdir()
            return backup
        fd, raw = tempfile.mkstemp(
            dir=artifact.path.parent,
            prefix=f".{artifact.path.name}.", suffix=".rollback")
        os.close(fd)
        return Path(raw)

    @staticmethod
    def _replace(source: Path, destination: Path) -> None:
        source.replace(destination)

    def _rollback(self, entries: list[_TransactionEntry]) -> list[str]:
        errors: list[str] = []
        for entry in reversed(entries):
            landed = entry.published or (
                entry.publishing
                and entry.artifact.path.exists()
                and not entry.temp_path.exists())
            if landed:
                try:
                    self._remove(entry.artifact.path)
                except BaseException as exc:  # preserve the original failure
                    errors.append(
                        f"remove {entry.artifact.path}: {exc}")
                entry.published = False
            entry.publishing = False
        for entry in reversed(entries):
            backup = entry.backup_path
            if backup is not None and backup.exists():
                try:
                    self._replace(backup, entry.artifact.path)
                except BaseException as exc:
                    errors.append(
                        f"restore {entry.artifact.path}: {exc}")
        for entry in reversed(entries):
            if entry.temp_path.exists():
                try:
                    self._remove(entry.temp_path)
                except BaseException as exc:
                    errors.append(f"remove {entry.temp_path}: {exc}")
        return errors

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

            self.plan.validate()
            for entry in entries:
                final = entry.artifact.path
                if final.exists():
                    if not self.plan.overwrite:
                        raise MediaError(
                            f"destination appeared before publication: {final}")
                    backup = self._backup_slot(entry.artifact)
                    entry.backup_path = backup
                    try:
                        self._replace(final, backup)
                    except BaseException:
                        if final.exists():
                            with contextlib.suppress(BaseException):
                                self._remove(backup)
                            entry.backup_path = None
                        elif backup.exists():
                            with contextlib.suppress(BaseException):
                                self._replace(backup, final)
                            if not backup.exists():
                                entry.backup_path = None
                        raise

            for entry in entries:
                entry.publishing = True
                self._replace(entry.temp_path, entry.artifact.path)
                entry.published = True
                entry.publishing = False
        except BaseException as exc:
            rollback_errors = self._rollback(entries)
            if rollback_errors:
                detail = "; ".join(rollback_errors)
                if not isinstance(exc, Exception):
                    exc.add_note(f"artifact rollback also failed: {detail}")
                    raise
                raise MediaError(
                    f"artifact publication failed ({exc}); rollback also "
                    f"failed: {detail}") from exc
            if isinstance(exc, PipelineError):
                raise
            if isinstance(exc, Exception):
                raise MediaError(
                    f"artifact publication failed: {exc}") from exc
            raise

        # Every destination has landed. This is the commit point: a later
        # interruption during backup cleanup must not roll back a complete set.
        self._committed = True
        cleanup_interrupt: BaseException | None = None
        for entry in entries:
            if entry.sink is not None:
                entry.sink._mark_published()
            backup = entry.backup_path
            if backup is not None and backup.exists():
                try:
                    self._remove(backup)
                except BaseException as exc:
                    _log.warning(
                        "could not remove rollback backup %s: %s",
                        backup, exc)
                    if not isinstance(exc, Exception):
                        with contextlib.suppress(BaseException):
                            self._remove(backup)
                        cleanup_interrupt = cleanup_interrupt or exc
        if cleanup_interrupt is not None:
            raise cleanup_interrupt

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
    with _OutputTransaction(plan, settings) as transaction:
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
        keyframes = vr.keyframe_display_indices(
            video_path, **keyframe_kwargs)
        if snap_start and keyframes:
            start = min(keyframes, key=lambda k: abs(k - start))
        if gop_align:
            from kinovsr.modeling.upscaler_base import plan_gop_windows

            if timing is not None:
                total = timing.sample_count
            else:
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
        track.save_wav(transaction.temp_file("audio sidecar"))
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
                with cut_log_path.open("a", encoding="utf-8") as log:
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
        save_image(mx.stack([u8, u8, u8], axis=-1), png)
        _log.info("[noise-map-debug] %s written: %s", suffix, final_png)

    transaction.commit()
    return FileRunResult(
        path=post_path, frames_in=source.frame_count, frames_out=frames_out,
        output_spec=session.output_spec,
        elapsed_s=time.perf_counter() - t0,
        comparison_path=comparison_path)


__all__ = ["FileRunResult", "FileSink", "FileSource", "run_file"]
