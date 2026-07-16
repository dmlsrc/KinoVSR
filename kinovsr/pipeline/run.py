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
from kinovsr.media.errors import is_native_operation_error
from kinovsr.media.errors import media_operation as _media_operation
from kinovsr.media.filesystem import rename_exclusive
from kinovsr.media.timing import (
    SampleTable,
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
    VariableCadence,
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


def _close_created_fd(fd: int) -> None:
    """Close a newly-created fd exactly once.

    A failed ``close`` has ambiguous ownership: the kernel may already have
    released the descriptor and another thread may reuse its number. Retrying
    could therefore close an unrelated resource, even when ``fstat`` reports
    the same inode through a newly-opened descriptor.
    """
    os.close(fd)


def _process_umask() -> int:
    cur = os.umask(0)
    os.umask(cur)
    return cur


# Captured once at import (os.umask is process-global; toggling it at temp
# creation time would race other threads).
_UMASK = _process_umask()


def _widen_temp_mode(path: Path, *, directory: bool = False) -> None:
    """Give a mkstemp/mkdtemp temporary the mode a normal creation would get.

    mkstemp/mkdtemp deliberately create 0o600/0o700; artifacts published by
    rename keep that mode, which breaks group conventions for anything the
    materializer does not itself recreate (cut logs, sidecars, frame dirs).
    """
    target = (0o777 if directory else 0o666) & ~_UMASK
    with contextlib.suppress(OSError):
        path.chmod(target)


def _unlink_temporary(path: Path) -> None:
    """Make one best-effort removal without replacing an active failure."""
    try:
        path.unlink(missing_ok=True)
    except BaseException as exc:
        # An unlink may complete before reporting an interruption. A retry can
        # delete a same-name replacement, so an ambiguous private residue is
        # safer than a second destructive operation.
        _log.warning("could not remove temporary %s: %s", path, exc)


def _create_private_temp(
    destination: Path,
    label: str,
    *,
    directory: bool = False,
) -> Path:
    """Create the hidden sibling temporary for ``destination``.

    Owns the complete acquisition dance shared by the writer sink and the
    transaction temps: mkstemp/mkdtemp beside the final name, close the
    descriptor exactly once, remove the raw name if any later acquisition
    step fails (Path conversion itself may be the failing step), widen the
    umask-narrowed mode, and normalize OSError to MediaError.
    """
    raw: str | None = None
    fd: int | None = None
    try:
        if directory:
            raw = tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".partial")
        else:
            fd, raw = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".partial")
            owned_fd, fd = fd, None
            _close_created_fd(owned_fd)
        temp_path = Path(raw)
        _widen_temp_mode(temp_path, directory=directory)
        return temp_path
    except BaseException as exc:
        if fd is not None:
            try:
                _close_created_fd(fd)
            except BaseException as cleanup_exc:
                exc.add_note(
                    f"temporary descriptor cleanup also failed: "
                    f"{cleanup_exc!r}")
        if raw is not None:
            if directory:
                with contextlib.suppress(BaseException):
                    shutil.rmtree(raw)
            else:
                _unlink_created_raw(raw, exc)
        if isinstance(exc, OSError):
            raise MediaError(
                f"cannot create {label} temporary for "
                f"{destination}: {exc}") from exc
        raise


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


def _probe_timing(path: Path, reader: Any) -> SampleTable | VideoTiming | None:
    """Inspect the source's exact sample clock before output mutation.

    Both built-in readers return the full :class:`SampleTable`; sources
    whose clock is not one uniform grid are then CARRIED verbatim by the
    endpoints instead of rejected. Custom reader adapters written before
    this contract may expose only the legacy ``probe_video_timing`` view -
    that view cannot carry variable timing, so a variable verdict there
    still refuses - or omit the probe entirely (their ``probe_video``
    cadence remains supported).
    """
    module = _reader_module(reader)
    probe = getattr(module, "read_sample_table", None)
    if probe is not None:
        with _media_operation("failed to inspect source timing for", path):
            return probe(path)
    probe = getattr(module, "probe_video_timing", None)
    if probe is None:
        return None
    with _media_operation("failed to inspect source timing for", path):
        timing = probe(path)
    if timing.cadence is None:
        raise MediaError(
            f"{path.name}: this reader adapter reports variable timing but "
            f"exposes no sample table to carry it ({timing.sample_count} "
            f"display samples over {float(timing.duration):.6g}s); use a "
            f"built-in reader, or extend the adapter with read_sample_table")
    return timing


def _timing_view(
    timing: SampleTable | VideoTiming | None,
) -> VideoTiming | None:
    """The legacy CFR-or-variable view of whichever probe result exists."""
    return timing.timing() if isinstance(timing, SampleTable) else timing


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
        timing: SampleTable | VideoTiming | None = None,
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
            timing = _probe_timing(self.path, self._vr)
        self._table = timing if isinstance(timing, SampleTable) else None
        self._timing = _timing_view(timing)

        with _media_operation("failed to probe video source", self.path):
            width, height, fps, total, transform, pixel_aspect = (
                self._vr.probe_video(self.path))
        if self._timing is not None:
            total = self._timing.sample_count
        cadence = (self._timing.cadence if self._timing is not None
                   else _cadence(fps))
        # A table whose clock is not one uniform grid is carried verbatim:
        # the spec below declares an explicit per-frame timeline instead of
        # a cadence, and units() stamps each frame's rebased source tick.
        self._explicit_carry = self._table is not None and cadence is None
        if not self._explicit_carry:
            assert cadence is not None
            fps = float(cadence)
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
        # source_cadence is None only under explicit carry; nominal_cadence
        # always exists (encoder rate hint, progress display, legacy math).
        self.source_cadence = cadence
        self.nominal_cadence = (cadence if cadence is not None
                                else self._nominal_from_table())
        self.source_fps = (fps if cadence is not None
                           else float(self.nominal_cadence))

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
            timeline=(
                TimelineSpec(
                    time_base=self._explicit_time_base(),
                    cadence=VariableCadence.VFR,
                    nominal_cadence=self.nominal_cadence,
                )
                if self._explicit_carry
                else TimelineSpec(
                    time_base=Fraction(1, _pb.VIDEO_TIME_SCALE),
                    cadence=cadence,
                )
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

    def _nominal_from_table(self) -> Fraction:
        """Bookkeeping rate for an explicit timeline (hints and display)."""
        table = self._table
        assert table is not None
        if table.grid_cadence is not None:
            return table.grid_cadence
        if table.duration > 0:
            return Fraction(table.sample_count) / table.duration
        return Fraction(25)

    def _explicit_time_base(self) -> Fraction:
        """One exact integer tick base for every carried stamp.

        The lcm of the product base and each sample's pts/duration
        denominator: rebased stamps divide it exactly, so unit ticks stay
        integers. Falls back to the source denominators alone when the
        combined base overflows CMTime's 32-bit timescale.
        """
        import math

        table = self._table
        assert table is not None
        denominators = {table.first_pts.denominator}
        for sample in table.samples:
            denominators.add(sample.pts.denominator)
            if sample.duration is not None:
                denominators.add(sample.duration.denominator)
        base = math.lcm(*denominators)
        combined = math.lcm(base, self._pb.VIDEO_TIME_SCALE)
        limit = 2**31 - 1
        if combined <= limit:
            return Fraction(1, combined)
        if base <= limit:
            return Fraction(1, base)
        raise MediaError(
            f"{self.path.name}: source sample clock cannot be represented "
            f"exactly within CMTime's 32-bit timescale")

    def _window_tail_duration(self) -> Fraction:
        """The final window frame's display duration on the source clock."""
        table = self._table
        assert table is not None
        last = table.samples[self.end - 1]
        if last.duration is not None:
            return last.duration
        if table.grid_cadence is not None:
            return 1 / table.grid_cadence
        if self.end >= 2:
            delta = last.pts - table.samples[self.end - 2].pts
            if delta > 0:
                return delta
        return Fraction(1, 25)

    def _window_stamps(self, read_start: int) -> list[tuple[int, int]]:
        """(pts, duration) ticks for [read_start, end): the rebased source
        clock. Durations are display-until-next across the window (spanning
        any dropped-frame gaps); the final frame keeps its own coded
        duration, falling back to the grid interval, then the last delta.
        """
        table = self._table
        assert table is not None
        base = self.spec.timeline.time_base
        origin = table.samples[self.start].pts

        def ticks(value: Fraction) -> int:
            scaled = value / base
            # The base is the lcm of every stamp denominator by
            # construction, so carried values always land on it.
            assert scaled.denominator == 1
            return scaled.numerator

        starts = [ticks(sample.pts - origin)
                  for sample in table.samples[read_start:self.end]]
        durations = [starts[i + 1] - starts[i]
                     for i in range(len(starts) - 1)]
        durations.append(max(round(self._window_tail_duration() / base), 1))
        return list(zip(starts, durations, strict=True))

    def window_span(self, frames: int) -> Fraction:
        """Duration of the window's first ``frames`` frames, in seconds."""
        count = min(max(frames, 0), self.frame_count)
        if count <= 0:
            return Fraction()
        if not self._explicit_carry:
            return Fraction(count) / self.source_cadence
        table = self._table
        assert table is not None
        origin = table.samples[self.start].pts
        if count == self.frame_count:
            return (table.samples[self.end - 1].pts - origin
                    + self._window_tail_duration())
        return table.samples[self.start + count].pts - origin

    def window_end_ticks(self) -> int:
        """Exact end of the emitted window in the spec's tick base."""
        assert self._explicit_carry
        pts, duration = self._window_stamps(self.start)[-1]
        return pts + duration

    def units(self) -> Iterator[FrameUnit]:
        """Decode the window and yield timestamped units, one per frame.

        Context frames (gop-align) extend the read BEFORE the window and
        carry negative pts; the window's first frame stays at pts 0.
        Uniform sources stamp the cadence grid; explicit carry stamps each
        frame's rebased source tick, preserving the source clock (gaps
        included) exactly.
        """
        decode_format = getattr(self._pb, _DECODE_FORMATS[self.layout])
        to_mlx = self.layout is Layout.MLX_RGB_HWC
        read_start = self.start - self.context_frames
        timing_kwargs: dict[str, Any] = {}
        if self._table is not None and hasattr(self._vr, "read_sample_table"):
            timing_kwargs["table"] = self._table
        elif self._timing is not None:
            timing_kwargs["timing"] = self._timing
        stamps = (self._window_stamps(read_start)
                  if self._explicit_carry else None)
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
        emitted = 0
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
                if stamps is not None:
                    if emitted >= len(stamps):
                        raise MediaError(
                            f"{self.path.name}: decode produced more frames "
                            f"than the {len(stamps)}-frame sample-table "
                            f"window; refusing to mislabel carried "
                            f"timestamps")
                    pts, duration = stamps[emitted]
                else:
                    pts = self._grid_ticks(index)
                    duration = self._grid_ticks(index + 1) - pts
                yield FrameUnit(payload=payload, pts=pts, duration=duration)
                index += 1
                emitted += 1
        if stamps is not None and emitted != len(stamps):
            raise MediaError(
                f"{self.path.name}: decode produced {emitted} frames for the "
                f"{len(stamps)}-frame sample-table window; the decoder and "
                f"the sample table disagree, refusing to mislabel carried "
                f"timestamps")

    def audio_track(self, *, max_duration: Fraction | None = None) -> Any:
        """Open the source's bounded audio window, or None when absent.

        The audio slice and its output placement derive from the video
        window's REAL bounds on the source clock, so staggered track
        origins and --start/--end trims compose through one formula:
        source audio before the window origin is skipped (video-anchored
        sync), audio starting after it is placed late by exactly the
        carried offset. Exact rational boundaries and an optional
        output-duration cap resolve before either backend decodes PCM.
        """
        from kinovsr.media.audio import read_audio_track_from_video

        if self._explicit_carry:
            # Real window times off the carried source clock: index/cadence
            # arithmetic has no meaning on a non-uniform grid.
            table = self._table
            assert table is not None
            win0 = table.samples[self.start].pts
            win1 = (table.samples[self.end - 1].pts
                    + self._window_tail_duration())
        else:
            vfirst = (self._timing.first_pts if self._timing is not None
                      else Fraction(0))
            win0 = vfirst + Fraction(self.start) / self.source_cadence
            win1 = vfirst + Fraction(self.end) / self.source_cadence
        audio_first = Fraction(0)
        probe = getattr(self._vr, "probe_audio_timing", None)
        if probe is not None:
            with _media_operation(
                    "failed to inspect source audio timing for", self.path):
                audio_timing = probe(self.path)
            if audio_timing is not None:
                audio_first = audio_timing.first_pts
        slice_start = max(Fraction(0), win0 - audio_first)
        slice_end = win1 - audio_first
        place = max(Fraction(0), audio_first - win0)
        if slice_end <= slice_start:
            _log.info(
                "audio track lies entirely outside the video window (audio "
                "starts at %.6gs, window covers %.6gs-%.6gs); output will "
                "be silent",
                float(audio_first), float(win0), float(win1))
            return None
        try:
            track = read_audio_track_from_video(
                self.path,
                self._vr,
                start_sec=slice_start,
                end_sec=slice_end,
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
        if track is None:
            return None
        lead = win0 - audio_first
        if lead > 0:
            _log.info(
                "audio leads the video window by %.1f ms; skipped %s "
                "leading samples to preserve sync",
                float(lead) * 1000.0, round(lead * track.sample_rate))
        elif place > 0:
            track.placement_samples = round(place * track.sample_rate)
            _log.info(
                "audio starts %.1f ms after the video window origin; "
                "placed %s samples late to preserve sync",
                float(place) * 1000.0, track.placement_samples)
        return track


class FileSink:
    """Output endpoint: consume ``FrameUnit``s into the native writer.

    Audio carry/mux policy lives here (planning 04): a supplied audio
    track is muxed only when the chain preserved clip duration, which is
    exactly what keeps it synchronized. Each appended unit's PTS is
    verified against the writer's cadence grid - or, for a carried
    explicit timeline, against strict monotonicity - so a mistimed chain
    fails loudly instead of writing a silently-drifting file.
    """

    # Class defaults keep partially-constructed sinks (tests stub the
    # writer) on the uniform-grid verification path.
    _explicit_timeline = False
    _last_explicit_pts: int | None = None

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
        self._explicit_timeline = not isinstance(timeline.cadence, Fraction)
        self._last_explicit_pts: int | None = None
        if self._explicit_timeline:
            # An explicit timeline is legal only as a carried source clock:
            # the file source supplies the nominal rate (encoder hint) and
            # the unit-fraction tick base the carried stamps live on.
            if source is None or getattr(
                    source, "nominal_cadence", None) is None:
                raise MediaError(
                    "output endpoint requires a CFR cadence unless the "
                    "chain carries an explicit file-source timeline")
            if timeline.time_base.numerator != 1:
                raise MediaError(
                    f"explicit timeline tick base {timeline.time_base} is "
                    f"not a unit fraction; it cannot map to a CMTime "
                    f"timescale")
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
        try:
            self._final_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MediaError(
                f"cannot create writer parent for "
                f"{self._final_path}: {exc}") from exc
        self._temp_path = _create_private_temp(self._final_path, "writer")
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
                fps=(source.nominal_cadence if self._explicit_timeline
                     else timeline.cadence),
                time_scale=(int(timeline.time_base.denominator)
                            if self._explicit_timeline else None),
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
        if self._explicit_timeline:
            # A carried source clock has no index grid to verify against;
            # the invariant a mistimed chain would break is monotonicity.
            last = self._last_explicit_pts
            if last is not None and unit.pts <= last:
                raise PipelineError(
                    f"unit {self.writer.frame_count} arrived at pts "
                    f"{unit.pts} but the carried source timeline requires "
                    f"strictly increasing stamps (previous {last}); the "
                    f"chain broke its declared timeline")
            if unit.duration < 0:
                raise PipelineError(
                    f"unit {self.writer.frame_count} carries negative "
                    f"duration {unit.duration}")
            self._last_explicit_pts = unit.pts
        else:
            expected = self._grid_ticks(self.writer.frame_count)
            if abs(unit.pts - expected) > 1:
                raise PipelineError(
                    f"unit {self.writer.frame_count} arrived at pts "
                    f"{unit.pts} but the output cadence grid expects "
                    f"{expected}; the chain broke its declared timeline")
        # The chain's timeline is the validated one: stamp the unit's own
        # ticks (the writer's index grid quantizes NTSC-family rates).
        if self._is_mlx:
            self._validate_mlx_frame(unit.payload)
            if self._direct_mlx_encode:
                try:
                    self.writer.append_mlx_rgb(
                        unit.payload, pts_ticks=unit.pts,
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

# One process-lifetime descriptor for the single shared lock file, plus
# per-offset refcounts.  POSIX fcntl range locks are process-global (closing
# ANY descriptor on the file would drop every range this process holds, and a
# range unlock releases it for the whole process), so concurrent reservations
# in one process must share the fd and only unlock an offset when its last
# in-process holder releases.  In-process conflicts are refused by
# _RESERVED_NAMESPACES before fcntl is ever consulted; the kernel ranges
# exist to exclude OTHER processes and die with this one.
_NAMESPACE_LOCK_FDS: dict[Path, int] = {}
_NAMESPACE_RANGE_HOLDERS: dict[tuple[Path, int], int] = {}


def _namespace_offset(path: Path) -> int:
    """Stable 62-bit byte offset for one namespace's fcntl range."""
    digest = hashlib.sha256(os.fsencode(str(path))).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 62) - 1)


class _ArtifactReservation:
    """Cross-thread/process advisory reservation for a complete artifact set.

    All processes rendezvous on ONE shared lock file and take fcntl
    byte-range locks at per-namespace offsets: exclusive on each artifact
    path, shared on every ancestor (a run publishing INTO a directory
    conflicts with a run producing that directory).  Ranges cost no disk
    entries beyond the single file, and the kernel releases them the moment
    the holding process dies - no stale-lock recovery, no per-path litter.
    """

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
            self._lock_file = (
                settings.shared_temp_dir.expanduser().resolve(strict=False)
                / "kinovsr-namespaces.lock")
        except (OSError, RuntimeError) as exc:
            raise MediaError(
                f"cannot resolve artifact lock file under "
                f"{settings.shared_temp_dir}: {exc}") from exc
        self._locked_offsets: list[int] = []
        self._held = False

    def _lock_fd(self) -> int:
        # Called under _RESERVATION_GUARD.  One descriptor per lock file,
        # kept for the life of the process: closing it would drop every
        # range this process holds on that file.  Keyed by path because the
        # lock file derives from Settings and one process may use several.
        fd = _NAMESPACE_LOCK_FDS.get(self._lock_file)
        if fd is None:
            self._lock_file.parent.mkdir(parents=True, exist_ok=True)
            fd = self._open_shared_lock(self._lock_file)
            _NAMESPACE_LOCK_FDS[self._lock_file] = fd
        return fd

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
                fd = self._lock_fd()
                for path, mode in self._lock_requests:
                    key = (self._lock_file, _namespace_offset(path))
                    if _NAMESPACE_RANGE_HOLDERS.get(key, 0) == 0:
                        try:
                            fcntl.lockf(
                                fd, mode | fcntl.LOCK_NB, 1, key[1])
                        except OSError as exc:
                            if exc.errno in (errno.EACCES, errno.EAGAIN):
                                raise MediaError(
                                    f"destination hierarchy is already "
                                    f"reserved: {path}") from exc
                            raise
                    _NAMESPACE_RANGE_HOLDERS[key] = (
                        _NAMESPACE_RANGE_HOLDERS.get(key, 0) + 1)
                    self._locked_offsets.append(key[1])
            except BaseException as exc:
                self._release_offsets()
                if isinstance(exc, OSError):
                    raise MediaError(
                        f"cannot reserve artifact destinations via "
                        f"{self._lock_file}: {exc}") from exc
                raise
            _RESERVED_NAMESPACES.update(self._namespaces)
            self._held = True

    @staticmethod
    def _open_shared_lock(lock_path: Path) -> int:
        """Open (creating if needed) a lock file all accounts can reuse.

        Lock files persist in a shared temp root and are keyed by destination
        namespace, so every account writing into a shared output tree touches
        the same files.  0o666 (umask applies) keeps the next account from
        being locked out permanently; the one-time migration below unlinks a
        pre-existing owner-only file from before this fix (safe: the locks
        are advisory and only meaningful while a run holds them - a file we
        cannot even open is not one we hold).
        """
        flags = os.O_RDWR | os.O_CREAT
        try:
            fd = os.open(lock_path, flags, 0o666)
        except PermissionError:
            try:
                lock_path.unlink()
            except OSError as exc:
                raise MediaError(
                    f"cannot reset unreadable reservation lock {lock_path}: "
                    f"{exc}; remove it manually") from exc
            fd = os.open(lock_path, flags, 0o666)
        # umask narrows the create mode (0o666 -> 0o644 under 022), which
        # would lock the sibling account out again; widen when we own the
        # file (fchmod by a non-owner fails and is suppressed - an existing
        # shared file needs nothing).
        with contextlib.suppress(OSError):
            if (os.fstat(fd).st_mode & 0o666) != 0o666:
                os.fchmod(fd, 0o666)
        return fd

    def _release_offsets(self) -> None:
        # Called under _RESERVATION_GUARD.
        fd = _NAMESPACE_LOCK_FDS.get(self._lock_file)
        for offset in reversed(self._locked_offsets):
            key = (self._lock_file, offset)
            remaining = _NAMESPACE_RANGE_HOLDERS.get(key, 0) - 1
            if remaining > 0:
                _NAMESPACE_RANGE_HOLDERS[key] = remaining
                continue
            _NAMESPACE_RANGE_HOLDERS.pop(key, None)
            if fd is not None:
                with contextlib.suppress(OSError):
                    fcntl.lockf(fd, fcntl.LOCK_UN, 1, offset)
        self._locked_offsets.clear()

    def release(self) -> None:
        if not self._held:
            return
        with _RESERVATION_GUARD:
            self._release_offsets()
            _RESERVED_NAMESPACES.difference_update(self._namespaces)
            self._held = False


@dataclasses.dataclass(slots=True)
class _TransactionEntry:
    """One artifact's publication state.

    Rename atomicity is the oracle: ``publishing`` is set before the
    publish rename and ``published`` after it returns, so rollback can
    decide "landed" from the flags plus plain existence.  The cooperative
    threat model (doc 15) assumes no other actor mutates our temp, backup,
    or destination paths mid-transaction; inode-identity verification and
    quarantine machinery were removed with it.
    """

    artifact: _Artifact
    temp_path: Path
    sink: FileSink | None = None
    publishing: bool = False
    published: bool = False
    backup_path: Path | None = None


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

    def temp_file(self, label: str) -> Path:
        existing = self._entries.get(label)
        if existing is not None:
            return existing.temp_path
        artifact = self._artifact(label)
        if artifact.directory:
            raise RuntimeError(f"{label} is a directory artifact")
        self._ensure_parent(artifact.path)
        temp_path = _create_private_temp(artifact.path, artifact.label)
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
        temp_path = _create_private_temp(
            artifact.path, artifact.label, directory=True)
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
        """Reserve, then vacate, a unique rollback pathname.

        The raw name is made absent before ``Path`` conversion so a
        conversion failure cannot orphan a slot, and because the exclusive
        rename that follows needs an absent destination.
        """
        if artifact.directory:
            raw = tempfile.mkdtemp(
                dir=artifact.path.parent,
                prefix=f".{artifact.path.name}.", suffix=".rollback")
            os.rmdir(raw)  # noqa: PTH106
            return Path(raw)
        fd, raw = tempfile.mkstemp(
            dir=artifact.path.parent,
            prefix=f".{artifact.path.name}.", suffix=".rollback")
        try:
            os.unlink(raw)  # noqa: PTH108 - Path conversion follows removal
        finally:
            _close_created_fd(fd)
        return Path(raw)

    @staticmethod
    def _replace(source: Path, destination: Path) -> None:
        rename_exclusive(source, destination)

    @staticmethod
    def _restore_replace(backup: Path, destination: Path) -> None:
        # Plain replace, not exclusive: restore may legitimately overwrite
        # our own failed artifact left behind when its removal failed.
        backup.replace(destination)

    def _remove_published(self, entry: _TransactionEntry) -> list[str]:
        """Remove a landed destination during rollback.

        The publish rename is atomic: it landed iff it returned
        (``published``) or - for an interrupt racing the syscall's return -
        the destination exists while our temp is gone.  A publish rename
        that failed (for example EEXIST against a foreign file) never
        removes the destination, because the temp still exists.
        """
        if not entry.publishing:
            return []
        final = entry.artifact.path
        landed = entry.published or (
            os.path.lexists(final) and not os.path.lexists(entry.temp_path))
        if landed:
            try:
                self._remove(final)
            except BaseException as exc:
                # Flags stay set: a later rollback pass (exit discard after a
                # commit failure) makes one more attempt.
                return [f"remove published artifact {final}: {exc}"]
        # Neutralize the entry either way.  Rollback runs again from the exit
        # discard, and by then the world has legitimately changed (temp
        # removed, original restored) in a way the landed heuristic would
        # misread as "landed" and delete a preserved or restored file.
        entry.published = False
        entry.publishing = False
        return []

    def _restore_backup(self, entry: _TransactionEntry) -> list[str]:
        """Put a displaced original back during rollback.

        Plain replace, not exclusive rename: in the only cooperative state
        where the destination is occupied at restore time (removal of our
        own landed artifact failed mid-rollback), replacing it with the
        user's original is the desired recovery.  Restore only ever runs
        for destinations the user explicitly opted to overwrite.
        """
        backup = entry.backup_path
        if backup is None:
            return []
        if not os.path.lexists(backup):
            # The backup move never landed; the original never left.
            entry.backup_path = None
            return []
        try:
            self._restore_replace(backup, entry.artifact.path)
        except BaseException as exc:
            return [
                f"restore {entry.artifact.path}: {exc}; original retained "
                f"at {backup}",
            ]
        entry.backup_path = None
        return []

    def _remove_temporary(self, entry: _TransactionEntry) -> list[str]:
        if entry.published:
            return []
        try:
            self._remove(entry.temp_path)
        except BaseException as exc:
            return [f"remove temporary artifact {entry.temp_path}: {exc}"]
        return []

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
        backup = entry.backup_path
        if backup is None:
            return None
        try:
            self._remove(backup)
        except BaseException as exc:
            return exc
        entry.backup_path = None
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

            self.plan.validate()
            # Backup pass: displace every pre-existing destination into a
            # private slot (overwrite is an explicit opt-in; a destination
            # appearing without it fails the whole set before anything moves).
            for entry in entries:
                final = entry.artifact.path
                if not os.path.lexists(final):
                    continue
                if not self.plan.overwrite:
                    raise MediaError(
                        f"destination appeared before publication: {final}")
                backup = self._backup_slot(entry.artifact)
                entry.backup_path = backup
                self._replace(final, backup)

            # Publish pass: exclusive rename into the (now absent) final
            # names.  Rename atomicity is the oracle for rollback.
            for entry in entries:
                entry.publishing = True
                self._replace(entry.temp_path, entry.artifact.path)
                entry.published = True
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
    from kinovsr.native.frameworks import autorelease_pool

    # One pool for the endpoint's complete span. The writer drains each of
    # its own phases, but objects autoreleased on this thread between those
    # phases (source probing, chain setup, publication) land in no pool at
    # all in a run-loop-less host and pin their encoder sessions; after ~32
    # leaked sessions VideoToolbox drops to a software capability set
    # without the 4:2:2 profile and writer creation fails (batch-processing
    # 33 files through this endpoint in one process reproduced it).
    with autorelease_pool():
        chunk_size = _validate_requested_chunk_size(chunk_size)
        if save_audio_sidecar and not audio:
            # Without audio carry there is no track to dump; silently reserving
            # (and preflighting) a sidecar that never gets written is worse than
            # refusing up front.
            raise MediaError(
                "save_audio_sidecar requires audio; enable audio carry or drop "
                "the sidecar request")
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
        # Inspect the complete source sample clock before entering the
        # artifact transaction. Uniform grids keep the regenerated cadence
        # timeline; non-uniform tables are carried verbatim downstream.
        # Staggered audio origins need no gate: the audio window derives
        # from the video window's real bounds (FileSource.audio_track).
        timing = _probe_timing(plan.input_path, reader)
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
    table = timing if isinstance(timing, SampleTable) else None
    timing_view = _timing_view(timing)
    if snap_start or gop_align:
        if table is not None and any(
                sample.is_sync is not None for sample in table.samples):
            # The walk that built the table read the real sync flags; table
            # positions are exact under jitter, gaps, and VFR, where the
            # legacy cadence-grid mapping desyncs from sample ordinals.
            keyframes = list(table.keyframe_indices) or [0]
        else:
            from kinovsr.media import video_reader as _native_vr

            vr = reader if reader is not None else _native_vr
            keyframe_kwargs = ({"timing": timing_view}
                               if timing_view is not None else {})
            with _media_operation(
                    "keyframe inspection failed for", video_path):
                keyframes = vr.keyframe_display_indices(
                    video_path, **keyframe_kwargs)
        if snap_start and keyframes:
            start = min(keyframes, key=lambda k: abs(k - start))
        if gop_align:
            from kinovsr.modeling.upscaler_base import plan_gop_windows

            if timing_view is not None:
                total = timing_view.sample_count
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
        if not isinstance(out_cadence, Fraction):
            raise MediaError(
                "a time-form output cap needs a constant output cadence; "
                "this chain carries the source's variable timeline - give "
                "the cap in frames (e.g. 500 or 500f)")
        max_output_frames = round(max_output_seconds * out_cadence)
    if max_output_frames is not None and max_output_frames < 1:
        raise MediaError(
            f"the output cap must be at least one frame; got "
            f"{max_output_frames}")
    audio_duration = None
    if max_output_frames is not None:
        audio_duration = (
            Fraction(max_output_frames) / out_cadence
            if isinstance(out_cadence, Fraction)
            else source.window_span(max_output_frames))
    track = (
        source.audio_track(max_duration=audio_duration)
        if audio else None
    )
    if plan.get("audio sidecar") is not None:
        if track is not None:
            # A WAV sidecar of the (trimmed) carried track, beside the output.
            sidecar_temp = transaction.temp_file("audio sidecar")
            with _media_operation(
                    "audio sidecar write failed for",
                    plan.path("audio sidecar")):
                track.save_wav(sidecar_temp)
        else:
            _log.warning(
                "source has no audio track; no audio sidecar written")
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
    if source.source_cadence is not None:
        source_end_ticks = grid_ticks(
            source.frame_count, source.source_cadence,
            session.output_spec.timeline.time_base)
    else:
        # Explicit carry is 1:1 (cadence-changing stages refuse a variable
        # timeline), so the source window's exact end IS the output end.
        source_end_ticks = source.window_end_ticks()
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
