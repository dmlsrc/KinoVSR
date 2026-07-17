"""ffmpeg (PyAV) compatibility reader for containers AVFoundation cannot open.

The native reader (video_reader.py, AVFoundation) is the default and preferred
path: zero-copy IOSurface decode, exact sync-sample metadata, full precision.
This module is the read-side analog of the ffmpeg encode fallback -- for the
esoteric containers and codecs AVFoundation refuses (MKV, VP9, AVI-era
material), it mirrors the native reader's exact surface so the entire
downstream chain (VSR sessions, denoisers, noise maps, gop-align, the native
encoder) runs unchanged:

    probe_video / probe_color / keyframe_display_indices
    iter_video_buffer_chunks / iter_forced_color_chunks
    read_audio_track_window / read_audio_track

Frames exit as IOSurface-backed CVPixelBuffers built with the same helpers the
native path uses, decoded via libavcodec and converted by libswscale honoring
(or, for the forced-color path, overriding) the stream's color tags. This
module is imported only when the ffmpeg reader is selected; install the
`ffmpeg` extra to enable it. No numpy: plane bytes cross into MLX through the
buffer protocol.

Rotation/flip display matrices propagate as CGAffineTransforms matching the
AVFoundation preferredTransform convention (verified component-wise against
the native probe at 90/180/270). >8-bit sources are read through rgb48le
(full precision kept through the fp16 pixel buffer).
"""
from __future__ import annotations

import bisect
import contextlib
import logging
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any, NoReturn

import av
import mlx.core as mx

from . import pixel_buffers as _pb
from .chunks import validate_decode_chunk_size
from .pixel_buffers import PIX_BGRA, PIX_RGBAHALF
from .timing import (
    AudioTiming,
    EpochUnwrapper,
    SampleTable,
    SampleTiming,
    VideoTiming,
    analyze_sample_table,
)

_log = logging.getLogger(__name__)


def _raise_ffmpeg_operation(
    operation: str, path: Path | str, exc: Exception,
) -> NoReturn:
    """Normalize a PyAV failure to RuntimeError at the module boundary.

    PyAV maps FFmpeg demux/decode failures onto ValueError subclasses
    (av.error.InvalidDataError is a builtins.ValueError), which upstream
    boundaries classify as programmer errors.  Corrupt or truncated media
    is operational: convert here so av's taxonomy stays private to this
    module.  OSError-flavored av errors (missing file, permission) already
    carry operational typing and pass through unchanged.
    """
    if isinstance(exc, OSError):
        raise exc
    raise RuntimeError(f"{operation} {path}: {exc}") from exc


def _open_container(path: Path | str, operation: str) -> Any:
    try:
        return av.open(str(path))
    except av.FFmpegError as exc:
        _raise_ffmpeg_operation(operation, path, exc)


def _open_video(path: Path):
    container = _open_container(path, "cannot open video in")
    streams = [s for s in container.streams if s.type == "video"]
    if not streams:
        container.close()
        raise RuntimeError(f"No video track in {path}")
    return container, streams[0]


def _fps(stream: Any) -> float:
    r = stream.average_rate or stream.guessed_rate or stream.base_rate
    return float(r) if r else 0.0


def _pts_index(
    pts: int,
    stream: Any,
    fps: float,
    *,
    origin: Fraction | None = None,
) -> int:
    """Display index of a pts, 0-based at the stream's first displayed frame.

    Some containers (FLV, ASF) start their display timeline at a nonzero pts;
    subtracting stream.start_time keeps keyframe indices and trim windows
    aligned with decode-order frame counting.
    """
    if origin is None:
        start = stream.start_time or 0
        seconds = Fraction(pts - start) * Fraction(stream.time_base)
    else:
        seconds = Fraction(pts) * Fraction(stream.time_base) - origin
    return int(round(seconds * fps))


def _display_transform(container: Any, vs: Any) -> Any:
    """The stream's display matrix as a CGAffineTransform (identity when absent).

    libavformat keeps the rotation/flip as DISPLAYMATRIX side data, which
    libavcodec propagates onto decoded frames -- so decode one frame and read
    it there (streams expose only setters in PyAV). The 3x3 matrix is row-major
    with the affine part in 16.16 fixed point; its top-left 2x2 + translation
    map directly onto CGAffineTransform's (a, b, c, d, tx, ty), the same
    convention as AVFoundation's preferredTransform (verified component-wise
    against the native probe on a rotated file).
    """
    import struct

    import Quartz
    try:
        for frame in container.decode(vs):
            for sd in getattr(frame, "side_data", []):
                if "DISPLAYMATRIX" in str(sd.type):
                    m = struct.unpack("<9i", bytes(sd))
                    s = 1.0 / 65536.0
                    return Quartz.CGAffineTransformMake(
                        m[0] * s, m[1] * s, m[3] * s, m[4] * s, m[6] * s, m[7] * s)
            break   # only the first frame carries what the stream declares
    except Exception:
        pass
    return Quartz.CGAffineTransformIdentity


# ---------------------------------------------------------------- probes
def probe_video(path: Path) -> tuple[int, int, float, int, Any, tuple[int, int] | None]:
    """(width, height, fps, n_frames, transform, pixel_aspect); mirrors the
    native probe. transform carries the stream's display matrix (rotation /
    flip) mapped to the AVFoundation preferredTransform convention."""
    container, vs = _open_video(path)
    try:
        w, h = int(vs.codec_context.width), int(vs.codec_context.height)
        fps = _fps(vs)
        n = int(vs.frames or 0)
        if n <= 0 and vs.duration is not None and fps > 0:
            n = int(round(float(vs.duration * vs.time_base) * fps))
        if n <= 0 and container.duration is not None and fps > 0:
            n = int(round(container.duration / 1_000_000 * fps))
        sar = vs.sample_aspect_ratio
        pixel_aspect = None
        if sar and sar.numerator > 0 and sar.denominator > 0 \
                and sar.numerator != sar.denominator:
            pixel_aspect = (int(sar.numerator), int(sar.denominator))
        transform = _display_transform(container, vs)
        return w, h, fps, n, transform, pixel_aspect
    except av.FFmpegError as exc:
        _raise_ffmpeg_operation("probing video in", path, exc)
    finally:
        container.close()


def read_sample_table(path: Path) -> SampleTable:
    """One metadata-only demux walk of the video stream's packets.

    Collects every packet's exact presentation time and duration plus its
    keyframe flag and coded size, then classifies the timing as one grid,
    gapped, or variable. No decode; mirrors the native reader's walk.
    """
    container, vs = _open_video(path)
    try:
        time_base = Fraction(vs.time_base)
        samples: list[SampleTiming] = []
        unwrapper = EpochUnwrapper()
        for packet in container.demux(vs):
            if packet.pts is None or packet.size <= 0:
                continue
            pts = unwrapper.push(Fraction(int(packet.pts)) * time_base)
            duration = (Fraction(int(packet.duration)) * time_base
                        if packet.duration and packet.duration > 0 else None)
            samples.append(SampleTiming(
                pts=pts,
                duration=duration,
                is_sync=bool(packet.is_keyframe),
                coded_size=int(packet.size),
            ))
        if unwrapper.resets:
            _log.info(
                "%s: unwrapped %d mid-file timestamp reset(s) into one "
                "declared monotonic clock", path.name, unwrapper.resets)
        try:
            return analyze_sample_table(
                samples, nominal_cadence=_fps(vs), source_tick=time_base,
                epoch_resets=unwrapper.resets)
        except ValueError as exc:
            raise RuntimeError(f"{path.name}: {exc}") from exc
    except av.FFmpegError as exc:
        _raise_ffmpeg_operation("probing video timing in", path, exc)
    finally:
        container.close()


def probe_video_timing(path: Path) -> VideoTiming:
    """Classify exact packet presentation timestamps without decoding.

    The legacy CFR-or-variable view over :func:`read_sample_table`'s walk.
    """
    return read_sample_table(path).timing()


# libav color tags -> the CoreVideo token strings the native probe_color emits
# (see color.py; the resolver matches on these exact strings).
_PRIMARIES = {
    1: "ITU_R_709_2",     # bt709
    5: "SMPTE_C",         # bt470bg (601 PAL primaries; closest CV token)
    6: "SMPTE_C",         # smpte170m
    7: "SMPTE_C",         # smpte240m
    9: "ITU_R_2020",      # bt2020
}
_MATRIX = {
    1: "ITU_R_709_2",     # bt709
    5: "ITU_R_601_4",     # bt470bg
    6: "ITU_R_601_4",     # smpte170m
    7: "ITU_R_601_4",     # smpte240m
    9: "ITU_R_2020",      # bt2020nc
    10: "ITU_R_2020",     # bt2020c
}
_TRANSFER = {
    1: "ITU_R_709_2",     # bt709
    6: "ITU_R_709_2",     # smpte170m (SDR gamma family)
    14: "ITU_R_709_2",    # bt2020-10 (SDR gamma family)
    15: "ITU_R_709_2",    # bt2020-12
    13: "sRGB",           # iec61966-2-1
}


def probe_color(path: Path) -> dict:
    """Source color tags mapped to the native probe_color dict shape."""
    container, vs = _open_video(path)
    try:
        cc = vs.codec_context
        prim = _PRIMARIES.get(int(cc.color_primaries or 2))
        mat = _MATRIX.get(int(cc.colorspace or 2))
        trans = _TRANSFER.get(int(cc.color_trc or 2))
        # color_range: 2 == JPEG/full
        full = int(cc.color_range or 0) == 2
        return {
            "primaries": prim,
            "transfer": trans,
            "matrix": mat,
            "full_range": full,
            # some codecs (VP9 in MKV) carry the matrix in-bitstream without
            # container primaries; the matrix is what steers the YUV read, so
            # either counts as tagged.
            "tagged": prim is not None or mat is not None,
        }
    finally:
        container.close()


def keyframe_display_indices(
    path: Path,
    *,
    timing: VideoTiming | None = None,
) -> list[int]:
    """Display-order keyframe indices from packet metadata (no decode).

    Table positions from the consolidated demux walk
    (:func:`read_sample_table`) - exact under jitter, gaps, and variable
    clocks, where the retired cadence-grid mapping desynced from sample
    ordinals after any dropped frame. ``timing`` is accepted for
    adapter-call compatibility and unused. All-intra codecs report every
    frame; codecs with no keyframe concept degrade to [0], and the gop
    planner handles both shapes.
    """
    del timing
    return list(read_sample_table(path).keyframe_indices) or [0]


def coded_frame_sizes(path: Path) -> list[int]:
    """Per-frame coded sizes in DISPLAY order (bytes), no decode.

    One entry per display SAMPLE from the consolidated walk
    (:func:`read_sample_table`); unknown sizes read 0. The retired
    grid-slot version inserted phantom zero entries at dropped-frame
    slots, misaligning sizes with decoded frame ordinals on gapped
    sources. Coded size is a LAST-GENERATION signal: a generous
    re-encode of damaged footage reads large. Returns [] when the
    stream cannot be walked.
    """
    try:
        table = read_sample_table(path)
    except RuntimeError:
        return []
    return [sample.coded_size or 0 for sample in table.samples]


# pix_fmt name -> (bit depth, chroma subsampling) for the descriptor.
_PIX_FMT_TRAITS = {
    "yuv420p": (8, "420"), "yuvj420p": (8, "420"), "nv12": (8, "420"),
    "yuv420p10le": (10, "420"), "yuv420p12le": (12, "420"),
    "yuv422p": (8, "422"), "yuvj422p": (8, "422"),
    "yuv422p10le": (10, "422"), "yuv422p12le": (12, "422"),
    "yuv444p": (8, "444"), "yuvj444p": (8, "444"),
    "yuv444p10le": (10, "444"), "yuv444p12le": (12, "444"),
}


def probe_stream_descriptor(path: Path) -> dict:
    """Coarse raw-stream identity for artifact-appropriate processing.

    ``{"codec": name, "profile": str | None, "bit_depth": int | None,
    "chroma": "420" | "422" | "444" | None}`` with ``None`` where the
    stream does not say. An MPEG-2 VOB and an H.264 web rip do not
    deserve the same restoration defaults; this is the cheap signal that
    tells them apart.
    """
    container, vs = _open_video(path)
    try:
        cc = vs.codec_context
        depth, chroma = _PIX_FMT_TRAITS.get(
            str(cc.pix_fmt or ""), (None, None))
        profile = cc.profile if isinstance(cc.profile, str) else None
        return {"codec": str(cc.name), "profile": profile,
                "bit_depth": depth, "chroma": chroma}
    finally:
        container.close()


# ---------------------------------------------------------------- frames
def _plane_to_mx(frame: Any, bytes_per_px: int, width_els: int) -> Any:
    """Packed plane 0 -> mx array (h, width_els), stride-cropped, no numpy."""
    plane = frame.planes[0]
    h = int(frame.height)
    stride = int(plane.line_size)
    raw = mx.array(memoryview(plane).cast("B"))
    if bytes_per_px == 2:
        return raw.view(mx.uint16).reshape(h, stride // 2)[:, :width_els]
    return raw.reshape(h, stride)[:, :width_els]


def _frame_to_buffer(frame: Any, out_format: int, attrs: dict,
                     reformat_kwargs: dict) -> Any:
    """Decode one av.VideoFrame into a CVPixelBuffer of out_format."""
    w, h = int(frame.width), int(frame.height)
    # >8-bit sources keep precision through rgb48le; 8-bit through rgb24.
    deep = "p10" in frame.format.name or "p12" in frame.format.name \
        or frame.format.name in ("yuv420p10le", "yuv422p10le", "yuv444p10le")
    if deep:
        rf = frame.reformat(format="rgb48le", **reformat_kwargs)
        flat = _plane_to_mx(rf, 2, w * 3)
        rgb = flat.reshape(h, w, 3).astype(mx.float32) * (1.0 / 65535.0)
    else:
        rf = frame.reformat(format="rgb24", **reformat_kwargs)
        flat = _plane_to_mx(rf, 1, w * 3)
        rgb = flat.reshape(h, w, 3).astype(mx.float32) * (1.0 / 255.0)

    pb = _pb.make_pixel_buffer_from_attrs(w, h, attrs)
    if out_format == PIX_RGBAHALF:
        rgba = mx.concatenate(
            [rgb.astype(mx.float16), mx.ones((h, w, 1), dtype=mx.float16)], axis=-1)
        mx.eval(rgba)
        _pb.write_fp16_rgba(rgba, pb)
    else:   # PIX_BGRA
        u8 = mx.clip(rgb * 255.0 + 0.5, 0, 255).astype(mx.uint8)
        bgra = mx.concatenate(
            [u8[..., 2:3], u8[..., 1:2], u8[..., 0:1],
             mx.full((h, w, 1), 255, dtype=mx.uint8)], axis=-1)
        mx.eval(bgra)
        _write_packed(bgra, pb, bytes_per_px=4)
    return pb


def _write_packed(frame: Any, pb: Any, bytes_per_px: int) -> None:
    """Memcpy a packed (H,W,C) frame into a CVPixelBuffer plane, honoring the
    destination's bytes-per-row pad (write_fp16_rgba's pattern, any layout)."""
    import Quartz
    h, w = int(frame.shape[0]), int(frame.shape[1])
    src = memoryview(mx.contiguous(frame)).cast("B")
    row = w * bytes_per_px
    Quartz.CVPixelBufferLockBaseAddress(pb, 0)
    try:
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        mv = base.as_buffer(h * bpr)
        if bpr == row:
            mv[:] = src
        else:
            for r in range(h):
                mv[r * bpr: r * bpr + row] = src[r * row: (r + 1) * row]
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)


def _buffer_attrs(out_format: int) -> dict:
    import Quartz
    if out_format not in (PIX_RGBAHALF, PIX_BGRA):
        raise ValueError(
            f"ffmpeg reader supports RGBAHalf and BGRA outputs; got format {out_format}. "
            "The NV12 fast path needs the native reader."
        )
    return {
        Quartz.kCVPixelBufferPixelFormatTypeKey: out_format,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
        Quartz.kCVPixelBufferMetalCompatibilityKey: True,
    }


def _iter_chunks(path: Path, out_format: int, chunk_size: int,
                 start_frame: int, end_frame: int | None,
                 reformat_kwargs: dict,
                 timing: VideoTiming | None = None,
                 table: SampleTable | None = None) -> Iterator[list]:
    container, vs = _open_video(path)
    try:
        if table is not None and timing is None:
            timing = table.timing()
        keys = ([sample.pts for sample in table.samples]
                if table is not None else None)
        fps = float(timing.cadence) if (
            timing is not None and timing.cadence is not None) else _fps(vs)
        origin = (timing.first_pts if timing is not None
                  else Fraction(int(vs.start_time or 0)) * Fraction(vs.time_base))
        time_base = Fraction(vs.time_base)
        attrs = _buffer_attrs(out_format)
        # Multi-epoch sources decode linearly: a coarse seek landing
        # inside a later epoch would start the decode-side unwrapper
        # without the history that maps raw stamps onto the table clock.
        seekable_clock = table is None or table.epoch_resets == 0
        if start_frame > 0 and seekable_clock and (keys is not None or fps > 0):
            # coarse keyframe seek, then exact per-frame trim below
            if keys is not None:
                back = min(max(start_frame - 1, 0), len(keys) - 1)
                sec = float(keys[back])
            else:
                sec = float(origin) + (start_frame - 1) / fps
            # Some edit timelines legitimately begin before zero. Seeking to
            # zero would skip their head; decode from the beginning instead
            # when the desired coarse seek point is negative.
            if sec >= 0:
                container.seek(
                    int(sec / vs.time_base), stream=vs, backward=True)
        chunk: list = []
        # Decoded frames carry raw stamps; the table's are unwrapped onto
        # one monotonic clock. Run the same streaming mapping here (reset
        # detection survives display reordering) so bisects stay aligned.
        trim_unwrapper = EpochUnwrapper() if keys is not None else None
        for frame in container.decode(vs):
            if frame.pts is None:
                continue
            if keys is not None:
                # Exact window indexing for jittered/gapped/variable clocks.
                assert trim_unwrapper is not None
                adjusted = trim_unwrapper.push(
                    Fraction(int(frame.pts)) * time_base)
                idx = bisect.bisect_right(keys, adjusted) - 1
            elif fps > 0:
                idx = _pts_index(frame.pts, vs, fps, origin=origin)
            else:
                idx = 0
            if idx < start_frame:
                continue
            if end_frame is not None and idx >= end_frame:
                break
            buffer = _frame_to_buffer(frame, out_format, attrs, reformat_kwargs)
            # With a table, label frames by SAMPLE IDENTITY so a decoder
            # that drops or multiplies frames (packed-bitstream AVIs,
            # dummy packets) cannot shift every later timestamp and
            # raw-stream tag by one.
            chunk.append((buffer, idx) if keys is not None else buffer)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
    except av.FFmpegError as exc:
        _raise_ffmpeg_operation("decoding video from", path, exc)
    finally:
        container.close()


def iter_video_buffer_chunks(
    path: Path, src_format: int, chunk_size: int = 8,
    *, start_frame: int = 0, end_frame: int | None = None,
    timing: VideoTiming | None = None,
    table: SampleTable | None = None,
) -> Iterator[list]:
    """Native-surface mirror: lists of CVPixelBuffers in src_format, honoring
    the stream's own color tags (libswscale reads them off each frame)."""
    chunk_size = validate_decode_chunk_size(chunk_size)
    return _iter_chunks(
        path, src_format, chunk_size, start_frame, end_frame, {}, timing,
        table)


def iter_forced_color_chunks(
    path: Path, out_format: int, matrix_cv: Any, full_range: bool,
    chunk_size: int = 8, *, start_frame: int = 0, end_frame: int | None = None,
    reinterpret_full_range: bool | None = None,
    timing: VideoTiming | None = None,
    table: SampleTable | None = None,
) -> Iterator[list]:
    """Forced-matrix read: override the YCbCr matrix / range at YUV->RGB time
    (libswscale src_colorspace/src_color_range), the --source-color fix for
    untagged or mis-tagged material."""
    chunk_size = validate_decode_chunk_size(chunk_size)
    m = str(matrix_cv)
    cs = av.video.reformatter.Colorspace.ITU709 if "709" in m else \
        av.video.reformatter.Colorspace.ITU601
    rng_full = full_range if reinterpret_full_range is None else reinterpret_full_range
    rng = av.video.reformatter.ColorRange.JPEG if rng_full \
        else av.video.reformatter.ColorRange.MPEG
    kw = {"src_colorspace": cs, "src_color_range": rng}
    return _iter_chunks(
        path, out_format, chunk_size, start_frame, end_frame, kw, timing,
        table)


# ---------------------------------------------------------------- audio
def _audio_origin(container: Any, stream: Any) -> Fraction | None:
    first = stream.start_time
    if first is None:
        for packet in container.demux(stream):
            if packet.pts is not None and packet.size > 0:
                first = packet.pts
                break
    if first is None:
        return None
    return Fraction(int(first)) * Fraction(stream.time_base)


def probe_audio_timing(path: Path) -> AudioTiming | None:
    """Return the first audio-stream PTS without decoding any packets."""
    container = _open_container(path, "cannot open audio in")
    try:
        streams = [stream for stream in container.streams
                   if stream.type == "audio"]
        if not streams:
            return None
        stream = streams[0]
        time_base = Fraction(stream.time_base)
        origin = _audio_origin(container, stream)
        if origin is None:
            return None
        return AudioTiming(
            first_pts=origin,
            source_tick=time_base,
        )
    except av.FFmpegError as exc:
        _raise_ffmpeg_operation("probing audio timing in", path, exc)
    finally:
        container.close()


def _audio_segment_start(
    reported_start: int,
    cumulative_start: int | None,
    tick_samples: Fraction,
    path: Path,
) -> int:
    """Snap timestamp quantization but reject a real backward edit."""
    if cumulative_start is None:
        return reported_start
    tolerance = max(
        1,
        (tick_samples.numerator + tick_samples.denominator - 1)
        // tick_samples.denominator,
    )
    delta = reported_start - cumulative_start
    if abs(delta) <= tolerance:
        return cumulative_start
    if delta < 0:
        raise RuntimeError(
            f"audio stream in {path} overlaps by {-delta} samples at "
            f"sample {cumulative_start}")
    return reported_start


def _audio_sample_position(
    pts: int,
    time_base: Fraction,
    origin: Fraction,
    sample_rate: int,
) -> int:
    return round((Fraction(pts) * time_base - origin) * sample_rate)


class _FFmpegAudioSource:
    """Seekable, bounded PyAV PCM cursor for a single writer/sidecar."""

    def __init__(
        self,
        path: Path,
        sample_rate: int,
        channels: int,
        layout: str,
        origin: Fraction,
    ) -> None:
        self._path = path
        self._sample_rate = sample_rate
        self._channels = channels
        self._layout = layout
        self._origin = origin
        self._bytes_per_frame = 4 * channels
        self._container: Any = None
        self._stream: Any = None
        self._segments: Any = None
        self._pending: tuple[int, int, bytes] | None = None
        self._cursor: int | None = None
        self._decoded_frame_count = 0

    def _close_cursor(self) -> None:
        container, self._container = self._container, None
        self._stream = None
        self._segments = None
        self._pending = None
        self._cursor = None
        if container is not None:
            try:
                container.close()
            except av.FFmpegError as exc:
                _raise_ffmpeg_operation(
                    "closing audio in", self._path, exc)

    def _segment_iter(self, resampler: Any) -> Iterator[tuple[int, int, bytes]]:
        assert self._container is not None
        assert self._stream is not None
        fallback_start: int | None = None

        def converted(frame: Any) -> tuple[int, int, bytes]:
            nonlocal fallback_start
            self._decoded_frame_count += 1
            if frame.pts is not None and frame.time_base is not None:
                time_base = Fraction(frame.time_base)
                reported_start = _audio_sample_position(
                    frame.pts,
                    time_base,
                    self._origin,
                    self._sample_rate,
                )
                # Coarse container clocks independently round adjacent PTS.
                # Snap deltas within one source tick to the cumulative sample
                # boundary; otherwise a normal 1 ms Matroska clock invents
                # tiny gaps/overlaps between 1024-sample AAC frames. Larger
                # deltas remain real edits and are preserved as silence below.
                tick = time_base * self._sample_rate
                start = _audio_segment_start(
                    reported_start, fallback_start, tick, self._path)
            elif fallback_start is not None:
                start = fallback_start
            else:
                raise RuntimeError(
                    f"audio frame from {self._path} has no timestamp")
            planes = [
                mx.array(memoryview(plane).cast("B")).view(mx.float32)[
                    :frame.samples]
                for plane in frame.planes
            ]
            if len(planes) != self._channels:
                raise RuntimeError(
                    f"audio channel layout changed while decoding "
                    f"{self._path}: expected {self._channels}, got "
                    f"{len(planes)}")
            raw = bytes(memoryview(mx.contiguous(mx.stack(planes, axis=1))))
            stop = start + int(frame.samples)
            fallback_start = stop
            return start, stop, raw

        for frame in self._container.decode(self._stream):
            for resampled in resampler.resample(frame):
                yield converted(resampled)
        for resampled in resampler.resample(None):
            yield converted(resampled)

    def _reset(self, target: int) -> None:
        self._close_cursor()
        container = _open_container(self._path, "cannot open audio in")
        streams = [stream for stream in container.streams
                   if stream.type == "audio"]
        if not streams:
            container.close()
            raise RuntimeError(f"no audio stream in {self._path}")
        stream = streams[0]
        rate = int(stream.rate or stream.codec_context.sample_rate or 0)
        channels = len(stream.layout.channels)
        if (rate, channels, stream.layout.name) != (
                self._sample_rate, self._channels, self._layout):
            container.close()
            raise RuntimeError(
                f"audio format changed while opening {self._path}")
        # Compressed audio needs decoder preroll (AAC's overlapping transform
        # is the common case). Seek a fixed one-second envelope before the
        # requested sample, decode/discard it, and still retain only the
        # caller's bounded PCM pull. Seeking directly to the preceding packet
        # produces a measurable onset transient versus a full decode.
        seek_sample = max(0, target - self._sample_rate)
        target_time = (
            self._origin
            + Fraction(seek_sample, self._sample_rate)
        )
        timestamp = int(target_time / Fraction(stream.time_base))
        try:
            container.seek(
                timestamp, stream=stream, backward=True, any_frame=False)
        except av.FFmpegError as exc:
            with contextlib.suppress(Exception):
                container.close()
            _raise_ffmpeg_operation("seeking audio in", self._path, exc)
        resampler = av.AudioResampler(
            format="fltp", layout=stream.layout, rate=self._sample_rate)
        self._container = container
        self._stream = stream
        self._segments = iter(self._segment_iter(resampler))
        self._pending = None
        self._cursor = target

    def read_frames(self, start_frame: int, end_frame: int) -> bytes:
        if end_frame <= start_frame:
            return b""
        if self._cursor != start_frame:
            self._reset(start_frame)
        output = bytearray()
        position = start_frame
        while position < end_frame:
            if self._pending is None or self._pending[1] <= position:
                try:
                    self._pending = next(self._segments)
                except av.FFmpegError as exc:
                    _raise_ffmpeg_operation(
                        "decoding audio from", self._path, exc)
                except StopIteration:
                    # Some containers omit stream.duration; the container's
                    # duration is only an upper bound when audio ends before
                    # video. Return a short final pull and let the writer/sidecar
                    # close the track cleanly at the actual decoded EOF.
                    self._cursor = position
                    return bytes(output)
                continue
            segment_start, segment_stop, raw = self._pending
            if segment_start > position:
                # Preserve timeline discontinuities as silence instead of
                # concatenating post-gap audio early. Limit the fill to this
                # caller's bounded request, so a long edit never becomes a
                # whole-track allocation.
                gap_stop = min(segment_start, end_frame)
                output.extend(
                    b"\0" * ((gap_stop - position) * self._bytes_per_frame))
                position = gap_stop
                continue
            take_stop = min(segment_stop, end_frame)
            byte_start = (position - segment_start) * self._bytes_per_frame
            byte_stop = (take_stop - segment_start) * self._bytes_per_frame
            output.extend(raw[byte_start:byte_stop])
            position = take_stop
        self._cursor = end_frame
        return bytes(output)

    def close(self) -> None:
        self._close_cursor()


def read_audio_track_window(
    path: Path,
    *,
    start_sec: Fraction | int | float = Fraction(0),
    end_sec: Fraction | int | float | None = None,
    max_duration_sec: Fraction | int | float | None = None,
) -> Any | None:
    """Open a bounded lazy PCM window from the first audio stream."""
    from .audio import StreamingAudioTrack, _sample_window

    container = _open_container(path, "cannot open audio in")
    try:
        astreams = [s for s in container.streams if s.type == "audio"]
        if not astreams:
            return None
        ast = astreams[0]
        origin = _audio_origin(container, ast)
        if origin is None:
            raise RuntimeError(
                f"audio stream in {path} has no usable presentation origin")
        sample_rate = int(ast.rate or ast.codec_context.sample_rate or 0)
        channels = len(ast.layout.channels)
        if sample_rate <= 0 or channels <= 0:
            return None
        total: int | None = None
        if ast.duration is not None:
            total = round(
                Fraction(ast.duration) * Fraction(ast.time_base) * sample_rate)
        elif container.duration is not None:
            total = round(Fraction(container.duration, 1_000_000) * sample_rate)
        try:
            start, stop = _sample_window(
                sample_rate=sample_rate,
                total_samples=total,
                start_sec=start_sec,
                end_sec=end_sec,
                max_duration_sec=max_duration_sec,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"audio stream in {path} has no usable duration") from exc
        if stop == start:
            return None
        layout = ast.layout.name

        def source_factory() -> _FFmpegAudioSource:
            return _FFmpegAudioSource(
                path, sample_rate, channels, layout, origin)

        return StreamingAudioTrack(
            sample_rate=sample_rate,
            channels=channels,
            n_samples=stop - start,
            source_factory=source_factory,
            offset=start,
        )
    except av.FFmpegError as exc:
        _raise_ffmpeg_operation("opening audio window in", path, exc)
    finally:
        container.close()


def read_audio_track(path: Path) -> Any | None:
    """Compatibility entry point for callers requesting the complete track."""
    return read_audio_track_window(path)


__all__ = [
    "probe_video", "probe_video_timing", "read_sample_table", "probe_color",
    "probe_stream_descriptor", "keyframe_display_indices",
    "iter_video_buffer_chunks", "iter_forced_color_chunks",
    "probe_audio_timing", "read_audio_track", "read_audio_track_window",
]
