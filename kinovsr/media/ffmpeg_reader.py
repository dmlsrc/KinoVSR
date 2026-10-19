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

Beyond the native surface, ``iter_coded_pictures`` reports the coded picture
each frame shows (its type and, where libavcodec exports it, its quantizer),
which AVFoundation does not expose. It decodes without converting pixels and
places pictures exactly as the frame readers do, so its frame numbers are
theirs.
"""

import bisect
import collections
import contextlib
import dataclasses
import heapq
import logging
import struct
from collections.abc import Callable, Iterable, Iterator, Mapping
from fractions import Fraction
from pathlib import Path
from typing import NoReturn, cast

import av
import mlx.core as mx
import Quartz
from av.sidedata.sidedata import Type as SideDataType

from . import pixel_buffers as _pb
from .audio import StreamingAudioTrack, _sample_window
from .buffer import mlx_array_from_buffer
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

_WRITER_AUDIO_RATES = {44_100, 48_000}


def _time_base(stream: av.VideoStream | av.AudioStream) -> Fraction:
    """A demuxed stream's time base. PyAV reports an unset one as the falsy
    AVRational(0, 1), which libavformat should not leave on a stream it
    opened; if it does, the media is broken, an operational failure like the
    others normalized here. PyAV's own rational type converts to an exact
    Fraction."""
    base = stream.time_base
    if not base:
        raise RuntimeError(f"{stream.type} stream #{stream.index} has no time base")
    return Fraction(base.numerator, base.denominator)


def _raise_ffmpeg_operation(
    operation: str,
    path: Path | str,
    exc: Exception,
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


def _open_container(path: Path | str, operation: str) -> av.container.InputContainer:
    try:
        return av.open(str(path))
    except av.FFmpegError as exc:
        _raise_ffmpeg_operation(operation, path, exc)


def _open_video(path: Path) -> tuple[av.container.InputContainer, av.VideoStream]:
    container = _open_container(path, "cannot open video in")
    streams = [s for s in container.streams if s.type == "video"]
    if not streams:
        container.close()
        raise RuntimeError(f"No video track in {path}")
    return container, cast(av.VideoStream, streams[0])  # PyAV wraps type "video" as VideoStream


# Demuxers whose containers store every frame's presentation time and index
# their keyframes: the stamps read after a seek are the stored ones, and the
# seek lands on a keyframe near its target. Near, not always before: FFmpeg
# searches an MP4 index by decode time, so with B frames a seek can land on
# the keyframe displayed after the target, which _iter_chunks detects. Every
# other demuxer is read from the head for a trimmed start. An MPEG program
# stream, for one, seeks to a pack boundary by timestamp, which can fall past
# the keyframe a start needs, and stamps each picture a PES header does not
# from where demuxing began, so after a seek a picture can read another's
# stamp. Names are libavformat's.
_SEEKABLE_FORMATS = frozenset(
    {"mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm", "avi", "flv", "asf", "dv", "mxf"}
)
# Video codecs whose parser hands over one packet per frame, with both fields
# of a field-coded frame in that packet, so the packets can be numbered as
# frames. The H.264 parser hands each field of a field pair over alone.
_FRAME_PACKET_CODECS = frozenset({"mpeg1video", "mpeg2video"})


def _seekable(container: av.container.InputContainer) -> bool:
    return container.format.name in _SEEKABLE_FORMATS


def _stampless(container: av.container.InputContainer) -> bool:
    """A raw elementary stream: libavformat reads no timestamps from it."""
    # Mask the raw flags: the enum's strict boundary rejects a flag it does
    # not name. The stub types a member's value as the enum; it is an int.
    no_timestamps = cast(int, av.format.Flags.no_timestamps.value)
    return bool(container.format.flags & no_timestamps)


def _fps(stream: av.VideoStream) -> float:
    r = stream.average_rate or stream.guessed_rate or stream.base_rate
    return float(r) if r else 0.0


def _pts_index(
    pts: int,
    stream: av.VideoStream,
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
        seconds = Fraction(pts - start) * Fraction(_time_base(stream))
    else:
        seconds = Fraction(pts) * Fraction(_time_base(stream)) - origin
    return round(seconds * fps)


def _display_transform(container: av.container.InputContainer, vs: av.VideoStream) -> object:
    """The stream's display matrix as a CGAffineTransform (identity when absent).

    libavformat keeps the rotation/flip as DISPLAYMATRIX side data, which
    libavcodec propagates onto decoded frames -- so decode one frame and read
    it there. The 3x3 matrix is row-major with the affine part in 16.16 fixed
    point; its top-left 2x2 + translation map directly onto
    CGAffineTransform's (a, b, c, d, tx, ty), the same convention as
    AVFoundation's preferredTransform (verified component-wise against the
    native probe on rotated and mirrored files).
    """
    try:
        frame = next(container.decode(vs), None)
    except av.FFmpegError:
        # A first frame that does not decode leaves the orientation unknown;
        # decoding proper reports the failure.
        return Quartz.CGAffineTransformIdentity
    # Side data is a mapping keyed by type; iterating it yields the keys.
    side_data = None if frame is None else frame.side_data
    if side_data is None or SideDataType.DISPLAYMATRIX not in side_data:
        return Quartz.CGAffineTransformIdentity
    m = struct.unpack("<9i", bytes(side_data[SideDataType.DISPLAYMATRIX]))
    s = 1.0 / 65536.0
    return Quartz.CGAffineTransformMake(m[0] * s, m[1] * s, m[3] * s, m[4] * s, m[6] * s, m[7] * s)


# ---------------------------------------------------------------- probes
def probe_video(path: Path) -> tuple[int, int, float, int, object, tuple[int, int] | None]:
    """(width, height, fps, n_frames, transform, pixel_aspect); mirrors the
    native probe. transform carries the stream's display matrix (rotation /
    flip) mapped to the AVFoundation preferredTransform convention."""
    container, vs = _open_video(path)
    try:
        w, h = int(vs.codec_context.width), int(vs.codec_context.height)
        fps = _fps(vs)
        n = int(vs.frames or 0)
        if n <= 0 and vs.duration is not None and fps > 0:
            n = round(float(vs.duration * _time_base(vs)) * fps)
        if n <= 0 and container.duration is not None and fps > 0:
            n = round(container.duration / 1_000_000 * fps)
        sar = vs.sample_aspect_ratio
        pixel_aspect = None
        if sar and sar.numerator > 0 and sar.denominator > 0 and sar.numerator != sar.denominator:
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

    Where the stamps cannot identify pictures, the table numbers the coded
    pictures in decode order instead (``SampleTable.positional_stamps``): a
    raw elementary stream carries no stamps, and a muxer can write a PES
    timestamp that disagrees with its picture, which gives two pictures one
    stamp.
    """
    container, vs = _open_video(path)
    try:
        time_base = Fraction(_time_base(vs))
        stampless = _stampless(container)
        samples: list[SampleTiming] = []
        # (keyframe, coded size) of each coded picture, in decode order.
        pictures: list[tuple[bool, int]] = []
        unwrapper = EpochUnwrapper()
        # Presentation-minus-decode offsets seen; one alone means the stamps
        # follow decode order (see SampleTable.decode_order_stamps).
        offsets: set[int] = set()
        for packet in container.demux(vs):
            if packet.size > 0:
                pictures.append((bool(packet.is_keyframe), int(packet.size)))
            if stampless or packet.pts is None:
                continue
            if packet.dts is not None and len(offsets) < 2:
                offsets.add(int(packet.pts) - int(packet.dts))
            # Zero-size packets that carry a timestamp are timeline slots
            # (Ogg/Theora "repeat previous frame" markers): record them so
            # the decode loop can place the re-materialized duplicate on
            # its own sample. Only the timestampless flush sentinel is
            # skipped. (The native reader never sees such containers, so
            # the walks stay mirrored in practice.)
            pts = unwrapper.push(Fraction(int(packet.pts)) * time_base)
            duration = (
                Fraction(int(packet.duration)) * time_base
                if packet.duration and packet.duration > 0
                else None
            )
            samples.append(
                SampleTiming(
                    pts=pts,
                    duration=duration,
                    is_sync=bool(packet.is_keyframe),
                    coded_size=int(packet.size),
                )
            )
        if stampless or (
            not _seekable(container)
            and vs.codec_context.name in _FRAME_PACKET_CODECS
            and len({sample.pts for sample in samples}) < len(samples)
        ):
            return _positional_table(path, vs, pictures, samples)
        if unwrapper.resets:
            _log.info(
                "%s: unwrapped %d mid-file timestamp reset(s) into one declared monotonic clock",
                path.name,
                unwrapper.resets,
            )
        try:
            return analyze_sample_table(
                samples,
                nominal_cadence=_fps(vs),
                source_tick=time_base,
                epoch_resets=unwrapper.resets,
                decode_order_stamps=bool(vs.codec_context.has_b_frames) and len(offsets) == 1,
            )
        except ValueError as exc:
            raise RuntimeError(f"{path.name}: {exc}") from exc
    except av.FFmpegError as exc:
        _raise_ffmpeg_operation("probing video timing in", path, exc)
    finally:
        container.close()


def _positional_table(
    path: Path,
    vs: av.VideoStream,
    pictures: list[tuple[bool, int]],
    stamped: list[SampleTiming],
) -> SampleTable:
    """The coded pictures in decode order, on a constant grid.

    The grid runs at the declared rate unless the stamps' mean spacing is
    more than 5% away from it: a soft-telecined MPEG-2 stream declares
    29.97 fps over 23.976 coded frames a second. It starts at the first
    stamp, which keeps the video clock where the audio's stamps expect it.
    """
    if not pictures:
        raise RuntimeError(f"{path.name}: video track contains no display samples")
    stamps = sorted(sample.pts for sample in stamped)
    declared = vs.average_rate or vs.guessed_rate or vs.base_rate
    rate = Fraction(declared.numerator, declared.denominator) if declared else None
    if len(stamps) > 1 and stamps[-1] > stamps[0]:
        mean = Fraction(len(stamps) - 1) / (stamps[-1] - stamps[0])
        if rate is None or abs(mean / rate - 1) > Fraction(1, 20):
            rate = mean.limit_denominator(1001)
    if rate is None or rate <= 0:
        raise RuntimeError(f"{path.name}: video stream declares no frame rate")
    if stamps:
        _log.warning(
            "%s: %d picture(s) share a presentation timestamp with another, so the stamps "
            "cannot tell pictures apart; numbering them in bitstream order at %s fps instead",
            path.name,
            len(stamps) - len(set(stamps)),
            rate,
        )
    origin = stamps[0] if stamps else Fraction(0)
    tick = 1 / rate
    try:
        return analyze_sample_table(
            (
                SampleTiming(pts=origin + index * tick, duration=tick, is_sync=key, coded_size=size)
                for index, (key, size) in enumerate(pictures)
            ),
            nominal_cadence=rate,
            source_tick=Fraction(_time_base(vs)),
            positional_stamps=True,
        )
    except ValueError as exc:
        raise RuntimeError(f"{path.name}: {exc}") from exc


def probe_video_timing(path: Path) -> VideoTiming:
    """Classify exact packet presentation timestamps without decoding.

    The legacy CFR-or-variable view over :func:`read_sample_table`'s walk.
    """
    return read_sample_table(path).timing()


# libav color tags -> the CoreVideo tokens the native probe_color emits (see
# color.py; the resolver and the writer match on these exact strings, so they
# are CoreVideo's own constants rather than spelled-out copies).
# Each code maps to what the native reader reports for it (after
# color.resolve's aliases), so both readers tag an output alike.
_PRIMARIES = {
    1: Quartz.kCVImageBufferColorPrimaries_ITU_R_709_2,  # bt709
    5: Quartz.kCVImageBufferColorPrimaries_EBU_3213,  # bt470bg (PAL)
    6: Quartz.kCVImageBufferColorPrimaries_SMPTE_C,  # smpte170m
    7: Quartz.kCVImageBufferColorPrimaries_SMPTE_C,  # smpte240m (the SMPTE C set)
    9: Quartz.kCVImageBufferColorPrimaries_ITU_R_2020,  # bt2020
    11: Quartz.kCVImageBufferColorPrimaries_DCI_P3,  # smpte431
    12: Quartz.kCVImageBufferColorPrimaries_P3_D65,  # smpte432
    22: Quartz.kCVImageBufferColorPrimaries_P22,  # jedec-p22
}
_MATRIX = {
    1: Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2,  # bt709
    5: Quartz.kCVImageBufferYCbCrMatrix_ITU_R_601_4,  # bt470bg (BT.601 coefficients)
    6: Quartz.kCVImageBufferYCbCrMatrix_ITU_R_601_4,  # smpte170m
    7: Quartz.kCVImageBufferYCbCrMatrix_SMPTE_240M_1995,  # smpte240m
    9: Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020,  # bt2020nc
    10: Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020,  # bt2020c (BT.2020 coefficients)
}
_TRANSFER = {
    1: Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,  # bt709
    6: Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,  # smpte170m (SDR gamma family)
    7: Quartz.kCVImageBufferTransferFunction_SMPTE_240M_1995,  # smpte240m
    8: Quartz.kCVImageBufferTransferFunction_Linear,  # linear
    13: Quartz.kCVImageBufferTransferFunction_sRGB,  # iec61966-2-1 ("IEC_sRGB")
    14: Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,  # bt2020-10 (SDR gamma family)
    15: Quartz.kCVImageBufferTransferFunction_ITU_R_709_2,  # bt2020-12
    16: Quartz.kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ,  # smpte2084
    17: Quartz.kCVImageBufferTransferFunction_SMPTE_ST_428_1,  # smpte428
    18: Quartz.kCVImageBufferTransferFunction_ITU_R_2100_HLG,  # arib-std-b67
}


def probe_color(path: Path) -> dict[str, object]:
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
    "yuv420p": (8, "420"),
    "yuvj420p": (8, "420"),
    "nv12": (8, "420"),
    "yuv420p10le": (10, "420"),
    "yuv420p12le": (12, "420"),
    "yuv422p": (8, "422"),
    "yuvj422p": (8, "422"),
    "yuv422p10le": (10, "422"),
    "yuv422p12le": (12, "422"),
    "yuv444p": (8, "444"),
    "yuvj444p": (8, "444"),
    "yuv444p10le": (10, "444"),
    "yuv444p12le": (12, "444"),
}


def is_program_stream(path: Path) -> bool:
    """Whether libavformat demuxes ``path`` as an MPEG program stream.

    Identified by the demuxer's probe of the content (VOB and VCD files
    included), not by the file name. False when the file cannot be opened.
    """
    try:
        container = _open_container(path, "cannot open")
    except OSError, RuntimeError:
        return False
    try:
        return container.format.name == "mpeg"
    finally:
        container.close()


def probe_stream_descriptor(path: Path) -> dict[str, str | int | None]:
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
        depth, chroma = _PIX_FMT_TRAITS.get(str(cc.pix_fmt or ""), (None, None))
        profile = cc.profile if isinstance(cc.profile, str) else None
        return {"codec": str(cc.name), "profile": profile, "bit_depth": depth, "chroma": chroma}
    finally:
        container.close()


# ---------------------------------------------------------------- frames
def _plane_to_mx(frame: av.VideoFrame, bytes_per_px: int, width_els: int) -> mx.array:
    """Packed plane 0 -> mx array (h, width_els), stride-cropped, no numpy."""
    plane = frame.planes[0]
    h = int(frame.height)
    stride = int(plane.line_size)
    raw = mlx_array_from_buffer(memoryview(plane).cast("B"))
    if bytes_per_px == 2:
        return raw.view(mx.uint16).reshape(h, stride // 2)[:, :width_els]
    return raw.reshape(h, stride)[:, :width_els]


def _frame_to_buffer(
    frame: av.VideoFrame,
    out_format: int,
    attrs: dict[str, object],
    reformat_kwargs: Mapping[str, int],
) -> object:
    """Decode one av.VideoFrame into a CVPixelBuffer of out_format."""
    w, h = int(frame.width), int(frame.height)
    # >8-bit sources keep precision through rgb48le; 8-bit through rgb24.
    deep = (
        "p10" in frame.format.name
        or "p12" in frame.format.name
        or frame.format.name in ("yuv420p10le", "yuv422p10le", "yuv444p10le")
    )
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
            [rgb.astype(mx.float16), mx.ones((h, w, 1), dtype=mx.float16)], axis=-1
        )
        mx.eval(rgba)
        _pb.write_fp16_rgba(rgba, pb)
    else:  # PIX_BGRA
        u8 = mx.clip(rgb * 255.0 + 0.5, 0, 255).astype(mx.uint8)
        bgra = mx.concatenate(
            [u8[..., 2:3], u8[..., 1:2], u8[..., 0:1], mx.full((h, w, 1), 255, dtype=mx.uint8)],
            axis=-1,
        )
        mx.eval(bgra)
        _write_packed(bgra, pb, bytes_per_px=4)
    return pb


# How the frame readers hand over each decoded picture.
_Convert = Callable[[av.VideoFrame], object]


def _to_buffer(out_format: int, reformat_kwargs: Mapping[str, int]) -> _Convert:
    """Pictures as CVPixelBuffers of ``out_format``; refuses a format this reader cannot write."""
    attrs = _buffer_attrs(out_format)
    return lambda frame: _frame_to_buffer(frame, out_format, attrs, reformat_kwargs)


def _write_packed(frame: mx.array, pb: object, bytes_per_px: int) -> None:
    """Memcpy a packed (H,W,C) frame into a CVPixelBuffer plane, honoring the
    destination's bytes-per-row pad (write_fp16_rgba's pattern, any layout)."""

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
                mv[r * bpr : r * bpr + row] = src[r * row : (r + 1) * row]
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)


def _buffer_attrs(out_format: int) -> dict[str, object]:

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


# Deeper than any B-frame run a packed stream holds (DivX writes one, Xvid up
# to four); libavcodec reorders at most 16 pictures for any codec.
_SLOT_ORDER_DEPTH = 16


class _SlotOrder:
    """Slot-stamped pictures in display order, each given its own slot.

    The pictures arrive in display order carrying slot stamps in decode order.
    Held in a window deeper than any B-frame run, the n-th picture takes the
    n-th smallest stamp.
    """

    def __init__(self, depth: int = _SLOT_ORDER_DEPTH) -> None:
        self.depth = depth
        self._held: collections.deque[tuple[av.VideoFrame, int]] = collections.deque()
        self._stamps: list[int] = []
        # Pictures placed in a slot other than the one they were stamped with.
        self.moved = 0

    def push(self, picture: av.VideoFrame, stamp: int) -> list[tuple[av.VideoFrame, int]]:
        """Hold one picture; return the pictures now placed, with their slots."""
        self._held.append((picture, stamp))
        heapq.heappush(self._stamps, stamp)
        return [self._place() for _ in range(len(self._held) - self.depth)]

    def drain(self) -> list[tuple[av.VideoFrame, int]]:
        """Place every held picture, once the stream has ended."""
        return [self._place() for _ in range(len(self._held))]

    def _place(self) -> tuple[av.VideoFrame, int]:
        picture, stamp = self._held.popleft()
        slot = heapq.heappop(self._stamps)
        self.moved += slot != stamp
        return picture, slot


def _slot_order(
    container: av.container.InputContainer,
    stream: av.VideoStream,
    table: SampleTable | None,
    positional: bool = False,
) -> _SlotOrder | None:
    """The display-order window for a stream whose pictures carry packet slots.

    libavcodec hands pictures over in display order, each stamped with the
    stamp of the packet being decoded when it was made. Where the packets'
    stamps follow decode order, the pictures' stamps arrive permuted, and the
    pictures take the stamps in sorted order instead
    (``SampleTable.decode_order_stamps``; FLV muxers that zero H.264's
    composition offsets write such stamps).

    AVI stores no presentation times at all: each index entry is one display
    slot, and a stream that reorders pictures is written in decode order.
    FFmpeg derives display times for plain MPEG-4 Part 2, but not for H.264,
    nor for the "packed bitstream" in which DivX and Xvid store MPEG-4 Part 2:
    a P frame and the B frame displayed before it share one packet, a
    placeholder N-VOP holds the next slot, and libavcodec decodes the B in the
    placeholder's place, so each pair arrives with swapped stamps. A VfW drop
    frame, a packet without a picture such as the one-byte delay frame DivX
    writes after the first I frame, leaves its slot without one, and the slot
    repeats the picture before it. In AVI a stream that does not reorder
    needs no window, only that fill.

    Pictures numbered by decode position (``positional``) carry slots by
    construction. Their window always runs at full depth, because a raw
    elementary stream need not declare whether it reorders.
    """
    if positional:
        return _SlotOrder()
    reorders = bool(stream.codec_context.has_b_frames)
    if container.format.name == "avi":
        return _SlotOrder(_SLOT_ORDER_DEPTH if reorders else 0)
    if table is not None and table.decode_order_stamps:
        return _SlotOrder()
    return None


class _LateStart(Exception):
    """The first picture decoded after a seek or skip lies past the one needed."""

    def __init__(self, route: str, landed: int | None, needed: int) -> None:
        super().__init__(route, landed, needed)
        self.route = route
        self.landed = landed
        self.needed = needed


def _iter_chunks(
    path: Path,
    convert: _Convert,
    chunk_size: int,
    start_frame: int,
    end_frame: int | None,
    timing: VideoTiming | None = None,
    table: SampleTable | None = None,
    *,
    codec_options: Mapping[str, str] | None = None,
) -> Iterator[list[object]]:
    """Chunks of the window [start_frame, end_frame), by the cheapest route that reaches it.

    A trimmed read seeks in a container that stores its stamps
    (``_SEEKABLE_FORMATS``) and otherwise walks the packets from the head,
    decoding from the table's keyframe before the start. Either way the
    first picture decoded must lie at or before the frame the start needs.
    When it lies past it, nothing has been yielded yet, and the read falls
    back: from the seek to the walk, and from the walk to decoding every
    picture from the head, which needs no check.
    """
    seek = skip = True
    while True:
        try:
            yield from _decode_window(
                path,
                convert,
                chunk_size,
                start_frame,
                end_frame,
                timing,
                table,
                seek=seek,
                skip=skip,
                codec_options=codec_options,
            )
            return
        except _LateStart as late:
            _log.info(
                "%s: the %s toward frame %d first decoded %s; %s",
                path.name,
                late.route,
                start_frame,
                "no frame"
                if late.landed is None
                else f"frame {late.landed}, after frame {late.needed}",
                "walking the packets from the head instead"
                if late.route == "seek"
                else "decoding from the head instead",
            )
            if late.route == "seek":
                seek = False
            else:
                seek = skip = False


def _decode_window(
    path: Path,
    convert: _Convert,
    chunk_size: int,
    start_frame: int,
    end_frame: int | None,
    timing: VideoTiming | None,
    table: SampleTable | None,
    *,
    seek: bool,
    skip: bool,
    codec_options: Mapping[str, str] | None = None,
) -> Iterator[list[object]]:
    container, vs = _open_video(path)
    try:
        if codec_options:
            # Before the first decode, which opens the codec context.
            vs.codec_context.options = dict(codec_options)
        # Pictures numbered by decode position (SampleTable.positional_stamps);
        # without a table, a raw stream has no stamps to number them by.
        positional = table.positional_stamps if table is not None else _stampless(container)
        if table is not None and timing is None:
            timing = table.timing()
        keys = (
            [sample.pts for sample in table.samples]
            if table is not None and not positional
            else None
        )
        fps = (
            float(timing.cadence)
            if (timing is not None and timing.cadence is not None)
            else _fps(vs)
        )
        origin = (
            timing.first_pts
            if timing is not None
            else Fraction(int(vs.start_time or 0)) * Fraction(_time_base(vs))
        )
        time_base = Fraction(_time_base(vs))
        # Multi-epoch sources decode linearly: a coarse seek landing
        # inside a later epoch would start the decode-side unwrapper
        # without the history that maps raw stamps onto the table clock.
        seekable_clock = table is None or table.epoch_resets == 0
        # A slot-stamped stream's pictures pass through a sorting window,
        # and each slot left without a picture repeats the one before it.
        slots = _slot_order(container, vs, table, positional)
        # A coarse start (a keyframe seek, or a walk that decodes from the
        # keyframe before the start), then the exact per-frame trim below. A
        # slot stamp finds a keyframe by decode position, and the B frames
        # after it in decode order but before it in display order cannot
        # decode from it: land a sorting window earlier, so their slots fall
        # before the start.
        lead = 1 + (slots.depth if slots is not None else 0)
        back = start_frame - lead
        route = "head"
        # The first picture decoded after a seek or skip must lie at or
        # before `back`; None once checked, or for a decode from the head.
        verify: int | None = None
        skip_to: Fraction | None = None
        skip_position: int | None = None
        if back >= 0 and seekable_clock:
            if seek and not positional and _seekable(container) and (keys is not None or fps > 0):
                if keys is not None:
                    sec = float(keys[min(back, len(keys) - 1)])
                else:
                    sec = float(origin) + back / fps
                # Some edit timelines legitimately begin before zero. Seeking to
                # zero would skip their head; decode from the beginning instead
                # when the desired coarse seek point is negative.
                if sec >= 0:
                    container.seek(int(sec / _time_base(vs)), stream=vs, backward=True)
                    route, verify = "seek", back
            elif skip and table is not None:
                # Demux from the head, as the table's walk did, so every
                # packet reads the stamp the table recorded, and decode from
                # the table's last keyframe at or before `back`.
                sync = [index for index, s in enumerate(table.samples[: back + 1]) if s.is_sync]
                if sync:
                    if positional:
                        skip_position = sync[-1]
                    else:
                        skip_to = table.samples[sync[-1]].pts
                    route, verify = "keyframe skip", back
        chunk: list[object] = []
        # Decoded frames carry raw stamps; the table's are unwrapped onto
        # one monotonic clock. Run the same streaming mapping here (reset
        # detection survives display reordering) so bisects stay aligned.
        trim_unwrapper = EpochUnwrapper() if keys is not None else None

        def frame_index(pts: int) -> int:
            if positional:
                return pts
            if keys is not None:
                # Exact window indexing for jittered/gapped/variable clocks.
                assert trim_unwrapper is not None
                adjusted = trim_unwrapper.push(Fraction(int(pts)) * time_base)
                return bisect.bisect_right(keys, adjusted) - 1
            if fps > 0:
                return _pts_index(pts, vs, fps, origin=origin)
            return 0

        def check_landing(index: int) -> None:
            """Raise when the first picture after a seek or skip lies past `back`."""
            nonlocal verify
            if verify is not None:
                if index > verify:
                    raise _LateStart(route, index, verify)
                verify = None

        def item(frame: av.VideoFrame, idx: int) -> object:
            value = convert(frame)
            # With a table, label frames by SAMPLE IDENTITY so a decoder
            # that drops or multiplies frames (dummy packets) cannot shift
            # every later timestamp and raw-stream tag by one.
            return (value, idx) if table is not None else value

        def filled(items: Iterable[object]) -> Iterator[list[object]]:
            """Add frames to the chunk, handing it over each time it fills."""
            nonlocal chunk
            for entry in items:
                chunk.append(entry)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []

        # Ogg/Theora encodes "this frame repeats the previous one" as a
        # ZERO-BYTE packet carrying a real timestamp. libavcodec's
        # send_packet rejects empty packets (EINVAL) even though
        # libavformat's own demuxer emits them; ffmpeg's CLI skips them
        # in application code and silently DROPS the repeat frames. This
        # loop re-materializes each marker as a duplicate of the previous
        # decoded frame at the marker's timestamp, so quirky-but-legal
        # old files keep their exact timeline. PyAV's end-of-stream flush
        # sentinel is also zero-size but carries no timestamps and must
        # still reach decode() to drain reordering codecs.
        last_frame: av.VideoFrame | None = None
        last_index: int | None = None
        dup_emitted = 0
        dup_dropped = 0
        stop = False
        slot_end = (
            end_frame
            if end_frame is not None
            else len(table.samples)
            if table is not None
            else None
        )
        repeated = 0

        def in_slot(frame: av.VideoFrame, idx: int) -> Iterator[object]:
            """A slot-stamped picture in its slot, after the empty slots before it.

            An empty slot repeats the picture before it, as players show a
            VfW drop frame; ahead of the first picture it takes that picture.
            """
            nonlocal last_frame, last_index, stop, repeated
            if last_index is None or idx > last_index:
                first = start_frame if last_index is None else max(last_index + 1, start_frame)
                for empty in range(first, idx if slot_end is None else min(idx, slot_end)):
                    repeated += 1
                    yield item(frame if last_frame is None else last_frame, empty)
                last_frame, last_index = frame, idx
            if idx < start_frame:
                return
            if end_frame is not None and idx >= end_frame:
                stop = True
                return
            yield item(frame, idx)

        def trailing() -> Iterator[object]:
            """The empty slots after the last picture, each repeating it."""
            nonlocal repeated
            if last_frame is None or last_index is None or slot_end is None:
                return
            for empty in range(max(last_index + 1, start_frame), slot_end):
                repeated += 1
                yield item(last_frame, empty)

        position = -1  # the decode position of the current coded picture
        skipping = route == "keyframe skip"
        for packet in container.demux(vs):
            if packet.size > 0:
                position += 1
                if positional:
                    packet.pts = packet.dts = position
            if skipping:
                if packet.size == 0 or not packet.is_keyframe:
                    continue
                if positional:
                    reached = skip_position is not None and position >= skip_position
                else:
                    reached = (
                        skip_to is not None
                        and packet.pts is not None
                        and Fraction(int(packet.pts)) * time_base >= skip_to
                    )
                if not reached:
                    continue
                skipping = False
            if packet.size == 0 and not (packet.pts is None and packet.dts is None):
                if slots is not None:
                    # An empty slot, filled once the pictures are in display
                    # order.
                    continue
                pts = packet.pts if packet.pts is not None else packet.dts
                assert pts is not None  # the enclosing check rules out both stamps missing
                idx = frame_index(int(pts))
                if last_frame is None or (last_index is not None and idx <= last_index):
                    # No decodable predecessor (marker leads the stream or
                    # follows a seek), or the index grid has no slot for
                    # it: dropping is the only safe move.
                    dup_dropped += 1
                    if dup_dropped == 1:
                        _log.warning(
                            "%s: zero-byte repeat-frame packet at pts %s "
                            "has no usable predecessor; dropping it "
                            "(ffmpeg CLI parity)",
                            path.name,
                            pts,
                        )
                    continue
                dup_emitted += 1
                if dup_emitted == 1:
                    _log.warning(
                        "%s: zero-byte repeat-frame packet(s) (Ogg/Theora "
                        "duplicate markers; libavcodec rejects empty "
                        "packets): re-materializing each as a duplicate "
                        "of the previous frame at the marker's timestamp, "
                        "first at pts %s",
                        path.name,
                        pts,
                    )
                last_index = idx
                if idx < start_frame:
                    continue
                if end_frame is not None and idx >= end_frame:
                    break
                yield from filled([item(last_frame, idx)])
                continue
            try:
                frames = packet.decode()
            except av.FFmpegError:
                if packet.size == 0:
                    # A malformed empty packet that dodged both branches
                    # (e.g. timestamped like a flush sentinel): skip it
                    # rather than kill the run.
                    dup_dropped += 1
                    _log.warning("%s: undecodable zero-byte packet skipped", path.name)
                    continue
                raise
            for frame in frames:
                if frame.pts is None:
                    continue
                idx = frame_index(int(frame.pts))
                if slots is not None:
                    for picture, slot in slots.push(frame, idx):
                        check_landing(slot)
                        yield from filled(in_slot(picture, slot))
                        if stop:
                            break
                    if stop:
                        break
                    continue
                check_landing(idx)
                last_frame = frame
                last_index = idx
                if idx < start_frame:
                    continue
                if end_frame is not None and idx >= end_frame:
                    stop = True
                    break
                yield from filled([item(frame, idx)])
            if stop:
                break
        if slots is not None:
            if not stop:
                for picture, slot in slots.drain():
                    check_landing(slot)
                    yield from filled(in_slot(picture, slot))
                    if stop:
                        break
            if not stop:
                yield from filled(trailing())
            if slots.moved or repeated:
                _log.info(
                    "%s: decode-order stamps read in display order: %d "
                    "picture(s) moved to their display slot, %d slot(s) "
                    "without a picture (drop frames) repeat the one before",
                    path.name,
                    slots.moved,
                    repeated,
                )
        if verify is not None:
            # The seek or skip decoded no picture at all; nothing was yielded.
            raise _LateStart(route, None, verify)
        if dup_emitted or dup_dropped:
            _log.warning(
                "%s: %d duplicate frame(s) re-materialized from zero-byte "
                "packets to keep the source timeline exact; %d dropped",
                path.name,
                dup_emitted,
                dup_dropped,
            )
        if chunk:
            yield chunk
    except av.FFmpegError as exc:
        _raise_ffmpeg_operation("decoding video from", path, exc)
    finally:
        container.close()


def iter_video_buffer_chunks(
    path: Path,
    src_format: int,
    chunk_size: int = 8,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    timing: VideoTiming | None = None,
    table: SampleTable | None = None,
) -> Iterator[list[object]]:
    """Native-surface mirror: lists of CVPixelBuffers in src_format, honoring
    the stream's own color tags (libswscale reads them off each frame)."""
    chunk_size = validate_decode_chunk_size(chunk_size)
    convert = _to_buffer(src_format, {})
    return _iter_chunks(path, convert, chunk_size, start_frame, end_frame, timing, table)


def iter_forced_color_chunks(
    path: Path,
    out_format: int,
    matrix_cv: object,
    full_range: bool,
    chunk_size: int = 8,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    reinterpret_full_range: bool | None = None,
    timing: VideoTiming | None = None,
    table: SampleTable | None = None,
) -> Iterator[list[object]]:
    """Forced-matrix read: override the YCbCr matrix / range at YUV->RGB time
    (libswscale src_colorspace/src_color_range), the --source-color fix for
    untagged or mis-tagged material."""
    chunk_size = validate_decode_chunk_size(chunk_size)
    # libswscale's matrix for each forced CoreVideo matrix; BT.601 otherwise.
    spaces = av.video.reformatter.Colorspace
    cs = {
        Quartz.kCVImageBufferYCbCrMatrix_ITU_R_709_2: spaces.ITU709,
        Quartz.kCVImageBufferYCbCrMatrix_ITU_R_2020: spaces.BT2020,
        Quartz.kCVImageBufferYCbCrMatrix_SMPTE_240M_1995: spaces.SMPTE240M,
    }.get(str(matrix_cv), spaces.ITU601)
    rng_full = full_range if reinterpret_full_range is None else reinterpret_full_range
    rng = av.video.reformatter.ColorRange.JPEG if rng_full else av.video.reformatter.ColorRange.MPEG
    kw = {"src_colorspace": cs, "src_color_range": rng}
    convert = _to_buffer(out_format, kw)
    return _iter_chunks(path, convert, chunk_size, start_frame, end_frame, timing, table)


# ---------------------------------------------------------------- coded pictures
# FFmpeg's AVPictureType values as PyAV reports them. An S picture (an MPEG-4
# S-VOP) is a P picture with global motion compensation; B and BI pictures are
# never references.
_PICTURE_KINDS = {1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}
REFERENCE_KINDS = frozenset({"I", "P", "S", "SI", "SP"})
# Pictures per decode chunk: no pixels are held, so a chunk is cheap.
_PICTURE_CHUNK = 64


@dataclasses.dataclass(frozen=True, slots=True)
class CodedPicture:
    """The coded picture a frame shows: its type and, where exported, its quantizer.

    ``kind`` is FFmpeg's picture type: I, P, B, S, SI, SP, BI, or "?" when the
    decoder sets none. ``quantizer`` is the mean block quantizer libavcodec
    exports with ``export_side_data=venc_params``, in the codec's own units and
    comparable within one stream: MPEG-1, -2 and -4 Part 2 give the MPEG-2
    quantiser scale (twice an MPEG-4 qscale), H.264 its QP, VP9 its quantizer
    index. It is None where the decoder exports none: other codecs, and a
    picture libavcodec drains at the end of a stream.
    """

    kind: str
    quantizer: float | None

    @property
    def reference(self) -> bool:
        """Whether a later picture may predict from this one (I, P, S, SI, SP)."""
        return self.kind in REFERENCE_KINDS


def _quantizer(frame: av.VideoFrame) -> float | None:
    """The picture's mean block quantizer from its video encoding parameters."""
    params = frame.side_data.get(SideDataType.VIDEO_ENC_PARAMS)
    if params is None:
        return None
    raw = bytes(memoryview(params))
    # AVVideoEncParams: uint32 nb_blocks, size_t blocks_offset, size_t
    # block_size, int type, int32 qp; each block (block_size bytes) holds
    # int src_x, src_y, w, h and then its int32 delta_qp.
    blocks = int(struct.unpack_from("<I", raw, 0)[0])
    offset, size = (int(value) for value in struct.unpack_from("<QQ", raw, 8))
    base = int(struct.unpack_from("<i", raw, 28)[0])
    if not blocks or size % 4:
        return float(base)
    cells = memoryview(raw)[offset : offset + blocks * size].cast("i")
    return base + sum(cells[4 :: size // 4]) / blocks


def _coded_picture(frame: av.VideoFrame) -> CodedPicture:
    return CodedPicture(_PICTURE_KINDS.get(int(frame.pict_type), "?"), _quantizer(frame))


def iter_coded_pictures(
    path: Path,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    table: SampleTable | None = None,
) -> Iterator[tuple[int, CodedPicture]]:
    """The coded picture each frame of [start_frame, end_frame) shows, in display order.

    Yields (table index, picture). Pictures are decoded without converting pixels
    and placed exactly as :func:`iter_video_buffer_chunks` places them: through a
    seek or a walk from the keyframe before the start, through the slot window of
    a stream stamped in decode order, and with a slot left without a picture (a
    drop frame) or a zero-byte repeat marker reporting the picture it repeats.
    Frames are numbered by ``table``, :func:`read_sample_table`'s by default.
    """
    if table is None:
        table = read_sample_table(path)
    chunks = _iter_chunks(
        path,
        _coded_picture,
        _PICTURE_CHUNK,
        start_frame,
        end_frame,
        None,
        table,
        codec_options={"export_side_data": "venc_params"},
    )
    for chunk in chunks:
        for item in chunk:
            picture, index = cast(tuple[CodedPicture, int], item)
            yield index, picture


# ---------------------------------------------------------------- audio
def _audio_origin(
    container: av.container.InputContainer,
    stream: av.AudioStream,
) -> Fraction | None:
    first = stream.start_time
    if first is None:
        for packet in container.demux(stream):
            if packet.pts is not None and packet.size > 0:
                first = packet.pts
                break
    if first is None:
        return None
    return Fraction(int(first)) * Fraction(_time_base(stream))


def probe_audio_timing(path: Path) -> AudioTiming | None:
    """Return the first audio-stream PTS, or None when no stream exists."""
    container = _open_container(path, "cannot open audio in")
    try:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if not streams:
            return None
        stream = cast(av.AudioStream, streams[0])  # PyAV wraps type "audio" as AudioStream
        time_base = Fraction(_time_base(stream))
        origin = _audio_origin(container, stream)
        if origin is None:
            raise RuntimeError(f"audio stream in {path} has no usable presentation origin")
        return AudioTiming(
            first_pts=origin,
            source_tick=time_base,
        )
    except av.FFmpegError as exc:
        _raise_ffmpeg_operation("probing audio timing in", path, exc)
    finally:
        container.close()


def _audio_clock_tolerance(tick_samples: Fraction) -> int:
    return max(
        1,
        (tick_samples.numerator + tick_samples.denominator - 1) // tick_samples.denominator,
    )


def _audio_segment_start(
    reported_start: int,
    cumulative_start: int | None,
    tick_samples: Fraction,
) -> int:
    """Snap clock rounding while retaining larger gaps and overlaps."""
    if cumulative_start is None:
        return reported_start
    if abs(reported_start - cumulative_start) <= _audio_clock_tolerance(tick_samples):
        return cumulative_start
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
        *,
        source_sample_rate: int | None = None,
    ) -> None:
        self._path = path
        self._sample_rate = sample_rate
        self._source_sample_rate = source_sample_rate or sample_rate
        self._channels = channels
        self._layout = layout
        self._origin = origin
        self._bytes_per_frame = 4 * channels
        self._container: av.container.InputContainer | None = None
        self._stream: av.AudioStream | None = None
        self._segments: Iterator[tuple[int, int, bytes]] | None = None
        self._pending: tuple[int, int, bytes] | None = None
        self._cursor: int | None = None
        self._decoded_frame_count = 0
        # After a seek, the first segment must start by this sample; None once
        # checked, or after a walk from the head.
        self._verify_at: int | None = None

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
                _raise_ffmpeg_operation("closing audio in", self._path, exc)

    def _decoded(self, skip_to: Fraction | None) -> Iterator[av.AudioFrame]:
        """The stream's frames, decoded from its first packet at or after ``skip_to``."""
        assert self._container is not None
        assert self._stream is not None
        if skip_to is None:
            yield from self._container.decode(self._stream)
            return
        time_base = Fraction(_time_base(self._stream))
        for packet in self._container.demux(self._stream):
            if skip_to is not None and packet.size > 0:
                if packet.pts is None or Fraction(int(packet.pts)) * time_base < skip_to:
                    continue
                skip_to = None
            yield from packet.decode()

    def _segment_iter(
        self,
        resampler: av.AudioResampler,
        skip_to: Fraction | None = None,
    ) -> Iterator[tuple[int, int, bytes]]:
        assert self._container is not None
        assert self._stream is not None
        fallback_start: int | None = None
        pending: tuple[int | None, Fraction, int, bytes] | None = None
        source_tick = Fraction(_time_base(self._stream)) * self._sample_rate

        def converted(frame: av.AudioFrame) -> tuple[int | None, Fraction, int, bytes]:
            self._decoded_frame_count += 1
            if frame.pts is not None and frame.time_base is not None:
                time_base = Fraction(frame.time_base.numerator, frame.time_base.denominator)
                reported_start = _audio_sample_position(
                    frame.pts,
                    time_base,
                    self._origin,
                    self._sample_rate,
                )
                # The resampler can advertise a finer clock than the source
                # container. Use the source tick when snapping its rounded PTS.
                tick = max(source_tick, time_base * self._sample_rate)
            else:
                reported_start = None
                tick = source_tick
            planes = [
                mlx_array_from_buffer(memoryview(plane).cast("B")).view(mx.float32)[: frame.samples]
                for plane in frame.planes
            ]
            if len(planes) != self._channels:
                raise RuntimeError(
                    f"audio channel layout changed while decoding "
                    f"{self._path}: expected {self._channels}, got "
                    f"{len(planes)}"
                )
            raw = bytes(memoryview(mx.contiguous(mx.stack(planes, axis=1))))
            return reported_start, tick, int(frame.samples), raw

        def placed(
            segment: tuple[int | None, Fraction, int, bytes],
            following: tuple[int | None, Fraction, int, bytes] | None,
        ) -> tuple[int, int, bytes]:
            nonlocal fallback_start
            reported, tick, samples, raw = segment
            if reported is None:
                if fallback_start is None:
                    raise RuntimeError(f"audio frame from {self._path} has no timestamp")
                start = fallback_start
            else:
                start = _audio_segment_start(reported, fallback_start, tick)
                if fallback_start is not None and start > fallback_start and following is not None:
                    next_reported, next_tick, _, _ = following
                    # Some Vorbis granule positions put one frame ahead, then
                    # return to the decoded sample count on the next frame.
                    # A one-frame lookahead distinguishes that pair from a
                    # sustained gap, which must remain silence on the timeline.
                    if next_reported is not None and abs(
                        next_reported - (fallback_start + samples)
                    ) <= _audio_clock_tolerance(next_tick):
                        start = fallback_start
            stop = start + samples
            # A backward edit retains the first decoded samples. read_frames
            # skips any overlapping prefix of this segment at its current
            # position, so never move the cumulative boundary backward.
            fallback_start = stop if fallback_start is None else max(fallback_start, stop)
            return start, stop, raw

        for frame in self._decoded(skip_to):
            for resampled in resampler.resample(frame):
                current = converted(resampled)
                if pending is not None:
                    yield placed(pending, current)
                pending = current
        for resampled in resampler.resample(None):
            current = converted(resampled)
            if pending is not None:
                yield placed(pending, current)
            pending = current
        if pending is not None:
            yield placed(pending, None)

    def _reset(self, target: int, *, walk: bool = False) -> None:
        self._close_cursor()
        container = _open_container(self._path, "cannot open audio in")
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if not streams:
            container.close()
            raise RuntimeError(f"no audio stream in {self._path}")
        stream = cast(av.AudioStream, streams[0])  # PyAV wraps type "audio" as AudioStream
        rate = int(stream.rate or stream.codec_context.sample_rate or 0)
        channels = len(stream.layout.channels)
        if (rate, channels, stream.layout.name) != (
            self._source_sample_rate,
            self._channels,
            self._layout,
        ):
            container.close()
            raise RuntimeError(f"audio format changed while opening {self._path}")
        # Compressed audio needs decoder preroll (AAC's overlapping transform
        # is the common case). Seek a fixed one-second envelope before the
        # requested sample, decode/discard it, and still retain only the
        # caller's bounded PCM pull. Seeking directly to the preceding packet
        # produces a measurable onset transient versus a full decode.
        seek_sample = max(0, target - self._sample_rate)
        target_time = self._origin + Fraction(seek_sample, self._sample_rate)
        skip_to: Fraction | None = None
        self._verify_at = None
        if seek_sample > 0 and (walk or not _seekable(container)):
            # Demux from the head, as a full decode does, and decode from the
            # envelope's start. After a seek in such a container the first
            # packet can be a torn frame, or the seek can land past the
            # envelope (see _SEEKABLE_FORMATS).
            skip_to = target_time
        elif seek_sample > 0:
            # A landing inside the envelope still costs the decoder its
            # preroll; allow the rounding of one container clock tick.
            tick = Fraction(_time_base(stream)) * self._sample_rate
            self._verify_at = seek_sample + _audio_clock_tolerance(tick)
            timestamp = int(target_time / Fraction(_time_base(stream)))
            try:
                container.seek(timestamp, stream=stream, backward=True, any_frame=False)
            except av.FFmpegError:
                # Some AVI PCM streams reject stream-relative seeks, while
                # container-level keyframe seeks still work. Both land before
                # the target; the cursor discards the decoded preroll.
                try:
                    container.seek(int(target_time * 1_000_000), backward=True, any_frame=False)
                except av.FFmpegError as exc:
                    with contextlib.suppress(Exception):
                        container.close()
                    _raise_ffmpeg_operation("seeking audio in", self._path, exc)
        resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=self._sample_rate)
        self._container = container
        self._stream = stream
        self._segments = iter(self._segment_iter(resampler, skip_to))
        self._pending = None
        self._cursor = target

    def read_frames(self, start_frame: int, end_frame: int) -> bytes:
        if end_frame <= start_frame:
            return b""
        if self._cursor != start_frame:
            self._reset(start_frame)
        assert self._segments is not None  # a set cursor always has its segment iterator
        output = bytearray()
        position = start_frame
        while position < end_frame:
            if self._pending is None or self._pending[1] <= position:
                try:
                    self._pending = next(self._segments)
                except av.FFmpegError as exc:
                    _raise_ffmpeg_operation("decoding audio from", self._path, exc)
                except StopIteration:
                    # Some containers omit stream.duration; the container's
                    # duration is only an upper bound when audio ends before
                    # video. Return a short final pull and let the writer/sidecar
                    # close the track cleanly at the actual decoded EOF.
                    self._cursor = position
                    return bytes(output)
                if self._verify_at is not None:
                    verify_at, self._verify_at = self._verify_at, None
                    if self._pending[0] > verify_at:
                        # The seek landed past the preroll envelope: the gap
                        # fill below would turn a landing past the requested
                        # start into silence, and one short of it leaves the
                        # decoder no preroll.
                        _log.info(
                            "%s: the audio seek for sample %d first decoded sample %d, "
                            "after sample %d; walking the packets from the head instead",
                            self._path.name,
                            position,
                            self._pending[0],
                            verify_at,
                        )
                        self._reset(position, walk=True)
                continue
            segment_start, segment_stop, raw = self._pending
            if segment_start > position:
                # Preserve timeline discontinuities as silence instead of
                # concatenating post-gap audio early. Limit the fill to this
                # caller's bounded request, so a long edit never becomes a
                # whole-track allocation.
                gap_stop = min(segment_start, end_frame)
                output.extend(b"\0" * ((gap_stop - position) * self._bytes_per_frame))
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
) -> StreamingAudioTrack | None:
    """Open a bounded lazy PCM window from the first audio stream.

    Resample uncommon source rates to 48 kHz for AVAssetWriter's ALAC/AAC
    encoders. An AVI can declare 11024 Hz, which neither writer accepts.
    """

    container = _open_container(path, "cannot open audio in")
    try:
        astreams = [s for s in container.streams if s.type == "audio"]
        if not astreams:
            return None
        ast = cast(av.AudioStream, astreams[0])  # PyAV wraps type "audio" as AudioStream
        origin = _audio_origin(container, ast)
        if origin is None:
            raise RuntimeError(f"audio stream in {path} has no usable presentation origin")
        source_sample_rate = int(ast.rate or ast.codec_context.sample_rate or 0)
        channels = len(ast.layout.channels)
        if source_sample_rate <= 0 or channels <= 0:
            return None
        sample_rate = source_sample_rate if source_sample_rate in _WRITER_AUDIO_RATES else 48_000
        if sample_rate != source_sample_rate:
            _log.info(
                "resampling audio from %s to %s Hz for output", source_sample_rate, sample_rate
            )
        total: int | None = None
        if ast.duration is not None:
            total = round(Fraction(ast.duration) * Fraction(_time_base(ast)) * sample_rate)
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
            raise RuntimeError(f"audio stream in {path} has no usable duration") from exc
        if stop == start:
            return None
        layout = ast.layout.name

        def source_factory() -> _FFmpegAudioSource:
            return _FFmpegAudioSource(
                path,
                sample_rate,
                channels,
                layout,
                origin,
                source_sample_rate=source_sample_rate,
            )

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


def read_audio_track(path: Path) -> StreamingAudioTrack | None:
    """Compatibility entry point for callers requesting the complete track."""
    return read_audio_track_window(path)


__all__ = [
    "iter_forced_color_chunks",
    "iter_video_buffer_chunks",
    "keyframe_display_indices",
    "probe_audio_timing",
    "probe_color",
    "probe_stream_descriptor",
    "probe_video",
    "probe_video_timing",
    "read_audio_track",
    "read_audio_track_window",
    "read_sample_table",
]
