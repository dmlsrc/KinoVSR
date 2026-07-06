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
    read_audio_track

Frames exit as IOSurface-backed CVPixelBuffers built with the same helpers the
native path uses, decoded via libavcodec and converted by libswscale honoring
(or, for the forced-color path, overriding) the stream's color tags. PyAV is
lazy-imported so the native path never touches it; install the `ffmpeg` extra
to enable this module. No numpy: plane bytes cross into MLX through the buffer
protocol.

Rotation/flip display matrices propagate as CGAffineTransforms matching the
AVFoundation preferredTransform convention (verified component-wise against
the native probe at 90/180/270). >8-bit sources are read through rgb48le
(full precision kept through the fp16 pixel buffer).
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlx.core as mx

from . import pixel_buffers as _pb
from .pixel_buffers import PIX_BGRA, PIX_RGBAHALF


def _av():
    try:
        import av
    except ImportError as e:   # pragma: no cover - environment-dependent
        raise RuntimeError(
            "the ffmpeg compatibility reader needs PyAV; install the 'ffmpeg' "
            "extra (uv pip install 'av>=18')"
        ) from e
    return av


def _open_video(path: Path):
    av = _av()
    container = av.open(str(path))
    streams = [s for s in container.streams if s.type == "video"]
    if not streams:
        container.close()
        raise RuntimeError(f"No video track in {path}")
    return container, streams[0]


def _fps(stream: Any) -> float:
    r = stream.average_rate or stream.guessed_rate or stream.base_rate
    return float(r) if r else 0.0


def _pts_index(pts: int, stream: Any, fps: float) -> int:
    """Display index of a pts, 0-based at the stream's first displayed frame.

    Some containers (FLV, ASF) start their display timeline at a nonzero pts;
    subtracting stream.start_time keeps keyframe indices and trim windows
    aligned with decode-order frame counting.
    """
    start = stream.start_time or 0
    return int(round(float((pts - start) * stream.time_base) * fps))


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
    finally:
        container.close()


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


def keyframe_display_indices(path: Path) -> list[int]:
    """Display-order keyframe indices from packet metadata (no decode).

    Mirrors the native sync-sample scan: demux the coded packets (decode
    order), keep the keyframes, map each to its display index via its pts.
    All-intra codecs report every frame; codecs with no keyframe concept
    degrade to [0], and the gop planner handles both shapes.
    """
    container, vs = _open_video(path)
    try:
        fps = _fps(vs)
        if fps <= 0:
            return [0]
        kf: set[int] = set()
        for pkt in container.demux(vs):
            if pkt.pts is None or not pkt.is_keyframe:
                continue
            kf.add(_pts_index(pkt.pts, vs, fps))
        return sorted(kf) if kf else [0]
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
                 reformat_kwargs: dict) -> Iterator[list]:
    container, vs = _open_video(path)
    try:
        fps = _fps(vs)
        attrs = _buffer_attrs(out_format)
        if start_frame > 0 and fps > 0:
            # coarse keyframe seek, then exact per-frame trim below
            sec = max(0.0, (start_frame - 1) / fps)
            container.seek(int(sec / vs.time_base), stream=vs, backward=True)
        chunk: list = []
        for frame in container.decode(vs):
            if frame.pts is None:
                continue
            idx = _pts_index(frame.pts, vs, fps) if fps > 0 else 0
            if idx < start_frame:
                continue
            if end_frame is not None and idx >= end_frame:
                break
            chunk.append(_frame_to_buffer(frame, out_format, attrs, reformat_kwargs))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
    finally:
        container.close()


def iter_video_buffer_chunks(
    path: Path, src_format: int, chunk_size: int = 8,
    *, start_frame: int = 0, end_frame: int | None = None,
) -> Iterator[list]:
    """Native-surface mirror: lists of CVPixelBuffers in src_format, honoring
    the stream's own color tags (libswscale reads them off each frame)."""
    return _iter_chunks(path, src_format, chunk_size, start_frame, end_frame, {})


def iter_forced_color_chunks(
    path: Path, out_format: int, matrix_cv: Any, full_range: bool,
    chunk_size: int = 8, *, start_frame: int = 0, end_frame: int | None = None,
    reinterpret_full_range: bool | None = None,
) -> Iterator[list]:
    """Forced-matrix read: override the YCbCr matrix / range at YUV->RGB time
    (libswscale src_colorspace/src_color_range), the --source-color fix for
    untagged or mis-tagged material."""
    av = _av()
    m = str(matrix_cv)
    cs = av.video.reformatter.Colorspace.ITU709 if "709" in m else \
        av.video.reformatter.Colorspace.ITU601
    rng_full = full_range if reinterpret_full_range is None else reinterpret_full_range
    rng = av.video.reformatter.ColorRange.JPEG if rng_full \
        else av.video.reformatter.ColorRange.MPEG
    kw = {"src_colorspace": cs, "src_color_range": rng}
    return _iter_chunks(path, out_format, chunk_size, start_frame, end_frame, kw)


# ---------------------------------------------------------------- audio
def read_audio_track(path: Path) -> Any | None:
    """Decode the first audio stream to an in-memory AudioTrack (fp32 PCM),
    the same object the native audio path produces. None when no audio."""
    from .audio import AudioTrack
    av = _av()
    container = av.open(str(path))
    try:
        astreams = [s for s in container.streams if s.type == "audio"]
        if not astreams:
            return None
        ast = astreams[0]
        resampler = av.AudioResampler(format="fltp", layout=ast.layout, rate=ast.rate)
        chans: list[list] = []
        for frame in container.decode(ast):
            for rf in resampler.resample(frame):
                planes = [mx.array(memoryview(p).cast("B")).view(mx.float32)[: rf.samples]
                          for p in rf.planes]
                chans.append(planes)
        if not chans:
            return None
        n_ch = len(chans[0])
        wave = mx.concatenate(
            [mx.concatenate([c[i] for c in chans])[None] for i in range(n_ch)], axis=0)
        return AudioTrack(wave, int(ast.rate))
    finally:
        container.close()


__all__ = [
    "probe_video", "probe_color", "keyframe_display_indices",
    "iter_video_buffer_chunks", "iter_forced_color_chunks", "read_audio_track",
]
