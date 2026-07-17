"""Native AVFoundation video decode for the `--video` VSR path.

`AVURLAsset` + `AVAssetReader` pull decoded frames straight out of the
container in VSR's own source pixel format (NV12 for the LowLatency `fast`
scaler, RGBAHalf for the HighQuality `balanced`/`image` scalers). The decoded
`CVPixelBuffer` is fed directly into VSR via `upscale_buffer_to_buffer` - no
intermediate RGB array, no re-quantization, no per-frame copy through MLX.

This preserves the source's bit depth and chroma:
  - fast: source YUV -> NV12 is a memory-layout change only, no color or
    chroma conversion (8-bit is the LowLatency scaler's ceiling regardless).
  - balanced/image: source YUV (including 10-bit 4:2:2 / 4:2:0) -> RGBAHalf is
    a single decode-time conversion at half-float precision and 4:4:4, so
    10-bit sources keep their precision instead of being clamped through an
    8-bit RGB intermediate.

A track's preferredTransform (rotation / flip) is returned by `probe_video`
and propagated to the output as container metadata by the writer, so rotated
inputs display correctly without ever rotating pixels - lossless.

No ffmpeg, no numpy.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

from kinovsr.native.frameworks import CoreMedia, Foundation, Quartz, av, vt

from . import pixel_buffers as _pb
from .chunks import validate_decode_chunk_size
from .timing import (
    AudioTiming,
    EpochUnwrapper,
    SampleTable,
    SampleTiming,
    VideoTiming,
    analyze_sample_table,
)

_log = logging.getLogger(__name__)


def _first_video_track(asset: Any) -> Any:
    tracks = asset.tracksWithMediaType_(av.AVMediaTypeVideo)
    if tracks is None or len(tracks) == 0:
        raise RuntimeError("no video track in asset")
    return tracks[0]


def _video_codec_fourcc(track: Any) -> str:
    """The track's codec as a 4-char tag ('hvc1', 'hev1', 'avc1', ...), or ''."""
    fmts = track.formatDescriptions()
    if not fmts or len(fmts) == 0:
        return ""
    code = CoreMedia.CMFormatDescriptionGetMediaSubType(fmts[0])
    return bytes((code >> s) & 0xFF for s in (24, 16, 8, 0)).decode("latin-1")


def _assert_decodable(track: Any, path: Path) -> None:
    """Reject 'hev1'-tagged HEVC up front with an actionable message.

    ffmpeg muxes HEVC into MP4 as 'hev1' by default, but AVFoundation can only
    decode 'hvc1' (parameter sets carried out of band in the hvcC box). An hev1
    track otherwise fails deep in the reader with a cryptic -11833 'Cannot
    Decode'; the fix is a lossless container re-tag, no re-encode."""
    if _video_codec_fourcc(track) == "hev1":
        out = path.with_name(f"{path.stem}_hvc1.mp4")
        raise RuntimeError(
            f"{path.name}: HEVC video is tagged 'hev1', which AVFoundation cannot "
            f"decode - it requires 'hvc1' (parameter sets out of band). Re-tag it "
            f"losslessly (no re-encode), then use the result:\n"
            f"    ffmpeg -i '{path}' -c copy -tag:v hvc1 '{out}'"
        )


def _encoded_dimensions(track: Any) -> tuple[int, int]:
    """Stored raster dimensions for the first video format description.

    ``AVAssetTrack.naturalSize`` is a presentation/display size. For anamorphic
    SD sources it can include pixel-aspect expansion (e.g. 352x288 with SAR
    128:117 reports as ~385x288), while ``AVAssetReader`` still decodes
    352x288 pixel buffers. VSR/model geometry must follow the encoded raster.
    """
    fmts = track.formatDescriptions()
    if fmts and len(fmts) > 0:
        dims = CoreMedia.CMVideoFormatDescriptionGetDimensions(fmts[0])
        w, h = int(dims.width), int(dims.height)
        if w > 0 and h > 0:
            return w, h
    size = track.naturalSize()
    return int(round(size.width)), int(round(size.height))


def _pixel_aspect_ratio(track: Any) -> tuple[int, int] | None:
    """Return source pixel aspect ratio as (horizontal, vertical), if tagged."""
    fmts = track.formatDescriptions()
    if not fmts or len(fmts) == 0:
        return None
    ext = CoreMedia.CMFormatDescriptionGetExtensions(fmts[0]) or {}
    par = ext.get(CoreMedia.kCMFormatDescriptionExtension_PixelAspectRatio)
    if not par:
        return None
    h = int(par.get(CoreMedia.kCMFormatDescriptionKey_PixelAspectRatioHorizontalSpacing, 0))
    v = int(par.get(CoreMedia.kCMFormatDescriptionKey_PixelAspectRatioVerticalSpacing, 0))
    if h <= 0 or v <= 0 or h == v:
        return None
    return h, v


def probe_video(path: Path) -> tuple[int, int, float, int, Any, tuple[int, int] | None]:
    """(width, height, fps, n_frames, transform, pixel_aspect) for the first track.

    Dimensions are encoded raster pixels, not display/presentation size.
    `transform` is the track's
    preferredTransform (a CGAffineTransform) - identity for upright content,
    a rotation/flip for camera footage; the writer applies it as output
    metadata so pixels never need rotating. n_frames is round(duration * fps),
    exact for constant-frame-rate content (everything VSR consumes).
    ``pixel_aspect`` carries anamorphic display geometry to the writer.
    """
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    _assert_decodable(track, path)
    w, h = _encoded_dimensions(track)
    fps = float(track.nominalFrameRate())
    duration = CoreMedia.CMTimeGetSeconds(asset.duration())
    n = int(round(duration * fps)) if fps > 0 else 0
    transform = track.preferredTransform()
    pixel_aspect = _pixel_aspect_ratio(track)
    return w, h, fps, n, transform, pixel_aspect


def _cm_time_fraction(value: Any) -> Fraction | None:
    """Convert a numeric CMTime to an exact Fraction, rejecting sentinels."""
    timescale = int(value.timescale)
    if timescale <= 0:
        return None
    return Fraction(int(value.value), timescale)


def read_sample_table(path: Path) -> SampleTable:
    """One metadata-only walk of the video track's coded samples.

    Collects every display sample's exact presentation time and duration
    plus its sync flag (``kCMSampleAttachmentKey_NotSync`` absent/false)
    and coded payload size, then classifies the timing as one grid,
    gapped, or variable. The compressed track output performs no decode.
    Output presentation time is intentional: unlike the raw sample PTS,
    it includes the asset's edit mapping and is the clock AVAssetReader
    presents to decoded consumers.
    """
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    _assert_decodable(track, path)
    reader, err = av.AVAssetReader.alloc().initWithAsset_error_(asset, None)
    if reader is None:
        raise RuntimeError(f"AVAssetReader init failed: {err}")
    output = av.AVAssetReaderTrackOutput.alloc().initWithTrack_outputSettings_(
        track, None)
    if not reader.canAddOutput_(output):
        raise RuntimeError("AVAssetReader cannot expose compressed timing")
    reader.addOutput_(output)
    if not reader.startReading():
        raise RuntimeError(f"AVAssetReader.startReading failed: {reader.error()}")

    samples: list[SampleTiming] = []
    natural_timescale = int(track.naturalTimeScale())
    source_tick = (Fraction(1, natural_timescale)
                   if natural_timescale > 0 else None)
    unwrapper = EpochUnwrapper()
    while True:
        sample = output.copyNextSampleBuffer()
        if sample is None:
            break
        # AVAssetReader can return zero-sample marker buffers around edit
        # boundaries. They are not display frames and carry invalid times.
        if CoreMedia.CMSampleBufferGetNumSamples(sample) < 1:
            del sample
            continue
        pts_time = CoreMedia.CMSampleBufferGetOutputPresentationTimeStamp(sample)
        duration_time = CoreMedia.CMSampleBufferGetOutputDuration(sample)
        pts = _cm_time_fraction(pts_time)
        duration = _cm_time_fraction(duration_time)
        if pts is not None:
            pts = unwrapper.push(pts)
            attachments = CoreMedia.CMSampleBufferGetSampleAttachmentsArray(
                sample, False)
            is_sync = True
            if attachments and len(attachments) > 0:
                is_sync = not bool(attachments[0].get(
                    CoreMedia.kCMSampleAttachmentKey_NotSync))
            coded_size = int(CoreMedia.CMSampleBufferGetTotalSampleSize(sample))
            samples.append(SampleTiming(
                pts=pts,
                duration=duration if duration and duration > 0 else None,
                is_sync=is_sync,
                coded_size=coded_size if coded_size > 0 else None,
            ))
            if natural_timescale <= 0:
                tick = Fraction(1, int(pts_time.timescale))
                source_tick = tick if source_tick is None else min(source_tick, tick)
        del sample

    if reader.error() is not None:
        raise RuntimeError(f"AVAssetReader timing scan failed: {reader.error()}")
    if source_tick is None:
        raise RuntimeError(f"{path.name}: video track contains no display samples")
    if unwrapper.resets:
        _log.info(
            "%s: unwrapped %d mid-file timestamp reset(s) into one "
            "declared monotonic clock", path.name, unwrapper.resets)
    try:
        return analyze_sample_table(
            samples,
            nominal_cadence=float(track.nominalFrameRate()),
            source_tick=source_tick,
            epoch_resets=unwrapper.resets,
        )
    except ValueError as exc:
        raise RuntimeError(f"{path.name}: {exc}") from exc


def probe_video_timing(path: Path) -> VideoTiming:
    """Read exact display PTS metadata and classify the track as CFR or VFR.

    The legacy CFR-or-variable view over :func:`read_sample_table`'s walk.
    """
    return read_sample_table(path).timing()


def probe_audio_timing(path: Path) -> AudioTiming | None:
    """Return the first edit-adjusted audio PTS without decoding audio.

    Audio carry currently owns PCM samples but not their source timestamps.
    The file endpoint uses this metadata probe to reject staggered track
    origins before it would silently place both tracks at output time zero.
    """
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    tracks = asset.tracksWithMediaType_(av.AVMediaTypeAudio)
    if tracks is None or len(tracks) == 0:
        return None
    track = tracks[0]
    reader, err = av.AVAssetReader.alloc().initWithAsset_error_(asset, None)
    if reader is None:
        raise RuntimeError(f"AVAssetReader init failed: {err}")
    output = av.AVAssetReaderTrackOutput.alloc().initWithTrack_outputSettings_(
        track, None)
    if not reader.canAddOutput_(output):
        raise RuntimeError("AVAssetReader cannot expose compressed audio timing")
    reader.addOutput_(output)
    if not reader.startReading():
        raise RuntimeError(f"AVAssetReader.startReading failed: {reader.error()}")

    while True:
        sample = output.copyNextSampleBuffer()
        if sample is None:
            break
        if CoreMedia.CMSampleBufferGetNumSamples(sample) < 1:
            del sample
            continue
        pts_time = CoreMedia.CMSampleBufferGetOutputPresentationTimeStamp(sample)
        pts = _cm_time_fraction(pts_time)
        timescale = int(pts_time.timescale)
        del sample
        if pts is not None and timescale > 0:
            reader.cancelReading()
            return AudioTiming(
                first_pts=pts,
                source_tick=Fraction(1, timescale),
            )

    if reader.error() is not None:
        raise RuntimeError(f"AVAssetReader audio timing scan failed: {reader.error()}")
    return None


def keyframe_display_indices(
    path: Path,
    *,
    timing: VideoTiming | None = None,
) -> list[int]:
    """Display-order frame indices of the source's keyframes (sync samples).

    Table positions from the consolidated sample walk
    (:func:`read_sample_table`): a sample is a keyframe iff its
    ``kCMSampleAttachmentKey_NotSync`` attachment is absent/false, and
    its index is its position in display order - exact under jitter,
    gaps, and variable clocks, where the retired cadence-grid mapping
    (``round((pts - origin) * cadence)``) desynced from sample ordinals
    after any dropped frame. ``timing`` is accepted for adapter-call
    compatibility and unused. Returns a sorted list; a clip with no
    reported sync samples returns ``[0]``.
    """
    del timing
    return list(read_sample_table(path).keyframe_indices) or [0]


def coded_frame_sizes(path: Path) -> list[int]:
    """Per-frame coded sizes in DISPLAY order (bytes), no decode.

    One entry per display SAMPLE from the consolidated walk
    (:func:`read_sample_table`); unknown sizes read 0. The retired
    grid-slot version placed sizes at ``round(pts * fps)`` positions,
    inserting phantom zero entries at dropped-frame slots and misaligning
    sizes with decoded frame ordinals on gapped sources. Coded size is a
    LAST-GENERATION signal: a generous re-encode of damaged footage reads
    large. Returns [] when the track cannot be walked.
    """
    try:
        table = read_sample_table(path)
    except RuntimeError:
        return []
    return [sample.coded_size or 0 for sample in table.samples]


def probe_stream_descriptor(path: Path) -> dict:
    """Coarse raw-stream identity for artifact-appropriate processing.

    ``{"codec": fourcc-or-name, "profile": str | None,
    "bit_depth": int | None, "chroma": "420" | "422" | "444" | None}``
    with ``None`` where the container does not say. The native format
    description exposes the codec type cheaply; depth/subsampling
    extensions are present only for some codecs, so the ffmpeg reader's
    descriptor is the richer of the two.
    """
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    descs = track.formatDescriptions()
    if not descs or len(descs) == 0:
        return {"codec": None, "profile": None,
                "bit_depth": None, "chroma": None}
    desc = descs[0]
    subtype = int(CoreMedia.CMFormatDescriptionGetMediaSubType(desc))
    codec = subtype.to_bytes(4, "big").decode("ascii", "replace").strip()
    bit_depth = None
    extensions = CoreMedia.CMFormatDescriptionGetExtensions(desc) or {}
    depth = extensions.get("BitsPerComponent")
    if depth is not None:
        bit_depth = int(depth)
    return {"codec": codec, "profile": None,
            "bit_depth": bit_depth, "chroma": None}


def probe_color(path: Path) -> dict:
    """Source color tags from the container, for output color propagation.

    Returns explicit primaries/transfer/matrix + full-range flag (None where
    untagged) so the encoder can tag the output to match the source instead of a
    hard-coded BT.709. See kinovsr/color.py.
    """
    from . import color
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    src = color.read_source_color(track.formatDescriptions()[0])
    if not src["tagged"]:
        # Untagged: read VideoToolbox's decode-time guess (its undocumented,
        # width-keyed choice) off a decoded frame, so 'auto' reports/tags what was
        # actually read instead of assuming BT.709.
        try:
            buf = next(iter(iter_video_buffer_chunks(path, _pb.PIX_RGBAHALF, chunk_size=1)))[0]
            att = Quartz.CVBufferCopyAttachments(buf, Quartz.kCVAttachmentMode_ShouldPropagate) or {}
            by = {str(k): att[k] for k in att}
            src["primaries"] = by.get("CVImageBufferColorPrimaries") or src["primaries"]
            src["transfer"] = by.get("CVImageBufferTransferFunction") or src["transfer"]
            src["matrix"] = by.get("CVImageBufferYCbCrMatrix") or src["matrix"]
            src["guessed"] = src["matrix"] is not None
        except Exception:
            pass
    return src


# 10-bit 4:2:2 YUV for the forced-decode path: a precision-preserving superset
# (4:2:0 / 8-bit sources upsample into it losslessly).
_YUV10_VIDEO = Quartz.kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
_YUV10_FULL = Quartz.kCVPixelFormatType_422YpCbCr10BiPlanarFullRange


def _retype_range_copy(yuv: Any, dst_fmt: int) -> Any:
    """Byte-identical copy of a planar YUV buffer into a buffer typed `dst_fmt`.

    The video/full-range variants of a YUV pixel format share one plane layout;
    only the format code (the range *interpretation*) differs. Copying the
    planes unchanged into a buffer of the other range type is what makes
    --source-range a REINTERPRETATION: asking the decoder for the other range
    format instead would RESCALE the code values -- exactly wrong when the
    container's flag is the thing that's mistaken.
    """
    w = Quartz.CVPixelBufferGetWidth(yuv)
    h = Quartz.CVPixelBufferGetHeight(yuv)
    dst = _pb.make_pixel_buffer_from_attrs(w, h, {
        Quartz.kCVPixelBufferPixelFormatTypeKey: dst_fmt,
        Quartz.kCVPixelBufferWidthKey: w, Quartz.kCVPixelBufferHeightKey: h,
        Quartz.kCVPixelBufferIOSurfacePropertiesKey: {}})
    Quartz.CVPixelBufferLockBaseAddress(yuv, 1)  # 1 = read-only lock
    Quartz.CVPixelBufferLockBaseAddress(dst, 0)
    try:
        for p in range(Quartz.CVPixelBufferGetPlaneCount(yuv)):
            rows = Quartz.CVPixelBufferGetHeightOfPlane(yuv, p)
            sbpr = Quartz.CVPixelBufferGetBytesPerRowOfPlane(yuv, p)
            dbpr = Quartz.CVPixelBufferGetBytesPerRowOfPlane(dst, p)
            smv = Quartz.CVPixelBufferGetBaseAddressOfPlane(yuv, p).as_buffer(rows * sbpr)
            dmv = Quartz.CVPixelBufferGetBaseAddressOfPlane(dst, p).as_buffer(rows * dbpr)
            if sbpr == dbpr:
                dmv[:] = smv
            else:
                row = min(sbpr, dbpr)
                for r in range(rows):
                    dmv[r * dbpr:r * dbpr + row] = smv[r * sbpr:r * sbpr + row]
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(dst, 0)
        Quartz.CVPixelBufferUnlockBaseAddress(yuv, 1)
    return dst


def iter_forced_color_chunks(
    path: Path, out_format: int, matrix_cv: Any, full_range: bool,
    chunk_size: int = 8, *, start_frame: int = 0, end_frame: int | None = None,
    reinterpret_full_range: bool | None = None,
    timing: VideoTiming | None = None,
    table: SampleTable | None = None,
) -> Iterator[list]:
    """Decode the source as raw 10-bit YUV, FORCE the YCbCr matrix (overriding the
    container tag / VideoToolbox's resolution-based guess), then convert to
    out_format ourselves.

    This makes --source-color control how the source is READ -- the fix for the
    untagged SD clips VideoToolbox mis-guesses as BT.601. The 4:2:2/10-bit YUV is a
    precision-preserving superset of any SDR source.

    `full_range` must be the CONTAINER's flag: the YUV decode requests the
    matching range format so the code values pass through unrescaled.
    `reinterpret_full_range` (--source-range) then presents those same values
    to the YUV->RGB conversion under a different range identity via a
    byte-identical retype copy; None or equal-to-container means no override.
    """
    chunk_size = validate_decode_chunk_size(chunk_size)
    yuv_fmt = _YUV10_FULL if full_range else _YUV10_VIDEO
    retype_fmt = None
    if reinterpret_full_range is not None and reinterpret_full_range != full_range:
        retype_fmt = _YUV10_FULL if reinterpret_full_range else _YUV10_VIDEO
    err, xfer = vt.VTPixelTransferSessionCreate(None, None)
    if err != 0 or xfer is None:
        raise RuntimeError(f"VTPixelTransferSessionCreate failed: {err}")
    for chunk in iter_video_buffer_chunks(
            path, yuv_fmt, chunk_size,
            start_frame=start_frame, end_frame=end_frame, timing=timing,
            table=table):
        out: list = []
        for item in chunk:
            yuv, idx = item if isinstance(item, tuple) else (item, None)
            if retype_fmt is not None:
                yuv = _retype_range_copy(yuv, retype_fmt)
            Quartz.CVBufferSetAttachment(
                yuv, Quartz.kCVImageBufferYCbCrMatrixKey, matrix_cv,
                Quartz.kCVAttachmentMode_ShouldPropagate)
            w = Quartz.CVPixelBufferGetWidth(yuv)
            h = Quartz.CVPixelBufferGetHeight(yuv)
            dst = _pb.make_pixel_buffer_from_attrs(w, h, {
                Quartz.kCVPixelBufferPixelFormatTypeKey: out_format,
                Quartz.kCVPixelBufferWidthKey: w, Quartz.kCVPixelBufferHeightKey: h,
                Quartz.kCVPixelBufferIOSurfacePropertiesKey: {}})
            e = vt.VTPixelTransferSessionTransferImage(xfer, yuv, dst)
            if e != 0:
                raise RuntimeError(f"forced-color YUV->{out_format:#x} transfer failed: {e}")
            out.append((dst, idx) if idx is not None else dst)
        yield out


def iter_video_buffer_chunks(
    path: Path, src_format: int, chunk_size: int = 8,
    *, start_frame: int = 0, end_frame: int | None = None,
    timing: VideoTiming | None = None,
    table: SampleTable | None = None,
) -> Iterator[list]:
    """Yield lists of up to `chunk_size` decoded CVPixelBuffers in `src_format`.

    Each buffer is IOSurface-backed and ready to feed straight into
    `VsrSession.upscale_buffer_to_buffer`. Decode is pull-based - the reader
    produces one frame at a time - so peak resident memory is bounded by
    `chunk_size` decoded frames (the harness sizes this to a memory budget and
    frees each frame as it is consumed).

    `start_frame`/`end_frame` trim the input to the half-open frame window
    [start_frame, end_frame) (end_frame=None means to the end). The reader's
    timeRange is seeked just before start_frame so the bulk of a long clip
    before the window is never decoded; the exact window boundary is then
    enforced per frame by presentation timestamp, so trimming is frame-exact
    even though the seek is approximate.

    With a `table` (the sample-table walk), window indexing bisects the real
    display timestamps - exact for jittered, gapped, and variable clocks -
    and the seek targets the window's true start time; without one it falls
    back to cadence-grid arithmetic, which requires a constant rate.

    The decoded CVPixelBuffer is retained independently of its owning
    CMSampleBuffer (pyobjc holds it for the wrapper's lifetime), so the sample
    buffer is released immediately after the image buffer is extracted; the
    image buffer stays valid until the consumer drops its reference.
    """
    chunk_size = validate_decode_chunk_size(chunk_size)
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    asset = av.AVURLAsset.alloc().initWithURL_options_(url, None)
    track = _first_video_track(asset)
    _assert_decodable(track, path)
    if table is not None and timing is None:
        timing = table.timing()
    keys = ([sample.pts for sample in table.samples]
            if table is not None else None)
    cadence = (timing.cadence if timing is not None
               else Fraction(float(track.nominalFrameRate())).limit_denominator(1001))
    if keys is None and (cadence is None or cadence <= 0):
        raise RuntimeError("video track reports no usable constant cadence")
    fps = float(cadence) if cadence is not None and cadence > 0 else 0.0
    origin = timing.first_pts if timing is not None else Fraction()

    reader, err = av.AVAssetReader.alloc().initWithAsset_error_(asset, None)
    if reader is None:
        raise RuntimeError(f"AVAssetReader init failed: {err}")

    trimming = start_frame > 0 or end_frame is not None
    # Multi-epoch sources decode linearly: a coarse seek landing inside a
    # later epoch would start the decode-side unwrapper without the
    # history that maps its raw stamps onto the table's monotonic clock.
    seekable_clock = table is None or table.epoch_resets == 0
    if trimming and seekable_clock and (keys is not None or fps > 0):
        # Seek the reader's timeRange to just before the window so the head of a
        # long clip isn't decoded. Back off one frame; the per-frame PTS check
        # below enforces the exact start. Compute the end from the asset
        # duration when the window is open-ended.
        ts = 24000
        if keys is not None:
            back = min(max(start_frame - 1, 0), len(keys) - 1)
            start_seconds = float(keys[back])
        else:
            start_seconds = float(origin) + (start_frame - 1) / fps
        # AVAssetReader time ranges are asset-time ranges. For a legitimate
        # negative edit origin, clamping to zero would skip source frames;
        # leave the range open and let the exact PTS trim below do the work.
        if start_seconds >= 0:
            start_t = CoreMedia.CMTimeMake(int(round(start_seconds * ts)), ts)
            if end_frame is None:
                dur_seconds = max(
                    0.0,
                    CoreMedia.CMTimeGetSeconds(asset.duration()) - start_seconds,
                )
            elif keys is not None:
                # Cover through the first excluded stamp plus slack; the
                # per-frame trim below is what enforces exactness.
                stop = min(end_frame, len(keys) - 1)
                dur_seconds = max(float(keys[stop]) - start_seconds, 0.0) + 1.0
            else:
                dur_seconds = (end_frame - start_frame + 2) / fps
            dur_t = CoreMedia.CMTimeMake(int(round(dur_seconds * ts)), ts)
            reader.setTimeRange_(CoreMedia.CMTimeRangeMake(start_t, dur_t))

    # Request IOSurface-backed, Metal-compatible buffers. Feeding the decoded
    # buffer straight to VSR bypasses the VSR source pool's attributes, so we
    # must ask for GPU-usable backing here instead - otherwise the Metal-based
    # super-resolution processor can reject the source frame (notably the
    # LowLatency 'fast' path: VTFrameProcessor error -19730).
    output = av.AVAssetReaderTrackOutput.alloc().initWithTrack_outputSettings_(
        track, {
            Quartz.kCVPixelBufferPixelFormatTypeKey: src_format,
            Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
            Quartz.kCVPixelBufferMetalCompatibilityKey: True,
        },
    )
    # Keep alwaysCopiesSampleData=YES (the default): we hold decoded buffers
    # past the copyNextSampleBuffer call - across a chunk, and across one
    # iteration for balanced mode's prev-frame chain - and feed them straight
    # into VSR. With NO, AVAssetReader can hand back references to volatile
    # decoder memory that gets recycled while we still reference it, which
    # corrupts the prev frame (flicker / dark output) or yields an invalid
    # source buffer (VTFrameProcessor -19730). The copy is one memcpy of the
    # already-decoded frame, in VSR's source format - no RGB conversion, cheap
    # next to the decode - so the fidelity and no-MLX-round-trip wins stand.
    output.setAlwaysCopiesSampleData_(True)
    if not reader.canAddOutput_(output):
        raise RuntimeError(
            f"AVAssetReader cannot output pixel format {src_format:#x}"
        )
    reader.addOutput_(output)
    if not reader.startReading():
        raise RuntimeError(f"AVAssetReader.startReading failed: {reader.error()}")

    chunk: list = []
    # The table's stamps are on the unwrapped monotonic clock; decoded
    # samples arrive with raw stamps, so the decode path runs the same
    # streaming mapping. Every sample pushes (kept or not) to keep the
    # two mappings aligned. With a table, each kept buffer is yielded as
    # a (buffer, table_index) pair so the consumer labels frames by
    # SAMPLE IDENTITY instead of arrival ordinal - a decoder that drops
    # or multiplies frames can then no longer shift every later label.
    trim_unwrapper = EpochUnwrapper() if keys is not None else None
    while True:
        sample_buf = output.copyNextSampleBuffer()
        if sample_buf is None:
            break
        image_buf = CoreMedia.CMSampleBufferGetImageBuffer(sample_buf)
        keep = image_buf is not None
        idx: int | None = None
        if keys is not None or (trimming and fps > 0):
            # Frame-exact window enforcement by presentation timestamp.
            pts = _cm_time_fraction(
                CoreMedia.CMSampleBufferGetOutputPresentationTimeStamp(
                    sample_buf))
            if pts is not None and trim_unwrapper is not None:
                pts = trim_unwrapper.push(pts)
            if pts is None:
                idx = 0
            elif keys is not None:
                idx = bisect.bisect_right(keys, pts) - 1
            else:
                idx = round((pts - origin) * cadence)
            if trimming and keep and idx < start_frame:
                keep = False
            elif trimming and keep and end_frame is not None \
                    and idx >= end_frame:
                del sample_buf
                break
        # Release the owning sample buffer now; the image buffer outlives it.
        del sample_buf
        if keep:
            chunk.append((image_buf, idx) if keys is not None
                         else image_buf)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

    if reader.status() == av.AVAssetReaderStatusFailed:
        raise RuntimeError(f"AVAssetReader failed: {reader.error()}")
    if chunk:
        yield chunk
