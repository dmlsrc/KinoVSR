"""In-memory and bounded file-backed PCM for AVAssetWriter."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import mlx.core as mx

from kinovsr.media.errors import is_native_operation_error
from kinovsr.native.frameworks import CoreAudio, CoreMedia

# CoreAudio FormatID constants (avoid importing the whole module just for these)
AUDIO_FORMAT_LPCM = 1819304813     # 'lpcm' kAudioFormatLinearPCM
AUDIO_FORMAT_AAC = 1633772320      # 'aac ' kAudioFormatMPEG4AAC
AUDIO_FORMAT_ALAC = 1634492771     # 'alac' kAudioFormatAppleLossless

_log = logging.getLogger(__name__)

_SIDECAR_PCM_BUDGET = 4 * 1024 * 1024


def _audio_format_description(sample_rate: int, channels: int) -> Any:
    bytes_per_frame = 4 * channels
    asbd = CoreAudio.AudioStreamBasicDescription(
        float(sample_rate),
        AUDIO_FORMAT_LPCM,
        CoreAudio.kAudioFormatFlagIsFloat | CoreAudio.kAudioFormatFlagIsPacked,
        bytes_per_frame,   # mBytesPerPacket
        1,                 # mFramesPerPacket
        bytes_per_frame,   # mBytesPerFrame
        channels,
        32,                # mBitsPerChannel
        0,
    )
    err, fmt = CoreMedia.CMAudioFormatDescriptionCreate(
        None, asbd, 0, None, 0, None, None, None,
    )
    if err != 0 or fmt is None:
        raise RuntimeError(
            f"CMAudioFormatDescriptionCreate failed: status={err}")
    return fmt


def _seconds_fraction(value: Fraction | int | float) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def _sample_index(seconds: Fraction | int | float, sample_rate: int) -> int:
    return round(_seconds_fraction(seconds) * sample_rate)


def _sample_window(
    *,
    sample_rate: int,
    total_samples: int | None,
    start_sec: Fraction | int | float,
    end_sec: Fraction | int | float | None,
    max_duration_sec: Fraction | int | float | None,
) -> tuple[int, int]:
    """Resolve absolute bounds, preserving cap-relative double rounding."""
    start = max(0, _sample_index(start_sec, sample_rate))
    if total_samples is not None:
        start = min(start, total_samples)
    if end_sec is None:
        if total_samples is None:
            raise ValueError("audio stream has no usable end time")
        stop = total_samples
    else:
        stop = max(0, _sample_index(end_sec, sample_rate))
        if total_samples is not None:
            stop = min(stop, total_samples)
    if max_duration_sec is not None:
        # The cap is relative to the already-rounded window origin. Combining
        # start+duration before rounding changes the selected sample count at
        # half-sample boundaries.
        stop = min(
            stop,
            start + max(0, _sample_index(max_duration_sec, sample_rate)),
        )
    return start, max(start, stop)


class AudioTrack:
    """PCM audio source for AVWriter.

    The public constructor keeps the existing in-memory waveform API. File
    carry uses :class:`StreamingAudioTrack`, which implements the same pull
    surface without retaining the complete source. Both build CMSampleBuffers
    on demand via ``make_sample_buffer(start_frame, end_frame)`` as the
    AVWriter's GCD audio pump drains.

    Format: interleaved 32-bit float PCM in the source sample rate. The
    writer's audio output settings (ALAC / AAC) handle the encode-time
    conversion.
    """

    def __init__(self, waveform: Any, sample_rate: int):
        # Accept an mlx or numpy (channels, samples) array; normalize to mlx f32.
        w = mx.array(waveform, dtype=mx.float32)
        if w.ndim != 2:
            raise ValueError(
                f"AudioTrack expects (channels, samples); got {w.shape}"
            )
        self.sample_rate = int(sample_rate)
        self.channels = int(w.shape[0])
        self.n_samples = int(w.shape[1])
        # Interleave: (channels, samples) -> (samples, channels) row-major bytes,
        # straight from the MLX buffer.
        self._bytes = bytes(memoryview(mx.contiguous(mx.transpose(w))))
        self.format_desc = _audio_format_description(
            self.sample_rate, self.channels)

    def _read_interleaved(self, start_frame: int, end_frame: int) -> memoryview:
        bytes_per_frame = 4 * self.channels
        return memoryview(self._bytes)[
            start_frame * bytes_per_frame:end_frame * bytes_per_frame]

    def fork(self) -> AudioTrack:
        """Return an independent reader cursor for a writer consumer.

        In-memory tracks are immutable and can safely share their backing
        bytes. Streaming tracks override this to create a fresh lazy decoder.
        """
        return self

    def close(self) -> None:
        """Release decoder state held by a file-backed track, if any."""

    def trimmed(
        self,
        start_sec: Fraction | int | float,
        end_sec: Fraction | int | float | None,
    ) -> AudioTrack:
        """Return a zero-copy view covering ``[start_sec, end_sec)``.

        Keeps muxed audio in sync when the video is trimmed with --start/--end.
        end_sec=None means to the end. Clamps to the available range.
        """
        s0 = max(0, _sample_index(start_sec, self.sample_rate))
        s1 = (
            self.n_samples if end_sec is None
            else min(self.n_samples, _sample_index(end_sec, self.sample_rate))
        )
        return _AudioTrackView(self, min(s0, s1), max(0, s1 - s0))

    def save_wav(self, path: Path) -> None:
        """Write float32 PCM in bounded chunks (for --save-audio-sidecar)."""
        cursor = self.fork()
        try:
            _write_track_wav_float32(cursor, path)
        finally:
            cursor.close()

    def make_sample_buffer(self, start_frame: int, end_frame: int) -> Any:
        """Build a CMSampleBuffer for audio frames [start_frame, end_frame).

        Returns None if the range is empty. Caller is responsible for
        appendSampleBuffer-ing it to an AVAssetWriterInput.
        """
        n = end_frame - start_frame
        if n <= 0:
            return None
        bytes_per_frame = 4 * self.channels
        chunk_bytes = self._read_interleaved(start_frame, end_frame)
        data_len = len(chunk_bytes)
        expected_len = n * bytes_per_frame
        if data_len > expected_len or data_len % bytes_per_frame:
            raise RuntimeError(
                f"audio source returned {data_len} bytes for {n} frames "
                f"({expected_len} maximum and whole frames required)")
        n = data_len // bytes_per_frame
        if n == 0:
            return None

        err, block = CoreMedia.CMBlockBufferCreateWithMemoryBlock(
            None, None, data_len, None, None, 0, data_len, 1, None,
        )
        if err != 0 or block is None:
            raise RuntimeError(f"CMBlockBufferCreateWithMemoryBlock failed: {err}")
        err = CoreMedia.CMBlockBufferReplaceDataBytes(chunk_bytes, block, 0, data_len)
        if err != 0:
            raise RuntimeError(f"CMBlockBufferReplaceDataBytes failed: {err}")

        pts = CoreMedia.CMTimeMake(start_frame, self.sample_rate)
        err, sample_buf = CoreMedia.CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            None, block, self.format_desc, n, pts, None, None,
        )
        if err != 0 or sample_buf is None:
            raise RuntimeError(
                f"CMAudioSampleBufferCreateReadyWithPacketDescriptions failed: {err}"
            )
        return sample_buf


class _AudioTrackView(AudioTrack):
    """Sample-window view which keeps the parent storage/decoder lazy."""

    def __init__(self, parent: AudioTrack, offset: int, n_samples: int):
        self._parent = parent
        self._offset = int(offset)
        self.sample_rate = parent.sample_rate
        self.channels = parent.channels
        self.n_samples = int(n_samples)
        self.format_desc = parent.format_desc

    def _read_interleaved(self, start_frame: int, end_frame: int) -> Any:
        return self._parent._read_interleaved(
            self._offset + start_frame, self._offset + end_frame)

    def fork(self) -> AudioTrack:
        return _AudioTrackView(
            self._parent.fork(), self._offset, self.n_samples)

    def close(self) -> None:
        self._parent.close()


class StreamingAudioTrack(AudioTrack):
    """Bounded, lazily decoded PCM track.

    ``source_factory`` returns an object with ``read_frames(start, end)`` and
    ``close()`` methods. Coordinates are absolute samples in the source track;
    ``offset`` selects the public window without decoding or copying it.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        n_samples: int,
        source_factory: Callable[[], Any],
        offset: int = 0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.n_samples = int(n_samples)
        self.format_desc = _audio_format_description(
            self.sample_rate, self.channels)
        self._source_factory = source_factory
        self._offset = int(offset)
        self._source: Any = None
        self._source_lock = threading.Lock()

    def _read_interleaved(self, start_frame: int, end_frame: int) -> Any:
        with self._source_lock:
            if self._source is None:
                self._source = self._source_factory()
            return self._source.read_frames(
                self._offset + start_frame, self._offset + end_frame)

    def fork(self) -> AudioTrack:
        return StreamingAudioTrack(
            sample_rate=self.sample_rate,
            channels=self.channels,
            n_samples=self.n_samples,
            source_factory=self._source_factory,
            offset=self._offset,
        )

    def close(self) -> None:
        with self._source_lock:
            source, self._source = self._source, None
        if source is not None:
            source.close()


class _AVAudioFileSource:
    """Seekable bounded PCM reader backed by AVAudioFile."""

    def __init__(self, path: Path, sample_rate: int, channels: int) -> None:
        from kinovsr.native.frameworks import Foundation, av

        url = Foundation.NSURL.fileURLWithPath_(str(path))
        audio_file, err = av.AVAudioFile.alloc().initForReading_error_(
            url, None)
        if audio_file is None:
            raise RuntimeError(f"AVAudioFile could not open {path}: {err}")
        fmt = audio_file.processingFormat()
        actual_rate = int(round(float(fmt.sampleRate())))
        actual_channels = int(fmt.channelCount())
        if (actual_rate, actual_channels) != (sample_rate, channels):
            raise RuntimeError(
                f"audio format changed while opening {path}: expected "
                f"{channels}ch/{sample_rate} Hz, got "
                f"{actual_channels}ch/{actual_rate} Hz")
        self._path = path
        self._audio_file = audio_file
        self._format = fmt
        self._channels = channels
        self._cursor = 0

    def read_frames(self, start_frame: int, end_frame: int) -> bytes:
        from kinovsr.native.frameworks import av

        n = end_frame - start_frame
        if n <= 0:
            return b""
        if start_frame != self._cursor:
            self._audio_file.setFramePosition_(start_frame)
        buf = av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
            self._format, n)
        ok, err = self._audio_file.readIntoBuffer_frameCount_error_(
            buf, n, None)
        if not ok:
            raise RuntimeError(
                f"AVAudioFile could not read {self._path} at frame "
                f"{start_frame}: {err}")
        frames = int(buf.frameLength())
        if frames != n:
            raise RuntimeError(
                f"AVAudioFile returned {frames} of {n} requested frames "
                f"from {self._path}")
        fcd = buf.floatChannelData()
        channels = [
            mx.array(memoryview(fcd[c].as_buffer(frames)).cast("f"))
            for c in range(self._channels)
        ]
        interleaved = mx.stack(channels, axis=1)
        self._cursor = end_frame
        return bytes(memoryview(mx.contiguous(interleaved)))

    def close(self) -> None:
        self._audio_file = None


def _native_streaming_audio_track(
    path: Path,
    *,
    start_sec: Fraction | int | float = Fraction(0),
    end_sec: Fraction | int | float | None = None,
    max_duration_sec: Fraction | int | float | None = None,
) -> StreamingAudioTrack | None:
    from kinovsr.native.frameworks import Foundation, av

    url = Foundation.NSURL.fileURLWithPath_(str(path))
    audio_file, err = av.AVAudioFile.alloc().initForReading_error_(url, None)
    if audio_file is None:
        raise RuntimeError(f"AVAudioFile could not open {path}: {err}")
    fmt = audio_file.processingFormat()
    sample_rate = int(round(float(fmt.sampleRate())))
    channels = int(fmt.channelCount())
    total = int(audio_file.length())
    if sample_rate <= 0 or channels <= 0 or total <= 0:
        return None
    start, stop = _sample_window(
        sample_rate=sample_rate,
        total_samples=total,
        start_sec=start_sec,
        end_sec=end_sec,
        max_duration_sec=max_duration_sec,
    )
    if stop == start:
        return None
    return StreamingAudioTrack(
        sample_rate=sample_rate,
        channels=channels,
        n_samples=stop - start,
        source_factory=lambda: _AVAudioFileSource(
            path, sample_rate, channels),
        offset=start,
    )


def read_wav(path: Any) -> tuple[int, mx.array]:
    """Read a WAV (or any AVFoundation-supported audio file) into
    ``(sample_rate, (channels, frames) float32 mlx array in [-1, 1])``.

    Uses AVFoundation's AVAudioFile, which reads PCM int16/24/32 AND IEEE float32
    - including the float32 sidecars that stdlib ``wave`` rejects - with no numpy /
    scipy / soundfile. Samples come straight from the AVAudioPCMBuffer's
    deinterleaved float channels via the buffer protocol.
    """
    from kinovsr.native.frameworks import Foundation, av

    url = Foundation.NSURL.fileURLWithPath_(str(path))
    audio_file, err = av.AVAudioFile.alloc().initForReading_error_(url, None)
    if audio_file is None:
        raise RuntimeError(f"AVAudioFile could not open {path}: {err}")
    fmt = audio_file.processingFormat()
    buf = av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
        fmt, int(audio_file.length()),
    )
    ok, err = audio_file.readIntoBuffer_error_(buf, None)
    if not ok:
        raise RuntimeError(f"AVAudioFile could not read {path}: {err}")
    channels = int(fmt.channelCount())
    frames = int(buf.frameLength())
    fcd = buf.floatChannelData()  # deinterleaved float32: channels x frames
    # as_buffer(n) exposes n elements (4 bytes each) as a uint8 view; cast to f32.
    chans = [
        mx.array(memoryview(fcd[c].as_buffer(frames)).cast("f"))
        for c in range(channels)
    ]
    samples = chans[0][None, :] if channels == 1 else mx.stack(chans, axis=0)
    return int(fmt.sampleRate()), samples


def read_audio_track_from_video(
    path: Path,
    reader: Any,
    *,
    start_sec: Fraction | int | float = Fraction(0),
    end_sec: Fraction | int | float | None = None,
    max_duration_sec: Fraction | int | float | None = None,
) -> AudioTrack | None:
    """Open a bounded, lazy audio window from an MP4/MOV.

    Native carry uses a seekable AVAudioFile cursor. When ``reader`` is the
    ffmpeg compatibility reader (it exposes ``read_audio_track_window``), it
    creates the equivalent PyAV cursor because AVFoundation cannot open those
    containers. Neither backend decodes before a writer or sidecar pulls a
    chunk, and both apply ``start_sec``/``end_sec`` before allocation.
    """
    window_reader = getattr(reader, "read_audio_track_window", None)
    if window_reader is not None:
        _log.info("reading audio track from %s (ffmpeg)", path)
        try:
            return window_reader(
                path,
                start_sec=start_sec,
                end_sec=end_sec,
                max_duration_sec=max_duration_sec,
            )
        except Exception as e:
            if is_native_operation_error(e):
                _log.warning(
                    "audio decode failed (%s); continuing without audio",
                    type(e).__name__,
                )
                return None
            raise
    if hasattr(reader, "read_audio_track"):
        # Never fall back to the original whole-track protocol: applying a
        # view after read_audio_track(path) still leaves the complete decoded
        # backing alive. Adapters must opt into the bounded window contract so
        # a short trim cannot silently allocate source-duration-sized PCM.
        raise RuntimeError(
            "audio carry requires the reader adapter to implement "
            "read_audio_track_window(path, start_sec=..., end_sec=..., "
            "max_duration_sec=...); refusing unbounded read_audio_track(path)")

    _log.info("reading audio track from %s", path)
    try:
        track = _native_streaming_audio_track(
            path,
            start_sec=start_sec,
            end_sec=end_sec,
            max_duration_sec=max_duration_sec,
        )
    except Exception as e:
        if is_native_operation_error(e):
            # No audio track (or an unsupported audio format) - carry on silent.
            _log.warning(
                "no usable audio track (%s: %s); output will be silent",
                type(e).__name__, e,
            )
            return None
        raise
    if track is not None:
        _log.info(
            "audio window: %sch, %s Hz, %s samples",
            track.channels, track.sample_rate, track.n_samples)
    return track


def _write_track_wav_float32(track: AudioTrack, path: Any) -> None:
    """Write a track without materializing more than a fixed PCM budget."""
    from kinovsr.native.frameworks import Foundation, av

    settings = {
        av.AVFormatIDKey: AUDIO_FORMAT_LPCM,
        av.AVSampleRateKey: float(track.sample_rate),
        av.AVNumberOfChannelsKey: track.channels,
        av.AVLinearPCMBitDepthKey: 32,
        av.AVLinearPCMIsFloatKey: True,
        av.AVLinearPCMIsBigEndianKey: False,
        av.AVLinearPCMIsNonInterleaved: False,
    }
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    out, err = av.AVAudioFile.alloc().initForWriting_settings_error_(
        url, settings, None)
    if out is None:
        raise RuntimeError(f"AVAudioFile could not open {path} for writing: {err}")
    bytes_per_frame = 4 * track.channels
    chunk_frames = max(1, _SIDECAR_PCM_BUDGET // bytes_per_frame)
    for start in range(0, track.n_samples, chunk_frames):
        stop = min(start + chunk_frames, track.n_samples)
        frames = stop - start
        raw = track._read_interleaved(start, stop)
        expected = frames * bytes_per_frame
        if len(raw) > expected or len(raw) % bytes_per_frame:
            raise RuntimeError(
                f"audio source returned {len(raw)} bytes for {frames} frames "
                f"while writing {path} ({expected} maximum and whole frames "
                f"required)")
        actual_frames = len(raw) // bytes_per_frame
        if actual_frames == 0:
            break
        samples = mx.array(
            memoryview(raw).cast("B").cast("f"), dtype=mx.float32,
        ).reshape(actual_frames, track.channels)
        buf = av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
            out.processingFormat(), actual_frames)
        buf.setFrameLength_(actual_frames)
        fcd = buf.floatChannelData()
        for channel in range(track.channels):
            channel_bytes = memoryview(
                mx.contiguous(samples[:, channel])).cast("B")
            memoryview(
                fcd[channel].as_buffer(actual_frames))[:] = channel_bytes
        ok, err = out.writeFromBuffer_error_(buf, None)
        if not ok:
            raise RuntimeError(f"AVAudioFile could not write {path}: {err}")
        if actual_frames < frames:
            break


def _write_wav(samples: Any, path: Any, sample_rate: int, *, float32: bool) -> None:
    """Write (B,C,T)/(C,T) mlx or numpy samples to a WAV via AVFoundation's
    AVAudioFile - native macOS, no struct/wave hand-rolling.

    float32=True writes an IEEE float32 WAV; otherwise int16 PCM. The samples are
    written into a float32 AVAudioPCMBuffer and AVAudioFile converts to the file
    format and writes the container/header.
    """
    from kinovsr.native.frameworks import Foundation, av

    w = mx.array(samples, dtype=mx.float32)
    if w.ndim == 3:
        w = w[0]
    if w.ndim != 2:
        raise ValueError(f"audio must be (B,C,T) or (C,T); got shape {w.shape}")
    channels, frames = int(w.shape[0]), int(w.shape[1])
    settings = {
        av.AVFormatIDKey: AUDIO_FORMAT_LPCM,
        av.AVSampleRateKey: float(sample_rate),
        av.AVNumberOfChannelsKey: channels,
        av.AVLinearPCMBitDepthKey: 32 if float32 else 16,
        av.AVLinearPCMIsFloatKey: float32,
        av.AVLinearPCMIsBigEndianKey: False,
        av.AVLinearPCMIsNonInterleaved: False,
    }
    url = Foundation.NSURL.fileURLWithPath_(str(path))
    out, err = av.AVAudioFile.alloc().initForWriting_settings_error_(url, settings, None)
    if out is None:
        raise RuntimeError(f"AVAudioFile could not open {path} for writing: {err}")
    buf = av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
        out.processingFormat(), frames,
    )
    buf.setFrameLength_(frames)
    fcd = buf.floatChannelData()  # deinterleaved float32, channels x frames
    for c in range(channels):
        # cast("B") is a zero-copy byte view of the float32 samples (byte-identical
        # to bytes(...)); the slice-assign copies it into the AVAudio channel buffer.
        memoryview(fcd[c].as_buffer(frames))[:] = memoryview(mx.contiguous(w[c])).cast("B")
    ok, err = out.writeFromBuffer_error_(buf, None)
    if not ok:
        raise RuntimeError(f"AVAudioFile could not write {path}: {err}")


def write_wav_int16(audio_waveform: Any, path: Any, sample_rate: int) -> None:
    """Write a stereo int16 PCM WAV (mlx/numpy (B,C,T) or (C,T))."""
    _write_wav(audio_waveform, path, sample_rate, float32=False)


def write_wav_float32(audio_waveform: Any, path: Any, sample_rate: int) -> None:
    """Write an IEEE float32 WAV (mlx/numpy (B,C,T) or (C,T)); no int16 quantization."""
    _write_wav(audio_waveform, path, sample_rate, float32=True)


def audio_writer_settings(codec: str, sample_rate: int, channels: int) -> dict:
    """AVAssetWriterInput output settings for the configured audio codec."""
    from kinovsr.native.frameworks import av

    if codec == "alac":
        return {
            av.AVFormatIDKey: AUDIO_FORMAT_ALAC,
            av.AVSampleRateKey: float(sample_rate),
            av.AVNumberOfChannelsKey: channels,
            av.AVEncoderBitDepthHintKey: 24,
        }
    if codec == "aac":
        return {
            av.AVFormatIDKey: AUDIO_FORMAT_AAC,
            av.AVSampleRateKey: float(sample_rate),
            av.AVNumberOfChannelsKey: channels,
            av.AVEncoderBitRateKey: 256000,
        }
    raise ValueError(f"Unknown audio codec {codec!r}")
