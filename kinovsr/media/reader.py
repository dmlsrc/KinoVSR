"""The decode contract the file endpoints read a video through.

The native reader (:mod:`kinovsr.media.video_reader`) and the ffmpeg reader
(:mod:`kinovsr.media.ffmpeg_reader`) are modules that satisfy it
structurally; a custom adapter may be a module, a class or an instance.
Optional hooks (``read_sample_table``, ``probe_video_timing``,
``probe_audio_timing``, ``read_audio_track_window``) are looked up with
getattr by their callers and are not part of the required surface.

Chunk items are CVPixelBuffers, or ``(buffer, table_index)`` pairs when a
``table`` is passed.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from . import video_reader
from .timing import SampleTable, VideoTiming


class VideoReader(Protocol):
    def probe_video(
        self, path: Path, /
    ) -> tuple[int, int, float, int, object, tuple[int, int] | None]: ...

    def probe_color(self, path: Path, /) -> dict[str, object]: ...

    def keyframe_display_indices(
        self, path: Path, /, *, timing: VideoTiming | None = None
    ) -> list[int]: ...

    def iter_video_buffer_chunks(
        self,
        path: Path,
        src_format: int,
        /,
        chunk_size: int = 8,
        *,
        start_frame: int = 0,
        end_frame: int | None = None,
        timing: VideoTiming | None = None,
        table: SampleTable | None = None,
    ) -> Iterator[list[object]]: ...

    def iter_forced_color_chunks(
        self,
        path: Path,
        out_format: int,
        matrix_cv: object,
        full_range: bool,
        /,
        chunk_size: int = 8,
        *,
        start_frame: int = 0,
        end_frame: int | None = None,
        reinterpret_full_range: bool | None = None,
        timing: VideoTiming | None = None,
        table: SampleTable | None = None,
    ) -> Iterator[list[object]]: ...


__all__ = ["VideoReader"]


def auto_reader(path: Path | str) -> VideoReader:
    """The reader automatic selection starts with for ``path``.

    The native reader, except for an MPEG program stream (VCD, DVD and other
    .mpg/.vob files), which the ffmpeg reader reads when the 'ffmpeg' extra is
    installed. AVFoundation reads such streams incompletely: it can stop short
    of the last frames of a video-only stream, and where a muxer stamped a PES
    timestamp on the wrong picture it drops a frame or fails, while the ffmpeg
    reader delivers every frame of both.
    """
    try:
        from kinovsr.media import ffmpeg_reader  # noqa: PLC0415 - ffmpeg extra
    except ImportError:
        return video_reader
    return ffmpeg_reader if ffmpeg_reader.is_program_stream(Path(path)) else video_reader
