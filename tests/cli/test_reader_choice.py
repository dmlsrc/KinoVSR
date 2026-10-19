"""The run command reads MPEG program streams with the ffmpeg reader unless told otherwise."""

from types import SimpleNamespace

import av
import pytest

from kinovsr.cli.args import build_parser
from kinovsr.media import ffmpeg_reader, video_reader
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


class _Captured(Exception):
    """Stops the run once it has handed its reader to the file run."""


def _write(path, container_format):
    out = av.open(str(path), "w", format=container_format)
    vs = out.add_stream("mpeg2video", rate=25)
    vs.width, vs.height = 64, 48
    vs.pix_fmt = "yuv420p"
    for t in range(24):
        frame = av.VideoFrame(64, 48, "gray")
        frame.planes[0].update(bytes((x * 3 + t * 5) % 256 for x in range(64)) * 48)
        for packet in vs.encode(frame.reformat(format="yuv420p")):
            out.mux(packet)
    for packet in vs.encode():
        out.mux(packet)
    out.close()


@pytest.mark.parametrize(
    ("name", "container_format", "choice", "expected"),
    [
        # Named as QuickTime: the automatic choice follows the content.
        ("source.mov", "mpeg", "auto", ffmpeg_reader),
        ("source.mpg", "mpeg", "auto", ffmpeg_reader),
        # AVFoundation picks its parser by extension, so the native cases
        # carry their real one.
        ("source.mpg", "mpeg", "native", video_reader),
        ("source.mpg", "mpeg", "ffmpeg", ffmpeg_reader),
        ("source.ts", "mpegts", "auto", video_reader),
    ],
)
def test_run_hands_over_the_reader_the_flag_and_content_choose(
    tmp_path, monkeypatch, name, container_format, choice, expected
):
    from kinovsr.cli.commands.run import _run_typed

    source = tmp_path / name
    _write(source, container_format)
    options = build_parser().parse_args(
        ["--video", str(source), "--output-dir", str(tmp_path / "out"), "--reader", choice]
    )
    invocation = SimpleNamespace(
        options=options,
        config={"pipeline": []},
        settings=Settings(mlx_cache_limit_gb=0, shared_temp_dir=tmp_path / "scratch"),
    )
    handed = []

    def capture(*_args, reader, **_kwargs):
        handed.append(reader)
        raise _Captured

    monkeypatch.setattr("kinovsr.cli.commands.run.process_video_file", capture)
    with pytest.raises(_Captured):
        _run_typed(invocation)
    assert handed == [expected]
