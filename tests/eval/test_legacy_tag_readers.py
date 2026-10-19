"""Evaluation frame readers accept a file whose metadata tag is not UTF-8."""

import av
import pytest

from kinovsr.eval import dover_score, musiq_score, niqe, perceptual_metrics
from kinovsr.processors.nafnet import probe as nafnet_probe

pytestmark = pytest.mark.unit

W, H, N = 64, 48, 3

READERS = {
    "niqe": lambda path: niqe._read_video_lumas(path),
    "musiq": lambda path: musiq_score._read_video_frames(path, 1),
    "dover": lambda path: dover_score._read_video(path, N),
    "perceptual": lambda path: perceptual_metrics.read_frames(path, N),
    "nafnet-probe": lambda path: nafnet_probe._read_frames(path, list(range(N))),
}


@pytest.fixture(scope="module")
def mac_roman_tagged(tmp_path_factory):
    # Old QuickTime writers store tag text as raw Mac Roman, so a copyright
    # sign arrives as the lone byte 0xa9, which is not valid UTF-8.
    marker = b"KINOVSR-TAG"
    folder = tmp_path_factory.mktemp("tags")
    source = folder / "tagged.mp4"
    out = av.open(str(source), "w")
    out.metadata["comment"] = marker.decode()
    vs = out.add_stream("mpeg4", rate=25)
    vs.width, vs.height = W, H
    vs.pix_fmt = "yuv420p"
    for _ in range(N):
        for pkt in vs.encode(av.VideoFrame(W, H, "yuv420p")):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()
    data = source.read_bytes()
    assert data.count(marker) == 1
    path = folder / "mac-roman-tag.mp4"
    path.write_bytes(data.replace(marker, b"\xa9" + marker[1:]))
    return path


@pytest.mark.parametrize("name", sorted(READERS))
def test_reader_decodes_every_frame(mac_roman_tagged, name):
    assert len(READERS[name](mac_roman_tagged)) == N
