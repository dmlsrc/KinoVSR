"""M2 acceptance fixture: synthetic file-to-file through the flag CLI.

Proves source probing (trim math over a probed window), audio carry,
output creation, and representative canonical CLI options through
``kinovsr.cli.main`` - since M6 the flag surface assembles into the typed
pipeline.

The fixture clip is muxed by PyAV (no binary fixtures): mpeg4 video plus
an AAC sine tone. The run itself uses the product's native path (reader
probe decides native vs ffmpeg; the writer is AVAssetWriter).
"""

import math

import av
import pytest

from kinovsr.cli.main import main

pytestmark = pytest.mark.integration

W, H, N, FPS = 160, 128, 24, 25
SAMPLE_RATE = 48000


@pytest.fixture(scope="module")
def clip_with_audio(tmp_path_factory):
    path = tmp_path_factory.mktemp("api_fixture") / "clip.mp4"
    out = av.open(str(path), "w")
    vs = out.add_stream("mpeg4", rate=FPS)
    vs.width, vs.height = W, H
    vs.pix_fmt = "yuv420p"
    vs.options = {"g": "8", "bf": "0", "qscale": "2"}
    audio = out.add_stream("aac", rate=SAMPLE_RATE)

    for t in range(N):
        rows = bytearray()
        for _y in range(H):
            rows += bytes(min(255, (x + 2 * t) % 256) for x in range(W))
        frame = av.VideoFrame(W, H, "gray")
        frame.planes[0].update(bytes(rows))
        frame = frame.reformat(format="yuv420p")
        for pkt in vs.encode(frame):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)

    total = SAMPLE_RATE * N // FPS
    chunk = 1024
    for start in range(0, total, chunk):
        n = min(chunk, total - start)
        pcm = b"".join(
            int(12000 * math.sin(2 * math.pi * 440 * (start + i) / SAMPLE_RATE)
                ).to_bytes(2, "little", signed=True)
            for i in range(n))
        af = av.AudioFrame(format="s16", layout="mono", samples=n)
        af.planes[0].update(pcm)
        af.sample_rate = SAMPLE_RATE
        af.pts = start
        for pkt in audio.encode(af):
            out.mux(pkt)
    for pkt in audio.encode():
        out.mux(pkt)
    out.close()
    return path


def _output_streams(path):
    with av.open(str(path)) as container:
        video = container.streams.video
        sound = container.streams.audio
        frames = sum(1 for _ in container.decode(video=0))
        return frames, len(video), len(sound)


def test_flag_cli_probes_trims_and_carries_audio(clip_with_audio, tmp_path):
    rc = main([
        "--video", str(clip_with_audio),
        "--output-dir", str(tmp_path),
        "--upscale", "none",
        "--start", "4", "--max-frames", "8",
        "--audio",
        "--mlx-cache-limit-gb", "0.25",
    ])
    assert rc == 0
    outputs = list(tmp_path.glob("*_post.mp4"))
    assert len(outputs) == 1
    frames, n_video, n_audio = _output_streams(outputs[0])
    assert frames == 8
    assert n_video == 1
    assert n_audio == 1, "muxed audio track missing from the output"

    # The facade (not just the CLI) applies the MLX cache cap from settings.
    import mlx.core as mx
    previous = mx.set_cache_limit(1000 ** 3)
    assert previous == int(0.25 * (1000 ** 3))


def test_main_runs_canonical_spellings_end_to_end(clip_with_audio, tmp_path):
    rc = main([
        "--video", str(clip_with_audio),
        "--output-dir", str(tmp_path),
        "--upscale", "none",
        "--reader", "ffmpeg",            # force the compatibility reader
        "--max-frames", "6",
        "--output-prefix", "canonical",
        "--mlx-cache-limit-gb", "0.25",
    ])
    assert rc == 0
    outputs = list(tmp_path.glob("canonical*_post.mp4"))
    assert len(outputs) == 1
    frames, _, n_audio = _output_streams(outputs[0])
    assert frames == 6
    assert n_audio == 0  # no --audio: silent output
