"""Probe-time auto geometry: bars="auto"/edges="auto" resolve before preflight.

The clips are synthesized with near-lossless x264 (qp=12) so bar and junk
bands survive the encode the way real mastering artifacts do.
"""

import av
import numpy as np
import pytest

from kinovsr.config import ConfigError
from kinovsr.pipeline import run_file
from kinovsr.pipeline.auto_geometry import (
    resolve_auto_geometry,
    wants_auto_geometry,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.integration

W, H, N = 160, 128, 24
SETTINGS = Settings()


def _write(path, *, bars=0, junk_top_inside=False, junk_bottom=False):
    out = av.open(str(path), "w")
    stream = out.add_stream("libx264", rate=25)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"qp": "12", "bf": "0"}
    rng = np.random.default_rng(9)
    for _ in range(N):
        frame = (rng.uniform(0.35, 0.85, (H, W, 3)) * 255).astype(np.uint8)
        if bars:
            frame[:bars] = 4
            frame[-bars:] = 4
        if junk_top_inside and bars:
            # junk band just inside the top bar: visible only post-crop
            frame[bars:bars + 2] = (frame[bars + 2:bars + 4] * 0.25
                                    ).astype(np.uint8)
        if junk_bottom:
            frame[-2:] = (frame[-4:-2] * 0.25).astype(np.uint8)
        for pkt in stream.encode(av.VideoFrame.from_ndarray(
                frame, format="rgb24")):
            out.mux(pkt)
    for pkt in stream.encode():
        out.mux(pkt)
    out.close()
    return path


@pytest.fixture(scope="module")
def letterboxed(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("ag") / "letter.mp4", bars=16)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("ag") / "clean.mp4")


def _out_geometry(path):
    with av.open(str(path)) as container:
        codec = container.streams.video[0].codec_context
        return codec.width, codec.height


class TestWantsAuto:
    def test_detects_auto_values(self):
        assert wants_auto_geometry(
            {"pipeline": ["c"], "c": {"processor": "crop", "bars": "auto"}})
        assert wants_auto_geometry(
            {"pipeline": ["s"],
             "s": {"processor": "sanitize_edges", "edges": "auto"}})

    def test_literal_configs_skip_the_probe(self):
        assert not wants_auto_geometry(
            {"pipeline": ["c"],
             "c": {"processor": "crop", "bars": "16,16,0,0"}})
        assert not wants_auto_geometry({"pipeline": []})


class TestResolve:
    def test_bars_auto_crops_the_detected_letterbox(self, letterboxed,
                                                    tmp_path):
        config = {"pipeline": ["c"], "c": {"processor": "crop",
                                           "bars": "auto"}}
        res = run_file(config, video=letterboxed, output=tmp_path / "o.mp4",
                       settings=SETTINGS)
        assert _out_geometry(res.path) == (W, H - 32)

    def test_nothing_detected_removes_the_stage(self, clean, tmp_path):
        config = {"pipeline": ["c"], "c": {"processor": "crop",
                                           "bars": "auto"}}
        res = run_file(config, video=clean, output=tmp_path / "o.mp4",
                       settings=SETTINGS)
        assert _out_geometry(res.path) == (W, H)   # full frame, no crop

    def test_sanitize_auto_detects_post_crop(self, tmp_path_factory,
                                             tmp_path):
        # Junk band INSIDE the letterbox: only visible on the post-crop
        # picture, so this passes only if the probe crops its samples
        # between detector passes (the harness's behavior).
        from kinovsr.media import video_reader

        clip = _write(tmp_path_factory.mktemp("ag") / "both.mp4",
                      bars=16, junk_top_inside=True)
        config = {
            "pipeline": ["c", "s"],
            "c": {"processor": "crop", "bars": "auto"},
            "s": {"processor": "sanitize_edges", "edges": "auto"},
        }
        resolved = resolve_auto_geometry(config, video=clip, vr=video_reader)
        assert resolved["c"]["bars"] == "16,16,0,0"
        assert resolved["s"]["edges"].startswith("2,")   # top junk, post-crop
        res = run_file(config, video=clip, output=tmp_path / "o.mp4",
                       settings=SETTINGS)
        assert _out_geometry(res.path) == (W, H - 32)

    def test_sanitize_auto_nothing_removes_stage(self, clean):
        from kinovsr.media import video_reader

        config = {"pipeline": ["s"],
                  "s": {"processor": "sanitize_edges", "edges": "auto"}}
        resolved = resolve_auto_geometry(config, video=clean, vr=video_reader)
        assert resolved["pipeline"] == []
        assert "s" not in resolved

    def test_auto_after_a_non_geometry_stage_is_rejected(self, letterboxed,
                                                         tmp_path):
        config = {
            "pipeline": ["cd", "c"],
            "cd": {"processor": "cut_detect"},
            "c": {"processor": "crop", "bars": "auto"},
        }
        with pytest.raises(ConfigError, match="auto detection samples"):
            run_file(config, video=letterboxed, output=tmp_path / "o.mp4",
                     settings=SETTINGS)

    def test_literal_stages_pass_through_untouched(self, letterboxed):
        from kinovsr.media import video_reader

        config = {
            "pipeline": ["c", "s"],
            "c": {"processor": "crop", "bars": "auto"},
            "s": {"processor": "sanitize_edges", "edges": "1,0,0,0"},
        }
        resolved = resolve_auto_geometry(
            config, video=letterboxed, vr=video_reader)
        assert resolved["s"] == {"processor": "sanitize_edges",
                                 "edges": "1,0,0,0"}
