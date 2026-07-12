"""AVWriter must not leak hardware-encoder session slots across writers.

AVFoundation autoreleases session-holding objects during a writer's
construction, appends, and finish. A long-lived host process without a
draining run loop (the API use case, this very test process) used to leak
them, and after ~32 writers the encoder fell back to a software capability
set without the 4:2:2 profile - AVAssetWriterInput creation then failed
with "video codec type hvc1 only allows ... HEVC_Main10_AutoLevel ...".
AVWriter now drains a pool per phase; this exercises well past the old cap.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from kinovsr.media import pixel_buffers as _pb
from kinovsr.native.writer import HEVC_PROFILE_MAIN422_10, AVWriter

pytestmark = pytest.mark.integration


def test_forty_writers_do_not_exhaust_encoder_sessions(tmp_path):
    quiet = io.StringIO()   # 40 setup lines are noise
    with contextlib.redirect_stdout(quiet):
        for i in range(40):
            writer = AVWriter(
                tmp_path / f"w{i}.mp4", width=160, height=128, fps=25.0,
                source_pixel_format=_pb.PIX_RGBAHALF,
                profile=HEVC_PROFILE_MAIN422_10, quality=0.65, label="probe")
            buffer = _pb.make_pixel_buffer_from_attrs(160, 128, {
                "PixelFormatType": _pb.PIX_RGBAHALF,
                "IOSurfaceProperties": {}})
            writer.append(buffer, pts_ticks=0, duration_ticks=960)
            writer.finish()
    assert (tmp_path / "w39.mp4").stat().st_size > 0
