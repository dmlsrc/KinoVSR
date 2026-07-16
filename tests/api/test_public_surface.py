"""The public API surface: exported names, import weight, and routing.

Pins the M5 contract: ``kinovsr.api`` exports exactly the documented
names (docs/API.md), stays light to import (no MLX/PyObjC at module
load), and both entries route through the one validated pipeline path.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

EXPECTED = {
    "PipelineSession",
    "VideoProcessResult",
    "open_pipeline",
    "process_video_file",
    "resolve_mlx_cache_limit_gb",
}


def test_all_matches_the_documented_surface():
    import kinovsr.api as api

    assert set(api.__all__) == EXPECTED


def test_every_exported_name_resolves():
    import kinovsr.api as api

    for name in api.__all__:
        assert getattr(api, name) is not None, name


def test_api_import_is_light():
    """Importing the API must not load MLX or PyObjC frameworks."""
    code = (
        "import sys, kinovsr.api; "
        "heavy = [m for m in ('mlx.core', 'AVFoundation', 'Quartz') "
        "if m in sys.modules]; "
        "print(','.join(heavy) or 'LIGHT')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "LIGHT", out.stdout


def test_annotations_resolve_at_runtime():
    """Host tooling introspects the API; postponed annotations must
    resolve without TYPE_CHECKING-only names."""
    import typing

    import kinovsr.api as api

    for func in (api.open_pipeline, api.process_video_file):
        hints = typing.get_type_hints(func)
        assert hints, func.__name__


def test_session_surface_is_video_only():
    """The M5 audio-surface review, pinned: the stream contract carries
    no audio. ``open_pipeline`` takes no audio argument and ``FrameUnit``
    has no audio field - audio is a file-endpoint concern handled by
    ``process_video_file`` (docs/API.md, Audio)."""
    import inspect

    from kinovsr.api import open_pipeline
    from kinovsr.processors import FrameUnit

    params = inspect.signature(open_pipeline).parameters
    assert not any("audio" in name.lower() for name in params), params
    # "source" is the raw-stream identity channel (sync flag, GOP
    # position, coded size) - video-stream metadata, not audio.
    assert set(FrameUnit.__dataclass_fields__) == {
        "payload", "pts", "duration", "boundaries", "source"}


def test_open_pipeline_is_the_session_constructor():
    from fractions import Fraction

    from kinovsr.api import open_pipeline
    from kinovsr.pipeline import PipelineSession
    from kinovsr.processors import Geometry, StreamSpec, TimelineSpec, frame_spec_for_matrix
    from kinovsr.settings import Settings

    spec = StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(64, 48)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))
    session = open_pipeline(
        {"pipeline": ["up"], "up": {"processor": "metalfx", "scale": 2}},
        spec, settings=Settings())
    assert isinstance(session, PipelineSession)
    assert session.output_spec.frame.geometry.width == 128
