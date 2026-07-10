"""Internal file-processing facade.

This is the M2 transitional surface: one call that takes a resolved
configuration and produces output files, so the CLI stops being the only
orchestration path. It wraps the inherited harness orchestration
(:mod:`kinovsr._harness`) and is NOT the frozen host API - the public
``process_video_file`` / ``open_pipeline`` contract freezes after the M3
typed-stream pipeline proves the underlying semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinovsr.settings import Settings


@dataclass(frozen=True)
class VideoFileConfig:
    """A resolved file-to-file run: global settings plus stage options.

    ``options`` is the transitional flat namespace of canonical CLI
    destinations (see :mod:`kinovsr.cli.config`); M3 replaces it with
    typed stage configuration.
    """

    settings: Settings
    options: Any


@dataclass(frozen=True)
class VideoProcessResult:
    """What a file-to-file run produced."""

    post_path: Path | None = None
    comparison_path: Path | None = None
    frames_out: int = 0
    elapsed_s: float = 0.0


def process_video_file(config: VideoFileConfig) -> VideoProcessResult:
    """Process one video file to output files, per the resolved config."""
    from kinovsr import _harness  # heavy import (MLX, native session setup)

    return _harness.run(config.options, settings=config.settings)
