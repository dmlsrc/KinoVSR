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


def resolve_mlx_cache_limit_gb(settings: Settings) -> float:
    """The effective MLX buffer-cache cap in GB: the setting when given,
    else the historical harness default of 1.0; 0 disables the cap."""
    limit = settings.mlx_cache_limit_gb
    return 1.0 if limit is None else limit


def process_video_file(config: VideoFileConfig) -> VideoProcessResult:
    """Process one video file to output files, per the resolved config.

    Runtime environment setup (the MLX cache cap, the PyObjC presence
    check) happens here, not in the CLI, so the same ``VideoFileConfig``
    behaves identically through the API and the console entry point.
    """
    from kinovsr import require_pyobjc

    require_pyobjc()
    limit = resolve_mlx_cache_limit_gb(config.settings)
    if limit > 0:
        import mlx.core as mx

        mx.set_cache_limit(int(limit * (1000 ** 3)))
        mx.clear_cache()

    from kinovsr import _harness  # heavy import (MLX, native session setup)

    return _harness.run(config.options, settings=config.settings)
