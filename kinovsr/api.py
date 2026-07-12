"""The public host API.

Two entry points, one validation path:

- :func:`process_video_file` runs a pipeline config file-to-file
  (probe, preflight, decode, chain, encode) and returns a
  :class:`VideoProcessResult`;
- :func:`open_pipeline` opens the same validated chain over a host's
  own frame units (:class:`~kinovsr.pipeline.PipelineSession`), so an
  engine can consume processed frames without KinoVSR encoding a file.

MLX RGB frames and CVPixelBuffers ride the same pipeline validation and
lifecycle - the input :class:`~kinovsr.processors.specs.StreamSpec`'s
layout states which payload the units carry. Errors are typed
(:class:`~kinovsr.config.ConfigError`,
:class:`~kinovsr.processors.errors.PipelineError` and subclasses,
:class:`~kinovsr.processors.errors.MediaError`); closing a session or
its iterator mid-stream cancels deterministically and releases every
stage exactly once.

Compatibility: this module is the supported import surface
(``__all__`` below, tested in ``tests/api``). Pre-1.0, additions land
freely; breaking changes to these names are called out in commit
history and the planning record. Underscore-prefixed names and every
other module are internal.

The transitional flat-options entry (:func:`process_video_options`)
runs the inherited orchestration for the flag-driven CLI surface until
it reaches typed-pipeline parity; hosts should not target it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kinovsr.pipeline import PipelineSession
from kinovsr.processors import StreamSpec
from kinovsr.reporting import Reporter
from kinovsr.settings import Settings


@dataclass(frozen=True)
class VideoFileConfig:
    """A resolved flat-options run: global settings plus stage options.

    The transitional shape consumed by :func:`process_video_options`;
    ``options`` is the flat namespace of canonical CLI destinations.
    """

    settings: Settings
    options: Any


@dataclass(frozen=True)
class VideoProcessResult:
    """What a file-to-file run produced."""

    post_path: Path | None = None
    comparison_path: Path | None = None
    frames_in: int = 0
    frames_out: int = 0
    elapsed_s: float = 0.0


def resolve_mlx_cache_limit_gb(settings: Settings) -> float:
    """The effective MLX buffer-cache cap in GB: the setting when given,
    else the historical default of 1.0; 0 disables the cap."""
    limit = settings.mlx_cache_limit_gb
    return 1.0 if limit is None else limit


def _runtime_setup(settings: Settings) -> None:
    """Environment setup shared by every processing entry: the PyObjC
    presence check and the MLX cache cap, applied here (not in the CLI)
    so API and console behavior match."""
    from kinovsr import require_pyobjc

    require_pyobjc()
    limit = resolve_mlx_cache_limit_gb(settings)
    if limit > 0:
        import mlx.core as mx

        mx.set_cache_limit(int(limit * (1000 ** 3)))
        mx.clear_cache()


def open_pipeline(
    config: Mapping[str, Any],
    input_spec: StreamSpec,
    *,
    settings: Settings | None = None,
    reporter: Reporter | None = None,
) -> PipelineSession:
    """Resolve and preflight-validate ``config`` against ``input_spec``;
    return a session over the caller's own frame units.

    See :class:`kinovsr.pipeline.PipelineSession`: ``process(units)``
    yields output units with natural backpressure, weights load at the
    first pull, and closing cancels deterministically.
    """
    from kinovsr.pipeline import open_pipeline as _open

    return _open(config, input_spec, settings=settings, reporter=reporter)


def process_video_file(
    config: Mapping[str, Any],
    *,
    video: Path | str,
    output: Path | str,
    settings: Settings | None = None,
    reporter: Reporter | None = None,
    start: int = 0,
    end: int | None = None,
    max_frames: int | None = None,
    max_output_frames: int | None = None,
    max_output_seconds: float | None = None,
    audio: bool = False,
    audio_codec: str = "alac",
    save_audio_sidecar: bool = False,
    quality: float = 0.65,
    chunk_size: int = 8,
    source_color: str = "auto",
    source_range: str = "auto",
    layout: Any = None,
    reader: Any = None,
) -> VideoProcessResult:
    """Run a pipeline config file-to-file and return what it produced.

    The input file is probed into a concrete StreamSpec, the chain is
    preflight-validated against it before any frame decodes, and the
    sink verifies the declared output timeline as units arrive.
    ``start``/``end``/``max_frames`` window the input in frames;
    ``max_output_frames`` / ``max_output_seconds`` cap what is written
    (the output-side cap for cadence-changing chains; the seconds form
    resolves against the OUTPUT cadence, and carried audio trims to the
    cap); ``audio`` carries the source audio track (trimmed to the
    window).
    """
    if settings is None:
        settings = Settings.from_env()
    _runtime_setup(settings)

    from kinovsr.pipeline import run_file
    from kinovsr.processors import Layout

    result = run_file(
        dict(config),
        video=video,
        output=output,
        settings=settings,
        reporter=reporter,
        layout=layout or Layout.MLX_RGB_HWC,
        start=start,
        end=end,
        max_frames=max_frames,
        max_output_frames=max_output_frames,
        max_output_seconds=max_output_seconds,
        audio=audio,
        audio_codec=audio_codec,
        save_audio_sidecar=save_audio_sidecar,
        quality=quality,
        chunk_size=chunk_size,
        source_color=source_color,
        source_range=source_range,
        reader=reader,
    )
    return VideoProcessResult(
        post_path=result.path,
        frames_in=result.frames_in,
        frames_out=result.frames_out,
        elapsed_s=result.elapsed_s,
    )


def process_video_options(config: VideoFileConfig) -> VideoProcessResult:
    """Process one video file per a resolved flat-options invocation.

    The transitional entry for the flag-driven CLI surface; it wraps the
    inherited harness orchestration and retires when that surface
    reaches typed-pipeline parity.
    """
    _runtime_setup(config.settings)

    from kinovsr import _harness  # heavy import (MLX, native session setup)

    return _harness.run(config.options, settings=config.settings)


__all__ = [
    "PipelineSession",
    "VideoFileConfig",
    "VideoProcessResult",
    "open_pipeline",
    "process_video_file",
    "process_video_options",
    "resolve_mlx_cache_limit_gb",
]
