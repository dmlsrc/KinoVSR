"""The processing command: parse, assemble, run, report.

This is the flat invocation (``kinovsr --video ...``), also reachable as
an explicit subcommand (``kinovsr run --video ...``). It owns the
terminal wiring for a processing run - logging configuration and the
MLX cache-limit echo - and hands the resolved invocation to the
:func:`kinovsr.api.process_video_options` facade (flag surface) or
the public :func:`kinovsr.api.process_video_file` (pipeline configs).
"""

from __future__ import annotations

from kinovsr.config import ConfigError
from kinovsr.ui.console import get_console

from ..args import build_parser, validate_args
from ..config import assemble


def run_video_command(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    try:
        invocation = assemble(args)
    except ConfigError as exc:
        get_console().print(f"config error: {exc}", style="bold red",
                            markup=False)
        return 2

    settings = invocation.settings
    from kinovsr.ui import configure_logging_from_settings

    configure_logging_from_settings(settings)

    # --probe-noise analyzes instead of processing; same implementation
    # as `kinovsr probe noise`.
    options = invocation.options
    if getattr(options, "probe_noise", False):
        from pathlib import Path

        from .probe_noise import probe_noise

        return probe_noise(Path(options.video), start_spec=options.start,
                           end_spec=options.end, reader=options.reader)

    # A config that declares a pipeline list runs through the typed
    # pipeline; the flag-driven stage surface stays on the inherited
    # orchestration until it reaches feature parity.
    if invocation.config.get("pipeline") is not None:
        return _run_typed(invocation)
    from kinovsr.api import (
        VideoFileConfig,
        process_video_options,
        resolve_mlx_cache_limit_gb,
    )

    limit = resolve_mlx_cache_limit_gb(settings)
    if limit > 0 and not settings.quiet:
        get_console().print(f"MLX cache limit: {limit:g} GB")

    process_video_options(
        VideoFileConfig(settings=settings, options=invocation.options))
    return 0


# Flags that select stages on the flag-driven surface; a [pipeline]
# config owns stage composition, so these cannot be combined with it.
_STAGE_SELECTORS = (
    ("upscale", "balanced"),
    ("denoise", "off"),
    ("deblock", "off"),
    ("restore", "off"),
    ("nafnet", "off"),
    ("deflicker", "off"),
    ("cut_detect", "off"),
)

# Geometry/orchestration flags a [pipeline] config also owns (crop, anamorphic,
# junk-edge sanitize, keyframe windowing). Silently ignoring them was a parity
# trap; reject them loudly like the stage selectors. A flag is "set" when its
# value is not one of these unset sentinels.
_GEOMETRY_FLAGS = (
    "crop_bars", "crop_aspect", "square_pixels", "sanitize_edges",
    "snap_start", "gop_align",
)
_UNSET = (None, False, "off", "")


def _pipeline_owned_flags(options) -> list[str]:
    """Flag names a ``[pipeline]`` config owns and that are set on ``options``.

    A ``[pipeline]`` config composes the whole chain, so any stage-selector or
    geometry/orchestration flag alongside it is a config error, not silently
    ignored.
    """
    owned = [name for name, default in _STAGE_SELECTORS
             if getattr(options, name, default) != default]
    owned += [name for name in _GEOMETRY_FLAGS
              if getattr(options, name, None) not in _UNSET]
    return owned


def _source_layout(config):
    """The file-source layout the FIRST stage accepts (MLX preferred).

    The input endpoint must decode into a payload the chain's head can
    consume; native families (videotoolbox) accept only CVPixelBuffer
    layouts. Unknown families fall back to the default so the builder
    reports the real error.
    """
    from kinovsr.pipeline.builder import _resolve_capability
    from kinovsr.processors import Layout, get_factory

    pipeline = config.get("pipeline") or []
    if not pipeline:
        return Layout.MLX_RGB_HWC
    head = config.get(pipeline[0]) or {}
    # Resolve the head's SPECIFIC capability (not a union over the family):
    # videotoolbox offers both an MLX-in upscale and a CV-in interpolate, so
    # the layout depends on which one the head selected.
    try:
        factory = get_factory(head.get("processor"))
        capability = _resolve_capability(
            pipeline[0], factory, head.get("capability"), head.get("profile"))
        layouts = factory.capabilities[capability].accepts.layouts
    except Exception:
        return Layout.MLX_RGB_HWC
    accepted = set(layouts or (Layout.MLX_RGB_HWC,))
    if Layout.MLX_RGB_HWC in accepted:
        return Layout.MLX_RGB_HWC
    for candidate in (Layout.CV_RGBA_HALF, Layout.CV_NV12, Layout.CV_BGRA):
        if candidate in accepted:
            return candidate
    return Layout.MLX_RGB_HWC


def _run_typed(invocation) -> int:
    """Run a [pipeline] config file-to-file through the typed pipeline."""
    from datetime import datetime
    from pathlib import Path

    options = invocation.options
    selected = _pipeline_owned_flags(options)
    if selected:
        flags = ", ".join("--" + n.replace("_", "-") for n in selected)
        get_console().print(
            f"config error: a [pipeline] config owns the full chain; "
            f"drop the flags ({flags}) or the pipeline table",
            style="bold red", markup=False)
        return 2
    if not options.output_dir:
        get_console().print("config error: --output-dir is required",
                            style="bold red", markup=False)
        return 2

    from kinovsr.api import process_video_file, resolve_mlx_cache_limit_gb
    from kinovsr.config import ConfigError
    from kinovsr.media import video_reader as _vr
    from kinovsr.media.naming import sanitize_output_prefix
    from kinovsr.media.timespec import resolve_trim
    from kinovsr.processors.errors import MediaError, PipelineError

    # Reader selection matches the flag surface: ffmpeg = forced, auto =
    # native with fallback, native = never fall back. The module that
    # probes is the module the run decodes with.
    video = Path(options.video)
    reader = None
    if options.reader == "ffmpeg":
        from kinovsr.media import ffmpeg_reader

        reader = ffmpeg_reader
    try:
        _w, _h, fps, total, _tf, _par = (reader or _vr).probe_video(video)
    except Exception as exc:
        if options.reader != "auto":
            get_console().print(f"error: cannot open {video}: {exc}",
                                style="bold red", markup=False)
            return 2
        from kinovsr.media import ffmpeg_reader

        reader = ffmpeg_reader
        try:
            _w, _h, fps, total, _tf, _par = reader.probe_video(video)
        except Exception as fallback_exc:
            get_console().print(
                f"error: cannot open {video}: {fallback_exc}",
                style="bold red", markup=False)
            return 2
    start, end = resolve_trim(options.start, options.end, fps, total)
    # --max-frames caps OUTPUT frames; a time spec is OUTPUT duration,
    # which only the resolved chain's cadence can convert - pass the
    # seconds form through. The input window is start/end.
    max_output_frames = max_output_seconds = None
    if options.max_frames is not None:
        from kinovsr.media.timespec import parse_frames_or_seconds

        try:
            max_output_frames, max_output_seconds = parse_frames_or_seconds(
                options.max_frames)
        except ValueError as exc:
            get_console().print(f"error: bad --max-frames value: {exc}",
                                style="bold red", markup=False)
            return 2

    stem = (f"{sanitize_output_prefix(options.output_prefix)}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root = Path(options.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    output = out_root / f"{stem}_post.mp4"

    limit = resolve_mlx_cache_limit_gb(invocation.settings)
    if limit > 0 and not invocation.settings.quiet:
        get_console().print(f"MLX cache limit: {limit:g} GB")

    try:
        result = process_video_file(
            invocation.config,
            video=video,
            output=output,
            settings=invocation.settings,
            start=start,
            end=end,
            max_output_frames=max_output_frames,
            max_output_seconds=max_output_seconds,
            audio=options.audio,
            audio_codec=options.audio_codec,
            quality=options.encode_quality,
            layout=_source_layout(invocation.config),
            reader=reader,
        )
    except (ConfigError, MediaError, PipelineError) as exc:
        get_console().print(f"error: {exc}", style="bold red", markup=False)
        return 2
    get_console().print(
        f"{result.frames_in} frames in -> {result.frames_out} out "
        f"in {result.elapsed_s:.2f}s", markup=False)
    get_console().print(f"Post: {result.post_path}", markup=False)
    return 0
