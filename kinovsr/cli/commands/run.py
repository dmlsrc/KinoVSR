"""The processing command: parse, assemble, run, report.

This is the flat invocation (``kinovsr --video ...``), also reachable as
an explicit subcommand (``kinovsr run --video ...``). It owns the
terminal wiring for a processing run - logging configuration and the
MLX cache-limit echo - and hands the resolved invocation to the
:func:`kinovsr.api.process_video_file` - flag invocations assemble
into the same typed config a ``[pipeline]`` TOML expresses (M6).
"""

from __future__ import annotations

import logging

from kinovsr.config import ConfigError

from ..args import build_parser, validate_args
from ..config import assemble

_log = logging.getLogger(__name__)


def run_video_command(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    try:
        invocation = assemble(args)
    except ConfigError as exc:
        _log.error("config error: %s", exc)
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

        return probe_noise(
            Path(options.video),
            start_spec=options.start,
            end_spec=options.end,
            reader=options.reader,
            chunk_size=options.video_chunk_size,
        )

    # A config that declares a pipeline list runs through the typed
    # pipeline as written; a flag-driven invocation assembles into the
    # same typed config (M6 step 3: one engine, two authoring surfaces).
    if invocation.config.get("pipeline") is not None:
        return _run_typed(invocation)
    return _run_typed(invocation, assemble_flags=True)


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

# Geometry flags a [pipeline] config also owns (crop, anamorphic, junk-edge
# sanitize compose as stages). Silently ignoring them was a parity trap;
# reject them loudly like the stage selectors. A flag is "set" when its value
# is not one of these unset sentinels. Keyframe windowing (--snap-start /
# --gop-align) is NOT here: it is run-level orchestration like --start, now
# threaded through the typed endpoints.
_GEOMETRY_FLAGS = (
    "crop_bars", "crop_aspect", "square_pixels", "sanitize_edges",
)
_PIPELINE_OPTIONS = (
    ("target_fps", None),
    ("temporal_mode", "normal"),
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
    owned += [name for name, default in _PIPELINE_OPTIONS
              if getattr(options, name, default) != default]
    return owned


def _source_layout(config, *, source_color: str = "auto",
                   source_range: str = "auto"):
    """The file-source layout the FIRST stage accepts (MLX preferred).

    The input endpoint must decode into a payload the chain's head can
    consume; native families (videotoolbox) accept only CVPixelBuffer
    layouts. Unknown families fall back to the default so the builder
    reports the real error.
    """
    from kinovsr.pipeline.builder import _resolve_capability
    from kinovsr.processors import Layout, get_factory
    from kinovsr.processors.errors import PipelineError

    # Forced color/range reinterpretation is implemented by the RGBAHalf MLX
    # decode route. Native VideoToolbox heads accept that upload bridge, so an
    # explicit correction request takes precedence over their zero-copy
    # preferred source layout.
    if source_color != "auto" or source_range != "auto":
        return Layout.MLX_RGB_HWC

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
        # A family may prefer a specific decode layout for its head position
        # (videotoolbox upscale wants its mode's own session format so the
        # decode feeds the scaler zero-copy - the harness's mainstream path).
        preferred = getattr(factory, "preferred_source_layout", None)
        if callable(preferred):
            layout = preferred(capability=capability,
                               profile=head.get("profile"))
            if layout is not None:
                return layout
        layouts = factory.capabilities[capability].accepts.layouts
    except PipelineError:
        return Layout.MLX_RGB_HWC
    accepted = set(layouts or (Layout.MLX_RGB_HWC,))
    if Layout.MLX_RGB_HWC in accepted:
        return Layout.MLX_RGB_HWC
    for candidate in (Layout.CV_RGBA_HALF, Layout.CV_NV12, Layout.CV_BGRA):
        if candidate in accepted:
            return candidate
    return Layout.MLX_RGB_HWC


def _run_typed(invocation, assemble_flags: bool = False) -> int:
    """Run file-to-file through the typed pipeline.

    ``assemble_flags=False``: the invocation carries a hand-written
    ``[pipeline]`` config; stage/geometry flags alongside it are a config
    error (C8). ``assemble_flags=True``: the flag surface IS the config
    source - the stage selectors and geometry flags assemble into the
    same typed config after the source is probed.
    """
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4

    options = invocation.options
    selected = () if assemble_flags else _pipeline_owned_flags(options)
    if selected:
        flags = ", ".join("--" + n.replace("_", "-") for n in selected)
        _log.error(
            "config error: a [pipeline] config owns the full chain; "
            "drop the flags (%s) or the pipeline table",
            flags,
        )
        return 2
    if not options.output_dir:
        _log.error("config error: --output-dir is required")
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
    except (OSError, RuntimeError, PipelineError) as exc:
        if options.reader != "auto":
            _log.error("cannot open %s: %s", video, exc)
            return 2
        from kinovsr.media import ffmpeg_reader

        reader = ffmpeg_reader
        try:
            _w, _h, fps, total, _tf, _par = reader.probe_video(video)
        except (OSError, RuntimeError, PipelineError) as fallback_exc:
            _log.error("cannot open %s: %s", video, fallback_exc)
            return 2
    # A time-form window needs the real clock when it is not uniform;
    # frame-form windows are ordinals and need no table walk.
    trim_table = None
    from kinovsr.media.timespec import parse_frames_or_seconds as _pfs

    try:
        wants_time_form = any(
            spec is not None and str(spec).strip() != ""
            and _pfs(str(spec))[1] is not None
            for spec in (options.start, options.end))
    except ValueError:
        wants_time_form = False  # resolve_trim reports the bad spec
    if wants_time_form:
        read_table = getattr(reader or _vr, "read_sample_table", None)
        if read_table is not None:
            try:
                trim_table = read_table(video)
            except (OSError, RuntimeError, PipelineError) as exc:
                _log.error("cannot inspect timing of %s: %s", video, exc)
                return 2
    try:
        start, end = resolve_trim(options.start, options.end, fps, total,
                                  table=trim_table)
    except ValueError as exc:
        _log.error("bad --start/--end value: %s", exc)
        return 2
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
            _log.error("bad --max-frames value: %s", exc)
            return 2

    stem = (f"{sanitize_output_prefix(options.output_prefix)}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{uuid4().hex[:8]}")
    out_root = Path(options.output_dir)
    output = out_root / f"{stem}_post.mp4"
    # Per-frame PNG dumps land in sibling {stem}_pre / {stem}_post dirs,
    # matching the harness naming (the _post dir sits beside _post.mp4).
    save_pre = out_root / f"{stem}_pre" if options.save_pre_frames else None
    save_post = out_root / f"{stem}_post" if options.save_post_frames else None
    comparison = (out_root / f"{stem}_comparison.mp4"
                  if options.comparison else None)

    limit = resolve_mlx_cache_limit_gb(invocation.settings)
    if limit > 0:
        _log.info("MLX cache limit: %g GB", limit)

    # The flag surface assembles into the same typed config a hand-written
    # [pipeline] TOML expresses (the probed dims feed the evenness bump).
    config = invocation.config
    if assemble_flags:
        from ..assemble_pipeline import assemble_pipeline

        try:
            config = assemble_pipeline(options, width=_w, height=_h)
        except ConfigError as exc:
            _log.error("config error: %s", exc)
            return 2

    try:
        from kinovsr.ui import RichReporter

        # Live per-frame progress (harness parity): Rich auto-detects the
        # console, so piped/logged runs degrade to the phase-summary INFO
        # line instead of drawing bars.
        with RichReporter() as reporter:
            result = process_video_file(
                config,
                video=video,
                output=output,
                settings=invocation.settings,
                reporter=reporter,
                start=start,
                end=end,
                max_output_frames=max_output_frames,
                max_output_seconds=max_output_seconds,
                audio=options.audio,
                audio_codec=options.audio_codec,
                save_audio_sidecar=options.save_audio_sidecar,
                quality=options.encode_quality,
                chunk_size=options.video_chunk_size,
                source_color=options.source_color,
                source_range=options.source_range,
                encode_chroma=options.encode_chroma,
                save_pre_frames=save_pre,
                save_post_frames=save_post,
                comparison=comparison,
                snap_start=options.snap_start,
                gop_align=options.gop_align,
                gop_min_window=options.gop_min_window,
                gop_max_window=options.gop_max_window,
                cut_log=options.cut_log,
                skip_post_mp4=options.skip_post_mp4,
                noise_map_debug=options.noise_map_debug,
                overwrite=options.overwrite,
                layout=_source_layout(
                    config,
                    source_color=options.source_color,
                    source_range=options.source_range,
                ),
                reader=reader,
            )
    except (ConfigError, MediaError, PipelineError) as exc:
        _log.error("processing failed: %s", exc)
        return 2
    _log.info(
        "%s frames in -> %s out in %.2fs",
        result.frames_in,
        result.frames_out,
        result.elapsed_s,
    )
    if result.post_path is not None:
        _log.info("Post: %s", result.post_path)
    if result.comparison_path is not None:
        _log.info("Comparison: %s", result.comparison_path)
    return 0
