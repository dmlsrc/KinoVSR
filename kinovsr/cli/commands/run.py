"""The processing command: parse, assemble, run, report.

This is the flat invocation (``kinovsr --video ...``), also reachable as
an explicit subcommand (``kinovsr run --video ...``). It owns the
terminal wiring for a processing run - logging configuration and the
MLX cache-limit echo - and hands the resolved invocation to the
:func:`kinovsr.api.process_video_file` facade.
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
        get_console().print(f"config error: {exc}", style="bold red")
        return 2

    settings = invocation.settings
    from kinovsr.ui import configure_logging_from_settings

    configure_logging_from_settings(settings)
    from kinovsr.api import (
        VideoFileConfig,
        process_video_file,
        resolve_mlx_cache_limit_gb,
    )

    limit = resolve_mlx_cache_limit_gb(settings)
    if limit > 0 and not settings.quiet:
        get_console().print(f"MLX cache limit: {limit:g} GB")

    process_video_file(
        VideoFileConfig(settings=settings, options=invocation.options))
    return 0
