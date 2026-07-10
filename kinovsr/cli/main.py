"""The installed console entry point: parse, assemble, run, report."""

from __future__ import annotations

from kinovsr.config import ConfigError
from kinovsr.ui.console import get_console

from .args import build_parser, validate_args
from .config import assemble


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if argv and argv[0] == "weights":
        # Subcommand surface (M3): read-only weight listing/verification.
        # The flat processing CLI remains the default invocation shape
        # until the M4 command split.
        from .weights_cmd import run_weights_command

        return run_weights_command(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    try:
        invocation = assemble(args)
    except ConfigError as exc:
        get_console().print(f"config error: {exc}", style="bold red")
        return 2

    settings = invocation.settings
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


if __name__ == "__main__":
    raise SystemExit(main())
