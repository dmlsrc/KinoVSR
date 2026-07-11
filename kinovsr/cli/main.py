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
    # Subcommand surface: the flat processing CLI remains the default
    # invocation shape until step 6 moves it onto the typed pipeline.
    if argv and argv[0] == "weights":
        from .commands.weights import run_weights_command

        return run_weights_command(argv[1:])
    if argv and argv[0] == "probe":
        from .commands.probe import run_probe_command

        return run_probe_command(argv[1:])
    if argv and argv[0] == "metrics":
        from .commands.metrics import run_metrics_command

        return run_metrics_command(argv[1:])
    if argv and argv[0] == "artifacts":
        from .commands.artifacts import run_artifacts_command

        return run_artifacts_command(argv[1:])

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


if __name__ == "__main__":
    raise SystemExit(main())
