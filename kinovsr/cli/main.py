"""The installed console entry point: parse, assemble, run, report."""

from __future__ import annotations

from kinovsr.config import ConfigError
from kinovsr.ui.console import get_console

from .args import build_parser, validate_args
from .config import assemble


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    try:
        invocation = assemble(args)
    except ConfigError as exc:
        get_console().print(f"config error: {exc}", style="bold red")
        return 2

    settings = invocation.settings
    limit = settings.mlx_cache_limit_gb
    if limit is None:
        limit = 1.0  # the historical harness default
    if limit > 0:
        import mlx.core as mx

        mx.set_cache_limit(int(limit * (1000 ** 3)))
        mx.clear_cache()
        if not settings.quiet:
            get_console().print(f"MLX cache limit: {limit:g} GB")

    from kinovsr import require_pyobjc
    from kinovsr.api import VideoFileConfig, process_video_file

    require_pyobjc()
    process_video_file(
        VideoFileConfig(settings=settings, options=invocation.options))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
