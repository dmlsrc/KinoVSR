"""The installed console entry point: subcommand dispatch only.

Each command owns its argument surface and terminal wiring; this module
just routes. The flat processing invocation (``kinovsr --video ...``)
remains the default route and is also reachable as ``kinovsr run``.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from kinovsr.ui.logging import configure_logging

    configure_logging()
    if argv is None:
        import sys

        argv = sys.argv[1:]
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

    from .commands.run import run_video_command

    if argv and argv[0] == "run":
        return run_video_command(argv[1:])
    return run_video_command(argv)


if __name__ == "__main__":
    raise SystemExit(main())
