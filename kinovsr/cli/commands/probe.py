"""kinovsr probe: source-analysis subcommands.

Dispatches to the probe implementations; each was a standalone runtime
script before the M4 command split.
"""

from __future__ import annotations

_SUBCOMMANDS = ("edges", "nafnet", "noise")


def run_probe_command(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        from kinovsr.ui.console import get_console

        get_console().print(
            f"usage: kinovsr probe {{{'|'.join(_SUBCOMMANDS)}}} ...",
            markup=False)
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name == "edges":
        from .probe_edges import run_probe_edges

        return run_probe_edges(rest) or 0
    if name == "nafnet":
        from .probe_nafnet import run_probe_nafnet

        return run_probe_nafnet(rest) or 0
    if name == "noise":
        from .probe_noise import run_probe_noise

        return run_probe_noise(rest) or 0
    from kinovsr.ui.console import get_console

    get_console().print(
        f"unknown probe subcommand {name!r} "
        f"(available: {', '.join(_SUBCOMMANDS)})", markup=False)
    return 2
