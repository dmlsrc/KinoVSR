"""kinovsr probe: source-analysis subcommands.

Dispatches to the probe implementations; each was a standalone runtime
script before the M4 command split.
"""

from __future__ import annotations

import logging

_SUBCOMMANDS = ("edges", "nafnet", "noise")
_log = logging.getLogger(__name__)


def run_probe_command(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        log = _log.info if argv else _log.error
        log("usage: kinovsr probe {%s} ...", "|".join(_SUBCOMMANDS))
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
    _log.error(
        "unknown probe subcommand %r (available: %s)",
        name,
        ", ".join(_SUBCOMMANDS),
    )
    return 2
