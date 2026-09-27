"""kinovsr probe: source-analysis subcommands.

Dispatches to the probe implementations; each was a standalone runtime
script before the M4 command split. A family's own diagnostic (nafnet)
lives in that family and is resolved through the processor catalog, so the
CLI loads the family only when its probe runs.
"""

import logging

from kinovsr.processors.catalog import get_probe, probe_families

from .probe_edges import run_probe_edges
from .probe_noise import run_probe_noise

_SUBCOMMANDS = tuple(sorted(("edges", "noise", *probe_families())))
_log = logging.getLogger(__name__)


def run_probe_command(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        log = _log.info if argv else _log.error
        log("usage: kinovsr probe {%s} ...", "|".join(_SUBCOMMANDS))
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name == "edges":
        return run_probe_edges(rest) or 0
    if name in probe_families():
        return get_probe(name)(rest) or 0
    if name == "noise":
        return run_probe_noise(rest) or 0
    _log.error(
        "unknown probe subcommand %r (available: %s)",
        name,
        ", ".join(_SUBCOMMANDS),
    )
    return 2
