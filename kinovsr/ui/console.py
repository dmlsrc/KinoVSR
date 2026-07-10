"""The shared Rich console - one terminal surface for logs and bars.

Progress bars (:mod:`kinovsr.ui.progress`) and the logging handler
(:mod:`kinovsr.ui.logging`) render to this one console so they interleave
cleanly instead of fighting over the terminal.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# Subdued level colors - readable in a long session, not a fireworks display.
# inherit=True (the default) keeps rich's styles for everything not named here.
_THEME = Theme({
    "logging.level.info": "cyan",
    "logging.level.warning": "yellow",
    "logging.level.error": "bold red",
    # rich's default log.time is *dim* cyan, which reads as washed-out gray on
    # a dark terminal; keep the timestamp a plain, readable cyan.
    "log.time": "cyan",
})


_console: Console | None = None


def get_console() -> Console:
    """Return the shared ``Console``, creating it lazily on first call.

    Renders to stderr so machine-readable output (a future ``--json``) can
    own stdout.
    """
    global _console
    if _console is None:
        _console = Console(stderr=True, theme=_THEME, highlight=False)
    return _console
