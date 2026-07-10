#!/usr/bin/env python3
"""Compatibility shim: the harness now lives in the kinovsr package.

The installed entry point is ``kinovsr`` (``kinovsr.cli.main:main``).
This file remains so source-checkout invocations of
``scripts/vsr_harness.py`` keep working, including legacy flag spellings
(parsed as hidden aliases). It is retired in M4 once nothing references
it.
"""

import sys
from pathlib import Path

# Source-checkout support: running this file directly puts scripts/ on
# sys.path, not the repository root that holds the kinovsr package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinovsr.cli.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
