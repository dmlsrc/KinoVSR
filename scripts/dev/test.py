#!/usr/bin/env python3
"""Run KinoVSR's quick developer lane or the complete pytest suite.

Extra arguments are passed through to pytest, for example:

    python scripts/dev/test.py quick -x tests/media/test_timing.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LANES = {
    "quick": ("-m", "not integration and not slow and not requires_weights"),
    "full": (),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lane",
        choices=tuple(LANES),
        default="quick",
        nargs="?",
        help="test selection (default: quick)",
    )
    args, pytest_args = parser.parse_known_args()
    os.chdir(REPO)
    return pytest.main([*LANES[args.lane], *pytest_args])


if __name__ == "__main__":
    raise SystemExit(main())
