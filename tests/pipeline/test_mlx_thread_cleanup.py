"""MLX 0.32.1 compile-cache teardown regression."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_retained_tuple_compile_cache_is_cleared_before_lane_exit() -> None:
    """A retained tuple-output compile used to segfault in pthread teardown."""
    root = Path(__file__).parents[2]
    code = textwrap.dedent(
        """
        import mlx.core as mx

        from kinovsr.pipeline.streaming import _AffinityLane

        retained = {}
        lane = _AffinityLane("mlx-teardown-regression")

        def compile_once():
            function = mx.compile(lambda value: (value + 1, value * 2))
            retained["function"] = function
            outputs = function(mx.array([1.0]))
            mx.eval(outputs)
            return tuple(output.item() for output in outputs)

        assert lane.call(compile_once) == (2.0, 2.0)
        lane.stop()
        print("clean")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "clean"
