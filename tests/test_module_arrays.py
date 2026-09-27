"""Module-level MLX arrays must be usable from any thread.

The streaming runtime runs MLX work on its own lane thread and stream. A
module-level array that is still a lazy graph binds to the importing thread's
default stream, and evaluating it from the lane fails with "There is no
Stream(gpu, 0) in current thread". FBCNN's auto quality estimate died that way
at its first refresh, on the DCT basis built at import.
"""

import importlib
import threading
from pathlib import Path

import mlx.core as mx
import pytest

pytestmark = pytest.mark.unit

_PACKAGE = Path(__file__).resolve().parents[1] / "kinovsr"
# Modules that need an optional extra the environment may lack are skipped,
# not failed; every other import error is a real failure.
_OPTIONAL = frozenset({"av", "cv2", "numpy"})


def _module_names() -> list[str]:
    names = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        parts = list(path.relative_to(_PACKAGE.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names.append(".".join(parts))
    return names


def _module_arrays() -> list[tuple[str, mx.array]]:
    found = []
    for name in _module_names():
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name in _OPTIONAL:
                continue
            raise
        found.extend(
            (f"{name}.{attr}", value)
            for attr, value in vars(module).items()
            if isinstance(value, mx.array)
        )
    return found


def test_module_level_arrays_evaluate_on_a_lane_stream():
    arrays = _module_arrays()
    failures: list[str] = []

    def lane() -> None:
        stream = mx.new_thread_unsafe_stream(mx.default_device())
        with mx.stream(stream):
            for label, value in arrays:
                try:
                    mx.eval(value + 0)
                except RuntimeError as exc:
                    failures.append(f"{label}: {exc}")

    worker = threading.Thread(target=lane)
    worker.start()
    worker.join()
    assert arrays, "found no module-level MLX arrays; the scan is broken"
    assert not failures, "module-level arrays bound to the importing thread:\n" + "\n".join(
        failures
    )
