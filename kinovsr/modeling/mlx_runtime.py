"""MLX lifecycle helpers for KinoVSR-managed threads."""

from __future__ import annotations

import sys
from typing import Any


def clear_mlx_thread_state() -> None:
    """Release MLX state owned by the current thread, if MLX was loaded.

    MLX 0.32.1 makes its thread-local compile-cache cleanup part of
    ``clear_streams()``.  Calling it while the Python thread state is still
    alive prevents cached Python objects from reaching their final decref in
    the later native TLS-destructor phase.

    Looking in ``sys.modules`` preserves KinoVSR's lightweight import boundary:
    importing KinoVSR solely for configuration or API inspection still does
    not initialize MLX or Metal.
    """
    mx: Any = sys.modules.get("mlx.core")
    if mx is not None:
        mx.clear_streams()
