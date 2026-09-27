"""MLX lifecycle helpers for KinoVSR-managed threads."""

import mlx.core as mx


def clear_mlx_thread_state() -> None:
    """Release MLX state owned by the current thread.

    MLX 0.32.1 makes its thread-local compile-cache cleanup part of
    ``clear_streams()``.  Calling it while the Python thread state is still
    alive prevents cached Python objects from reaching their final decref in
    the later native TLS-destructor phase.
    """
    mx.clear_streams()
