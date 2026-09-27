"""Up-front check that native optical flow is real and correctly signed.

Vision and VideoToolbox motion estimation can return all-zero flow, or flow
of the wrong sign, on some devices and geometries. Every consumer of a native
flow engine runs this check once when it opens the engine, so a bad engine
fails loudly instead of silently disabling motion compensation.
"""

from collections.abc import Callable

import mlx.core as mx


def self_test_frames(width: int, height: int, shift: int) -> tuple[mx.array, mx.array]:
    """Aperiodic self-test content for the Vision and vtme matchers.

    A modular-hash pattern admits EXACT false matches at small
    displacement combinations (37*dx + 17*dy wraps), which Vision's
    pyramid locks onto at many geometries. Irrational trig value noise
    has no exact aliases, and a few Gaussian blobs anchor the coarse
    levels isotropically (oriented sinusoids bias the unobservable axis
    via the aperture problem). Measured exact (+shift, 0) from 16x16
    through 1920x1080.
    """
    ys, xs = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
    xf, yf = xs.astype(mx.float32), ys.astype(mx.float32)
    n = mx.sin(xf * 12.9898 + yf * 78.233) * 43758.5453
    hashn = n - mx.floor(n)
    s2 = (0.22 * min(width, height)) ** 2

    def blob(cx: float, cy: float, sign: float) -> mx.array:
        return sign * mx.exp(-((xf - cx * width) ** 2 + (yf - cy * height) ** 2) / s2)

    blobs = (
        blob(0.30, 0.35, 1.0)
        + blob(0.72, 0.60, -1.0)
        + blob(0.45, 0.80, 0.8)
        + blob(0.80, 0.20, -0.7)
    )
    # The vtme block matcher also reads this content exactly at every
    # probed geometry: the trig value noise is block-distinctive (no
    # repeats inside the search window) and the smooth blobs cost it
    # nothing, so vision and vtme share these frames.
    base = 0.5 + (hashn - 0.5) * 0.5 + blobs * 0.25
    ref = mx.clip(
        mx.stack(
            [
                base,
                0.45 + (hashn - 0.5) * 0.4 + blobs * 0.22,
                0.55 + (hashn - 0.5) * 0.45 + blobs * 0.20,
            ],
            axis=-1,
        ),
        0.0,
        1.0,
    ).astype(mx.float32)
    left = mx.broadcast_to(ref[:, :1], (height, shift, 3))
    curr = mx.concatenate([left, ref[:, : width - shift]], axis=1)
    mx.eval(ref, curr)
    return ref, curr


def self_test_flow(
    forward: Callable[[mx.array, mx.array], mx.array],
    width: int,
    height: int,
    *,
    name: str,
    consumer: str,
) -> None:
    """Catch silent-zero and wrong-sign flow up front.

    ``forward(curr, ref)`` returns the flow of ``ref`` into ``curr``: the
    negated engine ``compute(curr, ref)``. ``name`` identifies the engine and
    ``consumer`` the feature that cannot run safely when the check fails.
    """
    if width < 16 or height < 16:
        raise RuntimeError(f"{name} self-test requires at least 16x16; got {width}x{height}")
    shift = 3 if width >= 32 else 1
    ref, curr = self_test_frames(width, height, shift)
    fwd = forward(curr, ref)
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    crop = fwd[y0:y1, x0:x1] if y1 > y0 and x1 > x0 else fwd
    mean_x = mx.mean(crop[..., 0])
    mean_y = mx.mean(crop[..., 1])
    max_abs = mx.max(mx.abs(fwd))
    mx.eval(mean_x, mean_y, max_abs)
    mean_x_f = float(mean_x)
    mean_y_f = float(mean_y)
    max_abs_f = float(max_abs)
    expected = float(shift)
    if max_abs_f < 0.25:
        raise RuntimeError(
            f"{name} self-test returned all-zero/near-zero flow for "
            f"{width}x{height}; {consumer} is unsafe on this clip/device."
        )
    if (
        mean_x_f < expected * 0.65
        or mean_x_f > expected * 1.35
        or abs(mean_y_f) > max(0.75, expected * 0.5)
    ):
        raise RuntimeError(
            f"{name} self-test failed for "
            f"{width}x{height}: expected about +{expected:.1f}px horizontal "
            f"flow, got mean=({mean_x_f:.3f}, {mean_y_f:.3f}), "
            f"max_abs={max_abs_f:.3f}."
        )


__all__ = ["self_test_flow", "self_test_frames"]
