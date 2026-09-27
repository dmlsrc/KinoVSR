"""Bilinear sampling primitives: sampling at arbitrary coordinates and resizing.

Pure MLX on NHWC arrays, following torch's coordinate conventions. Flow
warping, SpyNet on either backend, and several families build on these; the
module imports no other KinoVSR module, so any of them can depend on it.
"""

import mlx.core as mx


def bilinear_sample(x: mx.array, sy: mx.array, sx: mx.array, pad: str = "border") -> mx.array:
    """Sample x (N,H,W,C) at (sy,sx) (each (N,oH,oW)) -> (N,oH,oW,C). 'border'
    clamps out-of-range to the edge; 'zeros' returns 0 outside."""
    n, h, w, c = x.shape
    oh, ow = sy.shape[1], sy.shape[2]
    y0 = mx.floor(sy)
    x0 = mx.floor(sx)
    ly = (sy - y0)[..., None]
    lx = (sx - x0)[..., None]
    y0i = y0.astype(mx.int32)
    x0i = x0.astype(mx.int32)
    flat = x.reshape(n, h * w, c)

    def g(yi: mx.array, xi: mx.array) -> mx.array:
        idx = (mx.clip(yi, 0, h - 1) * w + mx.clip(xi, 0, w - 1)).reshape(n, oh * ow, 1)
        v = mx.take_along_axis(flat, mx.broadcast_to(idx, (n, oh * ow, c)), axis=1).reshape(
            n, oh, ow, c
        )
        if pad == "zeros":
            valid = ((yi >= 0) & (yi <= h - 1) & (xi >= 0) & (xi <= w - 1)).astype(x.dtype)
            v = v * valid[..., None]
        return v

    v00 = g(y0i, x0i)
    v01 = g(y0i, x0i + 1)
    v10 = g(y0i + 1, x0i)
    v11 = g(y0i + 1, x0i + 1)
    out = (1 - ly) * (1 - lx) * v00 + (1 - ly) * lx * v01 + ly * (1 - lx) * v10 + ly * lx * v11
    return out.astype(x.dtype)  # fp32 grid weights would otherwise upcast features


def resize(x: mx.array, oh: int, ow: int, align_corners: bool) -> mx.array:
    """Bilinear resize NHWC x to (oh, ow) (edge-clamped), matching torch's
    align_corners True/False coordinate maps."""
    n, h, w, _ = x.shape
    if align_corners:
        ry = (h - 1) / (oh - 1) if oh > 1 else 0.0
        rx = (w - 1) / (ow - 1) if ow > 1 else 0.0
        sy1 = mx.arange(oh, dtype=mx.float32) * ry
        sx1 = mx.arange(ow, dtype=mx.float32) * rx
    else:
        sy1 = (mx.arange(oh, dtype=mx.float32) + 0.5) * (h / oh) - 0.5
        sx1 = (mx.arange(ow, dtype=mx.float32) + 0.5) * (w / ow) - 0.5
    sy = mx.broadcast_to(sy1.reshape(1, oh, 1), (n, oh, ow))
    sx = mx.broadcast_to(sx1.reshape(1, 1, ow), (n, oh, ow))
    return bilinear_sample(x, sy, sx, "border")
