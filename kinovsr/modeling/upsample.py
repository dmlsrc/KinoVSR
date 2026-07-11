"""Shared torch-parity bicubic upsampling.

Promoted from the SAFMN net once toflow became a second consumer
(planning 05: shared code moves to modeling/ only on demonstrated
reuse). SAFMN uses it as its checkpoint-faithful upsampler; TOFlow SR
uses it to build the bicubic-upsampled base its residual is trained
against. Both need exact torch ``F.interpolate(mode="bicubic",
align_corners=False)`` semantics, which is what the phase tables
reproduce.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def _cubic_phases(r: int) -> list:
    """Per-phase (floor offset, 4 tap weights) for torch bicubic interpolation
    (Keys kernel, A=-0.75, align_corners=False) at integer upscale r. Output
    pixel j of each block maps to source coordinate (j + 0.5)/r - 0.5; the four
    taps sit at floor-1..floor+2."""
    a = -0.75
    phases = []
    for j in range(r):
        src = (j + 0.5) / r - 0.5
        f = -1 if src < 0 else 0                        # floor(src) for src in (-0.5, 0.5)
        t = src - f
        w_m1 = a * (1 + t) ** 3 - 5 * a * (1 + t) ** 2 + 8 * a * (1 + t) - 4 * a
        w_0 = (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1
        w_1 = (a + 2) * (1 - t) ** 3 - (a + 3) * (1 - t) ** 2 + 1
        w_2 = a * (2 - t) ** 3 - 5 * a * (2 - t) ** 2 + 8 * a * (2 - t) - 4 * a
        phases.append((f, (w_m1, w_0, w_1, w_2)))
    return phases


def _bicubic_axis_up(x: Any, r: int, axis: int) -> Any:
    """Bicubic upsample by integer r along axis 1 (H) or 2 (W), matching torch
    F.interpolate(mode="bicubic", align_corners=False). Edge-replicate padding by 2
    reproduces torch's tap-index clamping at the borders exactly (taps reach at most
    2 outside). Each of the r phases is a fixed 4-tap blend of shifted slices."""
    n = x.shape[1] if axis == 1 else x.shape[2]
    if axis == 1:
        edge0, edge1 = x[:, :1], x[:, -1:]
        xp = mx.concatenate([edge0, edge0, x, edge1, edge1], axis=1)
    else:
        edge0, edge1 = x[:, :, :1], x[:, :, -1:]
        xp = mx.concatenate([edge0, edge0, x, edge1, edge1], axis=2)

    def sl(o):
        return xp[:, o:o + n] if axis == 1 else xp[:, :, o:o + n]

    phases = []
    for f, wt in _cubic_phases(r):
        base = 2 + f - 1                                # first tap of block i is i + f - 1
        s = (wt[0] * sl(base) + wt[1] * sl(base + 1)
             + wt[2] * sl(base + 2) + wt[3] * sl(base + 3))
        phases.append(s)
    y = mx.stack(phases, axis=axis + 1)                 # (n, N, r, ...) on the chosen axis
    shape = list(x.shape)
    shape[axis] = shape[axis] * r
    return y.reshape(shape)


def bicubic_up(x: Any, r: int) -> Any:
    """r x r bicubic upsample (torch semantics), separable rows-then-columns."""
    return _bicubic_axis_up(_bicubic_axis_up(x, r, 1), r, 2)


__all__ = ["bicubic_up"]
