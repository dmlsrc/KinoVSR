"""Pure crop geometry and frame slicing owned by the crop family."""

from __future__ import annotations

from typing import Any

# Fraction of the slack placed left of / above the crop window.
_ANCHORS = {
    "top-left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top-right": (1.0, 0.0),
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "right": (1.0, 0.5),
    "bottom-left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom-right": (1.0, 1.0),
}


def compute_aspect_crop(
    w: int,
    h: int,
    ar_w: int,
    ar_h: int,
    dx: int = 0,
    dy: int = 0,
    anchor: str = "center",
) -> tuple[int, int, int, int]:
    """Return the largest even crop window matching ``ar_w:ar_h``.

    The result is ``(top, bottom, left, right)``. The window is placed at one
    of the nine anchors, shifted by ``(dx, dy)`` pixels, and clamped inside the
    frame. Exact ratios are approximated to the nearest even dimensions.
    """
    if ar_w <= 0 or ar_h <= 0:
        raise ValueError(f"aspect must be positive, got {ar_w}:{ar_h}")
    if anchor not in _ANCHORS:
        raise ValueError(
            f"anchor must be one of {sorted(_ANCHORS)}, got {anchor!r}")

    # Even-integer boxes can only approximate most ratios; evaluate the three
    # natural fit orders and keep the pair with the smallest relative ratio
    # error (ties broken toward the larger area).
    tw0 = min(w, (h * ar_w) // ar_h)
    th0 = min(h, (tw0 * ar_h) // ar_w)
    tw0 = min(tw0, (th0 * ar_w) // ar_h)
    chain = (tw0 - tw0 % 2, th0 - th0 % 2)

    fw = min(w, (h * ar_w) // ar_h)
    fw -= fw % 2
    fh = min(h, (fw * ar_h) // ar_w)
    fh -= fh % 2
    width_first = (fw, fh)

    gh = min(h, (w * ar_h) // ar_w)
    gh -= gh % 2
    gw = min(w, (gh * ar_w) // ar_h)
    gw -= gw % 2
    height_first = (gw, gh)

    target = ar_w / ar_h
    valid = [
        candidate
        for candidate in {chain, width_first, height_first}
        if candidate[0] >= 2 and candidate[1] >= 2
    ]
    if not valid:
        raise ValueError(f"aspect {ar_w}:{ar_h} leaves no picture in {w}x{h}")
    tw, th = min(
        valid,
        key=lambda candidate: (
            abs(candidate[0] / candidate[1] / target - 1.0),
            -(candidate[0] * candidate[1]),
        ),
    )
    ax, ay = _ANCHORS[anchor]
    left = max(0, min(int(round((w - tw) * ax)) + int(dx), w - tw))
    top = max(0, min(int(round((h - th) * ay)) + int(dy), h - th))
    return top, h - th - top, left, w - tw - left


def crop_rgb(rgb: Any, edges: tuple[int, int, int, int]) -> Any:
    """Crop edge bands from an ``HWC`` or ``NHWC`` frame tensor."""
    top, bottom, left, right = edges
    if rgb.ndim == 4:
        height, width = int(rgb.shape[1]), int(rgb.shape[2])
        return rgb[:, top:height - bottom, left:width - right]
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    return rgb[top:height - bottom, left:width - right]


__all__ = ["compute_aspect_crop", "crop_rgb"]
