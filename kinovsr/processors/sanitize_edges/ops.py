"""MLX pixel operations owned by the sanitize-edges family."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def _nn_up(x: Any, ratio: int) -> Any:
    height, width = x.shape[0], x.shape[1]
    y = mx.broadcast_to(
        x[:, None, :, None, :],
        (height, ratio, width, ratio, x.shape[-1]),
    )
    return y.reshape(height * ratio, width * ratio, x.shape[-1])


def _to_unit(rgb: Any) -> Any:
    f = rgb[..., :3].astype(mx.float32)
    return f / 255.0 if rgb.dtype == mx.uint8 else f


# Feather vectors are identical for a given output-pixel band and feather.
_FEATHER_CACHE: dict[tuple[int, int], Any] = {}


def _feather_weights(band_px: int, feather_px: int, dtype: Any) -> Any:
    """Return a solid band followed by a linear feather to zero."""
    key = (band_px, feather_px)
    base = _FEATHER_CACHE.get(key)
    if base is None:
        weights = [1.0] * band_px
        for i in range(feather_px):
            weights.append(1.0 - (i + 0.5) / feather_px)
        base = mx.array(weights, dtype=mx.float32)
        _FEATHER_CACHE[key] = base
    return base.astype(dtype)


def restore_borders(
    out_rgb: Any,
    src_rgb: Any,
    edges: tuple[int, int, int, int],
    feather: int = 2,
) -> Any:
    """Composite original edge bands over an integer-upscaled output frame.

    The source bands are nearest-upscaled to match and blended into the
    processed output. Splices are combined per axis, so a frame costs at most
    two full-size copies; only source edge zones are converted and upscaled.
    """
    top, bottom, left, right = edges
    feather = max(0, int(feather))
    src_h, src_w = int(src_rgb.shape[0]), int(src_rgb.shape[1])
    out_h, out_w = int(out_rgb.shape[0]), int(out_rgb.shape[1])
    if out_h % src_h or out_w % src_w or out_h // src_h != out_w // src_w:
        raise ValueError(
            f"output {out_h}x{out_w} is not an integer multiple of source "
            f"{src_h}x{src_w}")
    ratio = out_h // src_h
    dtype = out_rgb.dtype if out_rgb.dtype != mx.uint8 else mx.float32

    def mixed_zone(
        output: Any,
        axis: int,
        from_end: bool,
        band: int,
        zone_feather: int,
    ) -> tuple[Any, int]:
        zone_src = band + zone_feather
        zone_out = zone_src * ratio
        if axis == 0:
            src_slice = (
                src_rgb[src_h - zone_src:] if from_end else src_rgb[:zone_src]
            )
            out_slice = (
                output[out_h - zone_out:] if from_end else output[:zone_out]
            )
        else:
            src_slice = (
                src_rgb[:, src_w - zone_src:]
                if from_end
                else src_rgb[:, :zone_src]
            )
            out_slice = (
                output[:, out_w - zone_out:]
                if from_end
                else output[:, :zone_out]
            )
        band_up = _nn_up(_to_unit(src_slice), ratio).astype(dtype)
        weights = _feather_weights(
            band * ratio, zone_feather * ratio, dtype)
        if from_end:
            weights = weights[::-1]
        weights = (
            weights[:, None, None] if axis == 0 else weights[None, :, None]
        )
        mixed = (
            band_up * weights
            + out_slice.astype(dtype) * (1.0 - weights)
        ).astype(output.dtype)
        return mixed, zone_out

    feather_top = min(feather, src_h - top - 1) if top else 0
    feather_bottom = min(feather, src_h - bottom - 1) if bottom else 0
    feather_left = min(feather, src_w - left - 1) if left else 0
    feather_right = min(feather, src_w - right - 1) if right else 0

    # Rows first, then columns, so column zones blend over restored rows at the
    # corners. Overlapping zones fall back to sequential splices.
    if top or bottom:
        top_zone = (top + feather_top) * ratio
        bottom_zone = (bottom + feather_bottom) * ratio
        if top and bottom and top_zone + bottom_zone <= out_h:
            top_mix, _ = mixed_zone(out_rgb, 0, False, top, feather_top)
            bottom_mix, _ = mixed_zone(
                out_rgb, 0, True, bottom, feather_bottom)
            out_rgb = mx.concatenate(
                [top_mix, out_rgb[top_zone:out_h - bottom_zone], bottom_mix],
                axis=0,
            )
        else:
            if top:
                mixed, zone = mixed_zone(out_rgb, 0, False, top, feather_top)
                out_rgb = mx.concatenate([mixed, out_rgb[zone:]], axis=0)
            if bottom:
                mixed, zone = mixed_zone(
                    out_rgb, 0, True, bottom, feather_bottom)
                out_rgb = mx.concatenate(
                    [out_rgb[:out_h - zone], mixed], axis=0)
    if left or right:
        left_zone = (left + feather_left) * ratio
        right_zone = (right + feather_right) * ratio
        if left and right and left_zone + right_zone <= out_w:
            left_mix, _ = mixed_zone(out_rgb, 1, False, left, feather_left)
            right_mix, _ = mixed_zone(
                out_rgb, 1, True, right, feather_right)
            out_rgb = mx.concatenate(
                [left_mix, out_rgb[:, left_zone:out_w - right_zone], right_mix],
                axis=1,
            )
        else:
            if left:
                mixed, zone = mixed_zone(
                    out_rgb, 1, False, left, feather_left)
                out_rgb = mx.concatenate([mixed, out_rgb[:, zone:]], axis=1)
            if right:
                mixed, zone = mixed_zone(
                    out_rgb, 1, True, right, feather_right)
                out_rgb = mx.concatenate(
                    [out_rgb[:, :out_w - zone], mixed], axis=1)
    return out_rgb


def sanitize_rgb(rgb: Any, edges: tuple[int, int, int, int]) -> Any:
    """Replicate-fill edge bands while preserving shape and dtype."""
    top, bottom, left, right = edges
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if top + bottom >= height or left + right >= width:
        raise ValueError(
            f"edge bands {edges} do not leave an interior for {height}x{width}")
    if top or bottom:
        parts = []
        if top:
            parts.append(mx.broadcast_to(
                rgb[top:top + 1], (top, *rgb.shape[1:])))
        parts.append(rgb[top:height - bottom])
        if bottom:
            parts.append(mx.broadcast_to(
                rgb[height - bottom - 1:height - bottom],
                (bottom, *rgb.shape[1:]),
            ))
        rgb = mx.concatenate(parts, axis=0)
    if left or right:
        parts = []
        if left:
            parts.append(mx.broadcast_to(
                rgb[:, left:left + 1], (height, left, rgb.shape[-1])))
        parts.append(rgb[:, left:width - right])
        if right:
            parts.append(mx.broadcast_to(
                rgb[:, width - right - 1:width - right],
                (height, right, rgb.shape[-1]),
            ))
        rgb = mx.concatenate(parts, axis=1)
    return rgb


__all__ = ["restore_borders", "sanitize_rgb"]
