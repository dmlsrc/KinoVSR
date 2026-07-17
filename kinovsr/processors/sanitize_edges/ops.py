"""MLX pixel operations owned by the sanitize-edges family."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import mlx.core as mx

# Nearest-resample index vectors are identical per (geometry, zone) pair.
_INDEX_CACHE: dict[tuple, Any] = {}


def _resample_index(out_size: int, src_size: int, out_lo: int, out_n: int,
                    src_lo: int, src_n: int) -> Any:
    """Global nearest map of output positions [out_lo, out_lo+out_n) into
    the source slice [src_lo, src_lo+src_n), clamped to the slice. Using
    the GLOBAL mapping keeps band pixels aligned with where a uniform
    resample of the whole frame placed the surrounding content."""
    key = (out_size, src_size, out_lo, out_n, src_lo, src_n)
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        idx = mx.array(
            [min(max((p * src_size) // out_size - src_lo, 0), src_n - 1)
             for p in range(out_lo, out_lo + out_n)],
            dtype=mx.int32)
        _INDEX_CACHE[key] = idx
    return idx


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
    """Composite original edge bands over the processed output frame.

    The source bands are nearest-resampled onto the output geometry (any
    per-axis scale, integer or not - a square-pixels resample between the
    restore capture and this composite is fine) and blended into the
    processed output. Splices are combined per axis, so a frame costs at
    most two full-size copies; only source edge zones are converted and
    resampled.
    """
    top, bottom, left, right = edges
    feather = max(0, int(feather))
    src_h, src_w = int(src_rgb.shape[0]), int(src_rgb.shape[1])
    out_h, out_w = int(out_rgb.shape[0]), int(out_rgb.shape[1])
    dtype = out_rgb.dtype if out_rgb.dtype != mx.uint8 else mx.float32

    def zone_len(zone_src: int, axis: int) -> int:
        edge_out, edge_src = (out_h, src_h) if axis == 0 else (out_w, src_w)
        return max(1, int(round(Fraction(zone_src * edge_out, edge_src))))

    def mixed_zone(
        output: Any,
        axis: int,
        from_end: bool,
        band: int,
        zone_feather: int,
    ) -> tuple[Any, int]:
        zone_src = band + zone_feather
        edge_out, edge_src = (out_h, src_h) if axis == 0 else (out_w, src_w)
        oth_out, oth_src = (out_w, src_w) if axis == 0 else (out_h, src_h)
        zone_out = zone_len(zone_src, axis)
        band_out = min(
            int(round(Fraction(band * edge_out, edge_src))), zone_out)
        src_lo = edge_src - zone_src if from_end else 0
        out_lo = edge_out - zone_out if from_end else 0
        if axis == 0:
            src_slice = src_rgb[src_lo:src_lo + zone_src]
            out_slice = output[out_lo:out_lo + zone_out]
        else:
            src_slice = src_rgb[:, src_lo:src_lo + zone_src]
            out_slice = output[:, out_lo:out_lo + zone_out]
        edge_idx = _resample_index(
            edge_out, edge_src, out_lo, zone_out, src_lo, zone_src)
        oth_idx = _resample_index(oth_out, oth_src, 0, oth_out, 0, oth_src)
        band_unit = _to_unit(src_slice)
        if axis == 0:
            band_up = mx.take(
                mx.take(band_unit, edge_idx, axis=0), oth_idx, axis=1
            ).astype(dtype)
        else:
            band_up = mx.take(
                mx.take(band_unit, oth_idx, axis=0), edge_idx, axis=1
            ).astype(dtype)
        weights = _feather_weights(band_out, zone_out - band_out, dtype)
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
        top_zone = zone_len(top + feather_top, 0) if top else 0
        bottom_zone = zone_len(bottom + feather_bottom, 0) if bottom else 0
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
        left_zone = zone_len(left + feather_left, 1) if left else 0
        right_zone = zone_len(right + feather_right, 1) if right else 0
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
