"""Synthetic-border detection and in-place cleanup for the VSR harness.

Every learned processor in the chain is trained on photographic content run
through a synthetic degradation pipeline; a synthetic border structure inside
the frame (letterbox/pillarbox bar, junk capture row, hard stabilization edge)
has zero probability under that distribution. GAN-trained stages react by
hallucinating texture around the border ("blooming") and the corruption
propagates tens of rows inward through the receptive field.

The cleanup decouples what the NETS see from what the VIEWER sees. Input side:
the detected junk rows/columns are overwritten with the adjacent interior
row/column (replicate fill) before any processor sees the frame, so the nets
never see the synthetic edge. Output side: bands can be RESTORED -- the
original border content is composited back over the processed frame -- because
a replicate fill that reaches the screen turns a quiet static-dark anomaly
into moving light content, which is far more noticeable (a 1 px fill is
imperceptible and removes the junk; wider fills should be restored). Frame
dimensions are untouched either way -- the output geometry, and therefore the
aspect ratio and any pixel-aspect handling, are identical by construction.

Detection is conservative and per clip: synthetic borders are temporally
persistent edge anomalies, so every sampled frame must agree. Two rules per
edge, scanned outermost-first:

- bar rows: near-constant and extreme (letterbox black / matte white) in all
  samples;
- junk step: the outermost row(s) sit far DARKER than the adjacent interior
  row in all samples (the classic junk capture line -- attenuated garbage).

Bars deeper than the fix-up cap are reported but not touched: replicating
content over a real letterbox would fabricate imagery; those want cropping,
not filling.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx

# Deepest edge band the auto-detector will fix (px per edge). Junk capture
# lines are 1-4 px; anything deeper is letterbox-class and only reported.
MAX_FIX_DEPTH = 8
# Junk-step rule: outermost row(s) must be at least this much darker than the
# adjacent interior row, in every sample.
_STEP_THR = 0.10
# Bar rule: near-constant (std below) and extreme (mean beyond) in every sample.
_BAR_STD = 0.025
_BAR_DARK = 0.10
_BAR_BRIGHT = 0.90
# Junk-step scan depth (bars use MAX_FIX_DEPTH).
_STEP_DEPTH = 3
# Samples whose whole-frame luma std is below this are blank/fade frames and
# are skipped by the detector.
_BLANK_STD = 0.02


def parse_edges_spec(spec: str) -> tuple[int, int, int, int]:
    """Parse an explicit 'T,B,L,R' pixel spec."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"edge spec must be T,B,L,R (four integers), got {spec!r}")
    vals = []
    for p in parts:
        v = int(p)
        if v < 0:
            raise ValueError(f"edge trim counts must be >= 0, got {spec!r}")
        vals.append(v)
    return tuple(vals)  # type: ignore[return-value]


def _luma(rgb: Any) -> Any:
    f = rgb.astype(mx.float32)
    return f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114


def _edge_stats(samples: Sequence[Any]) -> list:
    """Per usable (non-blank) sample: {edge: (line_means, line_stds)} with the
    lists ordered outermost-first for that edge.

    All statistics for a sample are computed as four axis reductions and
    ONE batched eval -- per-line `float(mx.mean(row))` calls are a GPU sync
    per line and made the detection pre-pass take seconds at 1080p."""
    pending = []
    for fr in samples:
        lu = _luma(fr)
        mu_r = mx.mean(lu, axis=1)
        sd_r = mx.sqrt(mx.mean((lu - mu_r[:, None]) ** 2, axis=1))
        mu_c = mx.mean(lu, axis=0)
        sd_c = mx.sqrt(mx.mean((lu - mu_c[None, :]) ** 2, axis=0))
        mu = mx.mean(lu)
        sd = mx.sqrt(mx.mean((lu - mu) ** 2))
        pending.append((mu_r, sd_r, mu_c, sd_c, sd))
    mx.eval(*[t for tup in pending for t in tup])
    out = []
    for mu_r, sd_r, mu_c, sd_c, sd in pending:
        if float(sd) < _BLANK_STD:
            continue
        rm, rs = mu_r.tolist(), sd_r.tolist()
        cm, cs = mu_c.tolist(), sd_c.tolist()
        out.append({
            "top": (rm, rs), "bottom": (rm[::-1], rs[::-1]),
            "left": (cm, cs), "right": (cm[::-1], cs[::-1]),
        })
    return out


def _edge_depth(stats: list, max_dim: int) -> tuple[int, int]:
    """(fix_depth, bar_depth) for one edge given per-sample (means, stds)
    lists where index 0 is the outermost line of that edge."""
    n_rows = min(MAX_FIX_DEPTH + 1, max_dim - 1)

    def is_bar(k: int) -> bool:
        if any(s[1][k] >= _BAR_STD for s in stats):
            return False
        return (all(s[0][k] < _BAR_DARK for s in stats)
                or all(s[0][k] > _BAR_BRIGHT for s in stats))

    bar_depth = 0
    while bar_depth < min(MAX_FIX_DEPTH, n_rows - 1) and is_bar(bar_depth):
        bar_depth += 1

    # Junk step: lines 0..d-1 all far darker than line d, in every sample.
    step_depth = 0
    for d in range(1, min(_STEP_DEPTH, n_rows - 1) + 1):
        ok = True
        for means, _stds in stats:
            ref = means[d]
            if any(means[k] > ref - _STEP_THR for k in range(d)):
                ok = False
                break
        if ok:
            step_depth = d
    return max(bar_depth, step_depth), bar_depth


def _deep_bar_depth(stats: list, max_dim: int, limit: int | None = None) -> int:
    """Bar-rule depth scanned past the fix cap (letterbox/pillarbox bars)."""
    if limit is None:
        limit = max_dim // 4
    depth = 0
    while depth < limit:
        if any(s[1][depth] >= _BAR_STD for s in stats):
            break
        if not (all(s[0][depth] < _BAR_DARK for s in stats)
                or all(s[0][depth] > _BAR_BRIGHT for s in stats)):
            break
        depth += 1
    return depth


def detect_bars(samples: Sequence[Any]) -> tuple[int, int, int, int]:
    """Detect constant letterbox/pillarbox bars for cropping.

    Scans each edge with the bar rule up to 45% of the dimension (a 9:16
    portrait shoved into 16:9 leaves ~35% bars per side). Returns per-edge bar
    depths in px, adjusted so the remaining active area has EVEN dimensions
    (4:2:0 chroma and encoders want even); the adjustment eats one content
    pixel rather than leaving one bar line. Requires 3 usable (non-blank)
    samples; bars must be constant-extreme in every one."""
    stats = _edge_stats(samples)
    if len(stats) < 3:
        return (0, 0, 0, 0)
    h = len(stats[0]["top"][0])
    w = len(stats[0]["left"][0])
    dims = {"top": h, "bottom": h, "left": w, "right": w}
    bars = {e: _deep_bar_depth([s[e] for s in stats], dims[e],
                               limit=(dims[e] * 45) // 100)
            for e in dims}
    t, b = bars["top"], bars["bottom"]
    left, r = bars["left"], bars["right"]
    if (h - t - b) % 2:
        b += 1
    if (w - left - r) % 2:
        r += 1
    if t + b >= h or left + r >= w:
        return (0, 0, 0, 0)
    return t, b, left, r


# The nine window anchors: fraction of the slack placed left of / above the
# window (0 = flush to that edge, 0.5 = centered, 1 = flush to the other).
_ANCHORS = {
    "top-left": (0.0, 0.0), "top": (0.5, 0.0), "top-right": (1.0, 0.0),
    "left": (0.0, 0.5), "center": (0.5, 0.5), "right": (1.0, 0.5),
    "bottom-left": (0.0, 1.0), "bottom": (0.5, 1.0), "bottom-right": (1.0, 1.0),
}


def compute_aspect_crop(w: int, h: int, ar_w: int, ar_h: int,
                        dx: int = 0, dy: int = 0,
                        anchor: str = "center") -> tuple[int, int, int, int]:
    """(top, bottom, left, right) crop that windows WxH to the largest
    even-dimension rectangle with aspect ar_w:ar_h. The window is placed at
    `anchor` (one of the nine: top-left/top/top-right/left/center/right/
    bottom-left/bottom/bottom-right), then shifted by (dx, dy) px (right/down
    positive) and clamped inside the frame. Exact ratios are approximated to
    the nearest even dimensions."""
    if ar_w <= 0 or ar_h <= 0:
        raise ValueError(f"aspect must be positive, got {ar_w}:{ar_h}")
    if anchor not in _ANCHORS:
        raise ValueError(f"anchor must be one of {sorted(_ANCHORS)}, got {anchor!r}")
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
    valid = [c for c in {chain, width_first, height_first} if c[0] >= 2 and c[1] >= 2]
    if not valid:
        raise ValueError(f"aspect {ar_w}:{ar_h} leaves no picture in {w}x{h}")
    tw, th = min(valid, key=lambda c: (abs(c[0] / c[1] / target - 1.0), -(c[0] * c[1])))
    ax, ay = _ANCHORS[anchor]
    left = max(0, min(int(round((w - tw) * ax)) + int(dx), w - tw))
    top = max(0, min(int(round((h - th) * ay)) + int(dy), h - th))
    return (top, h - th - top, left, w - tw - left)


def crop_rgb(rgb: Any, bars: tuple[int, int, int, int]) -> Any:
    """Crop the bar bands off a frame (works for (H,W,C) and (N,H,W,C))."""
    t, b, left, r = bars
    if rgb.ndim == 4:
        h, w = int(rgb.shape[1]), int(rgb.shape[2])
        return rgb[:, t:h - b, left:w - r]
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    return rgb[t:h - b, left:w - r]


def detect_junk_edges(samples: Sequence[Any]) -> tuple[tuple[int, int, int, int], list]:
    """Detect junk edge bands across sampled frames (each (H,W,3), [0,1]).

    Returns ((top, bottom, left, right) fix depths in px, notices) where
    notices lists letterbox-class findings that are reported but not fixed.
    Blank/fade samples are skipped; with fewer than 3 usable samples nothing
    is detected (too little evidence to overwrite pixels)."""
    stats = _edge_stats(samples)
    if len(stats) < 3:
        return (0, 0, 0, 0), ["too few usable sample frames; no detection"]

    h = len(stats[0]["top"][0])
    w = len(stats[0]["left"][0])
    dims = {"top": h, "bottom": h, "left": w, "right": w}
    fix = {}
    notices = []
    for edge, dim in dims.items():
        es = [s[edge] for s in stats]
        d, bar = _edge_depth(es, dim)
        if bar >= MAX_FIX_DEPTH:
            deep = _deep_bar_depth(es, dim)
            notices.append(
                f"{edge}: {deep}px constant bar (letterbox-class) -- not filled; "
                "crop the active area instead"
            )
            d = 0
        fix[edge] = d
    return (fix["top"], fix["bottom"], fix["left"], fix["right"]), notices


def _nn_up(x: Any, r: int) -> Any:
    h, w = x.shape[0], x.shape[1]
    y = mx.broadcast_to(x[:, None, :, None, :], (h, r, w, r, x.shape[-1]))
    return y.reshape(h * r, w * r, x.shape[-1])


def _to_unit(rgb: Any) -> Any:
    f = rgb[..., :3].astype(mx.float32)
    return f / 255.0 if rgb.dtype == mx.uint8 else f


# Feather weight vectors are identical every frame for a given (band, feather)
# in output px -- build once, cast per use (they are tiny).
_FEATHER_CACHE: dict = {}


def _feather_weights(band_px: int, feather_px: int, dt: Any) -> Any:
    """Composite weights for one edge zone: 1.0 across the band, then a linear
    ramp to 0 across the feather. Length band_px + feather_px."""
    key = (band_px, feather_px)
    base = _FEATHER_CACHE.get(key)
    if base is None:
        w = [1.0] * band_px
        for i in range(feather_px):
            w.append(1.0 - (i + 0.5) / feather_px)
        base = mx.array(w, dtype=mx.float32)
        _FEATHER_CACHE[key] = base
    return base.astype(dt)


def restore_borders(out_rgb: Any, src_rgb: Any, edges: tuple[int, int, int, int],
                    feather: int = 2) -> Any:
    """Composite the ORIGINAL edge bands back over a processed frame.

    out_rgb is the processed frame at an integer multiple of src_rgb's size;
    the source bands are nearest-upscaled to match and blended in. The viewer
    then sees the border exactly as authentic as an unprocessed pipeline would
    show it (static, quiet), while the nets never saw it. `feather` (source px)
    crossfades from the restored band into the processed content so the seam
    between the soft original border and the crisply processed interior does
    not read as a hard line; 0 = hard splice.

    Splices are combined per axis (one concatenate for top+bottom, one for
    left+right) so a frame costs at most two full-size copies; only the edge
    zones of the source are converted/upscaled."""
    t, b, left, r = edges
    feather = max(0, int(feather))
    sh, sw = int(src_rgb.shape[0]), int(src_rgb.shape[1])
    oh, ow = int(out_rgb.shape[0]), int(out_rgb.shape[1])
    if oh % sh or ow % sw or (oh // sh) != (ow // sw):
        raise ValueError(f"output {oh}x{ow} is not an integer multiple of source {sh}x{sw}")
    ratio = oh // sh
    dt = out_rgb.dtype if out_rgb.dtype != mx.uint8 else mx.float32

    def mixed_zone(o, axis, from_end, band, fe):
        """The blended zone tensor for one edge (band+feather, output px)."""
        zone_src = band + fe
        zone_out = zone_src * ratio
        if axis == 0:
            s_slice = src_rgb[sh - zone_src:] if from_end else src_rgb[:zone_src]
            o_slice = o[oh - zone_out:] if from_end else o[:zone_out]
        else:
            s_slice = src_rgb[:, sw - zone_src:] if from_end else src_rgb[:, :zone_src]
            o_slice = o[:, ow - zone_out:] if from_end else o[:, :zone_out]
        band_up = _nn_up(_to_unit(s_slice), ratio).astype(dt)
        w = _feather_weights(band * ratio, fe * ratio, dt)
        if from_end:
            w = w[::-1]
        w = w[:, None, None] if axis == 0 else w[None, :, None]
        return (band_up * w + o_slice.astype(dt) * (1.0 - w)).astype(o.dtype), zone_out

    fe_t = min(feather, sh - t - 1) if t else 0
    fe_b = min(feather, sh - b - 1) if b else 0
    fe_l = min(feather, sw - left - 1) if left else 0
    fe_r = min(feather, sw - r - 1) if r else 0

    # Rows first, then columns (column zones blend over the already-restored
    # rows at the corners, same as the previous sequential order). Zones that
    # would overlap fall back to sequential splices.
    if t or b:
        zt = (t + fe_t) * ratio
        zb = (b + fe_b) * ratio
        if t and b and zt + zb <= oh:
            top_mix, _ = mixed_zone(out_rgb, 0, False, t, fe_t)
            bot_mix, _ = mixed_zone(out_rgb, 0, True, b, fe_b)
            out_rgb = mx.concatenate([top_mix, out_rgb[zt:oh - zb], bot_mix], axis=0)
        else:
            if t:
                mix, zo = mixed_zone(out_rgb, 0, False, t, fe_t)
                out_rgb = mx.concatenate([mix, out_rgb[zo:]], axis=0)
            if b:
                mix, zo = mixed_zone(out_rgb, 0, True, b, fe_b)
                out_rgb = mx.concatenate([out_rgb[:oh - zo], mix], axis=0)
    if left or r:
        zl = (left + fe_l) * ratio
        zr = (r + fe_r) * ratio
        if left and r and zl + zr <= ow:
            l_mix, _ = mixed_zone(out_rgb, 1, False, left, fe_l)
            r_mix, _ = mixed_zone(out_rgb, 1, True, r, fe_r)
            out_rgb = mx.concatenate([l_mix, out_rgb[:, zl:ow - zr], r_mix], axis=1)
        else:
            if left:
                mix, zo = mixed_zone(out_rgb, 1, False, left, fe_l)
                out_rgb = mx.concatenate([mix, out_rgb[:, zo:]], axis=1)
            if r:
                mix, zo = mixed_zone(out_rgb, 1, True, r, fe_r)
                out_rgb = mx.concatenate([out_rgb[:, :ow - zo], mix], axis=1)
    return out_rgb


def sanitize_rgb(rgb: Any, edges: tuple[int, int, int, int]) -> Any:
    """Overwrite the given edge bands (top, bottom, left, right px) with the
    adjacent interior row/column. (H,W,3) in, same shape out; dtype preserved.
    One concatenate per touched axis, so at most two full-frame copies."""
    t, b, left, r = edges
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if t + b >= h or left + r >= w:
        raise ValueError(f"edge bands {edges} do not leave an interior for {h}x{w}")
    if t or b:
        parts = []
        if t:
            parts.append(mx.broadcast_to(rgb[t:t + 1], (t, *rgb.shape[1:])))
        parts.append(rgb[t:h - b])
        if b:
            parts.append(mx.broadcast_to(rgb[h - b - 1:h - b], (b, *rgb.shape[1:])))
        rgb = mx.concatenate(parts, axis=0)
    if left or r:
        parts = []
        if left:
            parts.append(mx.broadcast_to(rgb[:, left:left + 1], (h, left, rgb.shape[-1])))
        parts.append(rgb[:, left:w - r])
        if r:
            parts.append(mx.broadcast_to(rgb[:, w - r - 1:w - r], (h, r, rgb.shape[-1])))
        rgb = mx.concatenate(parts, axis=1)
    return rgb
