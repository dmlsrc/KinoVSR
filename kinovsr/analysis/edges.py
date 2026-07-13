"""Shared frame-statistics analysis for persistent edge anomalies.

Every learned processor in the chain is trained on photographic content run
through a synthetic degradation pipeline; a synthetic border structure inside
the frame (letterbox/pillarbox bar, junk capture row, hard stabilization edge)
has zero probability under that distribution. GAN-trained stages react by
hallucinating texture around the border ("blooming") and the corruption
propagates tens of rows inward through the receptive field.

Detection is conservative and per clip: synthetic borders are temporally
persistent edge anomalies, so every sampled frame must agree. Two rules per
edge, scanned outermost-first:

- bar rows: near-constant and extreme (letterbox black / matte white) in all
  samples;
- junk step: the outermost row(s) sit far DARKER than the adjacent interior
  row in all samples (the classic junk capture line -- attenuated garbage).

Bars deeper than the fix-up cap are reported but not touched: replicating
content over a real letterbox would fabricate imagery; those want cropping,
not filling. This module detects only; the crop and sanitize processor
families own their respective pixel operations.
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


__all__ = ["detect_bars", "detect_junk_edges"]
