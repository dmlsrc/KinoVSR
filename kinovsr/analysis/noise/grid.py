"""Grid-period and blockiness-map detection.

Split of the noise_map module: coding-grid phase/period spectra and the
tile-excess blockiness map used by deblock conditioning.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .estimate import (
    _box_blur_full,
    _box_smooth_coarse,
    _edge_pad_hw,
    _to_luma_2d,
)


def _grid_phase(g: Any, period: int = 8) -> int:
    """Most-elevated gradient phase along an axis (the coding grid's offset)."""
    n = int(g.shape[1]) // period * period
    if n < period:
        return 0
    m = mx.mean(g[:, :n].reshape(g.shape[0], n // period, period), axis=(0, 1))  # (period,)
    vals = [float(v) for v in m.tolist()]
    return max(range(period), key=lambda i: vals[i])


def _grid_spectrum(g: Any, lo: float, hi: float):
    """Matched-projection spectrum of an axis's mean-gradient profile.

    Returns (periods, strengths, phases) or None when the profile is too
    short. Strength at P is |mean(hp(x) e^(2*pi*i*x/P))| of the high-passed
    profile; the phase falls out of the complex argument.
    """
    import math
    prof = mx.mean(g, axis=0)
    n = int(prof.shape[0])
    if n < int(hi) * 3:
        return None
    k = 17
    r = k // 2
    pad = mx.concatenate([mx.broadcast_to(prof[:1], (r,)), prof,
                          mx.broadcast_to(prof[-1:], (r,))], axis=0)
    smooth = mx.zeros_like(prof)
    for off in range(k):
        smooth = smooth + pad[off:off + n]
    hp = prof - smooth / k
    xs = mx.arange(n).astype(mx.float32)
    periods = [lo + 0.1 * i for i in range(int((hi - lo) / 0.1) + 1)]
    pv = mx.array(periods, dtype=mx.float32)[:, None]
    ang = (2.0 * 3.141592653589793) * xs[None, :] / pv
    zr = mx.mean(hp[None, :] * mx.cos(ang), axis=1)
    zi = mx.mean(hp[None, :] * mx.sin(ang), axis=1)
    s = [float(v) for v in mx.sqrt(zr * zr + zi * zi).tolist()]
    phases = [(math.atan2(float(bb), float(aa)) / (2.0 * 3.141592653589793)) * P % P
              for aa, bb, P in zip(zr.tolist(), zi.tolist(), periods, strict=True)]
    return periods, s, phases


def _grid_period_joint(gx: Any, gy_t: Any, lo: float = 6.0,
                       hi: float = 16.0) -> tuple[float, float, float, float, float] | None:
    """Detect a shared FRACTIONAL grid period across BOTH axes.

    Compressed-then-resized video carries its coding grid at 8 x scale -- a
    non-integer period an 8-locked detector cannot see. A genuine coding
    grid has the SAME period on both axes (resizes are isotropic), while
    content periodicity (stripes, blinds, railings) is almost always
    one-axis: requiring both axes to support the winning period is what
    keeps real-world texture from hijacking the detection. The band caps at
    16 (a 2x resize): beyond that a 32px tile holds a single grid line and
    the min-over-lines edge rejection collapses.

    Returns (period, phase_x, phase_y, strength_x, strength_y) or None
    (caller falls back to the legacy period-8 path; anamorphic one-axis
    resizes are deliberately not chased).
    """
    rx = _grid_spectrum(gx, lo, hi)
    ry = _grid_spectrum(gy_t, lo, hi)
    if rx is None or ry is None:
        return None
    periods, sx, phx = rx
    _, sy, phy = ry
    medx = sorted(sx)[len(sx) // 2]
    medy = sorted(sy)[len(sy) // 2]
    if medx <= 0.0 or medy <= 0.0:
        return None
    joint = [aa / medx + bb / medy for aa, bb in zip(sx, sy, strict=True)]
    jmax = max(joint)
    strong = sorted((i for i, v in enumerate(joint) if v >= 0.8 * jmax),
                    key=lambda j: periods[j])
    # cluster near-equal periods (one physical peak spans several 0.1-step
    # samples) and start from the strongest cluster's joint-argmax -- the
    # phase must come from the true lattice, not a neighboring sample whose
    # phase drifts off-grid across the frame. A positive line comb projects
    # fully onto every DIVISOR of its period, so if a strong cluster sits at
    # ~2x the winner, promote to it (the winner was the half-period
    # harmonic); unrelated competing periods resolve by strength, not size.
    clusters: list = []
    for i in strong:
        if clusters and periods[i] <= periods[clusters[-1][-1]] * 1.15:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    reps = [max(c, key=lambda j: joint[j]) for c in clusters]
    i = max(reps, key=lambda j: joint[j])
    for r in reps:
        ratio = periods[r] / periods[i]
        if 1.85 <= ratio <= 2.15:
            i = r
            break
    if joint[i] < 5.5 or sx[i] < 1.8 * medx or sy[i] < 1.8 * medy:
        return None
    return periods[i], phx[i], phy[i], sx[i], sy[i]


def detect_grid_period(frames: list, max_frames: int = 6) -> dict:
    """Grid period diagnostic for a window of frames (joint across axes).

    Returns {"px": (period, strength_x) | None, "py": (period, strength_y)
    | None} with a SHARED period; a period near 8 x s with s != 1 is the
    signature of compressed-then-resized footage. Used by the probe router;
    the blockiness map runs the same detection internally (period="auto").
    """
    if not frames:
        return {"px": None, "py": None}
    n = len(frames)
    take = (list(range(n)) if n <= max_frames else
            sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)}))
    lum = [_to_luma_2d(frames[i]) for i in take]
    y = mx.stack(lum, axis=0)
    gv = mx.mean(mx.abs(y[:, :, 1:] - y[:, :, :-1]), axis=0)
    gh = mx.mean(mx.abs(y[:, 1:, :] - y[:, :-1, :]), axis=0)
    r = _grid_period_joint(gv, mx.transpose(gh, (1, 0)))
    if r is None:
        return {"px": None, "py": None}
    P, _phx, _phy, sx, sy = r
    return {"px": (P, sx), "py": (P, sy)}


def _profile_spectrum(hp: Any, lo: float = 2.0, hi: float = 32.0):
    """Matched-projection spectrum of one or more high-passed 1-D profiles.

    hp: (K, N) high-passed profiles. Per candidate period the per-profile
    projection MAGNITUDES are averaged (phases jump between frames for
    jumping scanlines, so a coherent average would cancel exactly the
    artifact this exists to find). Returns (periods, strengths).
    """
    N = int(hp.shape[1])
    xs = mx.arange(N).astype(mx.float32)
    periods = [lo + 0.1 * i for i in range(int((hi - lo) / 0.1) + 1)]
    pv = mx.array(periods, dtype=mx.float32)[:, None]
    ang = (2.0 * 3.141592653589793) * xs[None, :] / pv           # (P, N)
    zr = mx.matmul(mx.cos(ang), mx.transpose(hp)) / N            # (P, K)
    zi = mx.matmul(mx.sin(ang), mx.transpose(hp)) / N
    s = mx.mean(mx.sqrt(zr * zr + zi * zi), axis=1)
    return periods, [float(v) for v in s.tolist()]


def _spectrum_peak(periods: list, s: list) -> tuple[float, float]:
    """(period, prominence-in-medians) of the dominant clustered peak; the
    same cluster + promote-at-2x fundamental rule as the grid detector."""
    med = sorted(s)[len(s) // 2]
    if med <= 0.0:
        return 0.0, 0.0
    smax = max(s)
    strong = sorted((i for i, v in enumerate(s) if v >= 0.8 * smax),
                    key=lambda j: periods[j])
    clusters: list = []
    for i in strong:
        if clusters and periods[i] <= periods[clusters[-1][-1]] * 1.15:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    reps = [max(c, key=lambda j: s[j]) for c in clusters]
    i = max(reps, key=lambda j: s[j])
    for r in reps:
        if 1.85 <= periods[r] / periods[i] <= 2.15:
            i = r
            break
    return periods[i], s[i] / med


def _highpass_rows(prof: Any, k: int = 17) -> Any:
    """(K, N) profiles minus their own moving mean of width k."""
    K, N = int(prof.shape[0]), int(prof.shape[1])
    r = k // 2
    pad = mx.concatenate([mx.broadcast_to(prof[:, :1], (K, r)), prof,
                          mx.broadcast_to(prof[:, -1:], (K, r))], axis=1)
    smooth = mx.zeros_like(prof)
    for off in range(k):
        smooth = smooth + pad[:, off:off + N]
    return prof - smooth / k


def _tile_excess_frac(g: Any, P: float, phase: float, tile: int,
                      transpose: bool) -> Any:
    """Fractional-period version of the tile grid-evidence statistic.

    Grid lines sit at global positions phase + k*P (non-integer after a
    resize), so within-tile offsets differ per tile and the integer
    reshape trick no longer applies. Lines are sampled from the full-width
    gradient by linear interpolation, averaged per 32-row band, grouped by
    the tile column they fall in, and reduced by the same min-minus-null
    rule (min over a tile's lines rejects single content edges; the
    quarter-period-shifted null rejects texture). The excess is scaled by
    P/8: resampling spreads each coding-grid step across neighboring
    columns, attenuating the per-column gradient by roughly the resize
    factor, so severity stays comparable to the unresized calibration.
    """
    if transpose:
        g = mx.transpose(g, (1, 0))
    h, w = int(g.shape[0]), int(g.shape[1])
    ph, pw = (-h) % tile, (-w) % tile
    g = _edge_pad_hw(g, ph, pw)
    hp, wp = h + ph, w + pw
    ht, wt = hp // tile, wp // tile
    band = mx.mean(g.reshape(ht, tile, wp), axis=1)             # (ht, wp)

    # the min-over-lines edge rejection needs ~4 lines to be meaningful (the
    # legacy 8-grid gives 4 per 32px tile); at larger fractional periods,
    # group neighboring tile columns until a group spans >= 4 lines and share
    # the group's min -- coarser localization, same robustness invariant
    import math as _math
    ge = max(1, _math.ceil((4.0 * P) / tile))

    def _group_min(off: float) -> list:
        ng = (wt + ge - 1) // ge
        per_group: dict = {}
        nk = max(0, int((wp - 1 - off) / P) + 1)
        for k in range(nk):
            x = off + k * P
            i0 = int(x)
            if i0 >= wp - 1:
                break
            fr = x - i0
            v = band[:, i0] * (1.0 - fr) + band[:, i0 + 1] * fr   # (ht,)
            g_idx = min(ng - 1, int(x // (tile * ge)))
            per_group.setdefault(g_idx, []).append(v)
        big = mx.full((ht,), 1e9, dtype=mx.float32)
        out = []
        for gi in range(ng):
            lv = per_group.get(gi)
            if not lv or len(lv) < 3:
                out.append(big)
                continue
            m = lv[0]
            for v in lv[1:]:
                m = mx.minimum(m, v)
            out.append(m)
        return out

    def _tile_min(off: float) -> list:
        groups = _group_min(off)
        return [groups[c // ge] for c in range(wt)]

    matched = _tile_min(phase % P)
    # three null phases, subtract their MAX: sparse fractional sampling can
    # alias against periodic content so that ONE null sits luckily low, but
    # a content wave cannot beat all three; true grid lines (1-2 px wide)
    # clear every null at these offsets
    nulls = [_tile_min((phase + f * P) % P) for f in (0.25, 0.5, 0.75)]
    cols = []
    for c in range(wt):
        nc = mx.maximum(mx.maximum(nulls[0][c], nulls[1][c]), nulls[2][c])
        ex = mx.maximum(matched[c] - nc, 0.0)
        bad = mx.maximum(matched[c], nc)
        ex = mx.where(bad >= 1e8, mx.zeros_like(ex), ex)         # no-evidence tiles
        cols.append(ex)
    ex = mx.stack(cols, axis=1) * (P / 8.0)                      # (ht, wt)
    return mx.transpose(ex, (1, 0)) if transpose else ex


def estimate_blockiness_map(
    frames: list,
    period: Any = "auto",
    tile: int = 32,
    max_frames: int = 6,
    severity_scale: float = 0.006,
    global_floor: float = 0.35,
    smooth: bool = True,
) -> Any | None:
    """Estimate a per-pixel blockiness mask in [0, 1] from a window of frames.

    Blocking is a purely spatial artifact, so unlike the noise-sigma estimator
    this needs no adjacent frames -- it samples up to max_frames spread across
    the window. period="auto" (default) first looks for a FRACTIONAL grid
    period per axis (compressed-then-resized footage carries its grid at
    8 x scale, invisible to an 8-locked detector) and uses interpolated
    fractional grid lines when one is found; otherwise -- including any
    detection within +-0.3 of 8 -- it takes the exact legacy integer path.
    Per axis, the coding-grid phase is detected globally from
    gradient energy (period-8 grids; period-16 boundaries are a subset), then
    each tile scores the mean gradient ON grid boundaries minus the mean OFF
    them -- content texture raises both and cancels, so only grid-aligned
    discontinuities register. The two axes combine by geometric mean, so a real
    vertical or horizontal content edge (one axis only) scores zero; DCT
    blocking (always a 2D grid) survives. `severity_scale` is the boundary
    excess (luma units) mapped to mask 1.0; the default is calibrated so
    loop-filtered modern H.264 blocking lands mid-scale and old-encoder /
    hard DCT grids saturate.
    `global_floor` is the maximum wetness kept for frame-wide saturated grids;
    only local excess above that frame baseline can remain fully wet.

    Returns (H,W,1) fp32 in [0,1], or None for an empty frame list.
    """
    if not frames:
        return None
    n = len(frames)
    take = list(range(n)) if n <= max_frames else \
        sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)})
    lum = [_to_luma_2d(frames[i]) for i in take]
    H, W = int(lum[0].shape[0]), int(lum[0].shape[1])
    y = mx.stack(lum, axis=0)                                   # (K,H,W)

    gv = mx.mean(mx.abs(y[:, :, 1:] - y[:, :, :-1]), axis=0)    # (H, W-1) col grads
    gh = mx.mean(mx.abs(y[:, 1:, :] - y[:, :-1, :]), axis=0)    # (H-1, W) row grads
    # period="auto": detect a fractional grid (compressed-then-resized
    # footage carries its grid at 8 x scale). Detection within +-0.3 of 8,
    # or no convincing periodicity, falls back to the exact legacy path.
    frac_x = frac_y = None
    if period == "auto":
        r = _grid_period_joint(gv, mx.transpose(gh, (1, 0)))
        if r is not None and abs(r[0] - 8.0) > 0.3:
            frac_x = (r[0], r[1])
            frac_y = (r[0], r[2])
        period = 8
    px = _grid_phase(gv, period)
    py = _grid_phase(mx.transpose(gh, (1, 0)), period)

    def _line_min(gt: Any, phase: int) -> Any:
        """Per-tile min over the grid lines of `phase`: blocking elevates EVERY
        line crossing a tile; a content edge elevates one, so the min rejects it."""
        lines = [gt[:, :, :, phase + k * period]
                 for k in range(max(1, (tile - phase) // period))
                 if phase + k * period < tile]
        line_means = mx.stack([mx.mean(ln, axis=1) for ln in lines], axis=0)  # (L,ht,wt)
        return mx.min(line_means, axis=0)                       # (ht,wt)

    def _tile_excess(g: Any, phase: int, transpose: bool) -> Any:
        """Per-tile grid evidence: line-min at the detected phase MINUS the same
        statistic at a null phase (offset by period/2), >= 0.

        The null phase is the matched control: content texture -- including
        periodic texture whose period divides the grid (a 2px checker elevates
        every other column, hence both phases equally) -- raises both and
        cancels; only a discontinuity locked to the detected phase survives.
        This replaces a global phase-unimodality gate, which real loop-filtered
        H.264 fails (its residual grid is weak) while still being visibly
        blocked.
        """
        if transpose:
            g = mx.transpose(g, (1, 0))
        h, w = int(g.shape[0]), int(g.shape[1])
        ph, pw = (-h) % tile, (-w) % tile
        g = _edge_pad_hw(g, ph, pw)
        hp, wp = h + ph, w + pw
        gt = g.reshape(hp // tile, tile, wp // tile, tile)      # (ht,tr,wt,tc)
        null = (phase + period // 2) % period
        ex = mx.maximum(_line_min(gt, phase) - _line_min(gt, null), 0.0)
        return mx.transpose(ex, (1, 0)) if transpose else ex

    bv = _tile_excess(gv, px, transpose=False)                  # (ht, wt-ish)
    bh = _tile_excess(gh, py, transpose=True)
    if frac_x is not None and frac_y is not None:
        # evaluate the detected fractional grid as a SECOND hypothesis and
        # keep the better-supported evidence per tile: twice-encoded resized
        # grids can be spectrally weaker than broad content periodicity, and
        # a mis-locked period would otherwise read ~0 and suppress the
        # legacy evidence -- max() turns a wrong detection into a no-op
        bv = mx.maximum(bv, _tile_excess_frac(gv, frac_x[0], frac_x[1], tile,
                                              transpose=False)[:bv.shape[0], :bv.shape[1]])
        bh = mx.maximum(bh, _tile_excess_frac(gh, frac_y[0], frac_y[1], tile,
                                              transpose=True)[:bh.shape[0], :bh.shape[1]])
    # tile grids can differ by one when (W-1) vs W pad differently; crop to match
    th = min(int(bv.shape[0]), int(bh.shape[0]))
    tw = min(int(bv.shape[1]), int(bh.shape[1]))
    b = mx.sqrt(bv[:th, :tw] * bh[:th, :tw])                    # 2D-grid evidence only
    if smooth:
        # single pass: a wet/dry mask wants peak fidelity (double smoothing
        # diluted mild real-world blocking ~1.5-1.8x); the full-res blur below
        # still removes tile edges.
        b = _box_smooth_coarse(b, passes=1)
    full = mx.repeat(mx.repeat(b, tile, axis=0), tile, axis=1)[:H, :W]
    if smooth:
        full = _box_blur_full(full, tile + 1)
    mask = mx.clip(full / severity_scale, 0.0, 1.0)
    # If much of the frame already carries grid evidence, an absolute mask can
    # quietly become "full-strength deblock everywhere" (the auto path's default
    # strength is 1.0). Keep a moderate global correction, but reserve full wet
    # values for regions that stand above the frame's own blockiness baseline.
    flat = mx.sort(mask.reshape(-1))
    nf = int(flat.shape[0])
    p25 = float(flat[int(0.25 * (nf - 1))])
    p50 = float(flat[nf // 2])
    p95 = float(flat[int(0.95 * (nf - 1))])
    area = float(mx.mean((mask > 0.5).astype(mx.float32)))
    if p95 > 0.90 and area > 0.30 and p25 > 0.20:
        local_excess = mx.clip((mask - p50) / max(1.0 - p50, 1e-6), 0.0, 1.0)
        tempered = float(global_floor) + (1.0 - float(global_floor)) * local_excess
        mask = mx.minimum(mask, tempered)
    return mask[:, :, None].astype(mx.float32)


