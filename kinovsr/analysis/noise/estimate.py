"""Sigma-map estimation and flat-pixel floors.

Split of the noise_map module (planning 04); the method rationale lives
in the package docstring. This module owns the per-block tail-RMS
sigma-map estimator and its luma/blur/motion-cap helpers.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

# Pooling statistic: tail RMS -- the RMS of the top 5% of |diff| samples per
# block, normalized so dense AWGN reads exactly sigma. The conditioning
# question is not "how much temporal energy is there" but "how large a sigma
# must the denoiser be told so it SUPPRESSES what flickers": a pixel flashing
# at amplitude A in a mostly-static block contributes only A*sqrt(density) to
# the block energy, yet a net conditioned below A treats the flash as signal
# and preserves it. The tail statistic reads the flicker's amplitude scale
# (sparse flash -> the tail IS the flicker); for dense AWGN the top-5% tail of
# |N(0,sigma)| has E[d^2 | tail] = 5.577 sigma^2, so dividing by sqrt(5.577)
# keeps it exact there. Quantiles fail both ways (a 10%-density flash reads 0
# at q75). Motion robustness is the luma-cap's job, not the statistic's.
_TAIL_FRACTION = 0.05
_TAIL_RMS_NORM = 2.3616  # sqrt(E[Z^2 | |Z| > z95]) for standard normal Z
# Rec.601 luma; the map conditions nets on broadcast-ish content.
_LUMA = (0.299, 0.587, 0.114)
# Map-conditioned nets are trained with per-CHANNEL AWGN sigma in the map plane,
# but the luma mix attenuates iid channel noise by sqrt(sum(c^2)) = 0.6686; scale
# the luma-domain estimate back to per-channel units.
_CHANNEL_FROM_LUMA = 1.0 / (0.299 ** 2 + 0.587 ** 2 + 0.114 ** 2) ** 0.5


def _to_luma_2d(frame: Any) -> Any:
    """(H,W,C) or (1,H,W,C), [0,1] -> (H,W) fp32 luma."""
    f = frame[0] if frame.ndim == 4 else frame
    f = f.astype(mx.float32)
    if f.shape[-1] == 1:
        return f[..., 0]
    return _LUMA[0] * f[..., 0] + _LUMA[1] * f[..., 1] + _LUMA[2] * f[..., 2]


def _edge_pad_hw(x: Any, ph: int, pw: int) -> Any:
    """Edge-replicate pad (H,W) or (K,H,W) on the bottom/right."""
    if ph:
        x = mx.concatenate([x, mx.broadcast_to(x[..., -1:, :], x.shape[:-2] + (ph, x.shape[-1]))], axis=-2)
    if pw:
        x = mx.concatenate([x, mx.broadcast_to(x[..., :, -1:], x.shape[:-1] + (pw,))], axis=-1)
    return x


def _box_smooth_coarse(g: Any, passes: int = 2) -> Any:
    """3x3 edge-padded box mean on a coarse (bh,bw) grid."""
    bh, bw = g.shape
    for _ in range(passes):
        p = mx.concatenate([g[:1], g, g[-1:]], axis=0)          # (bh+2, bw)
        p = mx.concatenate([p[:, :1], p, p[:, -1:]], axis=1)    # (bh+2, bw+2)
        acc = mx.zeros_like(g)
        for i in range(3):
            for j in range(3):
                acc = acc + p[i:i + bh, j:j + bw]
        g = acc / 9.0
    return g


def _box_blur_full(x: Any, k: int) -> Any:
    """Separable edge-padded box blur of odd width k on an (H,W) map."""
    h, w = x.shape
    r = k // 2
    x4 = x[None, :, :, None]
    x4 = mx.concatenate([mx.broadcast_to(x4[:, :1], (1, r, w, 1)), x4,
                         mx.broadcast_to(x4[:, -1:], (1, r, w, 1))], axis=1)
    x4 = mx.conv2d(x4, mx.full((1, k, 1, 1), 1.0 / k), stride=1, padding=0)
    x4 = mx.concatenate([mx.broadcast_to(x4[:, :, :1], (1, h, r, 1)), x4,
                         mx.broadcast_to(x4[:, :, -1:], (1, h, r, 1))], axis=2)
    x4 = mx.conv2d(x4, mx.full((1, 1, k, 1), 1.0 / k), stride=1, padding=0)
    return x4[0, :, :, 0]


def _select_runs(n: int, max_frames: int, run_len: int = 3) -> list[list[int]]:
    """Pick frame indices for diffing under a max_frames budget.

    Diffs must be between temporally ADJACENT frames (distant diffs measure
    motion, not noise), so when the window exceeds the budget we take short
    runs of `run_len` consecutive frames spread evenly across the window
    instead of just its head -- the estimate then reflects the whole window.
    """
    if n <= max_frames:
        return [list(range(n))]
    n_runs = max(2, max_frames // run_len)
    span = n - run_len
    starts = sorted({round(i * span / (n_runs - 1)) for i in range(n_runs)})
    return [list(range(s, s + run_len)) for s in starts]


def _frame_low_quantile_sigma(d: Any, block: int = 16, q: float = 0.30) -> float:
    """Robust global sigma for one luma diff frame.

    Uses the low quantile of block RMS values: global noise pulses lift even
    quiet blocks, while subject/camera motion usually occupies a smaller set of
    blocks and should not drive a whole-frame pulse gain.
    """
    H, W = int(d.shape[0]), int(d.shape[1])
    ph, pw = (-H) % block, (-W) % block
    dp = _edge_pad_hw(d[None, :, :], ph, pw)[0]
    hp, wp = H + ph, W + pw
    bh, bw = hp // block, wp // block
    db = dp.reshape(bh, block, bw, block)
    br = mx.sqrt(mx.mean(db * db, axis=(1, 3))) * _CHANNEL_FROM_LUMA
    flat = mx.sort(br.reshape(-1))
    idx = int(max(0.0, min(1.0, q)) * (int(flat.shape[0]) - 1))
    return float(flat[idx])


def _mc_pair_medians(y: Any, ia: list, ib: list, B: int = 32) -> tuple[Any, Any]:
    """Phase-correlation motion-compensated residual medians per frame pair.

    y: (T,H,W) luma stack; (ia[i], ib[i]) index the frames of pair i.  Each
    pair is cut into non-overlapping BxB blocks; per block, phase correlation
    (Hann-windowed, normalized cross power) finds the dominant displacement
    with parabolic subpixel refinement, block b is aligned to block a by a
    Fourier phase ramp, and the whitened aligned residual's central-region
    median is taken.  The unaligned residual median is computed the same way
    and the per-block MINIMUM of the two is returned: flat blocks (no peak
    structure) and failed alignments (periodic texture, occlusion, shifts
    beyond B/4) fall back to the plain diff instead of poisoning the floor.

    Returns (med, lum): (P, nb) whitened residual medians in luma-sigma-like
    units (scaled so iid per-frame noise of sigma reads 0.6745*sigma as a
    median, matching the flat floor's convention) and (P, nb) block lumas.
    """
    H, W = int(y.shape[1]), int(y.shape[2])
    S = B // 2                                   # half-overlap: model granularity
    ny, nx = (H - B) // S + 1, (W - B) // S + 1
    nb = ny * nx
    ys = (mx.arange(ny) * S)[:, None, None, None]
    xs = (mx.arange(nx) * S)[None, :, None, None]
    idx = ((ys + mx.arange(B)[None, None, :, None]) * W
           + (xs + mx.arange(B)[None, None, None, :])).reshape(nb * B * B)
    P = len(ia)
    ya = mx.take(y, mx.array(ia, dtype=mx.int32), axis=0).reshape(P, H * W)
    yb = mx.take(y, mx.array(ib, dtype=mx.int32), axis=0).reshape(P, H * W)
    A = mx.take(ya, idx, axis=1).reshape(P * nb, B, B)
    Bb = mx.take(yb, idx, axis=1).reshape(P * nb, B, B)

    n1 = mx.arange(B).astype(mx.float32)
    w1 = 0.5 - 0.5 * mx.cos(2.0 * 3.141592653589793 * n1 / (B - 1))
    win = (w1[:, None] * w1[None, :])[None]
    ky = mx.concatenate([mx.arange(B // 2 + 1),
                         mx.arange(-(B - B // 2 - 1), 0)]).astype(mx.float32)[None, :, None]
    kx = mx.arange(B // 2 + 1).astype(mx.float32)[None, None, :]

    Fa = mx.fft.rfft2(A * win)
    Fbw = mx.fft.rfft2(Bb * win)
    R = Fa * mx.conj(Fbw)
    R = R / (mx.abs(R) + 1e-9)
    corr = mx.fft.irfft2(R, s=(B, B)).reshape(P * nb, B * B)
    peak = mx.argmax(corr, axis=-1)
    py = (peak // B).astype(mx.int32)
    px = (peak % B).astype(mx.int32)

    def _at(dy: int, dx: int) -> Any:
        gi = (((py + dy) % B) * B + (px + dx) % B)[:, None]
        return mx.take_along_axis(corr, gi.astype(mx.int32), axis=-1)[:, 0]

    c0 = _at(0, 0)
    sub_y = 0.5 * (_at(-1, 0) - _at(1, 0)) / (_at(-1, 0) - 2 * c0 + _at(1, 0) - 1e-9)
    sub_x = 0.5 * (_at(0, -1) - _at(0, 1)) / (_at(0, -1) - 2 * c0 + _at(0, 1) - 1e-9)
    dy = py.astype(mx.float32)
    dx = px.astype(mx.float32)
    dy = mx.where(dy > B / 2, dy - B, dy) + mx.clip(sub_y, -0.5, 0.5)
    dx = mx.where(dx > B / 2, dx - B, dx) + mx.clip(sub_x, -0.5, 0.5)
    in_range = (mx.abs(dy) <= B / 4) & (mx.abs(dx) <= B / 4)

    ang = (-2.0 * 3.141592653589793 / B) * (ky * dy[:, None, None] + kx * dx[:, None, None])
    ramp = mx.cos(ang) + 1j * mx.sin(ang)
    b_al = mx.fft.irfft2(mx.fft.rfft2(Bb) * ramp, s=(B, B))

    box3 = mx.full((1, 3, 3, 1), 1.0 / 9.0)

    def _white_med(res: Any) -> Any:
        w = res - mx.conv2d(res[..., None], box3, padding=1)[..., 0]
        lo, hi = B // 4, B - B // 4
        # /sqrt2 pair convention, /0.9428 whitening attenuation for iid noise
        v = mx.abs(w[:, lo:hi, lo:hi]) * (1.0 / (1.4142135623730951 * 0.9428))
        v = v.reshape(v.shape[0], -1)
        return mx.sort(v, axis=-1)[:, v.shape[-1] // 2]

    med_mc = mx.where(in_range, _white_med(A - b_al), mx.full((P * nb,), 1e9))
    med_zero = _white_med(A - Bb)
    # hysteresis: on static noise the correlation peak is the noise's best
    # self-match and alignment shaves ~5-10% off the residual; real motion
    # cuts it by 2-5x.  Only trust the aligned reading when it undercuts the
    # plain diff decisively, so static blocks measure exactly the plain diff.
    med = mx.where(med_mc < 0.90 * med_zero, med_mc, med_zero).reshape(P, nb)
    lo, hi = B // 4, B - B // 4
    lum = mx.mean(A[:, lo:hi, lo:hi], axis=(1, 2)).reshape(P, nb)
    return med, lum


def estimate_sigma_map(
    frames: list,
    block: int = 16,
    max_frames: int = 12,
    luma_cap_headroom: float = 2.0,
    motion_cap: str = "strict",
    masking: float = 0.0,
    pulse_robust: bool = False,
    pulse_clip_ratio: float = 1.35,
    floor_mode: str = "mc",
    luma_bins: int = 8,
    sigma_floor: float = 0.002,
    sigma_ceil: float = 0.25,
    smooth: bool = True,
) -> Any | None:
    """Estimate a per-pixel noise sigma map from a window of frames.

    frames: list of (H,W,C) or (1,H,W,C) arrays in [0,1] (any float dtype).
    Returns an (H,W,1) fp32 sigma map in [0,1] luma units, or None when fewer
    than 2 frames are given. Square it for variance-conditioned nets (PVDD).
    Windows longer than max_frames are sampled as spread runs of consecutive
    frames, so the estimate covers the whole window, not just its head.
    When pulse_robust is enabled, whole-frame diff spikes are winsorized before
    the spatial map is estimated; use it together with PulseGain so GOP/noise
    pulses are represented as per-frame gains instead of baked into the base map.
    floor_mode picks the motion-immune noise-floor source for the luma model:
    "mc" (default) aligns 32px blocks by phase correlation first and measures
    the whitened aligned residual on all pixels; "flat" measures flat pixels
    only (aperture-gated whitened median). mc is stronger on pans over weak
    texture (the flat set there is thin or drift-contaminated) and cheaper;
    they agree wherever the flat floor is healthy.
    """
    if len(frames) < 2:
        return None
    if floor_mode not in ("flat", "mc"):
        raise ValueError(f"floor_mode must be 'flat' or 'mc'; got {floor_mode!r}")
    runs = _select_runs(len(frames), max_frames)
    lum_all: list = []
    diffs: list = []
    diffs2: list = []
    pair_means: list = []
    pair_means2: list = []
    sdiffs: list = []
    sdiffs2: list = []
    p1a: list = []
    p1b: list = []
    p2a: list = []
    p2b: list = []
    for run in runs:
        off = len(lum_all)
        lum = [_to_luma_2d(frames[i]) for i in run]
        lum_all.extend(lum)
        yr = mx.stack(lum, axis=0)
        diffs.append(mx.abs(yr[1:] - yr[:-1]) * (1.0 / 1.4142135623730951))
        sdiffs.append((yr[1:] - yr[:-1]) * (1.0 / 1.4142135623730951))
        pair_means.append((yr[1:] + yr[:-1]) * 0.5)
        p1a.extend(off + i for i in range(len(run) - 1))
        p1b.extend(off + i + 1 for i in range(len(run) - 1))
        if yr.shape[0] >= 3:
            diffs2.append(mx.abs(yr[2:] - yr[:-2]) * (1.0 / 1.4142135623730951))
            sdiffs2.append((yr[2:] - yr[:-2]) * (1.0 / 1.4142135623730951))
            pair_means2.append((yr[2:] + yr[:-2]) * 0.5)
            p2a.extend(off + i for i in range(len(run) - 2))
            p2b.extend(off + i + 2 for i in range(len(run) - 2))
    y = mx.stack(lum_all, axis=0)                               # sampled lumas
    d = mx.concatenate(diffs, axis=0)                           # (K,H,W) adjacent diffs
    sd = mx.concatenate(sdiffs, axis=0)                         # signed diffs
    pm = mx.concatenate(pair_means, axis=0)                     # (K,H,W) pair-mean lumas
    d2 = mx.concatenate(diffs2, axis=0) if diffs2 else None     # lag-2 diffs
    sd2 = mx.concatenate(sdiffs2, axis=0) if sdiffs2 else None
    pm2 = mx.concatenate(pair_means2, axis=0) if pair_means2 else None
    H, W = int(y.shape[1]), int(y.shape[2])
    K = d.shape[0]
    if pulse_robust and K >= 4:
        trace = mx.sqrt(mx.mean(d * d, axis=(1, 2)))
        ts = mx.sort(trace)
        med = float(ts[int(0.50 * (K - 1))])
        if med > 1e-6:
            limit = med * max(1.0, float(pulse_clip_ratio))
            scale = mx.minimum(1.0, limit / (trace + 1e-6))
            d = d * scale[:, None, None]

    # block-pool |d| over space and time; low quantile -> sigma per block
    ph, pw = (-H) % block, (-W) % block
    d = _edge_pad_hw(d, ph, pw)
    ylum = _edge_pad_hw(mx.mean(y, axis=0), ph, pw)
    hp, wp = H + ph, W + pw
    bh, bw = hp // block, wp // block
    db = d.reshape(K, bh, block, bw, block)
    db = mx.transpose(db, (1, 3, 0, 2, 4)).reshape(bh, bw, K * block * block)
    n = db.shape[-1]
    k = max(1, int(round(_TAIL_FRACTION * n)))
    tail = mx.sort(db, axis=-1)[..., n - k:]                    # top-5% |d| per block
    sig = mx.sqrt(mx.mean(tail * tail, axis=-1))         * (_CHANNEL_FROM_LUMA / _TAIL_RMS_NORM)                 # (bh,bw)
    blum = mx.mean(ylum.reshape(bh, block, bw, block), axis=(1, 3))   # (bh,bw)

    gx = mx.concatenate([mx.abs(ylum[:, 1:] - ylum[:, :-1]),
                         mx.zeros((hp, 1), dtype=mx.float32)], axis=1)
    gy = mx.concatenate([mx.abs(ylum[1:, :] - ylum[:-1, :]),
                         mx.zeros((1, wp), dtype=mx.float32)], axis=0)
    gsum = gx + gy
    dblk = mx.mean(gsum.reshape(bh, block, bw, block), axis=(1, 3))

    ratio = None
    if d2 is not None:
        d2p = _edge_pad_hw(d2, ph, pw)
        K2 = d2p.shape[0]
        db2 = d2p.reshape(K2, bh, block, bw, block)
        db2 = mx.transpose(db2, (1, 3, 0, 2, 4)).reshape(bh, bw, -1)
        r1 = mx.sqrt(mx.mean(db * db, axis=-1))
        r2 = mx.sqrt(mx.mean(db2 * db2, axis=-1))
        ratio = r2 / (r1 + 1e-6)

    # Aperture-gated flat-pixel noise floor.  A frame difference decomposes as
    # d ~ v.grad(I) + noise: motion can only create temporal change where
    # spatial gradient exists, while sensor/AWGN noise flickers everywhere,
    # including flat pixels.  Median |d| over pixels that are flat in the pair
    # (gradient of the blurred pair-mean below a per-pair adaptive threshold)
    # is therefore a per-block sigma that motion cannot lift -- with two escape
    # hatches for texture masquerading as flatness:
    #   * a large pan dragging weak texture whose gradient the blur hides is
    #     caught by the lag-2 signature inside the flat set itself (noise gives
    #     |d2| ~ |d1|, drifting texture ~ 2|d1|; blocks above 1.30 rejected);
    #   * a frame that is texture everywhere has no honest flat set at all --
    #     its adaptive threshold saturates, which marks the whole floor as
    #     untrustworthy (translated fine texture decorrelates within one shift
    #     and is then statistically identical to per-frame noise).
    def _flat_block_median(dd: Any, pmm: Any, kk: int) -> tuple[Any, Any, float, Any]:
        sm_ = mx.stack([_box_blur_full(pmm[i], 5) for i in range(kk)], axis=0)
        gx_ = mx.concatenate([mx.abs(sm_[:, :, 1:] - sm_[:, :, :-1]),
                              mx.zeros((kk, H, 1), dtype=mx.float32)], axis=2)
        gy_ = mx.concatenate([mx.abs(sm_[:, 1:, :] - sm_[:, :-1, :]),
                              mx.zeros((kk, 1, W), dtype=mx.float32)], axis=1)
        gp_ = gx_ + gy_
        gs_ = mx.sort(gp_.reshape(kk, -1), axis=-1)
        thr_ = mx.clip(gs_[:, int(0.35 * (H * W - 1))] * 1.5, 0.016, 0.060)[:, None, None]
        thr_s = sorted(float(v) for v in thr_.reshape(-1).tolist())
        thr_med_ = thr_s[len(thr_s) // 2]
        fl_ = (gp_ < thr_).astype(mx.float32)
        fb_ = _edge_pad_hw(fl_, ph, pw).reshape(kk, bh, block, bw, block)
        fb_ = mx.transpose(fb_, (1, 3, 0, 2, 4)).reshape(bh, bw, -1)
        nn = int(fb_.shape[-1])
        cnt_ = mx.sum(fb_, axis=-1)
        masked_ = mx.where(fb_ > 0.5, dd, mx.full(dd.shape, 1e9, dtype=mx.float32))
        msort_ = mx.sort(masked_, axis=-1)
        midx_ = mx.clip((cnt_ - 1.0) * 0.5, 0.0, float(nn - 1)).astype(mx.int32)
        med_ = mx.take_along_axis(msort_, midx_[..., None], axis=-1)[..., 0]
        return med_, cnt_, thr_med_, fl_

    if floor_mode == "mc":
        # Motion-compensated floor: per-block phase correlation aligns each
        # pair before the whitened residual median, so the floor is read on
        # ALL pixels rather than the gradient-free subset -- pans over weak
        # texture (the flat floor's residual blind spot) align away instead
        # of contaminating the estimate.  Validation mirrors the flat floor:
        # per-block lag-2 consistency plus a global drift median; the floor
        # blocks live on their own 32px grid and carry their own lumas.
        med1p, lum1p = _mc_pair_medians(y, p1a, p1b)
        P1 = int(med1p.shape[0])
        med1b = mx.sort(mx.transpose(med1p, (1, 0)), axis=-1)[:, P1 // 2]
        mlum = mx.mean(lum1p, axis=0)
        r_flat_med = 2.0
        if p2a:
            med2p, _ = _mc_pair_medians(y, p2a, p2b)
            P2 = int(med2p.shape[0])
            med2b = mx.sort(mx.transpose(med2p, (1, 0)), axis=-1)[:, P2 // 2]
            mc_static = med1b < (2.0 / 255.0)
            mc_ok = mc_static | (med2b <= 1.30 * med1b)
            rr = [m2 / max(m1, 1e-6) for m1, m2, st in zip(
                [float(v) for v in med1b.tolist()],
                [float(v) for v in med2b.tolist()],
                [bool(v) for v in mc_static.tolist()], strict=True) if not st]
            r_flat_med = sorted(rr)[len(rr) // 2] if rr else 1.0
        else:
            mc_ok = mx.zeros(med1b.shape, dtype=mx.bool_)
        flat_trusted = True
        fsig_mc = med1b * (_CHANNEL_FROM_LUMA / 0.6745)
        flat_sig_l = [float(v) for v in fsig_mc.tolist()]
        flat_ok_l = [bool(v) for v in mc_ok.tolist()]
        floor_lum_l = [float(v) for v in mlum.tolist()]
    else:
        # The floor is measured on the WHITENED diff: d minus its own 3x3
        # mean.  Clean-encode wobble, sub-pixel drift, and deformable-texture
        # shimmer (fur, foliage) produce a spatially smooth diff field (lag-1
        # correlation 0.92-0.98 measured on clean re-encodes) that the
        # high-pass removes, while sensor/AWGN noise is spatially white and
        # survives (x0.9428 for iid, corrected below).  This is what makes
        # the floor read the DENOISEABLE component, not any temporal change.
        def _whiten(sig_diff: Any) -> Any:
            w = mx.stack([_box_blur_full(sig_diff[i], 3)
                          for i in range(int(sig_diff.shape[0]))], axis=0)
            return mx.abs(sig_diff - w) * (1.0 / 0.9428)

        dw = _edge_pad_hw(_whiten(sd), ph, pw)
        dwb = mx.transpose(dw.reshape(K, bh, block, bw, block),
                           (1, 3, 0, 2, 4)).reshape(bh, bw, n)
        # value floor from the WHITENED diff; displacement test from the RAW
        # diff (whitening is symmetric in lag, so it would erase the lag-2
        # doubling signature that separates drift from noise)
        fmed1, fcnt1, flat_thr_med, _ = _flat_block_median(dwb, pm, K)
        fmed1r, _, _, _ = _flat_block_median(db, pm, K)
        flat_trusted = flat_thr_med <= 0.045
        r_flat_med = 2.0
        if sd2 is not None and pm2 is not None and flat_trusted:
            d2pf = _edge_pad_hw(d2, ph, pw)
            K2f = int(d2pf.shape[0])
            db2f = mx.transpose(d2pf.reshape(K2f, bh, block, bw, block),
                                (1, 3, 0, 2, 4)).reshape(bh, bw, -1)
            fmed2, fcnt2, _, _ = _flat_block_median(db2f, pm2, K2f)
            r_flat = fmed2 / mx.maximum(fmed1r, 1e-6)
            flat_static = fmed1r < (2.0 / 255.0)
            counted = (fcnt1 >= 24) & (fcnt2 >= 24)
            flat_ok = counted & (flat_static | (r_flat <= 1.30))
            # global drift signature over ALL counted blocks (not just the
            # passing ones): source-wide slow drift puts the median well
            # above 1.30 even when a tail of blocks slips under the per-block
            # gate.
            rf_l = [float(r) for r, c, st in zip(
                [float(v) for v in r_flat.reshape(-1).tolist()],
                [bool(v) for v in counted.reshape(-1).tolist()],
                [bool(v) for v in flat_static.reshape(-1).tolist()], strict=True)
                if c and not st]
            r_flat_med = sorted(rf_l)[len(rf_l) // 2] if rf_l else 1.0
        else:
            flat_ok = mx.zeros(fmed1.shape, dtype=mx.bool_)
        # |d| of iid per-frame noise is half-normal with median 0.6745 sigma
        fsig = fmed1 * (_CHANNEL_FROM_LUMA / 0.6745)
        flat_sig_l = [float(v) for v in fsig.reshape(-1).tolist()]
        flat_ok_l = [bool(v) for v in flat_ok.reshape(-1).tolist()]
        floor_lum_l = None

    # signal-dependence model: robust sigma-vs-luma from the quiet blocks, used to
    # cap motion-dominated blocks (their pooled quantile is inflated everywhere).
    sig_l = [float(v) for v in sig.reshape(-1).tolist()]
    lum_l = [float(v) for v in blum.reshape(-1).tolist()]
    det_l = [float(v) for v in dblk.reshape(-1).tolist()]
    def _p30(vals: list) -> float:
        s = sorted(vals)
        return s[int(0.30 * (len(s) - 1))]
    nd = len(det_l)
    det_s = sorted(det_l)
    # The luma cap needs a quiet spatial baseline. On clips where every block is
    # detailed and changing (textured subject/camera motion), the old all-block
    # p30 baseline was itself motion-contaminated, so strict mode could still
    # emit a huge map. Use genuinely low-detail blocks when available; if there
    # are none, fall back to a conservative strict-mode baseline and close the
    # flicker bypass below.
    detail_gate = max(0.006, min(det_s[int(0.40 * (nd - 1))], 0.035))
    quiet_idx = [i for i, dv in enumerate(det_l) if dv <= detail_gate]
    min_quiet = max(8, int(0.15 * len(sig_l)))
    have_quiet_detail = len(quiet_idx) >= min_quiet
    model_sig_l = [sig_l[i] for i in quiet_idx] if have_quiet_detail else sig_l
    model_lum_l = [lum_l[i] for i in quiet_idx] if have_quiet_detail else lum_l
    sig_s = sorted(sig_l)
    sig_p30 = sig_s[int(0.30 * (len(sig_s) - 1))]
    sig_med = sig_s[len(sig_s) // 2]
    sig_p95 = sig_s[int(0.95 * (len(sig_s) - 1))]
    den = mx.mean((db > (2.0 / 255.0)).astype(mx.float32), axis=-1)
    den_l = [float(v) for v in den.reshape(-1).tolist()]
    den_s = sorted(den_l)
    static_fraction = float(mx.mean((mx.max(d, axis=0) < (2.0 / 255.0)).astype(mx.float32)))
    den_med = den_s[len(den_s) // 2]
    ratio_med = 2.0
    if ratio is not None:
        ratio_l = [float(v) for v in ratio.reshape(-1).tolist()]
        ratio_s = sorted(ratio_l)
        ratio_med = ratio_s[len(ratio_s) // 2]
    # On dense-change content (subject/camera motion or dense noise) every
    # block's tail statistic is lifted, so the detail-gated model above is
    # itself contaminated -- measured on clean moving re-encodes it reads the
    # same ~0.06 as on sigma-0.06 noise.  The flat-pixel floor separates the
    # two: it stays at the true noise level regardless of motion.  Two regimes:
    #   * full flat mode -- the flat set is globally noise-like (block-median
    #     lag ratio <= 1.30) and physically plausible as noise (<= 0.12): the
    #     floor IS the model and gets a tight ceiling below.
    #   * floor-min mode -- the flat set carries drift (a pan), so only its
    #     ratio-validated subset is trusted, and only in the DOWNWARD
    #     direction: it may lower the legacy model (less denoising of clean
    #     content) but never raise it.  The legacy guards stay in charge.
    # Static/sparse-flash content keeps the legacy path untouched (the flat
    # median would under-read sparse flashes there).
    floor_lum = floor_lum_l if floor_lum_l is not None else lum_l
    min_flat_blocks = max(8, int(0.10 * len(flat_sig_l)))
    flat_idx = [i for i, ok in enumerate(flat_ok_l) if ok]
    have_flat = len(flat_idx) >= min_flat_blocks
    flat_floor_p30 = _p30([flat_sig_l[i] for i in flat_idx]) if have_flat else None
    flat_active = (
        motion_cap != "off"
        and have_flat
        and den_med >= 0.45
        and static_fraction <= 0.10
    )
    use_flat_model = (
        flat_active
        and r_flat_med <= 1.35
        and flat_floor_p30 is not None
        and flat_floor_p30 <= 0.12
    )
    use_flat_min = flat_active and not use_flat_model
    if use_flat_model:
        model_sig_l = [flat_sig_l[i] for i in flat_idx]
        model_lum_l = [floor_lum[i] for i in flat_idx]
    global_p30 = _p30(model_sig_l)
    dense_texture_ambiguous = (
        motion_cap == "strict"
        and not use_flat_model
        and ratio is not None
        and sig_med > 0.08
        and den_s[len(den_s) // 2] >= 0.55
        and static_fraction <= 0.06
        and ratio_med <= 1.35
    )
    # Uniform dense temporal change can be true sensor/AWGN noise. Real motion
    # contamination is usually spatially uneven, so only the high-spread case
    # gets the stricter cap; the uniform case keeps the older softer cap.
    dense_motion_heterogeneous = sig_p95 > max(0.12, 1.60 * max(sig_p30, 1e-6))
    source_wide_motion_contaminated = (
        motion_cap == "strict"
        and not use_flat_model
        and ratio is not None
        and sig_med > 0.06
        and den_med >= 0.55
        and static_fraction <= 0.15
        and ratio_med >= 1.35
    )
    if dense_texture_ambiguous:
        global_p30 = min(global_p30, 0.020 if dense_motion_heterogeneous else 0.035)
    if source_wide_motion_contaminated:
        global_p30 = min(global_p30, 0.020)
    model = [global_p30] * luma_bins
    for b in range(luma_bins):
        lo, hi = b / luma_bins, (b + 1) / luma_bins
        vals = [s for s, lv in zip(model_sig_l, model_lum_l, strict=True) if lo <= lv < hi]
        if len(vals) >= 8:
            model[b] = _p30(vals)
    if dense_texture_ambiguous:
        dense_cap = 0.020 if dense_motion_heterogeneous else 0.035
        model = [min(v, dense_cap) for v in model]
    if source_wide_motion_contaminated:
        model = [min(v, 0.020) for v in model]
    if use_flat_min and flat_floor_p30 is not None:
        flat_model = [flat_floor_p30] * luma_bins
        for b in range(luma_bins):
            lo, hi = b / luma_bins, (b + 1) / luma_bins
            vals = [flat_sig_l[i] for i in flat_idx if lo <= floor_lum[i] < hi]
            if len(vals) >= 8:
                flat_model[b] = _p30(vals)
        model = [min(v, fv) for v, fv in zip(model, flat_model, strict=True)]
    # absolute slack keeps the cap from crushing genuine local noise on clips
    # whose baseline sits near the floor (heavy compression): the cap's job is
    # rejecting motion blowups (0.1-0.3+), not flattening the map to 2x floor.
    if use_flat_model:
        # The flat floor is a direct sigma estimate, not a contaminated lower
        # bound, and on dense-change content the tail's elevation above it is
        # motion.  Keep the ceiling tight so motion cannot inflate the map.
        cap = [1.25 * model[min(luma_bins - 1, max(0, int(lv * luma_bins)))] + 0.005
               for lv in lum_l]
    else:
        cap = [luma_cap_headroom * model[min(luma_bins - 1, max(0, int(lv * luma_bins)))] + 0.01
               for lv in lum_l]
    capped = mx.minimum(sig, mx.array(cap, dtype=mx.float32).reshape(bh, bw))
    # flicker bypass: the cap exists for DISPLACEMENT motion, which doubles
    # over two frames (lag2/lag1 -> 2); per-frame flicker is temporally
    # independent (ratio -> 1). Spatially CONCENTRATED noise would otherwise be
    # indistinguishable from motion outliers and get flattened to the quiet
    # majority's cap -- exactly the blocks a noise map is for. Blend by
    # per-block flicker-ness so flicker keeps its measured level and
    # displacement still gets capped.
    # motion_cap modes: temporal statistics CANNOT cleanly separate flicker
    # that persists a couple of frames (compression artifacts refresh at coding
    # cadence -> lag ratio ~1.5) from occlusion edges (~1.41) -- the
    # distributions overlap, so the cap's aggressiveness is a material
    # decision the caller makes:
    #   strict (default) -- displacement-safe: only temporally independent
    #     flicker (ratio ~1) bypasses the cap; anything edge/motion-like is
    #     capped. Right for footage with real subject/camera motion.
    #   loose -- static-camera material: persistent compression flicker
    #     (ratio up to ~1.7) also bypasses; occlusion edges partially leak.
    #   off -- tripod/archival material: no motion cap at all; the map reports
    #     exactly what it measured.
    if motion_cap == "off":
        pass
    elif use_flat_model:
        # The flat-pixel model already measures the true noise floor under
        # motion, so the cap is trustworthy here; the bypass would only let
        # motion peaks (whose lag ratio noise pulls toward 1) leak past it.
        sig = capped
    elif ratio is not None:
        if motion_cap == "loose":
            flick = mx.clip((2.0 - ratio) * (1.0 / 0.3), 0.0, 1.0)   # open <=1.7
        else:
            flick = mx.clip((1.35 - ratio) * (1.0 / 0.35), 0.0, 1.0)  # open <=1.0
            if dense_texture_ambiguous or source_wide_motion_contaminated:
                # Dense textured motion can have ratio ~= 1, the same as
                # temporal noise; broad, source-wide motion can also occupy the
                # whole frame with no stable baseline. In either case strict
                # mode should prefer a conservative cap over a giant map that
                # denoises real detail as if it were noise.
                flick = flick * 0.0
        sig = flick * sig + (1.0 - flick) * capped
    else:
        sig = capped

    if masking > 0.0:
        # perceptual masking: noise VISIBILITY is highest in flat regions and
        # lowest near edges/texture (spatial masking), while over-conditioning
        # near detail is what reads as softness. Weight the measured map by
        # local detail: flat blocks get a suppression margin (up to ~1.75x at
        # masking=1, matching the visual-kill margin over measured amplitude),
        # detailed blocks are tempered toward ~0.5x. masking=0 disables.
        df = mx.sort(dblk.reshape(-1))
        nb = df.shape[0]
        lo_d, hi_d = float(df[int(0.2 * (nb - 1))]), float(df[int(0.9 * (nb - 1))])
        dn = mx.clip((dblk - lo_d) / max(hi_d - lo_d, 1e-6), 0.0, 1.0)
        margin = 1.0 + masking * (0.75 - 1.25 * dn)   # flat 1.75x .. detail 0.5x
        sig = sig * margin
    if smooth:
        sig = _box_smooth_coarse(sig)
    full = mx.repeat(mx.repeat(sig, block, axis=0), block, axis=1)[:H, :W]
    if smooth:
        full = _box_blur_full(full, block + 1)
    full = mx.clip(full, sigma_floor, sigma_ceil)
    return full[:, :, None].astype(mx.float32)


