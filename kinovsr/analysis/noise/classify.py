"""Noise analysis and classification diagnostics.

Split of the noise_map module: the analyze_noise statistic battery and
the classify_noise_analysis decision layer over it.
"""

from __future__ import annotations

import mlx.core as mx

from .estimate import (
    _CHANNEL_FROM_LUMA,
    _box_blur_full,
    _mc_pair_medians,
    _to_luma_2d,
)
from .grid import (
    _highpass_rows,
    _profile_spectrum,
    _spectrum_peak,
)


def analyze_noise(frames: list, thresh: float = 2.0 / 255.0) -> dict:
    """Diagnostic battery over one window of frames: every statistic family that
    can characterize temporal noise, for probing clips where the estimator and
    the eye disagree. Returns a dict of scalars (block statistics are reported
    as (median, p90) over 16px blocks; sigma-like values in per-channel units).
    """
    lum = [_to_luma_2d(f) for f in frames]
    y = mx.stack(lum, axis=0)
    d = mx.abs(y[1:] - y[:-1]) * (1.0 / 1.4142135623730951)
    K, H, W = int(d.shape[0]), int(d.shape[1]), int(d.shape[2])
    block = 16
    hb, wb = H // block * block, W // block * block
    db = d[:, :hb, :wb].reshape(K, hb // block, block, wb // block, block)
    db = mx.transpose(db, (1, 3, 0, 2, 4)).reshape(hb // block, wb // block, -1)
    n = int(db.shape[-1])
    srt = mx.sort(db, axis=-1)
    out: dict = {}

    def blkstats(v):
        f = mx.sort(v.reshape(-1))
        m = int(f.shape[0])
        return float(f[m // 2]), float(f[int(0.9 * (m - 1))])

    # quantiles, AWGN-normalized (sqrt2*erfinv(p))
    for pq, fac in ((0.50, 0.6745), (0.75, 1.1503), (0.90, 1.6449), (0.95, 1.9600)):
        q = srt[..., int(pq * (n - 1))] * (_CHANNEL_FROM_LUMA / fac)
        out[f"q{int(pq * 100)}"] = blkstats(q)
    out["rms"] = blkstats(mx.sqrt(mx.mean(db * db, axis=-1)) * _CHANNEL_FROM_LUMA)
    for frac, norm, name in ((0.05, 2.3616, "tail5"), (0.01, 2.9110, "tail1")):
        k = max(1, int(round(frac * n)))
        t = srt[..., n - k:]
        out[name] = blkstats(mx.sqrt(mx.mean(t * t, axis=-1)) * (_CHANNEL_FROM_LUMA / norm))
    out["max"] = blkstats(srt[..., -1] * _CHANNEL_FROM_LUMA)

    # sparseness: how many pixels flicker at all, and how hard THOSE flicker
    fmask = (db > thresh).astype(mx.float32)
    cnt = mx.sum(fmask, axis=-1)
    out["flicker_density"] = blkstats(cnt / n)
    amp = mx.sqrt(mx.sum(db * db * fmask, axis=-1) / mx.maximum(cnt, 1.0)) * _CHANNEL_FROM_LUMA
    out["flicker_amplitude"] = blkstats(amp)

    # aperture-gated noise floor: median |d| over pixels flat in the pair mean
    # (motion needs gradient to change a pixel; noise does not), whole-frame.
    # flat_lag21 is the lag-2/lag-1 ratio INSIDE the flat set: ~1 for noise,
    # ~2 when the flat set is contaminated by drifting weak texture, so a
    # high value marks flat_sigma itself as untrustworthy.
    def _flat_median(dd, pmm, kk):
        smw = mx.stack([_box_blur_full(pmm[i], 5) for i in range(kk)], axis=0)
        gfx = mx.concatenate([mx.abs(smw[:, :, 1:] - smw[:, :, :-1]),
                              mx.zeros((kk, H, 1), dtype=mx.float32)], axis=2)
        gfy = mx.concatenate([mx.abs(smw[:, 1:, :] - smw[:, :-1, :]),
                              mx.zeros((kk, 1, W), dtype=mx.float32)], axis=1)
        gf = gfx + gfy
        gfs = mx.sort(gf.reshape(kk, -1), axis=-1)
        fthr = mx.clip(gfs[:, int(0.35 * (H * W - 1))] * 1.5, 0.016, 0.060)[:, None, None]
        thr_s = sorted(float(v) for v in fthr.reshape(-1).tolist())
        fmask = (gf < fthr).astype(mx.float32)
        fsel = fmask.reshape(-1)
        dflat = mx.where(fsel > 0.5, dd.reshape(-1),
                         mx.full((int(dd.reshape(-1).shape[0]),), 1e9, dtype=mx.float32))
        dfs = mx.sort(dflat)
        nflat = int(mx.sum(fsel))
        med = float(dfs[nflat // 2]) if nflat >= 64 else 0.0
        return med, thr_s[len(thr_s) // 2], fmask

    sdw = (y[1:] - y[:-1]) * (1.0 / 1.4142135623730951)
    wme = mx.stack([_box_blur_full(sdw[i], 3) for i in range(K)], axis=0)
    dwhite = mx.abs(sdw - wme) * (1.0 / 0.9428)
    med1, thr1, fm1 = _flat_median(dwhite, (y[1:] + y[:-1]) * 0.5, K)
    out["flat_sigma"] = med1 * (_CHANNEL_FROM_LUMA / 0.6745)
    # a saturated flat threshold means the frame has no honest flat pixels;
    # flat_sigma is then texture in disguise and must not be trusted
    out["flat_thr"] = thr1
    # spatial whiteness of the flat-set signed diff (noise ~0; smooth encode
    # wobble / deformable-texture shimmer ~1)
    ca, cb = sdw[:, :, :-1], sdw[:, :, 1:]
    cm = fm1[:, :, :-1] * fm1[:, :, 1:]
    out["flat_diff_corr"] = (float(mx.sum(ca * cb * cm)) /
                             (float(mx.sum(0.5 * (ca * ca + cb * cb) * cm)) + 1e-9))
    # the lag ratio uses RAW diffs on both lags: whitening is lag-symmetric
    # and would erase the displacement-doubling signature
    med1r, _, _ = _flat_median(d, (y[1:] + y[:-1]) * 0.5, K)
    if K >= 2 and med1r > 1e-6:
        d2w = mx.abs(y[2:] - y[:-2]) * (1.0 / 1.4142135623730951)
        med2r, _, _ = _flat_median(d2w, (y[2:] + y[:-2]) * 0.5, K - 1)
        out["flat_lag21"] = med2r / med1r
    else:
        out["flat_lag21"] = 1.0

    # motion-compensated floor, mirroring the sigma map's DEFAULT floor
    # (floor_mode="mc"): probe verdicts should describe what the shipping
    # estimator actually measures, not just the flat-pixel diagnostic
    if H >= 32 and W >= 32 and K >= 1:
        p1a = list(range(K))
        p1b = list(range(1, K + 1))
        med1p, _ = _mc_pair_medians(y, p1a, p1b)
        P1 = int(med1p.shape[0])
        med1b = mx.sort(mx.transpose(med1p, (1, 0)), axis=-1)[:, P1 // 2]
        m1 = mx.sort(med1b)
        mc_med = float(m1[int(m1.shape[0]) // 2])
        out["mc_sigma"] = mc_med * (_CHANNEL_FROM_LUMA / 0.6745)
        if K >= 2 and mc_med > 1e-6:
            p2a = list(range(K - 1))
            p2b = list(range(2, K + 1))
            med2p, _ = _mc_pair_medians(y, p2a, p2b)
            P2 = int(med2p.shape[0])
            med2b = mx.sort(mx.transpose(med2p, (1, 0)), axis=-1)[:, P2 // 2]
            rr = sorted(float(b) / max(float(a), 1e-6)
                        for a, b in zip(med1b.tolist(), med2b.tolist(), strict=True)
                        if float(a) >= 2.0 / 255.0)
            out["mc_lag21"] = rr[len(rr) // 2] if rr else 1.0
        else:
            out["mc_lag21"] = 1.0
    else:
        out["mc_sigma"] = 0.0
        out["mc_lag21"] = 1.0

    # per-frame flash trace (whole-frame RMS sigma per diff)
    tr = [float(mx.sqrt(mx.mean(d[i] * d[i]))) * _CHANNEL_FROM_LUMA for i in range(K)]
    out["frame_trace"] = tr

    # persistence: lag-2 vs lag-1 energy (iid per-frame flicker -> ~1.0;
    # frame-alternating -> <1; slow drift/motion -> >1)
    if len(lum) >= 3:
        d2 = mx.abs(y[2:] - y[:-2]) * (1.0 / 1.4142135623730951)
        out["lag2_over_lag1"] = float(mx.sqrt(mx.mean(d2 * d2)) / (mx.sqrt(mx.mean(d * d)) + 1e-9))

    # edge correlation (mosquito noise rides edges): flicker energy near edges
    # vs in flat areas
    my = mx.mean(y, axis=0)
    gx = mx.abs(my[:, 1:] - my[:, :-1])[:-1, :]
    gy = mx.abs(my[1:, :] - my[:-1, :])[:, :-1]
    g = gx + gy
    gs = mx.sort(g.reshape(-1))
    ethr = float(gs[int(0.85 * (gs.shape[0] - 1))])
    em = (g > ethr).astype(mx.float32)
    de = mx.mean(d[:, :-1, :-1] ** 2, axis=0)
    e_edge = float(mx.sum(de * em) / mx.maximum(mx.sum(em), 1.0))
    e_flat = float(mx.sum(de * (1 - em)) / mx.maximum(mx.sum(1 - em), 1.0))
    out["edge_over_flat"] = (e_edge / (e_flat + 1e-12)) ** 0.5

    # luma correlation of block sigma (shadows-noisier structure)
    ylb = mx.mean(y[:, :hb, :wb].mean(axis=0).reshape(hb // block, block, wb // block, block),
                  axis=(1, 3)).reshape(-1)
    sb = mx.sqrt(mx.mean(db * db, axis=-1)).reshape(-1)
    ym, sm = mx.mean(ylb), mx.mean(sb)
    cov = mx.mean((ylb - ym) * (sb - sm))
    out["luma_corr"] = float(cov / (mx.sqrt(mx.mean((ylb - ym) ** 2) * mx.mean((sb - sm) ** 2)) + 1e-12))

    # static-grain proxy: spatial high-frequency energy where NOTHING flickers
    static = (mx.max(d, axis=0) < thresh).astype(mx.float32)
    lap = mx.abs(my[1:-1, 1:-1] * 4 - my[:-2, 1:-1] - my[2:, 1:-1] - my[1:-1, :-2] - my[1:-1, 2:])
    sm2 = static[1:-1, 1:-1]
    out["static_fraction"] = float(mx.mean(static))
    out["static_spatial_hf"] = float(mx.sum(lap * sm2) / mx.maximum(mx.sum(sm2), 1.0)) / 4.4721 * _CHANNEL_FROM_LUMA

    # Row-coherent flicker proxy: thin analog scanlines produce a signed
    # frame-diff profile that repeats at a row period. Dense sensor noise can
    # have row drift too, but it should not have strong short-lag periodicity.
    signed = (y[1:] - y[:-1]) * (1.0 / 1.4142135623730951)
    row = mx.mean(signed, axis=2)                               # (K,H)
    frame_rms = mx.sqrt(mx.mean(signed * signed, axis=(1, 2))) + 1e-9
    row_rms = mx.sqrt(mx.mean(row * row, axis=1))
    rc = mx.sort(row_rms / frame_rms)
    out["row_coherence"] = float(rc[int(rc.shape[0]) // 2])
    if H >= 96:
        # temporal row spectrum: matched projections of the per-pair row-mean
        # diff profiles, magnitude-averaged (jumping scanlines change phase
        # per frame). Reaches down to period 2 -- interlace/field residue --
        # which the old lag-3..25 autocorrelation could never see.
        hp = _highpass_rows(row)
        periods, s = _profile_spectrum(hp, 2.0, 32.0)
        P, prom = _spectrum_peak(periods, s)
        med = sorted(s)[len(s) // 2]
        out["row_period_px"] = P
        out["row_periodicity"] = min(1.0, prom / 8.0)   # ~1.0 = unambiguous comb
        out["row_interlace"] = (s[0] / med) if med > 0 else 0.0   # strength at P=2
        # STATIC row banding: baked scanline/banding patterns produce no
        # temporal diff at all -- measure the mean frame's row profile (the
        # temporal average also suppresses noise by 1/sqrt(T)). Content has
        # its own row-periodic structure (fences, siding, fur), so a peak
        # only counts as BANDING when it is width-coherent: the left and
        # right frame halves must carry the period at the SAME phase.
        import math as _math
        mfr0 = mx.mean(y, axis=0)
        mrow = mx.mean(mfr0, axis=1)[None, :]                    # (1, H)
        speriods, ss = _profile_spectrum(_highpass_rows(mrow), 2.0, 32.0)
        sP, sprom = _spectrum_peak(speriods, ss)
        coher = 0.0
        if sP > 0:
            halves = [mx.mean(mfr0[:, : W // 2], axis=1)[None, :],
                      mx.mean(mfr0[:, W // 2:], axis=1)[None, :]]
            phs, mags = [], []
            for hprof in halves:
                hh = _highpass_rows(hprof)[0]
                ys_ = mx.arange(H).astype(mx.float32)
                a = (2.0 * 3.141592653589793) * ys_ / sP
                zr = float(mx.mean(hh * mx.cos(a)))
                zi = float(mx.mean(hh * mx.sin(a)))
                phs.append(_math.atan2(zi, zr))
                mags.append((zr * zr + zi * zi) ** 0.5)
            dphi = abs(phs[0] - phs[1])
            dphi = min(dphi, 2.0 * 3.141592653589793 - dphi)
            if dphi < 3.141592653589793 / 3.0 and min(mags) > 0.35 * max(mags):
                coher = 1.0
        out["static_row_period_px"] = sP
        out["static_row_periodicity"] = min(1.0, sprom / 8.0) * coher
    else:
        out["row_periodicity"] = 0.0
        out["row_period_px"] = 0.0
        out["row_interlace"] = 0.0
        out["static_row_period_px"] = 0.0
        out["static_row_periodicity"] = 0.0

    # single-frame spatial noise floor on the MEAN frame: temporal noise
    # averages down by 1/sqrt(T) there while STATIC grain / fixed-pattern
    # noise survives fully. Same aperture + whitening conventions as the
    # temporal floors, so the sigmas are directly comparable; the static
    # component is what temporal denoisers cannot touch.
    mfr = mx.mean(y, axis=0)
    smf = _box_blur_full(mfr, 5)
    gmx_ = mx.concatenate([mx.abs(smf[:, 1:] - smf[:, :-1]),
                           mx.zeros((H, 1), dtype=mx.float32)], axis=1)
    gmy_ = mx.concatenate([mx.abs(smf[1:, :] - smf[:-1, :]),
                           mx.zeros((1, W), dtype=mx.float32)], axis=0)
    gm = (gmx_ + gmy_).reshape(-1)
    gs_ = mx.sort(gm)
    thr_sp = min(0.060, max(0.016, float(gs_[int(0.35 * (H * W - 1))]) * 1.5))
    wsp = (mfr - _box_blur_full(mfr, 3)).reshape(-1)
    sel = (gm < thr_sp).astype(mx.float32)
    nsel = int(mx.sum(sel))
    if nsel >= 64:
        vals = mx.sort(mx.where(sel > 0.5, mx.abs(wsp),
                                mx.full((H * W,), 1e9, dtype=mx.float32)))
        med_sp = float(vals[nsel // 2])
        sp_sigma = med_sp * (_CHANNEL_FROM_LUMA / (0.6745 * 0.9428))
    else:
        sp_sigma = 0.0
    out["spatial_sigma"] = sp_sigma
    mc_part = float(out.get("mc_sigma", 0.0)) / max(1.0, float(K)) ** 0.5
    out["static_grain_sigma"] = max(0.0, (sp_sigma ** 2 - mc_part ** 2)) ** 0.5

    # per-channel temporal RMS (is the flash chromatic?)
    for i, ch in enumerate("RGB"):
        c = mx.stack([f[0] if f.ndim == 4 else f for f in frames], axis=0).astype(mx.float32)[..., i]
        dc = (c[1:] - c[:-1]) * (1.0 / 1.4142135623730951)
        out[f"sigma_{ch}"] = float(mx.sqrt(mx.mean(dc * dc)))
    return out


def classify_noise_analysis(stats: dict) -> dict:
    """Interpret `analyze_noise` output for map/probe decisions.

    The classifier is intentionally heuristic. Its job is not to prove the
    artifact source; it calls out when the temporal estimator has enough signal
    to trust, when it is blind, and when motion/noise are statistically
    ambiguous. Values are in the same [0, 1] per-channel sigma-ish units printed
    by --probe-noise.
    """
    dens_med, dens_p90 = stats.get("flicker_density", (0.0, 0.0))
    _, amp_p90 = stats.get("flicker_amplitude", (0.0, 0.0))
    tail5_med, tail5_p90 = stats.get("tail5", (0.0, 0.0))
    _, rms_p90 = stats.get("rms", (0.0, 0.0))
    lag = float(stats.get("lag2_over_lag1", 0.0))
    edge = float(stats.get("edge_over_flat", 1.0))
    static = float(stats.get("static_fraction", 0.0))
    static_hf = float(stats.get("static_spatial_hf", 0.0))
    flat_sigma = float(stats.get("flat_sigma", 0.0))
    flat_lag21 = float(stats.get("flat_lag21", 1.0))
    flat_corr = float(stats.get("flat_diff_corr", 0.0))
    flat_trusted = (float(stats.get("flat_thr", 1.0)) <= 0.045
                    and flat_sigma <= 0.120)
    mc_sigma = float(stats.get("mc_sigma", 0.0))
    mc_lag21 = float(stats.get("mc_lag21", 1.0))
    mc_trusted = 0.0 < mc_sigma <= 0.120
    row_interlace = float(stats.get("row_interlace", 0.0))
    static_row_per = float(stats.get("static_row_period_px", 0.0))
    static_row_str = float(stats.get("static_row_periodicity", 0.0))
    static_grain = float(stats.get("static_grain_sigma", 0.0))
    spatial_sigma = float(stats.get("spatial_sigma", 0.0))
    row_coherence = float(stats.get("row_coherence", 0.0))
    row_periodicity = float(stats.get("row_periodicity", 0.0))
    row_period_px = float(stats.get("row_period_px", 0.0))
    trace = [float(v) for v in stats.get("frame_trace", [])]
    trace_sorted = sorted(trace)
    trace_med = trace_sorted[len(trace_sorted) // 2] if trace_sorted else 0.0
    trace_max = max(trace, default=0.0)

    labels: list[str] = []
    warnings: list[str] = []
    suggestions: list[str] = []

    if tail5_p90 < 0.012 and dens_p90 < 0.10:
        labels.append("low temporal noise")
    if (static_grain >= 0.015 and static >= 0.10) or \
            (static >= 0.15 and static_hf >= 0.006 and dens_p90 < 0.25):
        labels.append("static/structured grain")
        warnings.append(
            f"static grain sigma ~{static_grain:.3f}: temporal denoisers cannot "
            f"remove what does not flicker")
        suggestions.append("spatial cleanup (nafnet-class) or a small --noise-map-floor for the temporal path")
    if row_interlace >= 4.0 and dens_p90 >= 0.10:
        labels.append("interlace/field residue")
        warnings.append("row-alternating flicker (period 2): temporal denoisers smear combing")
        suggestions.append("deinterlace upstream before any denoise/deblock")
    if static_row_str >= 0.6 and static_row_per >= 2.0:
        labels.append("static row banding")
        warnings.append(
            f"baked row pattern at ~{static_row_per:.0f} px: invisible to every "
            f"temporal statistic, untouched by temporal denoisers")
    if edge >= 2.5 and dens_p90 < 0.45 and amp_p90 >= 0.012:
        labels.append("sparse edge flicker")
        warnings.append("edge-local mosquito flicker can be weak in a smooth sigma map")
        suggestions.append("inspect the deblock map and consider edge/compression cleanup before denoise")
    if row_periodicity >= 0.50 and row_coherence >= 0.05 and dens_p90 >= 0.05 and amp_p90 >= 0.012:
        labels.append("row-coherent scanline flicker")
        warnings.append("periodic row flicker can make temporal sigma maps over-condition line artifacts")
        suggestions.append("compare against the clean/reencode baseline; try --noise-map-gain 0.6-0.8 if it over-softens")
    if trace_med > 0 and trace_max >= max(0.030, 1.55 * trace_med):
        labels.append("pulsed temporal noise")
        suggestions.append("--noise-map-pulse is likely useful")
    if lag >= 1.35 and dens_p90 >= 0.25:
        labels.append("motion-like temporal change")
        warnings.append("motion can inflate frame-difference sigma estimates")
        if static <= 0.15 and dens_med >= 0.55 and tail5_med >= 0.08:
            labels.append("source-wide motion contamination")
            warnings.append("few stable pixels are available; temporal sigma may become a motion map")
            suggestions.append("compare against a reencode-only baseline or manual constant denoise")
    if static <= 0.06 and dens_p90 >= 0.55 and tail5_p90 >= 0.045:
        labels.append("dense temporal change")
        mc_says_noise = mc_trusted and mc_sigma >= 0.030 and mc_lag21 <= 1.35
        flat_says_noise = flat_trusted and flat_sigma >= 0.030 and flat_lag21 <= 1.30
        if mc_says_noise or flat_says_noise:
            # the MC floor aligns each pair before measuring, so it reads the
            # noise level on all pixels (matches the map's default floor);
            # the flat floor is the aperture-gated second opinion
            labels.append("dense sensor noise")
            suggestions.append("--noise-map auto should track this; the "
                               + ("motion-compensated" if mc_says_noise else "flat-pixel")
                               + " floor anchors the map")
        elif mc_trusted and mc_sigma < 0.020 and mc_lag21 <= 1.35:
            # the MC floor aligned the change away: dense MOTION with a
            # confidently low noise floor -- resolved, not ambiguous
            labels.append("dense motion, low noise floor")
            suggestions.append("auto map stays conservative here; heavy denoising would eat texture")
        elif edge < 1.8 and lag <= 1.30 and (
            not flat_trusted or flat_sigma < 0.020 or flat_lag21 > 1.30
        ) and (not mc_trusted or mc_sigma < 0.020 or mc_lag21 > 1.35):
            labels.append("motion/noise ambiguous")
            warnings.append("dense texture motion can look like dense noise; strict auto maps may over-condition")
            suggestions.append("compare the debug noise map against a constant/manual denoise run")
        elif lag <= 1.35:
            warnings.append("little static baseline is available for the motion cap")
    if amp_p90 > max(0.080, 2.5 * max(rms_p90, 1e-6)) and dens_p90 < 0.35:
        labels.append("sparse flashes")

    if not labels:
        labels.append("mixed/low-confidence")
    risk = "low"
    if (any("ambiguous" in label for label in labels)
            or "source-wide motion contamination" in labels):
        risk = "high"
    elif warnings:
        risk = "medium"
    return {
        "labels": labels,
        "risk": risk,
        "warnings": warnings,
        "suggestions": suggestions,
        "metrics": {
            "density_p90": float(dens_p90),
            "amplitude_p90": float(amp_p90),
            "tail5_p90": float(tail5_p90),
            "tail5_median": float(tail5_med),
            "lag2_over_lag1": lag,
            "edge_over_flat": edge,
            "flat_sigma": flat_sigma,
            "flat_lag21": flat_lag21,
            "flat_diff_corr": flat_corr,
            "mc_sigma": mc_sigma,
            "mc_lag21": mc_lag21,
            "row_interlace": row_interlace,
            "static_row_period_px": static_row_per,
            "static_row_periodicity": static_row_str,
            "spatial_sigma": spatial_sigma,
            "static_grain_sigma": static_grain,
            "static_fraction": static,
            "static_spatial_hf": static_hf,
            "row_coherence": row_coherence,
            "row_periodicity": row_periodicity,
            "row_period_px": row_period_px,
            "trace_median": float(trace_med),
            "trace_max": float(trace_max),
        },
    }
