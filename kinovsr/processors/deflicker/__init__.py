"""Verified-static temporal state integration (codec-junk deflicker).

Starved, loop-filtered encodes carry their damage as temporally UNSTABLE
quantization states on static content: blocks sit on a wrong-but-stable
value, then jump when the encoder re-codes them (GOP pulses, coding-cadence
flicker, 1-frame re-quantization flashes). The states interleave around the
truth, so along a verified-static trajectory the artifact is separable from
real content changes, which are one-sided (before differs from after, never
interleaved).

Rule, per pixel, over a +/-K frame window:

    admit sample j only if the 32px tile around the pixel is VERIFIED
    STATIC between frames j and t (phase-correlation displacement < tau);
    take the median of admitted samples; integrate (uniform mean) the
    admitted samples within band h of that median; replace the pixel only
    if >= frac of admitted samples are in-band, both existing temporal
    sides contribute, the temporal profile is OSCILLATORY (net change well
    below total variation -- monotone smooth profiles are lighting ramps or
    true drift and must pass through), the correction field is locally
    SIGN-MIXED (spatially coherent same-sign corrections are real
    oscillating illumination -- blinking signage, headlight sweeps, AGC --
    and are vetoed), and the correction is below max_fix (refused, not
    clamped).

Why this shape (measured on ground-truth crushed fixtures; see the v1-v6
prototype history in the planning evidence):

- Verification-only alignment: samples are admitted RAW or not at all --
  nothing is ever warped, so misalignment cannot poison the mixture and
  panning/deforming content is passthrough BY CONSTRUCTION (a warped-
  sample variant lost 3 dB on a pan fixture; this one is bit-identical).
- Band centered on the window median, not the current frame: every frame
  in the window converges to the same dwell-weighted state mixture, which
  collapses state STEPS (GOP pulses), not just minority excursions, and
  recovers sub-quantization detail by integrating the encoder's
  inadvertent temporal dither.
- No spatial mixing exists anywhere in the operator: a pixel is only ever
  replaced by an average of its own trajectory samples, so the stage has
  no softening budget; pixels that do not fire pass through bit-identical.

Scope (honest, measured): fixes flicker on verified-static content only --
which is where compressed-junk flicker offends (motion masks the rest).
Tracked-moving content is left alone; integration under subpixel alignment
residual measured net-harmful every time.

Streaming: K frames of lookahead (feed returns frame t once t+K arrives;
flush drains the tail). Frame dimensions are processed on the /16-cropped
interior for the validity grid; any bottom/right remainder margin simply
never fires.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import mlx.core as mx

from kinovsr.analysis.noise.estimate import _to_luma_2d
from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.processors.capabilities import (
    Capability,
    CapabilitySpec,
    TemporalMode,
)
from kinovsr.processors.feed_driver import FeedFlushProcessor
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Domain,
    DType,
    Layout,
    StreamConstraint,
    StreamSpec,
)
from kinovsr.settings import Settings

_PI = 3.141592653589793


class StaticStateDeflicker:
    """RGB-in / RGB-out streaming deflicker; K frames of lookahead."""

    BLOCK = 32          # phase-correlation verification tile
    STRIDE = 16         # tile stride (half-overlapped grid)

    def __init__(self, window: int = 8, band: float = 0.10,
                 frac: float = 0.5, max_fix: float = 0.25,
                 tau: float = 0.75, min_valid: int = 6,
                 strength: float = 1.0, jitter: bool = False,
                 jitter_max: float = 3.0, illum_veto: bool = True,
                 gop: bool = True):
        # window: +/-K frames integrated (latency = K).
        # band: luma half-width around the window median that counts as the
        #   same quantization-state cluster; the reference regime's state
        #   steps read 0.05-0.15.
        # frac: fraction of admitted samples that must sit in-band.
        # max_fix: corrections larger than this are refused (safety valve;
        #   genuine state flicker sits well below it).
        # tau: max phase-correlation displacement (px) for a tile pair to
        #   count as verified static.
        # min_valid: minimum admitted samples (incl. t) before acting.
        # jitter: compensate GLOBAL camera micro-jitter before verification:
        #   the median per-tile displacement of a pair is treated as camera
        #   shift; tiles are verified against the RESIDUAL and admitted
        #   samples are aligned by ONE whole-frame Fourier shift. This is
        #   not the per-tile warping that failed (median-of-tiles is robust
        #   to moving subjects, a single global translation cannot print
        #   seams, and Fourier translation is all-pass). Pairs whose global
        #   shift exceeds jitter_max px are real camera motion, not jitter,
        #   and fall back to the strict static rule; the wrap-contaminated
        #   border strip is invalidated per pair.
        self._k = max(1, int(window))
        self._band = float(band)
        self._frac = float(frac)
        self._max_fix = float(max_fix)
        self._tau = float(tau)
        self._min_valid = float(min_valid)
        self._strength = float(strength)
        self._jitter = bool(jitter)
        self._jitter_max = float(jitter_max)
        # illum_veto: refuse spatially sign-coherent corrections (real
        # oscillating illumination). Disable only on material with no real
        # light dynamics (e.g. archival scans) where every oscillation is
        # junk.
        self._illum_veto = bool(illum_veto)
        # gop: sync-keyed pumping rescue. The oscillatory gate refuses a
        # single state STEP (net ~ tv reads as a lighting step at window
        # scale), so a window containing exactly one GOP-boundary reset -
        # the I-frame pop, the most legible pumping artifact - never fires
        # when the window is shorter than the GOP. Frames that arrive with
        # sync flags say exactly where coding state resets: discounting
        # only the sync-pair deltas from the profile and re-testing makes
        # a flat -> step-at-I -> flat profile fixable while a real ramp
        # stays refused (its residual is still monotone after removing one
        # pair). Real light steps that happen to land on an I-frame are
        # still caught downstream by the sign-coherence veto.
        self._gop = bool(gop)
        # gate attribution (run averages); lives OUTSIDE _reset_state so it
        # survives flush() and cut-boundary resets -- it describes the run,
        # not the buffer
        self._stat_frames = 0
        self._stat_fired = 0.0
        self._stat_verified = 0.0
        self._stat_osc = 0.0
        self._stat_applied = 0.0
        self._stat_jit_px = 0.0
        self._stat_jit_pairs = 0
        self._stat_gop_rescued = 0.0

        self._reset_state()

    def _reset_state(self) -> None:
        # [(rgb (H,W,3) fp32, luma, token, is_sync)]
        self._buf: list = []
        self._base = 0                   # index of _buf[0]
        self._received = 0
        self._emitted = 0
        self._masks: dict = {}           # {(lo, hi): (H,W) static mask}

    def reset(self) -> None:
        self._reset_state()

    def stats(self) -> dict:
        """Run-average gate attribution: where firing dies on this clip.

        verified = pixels whose 32px tiles phase-correlate static against
        enough window neighbors (the tool's scope: this is bounded by real
        camera/subject motion, NOT by band or window -- micro-jitter
        accumulates with temporal distance, so far pairs fail verification
        and windows past the jitter horizon add nothing). oscillatory =
        verified pixels whose temporal profile is non-monotone (the
        fixable kind). fired = pixels actually corrected. applied = mean
        absolute luma correction per frame: the needle that moves with
        band/strength when the gate fractions do not (band changes WHAT
        the mixture averages far more than WHETHER pixels fire).
        """
        n = max(self._stat_frames, 1)
        return {"fired": self._stat_fired / n,
                "verified": self._stat_verified / n,
                "oscillatory": self._stat_osc / n,
                "applied": self._stat_applied / n,
                "gop_rescued": self._stat_gop_rescued / n,
                "jitter_px": (self._stat_jit_px / self._stat_jit_pairs
                              if self._stat_jit_pairs else 0.0)}

    def run_diagnostics(self) -> list:
        st = self.stats()
        jit = (f", compensated jitter avg {st['jitter_px']:.2f}px"
               if st["jitter_px"] else "")
        gop = (f", gop-rescued {st['gop_rescued'] * 100:.1f}%"
               if st["gop_rescued"] else "")
        return [
            f"[deflicker] run avg: static-verified "
            f"{st['verified'] * 100:.1f}% of pixels, oscillatory "
            f"{st['oscillatory'] * 100:.1f}%{gop}, fired "
            f"{st['fired'] * 100:.1f}%, applied "
            f"{st['applied'] * 1000:.2f}e-3 luma{jit} "
            f"(verification is the scope gate: bounded by "
            f"camera/subject motion, not band/window)"]

    def close(self) -> None:
        pass

    # ---- static verification ---------------------------------------------

    def _static_mask(self, ref_l: Any, src_l: Any) -> tuple:
        """((H,W) 0/1 mask, gy, gx): mask is 1 where every covering 32px
        tile is static; (gy, gx) is the global (camera) shift of src
        relative to ref -- the median per-tile displacement -- or (0, 0)
        when jitter compensation is off or the shift exceeds jitter_max.
        In jitter mode tiles verify against the RESIDUAL displacement."""
        H, W = int(ref_l.shape[0]), int(ref_l.shape[1])
        B, S = self.BLOCK, self.STRIDE
        Hc, Wc = (H // S) * S, (W // S) * S
        if Hc < B or Wc < B:
            return mx.zeros((H, W), dtype=mx.float32), 0.0, 0.0
        ny, nx = Hc // S - 1, Wc // S - 1        # regular tile origins
        rl = ref_l[:Hc, :Wc]
        sl = src_l[:Hc, :Wc]
        # (ny*nx, B, B) tile stacks via strided reshape: tile (i,j) spans
        # rows i*S..i*S+B = cells (i, i+1), cols likewise
        rc = rl.reshape(ny + 1, S, nx + 1, S).transpose(0, 2, 1, 3)
        sc = sl.reshape(ny + 1, S, nx + 1, S).transpose(0, 2, 1, 3)
        rt = mx.concatenate([mx.concatenate(
            [rc[:ny, :nx], rc[1:, :nx]], axis=2), mx.concatenate(
            [rc[:ny, 1:], rc[1:, 1:]], axis=2)], axis=3).reshape(ny * nx, B, B)
        st = mx.concatenate([mx.concatenate(
            [sc[:ny, :nx], sc[1:, :nx]], axis=2), mx.concatenate(
            [sc[:ny, 1:], sc[1:, 1:]], axis=2)], axis=3).reshape(ny * nx, B, B)
        n1 = mx.arange(B).astype(mx.float32)
        w1 = 0.5 - 0.5 * mx.cos(2.0 * _PI * n1 / (B - 1))
        win = (w1[:, None] * w1[None, :])[None]
        R = mx.fft.rfft2(rt * win) * mx.conj(mx.fft.rfft2(st * win))
        R = R / (mx.abs(R) + 1e-9)
        corr = mx.fft.irfft2(R, s=(B, B)).reshape(ny * nx, B * B)
        peak = mx.argmax(corr, axis=-1)
        py = (peak // B).astype(mx.float32)
        px = (peak % B).astype(mx.float32)
        dy = mx.where(py > B / 2, py - B, py)
        dx = mx.where(px > B / 2, px - B, px)
        gy = gx = 0.0
        if self._jitter:
            nt = int(dy.shape[0])
            gy = float(mx.sort(dy)[nt // 2])
            gx = float(mx.sort(dx)[nt // 2])
            if max(abs(gy), abs(gx)) > self._jitter_max or \
                    max(abs(gy), abs(gx)) < 0.2:
                # beyond the cap it is real camera motion (strict rule);
                # below 0.2 px compensation is not worth a resample
                gy = gx = 0.0
            else:
                # dominance rule: starved encoders shred camera shake into
                # piecewise skips and snaps, so the decoded displacement
                # field is often NOT a global translation -- half the frame
                # sits at 0 while coded regions snapped. Compensate only
                # when it verifies MORE tiles than not compensating, so
                # jitter mode can never do worse than the strict rule.
                n_g = float(mx.sum(((mx.abs(dy - gy) < self._tau)
                                    & (mx.abs(dx - gx) < self._tau))
                                   .astype(mx.float32)))
                n_0 = float(mx.sum(((mx.abs(dy) < self._tau)
                                    & (mx.abs(dx) < self._tau))
                                   .astype(mx.float32)))
                if n_0 >= n_g:
                    gy = gx = 0.0
        ok = ((mx.abs(dy - gy) < self._tau)
              & (mx.abs(dx - gx) < self._tau)).astype(mx.float32).reshape(ny, nx)
        # cell verdict = AND of covering tiles; cell grid is (ny+1, nx+1),
        # cell (i,j) covered by tile origins {i-1, i} x {j-1, j}
        okp = mx.pad(ok, ((1, 1), (1, 1)), constant_values=1.0)
        cells = mx.minimum(okp[:-1, :], okp[1:, :])
        cells = mx.minimum(cells[:, :-1], cells[:, 1:])     # (ny+1, nx+1)
        m = mx.broadcast_to(cells[:, None, :, None],
                            (ny + 1, S, nx + 1, S)).reshape(Hc, Wc)
        if Hc < H or Wc < W:
            m = mx.pad(m, ((0, H - Hc), (0, W - Wc)))
        if gy or gx:
            # the Fourier shift wraps content around the frame edge: kill
            # validity in the contaminated border strip for this pair
            s = int(max(abs(gy), abs(gx))) + 1
            inner = mx.ones((H - 2 * s, W - 2 * s), dtype=mx.float32)
            m = mx.minimum(m, mx.pad(inner, ((s, s), (s, s))))
        return m, gy, gx

    def _pair_mask(self, a: int, b: int) -> tuple:
        """Cached (mask, gy, gx) for the unordered pair; (gy, gx) aligns
        the HIGHER-index frame onto the lower one (negate to go the other
        way)."""
        key = (a, b) if a < b else (b, a)
        got = self._masks.get(key)
        if got is None:
            got = self._static_mask(self._buf[key[0] - self._base][1],
                                    self._buf[key[1] - self._base][1])
            mx.eval(got[0])
            self._masks[key] = got
            if got[1] or got[2]:
                self._stat_jit_px += max(abs(got[1]), abs(got[2]))
                self._stat_jit_pairs += 1
        return got

    def _warp_global(self, rgb: Any, dy: float, dx: float) -> Any:
        """Whole-frame Fourier translation of (H,W,3) by (dy, dx) px."""
        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        ky = mx.concatenate([mx.arange(H // 2 + 1),
                             mx.arange(-(H - H // 2 - 1), 0)]).astype(mx.float32)
        kx = mx.arange(W // 2 + 1).astype(mx.float32)
        ang = (-2.0 * _PI) * (ky[:, None] * (dy / H) + kx[None, :] * (dx / W))
        ramp = (mx.cos(ang) + 1j * mx.sin(ang))[None]
        chan = mx.transpose(rgb, (2, 0, 1))
        out = mx.fft.irfft2(mx.fft.rfft2(chan) * ramp, s=(H, W))
        return mx.transpose(out, (1, 2, 0))

    # ---- streaming ---------------------------------------------------------

    def _emit_one(self, last: int) -> tuple:
        t = self._emitted
        cur, cur_l, tok, _cur_sync = self._buf[t - self._base]
        lo, hi = max(self._base, t - self._k), min(last, t + self._k)
        idx = list(range(lo, hi + 1))
        if len(idx) < 2:
            self._emitted += 1
            return cur, tok
        Ss, SLs, Vs = [], [], []
        for j in idx:
            if j == t:
                Ss.append(cur)
                SLs.append(cur_l)
                Vs.append(mx.ones(cur_l.shape, dtype=mx.float32))
                continue
            m, gy, gx = self._pair_mask(t, j)
            if gy or gx:
                # cached shift aligns the higher-index frame onto the lower
                sgn = 1.0 if j > t else -1.0
                w = mx.clip(self._warp_global(self._buf[j - self._base][0],
                                              sgn * gy, sgn * gx), 0.0, 1.0)
                Ss.append(w)
                SLs.append(_to_luma_2d(w))
            else:
                rgbj, lj, _tok, _sync = self._buf[j - self._base]
                Ss.append(rgbj)
                SLs.append(lj)
            Vs.append(m)
        S = mx.stack(Ss, axis=0)
        SL = mx.stack(SLs, axis=0)
        V = mx.stack(Vs, axis=0)
        M = len(idx)
        n_valid = mx.sum(V, axis=0)
        # monotone-profile gate: lighting/exposure ramps and slow true drift
        # are monotone and smooth, so their net change ~ total variation;
        # fixable codec flicker is oscillatory (TV >> NET). A trend FIT
        # cannot make this separation -- a monotone step is a valid trend at
        # window scale and the fit absorbs the staircase (measured: both LS
        # and Theil-Sen detrending lose the flicker kill entirely) -- but
        # the profile shape can. Without this gate the raw mixture flattens
        # ramps wherever it fires, and since firing follows the 16px
        # validity cells the damage prints square seams on lighting changes
        # (user-reported); with it, ramp regions produce no correction at
        # any band setting.
        d_signed = (SL[1:] - SL[:-1]) * (V[1:] * V[:-1])
        dj = mx.abs(d_signed)
        tv = mx.sum(dj, axis=0)
        first_i = mx.argmax(V, axis=0).astype(mx.int32)[None]
        last_i = (M - 1 - mx.argmax(V[::-1], axis=0)).astype(mx.int32)[None]
        y_first = mx.take_along_axis(SL, first_i, axis=0)[0]
        y_last = mx.take_along_axis(SL, last_i, axis=0)[0]
        net = mx.abs(y_last - y_first)
        oscillatory = net <= 0.6 * tv + 1.5 / 255.0
        gop_rescued = None
        if self._gop:
            # Sync-keyed pumping rescue: a window holding exactly one
            # GOP-boundary state reset reads as a monotone step (net ~ tv)
            # and the base gate refuses it - correctly, for an unexplained
            # step, but this step sits exactly where the coded stream
            # declares a coding restart. Discount ONLY the deltas of pairs
            # that straddle a sync sample and re-test: a flat-step-flat
            # pump profile becomes oscillatory (residual net ~ 0), while a
            # real ramp stays refused (its residual is still monotone
            # after removing one pair). A genuine light step landing on an
            # I-frame is still vetoed downstream by sign-coherence.
            sync_pairs = [
                1.0 if self._buf[idx[p + 1] - self._base][3] else 0.0
                for p in range(M - 1)
            ]
            if any(sync_pairs):
                s = mx.array(sync_pairs, dtype=mx.float32).reshape(
                    M - 1, 1, 1)
                sync_net = mx.sum(d_signed * s, axis=0)
                tv_rest = tv - mx.sum(dj * s, axis=0)
                osc_gop = (mx.abs((y_last - y_first) - sync_net)
                           <= 0.6 * tv_rest + 1.5 / 255.0)
                gop_rescued = osc_gop & ~oscillatory
                oscillatory = oscillatory | osc_gop
        # median of admitted samples: invalid sort past the [0,1] range
        SL_m = mx.where(V > 0.5, SL, mx.full(SL.shape, 2.0))
        SL_s = mx.sort(SL_m, axis=0)
        mi = mx.maximum((n_valid - 1) // 2, 0).astype(mx.int32)[None]
        wmed = mx.take_along_axis(SL_s, mi, axis=0)[0]
        inl = ((mx.abs(SL - wmed[None]) < self._band)
               & (V > 0.5)).astype(mx.float32)
        n_inl = mx.sum(inl, axis=0)
        rel = mx.arange(lo - t, hi - t + 1).astype(mx.float32).reshape(M, 1, 1)
        left = mx.sum(inl * (rel < 0), axis=0)
        right = mx.sum(inl * (rel > 0), axis=0)
        need_l = 1.0 if lo < t else 0.0
        need_r = 1.0 if hi > t else 0.0
        wsum = mx.maximum(n_inl, 1e-6)
        mean_w = mx.sum(inl[..., None] * S, axis=0) / wsum[..., None]
        cur_dev = mx.abs(_to_luma_2d(mean_w) - cur_l)
        fire = ((n_inl >= self._frac * mx.maximum(n_valid, 1.0))
                & (n_valid >= self._min_valid)
                & (left >= need_l) & (right >= need_r)
                & oscillatory
                & (cur_dev < self._max_fix)).astype(mx.float32)[..., None]
        # illumination veto: real oscillating light (blinking signage,
        # headlight sweeps, AGC pumping) is per-pixel indistinguishable
        # from codec pulsing -- both are oscillatory luminance on static
        # geometry -- but its CORRECTION field is locally same-signed and
        # smooth (a light pool dims coherently), while codec block
        # re-rolls are sign-mixed at block scale. Where the local mean
        # correction keeps most of the local mean magnitude, the tool is
        # about to flatten real light: refuse. Without this the
        # flattening follows the 16px verification cells and prints
        # flickering boxes wherever lighting oscillates (user-reported).
        # the coherence window must sit BETWEEN the scales it separates:
        # larger than a codec block (8-16px, sign-constant within), smaller
        # than an illumination pool (typically 50px+); 17px reads sign
        # mixture across 2+ blocks while still seeing a light pool as
        # uniformly signed
        corr_l = (_to_luma_2d(mean_w) - cur_l)[None, :, :, None] * fire[None]
        k17 = mx.ones((1, 17, 17, 1), dtype=mx.float32) / 289.0
        v_num = mx.conv2d(corr_l, k17, padding=8)[0, :, :, 0]
        v_den = mx.conv2d(mx.abs(corr_l), k17, padding=8)[0, :, :, 0]
        coher = mx.abs(v_num) / mx.maximum(v_den, 1e-6)
        veto = ((coher > 0.65) & (v_den > 1.0 / 255.0))
        if self._illum_veto:
            fire = fire * (1.0 - veto.astype(mx.float32))[..., None]
        out = cur + (self._strength * fire) * (mean_w - cur)
        # anomalous-self fallback: a large 1-frame flash invalidates every
        # (t, j) verification in its own tiles (n_valid collapses to 1), so
        # the main path refuses exactly where the flash is. If the BRACKET
        # pair (t-1, t+1) verifies static against each other and agrees
        # in-band at this pixel, the current frame is a one-frame anomaly in
        # a static context: replace with the bracket mean. A 2-frame flash
        # leaves one bracket frame contaminated (pair fails or disagrees),
        # so persistent content is safe.
        if t - 1 >= self._base and t + 1 <= last:
            pv, pv_l, _, _ = self._buf[t - 1 - self._base]
            nx_, nx_l, _, _ = self._buf[t + 1 - self._base]
            # bracket values stay raw: adjacent-pair jitter is sub-tau and
            # the agree/anom gates refuse where a residual shift matters
            br_ok, _bgy, _bgx = self._pair_mask(t - 1, t + 1)
            agree = mx.abs(pv_l - nx_l) < self._band
            anom = mx.abs(cur_l - 0.5 * (pv_l + nx_l)) > self._band
            fixable = mx.abs(cur_l - 0.5 * (pv_l + nx_l)) < self._max_fix
            # serves two refusal modes of the main path: self-invalidating
            # large flashes (n_valid collapses) and coherence-vetoed sparse
            # same-sign flash blocks -- the bracket rule is itself safe
            # (static bracket pair + in-band agreement), and a genuine
            # 1-frame light strobe falling to it is the documented 1-frame
            # despot trade
            fire2 = ((fire[..., 0] < 0.5) & ((n_valid <= 2.0) | veto)
                     & (br_ok > 0.5)
                     & agree & anom & fixable).astype(mx.float32)[..., None]
            out = out + (self._strength * fire2) * (0.5 * (pv + nx_) - out)
            fire = mx.minimum(fire + fire2, 1.0)
        out = mx.clip(out, 0.0, 1.0)
        verified = (n_valid >= self._min_valid)
        rescued = (mx.mean((gop_rescued & verified).astype(mx.float32))
                   if gop_rescued is not None
                   else mx.zeros((), dtype=mx.float32))
        stat = mx.stack([mx.mean(fire), mx.mean(verified.astype(mx.float32)),
                         mx.mean((verified & oscillatory).astype(mx.float32)),
                         mx.mean(mx.abs(_to_luma_2d(out) - cur_l)),
                         rescued])
        mx.eval(out, stat)
        self._stat_frames += 1
        self._stat_fired += float(stat[0])
        self._stat_verified += float(stat[1])
        self._stat_osc += float(stat[2])
        self._stat_applied += float(stat[3])
        self._stat_gop_rescued += float(stat[4])
        self._emitted += 1
        keep = self._emitted - self._k
        while self._base < keep and self._buf:
            self._buf.pop(0)
            self._base += 1
        gone = self._base
        self._masks = {k: v for k, v in self._masks.items() if k[0] >= gone}
        return out, tok

    def feed(self, rgb: Any, token: Any = None) -> list:
        a = rgb[0] if rgb.ndim == 4 else rgb
        a = mx.clip(a[..., :3].astype(mx.float32), 0.0, 1.0)
        source = getattr(token, "source", None)
        is_sync = bool(getattr(source, "is_sync", None) or False)
        self._buf.append((a, _to_luma_2d(a), token, is_sync))
        self._received += 1
        last = self._received - 1
        ready = []
        while last - self._emitted >= self._k:
            ready.append(self._emit_one(last))
        return ready

    def flush(self) -> list:
        last = self._received - 1
        out = []
        while self._emitted <= last:
            out.append(self._emit_one(last))
        self._reset_state()
        return out


# ===========================================================================
# Processor family: self-buffered flicker suppression
# ===========================================================================

@dataclasses.dataclass(frozen=True, slots=True)
class DeflickerStageConfig:
    window: int
    band: float
    frac: float
    max_fix: float
    jitter: bool
    strength: float
    gop: bool


def _passthrough(spec: StreamSpec, config: object) -> StreamSpec:
    return spec


class DeflickerFactory:
    name = "deflicker"

    # CENTERED per the taxonomy: the +/-K integration window's future
    # half is self-buffered and paid as K frames of output delay - no
    # source lookahead demanded. temporal_radius records the default
    # delay; the configured window governs the driver.
    capabilities = {
        Capability.PREPROCESS: CapabilitySpec(
            capability=Capability.PREPROCESS,
            profiles=(),
            accepts=StreamConstraint(
                layouts=(Layout.MLX_RGB_HWC,),
                dtypes=(DType.FLOAT32,),
                domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
            ),
            produces=_passthrough,
            temporal_mode=TemporalMode.CENTERED,
            temporal_radius=8,
            stateful=True,
        ),
    }

    def parse_config(
        self,
        raw: Mapping[str, object],
        *,
        capability: Capability,
        profile: str | None,
        settings: Settings,
    ) -> DeflickerStageConfig:
        reject_unknown_keys(
            raw, ("window", "band", "frac", "max_fix", "jitter", "strength",
                  "gop"))
        window = typed_value(raw, "window", int, 8)
        if window < 1:
            raise ValueError("window must be >= 1")
        band = typed_value(raw, "band", float, 0.1)
        frac = typed_value(raw, "frac", float, 0.5)
        max_fix = typed_value(raw, "max_fix", float, 0.25)
        jitter = typed_value(raw, "jitter", bool, False)
        strength = typed_value(raw, "strength", float, 1.0)
        gop = typed_value(raw, "gop", bool, True)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        return DeflickerStageConfig(
            window=window, band=band, frac=frac, max_fix=max_fix,
            jitter=jitter, strength=strength, gop=gop)

    def build(self, config: DeflickerStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> StaticStateDeflicker:
            return StaticStateDeflicker(
                window=config.window, band=config.band, frac=config.frac,
                max_fix=config.max_fix, jitter=config.jitter,
                strength=config.strength, gop=config.gop)

        return FeedFlushProcessor(make_driver)


FACTORY = DeflickerFactory()
