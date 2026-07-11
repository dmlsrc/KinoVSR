"""DCT-coefficient comb detection: measure JPEG-family quantization from pixels.

JPEG (and MJPEG / MPEG-2 intra) quantizes 8x8 DCT coefficients with integer
steps, so decoded frames carry coefficient distributions that are combs at
multiples of each position's step. Detecting the comb recovers the step per
frequency position; voting the steps against libjpeg's scaled standard
luminance table recovers the JPEG quality factor (QF). The measurement is a
physical readout of how hard the content was quantized -- it drives FBCNN's
QF conditioning (replacing a hand-pinned value) and can scale deblock
strength. It survives one later H.264 re-encode at high quality (verified at
crf 18) and declines honestly on content with no JPEG history.

Method notes (each learned the hard way):
- Comb strength is measured in FREQUENCY domain as the SIGNED real part of
  the characteristic function E[cos(2*pi*f*c)] over nonzero coefficients: a
  comb of step D peaks exactly at f = m/D. The magnitude |CF| would also
  peak at half-frequencies for +-D-dominated distributions and rises
  monotonically toward large steps, which rails a step-domain sweep.
- The distribution envelope is smooth in f, so a moving-median baseline is
  subtracted and peaks are accepted by PROMINENCE.
- The fundamental is the lowest-f peak within 50% of the strongest peak.
  Accepting any lower-f peak "confirmed by its harmonic" endorses octave-down
  errors: a spurious half-frequency candidate always finds the true peak at
  its 2f.
- JPEG QF is table-equivalent in bands at high quality (e.g. 71..75 share
  the low-frequency table entries), so estimates are exact only up to that
  inherent resolution.
- Per-tile estimation needs temporal accumulation: 128 px tiles over a
  ~12-frame window are reliable; 96 px starts declining exactly on heavily
  compressed regions (coarse quantization zeroes most AC samples), and
  single-frame tiles decline at any size.

Public API:
  estimate_qf(frames)                  -> global QF dict for a frame window
  estimate_qf_map(frames, tile=128)    -> per-tile QF grid + filled float grid
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from .noise.estimate import _to_luma_2d


def _lum(frame: Any) -> Any:
    return frame.astype(mx.float32) if frame.ndim == 2 else _to_luma_2d(frame)

# standard JPEG luminance quantization table (Annex K), row-major
_T_STD = (
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 35, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99),
)

# low-frequency AC positions: usable energy, small steps, early zigzag order
_POSITIONS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0),
              (1, 2), (2, 1), (2, 2), (0, 3), (3, 0))

_D_MAX = 64            # largest detectable quantization step
_SUBSAMPLE = 20000     # CF sample cap per position


def jpeg_quality_table(q: int) -> list:
    """libjpeg-scaled luminance table for quality q (list of 8 rows)."""
    scale = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    return [[min(255.0, max(1.0, float(int((t * scale + 50.0) // 100.0))))
             for t in row] for row in _T_STD]


def _dct_basis() -> Any:
    n = mx.arange(8).astype(mx.float32)
    k = mx.arange(8).astype(mx.float32)
    c = mx.cos((2.0 * n[None, :] + 1.0) * k[:, None] * (3.141592653589793 / 16.0))
    s = mx.concatenate([mx.full((1, 8), (1.0 / 8.0) ** 0.5),
                        mx.full((7, 8), (2.0 / 8.0) ** 0.5)], axis=0)
    return c * s        # orthonormal 8-point DCT-II rows; matches JPEG scale


_C8 = _dct_basis()


def _block_dct(luma01: Any) -> tuple[Any, int, int]:
    """(H,W) luma in [0,1] -> ((nby*nbx, 8, 8) coefficients, nby, nbx)."""
    H, W = int(luma01.shape[0]), int(luma01.shape[1])
    h8, w8 = H // 8 * 8, W // 8 * 8
    x = luma01[:h8, :w8].astype(mx.float32) * 255.0 - 128.0
    b = x.reshape(h8 // 8, 8, w8 // 8, 8)
    b = mx.transpose(b, (0, 2, 1, 3)).reshape(-1, 8, 8)
    return mx.matmul(_C8[None], mx.matmul(b, mx.transpose(_C8)[None])), h8 // 8, w8 // 8


def _comb_step(samples: Any, min_samples: int) -> tuple[float, float]:
    """(fundamental step, prominence) from nonzero coefficients; (0,.) = none."""
    n = int(samples.shape[0])
    if n < min_samples:
        return 0.0, 0.0
    if n > _SUBSAMPLE:
        samples = samples[:: max(1, n // _SUBSAMPLE)][:_SUBSAMPLE]
    nf = 480
    f = mx.linspace(1.0 / _D_MAX, 0.55, nf)
    ph = (2.0 * 3.141592653589793) * f[:, None] * samples[None, :]
    s = mx.mean(mx.cos(ph), axis=1)
    sl = [float(v) for v in s.tolist()]
    w = 31
    base = []
    for i in range(nf):
        lo, hi = max(0, i - w), min(nf, i + w + 1)
        seg = sorted(sl[lo:hi])
        base.append(seg[len(seg) // 2])
    prom = [v - b for v, b in zip(sl, base, strict=True)]
    fl = [float(v) for v in f.tolist()]
    peaks = [i for i in range(2, nf - 2)
             if sl[i] >= sl[i - 1] and sl[i] >= sl[i + 1]
             and sl[i] >= sl[i - 2] and sl[i] >= sl[i + 2]
             and prom[i] >= 0.05]
    if not peaks:
        return 0.0, max(prom) if prom else 0.0
    pmax = max(prom[i] for i in peaks)
    strong = [i for i in peaks if prom[i] >= 0.5 * pmax]
    i = min(strong, key=lambda j: fl[j])
    return 1.0 / fl[i], prom[i]


def _steps_for(coeffs: Any, min_samples: int) -> tuple[dict, list]:
    steps: dict = {}
    contrasts: list = []
    for (u, v) in _POSITIONS:
        c = coeffs[:, u, v]
        # counted-sort nonzero filter (no boolean indexing in MLX); the CF is
        # order-independent, so the |c| >= 1 subset via a magnitude sort works
        nz = int(mx.sum((mx.abs(c) >= 1.0).astype(mx.float32)))
        if nz < min_samples:
            continue
        order = mx.argsort(-mx.abs(c))
        d, contrast = _comb_step(mx.take(c, order[:nz]), min_samples)
        if d > 0.0:
            steps[(u, v)] = d
            contrasts.append(contrast)
    return steps, contrasts


def _fit_qf(steps: dict) -> tuple[int | None, int]:
    if len(steps) < 3:
        return None, 0
    best_q, best_hits = None, -1
    for q in range(5, 99):
        t = jpeg_quality_table(q)
        hits = sum(1 for (u, v), d in steps.items()
                   if abs(d - t[u][v]) <= max(1.0, 0.08 * t[u][v]))
        if hits > best_hits:
            best_hits, best_q = hits, q
    if best_hits < 3:
        return None, best_hits
    return best_q, best_hits


def estimate_qf(frames: list, min_samples: int = 800) -> dict:
    """Global JPEG QF estimate over a window of frames.

    frames: list of (H,W)/(H,W,C)/(1,H,W,C) arrays in [0,1].
    Returns {"qf": int|None, "confidence": float, "positions": int,
    "hits": int, "steps": {"uv": step}}; qf None = no comb detected (content
    has no JPEG-family history the detector can see -- do not force a value).
    """
    coeffs = mx.concatenate(
        [_block_dct(_lum(f))[0] for f in frames], axis=0)
    steps, contrasts = _steps_for(coeffs, min_samples)
    qf, hits = _fit_qf(steps)
    conf = 0.0
    if qf is not None and contrasts:
        conf = (sum(contrasts) / len(contrasts)) * (hits / max(1, len(steps)))
    return {"qf": qf, "confidence": round(float(conf), 3),
            "positions": len(steps), "hits": hits,
            "steps": {f"{u}{v}": round(d, 2) for (u, v), d in steps.items()}}


def estimate_qf_map(frames: list, tile: int = 128, min_samples: int = 500,
                    fallback: float = 50.0) -> dict:
    """Per-tile QF over a window of frames, with declined tiles filled.

    Measurement tiles of `tile` px (>= 128 recommended; needs a multi-frame
    window). Declined tiles fill from the global estimate, then `fallback`.
    Returns {"qf_grid": (ty,tx) fp32 mx array (filled), "valid": (ty,tx) bool
    list grid, "tile": tile, "global": <estimate_qf dict>, "coverage": float}.
    """
    lum = [_lum(f) for f in frames]
    H, W = int(lum[0].shape[0]), int(lum[0].shape[1])
    ty, tx = max(1, H // tile), max(1, W // tile)
    per_frame = [_block_dct(f) for f in lum]
    coeffs = mx.concatenate([c for c, _, _ in per_frame], axis=0)
    nby, nbx = per_frame[0][1], per_frame[0][2]
    nb_frame = nby * nbx
    T = len(lum)

    gsteps, gcontrasts = _steps_for(coeffs, max(min_samples, 800))
    gqf, ghits = _fit_qf(gsteps)
    gconf = 0.0
    if gqf is not None and gcontrasts:
        gconf = (sum(gcontrasts) / len(gcontrasts)) * (ghits / max(1, len(gsteps)))

    grid = [[None] * tx for _ in range(ty)]
    bt = tile // 8
    for i in range(ty):
        r0, r1 = i * bt, min(nby, (i + 1) * bt) if i < ty - 1 else nby
        for j in range(tx):
            c0, c1 = j * bt, min(nbx, (j + 1) * bt) if j < tx - 1 else nbx
            idx = [t * nb_frame + r * nbx + c
                   for t in range(T) for r in range(r0, r1) for c in range(c0, c1)]
            sel = mx.take(coeffs, mx.array(idx, dtype=mx.int32), axis=0)
            steps, _ = _steps_for(sel, min_samples)
            q, _hits = _fit_qf(steps)
            grid[i][j] = q
    n_valid = sum(1 for row in grid for q in row if q is not None)
    fill = float(gqf) if gqf is not None else float(fallback)
    filled = mx.array([[float(q) if q is not None else fill for q in row]
                       for row in grid], dtype=mx.float32)
    return {"qf_grid": filled,
            "valid": [[q is not None for q in row] for row in grid],
            "tile": tile,
            "global": {"qf": gqf, "confidence": round(float(gconf), 3)},
            "coverage": round(n_valid / (ty * tx), 3)}


__all__ = ["estimate_qf", "estimate_qf_map", "jpeg_quality_table"]
