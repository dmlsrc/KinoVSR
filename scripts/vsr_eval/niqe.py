#!/usr/bin/env python3
"""NIQE: no-reference perceptual quality (natural-scene statistics).

Mittal, Soundararajan, Bovik -- "Making a 'Completely Blind' Image Quality
Analyzer" (IEEE SPL 2013), reimplemented from the paper. Lower is better.

NIQE scores an image by the Mahalanobis-style distance between the
multivariate-Gaussian statistics of its MSCN-domain features and those of a
PRISTINE corpus. No opinion training, no network, no downloads -- but it
needs the pristine model, which `--fit` builds locally from a folder of
clean images (the REDS sharp frames serve). The model ships beside this
script once fitted.

Scope notes for restoration eval, MEASURED on this workspace's fixtures:
NIQE rewards natural sharpness statistics, so over-smoothing scores WORSE
(blur sigma 1.0/2.5 scored 8.6/11.3 vs pristine 1.7 -- the opposite bias
to PSNR) -- but it reads codec junk (ringing, mosquito, blocking energy)
as TEXTURE: jpeg q20 scored a mild 2.6, and a truth-verified-better
deblock output scored WORSE than its crushed input (5.68 vs 5.31). USE IT
AS AN OVER-SMOOTHING TRIPWIRE, NOT AS AN OBJECTIVE for deblock tuning: a
recipe whose NIQE leaps toward blur-range values has been overcooked, but
ranking deblock recipes by NIQE selects for doing nothing. It is also
per-frame spatial only: pair it with a temporal-stability measure and a
fidelity anchor, and let opinion-trained metrics (MUSIQ / CLIP-IQA /
DOVER, downloads required) or eyes make the quality call.

Usage:
  fit:    niqe.py --fit <folder-of-pristine-images> [--out model.npz]
  score:  niqe.py <video-or-image> [...] [--model model.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import gamma as _gamma

MODEL_PATH = Path(__file__).resolve().parent / "weights" / "niqe_pristine_reds.npz"
PATCH = 96


def _mscn(luma: np.ndarray) -> np.ndarray:
    mu = gaussian_filter(luma, 7.0 / 6.0, truncate=3.0)
    sigma = np.sqrt(np.maximum(
        gaussian_filter(luma * luma, 7.0 / 6.0, truncate=3.0) - mu * mu, 0.0))
    return (luma - mu) / (sigma + 1.0)


_GAM = np.arange(0.2, 10.001, 0.001)
_R_GAM = (_gamma(1.0 / _GAM) * _gamma(3.0 / _GAM)) / (_gamma(2.0 / _GAM) ** 2)


def _aggd(x: np.ndarray) -> tuple[float, float, float]:
    """Asymmetric generalized Gaussian fit -> (alpha, sigma_l, sigma_r)."""
    left = x[x < 0]
    right = x[x >= 0]
    sl = np.sqrt(np.mean(left * left)) if left.size else 1e-6
    sr = np.sqrt(np.mean(right * right)) if right.size else 1e-6
    gh = sl / sr if sr > 0 else 1.0
    m1 = np.mean(np.abs(x))
    m2 = np.mean(x * x)
    rh = (m1 * m1) / m2 if m2 > 0 else 1e-6
    rhat = rh * (gh ** 3 + 1.0) * (gh + 1.0) / ((gh * gh + 1.0) ** 2)
    alpha = _GAM[int(np.argmin((_R_GAM - rhat) ** 2))]
    return float(alpha), float(sl), float(sr)


def _ggd(x: np.ndarray) -> tuple[float, float]:
    m2 = np.mean(x * x)
    m1 = np.mean(np.abs(x))
    rho = m2 / (m1 * m1 + 1e-12)
    rr = _gamma(1.0 / _GAM) * _gamma(3.0 / _GAM) / (_gamma(2.0 / _GAM) ** 2)
    alpha = _GAM[int(np.argmin((rr - rho) ** 2))]
    return float(alpha), float(np.sqrt(m2))


def _patch_features(mscn: np.ndarray) -> np.ndarray:
    feats = list(_ggd(mscn.reshape(-1)))
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = mscn[max(dy, 0):mscn.shape[0] - max(-dy, 0) or None,
                 max(dx, 0):mscn.shape[1] - max(-dx, 0) or None]
        b = mscn[max(-dy, 0):mscn.shape[0] - max(dy, 0) or None,
                 max(-dx, 0):mscn.shape[1] - max(dx, 0) or None]
        prod = (a * b).reshape(-1)
        alpha, sl, sr = _aggd(prod)
        const = np.sqrt(_gamma(1.0 / alpha) / _gamma(3.0 / alpha))
        mean = (sr - sl) * (_gamma(2.0 / alpha) / _gamma(1.0 / alpha)) * const
        feats += [alpha, mean, sl * sl, sr * sr]
    return np.asarray(feats, dtype=np.float64)


def _luma_of(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img.astype(np.float64)


def image_features(luma: np.ndarray) -> np.ndarray:
    """(n_patches, 36) two-scale NIQE features over PATCH-sized tiles."""
    rows = []
    for scale in (1, 2):
        lm = luma if scale == 1 else luma[::2, ::2]
        m = _mscn(lm)
        p = PATCH // scale
        ny, nx = m.shape[0] // p, m.shape[1] // p
        feats = [
            _patch_features(m[i * p:(i + 1) * p, j * p:(j + 1) * p])
            for i in range(ny) for j in range(nx)
        ]
        rows.append(np.stack(feats) if feats else np.zeros((0, 18)))
    # pair each scale-2 patch (half-res, half patch size = same region) with
    # its scale-1 patch
    s1, s2 = rows
    ny1 = luma.shape[0] // PATCH
    nx1 = luma.shape[1] // PATCH
    ny2 = (luma.shape[0] // 2) // (PATCH // 2)
    nx2 = (luma.shape[1] // 2) // (PATCH // 2)
    out = []
    for i in range(min(ny1, ny2)):
        for j in range(min(nx1, nx2)):
            out.append(np.concatenate([s1[i * nx1 + j], s2[i * nx2 + j]]))
    return np.stack(out) if out else np.zeros((0, 36))


def fit_model(folder: Path, out: Path, max_images: int = 200,
              sharp_frac: float = 0.75) -> None:
    from LTX_2_MLX.videotoolbox.images import load_image_rgb
    import mlx.core as mx
    paths = sorted(p for p in folder.rglob("*.png"))
    if len(paths) > max_images:
        paths = paths[:: max(1, len(paths) // max_images)][:max_images]
    if not paths:
        raise SystemExit(f"no .png images under {folder}")
    feats = []
    for p in paths:
        img = np.asarray(load_image_rgb(p).astype(mx.float32)) / 255.0
        luma = _luma_of(img)
        f = image_features(luma)
        if not len(f):
            continue
        # NIQE fits the pristine model on the SHARPEST patches only
        m = _mscn(luma)
        pch = PATCH
        ny, nx = m.shape[0] // pch, m.shape[1] // pch
        sharp = np.asarray([
            np.std(m[i * pch:(i + 1) * pch, j * pch:(j + 1) * pch])
            for i in range(ny) for j in range(nx)
        ])
        keep = sharp >= np.quantile(sharp, 1.0 - sharp_frac)
        feats.append(f[keep[: len(f)]])
    allf = np.concatenate(feats, axis=0)
    mu = allf.mean(axis=0)
    cov = np.cov(allf, rowvar=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mu=mu, cov=cov, n=allf.shape[0])
    print(f"pristine model: {allf.shape[0]} patches from {len(paths)} images -> {out}")


def _read_video_lumas(path: Path, every: int = 1) -> list[np.ndarray]:
    import av
    out = []
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i % every:
                continue
            out.append(_luma_of(f.to_ndarray(format="rgb24").astype(np.float64) / 255.0))
    return out


def score_lumas(lumas: list[np.ndarray], model: Path = MODEL_PATH) -> float:
    m = np.load(model)
    mu_p, cov_p = m["mu"], m["cov"]
    feats = np.concatenate([image_features(lu) for lu in lumas], axis=0)
    mu_d = feats.mean(axis=0)
    cov_d = np.cov(feats, rowvar=False)
    d = mu_p - mu_d
    pinv = np.linalg.pinv((cov_p + cov_d) / 2.0)
    return float(np.sqrt(max(d @ pinv @ d, 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="*", help="videos (or images) to score")
    ap.add_argument("--fit", metavar="FOLDER", help="fit the pristine model from images")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--out", default=str(MODEL_PATH))
    ap.add_argument("--every", type=int, default=3, help="score every Nth frame")
    args = ap.parse_args()
    if args.fit:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        fit_model(Path(args.fit), Path(args.out))
        return 0
    rows = {}
    for inp in args.inputs:
        lumas = _read_video_lumas(Path(inp), every=args.every)
        rows[inp] = score_lumas(lumas, Path(args.model))
        print(f"{rows[inp]:7.3f}  {inp}")
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
