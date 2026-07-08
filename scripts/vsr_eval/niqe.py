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

from typing import Any

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "weights" / "niqe_pristine_reds.safetensors"
PATCH = 96

# ---- MLX compute core (numpy only at the file/CLI boundary) -----------------
# gamma-function terms are precomputed per table entry, so nothing at runtime
# needs a special function: alpha fits reduce to an argmin against _R_GAM.
_GAM_NP = np.arange(0.2, 10.001, 0.001)
_lgamma = np.vectorize(math.lgamma)
_R_GAM_NP = np.exp(_lgamma(1.0 / _GAM_NP) + _lgamma(3.0 / _GAM_NP)
                   - 2.0 * _lgamma(2.0 / _GAM_NP))
_CONST_NP = np.sqrt(np.exp(_lgamma(1.0 / _GAM_NP) - _lgamma(3.0 / _GAM_NP)))
_MEANFAC_NP = np.exp(_lgamma(2.0 / _GAM_NP) - _lgamma(1.0 / _GAM_NP)) * _CONST_NP
_GAM = mx.array(_GAM_NP.astype(np.float32))
_R_GAM = mx.array(_R_GAM_NP.astype(np.float32))
_MEANFAC = mx.array(_MEANFAC_NP.astype(np.float32))


def _alpha_idx(rhat: Any) -> Any:
    return mx.argmin(mx.abs(_R_GAM[None, :] - rhat[:, None]), axis=1)


def _gauss_kernel(sigma: float = 7.0 / 6.0, truncate: float = 3.0) -> Any:
    r = int(truncate * sigma + 0.5)
    x = mx.arange(-r, r + 1).astype(mx.float32)
    k = mx.exp(-(x * x) / (2 * sigma * sigma))
    return k / mx.sum(k), r


_K1D, _KRAD = _gauss_kernel()


def _reflect_pad(x: Any, r: int, axis: int) -> Any:
    # scipy gaussian_filter default 'reflect' = mirror including the edge
    if axis == 0:
        return mx.concatenate([x[r - 1::-1, :], x, x[:-r - 1:-1, :]], axis=0)
    return mx.concatenate([x[:, r - 1::-1], x, x[:, :-r - 1:-1]], axis=1)


def _gauss_blur(x: Any) -> Any:
    r = _KRAD
    k = _K1D.reshape(1, 2 * r + 1, 1, 1)
    y = _reflect_pad(x, r, 0)
    y = mx.conv2d(y[None, :, :, None], k)[0, :, :, 0]
    y = _reflect_pad(y, r, 1)
    y = mx.conv2d(y[None, :, :, None], mx.transpose(k, (0, 2, 1, 3)))[0, :, :, 0]
    return y


def _mscn(luma: Any) -> Any:
    mu = _gauss_blur(luma)
    sigma = mx.sqrt(mx.maximum(_gauss_blur(luma * luma) - mu * mu, 0.0))
    return (luma - mu) / (sigma + 1.0)


def _scale_features(m: Any, p: int) -> Any:
    """(P, 18) GGD + 4x AGGD features for every p x p patch, vectorized."""
    ny, nx = int(m.shape[0]) // p, int(m.shape[1]) // p
    if not ny or not nx:
        return mx.zeros((0, 18))
    pt = m[: ny * p, : nx * p].reshape(ny, p, nx, p).transpose(0, 2, 1, 3) \
        .reshape(ny * nx, p, p)
    feats = []
    # GGD over the patch
    m1 = mx.mean(mx.abs(pt), axis=(1, 2))
    m2 = mx.mean(pt * pt, axis=(1, 2))
    rho = m2 / (m1 * m1 + 1e-12)
    gi = _alpha_idx(rho)
    feats += [_GAM[gi], mx.sqrt(m2)]
    # AGGD over the four neighbor products, computed inside each patch
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = pt[:, max(dy, 0):p - max(-dy, 0) or None,
               max(dx, 0):p - max(-dx, 0) or None]
        b = pt[:, max(-dy, 0):p - max(dy, 0) or None,
               max(-dx, 0):p - max(dx, 0) or None]
        prod = (a * b).reshape(ny * nx, -1)
        neg = (prod < 0).astype(mx.float32)
        pos = 1.0 - neg
        nn = mx.maximum(mx.sum(neg, axis=1), 1.0)
        np_ = mx.maximum(mx.sum(pos, axis=1), 1.0)
        sl = mx.sqrt(mx.sum(prod * prod * neg, axis=1) / nn) + 1e-6
        sr = mx.sqrt(mx.sum(prod * prod * pos, axis=1) / np_) + 1e-6
        gh = sl / sr
        m1p = mx.mean(mx.abs(prod), axis=1)
        m2p = mx.mean(prod * prod, axis=1)
        rh = (m1p * m1p) / (m2p + 1e-12)
        rhat = rh * (gh ** 3 + 1.0) * (gh + 1.0) / ((gh * gh + 1.0) ** 2)
        ai = _alpha_idx(rhat)
        mean = (sr - sl) * _MEANFAC[ai]
        feats += [_GAM[ai], mean, sl * sl, sr * sr]
    return mx.stack(feats, axis=1)


def _luma_of(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img.astype(np.float32)


def image_features(luma: np.ndarray) -> np.ndarray:
    """(n_patches, 36) two-scale NIQE features over PATCH-sized tiles."""
    lm = mx.array(np.ascontiguousarray(luma, dtype=np.float32))
    s1 = _scale_features(_mscn(lm), PATCH)
    s2 = _scale_features(_mscn(lm[::2, ::2]), PATCH // 2)
    ny1, nx1 = luma.shape[0] // PATCH, luma.shape[1] // PATCH
    ny2 = (luma.shape[0] // 2) // (PATCH // 2)
    nx2 = (luma.shape[1] // 2) // (PATCH // 2)
    ny, nx = min(ny1, ny2), min(nx1, nx2)
    if not ny or not nx:
        return np.zeros((0, 36))
    i1 = mx.array([i * nx1 + j for i in range(ny) for j in range(nx)])
    i2 = mx.array([i * nx2 + j for i in range(ny) for j in range(nx)])
    out = mx.concatenate([s1[i1], s2[i2]], axis=1)
    mx.eval(out)
    return np.asarray(out.astype(mx.float32), dtype=np.float64)


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
        m = _mscn(mx.array(np.ascontiguousarray(luma, dtype=np.float32)))
        pch = PATCH
        ny, nx = int(m.shape[0]) // pch, int(m.shape[1]) // pch
        pt = m[: ny * pch, : nx * pch].reshape(ny, pch, nx, pch) \
            .transpose(0, 2, 1, 3).reshape(ny * nx, -1)
        sharp = np.asarray(mx.sqrt(mx.var(pt, axis=1)).astype(mx.float32))
        keep = sharp >= np.quantile(sharp, 1.0 - sharp_frac)
        feats.append(f[keep[: len(f)]])
    allf = np.concatenate(feats, axis=0)
    mu = allf.mean(axis=0)
    cov = np.cov(allf, rowvar=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out), {
        "mu": mx.array(mu.astype(np.float32)),
        "cov": mx.array(cov.astype(np.float32)),
        "n": mx.array([allf.shape[0]], dtype=mx.int32),
    })
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
    m = mx.load(str(model))
    mu_p = np.asarray(m["mu"].astype(mx.float32), dtype=np.float64)
    cov_p = np.asarray(m["cov"].astype(mx.float32), dtype=np.float64)
    feats = np.concatenate([image_features(lu) for lu in lumas], axis=0)
    mu_d = feats.mean(axis=0)
    cov_d = np.cov(feats, rowvar=False)
    d = mu_p - mu_d
    pinv = np.linalg.pinv((cov_p + cov_d) / 2.0)
    return float(np.sqrt(max(d @ pinv @ d, 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="*", help="videos (or images) to score")
    ap.add_argument("--fit", metavar="FOLDER",
                    help="fit the pristine model (safetensors) from images")
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
