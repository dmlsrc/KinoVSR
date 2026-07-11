"""Per-frame FBCNN JPEG-artifact-removal deblocker, RGB in / RGB out.

FBCNN is a single-image network (no temporal window), so this is a stateless per-frame
stage -- denoise(rgb) restores each frame independently. Pair it before the scaler to
strip JPEG / intra-block artifacts. Because it is single-image, BLIND mode
(quality=None) re-estimates the quality factor per frame, which can flicker on video as
the estimate drifts shot to shot; pass a fixed `quality` (a JPEG quality factor, lower =
stronger removal) for a temporally stable result. Unlike the temporal STDF deblocker it
does no noise averaging, so it is a pure deblocker, not a partial denoiser.

quality="auto" measures the quantization instead of assuming it: the DCT
coefficient comb (quant_comb.py) estimates a per-tile JPEG QF over a rolling
frame window, and the net runs on overlapping 256px tiles with PER-TILE QF
conditioning (FBCNN's FiLM input accepts a batch vector) plus a per-tile
dry/wet derived from the measured severity -- FiLM alone is not gentle
enough: even QF-85-conditioned FBCNN costs a near-clean region over 1 dB, so
light tiles must also be mostly left alone. A single global QF -- pinned or
blind -- blunts one side or the other.

Decline semantics differ by scope, and the difference matters: when SOME
tiles carry combs, a declined tile is evidence of absence (the detector
demonstrably works on this footage) and fills GENTLE (QF 88, low wet) --
filling from the global estimate would condition clean regions at the
surviving heavy tiles' QF, which is exactly the over-smoothing this mode
exists to avoid. When the WHOLE window declines (no JPEG history detectable
anywhere, e.g. combs killed by a later heavy re-encode), the frame runs
through the compiled full-frame path at `quality_fallback` -- the caller's
statement about footage the measurement cannot see.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from kinovsr.analysis.noise.estimate import _box_blur_full, _to_luma_2d
from kinovsr.analysis.quant_comb import estimate_qf_map

from . import net


class FbcnnDeblocker:
    """Stateless per-frame RGB JPEG-artifact deblocker (FBCNN color)."""

    MAP_REFRESH = 64   # frames between blockiness-mask / QF-map refreshes
    QF_TILE = 128      # comb measurement tile (validated floor; see quant_comb)
    NET_TILE = 256     # per-tile-QF forward tile
    NET_OVERLAP = 32   # blend margin between net tiles
    QF_WINDOW = 12     # frames of luma buffered for the comb
    QF_MIN_FRAMES = 6  # first estimate once this many frames are buffered
    QF_GENTLE = 88.0   # QF for comb-declined tiles when siblings have combs
    WET_FLOOR = 0.12   # minimum per-tile dry/wet (never a hard bypass)

    def __init__(self, weights: Any = None, quality: Any = None, strength: float = 1.0,
                 compile: bool = True, dtype: Any = mx.float16,
                 blockiness_map: Any = None, quality_fallback: float = 50.0):
        self._p = net.load_params(weights, dtype=dtype)
        self._in_nc, self._nb = net._config(self._p)
        if self._in_nc != 3:
            raise ValueError(
                f"FbcnnDeblocker expects the RGB (color) FBCNN checkpoint (in_nc=3), got "
                f"{self._in_nc}; the grayscale variants would need a luma path not wired here.")
        # User-facing JPEG quality (1-100, lower = more compressed = stronger removal)
        # maps to the model's inverted-quality qf_input = 1 - quality/100; None = blind;
        # "auto" = per-tile comb-measured QF with `quality_fallback` where undetected.
        self._auto = isinstance(quality, str) and quality.strip().lower() == "auto"
        if isinstance(quality, str) and not self._auto:
            raise ValueError(f"quality must be None, a number, or 'auto'; got {quality!r}")
        self._quality = None if (quality is None or self._auto) else float(quality)
        self._qf_input = (None if self._quality is None
                          else max(0.0, min(1.0, 1.0 - self._quality / 100.0)))
        self._fallback = float(quality_fallback)
        # strength = linear dry/wet on the correction (the QF-independent knob); a fixed
        # quality also skips the QF predictor, so a pinned-quality run is the faster one.
        self._strength = float(strength)
        # optional blockiness tracker: per-pixel wet/dry mask on the correction.
        # FBCNN's strength is an output lerp, so full-strength net + outside
        # blend of mask*strength is exact.
        self._tracker = blockiness_map
        self._mask: Any = None
        self._recent: list = []
        self._since_refresh = 0
        self.last_blockiness_map: Any = None   # fp32 (H,W,1) (debug)
        self._net_strength = 1.0 if self._tracker is not None else self._strength
        self._compile = compile
        if self._auto:
            self._fwd = None                   # per-tile path builds batches itself
            self._fwd_fallback: Any = None     # lazy compiled path for clean footage
            self._qf_frames: list = []
            self._qf_grid: Any = None          # (ty,tx) fp32 user-QF, median-smoothed
            self._qf_cells: Any = None         # raw filled cells (wet-map source)
            self._wet_full: Any = None         # cached (H,W,1) per-pixel wet
            self._since_qf = 0
            self.last_qf_info: dict | None = None   # {"global", "coverage"} (debug)
        else:
            self._fwd = net.make_forward(self._p, self._qf_input, self._net_strength,
                                         self._nb, compile=compile)

    def _reset_conditioning(self, clear_debug: bool = False) -> None:
        self._mask = None
        self._recent = []
        self._since_refresh = 0
        if self._auto:
            self._qf_frames = []
            self._qf_grid = None
            self._qf_cells = None
            self._wet_full = None
            self._since_qf = 0
        if self._tracker is not None and hasattr(self._tracker, "reset"):
            self._tracker.reset()
        if clear_debug:
            self.last_blockiness_map = None
            if self._auto:
                self.last_qf_info = None

    def reset(self) -> None:
        self._reset_conditioning(clear_debug=True)

    def close(self) -> None:
        pass

    # ---- auto-QF machinery ---------------------------------------------------

    def _refresh_qf(self, inp: Any) -> None:
        self._qf_frames.append(_to_luma_2d(inp[0]))
        if len(self._qf_frames) > self.QF_WINDOW:
            self._qf_frames.pop(0)
        due = self._qf_grid is None or self._since_qf >= self.MAP_REFRESH
        if due and len(self._qf_frames) >= self.QF_MIN_FRAMES:
            m = estimate_qf_map(self._qf_frames, tile=self.QF_TILE,
                                fallback=self._fallback)
            # re-fill with the partial-coverage semantics: declined-next-to-
            # detected means no damage HERE, so gentle, not the global QF
            raw = m["qf_grid"]
            valid = m["valid"]
            cells = [[float(raw[i, j]) if v else self.QF_GENTLE
                      for j, v in enumerate(vrow)] for i, vrow in enumerate(valid)]
            # FiLM grid: 3x3 MEDIAN smoothing -- kills isolated outlier tiles
            # but, unlike a mean, does not smear a low QF across a genuine
            # quality boundary (a mean drags neighboring clean tiles down)
            ty, tx = len(cells), len(cells[0])
            med = [[0.0] * tx for _ in range(ty)]
            for i in range(ty):
                for j in range(tx):
                    neigh = [cells[a][b]
                             for a in range(max(0, i - 1), min(ty, i + 2))
                             for b in range(max(0, j - 1), min(tx, j + 2))]
                    neigh.sort()
                    med[i][j] = neigh[len(neigh) // 2]
            self._qf_grid = mx.array(med, dtype=mx.float32)
            # wet map stays UNsmoothed: localization at measurement-cell
            # resolution, independent of the coarser net-tile FiLM
            self._qf_cells = cells
            self._wet_full = None
            # mode: the gentle fill is only valid when the detector
            # demonstrably works on this clip. A failed global fit with only
            # scattered tile hits (< 25% coverage) is NOT that -- it reads as
            # "nothing measurable here" (native H.264/HEVC or combs killed by
            # re-encode), which is the caller's quality_fallback territory.
            if m["global"]["qf"] is not None:
                mode = "measured"
            elif m["coverage"] >= 0.25:
                mode = "gentle"
            else:
                mode = "fallback"
            self.last_qf_info = {"global": m["global"], "coverage": m["coverage"],
                                 "mode": mode}
            self._since_qf = 0
        else:
            self._since_qf += 1

    def _wet_map(self, H: int, W: int) -> Any:
        if self._wet_full is not None and tuple(self._wet_full.shape[:2]) == (H, W):
            return self._wet_full
        cells = getattr(self, "_qf_cells", None)
        if cells is None:
            w = mx.full((H, W, 1), self.WET_FLOOR, dtype=mx.float32)   # warmup: dry
            self._wet_full = w
            return w
        wet = [[max(self.WET_FLOOR, min(1.0, (92.0 - q) / 45.0)) for q in row]
               for row in cells]
        g = mx.array(wet, dtype=mx.float32)
        full = mx.repeat(mx.repeat(g, self.QF_TILE, axis=0), self.QF_TILE, axis=1)
        fh, fw = int(full.shape[0]), int(full.shape[1])
        if fh < H:
            full = mx.concatenate(
                [full, mx.broadcast_to(full[-1:, :], (H - fh, fw))], axis=0)
        if fw < W:
            full = mx.concatenate(
                [full, mx.broadcast_to(full[:, -1:], (int(full.shape[0]), W - fw))], axis=1)
        full = _box_blur_full(full[:H, :W], 33)
        self._wet_full = full[:, :, None]
        return self._wet_full

    def _tile_origins(self, size: int, span: int) -> list:
        if span <= size:
            return [0]
        stride = size - self.NET_OVERLAP
        outs = list(range(0, span - size, stride))
        outs.append(span - size)
        return sorted(set(outs))

    def _tile_qf(self, y0: int, x0: int, th: int, tw: int) -> float:
        g = self._qf_grid
        if g is None:
            # warmup (no estimate yet): be gentle, not aggressive -- the
            # fallback QF is a statement about MEASURED-and-declined footage,
            # not about frames we simply have not looked at yet
            return self.QF_GENTLE
        ty, tx = int(g.shape[0]), int(g.shape[1])
        i = min(ty - 1, (y0 + th // 2) // self.QF_TILE)
        j = min(tx - 1, (x0 + tw // 2) // self.QF_TILE)
        return float(g[i, j])

    def _ramp(self, n: int, lo_open: bool, hi_open: bool) -> Any:
        w = mx.ones((n,), dtype=mx.float32)
        ov = min(self.NET_OVERLAP, n)
        r = (mx.arange(ov).astype(mx.float32) + 1.0) / float(ov)
        if lo_open:
            w = mx.concatenate([r, w[ov:]], axis=0)
        if hi_open:
            w = mx.concatenate([w[:-ov], r[::-1]], axis=0)
        return w

    def _forward_auto(self, inp: Any) -> Any:
        H, W = int(inp.shape[1]), int(inp.shape[2])
        info = self.last_qf_info or {}
        if self._qf_grid is not None and info.get("mode") == "fallback":
            # no usable JPEG evidence: single compiled full-frame pass at the
            # fallback QF instead of the tiled batch (same speed as a pin)
            if self._fwd_fallback is None:
                self._fwd_fallback = net.make_forward(
                    self._p, 1.0 - self._fallback / 100.0, self._net_strength,
                    self._nb, compile=self._compile)
            return self._fwd_fallback(inp)
        ts = self.NET_TILE
        ys = self._tile_origins(ts, H)
        xs = self._tile_origins(ts, W)
        th = min(ts, H)
        tw = min(ts, W)
        # severity-scaled dry/wet, PER PIXEL at measurement-cell resolution:
        # FiLM alone is not gentle enough on light regions (even QF-85
        # conditioning costs a near-clean area over 1 dB), and a per-net-tile
        # scalar would smear treatment across quality boundaries inside a tile
        wet_full = self._wet_map(H, W)
        tiles = []
        qfs = []
        wets = []
        wins = []
        for y0 in ys:
            for x0 in xs:
                tiles.append(inp[:, y0:y0 + th, x0:x0 + tw, :])
                q = self._tile_qf(y0, x0, th, tw)
                qfs.append(max(0.0, min(1.0, 1.0 - q / 100.0)))
                wets.append(wet_full[y0:y0 + th, x0:x0 + tw, :])
                wy = self._ramp(th, y0 > 0, y0 + th < H)
                wx = self._ramp(tw, x0 > 0, x0 + tw < W)
                wins.append(wy[:, None] * wx[None, :])
        num = mx.zeros((H, W, 3), dtype=mx.float32)
        den = mx.zeros((H, W, 1), dtype=mx.float32)
        chunk = 8
        for s in range(0, len(tiles), chunk):
            batch = mx.concatenate(tiles[s:s + chunk], axis=0)
            qv = mx.array(qfs[s:s + chunk], dtype=mx.float32).reshape(-1, 1)
            out, _ = net.fbcnn(batch, self._p, qf_input=qv,
                               strength=self._net_strength, nb=self._nb)
            out = out.astype(mx.float32)
            wv = mx.stack(wets[s:s + chunk], axis=0)
            out = batch.astype(mx.float32) + wv * (out - batch.astype(mx.float32))
            for k in range(int(out.shape[0])):
                i = s + k
                y0 = ys[i // len(xs)]
                x0 = xs[i % len(xs)]
                w = wins[i][:, :, None]
                num[y0:y0 + th, x0:x0 + tw, :] = (
                    num[y0:y0 + th, x0:x0 + tw, :] + out[k] * w)
                den[y0:y0 + th, x0:x0 + tw, :] = den[y0:y0 + th, x0:x0 + tw, :] + w
            mx.eval(num, den)
        return (num / mx.maximum(den, 1e-6))[None]

    # ---- main entry ------------------------------------------------------------

    def denoise(self, rgb_f32: Any) -> Any:
        """Restore one RGB frame (H,W,3) in [0,1]; returns (H,W,3)."""
        a = rgb_f32 if rgb_f32.ndim == 4 else rgb_f32[None]
        inp = mx.clip(a[..., :3].astype(mx.float32), 0.0, 1.0)
        if self._auto:
            self._refresh_qf(inp)
            out = mx.clip(self._forward_auto(inp), 0.0, 1.0)
        else:
            out = mx.clip(self._fwd(inp), 0.0, 1.0)
        if self._tracker is not None:
            self._recent.append(inp)
            if len(self._recent) > 6:
                self._recent.pop(0)
            if self._mask is None or self._since_refresh >= self.MAP_REFRESH:
                m = self._tracker.update(self._recent)
                if m is not None:
                    self.last_blockiness_map = m
                    self._mask = m[None]           # (1,H,W,1)
                self._since_refresh = 0
            else:
                self._since_refresh += 1
            if self._mask is not None:
                # clamp at 1.0 post-gain: blend factors above 1 extrapolate
                out = inp + (mx.minimum(self._mask, 1.0) * self._strength) * (out - inp)
        mx.eval(out)
        return out[0]
