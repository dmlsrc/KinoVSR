"""Global exposure leveler: histogram matching to a temporal reference.

Targets GLOBAL luma pumping - auto-exposure hunting, camcorder AGC
flicker, analog gain wander - which the deflicker family deliberately
does not cover (deflicker corrects per-tile quantization-state flicker
on verified-static content; a global gain swing fails its photometric
verification wholesale). The two families compose: level first, then
deflicker sees a stable exposure.

Mechanism (each choice measured; see the planning record for numbers):

- Each frame's luma distribution is matched to the average distribution
  of a CENTERED +-window reference. Averaging happens in the QUANTILE
  domain (the 1D Wasserstein barycenter): averaging histograms instead
  builds a mixture that smears gain-shifted frames apart and measured
  6 dB worse. A causal reference (EMA) lags every trend and visibly
  flattens fades; the centered window follows them.
- Histograms come from vImage (CPU, ~0.2 ms, immune to GPU load) at
  1024 float bins: 256-bin u8 histograms floor the correction on
  subtle (about 2 percent) pumping.
- The transfer curve is applied as a linearly interpolated LUT on the
  fp32 luma plane, and RGB is scaled by the per-pixel luma ratio.
  Taking the native u8 specification output instead amplifies its
  quantization on dark pixels.
- Full-distribution matching (not scalar gain) because real exposure
  pumping is gamma-like, where the scalar gain leveler measures 4-5 dB
  worse; on pure gain swings the two are identical.

Limits, stated plainly: only global pumping (per-region flicker is
deflicker's job); slow random-walk wander is indistinguishable from a
legitimate lighting change and is deliberately followed, not fought;
clipped highlights cannot be un-clipped. The deadband keeps the stage
bit-exact passthrough on stable content.
"""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Mapping

import mlx.core as mx

from kinovsr.config.helpers import reject_unknown_keys, typed_value
from kinovsr.native.vimage import histogram_planarf
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

_BINS = 1024
_LUMA = (0.2126, 0.7152, 0.0722)
_RATIO_CLAMP = (0.5, 2.0)


def _quantile_fn(hist: list[int], n_px: float) -> list[float]:
    """Inverse CDF sampled at the bin count's quantile nodes, in bin
    units, with piecewise-linear interpolation across sparse bins."""
    bins = len(hist)
    cdf, acc = [], 0.0
    for count in hist:
        acc += count / n_px
        cdf.append(acc)
    qf, k = [], 0
    for j in range(bins):
        q = (j + 0.5) / bins
        while k < bins - 1 and cdf[k] < q:
            k += 1
        left = cdf[k - 1] if k else 0.0
        width = cdf[k] - left
        frac = (q - left) / width if width > 1e-9 else 0.0
        qf.append(min(float(bins - 1), max(0.0, k - 1 + frac + 0.5)))
    return qf


def _cdf(hist: list[int], n_px: float) -> list[float]:
    out, acc = [], 0.0
    for count in hist:
        acc += count / n_px
        out.append(acc)
    return out


def _transfer(cur_cdf: list[float], ref_qf: list[float]) -> list[float]:
    """256..N-entry transfer: current bin midpoint mass through the
    reference quantile function, normalized to [0, 1] luma."""
    bins = len(cur_cdf)
    out = []
    for b in range(bins):
        pmf_b = cur_cdf[b] - (cur_cdf[b - 1] if b else 0.0)
        q = cur_cdf[b] - 0.5 * pmf_b
        pos = min(float(bins - 1), max(0.0, q * bins - 0.5))
        j0 = int(pos)
        j1 = min(j0 + 1, bins - 1)
        frac = pos - j0
        out.append((ref_qf[j0] * (1.0 - frac) + ref_qf[j1] * frac)
                   / (bins - 1))
    return out


class HistogramLeveler:
    """Self-buffered centered-window leveler (feed/flush driver)."""

    def __init__(self, window: int = 5, deadband: float = 0.003) -> None:
        self._k = max(1, int(window))
        self._deadband = max(0.0, float(deadband))
        self._luma_w = mx.array(_LUMA, dtype=mx.float32)
        self._scratch: bytearray | None = None
        self._reset_state()
        # run stats survive reset (they describe the whole run)
        self._shifts: list[float] = []
        self._corrected = 0
        self._emitted_total = 0

    def _reset_state(self) -> None:
        self._frames: list = []       # (rgb, luma_plane, token)
        self._qfs: list[list[float]] = []
        self._cdfs: list[list[float]] = []
        self._base = 0                # absolute index of self._frames[0]
        self._received = 0
        self._emitted = 0

    # -- per-frame analysis -------------------------------------------------

    def _analyze(self, rgb) -> tuple:
        luma = (rgb[..., :3].astype(mx.float32) @ self._luma_w)
        mx.eval(luma)
        h, w = int(luma.shape[0]), int(luma.shape[1])
        need = h * w * 4
        if self._scratch is None or len(self._scratch) != need:
            self._scratch = bytearray(need)
        self._scratch[:] = memoryview(mx.contiguous(luma)).cast("B")
        hist = histogram_planarf(self._scratch, w, h, _BINS)
        n_px = float(h * w)
        return luma, _cdf(hist, n_px), _quantile_fn(hist, n_px)

    # -- emit ---------------------------------------------------------------

    def _emit_one(self, last: int):
        e = self._emitted
        # Symmetric shrink at segment boundaries: a lopsided window pulls
        # boundary frames toward the interior on any luma trend (a hard
        # fade's first frame measurably darkens), while a symmetric mean
        # of a linear trend equals the center value - so the window
        # narrows to what both sides can supply and the stage ramps in
        # over the first/last K frames instead of biasing them.
        m = min(self._k, e - self._base, last - e)
        lo = e - m
        hi = e + m + 1
        i0 = lo - self._base
        idx = e - self._base
        # Quantile functions live as tiny MLX vectors: the window average
        # is one mx.mean over at most (2K+1, BINS), and the python-side
        # transfer only pays a tolist() when a correction actually fires.
        ref_mx = (mx.mean(mx.stack(self._qfs[i0:hi - self._base]), axis=0)
                  if hi - lo > 1 else self._qfs[idx])
        rgb, luma, token = self._frames[idx]
        # Gate on the SIGNED net displacement between the distributions:
        # content churn moves quantiles both ways and cancels, while real
        # pumping displaces the whole distribution coherently - measured
        # on stable real footage the absolute-shift gate fired on 7% of
        # frames from content drift alone; the signed gate does not.
        shift = abs(float(mx.mean(ref_mx - self._qfs[idx]))) / (_BINS - 1)
        self._shifts.append(shift)
        self._emitted_total += 1
        if shift <= self._deadband:
            out = rgb
        else:
            self._corrected += 1
            lut = mx.array(_transfer(self._cdfs[idx], ref_mx.tolist()),
                           dtype=mx.float32)
            pos = mx.clip(luma * (_BINS - 1), 0.0, float(_BINS - 1))
            j0 = mx.floor(pos).astype(mx.int32)
            j1 = mx.minimum(j0 + 1, _BINS - 1)
            frac = pos - j0.astype(mx.float32)
            y_out = lut[j0] * (1.0 - frac) + lut[j1] * frac
            ratio = mx.clip(y_out / mx.maximum(luma, 1e-4),
                            _RATIO_CLAMP[0], _RATIO_CLAMP[1])
            out = mx.clip(rgb * ratio[..., None], 0.0, 1.0)
            mx.eval(out)
        self._emitted += 1
        # drop history the sliding window can no longer reach
        drop = (self._emitted - self._k) - self._base
        if drop > 0:
            del self._frames[:drop]
            del self._qfs[:drop]
            del self._cdfs[:drop]
            self._base += drop
        return out, token

    # -- feed/flush driver contract ----------------------------------------

    def feed(self, rgb, token=None) -> list:
        luma, cdf, qf = self._analyze(rgb)
        self._frames.append((rgb, luma, token))
        self._cdfs.append(cdf)
        qf_mx = mx.array(qf, dtype=mx.float32)
        mx.eval(qf_mx)
        self._qfs.append(qf_mx)
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

    def reset(self) -> None:
        """Drop the buffered window (scene cut): the reference must never
        straddle a discontinuity. The scheduler flushes the tail first."""
        self._reset_state()

    def run_diagnostics(self) -> list[str]:
        if not self._shifts:
            return []
        ordered = sorted(self._shifts)
        p95 = ordered[int(0.95 * (len(ordered) - 1))]
        return [
            f"[level] global pumping meter: mean shift "
            f"{statistics.fmean(self._shifts) * 1000:.2f}e-3, p95 "
            f"{p95 * 1000:.2f}e-3 luma; corrected "
            f"{self._corrected}/{self._emitted_total} frames "
            f"(deadband {self._deadband * 1000:.1f}e-3; stable footage "
            f"should read near zero corrected)"
        ]

    def close(self) -> None:
        self._reset_state()
        self._scratch = None


# ===========================================================================
# Processor family
# ===========================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class LevelStageConfig:
    window: int
    deadband: float


def _passthrough(spec: StreamSpec, config: object) -> StreamSpec:
    return spec


class LevelFactory:
    name = "level"

    # CENTERED: the +-window reference's future half is self-buffered and
    # paid as `window` frames of output delay; no source lookahead.
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
            temporal_radius=5,
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
    ) -> LevelStageConfig:
        reject_unknown_keys(raw, ("window", "deadband"))
        window = typed_value(raw, "window", int, 5)
        if window < 1:
            raise ValueError("window must be >= 1")
        deadband = typed_value(raw, "deadband", float, 0.003)
        if deadband < 0.0:
            raise ValueError("deadband must be >= 0")
        return LevelStageConfig(window=window, deadband=deadband)

    def build(self, config: LevelStageConfig, *,
              context: PipelineContext) -> FeedFlushProcessor:
        def make_driver() -> HistogramLeveler:
            return HistogramLeveler(window=config.window,
                                    deadband=config.deadband)

        return FeedFlushProcessor(make_driver)


FACTORY = LevelFactory()
