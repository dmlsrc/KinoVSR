"""Spatial noise-map estimation for map-conditioned (FFDNet-style) denoisers.

Estimates a per-pixel noise **sigma map** (units: [0,1] luma sigma, the same scale
FastDVDnet and BSVD take as their 4th input channel; PVDD's level checkpoints take
noise **variance**, so square this map for them) from a short window of frames.

Method: temporal, not spatial. Consecutive-frame differences on luma cancel static
content (including fine texture, which spatial estimators misread as noise) and
leave motion plus temporally-varying noise. Per coarse block, the pooled diff
samples reduce by tail RMS (top-5%, AWGN-normalized): conditioning must cover
the AMPLITUDE of what flickers -- sparse mosquito flicker reads zero at any
quantile and only sqrt(density) of its amplitude in plain energy, yet a net
conditioned below the amplitude preserves the flash as signal. Tail RMS reads
the flicker amplitude scale and stays exact on dense AWGN.
Motion-contaminated blocks are capped
by a signal-dependence model (sigma as a function of block luma, fitted from the
quiet blocks), which also encodes the shadows-are-noisier structure of real
footage. The coarse map is smoothed and bilinearly-ish upsampled so the
conditioning signal stays smooth (map-trained nets were trained on smooth maps).

Because the estimate is differential it measures the *temporally fluctuating*
noise component -- exactly what temporal denoisers can remove. Static grain and
fixed-pattern noise do not register (they also cannot be removed temporally).

Public API:
  estimate_sigma_map(frames) -> (H,W,1) fp32 sigma map, or None if < 2 frames
  NoiseMapTracker            -> stateful wrapper: gain + EMA across windows
"""

from .classify import analyze_noise, classify_noise_analysis
from .estimate import estimate_sigma_map
from .grid import detect_grid_period, estimate_blockiness_map
from .track import NoiseMapTracker, PulseGain

__all__ = ["analyze_noise", "estimate_sigma_map", "estimate_blockiness_map",
           "classify_noise_analysis", "detect_grid_period",
           "NoiseMapTracker", "PulseGain"]
