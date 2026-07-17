"""Typed noise-map conditioning config shared across denoise families.

The map-conditioned denoisers (bsvd, fastdvdnet, mc, and pvdd's level
variants) take an optional per-pixel sigma estimate plus a per-frame GOP
pulse gain. Both are stateful analysis objects the engine owns and feeds;
this module only turns the flat ``noise_map*`` stage keys (planning 07's
``--noise-map`` family) into a frozen config and builds those objects, so
the conditioning math stays in ``analysis.noise`` and the engines.

The default config - ``mode = "constant"`` with pulse off - builds
nothing, so a denoise stage with no ``noise_map*`` keys behaves exactly
as it did before conditioning was expressible as stage config.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from kinovsr.config.helpers import typed_value

# Flat stage keys, named to match the --noise-map* CLI flags one for one so
# the flag CLI maps onto them directly. --noise-map-debug is intentionally
# absent: it writes a post-run diagnostic PNG, an output concern rather than
# conditioning config.
#
# The tracker keys drive the sigma-map estimator and the pulse gain, and every
# map-capable family takes them. The stream keys are engine-side streaming
# controls (per-frame refresh cadence and the sigma floor) that only the
# frame-streaming denoisers take; windowed pvdd re-estimates per window and
# does not accept them.
NOISE_MAP_TRACKER_KEYS = (
    "noise_map",
    "noise_map_gain",
    "noise_map_masking",
    "noise_map_motion_cap",
    "noise_map_floor_mode",
    "noise_map_pulse",
    "noise_map_upsample",
)
NOISE_MAP_STREAM_KEYS = (
    "noise_map_refresh",
    "noise_map_floor",
)
NOISE_MAP_KEYS = (*NOISE_MAP_TRACKER_KEYS, *NOISE_MAP_STREAM_KEYS)

_MODES = ("constant", "auto")
_MOTION_CAPS = ("strict", "loose", "off")
_FLOOR_MODES = ("mc", "flat")
_UPSAMPLES = ("edge", "box")


@dataclasses.dataclass(frozen=True, slots=True)
class NoiseMapConfig:
    mode: str = "constant"       # "constant" = no map; "auto" = estimate one
    gain: float = 1.0            # multiplier on the estimated map
    refresh: int = 64            # streaming re-estimate cadence (frames); 0 holds
    masking: float = 0.0         # perceptual masking weight [0, 1]
    motion_cap: str = "strict"   # motion-likeness suppression: strict/loose/off
    floor_mode: str = "mc"       # motion-immune floor source: mc/flat
    floor: float = 0.0           # minimum sigma under the map [0, 1]
    pulse: bool = False          # per-frame GOP-phase pulse gain
    upsample: str = "edge"       # grid -> pixels: edge (guided) / box (repeat)


def parse_noise_map(raw: Mapping[str, Any]) -> NoiseMapConfig:
    """Parse the flat ``noise_map*`` keys into a frozen :class:`NoiseMapConfig`.

    Enum and range checks fail at parse (open time), not the first frame.
    Keys are validated by the family's own ``reject_unknown_keys`` before
    this runs; here each value is read and bounded.
    """
    mode = typed_value(raw, "noise_map", str, "constant")
    if mode not in _MODES:
        raise ValueError(f"noise_map must be one of {list(_MODES)}")
    gain = typed_value(raw, "noise_map_gain", float, 1.0)
    if gain <= 0.0:
        raise ValueError("noise_map_gain must be > 0")
    refresh = typed_value(raw, "noise_map_refresh", int, 64)
    if refresh < 0:
        raise ValueError("noise_map_refresh must be >= 0")
    masking = typed_value(raw, "noise_map_masking", float, 0.0)
    if not 0.0 <= masking <= 1.0:
        raise ValueError("noise_map_masking must be in [0, 1]")
    motion_cap = typed_value(raw, "noise_map_motion_cap", str, "strict")
    if motion_cap not in _MOTION_CAPS:
        raise ValueError(f"noise_map_motion_cap must be one of {list(_MOTION_CAPS)}")
    floor_mode = typed_value(raw, "noise_map_floor_mode", str, "mc")
    if floor_mode not in _FLOOR_MODES:
        raise ValueError(f"noise_map_floor_mode must be one of {list(_FLOOR_MODES)}")
    upsample = typed_value(raw, "noise_map_upsample", str, "edge")
    if upsample not in _UPSAMPLES:
        raise ValueError(f"noise_map_upsample must be one of {list(_UPSAMPLES)}")
    floor = typed_value(raw, "noise_map_floor", float, 0.0)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("noise_map_floor must be in [0, 1]")
    pulse = typed_value(raw, "noise_map_pulse", bool, False)
    return NoiseMapConfig(
        mode=mode, gain=gain, refresh=refresh, masking=masking,
        motion_cap=motion_cap, floor_mode=floor_mode, floor=floor, pulse=pulse,
        upsample=upsample)


def build_conditioning(config: NoiseMapConfig) -> tuple[Any | None, Any | None]:
    """Build ``(tracker, pulse)`` for the engine from a parsed config.

    ``mode = "auto"`` builds the sigma-map tracker; ``pulse = True`` builds
    the GOP-phase pulse gain (independent of mode, matching the flat CLI).
    Everything else - refresh cadence and the sigma floor - stays a scalar
    the family threads into its engine directly. Constructed lazily so an
    unconditioned stage imports no estimator code.
    """
    tracker = pulse = None
    if config.mode == "auto":
        from kinovsr.analysis.noise import NoiseMapTracker

        tracker = NoiseMapTracker(
            gain=config.gain, motion_cap=config.motion_cap,
            masking=config.masking, pulse_robust=config.pulse,
            floor_mode=config.floor_mode, upsample=config.upsample)
    if config.pulse:
        from kinovsr.analysis.noise import PulseGain

        pulse = PulseGain()
    return tracker, pulse


# The deblock-map keys condition stdf/fbcnn the way noise-map conditions the
# denoisers: a per-pixel blockiness mask that gates the correction. It reuses
# NoiseMapTracker with the blockiness estimator (min_frames=1, a spatial
# estimate needs no temporal warm-up), so it is a smaller surface - just the
# mode and a gain.
DEBLOCK_MAP_KEYS = ("deblock_map", "deblock_map_gain")


@dataclasses.dataclass(frozen=True, slots=True)
class DeblockMapConfig:
    mode: str = "constant"    # "constant" = no mask; "auto" = estimate one
    gain: float = 1.0


def parse_deblock_map(raw: Mapping[str, Any]) -> DeblockMapConfig:
    """Parse the flat ``deblock_map``/``deblock_map_gain`` keys."""
    mode = typed_value(raw, "deblock_map", str, "constant")
    if mode not in _MODES:
        raise ValueError(f"deblock_map must be one of {list(_MODES)}")
    gain = typed_value(raw, "deblock_map_gain", float, 1.0)
    if gain <= 0.0:
        raise ValueError("deblock_map_gain must be > 0")
    return DeblockMapConfig(mode=mode, gain=gain)


def build_blockiness_tracker(config: DeblockMapConfig) -> Any | None:
    """Build the blockiness tracker for a deblocker, or None for constant."""
    if config.mode != "auto":
        return None
    from kinovsr.analysis.noise import NoiseMapTracker, estimate_blockiness_map

    return NoiseMapTracker(gain=config.gain, min_frames=1,
                           estimator=estimate_blockiness_map)


# ---- End-of-run diagnostics (shared by every map-conditioned family) -----
#
# Families expose run_diagnostics()/debug_images() by delegating here: the
# attribute convention (last_noise_map, SIGMA_MIN/MAX, _map_floor, _pulse /
# _pulse_log, last_blockiness_map) is this module's conditioning contract,
# so the report logic lives next to it instead of as four copies. The line
# formats are the inherited harness's, verbatim, for output parity.

def noise_map_diagnostics(driver: Any) -> list[str]:
    """The harness's ``[noise-map]`` end-of-run report from a driver's own
    conditioning state; [] when no map conditioning ran."""
    import mlx.core as mx

    lines: list[str] = []
    nm = getattr(driver, "last_noise_map", None)
    if nm is not None:
        s = mx.sort(nm.reshape(-1))
        n = s.shape[0]
        lines.append(
            f"[noise-map] estimated sigma: min {float(s[0]):.4f}  "
            f"median {float(s[n // 2]):.4f}  "
            f"p95 {float(s[int(0.95 * (n - 1))]):.4f}  "
            f"max {float(s[-1]):.4f}")
        # what the net actually receives: the estimate clamped into the
        # consumer's conditioning bounds (trained range and/or user floor)
        lo = max(float(getattr(driver, "SIGMA_MIN", 0.0) or 0.0),
                 float(getattr(driver, "_map_floor", 0.0) or 0.0))
        hi = float(getattr(driver, "SIGMA_MAX", 0.0) or 0.0)
        if lo > 0.0 or hi > 0.0:
            e = mx.sort(mx.clip(nm, lo, hi if hi > 0 else 1.0).reshape(-1))
            lines.append(
                f"[noise-map] effective conditioning: min {float(e[0]):.4f}  "
                f"median {float(e[n // 2]):.4f}  max {float(e[-1]):.4f}  "
                f"(floor {lo:.4f}"
                f"{f', ceil {hi:.4f}' if hi > 0 else ''})")
    if getattr(driver, "_pulse", None) is not None:
        log = list(getattr(driver, "_pulse_log", ()) or ())
        if log:
            ps = sorted(log)
            lines.append(
                f"[noise-map] pulse gain over {len(ps)} frames: "
                f"min {ps[0]:.2f}  median {ps[len(ps) // 2]:.2f}  "
                f"max {ps[-1]:.2f}  "
                f"({sum(1 for g in ps if g > 1.2)} frames > 1.2)")
    return lines


def noise_map_debug_image(driver: Any) -> dict[str, Any]:
    """The harness's ``_noisemap.png`` content: sigma / 0.15, in [0,1]."""
    import mlx.core as mx

    nm = getattr(driver, "last_noise_map", None)
    if nm is None:
        return {}
    return {"noisemap": mx.clip(nm[:, :, 0] / 0.15, 0, 1)}


def blockiness_diagnostics(driver: Any) -> list[str]:
    """The harness's ``[deblock-map]`` mask stats; [] with no mask."""
    import mlx.core as mx

    bm = getattr(driver, "last_blockiness_map", None)
    if bm is None:
        return []
    s = mx.sort(bm.reshape(-1))
    n = s.shape[0]
    share = float(mx.mean((bm > 0.5).astype(mx.float32))) * 100
    return [
        f"[deblock-map] blockiness mask: median {float(s[n // 2]):.3f}  "
        f"p95 {float(s[int(0.95 * (n - 1))]):.3f}  max {float(s[-1]):.3f}  "
        f"({share:.0f}% of frame > 0.5)"]


def blockiness_debug_image(driver: Any) -> dict[str, Any]:
    """The harness's ``_blockmap.png`` content: the mask, in [0,1]."""
    import mlx.core as mx

    bm = getattr(driver, "last_blockiness_map", None)
    if bm is None:
        return {}
    return {"blockmap": mx.clip(bm[:, :, 0], 0, 1)}


__all__ = [
    "DEBLOCK_MAP_KEYS",
    "NOISE_MAP_KEYS",
    "NOISE_MAP_STREAM_KEYS",
    "NOISE_MAP_TRACKER_KEYS",
    "DeblockMapConfig",
    "NoiseMapConfig",
    "blockiness_debug_image",
    "blockiness_diagnostics",
    "build_blockiness_tracker",
    "build_conditioning",
    "noise_map_debug_image",
    "noise_map_diagnostics",
    "parse_deblock_map",
    "parse_noise_map",
]
