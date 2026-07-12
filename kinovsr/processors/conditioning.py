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
NOISE_MAP_KEYS = (
    "noise_map",
    "noise_map_gain",
    "noise_map_refresh",
    "noise_map_masking",
    "noise_map_motion_cap",
    "noise_map_floor_mode",
    "noise_map_floor",
    "noise_map_pulse",
)

_MODES = ("constant", "auto")
_MOTION_CAPS = ("strict", "loose", "off")
_FLOOR_MODES = ("mc", "flat")


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
    floor = typed_value(raw, "noise_map_floor", float, 0.0)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("noise_map_floor must be in [0, 1]")
    pulse = typed_value(raw, "noise_map_pulse", bool, False)
    return NoiseMapConfig(
        mode=mode, gain=gain, refresh=refresh, masking=masking,
        motion_cap=motion_cap, floor_mode=floor_mode, floor=floor, pulse=pulse)


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
            floor_mode=config.floor_mode)
    if config.pulse:
        from kinovsr.analysis.noise import PulseGain

        pulse = PulseGain()
    return tracker, pulse


__all__ = [
    "NOISE_MAP_KEYS",
    "NoiseMapConfig",
    "build_conditioning",
    "parse_noise_map",
]
