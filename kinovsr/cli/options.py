"""Option vocabulary: the spec type and the shared-key contract.

Every processor-family CLI flag is ``--<family>-<key>`` where ``<key>`` is
either a shared vocabulary key (same word = same concept in every family)
or a registered family-specific key. The registry in
:mod:`kinovsr.cli._registry` is data; the parser is generated from it, and
a conformance test rejects rows that invent a new name for a shared
concept. Stage tables and ``--set`` (M3) reuse the same key names in
snake_case.

Only canonical spellings are accepted.  The compatibility aliases retained
during the harness migration were retired once all in-repository callers used
the shared vocabulary.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

# Preprocess slots, in default execution order.
PP_STAGE_NAMES = ("level", "restore", "deflicker", "deblock", "denoise",
                  "nafnet")


def pp_order(spec: str) -> list:
    """argparse type for --preprocess-order: a comma-separated
    permutation/subset of the preprocess stage names."""
    names = [x.strip() for x in spec.split(",") if x.strip()]
    bad = [n for n in names if n not in PP_STAGE_NAMES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown preprocess stage(s) {bad}; valid: {list(PP_STAGE_NAMES)}")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"duplicate stage in {names}")
    return names

# Keys whose meaning is fixed across every family that exposes them.
SHARED_KEYS = frozenset({
    "profile",            # named family preset (checkpoint + defaults)
    "weights",            # explicit checkpoint path override
    "graph",              # interpreter graph JSON (toflow runtime)
    "strength",           # primary output dial, 1.0 = full trained effect
    "luma_strength",      # split output blend, luma half
    "chroma_strength",    # split output blend, chroma half
    "denoise_strength",   # checkpoint-interpolation denoise dial (dni)
    "dtype",              # float16 | float32
    "scale",              # spatial scale factor when it is a free parameter
    "window",             # the family's temporal-extent knob
    "trim",               # warm-up frames trimmed at window joins
    "passes",             # in-stage iteration count
    "ensemble",           # 8-way geometric self-ensemble
    "flow",               # optical-flow engine: spynet | zero | vt
    "flow_weights",       # stock SpyNet checkpoint path
    "flow_scale",         # flow-pyramid resolution: full | half | quarter
    "flow_consistency",   # forward-backward consistency masking strength
    "history_strength",
    "history_gate",
    "history_cleanup",
    "history_gate_drop",
    "history_risk_decay",
    "history_static_cap",
    "guard",
    "guard_threshold",
    "guard_fast_fraction",
    "guard_lockout",
    "guard_ramp",
    "guard_fall",
})

# Family-specific keys: concepts no other family shares. Adding a key here
# is a deliberate vocabulary decision (planning 07-cli-vocabulary.md).
FAMILY_KEYS: dict[str, frozenset[str]] = {
    "mc": frozenset({"sigma", "gate", "clamp", "occlusion", "confidence"}),
    "fbcnn": frozenset({"quality", "quality_fallback", "gop"}),
    "safmn": frozenset({"safm_up", "pool_clamp"}),
    "pvdd": frozenset({"noise_preset", "noise_variance"}),
    "realbasicvsr": frozenset({"clean_iters", "clean_threshold",
                               "residual_strength"}),
    "nafnet": frozenset({"pool"}),
    "deflicker": frozenset({"band", "frac", "max_fix", "jitter", "gop"}),
    "level": frozenset({"deadband"}),
    "denoise": frozenset({"first"}),
    # Conditioning subsystems and foundation prefixes lock their keys the
    # same way so the whole prefixed surface stays deliberate.
    "noise_map": frozenset({"gain", "debug", "refresh", "masking",
                            "motion_cap", "floor_mode", "floor", "pulse",
                            "upsample"}),
    "deblock_map": frozenset({"gain"}),
    "conform": frozenset({"fps"}),
    "cut": frozenset({"detect", "threshold", "log"}),
    "crop": frozenset({"bars", "aspect", "anchor", "offset"}),
    "sanitize_edges": frozenset({"fill", "feather"}),
    "gop": frozenset({"align", "min_window", "max_window"}),
}


@dataclass(frozen=True)
class Opt:
    """One canonical CLI option and its parse shape."""

    flag: str
    group: str
    help: str
    dest: str | None = None
    kind: str = "store"                     # "store" | "flag" (store_true)
    type: Callable | None = None            # None = str
    default: object = None
    choices: tuple | None = None
    # Named checkpoint tokens owned by a slot selector whose key is not
    # literally ``profile`` (for example --restore and --nafnet).
    profile_tokens: tuple[str, ...] = ()
    metavar: str | None = None
    required: bool = False
    family: str | None = None               # prefix for conformance checks
    key: str | None = None                  # vocabulary key (flag suffix)
    # Dest is a Settings field: the flag participates in the settings
    # trifecta (default < env < TOML < CLI), so its argparse default must
    # stay None for the env/TOML layers to show through.
    settings_backed: bool = False

    @property
    def resolved_dest(self) -> str:
        return self.dest or self.flag.lstrip("-").replace("-", "_")


def vocabulary_violations(registry: list[Opt]) -> list[str]:
    """Return conformance problems in a registry (empty = conformant).

    - a family option's key must be a shared key or registered for that
      family (slot/selector rows carry ``key=None`` and are exempt);
    - a settings-backed row must not declare a non-None argparse default;
    - canonical flags must stay unique.
    """
    problems: list[str] = []
    seen_flags: dict[str, str] = {}
    for opt in registry:
        if opt.flag in seen_flags:
            problems.append(
                f"{opt.flag} defined by both {seen_flags[opt.flag]} "
                f"and {opt.flag}")
        seen_flags[opt.flag] = opt.flag
        if opt.family is not None and opt.key is not None:
            allowed = SHARED_KEYS | FAMILY_KEYS.get(opt.family, frozenset())
            if opt.key not in allowed:
                problems.append(
                    f"{opt.flag}: key {opt.key!r} is neither a shared "
                    f"vocabulary key nor registered for family "
                    f"{opt.family!r}")
        if opt.settings_backed and opt.default is not None:
            problems.append(
                f"{opt.flag}: settings-backed options must default to None "
                f"so env/TOML layers apply")
    return problems
