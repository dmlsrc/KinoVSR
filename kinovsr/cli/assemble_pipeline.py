"""Flag -> [pipeline] config assembler (M6 root 2).

Maps the flag surface onto the same typed pipeline config a hand-written
TOML expresses, encoding the harness's semantics:

- the frame-loop stage order: crop -> sanitize fill -> square-pixels ->
  cut detection -> preprocess slots -> upscale -> frame-rate conversion;
- the preprocess slot order (planning C5): restore and deflicker first,
  deblock before denoise (``--denoise-first`` swaps them), nafnet last;
  ``--preprocess-order`` sets an explicit order and any enabled stage it
  omits is appended in the default order;
- comma-chained selectors (``--denoise mc,bsvd``) as one stage each, in
  order;
- junk-edge TRIM folded into the crop (the crop family's ``trim`` key;
  ``auto`` values resolve in the probe pass), with the harness's +1px
  even-bump applied here for explicit counts;
- every ``--<family>-<key>`` dial routed into its stage table via the
  registry's ``family=``/``key=`` metadata, with the shared groups
  (denoise/deblock/restore dials, noise-map and deblock-map
  conditioning) distributed across their slot's members exactly as the
  harness applied them. Dials for families not in the chain are ignored,
  as the harness ignored them.

Run-level flags (trim window, gop windowing, cut log, encode, audio,
dumps) are NOT mapped here - they already thread through
``process_video_file`` as run options.
"""

from __future__ import annotations

import logging
from typing import Any

from kinovsr.config import ConfigError

from .options import PP_STAGE_NAMES

_log = logging.getLogger(__name__)

# --upscale values that select the native VideoToolbox scaler; everything
# else (bar "none") is a learned family name.
_VT_MODES = ("fast", "balanced", "image")

# --nafnet value -> (capability, profile)
_NAFNET = {
    "gopro": ("deblur", "gopro"), "gopro32": ("deblur", "gopro32"),
    "sidd": ("denoise", "sidd"), "sidd32": ("denoise", "sidd32"),
    "reds": ("restore", "reds"),
}

# Denoise-slot families whose stage tables accept the noise-map
# conditioning keys (processors/conditioning.py consumers).
_NOISE_MAP_FAMILIES = ("mc", "bsvd", "fastdvdnet", "pvdd")
# Deblock-slot families accepting the deblock-map (blockiness) keys.
_DEBLOCK_MAP_FAMILIES = ("stdf", "fbcnn")

# Flag-groups consumed structurally below; the generic registry
# distribution skips them.
_SPECIAL_FAMILIES = frozenset({
    "crop", "sanitize_edges", "cut", "gop", "denoise", "deblock",
    "restore", "noise_map", "deblock_map", "toflow_sr",
})


def _set_flags(options: Any) -> list:
    """Registry rows carrying family metadata whose value the user set
    (differs from the registry default and is not an unset None)."""
    from ._registry import REGISTRY

    rows = []
    for opt in REGISTRY:
        family = getattr(opt, "family", None)
        if family is None:
            continue
        value = getattr(options, opt.resolved_dest, opt.default)
        if value is None or value == opt.default:
            continue
        rows.append((opt, value))
    return rows


def _coerce(opt: Any, value: Any) -> Any:
    """Untyped registry flags parse as strings; numeric ones become the
    numbers the stage parsers expect ('1.0' -> 1.0), tokens stay strings."""
    if getattr(opt, "type", None) is None and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _slot_values(opt: Any, raw_value: Any,
                 stage_names: list[str]) -> list[Any]:
    """Type and distribute a shared slot value over its selected stages.

    One scalar broadcasts. A comma list is positional and must have exactly
    one value per stage, matching the CLI help contract.
    """
    if not stage_names:
        return []
    if not isinstance(raw_value, str) or "," not in raw_value:
        value = _coerce(opt, raw_value)
        return [value] * len(stage_names)
    tokens = [token.strip() for token in raw_value.split(",")]
    if any(not token for token in tokens):
        raise ConfigError(
            f"{opt.flag} contains an empty positional value")
    if len(tokens) != len(stage_names):
        raise ConfigError(
            f"{opt.flag} has {len(tokens)} values for "
            f"{len(stage_names)} selected stages")
    return [_coerce(opt, token) for token in tokens]


def _bump_even(edges: list[int], width: int, height: int) -> list[int]:
    """The harness's evenness bump: eat one content pixel bottom/right so
    the active area keeps even dimensions."""
    if (height - edges[0] - edges[1]) % 2:
        edges[1] += 1
    if (width - edges[2] - edges[3]) % 2:
        edges[3] += 1
    return edges


def assemble_pipeline(options: Any, *, width: int, height: int) -> dict:
    """The [pipeline] config equivalent of a flag invocation.

    ``width``/``height`` are the probed source dimensions (the evenness
    bump for explicit crop counts needs them, exactly as the harness
    bumped against the probed size).
    """
    from kinovsr.config.helpers import parse_edge_counts

    config: dict[str, Any] = {}
    pipeline: list[str] = []
    slot_members: dict[str, list[str]] = {}   # slot -> stage names, in order

    def add(name: str, table: dict, slot: str | None = None) -> None:
        config[name] = table
        pipeline.append(name)
        if slot is not None:
            slot_members.setdefault(slot, []).append(name)

    # ---- timeline: explicit CFR conform (ingest normalization) -----------
    # First in the chain: it retains one frame of lookahead, which the
    # bounded zero-copy pools downstream (upscale -> writer) do not budget
    # for, and dropped frames should not be processed at all.
    conform_cfr = getattr(options, "conform_cfr", None)
    if conform_cfr and options.target_fps:
        raise ConfigError(
            "--conform-cfr and --target-fps are both timeline "
            "normalizations; pick one (--target-fps interpolates onto its "
            "grid, --conform-cfr duplicates/drops original frames)")
    if conform_cfr:
        add("conform", {"processor": "conform", "fps": str(conform_cfr)})

    # ---- geometry: crop (bars + trim + aspect), sanitize fill, square ----
    crop_table: dict[str, Any] = {"processor": "crop"}
    bars_spec = options.crop_bars
    trim_spec = (options.sanitize_edges
                 if options.sanitize_edges
                 and options.sanitize_edges_fill == "trim" else None)
    if bars_spec and bars_spec != "auto" and trim_spec and trim_spec != "auto":
        # Both explicit: fold with the harness's two-step bump (bars are
        # bumped against the source, trim against the post-bars area).
        bars = _bump_even(list(parse_edge_counts(bars_spec)), width, height)
        inner_w = width - bars[2] - bars[3]
        inner_h = height - bars[0] - bars[1]
        trim = _bump_even(list(parse_edge_counts(trim_spec)), inner_w, inner_h)
        crop_table["bars"] = ",".join(
            str(b + t) for b, t in zip(bars, trim, strict=True))
    else:
        if bars_spec:
            crop_table["bars"] = (
                bars_spec if bars_spec == "auto" else ",".join(map(str,
                    _bump_even(list(parse_edge_counts(bars_spec)),
                               width, height))))
        if trim_spec:
            # With auto bars ahead of an explicit trim the post-bars area
            # is unknown here; bump against the source (the probe re-evens
            # detected values itself).
            crop_table["trim"] = (
                trim_spec if trim_spec == "auto" else ",".join(map(str,
                    _bump_even(list(parse_edge_counts(trim_spec)),
                               width, height))))
    if options.crop_aspect:
        crop_table["aspect"] = options.crop_aspect
        crop_table["anchor"] = options.crop_anchor
        crop_table["offset"] = options.crop_offset
    if len(crop_table) > 1:
        add("crop", crop_table)

    if options.sanitize_edges and options.sanitize_edges_fill != "trim":
        add("sanitize", {
            "processor": "sanitize_edges",
            "edges": options.sanitize_edges,
            "fill": options.sanitize_edges_fill,
            "feather": options.sanitize_edges_feather,
        })

    if options.square_pixels:
        add("square", {"processor": "square_pixels"})

    if options.cut_detect != "off":
        add("cut", {"processor": "cut_detect",
                    "detect": options.cut_detect,
                    "threshold": options.cut_threshold})

    # ---- preprocess slots (C5 default order + explicit override) --------
    if options.preprocess_order:
        order = list(options.preprocess_order)
    else:
        order = (["denoise", "deblock"] if options.denoise_first
                 else ["deblock", "denoise"])
        order = ["restore", "deflicker", *order]
    for slot in PP_STAGE_NAMES:      # append any slot not listed
        if slot not in order:
            order.append(slot)

    def chain(value: str) -> list[str]:
        return [x.strip() for x in str(value).split(",")
                if x.strip() and x.strip() != "off"]

    for slot in order:
        if slot == "restore" and options.restore != "off":
            for variant in chain(options.restore):
                add(f"restore_{variant}", {
                    "processor": "basicvsrpp", "capability": "restore",
                    "profile": variant}, slot="restore")
        elif slot == "deflicker" and options.deflicker != "off":
            add("deflicker", {"processor": "deflicker"}, slot="deflicker")
        elif slot == "deblock" and options.deblock != "off":
            for family in chain(options.deblock):
                table = {
                    "processor": family,
                    "capability": "deblock",
                }
                add(f"deblock_{family}", table, slot="deblock")
        elif slot == "denoise" and options.denoise != "off":
            for family in chain(options.denoise):
                table = {
                    "processor": family,
                    "capability": "denoise",
                }
                add(f"denoise_{family}", table, slot="denoise")
        elif slot == "nafnet" and options.nafnet != "off":
            capability, profile = _NAFNET[options.nafnet]
            add("nafnet", {"processor": "nafnet", "capability": capability,
                           "profile": profile}, slot="nafnet")

    # ---- upscale + frame-rate conversion ---------------------------------
    if options.upscale in _VT_MODES:
        add("upscale", {"processor": "videotoolbox",
                        "capability": "upscale",
                        "profile": options.upscale}, slot="upscale")
    elif options.upscale != "none":
        table = {"processor": options.upscale, "capability": "upscale"}
        add("upscale", table, slot="upscale")

    if options.target_fps:
        add("fps", {"processor": "videotoolbox",
                    "capability": "interpolate",
                    "profile": options.temporal_mode,
                    "target_fps": options.target_fps})

    # ---- registry-driven dial routing ------------------------------------
    def members(slot: str, families: tuple[str, ...] | None = None) -> list:
        names = slot_members.get(slot, ())
        if families is None:
            return list(names)
        return [n for n in names if config[n]["processor"] in families]

    # Shared slot fills first, family-specific dials second, so a family
    # flag (--bsvd-strength) deterministically overrides its slot's shared
    # dial (--denoise-strength) regardless of registry row order.
    rows = _set_flags(options)
    ordered = ([r for r in rows if r[0].family in _SPECIAL_FAMILIES]
               + [r for r in rows if r[0].family not in _SPECIAL_FAMILIES])
    for opt, raw_value in ordered:
        family, key = opt.family, getattr(opt, "key", None)
        value = _coerce(opt, raw_value)
        if family in ("gop",):
            continue                      # run-level, threaded separately
        if family == "cut":
            if key in ("detect", "threshold") and "cut" in config:
                config["cut"][key] = value
            continue                      # log is run-level
        if family == "noise_map":
            if key == "debug":
                continue                  # run-level (noise_map_debug)
            stage_key = "noise_map" if key is None else f"noise_map_{key}"
            for name in members("denoise", _NOISE_MAP_FAMILIES):
                config[name][stage_key] = value
            continue
        if family == "deblock_map":
            stage_key = "deblock_map" if key is None else f"deblock_map_{key}"
            for name in members("deblock", _DEBLOCK_MAP_FAMILIES):
                config[name][stage_key] = value
            continue
        if family in ("denoise", "deblock", "restore"):
            if key is None or key == "first":
                continue                  # the selector / ordering itself
            names = members(family)
            for name, slot_value in zip(
                    names, _slot_values(opt, raw_value, names), strict=True):
                config[name][key] = slot_value
            continue
        if family == "toflow_sr":
            up = config.get("upscale")
            if up is not None and up["processor"] == "toflow":
                up[key] = value
            continue
        if family in _SPECIAL_FAMILIES or key is None:
            continue                      # geometry built structurally above
        # Family-specific dial: every stage of that processor gets it
        # (family flags override the shared-slot fills applied above
        # because the registry lists them after the shared rows).
        for name in pipeline:
            if config[name]["processor"] == family:
                config[name][key] = value

    # ---- broadcast conditioning is capability-scoped, not a build error --
    # The shared --noise-map dial fans out to every capable denoiser in the
    # slot; a stage whose loaded model cannot take a map (a blind PVDD
    # checkpoint) is skipped with a warning instead of refusing the whole
    # chain. An explicit per-stage noise_map in a hand-written [pipeline]
    # config still hits the family's loud guard.
    for name in pipeline:
        table = config[name]
        if table.get("processor") != "pvdd":
            continue
        if "level" in (table.get("profile") or "pvdd"):
            continue
        stripped = [k for k in table if k.startswith("noise_map")]
        if stripped:
            for k in stripped:
                del table[k]
            _log.warning(
                "noise map does not apply to %s: PVDD profile %r is blind "
                "(use --pvdd-profile pvdd_level for map conditioning); the "
                "stage runs unconditioned",
                name, table.get("profile", "pvdd"))
    if getattr(options, "noise_map", None) == "auto" and not any(
            config[n].get("noise_map") == "auto" for n in pipeline):
        raise ConfigError(
            "--noise-map auto applies to no stage in this chain (capable: "
            "mc, bsvd, fastdvdnet, and pvdd with a level profile)")

    config["pipeline"] = pipeline
    return config


__all__ = ["assemble_pipeline"]
