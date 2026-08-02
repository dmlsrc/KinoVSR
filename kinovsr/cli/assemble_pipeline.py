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
  registry's targeting metadata, with the shared groups
  (denoise/deblock/restore dials, noise-map and deblock-map
  conditioning) distributed across their slot's members exactly as the
  harness applied them. A set dial with no matching stage is an error.

Run-level flags (trim window, gop windowing, cut log, encode, audio,
dumps) are NOT mapped here - they already thread through
``process_video_file`` as run options.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from kinovsr.config import (
    ConfigError,
    apply_set_overrides,
    merge_configs,
    validate_config,
)

from .options import PP_STAGE_NAMES, option_roles

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

# Shared slot dials not every member family accepts: the broadcast skips
# these (family, key) -> processors with a warning (an explicit per-stage
# key in a hand-written [pipeline] config still hits the family's loud
# guard). PVDD has no dry/wet dial; its intensity is noise conditioning.
_SLOT_KEY_UNSUPPORTED = {
    ("denoise", "strength"): ("pvdd",),
}

_SLOT_FAMILIES = frozenset({"denoise", "deblock", "restore"})
_BROADCAST_FAMILIES = frozenset({"noise_map", "deblock_map"})


def compositional_flags(options: Any) -> list[str]:
    """Set options that would compete with a TOML-owned stage list.

    The ownership set lives on the parser registry rows consumed by this
    assembler.  That keeps parsing, assembly, and the conflict guard on one
    source of truth.
    """
    from ._registry import REGISTRY

    return [
        opt.resolved_dest
        for opt in REGISTRY
        if opt.compositional
        and getattr(options, opt.resolved_dest, opt.default) != opt.default
    ]


def _set_flags(options: Any) -> list:
    """Registry rows carrying family metadata whose value the user set
    (differs from the registry default and is not an unset None)."""
    from ._registry import REGISTRY

    rows = []
    for opt in REGISTRY:
        if "dial" not in option_roles(opt):
            continue
        value = getattr(options, opt.resolved_dest, opt.default)
        if value is None or value == opt.default:
            continue
        rows.append((opt, value))
    return rows


def _coerce(opt: Any, value: Any) -> Any:
    """Untyped registry flags parse as strings; numeric ones become the
    numbers the stage parsers expect ('1.0' -> 1.0), tokens stay strings.

    An ``off|on`` choice pair is a boolean spelled for the command line:
    the stage parsers type those keys with ``typed_value(..., bool, ...)``,
    which refuses a string, and the only routable value is the non-default
    one, so handing the family 'on' verbatim failed every such flag on the
    only value worth passing. A wider choice set (``off|on|auto``) is a real
    token set and stays a string.
    """
    if getattr(opt, "type", None) is None and isinstance(value, str):
        if tuple(getattr(opt, "choices", None) or ()) == ("off", "on"):
            return value == "on"
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


def _stage_capability(name: str, table: Mapping[str, Any]) -> str:
    """Return a stage's effective capability without parsing its config."""
    token = table.get("capability")
    if isinstance(token, str):
        return token

    from kinovsr.pipeline.builder import _resolve_capability
    from kinovsr.processors import get_factory

    factory = get_factory(table.get("processor"))
    return _resolve_capability(
        name, factory, None, table.get("profile")
    ).value


def apply_flag_dials(config: Mapping[str, Any], options: Any) -> dict[str, Any]:
    """Overlay set CLI dials on stages from either chain producer.

    Slot dials target effective capabilities in pipeline order.  Family dials
    target processors, with the few multi-capability exceptions declared on
    their registry rows.  The input mapping and its stage tables are not
    mutated.
    """
    out = dict(config)
    pipeline = list(out.get("pipeline", []))
    for name in set(pipeline):
        table = out.get(name)
        if isinstance(table, Mapping):
            out[name] = dict(table)

    def members(
        capability: str | None = None,
        processors: tuple[str, ...] | None = None,
    ) -> list[str]:
        names = []
        for name in pipeline:
            table = out.get(name)
            if not isinstance(table, Mapping):
                continue
            if processors is not None and table.get("processor") not in processors:
                continue
            if capability is not None and _stage_capability(name, table) != capability:
                continue
            names.append(name)
        return names

    # Broadcast/slot fills precede family-specific flags, so the narrower flag
    # wins regardless of registry fragment order.
    rows = _set_flags(options)
    broad = _SLOT_FAMILIES | _BROADCAST_FAMILIES
    ordered = ([row for row in rows if row[0].family in broad]
               + [row for row in rows if row[0].family not in broad])
    conditioned: list[tuple[Any, str]] = []
    for opt, raw_value in ordered:
        family, key = opt.family, opt.key
        value = _coerce(opt, raw_value)
        if family == "noise_map":
            stage_key = "noise_map" if key is None else f"noise_map_{key}"
            names = members("denoise", _NOISE_MAP_FAMILIES)
            if not names:
                raise ConfigError(
                    f"{opt.flag} applies to no stage accepting a noise map")
            for name in names:
                out[name][stage_key] = value
            conditioned.append((opt, stage_key))
            continue
        if family == "deblock_map":
            stage_key = "deblock_map" if key is None else f"deblock_map_{key}"
            names = members("deblock", _DEBLOCK_MAP_FAMILIES)
            if not names:
                raise ConfigError(
                    f"{opt.flag} targets no deblock stage accepting a block map")
            for name in names:
                out[name][stage_key] = value
            continue
        if family in _SLOT_FAMILIES:
            if key is None:
                continue
            names = members(family)
            if not names:
                raise ConfigError(
                    f"{opt.flag} targets no {family} stage in the pipeline")
            blocked = _SLOT_KEY_UNSUPPORTED.get((family, key), ())
            skipped = []
            for name, slot_value in zip(
                    names, _slot_values(opt, raw_value, names), strict=True):
                if out[name]["processor"] in blocked:
                    skipped.append(name)
                    continue
                out[name][key] = slot_value
            for name in skipped:
                _log.warning(
                    "%s does not apply to %s: PVDD has no dry/wet dial "
                    "(its intensity comes from noise conditioning: "
                    "--pvdd-noise-preset / --pvdd-noise-variance, or "
                    "--noise-map auto with --pvdd-profile pvdd_level); "
                    "the stage runs at full effect", opt.flag, name)
            if names and skipped and len(skipped) == len(names):
                raise ConfigError(
                    f"{opt.flag} applies to no stage in this chain")
            continue
        if key is None:
            continue
        if (family == "sanitize_edges" and key == "fill" and value == "trim"
                and getattr(options, "sanitize_edges", None)):
            continue                      # folded into the generated crop stage

        processor = opt.stage_processor or family
        capabilities = frozenset(opt.stage_capabilities)
        applied = 0
        for name in pipeline:
            table = out.get(name)
            if not isinstance(table, Mapping) or table.get("processor") != processor:
                continue
            if capabilities and _stage_capability(name, table) not in capabilities:
                continue
            if (processor == "crop" and key in {"anchor", "offset"}
                    and "aspect" not in table):
                continue
            out[name][key] = value
            applied += 1
        if not applied:
            target = processor
            if capabilities:
                target += " (" + ", ".join(sorted(capabilities)) + ")"
            raise ConfigError(f"{opt.flag} targets no {target} stage in the pipeline")

    # A broadcast noise map may be accepted by the family in general but not
    # by the selected blind PVDD profile.  Strip that one target, retaining the
    # existing warning/error contract for a broadcast that reaches nothing.
    for name in pipeline:
        table = out[name]
        if table.get("processor") != "pvdd":
            continue
        if "level" in (table.get("profile") or "pvdd"):
            continue
        stripped = [k for k in table if k.startswith("noise_map")]
        if stripped:
            for key in stripped:
                del table[key]
            _log.warning(
                "noise map does not apply to %s: PVDD profile %r is blind "
                "(use --pvdd-profile pvdd_level for map conditioning); the "
                "stage runs unconditioned",
                name, table.get("profile", "pvdd"))
    if getattr(options, "noise_map", None) == "auto" and not any(
            out[name].get("noise_map") == "auto" for name in pipeline):
        raise ConfigError(
            "--noise-map auto applies to no stage in this chain (capable: "
            "mc, bsvd, fastdvdnet, and pvdd with a level profile)")
    for opt, stage_key in conditioned:
        if not any(stage_key in out[name] for name in pipeline):
            raise ConfigError(f"{opt.flag} applies to no stage in this chain")
    return out


def _bump_even(edges: list[int], width: int, height: int) -> list[int]:
    """The harness's evenness bump: eat one content pixel bottom/right so
    the active area keeps even dimensions."""
    if (height - edges[0] - edges[1]) % 2:
        edges[1] += 1
    if (width - edges[2] - edges[3]) % 2:
        edges[3] += 1
    return edges


def assemble_pipeline(
    options: Any,
    *,
    width: int,
    height: int,
    base_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The [pipeline] config equivalent of a flag invocation.

    ``width``/``height`` are the probed source dimensions (the evenness
    bump for explicit crop counts needs them, exactly as the harness
    bumped against the probed size).
    """
    from kinovsr.config.helpers import parse_edge_counts

    config: dict[str, Any] = {}
    pipeline: list[str] = []

    def add(name: str, table: dict) -> None:
        # A family repeated in one chain (--denoise mc,mc) is a request for
        # two independently configured passes, so each instance needs its
        # own table. Sharing one would make the positional dial lists
        # (--denoise-strength 0.3,0.7) silently collapse to the last value.
        if name in config:
            suffix = 2
            while f"{name}_{suffix}" in config:
                suffix += 1
            name = f"{name}_{suffix}"
        config[name] = table
        pipeline.append(name)

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
        stage = {"processor": "cut_detect", "detect": options.cut_detect}
        if options.cut_threshold is not None:
            # Absent means the family resolves its per-mode default; the
            # modes' statistics live on different scales.
            stage["threshold"] = options.cut_threshold
        add("cut", stage)

    # ---- preprocess slots (C5 default order + explicit override) --------
    if options.preprocess_order:
        order = list(options.preprocess_order)
    else:
        order = (["denoise", "deblock"] if options.denoise_first
                 else ["deblock", "denoise"])
        # level runs first: a stable exposure improves every temporal
        # stage after it (deflicker's static verification, mc's gate,
        # the noise map, learned temporal denoisers).
        order = ["level", "restore", "deflicker", *order]
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
                    "profile": variant})
        elif slot == "level" and options.level != "off":
            add("level", {"processor": "level"})
        elif slot == "deflicker" and options.deflicker != "off":
            add("deflicker", {"processor": "deflicker"})
        elif slot == "deblock" and options.deblock != "off":
            for family in chain(options.deblock):
                table = {
                    "processor": family,
                    "capability": "deblock",
                }
                add(f"deblock_{family}", table)
        elif slot == "denoise" and options.denoise != "off":
            for family in chain(options.denoise):
                table = {
                    "processor": family,
                    "capability": "denoise",
                }
                add(f"denoise_{family}", table)
        elif slot == "nafnet" and options.nafnet != "off":
            capability, profile = _NAFNET[options.nafnet]
            add("nafnet", {"processor": "nafnet", "capability": capability,
                           "profile": profile})

    # ---- upscale + frame-rate conversion ---------------------------------
    if options.upscale in _VT_MODES:
        table = {
            "processor": "videotoolbox",
            "capability": "upscale",
            "profile": options.upscale,
        }
        if options.vt_sr_flow != "vt":
            if options.upscale != "balanced":
                raise ConfigError(
                    "--vt-sr-flow is only valid with --upscale balanced"
                )
            table["flow"] = options.vt_sr_flow
        add("upscale", table)
    elif options.upscale != "none":
        if options.vt_sr_flow != "vt":
            raise ConfigError(
                "--vt-sr-flow is only valid with --upscale balanced"
            )
        table = {"processor": options.upscale, "capability": "upscale"}
        add("upscale", table)
    elif options.vt_sr_flow != "vt":
        raise ConfigError(
            "--vt-sr-flow is only valid with --upscale balanced"
        )

    if options.target_fps:
        add("fps", {"processor": "videotoolbox",
                    "capability": "interpolate",
                    "profile": options.temporal_mode,
                    "target_fps": options.target_fps})

    config["pipeline"] = pipeline
    # A flag-owned chain still consumes matching TOML stage tables.  Generated
    # selectors win; table settings survive; explicit dials are overlaid last.
    composed = merge_configs(base_config or {}, config)
    return apply_flag_dials(composed, options)


def resolve_pipeline_config(
    options: Any,
    config: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Resolve either authoring surface through the same merge pipeline."""
    if "pipeline" in config:
        owned = compositional_flags(options)
        if owned:
            flags = ", ".join("--" + name.replace("_", "-") for name in owned)
            raise ConfigError(
                "a [pipeline] config owns the full chain; drop the flags "
                f"({flags}) or the pipeline table")
        resolved = apply_flag_dials(config, options)
    else:
        resolved = assemble_pipeline(
            options, width=width, height=height, base_config=config)
    resolved = apply_set_overrides(resolved, getattr(options, "set", None))
    validate_config(resolved)
    return resolved


__all__ = [
    "apply_flag_dials",
    "assemble_pipeline",
    "compositional_flags",
    "resolve_pipeline_config",
]
