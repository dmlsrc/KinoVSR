"""Build the argparse parser from the option registry.

The parser is generated, not hand-written: every option is an
:class:`~kinovsr.cli.options.Opt` row in :mod:`kinovsr.cli._registry`.
Canonical flags show in ``--help`` and are the only accepted spellings.
Settings-backed rows double as the CLI layer of the settings trifecta, so the
settings group only adds flags for fields the registry does not already own.
"""

from __future__ import annotations

import argparse

from kinovsr.settings import add_argparse_args

from ._registry import REGISTRY
from .options import Opt

_DESCRIPTION = """\
Read video and pump frames through KinoVSR spatial/temporal processing.
Writes the upscaled MP4 directly via AVAssetWriter - no PNG round-trip, no
disk WAV by default.

Usage
-----
    # VSR an existing clip. Add --audio to carry the source file's audio track
    # through to the upscaled MP4.
    kinovsr --video clip.mp4 \\
        --output-dir outputs/vsr/run2 --upscale balanced --audio

    # Process only the middle: upscale the [5s, 8s) window of a long clip.
    # --start/--end (and --max-frames) accept frames (120), seconds (5s / 1.5),
    # or a clock string (0:05, 1:02:03). --video seeks natively to the window.
    kinovsr --video clip.mp4 \\
        --output-dir outputs/vsr/run3 --start 0:05 --end 0:08 --audio

Option vocabulary
-----------------
Processor options are --<family>-<key> with one shared key vocabulary:
profile (named preset), weights (checkpoint path), strength, dtype, window,
trim, passes, ensemble, flow, flow-weights, history-*, guard-*. The same
word always means the same concept in every family. Chain-level dials
(--denoise-strength, --deblock-strength) distribute positionally over a
comma-chain; a family-level flag overrides the chain value for that family.

Known limitation on edited footage
----------------------------------
`--upscale balanced` chains previous-frame state through VSR for temporal
coherence. Across a hard cut that's the wrong context and can produce
ghosting around the cut frame. Enable `--cut-detect` to reset the chain at
hard cuts.
"""


def settings_owned_dests() -> frozenset[str]:
    """Registry dests that are Settings fields (skip in the settings group)."""
    return frozenset(o.resolved_dest for o in REGISTRY if o.settings_backed)


def _add_option(group: argparse._ArgumentGroup, opt: Opt) -> None:
    dest = opt.resolved_dest
    common: dict = {"dest": dest}
    if opt.kind == "flag":
        common["action"] = "store_true"
    else:
        if opt.type is not None:
            common["type"] = opt.type
        if opt.choices is not None:
            common["choices"] = opt.choices
        if opt.metavar is not None:
            common["metavar"] = opt.metavar
    group.add_argument(
        opt.flag, default=opt.default, required=opt.required,
        help=opt.help, **common)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinovsr",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    groups: dict[str, argparse._ArgumentGroup] = {}
    for opt in REGISTRY:
        group = groups.get(opt.group)
        if group is None:
            group = parser.add_argument_group(opt.group)
            groups[opt.group] = group
        _add_option(group, opt)

    config_group = parser.add_argument_group("Configuration Files")
    config_group.add_argument(
        "--base-config", action="append", default=[], metavar="TOML",
        help=(
            "Base TOML config file; repeatable, applied in order before "
            "--config. In M2 only the [settings] table is consumed (the "
            "global settings trifecta: defaults < env < base TOML(s) < "
            "specific TOML < CLI flags); pipeline/stage tables arrive with "
            "the M3 builder."
        ),
    )
    config_group.add_argument(
        "--config", metavar="TOML", default=None,
        help="Specific TOML overlay applied after every --base-config.",
    )

    add_argparse_args(parser, skip=settings_owned_dests())
    return parser


def validate_args(parser: argparse.ArgumentParser,
                  args: argparse.Namespace) -> None:
    """Cross-option checks that argparse cannot express per-flag."""
    if not args.output_dir and not (args.probe_noise and args.video):
        parser.error(
            "--output-dir is required unless --probe-noise is used with --video")
    if not args.output_dir and (args.save_pre_frames or args.save_post_frames
                                or args.save_audio_sidecar):
        parser.error(
            "--save-pre-frames/--save-post-frames/--save-audio-sidecar "
            "require --output-dir")
    if args.save_audio_sidecar and not args.audio:
        parser.error("--save-audio-sidecar requires --audio")
    if args.video_chunk_size < 1:
        parser.error("--video-chunk-size must be >= 1")
    if args.gop_min_window < 1:
        parser.error("--gop-min-window must be >= 1")
    if args.gop_max_window < 1:
        parser.error("--gop-max-window must be >= 1")
    if args.gop_min_window > args.gop_max_window:
        parser.error(
            "--gop-min-window must be <= --gop-max-window")
    if args.upscale == "realbasicvsr":
        if args.realbasicvsr_window < 1:
            parser.error("--realbasicvsr-window must be >= 1")
        if args.realbasicvsr_trim < 0:
            parser.error("--realbasicvsr-trim must be >= 0")
        if (args.realbasicvsr_trim
                and args.realbasicvsr_window <= 2 * args.realbasicvsr_trim):
            parser.error(
                "--realbasicvsr-window must be greater than "
                "2*--realbasicvsr-trim; use --realbasicvsr-trim 0 for "
                "reference-like chunks")
    if args.upscale == "realviformer":
        if args.realviformer_window < 0:
            parser.error("--realviformer-window must be >= 0")
        if args.realviformer_history_strength < 0:
            parser.error("--realviformer-history-strength must be >= 0")
        if not 0.0 <= args.realviformer_history_cleanup <= 1.0:
            parser.error("--realviformer-history-cleanup must be in [0, 1]")
        if not 0.0 <= args.realviformer_history_gate_drop <= 1.0:
            parser.error("--realviformer-history-gate-drop must be in [0, 1]")
        if not 0.0 <= args.realviformer_history_risk_decay < 1.0:
            parser.error("--realviformer-history-risk-decay must be in [0, 1)")
        if not 0.0 <= args.realviformer_history_static_cap <= 1.0:
            parser.error("--realviformer-history-static-cap must be in [0, 1]")
