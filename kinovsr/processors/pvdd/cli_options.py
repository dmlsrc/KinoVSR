"""CLI option rows owned by the PVDD family."""

from kinovsr.cli.options import Opt

# PVDD has no dry/wet strength dial: its intensity is noise conditioning
# (--pvdd-noise-preset / --pvdd-noise-variance on the level variants), so
# there is deliberately no --pvdd-strength row here. The shared
# --denoise-strength broadcast skips PVDD stages with a warning naming the
# real dials; see _SLOT_KEY_UNSUPPORTED in cli/assemble_pipeline.py.

PVDD_OPTIONS = [
    Opt(flag='--pvdd-profile',
        group='Denoise And Noise Maps',
        default='pvdd',
        choices=('pvdd', 'crvd', 'davis', 'pvdd_level', 'pvdd_raw', 'pvdd_raw_level'),
        family='pvdd',
        key='profile',
        help='Which local PVDD model for --denoise pvdd. pvdd (default) = real-world sRGB blind; crvd = real high-ISO sensor noise; davis = synthetic-AWGN sibling (baseline, behaves like fastdvdnet); pvdd_level = noise-level dial (non-blind, see --pvdd-noise-*); pvdd_raw / pvdd_raw_level = packed Bayer (need a raw pipeline, not sRGB video). Ignored when --pvdd-weights is given.'),
    Opt(flag='--pvdd-weights',
        group='Denoise And Noise Maps',
        metavar='PATH',
        family='pvdd',
        key='weights',
        settings_backed=True,
        help='Override PVDD weights (.safetensors) for --denoise pvdd. Optional - defaults to the local --pvdd-profile weights (or $PVDD_WEIGHTS). Not bundled; convert a .pth with kinovsr weights convert (see kinovsr/processors/pvdd/weights/README.md).'),
    Opt(flag='--pvdd-window',
        group='Denoise And Noise Maps',
        type=int,
        default=10,
        metavar='N',
        family='pvdd',
        key='window',
        help="Sliding-window length (frames) for --denoise pvdd's bidirectional recurrence (default 10). Larger windows give more temporal context at higher cost; use --pvdd-trim to overlap windows."),
    Opt(flag='--pvdd-trim',
        group='Denoise And Noise Maps',
        type=int,
        default=0,
        metavar='N',
        family='pvdd',
        key='trim',
        help='Overlap frames trimmed from each --denoise pvdd window edge (default 0 = reference-like non-overlapping chunks). Must be < window/2.'),
    Opt(flag='--pvdd-noise-preset',
        group='Denoise And Noise Maps',
        default='M',
        choices=('off', 'S', 'M', 'L'),
        family='pvdd',
        key='noise_preset',
        help='Noise-level preset for the pvdd_level variants (non-blind). S/M/L map to the reference noise-variance levels (0.00069 / 0.0022 / 0.0055); M is default. off disables (needs --pvdd-noise-variance). Ignored by blind variants.'),
    Opt(flag='--pvdd-noise-variance',
        group='Denoise And Noise Maps',
        type=float,
        metavar='V',
        family='pvdd',
        key='noise_variance',
        help='Explicit noise-variance value for the pvdd_level variants, overriding --pvdd-noise-preset. This is variance (sigma^2), not sigma. Ignored by blind variants.'),
    Opt(flag='--pvdd-dtype',
        group='Denoise And Noise Maps',
        default='float16',
        choices=('float16', 'float32'),
        family='pvdd',
        key='dtype',
        help='MLX dtype for --denoise pvdd (default float16; use float32 for parity probes).'),
]

__all__ = ["PVDD_OPTIONS"]
