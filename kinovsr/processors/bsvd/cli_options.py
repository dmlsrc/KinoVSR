"""CLI option rows owned by the BSVD family."""

from kinovsr.cli.options import Opt

BSVD_STRENGTH_OPTIONS = [
    Opt(flag='--bsvd-strength',
        group='Denoise And Noise Maps',
        type=float,
        metavar='S',
        family='bsvd',
        key='strength',
        help='Strength for --denoise bsvd (noise sigma, mapped onto sigma_255 in [5, 55]), overriding the positional --denoise-strength value for this family (default: the chain value).'),
]

BSVD_OPTIONS = [
    Opt(flag='--bsvd-weights',
        group='Denoise And Noise Maps',
        metavar='PATH',
        family='bsvd',
        key='weights',
        settings_backed=True,
        help='Override BSVD weights (.safetensors) for --denoise bsvd. Optional - defaults to the local --bsvd-profile weights (or $BSVD_WEIGHTS). Convert a .pth with kinovsr weights convert --param-key params.'),
    Opt(flag='--bsvd-profile',
        aliases=('--bsvd-variant',),
        group='Denoise And Noise Maps',
        default='c64',
        choices=('c64', 'c32'),
        family='bsvd',
        key='profile',
        help='Which local BSVD model token for --denoise bsvd. c64 matches the public unblind test config; c32 is a smaller mirror checkpoint with weaker provenance. Ignored when --bsvd-weights is given.'),
    Opt(flag='--bsvd-dtype',
        group='Denoise And Noise Maps',
        default='float16',
        choices=('float16', 'float32'),
        family='bsvd',
        key='dtype',
        help='MLX dtype for --denoise bsvd (default float16; use float32 for parity probes).'),
]

__all__ = ["BSVD_STRENGTH_OPTIONS", "BSVD_OPTIONS"]
