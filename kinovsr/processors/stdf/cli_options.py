"""CLI option rows owned by the STDF family."""

from kinovsr.cli.options import Opt

STDF_OPTIONS = [
    Opt(flag='--stdf-weights',
        group='Deblock',
        metavar='PATH',
        family='stdf',
        key='weights',
        settings_backed=True,
        help='Override STDF weights (.safetensors) for --deblock stdf (or $STDF_WEIGHTS). Optional - defaults to the bundled --stdf-profile weights.'),
    Opt(flag='--stdf-profile',
        group='Deblock',
        choices=('mfqev2', 'vimeo90k'),
        family='stdf',
        key='profile',
        help='Which bundled STDF model for --deblock stdf. mfqev2 (default) = HEVC multi-QP; vimeo90k = All-Intra QP37. Ignored when --stdf-weights is given.'),
    Opt(flag='--stdf-strength',
        group='Deblock',
        type=float,
        metavar='S',
        family='stdf',
        key='strength',
        help="Scale STDF's residual for --deblock stdf, overriding the positional --deblock-strength value for this family (default: the chain value)."),
]

__all__ = ["STDF_OPTIONS"]
