"""CLI option rows owned by the spatial denoiser family."""

from kinovsr.cli.options import Opt

SPATIAL_OPTIONS = [
    Opt(flag='--spatial-strength',
        group='Denoise And Noise Maps',
        type=float,
        metavar='S',
        family='spatial',
        key='strength',
        help='Strength for --denoise spatial, overriding the positional --denoise-strength value for this family (default: the chain value).'),
]

__all__ = ["SPATIAL_OPTIONS"]
