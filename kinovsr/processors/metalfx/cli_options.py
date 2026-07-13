"""CLI option rows owned by the MetalFX family."""

from kinovsr.cli.options import Opt

METALFX_OPTIONS = [
    Opt(flag='--metalfx-scale',
        group='Spatial Upscalers',
        type=int,
        default=2,
        choices=(2, 3, 4),
        metavar='N',
        family='metalfx',
        key='scale',
        help='Scale factor for --upscale metalfx (default 2). The scaler network upscales to any of 2x/3x/4x in one encode; unlike the checkpoint families the factor is a free parameter, not implied by weights.'),
]

__all__ = ["METALFX_OPTIONS"]
