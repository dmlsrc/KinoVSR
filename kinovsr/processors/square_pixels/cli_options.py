"""CLI option rows owned by the square-pixels family."""

from kinovsr.cli.options import Opt

SQUARE_PIXELS_OPTIONS = [
    Opt(flag='--square-pixels',
        group='Trim, Cropping, And Cuts',
        kind='flag',
        default=False,
        help='Resample anamorphic sources to square pixels (1:1 pixel aspect) before processing: a horizontal-only resample at SOURCE resolution (Lanczos-3, GPU-resident precomputed plan) -- the cheapest point, and the upscaler re-synthesizes the mild resample softness -- with the output tagged 1:1 for PAR-ignorant players and toolchains. Also a mild DISPLAY-domain sharpness win (~7 percent measured): the anamorphic stretch must happen somewhere, and pre-SR (here, then re-synthesized by the net) beats post-SR in the player, which dilutes rendered detail. Default behavior (off) passes the source pixel aspect through losslessly instead. No-op on square-pixel sources.'),
]

__all__ = ["SQUARE_PIXELS_OPTIONS"]
