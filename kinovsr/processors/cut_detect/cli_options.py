"""CLI option rows owned by the cut-detect family."""

from kinovsr.cli.options import Opt

CUT_OPTIONS = [
    Opt(flag='--cut-detect',
        group='Trim, Cropping, And Cuts',
        default='off',
        choices=('off', 'simple', 'hist'),
        family='cut',
        key='detect',
        help="Reset VSR's prev-frame chain at hard cuts. off = never reset (correct for continuous generated clips and single-shot sources). Only meaningful for edited --video input under --upscale balanced (which chains prev-frame state); a no-op under fast/image modes."),
    Opt(flag='--cut-threshold',
        group='Trim, Cropping, And Cuts',
        type=float,
        default=0.25,
        family='cut',
        key='threshold',
        help=''),
    Opt(flag='--cut-log',
        group='Trim, Cropping, And Cuts',
        family='cut',
        key='log',
        help='Write detected cut frame indices to this file (one per line).'),
]

__all__ = ["CUT_OPTIONS"]
