"""CLI option rows owned by the cut-detect family."""

from kinovsr.cli.options import Opt

CUT_OPTIONS = [
    Opt(flag='--cut-detect',
        group='Trim, Cropping, And Cuts',
        default='off',
        choices=('off', 'simple', 'hist', 'vtme'),
        family='cut',
        key='detect',
        compositional=True,
        help="Reset temporal state at hard cuts (VSR balanced's prev-frame chain, mc history, and every other stateful stage downstream of the mark). off = never reset (correct for continuous generated clips and single-shot sources). simple = downsampled-pixel difference (~1 ms). hist = per-channel histogram distance (~3 ms), more robust to fast motion. vtme = trackability via the VTMotionEstimationSession block-motion field on the media engine (macOS 26+, ~7-10 ms, zero GPU cost): consecutive frames stay coherent even under fast motion and global luma flicker, cuts do not. On deflicker-class alternating-gain footage simple and hist false-fire on EVERY frame pair while vtme stays quiet; prefer it for flickery or analog-degraded sources."),
    Opt(flag='--cut-threshold',
        group='Trim, Cropping, And Cuts',
        type=float,
        family='cut',
        key='threshold',
        stage_processor='cut_detect',
        help="Cut decision threshold; each mode has its own scale and default. simple: thumbnail MAD in [0,1], default 0.25, typical 0.2-0.35. hist: normalized chi-squared, default 0.25, typical 0.4-0.8 on busy footage. vtme: block-field neighbor incoherence over the frame diagonal, default 0.07, typical 0.05-0.09 (measured non-cut pairs at or under ~0.06 and true cuts at or over ~0.088). Lower catches subtler cuts; false positives only cost one frame of temporal context."),
    Opt(flag='--cut-log',
        group='Trim, Cropping, And Cuts',
        family='cut',
        key='log',
        config_table='diagnostics',
        help='Write detected cut frame indices to this file (one per line).'),
]

__all__ = ["CUT_OPTIONS"]
