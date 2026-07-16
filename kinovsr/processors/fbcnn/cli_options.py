"""CLI option rows owned by the FBCNN family."""

from kinovsr.cli.options import Opt

FBCNN_WEIGHT_OPTIONS = [
    Opt(flag='--fbcnn-weights',
        group='Deblock',
        metavar='PATH',
        family='fbcnn',
        key='weights',
        settings_backed=True,
        help='FBCNN weights path (fbcnn_color.safetensors) for --deblock fbcnn (or $FBCNN_WEIGHTS); not bundled - see kinovsr/processors/fbcnn/weights/README.md.'),
]

FBCNN_OPTIONS = [
    Opt(flag='--fbcnn-quality',
        group='Deblock',
        default='auto',
        metavar='QF',
        family='fbcnn',
        key='quality',
        help="JPEG quality factor for --deblock fbcnn (1-100, lower = more compressed = stronger removal). Default 'auto': the DCT-coefficient comb MEASURES the quantization per 128px tile over a rolling frame window and the net runs with per-tile QF conditioning, so lightly compressed regions get gentle treatment while heavy ones get strong removal; tiles with no detectable JPEG history fall back to --fbcnn-quality-fallback. A NUMBER pins one global QF (temporally stable, compiled, fastest). 'blind' uses the net's own per-frame estimator -- it reads loop-filtered H.264/HEVC as near-lossless (~QF 96) and barely acts, so prefer auto or a pin on video."),
    Opt(flag='--fbcnn-quality-fallback',
        group='Deblock',
        type=float,
        default=50.0,
        metavar='QF',
        family='fbcnn',
        key='quality_fallback',
        help='QF used by --fbcnn-quality auto where the comb declines (no measurable JPEG-family quantization): per tile when only some tiles decline, or for the whole frame (via the fast compiled path) when nothing in the window carries a comb. Default 50 = mild; lower it for footage you know is heavily compressed but too re-encoded for the comb to survive.'),
    Opt(flag='--fbcnn-strength',
        group='Deblock',
        type=float,
        metavar='A',
        family='fbcnn',
        key='strength',
        help="Linear dry/wet blend of FBCNN's correction for --deblock fbcnn: out = (1-A)*input + A*fbcnn(input). 1.0 = full; <1 keeps more original texture (and faint residual artifacts) uniformly; >1 over-drives (can ring). Overrides the positional --deblock-strength value for this family (default: the chain value). A QF-independent strength dial, complementary to --fbcnn-quality."),
    Opt(flag='--fbcnn-gop',
        group='Deblock',
        default='on',
        choices=('off', 'on'),
        family='fbcnn',
        key='gop',
        help="Key --fbcnn-quality auto's conditioning refreshes to the source's sync samples (default on). A QF-map or blockiness-mask refresh changes treatment; landing it on an I-frame hides the step under the source's own coding reset instead of printing it at an arbitrary mid-GOP frame. The frame counter stays as the ceiling, so refreshes never become rarer; off reproduces the counter-only cadence."),
]

__all__ = ["FBCNN_WEIGHT_OPTIONS", "FBCNN_OPTIONS"]
