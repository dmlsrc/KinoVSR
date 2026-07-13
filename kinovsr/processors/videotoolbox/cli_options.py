"""CLI option rows owned by the VideoToolbox family."""

from kinovsr.cli.options import Opt

VIDEOTOOLBOX_OPTIONS = [
    Opt(flag='--target-fps',
        group='Output, Encoding, And Audio',
        type=float,
        help='Target output fps. Defaults to the source fps (no temporal upscale). Setting a different value routes VSR output through VTFrameRateConversionConfiguration, motion-interpolating to the target rate. Arbitrary float values supported; 24->60, 15->30, 30->24 (downsample), etc. The CMTime base is 24000 so common rates land bit-exact.'),
    Opt(flag='--temporal-mode',
        group='Output, Encoding, And Audio',
        default='normal',
        choices=('normal', 'high'),
        help='VTFrameRateConversion mode. Only active when --target-fps is set. normal (default) = fast and adequate for 2x rate-up; high = QualityPrioritizationQuality, more compute for cleaner motion.'),
]

__all__ = ["VIDEOTOOLBOX_OPTIONS"]
