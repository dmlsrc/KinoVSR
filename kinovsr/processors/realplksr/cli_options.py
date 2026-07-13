"""CLI option rows owned by the RealPLKSR family."""

from kinovsr.cli.options import Opt

REALPLKSR_OPTIONS = [
    Opt(flag='--realplksr-weights',
        group='Spatial Upscalers',
        metavar='PATH',
        family='realplksr',
        key='weights',
        settings_backed=True,
        help='RealPLKSR weights path (.safetensors) for --upscale realplksr (or $REALPLKSR_WEIGHTS). Prefer --realplksr-profile for the named checkpoints.'),
    Opt(flag='--realplksr-profile',
        group='Spatial Upscalers',
        choices=('public2x', 'public2x-nn', 'nomos4x'),
        family='realplksr',
        key='profile',
        help='Which RealPLKSR model for --upscale realplksr. public2x (default; 2x LayerNorm+DySample, real-world photo/JPEG, Apache-2.0), public2x-nn (same trained without noise -- for cleaner sources), nomos4x (4xNomosWebPhoto, 4x GroupNorm+PixelShuffle photo restoration, CC-BY). Scale (2x/4x) is read from the checkpoint. Single-image, no pooled gate so no SAFMN block lattice; no denoising prior beyond its training, so prefer it on decent sources or pair with --denoise. None are bundled; see kinovsr/processors/realplksr/weights/README.md. Ignored when --realplksr-weights is given.'),
    Opt(flag='--realplksr-dtype',
        group='Spatial Upscalers',
        default='float16',
        choices=('float16', 'float32'),
        family='realplksr',
        key='dtype',
        help='Compute/storage dtype for --upscale realplksr. float16 (default) runs the convs in half precision with fp32 precision islands in the norm reductions and Mish (visually lossless, ~72 dB vs fp32, and it is these islands that make the fp16-flagged GroupNorm 4x checkpoint safe). float32 forces a full single-precision run (slower, more memory).'),
]

__all__ = ["REALPLKSR_OPTIONS"]
