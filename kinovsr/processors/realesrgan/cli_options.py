"""CLI option rows owned by the Real-ESRGAN family."""

from kinovsr.cli.options import Opt

REALESRGAN_OPTIONS = [
    Opt(flag='--realesrgan-scale',
        group='Spatial Upscalers',
        type=int,
        metavar='N',
        family='realesrgan',
        key='scale',
        help='Declared output scale for an explicit --realesrgan-weights path; must match the checkpoint and any named profile.'),
    Opt(flag='--realesrgan-weights',
        group='Spatial Upscalers',
        metavar='PATH',
        family='realesrgan',
        key='weights',
        settings_backed=True,
        help='RRDBNet/SRVGG weights path (.safetensors) for --upscale realesrgan (or $REALESRGAN_WEIGHTS). Prefer --realesrgan-profile for the named checkpoints.'),
    Opt(flag='--realesrgan-profile',
        group='Spatial Upscalers',
        choices=('general', 'x4plus', 'realesrnet', 'bsrnet', 'bsrgan',
                 'x2plus', 'anime', 'animevideo', 'esrgan'),
        family='realesrgan',
        key='profile',
        help='Which Real-ESRGAN model for --upscale realesrgan. general (default; SRVGG, fast/gentle), x4plus (RRDBNet crisp/GAN, ~20x slower), realesrnet / bsrnet (MSE, faithful/soft), bsrgan, x2plus (2x output), anime / animevideo (anime), esrgan (original ESRGAN). Only general is bundled; the rest download + convert (see kinovsr/processors/realesrgan/weights/README.md). Ignored when --realesrgan-weights is given.'),
    Opt(flag='--realesrgan-denoise-strength',
        group='Spatial Upscalers',
        type=float,
        default=1.0,
        metavar='S',
        family='realesrgan',
        key='denoise_strength',
        help='Denoise dial (dni) for realesr-general-x4v3 only, 0..1 (default 1.0 = pure general). Blends s*general + (1-s)*wdn; per Real-ESRGAN, higher = stronger denoise (smoother), lower keeps more real-world texture/grain. Needs the realesr_general_wdn_x4v3 companion weight.'),
]

__all__ = ["REALESRGAN_OPTIONS"]
