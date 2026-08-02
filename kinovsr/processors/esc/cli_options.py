"""CLI option rows owned by the ESC family."""

from kinovsr.cli.options import Opt

ESC_OPTIONS = [
    Opt(flag='--esc-scale',
        group='Spatial Upscalers',
        type=int,
        metavar='N',
        family='esc',
        key='scale',
        help='Declared output scale for an explicit --esc-weights path; must match the checkpoint. Named profiles declare this themselves.'),
    Opt(flag='--esc-weights',
        group='Spatial Upscalers',
        metavar='PATH',
        family='esc',
        key='weights',
        settings_backed=True,
        help='ESC-Real weights path (.safetensors) for --upscale esc (or $ESC_WEIGHTS). Prefer --esc-profile for the named checkpoints.'),
    Opt(flag='--esc-profile',
        group='Spatial Upscalers',
        choices=('gan', 'mse'),
        family='esc',
        key='profile',
        help='Which ESC-Real model for --upscale esc. gan (default; perceptual, Real-ESRGAN-style degradation training) and mse (fidelity twin). Neither is bundled; see kinovsr/processors/esc/weights/README.md. Ignored when --esc-weights is given.'),
]

__all__ = ["ESC_OPTIONS"]
