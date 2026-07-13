"""CLI option rows owned by the FastDVDnet family."""

from kinovsr.cli.options import Opt

FASTDVDNET_STRENGTH_OPTIONS = [
    Opt(flag='--fastdvdnet-strength',
        group='Denoise And Noise Maps',
        type=float,
        metavar='S',
        family='fastdvdnet',
        key='strength',
        help='Strength for --denoise fastdvdnet (noise sigma, mapped onto sigma_255 in [5, 55]), overriding the positional --denoise-strength value for this family (default: the chain value).'),
]

FASTDVDNET_OPTIONS = [
    Opt(flag='--fastdvdnet-weights',
        group='Denoise And Noise Maps',
        metavar='PATH',
        family='fastdvdnet',
        key='weights',
        settings_backed=True,
        help='Override FastDVDnet weights (.safetensors) for --denoise fastdvdnet. Optional - defaults to the bundled --fastdvdnet-profile weights (or $FASTDVD_WEIGHTS). Convert a .pth with kinovsr weights convert.'),
    Opt(flag='--fastdvdnet-profile',
        group='Denoise And Noise Maps',
        default='clipped',
        choices=('clipped', 'standard'),
        family='fastdvdnet',
        key='profile',
        help='Which bundled FastDVDnet model for --denoise fastdvdnet. clipped (default) is trained with clipped noise and stays clean on real footage at moderate strength; standard is the plain-AWGN model and shows a faint pixel-shuffle grid above ~0.1 strength on clean content. Ignored when --fastdvdnet-weights is given.'),
]

__all__ = ["FASTDVDNET_STRENGTH_OPTIONS", "FASTDVDNET_OPTIONS"]
