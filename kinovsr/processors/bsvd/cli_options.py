"""CLI option rows owned by the BSVD family."""

from kinovsr.cli.options import Opt

BSVD_STRENGTH_OPTIONS = [
    Opt(flag='--bsvd-strength',
        group='Denoise And Noise Maps',
        type=float,
        metavar='S',
        family='bsvd',
        key='strength',
        help='Strength for --denoise bsvd (noise sigma, mapped onto sigma_255 in [5, 55]), overriding the positional --denoise-strength value for this family (default: the chain value).'),
]

BSVD_OPTIONS = [
    Opt(flag='--bsvd-weights',
        group='Denoise And Noise Maps',
        metavar='PATH',
        family='bsvd',
        key='weights',
        settings_backed=True,
        help='Override BSVD weights (.safetensors) for --denoise bsvd. Optional - defaults to the local --bsvd-profile weights (or $BSVD_WEIGHTS). Convert a .pth with kinovsr weights convert --param-key params.'),
    Opt(flag='--bsvd-profile',
        group='Denoise And Noise Maps',
        default='c64',
        choices=('c64', 'c32'),
        family='bsvd',
        key='profile',
        help='Which local BSVD model token for --denoise bsvd. c64 matches the public unblind test config; c32 is a smaller mirror checkpoint with weaker provenance. Ignored when --bsvd-weights is given.'),
    Opt(flag='--bsvd-dtype',
        group='Denoise And Noise Maps',
        default='float16',
        choices=('float16', 'float32'),
        family='bsvd',
        key='dtype',
        help='MLX dtype for --denoise bsvd (default float16; use float32 for parity probes).'),
    Opt(flag='--bsvd-backend',
        group='Denoise And Noise Maps',
        choices=('mlx', 'ane'),
        family='bsvd',
        key='backend',
        settings_backed=True,
        help='Which BSVD implementation runs for --denoise bsvd. mlx (default): the reference MLX/GPU path. ane: the full per-step network as one Core ML dispatch pinned to the Neural Engine - slightly SLOWER standalone (about 91 vs 78 ms/frame at 640x480 c64) but it vacates the GPU, so chains with other GPU stages can hide its cost; numerics are marginally closer to fp32 than the MLX fp16 path. fp16 only; frames need at least 96 px per side, and widths are reflect-padded internally to the next multiple of 128 (the ANE alignment envelope - CIF 352 runs as 384) and cropped back. Fails loudly rather than fall back: a partial Core ML CPU placement computes this graph WRONG, so non-ANE placement is refused outright. The first run at a new padded geometry builds and compiles the model (seconds to a minute, cached under $KINOVSR_CACHE_DIR); later runs load it in about a second.'),
]

__all__ = ["BSVD_STRENGTH_OPTIONS", "BSVD_OPTIONS"]
