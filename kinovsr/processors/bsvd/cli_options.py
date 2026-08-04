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
        choices=('mlx', 'ane', 'mpsgraph'),
        family='bsvd',
        key='backend',
        settings_backed=True,
        help=(
            'Which BSVD implementation runs for --denoise bsvd. mlx '
            '(default): the reference MLX/GPU path. Under --gop-align it '
            'uses the same reactive GOP windows and per-window conditioning '
            'as the other backends. mpsgraph: MPSGraph compiles BSVD for the '
            'Neural Engine; cached ANECIR products execute directly, with no '
            'Core ML package or prediction. Recurrent state and skip rings '
            'stay in shared IOSurfaces. Streaming dispatches remain pipelined '
            'one deep. Under --gop-align one schedule-generic direct entry '
            'remains loaded through fill, steady state, drain, and reset '
            'windows, avoiding raw ANE program re-entry while keeping the '
            'one-frame cadence needed for downstream GPU overlap. The compiler '
            'cache records realized live port order. A request-liveness result '
            'is verified so a runtime '
            'no-op cannot return stale frames. fp16 only; sides must be '
            'multiples of four. '
            'The window path covers padded geometries from 128x256 pixels '
            'through 1024x576 and fails loudly outside that envelope. The '
            'first window at a new geometry builds an OS-specific executable '
            'cache under $KINOVSR_CACHE_DIR. ane: the full per-step network '
            'as one Core ML dispatch pinned to the Neural Engine, pipelined '
            'one dispatch deep so the prediction overlaps the rest of the '
            'per-frame work (one extra frame of output delay, absorbed into '
            'the family delay). Slightly SLOWER standalone (about 91 vs 78 '
            'ms/frame at 640x480 c64) but it vacates the GPU, so chains with '
            'other GPU stages can hide its cost; numerics are marginally '
            'closer to fp32 than fp16 MLX. Under --gop-align through the '
            'verified 640x480 padded envelope, fixed eight-step fill 0-7 and '
            'drain 8-15 functions omit the None work; the inner boundary '
            'steps stay on the full graph to keep only three ANE programs '
            'resident. Short light clips can still favor MLX because the '
            'extra Core ML functions have a one-time load cost. fp16 only; '
            'frames need at least 96 px per side, and widths are reflect-'
            'padded internally to the next multiple of 128 (the ANE alignment '
            'envelope - CIF 352 runs as 384) and cropped back. Fails loudly '
            'rather than fall back: a partial Core ML CPU placement computes '
            'this graph WRONG, so non-ANE placement is refused outright. The '
            'first run at a new padded geometry builds and compiles the model '
            '(seconds to a minute, cached under $KINOVSR_CACHE_DIR); later '
            'runs load it in about a second.')),
]

__all__ = ["BSVD_STRENGTH_OPTIONS", "BSVD_OPTIONS"]
