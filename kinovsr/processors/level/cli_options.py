"""CLI option rows owned by the exposure-leveler family."""

from kinovsr.cli.options import Opt

LEVEL_OPTIONS = [
    Opt(flag='--level',
        group='Temporal Restore And Deflicker',
        default='off',
        choices=('off', 'hist'),
        family='level',
        compositional=True,
        help="Global exposure/flicker leveler, run FIRST in the preprocess chain. For footage whose brightness pumps as a whole - auto-exposure hunting, camcorder AGC flicker, gamma-like exposure wobble - which deflicker deliberately does not cover (it fixes per-tile quantization flicker on verified-static content; a global gain swing fails its verification wholesale, so the two compose: level, then deflicker). hist = each frame's luma distribution is matched to a centered +-window reference (histograms via native vImage on the CPU, immune to GPU load; measured +11 to +21 dB pumping removal on oscillatory cases, 4-5 dB over plain gain matching on gamma-type pumping). Scope is honest: global pumping only; a slow drift is indistinguishable from a real lighting change and is followed, not fought; fades and dissolves pass through (the centered reference tracks them); clipped flash frames cannot be unclipped. Every downstream temporal stage benefits: the noise map stops over-reading pumping as noise, mc's photometric gate stops closing on gain swings, learned temporal denoisers see in-distribution brightness."),
    Opt(flag='--level-window',
        group='Temporal Restore And Deflicker',
        type=int,
        default=5,
        metavar='K',
        family='level',
        key='window',
        help='Leveler half-window in frames (default 5; also the stage latency). The reference is the average luma distribution over +-K frames, so oscillation with period up to about 2K is flattened while slower trends (fades, real lighting changes) are followed. Match to the pumping cadence: AGC flicker at 1-3 frame periods is covered by the default; slow exposure hunting may need 8-10.'),
    Opt(flag='--level-deadband',
        group='Temporal Restore And Deflicker',
        type=float,
        default=0.003,
        metavar='S',
        family='level',
        key='deadband',
        help="Minimum implied luma shift (0..1 scale) before a frame is corrected (default 0.003, i.e. 0.3 percent). Below it the frame passes through BIT-EXACT, keeping the stage inert on stable footage where content motion alone drifts the window distribution by ~0.003 on fast scenes. Lower it (0.001) to chase subtle 1-2 percent pumping on calm footage; raise it if fast-motion content is being touched. The end-of-run '[level] pumping meter' line reports measured shifts and the corrected-frame count - stable clips should read near zero corrected."),
]

__all__ = ["LEVEL_OPTIONS"]
