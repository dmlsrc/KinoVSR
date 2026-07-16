"""CLI option rows owned by the conform family."""

from kinovsr.cli.options import Opt

CONFORM_OPTIONS = [
    Opt(flag='--conform-cfr',
        group='Temporal Restore And Deflicker',
        default=None,
        metavar='RATE',
        family='conform',
        help="Map a carried non-uniform timeline (dropped frames, splices, jitter) onto a constant-rate grid: each output slot takes the NEAREST original frame - duplicated across gaps, dropped under jitter bursts - with timestamps regenerated on the grid. No frame is synthesized (use --target-fps for interpolation); the dup/drop/max-shift ledger is printed so the normalization is declared, never silent. RATE is a rate like 25 or 30000/1001, or 'auto' for the source's own nominal grid. By default the pipeline CARRIES the source clock exactly; use this only when the deliverable must be CFR."),
]

__all__ = ["CONFORM_OPTIONS"]
