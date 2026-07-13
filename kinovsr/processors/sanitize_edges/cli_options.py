"""CLI option rows owned by the sanitize-edges family."""

from kinovsr.cli.options import Opt

SANITIZE_EDGE_OPTIONS = [
    Opt(flag='--sanitize-edges',
        group='Trim, Cropping, And Cuts',
        metavar='auto|T,B,L,R',
        family='sanitize_edges',
        help='Detect and clean synthetic border junk (letterbox lines, capture garbage rows) BEFORE any processor sees the frame: the affected edge rows/cols are overwritten with the adjacent interior line (replicate fill), because learned restorers are trained on photographic content and hallucinate texture around synthetic edges. Frame dimensions and pixel-aspect are untouched, so the output geometry is identical. auto (needs --video) samples early frames and only trims edges that are anomalous in every sample, capped at 8 px per edge; thick constant bars (letterbox-class) are reported but never filled -- crop those instead. T,B,L,R forces explicit per-edge pixel counts. Default: off.'),
    Opt(flag='--sanitize-edges-fill',
        group='Trim, Cropping, And Cuts',
        default='restore',
        choices=('restore', 'extend', 'trim'),
        family='sanitize_edges',
        key='fill',
        help='What happens where junk edges were detected. restore (default) = the nets see a replicate-extended frame and the ORIGINAL border is composited back over the processed output, feathered into the content (--sanitize-edges-feather): the border stays exactly as quiet/static/dark as the source. extend = keep the replicated content in the output, removing the junk -- but replicated content MOVES with the interior, visible shimmer where the eye expects a static border. trim = CROP the junk lines off entirely (folded into the crop before --crop-aspect runs, so the aspect window is computed on the clean picture; bottom/right bumped 1 px if needed to keep even dimensions).'),
    Opt(flag='--sanitize-edges-feather',
        group='Trim, Cropping, And Cuts',
        type=int,
        default=2,
        metavar='N',
        family='sanitize_edges',
        key='feather',
        help='Crossfade width (source px) from a restored border band into the processed content (default 2). Softens the seam between the authentic soft border and the crisply processed interior. 0 = hard splice.'),
]

__all__ = ["SANITIZE_EDGE_OPTIONS"]
