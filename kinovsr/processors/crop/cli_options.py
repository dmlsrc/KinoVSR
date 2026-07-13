"""CLI option rows owned by the crop family."""

from kinovsr.cli.options import Opt

CROP_OPTIONS = [
    Opt(flag='--crop-bars',
        group='Trim, Cropping, And Cuts',
        metavar='auto|T,B,L,R',
        family='crop',
        key='bars',
        help="Crop constant letterbox/pillarbox bars off BEFORE processing and output only the active picture (e.g. 16:9 letterboxed in 4:3 becomes true 16:9 out; 9:16 pillarboxed in 16:9 becomes true 9:16). auto detects bars that are constant-extreme in every sampled frame, up to 45 percent per edge, and rounds so the active area keeps even dimensions; T,B,L,R forces explicit counts. The pixel aspect is unchanged -- the display aspect becomes the content's true aspect, which is the point. Composes with --sanitize-edges (junk detection runs on the cropped picture). Needs --video. Default: off."),
    Opt(flag='--crop-aspect',
        group='Trim, Cropping, And Cuts',
        metavar='W:H',
        family='crop',
        key='aspect',
        help='Crop the picture to the largest even-dimension window with this DISPLAY aspect (e.g. 16:9 on a 4:3 source, 1:1, 9:16 for a portrait extract). On anamorphic sources the pixel aspect is folded into the target automatically, so 16:9 means 16:9 on screen, not in storage pixels. Even-integer windows approximate most ratios; the closest fit is chosen. Applies AFTER --crop-bars, so a letterboxed source can be bar-cropped and then reframed in one run. Place with --crop-anchor, shift with --crop-offset. Needs --video.'),
    Opt(flag='--crop-anchor',
        group='Trim, Cropping, And Cuts',
        default='center',
        choices=('top-left', 'top', 'top-right', 'left', 'center', 'right', 'bottom-left', 'bottom', 'bottom-right'),
        family='crop',
        key='anchor',
        help="Where to place the --crop-aspect window (default center). E.g. 16:9 from a 4:3 source anchored at 'bottom' keeps the lower two-thirds; 'top-right' pins the window to that corner. --crop-offset nudges from the anchor."),
    Opt(flag='--crop-offset',
        group='Trim, Cropping, And Cuts',
        default='0,0',
        metavar='DX,DY',
        family='crop',
        key='offset',
        help='Pixel offset of the --crop-aspect window from its anchor (right/down positive, clamped so the window stays inside the frame). Default 0,0.'),
]

__all__ = ["CROP_OPTIONS"]
