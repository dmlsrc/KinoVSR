"""KinoVSR - MLX-native video super-resolution and restoration for Apple Silicon.

The names re-exported at the package top level are the native VideoToolbox /
AVFoundation bridge: VideoToolbox Super Resolution (`VsrSession`), Frame Rate
Conversion (`VtfrcSession`), and AVAssetWriter (`AVWriter`). These require the
PyObjC frameworks listed in the project dependencies; constructing them on a
base install raises a SystemExit with the install hint.

Learned restoration and upscaler families (denoise, deblock, toflow, nafnet,
basicvsrpp, realesrgan, ...) live in their own submodules and are imported
directly, for example `from kinovsr.denoise import ...`.

Public surface:

    from kinovsr import (
        VsrSession,        # spatial upscale via VTSuperResolutionScaler*
        VtfrcSession,      # temporal frame-rate conversion via VTFrameRateConversion*
        AVWriter,          # HEVC + audio encoder via AVAssetWriter
        AudioTrack,        # in-memory PCM -> CMSampleBuffer wrapper
        CutDetector,       # pure-MLX scene-cut detector for VSR reset
    )

Submodules expose lower-level helpers:

    pixel_buffers   CVPixelBuffer create/read/write, CMTime helpers
    comparison      Side-by-side composite for `comparison.mp4`

The stacked progress-bar primitives moved to `kinovsr.progress`
(was previously re-exported here when the bars only had VSR callers).
"""

from __future__ import annotations

from ._compat import autorelease_pool, require_pyobjc

# Re-export the main classes. Each submodule imports its own PyObjC bits via
# _compat, so importing this package only forces the pyobjc check at the
# point a class is actually constructed (via require_pyobjc in each ctor).
from .audio import AudioTrack
from .cut_detect import CutDetector
from .encode import encode_video_videotoolbox
from .temporal import VtfrcSession
from .vsr import VsrSession
from .writer import AVWriter

__all__ = [
    "AVWriter",
    "AudioTrack",
    "CutDetector",
    "VsrSession",
    "VtfrcSession",
    "autorelease_pool",
    "encode_video_videotoolbox",
    "require_pyobjc",
]
