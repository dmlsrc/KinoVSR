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

Progress is published through `kinovsr.reporting.Reporter`; the CLI wires
the Rich-backed implementation from `kinovsr.ui`.
"""

from __future__ import annotations

# Re-export the main classes. Each submodule imports its own PyObjC bits via
# native.compat, so importing this package only forces the pyobjc check at the
# point a class is actually constructed (via require_pyobjc in each ctor).
from .media.audio import AudioTrack
from .native.compat import autorelease_pool, require_pyobjc
from .native.encode import encode_video_videotoolbox
from .native.temporal import VtfrcSession
from .native.vsr import VsrSession
from .native.writer import AVWriter
from .processors.cut_detect import CutDetector

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
