"""KinoVSR - MLX-native video super-resolution and restoration for Apple Silicon.

The supported host API is :mod:`kinovsr.api`. The names re-exported at the
package top level are the native VideoToolbox / AVFoundation bridge:
VideoToolbox Super Resolution (`VsrSession`), Frame Rate Conversion
(`VtfrcSession`), and AVAssetWriter (`AVWriter`). Their PyObjC frameworks are
normal project dependencies and load with the package.

Processor families (cut detection, deblock, toflow, nafnet, basicvsrpp,
realesrgan, ...) live in their own submodules and are imported directly, for
example `from kinovsr.processors.cut_detect import CutDetector`. The pipeline
loads them by name through the processor catalog.

Top-level names:

    from kinovsr import (
        VsrSession,        # spatial upscale via VTSuperResolutionScaler*
        VtfrcSession,      # temporal frame-rate conversion via VTFrameRateConversion*
        AVWriter,          # HEVC + audio encoder via AVAssetWriter
        AudioTrack,        # in-memory PCM -> CMSampleBuffer wrapper
    )

Submodules expose lower-level helpers:

    pixel_buffers   CVPixelBuffer create/read/write, CMTime helpers

Progress is published through `kinovsr.reporting.Reporter`; the CLI wires
the Rich-backed implementation from `kinovsr.ui`.
"""

import atexit as _atexit

from .media.audio import AudioTrack
from .modeling.mlx_runtime import (
    clear_mlx_thread_state as _clear_mlx_thread_state,
)
from .native.frameworks import autorelease_pool
from .native.temporal import VtfrcSession
from .native.vsr import VsrSession
from .native.writer import AVWriter

# MLX 0.32.1 clears its Python-bearing compile cache through clear_streams().
# Register while the interpreter is healthy.
_atexit.register(_clear_mlx_thread_state)

__all__ = ["AVWriter", "AudioTrack", "VsrSession", "VtfrcSession", "autorelease_pool"]
