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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .media.audio import AudioTrack
    from .native.compat import autorelease_pool, require_pyobjc
    from .native.encode import encode_video_videotoolbox
    from .native.temporal import VtfrcSession
    from .native.vsr import VsrSession
    from .native.writer import AVWriter
    from .processors.cut_detect import CutDetector

# Re-exports resolve lazily (PEP 562): importing the package loads no
# PyObjC framework and no MLX, so `import kinovsr` / `kinovsr.api` stay
# light for hosts and CLI startup. The frameworks load at the first
# attribute access - typically a session constructor.
_EXPORTS = {
    "AVWriter": ("kinovsr.native.writer", "AVWriter"),
    "AudioTrack": ("kinovsr.media.audio", "AudioTrack"),
    "CutDetector": ("kinovsr.processors.cut_detect", "CutDetector"),
    "VsrSession": ("kinovsr.native.vsr", "VsrSession"),
    "VtfrcSession": ("kinovsr.native.temporal", "VtfrcSession"),
    "autorelease_pool": ("kinovsr.native.compat", "autorelease_pool"),
    "encode_video_videotoolbox": ("kinovsr.native.encode",
                                  "encode_video_videotoolbox"),
    "require_pyobjc": ("kinovsr.native.compat", "require_pyobjc"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value      # cache: subsequent access skips the hook
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
