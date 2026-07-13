"""Shared PyObjC framework imports for the native KinoVSR modules."""

from __future__ import annotations

import AVFoundation as av  # type: ignore
import CoreAudio  # type: ignore
import CoreMedia  # type: ignore
import Foundation  # type: ignore
import libdispatch  # type: ignore
import objc  # type: ignore
import Quartz  # type: ignore
import VideoToolbox  # type: ignore

__all__ = [
    "CoreAudio",
    "CoreMedia",
    "Foundation",
    "Quartz",
    "autorelease_pool",
    "av",
    "libdispatch",
    "objc",
    "vt",
]


# Alias to keep `vt.` references readable inside submodules.
vt = VideoToolbox


def autorelease_pool():
    """`with autorelease_pool():` to drain transient PyObjC objects per iter.

    PyObjC autoreleased objects (NSData, CIImage, ...) accumulate in the
    process's top-level autorelease pool, which doesn't drain until the
    interpreter exits. Long Python loops that allocate many such objects
    per iteration grow RSS unboundedly. Wrapping the inner-loop body in
    a fresh pool forces drainage at the end of each iteration.
    """
    return objc.autorelease_pool()
