"""Shared PyObjC framework bindings for KinoVSR's native modules."""

from contextlib import AbstractContextManager

import AVFoundation as av
import CoreAudio
import CoreMedia
import Foundation
import libdispatch
import objc
import Quartz
import VideoToolbox

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


def autorelease_pool() -> AbstractContextManager[None, None]:
    """`with autorelease_pool():` to drain transient PyObjC objects per iter.

    PyObjC autoreleased objects (NSData, CIImage, ...) accumulate in the
    process's top-level autorelease pool, which doesn't drain until the
    interpreter exits. Long Python loops that allocate many such objects
    per iteration grow RSS unboundedly. Wrapping the inner-loop body in
    a fresh pool forces drainage at the end of each iteration.
    """
    pool: AbstractContextManager[None, None] = objc.autorelease_pool()
    return pool
