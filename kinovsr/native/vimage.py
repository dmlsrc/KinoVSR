"""ctypes bindings for the vImage histogram primitives (Accelerate).

pyobjc ships no Accelerate framework wrapper (its full framework tree
carries none), so vImage is reached through ctypes. The C API suits
that unusually well: a four-field ``vImage_Buffer`` struct and plain
pointers, no Objective-C objects involved, and the framework is part
of every macOS install - no dependency is added.

Only the float histogram is bound: the measured leveler design needs
histogram CALCULATION only (the transfer curve is built from quantile
functions and applied as an fp32 LUT; the u8 specification call's
quantized output amplifies on dark pixels and is deliberately not
used).
"""

from __future__ import annotations

import ctypes
from functools import lru_cache


class VImageBuffer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p),
                ("height", ctypes.c_ulong),
                ("width", ctypes.c_ulong),
                ("rowBytes", ctypes.c_size_t)]


@lru_cache(maxsize=1)
def _accelerate() -> ctypes.CDLL:
    lib = ctypes.CDLL(
        "/System/Library/Frameworks/Accelerate.framework/Accelerate")
    fn = lib.vImageHistogramCalculation_PlanarF
    fn.restype = ctypes.c_ssize_t
    fn.argtypes = [
        ctypes.POINTER(VImageBuffer), ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_uint32, ctypes.c_float, ctypes.c_float, ctypes.c_uint32]
    return lib


def histogram_planarf(plane: bytearray, width: int, height: int,
                      bins: int, min_value: float = 0.0,
                      max_value: float = 1.0) -> list[int]:
    """Histogram of a contiguous fp32 plane held in a writable buffer.

    ``plane`` must be ``height * width * 4`` bytes of row-contiguous
    float32. Returns ``bins`` counts covering [min_value, max_value].
    """
    expected = height * width * 4
    if len(plane) != expected:
        raise ValueError(
            f"plane must be {expected} bytes of fp32, got {len(plane)}")
    hist = (ctypes.c_ulong * bins)()
    raw = (ctypes.c_char * len(plane)).from_buffer(plane)
    buf = VImageBuffer(ctypes.cast(raw, ctypes.c_void_p),
                       height, width, width * 4)
    err = _accelerate().vImageHistogramCalculation_PlanarF(
        ctypes.byref(buf), ctypes.cast(hist, ctypes.POINTER(ctypes.c_ulong)),
        bins, ctypes.c_float(min_value), ctypes.c_float(max_value), 0)
    if err != 0:
        raise RuntimeError(f"vImageHistogramCalculation_PlanarF -> {err}")
    return list(hist)


__all__ = ["histogram_planarf"]
