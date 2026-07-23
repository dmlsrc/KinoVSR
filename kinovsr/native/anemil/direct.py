"""Direct AppleNeuralEngine dispatch for compiled anemil programs.

Core ML is the default execution route for anemil-emitted models
(:mod:`.runtime`). This module is the second route: it drives the
private ``AppleNeuralEngine.framework`` (``_ANEInMemoryModel``)
directly, consuming exactly the artifacts the existing pipeline already
produces - the ``model.mil`` text and ``weights/weight.bin`` inside a
``coremlcompiler``-compiled ``.mlmodelc``.

Why it exists: Core ML keeps every loaded function's ANE program
resident and provides no unload, so two large programs cannot alternate
(re-entry fails with ``ANEProgramProcessRequestDirect`` status=0x16),
and its Espresso translator cannot build very large single graphs at
all (error -14, hence the fp32 island in the BSVD step). The direct
interface exposes the missing lifecycle: a program loads in ~13 ms and
unloads in ~3 ms on this machine, so two exact graph halves can
alternate within a frame, byte-identical to the Core ML result and
about 10% faster than the islanded single graph at 1080p (measured;
see the BSVD backend that builds on this).

Contracts and cautions:

- I/O is IOSurface-backed. Tensors bind by position through
  ``_ANERequest``; layouts come from the loaded model's live-IO lists
  and are strided, not necessarily contiguous.
- ONE large program resident at a time is a driver law, not a policy
  choice; loading a second one can poison ANE state for this
  executable identity (the venv Python) until the binary changes.
  :func:`DirectModel.load` enforces single residency structurally.
- Private framework: every entry point funnels through
  :func:`preflight`, which verifies the classes and selectors in
  ~40 ms once per process and raises :class:`DirectUnavailable`
  otherwise. Callers are expected to fall back to the Core ML route.
- Evaluation blocks the calling thread and releases the GIL; run it on
  a dispatch worker (:class:`.runtime.DispatchPipeline`) exactly like
  the Core ML route.
"""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger("kinovsr.anemil.direct")

_ANE_FRAMEWORK = ("/System/Library/PrivateFrameworks/"
                  "AppleNeuralEngine.framework/AppleNeuralEngine")
_IOSURFACE_FRAMEWORK = ("/System/Library/Frameworks/IOSurface.framework/"
                        "IOSurface")
_QOS = 21  # the QoS the OS stack itself uses for user-initiated inference

_REQUIRED_SELECTORS = (
    b"compileWithQoS:options:error:",
    b"loadWithQoS:options:error:",
    b"unloadWithQoS:error:",
    b"evaluateWithQoS:options:request:error:",
    b"mapIOSurfacesWithRequest:cacheInference:error:",
    b"unmapIOSurfacesWithRequest:",
)


class DirectUnavailable(RuntimeError):
    """The private dispatch route is not usable on this system."""


class _Bridge:
    """Lazily initialized objc/ctypes surface (one per process)."""

    def __init__(self):
        import objc

        try:
            ctypes.CDLL(_ANE_FRAMEWORK)
        except OSError as error:
            raise DirectUnavailable(
                f"AppleNeuralEngine framework did not load: {error}"
            ) from error
        self.iosurface_c = ctypes.CDLL(_IOSURFACE_FRAMEWORK)
        self.iosurface_c.IOSurfaceGetBaseAddress.restype = ctypes.c_void_p
        self.iosurface_c.IOSurfaceGetBaseAddress.argtypes = [ctypes.c_void_p]
        self.iosurface_c.IOSurfaceGetAllocSize.restype = ctypes.c_size_t
        self.iosurface_c.IOSurfaceGetAllocSize.argtypes = [ctypes.c_void_p]
        self.iosurface_c.IOSurfaceLock.restype = ctypes.c_int
        self.iosurface_c.IOSurfaceLock.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        self.iosurface_c.IOSurfaceUnlock.restype = ctypes.c_int
        self.iosurface_c.IOSurfaceUnlock.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]

        try:
            self.descriptor_class = objc.lookUpClass(
                "_ANEInMemoryModelDescriptor")
            self.model_class = objc.lookUpClass("_ANEInMemoryModel")
            self.surface_class = objc.lookUpClass("_ANEIOSurfaceObject")
            self.request_class = objc.lookUpClass("_ANERequest")
            self.iosurface_class = objc.lookUpClass("IOSurface")
        except objc.nosuchclass_error as error:
            raise DirectUnavailable(
                f"private ANE class missing: {error}") from error
        missing = [selector.decode() for selector in _REQUIRED_SELECTORS
                   if not self.model_class.instancesRespondToSelector_(
                       selector)]
        if missing:
            raise DirectUnavailable(
                f"private ANE selectors missing: {', '.join(missing)}")

        # NSError** out-arguments on the private selectors; the runtime
        # carries the type encodings, the metadata adds the out modifier
        # so calls return (ok, error) tuples.
        for selector, err_index in (
                (b"compileWithQoS:options:error:", 4),
                (b"loadWithQoS:options:error:", 4),
                (b"unloadWithQoS:error:", 3),
                (b"evaluateWithQoS:options:request:error:", 5),
                (b"mapIOSurfacesWithRequest:cacheInference:error:", 4)):
            objc.registerMetaDataForSelector(
                b"_ANEInMemoryModel", selector,
                {"arguments": {err_index: {"type_modifier": b"o"}}})

        # objectWithIOSurface: takes a raw IOSurfaceRef, which PyObjC
        # will not coerce from the (toll-free identical) ObjC IOSurface
        # object - route this single call through raw objc_msgSend.
        self._objc = objc
        libobjc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        libobjc.sel_registerName.restype = ctypes.c_void_p
        libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
        libobjc.objc_getClass.restype = ctypes.c_void_p
        libobjc.objc_getClass.argtypes = [ctypes.c_char_p]
        libobjc.objc_msgSend.restype = ctypes.c_void_p
        libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p]
        self._libobjc = libobjc
        self._wrap_sel = libobjc.sel_registerName(b"objectWithIOSurface:")
        self._wrap_cls = libobjc.objc_getClass(b"_ANEIOSurfaceObject")

    def wrap_surface(self, surface) -> Any:
        raw = self._libobjc.objc_msgSend(
            ctypes.c_void_p(self._wrap_cls),
            ctypes.c_void_p(self._wrap_sel),
            ctypes.c_void_p(self._objc.pyobjc_id(surface)))
        if not raw:
            raise RuntimeError("_ANEIOSurfaceObject wrap returned nil")
        return self._objc.objc_object(c_void_p=raw)

    def surface_ref(self, surface) -> ctypes.c_void_p:
        return ctypes.c_void_p(self._objc.pyobjc_id(surface))


_BRIDGE: _Bridge | None = None
_BRIDGE_ERROR: DirectUnavailable | None = None
_BRIDGE_LOCK = threading.Lock()


def preflight() -> _Bridge:
    """Verify and return the private-framework bridge (~40 ms, cached).

    Raises :class:`DirectUnavailable` with the reason when the route
    cannot work; the result (either way) is cached for the process.
    """
    global _BRIDGE, _BRIDGE_ERROR
    with _BRIDGE_LOCK:
        if _BRIDGE is not None:
            return _BRIDGE
        if _BRIDGE_ERROR is not None:
            raise _BRIDGE_ERROR
        started = time.perf_counter()
        try:
            _BRIDGE = _Bridge()
        except DirectUnavailable as error:
            _BRIDGE_ERROR = error
            raise
        _log.debug("direct ANE preflight passed in %.1f ms",
                   (time.perf_counter() - started) * 1e3)
        return _BRIDGE


def available() -> bool:
    try:
        preflight()
    except DirectUnavailable:
        return False
    return True


# ------------------------------------------------------------- surfaces

class Surface:
    """One IOSurface-backed tensor backing with CPU access helpers."""

    __slots__ = ("nbytes", "nsobject", "ref", "wrapped")

    def __init__(self, nbytes: int):
        bridge = preflight()
        properties = {
            "IOSurfaceWidth": nbytes,
            "IOSurfaceHeight": 1,
            "IOSurfaceBytesPerElement": 1,
            "IOSurfaceBytesPerRow": nbytes,
            "IOSurfaceAllocSize": nbytes,
            "IOSurfacePixelFormat": 0,
        }
        self.nsobject = bridge.iosurface_class.alloc().initWithProperties_(
            properties)
        if self.nsobject is None:
            raise RuntimeError(f"IOSurface allocation failed ({nbytes} B)")
        self.nbytes = nbytes
        self.ref = bridge.surface_ref(self.nsobject)
        self.wrapped = bridge.wrap_surface(self.nsobject)

    def view(self) -> memoryview:
        bridge = preflight()
        base = bridge.iosurface_c.IOSurfaceGetBaseAddress(self.ref)
        size = bridge.iosurface_c.IOSurfaceGetAllocSize(self.ref)
        return memoryview(
            (ctypes.c_ubyte * size).from_address(base)).cast("B")

    def lock(self, readonly: bool = False) -> None:
        preflight().iosurface_c.IOSurfaceLock(
            self.ref, 1 if readonly else 0, None)

    def unlock(self, readonly: bool = False) -> None:
        preflight().iosurface_c.IOSurfaceUnlock(
            self.ref, 1 if readonly else 0, None)

    def zero(self) -> None:
        self.lock()
        view = self.view()
        view[:] = bytes(len(view))
        self.unlock()


# ----------------------------------------------------------------- ports

def _number(info, key: str, fallback: int) -> int:
    value = info.get(key)
    if value is not None:
        result = int(value)
        if result:
            return result
    return fallback


class Port:
    """One live input or output of a loaded direct model."""

    __slots__ = ("name", "info", "surface")

    def __init__(self, name: str, info: Any, surface: Surface):
        self.name = name
        self.info = info
        self.surface = surface

    @property
    def nbytes(self) -> int:
        stride = _number(self.info, "BatchStride", 0)
        batches = _number(self.info, "Batches", 1)
        return stride * batches if stride else 65536

    def logical_dims(self) -> tuple[int, int, int, int, int]:
        return tuple(_number(self.info, key, 1)
                     for key in ("Batches", "Channels", "Depth",
                                 "Height", "Width"))

    def strides(self) -> tuple[int, int, int, int]:
        batch = _number(self.info, "BatchStride", 65536)
        depth = _number(self.info, "DepthStride", batch)
        plane = _number(self.info, "PlaneStride", depth)
        row = _number(self.info, "RowStride", plane)
        return batch, depth, plane, row

    def logical_nbytes(self) -> int:
        dims = self.logical_dims()
        return dims[0] * dims[1] * dims[2] * dims[3] * dims[4] * 2

    def is_contiguous(self) -> bool:
        batches, channels, depth, height, width = self.logical_dims()
        batch_s, depth_s, plane_s, row_s = self.strides()
        return (row_s == width * 2
                and plane_s == height * row_s
                and depth_s == channels * plane_s
                and batch_s == depth * depth_s
                and self.logical_nbytes() == self.nbytes)


def _symbol(info) -> str:
    return str(info.get("Symbol") or "").split("@", 1)[0]


# ----------------------------------------------------------------- model

_RESIDENT: str | None = None
_RESIDENT_LOCK = threading.Lock()


class DirectModel:
    """One compiled anemil program on the direct dispatch route.

    Construction stages the program with the framework and performs the
    (bundle-cached) device compile; :meth:`discover_ports` allocates or
    attaches IOSurface backings from the model's live-IO lists;
    :meth:`dispatch` runs one load -> request -> map -> evaluate ->
    unmap -> unload cycle. Fresh requests per dispatch are deliberate:
    request caching was measured at under 1 ms per frame and rejected.
    """

    def __init__(self, compiled: Path, label: str):
        from Foundation import NSData, NSDictionary, NSNumber

        bridge = preflight()
        self.label = label
        compiled = Path(compiled)
        mil = NSData.dataWithContentsOfFile_(str(compiled / "model.mil"))
        blob = NSData.dataWithContentsOfFile_(
            str(compiled / "weights" / "weight.bin"))
        if mil is None or blob is None:
            raise FileNotFoundError(
                f"{label}: {compiled} lacks model.mil/weights/weight.bin")
        # Real Foundation containers: the descriptor's validator probes
        # this dictionary with NSData selectors and rejects PyObjC's
        # lazy number proxies.
        entry = NSDictionary.dictionaryWithObjectsAndKeys_(
            NSNumber.numberWithUnsignedLongLong_(0), "offset",
            blob, "data", None)
        weights = NSDictionary.dictionaryWithObject_forKey_(
            entry, "@model_path/weights/weight.bin")
        descriptor = (bridge.descriptor_class.
                      modelWithMILText_weights_optionsPlist_(
                          mil, weights, None))
        self.model = bridge.model_class.inMemoryModelWithDescriptor_(
            descriptor)
        if descriptor is None or self.model is None:
            raise RuntimeError(f"{label}: direct descriptor/model creation "
                               f"failed")
        self.descriptor = descriptor

        local = Path(str(self.model.localModelPath()))
        (local / "weights").mkdir(parents=True, exist_ok=True)
        (local / "model.mil").write_bytes(bytes(mil))
        (local / "weights" / "weight.bin").write_bytes(bytes(blob))

        started = time.perf_counter()
        ok, error = self.model.compileWithQoS_options_error_(_QOS, {}, None)
        if not ok:
            raise RuntimeError(f"{label} direct compile failed: {error}")
        _log.debug("%s direct compile %.2f s", label,
                   time.perf_counter() - started)
        self.inputs: list[Port] = []
        self.outputs: list[Port] = []
        self._loaded = False

    # ------------------------------------------------------- lifecycle

    def load(self) -> None:
        global _RESIDENT
        with _RESIDENT_LOCK:
            if _RESIDENT is not None:
                raise RuntimeError(
                    f"direct ANE program {_RESIDENT!r} is still resident; "
                    f"refusing to co-load {self.label!r} (co-residency "
                    f"corrupts ANE state for this executable identity)")
            _RESIDENT = self.label
        ok, error = self.model.loadWithQoS_options_error_(_QOS, {}, None)
        if not ok:
            with _RESIDENT_LOCK:
                _RESIDENT = None
            raise RuntimeError(f"{self.label} direct load failed: {error}")
        self._loaded = True

    def unload(self) -> None:
        global _RESIDENT
        ok, error = self.model.unloadWithQoS_error_(_QOS, None)
        self._loaded = False
        with _RESIDENT_LOCK:
            if self.label == _RESIDENT:
                _RESIDENT = None
        if not ok:
            raise RuntimeError(f"{self.label} direct unload failed: {error}")

    def close(self) -> None:
        if self._loaded:
            self.unload()

    # ----------------------------------------------------------- ports

    def discover_ports(self, shared: dict[str, Surface] | None = None) -> None:
        """Read live IO from a transient load and attach surfaces.

        ``shared`` maps input names to existing surfaces (zero-copy
        handoffs between chained programs).
        """
        self.load()
        try:
            attributes = self.model.modelAttributes()
            network = attributes["NetworkStatusList"][0]
            for info in network["LiveInputList"]:
                name = _symbol(info)
                port = Port(name, info, None)  # type: ignore[arg-type]
                if shared and name in shared:
                    surface = shared[name]
                    if surface.nbytes != port.nbytes:
                        raise RuntimeError(
                            f"{self.label}: shared surface for {name!r} is "
                            f"{surface.nbytes} B, model wants {port.nbytes}")
                    port.surface = surface
                else:
                    port.surface = Surface(port.nbytes)
                self.inputs.append(port)
            for info in network["LiveOutputList"]:
                port = Port(_symbol(info), info, None)  # type: ignore[arg-type]
                port.surface = Surface(port.nbytes)
                self.outputs.append(port)
        finally:
            self.unload()
        if not self.inputs or not self.outputs:
            raise RuntimeError(f"{self.label}: model reports no live IO")

    def input(self, name: str) -> Port:
        return _find(self.inputs, name, self.label)

    def output(self, name: str) -> Port:
        return _find(self.outputs, name, self.label)

    # -------------------------------------------------------- dispatch

    def _request(self) -> Any:
        bridge = preflight()
        inputs = [port.surface.wrapped for port in self.inputs]
        outputs = [port.surface.wrapped for port in self.outputs]
        request = (bridge.request_class.
                   requestWithInputs_inputIndices_outputs_outputIndices_procedureIndex_(
                       inputs, list(range(len(inputs))),
                       outputs, list(range(len(outputs))), 0))
        if request is None:
            raise RuntimeError(f"{self.label} direct request creation failed")
        return request

    def dispatch(self) -> None:
        """One full lifecycle: load, map, evaluate, unmap, unload."""
        self.load()
        try:
            request = self._request()
            ok, error = (self.model.
                         mapIOSurfacesWithRequest_cacheInference_error_(
                             request, True, None))
            if not ok:
                raise RuntimeError(f"{self.label} direct map failed: {error}")
            try:
                ok, error = self.model.evaluateWithQoS_options_request_error_(
                    _QOS, {}, request, None)
                if not ok:
                    raise RuntimeError(
                        f"{self.label} direct eval failed: {error}")
            finally:
                self.model.unmapIOSurfacesWithRequest_(request)
        finally:
            self.unload()


def _find(ports: list[Port], name: str, label: str) -> Port:
    for port in ports:
        if port.name == name:
            return port
    raise KeyError(f"{label}: no port named {name!r}")


__all__ = ["DirectModel", "DirectUnavailable", "Port", "Surface",
           "available", "preflight"]
