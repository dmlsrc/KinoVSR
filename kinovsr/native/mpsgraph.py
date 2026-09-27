"""Generic MPSGraph machinery with Neural Engine placement.

MPSGraph is Apple's graph compiler above Metal.  Its public header admits
only the Metal device type, but the shipping framework weak-links
AppleNeuralEngine and ANECompiler and carries a compile-time placement
pass that lowers eligible regions onto the ANE.  The knobs are SPI (absent
from the SDK header, exported by the binary and its .tbd link stub;
verified on macOS 26.x, M1 Max):

* device types: Metal 0, CPU 1, ANE 2;
* ``MPSGraphCompilationDescriptor.optimizationLevel = 1`` enables the
  placement pass and ``preferredDevice = 2`` requests the ANE;
* ``printANEPlacementAnalysis = True`` prints the resulting partition
  (one ``ANERegionCall`` per contiguous ANE region) - diagnosis only;
* the ANE ``MPSGraphDevice`` is NOT runnable: execution always goes
  through a Metal device, and placed regions dispatch to the ANE from
  inside the executable.

Everything here is model-agnostic plumbing: build an NCHW graph from MLX
arrays, compile it for the ANE (or the GPU), and step it with named feeds
and results.  Model graphs live with their processor families; this
module must not import any of them.

Known ANE lowering defect (measured, macOS 26.x): when a bias addition
feeds a ``depthToSpace2D`` pixel shuffle, ANECompiler sinks the bias
through the shuffle and indexes it by the POST-shuffle channel - output
channel ``c`` receives ``bias[c]`` instead of
``bias[c*r*r + r*(y%r) + (x%r)]``.  The error is additive and bias-sized,
and inserting ops between the bias and the shuffle does not help (the
fusion sinks through them).  Use :meth:`GraphBuilder.pixel_shuffle_biased`
for the exact spelling: keep the producing convolution bias-free and add
the bias AFTER the shuffle as a precomputed full-size constant.  (A
runtime tile op also works numerically but cannot be placed on the ANE;
folding conv+shuffle into a transposed convolution is exact too but
measured slower here.)

Everything fails loudly: compile or dispatch errors surface as ObjC
exceptions.  There is no fallback logic at this layer.
"""

import ctypes
import functools
import json
import logging
import shutil
import tempfile
from collections.abc import Callable, Hashable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, TypedDict, cast

import Metal
import MetalPerformanceShaders as MPS
import mlx.core as mx
import objc
from Foundation import NSURL, NSData, NSMutableDictionary

from kinovsr.media.buffer import mlx_array_from_buffer

# MPSDataType values from MPSCoreTypes.h: floatBit (0x10000000) | width.
FLOAT16 = 0x10000010
FLOAT32 = 0x10000020

_NCHW = 0  # MPSGraphTensorNamedDataLayoutNCHW
_OIHW = 2  # MPSGraphTensorNamedDataLayoutOIHW
_PADDING_EXPLICIT = 0  # MPSGraphPaddingStyleExplicit

DEVICE_GPU = 0  # MPSGraphDeviceTypeMetal (public)
DEVICE_ANE = 2  # MPSGraphDeviceTypeANE (SPI)

_FRAMEWORK = (
    "/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/MetalPerformanceShadersGraph"
)

_MX_OF = {FLOAT16: mx.float16, FLOAT32: mx.float32}
_CACHE_FORMAT = 1
_log = logging.getLogger("kinovsr.mpsgraph")


# Structural views of the ObjC objects reached through ``_fw()`` here and in
# mpsgraph_state; each lists only the selectors that code calls.
class _Class[T](Protocol):
    def alloc(self) -> T: ...


class _BufferContents(Protocol):
    def as_buffer(self, count: int) -> memoryview: ...


class _MTLBuffer(Protocol):
    def contents(self) -> _BufferContents: ...


class _MTLDevice(Protocol):
    def newCommandQueue(self) -> object: ...
    def newBufferWithLength_options_(self, length: int, options: int) -> _MTLBuffer: ...


class _AutoreleasePool(Protocol):
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: object, value: object, traceback: object) -> None: ...


class _ObjCModule(Protocol):
    def autorelease_pool(self) -> _AutoreleasePool: ...
    def lookUpClass(self, name: str) -> object: ...
    def objc_object(self, *, c_void_p: int) -> object: ...
    def createOpaquePointerType(self, name: str, typestr: bytes, doc: str) -> object: ...


class _MetalModule(Protocol):
    def MTLCreateSystemDefaultDevice(self) -> object: ...


class _NSDataClass(Protocol):
    def dataWithBytes_length_(self, payload: bytes, length: int) -> object: ...


class _NSMutableDictionaryClass(Protocol):
    def dictionary(self) -> object: ...


class _NSURLClass(Protocol):
    def fileURLWithPath_(self, path: str) -> object: ...


class _GraphDeviceClass(Protocol):
    def deviceWithMTLDevice_(self, device: object) -> object: ...


class _Convolution2DDescriptorClass(Protocol):
    def descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout_(
        self,
        stride_x: int,
        stride_y: int,
        dilation_x: int,
        dilation_y: int,
        groups: int,
        padding_left: int,
        padding_right: int,
        padding_top: int,
        padding_bottom: int,
        padding_style: int,
        data_layout: int,
        weights_layout: int,
    ) -> object: ...


class _ShapedType(Protocol):
    def initWithShape_dataType_(self, shape: object, data_type: int) -> object: ...


class _TensorData(Protocol):
    def initWithMTLBuffer_shape_dataType_(
        self, buffer: object, shape: object, data_type: int
    ) -> object: ...


class _EntryPoint(Protocol):
    def initWithEntryFunctionName_inputTypes_(self, name: str, input_types: object) -> object: ...


class _CompilationDescriptor(Protocol):
    def init(self) -> _CompilationDescriptor: ...
    def setOptimizationLevel_(self, level: int) -> None: ...
    def setPreferredDevice_(self, device: int) -> None: ...
    def setEnableANEFWToFWSignal_(self, enabled: bool) -> None: ...
    def setEnableANELateLatch_(self, enabled: bool) -> None: ...
    def setPrintANEPlacementAnalysis_(self, enabled: bool) -> None: ...
    def setWaitForCompilationCompletion_(self, enabled: bool) -> None: ...
    def setEnableMLIRDiagnostics_(self, enabled: bool) -> None: ...


class _AneSessionDescriptor(Protocol):
    def init(self) -> _AneSessionDescriptor: ...
    def setShapedEntryPoint_(self, entry_point: object) -> None: ...
    def setEnergyEffecientWorkload_(self, enabled: bool) -> None: ...
    def setAneSessionSignal_(self, signal: int) -> None: ...


class _SerializationDescriptor(Protocol):
    def init(self) -> _SerializationDescriptor: ...
    def setAppend_(self, append: bool) -> None: ...


class _Executable(Protocol):
    def initWithMPSGraphPackageAtURL_compilationDescriptor_(
        self, url: object, descriptor: object
    ) -> _Executable: ...
    def initWithMLIRSource_executableDescriptor_(
        self, source: object, descriptor: object
    ) -> _Executable: ...
    def initWithSpecializedMLIRModule_device_shapedEntryPoint_compilationDescriptor_executableDescriptor_(
        self,
        module: object,
        device: object,
        entry_point: object,
        compilation: object,
        descriptor: object,
    ) -> _Executable | None: ...
    def feedTensors(self) -> Iterable[object]: ...
    def functionNames(self) -> Iterable[object]: ...
    def getIR(self) -> object: ...
    def getMutableWeightsFilePath(self) -> object: ...
    def getStateInputPositionsWithEntryFunctionName_(self, name: str) -> Iterable[int]: ...
    def setOptions_(self, options: int) -> None: ...
    def setValue_forKey_(self, value: str, key: str) -> None: ...
    def dumpCompiledProducts(self) -> object: ...
    def specializeWithDevice_inputTypes_compilationDescriptor_(
        self, device: object, input_types: object, descriptor: object
    ) -> None: ...
    def specializeWithDevice_shapedEntryPoints_compilationDescriptor_error_(
        self, device: object, entry_points: object, descriptor: object, error: None
    ) -> object: ...
    def specializedModuleWithDevice_shapedEntryPoints_compilationDescriptor_error_(
        self, device: object, entry_points: object, descriptor: object, error: None
    ) -> object: ...
    def serializeToMPSGraphPackageAtURL_descriptor_(
        self, url: object, descriptor: object
    ) -> None: ...
    def runWithDevice_inputsArray_resultsArray_executionDescriptor_(
        self, device: object, inputs: object, results: object, execution: None
    ) -> object: ...
    def runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor_(
        self, queue: object, inputs: object, results: object, execution: None
    ) -> object: ...
    def sendANEStreamingSessionSignal_sessionDescriptor_report_(
        self, device: object, session: object, report: object
    ) -> object: ...


class _Graph(Protocol):
    def init(self) -> _Graph: ...
    def placeholderWithShape_dataType_name_(
        self, shape: object, data_type: int, name: str
    ) -> object: ...
    def constantWithData_shape_dataType_(
        self, data: object, shape: object, data_type: int
    ) -> object: ...
    def convolution2DWithSourceTensor_weightsTensor_descriptor_name_(
        self, source: object, weights: object, descriptor: object, name: str
    ) -> object: ...
    def additionWithPrimaryTensor_secondaryTensor_name_(
        self, primary: object, secondary: object, name: str
    ) -> object: ...
    def subtractionWithPrimaryTensor_secondaryTensor_name_(
        self, primary: object, secondary: object, name: str
    ) -> object: ...
    def multiplicationWithPrimaryTensor_secondaryTensor_name_(
        self, primary: object, secondary: object, name: str
    ) -> object: ...
    def reLUWithTensor_name_(self, tensor: object, name: str) -> object: ...
    def clampWithTensor_minValueTensor_maxValueTensor_name_(
        self, tensor: object, low: object, high: object, name: str
    ) -> object: ...
    def sliceTensor_dimension_start_length_name_(
        self, tensor: object, dimension: int, start: int, length: int, name: str
    ) -> object: ...
    def concatTensors_dimension_name_(
        self, tensors: object, dimension: int, name: str
    ) -> object: ...
    def reshapeTensor_withShape_name_(self, tensor: object, shape: object, name: str) -> object: ...
    def depthToSpace2DTensor_widthAxis_heightAxis_depthAxis_blockSize_usePixelShuffleOrder_name_(
        self,
        tensor: object,
        width_axis: int,
        height_axis: int,
        depth_axis: int,
        block_size: int,
        pixel_shuffle_order: bool,
        name: str,
    ) -> object: ...
    def compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor_(
        self,
        device: object,
        feeds: object,
        target_tensors: object,
        target_operations: None,
        descriptor: object,
    ) -> _Executable: ...


class _Framework(TypedDict):
    MPSGraph: _Class[_Graph]
    MPSGraphDevice: _GraphDeviceClass
    MPSGraphCompilationDescriptor: _Class[_CompilationDescriptor]
    MPSGraphShapedType: _Class[_ShapedType]
    MPSGraphTensorData: _Class[_TensorData]
    MPSGraphConvolution2DOpDescriptor: _Convolution2DDescriptorClass
    MPSNDArray: object
    MPSNDArrayDescriptor: object
    MPSGraphExecutableEntryPoint: _Class[_EntryPoint]
    MPSGraphAneSessionDescriptor: _Class[_AneSessionDescriptor]
    MPSGraphExecutable: _Class[_Executable]
    MPSGraphExecutableSerializationDescriptor: _Class[_SerializationDescriptor]
    NSData: _NSDataClass
    NSMutableDictionary: _NSMutableDictionaryClass
    NSURL: _NSURLClass
    Metal: _MetalModule
    objc: _ObjCModule


# JSON spelling of one (name, shape) pair: [name, [dims...]].
type _PortJson = list[str | list[int]]


class _GraphContract(TypedDict):
    format: int
    dtype: int
    order: list[_PortJson]
    targets: list[_PortJson]


# One graph-owned tensor-data object and the CPU view of its shared backing.
type _Backing = tuple[object, memoryview]


@functools.cache
def _fw() -> _Framework:
    """Load the framework once and resolve the ObjC classes.

    Class resolution goes through ``objc.lookUpClass`` rather than a
    pyobjc framework package: the generated binding carries none of the
    ANE SPI anyway, and the ObjC runtime's type encodings make every
    selector callable without it.
    """
    ctypes.CDLL(_FRAMEWORK)

    # Registers pyobjc metadata for MPSNDArray (readBytes:strideBytes:
    # takes a buffer, not an integer); also the authority for the
    # MPSDataType values hardcoded above.

    if (int(MPS.MPSDataTypeFloat16), int(MPS.MPSDataTypeFloat32)) != (FLOAT16, FLOAT32):
        raise RuntimeError("MPSDataType constants moved; update this module")

    names = (
        "MPSGraph",
        "MPSGraphDevice",
        "MPSGraphCompilationDescriptor",
        "MPSGraphShapedType",
        "MPSGraphTensorData",
        "MPSGraphConvolution2DOpDescriptor",
        "MPSNDArray",
        "MPSNDArrayDescriptor",
        "MPSGraphExecutableEntryPoint",
        "MPSGraphAneSessionDescriptor",
        "MPSGraphExecutable",
        "MPSGraphExecutableSerializationDescriptor",
    )
    handles: dict[str, object] = {name: objc.lookUpClass(name) for name in names}
    handles["NSData"] = NSData
    handles["NSMutableDictionary"] = NSMutableDictionary
    handles["NSURL"] = NSURL
    handles["Metal"] = Metal
    handles["objc"] = objc
    return cast(_Framework, handles)  # exactly the keys _Framework declares


def _nsdata(arr: mx.array, mx_dtype: mx.Dtype) -> object:
    payload = bytes(memoryview(mx.contiguous(arr.astype(mx_dtype))))
    return _fw()["NSData"].dataWithBytes_length_(payload, len(payload))


class GraphBuilder:
    """One MPSGraph under construction, NCHW, single dtype.

    Placeholders are registered with names; :func:`compile_graph` turns
    the builder plus a named target list into a :class:`CompiledGraph`
    that is fed and read by those names.
    """

    def __init__(self, dtype: int = FLOAT16):
        if dtype not in _MX_OF:
            raise ValueError(f"unsupported MPSDataType 0x{dtype:x}")
        fw = _fw()
        self.graph = fw["MPSGraph"].alloc().init()
        self.dtype = dtype
        self.mx_dtype = _MX_OF[dtype]
        self.feeds: list[tuple[object, tuple[int, ...], str]] = []
        self._conv_descs: dict[tuple[int, int], object] = {}
        # Cache keys carry id() of the source array, so each entry also holds
        # that array: a freed temporary's id is reused at once by the next
        # array, which would otherwise hit this cache with other data.
        self._constants: dict[tuple[Hashable, ...], tuple[object, mx.array]] = {}

    # ------------------------------------------------------------ inputs

    def placeholder(self, shape: tuple[int, ...], name: str) -> object:
        tensor = self.graph.placeholderWithShape_dataType_name_(list(shape), self.dtype, name)
        self.feeds.append((tensor, tuple(int(s) for s in shape), name))
        return tensor

    def constant(
        self,
        arr: mx.array,
        shape: tuple[int, ...] | None = None,
        *,
        cache_key: tuple[Hashable, ...] | None = None,
    ) -> object:
        if shape is None:
            shape = tuple(int(s) for s in arr.shape)
        if cache_key is not None and cache_key in self._constants:
            return self._constants[cache_key][0]
        tensor = self.graph.constantWithData_shape_dataType_(
            _nsdata(arr, self.mx_dtype), list(shape), self.dtype
        )
        if cache_key is not None:
            self._constants[cache_key] = (tensor, arr)
        return tensor

    # --------------------------------------------------------------- ops

    def _conv_desc(self, stride: int, pad: int) -> object:
        key = (stride, pad)
        if key not in self._conv_descs:
            self._conv_descs[key] = _fw()[
                "MPSGraphConvolution2DOpDescriptor"
            ].descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout_(
                stride, stride, 1, 1, 1, pad, pad, pad, pad, _PADDING_EXPLICIT, _NCHW, _OIHW
            )
        return self._conv_descs[key]

    def conv2d(
        self,
        x: object,
        weight: mx.array,
        bias: mx.array | None = None,
        *,
        stride: int = 1,
        pad: int = 1,
        name: str,
    ) -> object:
        """3x3-style convolution; ``weight`` is an MLX OIHW array.

        ``bias`` (length O) is added as a separate (1, O, 1, 1) constant.
        Leave it ``None`` when the result feeds a pixel shuffle and route
        the bias through :meth:`pixel_shuffle_biased` instead (see the
        module docstring for the ANE defect this avoids).
        """
        o, i, kh, kw = (int(s) for s in weight.shape)
        y = self.graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name_(
            x,
            self.constant(
                weight, (o, i, kh, kw), cache_key=("conv.weight", id(weight), o, i, kh, kw)
            ),
            self._conv_desc(stride, pad),
            name,
        )
        if bias is None:
            return y
        return self.add(
            y,
            self.constant(bias, (1, o, 1, 1), cache_key=("conv.bias", id(bias), o)),
            name + ".bias",
        )

    def add(self, a: object, b: object, name: str) -> object:
        return self.graph.additionWithPrimaryTensor_secondaryTensor_name_(a, b, name)

    def subtract(self, a: object, b: object, name: str) -> object:
        return self.graph.subtractionWithPrimaryTensor_secondaryTensor_name_(a, b, name)

    def multiply(self, a: object, b: object, name: str) -> object:
        return self.graph.multiplicationWithPrimaryTensor_secondaryTensor_name_(a, b, name)

    def relu(self, x: object, name: str) -> object:
        return self.graph.reLUWithTensor_name_(x, name)

    def clamp(self, x: object, low: float, high: float, name: str) -> object:
        lo = self.constant(mx.full((1,), low), cache_key=("scalar", float(low)))
        hi = self.constant(mx.full((1,), high), cache_key=("scalar", float(high)))
        return self.graph.clampWithTensor_minValueTensor_maxValueTensor_name_(x, lo, hi, name)

    def slice_channels(self, x: object, start: int, length: int, name: str) -> object:
        return self.graph.sliceTensor_dimension_start_length_name_(x, 1, start, length, name)

    def concat_channels(self, tensors: list[object], name: str) -> object:
        return self.graph.concatTensors_dimension_name_(tensors, 1, name)

    def reshape(self, x: object, shape: tuple[int, ...], name: str) -> object:
        """Return a static reshape without changing element order."""
        return self.graph.reshapeTensor_withShape_name_(x, [int(value) for value in shape], name)

    def pixel_shuffle(self, x: object, block: int, name: str) -> object:
        """Raw depth-to-space in pixel-shuffle order.

        HAZARD: on the ANE this miscomputes when a bias addition feeds it
        (see the module docstring).  Safe only when nothing upstream adds
        a per-channel bias; otherwise use :meth:`pixel_shuffle_biased`.
        """
        return self.graph.depthToSpace2DTensor_widthAxis_heightAxis_depthAxis_blockSize_usePixelShuffleOrder_name_(
            x, 3, 2, 1, block, True, name
        )

    def pixel_shuffle_biased(
        self,
        x: object,
        bias: mx.array,
        *,
        channels: int,
        height: int,
        width: int,
        block: int = 2,
        name: str,
    ) -> object:
        """Pixel shuffle of a bias-free tensor, bias applied after.

        ``x`` is (1, channels, height, width) with NO bias applied;
        ``bias`` is the length-``channels`` vector that would have been
        added before the shuffle.  The bias lands after the shuffle as a
        full-size constant precomputed with ``mx.tile`` - the exact
        arrangement, and the only spelling of it that places on the ANE.
        """
        if channels % (block * block):
            raise ValueError(f"{channels} channels not divisible by {block}x{block}")
        out_channels = channels // (block * block)
        y = self.pixel_shuffle(x, block, name)
        pattern = bias.reshape(1, out_channels, block, block)
        tiled = mx.tile(pattern, (1, 1, height, width))
        return self.add(
            y,
            self.constant(
                tiled, cache_key=("pixel_shuffle.bias", id(bias), channels, height, width, block)
            ),
            name + ".bias",
        )


class TensorBinding:
    """One reusable ``MPSGraphTensorData`` backed by shared Metal storage.

    Bindings are allocated by :meth:`CompiledGraph.bind` and may replace a
    feed or result carrying the same name on each dispatch.  This is the
    zero-copy primitive for delay rings: an old result binding can become a
    later feed binding without snapshotting it through Python.
    """

    __slots__ = ("_buffer", "_data", "_owner", "_view", "shape")

    def __init__(
        self,
        owner: CompiledGraph,
        data: object,
        buffer: object,
        view: memoryview,
        shape: tuple[int, ...],
    ):
        self._owner = owner
        self._data = data
        self._buffer = buffer
        self._view = view
        self.shape = shape

    def write(self, value: bytes | bytearray | memoryview | mx.array) -> None:
        """Copy bytes or an MLX array into this binding's shared storage."""
        if isinstance(value, (bytes, bytearray, memoryview)):
            payload = value
        else:
            shape = tuple(int(s) for s in value.shape)
            if shape != self.shape:
                raise ValueError(f"binding expects shape {self.shape}, got {shape}")
            payload = memoryview(mx.contiguous(value.astype(self._owner.mx_dtype))).cast("B")
        if len(payload) != len(self._view):
            raise ValueError(f"binding expects {len(self._view)} bytes, got {len(payload)}")
        self._view[:] = payload

    def array(self) -> mx.array:
        """Snapshot this binding as an MLX array."""
        return (
            mlx_array_from_buffer(bytes(self._view)).view(self._owner.mx_dtype).reshape(self.shape)
        )


class CompiledGraph:
    """A compiled executable stepped with named MLX feeds and results.

    All tensor data is allocated ONCE over shared-storage ``MTLBuffer``
    backings. Fixed feeds are memcpy'd into persistent views and fixed
    results can be snapshotted back out. Names declared ``dynamic`` instead
    take a :class:`TensorBinding` per dispatch, allowing a result backing to
    become a later feed with no host copy. Recreating the ObjC tensor-data
    graph every step instead (the obvious spelling) floods the autorelease
    pool and stretches host prep past the ~10 ms idle threshold where every
    ANE dispatch pays a power-state ramp - measured as a 5x whole-pipeline
    slowdown before this design.

    Loopback pairs (``result name -> feed name``) keep recurrent state on
    the device entirely: each pair gets a ping-pong buffer PAIR, the run
    at parity p reads the feed from buffer p and writes the result into
    buffer 1-p, and the parities alternate - so recurrence costs no host
    round trip and no in-place aliasing within a single execution (which
    MPSGraph does not promise to order).  Loopback state starts zeroed;
    :meth:`reset` rezeroes it.
    """

    def __init__(
        self,
        exe: object,
        device: object,
        metal_device: object,
        dtype: int,
        order: list[tuple[str, tuple[int, ...]]],
        targets: list[tuple[str, tuple[int, ...]]],
        loopback: Mapping[str, str] | None = None,
        dynamic: set[str] | None = None,
        loopback_in_place: bool = False,
        use_command_queue: bool = False,
        compilation_descriptor: object | None = None,
    ):
        self._exe = cast(_Executable, exe)  # an MPSGraphExecutable from compile_graph
        self._device = device
        self._metal_device = cast(_MTLDevice, metal_device)  # the MTLDevice compile_graph made
        self._command_queue = (
            cast(_MTLDevice, metal_device).newCommandQueue() if use_command_queue else None
        )
        self._compilation_descriptor = compilation_descriptor
        self._ane_session: _AneSessionDescriptor | None = None
        self._closed = False
        self.dtype = dtype
        self.mx_dtype = _MX_OF[dtype]
        self._order = order
        self._targets = targets
        self._element = 2 if dtype == FLOAT16 else 4
        loopback = dict(loopback or {})
        feed_shape = dict(order)
        target_shape = dict(targets)
        dynamic = set(dynamic or ())
        unknown = dynamic - (set(feed_shape) | set(target_shape))
        if unknown:
            raise ValueError(f"unknown dynamic tensor names: {sorted(unknown)}")
        loop_names = set(loopback) | set(loopback.values())
        overlap = dynamic & loop_names
        if overlap:
            raise ValueError(f"loopback tensors cannot also be dynamic: {sorted(overlap)}")
        for result_name, feed_name in loopback.items():
            if result_name not in target_shape or feed_name not in feed_shape:
                # Two unknown names would otherwise compare None == None and
                # pass, silently dropping the recurrence.
                raise ValueError(f"unknown loopback tensor names: {result_name} -> {feed_name}")
            if target_shape.get(result_name) != feed_shape.get(feed_name):
                raise ValueError(
                    f"loopback {result_name} -> {feed_name}: shapes "
                    f"{target_shape.get(result_name)} vs "
                    f"{feed_shape.get(feed_name)} do not match"
                )
        looped_feeds = set(loopback.values())

        self._buffers: list[object] = []  # keeps every backing alive
        self._data_buffers: dict[int, object] = {}
        self._feed_shapes = feed_shape
        self._target_shapes = target_shape

        # Every feed is double-buffered by parity: looped ones because the
        # loopback result lands in the opposite parity's buffer, host-fed
        # ones so `write_feeds` may fill the NEXT dispatch's buffers while
        # the current dispatch is still reading its own.
        feed_pair: dict[str, tuple[_Backing, _Backing]] = {}
        self._writes: list[list[tuple[str, memoryview]]] = [[], []]
        feed_data: list[list[object]] = [[], []]
        self._dynamic_feeds: dict[str, int] = {}
        for name, shape in order:
            if name in dynamic:
                self._dynamic_feeds[name] = len(feed_data[0])
                feed_data[0].append(None)
                feed_data[1].append(None)
                continue
            if name in looped_feeds and loopback_in_place:
                one = self._backing(shape)
                pair = (one, one)
            else:
                pair = (self._backing(shape), self._backing(shape))
            for parity in (0, 1):
                feed_data[parity].append(pair[parity][0])
            if name in looped_feeds:
                feed_pair[name] = pair
            else:
                for parity in (0, 1):
                    self._writes[parity].append((name, pair[parity][1]))
        self._feed_data = feed_data
        self._loop_views = [view for pair in feed_pair.values() for _, view in pair]

        # Results: a looped result writes the OPPOSITE parity's feed
        # buffer; other results get their own backing plus a read view.
        self._reads: dict[str, tuple[memoryview, tuple[int, ...]]] = {}
        result_data: list[list[object]] = [[], []]
        self._dynamic_results: dict[str, int] = {}
        for name, shape in targets:
            if name in loopback:
                pair = feed_pair[loopback[name]]
                result_data[0].append(pair[1][0])
                result_data[1].append(pair[0][0])
            elif name in dynamic:
                self._dynamic_results[name] = len(result_data[0])
                result_data[0].append(None)
                result_data[1].append(None)
            else:
                data, view = self._backing(shape)
                self._reads[name] = (view, shape)
                result_data[0].append(data)
                result_data[1].append(data)
        self._result_data = result_data
        self._parity = 0
        # MTLBuffer contents are undefined at creation; recurrent state
        # must begin as zeros.
        self.reset()

    def _backing(self, shape: tuple[int, ...]) -> _Backing:
        count = self._element
        for s in shape:
            count *= s
        # Storage mode shared (0): the CPU view and the graph see the same
        # memory, so fixed-feed writes are plain memcpys.
        buffer = self._metal_device.newBufferWithLength_options_(count, 0)
        if buffer is None:
            raise MemoryError(f"Metal could not allocate {count} bytes for an MPSGraph tensor")
        self._buffers.append(buffer)
        data = (
            _fw()["MPSGraphTensorData"]
            .alloc()
            .initWithMTLBuffer_shape_dataType_(buffer, list(shape), self.dtype)
        )
        self._data_buffers[id(data)] = buffer
        return data, buffer.contents().as_buffer(count)

    @property
    def feed_names(self) -> list[str]:
        return [name for name, _ in self._order]

    def bind(self, name: str, *, shared: TensorBinding | None = None) -> TensorBinding:
        """Bind a dynamic tensor to new or explicitly shared Metal storage.

        ``shared`` creates a graph-local ``MPSGraphTensorData`` view over
        another graph's backing. This is the zero-copy handoff used by
        schedule-specialized executables: each executable owns its tensor
        data object, while all of them address the same recurrent/ring
        ``MTLBuffer``.
        """
        shape = self._feed_shapes.get(name, self._target_shapes.get(name))
        if shape is None:
            raise KeyError(name)
        if name not in self._dynamic_feeds and name not in self._dynamic_results:
            raise ValueError(f"tensor '{name}' is not dynamic")
        if shared is None:
            data, view = self._backing(shape)
            buffer = self._data_buffers[id(data)]
        else:
            if not isinstance(shared, TensorBinding):
                raise TypeError("shared backing must be a TensorBinding")
            if shared.shape != shape:
                raise ValueError(f"shared binding for '{name}' expects {shape}, got {shared.shape}")
            if shared._owner.dtype != self.dtype:
                raise ValueError(f"shared binding for '{name}' has a different dtype")
            buffer, view = shared._buffer, shared._view
            self._buffers.append(buffer)
            data = (
                _fw()["MPSGraphTensorData"]
                .alloc()
                .initWithMTLBuffer_shape_dataType_(buffer, list(shape), self.dtype)
            )
            self._data_buffers[id(data)] = buffer
        return TensorBinding(self, data, buffer, view, shape)

    def _ane_session_signal(self, value: int) -> bool:
        if self._ane_session is None:
            return False
        self._ane_session.setAneSessionSignal_(value)
        report = _fw()["NSMutableDictionary"].dictionary()
        return bool(
            self._exe.sendANEStreamingSessionSignal_sessionDescriptor_report_(
                self._device, self._ane_session, report
            )
        )

    def start_ane_session(self, *, energy_efficient: bool = False) -> None:
        """Start MPSGraph's private ANE streaming session for this graph.

        The descriptor's public-looking property is misleading: the runtime
        expects an ``MPSGraphExecutableEntryPoint`` there, then derives its
        own shaped entry point. Signal 1 is the framework's client-session
        start hint; signal 2 stops it.
        """
        if self._ane_session is not None:
            return
        fw = _fw()
        input_types = [
            fw["MPSGraphShapedType"].alloc().initWithShape_dataType_(list(shape), self.dtype)
            for _name, shape in self._order
        ]
        entry = (
            fw["MPSGraphExecutableEntryPoint"]
            .alloc()
            .initWithEntryFunctionName_inputTypes_("main", input_types)
        )
        session = fw["MPSGraphAneSessionDescriptor"].alloc().init()
        session.setShapedEntryPoint_(entry)
        session.setEnergyEffecientWorkload_(bool(energy_efficient))
        self._ane_session = session
        if not self._ane_session_signal(1):
            self._ane_session = None
            raise RuntimeError("MPSGraph rejected the ANE streaming session")

    def stop_ane_session(self) -> None:
        """Stop a live ANE streaming hint without releasing the executable."""
        if self._ane_session is None:
            return
        try:
            if not self._ane_session_signal(2):
                raise RuntimeError("MPSGraph rejected the ANE session stop signal")
        finally:
            self._ane_session = None

    def close(self) -> None:
        """Stop a live ANE streaming session and release graph backings."""
        if self._closed:
            return
        self._closed = True
        self.stop_ane_session()

    def reset(self) -> None:
        """Zero the loopback state and restart the parity sequence."""
        for view in self._loop_views:
            view[:] = bytes(len(view))
        self._parity = 0

    def write_feeds(self, values: Mapping[str, mx.array]) -> None:
        """Memcpy every non-loopback feed for the NEXT dispatch.

        Targets that dispatch's own buffer parity, so this may run while
        the previous dispatch is still executing (its buffers are the
        other parity).  MLX arrays are materialized here, on the caller's
        thread - never on a dispatch worker.
        """
        for name, view in self._writes[self._parity]:
            view[:] = memoryview(mx.contiguous(values[name].astype(self.mx_dtype))).cast("B")

    def begin_dispatch(
        self, bindings: Mapping[str, TensorBinding] | None = None
    ) -> Callable[[], None]:
        """Capture one step over the current parity's feeds as a job.

        The parity advances HERE, on the caller's thread, so a following
        ``write_feeds`` targets the next dispatch's buffers even while
        the returned job is still running on a worker.  The job itself is
        pure ObjC - safe to hand to a dispatch worker thread.  Loopback
        results land in the next parity's feed buffers, continuing the
        chain.
        """
        bindings = bindings or {}
        required = set(self._dynamic_feeds) | set(self._dynamic_results)
        missing = required - set(bindings)
        if missing:
            raise ValueError(f"missing dynamic tensor bindings: {sorted(missing)}")
        extra = set(bindings) - required
        if extra:
            raise ValueError(f"unexpected dynamic tensor bindings: {sorted(extra)}")

        parity = self._parity
        self._parity = 1 - parity
        exe, device = self._exe, self._device
        command_queue = self._command_queue
        feeds = list(self._feed_data[parity])
        results = list(self._result_data[parity])
        for name, index in self._dynamic_feeds.items():
            binding = bindings[name]
            self._check_binding(name, binding, self._feed_shapes[name])
            feeds[index] = binding._data
        for name, index in self._dynamic_results.items():
            binding = bindings[name]
            self._check_binding(name, binding, self._target_shapes[name])
            results[index] = binding._data
        pool = _fw()["objc"].autorelease_pool

        def job() -> None:
            with pool():
                if command_queue is None:
                    exe.runWithDevice_inputsArray_resultsArray_executionDescriptor_(
                        device, feeds, results, None
                    )
                else:
                    exe.runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor_(
                        command_queue, feeds, results, None
                    )

        return job

    def _check_binding(self, name: str, binding: TensorBinding, shape: tuple[int, ...]) -> None:
        if not isinstance(binding, TensorBinding) or binding._owner is not self:
            raise ValueError(f"binding for '{name}' belongs to a different graph")
        if binding.shape != shape:
            raise ValueError(f"binding for '{name}' expects {shape}, got {binding.shape}")

    def dispatch(self, bindings: Mapping[str, TensorBinding] | None = None) -> None:
        """Execute one step synchronously (the non-pipelined form)."""
        self.begin_dispatch(bindings)()

    def read(self, read: set[str] | None = None) -> dict[str, mx.array]:
        """Snapshot non-loopback results of the last finished dispatch.

        ``read`` limits which targets are copied out (default: all) -
        skipping unneeded multi-megabyte reads is a real saving in a
        per-frame loop.  Only call with no dispatch in flight: result
        backings are single-buffered.
        """
        out: dict[str, mx.array] = {}
        for name, (view, shape) in self._reads.items():
            if read is not None and name not in read:
                continue
            # bytes() snapshots the shared buffer before the next
            # dispatch rewrites it; mx views the copy without another one.
            out[name] = mlx_array_from_buffer(bytes(view)).view(self.mx_dtype).reshape(shape)
        return out

    def run(
        self,
        values: Mapping[str, mx.array],
        read: set[str] | None = None,
        bindings: Mapping[str, TensorBinding] | None = None,
    ) -> dict[str, mx.array]:
        """The synchronous composite: write, dispatch, read."""
        self.write_feeds(values)
        self.dispatch(bindings)
        return self.read(read)


def _compilation_descriptor(
    fw: _Framework,
    device: int,
    *,
    placement_report: bool,
    ane_fw_to_fw_signal: bool,
    ane_late_latch: bool,
) -> _CompilationDescriptor:
    descriptor = fw["MPSGraphCompilationDescriptor"].alloc().init()
    if device == DEVICE_ANE:
        descriptor.setOptimizationLevel_(1)
        descriptor.setPreferredDevice_(DEVICE_ANE)
        if ane_fw_to_fw_signal:
            descriptor.setEnableANEFWToFWSignal_(True)
        if ane_late_latch:
            descriptor.setEnableANELateLatch_(True)
        if placement_report:
            descriptor.setPrintANEPlacementAnalysis_(True)
    else:
        descriptor.setOptimizationLevel_(0)
        descriptor.setPreferredDevice_(device)
    return descriptor


def _target_contract(
    targets: list[tuple[str, object, tuple[int, ...]]],
) -> list[tuple[str, tuple[int, ...]]]:
    return [(name, tuple(int(value) for value in shape)) for name, _tensor, shape in targets]


def _load_cached_executable(
    cache_directory: Path,
    *,
    fw: _Framework,
    graph_device: object,
    descriptor: object,
    builder: GraphBuilder,
    targets: list[tuple[str, object, tuple[int, ...]]],
) -> tuple[_Executable, list[tuple[str, tuple[int, ...]]]] | None:
    contract_path = cache_directory / "contract.json"
    package_path = cache_directory / "model.mpsgraphpackage"
    if not (contract_path.is_file() and package_path.is_dir()):
        return None
    record: _GraphContract = json.loads(contract_path.read_text())
    if record.get("format") != _CACHE_FORMAT:
        return None
    if record.get("dtype") != builder.dtype:
        raise RuntimeError(f"MPSGraph cache dtype changed at {cache_directory}")

    feed_shapes = {
        name: tuple(int(value) for value in shape) for _tensor, shape, name in builder.feeds
    }
    order = [
        (str(name), tuple(int(value) for value in shape)) for name, shape in record.get("order", ())
    ]
    if len(order) != len(feed_shapes) or dict(order) != feed_shapes:
        raise RuntimeError(f"MPSGraph cache feed contract changed at {cache_directory}")
    expected_targets = _target_contract(targets)
    cached_targets = [
        (str(name), tuple(int(value) for value in shape))
        for name, shape in record.get("targets", ())
    ]
    if cached_targets != expected_targets:
        raise RuntimeError(f"MPSGraph cache target contract changed at {cache_directory}")

    executable = (
        fw["MPSGraphExecutable"]
        .alloc()
        .initWithMPSGraphPackageAtURL_compilationDescriptor_(
            fw["NSURL"].fileURLWithPath_(str(package_path)), descriptor
        )
    )
    input_types = [
        fw["MPSGraphShapedType"].alloc().initWithShape_dataType_(list(shape), builder.dtype)
        for _name, shape in order
    ]
    executable.specializeWithDevice_inputTypes_compilationDescriptor_(
        graph_device, input_types, descriptor
    )
    return executable, order


def _publish_executable_cache(
    cache_directory: Path,
    *,
    fw: _Framework,
    executable: _Executable,
    dtype: int,
    order: list[tuple[str, tuple[int, ...]]],
    targets: list[tuple[str, tuple[int, ...]]],
) -> None:
    """Atomically publish one derived MPSGraph package.

    Concurrent builders may both compile, but only a complete sibling
    directory becomes visible. The loser discards its derived staging
    directory and keeps using its already-compiled in-memory executable.
    """
    parent = cache_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cache_directory.name}.partial-", dir=parent))
    try:
        package = staging / "model.mpsgraphpackage"
        serialization = fw["MPSGraphExecutableSerializationDescriptor"].alloc().init()
        serialization.setAppend_(False)
        executable.serializeToMPSGraphPackageAtURL_descriptor_(
            fw["NSURL"].fileURLWithPath_(str(package)), serialization
        )
        (staging / "contract.json").write_text(
            json.dumps(
                {
                    "format": _CACHE_FORMAT,
                    "dtype": dtype,
                    "order": [[name, list(shape)] for name, shape in order],
                    "targets": [[name, list(shape)] for name, shape in targets],
                },
                indent=2,
            )
        )
        try:
            staging.replace(cache_directory)
        except OSError:
            # Another process may have won the same cold-cache race.
            if not cache_directory.is_dir():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def compile_graph(
    builder: GraphBuilder,
    targets: list[tuple[str, object, tuple[int, ...]]],
    *,
    device: int = DEVICE_ANE,
    placement_report: bool = False,
    loopback: Mapping[str, str] | None = None,
    dynamic: set[str] | None = None,
    loopback_in_place: bool = False,
    synchronize_results: bool = True,
    use_command_queue: bool = False,
    ane_fw_to_fw_signal: bool = False,
    ane_late_latch: bool = False,
    ane_streaming_session: bool = False,
    ane_energy_efficient: bool = False,
    executable_cache: str | Path | None = None,
) -> CompiledGraph:
    """Compile ``builder`` for ``device`` (`DEVICE_ANE` or `DEVICE_GPU`).

    ``targets`` is a list of (name, tensor, shape). ``loopback`` maps result
    names onto placeholder names whose next-step value they become (see
    :class:`CompiledGraph`). ``loopback_in_place`` aliases each loopback
    feed/result pair rather than ping-ponging; callers must independently
    establish that their executable consumes every input before writing
    results. ``dynamic`` names feeds/results whose backing is supplied per
    dispatch through :class:`TensorBinding`. ANE placement is a compiler
    preference, not a guarantee; ``placement_report=True`` makes the
    compiler print the realized partition to stderr for inspection.
    """
    fw = _fw()
    # Framework calls here return autoreleased objects, the executable
    # among them, and a caller's thread may never drain a pool (a script's
    # or test's main thread). Drained here, the executable lives exactly as
    # long as the returned graph, and MPSGraph removes its temp archive when
    # it is freed.
    with fw["objc"].autorelease_pool():
        metal = fw["Metal"].MTLCreateSystemDefaultDevice()
        graph_device = fw["MPSGraphDevice"].deviceWithMTLDevice_(metal)

        descriptor = _compilation_descriptor(
            fw,
            device,
            placement_report=placement_report,
            ane_fw_to_fw_signal=ane_fw_to_fw_signal,
            ane_late_latch=ane_late_latch,
        )

        shaped = {
            tensor: fw["MPSGraphShapedType"]
            .alloc()
            .initWithShape_dataType_(list(shape), builder.dtype)
            for tensor, shape, _ in builder.feeds
        }
        cached = None
        cache_directory = None if executable_cache is None else Path(executable_cache)
        # A placement report is an inspection request: compile the live graph so
        # MPSGraph emits the realized partition rather than silently reusing a
        # package compiled without diagnostics.
        if cache_directory is not None and not placement_report:
            cached = _load_cached_executable(
                cache_directory,
                fw=fw,
                graph_device=graph_device,
                descriptor=descriptor,
                builder=builder,
                targets=targets,
            )
        if cached is None:
            exe = builder.graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor_(
                graph_device, shaped, [tensor for _, tensor, _ in targets], None, descriptor
            )
            by_id = {id(tensor): (name, shape) for tensor, shape, name in builder.feeds}
            order = [by_id[id(tensor)] for tensor in exe.feedTensors()]
            if cache_directory is not None and not placement_report:
                try:
                    _publish_executable_cache(
                        cache_directory,
                        fw=fw,
                        executable=exe,
                        dtype=builder.dtype,
                        order=order,
                        targets=_target_contract(targets),
                    )
                except (OSError, ValueError) as exc:
                    _log.warning(
                        "could not cache MPSGraph executable at %s: %s", cache_directory, exc
                    )
        else:
            exe, order = cached
        if not synchronize_results:
            # Every backing this module supplies uses coherent shared storage and
            # the synchronous dispatch returns only after execution. The default
            # SynchronizeResults option is therefore a redundant result blit.
            exe.setOptions_(0)

        compiled = CompiledGraph(
            exe,
            graph_device,
            metal,
            builder.dtype,
            order,
            [(name, tuple(shape)) for name, _, shape in targets],
            loopback=loopback,
            dynamic=dynamic,
            loopback_in_place=loopback_in_place,
            use_command_queue=use_command_queue,
            compilation_descriptor=descriptor,
        )
        if ane_streaming_session:
            if device != DEVICE_ANE:
                raise ValueError("ANE streaming sessions require DEVICE_ANE")
            compiled.start_ane_session(energy_efficient=ane_energy_efficient)
        return compiled


__all__ = [
    "DEVICE_ANE",
    "DEVICE_GPU",
    "FLOAT16",
    "FLOAT32",
    "CompiledGraph",
    "GraphBuilder",
    "TensorBinding",
    "compile_graph",
]
