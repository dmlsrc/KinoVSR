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

from __future__ import annotations

import ctypes
import functools
from collections.abc import Mapping
from typing import Any

import mlx.core as mx

# MPSDataType values from MPSCoreTypes.h: floatBit (0x10000000) | width.
FLOAT16 = 0x10000010
FLOAT32 = 0x10000020

_NCHW = 0                # MPSGraphTensorNamedDataLayoutNCHW
_OIHW = 2                # MPSGraphTensorNamedDataLayoutOIHW
_PADDING_EXPLICIT = 0    # MPSGraphPaddingStyleExplicit

DEVICE_GPU = 0           # MPSGraphDeviceTypeMetal (public)
DEVICE_ANE = 2           # MPSGraphDeviceTypeANE (SPI)

_FRAMEWORK = ("/System/Library/Frameworks/MetalPerformanceShadersGraph"
              ".framework/MetalPerformanceShadersGraph")

_MX_OF = {FLOAT16: mx.float16, FLOAT32: mx.float32}


@functools.cache
def _fw() -> dict[str, Any]:
    """Load the framework once and resolve the ObjC classes.

    Class resolution goes through ``objc.lookUpClass`` rather than a
    pyobjc framework package: the generated binding carries none of the
    ANE SPI anyway, and the ObjC runtime's type encodings make every
    selector callable without it.
    """
    ctypes.CDLL(_FRAMEWORK)
    import Metal

    # Registers pyobjc metadata for MPSNDArray (readBytes:strideBytes:
    # takes a buffer, not an integer); also the authority for the
    # MPSDataType values hardcoded above.
    import MetalPerformanceShaders as MPS
    import objc
    from Foundation import NSData

    if (int(MPS.MPSDataTypeFloat16), int(MPS.MPSDataTypeFloat32)) != (
            FLOAT16, FLOAT32):
        raise RuntimeError("MPSDataType constants moved; update this module")

    names = ("MPSGraph", "MPSGraphDevice", "MPSGraphCompilationDescriptor",
             "MPSGraphShapedType", "MPSGraphTensorData",
             "MPSGraphConvolution2DOpDescriptor", "MPSNDArray",
             "MPSNDArrayDescriptor")
    handles: dict[str, Any] = {name: objc.lookUpClass(name)
                               for name in names}
    handles["NSData"] = NSData
    handles["Metal"] = Metal
    handles["objc"] = objc
    return handles


def _nsdata(arr: Any, mx_dtype: Any) -> Any:
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
        self.feeds: list[tuple[Any, tuple[int, ...], str]] = []
        self._conv_descs: dict[tuple[int, int], Any] = {}

    # ------------------------------------------------------------ inputs

    def placeholder(self, shape: tuple[int, ...], name: str) -> Any:
        tensor = self.graph.placeholderWithShape_dataType_name_(
            list(shape), self.dtype, name)
        self.feeds.append((tensor, tuple(int(s) for s in shape), name))
        return tensor

    def constant(self, arr: Any, shape: tuple[int, ...] | None = None) -> Any:
        if shape is None:
            shape = tuple(int(s) for s in arr.shape)
        return self.graph.constantWithData_shape_dataType_(
            _nsdata(arr, self.mx_dtype), list(shape), self.dtype)

    # --------------------------------------------------------------- ops

    def _conv_desc(self, stride: int, pad: int) -> Any:
        key = (stride, pad)
        if key not in self._conv_descs:
            self._conv_descs[key] = _fw()[
                "MPSGraphConvolution2DOpDescriptor"
            ].descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout_(
                stride, stride, 1, 1, 1, pad, pad, pad, pad,
                _PADDING_EXPLICIT, _NCHW, _OIHW)
        return self._conv_descs[key]

    def conv2d(self, x: Any, weight: Any, bias: Any | None = None, *,
               stride: int = 1, pad: int = 1, name: str) -> Any:
        """3x3-style convolution; ``weight`` is an MLX OIHW array.

        ``bias`` (length O) is added as a separate (1, O, 1, 1) constant.
        Leave it ``None`` when the result feeds a pixel shuffle and route
        the bias through :meth:`pixel_shuffle_biased` instead (see the
        module docstring for the ANE defect this avoids).
        """
        o, i, kh, kw = (int(s) for s in weight.shape)
        y = self.graph.convolution2DWithSourceTensor_weightsTensor_descriptor_name_(
            x, self.constant(weight, (o, i, kh, kw)),
            self._conv_desc(stride, pad), name)
        if bias is None:
            return y
        return self.add(y, self.constant(bias.reshape(1, o, 1, 1)),
                        name + ".bias")

    def add(self, a: Any, b: Any, name: str) -> Any:
        return self.graph.additionWithPrimaryTensor_secondaryTensor_name_(
            a, b, name)

    def subtract(self, a: Any, b: Any, name: str) -> Any:
        return self.graph.subtractionWithPrimaryTensor_secondaryTensor_name_(
            a, b, name)

    def multiply(self, a: Any, b: Any, name: str) -> Any:
        return self.graph.multiplicationWithPrimaryTensor_secondaryTensor_name_(
            a, b, name)

    def relu(self, x: Any, name: str) -> Any:
        return self.graph.reLUWithTensor_name_(x, name)

    def clamp(self, x: Any, low: float, high: float, name: str) -> Any:
        lo = self.constant(mx.full((1,), low))
        hi = self.constant(mx.full((1,), high))
        return self.graph.clampWithTensor_minValueTensor_maxValueTensor_name_(
            x, lo, hi, name)

    def slice_channels(self, x: Any, start: int, length: int,
                       name: str) -> Any:
        return self.graph.sliceTensor_dimension_start_length_name_(
            x, 1, start, length, name)

    def concat_channels(self, tensors: list[Any], name: str) -> Any:
        return self.graph.concatTensors_dimension_name_(tensors, 1, name)

    def pixel_shuffle(self, x: Any, block: int, name: str) -> Any:
        """Raw depth-to-space in pixel-shuffle order.

        HAZARD: on the ANE this miscomputes when a bias addition feeds it
        (see the module docstring).  Safe only when nothing upstream adds
        a per-channel bias; otherwise use :meth:`pixel_shuffle_biased`.
        """
        return self.graph.depthToSpace2DTensor_widthAxis_heightAxis_depthAxis_blockSize_usePixelShuffleOrder_name_(
            x, 3, 2, 1, block, True, name)

    def pixel_shuffle_biased(self, x: Any, bias: Any, *, channels: int,
                             height: int, width: int, block: int = 2,
                             name: str) -> Any:
        """Pixel shuffle of a bias-free tensor, bias applied after.

        ``x`` is (1, channels, height, width) with NO bias applied;
        ``bias`` is the length-``channels`` vector that would have been
        added before the shuffle.  The bias lands after the shuffle as a
        full-size constant precomputed with ``mx.tile`` - the exact
        arrangement, and the only spelling of it that places on the ANE.
        """
        if channels % (block * block):
            raise ValueError(f"{channels} channels not divisible by "
                             f"{block}x{block}")
        out_channels = channels // (block * block)
        y = self.pixel_shuffle(x, block, name)
        pattern = bias.reshape(1, out_channels, block, block)
        tiled = mx.tile(pattern, (1, 1, height, width))
        return self.add(y, self.constant(tiled), name + ".bias")


class CompiledGraph:
    """A compiled executable stepped with named MLX feeds and results.

    All tensor data is allocated ONCE over shared-storage ``MTLBuffer``
    backings; each ``run`` memcpys feed bytes into the persistent views,
    executes with preallocated result backings, and snapshots requested
    targets back out.  Recreating the ObjC tensor-data graph every step
    instead (the obvious spelling) floods the autorelease pool and
    stretches host prep past the ~10 ms idle threshold where every ANE
    dispatch pays a power-state ramp - measured as a 5x whole-pipeline
    slowdown before this design.

    Loopback pairs (``result name -> feed name``) keep recurrent state on
    the device entirely: each pair gets a ping-pong buffer PAIR, the run
    at parity p reads the feed from buffer p and writes the result into
    buffer 1-p, and the parities alternate - so recurrence costs no host
    round trip and no in-place aliasing within a single execution (which
    MPSGraph does not promise to order).  Loopback state starts zeroed;
    :meth:`reset` rezeroes it.
    """

    def __init__(self, exe: Any, device: Any, metal_device: Any, dtype: int,
                 order: list[tuple[str, tuple[int, ...]]],
                 targets: list[tuple[str, tuple[int, ...]]],
                 loopback: Mapping[str, str] | None = None):
        fw = _fw()
        self._exe = exe
        self._device = device
        self.dtype = dtype
        self.mx_dtype = _MX_OF[dtype]
        self._order = order
        self._targets = targets
        element = 2 if dtype == FLOAT16 else 4
        loopback = dict(loopback or {})
        feed_shape = dict(order)
        target_shape = dict(targets)
        for result_name, feed_name in loopback.items():
            if target_shape.get(result_name) != feed_shape.get(feed_name):
                raise ValueError(
                    f"loopback {result_name} -> {feed_name}: shapes "
                    f"{target_shape.get(result_name)} vs "
                    f"{feed_shape.get(feed_name)} do not match")
        looped_feeds = set(loopback.values())

        self._buffers: list[Any] = []           # keeps every backing alive

        def backing(shape: tuple[int, ...]) -> tuple[Any, Any]:
            count = element
            for s in shape:
                count *= s
            # Storage mode shared (0): the CPU view and the graph see the
            # same memory - feed writes are plain memcpys.
            buffer = metal_device.newBufferWithLength_options_(count, 0)
            self._buffers.append(buffer)
            data = fw["MPSGraphTensorData"].alloc(
            ).initWithMTLBuffer_shape_dataType_(buffer, list(shape), dtype)
            return data, buffer.contents().as_buffer(count)

        # Every feed is double-buffered by parity: looped ones because the
        # loopback result lands in the opposite parity's buffer, host-fed
        # ones so `write_feeds` may fill the NEXT dispatch's buffers while
        # the current dispatch is still reading its own.
        feed_pair: dict[str, tuple] = {}
        self._writes: list[list[tuple[str, Any]]] = [[], []]
        feed_data: list[list[Any]] = [[], []]
        for name, shape in order:
            pair = (backing(shape), backing(shape))
            for parity in (0, 1):
                feed_data[parity].append(pair[parity][0])
            if name in looped_feeds:
                feed_pair[name] = pair
            else:
                for parity in (0, 1):
                    self._writes[parity].append((name, pair[parity][1]))
        self._feed_data = feed_data
        self._loop_views = [view for pair in feed_pair.values()
                            for _, view in pair]

        # Results: a looped result writes the OPPOSITE parity's feed
        # buffer; other results get their own backing plus a read view.
        self._reads: dict[str, tuple[Any, tuple[int, ...]]] = {}
        result_data: list[list[Any]] = [[], []]
        for name, shape in targets:
            if name in loopback:
                pair = feed_pair[loopback[name]]
                result_data[0].append(pair[1][0])
                result_data[1].append(pair[0][0])
            else:
                data, view = backing(shape)
                self._reads[name] = (view, shape)
                result_data[0].append(data)
                result_data[1].append(data)
        self._result_data = result_data
        self._parity = 0
        # MTLBuffer contents are undefined at creation; recurrent state
        # must begin as zeros.
        self.reset()

    @property
    def feed_names(self) -> list[str]:
        return [name for name, _ in self._order]

    def reset(self) -> None:
        """Zero the loopback state and restart the parity sequence."""
        for view in self._loop_views:
            view[:] = bytes(len(view))
        self._parity = 0

    def write_feeds(self, values: Mapping[str, Any]) -> None:
        """Memcpy every non-loopback feed for the NEXT dispatch.

        Targets that dispatch's own buffer parity, so this may run while
        the previous dispatch is still executing (its buffers are the
        other parity).  MLX arrays are materialized here, on the caller's
        thread - never on a dispatch worker.
        """
        for name, view in self._writes[self._parity]:
            view[:] = memoryview(
                mx.contiguous(values[name].astype(self.mx_dtype))).cast("B")

    def begin_dispatch(self):
        """Capture one step over the current parity's feeds as a job.

        The parity advances HERE, on the caller's thread, so a following
        ``write_feeds`` targets the next dispatch's buffers even while
        the returned job is still running on a worker.  The job itself is
        pure ObjC - safe to hand to a dispatch worker thread.  Loopback
        results land in the next parity's feed buffers, continuing the
        chain.
        """
        parity = self._parity
        self._parity = 1 - parity
        exe, device = self._exe, self._device
        feeds = self._feed_data[parity]
        results = self._result_data[parity]
        pool = _fw()["objc"].autorelease_pool

        def job() -> None:
            with pool():
                exe.runWithDevice_inputsArray_resultsArray_executionDescriptor_(
                    device, feeds, results, None)

        return job

    def dispatch(self) -> None:
        """Execute one step synchronously (the non-pipelined form)."""
        self.begin_dispatch()()

    def read(self, read: set[str] | None = None) -> dict[str, Any]:
        """Snapshot non-loopback results of the last finished dispatch.

        ``read`` limits which targets are copied out (default: all) -
        skipping unneeded multi-megabyte reads is a real saving in a
        per-frame loop.  Only call with no dispatch in flight: result
        backings are single-buffered.
        """
        out: dict[str, Any] = {}
        for name, (view, shape) in self._reads.items():
            if read is not None and name not in read:
                continue
            # bytes() snapshots the shared buffer before the next
            # dispatch rewrites it; mx views the copy without another one.
            out[name] = mx.array(bytes(view)).view(
                self.mx_dtype).reshape(shape)
        return out

    def run(self, values: Mapping[str, Any],
            read: set[str] | None = None) -> dict[str, Any]:
        """The synchronous composite: write, dispatch, read."""
        self.write_feeds(values)
        self.dispatch()
        return self.read(read)


def compile_graph(builder: GraphBuilder,
                  targets: list[tuple[str, Any, tuple[int, ...]]], *,
                  device: int = DEVICE_ANE,
                  placement_report: bool = False,
                  loopback: Mapping[str, str] | None = None) -> CompiledGraph:
    """Compile ``builder`` for ``device`` (`DEVICE_ANE` or `DEVICE_GPU`).

    ``targets`` is a list of (name, tensor, shape).  ``loopback`` maps
    result names onto placeholder names whose next-step value they become
    (see :class:`CompiledGraph`).  ANE placement is a compiler
    preference, not a guarantee; ``placement_report=True`` makes the
    compiler print the realized partition to stderr for inspection.
    """
    fw = _fw()
    metal = fw["Metal"].MTLCreateSystemDefaultDevice()
    graph_device = fw["MPSGraphDevice"].deviceWithMTLDevice_(metal)

    descriptor = fw["MPSGraphCompilationDescriptor"].alloc().init()
    if device == DEVICE_ANE:
        descriptor.setOptimizationLevel_(1)
        descriptor.setPreferredDevice_(DEVICE_ANE)
        if placement_report:
            descriptor.setPrintANEPlacementAnalysis_(True)
    else:
        descriptor.setOptimizationLevel_(0)
        descriptor.setPreferredDevice_(device)

    shaped = {
        tensor: fw["MPSGraphShapedType"].alloc().initWithShape_dataType_(
            list(shape), builder.dtype)
        for tensor, shape, _ in builder.feeds
    }
    exe = builder.graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor_(
        graph_device, shaped, [tensor for _, tensor, _ in targets], None,
        descriptor)

    by_id = {id(tensor): (name, shape)
             for tensor, shape, name in builder.feeds}
    order = [by_id[id(tensor)] for tensor in exe.feedTensors()]
    return CompiledGraph(
        exe, graph_device, metal, builder.dtype, order,
        [(name, tuple(shape)) for name, _, shape in targets],
        loopback=loopback)


__all__ = ["FLOAT16", "FLOAT32", "DEVICE_ANE", "DEVICE_GPU",
           "GraphBuilder", "CompiledGraph", "compile_graph"]
