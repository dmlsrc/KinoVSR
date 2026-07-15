"""CVPixelBuffer + CMTime helpers for the VideoToolbox bridge.

VSR's source/dst formats (NV12, RGBAHalf), the BGRA buffer used for the
side-by-side comparison, and the CoreImage-based upload path for converting
numpy frames into IOSurface-backed CVPixelBuffers all live here. Plus the
fixed-timescale `_frame_pts` so VSR and AVWriter agree on PTSes for any
arbitrary fps.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import mlx.core as mx

from kinovsr.native.frameworks import (
    CoreMedia,
    Foundation,
    Quartz,
    autorelease_pool,
)

from .timing import grid_ticks

_log = logging.getLogger(__name__)

# FourCC pixel-format constants ----------------------------------------------
#
# CV uses big-endian four-character codes packed into a uint32. PIX_BGRA is
# the common 8-bit RGBA destination used for the comparison composite;
# PIX_RGBAHALF is the half-float RGBA source VSR HighQuality expects;
# PIX_NV12 is what LL VSR (and HEVC encoders) consume.
PIX_BGRA = int.from_bytes(b"BGRA", "big")        # 0x42475241
PIX_RGBAHALF = 1380411457                         # 'RGhA' kCVPixelFormatType_64RGBAHalf
PIX_NV12 = 875704438                              # '420v' kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange


# CMTime base for video PTS --------------------------------------------------
#
# 24000 lands bit-exact for 24/25/30/48/50/60 and 24000/1001. Other NTSC
# rates alternate adjacent integer durations on an exact rational index grid;
# no per-frame rounding error accumulates. Picked over 600, whose coarse ticks
# cannot represent those presentation times closely enough.
VIDEO_TIME_SCALE = 24000


_ci_context: Any = None
_srgb: Any = None
_ci_singleton_lock = threading.Lock()


def _append_exception_context(
    winner: BaseException,
    losers: tuple[BaseException | None, ...] | list[BaseException | None],
) -> None:
    """Build one identity-deduplicated, acyclic exception context chain."""
    ordered: list[BaseException] = []
    seen: set[int] = set()

    def collect(exc: BaseException | None) -> None:
        node = exc
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            ordered.append(node)
            node = node.__context__

    collect(winner)
    for loser in losers:
        collect(loser)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        earlier.__context__ = later
    ordered[-1].__context__ = None


def _cleanup_winner(
    active: BaseException | None,
    cleanup_errors: list[BaseException],
) -> BaseException | None:
    """Apply pipeline cleanup precedence without importing pipeline modules."""
    if not cleanup_errors:
        return active
    interrupts = [exc for exc in cleanup_errors if not isinstance(exc, Exception)]
    ordinary = [exc for exc in cleanup_errors if isinstance(exc, Exception)]
    if interrupts:
        winner = interrupts[0]
        _append_exception_context(
            winner,
            [active, *ordinary, *interrupts[1:]],
        )
        return winner
    if active is not None:
        _append_exception_context(active, ordinary)
        return active
    winner = ordinary[0]
    _append_exception_context(winner, ordinary[1:])
    return winner


class _CiCacheJanitor:
    """Serialize cache maintenance around process-global Core Image renders.

    Plain counters under one lock: renders in flight, completed renders not
    yet covered by a clear (dirty), and live owner leases.  A clear executes
    outside the lock with ``_clearing`` set; render entry waits while it is.
    The product renders from a single thread (verified at runtime on CI
    chains), so the waits exist for host embedders, not for scheduling.

    Failed clears leave the dirty count in place: the work stays eligible
    for the next threshold, the final owner release, or an explicit
    :func:`clear_ci_caches`.
    """

    def __init__(self, interval: int = 64) -> None:
        if interval < 1:
            raise ValueError("Core Image cleanup interval must be positive")
        self._interval = int(interval)
        self._condition = threading.Condition()
        self._active = 0
        self._dirty = 0
        self._cleared = 0
        self._backoff = 0
        self._owners = 0
        self._clearing = False

    def begin_render(self) -> None:
        with self._condition:
            while self._clearing:
                self._condition.wait()
            self._active += 1

    def finish_render(self, clear: Callable[[], None]) -> None:
        with self._condition:
            self._active -= 1
            self._dirty += 1
            # A failed attempt sets _backoff so the next periodic attempt
            # waits one further interval instead of retrying every render.
            due = (self._active == 0
                   and not self._clearing
                   and self._dirty - self._backoff >= self._interval)
            if due:
                self._clearing = True
            if self._active == 0:
                self._condition.notify_all()
        if due:
            self._run_clear(clear)

    def acquire_owner(self, clear: Callable[[], None]) -> _CiCacheOwner:
        with self._condition:
            self._owners += 1
        return _CiCacheOwner(self, clear)

    def release_owner(self, clear: Callable[[], None]) -> None:
        with self._condition:
            self._owners -= 1
            if self._owners:
                return
            while self._clearing or self._active:
                self._condition.wait()
            if not self._dirty:
                return
            self._clearing = True
        self._run_clear(clear)

    def clear_if_dirty(self, clear: Callable[[], None]) -> None:
        """Clear every render completed before this call returns."""
        with self._condition:
            while self._clearing or self._active:
                self._condition.wait()
            if not self._dirty:
                return
            self._clearing = True
        self._run_clear(clear)

    def _run_clear(self, clear: Callable[[], None]) -> None:
        # Never hold the condition while Core Image performs maintenance.
        try:
            clear()
        except BaseException:
            with self._condition:
                self._backoff = self._dirty
                self._clearing = False
                self._condition.notify_all()
            raise
        with self._condition:
            self._cleared += self._dirty
            self._dirty = 0
            self._backoff = 0
            self._clearing = False
            self._condition.notify_all()

    def _snapshot(self) -> tuple[int, int, int, bool, int]:
        """Deterministic counter view for lifecycle regression tests."""
        with self._condition:
            return (
                self._active,
                self._cleared + self._dirty,
                self._cleared,
                self._clearing,
                self._owners,
            )


class _CiCacheOwner:
    """Idempotent lease that triggers cleanup when the final owner exits."""

    def __init__(
        self,
        janitor: _CiCacheJanitor,
        clear: Callable[[], None],
    ) -> None:
        self._janitor = janitor
        self._clear = clear
        self._lock = threading.Lock()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._janitor.release_owner(self._clear)
            except Exception as exc:  # cache maintenance is best-effort
                _log.warning(
                    "Core Image final cache cleanup failed; dirty work "
                    "remains eligible for retry: %s",
                    exc,
                )

    def __enter__(self) -> _CiCacheOwner:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        cleanup_errors: list[BaseException] = []
        try:
            self.close()
        except BaseException as cleanup:  # noqa: BLE001 - precedence below
            cleanup_errors.append(cleanup)
        winner = _cleanup_winner(exc, cleanup_errors)
        if winner is None or winner is exc:
            return False
        raise winner


class _CiRenderScope:
    """One gated Core Image conversion plus its PyObjC autorelease pool."""

    def __init__(
        self,
        janitor: _CiCacheJanitor,
        clear: Callable[[], None],
    ) -> None:
        self._janitor = janitor
        self._clear = clear
        self._pool: Any = None
        self._begun = False

    def __enter__(self) -> Any:
        active: BaseException | None = None
        try:
            self._janitor.begin_render()
            self._begun = True
            self._pool = autorelease_pool()
            self._pool.__enter__()
            return ci_context()
        except BaseException as exc:  # noqa: BLE001 - cleanup precedence below
            active = exc
        cleanup_errors = self._finish(type(active), active, active.__traceback__)
        winner = _cleanup_winner(active, cleanup_errors)
        raise winner

    def __exit__(self, exc_type, exc, tb) -> bool:
        cleanup_errors = self._finish(exc_type, exc, tb)
        winner = _cleanup_winner(exc, cleanup_errors)
        if winner is None or winner is exc:
            return False
        raise winner

    def _finish(self, exc_type, exc, tb) -> list[BaseException]:
        errors: list[BaseException] = []
        pool, self._pool = self._pool, None
        if pool is not None:
            try:
                pool.__exit__(exc_type, exc, tb)
            except BaseException as cleanup:  # noqa: BLE001 - collected below
                errors.append(cleanup)
        begun, self._begun = self._begun, False
        if begun:
            try:
                self._janitor.finish_render(self._clear)
            except Exception as cleanup:
                # Ordinary periodic-maintenance failures never fail a clean
                # render; the dirty count stays eligible for a later clear.
                _log.warning(
                    "Core Image periodic cache cleanup failed; dirty work "
                    "remains eligible for retry: %s",
                    cleanup,
                )
                if exc is not None or errors:
                    errors.append(cleanup)
            except BaseException as cleanup:
                errors.append(cleanup)
        return errors


_ci_janitor = _CiCacheJanitor()


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

def ci_context() -> Any:
    """Shared CIContext; render call sites must use :func:`ci_render_scope`."""
    global _ci_context
    with _ci_singleton_lock:
        if _ci_context is None:
            _ci_context = Quartz.CIContext.contextWithOptions_(None)
        return _ci_context


def _clear_ci_context() -> None:
    """Clear an existing context without creating one just for maintenance."""
    with _ci_singleton_lock:
        context = _ci_context
    if context is not None:
        context.clearCaches()


def clear_ci_caches() -> None:
    """Clear work recorded by :func:`ci_render_scope`, if any.

    CIContext caches intermediate compute resources (rendered tiles, GPU
    pipeline states, etc.) across render calls. The janitor normally clears
    every 64 conversions and when the final owner lease closes; this explicit
    hook is retained for diagnostics and retry after a failed clear.
    """
    _ci_janitor.clear_if_dirty(_clear_ci_context)


def ci_cache_owner() -> _CiCacheOwner:
    """Own the shared cache for a host/file run and clear on final release."""
    return _ci_janitor.acquire_owner(_clear_ci_context)


def ci_render_scope() -> _CiRenderScope:
    """Gate one Core Image conversion and drain its autoreleased objects."""
    return _CiRenderScope(_ci_janitor, _clear_ci_context)


def srgb_colorspace() -> Any:
    """Shared sRGB CGColorSpace handle (cheap to create but reused for clarity)."""
    global _srgb
    with _ci_singleton_lock:
        if _srgb is None:
            _srgb = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB)
        return _srgb


# ---------------------------------------------------------------------------
# CMTime helpers
# ---------------------------------------------------------------------------

def frame_ticks(
    frame_index: int,
    cadence: Fraction | int | float | str,
) -> int:
    """Exact rational frame position in the product's integer time base."""
    return grid_ticks(frame_index, cadence, Fraction(1, VIDEO_TIME_SCALE))


def frame_pts(
    frame_index: int,
    fps: Fraction | int | float | str,
) -> Any:
    """Build a CMTime for a video frame index at the given fps.

    The complete rational index is rounded once. Unlike multiplying a rounded
    one-frame duration, the error therefore stays within half one output tick
    for arbitrarily long NTSC-family sequences.
    """
    return CoreMedia.CMTimeMake(frame_ticks(frame_index, fps), VIDEO_TIME_SCALE)


def resolve_pixel_format(attrs: dict) -> int:
    """Extract the PixelFormatType from a VT config's attributes dict.

    Quirk: VTSuperResolutionScalerConfiguration returns its supported source
    formats as a single-element NSArray, not a bare int. Unwrap if needed.
    """
    fmt = attrs.get("PixelFormatType")
    if not isinstance(fmt, int) and hasattr(fmt, "__getitem__"):
        fmt = int(fmt[0])
    return int(fmt)


# ---------------------------------------------------------------------------
# CVPixelBuffer creation
# ---------------------------------------------------------------------------

def make_pixel_buffer_from_attrs(width: int, height: int, attrs: dict) -> Any:
    """Allocate a fresh CVPixelBuffer from a VT config's attributes dict.

    Used as a fallback when a CVPixelBufferPool isn't available (e.g., before
    AVAssetWriter has been started); pools are preferred for hot paths.
    """
    fmt = resolve_pixel_format(attrs)
    err, pb = Quartz.CVPixelBufferCreate(None, width, height, fmt, attrs, None)
    if err != 0:
        raise RuntimeError(
            f"CVPixelBufferCreate({width}x{height}, fmt={fmt:#x}) failed: status={err}"
        )
    return pb


def make_pool_from_attrs(attrs: dict) -> Any | None:
    """Try to create a CVPixelBufferPool for the given attrs; None on failure.

    Caller should fall back to make_pixel_buffer_from_attrs if this returns
    None - some attribute combos don't pool cleanly.
    """
    err, pool = Quartz.CVPixelBufferPoolCreate(None, None, attrs, None)
    if err != 0 or pool is None:
        return None
    return pool


def make_bounded_pool_from_attrs(attrs: dict, buffer_count: int) -> Any | None:
    """Create a pool whose reusable working set matches a hard acquire cap.

    The minimum count keeps returned surfaces cached instead of purging them
    between VideoToolbox calls; acquisitions pair this with
    :func:`pool_create_buffer_bounded` at the same count so the cache neither
    churns nor grows past the declared window.
    """
    count = int(buffer_count)
    if count < 1:
        raise ValueError("buffer_count must be positive")
    pool_attrs = {Quartz.kCVPixelBufferPoolMinimumBufferCountKey: count}
    err, pool = Quartz.CVPixelBufferPoolCreate(
        None, pool_attrs, attrs, None
    )
    if err != 0 or pool is None:
        return None
    return pool


def pool_create_buffer(pool: Any) -> Any | None:
    """Pull a fresh buffer from a CVPixelBufferPool. None on failure."""
    err, pb = Quartz.CVPixelBufferPoolCreatePixelBuffer(None, pool, None)
    if err != 0 or pb is None:
        return None
    return pb


class PixelBufferPoolExhausted(RuntimeError):
    """A bounded CVPixelBufferPool has reached its allocation ceiling."""


def pool_create_buffer_bounded(pool: Any, allocation_threshold: int) -> Any:
    """Acquire from ``pool`` without allowing its allocation count to grow.

    CoreVideo's ordinary pool acquisition is only a reuse hint: it may keep
    allocating indefinitely while prior buffers remain live. The auxiliary
    allocation-threshold attribute turns that hint into a hard bound and
    reports a distinct exhaustion status when downstream ownership exceeds
    the promised in-flight window.
    """
    threshold = int(allocation_threshold)
    if threshold < 1:
        raise ValueError("allocation_threshold must be positive")
    aux = {Quartz.kCVPixelBufferPoolAllocationThresholdKey: threshold}
    err, pb = Quartz.CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
        None, pool, aux, None
    )
    if err == Quartz.kCVReturnWouldExceedAllocationThreshold:
        raise PixelBufferPoolExhausted(
            "CVPixelBufferPool exhausted at allocation threshold "
            f"{threshold} (status={err})"
        )
    if err != 0 or pb is None:
        raise RuntimeError(
            "CVPixelBufferPoolCreatePixelBufferWithAuxAttributes failed: "
            f"status={err}"
        )
    return pb


def flush_pool(pool: Any) -> None:
    """Release any excess cached buffers in a CVPixelBufferPool.

    Pools cache returned buffers for reuse (default age threshold ~1s) and
    don't expose `kCVPixelBufferPoolAllocationThresholdKey` by default -
    they grow to whatever peak buffer count the workload demands and stay
    there. For long runs that's a memory leak from the user's perspective.
    Calling `CVPixelBufferPoolFlush` with `kCVPixelBufferPoolFlushExcessBuffers`
    aggressively releases the cached-but-currently-unused buffers back to
    the system.
    """
    if pool is None:
        return
    # kCVPixelBufferPoolFlushExcessBuffers = 1
    Quartz.CVPixelBufferPoolFlush(pool, 1)


def _copy_plane_bytes(src_base: Any, src_bpr: int,
                      dst_base: Any, dst_bpr: int, height: int) -> None:
    """Copy ``height`` rows between two locked planes. Copies min(bpr) bytes
    per row so differing row-padding on the two buffers is safe - the valid
    pixel bytes never exceed either stride, so they are always fully copied."""
    src = src_base.as_buffer(height * src_bpr)
    dst = dst_base.as_buffer(height * dst_bpr)
    if src_bpr == dst_bpr:
        dst[:height * src_bpr] = src[:height * src_bpr]
        return
    n = min(src_bpr, dst_bpr)
    for r in range(height):
        dst[r * dst_bpr:r * dst_bpr + n] = src[r * src_bpr:r * src_bpr + n]


def _apply_frame_spec_attachments(pb: Any, frame_spec: Any) -> None:
    """Make modeled output identity override propagated native metadata."""
    from .color import resolve_frame_spec

    primaries, transfer, matrix, _full_range = resolve_frame_spec(frame_spec)
    for key, value in (
        (Quartz.kCVImageBufferYCbCrMatrixKey, matrix),
        (Quartz.kCVImageBufferColorPrimariesKey, primaries),
        (Quartz.kCVImageBufferTransferFunctionKey, transfer),
    ):
        Quartz.CVBufferSetAttachment(
            pb, key, value, Quartz.kCVAttachmentMode_ShouldPropagate
        )
    pixel_aspect = frame_spec.geometry.pixel_aspect
    Quartz.CVBufferSetAttachment(
        pb,
        Quartz.kCVImageBufferPixelAspectRatioKey,
        {
            Quartz.kCVImageBufferPixelAspectRatioHorizontalSpacingKey:
                int(pixel_aspect.numerator),
            Quartz.kCVImageBufferPixelAspectRatioVerticalSpacingKey:
                int(pixel_aspect.denominator),
        },
        Quartz.kCVAttachmentMode_ShouldPropagate,
    )


def copy_pixel_buffer(pb: Any, *, frame_spec: Any = None) -> Any:
    """Deep-copy a CVPixelBuffer into a fresh IOSurface-backed buffer.

    Format-agnostic: copies every plane row by row, so it handles packed
    formats (BGRA, RGBAHalf) and planar ones (NV12) alike. Used to hand a
    session consumer an OWNED output that stays valid after the source buffer
    is recycled or overwritten by the input owner.
    """
    fmt = Quartz.CVPixelBufferGetPixelFormatType(pb)
    w = Quartz.CVPixelBufferGetWidth(pb)
    h = Quartz.CVPixelBufferGetHeight(pb)
    dst = make_pixel_buffer_from_attrs(w, h, {
        "PixelFormatType": fmt, "Width": w, "Height": h,
        "IOSurfaceProperties": {}, "MetalCompatibility": True,
    })
    Quartz.CVPixelBufferLockBaseAddress(pb, 1)   # read-only source
    Quartz.CVPixelBufferLockBaseAddress(dst, 0)  # writable destination
    try:
        if Quartz.CVPixelBufferIsPlanar(pb):
            for i in range(Quartz.CVPixelBufferGetPlaneCount(pb)):
                _copy_plane_bytes(
                    Quartz.CVPixelBufferGetBaseAddressOfPlane(pb, i),
                    Quartz.CVPixelBufferGetBytesPerRowOfPlane(pb, i),
                    Quartz.CVPixelBufferGetBaseAddressOfPlane(dst, i),
                    Quartz.CVPixelBufferGetBytesPerRowOfPlane(dst, i),
                    Quartz.CVPixelBufferGetHeightOfPlane(pb, i))
        else:
            _copy_plane_bytes(
                Quartz.CVPixelBufferGetBaseAddress(pb),
                Quartz.CVPixelBufferGetBytesPerRow(pb),
                Quartz.CVPixelBufferGetBaseAddress(dst),
                Quartz.CVPixelBufferGetBytesPerRow(dst), h)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(dst, 0)
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
    # Preserve native metadata that the producing buffer explicitly marked for
    # downstream ownership. Processor-produced values already present on ``pb``
    # therefore win; non-propagating/private attachments intentionally stay with
    # the borrowed source buffer.
    Quartz.CVBufferPropagateAttachments(pb, dst)
    if frame_spec is not None:
        _apply_frame_spec_attachments(dst, frame_spec)
    return dst


# ---------------------------------------------------------------------------
# frame -> CVPixelBuffer
# ---------------------------------------------------------------------------

def _frame_is_fp16(frame: Any) -> bool:
    """True for a float16 frame, whether it is a numpy or an mlx array."""
    return str(frame.dtype).split(".")[-1] == "float16"


def _frame_buffer(frame: Any) -> memoryview:
    """A contiguous uint8-format memoryview over a frame's bytes, no copy.

    mlx arrays go through the buffer protocol (mx.contiguous + memoryview); a
    contiguous numpy array returns a zero-copy view of its buffer. This lets the
    caller memcpy straight from the array's (unified) memory into an IOSurface
    plane instead of materializing an intermediate ``bytes`` object first.
    """
    if isinstance(frame, mx.array):
        return memoryview(mx.contiguous(frame)).cast("B")
    mv = memoryview(frame)
    if mv.c_contiguous:
        return mv.cast("B")
    return memoryview(frame.tobytes())  # non-contiguous numpy fallback


def write_fp16_rgba(rgba_fp16: Any, pb: Any) -> None:
    """Memcpy a (H,W,4) fp16 RGBA frame (mlx or numpy) into a RGBAHalf CVPixelBuffer.

    Used for the HQ VSR source upload (RGBAHalf format) and any other case where
    we already have the exact destination layout. The base address is an
    objc.varlist whose `.as_buffer(n)` is a writable memoryview into the IOSurface
    plane; the source bytes come straight from the frame's buffer.
    """
    h, w = int(rgba_fp16.shape[0]), int(rgba_fp16.shape[1])
    # Zero-copy view of the frame's buffer; the single mv[:] = src memcpy below
    # goes straight from MLX's unified memory into the IOSurface plane, with no
    # intermediate bytes object.
    src = _frame_buffer(rgba_fp16)
    row = w * 8
    Quartz.CVPixelBufferLockBaseAddress(pb, 0)
    try:
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        mv = base.as_buffer(h * bpr)
        if bpr == row:
            mv[:] = src
        else:
            # Row-pad case: copy each row's bytes, skipping the destination pad.
            for r in range(h):
                mv[r * bpr : r * bpr + row] = src[r * row : (r + 1) * row]
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 0)


def upload_frame_to_buffer(frame: Any, pb: Any) -> None:
    """Upload `frame` into `pb`, dispatching on the buffer's pixel format.

    Accepted inputs (mlx or numpy array):
      - (H,W,3) uint8 RGB           : video / ffmpeg rgb24 path
      - (H,W,4) fp16 RGBA           : native decoder or MLX frame source path

    Accepted destinations:
      - NV12 ('420v')               : LowLatency VSR source
      - RGBAHalf ('RGhA')           : HighQuality VSR source

    The NV12 destination always goes through CoreImage so the sRGB->BT.709
    YUV conversion is correct. CIImage's source format is RGBA8 for uint8
    input and RGBAh for fp16 input - using RGBAh defers quantization to
    CIContext's render pass so the single 8-bit cast happens in YUV space
    rather than once in RGB and once in YUV.

    The RGBAHalf destination is a direct memcpy when the source is already
    fp16 RGBA. For uint8 input we promote to fp16 inline. All channel math runs
    in MLX and the bytes come from the array buffer, so no numpy - and the
    CoreImage / memcpy calls are unchanged, so chroma is byte-for-byte identical.
    """
    pix_fmt = Quartz.CVPixelBufferGetPixelFormatType(pb)
    h, w = int(frame.shape[0]), int(frame.shape[1])

    if pix_fmt == PIX_RGBAHALF:
        if _frame_is_fp16(frame):
            write_fp16_rgba(frame, pb)
            return
        # uint8 RGB -> fp16 RGBA promotion (legacy / --video path).
        f = frame if isinstance(frame, mx.array) else mx.array(frame)
        rgb = f.astype(mx.float16) * mx.array(1.0 / 255.0, dtype=mx.float16)
        alpha = mx.ones((h, w, 1), dtype=mx.float16)
        write_fp16_rgba(mx.concatenate([rgb, alpha], axis=-1), pb)
        return

    # NV12 (and any other format CoreImage can render into). Pick the CIImage
    # source format from the input dtype: RGBAh for fp16, RGBA8 for uint8.
    if _frame_is_fp16(frame):
        src = _frame_buffer(frame)
        with ci_render_scope() as context:
            data = Foundation.NSData.dataWithBytes_length_(src, len(src))
            ci_image = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
                data, w * 8, (w, h), Quartz.kCIFormatRGBAh, srgb_colorspace(),
            )
            context.render_toCVPixelBuffer_(ci_image, pb)
        return

    # uint8 RGB -> opaque RGBA8 for CoreImage.
    f = frame if isinstance(frame, mx.array) else mx.array(frame)
    alpha = mx.full((h, w, 1), 255, dtype=mx.uint8)
    src = _frame_buffer(mx.concatenate([f, alpha], axis=-1))
    with ci_render_scope() as context:
        data = Foundation.NSData.dataWithBytes_length_(src, len(src))
        ci_image = Quartz.CIImage.alloc().initWithBitmapData_bytesPerRow_size_format_colorSpace_(
            data, w * 4, (w, h), Quartz.kCIFormatRGBA8, srgb_colorspace(),
        )
        context.render_toCVPixelBuffer_(ci_image, pb)


# ---------------------------------------------------------------------------
# CVPixelBuffer -> mlx
# ---------------------------------------------------------------------------

def read_pixel_buffer_rgb(pb: Any) -> Any:
    """Read any CVPixelBuffer into a (H, W, 3) uint8 RGB mlx array via CoreImage.

    Goes through CIImage(CVPixelBuffer) + CIContext.render_toBitmap, so any
    source format (NV12, RGBAHalf, BGRA, ...) is handled uniformly. Slower
    than a direct memcpy for the trivial cases but correct everywhere.
    """
    w = Quartz.CVPixelBufferGetWidth(pb)
    h = Quartz.CVPixelBufferGetHeight(pb)
    buf = bytearray(w * h * 4)
    with ci_render_scope() as context:
        ci_image = Quartz.CIImage.alloc().initWithCVPixelBuffer_(pb)
        context.render_toBitmap_rowBytes_bounds_format_colorSpace_(
            ci_image, buf, w * 4, ((0, 0), (w, h)),
            Quartz.kCIFormatRGBA8, srgb_colorspace(),
        )
    rgba = mx.array(memoryview(buf)).reshape(h, w, 4)
    return mx.contiguous(rgba[..., :3])


def read_rgbahalf_rgb(pb: Any) -> Any:
    """Read a RGBAHalf ('RGhA') CVPixelBuffer into (H,W,3) float32 RGB, direct.

    Memcpy of the fp16 plane (no CoreImage, no 8-bit quantization, no colorspace
    re-render), so the decoder's full half-float precision survives. Values are
    whatever the buffer holds - gamma-encoded RGB in roughly [0, 1] for SDR.
    The round trip is byte-exact against write_fp16_rgba.
    """
    w = Quartz.CVPixelBufferGetWidth(pb)
    h = Quartz.CVPixelBufferGetHeight(pb)
    Quartz.CVPixelBufferLockBaseAddress(pb, 1)
    try:
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pb)
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        raw = mx.array(memoryview(base.as_buffer(h * bpr)))
        half = raw.view(mx.float16).reshape(h, bpr // 2)[:, : w * 4].reshape(h, w, 4)
        rgb = mx.contiguous(half[..., :3]).astype(mx.float32)
        mx.eval(rgb)
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, 1)
    return rgb


def read_buffer_rgb_f32(pb: Any) -> Any:
    """Read any CVPixelBuffer into (H,W,3) float32 RGB, NOMINALLY [0, 1].

    RGBAHalf is read direct (fp16-preserving; see read_rgbahalf_rgb) and can
    carry legal YUV->RGB overshoot OUTSIDE [0,1] (measured -0.14..+1.25 at
    saturated color edges) -- consumers that need the training domain must
    clip (learned-net entries do; see upscaler_base.to_rgb_batch). Other
    formats (NV12, BGRA, ...) go through CoreImage and are 8-bit, hence
    clipped by construction. Lets the denoise path keep 10-bit precision when
    the decode is RGBAHalf (balanced/image/none) and degrade gracefully to
    8-bit for NV12 (fast).
    """
    if Quartz.CVPixelBufferGetPixelFormatType(pb) == PIX_RGBAHALF:
        return read_rgbahalf_rgb(pb)
    return read_pixel_buffer_rgb(pb).astype(mx.float32) / 255.0
