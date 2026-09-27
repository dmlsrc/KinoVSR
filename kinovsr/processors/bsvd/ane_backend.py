"""The BSVD network on the Apple Neural Engine (:class:`AneBSVD`).

This is the denoiser's ``ane`` backend. It sits above the graph and runner
primitives in :mod:`.ane` and the two ways of running them,
:mod:`.ane_direct` and :mod:`.ane_phases`, so none of those modules imports
it.
"""

import contextlib
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict, cast

import mlx.core as mx

from kinovsr.native.dispatch import DispatchPipeline

from . import ane_direct
from .ane import (
    ANE_WIDTH_QUANTUM,
    MIN_SIDE,
    PHASE_MAX_HEIGHT,
    PHASE_MAX_WIDTH,
    BsvdRunner,
    _cache_directory,
    _vector_bytes,
    build_runner,
)
from .ane_phases import PHASE_GRAPH_VERSION, ScheduledPhaseSuite, WindowMachine, build_suite
from .net import _pad_width_reflect, load_bsvd
from .schedule import NoneFlowNet as _NoneFlowNet


class _TailStep(TypedDict):
    gate: list[float]
    write: list[float]
    pushes: list[bool]
    emit: bool


class _Pending(TypedDict):
    emit: bool
    pushes: list[bool] | None


class AneBSVD:
    """Drop-in for :class:`kinovsr.processors.bsvd.BSVD` on the ANE.

    Same contract shape: ``step(frame_or_none)`` once per stream item with
    NHWC (1, H, W, C) inputs padded to multiples of four, ``None`` through
    the fill, real outputs through the drain, ``reset()`` between streams.
    The engine is built lazily at the first real frame, when the geometry
    is known; construction failures raise (this backend is explicitly
    requested - there is no silent fallback).

    **Dispatches are pipelined one step deep.** ``step(k)`` collects the
    result of dispatch ``k-1`` and SUBMITS dispatch ``k`` to a worker
    thread, so the Core ML prediction runs while the caller does the rest
    of its per-frame work. That both hides the dispatch latency and keeps
    the ANE busy through the inter-frame gap - critical, because an ANE
    dispatch issued after >= 10 ms of host idleness pays a 15-23 ms
    power-state ramp that no host-side warm-up avoids (measured; the
    synchronous arrangement ran 57 ms/dispatch inside light pipelines
    against 34 ms hot). The worker runs ONLY ``BsvdRunner.dispatch`` -
    pure Core ML - and every MLX operation stays on the caller's thread.
    The pipelining adds one step to the output delay: ``SHIFT_NUM`` is 17
    (the network's 16 plus one dispatch in flight).

    Frames are reflect-padded on the right up to the ANE width quantum
    (multiples of 128 - see the module docstring) and outputs are cropped
    back, so callers see the original geometry throughout. The cache and
    the compiled model key on the PADDED size, so every source width that
    maps to the same quantum shares one model.
    """

    SHIFT_NUM = 17  # 16-step BiBuffer delay + 1 dispatch in flight
    TAIL_STEPS = 16

    def __init__(self, weights_path: str | Path, dtype: mx.Dtype = mx.float16):
        if dtype != mx.float16:
            raise ValueError(
                "the BSVD ANE backend executes fp16 only; use "
                "--bsvd-dtype float16 or --bsvd-backend mlx"
            )

        self.dtype = mx.float16
        self.params, self.input_channels = load_bsvd(weights_path, dtype=mx.float32)
        self._runner: BsvdRunner | ane_direct.DirectChainRunner | None = None
        self._directory: Path | None = None
        self._phase_suite: ScheduledPhaseSuite | None = None
        self._geometry: tuple[int, int] | None = None
        self._width = 0
        self._padded_width = 0
        self._zero_frame: bytes | None = None
        self._mirror: _NoneFlowNet | None = _NoneFlowNet()
        self._tail: list[_TailStep] | None = None
        self._tail_cursor = 0
        self._window_complete = False
        self._pipeline = DispatchPipeline("bsvd-ane-dispatch")
        self._preheat: Future[None] | None = None
        self._preheat_pool: ThreadPoolExecutor | None = None
        self._closed = False
        self._dirty = False
        self._state_needs_zero = False
        self._pending: _Pending | None = None

    def reset(self) -> None:
        self._require_open()
        self._join_pending(discard=True)
        if self._phase_suite is not None:
            # A window machine abandoned mid-flight leaves its dispatch on
            # the suite pipeline; settle it before touching shared state.
            self._phase_suite.pipeline.drain()
        if self._runner is not None and self._dirty:
            # Scheduled windows reuse the MLState (see BsvdRunner.reset);
            # a later PER-STEP stream on this net must first zero it,
            # because the ordinary gated graph reads state through fill.
            reuse = self._phase_suite is not None
            self._runner.reset(reuse_state=reuse)
            self._state_needs_zero = reuse
            if self._phase_suite is not None:
                # With a suite, the runner is always the suite's own BsvdRunner.
                runner = cast(BsvdRunner, self._runner)
                self._phase_suite.set_state(runner.model._state)
        self._dirty = False
        self._mirror = _NoneFlowNet()
        self._tail = None
        self._tail_cursor = 0
        self._window_complete = False

    def close(self) -> None:
        """Release the dispatch worker and the runner's large state buffers.

        The pipeline owns this object through an explicit lifecycle, so do
        not rely on a destructor: the dispatch worker retains ``self`` (and
        therefore the Core ML state plus skip rings) until it shuts down.
        Cleanup remains complete even when an in-flight prediction failed;
        that error is re-raised after the worker has stopped.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._join_pending(discard=True)
        finally:
            with contextlib.suppress(BaseException):
                self._join_preheat()
            self._pipeline.close()
            if self._phase_suite is not None:
                self._phase_suite.close()
            self._phase_suite = None
            closer = getattr(self._runner, "close", None)
            if callable(closer):
                closer()
            self._runner = None
            self.params = {}
            self._geometry = None
            self._directory = None
            self._zero_frame = None
            self._tail = None
            self._mirror = None
            self._dirty = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("BSVD ANE backend is closed")

    def _configure_geometry(self, height: int, width: int) -> None:
        self._require_open()
        if self._geometry is not None:
            if self._geometry != (height, width):
                raise RuntimeError(
                    f"BSVD ANE stream changed resolution from {self._geometry} to {(height, width)}"
                )
            return
        if min(height, width) < MIN_SIDE:
            raise RuntimeError(
                f"{width}x{height} is below the verified ANE floor "
                f"({MIN_SIDE} px per side); use --bsvd-backend mlx for "
                f"smaller frames."
            )
        padded = -(-width // ANE_WIDTH_QUANTUM) * ANE_WIDTH_QUANTUM
        self._geometry = (height, width)
        self._width = width
        self._padded_width = padded

    def _build_continuous(self, height: int) -> None:

        if ane_direct.should_use(height, self._padded_width):
            # Large geometries: the direct two-half chain replaces the
            # islanded single graph (same numbers, no island tax; see
            # ane_direct). Same runner contract, so everything downstream
            # of this build is engine-agnostic.
            self._runner, self._directory = ane_direct.build_direct_runner(
                self.params, self.input_channels, height, self._padded_width
            )
        else:
            self._runner, self._directory = build_runner(
                self.params, self.input_channels, height, self._padded_width
            )
        self._zero_frame = bytes(len(self._runner.model.input_view("frame")))

    def _build_scheduled(self, height: int) -> None:

        self._directory = _cache_directory(self.params, height, self._padded_width)
        self._phase_suite = build_suite(
            self.params, self.input_channels, height, self._padded_width, self._directory
        )
        self._runner = self._phase_suite.runner
        self._zero_frame = bytes(len(self._runner.model.input_view("frame")))

    def _join_preheat(self) -> None:
        future, self._preheat = self._preheat, None
        pool, self._preheat_pool = self._preheat_pool, None
        if future is not None:
            future.result()  # surfaces a preheat failure at first use
        if pool is not None:
            pool.shutdown(wait=False)

    def preheat(self, height: int, width: int, scheduled: bool = False) -> None:
        """Start loading the engine in the background - warm caches only.

        Called at the pipeline's prepare edge, before the first frame:
        the multi-second Core ML function loads then overlap the source's
        startup and the first window's decode instead of serializing at
        the first dispatch. Cold builds stay on the first-use thread,
        where their float64 fold audit, verification drives, and progress
        logging belong; a cold cache makes this a no-op.
        """
        if self._closed or self._preheat is not None:
            return
        if self._runner is not None or self._phase_suite is not None:
            return
        try:
            self._configure_geometry(height, width)
        except RuntimeError:
            return  # first real use raises the descriptive error
        directory = _cache_directory(self.params, height, self._padded_width)
        if scheduled and self.window_capable(height, width):
            stem = f"scheduled8-v{PHASE_GRAPH_VERSION}"
            warm = all(
                (directory / name).exists()
                for name in (
                    f"{stem}.mlpackage",
                    f"{stem}-verify.json",
                    f"{stem}-replay.safetensors",
                )
            )
            target = self._build_scheduled
        else:
            names = ["model.mlpackage", "verify.json", "replay.safetensors"]
            if ane_direct.should_use(height, self._padded_width):
                # Cold direct builds (device compiles plus the one-time
                # exactness gate) stay on the first-use thread with the
                # other cold paths; preheat only warm ones.
                names += ane_direct.warm_names()
            warm = all((directory / name).exists() for name in names)
            target = self._build_continuous
        if not warm:
            return

        self._preheat_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bsvd-ane-preheat"
        )
        self._preheat = self._preheat_pool.submit(target, height)

    def _ensure_runner(self, height: int, width: int) -> None:
        self._configure_geometry(height, width)
        self._join_preheat()
        if self._runner is None:
            self._build_continuous(height)

    def _ensure_scheduled_runner(self, height: int, width: int) -> None:
        self._configure_geometry(height, width)
        self._join_preheat()
        if self._phase_suite is None:
            self._build_scheduled(height)

    # ------------------------------------------------ dispatch pipelining

    def _submit(
        self,
        frame_bytes: bytes | memoryview,
        gate_bytes: bytes | None,
        write_bytes: bytes | None,
        emit: bool,
        pushes: list[bool] | None = None,
    ) -> None:
        assert self._runner is not None
        self._runner.load_inputs(frame_bytes, gate_bytes, write_bytes)
        self._pending = {"emit": emit, "pushes": pushes}
        self._dirty = True
        self._pipeline.submit(self._runner.dispatch)

    def _join_pending(self, discard: bool = False) -> mx.array | None:
        """Wait out the in-flight dispatch; return its emitted output.

        Push gating and output materialization happen here, on the
        caller's thread, strictly before the next submit reloads the
        input buffers and the next dispatch rewrites the out backing.
        A failed dispatch re-raises here even when discarding.
        """
        if self._pending is None:
            return None
        pending, self._pending = self._pending, None
        self._pipeline.join()
        if discard:
            return None
        if pending["pushes"] is not None:
            assert self._runner is not None
            for line, pushed in enumerate(pending["pushes"]):
                if not pushed:
                    self._runner.zero_last_push(line)
        return self._materialize_out() if pending["emit"] else None

    def _materialize_array(self, raw: mx.array) -> mx.array:
        # Fresh copy, NCHW backing -> NHWC cropped to the caller's width:
        # the backing is rewritten by the next prediction, so no lazy
        # graph over it may leave this method.
        out = mx.contiguous(mx.transpose(raw, (0, 2, 3, 1))[:, :, : self._width, :])
        mx.eval(out)
        return out

    def _materialize_out(self) -> mx.array:
        assert self._runner is not None
        return self._materialize_array(self._runner.model.output_array("out"))

    def window_capable(self, height: int, width: int) -> bool:
        """Whether the phase-specialized window path applies here.

        Beyond the measured ``PHASE_MAX_WIDTH`` by ``PHASE_MAX_HEIGHT``
        envelope, the schedule-capable wrapper routes windows through the
        per-step gated path instead (the ordinary function is reliable at
        every size).
        """
        padded = -(-width // ANE_WIDTH_QUANTUM) * ANE_WIDTH_QUANTUM
        return height <= PHASE_MAX_HEIGHT and padded <= PHASE_MAX_WIDTH

    def begin_window(self, frames: list[mx.array]) -> WindowMachine:
        """Start one independently reset window; return its async handle.

        The handle implements the shared async window protocol (see
        ``kinovsr.processors.feed_driver.WindowWavefront``): call
        ``advance(block=False)`` opportunistically to make progress in the
        shadow of other work, ``advance(block=True)`` to complete, then
        read ``outputs`` (input-frame order, ``len(frames)`` entries).
        Every MLX operation runs inside ``advance`` on the caller's
        thread; only the Core ML dispatches run on the suite's worker.
        The stream is dirty from this point - ``reset()`` before the next
        window.

        The unrolled outer fill/drain functions remove convolutions whose
        product value is ``None`` and batch those dispatches eight steps at
        a time; the inner boundary and steady middle use the verified
        one-step runner.
        """
        self._require_open()
        if len(frames) < 16:
            raise ValueError("BSVD ANE phase path needs at least 16 frames")
        if self._dirty or self._pending is not None:
            raise RuntimeError("reset BSVD ANE before running a schedule window")
        first = frames[0]
        height, width = int(first.shape[1]), int(first.shape[2])
        if not self.window_capable(height, width):
            raise RuntimeError(
                f"{width}x{height} is outside the phase-window envelope "
                f"(verified padded maximum {PHASE_MAX_WIDTH}x"
                f"{PHASE_MAX_HEIGHT}); route this window through step() "
                f"instead"
            )
        self._ensure_scheduled_runner(height, width)
        for frame in frames:
            if (int(frame.shape[1]), int(frame.shape[2])) != (height, width):
                raise RuntimeError("BSVD ANE schedule window changed resolution")

        def provider(frame: mx.array) -> Callable[[], memoryview]:
            # Lazy host-side prep: resolved in the shadow of an in-flight
            # dispatch, so the pad/transpose/eval work (MLX, caller's
            # thread) overlaps the ANE instead of running serially first.
            def resolve() -> memoryview:
                padded = _pad_width_reflect(frame.astype(mx.float16), self._padded_width)
                nchw = mx.contiguous(mx.transpose(padded, (0, 3, 1, 2)))
                mx.eval(nchw)
                return memoryview(nchw).cast("B")

            return resolve

        self._dirty = True
        self._window_complete = True
        assert self._phase_suite is not None
        assert self._zero_frame is not None
        return self._phase_suite.machine(
            [provider(frame) for frame in frames],
            memoryview(self._zero_frame),
            self._materialize_array,
        )

    def run_window(self, frames: list[mx.array]) -> list[mx.array]:
        """Run one reset window to completion (the synchronous form)."""
        handle = self.begin_window(frames)
        handle.advance(block=True)
        if len(handle.outputs) != len(frames):
            raise RuntimeError(
                f"BSVD ANE phase path returned {len(handle.outputs)} "
                f"outputs for {len(frames)} frames"
            )
        return handle.outputs

    def _assemble_tail(self) -> list[_TailStep]:
        """Mirror the product's 16 drain steps and derive the schedule.

        ``write`` fires on a unit's priming step over the WHOLE timeline -
        units left unprimed by a stream shorter than the fill prime DURING
        the drain, so their write gate falls in the tail. The final tail
        step also writes-gates any unit still unprimed, matching the
        derivation the product schedule was verified against.
        """
        assert self._mirror is not None
        records = [self._mirror.step(False) for _ in range(self.TAIL_STEPS)]
        tail: list[_TailStep] = []
        for k, record in enumerate(records):
            writes = [1.0] * 16
            for i in range(16):
                if record.unprimed[i] and (
                    k + 1 >= self.TAIL_STEPS or not records[k + 1].unprimed[i]
                ):
                    writes[i] = 0.0
            tail.append(
                {
                    "gate": [0.0 if record.drained[i] else 1.0 for i in range(16)],
                    "write": writes,
                    "pushes": list(record.pushes),
                    "emit": record.out_real,
                }
            )
        return tail

    def step(self, x: mx.array | None) -> mx.array | None:
        self._require_open()
        if self._window_complete:
            raise RuntimeError("BSVD ANE schedule window is complete; reset() first")
        if x is None:
            if self._runner is None:
                return None  # drained before any input frame
            if self._tail is None:
                self._tail = self._assemble_tail()
                self._tail_cursor = 0
            out = self._join_pending()
            if self._tail_cursor < len(self._tail):
                entry = self._tail[self._tail_cursor]
                self._tail_cursor += 1
                assert self._zero_frame is not None
                self._submit(
                    self._zero_frame,
                    _vector_bytes(entry["gate"]),
                    _vector_bytes(entry["write"]),
                    entry["emit"],
                    entry["pushes"],
                )
            return out

        if self._tail is not None:
            raise RuntimeError("BSVD ANE received a real frame after draining began; reset() first")
        height, width = int(x.shape[1]), int(x.shape[2])
        self._ensure_runner(height, width)
        if self._state_needs_zero:
            # A scheduled window left its state behind (reused by design);
            # the ordinary gated graph reads state through fill, so a
            # per-step stream starts from zeros.
            assert self._runner is not None
            self._runner.model.reset_state()
            if self._phase_suite is not None:
                # With a suite, the runner is always the suite's own BsvdRunner.
                runner = cast(BsvdRunner, self._runner)
                self._phase_suite.set_state(runner.model._state)
            self._state_needs_zero = False
        out = self._join_pending()
        assert self._mirror is not None
        record = self._mirror.step(True)
        write = [0.0 if record.primes[i] else 1.0 for i in range(16)]
        padded = _pad_width_reflect(x.astype(mx.float16), self._padded_width)
        nchw = mx.contiguous(mx.transpose(padded, (0, 3, 1, 2)))
        mx.eval(nchw)
        self._submit(memoryview(nchw).cast("B"), None, _vector_bytes(write), record.out_real)
        return out


__all__ = ["AneBSVD"]
