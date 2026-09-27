"""BSVD video denoiser, ported to MLX.

Architecture ported from BSVD (C. Qi et al., "Real-time Streaming Video
Denoising with Bidirectional Buffers", ACM MM 2022). The forward pass is a clean
MLX reimplementation: NHWC tensors, plain convs, ReLU6, pixel shuffle, and the
reference bidirectional buffer streaming schedule.

The public RGB checkpoint is non-blind: each input frame is RGB plus a constant
noise map. The loader also supports blind RGB-only checkpoints by inferring the
first conv's input channels from the weights.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

import mlx.core as mx

from kinovsr.analysis.noise.track import NoiseMapTracker, PulseGain, source_since_sync
from kinovsr.modeling.compile_cache import cached
from kinovsr.modeling.window_buffer import WindowBuffer
from kinovsr.processors.conditioning import noise_map_debug_image, noise_map_diagnostics
from kinovsr.processors.feed_driver import (
    AsyncWindowHandle,
    CompletionSubmit,
    LaneSubmit,
    OwnerWait,
    WindowRouteUnavailable,
    WindowWavefront,
)
from kinovsr.processors.protocol import GopWindowPolicy

from .ane_backend import AneBSVD
from .mps import MpsGraphBSVD
from .net import (
    BSVD,
    _reflect_pad_to4,
    _strength_to_sigma,
    default_weights_path,
    load_bsvd,
)


class BsvdDenoiser:
    """Streaming BSVD denoiser for harness preprocess chains.

    feed(frame, token) and flush() return iterables of
    (denoised_frame, token). The network has a 16-step bidirectional-buffer
    delay; the first 16 intermediate outputs are discarded, then outputs are
    paired with the oldest input token. Flush stays lazy so downstream work
    can consume each tail frame before the next recurrent accelerator step.
    """

    MAP_WARMUP = 9  # frames buffered before estimating a spatial noise map

    def __init__(
        self,
        weights_path: str | Path | None = None,
        strength: float = 0.5,
        variant: str = "c64",
        dtype: mx.Dtype = mx.float16,
        noise_map: NoiseMapTracker | None = None,
        map_refresh: int = 64,
        pulse: PulseGain | None = None,
        map_floor: float = 0.0,
        backend: str = "mlx",
    ):
        if backend not in ("mlx", "ane", "mpsgraph"):
            raise ValueError(
                f"unknown BSVD backend {backend!r}; expected 'mlx', 'ane', or 'mpsgraph'"
            )
        wp = Path(weights_path) if weights_path else default_weights_path(variant)
        if not wp.is_file():
            raise FileNotFoundError(
                f"BSVD weights not found at {wp}. They are not bundled; convert the "
                "source .pth with kinovsr weights convert or pass --bsvd-weights."
            )
        if backend == "ane":
            # Explicitly requested; construction and first-step failures
            # raise rather than silently running a different backend.

            self.net: BSVD | AneBSVD | MpsGraphBSVD | None = AneBSVD(wp, dtype=dtype)
        elif backend == "mpsgraph":
            self.net = MpsGraphBSVD(wp, dtype=dtype)
        else:
            self.net = BSVD(wp, dtype=dtype)
        self.sigma = _strength_to_sigma(strength)
        # optional NoiseMapTracker: replaces the constant sigma plane with a
        # per-pixel estimate (sigma units, same scale as the constant).
        self._tracker = noise_map
        # optional PulseGain: per-frame scalar on the sigma plane tracking
        # GOP-phase noise pulsing (I-frame grain refresh).
        self._pulse = pulse
        if (self._tracker is not None or self._pulse is not None) and self.net.input_channels != 4:
            raise ValueError(
                "--noise-map / --noise-map-pulse need a non-blind (4-channel) "
                "BSVD checkpoint; this one is blind RGB-only."
            )
        # Streaming-mode map refresh cadence (frames); 0 disables. The
        # GOP-window path re-estimates per window instead and ignores this.
        self._map_refresh = max(0, int(map_refresh))
        # user sigma floor under the map: the temporal estimator only measures
        # FLICKERING noise, so static grain / structured junk reads near zero
        # even though conditioning the net higher visibly cleans it. The floor
        # guarantees a base denoise level (the manual dial's role) with the
        # map's spatial and pulse adaptation applying above it.
        self._map_floor = max(0.0, float(map_floor))
        self.last_noise_map: mx.array | None = None  # fp32 (H,W,1) actually used (debug)
        # diagnostics: gains across the whole run (survives the end-of-stream reset)
        self._pulse_log: list[float] = []
        self._gop: WindowBuffer[mx.array, object, tuple[mx.array, object]] | None = None

        # Cross-window pipelining for async-window-capable nets (the ANE
        # backend); a plain object with no thread until a window flies.
        self._wavefront: WindowWavefront[tuple[mx.array, object]] = WindowWavefront()
        self._reset_state()

    def _reset_conditioning(self, clear_debug: bool = False) -> None:
        if self._tracker is not None and hasattr(self._tracker, "reset"):
            self._tracker.reset()
        if clear_debug:
            self.last_noise_map = None

    def _reset_state(self) -> None:
        self._backend().reset()
        self._hw: tuple[int, int] | None = None
        self._nm: mx.array | None = None
        self._padded_hw: tuple[int, int] | None = None
        self._tokens: list[object] = []
        self._warm: list[
            tuple[mx.array, object, float]
        ] = []  # (frame3ch, token, gain) held until the map is estimated
        self._recent: list[mx.array] = []  # rolling last frames for the streaming map refresh
        self._since_refresh = 0
        if self._pulse is not None:
            self._pulse.reset()
        self._received = 0
        self._emitted = 0
        self._steps = 0
        if self._gop is not None:
            self._gop.reset()

    def reset(self) -> None:
        self._wavefront.abandon()
        self._reset_state()
        self._reset_conditioning(clear_debug=True)

    def preheat(self, height: int, width: int) -> None:
        """Start loading the backend engine for this geometry, if it can.

        Called at the pipeline's prepare edge (after GOP policy setup);
        the frame geometry maps to the net's padded working size exactly
        as ``_prepare`` will pad it. Backends without the hook (the MLX
        net) ignore this.
        """
        hook = getattr(self.net, "preheat", None)
        if callable(hook):
            hook(height + (-height) % 4, width + (-width) % 4, scheduled=self._gop is not None)

    def bind_background_submit(
        self,
        submit: LaneSubmit,
        completion_submit: CompletionSubmit | None = None,
        owner_wait: OwnerWait | None = None,
    ) -> None:
        """Bind window progress to the runtime's serial BSVD owner lane."""
        self._wavefront.bind_background_submit(
            submit,
            completion_submit,
            owner_wait,
        )

    def set_gop_policy(self, policy: GopWindowPolicy | None) -> None:
        """Apply the shared GOP-window policy on every BSVD backend."""
        if policy is None:
            self._gop = None
            return

        self._gop = WindowBuffer.gop(policy.min_window, policy.max_window, self._run_window)

    def run_diagnostics(self) -> list[str]:

        return noise_map_diagnostics(self)

    def debug_images(self) -> dict[str, mx.array]:

        return noise_map_debug_image(self)

    def _backend(self) -> BSVD | AneBSVD | MpsGraphBSVD:
        """The backend this denoiser runs; close() releases it."""
        net = self.net
        if net is None:
            raise RuntimeError("BSVD denoiser is closed")
        return net

    def close(self) -> None:
        self._wavefront.abandon()
        net, self.net = self.net, None
        try:
            close = getattr(net, "close", None)
            if callable(close):
                close()
        finally:
            # Release delayed frames/tokens and conditioning tensors even if
            # the backend reports an in-flight prediction failure while it
            # shuts down.  FeedFlushProcessor has already captured diagnostics
            # before calling this lifecycle edge.
            self._nm = None
            self._tokens = []
            self._warm = []
            self._recent = []
            self._gop = None

    def _prepare(self, frame: mx.array) -> mx.array:
        """Clip + pad one frame to a 3-channel net-dtype tensor (no noise map yet;
        the map channel is concatenated at step time so a spatial estimate made
        from the first frames can apply to those same frames)."""
        frame = mx.clip(frame[..., :3].astype(mx.float32), 0.0, 1.0)
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if self._hw is None:
            self._hw = (h, w)
        elif self._hw != (h, w):
            raise ValueError(f"BSVD stream changed resolution from {self._hw} to {(h, w)}")
        return _reflect_pad_to4(frame[None].astype(self._backend().dtype))[0]

    def _plane_from_map(self, sig_map: mx.array) -> mx.array:
        """(H,W,1) sigma map -> (1,hp,wp,1) net-dtype plane (reflect-padded)."""
        self.last_noise_map = sig_map.astype(mx.float32)
        return _reflect_pad_to4(sig_map[None].astype(self._backend().dtype))[0]

    def _ensure_nm(self, x: mx.array) -> None:
        """Make sure self._nm exists for this padded size (constant-sigma path)."""
        _, hp, wp, _ = x.shape
        if self._nm is None or self._padded_hw != (int(hp), int(wp)):
            self._nm = mx.full((1, hp, wp, 1), float(self.sigma), dtype=self._backend().dtype)
            self._padded_hw = (int(hp), int(wp))

    # the unblind checkpoints were trained with noise_ival [5, 55]: sigma
    # conditioning outside that dial is out of distribution (below the floor the
    # net no-ops, above the ceiling it over-smooths), so the plane is clamped
    # into it after the map gain and pulse gain -- the same range the manual
    # --denoise-strength dial spans.
    SIGMA_MIN = 5.0 / 255.0
    SIGMA_MAX = 55.0 / 255.0

    def _with_nm(self, x: mx.array, nm: mx.array | None = None, gain: float = 1.0) -> mx.array:
        if self._backend().input_channels != 4:
            return x
        if nm is None:
            if self._nm is None:
                self._ensure_nm(x)
            assert self._nm is not None
            nm = self._nm
        if gain != 1.0:
            nm = nm * gain
        if self._tracker is not None or self._pulse is not None:
            nm = mx.clip(nm, max(self.SIGMA_MIN, self._map_floor), self.SIGMA_MAX)
        return mx.concatenate([x, nm], axis=-1)

    def _pulse_gain(
        self, x3: mx.array, new_segment: bool = False, since_sync: int | None = None
    ) -> float:
        """Per-frame pulse gain from the cropped frame (1.0 when pulse is off)."""
        if self._pulse is None:
            return 1.0
        g = self._pulse.update(self._crop(x3), new_segment=new_segment, since_sync=since_sync)
        self._pulse_log.append(g)
        return g

    def _crop(self, x: mx.array) -> mx.array:
        assert self._hw is not None
        h, w = self._hw
        return x[:, :h, :w, :]

    def _estimate_from(self, frames3: list[mx.array]) -> None:
        """Estimate the map from padded 3ch frames; fall back to the constant
        sigma when the tracker cannot estimate (too few frames)."""
        assert self._tracker is not None
        sig = self._tracker.update([self._crop(f) for f in frames3])
        if sig is None:
            assert self._hw is not None
            h, w = self._hw
            sig = mx.full((h, w, 1), float(self.sigma), dtype=mx.float32)
        plane = self._plane_from_map(sig)
        self._nm = plane
        self._padded_hw = (int(plane.shape[1]), int(plane.shape[2]))

    def _drain_warm(self) -> list[tuple[mx.array, object]]:
        out: list[tuple[mx.array, object]] = []
        for x, tok, gain in self._warm:
            out += self._step(x, token=tok, real=True, gain=gain)
        self._warm = []
        return out

    def _emit(self, out: mx.array, token: object) -> tuple[mx.array, object]:
        if self._hw is None:
            raise RuntimeError("BSVD emitted before any input frame")
        h, w = self._hw
        out = mx.clip(out, 0.0, 1.0)[0, :h, :w, :3].astype(mx.float32)
        mx.eval(out)
        return out, token

    def _step(
        self, x: mx.array | None, token: object = None, real: bool = False, gain: float = 1.0
    ) -> list[tuple[mx.array, object]]:
        if real:
            self._tokens.append(token)
            self._received += 1
        out = self._backend().step(None if x is None else self._with_nm(x, gain=gain))
        self._steps += 1
        if (
            self._steps <= self._backend().SHIFT_NUM
            or out is None
            or self._emitted >= self._received
        ):
            return []
        tok = self._tokens.pop(0)
        self._emitted += 1
        return [self._emit(out, tok)]

    def feed(self, frame: mx.array, token: object = None) -> Iterable[tuple[mx.array, object]]:

        x = self._prepare(frame)
        if self._gop is not None:
            # Opportunistically advance the window in flight: completed
            # dispatches are consumed and the next submitted while the
            # source keeps decoding, so the accelerator never waits for
            # a full window boundary.
            self._wavefront.poll()
            return self._feed_gop(x, token)
        gain = self._pulse_gain(x, since_sync=source_since_sync(token))
        if self._tracker is not None and self._nm is None:
            # hold the first frames, estimate the spatial map from them, then
            # drain them through the net with that map attached.
            self._warm.append((x, token, gain))
            if len(self._warm) >= self.MAP_WARMUP:
                self._estimate_from([f for f, _, _ in self._warm])
                self._recent = [f for f, _, _ in self._warm]
                return self._drain_warm()
            return []
        if self._tracker is not None and self._map_refresh > 0:
            # periodic streaming refresh: re-estimate from a rolling buffer of
            # recent frames; the tracker's EMA keeps the transition gradual.
            self._recent.append(x)
            if len(self._recent) > self.MAP_WARMUP:
                self._recent.pop(0)
            self._since_refresh += 1
            if self._since_refresh >= self._map_refresh and len(self._recent) >= 2:
                self._estimate_from(self._recent)
                self._since_refresh = 0
        return self._step(x, token=token, real=True, gain=gain)

    def flush(self) -> Iterable[tuple[mx.array, object]]:
        if self._gop is not None:
            yield from self._gop.flush()
            yield from self._wavefront.drain()
            self._reset_state()
            self._reset_conditioning(clear_debug=False)
            return
        if self._warm:
            # short stream ended before the map warmup filled: estimate from
            # whatever arrived (the tracker falls back to constant below 2 frames)
            self._estimate_from([f for f, _, _ in self._warm])
            yield from self._drain_warm()
        guard = self._backend().SHIFT_NUM + self._received + 2
        while self._emitted < self._received:
            before = self._emitted
            yield from self._step(None)
            guard -= 1
            if guard <= 0 and self._emitted == before:
                raise RuntimeError("BSVD flush did not produce enough delayed frames")
        self._reset_state()
        self._reset_conditioning(clear_debug=False)

    def _feed_gop(self, frame: mx.array, token: object) -> Iterable[tuple[mx.array, object]]:
        # A feed-side poll may have materialized outputs since the previous
        # call. Emit them before a new boundary can submit another window.
        yield from self._wavefront.available()
        assert self._gop is not None
        yield from self._gop.feed(frame, token)

    def _run_window(
        self,
        frames: list[mx.array],
        tokens: list[object],
        emit_start: int,
        emit_end: int,
    ) -> Iterable[tuple[mx.array, object]]:
        nm = None
        if self._tracker is not None:
            # per-window estimate, EMA-blended across windows by the tracker so
            # the conditioning does not pump at gop-aligned window boundaries.
            # Conditioning needs no net state, so on the async path it runs
            # while the PREVIOUS window's dispatches are still in flight.
            sig = self._tracker.update([self._crop(f) for f in frames])
            if sig is not None:
                nm = self._plane_from_map(sig)

        # Window starts break temporal adjacency (proc ranges overlap), so the
        # pulse tracker restarts its diff chain at each window. Keep this lazy
        # for synchronous MLX; accelerator backends materialize their batch.
        conditioned: Iterable[mx.array] = (
            self._with_nm(
                x,
                nm,
                gain=self._pulse_gain(x, new_segment=(i == 0), since_sync=source_since_sync(token)),
            )
            for i, (x, token) in enumerate(zip(frames, tokens, strict=True))
        )

        begin_window = getattr(self.net, "begin_window", None)
        if callable(begin_window):
            conditioned = list(conditioned)
        capable = getattr(self.net, "window_capable", None)
        minimum_window = int(getattr(self.net, "MIN_WINDOW_FRAMES", 16))
        if (
            callable(begin_window)
            and len(frames) >= minimum_window
            and (not callable(capable) or capable(int(frames[0].shape[1]), int(frames[0].shape[2])))
        ):
            # Depth-one cross-window pipelining: submit completes the
            # window in flight (emitting it), then starts this one; its
            # dispatches then run while feed() buffers the NEXT window -
            # upstream decode, conditioning, and downstream encode all
            # hide under the accelerator. The net is reset by finalize,
            # never while its window is still in flight.
            count = len(frames)
            held_tokens = list(tokens)
            next_output = 0

            def collect(
                handle: AsyncWindowHandle, complete: bool
            ) -> Iterable[tuple[mx.array, object]]:
                nonlocal next_output
                outputs = cast(Sequence[mx.array], handle.outputs)  # window machines hold arrays
                if complete and len(outputs) != count:
                    raise RuntimeError(
                        f"BSVD window returned {len(outputs)} outputs for {count} frames"
                    )
                ready = min(len(outputs), count)
                selected_start = max(next_output, emit_start)
                selected_end = min(ready, emit_end)
                selected = tuple(
                    (outputs[index], held_tokens[index])
                    for index in range(selected_start, selected_end)
                )
                next_output = ready
                if complete:
                    # Window outputs are detached materialized arrays, so the
                    # recurrent state may reset before public crop/cast work.
                    self._backend().reset()
                return (self._emit(output, token) for output, token in selected)

            try:
                yield from self._wavefront.submit(lambda: begin_window(conditioned), collect)
            except WindowRouteUnavailable:
                pass  # the net refused before touching the stream: step this window
            else:
                yield from self._wavefront.available()
                return

        # The synchronous fallback (a window too short for the phase path,
        # or a net without begin_window) emits inline, so the window still
        # in flight must complete and emit FIRST - both for emission order
        # (its frames precede this window's) and because resetting the net
        # under an in-flight window corrupts shared runner state.
        yield from self._wavefront.drain()
        self._backend().reset()
        for i, x in enumerate(conditioned):
            y = self._backend().step(x)
            idx = i - self._backend().SHIFT_NUM
            if emit_start <= idx < emit_end:
                if y is None:
                    raise RuntimeError("BSVD GOP window emitted an empty frame")
                yield self._emit(y, tokens[idx])
        # The proc range may end with a shared anchor or forced-split trim
        # whose own outputs are deliberately not emitted. Stop the delay-line
        # drain as soon as the last requested output can emerge.
        tail_steps = max(0, emit_end + self._backend().SHIFT_NUM - len(frames))
        for i in range(tail_steps):
            y = self._backend().step(None)
            idx = len(frames) + i - self._backend().SHIFT_NUM
            if emit_start <= idx < emit_end:
                if y is None:
                    raise RuntimeError("BSVD GOP window emitted an empty frame")
                yield self._emit(y, tokens[idx])
        self._backend().reset()


__all__ = ["BSVD", "BsvdDenoiser", "default_weights_path", "load_bsvd"]
