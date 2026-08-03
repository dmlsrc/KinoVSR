"""Bounded actor runtime behind the synchronous pipeline iterator."""

from __future__ import annotations

import contextlib
import dataclasses
import heapq
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from kinovsr.processors import (
    Boundary,
    BoundaryKind,
    FrameUnit,
    Layout,
    PipelineContext,
    StreamSpec,
)

from .builder import ResolvedStage, _append_context, _wrap_stage_error
from .channels import EOS, BoundedChannel, ChannelClosed, EndOfStream
from .execution import (
    BufferingSpec,
    ExecutionSpec,
    PhysicalOperation,
    ResourceClaim,
    ResourceKind,
    estimate_frame_bytes,
    expand_operations,
    storage_descriptor_for_spec,
)
from .leases import Envelope, PayloadLease, StorageDescriptor

_LANE_STOP = object()
_ITERATION_DONE = object()


@dataclass(frozen=True, slots=True)
class _MlxSnapshot:
    data: bytes
    shape: tuple[int, ...]
    dtype: Any


class _AffinityLane:
    """One long-lived thread that serializes all calls for an affinity key."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: list[
            tuple[float, int, int, Future[Any] | None, Any]
        ] = []
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._sequence = 0
        self._stopped = False
        self._thread_id: int | None = None
        self._mlx_stream: Any = None

    def set_mlx_stream(self, stream: Any) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("cannot change an affinity lane stream after start")
            self._mlx_stream = stream

    def _start(self) -> None:
        with self._lock:
            if self._stopped:
                raise RuntimeError(f"affinity lane {self.name!r} is stopped")
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name=f"kinovsr-{self.name}",
                daemon=False,
            )
            self._thread.start()

    def _run(self) -> None:
        self._thread_id = threading.get_ident()
        while True:
            future, callback = self._take_ready()
            if callback is _LANE_STOP:
                return
            self._execute(future, callback)

    def _take_ready(self) -> tuple[Future[Any] | None, Any]:
        with self._condition:
            while True:
                while not self._items:
                    self._condition.wait()
                ready = self._items[0][0]
                remaining = ready - time.monotonic()
                if remaining > 0:
                    self._condition.wait(remaining)
                    continue
                ready_items = []
                now = time.monotonic()
                while self._items and self._items[0][0] <= now:
                    ready_items.append(heapq.heappop(self._items))
                chosen = min(
                    ready_items,
                    key=lambda item: (item[1], item[2]),
                )
                ready_items.remove(chosen)
                for item in ready_items:
                    heapq.heappush(self._items, item)
                _ready, _priority, _sequence, future, callback = chosen
                return future, callback

    def _execute(self, future: Future[Any] | None, callback: Any) -> None:
        from kinovsr.native.frameworks import autorelease_pool

        assert future is not None
        if not future.set_running_or_notify_cancel():
            return
        try:
            stream_scope = contextlib.nullcontext()
            if self._mlx_stream is not None:
                import mlx.core as mx

                stream_scope = mx.stream(self._mlx_stream)
            with autorelease_pool(), stream_scope:
                result = callback()
        except BaseException as exc:  # noqa: BLE001 - delivered through Future
            future.set_exception(exc)
        else:
            future.set_result(result)
            del result
        # A lane blocks on its next queue item. Do not let that wait pin the
        # previous callback closure, Future result, or a bounded native pool
        # slot captured by either one.
        del future, callback

    def run_until(self, predicate: Callable[[], bool]) -> None:
        """Cooperatively service this lane while its current call is suspended.

        Stateful processor code sometimes reaches a bounded window barrier
        from inside its owner callback. Waiting on a queued progress Future
        there would deadlock the same lane; finishing the handle inline would
        instead starve downstream MLX bridges. A nested owner turn preserves
        serial execution while allowing already-queued work to make progress.
        """
        if threading.get_ident() != self._thread_id:
            raise RuntimeError("run_until() must execute on its affinity lane")
        while not predicate():
            with self._condition:
                while not self._items and not predicate():
                    # A completion failure can change the predicate without
                    # submitting another lane callback, so retain a short
                    # bounded wakeup in addition to condition notifications.
                    self._condition.wait(0.01)
                if predicate():
                    return
                while True:
                    ready = self._items[0][0]
                    remaining = ready - time.monotonic()
                    if remaining <= 0:
                        ready_items = []
                        now = time.monotonic()
                        while self._items and self._items[0][0] <= now:
                            ready_items.append(heapq.heappop(self._items))
                        chosen = min(
                            ready_items,
                            key=lambda item: (item[1], item[2]),
                        )
                        ready_items.remove(chosen)
                        for item in ready_items:
                            heapq.heappush(self._items, item)
                        (
                            _ready,
                            _priority,
                            _sequence,
                            future,
                            callback,
                        ) = chosen
                        break
                    self._condition.wait(min(remaining, 0.01))
                    if predicate():
                        return
            if callback is _LANE_STOP:
                raise RuntimeError(
                    f"affinity lane {self.name!r} stopped during owner wait"
                )
            self._execute(future, callback)

    def call(
        self,
        callback: Callable[[], Any],
        *,
        priority: int = 0,
    ) -> Any:
        if threading.get_ident() == self._thread_id:
            return callback()
        return self.submit(callback, priority=priority).result()

    def submit(
        self,
        callback: Callable[[], Any],
        *,
        delay: float = 0.0,
        priority: int = 0,
    ) -> Future[Any]:
        """Queue work even when called by the lane itself.

        The always-queued form is what lets a physical owner schedule native
        progress behind its current callback, then return to the stage actor.
        A later logical-stage call is FIFO behind that progress task.
        """
        if delay < 0:
            raise ValueError("affinity lane delay must be nonnegative")
        self._start()
        future: Future[Any] = Future()
        with self._condition:
            if self._stopped:
                raise RuntimeError(f"affinity lane {self.name!r} is stopped")
            self._sequence += 1
            heapq.heappush(
                self._items,
                (
                    time.monotonic() + delay,
                    int(priority),
                    self._sequence,
                    future,
                    callback,
                ),
            )
            self._condition.notify()
        return future

    @property
    def thread_id(self) -> int | None:
        return self._thread_id

    def stop(self) -> None:
        with self._condition:
            if self._stopped:
                return
            self._stopped = True
            thread = self._thread
            if thread is None:
                return
            self._sequence += 1
            ready = max(
                (item[0] for item in self._items),
                default=time.monotonic(),
            )
            heapq.heappush(
                self._items,
                (ready, 0, self._sequence, None, _LANE_STOP),
            )
            self._condition.notify()
        if thread is not threading.current_thread():
            thread.join()


class _ResourceArbiter:
    """Deadlock-free weighted permits for coarse native resource claims."""

    _CAPACITIES = {
        ResourceKind.GPU: 4,
        ResourceKind.ANE: 2,
        ResourceKind.MEDIA: 2,
        ResourceKind.CPU: 4,
        ResourceKind.IO: 4,
        ResourceKind.MEMORY_BANDWIDTH: 4,
        ResourceKind.OPAQUE_NATIVE: 4,
    }

    def __init__(self) -> None:
        self._semaphores = {
            kind: threading.BoundedSemaphore(capacity)
            for kind, capacity in self._CAPACITIES.items()
        }

    @contextmanager
    def acquire(self, claims: tuple[ResourceClaim, ...]):
        acquired: list[threading.BoundedSemaphore] = []
        try:
            for claim in sorted(claims, key=lambda item: item.resource.value):
                capacity = self._CAPACITIES[claim.resource]
                count = capacity if claim.exclusive else claim.weight
                if count > capacity:
                    raise ValueError(
                        f"resource claim {claim.resource.value} weight {count} "
                        f"exceeds capacity {capacity}"
                    )
                semaphore = self._semaphores[claim.resource]
                for _ in range(count):
                    semaphore.acquire()
                    acquired.append(semaphore)
            yield
        finally:
            for semaphore in reversed(acquired):
                semaphore.release()


class _Supervisor:
    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self._failure: BaseException | None = None
        self._channels: list[BoundedChannel] = []
        self._lock = threading.Lock()

    def register(self, channel: BoundedChannel) -> None:
        with self._lock:
            self._channels.append(channel)

    @property
    def failure(self) -> BaseException | None:
        with self._lock:
            return self._failure

    def fail(self, failure: BaseException) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = failure
            channels = tuple(self._channels)
            self.cancelled.set()
        for channel in channels:
            channel.close()

    def cancel(self) -> None:
        with self._lock:
            channels = tuple(self._channels)
            self.cancelled.set()
        for channel in channels:
            channel.close()


@dataclass(frozen=True, slots=True)
class ChannelMetrics:
    name: str
    max_units: int
    max_bytes: int
    high_water_units: int
    high_water_bytes: int
    current_units: int
    current_bytes: int
    put_count: int
    get_count: int
    put_wait_count: int
    put_wait_seconds: float


@dataclass(frozen=True, slots=True)
class OperationMetrics:
    stage: str
    admissions: int
    submissions: int
    completions: int
    emissions: int
    callback_seconds: float
    resource_handoff_count: int
    resource_handoff_seconds: float


@dataclass(frozen=True, slots=True)
class RuntimeMetrics:
    channels: tuple[ChannelMetrics, ...]
    operations: tuple[OperationMetrics, ...]
    current_leased_bytes: int
    high_water_leased_bytes: int
    trace_events_dropped: int


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One opt-in, synchronization-free runtime observation."""

    timestamp_ns: int
    kind: str
    stage: str | None = None
    sequence: int | None = None
    name: str | None = None
    units: int | None = None
    payload_bytes: int | None = None
    seconds: float | None = None


class _TraceRecorder:
    def __init__(self, *, enabled: bool, limit: int) -> None:
        if limit <= 0:
            raise ValueError("runtime trace limit must be positive")
        self.enabled = enabled
        self.limit = int(limit)
        self.events: list[TraceEvent] = []
        self.dropped = 0
        self._lock = threading.Lock()

    def record(self, kind: str, **values: Any) -> None:
        if not self.enabled:
            return
        event = TraceEvent(
            timestamp_ns=time.perf_counter_ns(),
            kind=kind,
            **values,
        )
        with self._lock:
            if len(self.events) < self.limit:
                self.events.append(event)
            else:
                self.dropped += 1

    def channel(
        self,
        action: str,
        name: str,
        units: int,
        payload_bytes: int,
    ) -> None:
        self.record(
            f"queue_{action}",
            name=name,
            units=units,
            payload_bytes=payload_bytes,
        )


class _LeaseAccounting:
    def __init__(self, trace: _TraceRecorder) -> None:
        self._trace = trace
        self._lock = threading.Lock()
        self.current_bytes = 0
        self.high_water_bytes = 0

    def new(
        self,
        payload: Any,
        *,
        estimated_bytes: int,
        descriptor: StorageDescriptor,
        on_release: Callable[[], None] | None = None,
    ) -> PayloadLease:
        charge = max(0, int(estimated_bytes))
        with self._lock:
            self.current_bytes += charge
            self.high_water_bytes = max(
                self.high_water_bytes,
                self.current_bytes,
            )
            current = self.current_bytes
        self._trace.record(
            "lease_acquire",
            name=descriptor.kind.value,
            payload_bytes=current,
        )

        def release() -> None:
            with self._lock:
                self.current_bytes -= charge
                current = self.current_bytes
            self._trace.record(
                "lease_release",
                name=descriptor.kind.value,
                payload_bytes=current,
            )
            if on_release is not None:
                on_release()

        return PayloadLease(
            payload,
            estimated_bytes=charge,
            on_release=release,
            descriptor=descriptor,
        )


class _RunCancelHandle:
    def __init__(self, supervisor: _Supervisor) -> None:
        self._supervisor = supervisor

    def close(self) -> None:
        self._supervisor.cancel()


@dataclass(frozen=True, slots=True)
class _TerminalExecution:
    affinity: str
    claims: tuple[ResourceClaim, ...]
    input_affinity: str | None
    input_claims: tuple[ResourceClaim, ...]
    buffering: BufferingSpec


@dataclass(frozen=True, slots=True)
class _SourceExecution:
    affinity: str
    claims: tuple[ResourceClaim, ...]
    buffering: BufferingSpec


def _claims_from_metadata(
    values: Iterable[ResourceClaim | ResourceKind | str],
    *,
    owner: str,
) -> tuple[ResourceClaim, ...]:
    claims: list[ResourceClaim] = []
    for value in values:
        if isinstance(value, ResourceClaim):
            claims.append(value)
            continue
        try:
            kind = value if isinstance(value, ResourceKind) else ResourceKind(str(value))
        except ValueError as exc:
            raise ValueError(
                f"{owner} declared unknown execution resource {value!r}"
            ) from exc
        claims.append(ResourceClaim(kind))
    return tuple(claims)


def _terminal_execution(consumer: Any) -> _TerminalExecution:
    if not callable(getattr(consumer, "consume", None)):
        raise TypeError("terminal consumer must define consume(value)")
    affinity = getattr(consumer, "execution_affinity", "writer")
    if not isinstance(affinity, str) or not affinity:
        raise TypeError("terminal consumer affinity must be a nonempty string")
    input_affinity = getattr(consumer, "execution_input_affinity", None)
    if input_affinity is not None and (
        not isinstance(input_affinity, str) or not input_affinity
    ):
        raise TypeError(
            "terminal consumer input affinity must be a nonempty string"
        )
    claims = _claims_from_metadata(
        getattr(
            consumer,
            "execution_resources",
            (ResourceKind.IO, ResourceKind.MEDIA),
        ),
        owner="terminal consumer",
    )
    input_claims = _claims_from_metadata(
        getattr(
            consumer,
            "execution_input_resources",
            (ResourceKind.GPU, ResourceKind.MEMORY_BANDWIDTH),
        ),
        owner="terminal input bridge",
    ) if input_affinity is not None else ()
    buffering = getattr(
        consumer,
        "execution_buffering",
        BufferingSpec(
            retained_input_units=1,
            pending_output_units=0,
            native_slots=1,
            estimated_bytes=1,
        ),
    )
    if not isinstance(buffering, BufferingSpec):
        raise TypeError(
            "terminal consumer execution_buffering must be a BufferingSpec"
        )
    return _TerminalExecution(
        affinity=affinity,
        claims=claims,
        input_affinity=input_affinity,
        input_claims=input_claims,
        buffering=buffering,
    )


def _source_execution(bridge: Any) -> _SourceExecution:
    if not callable(getattr(bridge, "prepare_input", None)):
        raise TypeError("source bridge must define prepare_input(unit)")
    affinity = getattr(bridge, "execution_affinity", "source-bridge")
    if not isinstance(affinity, str) or not affinity:
        raise TypeError("source bridge affinity must be a nonempty string")
    claims = _claims_from_metadata(
        getattr(
            bridge,
            "execution_resources",
            (ResourceKind.CPU, ResourceKind.MEMORY_BANDWIDTH),
        ),
        owner="source bridge",
    )
    buffering = getattr(
        bridge,
        "execution_buffering",
        BufferingSpec(
            retained_input_units=1,
            pending_output_units=1,
            native_slots=0,
            estimated_bytes=1,
        ),
    )
    if not isinstance(buffering, BufferingSpec):
        raise TypeError("source bridge execution_buffering must be a BufferingSpec")
    return _SourceExecution(affinity, claims, buffering)


class _StreamingGraph:
    """The actor graph; deliberately does not retain its public iterator."""

    def __init__(
        self,
        built: tuple[tuple[ResolvedStage, Any], ...],
        units: Iterable[FrameUnit],
        context: PipelineContext,
        finalizers: tuple[Callable[[], None], ...] = (),
        *,
        input_spec: StreamSpec | None = None,
        source_bridge: Any = None,
        terminal_consumer: Any = None,
        trace: bool = False,
        trace_limit: int = 100_000,
    ) -> None:
        self._trace = _TraceRecorder(enabled=trace, limit=trace_limit)
        self._leases = _LeaseAccounting(self._trace)
        self._built = built
        self._operations = expand_operations(built, context)
        self._source_bridge = source_bridge
        self._source_execution = (
            _source_execution(source_bridge)
            if source_bridge is not None
            else None
        )
        self._terminal_consumer = terminal_consumer
        self._terminal_execution = (
            _terminal_execution(terminal_consumer)
            if terminal_consumer is not None
            else None
        )
        self._terminal_done = threading.Event()
        self._terminal_result: Any = None
        self._terminal_finished = False
        self._operation_indices = {
            id(operation): index
            for index, operation in enumerate(self._operations)
        }
        self._active_lock = threading.Lock()
        self._active_operations = [0] * len(self._operations)
        self._admissions = [0] * len(self._operations)
        self._submissions = [0] * len(self._operations)
        self._completions = [0] * len(self._operations)
        self._emissions = [0] * len(self._operations)
        self._callback_seconds = [0.0] * len(self._operations)
        self._resource_handoff_count = [0] * len(self._operations)
        self._resource_handoff_seconds = [0.0] * len(self._operations)
        self._downstream_conflicts = self._resolve_downstream_conflicts()
        self._units: Iterable[FrameUnit] | None = units
        self._context = context
        self._finalizers = finalizers
        self._supervisor = _Supervisor()
        self._arbiter = _ResourceArbiter()
        self._completion_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=max(1, len(self._operations)),
            thread_name_prefix="kinovsr-completion",
        )
        affinity_names = {
            spec.affinity.name
            for operation in self._operations
            for spec in (
                operation.execution,
                operation.input_bridge,
                operation.output_bridge,
            )
            if spec is not None
        }
        if self._terminal_execution is not None:
            affinity_names.add(self._terminal_execution.affinity)
            if self._terminal_execution.input_affinity is not None:
                affinity_names.add(self._terminal_execution.input_affinity)
        if self._source_execution is not None:
            affinity_names.add(self._source_execution.affinity)
        source_spec = (
            self._operations[0].stage.input_spec
            if self._operations
            else input_spec
        )
        if (
            source_spec is not None
            and source_spec.frame.layout is Layout.MLX_RGB_HWC
        ):
            affinity_names.add("mlx")
        self._lanes = {
            name: _AffinityLane(name)
            for name in sorted(affinity_names)
        }
        self._channels = self._make_channels(input_spec)
        self._pending_channels = self._make_pending_channels()
        bind_runtime_buffering = getattr(
            self._terminal_consumer,
            "bind_runtime_buffering",
            None,
        )
        if callable(bind_runtime_buffering):
            retained_units = sum(
                operation.execution.buffering.retained_input_units
                + operation.execution.buffering.pending_output_units
                + operation.execution.buffering.native_slots
                for operation in self._operations
            )
            channel_units = sum(
                channel.max_units
                for channel in (*self._channels, *self._pending_channels)
            )
            terminal_units = (
                self._terminal_execution.buffering.retained_input_units
                + self._terminal_execution.buffering.native_slots
                if self._terminal_execution is not None
                else 0
            )
            source_units = (
                self._source_execution.buffering.retained_input_units
                + self._source_execution.buffering.pending_output_units
                + self._source_execution.buffering.native_slots
                if self._source_execution is not None
                else 0
            )
            bind_runtime_buffering(
                max(
                    1,
                    retained_units
                    + channel_units
                    + terminal_units
                    + source_units,
                ),
                detach_input=(
                    bool(self._operations)
                    and self._operations[-1].output_slots == 0
                ),
            )
            resolved_buffering = getattr(
                self._terminal_consumer,
                "execution_buffering",
                None,
            )
            if (
                self._terminal_execution is not None
                and isinstance(resolved_buffering, BufferingSpec)
            ):
                self._terminal_execution = dataclasses.replace(
                    self._terminal_execution,
                    buffering=resolved_buffering,
                )
        self._output_slot_semaphores = {
            id(operation): threading.BoundedSemaphore(operation.output_slots)
            for operation in self._operations
            if operation.output_slots > 0
        }
        for channel in (*self._channels, *self._pending_channels):
            self._supervisor.register(channel)
        self._stage_threads: list[threading.Thread] = []
        self._publisher_threads: list[threading.Thread] = []
        self._source_thread: threading.Thread | None = None
        self._sink_thread: threading.Thread | None = None
        self._stream: Any = None  # retained for cleanup precedence compatibility
        # Keep two zero-copy terminal leases. The file endpoint's one-frame
        # duration holdback consumes frame N-1 only after pulling N; releasing
        # N-1 at that pull would return a bounded native pool slot while the
        # sink still owns the payload. Two remains finite and is a strict
        # extension of the public "valid until next pull" guarantee.
        self._terminal_leases: deque[Envelope] = deque()
        self._terminal_lease_limit = max(
            (1, *(operation.output_slots for operation in self._operations))
        )
        self._input_spec = (
            self._operations[0].stage.input_spec
            if self._operations
            else input_spec
        )
        self._terminal_spec = (
            self._operations[-1].stage.output_spec
            if self._operations
            else input_spec
        )
        self._source_storage = (
            storage_descriptor_for_spec(
                self._input_spec,
                borrowed=True,
                label="source",
            )
            if self._input_spec is not None
            else StorageDescriptor(label="source")
        )
        self._uses_mlx = any(
            Layout.MLX_RGB_HWC
            in {operation.stage.input_spec.frame.layout,
                operation.stage.output_spec.frame.layout}
            for operation in self._operations
        ) or (
            input_spec is not None
            and input_spec.frame.layout is Layout.MLX_RGB_HWC
        ) or (
            self._terminal_execution is not None
            and "mlx" in {
                self._terminal_execution.affinity,
                self._terminal_execution.input_affinity,
            }
        ) or (
            self._source_execution is not None
            and self._source_execution.affinity == "mlx"
        )
        self._mlx_stream: Any = None
        self._host_default_stream: Any = None
        # Nested owner turns execute on the same MLX thread and stream. The
        # lock remains exclusive across threads but must admit that deliberate
        # same-thread re-entry while a window barrier services queued bridges.
        self._mlx_lock = threading.RLock()
        self._started = False
        self._closed = False
        self._cleanup_lock = threading.RLock()

    def _make_channels(self, input_spec: StreamSpec | None) -> tuple[BoundedChannel, ...]:
        if self._operations:
            edge_specs = [self._operations[0].stage.input_spec]
            edge_specs.extend(operation.stage.output_spec for operation in self._operations)
        elif input_spec is not None:
            edge_specs = [input_spec]
        else:
            edge_specs = [None]
        channels: list[BoundedChannel] = []
        for index, spec in enumerate(edge_specs):
            # Cross-island pressure stays deliberately small. A processor's
            # finite multi-output burst is charged separately in its private
            # egress, so it can release its serial owner without turning this
            # edge into a whole-window queue.
            capacity = 2
            frame_bytes = estimate_frame_bytes(spec) if spec is not None else 1 << 28
            channels.append(
                BoundedChannel(
                    max_units=capacity,
                    max_bytes=frame_bytes * capacity,
                    name=f"edge-{index}",
                    observer=self._trace.channel,
                )
            )
        return tuple(channels)

    @staticmethod
    def _operation_resources(operation: PhysicalOperation) -> set[ResourceKind]:
        specs = (
            operation.execution,
            operation.input_bridge,
            operation.output_bridge,
        )
        return {
            claim.resource
            for spec in specs
            if spec is not None
            for claim in spec.claims
        }

    def _resolve_downstream_conflicts(self) -> tuple[tuple[int, ...], ...]:
        scarce = {
            ResourceKind.GPU,
            ResourceKind.ANE,
            ResourceKind.MEDIA,
            ResourceKind.MEMORY_BANDWIDTH,
            ResourceKind.OPAQUE_NATIVE,
        }
        resources = [
            self._operation_resources(operation) & scarce
            for operation in self._operations
        ]
        return tuple(
            tuple(
                downstream
                for downstream in range(index + 1, len(self._operations))
                if resources[index] & resources[downstream]
            )
            for index in range(len(self._operations))
        )

    def _resource_handoff(self, index: int) -> float:
        delay = self._operations[index].execution.resource_handoff_seconds
        conflicts = self._downstream_conflicts[index]
        if delay <= 0 or not conflicts:
            return 0.0
        with self._active_lock:
            if any(self._active_operations[item] for item in conflicts):
                return delay
        # Include queued work as pressure: a downstream actor can be between
        # calls when the completion thread makes this admission decision.
        if self._pending_channels[index].current_units or any(
            self._channels[item].current_units
            for item in conflicts
        ):
            return delay
        return 0.0

    def _progress_delay(self, index: int, requested: float) -> float:
        handoff = self._resource_handoff(index)
        effective = max(requested, handoff)
        if handoff > requested:
            with self._active_lock:
                self._resource_handoff_count[index] += 1
                self._resource_handoff_seconds[index] += handoff
            self._trace.record(
                "resource_handoff",
                stage=self._operations[index].stage.name,
                seconds=handoff,
            )
        return effective

    def _make_pending_channels(self) -> tuple[BoundedChannel, ...]:
        channels: list[BoundedChannel] = []
        for operation in self._operations:
            # Detach the processor's complete declared finite burst from its
            # owner. The courier may block, but only on its own thread. Window
            # barriers reached after that burst cooperatively service the
            # affinity lane, so this capacity cannot monopolize native/MLX
            # progress while it keeps ingress far enough ahead to submit the
            # next bounded window.
            capacity = max(
                1,
                operation.execution.buffering.pending_output_units,
            )
            frame_bytes = estimate_frame_bytes(operation.stage.output_spec)
            channels.append(
                BoundedChannel(
                    max_units=capacity,
                    max_bytes=frame_bytes * capacity,
                    name=f"pending:{operation.stage.name}",
                    observer=self._trace.channel,
                )
            )
        return tuple(channels)

    @property
    def metrics(self) -> tuple[ChannelMetrics, ...]:
        return tuple(
            ChannelMetrics(
                name=channel.name,
                max_units=channel.max_units,
                max_bytes=channel.max_bytes,
                high_water_units=channel.high_water_units,
                high_water_bytes=channel.high_water_bytes,
                current_units=channel.current_units,
                current_bytes=channel.current_bytes,
                put_count=channel.put_count,
                get_count=channel.get_count,
                put_wait_count=channel.put_wait_count,
                put_wait_seconds=channel.put_wait_seconds,
            )
            for channel in (*self._channels, *self._pending_channels)
        )

    @property
    def runtime_metrics(self) -> RuntimeMetrics:
        with self._active_lock:
            operations = tuple(
                OperationMetrics(
                    stage=operation.stage.name,
                    admissions=self._admissions[index],
                    submissions=self._submissions[index],
                    completions=self._completions[index],
                    emissions=self._emissions[index],
                    callback_seconds=self._callback_seconds[index],
                    resource_handoff_count=(
                        self._resource_handoff_count[index]
                    ),
                    resource_handoff_seconds=(
                        self._resource_handoff_seconds[index]
                    ),
                )
                for index, operation in enumerate(self._operations)
            )
        with self._leases._lock:
            current_leased = self._leases.current_bytes
            high_water_leased = self._leases.high_water_bytes
        return RuntimeMetrics(
            channels=self.metrics,
            operations=operations,
            current_leased_bytes=current_leased,
            high_water_leased_bytes=high_water_leased,
            trace_events_dropped=self._trace.dropped,
        )

    @property
    def trace_events(self) -> tuple[TraceEvent, ...]:
        with self._trace._lock:
            return tuple(self._trace.events)

    def __iter__(self) -> _StreamingGraph:
        return self

    def __next__(self) -> FrameUnit:
        if self._terminal_consumer is not None:
            raise TypeError(
                "a terminal-consumer run is completed with wait(), not iteration"
            )
        if self._closed:
            raise StopIteration
        if len(self._terminal_leases) >= self._terminal_lease_limit:
            self._terminal_leases.popleft().release()
        to_raise: BaseException | None = None
        message: Envelope | EndOfStream | None = None
        try:
            if not self._started:
                self._start()
            message = self._channels[-1].get()
        except ChannelClosed:
            failure = self._supervisor.failure
            if failure is None:
                failure = RuntimeError("streaming pipeline closed without EOS")
            to_raise = self._deliver_cleanup(failure)
        except BaseException as active:
            self._supervisor.fail(active)
            to_raise = self._deliver_cleanup(active)
        if to_raise is not None:
            raise to_raise
        assert message is not None
        if isinstance(message, EndOfStream):
            failure = self._supervisor.failure
            to_raise = self._deliver_cleanup(failure)
            if to_raise is None:
                raise StopIteration
            raise to_raise
        to_raise = None
        try:
            message.readiness.wait()
        except BaseException as active:
            message.release()
            self._supervisor.fail(active)
            to_raise = self._deliver_cleanup(active)
        if to_raise is not None:
            raise to_raise
        try:
            unit = self._rehome_terminal_unit(message.unit)
        except BaseException as active:
            message.release()
            self._supervisor.fail(active)
            failure = self._deliver_cleanup(active)
            if failure is active:
                raise
            raise failure from active
        self._terminal_leases.append(message)
        return unit

    def _rehome_terminal_unit(self, unit: FrameUnit) -> FrameUnit:
        """Snapshot a terminal MLX value on its owner, rebuild it for host use."""
        if (
            self._terminal_spec is None
            or self._terminal_spec.frame.layout is not Layout.MLX_RGB_HWC
            or self._mlx_stream is None
        ):
            return unit
        import mlx.core as mx

        def snapshot() -> _MlxSnapshot | None:
            if not isinstance(unit.payload, mx.array):
                return None
            with self._mlx_lock:
                payload = mx.contiguous(unit.payload)
                mx.eval(payload)
                return _MlxSnapshot(
                    bytes(memoryview(payload)),
                    tuple(payload.shape),
                    payload.dtype,
                )

        value = self._lanes["mlx"].call(
            snapshot,
            priority=-len(self._operations) - 3,
        )
        if value is None:
            return unit
        stream = self._host_default_stream
        if stream is None:
            raise RuntimeError("host MLX stream is unavailable")
        with mx.stream(stream):
            raw = mx.array(memoryview(value.data))
            payload = raw.view(value.dtype).reshape(value.shape)
            mx.eval(payload)
        return unit.with_payload(payload)

    def wait(self) -> Any:
        """Wait for an attached terminal consumer and return its result."""
        if self._terminal_consumer is None:
            raise TypeError("an iterator run has no terminal consumer to wait for")
        if self._closed:
            return self._terminal_result
        if not self._started:
            try:
                self._start()
            except BaseException as active:
                failure = self._deliver_cleanup(active)
                if failure is active:
                    raise
                raise failure from active
        self._terminal_done.wait()
        failure = self._supervisor.failure
        delivered = self._deliver_cleanup(failure)
        if delivered is not None:
            raise delivered
        return self._terminal_result

    def _start(self) -> None:
        self._started = True
        self._stream = _RunCancelHandle(self._supervisor)
        if self._uses_mlx:
            import mlx.core as mx

            # A concrete host sequence may contain lazy arrays constructed on
            # the caller's ordinary thread-local stream before process() was
            # entered. Such a graph cannot be evaluated from another thread.
            # Materialize those already-retained values here; generators are
            # consumed on the MLX lane and therefore construct their values on
            # the correct stream without eager traversal.
            if isinstance(self._units, Sequence):
                for unit in self._units:
                    if isinstance(unit.payload, mx.array):
                        mx.eval(unit.payload)
            device = mx.default_device()
            self._host_default_stream = mx.default_stream(device)
            self._mlx_stream = mx.new_thread_unsafe_stream(device)
            mlx_lane = self._lanes.get("mlx")
            if mlx_lane is not None:
                mlx_lane.set_mlx_stream(self._mlx_stream)
        try:
            if self._source_bridge is not None:
                prepare_source = getattr(self._source_bridge, "prepare", None)
                if callable(prepare_source):
                    self._source_invoke(
                        lambda: prepare_source(self._input_spec, self._context),
                        claim_resources=False,
                    )
            for index, operation in enumerate(self._operations):
                self._invoke(
                    operation,
                    lambda operation=operation: operation.processor.prepare(
                        operation.stage.input_spec, operation.context
                    ),
                )
                bind_background = getattr(
                    operation.processor, "bind_background_submit", None
                )
                if callable(bind_background):
                    self._invoke(
                        operation,
                        lambda operation=operation,
                        index=index,
                        bind_background=bind_background: bind_background(
                            lambda callback, delay=0.0: self._submit(
                                operation,
                                callback,
                                delay=self._progress_delay(index, delay),
                            ),
                            self._completion_executor.submit,
                            self._lane(operation.execution).run_until,
                        ),
                        claim_resources=False,
                    )
            if self._terminal_consumer is not None:
                prepare = getattr(self._terminal_consumer, "prepare", None)
                if callable(prepare):
                    self._terminal_invoke(
                        lambda: prepare(self._terminal_spec, self._context),
                        claim_resources=False,
                    )
        except BaseException as failure:
            self._supervisor.fail(failure)
            raise

        self._publisher_threads = [
            threading.Thread(
                target=self._publisher_worker,
                args=(index,),
                name=f"kinovsr-publish-{operation.stage.name}",
                daemon=False,
            )
            for index, operation in enumerate(self._operations)
        ]
        self._stage_threads = [
            threading.Thread(
                target=self._stage_worker,
                args=(index, operation),
                name=f"kinovsr-stage-{operation.stage.name}",
                daemon=False,
            )
            for index, operation in enumerate(self._operations)
        ]
        # Consumers wait before producers start, avoiding startup-only queue
        # pressure and making the finite read-ahead bound deterministic.
        if self._terminal_consumer is not None:
            self._sink_thread = threading.Thread(
                target=self._sink_worker,
                name="kinovsr-sink",
                daemon=False,
            )
            self._sink_thread.start()
        for thread in reversed(self._publisher_threads):
            thread.start()
        for thread in reversed(self._stage_threads):
            thread.start()
        self._source_thread = threading.Thread(
            target=self._source_worker,
            name="kinovsr-source",
            daemon=False,
        )
        self._source_thread.start()

    def _lane(self, execution: ExecutionSpec) -> _AffinityLane:
        return self._lanes[execution.affinity.name]

    def _guarded_callback(
        self,
        operation: PhysicalOperation,
        execution: ExecutionSpec,
        callback: Callable[[], Any],
        *,
        claim_resources: bool,
    ) -> Callable[[], Any]:
        def guarded() -> Any:
            index = self._operation_indices[id(operation)]
            started = time.perf_counter()
            with self._active_lock:
                self._active_operations[index] += 1
                self._submissions[index] += 1
            self._trace.record(
                "submission",
                stage=operation.stage.name,
            )
            try:
                mlx_scope = (
                    self._mlx_lock
                    if execution.affinity.name == "mlx"
                    else contextlib.nullcontext()
                )
                with mlx_scope:
                    if claim_resources:
                        with self._arbiter.acquire(execution.claims):
                            return callback()
                    return callback()
            except Exception as exc:
                wrapped = _wrap_stage_error(operation.stage, exc)
                if wrapped is exc:
                    raise
                raise wrapped from exc
            finally:
                elapsed = time.perf_counter() - started
                with self._active_lock:
                    self._active_operations[index] -= 1
                    self._completions[index] += 1
                    self._callback_seconds[index] += elapsed
                self._trace.record(
                    "completion",
                    stage=operation.stage.name,
                    seconds=elapsed,
                )

        return guarded

    def _invoke(
        self,
        operation: PhysicalOperation,
        callback: Callable[[], Any],
        *,
        claim_resources: bool = True,
        execution: ExecutionSpec | None = None,
    ) -> Any:
        execution = operation.execution if execution is None else execution
        guarded = self._guarded_callback(
            operation,
            execution,
            callback,
            claim_resources=claim_resources,
        )
        index = self._operation_indices[id(operation)]
        return self._lane(execution).call(guarded, priority=-index)

    def _submit(
        self,
        operation: PhysicalOperation,
        callback: Callable[[], Any],
        *,
        delay: float = 0.0,
    ) -> Future[Any]:
        """Queue owner-driven progress behind the current lane callback."""
        guarded = self._guarded_callback(
            operation,
            operation.execution,
            callback,
            claim_resources=True,
        )
        index = self._operation_indices[id(operation)]
        return self._lane(operation.execution).submit(
            guarded,
            delay=delay,
            priority=-index,
        )

    def _terminal_invoke(
        self,
        callback: Callable[[], Any],
        *,
        input_bridge: bool = False,
        claim_resources: bool = True,
    ) -> Any:
        execution = self._terminal_execution
        if execution is None:
            raise RuntimeError("terminal consumer execution is unavailable")
        affinity = (
            execution.input_affinity
            if input_bridge
            else execution.affinity
        )
        if affinity is None:
            raise RuntimeError("terminal consumer has no input bridge")
        claims = execution.input_claims if input_bridge else execution.claims

        def guarded() -> Any:
            started = time.perf_counter()
            self._trace.record(
                "submission",
                stage="sink:input" if input_bridge else "sink",
            )
            try:
                mlx_scope = (
                    self._mlx_lock
                    if affinity == "mlx"
                    else contextlib.nullcontext()
                )
                with mlx_scope:
                    if claim_resources:
                        with self._arbiter.acquire(claims):
                            return callback()
                    return callback()
            finally:
                self._trace.record(
                    "completion",
                    stage="sink:input" if input_bridge else "sink",
                    seconds=time.perf_counter() - started,
                )

        priority = -len(self._operations) - (2 if input_bridge else 1)
        return self._lanes[affinity].call(guarded, priority=priority)

    def _source_invoke(
        self,
        callback: Callable[[], Any],
        *,
        claim_resources: bool = True,
    ) -> Any:
        execution = self._source_execution
        if execution is None:
            raise RuntimeError("source bridge execution is unavailable")

        def guarded() -> Any:
            started = time.perf_counter()
            self._trace.record("submission", stage="source:bridge")
            try:
                mlx_scope = (
                    self._mlx_lock
                    if execution.affinity == "mlx"
                    else contextlib.nullcontext()
                )
                with mlx_scope:
                    if claim_resources:
                        with self._arbiter.acquire(execution.claims):
                            return callback()
                    return callback()
            finally:
                self._trace.record(
                    "completion",
                    stage="source:bridge",
                    seconds=time.perf_counter() - started,
                )

        return self._lanes[execution.affinity].call(guarded, priority=1)

    def _prepare_source(self, unit: FrameUnit) -> FrameUnit:
        if self._source_bridge is None:
            return unit
        prepared = self._source_invoke(
            lambda: self._source_bridge.prepare_input(unit)
        )
        if not isinstance(prepared, FrameUnit):
            raise TypeError("source bridge prepare_input() must return a FrameUnit")
        return prepared

    def _prepare_terminal(self, unit: FrameUnit) -> Any:
        execution = self._terminal_execution
        if execution is None or execution.input_affinity is None:
            return unit
        prepare = getattr(self._terminal_consumer, "prepare_input", None)
        if not callable(prepare):
            raise RuntimeError(
                "terminal consumer declares an input bridge without "
                "prepare_input()"
            )
        return self._terminal_invoke(
            lambda: prepare(unit),
            input_bridge=True,
        )

    def _finish_terminal(self) -> Any:
        if self._terminal_finished:
            return self._terminal_result
        finish = getattr(self._terminal_consumer, "finish", None)
        self._terminal_result = (
            self._terminal_invoke(finish)
            if callable(finish)
            else None
        )
        self._terminal_finished = True
        return self._terminal_result

    def _operation_iterator(
        self,
        operation: PhysicalOperation,
        callback: Callable[[], Iterable[FrameUnit]],
    ) -> Iterator[FrameUnit]:
        return self._invoke(operation, lambda: iter(callback()))

    def _advance(
        self,
        operation: PhysicalOperation,
        iterator: Iterator[FrameUnit],
    ) -> FrameUnit | object:
        def advance() -> FrameUnit | object:
            try:
                return next(iterator)
            except StopIteration:
                return _ITERATION_DONE

        return self._invoke(operation, advance)

    def _close_iterator(
        self,
        operation: PhysicalOperation,
        iterator: Iterator[FrameUnit],
    ) -> None:
        close = getattr(iterator, "close", None)
        if callable(close):
            self._invoke(operation, close, claim_resources=False)

    def _prepare_input(
        self,
        operation: PhysicalOperation,
        unit: FrameUnit,
    ) -> FrameUnit:
        execution = operation.input_bridge
        if execution is None:
            return unit
        hook = getattr(operation.processor, "prepare_input", None)
        if not callable(hook):
            raise RuntimeError(
                f"stage {operation.stage.name!r} declares an input bridge "
                "without prepare_input()"
            )
        prepared = self._invoke(
            operation,
            lambda: hook(unit, operation.context),
            execution=execution,
        )
        if not isinstance(prepared, FrameUnit):
            raise TypeError("prepare_input() must return a FrameUnit")
        return prepared

    def _prepare_output(
        self,
        operation: PhysicalOperation,
        unit: FrameUnit,
    ) -> FrameUnit:
        execution = operation.output_bridge
        if execution is None:
            return unit
        hook = getattr(operation.processor, "prepare_output", None)
        if not callable(hook):
            raise RuntimeError(
                f"stage {operation.stage.name!r} declares an output bridge "
                "without prepare_output()"
            )
        prepared = self._invoke(
            operation,
            lambda: hook(unit, operation.context),
            execution=execution,
        )
        if not isinstance(prepared, FrameUnit):
            raise TypeError("prepare_output() must return a FrameUnit")
        return prepared

    def _publish_iterator(
        self,
        operation: PhysicalOperation,
        iterator: Iterator[FrameUnit],
        output: BoundedChannel,
        pending: list[Boundary],
        source: Envelope | None,
        sequence: list[int],
    ) -> list[PayloadLease]:
        completed = False
        published: list[PayloadLease] = []
        active: BaseException | None = None
        cleanup: BaseException | None = None
        try:
            while not self._supervisor.cancelled.is_set():
                release_slot = self._acquire_output_slot(operation)
                try:
                    produced = self._advance(operation, iterator)
                except BaseException:
                    if release_slot is not None:
                        release_slot()
                    raise
                if produced is _ITERATION_DONE:
                    if release_slot is not None:
                        release_slot()
                    completed = True
                    break
                try:
                    produced = self._prepare_output(operation, produced)
                except BaseException:
                    if release_slot is not None:
                        release_slot()
                    raise
                if pending:
                    produced = dataclasses.replace(
                        produced,
                        boundaries=(*pending, *produced.boundaries),
                    )
                    pending.clear()
                if source is not None and produced.payload is source.unit.payload:
                    lease = source.lease.retain()
                    if release_slot is not None:
                        release_slot()
                else:
                    lease = self._leases.new(
                        produced.payload,
                        estimated_bytes=estimate_frame_bytes(operation.stage.output_spec),
                        descriptor=operation.output_storage,
                        on_release=release_slot,
                    )
                envelope = Envelope(
                    unit=produced,
                    lease=lease,
                    sequence=sequence[0],
                )
                published.append(lease)
                index = self._operation_indices[id(operation)]
                with self._active_lock:
                    self._emissions[index] += 1
                self._trace.record(
                    "emission",
                    stage=operation.stage.name,
                    sequence=sequence[0],
                    payload_bytes=lease.estimated_bytes,
                )
                sequence[0] += 1
                try:
                    output.put(envelope)
                except BaseException:
                    envelope.release()
                    raise
                del envelope, produced
        except BaseException as exc:  # noqa: BLE001 - precedence below
            active = exc
        if not completed:
            try:
                self._close_iterator(operation, iterator)
            except BaseException as exc:  # noqa: BLE001 - precedence below
                cleanup = exc
        if active is not None:
            if isinstance(active, ChannelClosed) and cleanup is not None:
                raise cleanup
            if cleanup is not None:
                if not isinstance(cleanup, Exception) and isinstance(active, Exception):
                    _append_context(cleanup, [active])
                    raise cleanup
                _append_context(active, [cleanup])
            raise active
        if cleanup is not None:
            raise cleanup
        return published

    def _wait_boundary_outputs(
        self,
        operation: PhysicalOperation,
        leases: list[PayloadLease],
    ) -> None:
        # Bounded native output slots are the stronger ownership mechanism:
        # reset may proceed while a terminal/file holdback keeps a lease, but
        # its backing cannot be recycled until the semaphore callback fires.
        # Compatibility outputs have no such ring contract, so preserve the
        # historical drain-before-reset lifetime exactly.
        if operation.output_slots > 0:
            return
        for lease in leases:
            while not lease.wait_released(0.05):
                if self._supervisor.cancelled.is_set():
                    return

    def _acquire_output_slot(
        self,
        operation: PhysicalOperation,
    ) -> Callable[[], None] | None:
        semaphore = self._output_slot_semaphores.get(id(operation))
        if semaphore is None:
            return None
        while not self._supervisor.cancelled.is_set():
            if semaphore.acquire(timeout=0.05):
                return semaphore.release
        raise ChannelClosed(
            f"output slots for {operation.stage.name!r} were cancelled"
        )

    def _stage_worker(self, index: int, operation: PhysicalOperation) -> None:
        input_channel = self._channels[index]
        output_channel = self._pending_channels[index]
        pending: list[Boundary] = []
        started = False
        sequence = [0]
        current: Envelope | None = None
        try:
            while not self._supervisor.cancelled.is_set():
                message = input_channel.get()
                if isinstance(message, EndOfStream):
                    iterator = self._operation_iterator(
                        operation,
                        lambda: operation.processor.flush(operation.context),
                    )
                    self._publish_iterator(
                        operation, iterator, output_channel, pending, None, sequence
                    )
                    output_channel.put(EOS)
                    return
                current = message
                current.readiness.wait()
                with self._active_lock:
                    self._admissions[index] += 1
                self._trace.record(
                    "admission",
                    stage=operation.stage.name,
                    sequence=current.sequence,
                    payload_bytes=current.estimated_bytes,
                )
                unit = current.unit
                if unit.boundaries:
                    if started:
                        iterator = self._operation_iterator(
                            operation,
                            lambda: operation.processor.flush(operation.context),
                        )
                        drained = self._publish_iterator(
                            operation,
                            iterator,
                            output_channel,
                            pending,
                            None,
                            sequence,
                        )
                        self._wait_boundary_outputs(operation, drained)
                    for boundary in unit.boundaries:
                        self._invoke(
                            operation,
                            lambda boundary=boundary: operation.processor.reset(
                                boundary, operation.context
                            ),
                        )
                    pending.extend(unit.boundaries)
                    unit = dataclasses.replace(unit, boundaries=())
                started = True
                unit = self._prepare_input(operation, unit)
                iterator = self._operation_iterator(
                    operation,
                    lambda unit=unit: operation.processor.process(
                        unit, operation.context
                    ),
                )
                self._publish_iterator(
                    operation,
                    iterator,
                    output_channel,
                    pending,
                    current,
                    sequence,
                )
                current.release()
                current = None
                del message, unit
        except ChannelClosed:
            return
        except BaseException as failure:  # noqa: BLE001 - supervisor owns delivery
            self._supervisor.fail(failure)
        finally:
            if current is not None:
                current.release()

    def _publisher_worker(self, index: int) -> None:
        """Move one stage's finite burst onto its two-unit public edge.

        Publication never runs on an affinity lane. A full downstream edge
        therefore blocks only this courier; the logical/native owner remains
        available to admit input or advance already-submitted work until its
        separately-accounted pending bound fills.
        """
        pending = self._pending_channels[index]
        output = self._channels[index + 1]
        current: Envelope | None = None
        try:
            while not self._supervisor.cancelled.is_set():
                message = pending.get()
                if isinstance(message, EndOfStream):
                    output.put(EOS)
                    return
                current = message
                output.put(current)
                current = None
                del message
        except ChannelClosed:
            return
        except BaseException as failure:  # noqa: BLE001
            self._supervisor.fail(failure)
        finally:
            if current is not None:
                current.release()

    def _sink_worker(self) -> None:
        """Consume the terminal edge without putting the writer on a pull stack."""
        channel = self._channels[-1]
        current: Envelope | None = None
        held: Envelope | None = None
        try:
            while not self._supervisor.cancelled.is_set():
                message = channel.get()
                if isinstance(message, EndOfStream):
                    self._finish_terminal()
                    return
                current = message
                current.readiness.wait()
                self._trace.record(
                    "admission",
                    stage="sink",
                    sequence=current.sequence,
                    payload_bytes=current.estimated_bytes,
                )
                value = self._prepare_terminal(current.unit)
                keep_going = self._terminal_invoke(
                    lambda value=value: self._terminal_consumer.consume(value)
                )
                if bool(
                    getattr(
                        self._terminal_consumer,
                        "retains_input_payload",
                        False,
                    )
                ):
                    previous, held = held, current
                    current = None
                    if previous is not None:
                        previous.release()
                else:
                    current.release()
                    current = None
                    if held is not None:
                        held.release()
                        held = None
                if keep_going is False:
                    self._finish_terminal()
                    self._supervisor.cancel()
                    return
        except ChannelClosed:
            return
        except BaseException as failure:  # noqa: BLE001
            self._supervisor.fail(failure)
        finally:
            if current is not None:
                current.release()
            if held is not None:
                held.release()
            self._terminal_done.set()

    def _source_worker(self) -> None:
        units, self._units = self._units, None
        iterator: Iterator[FrameUnit] | None = None
        first = True
        sequence = 0
        output = self._channels[0]
        # The channel was resolved from the typed input edge even for an empty
        # pipeline; charge exactly one of its slots rather than a fallback.
        estimated_bytes = output.max_bytes // output.max_units
        try:
            iterator = iter(units)
            while True:
                try:
                    if self._source_bridge is not None:
                        # File decode remains on its own admission actor. Only
                        # the explicit upload/debug bridge enters the MLX lane.
                        unit = next(iterator)
                    elif self._mlx_stream is None:
                        unit = next(iterator)
                    else:
                        # MLX-backed file and host sources are consumed on the
                        # same long-lived lane as every MLX processor. The
                        # source actor remains the bounded admission owner; it
                        # never constructs lazy MLX work on its own thread.
                        lane = self._lanes["mlx"]

                        def advance() -> FrameUnit:
                            with self._mlx_lock:
                                return next(iterator)

                        unit = lane.call(advance, priority=1)
                except StopIteration:
                    break
                if self._supervisor.cancelled.is_set():
                    return
                if self._source_bridge is not None:
                    self._trace.record(
                        "admission",
                        stage="source:bridge",
                        sequence=sequence,
                    )
                    unit = self._prepare_source(unit)
                    self._trace.record(
                        "emission",
                        stage="source:bridge",
                        sequence=sequence,
                        payload_bytes=estimated_bytes,
                    )
                if first:
                    first = False
                    if not any(
                        boundary.kind is BoundaryKind.STREAM_START
                        for boundary in unit.boundaries
                    ):
                        unit = unit.with_boundary(Boundary(BoundaryKind.STREAM_START))
                envelope = Envelope(
                    unit=unit,
                    lease=self._leases.new(
                        unit.payload,
                        estimated_bytes=estimated_bytes,
                        descriptor=self._source_storage,
                    ),
                    sequence=sequence,
                )
                sequence += 1
                try:
                    output.put(envelope)
                except BaseException:
                    envelope.release()
                    raise
            output.put(EOS)
        except ChannelClosed:
            return
        except BaseException as failure:  # noqa: BLE001 - supervisor owns delivery
            self._supervisor.fail(failure)
        finally:
            if iterator is not None and self._supervisor.cancelled.is_set():
                close = getattr(iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as failure:  # noqa: BLE001
                        self._supervisor.fail(failure)

    def _finish_graph(self, *, cancel: bool) -> None:
        if cancel:
            self._supervisor.cancel()
        sink = self._sink_thread
        if sink is not None and sink is not threading.current_thread():
            sink.join()
        source = self._source_thread
        if source is not None and source is not threading.current_thread():
            source.join()
        for thread in self._stage_threads:
            if thread is not threading.current_thread():
                thread.join()
        for thread in self._publisher_threads:
            if thread is not threading.current_thread():
                thread.join()
        completion, self._completion_executor = self._completion_executor, None
        if completion is not None:
            completion.shutdown(wait=True, cancel_futures=False)
        for channel in (*self._channels, *self._pending_channels):
            channel.close()
            channel.drain()
        while self._terminal_leases:
            self._terminal_leases.popleft().release()
        self._stream = None

    @staticmethod
    def _is_interrupt(exc: BaseException) -> bool:
        return not isinstance(exc, Exception)

    def _close_all(self) -> tuple[list[Exception], list[BaseException]]:
        if self._closed:
            return [], []
        self._closed = True
        close_errors: list[Exception] = []
        interrupts: list[BaseException] = []
        if self._terminal_consumer is not None:
            close = getattr(self._terminal_consumer, "close", None)
            if callable(close):
                try:
                    self._terminal_invoke(close, claim_resources=False)
                except Exception as exc:  # noqa: BLE001 - collected below
                    close_errors.append(exc)
                except BaseException as exc:
                    interrupts.append(exc)
                finally:
                    self._trace.record("close", stage="sink")
        for operation in reversed(self._operations):
            try:
                self._invoke(
                    operation,
                    lambda operation=operation: operation.processor.close(
                        operation.context
                    ),
                    claim_resources=False,
                )
            except Exception as exc:  # noqa: BLE001 - collected below
                close_errors.append(exc)
            except BaseException as exc:
                interrupts.append(exc)
            finally:
                self._trace.record(
                    "close",
                    stage=operation.stage.name,
                )
        if self._source_bridge is not None:
            close = getattr(self._source_bridge, "close", None)
            if callable(close):
                try:
                    self._source_invoke(close, claim_resources=False)
                except Exception as exc:  # noqa: BLE001 - collected below
                    close_errors.append(exc)
                except BaseException as exc:
                    interrupts.append(exc)
            self._trace.record("close", stage="source:bridge")
        finalizers, self._finalizers = self._finalizers, ()
        for finalizer in reversed(finalizers):
            try:
                finalizer()
            except Exception as exc:  # noqa: BLE001 - collected below
                close_errors.append(exc)
            except BaseException as exc:
                interrupts.append(exc)
        for lane in self._lanes.values():
            lane.stop()
        return close_errors, interrupts

    def _deliver_cleanup(
        self,
        active: BaseException | None,
    ) -> BaseException | None:
        graph_error: BaseException | None = None
        try:
            self._finish_graph(cancel=active is not None)
        except BaseException as exc:  # noqa: BLE001 - precedence below
            graph_error = exc
        close_errors, interrupts = self._close_all()
        cleanup = [item for item in (graph_error, *close_errors) if item is not None]
        if active is None:
            ordered = [*cleanup, *interrupts]
            if not ordered:
                return None
            winner = interrupts[0] if interrupts else ordered[0]
            _append_context(winner, [item for item in ordered if item is not winner])
            return winner
        if interrupts:
            winner = interrupts[0]
            _append_context(winner, [active, *cleanup, *interrupts[1:]])
            return winner
        _append_context(active, cleanup)
        return active

    def _shutdown(self) -> BaseException | None:
        with self._cleanup_lock:
            if self._closed:
                return None
            stream, self._stream = self._stream, None
            stream_error: BaseException | None = None
            if stream is not None:
                try:
                    stream.close()
                except BaseException as exc:  # noqa: BLE001 - precedence below
                    stream_error = exc
            graph_error: BaseException | None = None
            try:
                self._finish_graph(cancel=True)
            except BaseException as exc:  # noqa: BLE001 - precedence below
                graph_error = exc
            close_errors, interrupts = self._close_all()
            ordered = [
                item
                for item in (stream_error, graph_error, *close_errors, *interrupts)
                if item is not None
            ]
            if not ordered:
                return None
            winner = next((item for item in ordered if self._is_interrupt(item)), ordered[0])
            _append_context(winner, [item for item in ordered if item is not winner])
            return winner

    def close(self) -> None:
        failure = self._shutdown()
        if failure is not None:
            raise failure

    def __enter__(self) -> _StreamingGraph:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        failure = self._shutdown()
        if failure is None:
            return
        if exc is not None and not self._is_interrupt(failure):
            _append_context(exc, [failure])
            return
        raise failure

    def __del__(self) -> None:
        with contextlib.suppress(BaseException):
            self.close()


class StreamingChainRun:
    """Owning synchronous iterator over a bounded asynchronous graph.

    Worker threads retain the private graph, never this public owner. That
    separation lets CPython abandonment finalize the owner immediately and
    cancel workers even while a bounded channel is full.
    """

    def __init__(
        self,
        built: tuple[tuple[ResolvedStage, Any], ...],
        units: Iterable[FrameUnit],
        context: PipelineContext,
        finalizers: tuple[Callable[[], None], ...] = (),
        *,
        input_spec: StreamSpec | None = None,
        source_bridge: Any = None,
        terminal_consumer: Any = None,
        trace: bool = False,
        trace_limit: int = 100_000,
    ) -> None:
        self._graph = _StreamingGraph(
            built,
            units,
            context,
            finalizers,
            input_spec=input_spec,
            source_bridge=source_bridge,
            terminal_consumer=terminal_consumer,
            trace=trace,
            trace_limit=trace_limit,
        )

    @property
    def _stream(self) -> Any:
        return self._graph._stream

    @_stream.setter
    def _stream(self, value: Any) -> None:
        self._graph._stream = value

    @property
    def metrics(self) -> tuple[ChannelMetrics, ...]:
        return self._graph.metrics

    @property
    def runtime_metrics(self) -> RuntimeMetrics:
        return self._graph.runtime_metrics

    @property
    def trace_events(self) -> tuple[TraceEvent, ...]:
        return self._graph.trace_events

    def __iter__(self) -> StreamingChainRun:
        return self

    def __next__(self) -> FrameUnit:
        return next(self._graph)

    def wait(self) -> Any:
        return self._graph.wait()

    def close(self) -> None:
        self._graph.close()

    def __enter__(self) -> StreamingChainRun:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._graph.__exit__(exc_type, exc, tb)

    def __del__(self) -> None:
        with contextlib.suppress(BaseException):
            self.close()


__all__ = [
    "ChannelMetrics",
    "OperationMetrics",
    "RuntimeMetrics",
    "StreamingChainRun",
    "TraceEvent",
]
