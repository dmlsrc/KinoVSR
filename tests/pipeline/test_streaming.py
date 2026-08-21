"""M8 contracts: ownership, finite pressure, affinity, and overlap."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from fractions import Fraction
from types import SimpleNamespace

import pytest

from kinovsr.pipeline.builder import ResolvedStage
from kinovsr.pipeline.channels import EOS, BoundedChannel
from kinovsr.pipeline.execution import expand_operations, resolve_execution
from kinovsr.pipeline.leases import (
    Envelope,
    FutureCompletion,
    PayloadLease,
    StorageDescriptor,
    StorageKind,
)
from kinovsr.pipeline.scheduler import run_chain
from kinovsr.pipeline.streaming import _AffinityLane, _StreamingGraph
from kinovsr.processors import (
    Boundary,
    BoundaryKind,
    Capability,
    CapabilitySpec,
    FrameUnit,
    Geometry,
    GopWindowPolicy,
    Layout,
    PipelineContext,
    StreamConstraint,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
    preserve_stream,
)
from kinovsr.processors.capabilities import TemporalMode
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


def _spec(layout: Layout = Layout.CV_BGRA) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709",
            full_range=False,
            geometry=Geometry(16, 16),
            layout=layout,
        ),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24_000),
            cadence=Fraction(25),
        ),
    )


def _stage(
    processor,
    name: str,
    affinity: str,
    *,
    temporal_mode: TemporalMode = TemporalMode.PER_FRAME,
    temporal_radius: int = 0,
    input_bridge: str | None = None,
    output_bridge: str | None = None,
    layout: Layout = Layout.CV_BGRA,
    resource_handoff: float = 0.0,
    resources: tuple[str, ...] | None = None,
    output_slots: int = 0,
) -> ResolvedStage:
    stream = _spec(layout)
    factory = SimpleNamespace(
        name="fake",
        execution_affinity=affinity,
        execution_input_affinity=input_bridge,
        execution_output_affinity=output_bridge,
        execution_resource_handoff_seconds=resource_handoff,
        execution_resources=resources,
        execution_output_slots=output_slots,
    )
    return ResolvedStage(
        name=name,
        position=0,
        family="fake",
        factory=factory,
        capability=Capability.PREPROCESS,
        capability_spec=CapabilitySpec(
            capability=Capability.PREPROCESS,
            profiles=(),
            accepts=StreamConstraint(),
            produces=preserve_stream,
            temporal_mode=temporal_mode,
            temporal_radius=temporal_radius,
            stateful=True,
        ),
        profile=None,
        config=None,
        input_spec=stream,
        output_spec=stream,
    )


class _Pass:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.thread_ids: list[int] = []
        self.intervals: list[tuple[float, float]] = []
        self._active = False

    def _record(self) -> None:
        self.thread_ids.append(threading.get_ident())

    def prepare(self, input_spec, context) -> None:
        self._record()

    def process(self, unit, context):
        assert not self._active, "stateful stage was re-entered"
        self._active = True
        try:
            self._record()
            started = time.perf_counter()
            if self.delay:
                time.sleep(self.delay)
            self.intervals.append((started, time.perf_counter()))
            yield unit
        finally:
            self._active = False

    def reset(self, boundary, context) -> None:
        self._record()

    def flush(self, context):
        self._record()
        return ()

    def close(self, context) -> None:
        self._record()


def _units(count: int) -> list[FrameUnit]:
    return [
        FrameUnit(payload=f"frame-{index}", pts=index, duration=1)
        for index in range(count)
    ]


def test_payload_lease_frees_fanout_only_after_last_release() -> None:
    released: list[str] = []
    root = PayloadLease(
        object(),
        estimated_bytes=64,
        on_release=lambda: released.append("free"),
        descriptor=StorageDescriptor(
            StorageKind.CV_PIXEL_BUFFER,
            borrowed=True,
            reusable=True,
        ),
    )
    left = root.retain()
    right = root.retain()
    assert root.references == 3
    assert not root.wait_released(0)
    root.release()
    left.release()
    assert not root.wait_released(0)
    assert released == []
    right.release()
    assert root.wait_released(0)
    assert released == ["free"]


def test_payload_lease_promotes_reusable_storage() -> None:
    source = bytearray(b"abc")
    lease = PayloadLease(
        source,
        estimated_bytes=3,
        descriptor=StorageDescriptor(
            StorageKind.ANE_RAW_SLOT,
            borrowed=True,
            reusable=True,
        ),
    )
    durable = lease.promote(
        bytes,
        descriptor=StorageDescriptor(StorageKind.HOST_BUFFER),
    )
    source[:] = b"xxx"
    assert durable.payload == b"abc"
    assert durable.descriptor.kind is StorageKind.HOST_BUFFER


def test_channel_bounds_units_and_bytes_and_releases_drain() -> None:
    releases: list[int] = []
    channel = BoundedChannel(max_units=2, max_bytes=8, name="test")
    for index in range(2):
        unit = FrameUnit(payload=index, pts=index, duration=1)
        channel.put(
            Envelope(
                unit,
                PayloadLease(
                    index,
                    estimated_bytes=4,
                    on_release=lambda index=index: releases.append(index),
                ),
            )
        )
    assert channel.high_water_units == 2
    assert channel.high_water_bytes == 8
    assert channel.current_units == 2
    channel.drain()
    assert channel.current_units == 0
    assert sorted(releases) == [0, 1]


def test_envelope_release_clears_its_payload_carrier() -> None:
    payload = object()
    envelope = Envelope(
        FrameUnit(payload=payload, pts=0, duration=1),
        PayloadLease(payload, estimated_bytes=4),
    )
    envelope.release()
    assert envelope.unit.payload is None


def test_completion_wait_is_explicit() -> None:
    future: Future[None] = Future()
    completion = FutureCompletion(future)
    done = threading.Event()

    def wait() -> None:
        completion.wait()
        done.set()

    thread = threading.Thread(target=wait)
    thread.start()
    assert not done.wait(0.02)
    future.set_result(None)
    assert done.wait(1.0)
    thread.join()


def test_centered_gop_buffering_is_finite_and_run_resolved() -> None:
    processor = _Pass()
    stage = _stage(
        processor,
        "window",
        "mlx",
        temporal_mode=TemporalMode.CENTERED,
        temporal_radius=3,
    )
    context = PipelineContext(
        settings=Settings(),
        gop=GopWindowPolicy(min_window=16, max_window=48),
    )
    buffering = resolve_execution(stage, context).buffering
    assert buffering.retained_input_units == 50
    assert buffering.pending_output_units == 50
    assert buffering.native_slots == 0
    assert buffering.estimated_bytes > 0


def test_factory_resource_handoff_is_part_of_execution_contract() -> None:
    processor = _Pass()
    stage = _stage(
        processor,
        "window",
        "native:window",
        resource_handoff=0.01,
    )
    execution = resolve_execution(stage, PipelineContext(settings=Settings()))
    assert execution.resource_handoff_seconds == pytest.approx(0.01)


def test_resource_handoff_requires_conflicting_downstream_pressure() -> None:
    first, second = _Pass(), _Pass()
    stages = (
        (
            _stage(
                first,
                "first",
                "lane-a",
                resource_handoff=0.01,
                resources=("memory_bandwidth",),
            ),
            first,
        ),
        (
            _stage(
                second,
                "second",
                "lane-b",
                resources=("memory_bandwidth",),
            ),
            second,
        ),
    )
    graph = _StreamingGraph(
        stages,
        (),
        PipelineContext(settings=Settings()),
    )
    try:
        assert graph._resource_handoff(0) == 0.0
        graph._channels[1].put(EOS)
        assert graph._resource_handoff(0) == pytest.approx(0.01)
        assert graph._channels[1].get() is EOS
        assert graph._resource_handoff(0) == 0.0
    finally:
        graph.close()


def test_physical_bridges_expand_without_changing_logical_stage() -> None:
    processor = _Pass()
    stage = _stage(
        processor,
        "hybrid",
        "native:hybrid",
        input_bridge="mlx",
        output_bridge="mlx",
        layout=Layout.MLX_RGB_HWC,
    )
    operation = expand_operations(
        ((stage, processor),),
        PipelineContext(settings=Settings()),
    )[0]
    assert operation.stage is stage
    assert operation.execution.affinity.name == "native:hybrid"
    assert operation.input_bridge.affinity.name == "mlx"
    assert operation.output_bridge.affinity.name == "mlx"
    assert operation.output_storage.kind is StorageKind.MLX_ARRAY


def test_pooled_output_slots_declare_reusable_borrowed_storage() -> None:
    processor = _Pass()
    stage = _stage(
        processor,
        "native",
        "native:stage",
        output_slots=2,
    )
    operation = expand_operations(
        ((stage, processor),),
        PipelineContext(settings=Settings()),
    )[0]
    assert operation.output_storage.kind is StorageKind.CV_PIXEL_BUFFER
    assert operation.output_storage.borrowed
    assert operation.output_storage.reusable


def test_opt_in_trace_reports_lifecycle_pressure_and_lease_bytes() -> None:
    processor = _Pass()
    stream = run_chain(
        ((_stage(processor, "stage", "owner"), processor),),
        _units(3),
        PipelineContext(settings=Settings()),
        trace=True,
    )
    assert [unit.pts for unit in stream] == [0, 1, 2]
    kinds = {event.kind for event in stream.trace_events}
    assert {
        "admission",
        "submission",
        "completion",
        "emission",
        "lease_acquire",
        "lease_release",
        "queue_put",
        "close",
    } <= kinds
    metrics = stream.runtime_metrics
    assert metrics.current_leased_bytes == 0
    assert metrics.high_water_leased_bytes > 0
    assert metrics.operations[0].admissions == 3
    assert metrics.operations[0].emissions == 3
    assert all(channel.current_units == 0 for channel in metrics.channels)


def test_trace_capture_has_a_finite_event_bound() -> None:
    processor = _Pass()
    stream = run_chain(
        ((_stage(processor, "stage", "owner"), processor),),
        _units(3),
        PipelineContext(settings=Settings()),
        trace=True,
        trace_limit=2,
    )
    list(stream)
    assert len(stream.trace_events) == 2
    assert stream.runtime_metrics.trace_events_dropped > 0


def test_same_affinity_lifecycle_uses_one_long_lived_lane() -> None:
    first, second = _Pass(), _Pass()
    stages = (
        (_stage(first, "first", "shared-owner"), first),
        (_stage(second, "second", "shared-owner"), second),
    )
    assert [unit.pts for unit in run_chain(
        stages,
        _units(8),
        PipelineContext(settings=Settings()),
    )] == list(range(8))
    identities = set(first.thread_ids + second.thread_ids)
    assert len(identities) == 1
    assert identities != {threading.get_ident()}


def test_distinct_affinity_lanes_overlap_steady_state() -> None:
    count = 20
    delay = 0.008
    first, second = _Pass(delay=delay), _Pass(delay=delay)
    stages = (
        (_stage(first, "first", "lane-a"), first),
        (_stage(second, "second", "lane-b"), second),
    )
    output = list(run_chain(
        stages,
        _units(count),
        PipelineContext(settings=Settings()),
    ))
    assert len(output) == count
    lane_costs = [
        sum(end - start for start, end in processor.intervals)
        for processor in (first, second)
    ]
    sequential = sum(lane_costs)
    work_span = max(
        end for _start, end in (*first.intervals, *second.intervals)
    ) - min(
        start for start, _end in (*first.intervals, *second.intervals)
    )
    # Measure the steady actor span, excluding cold thread/PyObjC imports.
    # The loose ceiling still rejects the additive two-stage schedule.
    assert work_span < sequential * 0.70, (work_span, lane_costs)
    assert work_span < max(lane_costs) * 1.35, (work_span, lane_costs)


def test_terminal_consumer_is_a_bounded_graph_actor() -> None:
    processor = _Pass(delay=0.001)

    class Sink:
        execution_affinity = "sink-owner"
        execution_resources = ("io",)

        def __init__(self) -> None:
            self.values = []
            self.thread_ids = []
            self.closed = 0

        def consume(self, unit):
            self.thread_ids.append(threading.get_ident())
            self.values.append(unit.pts)
            return len(self.values) < 3

        def finish(self):
            self.thread_ids.append(threading.get_ident())
            return tuple(self.values)

        def close(self):
            self.thread_ids.append(threading.get_ident())
            self.closed += 1

    sink = Sink()
    run = run_chain(
        ((_stage(processor, "stage", "stage-owner"), processor),),
        _units(20),
        PipelineContext(settings=Settings()),
        terminal_consumer=sink,
    )
    assert run.wait() == (0, 1, 2)
    assert sink.values == [0, 1, 2]
    assert sink.closed == 1
    assert len(set(sink.thread_ids)) == 1
    assert sink.thread_ids[0] not in processor.thread_ids
    assert all(
        metric.high_water_units <= metric.max_units
        for metric in run.metrics
    )


def test_terminal_input_bridge_and_sink_have_distinct_owners() -> None:
    processor = _Pass()

    class Sink:
        execution_affinity = "writer-owner"
        execution_input_affinity = "bridge-owner"
        execution_resources = ("io",)
        execution_input_resources = ("cpu",)

        def __init__(self) -> None:
            self.bridge_threads = []
            self.writer_threads = []
            self.values = []

        def prepare_input(self, unit):
            self.bridge_threads.append(threading.get_ident())
            return unit.with_payload(f"prepared:{unit.payload}")

        def consume(self, unit):
            self.writer_threads.append(threading.get_ident())
            self.values.append(unit.payload)

        def finish(self):
            return tuple(self.values)

    sink = Sink()
    run = run_chain(
        ((_stage(processor, "stage", "stage-owner"), processor),),
        _units(2),
        PipelineContext(settings=Settings()),
        terminal_consumer=sink,
    )
    assert run.wait() == (
        "prepared:frame-0",
        "prepared:frame-1",
    )
    assert len(set(sink.bridge_threads)) == 1
    assert len(set(sink.writer_threads)) == 1
    assert sink.bridge_threads[0] != sink.writer_threads[0]


def test_source_decode_and_bridge_have_distinct_owners() -> None:
    processor = _Pass()

    class Source:
        def __init__(self) -> None:
            self.index = 0
            self.thread_ids = []

        def __iter__(self):
            return self

        def __next__(self):
            self.thread_ids.append(threading.get_ident())
            if self.index == 3:
                raise StopIteration
            unit = FrameUnit(
                payload=f"native-{self.index}",
                pts=self.index,
                duration=1,
            )
            self.index += 1
            return unit

    class Bridge:
        execution_affinity = "source-bridge-owner"
        execution_resources = ("cpu",)

        def __init__(self) -> None:
            self.thread_ids = []
            self.closed = 0

        def prepare(self, input_spec, context):
            self.thread_ids.append(threading.get_ident())

        def prepare_input(self, unit):
            self.thread_ids.append(threading.get_ident())
            return unit.with_payload(f"uploaded:{unit.payload}")

        def close(self):
            self.thread_ids.append(threading.get_ident())
            self.closed += 1

    source, bridge = Source(), Bridge()
    run = run_chain(
        ((_stage(processor, "stage", "stage-owner"), processor),),
        source,
        PipelineContext(settings=Settings()),
        source_bridge=bridge,
        trace=True,
    )
    assert [unit.payload for unit in run] == [
        "uploaded:native-0",
        "uploaded:native-1",
        "uploaded:native-2",
    ]
    assert bridge.closed == 1
    assert len(set(source.thread_ids)) == 1
    assert len(set(bridge.thread_ids)) == 1
    assert source.thread_ids[0] != bridge.thread_ids[0]
    assert bridge.thread_ids[0] not in processor.thread_ids
    source_events = {
        event.kind
        for event in run.trace_events
        if event.stage == "source:bridge"
    }
    assert {"admission", "submission", "completion", "emission", "close"} \
        <= source_events


def test_terminal_mlx_value_is_rebuilt_on_the_host_stream() -> None:
    import mlx.core as mx

    class AddOne(_Pass):
        def process(self, unit, context):
            self._record()
            yield unit.with_payload(unit.payload + 1)

    processor = AddOne()
    source = mx.arange(12, dtype=mx.float32).reshape(2, 2, 3)
    run = run_chain(
        (
            (
                _stage(
                    processor,
                    "mlx-stage",
                    "mlx",
                    layout=Layout.MLX_RGB_HWC,
                ),
                processor,
            ),
        ),
        [FrameUnit(payload=source, pts=0, duration=1)],
        PipelineContext(settings=Settings()),
    )
    output = next(run)
    with pytest.raises(StopIteration):
        next(run)
    mx.eval(output.payload)
    assert output.payload.shape == (2, 2, 3)
    assert output.payload.dtype == mx.float32
    assert output.payload.tolist() == (source + 1).tolist()


def test_shared_affinity_prefers_ready_downstream_work() -> None:
    lane = _AffinityLane("priority-test")
    entered = threading.Event()
    release = threading.Event()
    order = []

    def blocker():
        entered.set()
        release.wait()

    first = lane.submit(blocker)
    assert entered.wait(1.0)
    upstream = lane.submit(lambda: order.append("upstream"), priority=0)
    downstream = lane.submit(lambda: order.append("downstream"), priority=-1)
    release.set()
    first.result()
    downstream.result()
    upstream.result()
    lane.stop()
    assert order == ["downstream", "upstream"]


def test_affinity_lane_clears_loaded_mlx_state_before_thread_exit(
    monkeypatch,
) -> None:
    import sys

    calls = []
    fake_mx = SimpleNamespace(clear_streams=lambda: calls.append(threading.get_ident()))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    lane = _AffinityLane("mlx-cleanup-test")
    owner = lane.call(threading.get_ident)
    lane.stop()
    assert calls == [owner]


def test_terminal_consumption_releases_a_pre_boundary_payload() -> None:
    resets = []

    class Stage(_Pass):
        def reset(self, boundary, context):
            resets.append(boundary.kind)
            super().reset(boundary, context)

    class Sink:
        def __init__(self):
            self.values = []

        def consume(self, unit):
            self.values.append(unit.pts)

        def finish(self):
            return tuple(self.values)

    processor = Stage()
    feed = _units(2)
    feed[1] = feed[1].with_boundary(Boundary(BoundaryKind.HARD_CUT))
    sink = Sink()
    run = run_chain(
        ((_stage(processor, "stage", "owner"), processor),),
        feed,
        PipelineContext(settings=Settings()),
        terminal_consumer=sink,
    )
    assert run.wait() == (0, 1)
    assert BoundaryKind.HARD_CUT in resets
