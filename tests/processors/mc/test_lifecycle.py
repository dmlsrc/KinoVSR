"""Transactional native optical-flow construction and teardown."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _Processor:
    def __init__(self, *, fail_close: bool = False, close_message: str = "close failed") -> None:
        self.end_count = 0
        self.fail_close = fail_close
        self.close_message = close_message

    def endSession(self) -> None:
        self.end_count += 1
        if self.fail_close:
            raise RuntimeError(self.close_message)

    def startSessionWithConfiguration_error_(self, *_args):
        return True, None


def _worker(processor):
    return {
        "proc": processor,
        "ref": object(),
        "fwd": object(),
        "bwd": object(),
    }


def test_worker_allocation_failure_ends_started_session(monkeypatch):
    from kinovsr.processors import mc

    processor = _Processor(fail_close=True)

    class Config:
        def sourcePixelBufferAttributes(self):
            return {}

        def destinationPixelBufferAttributes(self):
            return {}

    class ConfigClass:
        @classmethod
        def alloc(cls):
            return cls()

        @classmethod
        def defaultRevision(cls):
            return 1

        def initWithFrameWidth_frameHeight_qualityPrioritization_revision_(self, *_args):
            return Config()

    class ProcessorClass:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return processor

    def fail_allocation(*_args):
        raise RuntimeError("buffer allocation failed")

    monkeypatch.setattr(mc.vt, "VTFrameProcessor", ProcessorClass)
    monkeypatch.setattr(
        processor,
        "startSessionWithConfiguration_error_",
        lambda *_args: (True, None),
        raising=False,
    )
    monkeypatch.setattr(mc._pb, "make_pixel_buffer_from_attrs", fail_allocation)
    engine = mc.McTemporalDenoiser.__new__(mc.McTemporalDenoiser)
    engine.w = engine.h = 16
    engine._src_attrs = None
    engine._dst_attrs = None

    with pytest.raises(RuntimeError, match="buffer allocation failed") as caught:
        engine._make_worker(ConfigClass)

    assert processor.end_count == 1
    assert str(caught.value.__context__) == "close failed"
    assert caught.value.__context__.__context__ is None


def test_later_worker_failure_closes_earlier_workers(monkeypatch):
    from kinovsr.processors import mc

    first = _Processor(fail_close=True)
    calls = 0

    class ConfigClass:
        @staticmethod
        def isSupported():
            return True

    def make_worker(self, _cls):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _worker(first)
        raise RuntimeError("second worker failed")

    monkeypatch.setattr(mc.vt, "VTOpticalFlowConfiguration", ConfigClass)
    monkeypatch.setattr(mc.McTemporalDenoiser, "_make_worker", make_worker)

    with pytest.raises(RuntimeError, match="second worker failed") as caught:
        mc.McTemporalDenoiser(16, 16, window=2, self_test=False)

    assert first.end_count == 1
    assert str(caught.value.__context__) == "close failed"
    assert caught.value.__context__.__context__ is None


def test_constructor_preserves_every_worker_cleanup_failure(monkeypatch):
    from kinovsr.processors import mc

    first = _Processor(fail_close=True, close_message="first cleanup failed")
    second = _Processor(fail_close=True, close_message="second cleanup failed")
    processors = [first, second]

    class Config:
        def sourcePixelBufferAttributes(self):
            return {}

        def destinationPixelBufferAttributes(self):
            return {}

    class ConfigClass:
        @classmethod
        def alloc(cls):
            return cls()

        @staticmethod
        def defaultRevision():
            return 1

        @staticmethod
        def isSupported():
            return True

        def initWithFrameWidth_frameHeight_qualityPrioritization_revision_(self, *_args):
            return Config()

    class ProcessorClass:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            return processors.pop(0)

    allocations = 0

    def allocate(*_args):
        nonlocal allocations
        allocations += 1
        if allocations == 4:
            raise RuntimeError("second worker allocation failed")
        return object()

    monkeypatch.setattr(mc.vt, "VTOpticalFlowConfiguration", ConfigClass)
    monkeypatch.setattr(mc.vt, "VTFrameProcessor", ProcessorClass)
    monkeypatch.setattr(mc._pb, "make_pixel_buffer_from_attrs", allocate)

    with pytest.raises(RuntimeError, match="second worker allocation failed") as caught:
        mc.McTemporalDenoiser(16, 16, window=2, self_test=False)

    second_cleanup = caught.value.__context__
    first_cleanup = second_cleanup.__context__
    assert str(second_cleanup) == "second cleanup failed"
    assert str(first_cleanup) == "first cleanup failed"
    assert first_cleanup.__context__ is None
    assert first.end_count == second.end_count == 1


def test_close_continues_after_a_processor_failure():
    from kinovsr.processors import mc

    first = _Processor(fail_close=True)
    second = _Processor(fail_close=True, close_message="second close failed")
    engine = mc.McTemporalDenoiser.__new__(mc.McTemporalDenoiser)
    engine._pool = None
    engine._workers = [_worker(first), _worker(second)]
    engine._curr_buf = object()
    engine._src_attrs = {}
    engine._dst_attrs = {}
    engine._prev = object()
    engine._hist = [object()]
    engine._warp_grid = (object(), object())

    with pytest.raises(RuntimeError, match="close failed") as caught:
        engine.close()

    assert first.end_count == second.end_count == 1
    assert str(caught.value.__context__) == "second close failed"
    assert engine._workers == []
    engine.close()


def test_gate_openness_accumulates_resident_no_per_frame_sync():
    """The run stat must ride the frame's own eval, never force its own
    host/device sync per reference (a float() in _weight serializes the
    pipeline mid-frame)."""
    import mlx.core as mx

    from kinovsr.processors.mc import McTemporalDenoiser

    d = McTemporalDenoiser.__new__(McTemporalDenoiser)
    d.strength = 0.8
    d.sigma = 0.05
    d.occlusion = False
    d.confidence = False
    d._gain = 1.0
    d._resid_scale = 1.0
    d._sigma_plane = None
    d._w_sum = None
    d._w_n = 0

    anchor = mx.zeros((4, 4, 3))
    warped = mx.full((4, 4, 3), 0.02)
    w = d._weight(anchor, warped, None, None)
    assert isinstance(d._w_sum, mx.array)     # resident, not a Python float
    assert d._w_n == 1
    expected = float(mx.mean(w))
    assert abs(d.gate_openness - expected / 0.8) < 1e-6

    d._weight(anchor, warped, None, None)
    assert d._w_n == 2
    assert abs(d.gate_openness - expected / 0.8) < 1e-6

    # the sync-free contract in source form: no host readback in _weight
    import inspect

    src = inspect.getsource(McTemporalDenoiser._weight)
    assert "float(" not in src and ".item()" not in src


def test_vision_occlusion_keeps_bidirectional_flow():
    import mlx.core as mx

    from kinovsr.processors.mc import McTemporalDenoiser

    class Engine:
        def __init__(self):
            self.calls = []

        def compute(self, from_frame, to_frame):
            self.calls.append((from_frame, to_frame))
            return mx.zeros((*from_frame.shape[:2], 2))

    current = mx.zeros((4, 6, 3))
    reference = mx.ones((4, 6, 3))
    engine = McTemporalDenoiser.__new__(McTemporalDenoiser)
    engine.flow_source = "vision"
    engine.occlusion = True
    engine._vision = Engine()

    flows = engine._compute_flows(current, [reference])

    assert len(flows) == 1
    assert flows[0][0].shape == flows[0][1].shape == (4, 6, 2)
    assert engine._vision.calls == [
        (current, reference),
        (reference, current),
    ]
