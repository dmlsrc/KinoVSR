"""mc engine bookkeeping: gate statistics and flow directions."""

import pytest

pytestmark = pytest.mark.unit


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
    assert isinstance(d._w_sum, mx.array)  # resident, not a Python float
    assert d._w_n == 1
    expected = float(mx.mean(w))
    assert abs(d.gate_openness - expected / 0.8) < 1e-6

    d._weight(anchor, warped, None, None)
    assert d._w_n == 2
    assert abs(d.gate_openness - expected / 0.8) < 1e-6

    # the sync-free contract in source form: no host readback in _weight
    import inspect

    src = inspect.getsource(McTemporalDenoiser._weight)
    assert "float(" not in src
    assert ".item()" not in src


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
