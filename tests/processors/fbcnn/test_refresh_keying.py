"""Sync-keyed conditioning-refresh cadence (weight-free).

The QF map refresh must land on the source's sync samples once the comb
window has refilled, keep the frame counter as the ceiling, and reproduce
the counter-only cadence with gop=False or without flags.
"""
from types import SimpleNamespace

import mlx.core as mx
import pytest

import kinovsr.processors.fbcnn.deblocker as deblocker_module
from kinovsr.processors.fbcnn.deblocker import FbcnnDeblocker

pytestmark = pytest.mark.unit


@pytest.fixture
def engine(monkeypatch):
    def make(gop=True):
        monkeypatch.setattr(deblocker_module.net, "load_params",
                            lambda *_a, **_k: {})
        monkeypatch.setattr(deblocker_module.net, "_config",
                            lambda _p: (3, 4))
        monkeypatch.setattr(
            deblocker_module, "estimate_qf_map",
            lambda *_a, **_k: {
                "qf_grid": mx.array([[50.0]]),
                "valid": [[True]],
                "global": {"qf": 50.0},
                "coverage": 1.0,
            })
        return FbcnnDeblocker(None, quality="auto", gop=gop)
    return make


def _drive(d, n, syncs):
    frame = mx.zeros((1, 64, 64, 3), dtype=mx.float32)
    refreshes = []
    for i in range(n):
        before = d.qf_refresh_count
        d._refresh_qf(frame, sync=i in syncs)
        if d.qf_refresh_count > before:
            refreshes.append(i)
    return refreshes


def test_refreshes_land_on_sync_samples(engine):
    d = engine(gop=True)
    # First estimate at QF_MIN_FRAMES-1 (buffer filled), then at each
    # sync sample once the comb window has refilled.
    syncs = {30, 60, 90}
    got = _drive(d, 100, syncs)
    assert got[0] == d.QF_MIN_FRAMES - 1
    assert set(got[1:]) == {30, 60, 90}


def test_counter_stays_the_ceiling_without_syncs(engine):
    d = engine(gop=True)
    got = _drive(d, 150, syncs=set())
    first = d.QF_MIN_FRAMES - 1
    assert got[0] == first
    # counter cadence: due once _since_qf reaches MAP_REFRESH
    assert got[1] == first + d.MAP_REFRESH + 1


def test_sync_before_window_refill_waits(engine):
    d = engine(gop=True)
    # GOP shorter than the comb window: a sync sample arriving before the
    # comb has refilled is skipped, so refreshes land on every SECOND
    # I-frame - still boundary-aligned, never comb-starved.
    got = _drive(d, 90, syncs=set(range(0, 90, 8)))
    assert got == [d.QF_MIN_FRAMES - 1, 24, 40, 56, 72, 88]
    gaps = [b - a for a, b in zip(got[1:], got[2:], strict=False)]
    assert all(gap >= d.QF_WINDOW for gap in gaps)


def test_gop_off_reproduces_counter_cadence(engine):
    d = engine(gop=False)
    # gop=False is enforced in denoise() (sync never reaches
    # _refresh_qf); driving with sync=False everywhere models it.
    got = _drive(d, 150, syncs=set())
    d2 = engine(gop=True)
    got2 = _drive(d2, 150, syncs=set())
    assert got == got2


def test_denoise_gop_off_ignores_source_flags(engine, monkeypatch):
    d = engine(gop=False)
    calls = []
    original = d._refresh_qf

    def spy(inp, *, sync=False):
        calls.append(sync)
        return original(inp, sync=sync)

    monkeypatch.setattr(d, "_refresh_qf", spy)
    monkeypatch.setattr(d, "_forward_auto", lambda inp: inp)
    frame = mx.zeros((64, 64, 3), dtype=mx.float32)
    source = SimpleNamespace(is_sync=True)
    d.denoise(frame, source=source)
    assert calls == [False]


def test_per_frame_driver_forwards_source_when_accepted():
    from kinovsr.processors.feed_driver import PerFrameDriver

    class WantsSource:
        def __init__(self):
            self.seen = []

        def denoise(self, frame, source=None):
            self.seen.append(source)
            return frame

    class Legacy:
        def denoise(self, frame):
            return frame

    unit = SimpleNamespace(source=SimpleNamespace(is_sync=True))
    wants = WantsSource()
    PerFrameDriver(wants).feed(mx.zeros((2, 2, 3)), token=unit)
    assert wants.seen == [unit.source]
    # legacy engines keep the bare call
    PerFrameDriver(Legacy()).feed(mx.zeros((2, 2, 3)), token=unit)
