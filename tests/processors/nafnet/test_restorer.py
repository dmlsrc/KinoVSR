"""NAFNet restorer and guard-policy tests."""

import mlx.core as mx
import pytest


def test_nafnet_model_rgb_clips_decode_overshoot():
    from kinovsr.processors.nafnet.restorer import model_rgb

    rgb = mx.array([[[[-0.25, 0.5, 1.25, 42.0]]]], dtype=mx.float32)
    inp = model_rgb(rgb)
    mx.eval(inp)
    assert inp.shape == (1, 1, 1, 3)
    assert float(mx.min(inp)) == 0.0
    assert float(mx.max(inp)) == 1.0
    assert float(inp[0, 0, 0, 1]) == 0.5


def test_nafnet_restorer_rejects_bad_pool_mode_before_loading_weights():
    from kinovsr.processors.nafnet.restorer import resolve_pool_mode

    with pytest.raises(ValueError, match="pool_mode"):
        resolve_pool_mode("gopro32", pool_mode="bogus")


def test_nafnet_pool_auto_matches_reference_variants():
    from kinovsr.processors.nafnet.restorer import resolve_pool_mode

    assert resolve_pool_mode("gopro", pool_mode="auto") == "local"
    assert resolve_pool_mode("gopro32", pool_mode="auto") == "local"
    assert resolve_pool_mode("reds", pool_mode="auto") == "local"
    assert resolve_pool_mode("sidd", pool_mode="auto") == "global"
    assert resolve_pool_mode("sidd32", pool_mode="auto") == "global"
    assert resolve_pool_mode("/tmp/nafnet_sidd_width64.safetensors", pool_mode="auto") == "global"

    with pytest.raises(ValueError, match="cannot infer"):
        resolve_pool_mode("/tmp/custom.safetensors", pool_mode="auto")


def test_nafnet_guard_auto_protects_gopro_variants_only():
    from kinovsr.processors.nafnet.restorer import resolve_guard_mode

    assert resolve_guard_mode("gopro", guard_mode="auto") == "reject"
    assert resolve_guard_mode("gopro32", guard_mode="auto") == "reject"
    assert resolve_guard_mode("sidd", guard_mode="auto") == "off"
    assert resolve_guard_mode("sidd32", guard_mode="auto") == "off"
    assert resolve_guard_mode("reds", guard_mode="auto") == "off"
    assert resolve_guard_mode("gopro32", guard_mode="residual") == "residual"
    assert resolve_guard_mode("sidd", guard_mode="control") == "control"
    assert resolve_guard_mode("gopro32", guard_mode="control-source") == "control-source"
    assert resolve_guard_mode("gopro32", guard_mode="fast") == "fast"
    assert resolve_guard_mode("gopro32", guard_mode="reject") == "reject"

    with pytest.raises(ValueError, match="guard_mode"):
        resolve_guard_mode("gopro32", guard_mode="bogus")


def test_nafnet_factory_routes_guard_notices_to_logging(
        monkeypatch, caplog):
    import kinovsr.processors.nafnet as nafnet_package
    from kinovsr.processors.nafnet.factory import FACTORY, NafnetStageConfig

    captured: dict = {}

    class StubRestorer:
        def __init__(self, *_args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(nafnet_package, "NafnetRestorer", StubRestorer)
    config = NafnetStageConfig(
        weights_spec="unused.safetensors",
        variant="gopro",
        strength=1.0,
        pool="auto",
        guard="reject",
        guard_threshold=0.12,
        guard_fast_fraction=0.85,
        guard_lockout=48,
        guard_ramp=12,
        guard_fall=None,
    )
    processor = FACTORY.build(config, context=None)
    processor._make_driver()

    with caplog.at_level(
            "WARNING", logger="kinovsr.processors.nafnet.factory"):
        captured["progress_message"]("[nafnet] guard notice")
    assert "[nafnet] guard notice" in caplog.messages


def test_nafnet_residual_guard_map_quadratic_knee():
    from kinovsr.processors.nafnet.restorer import residual_guard_map

    low = mx.full((1, 7, 7, 3), 0.06)
    high = mx.full((1, 7, 7, 3), 0.24)
    low_knee = residual_guard_map(low, guard=0.12)
    high_knee = residual_guard_map(high, guard=0.12)
    mx.eval(low_knee, high_knee)

    assert float(mx.min(low_knee)) == 1.0
    assert abs(float(mx.mean(high_knee)) - 0.25) < 1e-6


def test_nafnet_guard_probe_map_marks_framewide_risk():
    from kinovsr.processors.nafnet.restorer import guard_probe_map

    safe = mx.full((1, 9, 9, 3), 0.02)
    risky = mx.full((1, 9, 9, 3), 0.08)
    safe_probe = guard_probe_map(safe, guard=0.12)
    risky_probe = guard_probe_map(risky, guard=0.12)
    mx.eval(safe_probe, risky_probe)

    assert float(mx.max(safe_probe)) == 0.0
    assert 0.2 < float(mx.mean(risky_probe)) < 0.4


def test_nafnet_guard_probe_map_marks_low_amplitude_structure():
    from kinovsr.processors.nafnet.restorer import guard_probe_map

    rows = [[0.04 if r % 2 else -0.04 for _c in range(9)] for r in range(9)]
    residual = mx.array([[[[v, v, v] for v in row] for row in rows]], dtype=mx.float32)
    risk = guard_probe_map(residual, guard=0.12)
    mx.eval(risk)

    assert float(mx.max(risk)) > 0.1


def test_nafnet_luma_control_smooths_vertical_luma_not_chroma():
    from kinovsr.processors.nafnet.restorer import luma_control_input

    rgb = mx.array(
        [[
            [[0.0, 0.0, 0.0]],
            [[0.9, 0.9, 0.9]],
            [[0.0, 0.0, 0.0]],
        ]],
        dtype=mx.float32,
    )
    control = luma_control_input(rgb)
    mx.eval(control)

    assert control.shape == rgb.shape
    assert 0.29 < float(control[0, 1, 0, 0]) < 0.31
    assert 0.29 < float(control[0, 0, 0, 0]) < 0.31
    assert float(mx.max(mx.abs(control[..., :1] - control[..., 1:2]))) == 0.0


def test_nafnet_control_risk_map_uses_residual_disagreement():
    from kinovsr.processors.nafnet.restorer import control_risk_map

    residual = mx.full((1, 9, 9, 3), 0.08)
    control_residual = mx.zeros((1, 9, 9, 3))
    risk = control_risk_map(residual, control_residual, guard=0.12)
    mx.eval(risk)
    assert float(mx.min(risk[:, 2:-2, 2:-2])) > 0.7

    same = control_risk_map(residual, residual, guard=0.12)
    mx.eval(same)
    assert float(mx.max(same)) < 0.5


def test_nafnet_control_guard_framewide_risk_locks_to_control_source():
    from kinovsr.processors.nafnet.restorer import NafnetRestorer

    restorer = object.__new__(NafnetRestorer)
    restorer._guard = 0.12
    restorer._guard_fast_fraction = 0.5
    restorer._control_source_locked = False
    restorer._guarded_frames = 0
    messages: list[str] = []
    restorer._progress_message = messages.append

    def control_source(inp):
        return inp + 0.25

    restorer._control_source_fwd = control_source
    inp = mx.zeros((1, 9, 9, 3), dtype=mx.float32)
    out = mx.full((1, 9, 9, 3), 0.5, dtype=mx.float32)
    guarded = restorer._apply_control_guard(inp, out)
    mx.eval(guarded)

    assert restorer._control_source_locked
    assert abs(float(mx.mean(guarded)) - 0.25) < 1e-6
    assert messages and "control-source guard" in messages[0]


def _reject_test_restorer(messages: list[str], lockout: int = 48, ramp: int = 0,
                          fall: int | None = None):
    from kinovsr.processors.nafnet.restorer import NafnetRestorer, _derived_fall_frames

    restorer = object.__new__(NafnetRestorer)
    restorer._guard = 0.12
    restorer._guard_mode = "reject"
    restorer._guard_lockout_frames = lockout
    restorer._guard_ramp_frames = ramp
    restorer._guard_fall_frames = _derived_fall_frames(ramp) if fall is None else fall
    restorer._control_source_locked = True
    restorer._reject_locked = False
    restorer._reject_streak = 0
    restorer._reject_probe_countdown = 0
    restorer._reject_gain = 1.0
    restorer._reject_notices = 0
    restorer._guarded_frames = 0
    restorer._progress_message = messages.append
    return restorer


def test_nafnet_reject_guard_needs_consecutive_area_evidence():
    messages: list[str] = []
    restorer = _reject_test_restorer(messages)

    inp = mx.zeros((1, 16, 16, 3), dtype=mx.float32)
    # Isolated hot pixel: high raw peak, negligible smoothed area -- the healthy
    # deblur-residual shape that must never trip the guard.
    fluke = mx.zeros((1, 16, 16, 3), dtype=mx.float32)
    fluke[0, 8, 8, :] = 0.5
    guarded = restorer._apply_reject_guard(inp, fluke)
    mx.eval(guarded)
    assert not restorer._reject_locked
    assert float(mx.max(guarded)) > 0.0  # fluke frame passes through restored
    assert messages == []

    # Frame-wide moderate explosion: trips, first trip emits passthrough but
    # does not lock yet.
    explosion = mx.full((1, 16, 16, 3), 0.09, dtype=mx.float32)
    guarded = restorer._apply_reject_guard(inp, explosion)
    mx.eval(guarded)
    assert not restorer._reject_locked
    assert float(mx.max(guarded)) == 0.0
    # Second consecutive trip locks.
    guarded = restorer._apply_reject_guard(inp, explosion)
    mx.eval(guarded)
    assert restorer._reject_locked
    assert restorer._reject_probe_countdown > 0
    assert messages and "reject guard locked" in messages[0]

    # A clean probe unlocks and emits the restored frame.
    guarded = restorer._apply_reject_guard(inp, fluke)
    mx.eval(guarded)
    assert not restorer._reject_locked
    assert float(mx.max(guarded)) > 0.0
    assert len(messages) == 2 and "resumed" in messages[1]

    restorer.reset()
    assert not restorer._reject_locked
    assert not restorer._control_source_locked
    assert restorer._reject_streak == 0
    assert restorer._guarded_frames == 0


def test_nafnet_reject_guard_catastrophic_frame_locks_immediately():
    messages: list[str] = []
    restorer = _reject_test_restorer(messages)

    inp = mx.zeros((1, 9, 9, 3), dtype=mx.float32)
    out = mx.full((1, 9, 9, 3), 0.5, dtype=mx.float32)
    guarded = restorer._apply_reject_guard(inp, out)
    mx.eval(guarded)

    assert restorer._reject_locked
    assert float(mx.max(guarded)) == 0.0
    assert messages and "reject guard locked" in messages[0]


def test_nafnet_reject_guard_lockout_sets_probe_cadence():
    messages: list[str] = []
    restorer = _reject_test_restorer(messages, lockout=3)
    calls = []
    healthy = {"flag": False}

    def fake_fwd(x):
        calls.append(1)
        if healthy["flag"]:
            return x
        return x + 0.5  # catastrophic explosion

    restorer._fwd = fake_fwd
    frame = mx.zeros((9, 9, 3), dtype=mx.float32)

    restorer.denoise(frame)  # catastrophic -> locks, countdown 3
    assert restorer._reject_locked and len(calls) == 1
    restorer.denoise(frame)  # locked, countdown 3->2: passthrough, no net
    restorer.denoise(frame)  # locked, countdown 2->1: passthrough, no net
    assert len(calls) == 1
    healthy["flag"] = True
    out = restorer.denoise(frame)  # countdown 1->0: probe runs, clean -> unlock
    assert len(calls) == 2
    assert not restorer._reject_locked
    mx.eval(out)
    assert len(messages) == 2 and "re-probing every 3 frames" in messages[0] \
        and "resumed" in messages[1]


def test_nafnet_reject_guard_lockout_zero_never_reprobes():
    messages: list[str] = []
    restorer = _reject_test_restorer(messages, lockout=0)
    calls = []
    restorer._fwd = lambda x: (calls.append(1), x + 0.5)[1]
    frame = mx.zeros((9, 9, 3), dtype=mx.float32)

    restorer.denoise(frame)  # catastrophic -> locks
    assert restorer._reject_locked and len(calls) == 1
    for _ in range(5):
        out = restorer.denoise(frame)
    assert len(calls) == 1  # net never re-runs
    assert restorer._reject_locked
    assert float(mx.max(mx.abs(out - frame))) == 0.0
    assert len(messages) == 1 and "for the rest of the clip" in messages[0]


def test_nafnet_restorer_rejects_negative_lockout_before_loading_weights():
    from kinovsr.processors.nafnet.restorer import NafnetRestorer

    with pytest.raises(ValueError, match="guard_lockout_frames"):
        NafnetRestorer("gopro32", guard_lockout_frames=-1)
    with pytest.raises(ValueError, match="guard_ramp_frames"):
        NafnetRestorer("gopro32", guard_ramp_frames=-1)
    with pytest.raises(ValueError, match="guard_fall_frames"):
        NafnetRestorer("gopro32", guard_fall_frames=-1)


def test_nafnet_reject_guard_explicit_fall_overrides_derived():
    from kinovsr.processors.nafnet.restorer import (
        _REJECT_HARD_CUT_AREA,
        _REJECT_TRIP_AREA,
        _derived_fall_frames,
        _local_mag,
    )

    assert _derived_fall_frames(12) == 3
    assert _derived_fall_frames(8) == 2
    assert _derived_fall_frames(0) == 0

    def moderate_trip(restorer):
        for size in (5, 6, 7, 8):
            cand = mx.zeros((1, 48, 48, 3), dtype=mx.float32)
            cand[0, 20:20 + size, 20:20 + size, :] = 0.3
            mag = _local_mag(cand)
            area = float(mx.mean((mag > 0.5 * restorer._guard).astype(mx.float32)))
            if _REJECT_TRIP_AREA <= area < _REJECT_HARD_CUT_AREA:
                return cand
        raise AssertionError("no candidate landed in the moderate-trip band")

    inp = mx.zeros((1, 48, 48, 3), dtype=mx.float32)

    # Explicit long fall: one moderate trip only drops the gain by 1/8.
    restorer = _reject_test_restorer([], ramp=8, fall=8)
    trip = moderate_trip(restorer)
    restorer._apply_reject_guard(inp, trip)
    assert abs(restorer._reject_gain - 0.875) < 1e-9

    # fall=0 with an eased rise: trips hard-cut, recovery still ramps.
    restorer = _reject_test_restorer([], ramp=8, fall=0)
    guarded = restorer._apply_reject_guard(inp, trip)
    mx.eval(guarded)
    assert restorer._reject_gain == 0.0
    assert float(mx.max(guarded)) == 0.0
    clean = mx.full((1, 48, 48, 3), 0.05, dtype=mx.float32)
    guarded = restorer._apply_reject_guard(inp, clean)
    mx.eval(guarded)
    assert 0.0 < float(mx.mean(guarded)) < 0.05  # partial strength, easing back in


def test_nafnet_reject_guard_marginal_probe_stays_locked():
    from kinovsr.processors.nafnet.restorer import (
        _REJECT_RESUME_AREA_RATIO,
        _REJECT_TRIP_AREA,
        _local_mag,
    )

    messages: list[str] = []
    restorer = _reject_test_restorer(messages, lockout=5, ramp=3)
    restorer._reject_locked = True

    # Self-tuned marginal residual: explosion area between the resume line and
    # the trip line (the hysteresis band).
    inp = mx.zeros((1, 64, 64, 3), dtype=mx.float32)
    marginal = None
    for size in (2, 3, 4, 5):
        cand = mx.zeros((1, 64, 64, 3), dtype=mx.float32)
        cand[0, 20:20 + size, 20:20 + size, :] = 0.3
        mag = _local_mag(cand)
        area = float(mx.mean((mag > 0.5 * restorer._guard).astype(mx.float32)))
        if _REJECT_TRIP_AREA * _REJECT_RESUME_AREA_RATIO < area < _REJECT_TRIP_AREA:
            marginal = cand
            break
    assert marginal is not None, "no candidate landed in the hysteresis band"

    guarded = restorer._apply_reject_guard(inp, marginal)
    mx.eval(guarded)
    assert restorer._reject_locked           # marginal probe does not resume
    assert float(mx.max(guarded)) == 0.0     # emits passthrough
    assert restorer._reject_probe_countdown == 5  # next probe rescheduled
    assert messages == []                    # no resume notice

    # A clearly clean probe (below the resume line) does resume.
    guarded = restorer._apply_reject_guard(inp, inp)
    assert not restorer._reject_locked
    assert messages and "resumed" in messages[0]


def test_nafnet_reject_guard_gain_eases_out_and_back_in():
    from kinovsr.processors.nafnet.restorer import (
        _REJECT_HARD_CUT_AREA,
        _REJECT_TRIP_AREA,
        _local_mag,
    )

    messages: list[str] = []
    restorer = _reject_test_restorer(messages, ramp=8)  # fall = max(2, 8//4) = 2

    inp = mx.zeros((1, 48, 48, 3), dtype=mx.float32)
    clean = mx.full((1, 48, 48, 3), 0.05, dtype=mx.float32)  # healthy residual

    # Self-tuned moderate trip: explosion area in [trip, hard-cut) so the
    # fade-out path (not the hard cut) is exercised.
    moderate = None
    for size in (5, 6, 7, 8):
        cand = mx.zeros((1, 48, 48, 3), dtype=mx.float32)
        cand[0, 20:20 + size, 20:20 + size, :] = 0.3
        mag = _local_mag(cand)
        area = float(mx.mean((mag > 0.5 * restorer._guard).astype(mx.float32)))
        if _REJECT_TRIP_AREA <= area < _REJECT_HARD_CUT_AREA:
            moderate = cand
            break
    assert moderate is not None, "no candidate landed in the moderate-trip band"

    # Moderate trip from settled gain: fades out (knee-damped, partial), not a cut.
    guarded = restorer._apply_reject_guard(inp, moderate)
    mx.eval(guarded)
    peak = float(mx.max(guarded))
    assert 0.0 < peak < float(mx.max(moderate))
    assert restorer._reject_streak == 1 and not restorer._reject_locked

    # Recovery eases back in monotonically and settles bit-exact at full strength.
    means = []
    for _ in range(4):
        guarded = restorer._apply_reject_guard(inp, clean)
        means.append(float(mx.mean(guarded)))
    assert all(b > a for a, b in zip(means, means[1:], strict=False))
    assert abs(means[-1] - 0.05) < 1e-7  # gain settled -> exact `out`

    # Catastrophic frame still cuts hard to passthrough.
    catastrophic = mx.full((1, 48, 48, 3), 0.5, dtype=mx.float32)
    guarded = restorer._apply_reject_guard(inp, catastrophic)
    mx.eval(guarded)
    assert float(mx.max(guarded)) == 0.0
    assert restorer._reject_gain == 0.0


def test_nafnet_guard_notice_uses_progress_callback(capsys):
    from kinovsr.processors.nafnet.restorer import NafnetRestorer

    restorer = object.__new__(NafnetRestorer)
    restorer._guard = 0.12
    restorer._guarded_frames = 0
    messages: list[str] = []
    restorer._progress_message = messages.append

    restorer._notice_once("control", peak=0.5, frac=0.25)
    restorer._notice_once("control", peak=0.6, frac=0.5)

    assert capsys.readouterr().out == ""
    assert len(messages) == 1
    assert messages[0].startswith("[nafnet] control guard:")


def test_nafnet_tlsc_windows_match_reference_dummy_crop_scales():
    from kinovsr.processors.nafnet import net

    cfg = (32, (1, 1, 1, 28), 1, (1, 1, 1, 1))
    assert net._tlsc_kernel("encoders.0.0.sca.1", cfg) == (384, 384)
    assert net._tlsc_kernel("encoders.1.0.sca.1", cfg) == (192, 192)
    assert net._tlsc_kernel("encoders.3.0.sca.1", cfg) == (48, 48)
    assert net._tlsc_kernel("middle_blks.0.sca.1", cfg) == (24, 24)
    assert net._tlsc_kernel("decoders.0.0.sca.1", cfg) == (48, 48)
    assert net._tlsc_kernel("decoders.3.0.sca.1", cfg) == (384, 384)


def test_nafnet_local_avg_pool_matches_reference_padding():
    from kinovsr.processors.nafnet import net

    h, w = 5, 6
    values = [[r * w + c for c in range(w)] for r in range(h)]
    x = mx.array([[[[float(values[r][c])] for c in range(w)] for r in range(h)]])
    out = net._local_avg_pool2d(x, (3, 4))
    mx.eval(out)

    inner = []
    for r in range(h - 3 + 1):
        row = []
        for c in range(w - 4 + 1):
            total = sum(values[rr][cc] for rr in range(r, r + 3) for cc in range(c, c + 4))
            row.append(total / 12.0)
        inner.append(row)

    expected = []
    for r in [0, 0, 1, 2, 2]:
        expected.append([inner[r][c] for c in [0, 0, 1, 2, 2, 2]])

    actual = [[float(out[0, r, c, 0]) for c in range(w)] for r in range(h)]
    assert actual == expected
