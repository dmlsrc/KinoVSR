import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from kinovsr.basicvsrpp.upscaler import BasicVsrUpscaler
from kinovsr.processors.realviformer.upscaler import RealViformerUpscaler
from kinovsr.realbasicvsr.upscaler import RealBasicVsrUpscaler
from kinovsr.vsr_blocks import _compute_flows, box3, history_improve_gate

ROOT = Path(__file__).resolve().parents[2]


def test_compute_flows_zero_mode_returns_zero_fields():
    frames = [
        mx.zeros((1, 4, 5, 3), dtype=mx.float16),
        mx.ones((1, 4, 5, 3), dtype=mx.float16),
    ]

    flows_forward, flows_backward = _compute_flows(frames, {}, flow_mode="zero")
    mx.eval(flows_forward[0], flows_backward[0])

    assert flows_forward[0].shape == (1, 4, 5, 2)
    assert flows_backward[0].shape == (1, 4, 5, 2)
    assert float(mx.sum(mx.abs(flows_forward[0]))) == 0.0
    assert float(mx.sum(mx.abs(flows_backward[0]))) == 0.0


def test_compute_flows_rejects_unknown_flow_mode():
    frames = [
        mx.zeros((1, 4, 5, 3), dtype=mx.float16),
        mx.ones((1, 4, 5, 3), dtype=mx.float16),
    ]

    with pytest.raises(ValueError, match="unknown flow_mode"):
        _compute_flows(frames, {}, flow_mode="bogus")


@pytest.mark.parametrize(
    ("cls", "name"),
    [
        (BasicVsrUpscaler, "BasicVSR"),
        (RealBasicVsrUpscaler, "RealBasicVSR"),
        (RealViformerUpscaler, "RealViformer"),
    ],
)
def test_upscaler_wrappers_reject_unknown_flow_mode_before_loading_weights(cls, name):
    with pytest.raises(ValueError, match=name):
        cls(flow_mode="bogus")


def test_realviformer_rejects_bad_history_controls_before_loading_weights():
    with pytest.raises(ValueError, match="history_gate"):
        RealViformerUpscaler(history_gate="bogus")
    with pytest.raises(ValueError, match="history_strength"):
        RealViformerUpscaler(history_strength=-0.1)
    with pytest.raises(ValueError, match="history_cleanup"):
        RealViformerUpscaler(history_cleanup=-0.1)
    with pytest.raises(ValueError, match="history_gate_drop"):
        RealViformerUpscaler(history_gate_drop=1.1)
    with pytest.raises(ValueError, match="history_risk_decay"):
        RealViformerUpscaler(history_risk_decay=1.0)
    with pytest.raises(ValueError, match="history_static_cap"):
        RealViformerUpscaler(history_static_cap=-0.1)


def test_realbasicvsr_rejects_bad_history_controls_before_loading_weights():
    with pytest.raises(ValueError, match="history_gate"):
        RealBasicVsrUpscaler(history_gate="bogus")
    with pytest.raises(ValueError, match="history_strength"):
        RealBasicVsrUpscaler(history_strength=-0.1)


def test_basicvsrpp_rejects_bad_history_controls_before_loading_weights():
    with pytest.raises(ValueError, match="history_gate"):
        BasicVsrUpscaler(history_gate="bogus")
    with pytest.raises(ValueError, match="history_strength"):
        BasicVsrUpscaler(history_strength=-0.1)


def test_history_improve_gate_closes_on_static_content():
    # Identical frames + zero flow: warping cannot improve the residual, so the
    # gate must close (this is the anti-etch property).
    mx.random.seed(0)
    curr = mx.random.uniform(shape=(1, 12, 16, 3))
    flow = mx.zeros((1, 12, 16, 2))
    gate = history_improve_gate(curr, curr, flow, mx.float32)
    mx.eval(gate)
    assert gate.shape == (1, 12, 16, 1)
    assert float(mx.max(gate)) == 0.0


def test_history_improve_gate_opens_on_well_tracked_motion():
    # prev shifted by exactly +2 px, flow pointing back at it: the warp
    # reconstructs curr almost exactly while the unwarped residual is large,
    # so interior gate values saturate toward strength.
    mx.random.seed(1)
    prev = mx.random.uniform(shape=(1, 16, 24, 3))
    curr = mx.roll(prev, 2, axis=2)
    flow = mx.concatenate(
        [mx.full((1, 16, 24, 1), -2.0), mx.zeros((1, 16, 24, 1))], axis=-1)
    gate = history_improve_gate(curr, prev, flow, mx.float32, strength=0.75)
    mx.eval(gate)
    interior = gate[:, 2:-2, 4:-4]
    assert float(mx.min(interior)) > 0.7
    assert float(mx.max(gate)) <= 0.75 + 1e-6


def test_box3_replicate_padded_mean():
    vals = mx.arange(3 * 3).reshape(1, 3, 3, 1).astype(mx.float32)
    out = box3(vals)
    mx.eval(out)

    # Replicate padding around:
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # makes the top-left 3x3 neighbourhood [0,0,1; 0,0,1; 3,3,4].
    assert abs(float(out[0, 0, 0, 0]) - (12.0 / 9.0)) < 1e-6
    assert abs(float(out[0, 1, 1, 0]) - 4.0) < 1e-6


def _holistic_test_upscaler(**kwargs):
    up = RealViformerUpscaler.__new__(RealViformerUpscaler)
    up._history_strength = kwargs.get("history_strength", 1.0)
    up._history_cleanup = kwargs.get("history_cleanup", 0.25)
    up._history_gate_drop = kwargs.get("history_gate_drop", 0.85)
    up._history_risk_decay = kwargs.get("history_risk_decay", 0.0)
    up._history_static_cap = kwargs.get("history_static_cap", 0.0)
    up._risk = None
    return up


def test_realviformer_holistic_policy_is_gentle_on_static_content():
    mx.random.seed(2)
    up = _holistic_test_upscaler()
    curr = mx.random.uniform(shape=(1, 12, 16, 3))
    flow = mx.zeros((1, 12, 16, 2))
    warped = mx.random.uniform(shape=(1, 12, 16, 48))

    cleaned, gate = up._holistic_history_policy(curr, curr, flow, warped, mx.float32)
    mx.eval(cleaned, gate)

    assert gate.shape == (1, 12, 16, 1)
    assert cleaned.shape == warped.shape
    assert 0.14 < float(mx.min(gate)) < 0.16
    assert 0.14 < float(mx.max(gate)) < 0.16
    assert float(mx.mean(mx.abs(cleaned - warped))) > 0.0


def test_realviformer_holistic_policy_opens_on_well_tracked_motion():
    mx.random.seed(3)
    up = _holistic_test_upscaler()
    prev = mx.random.uniform(shape=(1, 16, 24, 3))
    curr = mx.roll(prev, 2, axis=2)
    flow = mx.concatenate(
        [mx.full((1, 16, 24, 1), -2.0), mx.zeros((1, 16, 24, 1))], axis=-1)
    warped = mx.random.uniform(shape=(1, 16, 24, 48))

    _, gate = up._holistic_history_policy(curr, prev, flow, warped, mx.float32)
    mx.eval(gate)

    interior = gate[:, 2:-2, 4:-4]
    assert float(mx.median(interior)) > 0.95


def test_realviformer_pad4_matches_reference_left_top_reflect():
    vals = mx.arange(5 * 6).reshape(1, 5, 6, 1)

    padded, pad_top, pad_left = RealViformerUpscaler._pad4(vals)
    mx.eval(padded)

    assert (pad_top, pad_left) == (3, 2)
    assert padded.shape == (1, 8, 8, 1)

    row_indices = [3, 2, 1, 0, 1, 2, 3, 4]
    col_indices = [2, 1, 0, 1, 2, 3, 4, 5]
    expected = [
        [r * 6 + c for c in col_indices]
        for r in row_indices
    ]
    assert padded[0, :, :, 0].tolist() == expected


def test_pth_converter_refuses_ambiguous_params_dict(tmp_path):
    torch = pytest.importorskip("torch")
    ckpt = tmp_path / "ambiguous.pth"
    out = tmp_path / "ambiguous.safetensors"
    torch.save(
        {
            "params": {"w": torch.ones(1)},
            "params_ema": {"w": torch.zeros(1)},
        },
        ckpt,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "pth_to_safetensors.py"),
            str(ckpt),
            "-o",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "checkpoint carries BOTH 'params' and 'params_ema'" in result.stderr
    assert not out.exists()


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


def test_safmn_bicubic_up_matches_torch_reference():
    from kinovsr.processors.safmn.net import _bicubic_up

    # Reference computed once with torch F.interpolate(scale_factor=2,
    # mode="bicubic", align_corners=False) on this exact input.
    vals = [v * 0.7 - 2.0 for v in range(12)]
    x = mx.array(vals, dtype=mx.float32).reshape(1, 3, 4, 1)
    expected = [
        [-2.3691406, -2.1613283, -1.8277344, -1.3874999, -1.1031251, -0.66289067, -0.32929698, -0.12148452],
        [-1.5378907, -1.3300781, -0.99648434, -0.55624998, -0.27187502, 0.16835934, 0.50195312, 0.70976561],
        [-0.20351566, 0.0042968662, 0.33789068, 0.77812499, 1.0625000, 1.5027344, 1.8363283, 2.0441408],
        [1.6558595, 1.8636718, 2.1972656, 2.6374998, 2.9218750, 3.3621094, 3.6957033, 3.9035158],
        [2.9902344, 3.1980467, 3.5316405, 3.9718747, 4.2562499, 4.6964846, 5.0300775, 5.2378907],
        [3.8214846, 4.0292969, 4.3628907, 4.8031244, 5.0874996, 5.5277343, 5.8613276, 6.0691404],
    ]
    out = _bicubic_up(x, 2)
    mx.eval(out)
    assert out.shape == (1, 6, 8, 1)
    for r in range(6):
        for c in range(8):
            assert abs(float(out[0, r, c, 0]) - expected[r][c]) < 2e-6, (r, c)


def test_safmn_bicubic_up_reproduces_constants_and_ramps():
    from kinovsr.processors.safmn.net import _bicubic_up

    # Weights sum to 1 -> constants reproduce exactly, including at the
    # replicate-padded borders.
    const = mx.full((1, 4, 4, 2), 0.37, dtype=mx.float32)
    out = _bicubic_up(const, 4)
    mx.eval(out)
    assert float(mx.max(mx.abs(out - 0.37))) < 1e-6

    # On a linear ramp the A=-0.75 cubic kernel has a fixed alternating phase
    # bias of -/+ 3/64 at r=2 (it does not reproduce linears exactly; torch
    # produces these same values). Interior rows, away from tap clamping:
    ramp = mx.broadcast_to(mx.arange(8, dtype=mx.float32)[None, :, None, None], (1, 8, 4, 1))
    out = _bicubic_up(ramp, 2)
    mx.eval(out)
    inner = out[0, 4:12, 2:6, 0]  # away from H borders
    linear = 1.75 + mx.arange(8, dtype=mx.float32) * 0.5
    bias = mx.array([-1.0, 1.0] * 4, dtype=mx.float32) * (3.0 / 64.0)
    expect = linear + bias
    assert float(mx.max(mx.abs(inner - expect[:, None]))) < 1e-5


def test_safmn_safm_mode_inferred_from_filename():
    from kinovsr.processors.safmn.net import _VARIANTS, _safm_mode_for

    assert _safm_mode_for("safmn_purescale_x4.safetensors") == "fixed"
    assert _safm_mode_for("/a/b/Safmn_PureScale_sharper_x2.safetensors") == "fixed"
    assert _safm_mode_for("safmn_l_real_lsdir_x4.safetensors") == "stock"
    assert _safm_mode_for("light_safmnpp.safetensors") == "stock"
    for token in ("purescale", "purescale2x", "purescale2x-sharp"):
        assert token in _VARIANTS
        assert _safm_mode_for(_VARIANTS[token]) == "fixed"
    for token in ("light", "real", "real2x"):
        assert _safm_mode_for(_VARIANTS[token]) == "stock"


def test_safmn_config_carries_safm_mode_and_trained_upsampler():
    from kinovsr.processors.safmn.net import _config

    p = {
        "to_feat.weight": mx.zeros((128, 3, 3, 3)),
        "feats.0.norm1.weight": mx.zeros((128,)),
        "feats.1.norm1.weight": mx.zeros((128,)),
        "to_img.0.weight": mx.zeros((48, 3, 3, 128)),
    }
    assert _config(p) == ("real", 128, 2, 4, "stock", "nearest", 0.0)
    p["__safm_mode__"] = "fixed"
    assert _config(p) == ("real", 128, 2, 4, "fixed", "bicubic", 0.0)


def test_safmn_upscaler_validates_args_before_loading_weights():
    from kinovsr.processors.safmn.upscaler import SafmnUpscaler

    with pytest.raises(ValueError, match="safm_up"):
        SafmnUpscaler(safm_up="bogus")
    with pytest.raises(ValueError, match="pool_clamp"):
        SafmnUpscaler(pool_clamp=-1.0)


def test_safmn_pool_clamp_touches_only_interior_outliers():
    from kinovsr.processors.safmn.net import _pool_clamp

    mx.random.seed(4)
    s = mx.random.normal(shape=(1, 12, 16, 4)) * 0.1
    spiked = s[:]
    spiked[0, 6, 8, 2] = 25.0    # interior spike, far beyond k sigma
    spiked[0, 0, 5, 3] = 25.0    # frame-boundary spike (exempt margin)
    out = _pool_clamp(spiked, 4.0)
    mx.eval(out)

    # The interior spike is bounded...
    assert float(out[0, 6, 8, 2]) < 25.0
    # ...to roughly mu + 4 sigma of its channel's INTERIOR statistics...
    core = spiked.astype(mx.float32)[:, 1:-1, 1:-1, 2]
    mu = float(mx.mean(core))
    sd = float(mx.sqrt(mx.mean((core - mu) ** 2)))
    assert abs(float(out[0, 6, 8, 2]) - (mu + 4.0 * sd)) < 1e-4
    # ...an untouched channel passes through numerically unchanged...
    assert float(mx.max(mx.abs(out[..., 0] - spiked[..., 0]))) < 1e-6
    # ...and the frame-boundary margin is exempt (synthetic borders saturate
    # there; clamping them re-engages texture hallucination).
    assert float(out[0, 0, 5, 3]) == 25.0

    # Degenerate pooled maps (no interior) pass through whole.
    tiny = mx.full((1, 2, 2, 1), 9.0)
    assert float(mx.max(mx.abs(_pool_clamp(tiny, 4.0) - tiny))) == 0.0


def test_edge_sanitize_parse_spec():
    from kinovsr.edge_sanitize import parse_edges_spec

    assert parse_edges_spec("0,1,0,0") == (0, 1, 0, 0)
    assert parse_edges_spec(" 2, 3 ,4,5 ") == (2, 3, 4, 5)
    with pytest.raises(ValueError, match="four integers"):
        parse_edges_spec("1,2,3")
    with pytest.raises(ValueError, match=">= 0"):
        parse_edges_spec("1,-2,3,4")


def _sanitize_samples(junk_bottom=False, bar_rows=0):
    mx.random.seed(7)
    out = []
    for _ in range(5):
        fr = mx.clip(mx.random.uniform(shape=(48, 64, 3)) * 0.5 + 0.35, 0, 1)
        if junk_bottom:
            fr[-1:] = fr[-2:-1] * 0.3   # dark junk line, tracks content
        if bar_rows:
            fr[:bar_rows] = 0.02        # constant near-black bar at the top
            fr[-bar_rows:] = 0.02
        out.append(fr)
    return out


def test_edge_sanitize_detects_dark_junk_row():
    from kinovsr.edge_sanitize import detect_junk_edges

    edges, notices = detect_junk_edges(_sanitize_samples(junk_bottom=True))
    assert edges == (0, 1, 0, 0)
    assert notices == []


def test_edge_sanitize_clean_content_untouched():
    from kinovsr.edge_sanitize import detect_junk_edges

    edges, notices = detect_junk_edges(_sanitize_samples())
    assert edges == (0, 0, 0, 0)


def test_edge_sanitize_letterbox_reported_not_filled():
    from kinovsr.edge_sanitize import detect_junk_edges

    edges, notices = detect_junk_edges(_sanitize_samples(bar_rows=12))
    assert edges == (0, 0, 0, 0)
    assert any("letterbox-class" in n for n in notices)


def test_edge_sanitize_blank_samples_yield_no_detection():
    from kinovsr.edge_sanitize import detect_junk_edges

    blanks = [mx.full((32, 32, 3), 0.02) for _ in range(5)]
    edges, notices = detect_junk_edges(blanks)
    assert edges == (0, 0, 0, 0)
    assert any("too few" in n for n in notices)


def test_edge_sanitize_rgb_replaces_bands_and_keeps_dims():
    from kinovsr.edge_sanitize import sanitize_rgb

    fr = mx.broadcast_to(mx.arange(10, dtype=mx.float32)[:, None, None], (10, 8, 3))
    out = sanitize_rgb(fr, (2, 1, 0, 0))
    mx.eval(out)
    assert out.shape == fr.shape
    assert float(mx.max(mx.abs(out[0] - fr[2]))) == 0.0   # top rows <- first interior
    assert float(mx.max(mx.abs(out[1] - fr[2]))) == 0.0
    assert float(mx.max(mx.abs(out[-1] - fr[-2]))) == 0.0  # bottom row <- neighbor
    assert float(mx.max(mx.abs(out[2:-1] - fr[2:-1]))) == 0.0  # interior untouched

    with pytest.raises(ValueError, match="interior"):
        sanitize_rgb(fr, (5, 5, 0, 0))


def test_edge_sanitize_restore_borders_composites_original():
    from kinovsr.edge_sanitize import restore_borders

    mx.random.seed(9)
    src = mx.random.uniform(shape=(8, 10, 3))
    out = mx.random.uniform(shape=(16, 20, 3))       # processed at 2x
    res = restore_borders(out, src, (0, 1, 2, 0), feather=0)
    mx.eval(res)

    assert res.shape == out.shape
    # bottom band (2 out rows) = nearest-2x of src's last row
    assert float(mx.max(mx.abs(res[14] - mx.repeat(src[7], 2, axis=0)))) < 1e-6
    assert float(mx.max(mx.abs(res[15] - res[14]))) < 1e-6
    # left band (4 out cols) = nearest-2x of src's first 2 cols
    assert float(mx.max(mx.abs(res[0, 0] - src[0, 0]))) < 1e-6
    assert float(mx.max(mx.abs(res[0, 3] - src[0, 1]))) < 1e-6
    # interior untouched
    assert float(mx.max(mx.abs(res[2:14, 4:] - out[2:14, 4:]))) < 1e-6

    # uint8 sources normalize to [0,1]
    src8 = mx.full((8, 10, 3), 255, dtype=mx.uint8)
    res = restore_borders(out, src8, (1, 0, 0, 0), feather=0)
    assert abs(float(mx.mean(res[0])) - 1.0) < 1e-6

    with pytest.raises(ValueError, match="integer multiple"):
        restore_borders(mx.zeros((15, 20, 3)), src, (1, 0, 0, 0), feather=0)


def test_edge_sanitize_restore_feather_ramps_into_content():
    from kinovsr.edge_sanitize import restore_borders

    src = mx.zeros((8, 6, 3), dtype=mx.float32)
    out = mx.ones((16, 12, 3), dtype=mx.float32)   # processed at 2x
    res = restore_borders(out, src, (1, 0, 0, 0), feather=2)
    mx.eval(res)

    col = [float(res[r, 5, 0]) for r in range(8)]
    # band rows (2 at 2x) fully restored to source (0), then a linear ramp
    # across the 4-row feather zone, then untouched processed content (1).
    assert col[0] == 0.0 and col[1] == 0.0
    expect = [0.125, 0.375, 0.625, 0.875]
    for got, want in zip(col[2:6], expect, strict=True):
        assert abs(got - want) < 1e-6
    assert col[6] == 1.0 and col[7] == 1.0


def test_edge_sanitize_detect_bars_letterbox_and_pillarbox():
    from kinovsr.edge_sanitize import detect_bars

    mx.random.seed(11)
    letter, pillar = [], []
    for _ in range(5):
        fr = mx.clip(mx.random.uniform(shape=(48, 64, 3)) * 0.5 + 0.35, 0, 1)
        lb = fr[:]
        lb[:12] = 0.02
        lb[-12:] = 0.02
        letter.append(lb)
        pb_ = fr[:]
        pb_[:, :22] = 0.02
        pb_[:, -22:] = 0.02
        pillar.append(pb_)
    assert detect_bars(letter) == (12, 12, 0, 0)
    assert detect_bars(pillar) == (0, 0, 22, 22)

    # even-ization: an 11-row top bar leaves 37 rows -> bottom bumped by 1
    odd = []
    for _ in range(5):
        fr = mx.clip(mx.random.uniform(shape=(48, 64, 3)) * 0.5 + 0.35, 0, 1)
        fr[:11] = 0.02
        odd.append(fr)
    assert detect_bars(odd) == (11, 1, 0, 0)

    # clean content: no bars
    clean = [mx.clip(mx.random.uniform(shape=(48, 64, 3)) * 0.5 + 0.35, 0, 1)
             for _ in range(5)]
    assert detect_bars(clean) == (0, 0, 0, 0)


def test_edge_sanitize_crop_rgb():
    from kinovsr.edge_sanitize import crop_rgb

    fr = mx.arange(6 * 8 * 3, dtype=mx.float32).reshape(6, 8, 3)
    out = crop_rgb(fr, (1, 2, 3, 0))
    assert out.shape == (3, 5, 3)
    assert float(mx.max(mx.abs(out - fr[1:4, 3:]))) == 0.0
    batched = crop_rgb(fr[None], (1, 2, 3, 0))
    assert batched.shape == (1, 3, 5, 3)


def test_edge_sanitize_compute_aspect_crop():
    from kinovsr.edge_sanitize import compute_aspect_crop

    # 16:9 window on a 4:3 frame: full width, centered vertically.
    assert compute_aspect_crop(640, 480, 16, 9) == (60, 60, 0, 0)
    # 9:16 portrait extract from 16:9: nearest even width, centered.
    t, b, left, r = compute_aspect_crop(1920, 1080, 9, 16)
    assert (left + r) + (1920 - left - r) == 1920
    w = 1920 - left - r
    h = 1080 - t - b
    assert w % 2 == 0 and h % 2 == 0
    assert abs(w / h - 9 / 16) < 0.01
    assert abs(left - r) <= 2  # centered
    # offsets shift and clamp
    t2, b2, l2, r2 = compute_aspect_crop(1920, 1080, 9, 16, dx=-10000)
    assert l2 == 0 and r2 == left + r
    # same aspect as the frame: no crop
    assert compute_aspect_crop(640, 480, 4, 3) == (0, 0, 0, 0)
    with pytest.raises(ValueError, match="positive"):
        compute_aspect_crop(640, 480, 0, 9)


def test_edge_sanitize_aspect_crop_anchors():
    from kinovsr.edge_sanitize import compute_aspect_crop

    # 16:9 on 4:3 (640x480 -> 640x360): the vertical slack is 120.
    assert compute_aspect_crop(640, 480, 16, 9, anchor="top") == (0, 120, 0, 0)
    assert compute_aspect_crop(640, 480, 16, 9, anchor="bottom") == (120, 0, 0, 0)
    assert compute_aspect_crop(640, 480, 16, 9, anchor="center") == (60, 60, 0, 0)
    # corners on a window with slack in both axes: 1:1 on 640x480 -> 480x480.
    assert compute_aspect_crop(640, 480, 1, 1, anchor="top-left") == (0, 0, 0, 160)
    assert compute_aspect_crop(640, 480, 1, 1, anchor="bottom-right") == (0, 0, 160, 0)
    assert compute_aspect_crop(640, 480, 1, 1, anchor="right") == (0, 0, 160, 0)
    # offset nudges from the anchor and clamps.
    assert compute_aspect_crop(640, 480, 16, 9, anchor="bottom", dy=-20) == (100, 20, 0, 0)
    assert compute_aspect_crop(640, 480, 16, 9, anchor="bottom", dy=50) == (120, 0, 0, 0)
    with pytest.raises(ValueError, match="anchor"):
        compute_aspect_crop(640, 480, 16, 9, anchor="middle-ish")


def test_edge_sanitize_aspect_crop_picks_closest_even_fit():
    from kinovsr.edge_sanitize import compute_aspect_crop

    # 16:9 in storage px on 348x288: even boxes can only approximate;
    # 344x194 (-0.26%) beats 346x194 (+0.32%) and 348x194 (+0.90%).
    assert compute_aspect_crop(348, 288, 16, 9) == (47, 47, 2, 2)
    # Display 16:9 on 128:117 anamorphic pixels = storage 13:8 (16*117:9*128
    # reduced): full width, 214 rows.
    assert compute_aspect_crop(348, 288, 16 * 117, 9 * 128) == (37, 37, 0, 0)


def test_lanczos_resample_plan_properties():
    from kinovsr.vsr_blocks import make_lanczos_plan, resample_width

    # identity when sizes match
    plan = make_lanczos_plan(12, 12)
    x = mx.random.uniform(shape=(4, 12, 3))
    assert float(mx.max(mx.abs(resample_width(x, plan) - x))) < 1e-6

    # constants preserved exactly on up AND down (weights sum to 1)
    up = make_lanczos_plan(87, 95)      # the 128:117 PAR ratio reduced
    dn = make_lanczos_plan(95, 87)
    const = mx.full((3, 87, 2), 0.4)
    out = resample_width(const, up)
    assert out.shape == (3, 95, 2)
    assert float(mx.max(mx.abs(out - 0.4))) < 1e-5
    constd = mx.full((3, 95, 2), 0.4)
    outd = resample_width(constd, dn)
    assert outd.shape == (3, 87, 2)
    assert float(mx.max(mx.abs(outd - 0.4))) < 1e-5

    # downscale kernel widens for antialiasing (more taps than upscale)
    assert dn[0].shape[0] > up[0].shape[0]

    # a linear ramp is reproduced closely in the interior on upscale
    ramp = mx.broadcast_to(mx.arange(87, dtype=mx.float32)[None, :, None], (2, 87, 1))
    r = resample_width(ramp, up)
    j = mx.arange(95, dtype=mx.float32)
    expect = (j + 0.5) * (87.0 / 95.0) - 0.5
    err = mx.abs(r[0, :, 0] - expect)[8:-8]
    assert float(mx.max(err)) < 0.02

    # batched 4D input works too
    x4 = mx.random.uniform(shape=(2, 4, 87, 3))
    assert resample_width(x4, up).shape == (2, 4, 95, 3)


def test_to_rgb_batch_clips_decode_overshoot():
    from kinovsr.upscaler_base import to_rgb_batch

    # decoded RGBAHalf carries legal YUV->RGB overshoot; every learned
    # upscaler entry must clip it (measured 56x confetti-speck area unclipped)
    fr = mx.array([[[-0.14, 0.5, 1.25, 1.0]] * 4] * 3).reshape(3, 4, 4)
    out = to_rgb_batch(fr)
    assert out.shape == (1, 3, 4, 3)
    assert float(mx.min(out)) >= 0.0
    assert float(mx.max(out)) <= 1.0
    # in-range values untouched
    assert abs(float(out[0, 0, 0, 1]) - 0.5) < 1e-7


def test_source_range_resolve_override():
    from kinovsr.media import color

    src = {"primaries": None, "transfer": None, "matrix": None,
           "full_range": False, "tagged": False}
    # auto trusts the container flag
    assert color.resolve(src, "auto", "auto")[3] is False
    assert color.resolve(dict(src, full_range=True), "auto", "auto")[3] is True
    # forcing overrides the flag in both directions
    assert color.resolve(src, "auto", "full")[3] is True
    assert color.resolve(dict(src, full_range=True), "auto", "video")[3] is False
    # range override composes with a colorimetry override
    resolved = color.resolve(src, "bt601", "full")
    assert resolved[3] is True
    assert "range=full" in color.describe(resolved)
