import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from LTX_2_MLX.videotoolbox.basicvsrpp.upscaler import BasicVsrUpscaler
from LTX_2_MLX.videotoolbox.realbasicvsr.upscaler import RealBasicVsrUpscaler
from LTX_2_MLX.videotoolbox.realviformer.upscaler import RealViformerUpscaler
from LTX_2_MLX.videotoolbox.vsr_blocks import _compute_flows, box3, history_improve_gate

ROOT = Path(__file__).resolve().parents[1]


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
    from LTX_2_MLX.videotoolbox.nafnet.restorer import model_rgb

    rgb = mx.array([[[[-0.25, 0.5, 1.25, 42.0]]]], dtype=mx.float32)
    inp = model_rgb(rgb)
    mx.eval(inp)
    assert inp.shape == (1, 1, 1, 3)
    assert float(mx.min(inp)) == 0.0
    assert float(mx.max(inp)) == 1.0
    assert float(inp[0, 0, 0, 1]) == 0.5


def test_nafnet_restorer_rejects_bad_pool_mode_before_loading_weights():
    from LTX_2_MLX.videotoolbox.nafnet.restorer import resolve_pool_mode

    with pytest.raises(ValueError, match="pool_mode"):
        resolve_pool_mode("gopro32", pool_mode="bogus")


def test_nafnet_pool_auto_matches_reference_variants():
    from LTX_2_MLX.videotoolbox.nafnet.restorer import resolve_pool_mode

    assert resolve_pool_mode("gopro", pool_mode="auto") == "local"
    assert resolve_pool_mode("gopro32", pool_mode="auto") == "local"
    assert resolve_pool_mode("reds", pool_mode="auto") == "local"
    assert resolve_pool_mode("sidd", pool_mode="auto") == "global"
    assert resolve_pool_mode("sidd32", pool_mode="auto") == "global"
    assert resolve_pool_mode("/tmp/nafnet_sidd_width64.safetensors", pool_mode="auto") == "global"

    with pytest.raises(ValueError, match="cannot infer"):
        resolve_pool_mode("/tmp/custom.safetensors", pool_mode="auto")


def test_nafnet_guard_auto_protects_gopro_variants_only():
    from LTX_2_MLX.videotoolbox.nafnet.restorer import resolve_guard_mode

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
    from LTX_2_MLX.videotoolbox.nafnet.restorer import residual_guard_map

    low = mx.full((1, 7, 7, 3), 0.06)
    high = mx.full((1, 7, 7, 3), 0.24)
    low_knee = residual_guard_map(low, guard=0.12)
    high_knee = residual_guard_map(high, guard=0.12)
    mx.eval(low_knee, high_knee)

    assert float(mx.min(low_knee)) == 1.0
    assert abs(float(mx.mean(high_knee)) - 0.25) < 1e-6


def test_nafnet_guard_probe_map_marks_framewide_risk():
    from LTX_2_MLX.videotoolbox.nafnet.restorer import guard_probe_map

    safe = mx.full((1, 9, 9, 3), 0.02)
    risky = mx.full((1, 9, 9, 3), 0.08)
    safe_probe = guard_probe_map(safe, guard=0.12)
    risky_probe = guard_probe_map(risky, guard=0.12)
    mx.eval(safe_probe, risky_probe)

    assert float(mx.max(safe_probe)) == 0.0
    assert 0.2 < float(mx.mean(risky_probe)) < 0.4


def test_nafnet_guard_probe_map_marks_low_amplitude_structure():
    from LTX_2_MLX.videotoolbox.nafnet.restorer import guard_probe_map

    rows = [[0.04 if r % 2 else -0.04 for _c in range(9)] for r in range(9)]
    residual = mx.array([[[[v, v, v] for v in row] for row in rows]], dtype=mx.float32)
    risk = guard_probe_map(residual, guard=0.12)
    mx.eval(risk)

    assert float(mx.max(risk)) > 0.1


def test_nafnet_luma_control_smooths_vertical_luma_not_chroma():
    from LTX_2_MLX.videotoolbox.nafnet.restorer import luma_control_input

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
    assert float(mx.max(mx.abs((control[..., :1] - control[..., 1:2])))) == 0.0


def test_nafnet_control_risk_map_uses_residual_disagreement():
    from LTX_2_MLX.videotoolbox.nafnet.restorer import control_risk_map

    residual = mx.full((1, 9, 9, 3), 0.08)
    control_residual = mx.zeros((1, 9, 9, 3))
    risk = control_risk_map(residual, control_residual, guard=0.12)
    mx.eval(risk)
    assert float(mx.min(risk[:, 2:-2, 2:-2])) > 0.7

    same = control_risk_map(residual, residual, guard=0.12)
    mx.eval(same)
    assert float(mx.max(same)) < 0.5


def test_nafnet_control_guard_framewide_risk_locks_to_control_source():
    from LTX_2_MLX.videotoolbox.nafnet.restorer import NafnetRestorer

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
    from LTX_2_MLX.videotoolbox.nafnet.restorer import NafnetRestorer, _derived_fall_frames

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
    from LTX_2_MLX.videotoolbox.nafnet.restorer import NafnetRestorer

    with pytest.raises(ValueError, match="guard_lockout_frames"):
        NafnetRestorer("gopro32", guard_lockout_frames=-1)
    with pytest.raises(ValueError, match="guard_ramp_frames"):
        NafnetRestorer("gopro32", guard_ramp_frames=-1)
    with pytest.raises(ValueError, match="guard_fall_frames"):
        NafnetRestorer("gopro32", guard_fall_frames=-1)


def test_nafnet_reject_guard_explicit_fall_overrides_derived():
    from LTX_2_MLX.videotoolbox.nafnet.restorer import (
        _REJECT_TRIP_AREA, _REJECT_HARD_CUT_AREA, _local_mag, _derived_fall_frames,
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
    from LTX_2_MLX.videotoolbox.nafnet.restorer import (
        _REJECT_TRIP_AREA, _REJECT_RESUME_AREA_RATIO, _local_mag,
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
    from LTX_2_MLX.videotoolbox.nafnet.restorer import (
        _REJECT_TRIP_AREA, _REJECT_HARD_CUT_AREA, _local_mag,
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
    assert all(b > a for a, b in zip(means, means[1:]))
    assert abs(means[-1] - 0.05) < 1e-7  # gain settled -> exact `out`

    # Catastrophic frame still cuts hard to passthrough.
    catastrophic = mx.full((1, 48, 48, 3), 0.5, dtype=mx.float32)
    guarded = restorer._apply_reject_guard(inp, catastrophic)
    mx.eval(guarded)
    assert float(mx.max(guarded)) == 0.0
    assert restorer._reject_gain == 0.0


def test_nafnet_guard_notice_uses_progress_callback(capsys):
    from LTX_2_MLX.videotoolbox.nafnet.restorer import NafnetRestorer

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
    from LTX_2_MLX.videotoolbox.nafnet import net

    cfg = (32, (1, 1, 1, 28), 1, (1, 1, 1, 1))
    assert net._tlsc_kernel("encoders.0.0.sca.1", cfg) == (384, 384)
    assert net._tlsc_kernel("encoders.1.0.sca.1", cfg) == (192, 192)
    assert net._tlsc_kernel("encoders.3.0.sca.1", cfg) == (48, 48)
    assert net._tlsc_kernel("middle_blks.0.sca.1", cfg) == (24, 24)
    assert net._tlsc_kernel("decoders.0.0.sca.1", cfg) == (48, 48)
    assert net._tlsc_kernel("decoders.3.0.sca.1", cfg) == (384, 384)


def test_nafnet_local_avg_pool_matches_reference_padding():
    from LTX_2_MLX.videotoolbox.nafnet import net

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
