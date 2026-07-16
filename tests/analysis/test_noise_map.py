"""Weight-free tests for the spatial noise-map estimator (noise_map.py).

Pins the estimator's contract: recovery of known sigma fields, immunity to static
texture (the property that separates a temporal estimator from spatial ones), the
motion cap, unit conversion, and the tracker's gain/EMA behavior.
"""
import math

import mlx.core as mx

from kinovsr.analysis.noise import (
    NoiseMapTracker,
    PulseGain,
    analyze_noise,
    classify_noise_analysis,
    estimate_sigma_map,
)

H, W, T = 128, 160, 8


def _content():
    ys = mx.linspace(0, 1, H).reshape(H, 1)
    xs = mx.linspace(0, 1, W).reshape(1, W)
    img = 0.35 + 0.3 * ys + 0.15 * mx.sin(4 * math.pi * xs)
    return mx.clip(mx.broadcast_to(img[..., None], (H, W, 3)), 0, 1)


def _noisy_clip(sigma_field):
    mx.random.seed(0)
    base = _content()
    return [mx.clip(base + mx.random.normal(shape=base.shape) * sigma_field[..., None], 0, 1)
            for _ in range(T)]


def test_recovers_constant_sigma():
    sig = mx.full((H, W), 0.05)
    est = estimate_sigma_map(_noisy_clip(sig))
    assert est.shape == (H, W, 1) and est.dtype == mx.float32
    med = float(mx.sort(est.reshape(-1))[H * W // 2])
    assert abs(med - 0.05) / 0.05 < 0.15   # within 15%


def test_recovers_gradient_and_orientation():
    xs = mx.linspace(0, 1, W).reshape(1, W)
    sig = mx.broadcast_to(0.01 + 0.07 * xs, (H, W))
    est = estimate_sigma_map(_noisy_clip(sig))[:, :, 0]
    left = float(mx.mean(est[:, : W // 4]))
    right = float(mx.mean(est[:, -W // 4:]))
    assert right > 2.0 * left              # ramp direction and spread survive


def test_static_texture_does_not_register():
    # a strong checkerboard is pure signal; frame diffs cancel it. A spatial
    # estimator would read it as huge noise; this one must stay at the true 0.01.
    ys = mx.arange(H).reshape(H, 1)
    xs = mx.arange(W).reshape(1, W)
    checker = ((ys // 2 + xs // 2) % 2).astype(mx.float32) * 0.5 + 0.25
    tex = mx.broadcast_to(checker[..., None], (H, W, 3))
    mx.random.seed(1)
    clip = [mx.clip(tex + 0.01 * mx.random.normal(shape=tex.shape), 0, 1) for _ in range(T)]
    est = estimate_sigma_map(clip)
    p95 = float(mx.sort(est.reshape(-1))[int(0.95 * (H * W - 1))])
    assert p95 < 0.02                       # NOT the ~0.5 a spatial reading would give


def test_motion_is_capped():
    # DISPLACEMENT motion (a drifting block, constant fill): lag-2 diffs double
    # over lag-1, so the flicker bypass stays closed and the luma-model cap
    # holds the map near the true noise. (A photometric STROBE is temporally
    # independent and now intentionally reads as noise -- see the bypass note.)
    mx.random.seed(2)
    base = _content()
    yy = mx.arange(H).reshape(H, 1)
    xx = mx.arange(W).reshape(1, W)
    clip = []
    for t in range(T):
        x0 = 10 + 10 * t
        m = ((yy >= 40) & (yy < 90) & (xx >= x0) & (xx < x0 + 40))[..., None]
        f = mx.where(m, 0.85, base)                    # constant fill, moving position
        clip.append(mx.clip(f + 0.03 * mx.random.normal(shape=f.shape), 0, 1))
    est = estimate_sigma_map(clip)
    assert float(mx.max(est)) < 0.12


def test_dense_textured_motion_is_conservative_in_strict_mode():
    # Full-frame textured translation has dense frame differences but no
    # injected noise. Its lag ratio is ~1, so a ratio-only flicker bypass used
    # to let it through as a huge sigma map.
    mx.random.seed(42)
    base = mx.clip(mx.random.uniform(shape=(H, W, 3)) * 0.6 + 0.2, 0, 1)
    clip = [mx.roll(base, shift=t * 3, axis=1) for t in range(T)]
    strict = estimate_sigma_map(clip, motion_cap="strict", masking=1.0)
    off = estimate_sigma_map(clip, motion_cap="off", masking=1.0)
    p95_strict = float(mx.sort(strict.reshape(-1))[int(0.95 * (H * W - 1))])
    p95_off = float(mx.sort(off.reshape(-1))[int(0.95 * (H * W - 1))])
    assert p95_strict < 0.12
    assert p95_off > 1.7 * p95_strict


def test_heterogeneous_dense_motion_gets_stricter_cap():
    # Real motion contamination is usually spatially uneven (some blocks are
    # texture/occlusion heavy, others are not). That should get a stricter cap
    # than spatially uniform dense noise.
    mx.random.seed(44)
    left = mx.random.uniform(shape=(H, W // 2, 3)) * 0.6 + 0.2
    right = mx.random.uniform(shape=(H, W - W // 2, 3)) * 0.12 + 0.44
    base = mx.clip(mx.concatenate([left, right], axis=1), 0, 1)
    clip = [mx.roll(base, shift=t * 3, axis=1) for t in range(T)]
    strict = estimate_sigma_map(clip, motion_cap="strict", masking=1.0)
    off = estimate_sigma_map(clip, motion_cap="off", masking=1.0)
    p95_strict = float(mx.sort(strict.reshape(-1))[int(0.95 * (H * W - 1))])
    p95_off = float(mx.sort(off.reshape(-1))[int(0.95 * (H * W - 1))])
    assert p95_strict < 0.08
    assert p95_off > 1.7 * p95_strict


def test_dense_noise_on_smooth_content_is_not_motion_capped():
    sig = mx.full((H, W), 0.10)
    est = estimate_sigma_map(_noisy_clip(sig), motion_cap="strict", masking=1.0)
    med = float(mx.sort(est.reshape(-1))[H * W // 2])
    assert med > 0.08


def test_source_wide_motion_contamination_is_conservative_in_strict_mode():
    # A lightly blurred texture drifting one pixel per frame has lag2/lag1 > 1
    # and no stable pixels. It is source-wide motion, not denoiseable noise.
    mx.random.seed(301)
    x = mx.random.uniform(shape=(H, W, 1))
    p = mx.concatenate([x[:1], x, x[-1:]], axis=0)
    p = mx.concatenate([p[:, :1], p, p[:, -1:]], axis=1)
    acc = mx.zeros_like(x)
    for i in range(3):
        for j in range(3):
            acc = acc + p[i:i + H, j:j + W]
    base = mx.broadcast_to(acc / 9.0, (H, W, 3))
    clip = [mx.roll(base, shift=t, axis=1) for t in range(T)]
    strict = estimate_sigma_map(clip, motion_cap="strict", masking=1.0)
    off = estimate_sigma_map(clip, motion_cap="off", masking=1.0)
    p95_strict = float(mx.sort(strict.reshape(-1))[int(0.95 * (H * W - 1))])
    p95_off = float(mx.sort(off.reshape(-1))[int(0.95 * (H * W - 1))])
    assert p95_strict < 0.08
    assert p95_off > 1.5 * p95_strict


def test_strobe_reads_as_noise():
    # a photometrically strobing region (no displacement) IS temporal noise for
    # a denoiser: the lag-ratio bypass must let it through the motion cap
    mx.random.seed(3)
    base = _content()
    yy = mx.arange(H).reshape(H, 1)
    xx = mx.arange(W).reshape(1, W)
    m = ((yy >= 40) & (yy < 90) & (xx >= 50) & (xx < 100))[..., None]
    clip = [mx.clip(mx.where(m, 0.6 if t % 2 == 0 else 0.4, base)
                    + 0.01 * mx.random.normal(shape=base.shape), 0, 1) for t in range(T)]
    est = estimate_sigma_map(clip)
    assert float(mx.max(est)) > 0.05      # the strobing region registers high


def test_sparse_flicker_registers():
    # mosquito-style compression flicker: 10% of pixels flash per frame, the
    # rest are temporally identical (skip blocks). Plainly visible, yet ANY
    # quantile of the diff samples reads ~0 (the zero spike swallows it). The
    # RMS statistic must read the energy: amp * sqrt(density).
    mx.random.seed(12)
    base = _content()
    amp, density = 0.06, 0.10
    clip = []
    for _ in range(T):
        m = (mx.random.uniform(shape=(H, W, 1)) < density).astype(mx.float32)
        clip.append(mx.clip(base + m * amp * mx.random.normal(shape=base.shape), 0, 1))
    est = estimate_sigma_map(clip)
    med = float(mx.sort(est.reshape(-1))[H * W // 2])
    # tail-RMS reads the flicker's AMPLITUDE scale (what conditioning must
    # cover to suppress it), well above the plain energy amp*sqrt(density)
    assert med > 0.35 * amp                   # amplitude scale (quantiles read ~0)
    assert med < 0.95 * amp                   # bounded below the raw amplitude


def test_probe_classifier_resolves_dense_motion_via_mc_floor():
    from kinovsr.analysis.noise import analyze_noise

    # rolling white-noise texture: statistically identical to per-frame noise
    # for every unaligned statistic (the old classifier said "ambiguous"),
    # but phase correlation aligns the roll away -- the MC floor resolves it
    # as MOTION with a confidently low noise floor
    mx.random.seed(43)
    base = mx.clip(mx.random.uniform(shape=(H, W, 3)) * 0.6 + 0.2, 0, 1)
    clip = [mx.roll(base, shift=t * 3, axis=1) for t in range(T)]
    stats = analyze_noise(clip)
    assert stats["mc_sigma"] < 0.02
    diag = classify_noise_analysis(stats)
    assert "dense motion, low noise floor" in diag["labels"]
    assert "motion/noise ambiguous" not in diag["labels"]
    assert "dense sensor noise" not in diag["labels"]


def test_probe_classifier_flags_source_wide_motion_contamination():
    mx.random.seed(301)
    x = mx.random.uniform(shape=(H, W, 1))
    p = mx.concatenate([x[:1], x, x[-1:]], axis=0)
    p = mx.concatenate([p[:, :1], p, p[:, -1:]], axis=1)
    acc = mx.zeros_like(x)
    for i in range(3):
        for j in range(3):
            acc = acc + p[i:i + H, j:j + W]
    base = mx.broadcast_to(acc / 9.0, (H, W, 3))
    clip = [mx.roll(base, shift=t, axis=1) for t in range(T)]
    diag = classify_noise_analysis(analyze_noise(clip))
    assert "source-wide motion contamination" in diag["labels"]
    assert any("motion map" in warning for warning in diag["warnings"])


def test_probe_classifier_flags_row_coherent_scanlines():
    base = _content()
    ys = mx.arange(H).reshape(H, 1)
    clip = []
    for t in range(T):
        phase = (t * 3) % 11
        lines = (((ys + phase) % 11) == 0).astype(mx.float32)
        lines = mx.broadcast_to(lines[:, :, None], (H, W, 3))
        clip.append(mx.clip(base - 0.035 * lines, 0, 1))
    stats = analyze_noise(clip)
    diag = classify_noise_analysis(stats)
    assert stats["row_periodicity"] > 0.5
    assert "row-coherent scanline flicker" in diag["labels"]
    assert any("row" in warning for warning in diag["warnings"])


def test_too_few_frames_returns_none():
    assert estimate_sigma_map([_content()]) is None
    assert estimate_sigma_map([]) is None


def test_tracker_gain_and_ema():
    sig = mx.full((H, W), 0.04)
    clip = _noisy_clip(sig)
    tr = NoiseMapTracker(gain=2.0, ema=0.5)
    m1 = tr.update(clip)
    med1 = float(mx.sort(m1.reshape(-1))[H * W // 2])
    assert abs(med1 - 2.0 * 0.04) / 0.08 < 0.2       # gain applied
    # feed a much noisier window: EMA must land between the two estimates
    mx.random.seed(3)
    base = _content()
    clip2 = [mx.clip(base + 0.10 * mx.random.normal(shape=base.shape), 0, 1) for _ in range(T)]
    m2 = tr.update(clip2)
    med2 = float(mx.sort(m2.reshape(-1))[H * W // 2]) / 2.0   # un-gain
    assert 0.05 < med2 < 0.09                         # between 0.04 and ~0.10
    # reset drops state; update with too-few frames returns None
    tr.reset()
    assert tr.current() is None
    assert tr.update([base]) is None


def test_tracker_rejects_bad_params():
    for bad in ({"gain": 0.0}, {"gain": -1.0}, {"ema": 0.0}, {"ema": 1.5}):
        try:
            NoiseMapTracker(**bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"NoiseMapTracker accepted {bad}"


def test_tracker_holds_map_on_degenerate_window():
    # a tiny (e.g. 6-frame gop-align tail) window must not crater the map: once
    # a map exists, windows below min_frames reuse it instead of updating.
    sig = mx.full((H, W), 0.05)
    clip = _noisy_clip(sig)
    tr = NoiseMapTracker(min_frames=8)
    m1 = tr.update(clip)
    med1 = float(mx.sort(m1.reshape(-1))[H * W // 2])
    clean = [_content() for _ in range(4)]          # 4 near-noiseless frames
    m2 = tr.update(clean)                            # below min_frames: held
    med2 = float(mx.sort(m2.reshape(-1))[H * W // 2])
    assert abs(med2 - med1) < 1e-6
    # but a first estimate from a tiny window is still allowed (better than none)
    tr2 = NoiseMapTracker(min_frames=8)
    assert tr2.update(clip[:4]) is not None


def test_select_runs_spread_and_adjacent():
    from kinovsr.analysis.noise.estimate import _select_runs
    # short windows: one consecutive run, unchanged behavior
    assert _select_runs(8, 12) == [list(range(8))]
    # long windows: runs of consecutive indices spread across the whole span
    runs = _select_runs(96, 12)
    assert all(r == list(range(r[0], r[0] + 3)) for r in runs)   # adjacent within runs
    assert runs[0][0] == 0 and runs[-1][-1] == 95                # covers both ends
    assert len(runs) >= 3


def test_long_window_sampling_sees_late_noise():
    # sigma jumps 0.01 -> 0.06 halfway through a 60-frame window. Head-only
    # sampling would read ~0.01; spread runs must see the noisy second half.
    mx.random.seed(4)
    base = _content()
    clip = [mx.clip(base + (0.01 if t < 30 else 0.06) * mx.random.normal(shape=base.shape), 0, 1)
            for t in range(60)]
    est_spread = estimate_sigma_map(clip)
    est_head = estimate_sigma_map(clip[:12])       # what head-only sampling saw
    med_s = float(mx.sort(est_spread.reshape(-1))[H * W // 2])
    med_h = float(mx.sort(est_head.reshape(-1))[H * W // 2])
    assert med_h < 0.02                            # head-only is blind to the jump
    assert med_s > med_h * 1.5                     # spread sampling is not


# blockiness tests run at the validated geometry (32px tiles need room; the
# estimator is meant for real frame sizes, not tiny buffers)
_BH, _BW = 256, 384


def _bcontent():
    ys = mx.linspace(0, 1, _BH).reshape(_BH, 1)
    xs = mx.linspace(0, 1, _BW).reshape(1, _BW)
    img = (0.4 + 0.25 * mx.sin(5 * math.pi * xs) * mx.cos(4 * math.pi * ys)
           + 0.2 * ys + 0.05 * mx.sin(23 * math.pi * xs))
    return mx.clip(mx.broadcast_to(img[..., None], (_BH, _BW, 3)), 0, 1)


def _blockify(img, size=8):
    h, w, c = img.shape
    hb, wb = h // size * size, w // size * size
    core = img[:hb, :wb].reshape(hb // size, size, wb // size, size, c)
    m = mx.mean(core, axis=(1, 3), keepdims=True)
    core = mx.broadcast_to(m, core.shape).reshape(hb, wb, c)
    return mx.concatenate([mx.concatenate([core, img[:hb, wb:]], axis=1), img[hb:]], axis=0)


def test_blockiness_map_localizes_and_rejects():
    from kinovsr.analysis.noise import estimate_blockiness_map
    mx.random.seed(10)
    base = _bcontent()
    # half blocked, half clean: mask lights the blocked half only
    half = mx.concatenate([_blockify(base)[:, : _BW // 2], base[:, _BW // 2:]], axis=1)
    frames = [mx.clip(half + 0.004 * mx.random.normal(shape=half.shape), 0, 1)
              for _ in range(3)]
    m = estimate_blockiness_map(frames)
    assert m.shape == (_BH, _BW, 1)
    # the mask is deliberately smooth (coarse 3x3 smoothing twice + a tile-wide
    # blur gives a ~2.5-tile soft skirt), so judge the clean half beyond it.
    # Safety cases assert in RAW excess units (severity_scale=1.0) so the
    # default severity calibration can move without weakening the guarantees.
    blocked = mx.sort(m[:, : _BW // 2 - 16, 0].reshape(-1))
    assert float(blocked[blocked.shape[0] // 2]) > 0.5      # hard blocking saturates
    raw = estimate_blockiness_map(frames, severity_scale=1.0)
    clean = mx.sort(raw[:, _BW // 2 + 80:, 0].reshape(-1))
    assert float(clean[int(0.95 * (clean.shape[0] - 1))]) < 1.5e-3
    # strong periodic texture (elevates BOTH the detected and the null phase,
    # so the per-tile null-phase subtraction cancels it) -> ~0 raw excess
    ys = mx.arange(_BH).reshape(_BH, 1)
    xs = mx.arange(_BW).reshape(1, _BW)
    checker = mx.broadcast_to(
        (((ys // 2 + xs // 2) % 2).astype(mx.float32) * 0.5 + 0.25)[..., None],
        (_BH, _BW, 3))
    assert float(mx.max(estimate_blockiness_map([checker] * 2, severity_scale=1.0))) < 1e-3
    # 1D content bars (not a 2D grid): min-over-lines + geometric mean -> ~0
    bars = mx.where(mx.broadcast_to(((xs % 40) < 2).reshape(1, _BW, 1), (_BH, _BW, 1)), 0.9, base)
    bars = mx.where(mx.broadcast_to(((ys % 56) < 2).reshape(_BH, 1, 1), (_BH, _BW, 1)), 0.1, bars)
    assert float(mx.max(estimate_blockiness_map([bars] * 2, severity_scale=1.0))) < 3e-3


def test_blockiness_grid_phase_offset_detected():
    from kinovsr.analysis.noise import estimate_blockiness_map
    mx.random.seed(11)
    base = _bcontent()
    # blocking on a grid shifted by 3px must still be found
    x = mx.concatenate([base[:, 3:], base[:, :3]], axis=1)
    x = mx.concatenate([x[3:], x[:3]], axis=0)
    b = _blockify(x)
    b = mx.concatenate([b[-3:], b[:-3]], axis=0)
    b = mx.concatenate([b[:, -3:], b[:, :-3]], axis=1)
    frames = [mx.clip(b + 0.004 * mx.random.normal(shape=b.shape), 0, 1) for _ in range(3)]
    m = estimate_blockiness_map(frames)
    mid = mx.sort(m[16:-16, 16:-16, 0].reshape(-1))
    assert float(mid[mid.shape[0] // 2]) > 0.3


def test_blockiness_global_saturation_is_tempered():
    from kinovsr.analysis.noise import estimate_blockiness_map
    mx.random.seed(13)
    base = _bcontent()
    blocked = _blockify(base)
    frames = [mx.clip(blocked + 0.004 * mx.random.normal(shape=blocked.shape), 0, 1)
              for _ in range(3)]
    m = estimate_blockiness_map(frames)
    flat = mx.sort(m[:, :, 0].reshape(-1))
    med = float(flat[flat.shape[0] // 2])
    p95 = float(flat[int(0.95 * (flat.shape[0] - 1))])
    # Whole-frame blocking should not become a silent "strength 1 everywhere"
    # mask; reserve high wetness for local excess above the frame baseline.
    assert 0.25 < med < 0.75
    assert p95 < 0.75


def test_pulse_gain_tracks_noise_pulse():
    # settled sigma 0.03; frames 24-27 carry sigma 0.09 (an I-frame grain
    # refresh). Settled gains ~1.0; pulse frames must rise and clamp at hi.
    mx.random.seed(6)
    base = _content()
    pg = PulseGain(lo=0.6, hi=1.8, min_history=8)
    gains = []
    for t in range(32):
        s = 0.09 if 24 <= t < 28 else 0.03
        f = mx.clip(base + s * mx.random.normal(shape=base.shape), 0, 1)
        gains.append(pg.update(f))
    settled = gains[10:23]
    assert all(abs(g - 1.0) < 0.25 for g in settled)
    # the diff spanning the sigma jump and the pulse frames read high
    assert max(gains[24:28]) > 1.5
    # new_segment restarts the chain neutrally
    assert pg.update(base, new_segment=True) == 1.0


def test_pulse_gain_ignores_localized_motion():
    # PulseGain should measure a global quiet-block noise pulse, not subject
    # motion. A moving square covers too few blocks to move the low quantile.
    base = _content()
    yy = mx.arange(H).reshape(H, 1)
    xx = mx.arange(W).reshape(1, W)
    pg = PulseGain(lo=0.6, hi=1.8, min_history=8)
    gains = []
    for t in range(32):
        x0 = 4 + 3 * t
        mask = ((yy >= 42) & (yy < 86) & (xx >= x0) & (xx < x0 + 36))[..., None]
        frame = mx.where(mask, 0.9, base)
        gains.append(pg.update(frame))
    assert max(gains[10:]) < 1.2


def test_pulse_gain_neutral_until_history():
    mx.random.seed(7)
    base = _content()
    pg = PulseGain(min_history=8)
    for _t in range(7):
        f = mx.clip(base + 0.05 * mx.random.normal(shape=base.shape), 0, 1)
        assert pg.update(f) == 1.0     # not enough history yet


def test_pulse_gain_rejects_bad_bounds():
    for bad in ({"lo": 0.0}, {"lo": 1.2}, {"hi": 0.9}):
        try:
            PulseGain(**bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"PulseGain accepted {bad}"


def test_pulse_robust_map_damps_global_noise_pulse():
    mx.random.seed(16)
    base = _content()
    clip = []
    for t in range(T):
        sigma = 0.11 if t == T // 2 else 0.03
        clip.append(mx.clip(base + sigma * mx.random.normal(shape=base.shape), 0, 1))
    plain = estimate_sigma_map(clip, motion_cap="off", smooth=False)
    robust = estimate_sigma_map(clip, motion_cap="off", pulse_robust=True, smooth=False)
    plain_med = float(mx.sort(plain.reshape(-1))[H * W // 2])
    robust_med = float(mx.sort(robust.reshape(-1))[H * W // 2])
    assert plain_med > 1.25 * robust_med
    assert 0.02 < robust_med < 0.055


def test_fastdvd_pulse_emits_all_and_varies():
    # integration: pulse on, sigma jumps mid-stream; all frames out in order and
    # the logged gains actually respond.
    from kinovsr.processors.fastdvdnet import FastDvdDenoiser
    mx.random.seed(8)
    base = mx.clip(_content()[:48, :80, :], 0, 1)
    den = FastDvdDenoiser(strength=0.3, pulse=PulseGain(min_history=6))
    em = []
    for t in range(40):
        s = 0.08 if 30 <= t < 34 else 0.02
        f = mx.clip(base + s * mx.random.normal(shape=base.shape), 0, 1)
        em += den.feed(f, token=t)
    gains = list(den._pulse_log)
    em += den.flush()
    assert [t for _, t in sorted(em, key=lambda e: e[1])] == list(range(40))
    assert max(gains[30:34]) > 1.4 and abs(gains[20] - 1.0) < 0.3


def test_mc_sigma_plane_and_pulse():
    # mc's residual gate takes the map as a per-pixel sigma plane (in residual
    # units) and the pulse as a per-frame scale. Needs VTOpticalFlow; skip where
    # the flow self-test refuses (unsupported device / too-small buffers).
    import pytest

    from kinovsr.processors.mc import McTemporalDenoiser
    mx.random.seed(9)
    h, w = 240, 320
    base = mx.clip(mx.full((h, w, 3), 0.45) + 0.15 * mx.random.uniform(shape=(h, w, 3)), 0, 1)
    xs = mx.linspace(0, 1, w).reshape(1, w, 1)
    try:
        den = McTemporalDenoiser(w, h, strength=0.8, sigma=0.06,
                                 noise_map=NoiseMapTracker(),
                                 pulse=PulseGain(min_history=6))
    except (RuntimeError, SystemExit) as e:
        pytest.skip(f"VTOpticalFlow unavailable: {e}")
    try:
        for t in range(20):
            s = (0.01 + 0.07 * xs) * (2.2 if 14 <= t < 17 else 1.0)
            f = mx.clip(base + s * mx.random.normal(shape=(h, w, 3)), 0, 1)
            out = den.denoise(f)
            assert bool(mx.all(mx.isfinite(out)))
        nm = den.last_noise_map
        assert nm is not None and nm.shape == (h, w, 1)
        # spatial: the sigma plane follows the gradient noise field
        assert float(mx.mean(nm[:, -40:, 0])) > 2.0 * float(mx.mean(nm[:, :40, 0]))
        # temporal: the pulse burst registers, settled frames stay neutral
        log = den._pulse_log
        assert max(log[14:17]) > 1.4 and abs(log[11] - 1.0) < 0.3
    finally:
        den.close()


def test_fastdvd_streaming_refresh_adapts():
    # noise jumps 0.01 -> 0.08 at frame 20 of a 60-frame stream. With refresh on,
    # the held map must climb toward the new level; with refresh off it must not.
    from kinovsr.processors.fastdvdnet import FastDvdDenoiser
    mx.random.seed(5)
    base = mx.clip(_content()[:48, :80, :], 0, 1)
    frames = [mx.clip(base + (0.01 if t < 20 else 0.08) * mx.random.normal(shape=base.shape), 0, 1)
              for t in range(60)]

    def run(refresh):
        den = FastDvdDenoiser(strength=0.3, noise_map=NoiseMapTracker(ema=0.7),
                              map_refresh=refresh)
        em = []
        for i, f in enumerate(frames):
            em += den.feed(f, token=i)
        last = den.last_noise_map
        em += den.flush()
        assert [t for _, t in sorted(em, key=lambda e: e[1])] == list(range(60))
        return float(mx.sort(last.reshape(-1))[last.shape[0] * last.shape[1] // 2])

    med_off = run(0)       # estimate-once: stuck at the quiet start
    med_on = run(16)       # refreshing: adapts toward 0.08
    assert med_off < 0.025
    assert med_on > 2.0 * med_off


def _panning_clip(noise_sigma):
    # 65% smoothed texture panning 2 px/frame over a flat lower band; every
    # frame carries a small encode-like wobble so static_fraction ~ 0 (as in
    # real footage). Extra width feeds the pan.
    mx.random.seed(77)
    tex = mx.random.uniform(shape=(H, W + 2 * T))
    k = mx.ones((1, 5, 5, 1)) / 25.0
    tex = mx.conv2d(tex[None, :, :, None], k, padding=2)[0, :, :, 0]
    split = int(H * 0.65)
    base_sigma = max(noise_sigma, 0.004)
    clip = []
    for t in range(T):
        f = tex[:, t * 2:t * 2 + W]
        f = mx.concatenate([f[:split], mx.full((H - split, W), 0.45)], axis=0)
        rgb = mx.broadcast_to(f[..., None], (H, W, 3))
        clip.append(mx.clip(rgb + mx.random.normal(shape=(H, W, 3)) * base_sigma, 0, 1))
    return clip


def test_flat_floor_separates_clean_pan_from_noisy_pan():
    # The aperture-gated flat floor: motion cannot flicker gradient-free
    # pixels, so a clean pan must read LOW while the same pan with real dense
    # noise reads the noise level. Before the flat floor both read ~0.06
    # (motion-dominated), which made auto denoise soften clean footage.
    clean = estimate_sigma_map(_panning_clip(0.0), motion_cap="strict",
                               floor_mode="flat")
    noisy = estimate_sigma_map(_panning_clip(0.05), motion_cap="strict",
                               floor_mode="flat")
    med_c = float(mx.sort(clean.reshape(-1))[H * W // 2])
    med_n = float(mx.sort(noisy.reshape(-1))[H * W // 2])
    assert med_c < 0.035                      # clean pan stays near the wobble
    assert 0.035 < med_n < 0.095              # dense noise is preserved
    assert med_n > 1.8 * med_c                # and the two separate cleanly


def test_probe_flat_floor_flags_dense_noise_not_motion():
    clean_stats = analyze_noise(_panning_clip(0.0))
    noisy_stats = analyze_noise(_panning_clip(0.05))
    # whitened flat floor: near zero on the clean pan, near sigma on the noisy
    assert clean_stats["flat_sigma"] < 0.02
    assert noisy_stats["flat_sigma"] > 0.03
    noisy_diag = classify_noise_analysis(noisy_stats)
    assert "dense sensor noise" in noisy_diag["labels"]
    clean_diag = classify_noise_analysis(clean_stats)
    assert "dense sensor noise" not in clean_diag["labels"]


def _full_texture_pan_clip(noise_sigma):
    # texture EVERYWHERE (no flat pixels), panning 2 px/frame: the flat floor
    # has nothing to measure here, but phase correlation aligns the pan away.
    mx.random.seed(99)
    tex = mx.random.uniform(shape=(H, W + 2 * T))
    k = mx.ones((1, 5, 5, 1)) / 25.0
    tex = mx.conv2d(tex[None, :, :, None], k, padding=2)[0, :, :, 0]
    base_sigma = max(noise_sigma, 0.004)
    clip = []
    for t in range(T):
        f = tex[:, t * 2:t * 2 + W]
        rgb = mx.broadcast_to(f[..., None], (H, W, 3))
        clip.append(mx.clip(rgb + mx.random.normal(shape=(H, W, 3)) * base_sigma, 0, 1))
    return clip


def test_mc_floor_separates_clean_pan_from_noisy_pan_without_flat_pixels():
    clean = estimate_sigma_map(_full_texture_pan_clip(0.0),
                               motion_cap="strict", floor_mode="mc")
    noisy = estimate_sigma_map(_full_texture_pan_clip(0.05),
                               motion_cap="strict", floor_mode="mc")
    med_c = float(mx.sort(clean.reshape(-1))[H * W // 2])
    med_n = float(mx.sort(noisy.reshape(-1))[H * W // 2])
    assert med_c < 0.035                      # aligned pan leaves only wobble
    assert 0.035 < med_n < 0.095              # dense noise survives alignment
    assert med_n > 1.8 * med_c


def test_mc_floor_matches_flat_floor_on_flat_band_pan():
    # where the flat floor CAN measure (pan + flat band), the two floor modes
    # must agree on the level; they share every downstream stage.
    for sigma in (0.0, 0.05):
        flat = estimate_sigma_map(_panning_clip(sigma), motion_cap="strict",
                                  floor_mode="flat")
        mc = estimate_sigma_map(_panning_clip(sigma), motion_cap="strict",
                                floor_mode="mc")
        med_f = float(mx.sort(flat.reshape(-1))[H * W // 2])
        med_m = float(mx.sort(mc.reshape(-1))[H * W // 2])
        assert abs(med_m - med_f) < max(0.015, 0.6 * med_f)


def test_mc_floor_rejects_bad_mode():
    import pytest
    with pytest.raises(ValueError):
        estimate_sigma_map(_panning_clip(0.0), floor_mode="phase")


def test_probe_mc_floor_resolves_full_texture_pan():
    # no flat pixels anywhere: the flat floor cannot decide, but the MC floor
    # (the map's default) aligns the pan away -- the probe verdict must follow
    # the shipping estimator, not just the flat diagnostic
    clean_stats = analyze_noise(_full_texture_pan_clip(0.0))
    noisy_stats = analyze_noise(_full_texture_pan_clip(0.05))
    assert clean_stats["mc_sigma"] < 0.02
    assert noisy_stats["mc_sigma"] > 0.03
    noisy_diag = classify_noise_analysis(noisy_stats)
    assert "dense sensor noise" in noisy_diag["labels"]
    assert any("motion-compensated" in s for s in noisy_diag["suggestions"])
    clean_diag = classify_noise_analysis(clean_stats)
    assert "dense sensor noise" not in clean_diag["labels"]


def _bilinear_up(img, s=1.5):
    h, w = int(img.shape[0]), int(img.shape[1])
    oh, ow = int(h * s), int(w * s)
    fy = mx.arange(oh).astype(mx.float32) / s
    fx = mx.arange(ow).astype(mx.float32) / s
    y0 = mx.minimum(fy.astype(mx.int32), h - 2)
    x0 = mx.minimum(fx.astype(mx.int32), w - 2)
    wy = (fy - y0.astype(mx.float32)).reshape(oh, 1, 1)
    wx = (fx - x0.astype(mx.float32)).reshape(1, ow, 1)
    r0 = mx.take(img, y0, axis=0)
    r1 = mx.take(img, y0 + 1, axis=0)
    g0 = mx.take(r0, x0, axis=1) * (1 - wx) + mx.take(r0, x0 + 1, axis=1) * wx
    g1 = mx.take(r1, x0, axis=1) * (1 - wx) + mx.take(r1, x0 + 1, axis=1) * wx
    return g0 * (1 - wy) + g1 * wy


def test_blockiness_auto_detects_resized_grid():
    # compressed-then-resized: an 8-grid upscaled 1.5x lives at period 12,
    # invisible to the 8-locked legacy detector but found by period="auto"
    from kinovsr.analysis.noise import estimate_blockiness_map

    mx.random.seed(21)
    base = _bcontent()
    blocked = _bilinear_up(_blockify(base))
    clean = _bilinear_up(base)
    bframes = [mx.clip(blocked + 0.003 * mx.random.normal(shape=blocked.shape), 0, 1)
               for _ in range(3)]
    cframes = [mx.clip(clean + 0.003 * mx.random.normal(shape=clean.shape), 0, 1)
               for _ in range(3)]
    m_auto = estimate_blockiness_map(bframes, period="auto", severity_scale=1.0)
    m_legacy = estimate_blockiness_map(bframes, period=8, severity_scale=1.0)
    m_clean = estimate_blockiness_map(cframes, period="auto", severity_scale=1.0)
    med_auto = float(mx.sort(m_auto.reshape(-1))[int(m_auto.size) // 2])
    med_legacy = float(mx.sort(m_legacy.reshape(-1))[int(m_legacy.size) // 2])
    p95_clean = float(mx.sort(m_clean.reshape(-1))[int(0.95 * (int(m_clean.size) - 1))])
    assert med_auto > 3e-3                    # resized grid registers
    assert med_auto > 3.0 * max(med_legacy, 1e-6)   # ...where 8-locked missed it
    assert p95_clean < 1.5e-3                 # resized CLEAN content stays quiet


def test_row_spectrum_detects_interlace_residue():
    # alternating-row flicker (comb at period 2) -- below the old lag-3
    # autocorrelation floor, squarely in the spectrum's band
    mx.random.seed(31)
    base = _content()
    comb = mx.broadcast_to(((mx.arange(H) % 2).astype(mx.float32) * 2 - 1)
                           .reshape(H, 1, 1), (H, W, 3))
    clip = [mx.clip(base + (0.02 if t % 2 == 0 else -0.02) * comb
                    + 0.004 * mx.random.normal(shape=base.shape), 0, 1)
            for t in range(T)]
    stats = analyze_noise(clip)
    assert stats["row_interlace"] >= 4.0
    diag = classify_noise_analysis(stats)
    assert "interlace/field residue" in diag["labels"]


def test_row_spectrum_detects_jumping_scanlines():
    # dark 1px lines every 11 rows, phase jumping per frame, 30% dropout --
    # the analog-corpus scanline recipe
    mx.random.seed(33)
    base = _content()
    clip = []
    for t in range(T):
        phase = (t * 5) % 11
        rows = ((mx.arange(H) + phase) % 11 < 1).astype(mx.float32)
        keep = (mx.random.uniform(shape=(H,)) >= 0.3).astype(mx.float32)
        lines = (rows * keep).reshape(H, 1, 1)
        clip.append(mx.clip(base - 0.035 * lines
                            + 0.004 * mx.random.normal(shape=base.shape), 0, 1))
    stats = analyze_noise(clip)
    assert abs(stats["row_period_px"] - 11.0) <= 0.6
    assert stats["row_periodicity"] >= 0.5


def test_static_row_banding_detected_without_temporal_signal():
    # banding BAKED into every frame identically: zero temporal diff, so only
    # the static row spectrum can see it
    mx.random.seed(35)
    base = _content()
    band = (0.02 * mx.cos(2.0 * 3.141592653589793 * mx.arange(H).astype(mx.float32) / 7.0)
            ).reshape(H, 1, 1)
    clip = [mx.clip(base + band + 0.003 * mx.random.normal(shape=base.shape), 0, 1)
            for _ in range(T)]
    stats = analyze_noise(clip)
    assert abs(stats["static_row_period_px"] - 7.0) <= 0.5
    assert stats["static_row_periodicity"] >= 0.5
    diag = classify_noise_analysis(stats)
    assert "static row banding" in diag["labels"]


def test_static_grain_floor_separates_from_temporal_noise():
    # grain field added IDENTICALLY to every frame (static) + small temporal
    # wobble: the mean-frame spatial floor must read the grain while the
    # temporal (mc) floor reads only the wobble
    mx.random.seed(37)
    base = _content()
    grain = 0.03 * mx.random.normal(shape=(H, W, 3))
    clip = [mx.clip(base + grain + 0.004 * mx.random.normal(shape=base.shape), 0, 1)
            for _ in range(T)]
    stats = analyze_noise(clip)
    assert stats["spatial_sigma"] > 0.018
    assert stats["static_grain_sigma"] > 0.015
    assert stats["mc_sigma"] < 0.012                  # temporal floor: wobble only
    diag = classify_noise_analysis(stats)
    assert "static/structured grain" in diag["labels"]
    # clean control: no static grain reported on grain-free content
    clean = [mx.clip(base + 0.004 * mx.random.normal(shape=base.shape), 0, 1)
             for _ in range(T)]
    cstats = analyze_noise(clean)
    assert cstats["static_grain_sigma"] < 0.012


def test_mc_floor_handles_frames_smaller_than_the_block():
    # The MC floor's 32px block must shrink to fit small frames instead
    # of building a zero-block grid (crashed on any dimension under 32).
    from kinovsr.analysis.noise.estimate import estimate_sigma_map

    mx.random.seed(0)
    for h, w in ((24, 64), (64, 24), (24, 24), (8, 8)):
        frames = [mx.random.uniform(shape=(h, w, 3)).astype(mx.float32)
                  for _ in range(6)]
        result = estimate_sigma_map(frames)
        sig = result[0] if isinstance(result, tuple) else result
        assert sig.shape == (h, w, 1)
        assert bool(mx.all(mx.isfinite(sig)))


def test_pulse_gain_sync_veto_blocks_mid_gop_boosts():
    # The same sigma spike, once inside the post-sync pulse zone and once
    # deep mid-GOP: raw-stream positions say only the first is an I-frame
    # grain refresh; the second (content/motion the low-quantile statistic
    # did not reject) must not boost denoise conditioning.
    mx.random.seed(6)
    base = _content()

    def run(spike_at, since_sync_of):
        pg = PulseGain(lo=0.6, hi=1.8, min_history=8)
        gains = []
        for t in range(40):
            s = 0.09 if spike_at <= t < spike_at + 3 else 0.03
            f = mx.clip(base + s * mx.random.normal(shape=base.shape), 0, 1)
            gains.append(pg.update(f, since_sync=since_sync_of(t)))
        return gains

    # spike lands at ordinals 0..2 of a GOP: boost allowed
    at_sync = run(30, lambda t: max(t - 30, 0) if t >= 30 else t + 2)
    assert max(at_sync[30:33]) > 1.5
    # identical spike at ordinals 10..12: vetoed to neutral
    mid_gop = run(30, lambda t: t - 20)
    assert all(g <= 1.0 for g in mid_gop[30:33])


def test_pulse_gain_sync_veto_keeps_suppression_and_blind_mode():
    mx.random.seed(6)
    base = _content()
    pg = PulseGain(lo=0.6, hi=1.8, min_history=8)
    gains = []
    for t in range(40):
        # settled 0.06; frames 30+ drop to 0.02 (prediction settled)
        s = 0.02 if t >= 30 else 0.06
        f = mx.clip(base + s * mx.random.normal(shape=base.shape), 0, 1)
        gains.append(pg.update(f, since_sync=25))  # deep mid-GOP
    # suppression below 1.0 stays allowed at any phase
    assert min(gains[31:36]) < 0.8

    # blind mode (since_sync=None) is unchanged: boosts allowed anywhere
    pg2 = PulseGain(lo=0.6, hi=1.8, min_history=8)
    gains2 = []
    for t in range(36):
        s = 0.09 if t >= 30 else 0.03
        f = mx.clip(base + s * mx.random.normal(shape=base.shape), 0, 1)
        gains2.append(pg2.update(f))
    assert max(gains2[30:34]) > 1.5


def test_source_since_sync_duck_typing():
    from types import SimpleNamespace

    from kinovsr.analysis.noise.track import source_since_sync

    assert source_since_sync(None) is None
    assert source_since_sync(7) is None
    assert source_since_sync(SimpleNamespace(source=None)) is None
    assert source_since_sync(
        SimpleNamespace(source=SimpleNamespace(gop_ordinal=4))) == 4
    assert source_since_sync(
        SimpleNamespace(source=SimpleNamespace(gop_ordinal=None))) is None
