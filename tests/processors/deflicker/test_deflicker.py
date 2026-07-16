"""Weight-free tests for the verified-static state deflicker (deflicker.py).

Pins the contract: quantization-state flicker on verified-static content is
collapsed to the dwell-weighted state mixture; moving content and one-sided
true content changes pass through bit-identical; the operator never mixes
across space.
"""
import mlx.core as mx

from kinovsr.processors.deflicker import StaticStateDeflicker

H, W, T = 192, 256, 24


def _base():
    mx.random.seed(7)
    x = mx.random.uniform(shape=(H, W))
    k = mx.ones((1, 7, 7, 1)) / 49.0
    x = mx.conv2d(x[None, :, :, None], k, padding=3)[0, :, :, 0]
    x = mx.clip(x * 0.5 + 0.25, 0, 1)
    return mx.broadcast_to(x[..., None], (H, W, 3))


def _run(clip, **kw):
    s = StaticStateDeflicker(**kw)
    outs = []
    for i, f in enumerate(clip):
        outs += s.feed(f, token=i)
    outs += s.flush()
    assert [tok for _o, tok in outs] == list(range(len(clip)))
    return [o for o, _tok in outs]


def _blocky(shape_hw, scale=8):
    # codec junk is block-structured: DCT blocks flip together, they are not
    # per-pixel white (white fields at texture amplitude would also defeat
    # the phase-correlation verification, which real junk does not)
    h, w = shape_hw
    r = mx.random.uniform(shape=(h // scale, w // scale, 1))
    r = mx.broadcast_to(r[:, None, :, None, :], (h // scale, scale, w // scale, scale, 1))
    return r.reshape(h, w, 1)


def test_collapses_state_flicker_on_static_content():
    # GOP-pulse model: 8px blocks flip between base+d and base-d with a
    # 3-frame cadence; the truth is base. Integration must both kill the
    # temporal flicker and land closer to the truth than either state.
    # Integration-machinery test: same-sign field, illumination veto off
    # (sign-mixed flips at texture-comparable energy decorrelate the
    # phase-correlation verification itself on this synthetic -- real junk
    # rides on real texture; the veto has its own dedicated test).
    mx.random.seed(11)
    base = _base()
    d = 0.02 * (1.0 + _blocky((H, W)))    # steps 2d in [0.04, 0.08] < band
    clip = [mx.clip(base + (d if (t // 3) % 2 == 0 else -d), 0, 1)
            for t in range(T)]
    outs = _run(clip, illum_veto=False)
    lo, hi = 8, T - 8
    err_in = sum(float(mx.mean(mx.abs(clip[t] - base))) for t in range(lo, hi))
    err_out = sum(float(mx.mean(mx.abs(outs[t] - base))) for t in range(lo, hi))
    fl_in = sum(float(mx.mean(mx.abs(clip[t] - clip[t - 1]))) for t in range(lo, hi))
    fl_out = sum(float(mx.mean(mx.abs(outs[t] - outs[t - 1]))) for t in range(lo, hi))
    assert err_out < 0.4 * err_in        # integrated toward the truth
    assert fl_out < 0.25 * fl_in         # flicker collapsed


def test_panning_motion_is_bit_identical():
    mx.random.seed(13)
    tex = mx.random.uniform(shape=(H, W + 2 * T))
    k = mx.ones((1, 5, 5, 1)) / 25.0
    tex = mx.conv2d(tex[None, :, :, None], k, padding=2)[0, :, :, 0]
    clip = [mx.broadcast_to(mx.clip(tex[:, 2 * t:2 * t + W], 0, 1)[..., None],
                            (H, W, 3)) for t in range(T)]
    outs = _run(clip)
    for t in range(T):
        assert float(mx.max(mx.abs(outs[t] - clip[t]))) < 1e-6


def test_true_content_change_is_one_sided_and_preserved():
    # an object appears at t=10 and STAYS: at every frame near the onset one
    # temporal side has no in-band support, so the gate refuses; away from
    # the onset the integration is over identical values (no-op)
    base = _base()
    yy = mx.arange(H).reshape(H, 1, 1)
    xx = mx.arange(W).reshape(1, W, 1)
    obj = ((yy >= 90) & (yy < 104) & (xx >= 120) & (xx < 136)).astype(mx.float32)
    clip = [mx.clip(base + (0.3 * obj if t >= 10 else 0.0), 0, 1)
            for t in range(T)]
    outs = _run(clip)
    for t in range(T):
        diff = float(mx.max(mx.abs((outs[t] - clip[t]) * obj)))
        assert diff < 5e-3, f"object region altered by {diff} at t={t}"


def test_one_frame_flash_removed():
    # deflash's old job, subsumed: sparse 1-frame block flashes outside the
    # state band are excluded from the mixture entirely
    mx.random.seed(17)
    base = _base()
    m = (_blocky((H, W)) < 0.08).astype(mx.float32)
    clip = [mx.clip(base + (0.15 * m if t == 12 else 0.0), 0, 1)
            for t in range(T)]
    outs = _run(clip)
    before = float(mx.mean(mx.abs(clip[12] - base)))
    after = float(mx.mean(mx.abs(outs[12] - base)))
    assert after < 0.1 * before


def test_lighting_ramp_is_passthrough():
    # multiplicative lighting/exposure ramp on static content: monotone
    # smooth temporal profile -> the oscillatory gate refuses, at ANY band
    # (v6 without the gate flattened ramps and printed square seams along
    # the 16px validity cells)
    base = _base()
    tt = mx.arange(T).astype(mx.float32) / (T - 1)
    clip = [mx.clip(base * (0.88 + 0.20 * float(g)), 0, 1) for g in tt]
    for band in (0.10, 0.25):
        outs = _run(clip, band=band)
        worst = max(float(mx.max(mx.abs(o - c))) for o, c in zip(outs, clip, strict=True))
        assert worst < 0.012, f"ramp altered by {worst} at band {band}"


def test_flicker_on_lighting_ramp_fixed_ramp_kept():
    # blocky state flicker riding ON a lighting ramp: the flicker must
    # collapse toward the ramped truth while the ramp itself is preserved
    mx.random.seed(23)
    base = _base()
    d = 0.02 * (1.0 + _blocky((H, W)))
    tt = mx.arange(T).astype(mx.float32) / (T - 1)
    truth = [mx.clip(base * (0.88 + 0.20 * float(g)), 0, 1) for g in tt]
    clip = [mx.clip(truth[t] + (d if (t // 3) % 2 == 0 else -d), 0, 1)
            for t in range(T)]
    outs = _run(clip, illum_veto=False)
    lo, hi = 8, T - 8
    err_in = sum(float(mx.mean(mx.abs(clip[t] - truth[t]))) for t in range(lo, hi))
    err_out = sum(float(mx.mean(mx.abs(outs[t] - truth[t]))) for t in range(lo, hi))
    assert err_out < 0.5 * err_in


def test_spatially_coherent_flicker_is_vetoed():
    # oscillating ILLUMINATION (blinking signage, AGC pumping) is per-pixel
    # indistinguishable from codec pulsing, but its correction field is
    # locally same-signed -- the coherence veto must refuse it, or the
    # flattening follows the 16px verification cells and prints flickering
    # boxes (user-reported failure)
    base = _base()
    d = mx.full((H, W, 1), 0.04)
    clip = [mx.clip(base * (1.0 + (d if (t // 3) % 2 == 0 else -d)), 0, 1)
            for t in range(T)]
    outs = _run(clip)
    worst = max(float(mx.max(mx.abs(o - c))) for o, c in zip(outs, clip, strict=True))
    assert worst < 0.012, f"coherent oscillation altered by {worst}"


def test_latency_and_flush():
    base = _base()
    s = StaticStateDeflicker()
    n_out = 0
    for _ in range(12):
        n_out += len(s.feed(base))
    assert n_out == 4                     # window=8 frames of lookahead
    n_out += len(s.flush())
    assert n_out == 12
    # gate stats must survive the flush (the harness reads them after):
    # identical static frames fully verify, and applied is ~zero
    st = s.stats()
    assert st["verified"] > 0.9
    assert st["applied"] < 1e-5


def test_jitter_compensation_recovers_shaky_static_scene():
    # global integer camera jitter (max 2px) on a static flickering scene:
    # without compensation, far pairs fail verification and the kill dies;
    # with it, the median-shift residual verifies and samples align
    mx.random.seed(29)
    base = _base()
    # same-sign field with the veto disabled: this test isolates the JITTER
    # machinery. (Sign-mixed flips at texture-comparable energy decorrelate
    # the displaced-pair phase correlation itself -- a synthetic artifact;
    # the veto has its own dedicated test below.)
    d = 0.02 * (1.0 + _blocky((H, W)))
    shifts = [(0, 0), (1, 0), (2, 1), (1, 2), (0, 1), (-1, 0), (-2, -1),
              (-1, -2)]
    clip = []
    for t_i in range(T):
        f = mx.clip(base + (d if (t_i // 3) % 2 == 0 else -d), 0, 1)
        sy, sx = shifts[t_i % len(shifts)]
        clip.append(mx.roll(mx.roll(f, sy, axis=0), sx, axis=1))
    # the output is intentionally NOT stabilized (only samples are
    # aligned), so adjacent-frame diffs keep the content displacement of
    # the jitter itself; the kill is judged on the flicker component above
    # the jittered-CLEAN floor, and on error to the jittered truth
    truth = []
    for t_i in range(T):
        sy, sx = shifts[t_i % len(shifts)]
        truth.append(mx.roll(mx.roll(base, sy, axis=0), sx, axis=1))
    lo, hi = 8, T - 8
    c = (slice(16, H - 16), slice(16, W - 16))   # avoid border strips

    def flick(frames):
        return sum(float(mx.mean(mx.abs(frames[t][c] - frames[t - 1][c])))
                   for t in range(lo, hi))

    def err(frames):
        return sum(float(mx.mean(mx.abs(frames[t][c] - truth[t][c])))
                   for t in range(lo, hi))

    floor = flick(truth)
    fl_in = flick(clip) - floor
    outs_off = _run(clip, illum_veto=False)
    outs_on = _run(clip, jitter=True, illum_veto=False)
    assert flick(outs_off) - floor > 0.7 * fl_in   # jitter defeats strict path
    assert flick(outs_on) - floor < 0.35 * fl_in   # compensation restores kill
    assert err(outs_on) < 0.5 * err(clip)          # and lands nearer the truth


def test_pan_stays_passthrough_with_jitter_on():
    # a real 2px/frame pan: pairs beyond distance 1 exceed the jitter cap,
    # so too few samples verify and nothing fires
    mx.random.seed(13)
    tex = mx.random.uniform(shape=(H, W + 2 * T))
    k = mx.ones((1, 5, 5, 1)) / 25.0
    tex = mx.conv2d(tex[None, :, :, None], k, padding=2)[0, :, :, 0]
    clip = [mx.broadcast_to(mx.clip(tex[:, 2 * t:2 * t + W], 0, 1)[..., None],
                            (H, W, 3)) for t in range(T)]
    outs = _run(clip, jitter=True)
    for t in range(T):
        assert float(mx.max(mx.abs(outs[t] - clip[t]))) < 1e-6


def test_nonmultiple16_margin_passes_through():
    # frames not a multiple of 16: the bottom/right remainder margin has no
    # validity grid and must never fire
    mx.random.seed(19)
    h, w = 190, 250
    base = mx.clip(mx.random.uniform(shape=(h, w, 3)) * 0.3 + 0.3, 0, 1)
    d = 0.02 * (1.0 + _blocky((192, 256)))[:h, :w]
    clip = [mx.clip(base + (d if (t // 3) % 2 == 0 else -d), 0, 1)
            for t in range(T)]
    outs = _run(clip, illum_veto=False)
    t = T // 2
    interior_fixed = float(mx.mean(mx.abs(outs[t][:176, :240] - clip[t][:176, :240])))
    margin_dy = float(mx.max(mx.abs(outs[t][176:, :] - clip[t][176:, :])))
    margin_dx = float(mx.max(mx.abs(outs[t][:, 240:] - clip[t][:, 240:])))
    assert interior_fixed > 0.01          # the interior integrates
    assert margin_dy < 1e-6 and margin_dx < 1e-6


# ---- sync-keyed GOP-pumping rescue ----------------------------------------


def _run_sync(clip, sync_at, **kw):
    """Run with tokens carrying raw-stream sync flags (the file endpoint's
    FrameUnit.source shape)."""
    from types import SimpleNamespace

    from kinovsr.processors.units import SourceFrameInfo

    s = StaticStateDeflicker(**kw)
    outs = []
    for i, f in enumerate(clip):
        token = SimpleNamespace(source=SourceFrameInfo(
            index=i, is_sync=i in sync_at))
        outs += s.feed(f, token=token)
    outs += s.flush()
    assert [tok.source.index for _o, tok in outs] == list(range(len(clip)))
    return [o for o, _tok in outs], s


def _sign_mixed_step():
    # A persistent block-scale re-quantization step: half the 8px blocks
    # land above the old state, half below (sign-mixed, like real codec
    # block re-rolls; a same-signed field is the illumination veto's job).
    mx.random.seed(31)
    sign = mx.where(_blocky((H, W)) > 0.5, 1.0, -1.0)
    return 0.04 * sign


def test_single_gop_step_is_refused_without_sync_keying():
    # The base oscillatory gate reads one state step as a possible
    # lighting step (net ~ tv) and correctly refuses it; without sync
    # flags this footage keeps its I-frame pop. Pins the baseline the
    # rescue is measured against.
    base = _base()
    step = _sign_mixed_step()
    cut = T // 2
    clip = [mx.clip(base + (step if t >= cut else 0.0), 0, 1)
            for t in range(T)]
    outs, _s = _run_sync(clip, sync_at=set(), gop=True)
    worst = max(float(mx.max(mx.abs(o - c)))
                for o, c in zip(outs, clip, strict=True))
    assert worst < 0.012, f"unexplained step altered by {worst}"


def test_gop_step_at_sync_sample_is_softened():
    # The same step, now landing exactly where the stream declares a
    # coding restart: the sync-keyed re-test discounts the sync-pair
    # delta, the profile reads flat-step-flat, and the boundary pop is
    # integrated toward the dwell mixture.
    base = _base()
    step = _sign_mixed_step()
    cut = T // 2
    clip = [mx.clip(base + (step if t >= cut else 0.0), 0, 1)
            for t in range(T)]
    outs, s = _run_sync(clip, sync_at={cut}, gop=True)
    pop_in = float(mx.mean(mx.abs(clip[cut] - clip[cut - 1])))
    pop_out = float(mx.mean(mx.abs(outs[cut] - outs[cut - 1])))
    assert pop_out < 0.5 * pop_in, (pop_in, pop_out)
    assert s.stats()["gop_rescued"] > 0.0
    # far from the boundary the window sees no step: passthrough
    tail = float(mx.max(mx.abs(outs[T - 2] - clip[T - 2])))
    assert tail < 0.012, f"far tail altered by {tail}"


def test_gop_rescue_off_reproduces_baseline_exactly():
    base = _base()
    step = _sign_mixed_step()
    cut = T // 2
    clip = [mx.clip(base + (step if t >= cut else 0.0), 0, 1)
            for t in range(T)]
    with_flags, _s = _run_sync(clip, sync_at={cut}, gop=False)
    without_flags, _s2 = _run_sync(clip, sync_at=set(), gop=False)
    for a, b in zip(with_flags, without_flags, strict=True):
        assert float(mx.max(mx.abs(a - b))) == 0.0


def test_lighting_ramp_with_sync_frames_stays_passthrough():
    # A ramp crossing an I-frame only loses one pair's delta in the
    # re-test; the residual profile is still monotone, so the rescue must
    # not flatten it (the ramp-safety property of discounting ONLY
    # sync-pair deltas).
    base = _base()
    tt = mx.arange(T).astype(mx.float32) / (T - 1)
    clip = [mx.clip(base * (0.88 + 0.20 * float(g)), 0, 1) for g in tt]
    outs, _s = _run_sync(clip, sync_at={T // 2}, gop=True)
    worst = max(float(mx.max(mx.abs(o - c)))
                for o, c in zip(outs, clip, strict=True))
    assert worst < 0.012, f"ramp altered by {worst}"


def test_coherent_light_step_at_sync_sample_stays_refused():
    # A real light STEP that happens to land on an I-frame passes the
    # sync-keyed gate but its correction field is same-signed everywhere:
    # the sign-coherence veto still refuses it. The residual-risk pin.
    base = _base()
    cut = T // 2
    clip = [mx.clip(base + (0.04 if t >= cut else 0.0), 0, 1)
            for t in range(T)]
    outs, _s = _run_sync(clip, sync_at={cut}, gop=True)
    pop_in = float(mx.mean(mx.abs(clip[cut] - clip[cut - 1])))
    pop_out = float(mx.mean(mx.abs(outs[cut] - outs[cut - 1])))
    assert pop_out > 0.85 * pop_in, (pop_in, pop_out)
