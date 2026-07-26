"""Native checks for small and portrait public optical-flow handling."""

from __future__ import annotations

import mlx.core as mx
import pytest


@pytest.mark.integration
@pytest.mark.parametrize("width,height", [(64, 48), (127, 192)])
def test_mc_flow_scales_and_rotation_normalizes_to_source(width, height):
    from kinovsr.processors.mc import McTemporalDenoiser

    engine = None
    try:
        engine = McTemporalDenoiser(
            width,
            height,
            strength=0.0,
            window=1,
            self_test=True,
            flow="vt",
        )
        reference, current = engine._self_test_frames(3)
        forward, backward = engine._compute_flows(current, [reference])[0]
        crop = forward[
            height // 4 : height - height // 4,
            width // 4 : width - width // 4,
        ]
        mean = mx.mean(crop, axis=(0, 1))
        mx.eval(mean)

        assert forward.shape == (height, width, 2)
        assert backward is None
        assert 2.5 < float(mean[0]) < 3.5
        assert abs(float(mean[1])) < 0.75
    finally:
        if engine is not None:
            engine.close()


@pytest.mark.integration
def test_small_balanced_vsr_uses_deterministic_image_fallback():
    from kinovsr.native.vsr import VsrSession

    session = VsrSession(
        64,
        48,
        mode="balanced",
        fps=30.0,
        explicit_flow=True,
    )
    try:
        assert session._explicit_flow is False
        assert session._image_fallback is True
        assert session._temporal_video is False
        assert session._flow_processor is None
        assert session._flow_pairs is None
        assert session._src_pool_allocation_limit == 1
    finally:
        session.close()


@pytest.mark.integration
def test_portrait_balanced_vsr_repairs_the_advertised_flow_geometry():
    """A 128x192 source advertises 80x60, so both dimensions need raising.

    The advertisement is already landscape, carrying VT's rotation-normalized
    orientation, so repairing it needs no axis swap. Measured on a 360x640
    portrait clip, whose advertisement is 160x90, the repaired 160x128 scored
    edge temporal instability 11.33 with gradient energy 6.64, against 13.95 and
    6.59 for the previous rotation-normalized full-source 640x360 - better on
    both axes. Non-temporal Image mode scored 11.36 with only 6.16 gradient
    energy, so the repair matches Image's stability while keeping more detail.
    """
    from kinovsr.native.frameworks import Quartz
    from kinovsr.native.vsr import VsrSession

    session = VsrSession(
        128,
        192,
        mode="balanced",
        fps=30.0,
        explicit_flow=True,
    )
    try:
        assert session._explicit_flow is True
        assert session._image_fallback is False
        assert session._flow_pairs is not None
        forward = session._flow_pairs[0][0]
        assert Quartz.CVPixelBufferGetWidth(forward) == 128
        assert Quartz.CVPixelBufferGetHeight(forward) == 128
    finally:
        session.close()


@pytest.mark.integration
@pytest.mark.parametrize("width,height", [(64, 48), (128, 192)])
def test_vision_balanced_vsr_runs_source_geometry_without_image_fallback(
    width,
    height,
):
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native.frameworks import Quartz
    from kinovsr.native.vsr import VsrSession

    rgb = mx.random.uniform(
        low=0.05,
        high=0.95,
        shape=(height, width, 3),
        key=mx.random.key(width * 1000 + height),
    ).astype(mx.float16)
    rgba = mx.concatenate(
        [rgb, mx.ones((height, width, 1), mx.float16)],
        axis=-1,
    )
    frames = [mx.roll(rgba, shift, axis=1) for shift in (0, 1, 2)]
    mx.eval(*frames)

    session = VsrSession(
        width,
        height,
        mode="balanced",
        fps=30.0,
        explicit_flow=True,
        flow_backend="vision",
    )
    output_count = 0
    try:
        assert session._explicit_flow is True
        assert session._image_fallback is False
        assert session._temporal_video is True
        assert session._flow_processor is None
        assert session._flow_pairs is not None

        for index, frame in enumerate(frames):
            output = session.submit_upscale_to_buffer(frame, index)
            if output is not None:
                array = pb.read_rgbahalf_rgb(output)
                mx.eval(array)
                output_count += 1
                del output
        output = session.finish_pending_upscale()
        if output is not None:
            array = pb.read_rgbahalf_rgb(output)
            mx.eval(array)
            output_count += 1
            del output

        assert output_count == len(frames)
        forward = session._flow_pairs[0][0]
        assert Quartz.CVPixelBufferGetWidth(forward) == width
        assert Quartz.CVPixelBufferGetHeight(forward) == height
        zero_pair = session._flow_zero_pair
        assert zero_pair is not None
        assert all(
            pair[0] is zero_pair[0]
            for pair in session._flow_pairs
        )
        assert all(
            pair[1] is not zero_pair[1]
            for pair in session._flow_pairs
        )
    finally:
        session.close()


@pytest.mark.integration
def test_small_balanced_fallback_is_bit_exact_with_image_mode():
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native.vsr import VsrSession

    width, height = 64, 48
    rgb = mx.random.uniform(
        low=0.05,
        high=0.95,
        shape=(height, width, 3),
        key=mx.random.key(1729),
    ).astype(mx.float16)
    rgba = mx.concatenate(
        [rgb, mx.ones((height, width, 1), mx.float16)],
        axis=-1,
    )
    frames = [rgba, mx.roll(rgba, 1, axis=1)]
    mx.eval(*frames)

    def run(mode, explicit_flow):
        session = VsrSession(
            width,
            height,
            mode=mode,
            fps=30.0,
            explicit_flow=explicit_flow,
        )
        outputs = []
        try:
            for index, frame in enumerate(frames):
                buffer = session.upscale_to_buffer(frame, index)
                array = pb.read_rgbahalf_rgb(buffer).astype(mx.float16)
                outputs.append(bytes(memoryview(mx.contiguous(array))))
        finally:
            session.close()
        return outputs

    assert run("balanced", True) == run("image", False)


@pytest.mark.integration
def test_balanced_flow_stays_near_zero_on_a_repeated_frame():
    """A repeated frame must keep producing the same near-zero field.

    The true field for a repeated frame is exactly zero, so this pins the
    end-to-end invariant that a static input produces a small, non-growing
    field. It caught a real runaway: with an oversized destination and
    Sequential submission the mean magnitude grew 0.07 -> 313 px over eleven
    identical pairs on real video.

    Scope, so this is not read as more than it is: at 640x480 the advertised
    destination is 160x120, below the 128 px writer floor, so this exercises the
    forced full-source fallback. The runaway does NOT reproduce here with
    synthetic texture, and it does not reproduce at an advertised destination at
    all, so this test does not by itself discriminate Random from Sequential.
    It guards the invariant, not the mode.
    """
    import numpy as np

    from kinovsr.native.frameworks import Quartz
    from kinovsr.native.vsr import VsrSession

    width, height, count = 640, 480, 8
    # Aperiodic but trackable: white noise gives the matcher nothing to lock
    # onto and produces a degenerate field, while modular/periodic patterns
    # admit exact false matches. Incommensurate sinusoids plus blobs avoid both.
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    texture = (
        0.5
        + 0.18 * np.sin(xx / 11.3 + yy / 29.7)
        + 0.14 * np.sin(xx / 4.7 - yy / 7.1)
        + 0.10 * np.sin((xx + yy) / 17.9)
    )
    rng = np.random.default_rng(11)
    for _ in range(24):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        texture += 0.25 * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 19.0**2))
        )
    texture = np.clip(texture, 0.0, 1.0)
    rgb = mx.array(np.repeat(texture[..., None], 3, axis=2)).astype(mx.float16)
    frame = mx.concatenate(
        [rgb, mx.ones((height, width, 1), dtype=mx.float16)], axis=-1
    )
    mx.eval(frame)

    def flow_mean(buffer):
        # Read the buffer's own geometry: the flow destination is the
        # configuration's advertised shape, not the source shape.
        flow_w = int(Quartz.CVPixelBufferGetWidth(buffer))
        flow_h = int(Quartz.CVPixelBufferGetHeight(buffer))
        row_bytes = int(Quartz.CVPixelBufferGetBytesPerRow(buffer))
        Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
        try:
            view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(
                flow_h * row_bytes
            )
            field = (
                np.frombuffer(view, dtype=np.float16)
                .reshape(flow_h, row_bytes // 2)[:, : flow_w * 2]
                .reshape(flow_h, flow_w, 2)
                .astype(np.float32)
            )
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)
        return float(np.linalg.norm(field, axis=-1).mean())

    session = VsrSession(width, height, mode="balanced", fps=30.0,
                         explicit_flow=True)
    observed = []
    original = session._run_explicit_flow

    def record(previous_frame, current_frame, slot, submission_mode, index):
        original(previous_frame, current_frame, slot, submission_mode, index)
        observed.append(flow_mean(session._flow_pairs[slot][1]))

    session._run_explicit_flow = record
    try:
        for index in range(count):
            session.submit_upscale_to_buffer(frame, index)
        while session.finish_pending_upscale() is not None:
            pass
    finally:
        session.close()

    assert len(observed) == count - 1
    # Every pair is the same pair, so every field must match the first one and
    # stay small. A compounding chain fails the ratio long before the bound.
    assert max(observed) < 1.0, observed
    assert max(observed) <= 4.0 * max(observed[0], 1e-4), observed
