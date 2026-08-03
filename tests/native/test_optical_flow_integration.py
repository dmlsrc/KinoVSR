"""Native checks for small and portrait public optical-flow handling."""

from __future__ import annotations

import mlx.core as mx
import pytest


def _write_flow_buffer(buffer, field) -> None:
    import numpy as np

    from kinovsr.native.frameworks import Quartz

    height, width, components = field.shape
    assert components == 2
    assert int(Quartz.CVPixelBufferGetWidth(buffer)) == width
    assert int(Quartz.CVPixelBufferGetHeight(buffer)) == height
    row_bytes = int(Quartz.CVPixelBufferGetBytesPerRow(buffer))
    encoded = np.ascontiguousarray(field.astype(np.float16)).view(np.uint8)
    encoded = encoded.reshape(height, width * 4)
    Quartz.CVPixelBufferLockBaseAddress(buffer, 0)
    try:
        view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(
            height * row_bytes
        )
        packed = np.frombuffer(view, dtype=np.uint8).reshape(height, row_bytes)
        packed[:, : width * 4] = encoded
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 0)


def _read_flow_buffer(buffer):
    import numpy as np

    from kinovsr.native.frameworks import Quartz

    width = int(Quartz.CVPixelBufferGetWidth(buffer))
    height = int(Quartz.CVPixelBufferGetHeight(buffer))
    row_bytes = int(Quartz.CVPixelBufferGetBytesPerRow(buffer))
    Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
    try:
        view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(
            height * row_bytes
        )
        packed = np.frombuffer(view, dtype=np.uint8).reshape(height, row_bytes)
        active = packed[:, : width * 4].copy()
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)
    return active.view(np.float16).reshape(height, width, 2).astype(np.float32)


@pytest.mark.integration
@pytest.mark.parametrize("portrait", [False, True])
def test_vision_flow_metal_conversion_matches_vt_geometry_units(portrait):
    import numpy as np

    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native.vision_flow import FLOW_16H, VisionFlowToVtConverter

    attrs = {
        "PixelFormatType": FLOW_16H,
        "IOSurfaceProperties": {},
        "MetalCompatibility": True,
    }
    if portrait:
        source_width, source_height = 2, 3
        destination_width, destination_height = 3, 2
        source = np.empty((3, 2, 2), dtype=np.float32)
        for y in range(3):
            for x in range(2):
                value = 10 * y + x
                source[y, x] = (value, 100 + value)
        expected = np.array(
            [
                [[101, -1], [111, -11], [121, -21]],
                [[100, 0], [110, -10], [120, -20]],
            ],
            dtype=np.float32,
        )
    else:
        source_width, source_height = 4, 2
        destination_width, destination_height = 2, 1
        source = np.empty((2, 4, 2), dtype=np.float32)
        for y in range(2):
            for x in range(4):
                source[y, x] = (x + 10 * y, 2)
        expected = np.array(
            [[[2.75, 1.0], [3.75, 1.0]]],
            dtype=np.float32,
        )

    source_buffer = pb.make_pixel_buffer_from_attrs(
        source_width,
        source_height,
        attrs,
    )
    destination_buffer = pb.make_pixel_buffer_from_attrs(
        destination_width,
        destination_height,
        attrs,
    )
    _write_flow_buffer(source_buffer, source)
    converter = VisionFlowToVtConverter(
        source_width,
        source_height,
        destination_width,
        destination_height,
        rotate_counterclockwise=portrait,
    )
    try:
        converter.convert(source_buffer, destination_buffer)
        actual = _read_flow_buffer(destination_buffer)
    finally:
        converter.close()

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)


@pytest.mark.integration
@pytest.mark.parametrize(
    "source_width,source_height,destination_width,destination_height,portrait",
    [
        (854, 480, 240, 135, False),
        (240, 320, 80, 60, True),
    ],
)
def test_vision_flow_metal_conversion_matches_area_oracle(
    source_width,
    source_height,
    destination_width,
    destination_height,
    portrait,
):
    import cv2
    import numpy as np

    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native.vision_flow import FLOW_16H, VisionFlowToVtConverter

    attrs = {
        "PixelFormatType": FLOW_16H,
        "IOSurfaceProperties": {},
        "MetalCompatibility": True,
    }
    source = np.random.default_rng(1729).uniform(
        -8,
        8,
        size=(source_height, source_width, 2),
    ).astype(np.float32)
    source_buffer = pb.make_pixel_buffer_from_attrs(
        source_width,
        source_height,
        attrs,
    )
    destination_buffer = pb.make_pixel_buffer_from_attrs(
        destination_width,
        destination_height,
        attrs,
    )
    _write_flow_buffer(source_buffer, source)
    converter = VisionFlowToVtConverter(
        source_width,
        source_height,
        destination_width,
        destination_height,
        rotate_counterclockwise=portrait,
    )
    try:
        converter.convert(source_buffer, destination_buffer)
        actual = _read_flow_buffer(destination_buffer)
    finally:
        converter.close()

    if portrait:
        rotated = np.rot90(source, k=1)
        oriented = np.stack(
            (rotated[..., 1], -rotated[..., 0]),
            axis=-1,
        )
    else:
        oriented = source
    expected = cv2.resize(
        oriented,
        (destination_width, destination_height),
        interpolation=cv2.INTER_AREA,
    )
    expected[..., 0] *= destination_width / oriented.shape[1]
    expected[..., 1] *= destination_height / oriented.shape[0]

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)


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
def test_small_explicit_vt_vsr_uses_deterministic_image_fallback():
    from kinovsr.native.vsr import VsrSession

    session = VsrSession(
        64,
        48,
        mode="balanced",
        fps=30.0,
        explicit_flow=True,
        flow_backend="vt",
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
def test_small_internal_balanced_vsr_keeps_temporal_video_mode():
    from kinovsr.native.vsr import VsrSession

    session = VsrSession(
        64,
        48,
        mode="balanced",
        fps=30.0,
        explicit_flow=False,
        flow_backend="internal",
    )
    try:
        assert session._explicit_flow is False
        assert session._image_fallback is False
        assert session._temporal_video is True
        assert session._flow_processor is None
        assert session._flow_pairs is None
        assert session._src_pool_allocation_limit > 1
    finally:
        session.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    "width,height,expected_flow",
    [
        (854, 480, (240, 135)),
        (640, 480, (160, 128)),
        (640, 360, (160, 128)),
        (320, 240, (128, 128)),
        (128, 128, (128, 128)),
    ],
)
def test_sequential_flow_estimator_stays_bounded_on_repaired_destinations(
    width,
    height,
    expected_flow,
):
    """A minimally enlarged destination must not revive estimator divergence.

    The old forced-full destination grew beyond 313 pixels on byte-identical
    frames in eleven Sequential submissions. Exercise both the 854x480
    advertised control and destinations raised to the 128-pixel writer floor
    for long enough to expose that failure mode.

    This deliberately pins only the public flow estimator's numerical state.
    A sane field is not sufficient evidence of correct VSR behavior: VT SR can
    turn a geometrically incompatible field into breathing/wavy output, so the
    rendered path requires separate video-level validation.
    """
    import numpy as np

    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native.frameworks import Quartz, vt
    from kinovsr.native.vsr import VsrSession

    def center_mean_magnitude(buffer):
        flow_width = int(Quartz.CVPixelBufferGetWidth(buffer))
        flow_height = int(Quartz.CVPixelBufferGetHeight(buffer))
        row_bytes = int(Quartz.CVPixelBufferGetBytesPerRow(buffer))
        Quartz.CVPixelBufferLockBaseAddress(buffer, 1)
        try:
            view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(
                flow_height * row_bytes
            )
            field = (
                np.frombuffer(view, dtype=np.float16)
                .reshape(flow_height, row_bytes // 2)[:, : flow_width * 2]
                .reshape(flow_height, flow_width, 2)
                .astype(np.float32)
            )
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)
        center = field[
            flow_height // 4 : flow_height - flow_height // 4,
            flow_width // 4 : flow_width - flow_width // 4,
        ]
        return float(np.sqrt(np.sum(center * center, axis=-1)).mean())

    session = VsrSession(
        width,
        height,
        mode="balanced",
        fps=30.0,
        explicit_flow=True,
        flow_backend="vt",
    )
    try:
        rgb = mx.random.uniform(
            low=0.05,
            high=0.95,
            shape=(height, width, 3),
            key=mx.random.key(width * 1000 + height),
        ).astype(mx.float16)
        frame = mx.concatenate(
            [rgb, mx.ones((height, width, 1), dtype=mx.float16)],
            axis=-1,
        )
        mx.eval(frame)
        sources = [
            session._upload_src_buffer(frame),
            session._upload_src_buffer(frame),
        ]

        def wrap(buffer, index):
            return vt.VTFrameProcessorFrame.alloc(
            ).initWithBuffer_presentationTimeStamp_(
                buffer,
                pb.frame_pts(index, session.fps),
            )

        previous = wrap(sources[0], 0)
        trace = []
        # One Random submission followed by the eleven Sequential submissions
        # that exposed the old divergence is the complete regression horizon.
        for index in range(1, 13):
            current = wrap(sources[index % 2], index)
            mode = (
                vt.VTOpticalFlowParametersSubmissionModeRandom
                if index == 1
                else vt.VTOpticalFlowParametersSubmissionModeSequential
            )
            slot = (index - 1) % 2
            session._run_explicit_flow(
                previous,
                current,
                slot,
                mode,
                index,
            )
            pair = session._flow_pairs[slot]
            trace.append(
                max(
                    center_mean_magnitude(pair[0]),
                    center_mean_magnitude(pair[1]),
                )
            )
            previous = current

        flow_width = int(Quartz.CVPixelBufferGetWidth(session._flow_pairs[0][0]))
        flow_height = int(Quartz.CVPixelBufferGetHeight(session._flow_pairs[0][0]))
        assert (flow_width, flow_height) == expected_flow
        assert max(trace) < 2.0, trace
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
@pytest.mark.parametrize(
    "width,height,expected_flow",
    [
        (64, 48, (80, 60)),
        (128, 192, (80, 60)),
    ],
)
def test_vision_balanced_vsr_runs_vt_geometry_without_image_fallback(
    width,
    height,
    expected_flow,
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
        assert Quartz.CVPixelBufferGetWidth(forward) == expected_flow[0]
        assert Quartz.CVPixelBufferGetHeight(forward) == expected_flow[1]
        zero_pair = session._flow_zero_pair
        destinations = session._vision_flow_destinations
        assert zero_pair is not None
        assert destinations is not None
        assert all(
            pair[0] is zero_pair[0]
            for pair in session._flow_pairs
        )
        assert tuple(pair[1] for pair in session._flow_pairs) == destinations
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
    minimally repaired 160x128 field through VSR. The runaway does not reproduce
    here with synthetic texture, so this test guards the estimator invariant and
    end-to-end execution, not rendered quality on moving real content.
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
