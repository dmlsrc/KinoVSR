"""Native checks for balanced VSR flow paths and the Vision-to-VT SR conversion."""

import mlx.core as mx
import pytest


def _write_flow_buffer(buffer, field) -> None:
    # Deliberately numpy, not the product's MLX marshaling: these helpers are
    # an INDEPENDENT implementation of the row-strided fp16 flow layout, so a
    # bug in the product's buffer views cannot cancel out in the assertions
    # (the advertised-geometry arc caught exactly that class of shared-bug
    # masking). Do not "clean this up" to the mx idiom.
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
        view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(height * row_bytes)
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
        view = Quartz.CVPixelBufferGetBaseAddress(buffer).as_buffer(height * row_bytes)
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
    ("source_width", "source_height", "destination_width", "destination_height", "portrait"),
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
    source = (
        np.random.default_rng(1729)
        .uniform(
            -8,
            8,
            size=(source_height, source_width, 2),
        )
        .astype(np.float32)
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
@pytest.mark.parametrize("portrait", [False, True])
def test_vision_flow_unscaled_conversion_matches_block_oracle(portrait):
    """A source smaller than VT's canvas sits unscaled at its top-left."""
    import numpy as np

    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native.vision_flow import FLOW_16H, VisionFlowToVtConverter

    attrs = {
        "PixelFormatType": FLOW_16H,
        "IOSurfaceProperties": {},
        "MetalCompatibility": True,
    }
    # 10x6 oriented source in a 16x12 canvas: the last grid column and row lie
    # beyond the image and the middle ones are partial.
    oriented_width, oriented_height = 10, 6
    destination_width, destination_height = 4, 3
    source = (
        np.random.default_rng(7)
        .uniform(
            -8,
            8,
            size=(oriented_width, oriented_height, 2)
            if portrait
            else (oriented_height, oriented_width, 2),
        )
        .astype(np.float32)
    )
    source_height, source_width = source.shape[:2]
    source_buffer = pb.make_pixel_buffer_from_attrs(source_width, source_height, attrs)
    destination_buffer = pb.make_pixel_buffer_from_attrs(
        destination_width, destination_height, attrs
    )
    _write_flow_buffer(source_buffer, source)
    converter = VisionFlowToVtConverter(
        source_width,
        source_height,
        destination_width,
        destination_height,
        rotate_counterclockwise=portrait,
        unscaled=True,
    )
    try:
        converter.convert(source_buffer, destination_buffer)
        actual = _read_flow_buffer(destination_buffer)
    finally:
        converter.close()

    if portrait:
        rotated = np.rot90(source, k=1)
        oriented = np.stack((rotated[..., 1], -rotated[..., 0]), axis=-1)
    else:
        oriented = source
    expected = np.zeros((destination_height, destination_width, 2), np.float32)
    for y in range(destination_height):
        for x in range(destination_width):
            block = oriented[4 * y : 4 * y + 4, 4 * x : 4 * x + 4]
            if block.size:
                expected[y, x] = block.reshape(-1, 2).mean(axis=0) * 0.25

    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-3)


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
        assert session._temporal_video is True
        assert session._flow_pairs is None
        assert session._src_pool_allocation_limit > 1
    finally:
        session.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("width", "height", "expected_flow"),
    [
        (64, 48, (80, 60)),
        (128, 192, (80, 60)),
        (213, 121, (80, 60)),
    ],
)
def test_vision_balanced_vsr_runs_in_the_vt_sr_flow_geometry(
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
        assert session._temporal_video is True
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
        assert all(pair[0] is zero_pair[0] for pair in session._flow_pairs)
        assert tuple(pair[1] for pair in session._flow_pairs) == destinations
    finally:
        session.close()
