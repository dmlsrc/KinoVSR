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
def test_portrait_balanced_vsr_uses_rotation_normalized_flow_geometry():
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
        assert Quartz.CVPixelBufferGetWidth(forward) == 192
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
