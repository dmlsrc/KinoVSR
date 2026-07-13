"""MetalFX family: config parsing, spec transform, driver, and e2e runs.

The family has no weights (the model ships inside the OS framework), so
unlike the checkpoint families the integration tests need no
requires_weights gate - only a Metal device that supports MetalFX
spatial scaling; runs skip cleanly where it is unsupported.
"""

from fractions import Fraction

import av
import mlx.core as mx
import pytest

from kinovsr.processors import (
    Capability,
    FrameUnit,
    Geometry,
    PipelineContext,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
    get_factory,
)
from kinovsr.processors.errors import MediaError
from kinovsr.processors.metalfx import (
    FACTORY,
    MetalFxSpatialUpscaler,
    MetalFxStageConfig,
)
from kinovsr.settings import Settings

SETTINGS = Settings()


def parse(raw, profile=None):
    return FACTORY.parse_config(
        raw, capability=Capability.UPSCALE, profile=profile,
        settings=SETTINGS)


def stream(width=64, height=48) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(width, height)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


@pytest.mark.unit
class TestParse:
    def test_default_scale(self):
        assert parse({}) == MetalFxStageConfig(scale=2)

    def test_explicit_scales(self):
        for scale in (2, 3, 4):
            assert parse({"scale": scale}).scale == scale

    def test_rejects_out_of_range_scale(self):
        for scale in (0, 1, 5):
            with pytest.raises(ValueError, match="scale must be"):
                parse({"scale": scale})

    def test_unknown_key_suggests(self):
        with pytest.raises(ValueError, match="did you mean 'scale'"):
            parse({"scal": 2})


@pytest.mark.unit
class TestSpec:
    def test_produces_scales_geometry(self):
        spec = FACTORY.capabilities[Capability.UPSCALE]
        out = spec.produces(stream(), parse({"scale": 4}))
        assert out.frame.geometry == Geometry(256, 192)
        assert out.timeline == stream().timeline

    def test_no_profiles_no_weights(self):
        assert FACTORY.capabilities[Capability.UPSCALE].profiles == ()

    def test_catalog_resolves(self):
        assert get_factory("metalfx") is FACTORY

    def test_driver_rejects_bad_scale(self):
        with pytest.raises(ValueError, match="scale must be"):
            MetalFxSpatialUpscaler(scale=5)


def skip_if_unsupported(exc: MediaError):
    if "not supported" in str(exc):
        pytest.skip(f"MetalFX unavailable: {exc}")
    raise exc


@pytest.mark.integration
class TestDriver:
    def test_per_frame_shapes_and_range(self):
        frame = mx.random.uniform(shape=(48, 64, 3)).astype(mx.float32)
        for scale in (2, 3, 4):
            up = MetalFxSpatialUpscaler(scale=scale)
            try:
                (sr, token), = up.feed(frame, token="t")
            except MediaError as exc:
                skip_if_unsupported(exc)
            assert token == "t"
            assert sr.shape == (48 * scale, 64 * scale, 3)
            assert sr.dtype == mx.float32
            assert float(sr.min()) >= 0.0 and float(sr.max()) <= 1.0
            assert up.flush() == []
            up.close()

    def test_fp16_roundtrip_and_determinism(self):
        frame = mx.random.uniform(shape=(48, 64, 3)).astype(mx.float16)
        up = MetalFxSpatialUpscaler(scale=2)
        try:
            a = up.feed(frame)[0][0]
        except MediaError as exc:
            skip_if_unsupported(exc)
        b = up.feed(frame)[0][0]
        assert a.dtype == mx.float16
        assert bool(mx.array_equal(a, b))
        up.close()

    def test_unaligned_output_rows_read_back_correctly(self):
        # Readback rows are ow*8 bytes with no padding. On Apple-family
        # GPUs the texture-to-buffer copy alignment (16 B, feature-set
        # tables) constrains the buffer OFFSET (always 0 here), and blit
        # destinationBytesPerRow needs only a pixel-size multiple - so
        # widths whose rows are not 64/256-byte aligned must still read
        # back unsheared. A vertical gradient turns any row-stride
        # mistake into a non-monotonic column profile.
        for h, w, scale in ((54, 77, 2), (51, 51, 3)):
            gradient = mx.contiguous(mx.broadcast_to(
                (mx.arange(h) / (h - 1))[:, None, None],
                (h, w, 3)).astype(mx.float32))
            up = MetalFxSpatialUpscaler(scale=scale)
            try:
                (sr, _), = up.feed(gradient)
            except MediaError as exc:
                skip_if_unsupported(exc)
            assert sr.shape == (h * scale, w * scale, 3)
            col_mean = mx.mean(sr, axis=(1, 2))
            assert bool(mx.all(col_mean[1:] >= col_mean[:-1] - 1e-3))
            up.close()

    def test_geometry_change_mid_stream_raises(self):
        up = MetalFxSpatialUpscaler(scale=2)
        try:
            up.feed(mx.zeros((48, 64, 3), dtype=mx.float32))
        except MediaError as exc:
            skip_if_unsupported(exc)
        with pytest.raises(MediaError, match="geometry changed"):
            up.feed(mx.zeros((24, 32, 3), dtype=mx.float32))
        up.close()


@pytest.mark.integration
def test_end_to_end_upscale_through_the_chain():
    from kinovsr.pipeline import resolve_pipeline, run_plan

    config = {
        "pipeline": ["up"],
        "up": {"processor": "metalfx", "scale": 2},
    }
    plan = resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)
    context = PipelineContext(settings=SETTINGS)
    frames = [FrameUnit(payload=mx.zeros((48, 64, 3), dtype=mx.float32),
                        pts=i * 960, duration=960) for i in range(3)]
    try:
        out = list(run_plan(plan, frames, context))
    except MediaError as exc:
        skip_if_unsupported(exc)
    assert [u.pts for u in out] == [0, 960, 1920]
    assert all(u.payload.shape == (96, 128, 3) for u in out)
    assert plan.output_spec.frame.geometry == Geometry(128, 96)


@pytest.mark.integration
def test_flag_cli_end_to_end(tmp_path):
    from kinovsr.cli.main import main

    w, h, n, fps = 160, 128, 6, 25
    clip = tmp_path / "clip.mp4"
    out = av.open(str(clip), "w")
    vs = out.add_stream("mpeg4", rate=fps)
    vs.width, vs.height = w, h
    vs.pix_fmt = "yuv420p"
    for t in range(n):
        rows = bytearray()
        for _y in range(h):
            rows += bytes(min(255, (x + 2 * t) % 256) for x in range(w))
        frame = av.VideoFrame(w, h, "gray")
        frame.planes[0].update(bytes(rows))
        for pkt in vs.encode(frame.reformat(format="yuv420p")):
            out.mux(pkt)
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()

    try:
        rc = main([
            "--video", str(clip),
            "--output-dir", str(tmp_path),
            "--upscale", "metalfx", "--metalfx-scale", "2",
            "--mlx-cache-limit-gb", "0.25",
        ])
    except MediaError as exc:
        skip_if_unsupported(exc)
    assert rc == 0
    outputs = list(tmp_path.glob("*_post.mp4"))
    assert len(outputs) == 1
    with av.open(str(outputs[0])) as container:
        stream_info = container.streams.video[0]
        assert (stream_info.width, stream_info.height) == (w * 2, h * 2)
        assert sum(1 for _ in container.decode(video=0)) == n
