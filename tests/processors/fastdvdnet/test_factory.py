"""FastDVDnet factory: noise-map conditioning end to end.

The parse/defaults surface is covered cross-family in
tests/processors/test_factory_sweep.py; this proves the wiring actually
reaches the (inherently non-blind) engine through the real pipeline.
"""
from fractions import Fraction

import pytest

from kinovsr.processors import (
    FrameUnit,
    Geometry,
    PipelineContext,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.settings import Settings

SETTINGS = Settings()


def stream() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(64, 48)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


@pytest.mark.requires_weights
@pytest.mark.integration
class TestConditioning:
    @staticmethod
    def run(units, config=None):
        from kinovsr.pipeline import resolve_pipeline, run_plan

        table = {"processor": "fastdvdnet", "strength": 0.3}
        table.update(config or {})
        try:
            plan = resolve_pipeline(
                {"pipeline": ["d"], "d": table},
                input_spec=stream(), settings=SETTINGS)
            return list(run_plan(
                plan, list(units), PipelineContext(settings=SETTINGS)))
        except FileNotFoundError as exc:
            pytest.skip(f"fastdvdnet weights not available: {exc}")

    def test_noise_map_auto_conditions_the_output(self):
        # The estimated per-pixel sigma map replaces the constant sigma, so an
        # auto run diverges from the constant run on the same frames while
        # staying frame-aligned. FastDVDnet is non-blind, so it always accepts
        # a map; 16 frames clears the tracker's warm-up.
        import mlx.core as mx

        units = [
            FrameUnit(
                payload=mx.random.uniform(shape=(48, 64, 3)).astype(mx.float32)
                * 0.5 + 0.25,
                pts=i * 960, duration=960)
            for i in range(16)]
        constant = self.run(units)
        auto = self.run(units, config={"noise_map": "auto"})
        assert [u.pts for u in constant] == [u.pts for u in auto]
        deltas = [float(mx.max(mx.abs(c.payload - a.payload)).item())
                  for c, a in zip(constant, auto, strict=True)]
        assert max(deltas) > 1e-3
