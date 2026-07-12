"""STDF deblocker: blockiness-map conditioning wiring.

The parse/defaults surface is covered cross-family in
tests/processors/test_factory_sweep.py; this proves the auto path actually
builds the tracker and threads it into StdfDeblocker end to end.
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
class TestDeblockMap:
    @staticmethod
    def run(table):
        import mlx.core as mx

        from kinovsr.pipeline import resolve_pipeline, run_plan

        stage = {"processor": "stdf"}
        stage.update(table)
        try:
            plan = resolve_pipeline(
                {"pipeline": ["d"], "d": stage},
                input_spec=stream(), settings=SETTINGS)
            units = [
                FrameUnit(
                    payload=mx.random.uniform(shape=(48, 64, 3)).astype(
                        mx.float32),
                    pts=i * 960, duration=960)
                for i in range(8)]
            return list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))
        except FileNotFoundError as exc:
            pytest.skip(f"stdf weights not available: {exc}")

    def test_auto_blockiness_conditioning_runs_end_to_end(self):
        # The blockiness estimator keys on coding-grid structure (absent from
        # random frames), so this asserts the WIRING runs - the tracker is
        # built, passed to StdfDeblocker, and estimated per window without
        # error - not a pixel delta. The math itself is unchanged from the
        # harness. 8 frames clear the 7-frame centered window.
        out = self.run({"deblock_map": "auto", "deblock_map_gain": 1.2,
                        "strength": 1.0})
        assert [u.pts for u in out] == [i * 960 for i in range(8)]
        assert all(u.payload.shape == (48, 64, 3) for u in out)
