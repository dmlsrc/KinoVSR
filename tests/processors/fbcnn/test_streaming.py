"""FBCNN through the streaming runtime, end to end.

Auto quality refreshes its QF map on the runtime's MLX lane. The first refresh
used to fail with "There is no Stream(gpu, 0) in current thread", because the
QF estimator's DCT basis was a lazy graph built at import on another thread.
The whole-file FBCNN test pins the quality, so no test reached a refresh before.
"""

from fractions import Fraction

import mlx.core as mx
import pytest

from kinovsr.processors import (
    FrameUnit,
    Geometry,
    PipelineContext,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.processors.fbcnn.deblocker import FbcnnDeblocker
from kinovsr.settings import Settings

SETTINGS = Settings()
W, H = 128, 96


def stream() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix("bt709", full_range=False, geometry=Geometry(W, H)),
        timeline=TimelineSpec(time_base=Fraction(1, 24000), cadence=Fraction(25)),
    )


@pytest.mark.requires_weights
@pytest.mark.integration
def test_auto_quality_refreshes_on_the_runtime_lane():
    from kinovsr.pipeline import resolve_pipeline, run_plan

    count = FbcnnDeblocker.QF_MIN_FRAMES + 2  # past the first QF refresh
    try:
        plan = resolve_pipeline(
            {"pipeline": ["d"], "d": {"processor": "fbcnn"}},
            input_spec=stream(),
            settings=SETTINGS,
        )
        mx.random.seed(7)
        units = [
            FrameUnit(
                payload=mx.random.uniform(shape=(H, W, 3)).astype(mx.float32),
                pts=i * 960,
                duration=960,
            )
            for i in range(count)
        ]
        out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))
    except FileNotFoundError as exc:
        pytest.skip(f"fbcnn weights not available: {exc}")
    assert len(out) == count
    assert all(tuple(unit.payload.shape) == (H, W, 3) for unit in out)
