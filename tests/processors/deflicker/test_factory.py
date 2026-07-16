"""deflicker factory: a CENTERED (self-buffered) preprocess family.

CENTERED means the +/-K window's future half is self-buffered and paid
as output delay, so no source lookahead is demanded and live edges
remain legal inputs.
"""

from fractions import Fraction

import pytest

from kinovsr.processors import (
    Capability,
    FrameUnit,
    Geometry,
    PipelineContext,
    StreamSpec,
    TemporalMode,
    TimelineSpec,
    frame_spec_for_matrix,
    get_factory,
)
from kinovsr.processors.deflicker import FACTORY, DeflickerStageConfig
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw):
    return FACTORY.parse_config(
        raw, capability=Capability.PREPROCESS, profile=None,
        settings=SETTINGS)


def stream(lookahead=True) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(64, 48)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)),
        seekable=True, lookahead_available=lookahead)


class TestParse:
    def test_defaults_match_the_cli(self):
        assert parse({}) == DeflickerStageConfig(
            window=8, band=0.1, frac=0.5, max_fix=0.25,
            jitter=False, strength=1.0, gop=True)

    def test_bounds(self):
        with pytest.raises(ValueError, match="window"):
            parse({"window": 0})
        with pytest.raises(ValueError, match="strength"):
            parse({"strength": 1.5})


class TestSpec:
    def test_no_source_lookahead_demanded(self):
        # The window is self-buffered as output delay, so a source
        # without lookahead (a live edge) is a legal input.
        from kinovsr.pipeline import resolve_pipeline

        config = {"pipeline": ["df"], "df": {"processor": "deflicker"}}
        resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)
        resolve_pipeline(config, input_spec=stream(lookahead=False),
                         settings=SETTINGS)

    def test_temporal_declaration(self):
        spec = FACTORY.capabilities[Capability.PREPROCESS]
        assert spec.temporal_mode is TemporalMode.CENTERED
        assert spec.stateful

    def test_catalog_resolves(self):
        assert get_factory("deflicker") is FACTORY


@pytest.mark.integration
def test_window_delay_rides_tokens_and_flush_drains():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan

    n, window = 10, 3
    units = [FrameUnit(payload=mx.full((48, 64, 3), i / n, dtype=mx.float32),
                       pts=i * 960, duration=960) for i in range(n)]
    plan = resolve_pipeline(
        {"pipeline": ["df"],
         "df": {"processor": "deflicker", "window": window}},
        input_spec=stream(), settings=SETTINGS)
    out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))

    # every input emerges exactly once, in order, on its own timestamps
    assert [u.pts for u in out] == [i * 960 for i in range(n)]
    assert all(u.payload.shape == (48, 64, 3) for u in out)
    # static gradient content: nothing fires, frames pass bit-identical
    assert all(bool(mx.array_equal(u.payload, units[i].payload))
               for i, u in enumerate(out))
