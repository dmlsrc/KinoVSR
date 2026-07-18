"""level factory: a centered-window global exposure leveler."""

from fractions import Fraction

import pytest

from kinovsr.processors import (
    Capability,
    Geometry,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
    get_factory,
)
from kinovsr.processors.capabilities import TemporalMode
from kinovsr.processors.level import FACTORY, LevelStageConfig
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw):
    return FACTORY.parse_config(
        raw, capability=Capability.PREPROCESS, profile=None,
        settings=SETTINGS)


def stream(width=64, height=48) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(width, height)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)),
        seekable=True, lookahead_available=True)


class TestParse:
    def test_defaults(self):
        assert parse({}) == LevelStageConfig(window=5, deadband=0.003)

    def test_bounds(self):
        with pytest.raises(ValueError, match="window"):
            parse({"window": 0})
        with pytest.raises(ValueError, match="deadband"):
            parse({"deadband": -0.1})
        assert parse({"deadband": 0.0}).deadband == 0.0

    def test_unknown_keys_rejected(self):
        with pytest.raises(ValueError):
            parse({"strength": 1.0})


class TestSpec:
    def test_centered_passthrough_spec(self):
        spec = FACTORY.capabilities[Capability.PREPROCESS]
        assert spec.temporal_mode is TemporalMode.CENTERED
        assert spec.stateful
        assert spec.produces(stream(), parse({})) == stream()

    def test_catalog_resolves(self):
        assert get_factory("level") is FACTORY
