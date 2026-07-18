"""cut_detect factory: a HARD_CUT boundary emitter (content-difference
based; other families may emit the same boundary from other signals,
e.g. keyframe cadence)."""

from fractions import Fraction

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
from kinovsr.processors.boundaries import BoundaryKind
from kinovsr.processors.cut_detect import FACTORY, CutDetectStageConfig
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
        assert parse({}) == CutDetectStageConfig(
            detect="simple", threshold=0.25)

    def test_off_is_not_a_stage(self):
        with pytest.raises(ValueError, match="omits this stage"):
            parse({"detect": "off"})

    def test_threshold_positive(self):
        with pytest.raises(ValueError, match="positive"):
            parse({"threshold": 0.0})

    def test_per_mode_threshold_defaults(self):
        # The modes' statistics live on different scales; an absent
        # threshold resolves per mode, an explicit one always wins.
        assert parse({"detect": "hist"}).threshold == 0.25
        assert parse({"detect": "vtme"}).threshold == 0.07
        assert parse({"detect": "vtme", "threshold": 0.1}).threshold == 0.1


class TestSpec:
    def test_emits_hard_cut_and_passes_spec_through(self):
        spec = FACTORY.capabilities[Capability.PREPROCESS]
        assert BoundaryKind.HARD_CUT in spec.emits_boundaries
        assert spec.produces(stream(), parse({})) == stream()

    def test_catalog_resolves(self):
        assert get_factory("cut_detect") is FACTORY


def test_boundary_lands_on_the_first_frame_of_the_new_shot():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan

    dark = mx.zeros((48, 64, 3), dtype=mx.float32)
    bright = mx.ones((48, 64, 3), dtype=mx.float32)
    frames = [dark, dark, dark, bright, bright, bright]
    units = [FrameUnit(payload=f, pts=i * 960, duration=960)
             for i, f in enumerate(frames)]

    plan = resolve_pipeline(
        {"pipeline": ["cut"], "cut": {"processor": "cut_detect"}},
        input_spec=stream(), settings=SETTINGS)
    out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))

    assert len(out) == 6
    marked = [i for i, u in enumerate(out)
              if any(b.kind is BoundaryKind.HARD_CUT for b in u.boundaries)]
    assert marked == [3]
    cut_boundary = next(b for b in out[3].boundaries
                        if b.kind is BoundaryKind.HARD_CUT)
    assert cut_boundary.source_index == 3
    # payloads pass through untouched
    assert all(bool(mx.array_equal(u.payload, f))
               for u, f in zip(out, frames, strict=True))
