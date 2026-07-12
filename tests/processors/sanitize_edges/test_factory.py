"""sanitize_edges factory: a declared edge replicate-fill family.

extend replaces the bands and keeps the replication; restore brackets the
chain - a companion post-pass composites the ORIGINAL border back at output
geometry, paired to the pre-pass input by PTS.
"""
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
from kinovsr.processors.sanitize_edges import FACTORY
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def parse(raw):
    return FACTORY.parse_config(
        raw, capability=Capability.PREPROCESS, profile=None,
        settings=SETTINGS)


def stream() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(64, 48)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


class TestParse:
    def test_edges_required(self):
        with pytest.raises(ValueError, match="probe-time"):
            parse({})
        with pytest.raises(ValueError, match="no bands"):
            parse({"edges": "0,0,0,0"})

    def test_fill_vocabulary(self):
        assert parse({"edges": "2,0,0,0"}).fill == "extend"  # default
        assert parse({"edges": "2,0,0,0", "fill": "restore"}).fill == "restore"
        with pytest.raises(ValueError, match="crop"):
            parse({"edges": "2,0,0,0", "fill": "trim"})

    def test_only_restore_declares_a_companion(self):
        spec = FACTORY.capabilities[Capability.PREPROCESS]
        assert spec.companion(parse({"edges": "2,0,0,0"})) is None
        companion = spec.companion(parse({"edges": "2,0,0,0", "fill": "restore"}))
        assert companion is not None

    def test_catalog_resolves(self):
        assert get_factory("sanitize_edges") is FACTORY


def test_bands_replicate_and_geometry_is_untouched():
    import mlx.core as mx

    from kinovsr.pipeline import resolve_pipeline, run_plan

    frame = mx.broadcast_to(
        mx.linspace(0, 1, 48)[:, None, None], (48, 64, 3)
    ).astype(mx.float32)
    # poison the top two rows
    poisoned = mx.concatenate(
        [mx.ones((2, 64, 3), dtype=mx.float32), frame[2:]], axis=0)
    units = [FrameUnit(payload=poisoned, pts=0, duration=960)]
    plan = resolve_pipeline(
        {"pipeline": ["san"],
         "san": {"processor": "sanitize_edges", "edges": "2,0,0,0"}},
        input_spec=stream(), settings=SETTINGS)
    out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))
    assert out[0].payload.shape == (48, 64, 3)
    assert plan.output_spec == stream()
    # bands now replicate the first interior row; the poison is gone
    result = out[0].payload
    assert bool(mx.array_equal(result[0], result[2]))
    assert bool(mx.array_equal(result[1], result[2]))
    assert float(result[0, 0, 0]) != 1.0


def test_full_axis_bands_fail_preflight():
    from kinovsr.pipeline import resolve_pipeline

    config = {
        "pipeline": ["san"],
        "san": {"processor": "sanitize_edges", "edges": "48,0,0,0"},
    }
    with pytest.raises(ValueError, match="leave no interior"):
        resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)


class TestRestore:
    @staticmethod
    def _poisoned(mx, top_value=1.0):
        frame = mx.broadcast_to(
            mx.linspace(0, 1, 48)[:, None, None], (48, 64, 3)).astype(mx.float32)
        return mx.concatenate(
            [mx.full((2, 64, 3), top_value, dtype=mx.float32), frame[2:]],
            axis=0)

    def test_companion_is_appended_and_restores_the_original(self):
        import mlx.core as mx

        from kinovsr.pipeline import resolve_pipeline, run_plan

        units = [FrameUnit(payload=self._poisoned(mx), pts=0, duration=960)]
        plan = resolve_pipeline(
            {"pipeline": ["san"],
             "san": {"processor": "sanitize_edges", "edges": "2,0,0,0",
                     "fill": "restore"}},
            input_spec=stream(), settings=SETTINGS)
        # the builder appended a synthetic companion post-stage at the end
        assert [(s.name, s.companion_of) for s in plan.stages] == [
            ("san", None), ("san:post", "san")]
        out = list(run_plan(plan, units, PipelineContext(settings=SETTINGS)))
        # restore puts the ORIGINAL top rows back (extend would replicate the
        # interior gradient); geometry unchanged.
        result = out[0].payload
        assert result.shape == (48, 64, 3)
        assert float(result[0, 0, 0]) == pytest.approx(1.0)

    def test_restore_pairs_each_output_with_its_own_input_through_delay(self):
        # sanitize_edges(restore) -> bsvd: bsvd's 16-frame delay reorders WHEN
        # outputs emerge, but the PTS-keyed buffer must still hand each output
        # its own frame's original border, not a neighbour's.
        import mlx.core as mx

        from kinovsr.pipeline import resolve_pipeline, run_plan

        # each frame's top band carries a distinct value; interiors are noise
        units = []
        for i in range(6):
            interior = mx.random.uniform(shape=(46, 64, 3)).astype(mx.float32)
            top = mx.full((2, 64, 3), 0.1 * (i + 1), dtype=mx.float32)
            units.append(FrameUnit(
                payload=mx.concatenate([top, interior], axis=0),
                pts=i * 960, duration=960))
        try:
            plan = resolve_pipeline(
                {"pipeline": ["san", "den"],
                 "san": {"processor": "sanitize_edges", "edges": "2,0,0,0",
                         "fill": "restore"},
                 "den": {"processor": "bsvd", "strength": 0.3}},
                input_spec=stream(), settings=SETTINGS)
            out = list(run_plan(plan, list(units),
                                PipelineContext(settings=SETTINGS)))
        except FileNotFoundError as exc:
            pytest.skip(f"bsvd weights not available: {exc}")
        assert [u.pts for u in out] == [i * 960 for i in range(6)]
        for i, u in enumerate(out):
            # top band restored to THIS frame's original value
            assert float(u.payload[0, 0, 0]) == pytest.approx(0.1 * (i + 1),
                                                              abs=2e-3)

    def test_restore_rejects_a_non_one_to_one_timeline(self):
        # The PTS pairing only holds on a 1:1 timeline, so a cardinality
        # change (interpolation) upstream of the companion is rejected at
        # preflight naming the post-stage. Exercised directly on a non-1:1
        # input, since no MLX-domain interpolator exists to build one in a
        # pure resolve.
        from kinovsr.pipeline import resolve_pipeline
        from kinovsr.processors import Cardinality
        from kinovsr.processors.errors import StreamEdgeError

        interpolated = StreamSpec(
            frame=stream().frame,
            timeline=TimelineSpec(
                time_base=Fraction(1, 24000), cadence=Fraction(50),
                cardinality=Cardinality.ONE_TO_MANY))
        config = {
            "pipeline": ["san"],
            "san": {"processor": "sanitize_edges", "edges": "2,0,0,0",
                    "fill": "restore"},
        }
        with pytest.raises(StreamEdgeError, match="cardinality|san:post"):
            resolve_pipeline(config, input_spec=interpolated, settings=SETTINGS)
