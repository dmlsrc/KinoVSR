"""Builder preflight: arbitrary valid orders resolve; invalid ones fail
before processing with field-level errors naming both edge sides."""

import dataclasses
import sys
import types
from fractions import Fraction

import pytest

from kinovsr.pipeline import (
    BuildPlan,
    OutputEndpointSpec,
    build_processors,
    resolve_pipeline,
)
from kinovsr.processors import (
    Boundary,
    BoundaryKind,
    Capability,
    CapabilitySpec,
    Cardinality,
    Domain,
    FrameUnit,
    Geometry,
    Layout,
    PipelineContext,
    StageConfigError,
    StreamConstraint,
    StreamEdgeError,
    StreamSpec,
    TemporalMode,
    TimelineSpec,
    TimestampPolicy,
    UnknownStageError,
    catalog,
    frame_spec_for_matrix,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

MLX_CONSTRAINT = StreamConstraint(
    layouts=(Layout.MLX_RGB_HWC,),
    domains=(Domain.UNIT, Domain.UNIT_SANITIZED),
)


class Passthrough:
    def __init__(self):
        self.prepared = False

    def prepare(self, input_spec, context):
        self.prepared = True

    def process(self, unit, context):
        yield unit

    def reset(self, boundary, context):
        pass

    def flush(self, context):
        return ()

    def close(self, context):
        pass


class FakeFamily:
    """A configurable contract-only family for builder tests."""

    def __init__(self, name, capabilities, presets=None):
        self.name = name
        self.capabilities = capabilities
        self.parsed = []
        self._presets = presets or {}

    def profile_defaults(self, *, capability, profile):
        return self._presets.get(profile, {})

    def parse_config(self, raw, *, capability, profile, settings):
        if "explode" in raw:
            raise ValueError("explode is not a setting")
        self.parsed.append((dict(raw), capability, profile))
        return dict(raw)

    def build(self, config, *, context):
        return Passthrough()


def upscale4x(spec: StreamSpec, config: object = None) -> StreamSpec:
    frame = dataclasses.replace(
        spec.frame, geometry=spec.frame.geometry.scaled(4))
    return dataclasses.replace(spec, frame=frame)


def interpolate_2x(spec: StreamSpec, config: object = None) -> StreamSpec:
    timeline = dataclasses.replace(
        spec.timeline,
        cadence=spec.timeline.cadence * 2,
        timestamp_policy=TimestampPolicy.REGENERATED,
        cardinality=Cardinality.ONE_TO_MANY,
    )
    return dataclasses.replace(spec, timeline=timeline)


def cap(capability, *, accepts=MLX_CONSTRAINT, produces=None, profiles=(),
        **kwargs):
    return CapabilitySpec(
        capability=capability, profiles=tuple(profiles), accepts=accepts,
        **({"produces": produces} if produces else {}), **kwargs)


@pytest.fixture
def families(monkeypatch):
    """A tiny universe of fake families registered in a clean catalog."""
    monkeypatch.setattr(catalog, "_CATALOG", {})
    monkeypatch.setattr(catalog, "_loaded", {})

    denoise = FakeFamily("fakedenoise", {
        Capability.DENOISE: cap(Capability.DENOISE, profiles=("mild",),
                                stateful=True,
                                temporal_mode=TemporalMode.CAUSAL),
    }, presets={"mild": {"strength": 0.2}})
    upscaler = FakeFamily("fakeupscale", {
        Capability.UPSCALE: cap(Capability.UPSCALE, produces=upscale4x),
    })
    interpolator = FakeFamily("fakeinterp", {
        Capability.INTERPOLATE: cap(
            Capability.INTERPOLATE, produces=interpolate_2x),
    })
    centered = FakeFamily("fakecentered", {
        Capability.DENOISE: cap(
            Capability.DENOISE, temporal_mode=TemporalMode.CENTERED,
            temporal_radius=3, stateful=True),
    })
    cutter = FakeFamily("fakecuts", {
        Capability.PREPROCESS: cap(
            Capability.PREPROCESS,
            emits_boundaries=(BoundaryKind.HARD_CUT,)),
    })
    needs_cuts = FakeFamily("fakeneedscuts", {
        Capability.RESTORE: cap(
            Capability.RESTORE, stateful=True,
            requires_boundaries=(BoundaryKind.HARD_CUT,)),
    })
    nv12_only = FakeFamily("fakenv12", {
        Capability.PREPROCESS: cap(
            Capability.PREPROCESS,
            accepts=StreamConstraint(layouts=(Layout.CV_NV12,),
                                     domains=(Domain.CODED,))),
    })
    bad_tap = FakeFamily("fakebadtap", {
        Capability.METRIC: cap(Capability.METRIC, produces=upscale4x,
                               is_tap=True),
    })
    multi = FakeFamily("fakemulti", {
        Capability.DENOISE: cap(Capability.DENOISE, profiles=("den",)),
        Capability.DEBLOCK: cap(Capability.DEBLOCK, profiles=("deb",)),
    })

    module = types.ModuleType("fake_builder_families")
    for factory in (denoise, upscaler, interpolator, centered, cutter,
                    needs_cuts, nv12_only, bad_tap, multi):
        setattr(module, factory.name, factory)
        catalog.register(factory.name,
                         f"fake_builder_families:{factory.name}")
    monkeypatch.setitem(sys.modules, "fake_builder_families", module)
    return module


def stream(width=640, height=480, *, lookahead=True) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(width, height)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)),
        seekable=True,
        lookahead_available=lookahead,
    )


SETTINGS = Settings()


class TestValidChains:
    def test_chain_threads_geometry_and_timeline(self, families):
        config = {
            "pipeline": ["denoise", "upscale", "interp"],
            "denoise": {"processor": "fakedenoise", "strength": 0.4},
            "upscale": {"processor": "fakeupscale"},
            "interp": {"processor": "fakeinterp"},
        }
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        assert isinstance(plan, BuildPlan)
        assert [s.family for s in plan.stages] == [
            "fakedenoise", "fakeupscale", "fakeinterp"]
        geo = plan.output_spec.frame.geometry
        assert (geo.width, geo.height) == (2560, 1920)
        assert plan.output_spec.timeline.cadence == Fraction(50)
        assert (plan.output_spec.timeline.cardinality
                is Cardinality.ONE_TO_MANY)
        # per-stage edges recorded
        assert plan.stages[1].input_spec.frame.geometry.width == 640
        assert plan.stages[1].output_spec.frame.geometry.width == 2560

    def test_arbitrary_reorder_also_valid(self, families):
        config = {
            "pipeline": ["upscale", "denoise"],   # denoise AFTER upscale
            "denoise": {"processor": "fakedenoise"},
            "upscale": {"processor": "fakeupscale"},
        }
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        assert plan.output_spec.frame.geometry.width == 2560

    def test_duplicate_stage_names_get_independent_instances(self, families):
        config = {
            "pipeline": ["denoise", "denoise"],
            "denoise": {"processor": "fakedenoise"},
        }
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        assert [s.position for s in plan.stages] == [0, 1]
        built = build_processors(plan, PipelineContext(settings=SETTINGS))
        assert built[0][1] is not built[1][1]

    def test_profile_preset_feeds_family_parser(self, families):
        config = {
            "pipeline": ["denoise"],
            "denoise": {"processor": "fakedenoise", "profile": "mild"},
        }
        resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)
        raw, capability, profile = families.fakedenoise.parsed[-1]
        assert raw == {"strength": 0.2}      # preset landed
        assert profile == "mild"
        assert capability is Capability.DENOISE

    def test_stage_table_overrides_profile_preset(self, families):
        config = {
            "pipeline": ["denoise"],
            "denoise": {"processor": "fakedenoise", "profile": "mild",
                        "strength": 0.7},
        }
        resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)
        raw, _, _ = families.fakedenoise.parsed[-1]
        assert raw == {"strength": 0.7}

    def test_boundary_requirement_satisfied_by_upstream_emitter(
            self, families):
        config = {
            "pipeline": ["cuts", "restore"],
            "cuts": {"processor": "fakecuts"},
            "restore": {"processor": "fakeneedscuts"},
        }
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        assert len(plan.stages) == 2


class TestInvalidChains:
    def test_edge_error_names_both_sides_and_field(self, families):
        config = {
            "pipeline": ["denoise", "nv12"],
            "denoise": {"processor": "fakedenoise"},
            "nv12": {"processor": "fakenv12"},
        }
        with pytest.raises(StreamEdgeError) as exc:
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)
        message = str(exc.value)
        assert "invalid edge denoise -> nv12:" in message
        assert "frame.layout" in message
        assert "cv_nv12" in message and "mlx_rgb_hwc" in message
        assert exc.value.upstream == "denoise"
        assert exc.value.downstream == "nv12"

    def test_failure_is_preflight_not_mid_run(self, families):
        """No factory.build happens when validation fails."""
        config = {
            "pipeline": ["nv12"],
            "nv12": {"processor": "fakenv12"},
        }
        with pytest.raises(StreamEdgeError) as exc:
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)
        assert exc.value.upstream == "input"

    def test_centered_window_is_self_buffered_no_source_demand(
            self, families):
        # CENTERED pays its future reach as output delay (emit t once
        # t+radius arrived), so it demands nothing of the source: a
        # live edge without lookahead is a legal input.
        config = {
            "pipeline": ["centered"],
            "centered": {"processor": "fakecentered"},
        }
        plan = resolve_pipeline(
            config, input_spec=stream(lookahead=True), settings=SETTINGS)
        assert plan.stages[0].capability_spec.temporal_radius == 3
        plan = resolve_pipeline(
            config, input_spec=stream(lookahead=False), settings=SETTINGS)
        assert plan.stages[0].capability_spec.temporal_radius == 3

    def test_boundary_requirement_unmet_is_rejected(self, families):
        config = {
            "pipeline": ["restore"],
            "restore": {"processor": "fakeneedscuts"},
        }
        with pytest.raises(StreamEdgeError) as exc:
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)
        assert "hard_cut" in str(exc.value)

    def test_emitter_after_the_requirer_does_not_count(self, families):
        config = {
            "pipeline": ["restore", "cuts"],
            "restore": {"processor": "fakeneedscuts"},
            "cuts": {"processor": "fakecuts"},
        }
        with pytest.raises(StreamEdgeError):
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)

    def test_output_endpoint_constraint_applies(self, families):
        config = {
            "pipeline": ["denoise"],
            "denoise": {"processor": "fakedenoise"},
        }
        output = OutputEndpointSpec(
            accepts=StreamConstraint(require_even_dims=True))
        with pytest.raises(StreamEdgeError) as exc:
            resolve_pipeline(
                config, input_spec=stream(641, 480), settings=SETTINGS,
                output=output)
        assert exc.value.downstream == "output"
        assert "even width and height" in str(exc.value)

    def test_tap_rewriting_the_stream_is_a_config_error(self, families):
        config = {
            "pipeline": ["tap"],
            "tap": {"processor": "fakebadtap"},
        }
        with pytest.raises(StageConfigError, match="tap"):
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)


class TestSelectors:
    def test_unknown_family_is_wrapped_with_stage_name(self, families):
        config = {"pipeline": ["x"], "x": {"processor": "fakedenoize"}}
        with pytest.raises(UnknownStageError) as exc:
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)
        message = str(exc.value)
        assert message.startswith("[x]")
        assert "did you mean 'fakedenoise'" in message

    def test_multi_capability_needs_selector(self, families):
        config = {"pipeline": ["m"], "m": {"processor": "fakemulti"}}
        with pytest.raises(UnknownStageError, match="state capability"):
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)

    def test_profile_selects_capability_unambiguously(self, families):
        config = {"pipeline": ["m"],
                  "m": {"processor": "fakemulti", "profile": "deb"}}
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        assert plan.stages[0].capability is Capability.DEBLOCK

    def test_explicit_capability_wins(self, families):
        config = {"pipeline": ["m"],
                  "m": {"processor": "fakemulti", "capability": "denoise"}}
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        assert plan.stages[0].capability is Capability.DENOISE

    def test_unknown_capability_token_lists_valid(self, families):
        config = {"pipeline": ["m"],
                  "m": {"processor": "fakemulti", "capability": "sharpen"}}
        with pytest.raises(UnknownStageError, match="unknown capability"):
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)

    def test_unoffered_capability_lists_offers(self, families):
        config = {"pipeline": ["d"],
                  "d": {"processor": "fakedenoise",
                        "capability": "upscale"}}
        with pytest.raises(UnknownStageError, match="does not offer"):
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)

    def test_unknown_profile_lists_profiles(self, families):
        config = {"pipeline": ["d"],
                  "d": {"processor": "fakedenoise", "profile": "extreme"}}
        with pytest.raises(UnknownStageError,
                           match="no profile 'extreme'"):
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)

    def test_family_parse_errors_carry_the_stage_name(self, families):
        config = {"pipeline": ["d"],
                  "d": {"processor": "fakedenoise", "explode": True}}
        with pytest.raises(StageConfigError, match=r"\[d\] explode"):
            resolve_pipeline(config, input_spec=stream(), settings=SETTINGS)


class TestBuildProcessors:
    def test_builds_in_order_with_stage_contexts(self, families):
        config = {
            "pipeline": ["denoise", "upscale"],
            "denoise": {"processor": "fakedenoise"},
            "upscale": {"processor": "fakeupscale"},
        }
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        built = build_processors(plan, PipelineContext(settings=SETTINGS))
        assert [stage.name for stage, _ in built] == ["denoise", "upscale"]
        for _, instance in built:
            unit = FrameUnit(payload="p", pts=0, duration=1)
            context = PipelineContext(settings=SETTINGS)
            instance.reset(Boundary(BoundaryKind.STREAM_START), context)
            assert list(instance.process(unit, context)) == [unit]


class TestBuildRollback:
    def test_partial_build_failure_closes_built_stages(self, families):
        closed = []

        class Session:
            def __init__(self, name):
                self.name = name

            def prepare(self, input_spec, context):
                pass

            def process(self, unit, context):
                yield unit

            def reset(self, boundary, context):
                pass

            def flush(self, context):
                return ()

            def close(self, context):
                closed.append((self.name, context.stage_id))

        module = sys.modules["fake_builder_families"]
        module.fakedenoise.build = (
            lambda config, *, context: Session("first"))
        original_upscale_build = module.fakeupscale.build

        def failing_build(config, *, context):
            raise RuntimeError("weights exploded")

        module.fakeupscale.build = failing_build
        try:
            config = {
                "pipeline": ["denoise", "upscale"],
                "denoise": {"processor": "fakedenoise"},
                "upscale": {"processor": "fakeupscale"},
            }
            plan = resolve_pipeline(config, input_spec=stream(),
                                    settings=SETTINGS)
            with pytest.raises(RuntimeError, match="weights exploded"):
                build_processors(plan, PipelineContext(settings=SETTINGS))
        finally:
            module.fakeupscale.build = original_upscale_build
        assert closed == [("first", "denoise")]


def _ctx_chain(exc):
    """Errors reachable via __cause__/__context__ from exc (cycle-safe)."""
    out, seen, node = [], set(), exc
    while node is not None and id(node) not in seen:
        out.append(node)
        seen.add(id(node))
        node = node.__cause__ or node.__context__
    return out


class TestBuildRollbackUnderInterrupts:
    @staticmethod
    def _plan_and_module(families):
        module = sys.modules["fake_builder_families"]
        config = {
            "pipeline": ["denoise", "upscale", "interp"],
            "denoise": {"processor": "fakedenoise"},
            "upscale": {"processor": "fakeupscale"},
            "interp": {"processor": "fakeinterp"},
        }
        return config, module

    def test_close_failure_during_rollback_does_not_stop_it(self, families):
        closed = []

        def make_session(name, error=None):
            class Session(Passthrough):
                def close(self, context):
                    closed.append(name)
                    if error is not None:
                        raise error

            return Session()

        config, module = self._plan_and_module(families)
        module.fakedenoise.build = (
            lambda cfg, *, context: make_session(
                "first", RuntimeError("close failed")))
        module.fakeupscale.build = (
            lambda cfg, *, context: make_session("second"))

        def failing_build(cfg, *, context):
            raise RuntimeError("weights exploded")

        module.fakeinterp.build = failing_build
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        with pytest.raises(RuntimeError, match="weights exploded"):
            build_processors(plan, PipelineContext(settings=SETTINGS))
        assert closed == ["second", "first"]

    @pytest.mark.parametrize("failure_type", [TypeError, AssertionError])
    def test_programmer_close_failure_is_preserved_during_rollback(
            self, families, failure_type):
        closed = []
        close_failure = failure_type("injected close defect")
        build_failure = RuntimeError("weights exploded")

        def make_session(name, error=None):
            class Session(Passthrough):
                def close(self, context):
                    closed.append(name)
                    if error is not None:
                        raise error

            return Session()

        config, module = self._plan_and_module(families)
        module.fakedenoise.build = (
            lambda cfg, *, context: make_session("first", close_failure))
        module.fakeupscale.build = (
            lambda cfg, *, context: make_session("second"))

        def failing_build(cfg, *, context):
            raise build_failure

        module.fakeinterp.build = failing_build
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)

        with pytest.raises(RuntimeError) as caught:
            build_processors(plan, PipelineContext(settings=SETTINGS))

        assert caught.value is build_failure
        assert closed == ["second", "first"]
        assert any(node is close_failure for node in _ctx_chain(caught.value))

    def test_interrupt_during_rollback_finishes_then_chains(self, families):
        closed = []

        def make_session(name, error=None):
            class Session(Passthrough):
                def close(self, context):
                    closed.append(name)
                    if error is not None:
                        raise error

            return Session()

        config, module = self._plan_and_module(families)
        module.fakedenoise.build = (
            lambda cfg, *, context: make_session(
                "first", KeyboardInterrupt()))
        module.fakeupscale.build = (
            lambda cfg, *, context: make_session("second"))

        def failing_build(cfg, *, context):
            raise RuntimeError("weights exploded")

        module.fakeinterp.build = failing_build
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        with pytest.raises(KeyboardInterrupt) as exc:
            build_processors(plan, PipelineContext(settings=SETTINGS))
        assert closed == ["second", "first"]  # rollback finished anyway
        # the build error is preserved on the delivered interrupt's chain
        assert any(isinstance(c, RuntimeError) and "weights exploded" in str(c)
                   for c in _ctx_chain(exc.value))

    def test_all_rollback_interrupts_are_preserved(self, families):
        # Two built stages raise interrupts on close during rollback: the first
        # in reverse ownership order wins, and the other remains reachable.
        def make_session(name, error=None):
            class Session(Passthrough):
                def close(self, context):
                    if error is not None:
                        raise error

            return Session()

        config, module = self._plan_and_module(families)
        module.fakedenoise.build = (
            lambda cfg, *, context: make_session(
                "first", KeyboardInterrupt("build-first")))
        module.fakeupscale.build = (
            lambda cfg, *, context: make_session(
                "second", SystemExit("build-second")))

        def failing_build(cfg, *, context):
            raise RuntimeError("weights exploded")

        module.fakeinterp.build = failing_build
        plan = resolve_pipeline(config, input_spec=stream(),
                                settings=SETTINGS)
        with pytest.raises(SystemExit, match="build-second") as exc:
            build_processors(plan, PipelineContext(settings=SETTINGS))
        chain = [str(c) for c in _ctx_chain(exc.value)]
        assert any("build-first" in s for s in chain)   # later interrupt kept
        assert any("weights exploded" in s for s in chain)  # build error kept
