"""The host session: open-time validation, once-only runs, early-close
cancellation, reporter plumbing, and layout parity through one path."""

from fractions import Fraction

import pytest

from kinovsr.pipeline import open_pipeline
from kinovsr.processors import (
    Capability,
    CapabilitySpec,
    FrameUnit,
    Geometry,
    PipelineError,
    StreamConstraint,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.processors import catalog as catalog_module
from kinovsr.processors.errors import StreamEdgeError
from kinovsr.reporting import RecordingReporter
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()


def spec(width: int = 64, height: int = 48, layout=None) -> StreamSpec:
    kwargs = {} if layout is None else {"layout": layout}
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(width, height),
            **kwargs),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


def units(n: int = 4, value: str = "frame"):
    for index in range(n):
        yield FrameUnit(payload=f"{value}-{index}", pts=index * 960,
                        duration=960)


class Tracker:
    """Lifecycle log shared by every fake stage instance."""

    def __init__(self) -> None:
        self.events: list[str] = []


class _FakeProcessor:
    def __init__(self, name: str, tracker: Tracker) -> None:
        self._name = name
        self._tracker = tracker

    def prepare(self, input_spec, context) -> None:
        self._tracker.events.append(f"prepare:{self._name}")
        self._reporter = context.reporter

    def process(self, unit, context):
        self._tracker.events.append(f"process:{self._name}")
        context.reporter.phase_advance(self._name)
        yield unit

    def reset(self, boundary, context) -> None:
        self._tracker.events.append(f"reset:{self._name}")

    def flush(self, context):
        self._tracker.events.append(f"flush:{self._name}")
        return ()

    def close(self, context) -> None:
        self._tracker.events.append(f"close:{self._name}")


class _FakeFactory:
    def __init__(self, name: str, tracker: Tracker) -> None:
        self.name = name
        self._tracker = tracker
        self.capabilities = {
            Capability.PREPROCESS: CapabilitySpec(
                capability=Capability.PREPROCESS,
                profiles=(),
                accepts=StreamConstraint(),
            ),
        }

    def parse_config(self, raw, *, capability, profile, settings):
        return {}

    def build(self, config, *, context):
        return _FakeProcessor(self.name, self._tracker)


@pytest.fixture
def tracker(monkeypatch) -> Tracker:
    """Register a close-tracking fake family for the test's duration."""
    log = Tracker()
    factory = _FakeFactory("fakestage", log)
    monkeypatch.setitem(catalog_module._FACTORY_TARGETS, "fakestage", "<test>")
    monkeypatch.setitem(catalog_module._loaded, "fakestage", factory)
    return log


CONFIG = {"pipeline": ["a", "b"],
          "a": {"processor": "fakestage"},
          "b": {"processor": "fakestage"}}


class TestOpen:
    def test_open_validates_before_any_processing(self, tracker):
        bad = {"pipeline": ["up"], "up": {"processor": "metalfx", "scale": 2,
                                          "nonsense": 1}}
        with pytest.raises(PipelineError):
            open_pipeline(bad, spec(), settings=SETTINGS)
        assert tracker.events == []   # nothing was built

    def test_edge_violation_surfaces_at_open(self):
        from kinovsr.processors import DType, Layout

        cv_spec = spec(layout=Layout.CV_NV12)
        config = {"pipeline": ["dn"],
                  "dn": {"processor": "fastdvdnet"}}
        with pytest.raises(StreamEdgeError):
            open_pipeline(config, cv_spec, settings=SETTINGS)
        assert DType  # silence unused-import pedantry

    def test_specs_exposed(self, tracker):
        session = open_pipeline(CONFIG, spec(), settings=SETTINGS)
        assert session.input_spec == spec()
        assert session.output_spec == spec()
        assert [s.name for s in session.plan.stages] == ["a", "b"]


class TestProcess:
    def test_units_flow_and_lifecycle_orders(self, tracker):
        session = open_pipeline(CONFIG, spec(), settings=SETTINGS)
        with session, session.process(units(3)) as run:
            outputs = list(run)
        assert len(outputs) == 3
        assert tracker.events.count("close:fakestage") == 2
        assert tracker.events.index("prepare:fakestage") < \
            tracker.events.index("process:fakestage")

    def test_session_is_once_only(self, tracker):
        session = open_pipeline(CONFIG, spec(), settings=SETTINGS)
        list(session.process(units(1)))
        with pytest.raises(PipelineError, match="already consumed"):
            session.process(units(1))

    def test_early_close_cancels_and_releases(self, tracker):
        session = open_pipeline(CONFIG, spec(), settings=SETTINGS)
        run = session.process(units(100))
        next(run)
        next(run)
        run.close()
        assert tracker.events.count("close:fakestage") == 2
        # closing again (via the session) stays safe
        session.close()
        assert tracker.events.count("close:fakestage") == 2

    def test_close_before_first_pull_releases(self, tracker):
        session = open_pipeline(CONFIG, spec(), settings=SETTINGS)
        session.process(units(5))
        session.close()
        assert tracker.events.count("close:fakestage") == 2

    def test_context_exit_cancels_midstream(self, tracker):
        session = open_pipeline(CONFIG, spec(), settings=SETTINGS)
        with session:
            run = session.process(units(50))
            next(run)
        assert tracker.events.count("close:fakestage") == 2

    def test_reporter_reaches_stages(self, tracker):
        reporter = RecordingReporter()
        session = open_pipeline(CONFIG, spec(), settings=SETTINGS,
                                reporter=reporter)
        list(session.process(units(2)))
        advances = [e for e in reporter.events if e[0] == "advance"]
        assert len(advances) == 4   # 2 units x 2 stages


class TestLayoutParity:
    """MLX and CVPixelBuffer layouts validate through the same path."""

    def test_same_chain_validates_for_both_layouts(self):
        from kinovsr.processors import Layout

        config = {"pipeline": ["fps"],
                  "fps": {"processor": "videotoolbox", "profile": "normal",
                          "target_fps": 50}}
        for layout in (Layout.CV_RGBA_HALF,):
            session = open_pipeline(config, spec(layout=layout),
                                    settings=SETTINGS)
            assert session.output_spec.timeline.cadence == Fraction(50)
        mlx_spec = spec()   # MLX layout refused by the CV-only family...
        with pytest.raises(StreamEdgeError):
            open_pipeline(config, mlx_spec, settings=SETTINGS)
        # ...and accepted by an MLX family, through the same validator.
        mlx_config = {"pipeline": ["up"],
                      "up": {"processor": "metalfx", "scale": 2}}
        session = open_pipeline(mlx_config, mlx_spec, settings=SETTINGS)
        assert session.output_spec.frame.geometry.width == 128
