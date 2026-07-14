"""Terminal writer-pool offers stay scoped to the final native stage."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest

from kinovsr.processors import Geometry, PipelineContext, StreamSpec, TimelineSpec
from kinovsr.processors.errors import PipelineError
from kinovsr.processors.specs import frame_spec_for_matrix
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


class _Processor:
    def __init__(self) -> None:
        self.bindings = []

    def _bind_output_pool(self, *binding) -> None:
        self.bindings.append(binding)


class _Run:
    def __init__(self, finalizers=()) -> None:
        self._finalizers = finalizers
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for finalizer in reversed(self._finalizers):
            finalizer()


def _spec() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix("bt709", full_range=False, geometry=Geometry(8, 6)),
        timeline=TimelineSpec(time_base=Fraction(1, 24000), cadence=Fraction(25)),
    )


def test_pipeline_session_offers_pool_only_to_terminal_stage(monkeypatch):
    from kinovsr.pipeline import session as session_module

    first, terminal = _Processor(), _Processor()
    built = (
        (SimpleNamespace(name="first"), first),
        (SimpleNamespace(name="terminal"), terminal),
    )
    monkeypatch.setattr(session_module, "build_processors", lambda *_args: built)
    monkeypatch.setattr(
        session_module,
        "run_chain",
        lambda *_args, **kwargs: _Run(kwargs.get("finalizers", ())),
    )
    spec = _spec()
    plan = SimpleNamespace(input_spec=spec, output_spec=spec)
    session = session_module.PipelineSession(plan, PipelineContext(settings=Settings()))
    pool = object()

    session._bind_terminal_output_pool(pool, 123, 8, 6)
    session.process([], retain_outputs=False)

    assert first.bindings == []
    assert terminal.bindings == [(pool, 123, 8, 6)]
    assert session._terminal_pool_binding is None
    with pytest.raises(PipelineError, match="before processing"):
        session._bind_terminal_output_pool(pool, 123, 8, 6)
    session.close()


def test_pipeline_session_drops_pool_offer_when_build_fails(monkeypatch):
    from kinovsr.pipeline import session as session_module

    def fail_build(*_args):
        raise RuntimeError("build failed")

    monkeypatch.setattr(session_module, "build_processors", fail_build)
    spec = _spec()
    plan = SimpleNamespace(input_spec=spec, output_spec=spec)
    session = session_module.PipelineSession(plan, PipelineContext(settings=Settings()))
    session._bind_terminal_output_pool(object(), 123, 8, 6)

    with pytest.raises(RuntimeError, match="build failed"):
        session.process([], retain_outputs=False)

    assert session._terminal_pool_binding is None


def test_pipeline_session_closes_owner_when_pool_binding_fails(monkeypatch):
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.pipeline import session as session_module

    closed = []

    class Lease:
        def close(self):
            closed.append("owner")

    class BadTerminal(_Processor):
        def _bind_output_pool(self, *_binding) -> None:
            raise RuntimeError("binding failed")

    monkeypatch.setattr(pb, "ci_cache_owner", Lease)
    built = (
        (SimpleNamespace(name="first"), _Processor()),
        (SimpleNamespace(name="terminal"), BadTerminal()),
    )
    monkeypatch.setattr(session_module, "build_processors", lambda *_args: built)
    monkeypatch.setattr(
        session_module,
        "run_chain",
        lambda *_args, **kwargs: _Run(kwargs.get("finalizers", ())),
    )
    stream_spec = _spec()
    plan = SimpleNamespace(input_spec=stream_spec, output_spec=stream_spec)
    session = session_module.PipelineSession(
        plan,
        PipelineContext(settings=Settings()),
    )
    session._bind_terminal_output_pool(object(), 123, 8, 6)

    with pytest.raises(RuntimeError, match="binding failed"):
        session.process([], retain_outputs=False)

    assert closed == ["owner"]
    assert session._run is None


def test_pipeline_session_close_drops_unconsumed_pool_offer():
    from kinovsr.pipeline import session as session_module

    spec = _spec()
    plan = SimpleNamespace(input_spec=spec, output_spec=spec)
    session = session_module.PipelineSession(plan, PipelineContext(settings=Settings()))
    session._bind_terminal_output_pool(object(), 123, 8, 6)

    session.close()

    assert session._terminal_pool_binding is None
