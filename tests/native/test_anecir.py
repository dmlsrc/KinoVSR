"""Pure contract tests for the explicit ANECIR runtime."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kinovsr.native import anecir
from kinovsr.native.anemil import direct

pytestmark = pytest.mark.unit


def test_live_port_table_order_overrides_numeric_netplist_order():
    infos = [
        {"Symbol": "__arg0"},
        {"Symbol": "__arg10"},
        {"Symbol": "__arg2"},
    ]

    ordered = anecir._ordered_infos(
        label="entry",
        semantic_names=("frame", "gate", "skip"),
        product_symbols=("__arg0", "__arg2", "__arg10"),
        live_infos=infos,
        kind="input",
    )

    assert [name for name, _info in ordered] == ["frame", "skip", "gate"]


def test_live_port_table_rejects_missing_or_duplicate_symbols():
    with pytest.raises(RuntimeError, match="live output symbols changed"):
        anecir._ordered_infos(
            label="entry",
            semantic_names=("first", "second"),
            product_symbols=("__out:0", "__out:1"),
            live_infos=(
                {"Symbol": "__out:0"},
                {"Symbol": "__out:0"},
            ),
            kind="output",
        )


def test_raw_model_populates_a_missing_deterministic_cache(monkeypatch):
    calls = []

    class CacheMiss:
        @staticmethod
        def code():
            return 16

    class Client:
        def loadModel_options_qos_error_(self, model, options, qos, error):
            del model, qos, error
            calls.append(bool(options[anecir._CACHE_FLAG]))
            return (False, CacheMiss()) if len(calls) == 1 else (True, None)

        @staticmethod
        def unloadModel_options_qos_error_(model, options, qos, error):
            del model, options, qos, error
            return True, None

    model = object.__new__(anecir._ANECIRModel)
    model.label = "entry"
    model._client = Client()
    model._options = {anecir._CACHE_FLAG: True}
    model.model = object()
    model._active_options = None
    model._refresh_after_unload = False
    model._loaded = False
    model._mapped_requests = {}
    monkeypatch.setattr(model, "_new_model", lambda: object())
    monkeypatch.setattr(anecir, "_RESIDENT", None)

    model.load()
    try:
        assert calls == [True, False]
        assert model._loaded
        assert model._active_options[anecir._CACHE_FLAG] is False
    finally:
        model.unload()
    assert anecir._RESIDENT is None


def test_raw_model_releases_residency_when_load_or_unload_raises(monkeypatch):
    class Client:
        fail_load = True

        def loadModel_options_qos_error_(self, model, options, qos, error):
            del model, options, qos, error
            if self.fail_load:
                raise RuntimeError("load bridge failed")
            return True, None

        @staticmethod
        def unloadModel_options_qos_error_(model, options, qos, error):
            del model, options, qos, error
            raise RuntimeError("unload bridge failed")

    model = object.__new__(anecir._ANECIRModel)
    model.label = "entry"
    model._client = Client()
    model._options = {anecir._CACHE_FLAG: True}
    model.model = object()
    model._active_options = None
    model._refresh_after_unload = False
    model._loaded = False
    model._mapped_requests = {}
    monkeypatch.setattr(anecir, "_RESIDENT", None)

    with pytest.raises(RuntimeError, match="load bridge failed"):
        model.load()
    assert anecir._RESIDENT is None

    model._client.fail_load = False
    model.load()
    with pytest.raises(RuntimeError, match="unload bridge failed"):
        model.unload()
    assert not model._loaded
    assert anecir._RESIDENT is None


def test_raw_model_reuses_request_for_the_same_surface_map(monkeypatch):
    requests = []

    class Request:
        @staticmethod
        def requestWithInputs_inputIndices_outputs_outputIndices_procedureIndex_(
            inputs, input_indices, outputs, output_indices, procedure
        ):
            request = object()
            requests.append((
                tuple(inputs), tuple(input_indices),
                tuple(outputs), tuple(output_indices), procedure, request,
            ))
            return request

    monkeypatch.setattr(
        anecir.direct,
        "preflight",
        lambda: SimpleNamespace(request_class=Request),
    )
    model = object.__new__(anecir._ANECIRModel)
    model.label = "entry"
    model._requests = {}
    first = SimpleNamespace(wrapped=object())
    second = SimpleNamespace(wrapped=object())
    inputs = (SimpleNamespace(surface=first),)
    outputs = (SimpleNamespace(surface=second),)

    initial = model._request(inputs, outputs)
    repeated = model._request(inputs, outputs)
    changed = model._request(
        (SimpleNamespace(surface=SimpleNamespace(wrapped=object())),),
        outputs,
    )

    assert repeated is initial
    assert changed is not initial
    assert len(requests) == 2


def test_raw_model_maps_a_cached_request_once(monkeypatch):
    calls = []
    request = object()

    class Client:
        @staticmethod
        def mapIOSurfacesWithModel_request_cacheInference_error_(
            model, mapped_request, cache, error
        ):
            del model, cache, error
            calls.append(("map", mapped_request))
            return True, None

        @staticmethod
        def evaluateWithModel_options_request_qos_error_(
            model, options, evaluated_request, qos, error
        ):
            del model, options, qos, error
            calls.append(("evaluate", evaluated_request))
            return True, None

    model = object.__new__(anecir._ANECIRModel)
    model.label = "entry"
    model.model = object()
    model._client = Client()
    model._loaded = True
    model._mapped_requests = {}
    monkeypatch.setattr(model, "_request", lambda inputs, outputs: request)

    model.evaluate((), ())
    model.evaluate((), ())

    assert calls == [
        ("map", request),
        ("evaluate", request),
        ("evaluate", request),
    ]


def test_stateful_executable_retains_one_model_until_entry_switch():
    events = []

    class Model:
        def __init__(self, name):
            self.name = name

        def load(self):
            events.append((self.name, "load"))

        def evaluate(self, inputs, outputs):
            events.append((self.name, "evaluate", inputs, outputs))

        def unload(self):
            events.append((self.name, "unload"))

    executable = object.__new__(anecir.StatefulExecutable)
    executable._closed = False
    executable._active_model = None
    first = Model("first")
    second = Model("second")

    executable._evaluate(first, (1,), (2,))
    executable._evaluate(first, (3,), (4,))
    executable._evaluate(second, (5,), (6,))

    assert events == [
        ("first", "load"),
        ("first", "evaluate", (1,), (2,)),
        ("first", "evaluate", (3,), (4,)),
        ("first", "unload"),
        ("second", "load"),
        ("second", "evaluate", (5,), (6,)),
    ]


def test_port_packs_and_unpacks_strided_fp16_lanes():
    class Surface:
        def __init__(self, size):
            self.nbytes = size
            self.storage = bytearray(size)

        def lock(self, readonly=False):
            del readonly

        def unlock(self, readonly=False):
            del readonly

        def view(self):
            return memoryview(self.storage)

    info = {
        "Batches": 1,
        "Channels": 16,
        "Depth": 1,
        "Height": 1,
        "Width": 1,
        "BatchStride": 1024,
        "DepthStride": 1024,
        "PlaneStride": 64,
        "RowStride": 64,
    }
    surface = Surface(1024)
    port = direct.Port("gate", info, surface)
    payload = bytes(range(32))

    port.write(payload)

    assert port.read() == payload
    assert all(
        surface.storage[lane * 64:lane * 64 + 2]
        == payload[lane * 2:lane * 2 + 2]
        for lane in range(16)
    )
    assert set(port.probe_offsets()) <= {lane * 64 for lane in range(16)}


class _GuardSurface:
    def __init__(self, size):
        self.storage = bytearray(size)
        self.locks = []

    def lock(self, readonly=False):
        self.locks.append(("lock", readonly))

    def unlock(self, readonly=False):
        self.locks.append(("unlock", readonly))

    def view(self):
        return memoryview(self.storage)


class _GuardPort:
    def __init__(self, name, size):
        self.name = name
        self.nbytes = size
        self.surface = _GuardSurface(size)

    @staticmethod
    def probe_offsets():
        return 0, 2, 4


def _guard_entry(evaluate):
    entry = object.__new__(anecir.StatefulEntry)
    entry._contract = SimpleNamespace(name="entry")
    entry._model = object()
    entry._owner = SimpleNamespace(_evaluate=evaluate)
    return entry


def test_request_liveness_guard_checks_only_the_smallest_result():
    large = _GuardPort("large", 64)
    small = _GuardPort("small", 16)

    def evaluate(_model, _inputs, outputs):
        for port in outputs:
            port.surface.storage[:] = bytes(len(port.surface.storage))

    entry = _guard_entry(evaluate)
    entry._guarded_dispatch((), (large, small), 1)

    assert large.surface.locks == []
    assert small.surface.locks == [
        ("lock", False),
        ("unlock", False),
        ("lock", True),
        ("unlock", True),
    ]


def test_request_liveness_guard_rejects_a_successful_no_op():
    entry = _guard_entry(lambda _model, _inputs, _outputs: None)

    with pytest.raises(RuntimeError, match="did not completely write"):
        entry._guarded_dispatch((), (_GuardPort("result", 16),), 1)
