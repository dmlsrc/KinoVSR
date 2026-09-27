"""Pure contract tests for the direct anemil dispatch route's residency claim."""

import pytest

from kinovsr.native.anemil import direct

pytestmark = pytest.mark.unit


class _Bridge:
    """Stands in for the in-memory model; each call can be made to raise."""

    fail_load = False
    fail_unload = False

    def loadWithQoS_options_error_(self, qos, options, error):
        del qos, options, error
        if self.fail_load:
            raise RuntimeError("load bridge failed")
        return True, None

    def unloadWithQoS_error_(self, qos, error):
        del qos, error
        if self.fail_unload:
            raise RuntimeError("unload bridge failed")
        return True, None


def _model(label: str) -> direct.DirectModel:
    model = object.__new__(direct.DirectModel)
    model.label = label
    model.model = _Bridge()
    model._loaded = False
    return model


def test_a_raising_load_releases_the_claim(monkeypatch):
    # The claim used to stay set, so every later direct-route load in the
    # process was refused as "still resident".
    monkeypatch.setattr(direct, "_RESIDENT", None)
    first = _model("first")
    first.model.fail_load = True

    with pytest.raises(RuntimeError, match="load bridge failed"):
        first.load()
    assert direct._RESIDENT is None
    assert not first._loaded

    second = _model("second")
    second.load()
    try:
        assert direct._RESIDENT == "second"
    finally:
        second.unload()


def test_a_raising_unload_releases_the_claim(monkeypatch):
    monkeypatch.setattr(direct, "_RESIDENT", None)
    model = _model("entry")
    model.load()
    model.model.fail_unload = True

    with pytest.raises(RuntimeError, match="unload bridge failed"):
        model.unload()
    assert not model._loaded
    assert direct._RESIDENT is None
