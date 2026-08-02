"""Recurrent VSR drivers forward their owned VT flow-service manager."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _Manager:
    instances = []

    def __init__(self, capacity):
        self.capacity = capacity
        self.close_count = 0
        self.instances.append(self)

    def close(self):
        self.close_count += 1


def test_driver_capacities_and_independent_ownership(monkeypatch):
    from kinovsr.modeling import vt_flow
    from kinovsr.processors.basicvsrpp import restorer, upscaler
    from kinovsr.processors.realbasicvsr import upscaler as real_upscaler

    _Manager.instances = []
    monkeypatch.setattr(vt_flow, "VtFlowServices", _Manager)
    monkeypatch.setattr(upscaler.net, "resolve_weights", lambda _value: object())
    monkeypatch.setattr(upscaler.net, "load_params", lambda _value: {})
    monkeypatch.setattr(restorer.net, "resolve_restore_weights", lambda _value: object())
    monkeypatch.setattr(restorer.net, "load_params", lambda _value: {})
    monkeypatch.setattr(restorer.net, "is_low_res_input", lambda _params: False)
    monkeypatch.setattr(real_upscaler.net, "resolve_weights", lambda _value: object())
    monkeypatch.setattr(real_upscaler.net, "load_params", lambda _value: {})

    first = upscaler.BasicVsrUpscaler(flow_mode="vt")
    second = upscaler.BasicVsrUpscaler(flow_mode="vt")
    ensemble = upscaler.BasicVsrUpscaler(flow_mode="vt", ensemble=True)
    restore = restorer.BasicVsrRestorer(flow_mode="vt", ensemble=True)
    real = real_upscaler.RealBasicVsrUpscaler(flow_mode="vt")
    spynet = upscaler.BasicVsrUpscaler(flow_mode="spynet")

    assert [manager.capacity for manager in _Manager.instances] == [1, 1, 2, 2, 1]
    assert first._vt_flow_services is not second._vt_flow_services
    assert spynet._vt_flow_services is None

    first.close()
    second.close()
    ensemble.close()
    restore.close()
    real.close()
    spynet.close()
    assert all(manager.close_count == 1 for manager in _Manager.instances)


def test_basicvsr_upscale_forwards_owned_services(monkeypatch):
    from kinovsr.processors.basicvsrpp import upscaler

    driver = upscaler.BasicVsrUpscaler.__new__(upscaler.BasicVsrUpscaler)
    driver._ensemble = False
    driver._p = object()
    driver._flow_mode = "vt"
    driver._history_strength = 1.0
    driver._history_gate = "off"
    driver._vt_flow_services = object()
    observed = {}

    def run(*_args, **kwargs):
        observed.update(kwargs)
        return []

    monkeypatch.setattr(upscaler.net, "upscale", run)
    list(driver._upscale_window([]))

    assert observed["vt_flow_services"] is driver._vt_flow_services


def test_basicvsr_restore_forwards_owned_services(monkeypatch):
    from kinovsr.processors.basicvsrpp import restorer

    driver = restorer.BasicVsrRestorer.__new__(restorer.BasicVsrRestorer)
    driver._ensemble = True
    driver._p = object()
    driver._flow_mode = "vt"
    driver._strength = 1.0
    driver._vt_flow_services = object()
    observed = {}

    def run(*_args, **kwargs):
        observed.update(kwargs)
        return []

    monkeypatch.setattr(restorer.net, "restore_ensemble", run)
    list(driver._upscale_window([]))

    assert observed["vt_flow_services"] is driver._vt_flow_services


def test_realbasicvsr_forwards_owned_services(monkeypatch):
    from kinovsr.processors.realbasicvsr import upscaler

    driver = upscaler.RealBasicVsrUpscaler.__new__(upscaler.RealBasicVsrUpscaler)
    driver._p = object()
    driver._dynamic_refine_thres = 5.0
    driver._clean_iters = 1
    driver._residual_strength = 1.0
    driver._flow_consistency = 0.0
    driver._flow_mode = "vt"
    driver._history_strength = 1.0
    driver._history_gate = "off"
    driver._vt_flow_services = object()
    observed = {}

    def run(*_args, **kwargs):
        observed.update(kwargs)
        return []

    monkeypatch.setattr(upscaler.net, "upscale", run)
    list(driver._upscale_window([]))

    assert observed["vt_flow_services"] is driver._vt_flow_services
