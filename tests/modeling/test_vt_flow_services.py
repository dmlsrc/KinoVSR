"""Driver-owned VideoToolbox flow-service lifetime and concurrency."""

from __future__ import annotations

import threading

import pytest

pytestmark = pytest.mark.unit


class _Service:
    def __init__(self, key: tuple) -> None:
        self.key = key
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _manager(monkeypatch, capacity: int):
    from kinovsr.modeling.vt_flow import VtFlowServices

    made = []

    def make(width, height, backend="vt"):
        service = _Service((backend, width, height))
        made.append(service)
        return service

    monkeypatch.setattr(VtFlowServices, "_make_service", staticmethod(make))
    return VtFlowServices(capacity), made


def test_geometry_lru_is_bounded_reuses_and_closes_exactly_once(monkeypatch):
    services, made = _manager(monkeypatch, 2)

    with services.borrow(16, 12) as first:
        pass
    with services.borrow(32, 24) as second:
        pass
    with services.borrow(16, 12) as reused:
        assert reused is first
    with services.borrow(48, 36) as third:
        pass

    assert services.size == 2
    assert made == [first, second, third]
    assert second.close_count == 1
    assert first.close_count == third.close_count == 0

    services.close()
    services.close()

    assert first.close_count == second.close_count == third.close_count == 1



def test_active_oldest_entry_skips_to_idle_victim(monkeypatch):
    services, made = _manager(monkeypatch, 2)

    with services.borrow(16, 12) as active:
        with services.borrow(32, 24) as idle:
            pass
        with services.borrow(48, 36):
            pass
        assert active.close_count == 0
        assert idle.close_count == 1

    assert services.size == 2
    services.close()
    assert all(service.close_count == 1 for service in made)




def test_different_geometry_construction_can_overlap(monkeypatch):
    from kinovsr.modeling.vt_flow import VtFlowServices

    construction_barrier = threading.Barrier(2)
    made = []

    def make(width, height, backend="vt"):
        construction_barrier.wait(2)
        service = _Service((width, height))
        made.append(service)
        return service

    monkeypatch.setattr(VtFlowServices, "_make_service", staticmethod(make))
    services = VtFlowServices(2)

    first = threading.Thread(target=lambda: _borrow_once(services, 16, 12))
    second = threading.Thread(target=lambda: _borrow_once(services, 32, 24))
    first.start()
    second.start()
    first.join(2)
    second.join(2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(made) == 2
    services.close()






def test_close_continues_after_service_failure(monkeypatch):
    from kinovsr.modeling.vt_flow import VtFlowServices

    class FailingService(_Service):
        def close(self):
            super().close()
            raise RuntimeError("service close failed")

    first = FailingService((16, 12))
    second = _Service((32, 24))
    queue = [first, second]
    monkeypatch.setattr(
        VtFlowServices,
        "_make_service",
        staticmethod(lambda *_args: queue.pop(0)),
    )
    services = VtFlowServices(2)
    _borrow_once(services, 16, 12)
    _borrow_once(services, 32, 24)

    with pytest.raises(RuntimeError, match="service close failed"):
        services.close()
    services.close()

    assert first.close_count == second.close_count == 1



def test_windowed_driver_close_releases_owned_manager(monkeypatch):
    from kinovsr.modeling import upscaler_base, vt_flow

    created = []

    class Manager:
        def __init__(self, capacity):
            self.capacity = capacity
            self.close_count = 0
            created.append(self)

        def close(self):
            self.close_count += 1

    monkeypatch.setattr(vt_flow, "VtFlowServices", Manager)
    driver = upscaler_base.WindowedUpscaler(window=5, trim=1, vt_flow_geometries=2)
    driver._frames = [object()]

    driver.close()
    driver.close()

    assert created[0].capacity == 2
    assert created[0].close_count == 1
    assert driver._frames == []
    assert driver._vt_flow_services is None


def _borrow_once(services, width, height):
    with services.borrow(width, height):
        pass


def test_temporary_scope_closes_and_preserves_active_error(monkeypatch):
    from kinovsr.modeling import vt_flow

    managers = []

    class Manager:
        def __init__(self, capacity):
            self.capacity = capacity
            self.close_count = 0
            managers.append(self)

        def close(self):
            self.close_count += 1
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(vt_flow, "VtFlowServices", Manager)

    with (
        pytest.raises(ValueError, match="compute failed") as caught,
        vt_flow.vt_flow_services_scope(None, max_geometries=2),
    ):
        raise ValueError("compute failed")

    assert isinstance(caught.value.__context__, RuntimeError)
    assert caught.value.__context__.__context__ is None
    assert managers[0].capacity == 2
    assert managers[0].close_count == 1


def test_single_frame_vt_flow_constructs_no_service(monkeypatch):
    import mlx.core as mx

    from kinovsr.modeling import vsr_blocks, vt_flow

    class ForbiddenManager:
        def __init__(self, *_args):
            pytest.fail("one frame has no flow pairs")

    monkeypatch.setattr(vt_flow, "VtFlowServices", ForbiddenManager)
    frame = mx.zeros((1, 12, 16, 3))

    assert vsr_blocks._vt_flows([frame]) == ([], [])


def test_closed_during_construction_never_publishes(monkeypatch):
    from kinovsr.modeling.vt_flow import VtFlowServices

    services = VtFlowServices(1)
    service = _Service((16, 12))

    def make(_width, _height, _backend="vt"):
        # A host closes the manager while a service is mid-construction.
        services.close()
        return service

    monkeypatch.setattr(VtFlowServices, "_make_service", staticmethod(make))
    with (
        pytest.raises(RuntimeError, match="closed during construction"),
        services.borrow(16, 12),
    ):
        pytest.fail("a service created during close must not publish")
    assert service.close_count == 1
    assert services.size == 0


def test_closed_during_construction_chains_cleanup_failure(monkeypatch):
    from kinovsr.modeling.vt_flow import VtFlowServices

    services = VtFlowServices(1)
    cleanup_failure = OSError("unpublished cleanup failed")

    class _FailingClose:
        def close(self):
            raise cleanup_failure

    def make(_width, _height, _backend="vt"):
        services.close()
        return _FailingClose()

    monkeypatch.setattr(VtFlowServices, "_make_service", staticmethod(make))
    with (
        pytest.raises(RuntimeError, match="closed during construction") as caught,
        services.borrow(16, 12),
    ):
        pytest.fail("must not publish")
    context_chain = []
    node = caught.value.__context__
    while node is not None:
        context_chain.append(node)
        node = node.__context__
    assert cleanup_failure in context_chain


def test_double_borrow_of_one_geometry_raises(monkeypatch):
    services, _made = _manager(monkeypatch, 1)
    with (
        services.borrow(16, 12),
        pytest.raises(RuntimeError, match="already borrowed"),
        services.borrow(16, 12),
    ):
        pytest.fail("exclusive lease must not be shared")
    services.close()


def test_eviction_with_every_service_borrowed_raises(monkeypatch):
    services, _made = _manager(monkeypatch, 1)
    with (
        services.borrow(16, 12),
        pytest.raises(RuntimeError, match="raise max_geometries"),
        services.borrow(32, 24),
    ):
        pytest.fail("no idle victim exists")
    services.close()


def test_close_with_live_borrow_raises_then_close_succeeds(monkeypatch):
    services, _made = _manager(monkeypatch, 1)
    with services.borrow(16, 12), \
            pytest.raises(RuntimeError, match="while a service is borrowed"):
        services.close()
    services.close()
    assert services.size == 0


def test_backend_is_part_of_the_service_key(monkeypatch):
    services, made = _manager(monkeypatch, 2)

    with services.borrow(16, 12) as vt_svc:
        pass
    with services.borrow(16, 12, backend="vision") as vision_svc:
        pass
    assert vt_svc is not vision_svc
    assert vt_svc.key == ("vt", 16, 12)
    assert vision_svc.key == ("vision", 16, 12)
    with services.borrow(16, 12) as again:
        assert again is vt_svc
    assert services.size == 2
    services.close()


def test_unknown_backend_is_rejected(monkeypatch):
    services, _ = _manager(monkeypatch, 1)
    with (
        pytest.raises(ValueError, match="flow backend"),
        services.borrow(16, 12, backend="raft"),
    ):
        pass
    services.close()
