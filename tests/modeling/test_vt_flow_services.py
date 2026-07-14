"""Driver-owned VideoToolbox flow-service lifetime and concurrency."""

from __future__ import annotations

import threading

import pytest

pytestmark = pytest.mark.unit


class _Service:
    def __init__(self, key: tuple[int, int]) -> None:
        self.key = key
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _manager(monkeypatch, capacity: int):
    from kinovsr.modeling.vt_flow import VtFlowServices

    made = []

    def make(width, height):
        service = _Service((width, height))
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


def test_active_entry_blocks_eviction_until_lease_releases(monkeypatch):
    services, made = _manager(monkeypatch, 1)
    active = threading.Event()
    release = threading.Event()
    replacement_acquired = threading.Event()

    def hold_first():
        with services.borrow(16, 12):
            active.set()
            assert release.wait(2)

    def borrow_replacement():
        with services.borrow(32, 24):
            replacement_acquired.set()

    first_thread = threading.Thread(target=hold_first)
    first_thread.start()
    assert active.wait(2)
    second_thread = threading.Thread(target=borrow_replacement)
    second_thread.start()

    assert not replacement_acquired.wait(0.05)
    assert made[0].close_count == 0
    release.set()
    first_thread.join(2)
    second_thread.join(2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert replacement_acquired.is_set()
    assert made[0].close_count == 1
    services.close()


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


def test_same_geometry_waiter_is_reserved_and_serialized(monkeypatch):
    services, _ = _manager(monkeypatch, 1)
    first_acquired = threading.Event()
    release = threading.Event()
    second_acquired = threading.Event()

    def first():
        with services.borrow(16, 12):
            first_acquired.set()
            assert release.wait(2)

    def second():
        with services.borrow(16, 12):
            second_acquired.set()

    first_thread = threading.Thread(target=first)
    first_thread.start()
    assert first_acquired.wait(2)
    second_thread = threading.Thread(target=second)
    second_thread.start()
    with services._condition:
        assert services._condition.wait_for(
            lambda: services._entries[(16, 12)].users == 2,
            timeout=2,
        )
    assert not second_acquired.is_set()

    release.set()
    first_thread.join(2)
    second_thread.join(2)
    assert second_acquired.is_set()
    services.close()


def test_same_geometry_construction_is_single_flight(monkeypatch):
    from kinovsr.modeling.vt_flow import VtFlowServices

    factory_entered = threading.Event()
    release_factory = threading.Event()
    factory_calls = []
    service = _Service((16, 12))

    def make(width, height):
        factory_calls.append((width, height))
        factory_entered.set()
        assert release_factory.wait(2)
        return service

    monkeypatch.setattr(VtFlowServices, "_make_service", staticmethod(make))
    services = VtFlowServices(1)
    completed = []

    def use():
        with services.borrow(16, 12) as borrowed:
            completed.append(borrowed)

    first = threading.Thread(target=use)
    second = threading.Thread(target=use)
    first.start()
    assert factory_entered.wait(2)
    second.start()
    second.join(0.05)
    assert second.is_alive()
    assert factory_calls == [(16, 12)]

    release_factory.set()
    first.join(2)
    second.join(2)
    assert completed == [service, service]
    assert factory_calls == [(16, 12)]
    services.close()


def test_different_geometry_construction_can_overlap(monkeypatch):
    from kinovsr.modeling.vt_flow import VtFlowServices

    construction_barrier = threading.Barrier(2)
    made = []

    def make(width, height):
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


def test_close_waits_for_active_lease(monkeypatch):
    services, made = _manager(monkeypatch, 1)
    active = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def use():
        with services.borrow(16, 12):
            active.set()
            assert release.wait(2)

    def close():
        services.close()
        closed.set()

    user = threading.Thread(target=use)
    user.start()
    assert active.wait(2)
    closer = threading.Thread(target=close)
    closer.start()
    assert not closed.wait(0.05)
    assert made[0].close_count == 0

    release.set()
    user.join(2)
    closer.join(2)
    assert closed.is_set()
    assert made[0].close_count == 1


def test_interrupted_close_wait_can_be_retried(monkeypatch):
    services, made = _manager(monkeypatch, 1)
    active = threading.Event()
    release = threading.Event()

    class InterruptOnceCondition(threading.Condition):
        interrupt_next_wait = False

        def wait(self, timeout=None):
            if self.interrupt_next_wait:
                self.interrupt_next_wait = False
                raise KeyboardInterrupt("close wait interrupted")
            return super().wait(timeout)

    condition = InterruptOnceCondition()
    services._condition = condition

    def use():
        with services.borrow(16, 12):
            active.set()
            assert release.wait(2)

    user = threading.Thread(target=use)
    user.start()
    assert active.wait(2)

    condition.interrupt_next_wait = True
    with pytest.raises(KeyboardInterrupt, match="close wait interrupted"):
        services.close()

    release.set()
    user.join(2)
    assert not user.is_alive()

    services.close()
    services.close()
    assert services.size == 0
    assert made[0].close_count == 1


def test_close_racing_construction_closes_unpublished_service(monkeypatch):
    from kinovsr.modeling.vt_flow import VtFlowServices

    factory_entered = threading.Event()
    release_factory = threading.Event()
    service = _Service((16, 12))

    def make(_width, _height):
        factory_entered.set()
        assert release_factory.wait(2)
        return service

    monkeypatch.setattr(VtFlowServices, "_make_service", staticmethod(make))
    services = VtFlowServices(1)
    errors = []

    def borrow():
        try:
            with services.borrow(16, 12):
                pytest.fail("a service created during close must not publish")
        except RuntimeError as exc:
            errors.append(str(exc))

    borrower = threading.Thread(target=borrow)
    borrower.start()
    assert factory_entered.wait(2)
    closer = threading.Thread(target=services.close)
    closer.start()
    with services._condition:
        assert services._condition.wait_for(lambda: services._closed, timeout=2)
    release_factory.set()
    borrower.join(2)
    closer.join(2)

    assert errors == ["VT optical-flow services closed during construction"]
    assert service.close_count == 1
    assert services.size == 0


def test_close_racing_construction_preserves_cleanup_failure_as_context(monkeypatch):
    from kinovsr.modeling.vt_flow import VtFlowServices

    factory_entered = threading.Event()
    release_factory = threading.Event()

    class FailingCloseService(_Service):
        def close(self) -> None:
            self.close_count += 1
            raise OSError("unpublished cleanup failed")

    service = FailingCloseService((16, 12))

    def make(_width, _height):
        factory_entered.set()
        assert release_factory.wait(2)
        return service

    monkeypatch.setattr(VtFlowServices, "_make_service", staticmethod(make))
    services = VtFlowServices(1)
    errors = []

    def borrow():
        try:
            with services.borrow(16, 12):
                pytest.fail("a service created during close must not publish")
        except RuntimeError as exc:
            errors.append(exc)

    borrower = threading.Thread(target=borrow)
    borrower.start()
    assert factory_entered.wait(2)
    closer = threading.Thread(target=services.close)
    closer.start()
    with services._condition:
        assert services._condition.wait_for(lambda: services._closed, timeout=2)
    release_factory.set()
    borrower.join(2)
    closer.join(2)

    assert len(errors) == 1
    assert str(errors[0]) == "VT optical-flow services closed during construction"
    assert isinstance(errors[0].__context__, OSError)
    assert str(errors[0].__context__) == "unpublished cleanup failed"
    assert errors[0].__context__.__context__ is None
    assert service.close_count == 1
    assert services.size == 0


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


def test_interrupted_use_lock_acquire_releases_reservation(monkeypatch):
    services, made = _manager(monkeypatch, 1)
    with services.borrow(16, 12):
        pass
    entry = services._entries[(16, 12)]

    class InterruptedLock:
        def acquire(self):
            raise KeyboardInterrupt

        def release(self):
            pytest.fail("an unacquired lock must not be released")

    entry.use_lock = InterruptedLock()
    with pytest.raises(KeyboardInterrupt), services.borrow(16, 12):
        pass

    assert entry.users == 0
    services.close()
    assert made[0].close_count == 1


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
