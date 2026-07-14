"""Core Image cache accounting, ownership, and maintenance concurrency."""

from __future__ import annotations

import importlib
import threading
from types import SimpleNamespace

import pytest

from kinovsr.media import pixel_buffers as pb


class _FakeContext:
    def __init__(self) -> None:
        self.clear_calls = 0
        self.clear_at_pool_exits: list[int] = []
        self.fail_with: BaseException | None = None
        self.pool_exits: list[int] | None = None

    def clearCaches(self) -> None:
        if self.pool_exits is not None:
            self.clear_at_pool_exits.append(len(self.pool_exits))
        self.clear_calls += 1
        if self.fail_with is not None:
            failure, self.fail_with = self.fail_with, None
            raise failure


class _FakePool:
    def __init__(self, enters: list[int], exits: list[int]) -> None:
        self._enters = enters
        self._exits = exits

    def __enter__(self):
        self._enters.append(len(self._enters))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._exits.append(len(self._exits))
        return False


@pytest.fixture
def fake_ci(monkeypatch):
    janitor = pb._CiCacheJanitor(interval=64)
    context = _FakeContext()
    enters: list[int] = []
    exits: list[int] = []
    context.pool_exits = exits
    monkeypatch.setattr(pb, "_ci_janitor", janitor)
    monkeypatch.setattr(pb, "_ci_context", context)
    monkeypatch.setattr(
        pb,
        "autorelease_pool",
        lambda: _FakePool(enters, exits),
    )
    return janitor, context, enters, exits


@pytest.mark.unit
def test_periodic_clear_tracks_render_work_and_drains_pool_first(fake_ci):
    janitor, context, enters, exits = fake_ci

    for _ in range(63):
        with pb.ci_render_scope() as rendered_by:
            assert rendered_by is context
    assert context.clear_calls == 0

    with pb.ci_render_scope():
        pass
    assert context.clear_calls == 1

    for _ in range(64):
        with pb.ci_render_scope():
            pass
    assert context.clear_calls == 2
    assert context.clear_at_pool_exits == [64, 128]
    assert len(enters) == len(exits) == 128
    assert janitor._snapshot() == (0, 128, 128, False, 0)


@pytest.mark.unit
def test_nested_owners_clear_only_after_outer_release(fake_ci):
    janitor, context, _enters, _exits = fake_ci
    outer = pb.ci_cache_owner()
    inner = pb.ci_cache_owner()

    with pb.ci_render_scope():
        pass
    inner.close()
    inner.close()
    assert context.clear_calls == 0
    assert janitor._snapshot()[-1] == 1

    # Models file work after the inner ChainRun/session has exhausted.
    with pb.ci_render_scope():
        pass
    outer.close()
    assert context.clear_calls == 1
    assert janitor._snapshot() == (0, 2, 2, False, 0)


@pytest.mark.unit
def test_unused_owner_does_not_create_or_clear_context(monkeypatch):
    janitor = pb._CiCacheJanitor(interval=64)
    monkeypatch.setattr(pb, "_ci_janitor", janitor)
    monkeypatch.setattr(pb, "_ci_context", None)

    pb.ci_cache_owner().close()
    pb.clear_ci_caches()

    assert pb._ci_context is None
    assert janitor._snapshot() == (0, 0, 0, False, 0)


@pytest.mark.unit
def test_failed_clear_stays_dirty_and_explicit_retry_succeeds(fake_ci, caplog):
    janitor, context, _enters, _exits = fake_ci
    janitor._interval = 1
    context.fail_with = RuntimeError("clear failed")

    with pb.ci_render_scope():
        pass
    assert janitor._snapshot() == (0, 1, 0, False, 0)
    assert "eligible for retry" in caplog.text

    pb.clear_ci_caches()
    assert context.clear_calls == 2
    assert janitor._snapshot() == (0, 1, 1, False, 0)


@pytest.mark.unit
def test_failed_periodic_clear_retries_after_another_interval(fake_ci):
    _janitor, context, _enters, _exits = fake_ci
    pb._ci_janitor._interval = 2
    context.fail_with = RuntimeError("clear failed")

    for _ in range(2):
        with pb.ci_render_scope():
            pass
    assert context.clear_calls == 1
    with pb.ci_render_scope():
        pass
    assert context.clear_calls == 1
    with pb.ci_render_scope():
        pass
    assert context.clear_calls == 2
    assert pb._ci_janitor._snapshot() == (0, 4, 4, False, 0)


@pytest.mark.unit
def test_render_error_outranks_ordinary_clear_error(fake_ci):
    janitor, context, _enters, _exits = fake_ci
    janitor._interval = 1
    context.fail_with = RuntimeError("clear failed")

    with (
        pytest.raises(ValueError, match="render failed") as caught,
        pb.ci_render_scope(),
    ):
        raise ValueError("render failed")

    assert caught.value.__context__ is not None
    assert "clear failed" in str(caught.value.__context__)


@pytest.mark.unit
def test_pool_exit_error_preserves_clear_error(fake_ci, monkeypatch):
    janitor, context, _enters, _exits = fake_ci
    janitor._interval = 1
    context.fail_with = RuntimeError("clear failed")

    class BadPool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            raise RuntimeError("pool exit failed")

    monkeypatch.setattr(pb, "autorelease_pool", BadPool)
    with (
        pytest.raises(RuntimeError, match="pool exit failed") as caught,
        pb.ci_render_scope(),
    ):
        pass

    assert caught.value.__context__ is not None
    assert "clear failed" in str(caught.value.__context__)


@pytest.mark.unit
def test_final_owner_cleanup_failure_is_best_effort_and_retryable(fake_ci, caplog):
    janitor, context, _enters, _exits = fake_ci
    owner = pb.ci_cache_owner()
    with pb.ci_render_scope():
        pass
    context.fail_with = RuntimeError("final clear failed")

    owner.close()

    assert "final cache cleanup failed" in caplog.text
    assert janitor._snapshot() == (0, 1, 0, False, 0)
    pb.clear_ci_caches()
    assert janitor._snapshot() == (0, 1, 1, False, 0)


@pytest.mark.unit
def test_final_clear_waits_for_active_render_and_gates_new_work():
    janitor = pb._CiCacheJanitor(interval=100)
    clear_started = threading.Event()
    allow_clear = threading.Event()
    second_entered = threading.Event()

    def clear() -> None:
        assert janitor._snapshot()[0] == 0
        clear_started.set()
        assert allow_clear.wait(timeout=2)

    owner = janitor.acquire_owner(clear)
    first_token = object()
    janitor.begin_render(first_token)
    closer = threading.Thread(target=owner.close)
    closer.start()
    with janitor._condition:
        assert janitor._condition.wait_for(
            lambda: janitor._clear_claim is not None,
            timeout=2,
        )

    def second_render() -> None:
        token = object()
        janitor.begin_render(token)
        second_entered.set()
        janitor.finish_render(token, clear)

    entrant = threading.Thread(target=second_render)
    entrant.start()
    assert not second_entered.wait(timeout=0.05)
    janitor.finish_render(first_token, clear)
    assert clear_started.wait(timeout=2)
    assert not second_entered.is_set()

    allow_clear.set()
    closer.join(timeout=2)
    entrant.join(timeout=2)
    assert not closer.is_alive()
    assert not entrant.is_alive()
    assert second_entered.is_set()
    # The second render began after the owner's final clear, so it remains a
    # new dirty generation for periodic maintenance rather than being lost.
    assert janitor._snapshot() == (0, 2, 1, False, 0)


@pytest.mark.unit
def test_periodic_clear_gates_and_drains_continuously_overlapping_renders():
    janitor = pb._CiCacheJanitor(interval=2)
    clears = []
    clear_started = threading.Event()
    fourth_entered = threading.Event()

    def clear() -> None:
        clears.append(janitor._snapshot())
        clear_started.set()

    first, second, third, fourth = (object() for _ in range(4))
    janitor.begin_render(first)
    janitor.begin_render(second)
    janitor.finish_render(first, clear)
    # The third render enters before the second completes, so the active set
    # has never reached zero when the interval threshold is crossed.
    janitor.begin_render(third)

    threshold_finisher = threading.Thread(
        target=janitor.finish_render,
        args=(second, clear),
    )
    threshold_finisher.start()
    with janitor._condition:
        assert janitor._condition.wait_for(
            lambda: janitor._clear_claim is not None,
            timeout=2,
        )

    def enter_fourth() -> None:
        janitor.begin_render(fourth)
        fourth_entered.set()
        janitor.finish_render(fourth, clear)

    later_entrant = threading.Thread(target=enter_fourth)
    later_entrant.start()
    assert not fourth_entered.wait(timeout=0.05)

    janitor.finish_render(third, clear)
    assert clear_started.wait(timeout=2)
    threshold_finisher.join(timeout=2)
    later_entrant.join(timeout=2)

    assert not threshold_finisher.is_alive()
    assert not later_entrant.is_alive()
    assert fourth_entered.is_set()
    assert len(clears) == 1
    # The clear owns the complete three-render admitted cohort. The fourth
    # render starts only afterward and remains the next dirty generation.
    assert clears[0][:3] == (0, 3, 0)
    assert janitor._snapshot() == (0, 4, 3, False, 0)


@pytest.mark.unit
def test_periodic_threshold_finisher_does_not_wait_on_application_lock():
    janitor = pb._CiCacheJanitor(interval=2)
    application_lock = threading.Lock()
    threshold_holds_lock = threading.Event()
    peer_waiting_for_lock = threading.Event()
    threshold_done = threading.Event()
    clears = []

    seed, peer, threshold = (object() for _ in range(3))
    janitor.begin_render(seed)
    janitor.begin_render(peer)
    janitor.finish_render(seed, lambda: clears.append("clear"))

    def finish_threshold() -> None:
        janitor.begin_render(threshold)
        with application_lock:
            threshold_holds_lock.set()
            assert peer_waiting_for_lock.wait(timeout=2)
            janitor.finish_render(threshold, lambda: clears.append("clear"))
            threshold_done.set()

    def finish_peer() -> None:
        assert threshold_holds_lock.wait(timeout=2)
        peer_waiting_for_lock.set()
        with application_lock:
            janitor.finish_render(peer, lambda: clears.append("clear"))

    threshold_thread = threading.Thread(target=finish_threshold)
    peer_thread = threading.Thread(target=finish_peer)
    threshold_thread.start()
    peer_thread.start()

    assert threshold_done.wait(timeout=2)
    threshold_thread.join(timeout=2)
    peer_thread.join(timeout=2)

    assert not threshold_thread.is_alive()
    assert not peer_thread.is_alive()
    assert clears == ["clear"]
    assert janitor._snapshot() == (0, 3, 3, False, 0)


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["owner", "explicit"])
def test_interrupted_wait_rolls_back_render_gate(monkeypatch, operation):
    janitor = pb._CiCacheJanitor(interval=100)
    clears = []
    token = object()
    janitor.begin_render(token)
    owner = janitor.acquire_owner(lambda: clears.append("clear"))
    original_wait = janitor._condition.wait

    def interrupt_wait(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(janitor._condition, "wait", interrupt_wait)
    if operation == "owner":
        with pytest.raises(KeyboardInterrupt):
            owner.close()
        expected_owners = 0
    else:
        with pytest.raises(KeyboardInterrupt):
            janitor.clear_if_dirty(lambda: clears.append("clear"))
        expected_owners = 1

    assert janitor._snapshot() == (1, 0, 0, False, expected_owners)
    monkeypatch.setattr(janitor._condition, "wait", original_wait)
    janitor.finish_render(token, lambda: clears.append("clear"))
    owner.close()
    assert clears == ["clear"]
    assert janitor._snapshot() == (0, 1, 1, False, 0)


@pytest.mark.unit
def test_render_finish_entry_interrupt_is_retried_without_stranding(fake_ci):
    janitor, _context, _enters, _exits = fake_ci
    scope = pb.ci_render_scope()
    scope.__enter__()
    base = janitor._condition

    class InterruptingCondition:
        def __init__(self):
            self.interrupt = True

        def __enter__(self):
            if self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt
            return base.__enter__()

        def __exit__(self, exc_type, exc, tb):
            return base.__exit__(exc_type, exc, tb)

        def wait(self, timeout=None):
            return base.wait(timeout)

        def notify_all(self):
            return base.notify_all()

    janitor._condition = InterruptingCondition()

    with pytest.raises(KeyboardInterrupt):
        scope.__exit__(None, None, None)

    assert janitor._snapshot() == (0, 1, 0, False, 0)
    janitor.clear_if_dirty(lambda: None)
    assert janitor._snapshot() == (0, 1, 1, False, 0)


@pytest.mark.unit
def test_interrupt_between_clear_claim_and_callback_releases_gate(monkeypatch):
    janitor = pb._CiCacheJanitor(interval=1)
    token = object()
    janitor.begin_render(token)
    original_complete = janitor._complete_clear

    def interrupt_before_callback(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(janitor, "_complete_clear", interrupt_before_callback)
    with pytest.raises(KeyboardInterrupt):
        janitor.finish_render(token, lambda: None)

    assert janitor._snapshot() == (0, 1, 0, False, 0)
    monkeypatch.setattr(janitor, "_complete_clear", original_complete)
    janitor.clear_if_dirty(lambda: None)
    assert janitor._snapshot() == (0, 1, 1, False, 0)


@pytest.mark.unit
def test_owner_release_entry_interrupt_is_retryable():
    janitor = pb._CiCacheJanitor(interval=100)
    owner = janitor.acquire_owner(lambda: None)
    base = janitor._condition

    class InterruptingCondition:
        def __init__(self):
            self.interrupt = True

        def __enter__(self):
            if self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt
            return base.__enter__()

        def __exit__(self, exc_type, exc, tb):
            return base.__exit__(exc_type, exc, tb)

        def wait(self, timeout=None):
            return base.wait(timeout)

        def notify_all(self):
            return base.notify_all()

    janitor._condition = InterruptingCondition()

    with pytest.raises(KeyboardInterrupt):
        owner.close()
    assert janitor._snapshot() == (0, 0, 0, False, 1)

    owner.close()
    assert janitor._snapshot() == (0, 0, 0, False, 0)


@pytest.mark.unit
def test_owner_construction_failure_does_not_register_token(monkeypatch):
    janitor = pb._CiCacheJanitor(interval=100)

    def fail_owner(*_args):
        raise MemoryError("owner construction failed")

    monkeypatch.setattr(pb, "_CiCacheOwner", fail_owner)
    with pytest.raises(MemoryError, match="owner construction failed"):
        janitor.acquire_owner(lambda: None)

    assert janitor._snapshot() == (0, 0, 0, False, 0)


@pytest.mark.unit
def test_interrupt_after_render_transition_retries_missed_notification(fake_ci):
    janitor, _context, _enters, _exits = fake_ci
    scope = pb.ci_render_scope()
    scope.__enter__()
    original_notify = janitor._condition.notify_all
    calls = 0

    def interrupt_first_notify():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        original_notify()

    janitor._condition.notify_all = interrupt_first_notify

    with pytest.raises(KeyboardInterrupt):
        scope.__exit__(None, None, None)

    assert calls >= 2
    assert janitor._snapshot() == (0, 1, 0, False, 0)


@pytest.mark.unit
def test_interrupt_after_clear_transition_retries_missed_notification(fake_ci):
    janitor, context, _enters, _exits = fake_ci
    janitor._interval = 1
    original_notify = janitor._condition.notify_all
    calls = 0

    def interrupt_second_notify():
        nonlocal calls
        calls += 1
        # First notify removes the render token. The second releases the clear
        # gate after the cache callback and ledger transition have completed.
        if calls == 2:
            raise KeyboardInterrupt
        original_notify()

    janitor._condition.notify_all = interrupt_second_notify

    with pytest.raises(KeyboardInterrupt), pb.ci_render_scope():
        pass

    assert context.clear_calls == 1
    assert calls >= 3
    assert janitor._snapshot() == (0, 1, 1, False, 0)


@pytest.mark.unit
def test_direct_rgbahalf_upload_does_not_count_core_image_work(
    fake_ci,
    monkeypatch,
):
    janitor, _context, _enters, _exits = fake_ci
    frame = SimpleNamespace(shape=(2, 2, 4), dtype="float16")
    writes: list[tuple[object, object]] = []
    monkeypatch.setattr(
        pb.Quartz,
        "CVPixelBufferGetPixelFormatType",
        lambda _buffer: pb.PIX_RGBAHALF,
    )
    monkeypatch.setattr(
        pb,
        "write_fp16_rgba",
        lambda source, buffer: writes.append((source, buffer)),
    )

    buffer = object()
    pb.upload_frame_to_buffer(frame, buffer)

    assert writes == [(frame, buffer)]
    assert janitor._snapshot()[1] == 0


@pytest.mark.unit
def test_file_owner_covers_work_after_inner_session_release(
    fake_ci,
    monkeypatch,
    tmp_path,
):
    run_module = importlib.import_module("kinovsr.pipeline.run")
    janitor, context, _enters, _exits = fake_ci

    class Transaction:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(run_module, "_OutputTransaction", Transaction)
    monkeypatch.setattr(run_module, "_probe_cfr_timing", lambda *_args: None)

    def reserved(*_args, **_kwargs):
        assert janitor._snapshot()[-1] == 1
        inner = pb.ci_cache_owner()
        with pb.ci_render_scope():
            pass
        inner.close()
        assert context.clear_calls == 0
        assert janitor._snapshot()[-1] == 1
        with pb.ci_render_scope():
            pass
        return "complete"

    monkeypatch.setattr(run_module, "_run_file_reserved", reserved)
    result = run_module.run_file(
        {"pipeline": []},
        video=tmp_path / "input.mov",
        output=tmp_path / "output.mov",
        settings=SimpleNamespace(),
        skip_post_mp4=True,
    )

    assert result == "complete"
    assert context.clear_calls == 1
    assert janitor._snapshot() == (0, 2, 2, False, 0)


@pytest.mark.integration
def test_native_conversion_call_sites_each_count_once(monkeypatch):
    import mlx.core as mx

    from kinovsr.media.images import resize_lanczos
    from kinovsr.native.frameworks import Quartz
    from kinovsr.processors.spatial import SpatialDenoiser

    janitor = pb._CiCacheJanitor(interval=1_000)
    monkeypatch.setattr(pb, "_ci_janitor", janitor)

    def make_buffer(pixel_format):
        return pb.make_pixel_buffer_from_attrs(8, 8, {
            Quartz.kCVPixelBufferPixelFormatTypeKey: pixel_format,
            Quartz.kCVPixelBufferWidthKey: 8,
            Quartz.kCVPixelBufferHeightKey: 8,
            Quartz.kCVPixelBufferIOSurfacePropertiesKey: {},
        })

    nv12 = make_buffer(pb.PIX_NV12)
    pb.upload_frame_to_buffer(mx.zeros((8, 8, 3), dtype=mx.uint8), nv12)
    assert janitor._snapshot()[1] == 1
    pb.read_pixel_buffer_rgb(nv12)
    assert janitor._snapshot()[1] == 2

    SpatialDenoiser().denoise(mx.zeros((8, 8, 3), dtype=mx.float32))
    assert janitor._snapshot()[1] == 3
    resize_lanczos(mx.zeros((8, 8, 3), dtype=mx.uint8), 6, 5)
    assert janitor._snapshot()[1] == 4

    rgba_half = make_buffer(pb.PIX_RGBAHALF)
    pb.upload_frame_to_buffer(
        mx.zeros((8, 8, 4), dtype=mx.float16),
        rgba_half,
    )
    pb.read_rgbahalf_rgb(rgba_half)
    assert janitor._snapshot()[1] == 4

    pb.clear_ci_caches()
    assert janitor._snapshot() == (0, 4, 4, False, 0)
