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
def test_final_clear_waits_for_active_render_and_blocks_renders_while_clearing():
    janitor = pb._CiCacheJanitor(interval=100)
    clear_started = threading.Event()
    allow_clear = threading.Event()
    second_done = threading.Event()

    def clear() -> None:
        clear_started.set()
        assert allow_clear.wait(timeout=2)

    owner = janitor.acquire_owner(clear)
    janitor.begin_render()
    closer = threading.Thread(target=owner.close)
    closer.start()
    # The final clear must wait for the active render.
    assert not clear_started.wait(timeout=0.05)
    janitor.finish_render(clear)
    assert clear_started.wait(timeout=2)

    def second_render() -> None:
        janitor.begin_render()
        janitor.finish_render(clear)
        second_done.set()

    entrant = threading.Thread(target=second_render)
    entrant.start()
    # New renders are blocked while the clear executes.
    assert not second_done.wait(timeout=0.05)
    allow_clear.set()
    closer.join(timeout=2)
    entrant.join(timeout=2)
    assert not closer.is_alive()
    assert not entrant.is_alive()
    assert second_done.is_set()
    # The second render began after the final clear, so it remains a new
    # dirty generation for periodic maintenance rather than being lost.
    assert janitor._snapshot() == (0, 2, 1, False, 0)


@pytest.mark.unit
def test_periodic_clear_defers_until_the_first_idle_moment_under_overlap():
    janitor = pb._CiCacheJanitor(interval=2)
    active_at_clear = []

    def clear() -> None:
        active_at_clear.append(janitor._snapshot()[0])

    janitor.begin_render()
    janitor.begin_render()
    janitor.finish_render(clear)
    # A third render enters before the second completes: the threshold is
    # crossed while a peer is active, so the clear defers.
    janitor.begin_render()
    janitor.finish_render(clear)
    assert active_at_clear == []
    # The last finisher out runs the deferred clear at zero active renders.
    janitor.finish_render(clear)
    assert active_at_clear == [0]
    assert janitor._snapshot() == (0, 3, 3, False, 0)


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
