"""Typed VideoToolbox output-pool compatibility checks."""

from __future__ import annotations

import pytest

from kinovsr.processors.videotoolbox import pools
from kinovsr.processors.videotoolbox.pools import MlxUploadPool, apply_output_pool

pytestmark = pytest.mark.unit


class _Session:
    def __init__(self, pixel_format: int, **attrs) -> None:
        self.dst_attrs = {"PixelFormatType": pixel_format, **attrs}
        self.pools = []

    def use_dst_pool(self, pool) -> None:
        self.pools.append(pool)


def test_exact_format_geometry_and_padding_bind_external_pool(monkeypatch):
    session = _Session(123)
    pool = object()
    monkeypatch.setattr(pools, "_pool_descriptor", lambda actual: (123, 1920, 1080, (0, 0, 0, 0)))

    apply_output_pool(session, (pool, 123, 1920, 1080), 1920, 1080)

    assert session.pools == [pool]


@pytest.mark.parametrize(
    "binding",
    [
        (object(), 456, 1920, 1080),
        (object(), 123, 1280, 720),
    ],
)
def test_mismatched_writer_pool_is_rejected(monkeypatch, binding):
    session = _Session(123)
    monkeypatch.setattr(
        pools,
        "_pool_descriptor",
        lambda _pool: pytest.fail("metadata mismatch should reject before probing"),
    )

    apply_output_pool(session, binding, 1920, 1080)

    assert session.pools == []


def test_actual_pool_descriptor_mismatch_is_rejected(monkeypatch):
    session = _Session(123)
    pool = object()
    monkeypatch.setattr(pools, "_pool_descriptor", lambda actual: (456, 1920, 1080, (0, 0, 0, 0)))

    apply_output_pool(session, (pool, 123, 1920, 1080), 1920, 1080)

    assert session.pools == []


def test_insufficient_extended_pixel_padding_is_rejected(monkeypatch):
    from kinovsr.native.frameworks import Quartz

    session = _Session(
        123,
        **{Quartz.kCVPixelBufferExtendedPixelsBottomKey: 16},
    )
    pool = object()
    monkeypatch.setattr(pools, "_pool_descriptor", lambda actual: (123, 1920, 1080, (0, 0, 0, 8)))

    apply_output_pool(session, (pool, 123, 1920, 1080), 1920, 1080)

    assert session.pools == []


def test_larger_extended_pixel_padding_is_compatible(monkeypatch):
    from kinovsr.native.frameworks import Quartz

    session = _Session(
        123,
        **{Quartz.kCVPixelBufferExtendedPixelsBottomKey: 8},
    )
    pool = object()
    monkeypatch.setattr(pools, "_pool_descriptor", lambda actual: (123, 1920, 1080, (0, 0, 0, 16)))

    apply_output_pool(session, (pool, 123, 1920, 1080), 1920, 1080)

    assert session.pools == [pool]


def test_mlx_upload_reuses_pool_and_flushes_on_close(monkeypatch):
    import mlx.core as mx

    pool, buffer = object(), object()
    uploaded = []
    flushed = []
    monkeypatch.setattr(
        pools.pb,
        "make_bounded_pool_from_attrs",
        lambda _attrs, _count: pool,
    )
    pulled = []
    monkeypatch.setattr(
        pools.pb,
        "pool_create_buffer_bounded",
        lambda actual, limit: pulled.append((actual, limit)) or buffer,
    )
    monkeypatch.setattr(
        pools.pb,
        "make_pixel_buffer_from_attrs",
        lambda *_args: pytest.fail("upload pool unexpectedly allocated fresh"),
    )
    monkeypatch.setattr(
        pools.pb, "upload_frame_to_buffer", lambda frame, dst: uploaded.append((frame.shape, dst))
    )
    monkeypatch.setattr(pools.pb, "flush_pool", flushed.append)

    bridge = MlxUploadPool(8, 6)
    assert bridge.upload(mx.zeros((6, 8, 3))) is buffer
    bridge.close()
    bridge.close()

    assert uploaded == [((6, 8, 4), buffer)]
    assert pulled == [(pool, pools.UPLOAD_POOL_ALLOCATION_LIMIT)]
    assert flushed == [pool]


def test_mlx_upload_requires_a_bounded_pool(monkeypatch):
    monkeypatch.setattr(
        pools.pb,
        "make_bounded_pool_from_attrs",
        lambda _attrs, _count: None,
    )
    monkeypatch.setattr(
        pools.pb,
        "make_pixel_buffer_from_attrs",
        lambda *_args: pytest.fail("fresh upload allocation must not be used"),
    )

    with pytest.raises(RuntimeError, match="bounded source allocation"):
        MlxUploadPool(8, 6)
