"""Session-owned VideoToolbox destination-pool lifecycle."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _Processor:
    def __init__(self) -> None:
        self.end_count = 0

    def endSession(self) -> None:
        self.end_count += 1


def test_vsr_source_pool_has_a_hard_allocation_limit(monkeypatch):
    from kinovsr.native import vsr

    session = vsr.VsrSession.__new__(vsr.VsrSession)
    pool, output = object(), object()
    session._src_pool = pool
    session._src_pool_allocation_limit = vsr.TEMPORAL_SRC_POOL_ALLOCATION_LIMIT
    pulled = []
    monkeypatch.setattr(
        vsr._pb,
        "pool_create_buffer_bounded",
        lambda actual, limit: pulled.append((actual, limit)) or output,
    )
    monkeypatch.setattr(
        vsr._pb,
        "make_pixel_buffer_from_attrs",
        lambda *_args: pytest.fail("bounded VSR source allocated fresh"),
    )

    assert session._make_src_buffer() is output
    assert pulled == [(pool, vsr.TEMPORAL_SRC_POOL_ALLOCATION_LIMIT)]


def test_vsr_reuses_owned_pool_and_never_flushes_external_pool(monkeypatch):
    from kinovsr.native import vsr

    session = vsr.VsrSession.__new__(vsr.VsrSession)
    owned, external, source, output = object(), object(), object(), object()
    session._src_pool = source
    session._dst_pool = owned
    session._owns_dst_pool = True
    session._prev_src_frame = object()
    session._prev_dst_frame = object()
    session._xfer = None
    session.processor = _Processor()
    session.out_w = session.out_h = 8
    session.dst_attrs = {"PixelFormatType": 1}
    flushed = []
    pulled = []

    monkeypatch.setattr(vsr._pb, "flush_pool", flushed.append)
    monkeypatch.setattr(
        vsr._pb,
        "pool_create_buffer_bounded",
        lambda pool, limit: pulled.append((pool, limit)) or output,
    )
    monkeypatch.setattr(
        vsr._pb,
        "pool_create_buffer",
        lambda pool: pulled.append((pool, None)) or output,
    )
    monkeypatch.setattr(
        vsr._pb,
        "make_pixel_buffer_from_attrs",
        lambda *_args: pytest.fail("pooled VSR output allocated a fresh buffer"),
    )

    assert session._make_dst_buffer() is output
    assert pulled == [(owned, vsr.DST_POOL_ALLOCATION_LIMIT)]
    session.use_dst_pool(external)
    assert flushed == [owned]
    assert session._make_dst_buffer() is output
    assert pulled[-1] == (external, None)

    processor = session.processor
    session.close()
    assert processor.end_count == 1
    assert flushed == [owned, source]
    assert session._prev_src_frame is None
    assert session._prev_dst_frame is None
    assert session._src_pool is None
    assert session._dst_pool is None


def test_frc_reuses_owned_pool_and_closes_it_exactly_once(monkeypatch):
    from kinovsr.native import temporal

    session = temporal.VtfrcSession.__new__(temporal.VtfrcSession)
    owned, output = object(), object()
    session._dst_pool = owned
    session._owns_dst_pool = True
    session._prev_src_pb = object()
    session.processor = _Processor()
    session.in_w = session.in_h = 8
    session.dst_attrs = {"PixelFormatType": 1}
    flushed = []
    pulled = []

    monkeypatch.setattr(temporal._pb, "flush_pool", flushed.append)
    monkeypatch.setattr(
        temporal._pb,
        "pool_create_buffer_bounded",
        lambda pool, limit: pulled.append((pool, limit)) or output,
    )
    monkeypatch.setattr(
        temporal._pb,
        "make_pixel_buffer_from_attrs",
        lambda *_args: pytest.fail("pooled FRC output allocated a fresh buffer"),
    )

    assert session._make_dst_buffer() is output
    assert pulled == [(owned, temporal.DST_POOL_ALLOCATION_LIMIT)]
    processor = session.processor
    session.close()
    session.close()

    assert processor.end_count == 1
    assert flushed == [owned]
    assert session._prev_src_pb is None
    assert session._dst_pool is None


def test_external_destination_pool_failure_does_not_allocate_fresh(monkeypatch):
    from kinovsr.native import temporal

    session = temporal.VtfrcSession.__new__(temporal.VtfrcSession)
    session._dst_pool = object()
    session._owns_dst_pool = False
    monkeypatch.setattr(temporal._pb, "pool_create_buffer", lambda _pool: None)
    monkeypatch.setattr(
        temporal._pb,
        "make_pixel_buffer_from_attrs",
        lambda *_args: pytest.fail("external pool failure allocated fresh"),
    )

    with pytest.raises(RuntimeError, match="external FRC destination"):
        session._make_dst_buffer()
