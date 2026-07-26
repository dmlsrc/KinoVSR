"""Unit coverage for public VT optical-flow destination safeguards."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _BaseAddress:
    def __init__(self) -> None:
        self.data = bytearray(16)

    def as_buffer(self, size: int):
        return memoryview(self.data)[:size]


class _Buffer:
    def __init__(self, width: int = 128, height: int = 128) -> None:
        self.width = width
        self.height = height
        self.base = _BaseAddress()


def _fake_quartz(monkeypatch):
    from kinovsr.native import optical_flow

    monkeypatch.setattr(
        optical_flow.Quartz,
        "CVPixelBufferLockBaseAddress",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        optical_flow.Quartz,
        "CVPixelBufferUnlockBaseAddress",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        optical_flow.Quartz,
        "CVPixelBufferGetBaseAddress",
        lambda buffer: buffer.base,
    )
    monkeypatch.setattr(
        optical_flow.Quartz,
        "CVPixelBufferGetWidth",
        lambda buffer: buffer.width,
    )
    monkeypatch.setattr(
        optical_flow.Quartz,
        "CVPixelBufferGetHeight",
        lambda buffer: buffer.height,
    )
    return optical_flow


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (64, 48, (128, 128)),
        (127, 192, (192, 128)),
        (128, 127, (128, 128)),
        (640, 480, (640, 480)),
        (480, 640, (640, 480)),
    ],
)
def test_flow_destination_geometry_enforces_writer_floor_and_orientation(
    width,
    height,
    expected,
):
    from kinovsr.native.optical_flow import flow_destination_geometry

    assert flow_destination_geometry(width, height) == expected


def test_pending_marker_distinguishes_zero_flow_from_no_write(monkeypatch):
    optical_flow = _fake_quartz(monkeypatch)
    pair = (_Buffer(), _Buffer())

    optical_flow.mark_flow_pair_pending(pair)
    with pytest.raises(RuntimeError, match="without writing forward/backward"):
        optical_flow.require_flow_pair_written(pair, context="injected")

    pair[0].base.data[:4] = bytes(4)
    pair[1].base.data[:4] = bytes(4)
    optical_flow.require_flow_pair_written(pair, context="real zero flow")


def test_source_sized_reliability_requires_both_dimensions():
    from kinovsr.native.optical_flow import source_sized_flow_is_reliable

    assert source_sized_flow_is_reliable(128, 128)
    assert source_sized_flow_is_reliable(1920, 1080)
    assert not source_sized_flow_is_reliable(127, 192)
    assert not source_sized_flow_is_reliable(192, 127)
