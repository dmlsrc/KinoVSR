"""CoreVideo pool acquisition has a real, observable allocation ceiling."""

from __future__ import annotations

import gc

import pytest

from kinovsr.media.pixel_buffers import (
    PIX_RGBAHALF,
    PixelBufferPoolExhausted,
    make_bounded_pool_from_attrs,
    make_pool_from_attrs,
    pool_create_buffer_bounded,
)
from kinovsr.native.frameworks import Quartz

pytestmark = pytest.mark.integration


def _pool():
    pool = make_pool_from_attrs(
        {
            "PixelFormatType": PIX_RGBAHALF,
            "Width": 16,
            "Height": 12,
            "IOSurfaceProperties": {},
            "MetalCompatibility": True,
        }
    )
    assert pool is not None
    return pool


def test_bounded_acquire_reports_exhaustion_and_recovers_after_release():
    pool = _pool()
    first = pool_create_buffer_bounded(pool, 1)

    with pytest.raises(
        PixelBufferPoolExhausted,
        match=r"allocation threshold 1 \(status=-6689\)",
    ):
        pool_create_buffer_bounded(pool, 1)

    del first
    gc.collect()
    replacement = pool_create_buffer_bounded(pool, 1)
    assert Quartz.CVPixelBufferGetWidth(replacement) == 16
    assert Quartz.CVPixelBufferGetHeight(replacement) == 12


def test_bounded_acquire_rejects_a_nonpositive_threshold():
    with pytest.raises(ValueError, match="must be positive"):
        pool_create_buffer_bounded(_pool(), 0)


def test_bounded_pool_declares_its_reusable_working_set():
    pool = make_bounded_pool_from_attrs(
        {
            "PixelFormatType": PIX_RGBAHALF,
            "Width": 16,
            "Height": 12,
            "IOSurfaceProperties": {},
        },
        3,
    )
    assert pool is not None
    attrs = dict(Quartz.CVPixelBufferPoolGetAttributes(pool))
    assert attrs[Quartz.kCVPixelBufferPoolMinimumBufferCountKey] == 3


def test_output_pool_probe_observes_real_extended_padding():
    from kinovsr.processors.videotoolbox.pools import apply_output_pool

    attrs = {
        "PixelFormatType": PIX_RGBAHALF,
        "Width": 16,
        "Height": 12,
        "IOSurfaceProperties": {},
        Quartz.kCVPixelBufferExtendedPixelsBottomKey: 16,
    }
    pool = make_pool_from_attrs(attrs)
    assert pool is not None

    class Session:
        def __init__(self, bottom: int) -> None:
            self.dst_attrs = {
                "PixelFormatType": PIX_RGBAHALF,
                Quartz.kCVPixelBufferExtendedPixelsBottomKey: bottom,
            }
            self.bound = []

        def use_dst_pool(self, actual) -> None:
            self.bound.append(actual)

    compatible = Session(16)
    apply_output_pool(compatible, (pool, PIX_RGBAHALF, 16, 12), 16, 12)
    assert compatible.bound == [pool]

    insufficient = Session(17)
    apply_output_pool(insufficient, (pool, PIX_RGBAHALF, 16, 12), 16, 12)
    assert insufficient.bound == []
