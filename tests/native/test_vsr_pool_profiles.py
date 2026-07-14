"""VSR geometry/profile changes replace and close bounded native pools."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _minimum_count(pool) -> int:
    from kinovsr.native.frameworks import Quartz

    attrs = dict(Quartz.CVPixelBufferPoolGetAttributes(pool))
    return int(attrs[Quartz.kCVPixelBufferPoolMinimumBufferCountKey])


def test_mode_and_geometry_change_replaces_bounded_source_and_destination_pools():
    from kinovsr.media import pixel_buffers as pb
    from kinovsr.native import vsr
    from kinovsr.native.frameworks import Quartz

    fast = balanced = None
    try:
        fast = vsr.VsrSession(128, 96, mode="fast", fps=24)
        fast_src = fast._src_pool
        fast_dst = fast._dst_pool
        assert _minimum_count(fast_src) == vsr.STATELESS_SRC_POOL_ALLOCATION_LIMIT
        assert _minimum_count(fast_dst) == vsr.DST_POOL_ALLOCATION_LIMIT
        fast_dst_attrs = dict(Quartz.CVPixelBufferPoolGetPixelBufferAttributes(fast_dst))
        assert pb.resolve_pixel_format(fast_dst_attrs) == pb.PIX_NV12
        assert fast_dst_attrs[Quartz.kCVPixelBufferWidthKey] == 256
        assert fast_dst_attrs[Quartz.kCVPixelBufferHeightKey] == 192
        closed_fast = fast
        fast.close()
        fast = None
        assert closed_fast._src_pool is None
        assert closed_fast._dst_pool is None

        balanced = vsr.VsrSession(160, 120, mode="balanced", fps=24)
        assert balanced._src_pool is not fast_src
        assert balanced._dst_pool is not fast_dst
        assert _minimum_count(balanced._src_pool) == vsr.TEMPORAL_SRC_POOL_ALLOCATION_LIMIT
        assert _minimum_count(balanced._dst_pool) == vsr.DST_POOL_ALLOCATION_LIMIT
        balanced_dst_attrs = dict(
            Quartz.CVPixelBufferPoolGetPixelBufferAttributes(balanced._dst_pool)
        )
        assert pb.resolve_pixel_format(balanced_dst_attrs) == pb.PIX_RGBAHALF
        assert balanced_dst_attrs[Quartz.kCVPixelBufferWidthKey] == 640
        assert balanced_dst_attrs[Quartz.kCVPixelBufferHeightKey] == 480
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    finally:
        if fast is not None:
            fast.close()
        if balanced is not None:
            balanced.close()

    assert balanced._src_pool is None
    assert balanced._dst_pool is None
