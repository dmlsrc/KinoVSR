"""Native smoke coverage for driver-owned VTOpticalFlow services."""

from __future__ import annotations

import mlx.core as mx
import pytest


@pytest.mark.integration
def test_owned_service_computes_reuses_and_closes():
    from kinovsr.modeling.vsr_blocks import _compute_flows
    from kinovsr.modeling.vt_flow import VtFlowServices

    height, width = 240, 320
    first = mx.random.uniform(shape=(1, height, width, 3)).astype(mx.float16)
    second = mx.roll(first, 1, axis=2)
    mx.eval(first, second)
    services = VtFlowServices(1)
    service = None
    try:
        forward, backward = _compute_flows(
            [first, second],
            {},
            flow_mode="vt",
            vt_flow_services=services,
        )
        mx.eval(forward[0], backward[0])
        assert (
            forward[0].shape
            == backward[0].shape
            == (
                1,
                height,
                width,
                2,
            )
        )
        assert services.size == 1
        with services.borrow(width, height) as service:
            pass
    except (RuntimeError, SystemExit) as exc:
        pytest.skip(str(exc))
    finally:
        services.close()

    assert service is not None
    assert service._workers == []
    assert services.size == 0
