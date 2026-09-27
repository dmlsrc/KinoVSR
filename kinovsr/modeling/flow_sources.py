"""Optical-flow sources for the recurrent VSR families.

BasicVSR++ and RealBasicVSR propagate features along flows between
neighboring frames, and each run chooses where those flows come from:
SpyNet (the checkpoint's own flow network), Vision optical flow, or zero
flow. This module sits above both sources, so the SpyNet blocks and the
Vision flow services never import each other.
"""

from collections.abc import Sequence

import mlx.core as mx

from .vision_flow_services import VisionFlowServices, vision_flow_services_scope
from .vsr_blocks import compiled_spynet_flow


def vision_flows(
    frames: Sequence[mx.array],
    services: VisionFlowServices | None = None,
) -> tuple[list[mx.array], list[mx.array]]:
    """Vision optical flow (revision 1) for both propagation directions, in
    compute_flows' conventions: flows_forward[i] pulls frame i into frame
    i+1's geometry (anchored at i+1); flows_backward[i] pulls frame i+1 into
    frame i's.

    One native call per direction per pair: the engine's ``compute(b, a)``
    satisfies ``b[p] ~= a[p + flow[p]]``, which is the pull-flow anchored at b.
    """
    if not frames:
        return [], []
    if frames[0].shape[0] != 1:
        raise ValueError("flow_mode='vision' supports batch-1 frames only")
    if len(frames) == 1:
        return [], []
    h, w = int(frames[0].shape[1]), int(frames[0].shape[2])
    with (
        vision_flow_services_scope(services, max_geometries=1) as flow_services,
        flow_services.borrow(w, h) as svc,
    ):
        dt = frames[0].dtype
        ff: list[mx.array] = []
        fb: list[mx.array] = []
        for i in range(len(frames) - 1):
            a = frames[i][0].astype(mx.float32)
            b = frames[i + 1][0].astype(mx.float32)
            ff.append(svc.compute(b, a)[None].astype(dt))
            fb.append(svc.compute(a, b)[None].astype(dt))
            mx.eval(ff[-1], fb[-1])
        return ff, fb


def compute_flows(
    frames: Sequence[mx.array],
    p: dict[str, mx.array],
    flow_mode: str = "spynet",
    vision_flow_services: VisionFlowServices | None = None,
) -> tuple[list[mx.array], list[mx.array]]:
    """flows_forward[i] = flow(i+1 -> i); flows_backward[i] = flow(i -> i+1).

    Each flow is materialized as computed: SPyNet upsizes to a multiple of 32 and
    builds the BasicSR pyramid, so holding all 2*(T-1) of them as one lazy graph
    spikes memory; per-flow eval keeps only the small (H,W,2) results alive."""
    if flow_mode == "zero":
        zeros = [
            mx.zeros((*frames[0].shape[:3], 2), dtype=frames[0].dtype)
            for _ in range(len(frames) - 1)
        ]
        if zeros:
            mx.eval(*zeros)
        return list(zeros), list(zeros)
    if flow_mode == "vision":
        return vision_flows(frames, vision_flow_services)
    if flow_mode != "spynet":
        raise ValueError(f"unknown flow_mode {flow_mode!r}; expected 'spynet', 'zero', or 'vision'")
    fb: list[mx.array] = []
    ff: list[mx.array] = []
    for i in range(len(frames) - 1):
        b = compiled_spynet_flow(p, frames[i], frames[i + 1])
        f = compiled_spynet_flow(p, frames[i + 1], frames[i])
        mx.eval(b, f)
        fb.append(b)
        ff.append(f)
    return ff, fb
