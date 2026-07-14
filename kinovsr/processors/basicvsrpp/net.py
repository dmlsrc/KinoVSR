"""MLX BasicVSR++ x4 net (port of OpenMMLab mmagic basicvsr_plusplus_net).

The shared BasicVSR backbone (conv/activation helpers, bilinear sample / flow_warp
/ resize, SPyNet, residual blocks, pixel-shuffle) lives in ../vsr_blocks; this
module adds the BasicVSR++-specific pieces: the weight loader, the second-order
deformable alignment, and the bidirectional recurrent forward.

Convention: MLX-native NHWC throughout. Conv weights are transposed to MLX's
(O,kH,kW,I) at load; the deformable-conv weight stays torch NCHW (O,I,kH,kW) for
deform_conv2d. Flow is (N,H,W,2) = (x-offset, y-offset), matching flow_warp.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import mlx.core as mx

from kinovsr.modeling.compile_cache import cached as _cached
from kinovsr.modeling.deform_conv import deform_conv2d
from kinovsr.modeling.vsr_blocks import (
    _compute_flows,
    _pixelshuffle_pack,
    _resblocks_with_input,
    compiled_resblocks,
    conv,
    flow_warp,
    history_improve_gate,
    lrelu,
    pad_spynet_gates,
    resize,
)
from kinovsr.modeling.vt_flow import vt_flow_services_scope
from kinovsr.modeling.weights import resolve_weights as _resolve_weights

# Per-checkpoint compiled reconstruction/upsample tail (keyed by id(p)).
_UPSAMPLE_COMPILE_CACHE: dict = {}


def _compiled_upsample(p: dict):
    """Compiled reconstruction resblocks + pixel-shuffle upsample tail -> HR residual.
    The cheap base resize + clip stay in the loop. Pure, byte-identical (profiled)."""

    def make():
        def step(hr):
            hr = _resblocks_with_input(hr, p, "reconstruction")
            hr = lrelu(_pixelshuffle_pack(hr, p, "upsample1"))
            hr = lrelu(_pixelshuffle_pack(hr, p, "upsample2"))
            hr = lrelu(conv(hr, p, "conv_hr"))
            return conv(hr, p, "conv_last")

        return mx.compile(step)

    return _cached(_UPSAMPLE_COMPILE_CACHE, id(p), make)


_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Bundled 4x-SR checkpoints (all is_low_res_input=True). reds4/vimeo90k_bi are
# c64n7 (7.3M); ntire_vsr is c128n25 (44M, NTIRE'21). Block counts auto-detect.
_VARIANTS = {
    "reds4": "basicvsrpp_reds4.safetensors",
    "vimeo90k_bi": "basicvsrpp_vimeo90k_bi.safetensors",
    "vimeo90k_bd": "basicvsrpp_vimeo90k_bd.safetensors",
    "ntire_vsr": "basicvsrpp_ntire_vsr.safetensors",
}


# 1x restoration checkpoints (is_low_res_input=False: the net downsamples the
# input 4x, propagates cheap at 1/4 res, upsamples back to the SAME size). Same
# architecture as the SR variants apart from the feature-extractor stem and the
# terminal residual; auto-detected at load. Not bundled (large; download +
# convert -- see weights/README.md).
_RESTORE_VARIANTS = {
    "decompress_track1": "basicvsrpp_decompress_track1.safetensors",  # NTIRE'21 compressed-video, fixed-QP fidelity
    "decompress_track2": "basicvsrpp_decompress_track2.safetensors",  # heavier compression
    "decompress_track3": "basicvsrpp_decompress_track3.safetensors",  # fixed bit-rate
    "denoise": "basicvsrpp_denoise.safetensors",  # temporal video denoise
    "deblur_dvd": "basicvsrpp_deblur_dvd.safetensors",  # real handheld-video deblur
    "deblur_gopro": "basicvsrpp_deblur_gopro.safetensors",  # synthetic GoPro deblur
}


def default_weights_path(variant: str = "reds4") -> Path:
    if variant not in _VARIANTS:
        raise ValueError(f"unknown basicvsrpp variant {variant!r}; choose from {list(_VARIANTS)}")
    return _WEIGHTS_DIR / _VARIANTS[variant]


def resolve_weights(spec: Any = None) -> Path:
    """Bundled variant token (reds4/vimeo90k_bi/vimeo90k_bd/ntire_vsr) or a path."""
    return _resolve_weights(spec, _VARIANTS, _WEIGHTS_DIR, "reds4")


def resolve_restore_weights(spec: Any = None) -> Path:
    """1x-restoration variant token (decompress_track*/denoise/deblur_*) or a path."""
    return _resolve_weights(spec, _RESTORE_VARIANTS, _WEIGHTS_DIR, "decompress_track1")


def is_low_res_input(p: dict) -> bool:
    """True for the 4x-SR checkpoints (feat_extract is a bare ResidualBlocksWithInputConv),
    False for the 1x-restoration checkpoints (feat_extract is a strided downsampling stem
    feat_extract.0/.2/.4). The two differ only in this stem and the terminal residual."""
    return "feat_extract.main.0.weight" in p


def load_params(path: str | Path | None = None, dtype: Any = mx.float16) -> dict:
    """Load + lay out the checkpoint: conv weights -> NHWC, DCN weight kept NCHW,
    SPyNet mean/std -> NHWC, step_counter dropped, all cast to `dtype`.

    Default fp16 halves activation memory and is ~1.5x faster; the deformable
    conv follows the input dtype (fp16 sampling/columns/GEMM with fp32
    accumulation -- see deform_conv2d; ~3.8x on the op, SR shift 58 dB). Pass
    dtype=mx.float32 for the full-fp32 validation reference."""
    src = Path(path or default_weights_path())
    w = mx.load(str(src))
    p: dict = {}
    for k, v in w.items():
        if k == "step_counter":
            continue
        if k in ("spynet.mean", "spynet.std"):
            a = v.reshape(1, 1, 1, 3)
        elif v.ndim == 4 and not (
            k.startswith("deform_align.") and k.endswith(".weight") and "conv_offset" not in k
        ):
            a = mx.transpose(v, (0, 2, 3, 1))  # (O,I,kH,kW) torch -> (O,kH,kW,I) MLX
        else:
            a = v
        p[k] = a.astype(dtype)
    pad_spynet_gates(p)  # SPyNet first convs 8->16 (see vsr_blocks)
    _pad_offset_gates(p)
    return p


def _pad_offset_gates(p: dict) -> None:
    """Zero-pad each deform_align conv_offset.0 weight from 196 to 208 input channels
    (in place, same key). Its input is concat(cond 192, flow1 2, flow2 2) = 196, and
    196%16 != 0 fails MLX's implicit-GEMM gate -- the heaviest conv of the offset stack
    ran on the ~1.45x-slower general kernel, four times per frame. _deform_align appends
    matching zero channels, so the math is exact (zero columns x zero channels)."""
    for k in list(p):
        if k.endswith(".conv_offset.0.weight"):
            w = p[k]
            cin = w.shape[-1]
            if cin > 4 and cin % 16:
                pad = 16 - cin % 16
                p[k] = mx.concatenate([w, mx.zeros((*w.shape[:3], pad), dtype=w.dtype)], axis=-1)


# ---- second-order deformable alignment -------------------------------------
def _flow_yx_tiled(flow: Any, reps: int) -> Any:
    """flow (N,H,W,2)=[x,y] -> [y,x] tiled `reps` times along the channel axis."""
    return mx.tile(mx.concatenate([flow[..., 1:2], flow[..., 0:1]], axis=-1), (1, 1, 1, reps))


def _deform_align(
    feat_cat: Any, cond: Any, flow1: Any, flow2: Any, p: dict, key: str, max_res: float = 10.0
) -> Any:
    """SecondOrderDeformableAlignment: predict offsets/mask from cond+flows, add
    the flows (the deformable offset is relative to the optical flow), then a
    modulated deform conv on feat_cat (NHWC <-> NCHW only at the DCN call)."""
    parts = [cond, flow1, flow2]
    pad = p[f"{key}.conv_offset.0.weight"].shape[-1] - sum(t.shape[-1] for t in parts)
    if pad:  # gate-padded weight (_pad_offset_gates)
        parts.append(mx.zeros((*cond.shape[:3], pad), dtype=cond.dtype))
    extra = mx.concatenate(parts, axis=-1)
    o = lrelu(conv(extra, p, f"{key}.conv_offset.0"))
    o = lrelu(conv(o, p, f"{key}.conv_offset.2"))
    o = lrelu(conv(o, p, f"{key}.conv_offset.4"))
    o = conv(o, p, f"{key}.conv_offset.6")  # (N,H,W,27*dg)
    o1, o2, mask = mx.split(o, 3, axis=-1)  # dg*K*K each
    off = max_res * mx.tanh(mx.concatenate([o1, o2], axis=-1))
    off1, off2 = mx.split(off, 2, axis=-1)
    off1 = off1 + _flow_yx_tiled(flow1, off1.shape[-1] // 2)
    off2 = off2 + _flow_yx_tiled(flow2, off2.shape[-1] // 2)
    offset = mx.concatenate([off1, off2], axis=-1)  # (N,H,W,dg*2*K*K)
    mask = mx.sigmoid(mask)
    dg = o.shape[-1] // 27
    out = deform_conv2d(
        mx.transpose(feat_cat, (0, 3, 1, 2)),
        mx.transpose(offset, (0, 3, 1, 2)),
        p[f"{key}.weight"],
        p.get(f"{key}.bias"),
        mx.transpose(mask, (0, 3, 1, 2)),
        stride=1,
        padding=1,
        dilation=1,
        deform_groups=dg,
    )
    return mx.transpose(out, (0, 2, 3, 1)).astype(feat_cat.dtype)  # DCN follows input dtype


_DEFORM_COMPILE_CACHE: dict = {}


def _compiled_deform_align(p: dict, key: str):
    """_deform_align (offset convs + the deform_conv kernel), compiled + cached per
    (checkpoint, module). ~1.02x byte-identical: the custom kernel is one big dispatch
    with nothing to fuse, but the offset conv stack around it fuses. Keyed by (id(p), key)."""
    return _cached(
        _DEFORM_COMPILE_CACHE,
        (id(p), key),
        lambda: mx.compile(lambda fc, c, f1, f2: _deform_align(fc, c, f1, f2, p, key)),
    )


# ---- recurrent forward -----------------------------------------------------
def _propagate(
    feats: dict,
    flows: list,
    module: str,
    p: dict,
    frames: list | None = None,
    history_strength: float = 1.0,
    history_gate: str = "off",
) -> dict:
    nf = len(feats["spatial"])
    frame_idx = list(range(nf))
    flow_idx = list(range(-1, nf - 1))
    if "backward" in module:
        frame_idx = frame_idx[::-1]
        flow_idx = frame_idx
    n, h, w, mid = feats["spatial"][0].shape
    dt = feats["spatial"][0].dtype
    # History admission (see vsr_blocks.history_improve_gate): the aligned
    # history bundle is multiplied per pixel by how much the warp actually
    # explains the current frame. Second-order alignment can draw from either
    # source, so the two per-source gates combine with max. A zero gate equals
    # the branch's window-start zero features (in-distribution). The scalar
    # history_strength path scales history unconditionally; the default
    # (1.0, gate off) leaves the reference math untouched.
    use_gate = history_gate == "improve" and frames is not None
    use_scalar = (not use_gate) and history_strength != 1.0
    feat_prop = mx.zeros((n, h, w, mid), dtype=dt)
    out: list = []
    for i, idx in enumerate(frame_idx):
        feat_current = feats["spatial"][idx]
        if i > 0:
            flow_n1 = flows[flow_idx[i]]
            cond_n1 = flow_warp(feat_prop, flow_n1)
            feat_n2 = mx.zeros_like(feat_prop)
            flow_n2 = mx.zeros_like(flow_n1)
            cond_n2 = mx.zeros_like(cond_n1)
            if i > 1:
                feat_n2 = out[-2]
                flow_n2 = flow_n1 + flow_warp(flows[flow_idx[i - 1]], flow_n1)
                cond_n2 = flow_warp(feat_n2, flow_n2)
            cond = mx.concatenate([cond_n1, feat_current, cond_n2], axis=-1)
            feat_prop = _compiled_deform_align(p, f"deform_align.{module}")(
                mx.concatenate([feat_prop, feat_n2], axis=-1), cond, flow_n1, flow_n2
            )
            if use_gate:
                gate = history_improve_gate(
                    frames[idx], frames[frame_idx[i - 1]], flow_n1, dt, history_strength
                )
                if i > 1:
                    gate = mx.maximum(
                        gate,
                        history_improve_gate(
                            frames[idx], frames[frame_idx[i - 2]], flow_n2, dt, history_strength
                        ),
                    )
                feat_prop = feat_prop * gate
            elif use_scalar:
                feat_prop = feat_prop * float(history_strength)
        feat = (
            [feat_current]
            + [feats[k][idx] for k in feats if k not in ("spatial", module)]
            + [feat_prop]
        )
        feat_prop = feat_prop + compiled_resblocks(
            mx.concatenate(feat, axis=-1), p, f"backbone.{module}"
        )
        # Materialize each step so the recurrent graph (and the large transient
        # DCN im2col columns) frees per frame instead of accumulating the whole
        # clip's forward into one lazy graph - that peaks memory catastrophically.
        mx.eval(feat_prop)
        out.append(feat_prop)
    if "backward" in module:
        out = out[::-1]
    feats[module] = out
    return feats


def _upsample(frames: list, feats: dict, p: dict) -> list:
    outs = []
    for i in range(len(feats["spatial"])):
        hr = [feats["spatial"][i]] + [feats[k][i] for k in feats if k != "spatial"]
        residual = _compiled_upsample(p)(mx.concatenate(hr, axis=-1))
        _, fh, fw, _ = frames[i].shape
        # Clip the terminal SR to [0,1]: the residual overshoots slightly at edges
        # (ringing) and this frame goes straight to the encoder. Not fed back into
        # the recurrence, so clipping here is safe.
        out_frame = mx.clip(residual + resize(frames[i], fh * 4, fw * 4, False), 0.0, 1.0)
        mx.eval(out_frame)  # free each frame's upsample graph before the next
        outs.append(out_frame)
    return outs


def upscale(
    frames: list,
    p: dict,
    flow_mode: str = "spynet",
    history_strength: float = 1.0,
    history_gate: str = "off",
    vt_flow_services: Any = None,
) -> list:
    """Upscale an LR clip 4x. frames: list of (N,H,W,3) f32 [0,1]; out: same len,
    each (N,4H,4W,3). Bidirectional + second-order, so the whole clip is needed.
    ``history_gate="improve"`` admits aligned history per pixel only where the
    flow warp measurably improves the photometric residual; ``history_strength``
    scales the aligned history (1.0 = reference)."""
    dt = p["conv_last.weight"].dtype
    # Clip to [0,1] - the model trained on uint8-derived [0,1] LR, but the
    # RGBAHalf decode can overshoot (~[-0.07, 1.04]); keep input in-distribution.
    frames = [mx.clip(f, 0.0, 1.0).astype(dt) for f in frames]
    spatial = []
    for f in frames:
        s = compiled_resblocks(f, p, "feat_extract")
        mx.eval(s)  # materialize per frame, not all at once
        spatial.append(s)
    feats: dict = {"spatial": spatial}
    ff, fb = _compute_flows(
        frames,
        p,
        flow_mode=flow_mode,
        vt_flow_services=vt_flow_services,
    )
    for it in (1, 2):
        for direction in ("backward", "forward"):
            mod = f"{direction}_{it}"
            feats = _propagate(
                feats,
                fb if direction == "backward" else ff,
                mod,
                p,
                frames=frames,
                history_strength=history_strength,
                history_gate=history_gate,
            )
            # _propagate already mx.eval's each step internally (see net.py:176), so every
            # element of feats[mod] is materialized here -- no extra sync barrier needed.
    return _upsample(frames, feats, p)


# ---- 1x restoration path (is_low_res_input=False) --------------------------
# torch F.interpolate(scale_factor=0.25, mode='bicubic', align_corners=False):
# because the factor is an exact 4, every output pixel maps to input coord 4j+1.5
# with the SAME fractional offset 0.5, so the cubic (A=-0.75) weights are constant
# -- a fixed 4-tap [w(-1),w(0),w(1),w(2)] over input taps [4j,4j+1,4j+2,4j+3].
_BICUBIC_DOWN4 = (-0.09375, 0.59375, 0.59375, -0.09375)


def _bicubic_down4(x: Any) -> Any:
    """Separable bicubic 1/4 downsample, exact for H,W multiples of 4 (taps never
    leave the image so no edge clamp is needed). Matches torch's flow-input downsample."""
    w0, w1, w2, w3 = _BICUBIC_DOWN4
    n, h, wd, c = x.shape
    r = x.reshape(n, h // 4, 4, wd, c)
    y = w0 * r[:, :, 0] + w1 * r[:, :, 1] + w2 * r[:, :, 2] + w3 * r[:, :, 3]
    cc = y.reshape(n, h // 4, wd // 4, 4, c)
    return w0 * cc[:, :, :, 0] + w1 * cc[:, :, :, 1] + w2 * cc[:, :, :, 2] + w3 * cc[:, :, :, 3]


def _feat_extract_1x(f: Any, p: dict) -> Any:
    """Downsampling feature-extractor stem (is_low_res_input=False): two stride-2
    convs (4x down) then the ResidualBlocksWithInputConv at feat_extract.4."""
    x = lrelu(conv(f, p, "feat_extract.0", stride=2, pad=1))
    x = lrelu(conv(x, p, "feat_extract.2", stride=2, pad=1))
    return compiled_resblocks(x, p, "feat_extract.4")


def _pad_mult4(f: Any) -> Any:
    """Replicate-pad bottom/right so H, W are multiples of 4 (the downsample factor)."""
    _, h, w, _ = f.shape
    ph, pw = (-h) % 4, (-w) % 4
    if ph:
        f = mx.concatenate(
            [f, mx.broadcast_to(f[:, h - 1 : h], (f.shape[0], ph, f.shape[2], f.shape[3]))], axis=1
        )
    if pw:
        f = mx.concatenate(
            [f, mx.broadcast_to(f[:, :, w - 1 : w], (f.shape[0], f.shape[1], pw, f.shape[3]))],
            axis=2,
        )
    return f


def restore(
    frames: list,
    p: dict,
    flow_mode: str = "spynet",
    vt_flow_services: Any = None,
) -> list:
    """1x recurrent restoration (decompress / denoise / deblur checkpoints). frames:
    list of (N,H,W,3) f32 [0,1]; out: same length and SAME size, restored. The net
    downsamples the input 4x, runs bidirectional second-order propagation at 1/4 res,
    upsamples back, and adds the original frame as the global residual (no bicubic
    upscale, unlike the SR path). Input is padded to a multiple of 4 and cropped back."""
    dt = p["conv_last.weight"].dtype
    orig = [(f.shape[1], f.shape[2]) for f in frames]
    padded = [_pad_mult4(mx.clip(f, 0.0, 1.0).astype(dt)) for f in frames]
    # Optical flow is computed on a bicubic-1/4 downsample of the input (reference).
    down = [_bicubic_down4(f) for f in padded]
    spatial = []
    for f in padded:
        s = _feat_extract_1x(f, p)
        mx.eval(s)  # materialize per frame, not all at once
        spatial.append(s)
    feats: dict = {"spatial": spatial}
    ff, fb = _compute_flows(
        down,
        p,
        flow_mode=flow_mode,
        vt_flow_services=vt_flow_services,
    )
    for it in (1, 2):
        for direction in ("backward", "forward"):
            mod = f"{direction}_{it}"
            feats = _propagate(feats, fb if direction == "backward" else ff, mod, p)
    outs = []
    for i in range(len(spatial)):
        hr = [feats["spatial"][i]] + [feats[k][i] for k in feats if k != "spatial"]
        residual = _compiled_upsample(p)(mx.concatenate(hr, axis=-1))
        oh, ow = orig[i]
        out = mx.clip(residual + padded[i], 0.0, 1.0)[:, :oh, :ow, :]
        mx.eval(out)
        outs.append(out)
    return outs


# ---- spatial self-ensemble (the reference's inference-time trick) -----------
# The NTIRE decompress + ntire-vsr configs run BasicVSR++ through an 8-way
# geometric self-ensemble (SpatialTemporalEnsemble, is_temporal_ensemble=False)
# and average -- how the challenge leaderboard numbers were reached. The original
# repo applies it in forward_test; mmagic's re-port left it as dead config. It is
# a genuine artifact-reducer: averaging 8 orientations cancels orientation-specific
# hallucinated texture (the aggressive checkpoints' flat-region "alligator skin")
# while keeping the orientation-consistent real signal -- measured ~2.5x less
# hallucination + ~1.7x less temporal crawl on track2, at 8x the compute.
def _flip(f: Any, ax: int) -> Any:
    return mx.take(f, (f.shape[ax] - 1) - mx.arange(f.shape[ax]), axis=ax)


def _geo_tf(f: Any, mode: str) -> Any:
    if mode == "v":
        return _flip(f, 1)  # flip H
    if mode == "h":
        return _flip(f, 2)  # flip W
    if mode == "t":
        return mx.transpose(f, (0, 2, 1, 3))  # swap H,W
    return f


def _spatial_ensemble(frames: list, run_fn) -> list:
    """8-way geometric self-ensemble, exact scheme from the reference
    mmedit/models/common/ensemble.py: build the 8 variants by successively applying
    vertical/horizontal/transpose, run_fn each, invert the transforms, average.
    run_fn(list_of_frames) -> list_of_outputs (same length). Works for the 1x
    restore and the 4x upscale paths alike (transforms are resolution-agnostic)."""
    lists = [frames]
    for mode in ("v", "h", "t"):
        lists = lists + [[_geo_tf(f, mode) for f in fl] for fl in lists]
    acc: list | None = None
    for i, fl in enumerate(lists):
        o = run_fn(fl)
        if i > 3:
            o = [_geo_tf(f, "t") for f in o]
        if i % 4 > 1:
            o = [_geo_tf(f, "h") for f in o]
        if (i % 4) % 2 == 1:
            o = [_geo_tf(f, "v") for f in o]
        acc = o if acc is None else [a + b for a, b in zip(acc, o, strict=True)]
        for a in acc:
            mx.eval(a)  # free each variant's graph before the next
    return [mx.clip(a * 0.125, 0.0, 1.0) for a in acc]


def restore_ensemble(
    frames: list,
    p: dict,
    flow_mode: str = "spynet",
    vt_flow_services: Any = None,
) -> list:
    """1x restoration under the reference's 8-way spatial self-ensemble (8x the
    cost of restore()). See _spatial_ensemble."""
    scope = (
        vt_flow_services_scope(vt_flow_services, max_geometries=2)
        if flow_mode == "vt"
        else nullcontext(None)
    )
    with scope as services:
        return _spatial_ensemble(
            frames,
            lambda fl: restore(
                fl,
                p,
                flow_mode=flow_mode,
                vt_flow_services=services,
            ),
        )


def upscale_ensemble(
    frames: list,
    p: dict,
    flow_mode: str = "spynet",
    history_strength: float = 1.0,
    history_gate: str = "off",
    vt_flow_services: Any = None,
) -> list:
    """4x SR under the reference's 8-way spatial self-ensemble -- the NTIRE
    ntire_vsr config declares it (the small reds4/vimeo SR configs do not). 8x the
    cost of upscale(). See _spatial_ensemble."""
    scope = (
        vt_flow_services_scope(vt_flow_services, max_geometries=2)
        if flow_mode == "vt"
        else nullcontext(None)
    )
    with scope as services:
        return _spatial_ensemble(
            frames,
            lambda fl: upscale(
                fl,
                p,
                flow_mode=flow_mode,
                history_strength=history_strength,
                history_gate=history_gate,
                vt_flow_services=services,
            ),
        )


_log = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = load_params()
    mx.random.seed(0)
    frames = [mx.clip(mx.random.uniform(shape=(1, 48, 64, 3)), 0, 1) for _ in range(5)]
    mx.eval(*frames)
    outs = upscale(frames, p)
    mx.eval(*outs)
    _log.info(
        f"upscale: {len(outs)} frames, 48x64 -> {outs[0].shape[1]}x{outs[0].shape[2]}, "
        f"center range [{float(mx.min(outs[2])):.3f}, {float(mx.max(outs[2])):.3f}]"
    )
