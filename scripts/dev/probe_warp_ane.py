"""Does the ANE lower resample/gather today? Re-run after OS updates.

The shipped SpyNet hybrid (kinovsr/modeling/spynet_ane.py) rests on
"resample/gather have no ANE lowering, so the warps stay in MLX" - and on
the measured corollary that embedding a warp in a graph drags the
NEIGHBORING convolutions off the ANE with it. This probe re-asks the
question empirically with the first-party stack: build
conv -> resample -> conv (a warp-in-graph pyramid level in miniature) and
conv -> gather -> conv, compile, and report the preferred and supported
devices per operation. It takes seconds; if a lowering ever appears after
an OS or silicon change, revisit the hybrid split then.

Reading the output: on macOS 26 / M1 Max both probes report the resample
and gather ops as supported by the CPU only, and the surrounding convs
prefer the CPU too - that is the drag effect, and the reason the split is
not "move just the warp".

Run: "$KINOVSR_PYTHON" scripts/dev/probe_warp_ane.py
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

import mlx.core as mx

from kinovsr.native.anemil import builder, runtime

_log = logging.getLogger("kinovsr.dev.probe_warp_ane")


def conv_weight(out_c, in_c, key):
    w = mx.random.normal(shape=(out_c, in_c, 3, 3), key=mx.random.key(key))
    return (w * 0.05).astype(mx.float32)


def emit_resample():
    g = builder.Graph()
    g.register_input("x", (1, 16, 64, 64))
    g.register_input("coordinates", (1, 64, 64, 2))
    t = g.conv2d("x", conv_weight(16, 16, 1), None, tag="pre")
    opname = g.n("warp")
    inputs = {
        "x": t,
        "coordinates": "coordinates",
        "sampling_mode": g.const_str(f"{opname}_sampling_mode_0", "bilinear"),
        "padding_mode": g.const_str(f"{opname}_padding_mode_0", "constant"),
        "padding_value": g.fp16_const(f"{opname}_padding_value_0",
                                      mx.array(0.0, dtype=mx.float16)),
        "coordinates_mode": g.const_str(f"{opname}_coordinates_mode_0",
                                        "unnormalized"),
        "align_corners": g.const_bool_scalar(f"{opname}_align_corners_0",
                                             False),
    }
    t = g.op("resample", inputs, opname, (1, 16, 64, 64))
    t = g.conv2d(t, conv_weight(16, 16, 2), None, tag="post")
    out = g.op("relu", {"x": t}, "out", g.shape[t])
    model = g.finish([("x", (1, 16, 64, 64)),
                      ("coordinates", (1, 64, 64, 2))], [], [out],
                     "warp probe: resample")
    return model, g.blob


def emit_gather():
    # Indices are a baked const here (the builder types model inputs fp16);
    # the real warp uses computed indices, so this is the EASIER case for
    # the ANE - if even const-index gather is refused, dynamic surely is.
    g = builder.Graph()
    g.register_input("x", (1, 16, 64, 64))
    t = g.conv2d("x", conv_weight(16, 16, 3), None, tag="pre")
    opname = g.n("pick")
    permuted = [(i * 7) % 64 for i in range(64)]
    inputs = {
        "x": t,
        "indices": g.const_i32(f"{opname}_indices_0", permuted),
        "axis": g.const_i32_scalar(f"{opname}_axis_0", 3),
        "batch_dims": g.const_i32_scalar(f"{opname}_batch_dims_0", 0),
        "validate_indices": g.const_bool_scalar(
            f"{opname}_validate_indices_0", False),
    }
    t = g.op("gather", inputs, opname, (1, 16, 64, 64))
    t = g.conv2d(t, conv_weight(16, 16, 4), None, tag="post")
    out = g.op("relu", {"x": t}, "out", g.shape[t])
    model = g.finish([("x", (1, 16, 64, 64))], [], [out],
                     "warp probe: gather")
    return model, g.blob


def probe(root: Path, name: str, emit) -> None:
    package = root / name / "model.mlpackage"
    model_bytes, blob = emit()
    builder.write_package(package, model_bytes, blob)
    try:
        compiled = runtime.compile_package(package)
    except RuntimeError as exc:
        _log.info("%s: DID NOT COMPILE (%s)", name, str(exc)[:160])
        return
    _log.info("%s: placement %s", name, runtime.placement(compiled))

    # Per-op detail: the SUPPORTED device list is the authoritative signal
    # on tiny probe models - every op prefers the CPU at this size because
    # there is not enough arithmetic to pull the cost model anywhere else.
    import CoreML
    from Foundation import NSURL

    config = CoreML.MLModelConfiguration.alloc().init()
    config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)
    holder, done = {}, threading.Event()

    def handler(plan, error):
        holder["plan"] = plan
        done.set()

    CoreML.MLComputePlan.loadContentsOfURL_configuration_completionHandler_(
        NSURL.fileURLWithPath_(str(compiled)), config, handler)
    if not done.wait(120) or holder.get("plan") is None:
        _log.info("%s: MLComputePlan did not load", name)
        return
    plan = holder["plan"]
    main_fn = plan.modelStructure().program().functions()["main"]
    for op in main_fn.block().operations():
        usage = plan.computeDeviceUsageForMLProgramOperation_(op)
        if usage is None:
            continue
        device = type(usage.preferredComputeDevice()).__name__
        supported = [type(d).__name__
                     for d in usage.supportedComputeDevices()]
        _log.info("  %s: preferred %s, supported %s",
                  op.operatorName(), device, supported)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Probe artifacts are disposable; the output IS the printed placement.
    # The models are only compiled and placement-planned, never predicted,
    # so the fp16-only ModelRunner constraint does not apply here.
    root = Path(tempfile.mkdtemp(prefix="kinovsr-warp-probe-"))
    probe(root, "resample", emit_resample)
    probe(root, "gather", emit_gather)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
