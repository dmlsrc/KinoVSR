"""SpyNet optical flow with the convolutions on the Neural Engine.

Same math as :func:`kinovsr.modeling.vsr_blocks.spynet_flow`, split across
two compute pools: each pyramid level's five 7x7 convolutions (plus the flow
accumulate and 2x upsample, folded into the same graph) run as a CoreML model
pinned to the ANE, while MLX keeps the pyramid build, the warps, the channel
concat, and the final resize. Measured at 640x352: 20.8 ms against 28.5 ms
for the same split done naively and 52.2 ms for the pure-MLX path, at mean
end-point error 0.0037 px versus that path.

Why the split: the ANE runs this conv workload about 1.8x faster than the GPU
(16.3 vs 28.9 ms for SpyNet's 144 GFLOP), and it refuses the warps outright -
`resample`/`gather` have no ANE lowering at any layer of the stack, verified
against the private compiler, so the warps must live elsewhere regardless.
Keeping them in MLX also leaves the GPU mostly idle during flow, which is what
makes this composable with the learned stages that need it.

Two invariants that carry the performance, both easy to lose in a refactor:

- **Cross contiguous.** ``mx.transpose`` is lazy, and handing CoreML a strided
  buffer makes it repack internally, inside ``predict`` where no profiler
  points at it: 27.6 ms versus 17.7 ms for six dispatches, against 1.9 ms to
  call ``mx.contiguous`` here.
- **No numpy on the flow path.** Inputs cross as ``memoryview`` over evaluated
  MLX arrays wrapped in ``MLMultiArray``; outputs land directly in persistent
  MLX arrays via ``MLPredictionOptions.outputBackings``. Besides the workspace
  convention, going through coremltools' own ``predict`` silently upcasts fp16
  inputs to fp32 and back, which cost 0.9 ms.

Output-backing aliasing: CoreML rewrites level ``l``'s backing on the next
call to that level. Every consumer of a backing is evaluated within the same
:meth:`AneSpyNet.flow` call, and the returned array is freshly materialized,
so no lazy graph referencing a backing survives the call.

One model set is built per padded geometry and cached on disk (about 1.3 s to
convert, 0.5 s to load thereafter). Fixed shapes are a deliberate choice, not
a CoreML limitation: ``EnumeratedShapes`` also stays entirely on the ANE and
costs only 2-3 percent, but it still rejects sizes outside its menu, and since
the models are converted on first use and cached either way, a fixed shape is
the simplest thing that is also the fastest. (``RangeDim`` is the one to
avoid: same placement, but 3.7x slower away from its default shape.)
Conversion needs ``coremltools``; inference does not. Anything unavailable or
unsupported returns ``None`` from :func:`engine_for` and the caller stays on
the pure-MLX path.
"""

from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx

_MLMULTIARRAY_FLOAT16 = 0x10000 | 16  # MLMultiArrayDataTypeFloat16

# Engines are keyed by (weights identity, padded geometry, level count) and
# reused across frames; a run touches one or two geometries at most.
_ENGINES: dict[tuple, Any] = {}
_UNAVAILABLE: str | None = None


def _strides(shape: list[int]) -> list[int]:
    """Element strides (not bytes) for a C-contiguous shape."""
    out = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        out[i] = out[i + 1] * shape[i + 1]
    return out


def padded_geometry(height: int, width: int) -> tuple[int, int, int]:
    """Padded size and level count, matching ``spynet_flow`` exactly."""
    w_up = width if width % 32 == 0 else 32 * (width // 32 + 1)
    h_up = height if height % 32 == 0 else 32 * (height // 32 + 1)
    levels = 6 if w_up > 32 else max(1, int(math.log2(w_up)))
    return h_up, w_up, levels


def _weights_key(params: dict) -> str:
    """Stable short digest of the checkpoint, so a different SpyNet cannot
    collide with a cached conversion."""
    digest = hashlib.sha256()
    for key in sorted(params):
        if not key.startswith("spynet."):
            continue
        value = params[key]
        digest.update(key.encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
    sample = params.get("spynet.basic_module.0.basic_module.0.conv.weight")
    if sample is not None:
        flat = sample.reshape(-1)[:64].astype(mx.float32)
        mx.eval(flat)
        digest.update(bytes(memoryview(mx.contiguous(flat)).cast("B")))
    return digest.hexdigest()[:16]


def cache_root() -> Path:
    """Where converted models live (``KINOVSR_CACHE_DIR`` overrides)."""
    from kinovsr.settings import default_settings

    return Path(default_settings().cache_dir).expanduser()


# ---------------------------------------------------------------- conversion

def _build_level_program(params: dict, lvl: int, levels: int,
                         h: int, w: int):
    """One level's basic module. Every level but the last also folds in the
    flow accumulate and the 2x upsample the next level needs, so those run on
    the ANE instead of costing an extra MLX round trip."""
    import numpy as np  # dev/conversion path only, never on the flow path
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    import coremltools as ct

    base = f"spynet.basic_module.{lvl}.basic_module"
    weights, biases = [], []
    for j in range(5):
        wt = np.asarray(params[f"{base}.{j}.conv.weight"].astype(mx.float32))
        weights.append(wt.transpose(0, 3, 1, 2).astype(np.float16))
        biases.append(np.asarray(
            params[f"{base}.{j}.conv.bias"].astype(mx.float32)
        ).astype(np.float16))
    is_last = lvl == levels - 1

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 8, h, w),
                                           dtype=types.fp16)],
                opset_version=ct.target.iOS18)
    def program(x):
        t = x
        for j in range(5):
            t = mb.conv(x=t, weight=weights[j], bias=biases[j],
                        strides=[1, 1], pad_type="custom", pad=[3, 3, 3, 3],
                        name=f"c{j}")
            if j < 4:
                t = mb.relu(x=t, name=f"r{j}")
        if is_last:
            return t
        carried = mb.slice_by_index(x=x, begin=[0, 6, 0, 0], end=[1, 8, h, w],
                                    name="carried")
        flow = mb.add(x=t, y=carried, name="accumulate")
        up = mb.upsample_bilinear(x=flow, scale_factor_height=2,
                                  scale_factor_width=2, align_corners=True,
                                  name="upsample")
        return mb.mul(x=up, y=np.float16(2.0), name="scale")

    return program


def _convert_models(params: dict, levels: int, h_up: int, w_up: int,
                    directory: Path) -> None:
    import coremltools as ct

    directory.mkdir(parents=True, exist_ok=True)
    for lvl in range(levels):
        package = directory / f"level{lvl}.mlpackage"
        if package.exists():
            continue
        h = h_up >> (levels - 1 - lvl)
        w = w_up >> (levels - 1 - lvl)
        model = ct.convert(
            _build_level_program(params, lvl, levels, h, w),
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        # Write beside the target and rename, so a crashed or concurrent
        # conversion cannot leave a half-written package in the cache. The
        # staging name keeps the .mlpackage suffix that the writer requires.
        staging = directory / f"level{lvl}.partial.mlpackage"
        shutil.rmtree(staging, ignore_errors=True)
        model.save(str(staging))
        staging.replace(package)


# ------------------------------------------------------------------- runtime

class _LevelRunner:
    """Compiled CoreML models driven directly through pyobjc, with MLX
    buffers on both sides of the fence."""

    def __init__(self, packages: list[Path]):
        import CoreML

        self._CoreML = CoreML
        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)
        self.levels: list[dict] = []
        for package in packages:
            model = self._load(package, config)
            self.levels.append(self._bind(model, model.modelDescription()))

    def _load(self, package: Path, config: Any) -> Any:
        """Load a level, preferring a persisted compiled model.

        Compiling an .mlpackage costs about 110 ms per level; loading the
        compiled .mlmodelc beside it costs about 10 ms, so persisting the
        compiled form takes engine construction from ~655 ms to ~57 ms for
        the six levels. Compiled models are tied to the OS and hardware, so
        a stale one that fails to load is rebuilt rather than trusted.
        """
        import CoreML
        from Foundation import NSURL

        compiled_dir = package.with_suffix(".mlmodelc")
        if compiled_dir.exists():
            model, _ = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
                NSURL.fileURLWithPath_(str(compiled_dir)), config, None)
            if model is not None:
                return model
            shutil.rmtree(compiled_dir, ignore_errors=True)

        compiled, err = CoreML.MLModel.compileModelAtURL_error_(
            NSURL.fileURLWithPath_(str(package)), None)
        if compiled is None:
            raise RuntimeError(f"CoreML compile failed: {err}")
        try:
            staging = package.with_suffix(".partial.mlmodelc")
            shutil.rmtree(staging, ignore_errors=True)
            shutil.copytree(compiled.path(), staging)
            staging.replace(compiled_dir)
        except OSError:
            # A cache that cannot be written costs startup time, not
            # correctness; the freshly compiled model is still usable.
            pass
        model, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
            compiled, config, None)
        if model is None:
            raise RuntimeError(f"CoreML load failed: {err}")
        return model

    def _multi_array(self, array: Any, shape: list[int]) -> Any:
        view = memoryview(array)
        multi, err = (self._CoreML.MLMultiArray.alloc()
                      .initWithDataPointer_shape_dataType_strides_deallocator_error_(
                          view, shape, _MLMULTIARRAY_FLOAT16,
                          _strides(shape), None, None))
        if multi is None:
            raise RuntimeError(f"MLMultiArray wrap failed: {err}")
        return multi

    def _bind(self, model: Any, description: Any) -> dict:
        CoreML = self._CoreML
        inputs = []
        features = {}
        for name in description.inputDescriptionsByName():
            constraint = description.inputDescriptionsByName()[name].multiArrayConstraint()
            shape = [int(d) for d in constraint.shape()]
            array = mx.zeros(shape, dtype=mx.float16)
            mx.eval(array)
            multi = self._multi_array(array, shape)
            features[name] = CoreML.MLFeatureValue.featureValueWithMultiArray_(multi)
            inputs.append((array, memoryview(array).cast("B"), multi))
        provider, err = (CoreML.MLDictionaryFeatureProvider.alloc()
                         .initWithDictionary_error_(features, None))
        if provider is None:
            raise RuntimeError(f"feature provider failed: {err}")

        backings = {}
        outputs = []
        retained = []
        for name in description.outputDescriptionsByName():
            constraint = description.outputDescriptionsByName()[name].multiArrayConstraint()
            shape = [int(d) for d in constraint.shape()]
            array = mx.zeros(shape, dtype=mx.float16)
            mx.eval(array)
            multi = self._multi_array(array, shape)
            backings[name] = multi
            outputs.append(array)
            retained.append(multi)
        options = CoreML.MLPredictionOptions.alloc().init()
        options.setOutputBackings_(backings)
        return {"model": model, "provider": provider, "inputs": inputs,
                "options": options, "outputs": outputs, "_retained": retained}

    def run(self, lvl: int, tensor: Any) -> Any:
        """Feed one evaluated, contiguous fp16 (1,8,h,w) tensor; return the
        persistent output array (consume before this level runs again)."""
        entry = self.levels[lvl]
        entry["inputs"][0][1][:] = memoryview(tensor).cast("B")
        result, err = entry["model"].predictionFromFeatures_options_error_(
            entry["provider"], entry["options"], None)
        if result is None:
            raise RuntimeError(f"ANE prediction failed: {err}")
        return entry["outputs"][0]


class AneSpyNet:
    """SpyNet flow for one padded geometry, ANE convolutions + MLX glue."""

    def __init__(self, params: dict, packages: list[Path], levels: int,
                 h_up: int, w_up: int):
        self.levels = levels
        self.h_up, self.w_up = h_up, w_up
        self._runner = _LevelRunner(packages)

        mean = params["spynet.mean"].reshape(1, 3, 1, 1).astype(mx.float32)
        std = params["spynet.std"].reshape(1, 3, 1, 1).astype(mx.float32)
        h0, w0 = h_up >> (levels - 1), w_up >> (levels - 1)
        zeros = mx.zeros((1, 2, h0, w0), dtype=mx.float16)
        mx.eval(mean, std, zeros)

        def start(ref, supp):
            # Level 0 carries a zero flow, so its warp is the identity and
            # its input is just [ref, supp, 0] - the warp is skipped.
            refs = [(mx.transpose(ref, (0, 3, 1, 2)) - mean) / std]
            supps = [(mx.transpose(supp, (0, 3, 1, 2)) - mean) / std]
            for _ in range(levels - 1):
                for pyramid in (refs, supps):
                    n, c, hh, ww = pyramid[-1].shape
                    pyramid.append(pyramid[-1].reshape(
                        n, c, hh // 2, 2, ww // 2, 2).mean(axis=(3, 5)))
            refs = refs[::-1]
            supps = supps[::-1]
            first = mx.concatenate([refs[0].astype(mx.float16),
                                    supps[0].astype(mx.float16), zeros],
                                   axis=1)
            return ([first]
                    + [r.astype(mx.float16) for r in refs[1:]]
                    + [s.astype(mx.float16) for s in supps[1:]])

        def make_prep(lvl):
            h = h_up >> (levels - 1 - lvl)
            w = w_up >> (levels - 1 - lvl)
            gy, gx = mx.meshgrid(mx.arange(h, dtype=mx.float32),
                                 mx.arange(w, dtype=mx.float32),
                                 indexing="ij")
            mx.eval(gy, gx)

            def prep(up16, supp, ref16):
                up = up16.astype(mx.float32)
                sx = gx + up[:, 0]
                sy = gy + up[:, 1]
                fy = mx.floor(sy)
                fx = mx.floor(sx)
                wy = (sy - fy)[:, None]
                wx = (sx - fx)[:, None]
                iy, ix = fy.astype(mx.int32), fx.astype(mx.int32)
                y0 = mx.clip(iy, 0, h - 1)
                y1 = mx.clip(iy + 1, 0, h - 1)
                x0 = mx.clip(ix, 0, w - 1)
                x1 = mx.clip(ix + 1, 0, w - 1)
                flat = supp.reshape(3, h * w)

                def gather(yc, xc):
                    return mx.take(flat, (yc * w + xc).reshape(-1),
                                   axis=1).reshape(1, 3, h, w)

                warped = ((1 - wy) * (1 - wx) * gather(y0, x0)
                          + (1 - wy) * wx * gather(y0, x1)
                          + wy * (1 - wx) * gather(y1, x0)
                          + wy * wx * gather(y1, x1))
                return mx.concatenate(
                    [ref16, warped.astype(mx.float16), up16], axis=1)
            return prep

        def finish(carried, residual):
            flow = carried.astype(mx.float32) + residual.astype(mx.float32)
            return mx.contiguous(mx.transpose(flow, (0, 2, 3, 1)))

        self._start = mx.compile(start)
        self._prep = [None] + [mx.compile(make_prep(lvl))
                               for lvl in range(1, levels)]
        self._finish = mx.compile(finish)

    def flow(self, ref: Any, supp: Any) -> Any:
        """(1,H,W,3) in [0,1] -> (1,H,W,2), same convention as spynet_flow."""
        from kinovsr.modeling.vsr_blocks import resize

        h, w = int(ref.shape[1]), int(ref.shape[2])
        ref32 = ref.astype(mx.float32)
        supp32 = supp.astype(mx.float32)
        if (h, w) != (self.h_up, self.w_up):
            ref32 = resize(ref32, self.h_up, self.w_up, False)
            supp32 = resize(supp32, self.h_up, self.w_up, False)

        stages = self._start(ref32, supp32)
        tensor = stages[0]
        refs = stages[1:self.levels]
        supps = stages[self.levels:]
        mx.eval(tensor)
        carried = self._runner.run(0, tensor)
        previous = None
        for lvl in range(1, self.levels):
            previous = carried
            tensor = self._prep[lvl](carried, supps[lvl - 1], refs[lvl - 1])
            mx.eval(tensor)
            carried = self._runner.run(lvl, tensor)
        flow = self._finish(previous, carried)

        if (h, w) != (self.h_up, self.w_up):
            flow = resize(flow, h, w, False)
            flow = mx.stack([flow[..., 0] * (w / self.w_up),
                             flow[..., 1] * (h / self.h_up)], axis=-1)
        mx.eval(flow)
        return flow


# ------------------------------------------------------------ entry point

def unavailable_reason() -> str | None:
    """Why the ANE backend cannot be used here, or ``None`` if it can."""
    global _UNAVAILABLE
    if _UNAVAILABLE is not None:
        return _UNAVAILABLE or None
    try:
        import CoreML  # noqa: F401
        from Foundation import NSURL  # noqa: F401
    except ImportError as exc:
        _UNAVAILABLE = f"CoreML bindings unavailable ({exc})"
        return _UNAVAILABLE
    _UNAVAILABLE = ""
    return None


def engine_for(params: dict, shape: tuple) -> AneSpyNet | None:
    """Engine for this batch/geometry, or ``None`` to stay on pure MLX.

    Conversion happens once per geometry and is cached on disk; a failure
    anywhere here is reported once and then remembered as unavailable, so a
    run never pays a broken path twice.
    """
    global _UNAVAILABLE
    if unavailable_reason() is not None:
        return None
    n, h, w = int(shape[0]), int(shape[1]), int(shape[2])
    if n != 1:
        return None                       # batched flow stays on MLX
    h_up, w_up, levels = padded_geometry(h, w)
    # Below one pyramid stride the frame is mostly padding and the per-level
    # dispatches cost more than the convolutions they carry; MLX wins.
    if levels < 2 or min(h, w) < 32:
        return None
    key = (_weights_key(params), h_up, w_up, levels)
    engine = _ENGINES.get(key)
    if engine is not None:
        return engine
    directory = cache_root() / "spynet-ane" / f"{key[0]}-{w_up}x{h_up}"
    packages = [directory / f"level{i}.mlpackage" for i in range(levels)]
    # Import coremltools ONLY when something actually has to be converted: it
    # pulls in its torch integration and costs over a second, which would be
    # charged to every warm start for no reason.
    needs_conversion = not all(p.exists() for p in packages)
    if needs_conversion:
        try:
            import coremltools  # noqa: F401
        except ImportError as exc:
            _UNAVAILABLE = (
                f"coremltools is needed once per geometry to build the ANE "
                f"models ({exc})")
            return None
    try:
        if needs_conversion:
            _convert_models(params, levels, h_up, w_up, directory)
        engine = AneSpyNet(params, packages, levels, h_up, w_up)
    except Exception as exc:  # noqa: BLE001 - any failure falls back
        _UNAVAILABLE = f"ANE SpyNet setup failed ({type(exc).__name__}: {exc})"
        return None
    _ENGINES[key] = engine
    return engine


def reset_cache() -> None:
    """Drop live engines and the availability verdict (tests)."""
    global _UNAVAILABLE
    _ENGINES.clear()
    _UNAVAILABLE = None


__all__ = ["AneSpyNet", "engine_for", "padded_geometry", "cache_root",
           "reset_cache", "unavailable_reason"]
