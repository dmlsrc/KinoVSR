"""Native Core ML compile / placement / prediction plumbing, pyobjc only.

Shared by every anemil processor: compile an .mlpackage and persist the
.mlmodelc, refuse models with any non-ANE-preferred operation (native
MLComputePlan - the same API coremltools wraps), and drive predictions with
MLX-backed buffers on both sides of the fence (memoryview in, output
backings out, optional MLState). Features can opt out of fixed binding
(`dynamic=` prefixes) and be bound per dispatch via `predict_with` - the
copy-free path BSVD's skip rings ride.
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

import mlx.core as mx

from kinovsr.native.dispatch import (  # noqa: F401  (re-export for
    DispatchPipeline,  # this module's existing consumers)
)

_MLMULTIARRAY_FLOAT16 = 0x10000 | 16  # MLMultiArrayDataTypeFloat16


def bind_array(shape):
    """MLX fp16 buffer wrapped as an MLMultiArray without copying.

    Returns (mlx array, flat byte view, MLMultiArray). The caller must
    keep the tuple alive for as long as the MLMultiArray is in use: the
    MLMultiArray borrows the MLX buffer (deallocator None).
    """
    import CoreML

    array = mx.zeros(shape, dtype=mx.float16)
    mx.eval(array)
    strides, acc = [1] * len(shape), 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i], acc = acc, acc * shape[i]
    multi, error = (
        CoreML.MLMultiArray.alloc()
        .initWithDataPointer_shape_dataType_strides_deallocator_error_(
            memoryview(array), [int(d) for d in shape],
            _MLMULTIARRAY_FLOAT16, strides, None, None))
    if multi is None:
        raise RuntimeError(f"MLMultiArray wrap failed: {error}")
    return array, memoryview(array).cast("B"), multi


def compile_package(package: Path) -> Path:
    """Compile an .mlpackage and persist the .mlmodelc beside it."""
    import CoreML
    from Foundation import NSURL

    compiled_dir = package.with_suffix(".mlmodelc")
    if compiled_dir.is_dir():
        return compiled_dir
    compiled, error = CoreML.MLModel.compileModelAtURL_error_(
        NSURL.fileURLWithPath_(str(package)), None)
    if compiled is None:
        raise RuntimeError(f"Core ML compile failed: {error}")
    staging = package.with_suffix(".partial.mlmodelc")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(str(compiled.path()), staging)
    staging.replace(compiled_dir)
    return compiled_dir


def _plan_operations(compiled: Path, timeout: float,
                     function_name: str | None):
    """Yield (operation, preferred-device class name) over the selected
    function of a compiled ML program's MLComputePlan."""
    import CoreML
    from Foundation import NSURL

    config = CoreML.MLModelConfiguration.alloc().init()
    config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)
    if function_name is not None:
        config.setFunctionName_(function_name)
    holder: dict[str, Any] = {}
    done = threading.Event()

    def handler(plan, error):
        holder["plan"], holder["error"] = plan, error
        done.set()

    CoreML.MLComputePlan.loadContentsOfURL_configuration_completionHandler_(
        NSURL.fileURLWithPath_(str(compiled)), config, handler)
    if not done.wait(timeout):
        raise RuntimeError("MLComputePlan load timed out")
    plan = holder.get("plan")
    if plan is None:
        raise RuntimeError(f"MLComputePlan failed: {holder.get('error')}")
    program = plan.modelStructure().program()
    if program is None:
        raise RuntimeError("compiled model is not an ML program")
    selected = function_name or "main"
    main = program.functions()[selected]
    for operation in main.block().operations():
        usage = plan.computeDeviceUsageForMLProgramOperation_(operation)
        if usage is None:
            continue
        yield operation, type(usage.preferredComputeDevice()).__name__


def _operation_outputs(operation) -> list[str]:
    """Output value names of an MLModelStructureProgramOperation."""
    return [str(named.name()) for named in operation.outputs()]


def placement(compiled: Path, timeout: float = 300.0,
              function_name: str | None = None) -> dict:
    """Preferred compute device counts, by device class name."""
    preferred: dict[str, int] = {}
    for _operation, device in _plan_operations(
            compiled, timeout, function_name):
        preferred[device] = preferred.get(device, 0) + 1
    return preferred


def assert_all_ane(compiled: Path,
                   function_name: str | None = None,
                   allow_prefixes: tuple[str, ...] = ()) -> dict:
    """Refuse a model that would run any operation off the ANE.

    `allow_prefixes` exempts operations whose every output name starts
    with one of the prefixes - the deliberate float32 translation-island
    ops ("island_") are value-exact on any device, unlike the fp16
    convolutions this gate exists to keep off the CPU. Placement is a
    preference, not a realized trace; pair this with a runtime oracle
    (replay or differential canary) per processor.
    """
    preferred: dict[str, int] = {}
    stray: dict[str, int] = {}
    stray_names: list[str] = []
    for operation, device in _plan_operations(compiled, 300.0,
                                              function_name):
        preferred[device] = preferred.get(device, 0) + 1
        if "NeuralEngine" in device:
            continue
        outputs = _operation_outputs(operation)
        if allow_prefixes and outputs and all(
                name.startswith(allow_prefixes) for name in outputs):
            continue
        stray[device] = stray.get(device, 0) + 1
        stray_names.extend(outputs[:2])
    if stray:
        raise RuntimeError(
            f"{stray} operations are not ANE-preferred (first: "
            f"{stray_names[:4]}); refusing because a partial fallback can "
            f"violate the requesting processor's performance or numerical "
            f"contract.")
    return preferred


class ModelRunner:
    """Compiled model with every fp16 input/output bound to an MLX buffer.

    Byte views blit data in; output backings land results in persistent MLX
    arrays. `predict()` runs one dispatch (with MLState when the model
    declares state). Features whose name starts with one of the `dynamic`
    prefixes get NO fixed buffer: the caller binds them per dispatch with
    `predict_with` (allocate via `bind_array`) - this is how BSVD's skip
    rings avoid per-step byte copies. Processor-specific stepping (FIFOs,
    gate defaults) wraps this rather than living in it.
    """

    def __init__(self, compiled: Path, compute_units: str = "ane",
                 fast_prediction: bool = False,
                 dynamic: tuple[str, ...] = (),
                 function_name: str | None = None,
                 state: Any | None = None):
        import CoreML
        from Foundation import NSURL

        self._CoreML = CoreML
        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(
            CoreML.MLComputeUnitsCPUOnly if compute_units == "cpu"
            else CoreML.MLComputeUnitsCPUAndNeuralEngine)
        if function_name is not None:
            config.setFunctionName_(function_name)
        if fast_prediction:
            # MLSpecializationStrategyFastPrediction: spend more at load
            # specializing the plan for faster predictions afterward.
            # Measured 2026-07-19 (macOS 26, M1 Max): FAILS the plan build
            # with E5RT -14 on the stateful BSVD graph (a model that loads
            # fine under the default strategy), and is measurement-noise
            # neutral on the stateless SpyNet levels. Kept as an opt-in
            # because the tradeoff is documented to pay on other model
            # shapes; never set it on an MLState graph.
            hints = CoreML.MLOptimizationHints.alloc().init()
            hints.setSpecializationStrategy_(
                CoreML.MLSpecializationStrategyFastPrediction)
            config.setOptimizationHints_(hints)
        model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
            NSURL.fileURLWithPath_(str(compiled)), config, None)
        if model is None:
            raise RuntimeError(f"model load failed: {error}")
        self._model = model
        self._dynamic = tuple(dynamic)
        description = model.modelDescription()
        self._stateful = len(description.stateDescriptionsByName()) > 0
        if state is not None and not self._stateful:
            raise ValueError("cannot attach MLState to a stateless function")
        self._state = (state if state is not None
                       else model.newState() if self._stateful else None)
        self.function_name = function_name

        self.inputs, self.dynamic_inputs = self._bind(
            description.inputDescriptionsByName())
        self.outputs, self.dynamic_outputs = self._bind(
            description.outputDescriptionsByName())
        provider, error = (
            CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
                {n: CoreML.MLFeatureValue.featureValueWithMultiArray_(
                    self.inputs[n][2]) for n in self.inputs}, None))
        if provider is None:
            raise RuntimeError(f"feature provider failed: {error}")
        self._provider = provider
        options = CoreML.MLPredictionOptions.alloc().init()
        options.setOutputBackings_({n: self.outputs[n][2]
                                    for n in self.outputs})
        self._options = options

    def _bind(self, descriptions):
        """Fixed features get persistent MLX-backed bindings; features
        matching a `dynamic` prefix are recorded shape-only for the
        caller to bind per dispatch."""
        bound, dynamic = {}, {}
        for name in descriptions:
            shape = tuple(int(d) for d in
                          descriptions[name].multiArrayConstraint().shape())
            if self._dynamic and name.startswith(self._dynamic):
                dynamic[name] = shape
                continue
            bound[name] = bind_array(shape)
        return bound, dynamic

    def input_view(self, name: str):
        return self.inputs[name][1]

    def output_view(self, name: str):
        return self.outputs[name][1]

    def output_array(self, name: str):
        return self.outputs[name][0]

    def reset_state(self) -> None:
        if self._stateful:
            self._state = self._model.newState()

    def _dispatch(self, provider, options):
        if self._stateful:
            result, error = (self._model
                             .predictionFromFeatures_usingState_options_error_(
                                 provider, self._state, options, None))
        else:
            result, error = self._model.predictionFromFeatures_options_error_(
                provider, options, None)
        if result is None:
            raise RuntimeError(f"prediction failed: {error}")
        return result

    def predict(self):
        if self.dynamic_inputs or self.dynamic_outputs:
            raise RuntimeError(
                "model has dynamic features; bind them via predict_with()")
        return self._dispatch(self._provider, self._options)

    def predict_with(self, features, backings):
        """One dispatch with caller-supplied MLMultiArray bindings for the
        dynamic features; the fixed bindings are merged in automatically.

        Every dynamic output MUST appear in `backings`: Core ML would
        otherwise write that output to its own allocation and the caller's
        buffer would silently keep stale data. Building the provider and
        options per call costs ~0.03 ms; prebinding the whole binding
        cycle measured no faster (2026-07-19), so keep this dynamic.
        """
        missing = self.dynamic_inputs.keys() - features.keys()
        if missing:
            raise RuntimeError(f"missing dynamic inputs: {sorted(missing)}")
        missing = self.dynamic_outputs.keys() - backings.keys()
        if missing:
            raise RuntimeError(
                f"missing dynamic output backings: {sorted(missing)}")
        CoreML = self._CoreML
        merged = {name: bound[2] for name, bound in self.inputs.items()}
        merged.update(features)
        provider, error = (
            CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
                {name: CoreML.MLFeatureValue.featureValueWithMultiArray_(multi)
                 for name, multi in merged.items()}, None))
        if provider is None:
            raise RuntimeError(f"feature provider failed: {error}")
        options = CoreML.MLPredictionOptions.alloc().init()
        merged = {name: bound[2] for name, bound in self.outputs.items()}
        merged.update(backings)
        options.setOutputBackings_(merged)
        return self._dispatch(provider, options)


# DispatchPipeline moved to kinovsr.native.dispatch (it is shared by
# the MPSGraph backend); re-exported here for its existing consumers.


def mean_abs(a: list, b: list, skip_first: int) -> float:
    """Mean over steps of mean absolute difference, in fp32 MLX."""
    diffs = [mx.abs(x - y).mean() for x, y in
             zip(a[skip_first:], b[skip_first:], strict=True)]
    value = mx.mean(mx.stack(diffs))
    mx.eval(value)
    return float(value)
