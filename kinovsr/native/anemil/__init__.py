"""First-party Core ML mlprogram authoring for the Neural Engine.

Author, compile, placement-gate, and drive ANE models with mlx + pyobjc
CoreML + protobuf - no coremltools anywhere. Every wheel involved is
either pure or abi3, so the stack installs on any CPython this project
supports, which is the property coremltools lacks and the reason this
package exists.

Layers:

- ``schema``  - Core ML protobuf message classes from the vendored
  descriptor set (``coreml_spec.fds``; provenance in the sidecar json).
  protobuf loads lazily, on first use, never at import.
- ``builder`` - ``Graph``: MIL op emitter with the ANE spellings verified
  at production scale, the fp16 weights blob writer, and the .mlpackage
  writer. Deterministic serialization: identical emissions produce
  identical packages byte for byte.
- ``runtime`` - native compile-and-persist, the ``MLComputePlan``
  placement gate, and an MLX-buffered ``ModelRunner`` (fixed fp16
  bindings, output backings, optional MLState, and per-dispatch dynamic
  rebinding via ``predict_with`` for caller-managed buffers).

Consumers: ``kinovsr.modeling.spynet_ane`` (stateless per-level flow
graphs) and ``kinovsr.processors.bsvd.ane`` (the stateful streaming
denoiser). New ANE processors should express their graph against
``builder`` and drive it with ``runtime``; the emission spellings here
carry the compiler constraints so a new port does not rediscover them.
"""

from . import builder, runtime, schema  # noqa: F401

__all__ = ["builder", "runtime", "schema"]
