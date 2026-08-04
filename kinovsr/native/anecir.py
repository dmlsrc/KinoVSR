"""Explicit lifecycle execution for cached MPSGraph ANECIR products.

MPSGraph remains the compiler.  This module consumes its durable netplist,
weights, compiler options, and semantic I/O contract, then executes each
entry through ``_ANEClient`` with exactly one program resident at a time.
That is the lifecycle Core ML provides around ``MLState`` and the mapped
MPSGraph runtime does not provide reliably when large entries alternate.

The public surface mirrors ``mpsgraph_state.StatefulExecutable`` so model
families can select this runtime without owning IOSurface, live-port-order,
cache-miss, or load/evaluate/unload details.
"""

from __future__ import annotations

import hashlib
import logging
import plistlib
import struct
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx

from .anemil import direct

_QOS = 33
_CACHE_FLAG = "kANEFModelHasCacheURLIdentifierKey"
_log = logging.getLogger("kinovsr.anecir")
_CLIENT_LOCK = threading.Lock()
_CLIENT_READY = False
_RESIDENT: str | None = None
_RESIDENT_LOCK = threading.Lock()


def _error_code(error: Any) -> int | None:
    if error is None:
        return None
    value = getattr(error, "code", None)
    try:
        return int(value() if callable(value) else value)
    except (TypeError, ValueError):
        return None


def _client_bridge() -> tuple[Any, Any, Any]:
    """Return the private client/model classes after one metadata setup."""
    global _CLIENT_READY
    bridge = direct.preflight()
    objc = bridge._objc
    with _CLIENT_LOCK:
        if not _CLIENT_READY:
            for selector, error_index in (
                (b"loadModel:options:qos:error:", 5),
                (b"unloadModel:options:qos:error:", 5),
                (b"mapIOSurfacesWithModel:request:cacheInference:error:", 5),
                (b"evaluateWithModel:options:request:qos:error:", 6),
            ):
                objc.registerMetaDataForSelector(
                    b"_ANEClient",
                    selector,
                    {"arguments": {error_index: {"type_modifier": b"o"}}},
                )
            _CLIENT_READY = True
    return (
        objc.lookUpClass("_ANEClient").sharedConnection(),
        objc.lookUpClass("_ANEModel"),
        objc,
    )


def _cache_identifier(product: Path, compiler_options: Path) -> str:
    digest = hashlib.sha256()
    for path in (product, product.with_suffix(".weights"), compiler_options):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest().upper()


class _ANECIRModel:
    """One raw ANECIR model with cache-aware explicit residency."""

    def __init__(self, product: Path, region: str, label: str):
        from Foundation import NSURL

        self.product = Path(product)
        self.region = region
        self.label = label
        self.compiler_options = (
            self.product.parent / f"compiler_options_{region}.plist"
        )
        for path in (
            self.product,
            self.product.with_suffix(".weights"),
            self.compiler_options,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label}: missing ANECIR artifact {path}")
        self._client, self._model_class, _objc = _client_bridge()
        self._url = NSURL.fileURLWithPath_isDirectory_(
            str(self.product.parent), True
        )
        self._identity = _cache_identifier(
            self.product, self.compiler_options
        )
        self._options = {
            "kANEFCompilerOptionsFilenameKey": self.compiler_options.name,
            "kANEFEnableFWToFWSignal": False,
            "kANEFEnableLateLatchKey": False,
            _CACHE_FLAG: True,
            "kANEFModelType": "kANEFModelANECIR",
            "kANEFNetPlistFilenameKey": self.product.name,
            "kANEFRetainModelsWithoutSourceURLKey": True,
        }
        self.model = self._new_model()
        self._active_options: dict[str, Any] | None = None
        self._refresh_after_unload = False
        self._loaded = False
        self._requests: dict[tuple[tuple[int, ...], tuple[int, ...]], Any] = {}
        self._mapped_requests: dict[int, Any] = {}

    def _new_model(self) -> Any:
        model = self._model_class.alloc(
        ).initWithModelAtURL_key_identifierSource_cacheURLIdentifier_modelAttributes_standardizeURL_(
            self._url, self.region, 3, self._identity, {}, True
        )
        if model is None:
            raise RuntimeError(f"{self.label}: raw ANECIR model creation failed")
        return model

    def _claim(self) -> None:
        global _RESIDENT
        with _RESIDENT_LOCK:
            if _RESIDENT is not None:
                raise RuntimeError(
                    f"ANECIR program {_RESIDENT!r} is still resident; "
                    f"refusing to co-load {self.label!r}"
                )
            _RESIDENT = self.label

    def _release(self) -> None:
        global _RESIDENT
        with _RESIDENT_LOCK:
            if self.label == _RESIDENT:
                _RESIDENT = None

    def load(self) -> None:
        if self._loaded:
            raise RuntimeError(f"{self.label}: ANECIR model is already loaded")
        self._claim()
        try:
            cached = dict(self._options)
            ok, error = self._client.loadModel_options_qos_error_(
                self.model, cached, _QOS, None
            )
            if not ok and _error_code(error) == 16:
                # A deterministic identifier is also how ANE names its compiled
                # cache.  Code 16 here means that cache has not been populated;
                # a fresh model with the flag cleared performs the one-time
                # compile.
                self.model = self._new_model()
                uncached = dict(self._options)
                uncached[_CACHE_FLAG] = False
                ok, error = self._client.loadModel_options_qos_error_(
                    self.model, uncached, _QOS, None
                )
                if ok:
                    self._active_options = uncached
                    self._refresh_after_unload = True
                    _log.debug("%s populated raw ANECIR cache", self.label)
            else:
                self._active_options = cached if ok else None
            if not ok:
                raise RuntimeError(f"{self.label}: ANECIR load failed: {error}")
        except BaseException:
            self._active_options = None
            self._refresh_after_unload = False
            self._release()
            raise
        self._loaded = True

    def unload(self) -> None:
        if not self._loaded:
            return
        options = self._active_options or self._options
        refresh = self._refresh_after_unload
        unmap_error: BaseException | None = None
        for request in self._mapped_requests.values():
            try:
                self._client.unmapIOSurfacesWithModel_request_(
                    self.model, request
                )
            except BaseException as error:
                if unmap_error is None:
                    unmap_error = error
        self._mapped_requests.clear()
        try:
            ok, error = self._client.unloadModel_options_qos_error_(
                self.model, options, _QOS, None
            )
        finally:
            self._loaded = False
            self._active_options = None
            self._refresh_after_unload = False
            self._release()
        if not ok:
            raise RuntimeError(f"{self.label}: ANECIR unload failed: {error}")
        if unmap_error is not None:
            raise RuntimeError(
                f"{self.label}: ANECIR request unmap failed"
            ) from unmap_error
        if refresh:
            # The object that performed compile-as-needed can retain a stale
            # program/intermediate handle.  Reopen the now-populated cache so
            # the first inference uses the same clean model state as every
            # subsequent process.
            self.model = self._new_model()

    def attributes(self) -> Mapping[str, Any]:
        if not self._loaded:
            raise RuntimeError(f"{self.label}: ANECIR model is not loaded")
        return self.model.modelAttributes()

    def _request(
        self,
        inputs: Sequence[direct.Port],
        outputs: Sequence[direct.Port],
    ) -> Any:
        key = (
            tuple(id(port.surface) for port in inputs),
            tuple(id(port.surface) for port in outputs),
        )
        request = self._requests.get(key)
        if request is not None:
            return request
        bridge = direct.preflight()
        request = bridge.request_class.requestWithInputs_inputIndices_outputs_outputIndices_procedureIndex_(
            [port.surface.wrapped for port in inputs],
            list(range(len(inputs))),
            [port.surface.wrapped for port in outputs],
            list(range(len(outputs))),
            0,
        )
        if request is None:
            raise RuntimeError(f"{self.label}: ANECIR request creation failed")
        self._requests[key] = request
        return request

    def evaluate(
        self,
        inputs: Sequence[direct.Port],
        outputs: Sequence[direct.Port],
    ) -> None:
        """Evaluate one stable binding map while this program is resident."""
        if not self._loaded:
            raise RuntimeError(f"{self.label}: ANECIR model is not loaded")
        request = self._request(inputs, outputs)
        request_id = id(request)
        if request_id not in self._mapped_requests:
            ok, error = self._client.mapIOSurfacesWithModel_request_cacheInference_error_(
                self.model, request, True, None
            )
            if not ok:
                raise RuntimeError(f"{self.label}: ANECIR map failed: {error}")
            self._mapped_requests[request_id] = request
        ok, error = self._client.evaluateWithModel_options_request_qos_error_(
            self.model, {}, request, _QOS, None
        )
        if not ok:
            raise RuntimeError(
                f"{self.label}: ANECIR evaluation failed: {error}"
            )

    def dispatch(
        self,
        inputs: Sequence[direct.Port],
        outputs: Sequence[direct.Port],
    ) -> None:
        """Compatibility wrapper for one isolated load/evaluate/unload."""
        self.load()
        try:
            self.evaluate(inputs, outputs)
        finally:
            self.unload()

    def close(self) -> None:
        if self._loaded:
            self.unload()
        self._requests.clear()
        self._mapped_requests.clear()


def _symbol(info: Mapping[str, Any]) -> str:
    return str(info.get("Symbol") or "").split("@", 1)[0]


def _shape_for_port(port: direct.Port) -> tuple[int, ...]:
    batch, channels, depth, height, width = port.logical_dims()
    if depth != 1:
        raise RuntimeError(
            f"ANECIR port {port.name!r} has unsupported depth {depth}"
        )
    return batch, channels, height, width


def _ordered_infos(
    *,
    label: str,
    semantic_names: Sequence[str],
    product_symbols: Sequence[str],
    live_infos: Sequence[Mapping[str, Any]],
    kind: str,
) -> list[tuple[str, Mapping[str, Any]]]:
    """Map semantic names onto ANE's live table, preserving live order."""
    if len(semantic_names) != len(product_symbols):
        raise RuntimeError(
            f"{label}: {kind} contract has {len(semantic_names)} names for "
            f"{len(product_symbols)} product symbols"
        )
    names_by_symbol = dict(zip(product_symbols, semantic_names, strict=True))
    live_symbols = [_symbol(info) for info in live_infos]
    if len(set(live_symbols)) != len(live_symbols) or set(live_symbols) != set(
        product_symbols
    ):
        raise RuntimeError(f"{label}: live {kind} symbols changed")
    # The live table is intentionally authoritative.  ANE sorts numeric-looking
    # symbols lexically (__arg10 before __arg2); request indices address this
    # order rather than the netplist Inputs/Outputs arrays.
    return [
        (names_by_symbol[symbol], info)
        for symbol, info in zip(live_symbols, live_infos, strict=True)
    ]


class TensorBinding:
    """An entry-specific port view over a reusable IOSurface."""

    __slots__ = ("_owner", "_port", "shape")

    def __init__(
        self, owner: StatefulEntry, port: direct.Port, shape: tuple[int, ...]
    ):
        self._owner = owner
        self._port = port
        self.shape = shape

    @property
    def surface(self) -> direct.Surface:
        return self._port.surface

    def write(self, value: Any) -> None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            payload = value
        else:
            actual = tuple(int(item) for item in value.shape)
            if actual != self.shape:
                raise ValueError(
                    f"binding expects shape {self.shape}, got {actual}"
                )
            payload = memoryview(
                mx.contiguous(value.astype(self._owner.mx_dtype))
            ).cast("B")
        self._port.write(payload)

    def array(self) -> Any:
        return mx.array(self._port.read()).view(
            self._owner.mx_dtype
        ).reshape(self.shape)


class StatefulEntry:
    """One semantic entry backed by one explicit-lifecycle ANECIR model."""

    def __init__(
        self,
        owner: StatefulExecutable,
        contract: Any,
        shared_states: dict[str, direct.Surface],
    ):
        self._owner = owner
        self._contract = contract
        self.dtype = owner.dtype
        self.mx_dtype = owner.mx_dtype
        self._element = owner._element
        product = owner._cache / contract.product
        self._model = _ANECIRModel(product, contract.region, contract.name)
        raw = plistlib.load(product.open("rb"))
        network = raw[raw["Networks"][0]]

        self._model.load()
        try:
            status = self._model.attributes()["NetworkStatusList"][0]
            input_infos = list(status.get("LiveInputList", ())) + list(
                status.get("LiveStateList", ())
            )
            output_infos = list(status.get("LiveOutputList", ()))
        finally:
            self._model.unload()

        state_names = owner._state_names
        # Netplist ports follow the lowered ANE region ABI, not necessarily
        # the public function signature.  In particular, state arguments that
        # placement pruned and this compiler re-injected are appended after
        # pre-existing state arguments.  The compiler contract records that
        # realized order so raw requests share the same state surfaces as
        # MPSGraph's public entry point.
        input_names = list(contract.ane_input_order)
        input_symbols = list(network.get("Inputs", ())) + list(
            network.get("States", ())
        )
        named_inputs = _ordered_infos(
            label=contract.name,
            semantic_names=input_names,
            product_symbols=input_symbols,
            live_infos=input_infos,
            kind="input",
        )
        output_names = list(contract.ane_output_order)
        named_outputs = _ordered_infos(
            label=contract.name,
            semantic_names=output_names,
            product_symbols=list(network.get("Outputs", ())),
            live_infos=output_infos,
            kind="output",
        )
        self._input_info = dict(named_inputs)
        self._output_info = dict(named_outputs)

        self._feed_shapes = dict(contract.order)
        self._target_shapes = dict(contract.runtime_targets)
        dynamic = set(contract.dynamic) - state_names - contract.state_result_names
        unknown = dynamic - (set(self._feed_shapes) | set(self._target_shapes))
        if unknown:
            raise ValueError(
                f"{contract.name}: unknown dynamic tensors {sorted(unknown)}"
            )

        self._input_ports: list[direct.Port | None] = []
        self._input_indices: dict[str, int] = {}
        self._writes: dict[str, direct.Port] = {}
        for name, info in named_inputs:
            shape = self._feed_shapes[name]
            template = direct.Port(name, info, None)  # type: ignore[arg-type]
            if _shape_for_port(template) != shape:
                raise RuntimeError(
                    f"{contract.name}: input {name!r} live shape changed"
                )
            self._input_indices[name] = len(self._input_ports)
            if name in state_names:
                layout = (
                    template.logical_dims(),
                    template.strides(),
                    template.nbytes,
                )
                prior_layout = owner._state_layouts.setdefault(name, layout)
                if prior_layout != layout:
                    raise RuntimeError(
                        f"{contract.name}: shared state {name!r} physical "
                        "layout changed"
                    )
                surface = shared_states.get(name)
                if surface is None:
                    surface = direct.Surface(template.nbytes)
                    surface.zero()
                    shared_states[name] = surface
                elif surface.nbytes != template.nbytes:
                    raise RuntimeError(
                        f"{contract.name}: shared state {name!r} size changed"
                    )
                port = direct.Port(name, info, surface)
            elif name in dynamic:
                port = None
            else:
                port = direct.Port(name, info, direct.Surface(template.nbytes))
                self._writes[name] = port
            self._input_ports.append(port)

        self._output_ports: list[direct.Port | None] = []
        self._output_indices: dict[str, int] = {}
        self._reads: dict[str, direct.Port] = {}
        for name, info in named_outputs:
            shape = self._target_shapes[name]
            template = direct.Port(name, info, None)  # type: ignore[arg-type]
            if _shape_for_port(template) != shape:
                raise RuntimeError(
                    f"{contract.name}: output {name!r} live shape changed"
                )
            self._output_indices[name] = len(self._output_ports)
            if name in dynamic:
                port = None
            else:
                port = direct.Port(name, info, direct.Surface(template.nbytes))
                self._reads[name] = port
            self._output_ports.append(port)
        self._dynamic_feeds = set(self._input_indices) & dynamic
        self._dynamic_results = set(self._output_indices) & dynamic
        self._dispatch_nonce = 0

    @property
    def feed_names(self) -> tuple[str, ...]:
        return tuple(name for name, _shape in self._contract.order)

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(name for name, _shape in self._contract.runtime_targets)

    def bind(
        self, name: str, *, shared: TensorBinding | None = None
    ) -> TensorBinding:
        shape = self._feed_shapes.get(name, self._target_shapes.get(name))
        if shape is None:
            raise KeyError(f"unknown entry tensor {name!r}")
        if name not in self._dynamic_feeds and name not in self._dynamic_results:
            raise ValueError(f"entry tensor {name!r} is not dynamic")
        if name in self._input_indices:
            template_info = self._live_info(name, input_port=True)
        else:
            template_info = self._live_info(name, input_port=False)
        template = direct.Port(name, template_info, None)  # type: ignore[arg-type]
        if shared is None:
            surface = direct.Surface(template.nbytes)
        else:
            if not isinstance(shared, TensorBinding):
                raise TypeError("shared backing must be an ANECIR TensorBinding")
            if (
                shared.shape != shape
                or shared.surface.nbytes != template.nbytes
                or shared._port.logical_dims() != template.logical_dims()
                or shared._port.strides() != template.strides()
            ):
                raise ValueError(
                    f"shared binding for {name!r} is incompatible with {shape}"
                )
            surface = shared.surface
        return TensorBinding(
            self, direct.Port(name, template_info, surface), shape
        )

    def _live_info(self, name: str, *, input_port: bool) -> Mapping[str, Any]:
        table = self._input_info if input_port else self._output_info
        return table[name]

    def write_feeds(self, values: Mapping[str, Any]) -> None:
        for name, port in self._writes.items():
            value = values[name]
            actual = tuple(int(item) for item in value.shape)
            expected = self._feed_shapes[name]
            if actual != expected:
                raise ValueError(
                    f"feed {name!r} expects {expected}, got {actual}"
                )
            port.write(memoryview(
                mx.contiguous(value.astype(self.mx_dtype))
            ).cast("B"))

    def begin_dispatch(
        self, bindings: Mapping[str, TensorBinding] | None = None
    ):
        bindings = bindings or {}
        required = self._dynamic_feeds | self._dynamic_results
        missing = required - set(bindings)
        extra = set(bindings) - required
        if missing:
            raise ValueError(f"missing dynamic tensor bindings: {sorted(missing)}")
        if extra:
            raise ValueError(f"unexpected dynamic tensor bindings: {sorted(extra)}")
        inputs = list(self._input_ports)
        outputs = list(self._output_ports)
        for name in self._dynamic_feeds:
            binding = bindings[name]
            self._check_binding(name, binding, self._feed_shapes[name])
            inputs[self._input_indices[name]] = binding._port
        for name in self._dynamic_results:
            binding = bindings[name]
            self._check_binding(name, binding, self._target_shapes[name])
            outputs[self._output_indices[name]] = binding._port
        if any(port is None for port in inputs) or any(
            port is None for port in outputs
        ):
            raise AssertionError("ANECIR dispatch retained an unbound port")
        concrete_inputs = tuple(inputs)  # type: ignore[arg-type]
        concrete_outputs = tuple(outputs)  # type: ignore[arg-type]
        self._dispatch_nonce += 1
        nonce = self._dispatch_nonce

        def job() -> None:
            self._guarded_dispatch(concrete_inputs, concrete_outputs, nonce)

        return job

    def _guarded_dispatch(
        self,
        inputs: Sequence[direct.Port],
        outputs: Sequence[direct.Port],
        nonce: int,
    ) -> None:
        if not outputs:
            raise RuntimeError(
                f"{self._contract.name}: ANECIR entry has no liveness output"
            )
        # The observed raw-runtime failure is request-wide: evaluate returns
        # success without executing the activation, leaving every result
        # untouched. Synchronizing every IOSurface to prove that same fact is
        # expensive enough to erase the accelerator overlap. One deterministic
        # result therefore acts as the request liveness witness; choose the
        # smallest physical port to minimize its cache-coherency cost.
        port = min(outputs, key=lambda item: (item.nbytes, item.name))
        marker = struct.pack("<H", 0x7E01 + (nonce & 0xFF))
        offsets = port.probe_offsets()
        port.surface.lock()
        try:
            view = port.surface.view()
            for offset in offsets:
                view[offset:offset + 2] = marker
        finally:
            port.surface.unlock()
        self._owner._evaluate(self._model, inputs, outputs)
        port.surface.lock(readonly=True)
        try:
            view = port.surface.view()
            marked = [
                bytes(view[offset:offset + 2]) == marker
                for offset in offsets
            ]
        finally:
            port.surface.unlock(readonly=True)
        if any(marked):
            raise RuntimeError(
                f"{self._contract.name}: ANECIR request did not completely "
                f"write liveness output {port.name!r}"
            )

    def _check_binding(
        self, name: str, binding: TensorBinding, shape: tuple[int, ...]
    ) -> None:
        if not isinstance(binding, TensorBinding) or binding._owner is not self:
            raise ValueError(f"binding for {name!r} belongs to another entry")
        if binding.shape != shape:
            raise ValueError(
                f"binding for {name!r} expects {shape}, got {binding.shape}"
            )

    def read(self, wanted: set[str] | None = None) -> dict[str, Any]:
        return {
            name: mx.array(port.read()).view(self.mx_dtype).reshape(
                self._target_shapes[name]
            )
            for name, port in self._reads.items()
            if wanted is None or name in wanted
        }

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._model.close()


class StatefulExecutable:
    """Multiple explicit-lifecycle entries sharing persistent state surfaces."""

    def __init__(
        self,
        cache: str | Path,
        dtype: int,
        states: Sequence[Any],
        contracts: Sequence[Any],
    ):
        from . import mpsgraph as mg

        if dtype != mg.FLOAT16:
            raise ValueError("direct ANECIR state currently supports fp16 only")
        self._cache = Path(cache)
        self.dtype = dtype
        self.mx_dtype = mx.float16
        self._element = 2
        self._states = tuple(states)
        self._state_names = {state.name for state in states}
        self._state_layouts: dict[str, tuple[Any, ...]] = {}
        self._state_surfaces: dict[str, direct.Surface] = {}
        self._entries: dict[str, StatefulEntry] = {}
        self._active_model: _ANECIRModel | None = None
        self._closed = False
        try:
            for contract in contracts:
                self._entries[contract.name] = StatefulEntry(
                    self, contract, self._state_surfaces
                )
        except BaseException:
            self.close()
            raise
        if set(self._state_surfaces) != self._state_names:
            raise RuntimeError("ANECIR entries did not expose every state port")

    @property
    def state_specs(self) -> tuple[Any, ...]:
        return self._states

    def entry(self, name: str) -> StatefulEntry:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"unknown ANECIR state entry {name!r}") from exc

    def prepare(self) -> None:
        if self._closed:
            raise RuntimeError("ANECIR state executable is closed")

    def _evaluate(
        self,
        model: _ANECIRModel,
        inputs: Sequence[direct.Port],
        outputs: Sequence[direct.Port],
    ) -> None:
        """Switch only at a semantic entry boundary, then retain residency."""
        if self._closed:
            raise RuntimeError("ANECIR state executable is closed")
        if self._active_model is not model:
            prior = self._active_model
            self._active_model = None
            if prior is not None:
                prior.unload()
            model.load()
            self._active_model = model
        try:
            model.evaluate(inputs, outputs)
        except BaseException:
            # A failed evaluation cannot be reused safely. Preserve the
            # primary exception even if teardown also fails.
            import contextlib

            with contextlib.suppress(BaseException):
                model.unload()
            self._active_model = None
            raise

    def reset(self) -> None:
        for surface in self._state_surfaces.values():
            surface.zero()
        for entry in self._entries.values():
            entry.reset()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        active = self._active_model
        self._active_model = None
        try:
            if active is not None:
                active.unload()
        finally:
            for entry in self._entries.values():
                entry.close()
            self._entries.clear()
            self._state_layouts.clear()
            self._state_surfaces.clear()
            self._states = ()


__all__ = ["StatefulEntry", "StatefulExecutable", "TensorBinding"]
