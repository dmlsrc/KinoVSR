"""Restricted torch checkpoint parsing rejects hostile metadata and bounds."""

import io
import pickle
import sys
import types
import zipfile
from collections import OrderedDict, defaultdict
from typing import ClassVar

import mlx.core as mx
import numpy as np
import pytest
import torch

import kinovsr.modeling.torch_checkpoint as checkpoint_module
from kinovsr.modeling.pickle_scan import scan_pickle_globals, suspicious_globals
from kinovsr.modeling.torch_checkpoint import (
    CheckpointFormatError,
    load_restricted_checkpoint,
)


class _FakeStorage:
    def __init__(self, key: str, numel: int) -> None:
        self.key = key
        self.numel = numel


class _TensorSpec:
    def __init__(
        self,
        storage: _FakeStorage,
        *,
        offset: int = 0,
        shape: tuple[int, ...] = (1,),
        strides: tuple[int, ...] = (1,),
    ) -> None:
        self.storage = storage
        self.offset = offset
        self.shape = shape
        self.strides = strides

    def __reduce_ex__(self, protocol: int):
        rebuild = sys.modules["torch._utils"]._rebuild_tensor_v2
        return (
            rebuild,
            (
                self.storage,
                self.offset,
                self.shape,
                self.strides,
                False,
                None,
            ),
        )


class _StoragePickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _FakeStorage):
            storage_type = sys.modules["torch"].FloatStorage
            return ("storage", storage_type, obj.key, "cpu", obj.numel)
        return None


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    utils = types.ModuleType("torch._utils")

    def _rebuild_tensor_v2(*_args):
        raise AssertionError("fixture rebuild function must not execute")

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    torch_module = types.ModuleType("torch")

    class FloatStorage:
        pass

    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"
    torch_module.FloatStorage = FloatStorage
    torch_module._utils = utils
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch._utils", utils)


def _write_fake_zip(
    path,
    tree: object,
    storages: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> None:
    data = io.BytesIO()
    _StoragePickler(data, protocol=4).dump(tree)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", data.getvalue())
        for key, raw in storages.items():
            archive.writestr(
                zipfile.ZipInfo(f"archive/data/{key}"),
                raw,
                compress_type=compression,
            )


def test_protocol_four_stack_global_is_resolved_exactly() -> None:
    references = scan_pickle_globals(pickle.dumps(defaultdict, protocol=4))

    assert any(reference.endswith(" defaultdict") for reference in references)
    assert suspicious_globals(references)


def test_exact_allowlist_does_not_trust_an_entire_module() -> None:
    references = {
        "collections OrderedDict",
        "collections defaultdict",
        "torch._utils _rebuild_tensor_v2",
    }

    assert suspicious_globals(references) == ["collections defaultdict"]


def test_zip_checkpoint_round_trips_contiguous_and_strided_tensors(tmp_path) -> None:
    source = tmp_path / "valid.pth"
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save(
        {
            "contiguous": expected,
            "transposed": expected.transpose(0, 1),
        },
        source,
    )

    checkpoint = load_restricted_checkpoint(source)

    np.testing.assert_array_equal(
        np.array(checkpoint.tree["contiguous"]),
        expected.numpy(),
    )
    np.testing.assert_array_equal(
        np.array(checkpoint.tree["transposed"]),
        expected.transpose(0, 1).numpy(),
    )
    assert not checkpoint.flagged_globals


def test_legacy_checkpoint_round_trips_without_reading_the_storage_as_pickle(
    tmp_path,
) -> None:
    source = tmp_path / "legacy.pth"
    expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.save(
        {"weight": expected},
        source,
        _use_new_zipfile_serialization=False,
    )

    checkpoint = load_restricted_checkpoint(source)

    np.testing.assert_array_equal(np.array(checkpoint.tree["weight"]), expected.numpy())


def test_truncated_legacy_storage_is_refused(tmp_path) -> None:
    source = tmp_path / "legacy.pth"
    torch.save(
        {"weight": torch.ones(8)},
        source,
        _use_new_zipfile_serialization=False,
    )
    source.write_bytes(source.read_bytes()[:-1])

    with pytest.raises(CheckpointFormatError, match="was truncated"):
        load_restricted_checkpoint(source)


def test_storage_bytes_must_match_declared_metadata(tmp_path, monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    source = tmp_path / "truncated.pth"
    tensor = _TensorSpec(_FakeStorage("0", 2), shape=(2,), strides=(1,))
    _write_fake_zip(source, OrderedDict(weight=tensor), {"0": b"\0\0\0\0"})

    with pytest.raises(CheckpointFormatError, match="metadata declares 8"):
        load_restricted_checkpoint(source)


def test_compressed_storage_is_refused_before_decompression(
    tmp_path,
    monkeypatch,
) -> None:
    _install_fake_torch(monkeypatch)
    source = tmp_path / "compressed.pth"
    tensor = _TensorSpec(_FakeStorage("0", 1))
    _write_fake_zip(
        source,
        OrderedDict(weight=tensor),
        {"0": b"\0\0\0\0"},
        compression=zipfile.ZIP_DEFLATED,
    )

    with pytest.raises(CheckpointFormatError, match="must be uncompressed"):
        load_restricted_checkpoint(source)


def test_conflicting_storage_declarations_are_refused(tmp_path, monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    source = tmp_path / "conflicting.pth"
    tree = OrderedDict(
        first=_TensorSpec(_FakeStorage("0", 1)),
        second=_TensorSpec(_FakeStorage("0", 2), shape=(2,), strides=(1,)),
    )
    _write_fake_zip(source, tree, {"0": b"\0" * 8})

    with pytest.raises(CheckpointFormatError, match="conflicting declarations"):
        load_restricted_checkpoint(source)


def test_tensor_shape_cannot_request_an_allocation_bomb(tmp_path, monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    source = tmp_path / "shape-bomb.pth"
    tensor = _TensorSpec(
        _FakeStorage("0", 1),
        shape=(2**32, 2**32),
        strides=(2**32, 1),
    )
    _write_fake_zip(source, OrderedDict(weight=tensor), {"0": b"\0\0\0\0"})

    with pytest.raises(CheckpointFormatError, match="per-tensor limit"):
        load_restricted_checkpoint(source)


@pytest.mark.parametrize(
    ("offset", "shape", "strides"),
    [
        (2, (1,), (1,)),
        (0, (2,), (1,)),
        (0, (2,), (-1,)),
    ],
)
def test_tensor_views_must_stay_inside_declared_storage(
    tmp_path,
    monkeypatch,
    offset,
    shape,
    strides,
) -> None:
    _install_fake_torch(monkeypatch)
    source = tmp_path / "bad-view.pth"
    tensor = _TensorSpec(
        _FakeStorage("0", 1),
        offset=offset,
        shape=shape,
        strides=strides,
    )
    _write_fake_zip(source, OrderedDict(weight=tensor), {"0": b"\0\0\0\0"})

    with pytest.raises(CheckpointFormatError, match="outside storage"):
        load_restricted_checkpoint(source)


def test_non_allowlisted_global_is_refused_even_when_scan_override_is_set(
    tmp_path,
) -> None:
    source = tmp_path / "object.pth"
    torch.save({"object": defaultdict(list)}, source)

    with pytest.raises(CheckpointFormatError, match="outside.*allowlist"):
        load_restricted_checkpoint(source)
    with pytest.raises(CheckpointFormatError, match="outside.*allowlist"):
        load_restricted_checkpoint(source, allow_suspicious=True)


def test_recursive_unpickler_failure_is_reported_as_a_typed_refusal(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "recursive.pth"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({}))

    class RecursiveUnpickler:
        storage_meta: ClassVar[dict[str, object]] = {}

        def load(self):
            raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(
        checkpoint_module,
        "_make_unpickler",
        lambda *_args: RecursiveUnpickler(),
    )

    with pytest.raises(CheckpointFormatError, match="maximum recursion depth"):
        load_restricted_checkpoint(source)


def test_spynet_converter_uses_the_shared_restricted_reader(
    tmp_path,
    monkeypatch,
) -> None:
    from kinovsr.modeling.spynet import convert_spynet

    source = tmp_path / "spynet.pth"
    source.write_bytes(b"reader is mocked")
    output = tmp_path / "spynet.safetensors"
    calls = []

    def fake_load(path):
        calls.append(path)
        return type(
            "Checkpoint",
            (),
            {"tree": {"basic_module.0.weight": mx.ones((1, 1, 1, 1))}},
        )()

    # The converter holds the shared restricted reader itself.
    assert convert_spynet.load_restricted_checkpoint is checkpoint_module.load_restricted_checkpoint
    monkeypatch.setattr(convert_spynet, "load_restricted_checkpoint", fake_load)
    monkeypatch.setattr(sys, "argv", ["convert_spynet.py", str(source), str(output)])

    assert convert_spynet.main() == 0
    assert calls == [source]
    assert output.is_file()
