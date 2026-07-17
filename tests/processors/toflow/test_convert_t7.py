"""The bespoke Torch7 reader is the gate for hostile .t7 bytes: a lying
tensor record (size/stride/offset exceeding its storage) must raise, never
read out of bounds."""

import io
import struct

import numpy as np
import pytest

from kinovsr.processors.toflow.convert_t7_to_safetensors import _T7Reader

pytestmark = pytest.mark.unit


def _tensor_bytes(ndim, size=(), stride=(), offset_raw=1):
    payload = struct.pack("i", ndim)
    payload += b"".join(struct.pack("q", s) for s in size)
    payload += b"".join(struct.pack("q", s) for s in stride)
    payload += struct.pack("q", offset_raw)
    return payload


def _reader(payload, storage):
    reader = _T7Reader(io.BytesIO(payload))
    reader.read_obj = lambda: storage
    return reader


def test_well_formed_record_reads_exactly():
    storage = np.arange(6, dtype=np.float32)
    reader = _reader(_tensor_bytes(2, (2, 3), (3, 1)), storage)
    out = reader._read_tensor(np.float32)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, storage.reshape(2, 3))


def test_oversized_stride_is_rejected():
    storage = np.zeros(16, dtype=np.float32)
    reader = _reader(_tensor_bytes(2, (4, 4), (1000, 1)), storage)
    with pytest.raises(ValueError, match="reaches element"):
        reader._read_tensor(np.float32)


def test_oversized_size_is_rejected():
    storage = np.zeros(16, dtype=np.float32)
    reader = _reader(_tensor_bytes(1, (10_000,), (1,)), storage)
    with pytest.raises(ValueError, match="reaches element"):
        reader._read_tensor(np.float32)


def test_negative_storage_offset_is_rejected():
    storage = np.zeros(16, dtype=np.float32)
    reader = _reader(_tensor_bytes(1, (4,), (1,), offset_raw=0), storage)
    with pytest.raises(ValueError, match="negative storage offset"):
        reader._read_tensor(np.float32)


def test_negative_stride_is_rejected():
    storage = np.zeros(16, dtype=np.float32)
    reader = _reader(_tensor_bytes(1, (4,), (-1,)), storage)
    with pytest.raises(ValueError, match="negative size/stride"):
        reader._read_tensor(np.float32)


def test_implausible_ndim_is_rejected_before_reading_arrays():
    reader = _reader(struct.pack("i", 200), np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError, match="implausible ndim"):
        reader._read_tensor(np.float32)


def test_zero_sized_dimension_yields_empty_tensor():
    storage = np.zeros(4, dtype=np.float32)
    reader = _reader(_tensor_bytes(2, (0, 3), (3, 1)), storage)
    out = reader._read_tensor(np.float32)
    assert out.shape == (0, 3)
    assert out.size == 0
