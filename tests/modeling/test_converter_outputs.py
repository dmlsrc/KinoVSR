"""Converter scripts refuse an output name MLX would rename.

MLX's save_safetensors appends .safetensors to any other name, so these
scripts wrote x.bin.safetensors for -o x.bin (SpyNet and the Torch7
converter then failed reloading x.bin). Each now refuses before reading its
source.
"""

import sys

import pytest

from kinovsr.eval.models.dover import convert_dover
from kinovsr.eval.models.musiq import convert_musiq
from kinovsr.eval.niqe import run_niqe
from kinovsr.modeling.spynet import convert_spynet
from kinovsr.processors.toflow import convert_t7_to_safetensors

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("module", [convert_dover, convert_musiq, convert_spynet])
def test_positional_converters_refuse_a_renamed_output(tmp_path, monkeypatch, module):
    source = tmp_path / "model.pth"
    source.write_bytes(b"never read")
    monkeypatch.setattr(sys, "argv", ["convert.py", str(source), str(tmp_path / "x.bin")])
    assert module.main() == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.pth"]


def test_torch7_converter_refuses_a_renamed_output(tmp_path, monkeypatch):
    source = tmp_path / "net.t7"
    source.write_bytes(b"never read")
    argv = ["convert_t7_to_safetensors.py", str(source), "-o", str(tmp_path / "x.bin")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        convert_t7_to_safetensors.main()
    assert exit_info.value.code == 2
    assert sorted(path.name for path in tmp_path.iterdir()) == ["net.t7"]


def test_niqe_fit_refuses_a_renamed_output(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        run_niqe(["--fit", str(tmp_path), "--out", str(tmp_path / "model.bin")])
    assert exit_info.value.code == 2
    assert list(tmp_path.iterdir()) == []
