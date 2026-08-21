"""MLX 0.32.1 owns eligible ungrouped convolution input padding."""

from pathlib import Path

import mlx.core as mx

from kinovsr.processors.basicvsrpp import net as basicvsrpp
from kinovsr.processors.realbasicvsr import net as realbasicvsr
from kinovsr.processors.realviformer import net as realviformer
from kinovsr.processors.stdf import net as stdf


def test_basicvsrpp_loader_keeps_automatic_input_pad_widths(monkeypatch):
    weights = {
        "spynet.basic_module.0.basic_module.0.conv.weight": mx.zeros((32, 8, 7, 7)),
        "deform_align.backward_1.conv_offset.0.weight": mx.zeros((64, 196, 3, 3)),
    }
    monkeypatch.setattr(basicvsrpp.mx, "load", lambda _path: weights)

    params = basicvsrpp.load_params("unused.safetensors", dtype=mx.float32)

    assert params["spynet.basic_module.0.basic_module.0.conv.weight"].shape == (32, 7, 7, 8)
    assert params["deform_align.backward_1.conv_offset.0.weight"].shape == (64, 3, 3, 196)


def test_realbasicvsr_loader_keeps_automatic_input_pad_widths(monkeypatch):
    weights = {
        "image_cleaning.0.main.0.weight": mx.zeros((64, 3, 3, 3)),
        "basicvsr.conv_last.weight": mx.zeros((3, 64, 3, 3)),
        "basicvsr.spynet.mean": mx.zeros((3,)),
        "basicvsr.spynet.basic_module.0.basic_module.0.conv.weight": mx.zeros((32, 8, 7, 7)),
        "basicvsr.backward_resblocks.main.0.weight": mx.zeros((64, 67, 3, 3)),
        "basicvsr.forward_resblocks.main.0.weight": mx.zeros((64, 67, 3, 3)),
    }
    monkeypatch.setattr(realbasicvsr, "_load_safetensors", lambda _path: weights)

    params = realbasicvsr.load_params(Path("unused.safetensors"), dtype=mx.float32)

    assert params["spynet.basic_module.0.basic_module.0.conv.weight"].shape == (32, 7, 7, 8)
    assert params["basicvsr.backward_resblocks.main.0.weight"].shape == (64, 3, 3, 67)
    assert params["basicvsr.forward_resblocks.main.0.weight"].shape == (64, 3, 3, 67)


def test_stdf_loader_keeps_input_pad_native_but_retains_output_pad(monkeypatch):
    weights = {
        "ffnet.in_conv.0.weight": mx.zeros((32, 7, 3, 3)),
        "ffnet.offset_mask.weight": mx.zeros((189, 32, 3, 3)),
        "ffnet.offset_mask.bias": mx.zeros((189,)),
    }
    monkeypatch.setattr(stdf, "resolve_weights", lambda _path: Path("unused.safetensors"))
    monkeypatch.setattr(stdf.mx, "load", lambda _path: weights)

    params = stdf.load_params("unused.safetensors", dtype=mx.float32)

    assert params["ffnet.in_conv.0.weight"].shape == (32, 3, 3, 7)
    assert "ffnet.in_conv.0.weight_gp" not in params
    assert params["ffnet.offset_mask.weight_gp"].shape == (192, 3, 3, 32)
    assert params["ffnet.offset_mask.bias_gp"].shape == (192,)


def test_realviformer_loader_keeps_spynet_input_width(monkeypatch):
    weights = {
        "shallow_extraction.0.weight": mx.zeros((48, 3, 3, 3)),
        "spynet.basic_module.0.basic_module.0.weight": mx.zeros((32, 8, 7, 7)),
    }
    monkeypatch.setattr(realviformer, "resolve_weights", lambda _path: Path("unused.safetensors"))
    monkeypatch.setattr(realviformer.mx, "load", lambda _path: weights)

    params = realviformer.load_params("unused.safetensors", dtype=mx.float32)

    assert params["spynet.basic_module.0.basic_module.0.conv.weight"].shape == (32, 7, 7, 8)
