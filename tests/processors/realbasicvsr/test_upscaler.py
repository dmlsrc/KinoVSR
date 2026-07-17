"""RealBasicVSR upscaler validation tests."""

import pytest

from kinovsr.processors.realbasicvsr.upscaler import RealBasicVsrUpscaler


def test_rejects_unknown_flow_mode_before_loading_weights():
    with pytest.raises(ValueError, match="RealBasicVSR"):
        RealBasicVsrUpscaler(flow_mode="bogus")


def test_rejects_bad_history_controls_before_loading_weights():
    with pytest.raises(ValueError, match="history_gate"):
        RealBasicVsrUpscaler(history_gate="bogus")
    with pytest.raises(ValueError, match="history_strength"):
        RealBasicVsrUpscaler(history_strength=-0.1)


def test_vision_flow_mode_passes_the_driver_guard():
    # Regression: the driver-level flow guard lagged the factory token
    # list when the vision backend landed, killing runs at build time.
    # Validation passes and the failure is weight resolution, not flow.
    with pytest.raises(FileNotFoundError):
        RealBasicVsrUpscaler(flow_mode="vision",
                             weights="/nonexistent/weights.safetensors")
