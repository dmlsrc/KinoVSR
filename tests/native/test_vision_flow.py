"""Vision optical-flow request-policy contracts."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_unknown_accuracy_is_rejected_before_native_request():
    from kinovsr.native.vision_flow import generate_vision_flow

    with pytest.raises(ValueError, match="accuracy must be one of"):
        generate_vision_flow(object(), object(), accuracy="very-high")


def test_bidirectional_helper_keeps_both_requests_and_accuracy(monkeypatch):
    from kinovsr.native import vision_flow

    calls = []

    def generate(from_buffer, to_buffer, **kwargs):
        calls.append((from_buffer, to_buffer, kwargs))
        return f"{from_buffer}->{to_buffer}"

    monkeypatch.setattr(vision_flow, "generate_vision_flow", generate)

    assert vision_flow.generate_bidirectional_vision_flow(
        "previous",
        "current",
        accuracy="high",
    ) == (
        "previous->current",
        "current->previous",
    )
    assert calls == [
        ("previous", "current", {"accuracy": "high"}),
        ("current", "previous", {"accuracy": "high"}),
    ]
