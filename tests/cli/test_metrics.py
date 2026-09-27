"""Metric-command validation."""

import pytest

pytestmark = pytest.mark.unit


def test_shimmer_rejects_mismatched_frame_rates(monkeypatch, capsys):
    from kinovsr.cli.commands import metrics

    probes = {
        "a.mp4": (640, 480, 24.0, 12),
        "b.mp4": (640, 480, 30.0, 12),
    }
    monkeypatch.setattr(metrics, "probe_dimensions", probes.__getitem__)

    with pytest.raises(SystemExit) as exc:
        metrics.run_shimmer(["a.mp4", "b.mp4"])
    assert exc.value.code == 2
    assert "frame rates differ" in capsys.readouterr().err
