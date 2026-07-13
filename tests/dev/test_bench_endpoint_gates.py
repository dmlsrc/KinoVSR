"""Post-M6 endpoint baseline-gate behavior."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).parents[2] / "scripts" / "dev" / "bench_endpoint_gates.py"
SPEC = importlib.util.spec_from_file_location("kinovsr_bench_endpoint_gates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def _baseline():
    return {
        "schema": bench.BASELINE_SCHEMA,
        "chip": "Apple M1 Max",
        "clip_sha256": "abc123",
        "clip_probe": [320, 180, 24.0, 48],
        "runs": 4,
        "frames": 36,
        "gates": {
            "pass": {
                "baseline_ms_per_frame": 2.0,
                "output_probe": [320, 180, 24.0, 36],
            },
        },
    }


def test_matching_baseline_is_accepted():
    bench._validate_baseline(
        _baseline(),
        chip="Apple M1 Max",
        clip_sha256="abc123",
        clip_probe=[320, 180, 24.0, 48],
        runs=4,
        frames=36,
        selected={"pass"},
    )


def test_baseline_mismatch_names_every_incompatible_field():
    with pytest.raises(ValueError) as exc:
        bench._validate_baseline(
            _baseline(),
            chip="Apple M2 Max",
            clip_sha256="different",
            clip_probe=[640, 360, 24.0, 48],
            runs=2,
            frames=40,
            selected={"pass", "learned"},
        )
    message = str(exc.value)
    for field in (
        "chip",
        "clip_sha256",
        "clip_probe",
        "runs",
        "frames",
    ):
        assert field in message
    assert "missing gates: ['learned']" in message


def test_gate_uses_existing_margin_and_requires_matching_output_probe():
    baseline = {
        "baseline_ms_per_frame": 100.0,
        "output_probe": [640, 360, 24.0, 36],
    }
    passing = bench._evaluate_gate(
        104.0,
        [103.0, 105.0],
        [640, 360, 24.0, 36],
        baseline,
        floor_ms=2.0,
        fraction=0.05,
    )
    assert passing["allowed_margin_ms"] == 5.0
    assert passing["pass"] is True

    too_slow = bench._evaluate_gate(
        105.1,
        [105.1],
        [640, 360, 24.0, 36],
        baseline,
        floor_ms=2.0,
        fraction=0.05,
    )
    assert too_slow["timing_pass"] is False
    assert too_slow["pass"] is False

    wrong_output = bench._evaluate_gate(
        99.0,
        [99.0],
        [640, 360, 30.0, 36],
        baseline,
        floor_ms=2.0,
        fraction=0.05,
    )
    assert wrong_output["timing_pass"] is True
    assert wrong_output["behavior_pass"] is False
    assert wrong_output["pass"] is False
