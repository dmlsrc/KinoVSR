"""Baseline provenance and compatibility validation for endpoint gates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from endpoint_gate_protocol import (
    _MISSING,
    BASELINE_KIND,
    BASELINE_SCHEMA,
    _signature_mismatches,
)

_INFORMATIONAL_FINGERPRINT_PATHS = (
    ("protocol", "timing_boundary"),
    ("protocol", "steady_interval"),
    ("protocol", "instrumentation_host_device_sync"),
    ("protocol", "quality_policy", "method"),
    ("protocol", "quality_policy", "decoder"),
    ("protocol", "quality_policy", "compression"),
    ("power_thermal", "thermal_precondition"),
)


def _comparable_view(fingerprint: Any) -> Any:
    """Strip informational prose from the compared view.

    The prose is still recorded for readers; semantic changes are gated
    by the schema and the numeric protocol fields, so rewording a
    description must not force a baseline re-record.
    """
    if not isinstance(fingerprint, Mapping):
        return fingerprint
    import copy

    stripped = copy.deepcopy(dict(fingerprint))
    for path in _INFORMATIONAL_FINGERPRINT_PATHS:
        node = stripped
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, Mapping) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return stripped


def _validate_baseline(
    baseline: Any,
    *,
    fingerprint: dict[str, Any],
    workloads: Mapping[str, dict[str, Any]],
    selected: set[str],
) -> None:
    if not isinstance(baseline, Mapping):
        raise ValueError("baseline must be a JSON object")
    mismatches: list[str] = []
    if baseline.get("kind") != BASELINE_KIND:
        mismatches.append(f"kind: baseline={baseline.get('kind')!r}, current={BASELINE_KIND!r}")
    if baseline.get("schema") != BASELINE_SCHEMA:
        mismatches.append(
            f"schema: baseline={baseline.get('schema')!r}, current={BASELINE_SCHEMA!r}"
        )
    mismatches.extend(
        _signature_mismatches(
            _comparable_view(baseline.get("fingerprint", _MISSING)),
            _comparable_view(fingerprint),
            prefix="fingerprint",
        )
    )
    gates = baseline.get("gates")
    if not isinstance(gates, Mapping):
        mismatches.append("gates: baseline must contain an object")
        gates = {}
    for chain in sorted(selected):
        entry = gates.get(chain)
        if not isinstance(entry, Mapping):
            mismatches.append(f"gates.{chain}: missing baseline gate")
            continue
        required = ("workload", "measurement", "output_behavior")
        for field in required:
            if field not in entry:
                mismatches.append(f"gates.{chain}.{field}: missing baseline field")
        if "workload" in entry:
            mismatches.extend(
                _signature_mismatches(
                    entry["workload"],
                    workloads[chain],
                    prefix=f"gates.{chain}.workload",
                )
            )
        # The recorded file is this script's own output; deep re-validation
        # of its arithmetic defended same-account tampering, which the
        # charter's cooperative tier excludes. Shape-check the gated number.
        measurement = entry.get("measurement")
        if isinstance(measurement, Mapping):
            runs = measurement.get("runs")
            expected_runs = fingerprint["protocol"]["runs"]
            if not isinstance(runs, list) or len(runs) != expected_runs:
                mismatches.append(
                    f"gates.{chain}.measurement.runs: expected "
                    f"{expected_runs} runs")
            gated = (measurement.get("median") or {}).get(
                "steady_ms_per_frame")
            if (not isinstance(gated, (int, float))
                    or isinstance(gated, bool)
                    or not math.isfinite(gated) or gated <= 0):
                mismatches.append(
                    f"gates.{chain}.measurement.median.steady_ms_per_frame: "
                    f"must be a finite positive number, got {gated!r}")
    if mismatches:
        raise ValueError("baseline does not match this run: " + "; ".join(mismatches))
