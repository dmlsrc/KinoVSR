"""Baseline provenance and compatibility validation for endpoint gates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from endpoint_gate_protocol import (
    _MISSING,
    BASELINE_KIND,
    BASELINE_SCHEMA,
    _measurement_errors,
    _signature_mismatches,
)


def _revision_errors(revision: Any, *, prefix: str) -> list[str]:
    if not isinstance(revision, Mapping):
        return [f"{prefix}: must be an object"]
    errors = []
    commit = revision.get("commit")
    dirty = revision.get("dirty")
    diff_sha256 = revision.get("diff_sha256", _MISSING)
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        errors.append(f"{prefix}.commit: must be a 40-64 digit lowercase hex hash")
    if not isinstance(dirty, bool):
        errors.append(f"{prefix}.dirty: must be a boolean")
    if dirty is True:
        if not isinstance(diff_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", diff_sha256) is None:
            errors.append(f"{prefix}.diff_sha256: dirty revisions require a SHA-256")
    elif dirty is False and diff_sha256 is not None:
        errors.append(f"{prefix}.diff_sha256: clean revisions require null")
    return errors


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
    mismatches.extend(_revision_errors(baseline.get("product_revision"), prefix="product_revision"))
    if baseline.get("schema") != BASELINE_SCHEMA:
        mismatches.append(
            f"schema: baseline={baseline.get('schema')!r}, current={BASELINE_SCHEMA!r}"
        )
    mismatches.extend(
        _signature_mismatches(
            baseline.get("fingerprint", _MISSING),
            fingerprint,
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
        required = (
            "workload",
            "measurement",
            "output_behavior",
            "output_behavior_run_count",
        )
        for field in required:
            if field not in entry:
                mismatches.append(f"gates.{chain}.{field}: missing baseline field")
        if "baseline_steady_ms_per_frame" in entry:
            mismatches.append(
                f"gates.{chain}.baseline_steady_ms_per_frame: duplicate timing "
                "source is forbidden; measurement.median is authoritative"
            )
        if "workload" in entry:
            mismatches.extend(
                _signature_mismatches(
                    entry["workload"],
                    workloads[chain],
                    prefix=f"gates.{chain}.workload",
                )
            )
        if "measurement" in entry:
            mismatches.extend(
                _measurement_errors(
                    entry["measurement"],
                    protocol=fingerprint["protocol"],
                    expected_conditions=fingerprint["power_thermal"],
                    prefix=f"gates.{chain}.measurement",
                )
            )
        expected_runs = fingerprint["protocol"]["runs"]
        if entry.get("output_behavior_run_count") != expected_runs:
            mismatches.append(
                f"gates.{chain}.output_behavior_run_count: expected "
                f"{expected_runs}, got {entry.get('output_behavior_run_count')!r}"
            )
    if mismatches:
        raise ValueError("baseline does not match this run: " + "; ".join(mismatches))
