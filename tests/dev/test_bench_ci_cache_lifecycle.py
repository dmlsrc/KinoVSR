"""Core Image lifecycle benchmark report math."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).parents[2] / "scripts" / "dev" / "bench_ci_cache_lifecycle.py"
SPEC = importlib.util.spec_from_file_location("kinovsr_bench_ci_cache_lifecycle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def test_expected_clear_count_includes_final_partial_interval():
    assert bench._expected_clears(0) == 0
    assert bench._expected_clears(63) == 1
    assert bench._expected_clears(64) == 1
    assert bench._expected_clears(65) == 2
    assert bench._expected_clears(128) == 2


def test_rss_summary_uses_only_requested_tail_and_reports_growth():
    samples = [
        {"renders": 64.0, "rss_mib": 100.0},
        {"renders": 128.0, "rss_mib": 120.0},
        {"renders": 192.0, "rss_mib": 121.0},
        {"renders": 256.0, "rss_mib": 122.0},
    ]

    summary = bench._rss_summary(samples, 3)

    assert summary["tail_samples"] == 3
    assert summary["tail_render_span"] == 128.0
    assert summary["tail_rss_first_mib"] == 120.0
    assert summary["tail_rss_last_mib"] == 122.0
    assert summary["tail_rss_span_mib"] == 2.0
    assert summary["tail_rss_growth_mib"] == 2.0
    assert summary["tail_rss_slope_mib_per_1000_renders"] == pytest.approx(15.625)


def _baseline(*, managed=False, cleanup_interval=None):
    environment = {
        "machine": "arm64",
        "hardware_model": "Mac14,6",
        "macos": "26.5.2",
        "python": "3.14.6",
        "mlx": "0.32.0",
    }
    protocol = {
        "fresh_process_required": True,
        "path": "host",
        "geometry": [320, 180],
        "warmup_iterations": 64,
        "measured_iterations": 256,
        "renders_per_iteration": 1,
        "sample_every_iterations": 64,
        "managed_lifecycle_available": managed,
        "cleanup_interval_renders": cleanup_interval,
    }
    return {
        "schema": 1,
        "environment": dict(environment),
        "protocol": protocol,
    }, dict(environment), {
        **protocol,
        "managed_lifecycle_available": True,
        "cleanup_interval_renders": 64,
    }


def test_baseline_accepts_only_intentional_unmanaged_cleanup_difference():
    baseline, environment, protocol = _baseline()

    bench._validate_baseline(
        baseline,
        protocol=protocol,
        environment=environment,
    )


@pytest.mark.parametrize(
    ("section", "key", "replacement", "message"),
    [
        ("environment", "macos", "99.0", "environment mismatch"),
        ("environment", "hardware_model", "Mac99,1", "environment mismatch"),
        ("protocol", "sample_every_iterations", 32, "sample_every_iterations"),
        ("protocol", "cleanup_interval_renders", 128, "cleanup policy mismatch"),
    ],
)
def test_baseline_rejects_environment_protocol_and_cleanup_drift(
    section,
    key,
    replacement,
    message,
):
    baseline, environment, protocol = _baseline(
        managed=True,
        cleanup_interval=64,
    )
    baseline[section][key] = replacement

    with pytest.raises(RuntimeError, match=message):
        bench._validate_baseline(
            baseline,
            protocol=protocol,
            environment=environment,
        )


def test_vm_walk_accepts_only_invalid_address_as_end_of_map():
    def task_self():
        return 1

    empty = bench._walk_vm_tags(
        lambda *_args: bench.KERN_INVALID_ADDRESS,
        task_self,
        page_size=4096,
    )

    assert all(values["regions"] == 0 for values in empty.values())
    with pytest.raises(RuntimeError, match="status=5"):
        bench._walk_vm_tags(
            lambda *_args: 5,
            task_self,
            page_size=4096,
        )


def test_protocol_rejects_tail_shorter_than_two_cleanup_intervals():
    assert not bench._protocol_sufficient(
        iterations=4096,
        sample_every=1,
        requested_tail_samples=4,
        renders_per_iteration=1,
        sample_count=4096,
        tail_render_span=3,
    )
    assert bench._protocol_sufficient(
        iterations=4096,
        sample_every=64,
        requested_tail_samples=4,
        renders_per_iteration=1,
        sample_count=64,
        tail_render_span=192,
    )
