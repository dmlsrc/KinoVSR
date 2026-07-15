"""Post-M6 steady-state endpoint baseline-gate contracts."""

import copy
import importlib.util
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).parents[2] / "scripts" / "dev" / "bench_endpoint_gates.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("kinovsr_bench_endpoint_gates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def _fingerprint():
    return {
        "machine": {
            "hardware_model": "Mac14,6",
            "chip": "Apple M1 Max",
            "architecture": "arm64",
        },
        "os": {
            "name": "macOS",
            "version": "26.5.1",
            "build": "25F90",
            "darwin_release": "25.5.0",
        },
        "runtime": {
            "python_implementation": "CPython",
            "python_version": "3.14.6",
            "python_build": ["main", "Jul 1 2026"],
            "python_compiler": "Clang 17",
            "mlx_version": "0.32.0",
            "pyav_version": "18.0.0",
            "libav_versions": {
                "libavcodec": [62, 28, 102],
                "libavformat": [62, 12, 102],
                "libavutil": [60, 26, 102],
                "libswscale": [9, 5, 102],
            },
        },
        "protocol": {
            "schema": 1,
            "fresh_process_per_run": True,
            "runs": 4,
            "warmup_frames": 30,
            "measured_frames": 120,
            "tail_frames": 17,
            "sink_holdback_frames": 1,
            "total_frames": 167,
            "timing_boundary": "after synchronous FileSink.append",
            "steady_interval": "after warmup output through measured output",
            "instrumentation_host_device_sync": (
                "none; CPU clock only, decoded quality probe runs after timing"
            ),
            "quality_policy": {
                "method": "decoded-rgb10-full-v4",
                "decoder": "PyAV/libav rgb48le to RGBAHalf",
                "compression": "zlib",
                "max_abs_rgb10": 2,
                "min_psnr_db": 60.0,
            },
        },
        "clip": {
            "sha256": "abc123",
            "track": {
                "width": 320,
                "height": 180,
                "nominal_fps": 24.0,
                "sample_count": 200,
                "cadence": "24/1",
                "first_pts": "0/1",
                "duration": "25/3",
                "source_tick": "1/24000",
                "codec_fourcc": "hvc1",
                "codec_details": {
                    "bits_per_component": 10,
                    "configuration_atom": "hvcC",
                    "configuration_sha256": "source-codec-hash",
                    "hevc": {
                        "profile_idc": 2,
                        "profile": "main10",
                        "tier": "main",
                        "level_idc": 30,
                        "chroma_format_idc": 1,
                        "chroma": "4:2:0",
                        "bit_depth_luma": 10,
                        "bit_depth_chroma": 10,
                    },
                },
                "pixel_aspect": None,
                "transform": {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "tx": 0.0, "ty": 0.0},
                "color": {"matrix": "ITU_R_709_2", "full_range": False},
            },
        },
        "cache_compile_policy": {
            "settings_source": "Settings constructor defaults",
            "mlx_cache_limit_gb": 1.0,
            "clear_mlx_cache_at_endpoint_start": True,
            "fresh_process_per_run": True,
            "model_compile": True,
            "system_compilation_cache": "retained",
        },
        "power_thermal": {
            "power_source": "AC Power",
            "low_power_mode": False,
            "thermal_state": "nominal",
            "thermal_policy": "nominal_only",
        },
    }


def _workload():
    return {
        "definition": {
            "schema": 1,
            "chain": "pass",
            "input_layout": "cv_rgba_half",
            "endpoint_args": {"quality": 0.65, "encode_chroma": "auto"},
        },
        "resolved": {
            "input_spec": {"layout": "cv_rgba_half"},
            "output_spec": {"layout": "cv_rgba_half"},
            "stages": [],
        },
        "measurement_contract": {
            "self_buffered_delay_frames": 0,
            "sink_holdback_frames": 1,
            "required_tail_frames": 1,
        },
    }


def _quality_sample(value=0, *, index=0, one_pixel=False):
    grid = np.zeros((2, 2, 3), dtype=np.uint16)
    if one_pixel:
        grid[0, 0, 0] = value
    else:
        grid.fill(value)
    return {
        "index": index,
        "shape": [2, 2, 3],
        "rgb10_zlib_b64": bench._encode_rgb10_frame(grid),
        "mean_rgb": [0.0, 0.0, 0.0],
        "std_rgb": [0.0, 0.0, 0.0],
    }


def _output_behavior(value=0, *, one_pixel=False, diagnostic_hash="quality-hash"):
    return {
        "track": {
            "width": 320,
            "height": 180,
            "sample_count": 167,
            "cadence": "24/1",
            "first_pts": "0/1",
            "duration": "167/24",
            "codec_fourcc": "hvc1",
            "codec_details": {
                "bits_per_component": 10,
                "configuration_atom": "hvcC",
                "configuration_sha256": "codec-hash",
                "hevc": {
                    "profile_idc": 4,
                    "profile": "range_extensions",
                    "tier": "main",
                    "level_idc": 30,
                    "chroma_format_idc": 2,
                    "chroma": "4:2:2",
                    "bit_depth_luma": 10,
                    "bit_depth_chroma": 10,
                },
            },
        },
        "quality": {
            "method": "decoded-rgb10-full-v4",
            "decoder": "PyAV/libav rgb48le to RGBAHalf",
            "indices": [0],
            "compression": "zlib",
            "diagnostic_full_frame_sha256": diagnostic_hash,
            "samples": [_quality_sample(value, one_pixel=one_pixel)],
        },
    }


def _output_runs(value=0, *, one_pixel=False, diagnostic_hash="quality-hash"):
    return [
        _output_behavior(
            value,
            one_pixel=one_pixel,
            diagnostic_hash=diagnostic_hash,
        )
        for _ in range(4)
    ]


def _measurement(steady_ms=100.0, setup_ms=5000.0):
    conditions = {
        "power_source": "AC Power",
        "low_power_mode": False,
        "thermal_state": "nominal",
    }
    run = {
        "setup_compile_ms": setup_ms,
        "warmup_ms": 6000.0,
        "steady_total_ms": steady_ms * 120,
        "steady_ms_per_frame": steady_ms,
        "tail_finalize_ms": 3000.0,
        "total_ms": setup_ms + 9000.0 + steady_ms * 120,
        "peak_rss_mib": 512.0,
        "peak_mlx_mib": 256.0,
        "measured_before_source_exhaustion_ms": 1.0,
        "frames_in": 167,
        "frames_written": 167,
        "conditions_start": conditions,
        "conditions_end": conditions,
    }
    return {
        "median": {
            key: value
            for key, value in run.items()
            if key not in {"frames_in", "frames_written", "conditions_start", "conditions_end"}
        },
        "runs": [{**run, "run": index} for index in range(1, 5)],
    }


def _baseline():
    workload = _workload()
    measurement = _measurement()
    return {
        "schema": bench.BASELINE_SCHEMA,
        "kind": bench.BASELINE_KIND,
        "product_revision": {
            "commit": "a" * 40,
            "dirty": False,
            "diff_sha256": None,
        },
        "fingerprint": _fingerprint(),
        "gates": {
            "pass": {
                "workload": workload,
                "measurement": measurement,
                "output_behavior": _output_behavior(),
                "output_behavior_run_count": 4,
            },
        },
    }


def _set_path(value, path, replacement):
    target = value
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement


@pytest.mark.parametrize(
    ("runs", "warmup", "measured", "tail", "field"),
    [
        (2, 30, 120, 17, "runs"),
        (3, 29, 120, 17, "warmup_frames"),
        (3, 30, 119, 17, "measured_frames"),
        (3, 30, 120, 16, "tail_frames"),
    ],
)
def test_protocol_minimums_are_enforced(runs, warmup, measured, tail, field):
    with pytest.raises(ValueError, match=field):
        bench._validate_protocol(runs, warmup, measured, tail)


def test_protocol_minimums_and_total_are_accepted():
    bench._validate_protocol(3, 30, 120, 17)
    assert bench._total_frames(30, 120, 17) == 167


def test_short_clip_and_truncated_worker_are_rejected():
    with pytest.raises(ValueError, match="clip.sample_count"):
        bench._require_clip_frames(166, 167)
    with pytest.raises(RuntimeError, match="frames_in=166, frames_out=166"):
        bench._assert_frame_counts("learned", 167, 166, 166)
    bench._assert_frame_counts("learned", 167, 167, 167)


def test_timing_excludes_setup_warmup_and_tail_from_steady_state():
    timing = bench._OutputTiming(warmup_frames=30, measured_frames=120)
    for count in range(1, 168):
        if count <= 30:
            timestamp = 5.0 + (count - 1) * (3.0 / 29.0)
        elif count <= 150:
            timestamp = 8.0 + (count - 30) * 0.25
        else:
            timestamp = 38.0 + (count - 150) * 0.1
        timing.record_append(timestamp)

    metrics = timing.metrics(start_s=0.0, end_s=42.0, expected_frames=167)
    assert metrics["setup_compile_ms"] == pytest.approx(5000.0)
    assert metrics["warmup_ms"] == pytest.approx(3000.0)
    assert metrics["steady_total_ms"] == pytest.approx(30000.0)
    assert metrics["steady_ms_per_frame"] == pytest.approx(250.0)
    assert metrics["tail_finalize_ms"] == pytest.approx(4000.0)
    assert metrics["total_ms"] == pytest.approx(42000.0)


def test_timing_requires_every_output_and_every_boundary():
    timing = bench._OutputTiming(warmup_frames=30, measured_frames=120)
    for count in range(1, 150):
        timing.record_append(float(count))
    with pytest.raises(RuntimeError, match="observed 149 frames, expected 167"):
        timing.metrics(start_s=0.0, end_s=170.0, expected_frames=167)


def test_measured_boundary_must_precede_natural_source_exhaustion():
    output = bench._OutputTiming(warmup_frames=30, measured_frames=120)
    output.measured_end_s = 167.0
    safe = bench._SourceTiming(count=167, exhausted_s=168.0)
    assert safe.measured_headroom_ms(output, expected_frames=167) == 1000.0

    flush_contaminated = bench._SourceTiming(count=166, exhausted_s=166.0)
    with pytest.raises(RuntimeError, match="source timing observed 166"):
        flush_contaminated.measured_headroom_ms(output, expected_frames=167)
    flush_contaminated.count = 167
    with pytest.raises(RuntimeError, match="after source exhaustion"):
        flush_contaminated.measured_headroom_ms(output, expected_frames=167)


def test_tail_requirement_tracks_resolved_temporal_delay_and_sink_holdback():
    passthrough = _workload()
    learned = copy.deepcopy(_workload())
    learned["measurement_contract"] = {
        "self_buffered_delay_frames": 16,
        "sink_holdback_frames": 1,
        "required_tail_frames": 17,
    }
    bench._validate_workload_tail(17, {"pass": passthrough, "learned": learned})
    with pytest.raises(ValueError, match=r"learned.tail_frames.*17"):
        bench._validate_workload_tail(16, {"learned": learned})

    learned["measurement_contract"]["self_buffered_delay_frames"] = 17
    learned["measurement_contract"]["required_tail_frames"] = 18
    with pytest.raises(ValueError, match=r"learned.tail_frames.*18"):
        bench._validate_workload_tail(17, {"learned": learned})


def test_learned_workload_is_explicit_and_quality_samples_span_all_intervals():
    definition = bench._gate_definition("learned")
    assert definition["config"]["den"] == {
        "processor": "bsvd",
        "profile": "c64",
        "strength": 0.5,
        "dtype": "float16",
    }
    assert definition["config"]["up"]["profile"] == "public2x"
    assert definition["config"]["up"]["dtype"] == "float16"
    assert definition["endpoint_args"]["encode_chroma"] == "auto"
    assert definition["output_encoder"]["profile"] == "HEVC_Main42210_AutoLevel"
    assert definition["output_encoder"]["chroma"] == "4:2:2"
    assert bench._quality_indices(
        warmup_frames=30,
        measured_frames=120,
        total_frames=167,
    ) == [0, 29, 30, 149, 166]


def test_run_summary_retains_every_sample_and_separate_memory_medians():
    runs = [
        {
            "setup_compile_ms": 1000.0,
            "warmup_ms": 2000.0,
            "steady_total_ms": 12000.0,
            "steady_ms_per_frame": 100.0,
            "tail_finalize_ms": 3000.0,
            "total_ms": 18000.0,
            "peak_rss_mib": 400.0,
            "peak_mlx_mib": 200.0,
            "measured_before_source_exhaustion_ms": 10.0,
            "conditions_start": {},
            "conditions_end": {},
            "path": "/temporary/one.mp4",
        },
        {
            "setup_compile_ms": 9000.0,
            "warmup_ms": 8000.0,
            "steady_total_ms": 14400.0,
            "steady_ms_per_frame": 120.0,
            "tail_finalize_ms": 7000.0,
            "total_ms": 38400.0,
            "peak_rss_mib": 600.0,
            "peak_mlx_mib": 300.0,
            "measured_before_source_exhaustion_ms": 30.0,
            "conditions_start": {},
            "conditions_end": {},
            "path": "/temporary/two.mp4",
        },
    ]
    summary = bench._summarize_runs(runs)
    assert summary["median"]["steady_ms_per_frame"] == 110.0
    assert summary["median"]["setup_compile_ms"] == 5000.0
    assert summary["median"]["peak_rss_mib"] == 500.0
    assert summary["median"]["peak_mlx_mib"] == 250.0
    assert summary["median"]["measured_before_source_exhaustion_ms"] == 20.0
    assert [run["run"] for run in summary["runs"]] == [1, 2]
    assert all("path" not in run for run in summary["runs"])


def test_runtime_condition_drift_is_rejected_with_exact_field():
    expected = {
        "power_source": "AC Power",
        "low_power_mode": False,
        "thermal_state": "nominal",
    }
    conditions = {
        "power_source": "AC Power",
        "low_power_mode": False,
        "thermal_state": "nominal",
    }
    runs = [
        {"conditions_start": conditions, "conditions_end": {**conditions, "low_power_mode": True}}
    ]
    with pytest.raises(RuntimeError, match=r"runs\[0\].conditions_end.low_power_mode"):
        bench._assert_run_conditions(runs, expected)


def test_non_nominal_thermal_state_cannot_be_recorded(monkeypatch):
    monkeypatch.setattr(
        bench._environment,
        "_runtime_conditions",
        lambda: {
            "power_source": "AC Power",
            "low_power_mode": False,
            "thermal_state": "serious",
            "thermal_precondition": "throttled",
        },
    )
    with pytest.raises(RuntimeError, match="nominal thermal state"):
        bench._comparison_conditions()


def test_matching_complete_baseline_is_accepted_and_revision_is_not_compared():
    baseline = _baseline()
    baseline["product_revision"] = {
        "commit": "b" * 40,
        "dirty": True,
        "diff_sha256": "c" * 64,
    }
    bench._validate_baseline(
        baseline,
        fingerprint=_fingerprint(),
        workloads={"pass": _workload()},
        selected={"pass"},
    )


@pytest.mark.parametrize(
    ("revision", "field"),
    [
        (None, "product_revision"),
        ({"commit": "short", "dirty": False, "diff_sha256": None}, "product_revision.commit"),
        ({"commit": "a" * 40, "dirty": "no", "diff_sha256": None}, "product_revision.dirty"),
        ({"commit": "a" * 40, "dirty": True, "diff_sha256": None}, "product_revision.diff_sha256"),
        (
            {"commit": "a" * 40, "dirty": False, "diff_sha256": "b" * 64},
            "product_revision.diff_sha256",
        ),
    ],
)
def test_missing_or_malformed_baseline_revision_is_rejected(revision, field):
    baseline = _baseline()
    baseline["product_revision"] = revision
    with pytest.raises(ValueError, match=field):
        bench._validate_baseline(
            baseline,
            fingerprint=_fingerprint(),
            workloads={"pass": _workload()},
            selected={"pass"},
        )


def test_baseline_kind_is_required():
    baseline = _baseline()
    baseline["kind"] = "not-a-kinovsr-endpoint-baseline"
    with pytest.raises(ValueError, match="kind"):
        bench._validate_baseline(
            baseline,
            fingerprint=_fingerprint(),
            workloads={"pass": _workload()},
            selected={"pass"},
        )


@pytest.mark.parametrize("baseline", [None, [], "baseline", 1])
def test_baseline_root_must_be_an_object(baseline):
    with pytest.raises(ValueError, match="JSON object"):
        bench._validate_baseline(
            baseline,
            fingerprint=_fingerprint(),
            workloads={"pass": _workload()},
            selected={"pass"},
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("machine.hardware_model", "Mac15,8"),
        ("machine.chip", "Apple M3 Max"),
        ("machine.architecture", "x86_64"),
        ("os.version", "27.0"),
        ("os.build", "26A1"),
        ("os.darwin_release", "26.0.0"),
        ("runtime.python_implementation", "PyPy"),
        ("runtime.python_version", "3.15.0"),
        ("runtime.python_build", ["other", "date"]),
        ("runtime.python_compiler", "Clang 18"),
        ("runtime.mlx_version", "0.33.0"),
        ("runtime.pyav_version", "19.0.0"),
        ("runtime.libav_versions.libswscale", [10, 0, 0]),
        ("protocol.schema", 2),
        ("protocol.fresh_process_per_run", False),
        ("protocol.runs", 3),
        ("protocol.warmup_frames", 31),
        ("protocol.measured_frames", 121),
        ("protocol.tail_frames", 18),
        ("protocol.sink_holdback_frames", 2),
        ("protocol.quality_policy.method", "different"),
        ("protocol.quality_policy.decoder", "native"),
        ("protocol.quality_policy.compression", "none"),
        ("protocol.quality_policy.max_abs_rgb10", 3),
        ("protocol.quality_policy.min_psnr_db", 50.0),
        ("protocol.total_frames", 170),
        ("protocol.timing_boundary", "different hook"),
        ("protocol.instrumentation_host_device_sync", "per-frame sync"),
        ("clip.sha256", "different"),
        ("clip.track.codec_fourcc", "avc1"),
        ("clip.track.codec_details.hevc.profile_idc", 4),
        ("clip.track.codec_details.hevc.chroma_format_idc", 2),
        ("clip.track.codec_details.hevc.bit_depth_luma", 8),
        ("clip.track.cadence", "24000/1001"),
        ("clip.track.first_pts", "1/24"),
        ("clip.track.duration", "50/3"),
        ("clip.track.sample_count", 201),
        ("clip.track.color", {"matrix": "ITU_R_2020"}),
        ("cache_compile_policy.mlx_cache_limit_gb", 0.5),
        ("cache_compile_policy.clear_mlx_cache_at_endpoint_start", False),
        ("cache_compile_policy.model_compile", False),
        ("cache_compile_policy.system_compilation_cache", "cleared"),
        ("power_thermal.power_source", "Battery Power"),
        ("power_thermal.low_power_mode", True),
        ("power_thermal.thermal_state", "fair"),
        ("power_thermal.thermal_policy", "anything"),
    ],
)
def test_every_environment_fingerprint_mutation_names_exact_field(path, replacement):
    current = _fingerprint()
    _set_path(current, path, replacement)
    with pytest.raises(ValueError) as exc:
        bench._validate_baseline(
            _baseline(),
            fingerprint=current,
            workloads={"pass": _workload()},
            selected={"pass"},
        )
    assert f"fingerprint.{path}" in str(exc.value)


def test_missing_and_extra_fingerprint_keys_are_incompatible():
    missing = _fingerprint()
    del missing["runtime"]["mlx_version"]
    extra = _fingerprint()
    extra["runtime"]["unexpected"] = "value"
    for current, field in (
        (missing, "fingerprint.runtime.mlx_version"),
        (extra, "fingerprint.runtime.unexpected"),
    ):
        with pytest.raises(ValueError) as exc:
            bench._validate_baseline(
                _baseline(),
                fingerprint=current,
                workloads={"pass": _workload()},
                selected={"pass"},
            )
        assert field in str(exc.value)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        ("definition.input_layout", "mlx_rgb_hwc"),
        ("definition.endpoint_args", {"quality": 0.4, "encode_chroma": "420"}),
        ("resolved.input_spec", {"layout": "mlx_rgb_hwc"}),
        ("resolved.output_spec", {"layout": "mlx_rgb_hwc"}),
        ("resolved.stages", [{"family": "bsvd", "profile": "c32"}]),
        ("measurement_contract.required_tail_frames", 18),
    ],
)
def test_workload_mutation_names_exact_field(path, replacement):
    workload = _workload()
    _set_path(workload, path, replacement)
    with pytest.raises(ValueError) as exc:
        bench._validate_baseline(
            _baseline(),
            fingerprint=_fingerprint(),
            workloads={"pass": workload},
            selected={"pass"},
        )
    assert f"gates.pass.workload.{path}" in str(exc.value)


@pytest.mark.parametrize(
    "field",
    ["workload", "measurement", "output_behavior"],
)
def test_missing_baseline_gate_fields_are_rejected(field):
    baseline = _baseline()
    del baseline["gates"]["pass"][field]
    with pytest.raises(ValueError, match=rf"gates.pass.{field}"):
        bench._validate_baseline(
            baseline,
            fingerprint=_fingerprint(),
            workloads={"pass": _workload()},
            selected={"pass"},
        )


def test_missing_selected_gate_and_wrong_schema_are_both_named():
    baseline = _baseline()
    baseline["schema"] = 2
    with pytest.raises(ValueError) as exc:
        bench._validate_baseline(
            baseline,
            fingerprint=_fingerprint(),
            workloads={"pass": _workload(), "learned": {"any": "value"}},
            selected={"pass", "learned"},
        )
    assert "schema" in str(exc.value)
    assert "gates.learned" in str(exc.value)


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (
            lambda measurement: measurement["median"].__setitem__(
                "steady_ms_per_frame", float("inf")
            ),
            "measurement.median.steady_ms_per_frame",
        ),
        (
            lambda measurement: measurement["median"].__setitem__("peak_rss_mib", float("nan")),
            "measurement.median.peak_rss_mib",
        ),
        (
            lambda measurement: measurement["median"].__setitem__(
                "steady_ms_per_frame", 1_000_000_000.0
            ),
            "measurement.median.steady_ms_per_frame",
        ),
        (lambda measurement: measurement["runs"].pop(), "measurement.runs.length"),
        (
            lambda measurement: measurement["runs"][0].__setitem__(
                "steady_ms_per_frame", float("inf")
            ),
            "measurement.runs[0].steady_ms_per_frame",
        ),
        (
            lambda measurement: measurement["runs"][0].__setitem__("peak_mlx_mib", -1.0),
            "measurement.runs[0].peak_mlx_mib",
        ),
        (
            lambda measurement: measurement["runs"][0].__setitem__("frames_written", 166),
            "measurement.runs[0].frames_written",
        ),
        (
            lambda measurement: measurement["runs"][0].__setitem__("total_ms", 1.0),
            "measurement.runs[0].total_ms",
        ),
        (
            lambda measurement: measurement["runs"][0]["conditions_end"].__setitem__(
                "thermal_state", "fair"
            ),
            "measurement.runs[0].conditions_end.thermal_state",
        ),
    ],
)
def test_baseline_measurement_must_be_finite_complete_and_internally_consistent(
    mutate,
    field,
):
    baseline = _baseline()
    mutate(baseline["gates"]["pass"]["measurement"])
    with pytest.raises(ValueError) as exc:
        bench._validate_baseline(
            baseline,
            fingerprint=_fingerprint(),
            workloads={"pass": _workload()},
            selected={"pass"},
        )
    assert field in str(exc.value)


def test_duplicate_baseline_timing_source_is_rejected_and_never_evaluated():
    baseline = _baseline()
    baseline["gates"]["pass"]["baseline_steady_ms_per_frame"] = float("inf")
    with pytest.raises(ValueError, match="duplicate timing source"):
        bench._validate_baseline(
            baseline,
            fingerprint=_fingerprint(),
            workloads={"pass": _workload()},
            selected={"pass"},
        )

    evaluated = bench._evaluate_gate(
        _measurement(steady_ms=1_000_000_000.0),
        _output_runs(),
        baseline["gates"]["pass"],
        fraction=0.05,
    )
    assert evaluated["baseline_steady_ms_per_frame"] == 100.0
    assert evaluated["timing_pass"] is False


def test_baseline_requires_one_output_probe_per_timing_run():
    baseline = _baseline()
    baseline["gates"]["pass"]["output_behavior_run_count"] = 3
    with pytest.raises(ValueError, match="output_behavior_run_count"):
        bench._validate_baseline(
            baseline,
            fingerprint=_fingerprint(),
            workloads={"pass": _workload()},
            selected={"pass"},
        )


def test_gate_uses_only_steady_median_and_requires_exact_output_behavior():
    baseline = _baseline()["gates"]["pass"]
    passing = bench._evaluate_gate(
        _measurement(steady_ms=104.0, setup_ms=500000.0),
        _output_runs(),
        baseline,
        fraction=0.05,
    )
    assert passing["allowed_margin_ms"] == 5.0
    assert passing["timing_pass"] is True
    assert passing["behavior_pass"] is True
    assert passing["pass"] is True

    within_tolerance = bench._evaluate_gate(
        _measurement(steady_ms=99.0),
        _output_runs(value=1, one_pixel=True, diagnostic_hash="different"),
        baseline,
        fraction=0.05,
    )
    assert within_tolerance["behavior_pass"] is True
    assert within_tolerance["quality_comparison"]["diagnostic_full_frame_hash_match"] is False
    assert within_tolerance["quality_comparison"]["samples"][0]["max_abs_rgb10"] == 1
    assert within_tolerance["pass"] is True

    too_slow = bench._evaluate_gate(
        _measurement(steady_ms=105.1, setup_ms=1.0),
        _output_runs(),
        baseline,
        fraction=0.05,
    )
    assert too_slow["timing_pass"] is False
    assert too_slow["pass"] is False

    wrong = _output_runs()
    for output in wrong:
        output["track"]["first_pts"] = "1/24"
    wrong_output = bench._evaluate_gate(
        _measurement(steady_ms=99.0),
        wrong,
        baseline,
        fraction=0.05,
    )
    assert wrong_output["timing_pass"] is True
    assert wrong_output["behavior_pass"] is False
    assert any(
        "output_behavior.track.first_pts" in item for item in wrong_output["behavior_mismatches"]
    )
    assert wrong_output["pass"] is False


def test_full_frame_quality_check_catches_single_pixel_corruption():
    baseline = _baseline()["gates"]["pass"]
    current = _output_runs(
        value=3,
        one_pixel=True,
        diagnostic_hash="localized-corruption",
    )
    evaluated = bench._evaluate_gate(
        _measurement(steady_ms=99.0),
        current,
        baseline,
        fraction=0.05,
    )
    assert evaluated["timing_pass"] is True
    assert evaluated["behavior_pass"] is False
    assert evaluated["quality_comparison"]["samples"][0]["max_abs_rgb10"] == 3
    assert any(
        "output_behavior.quality.samples[0]" in item for item in evaluated["behavior_mismatches"]
    )
    assert evaluated["pass"] is False


def test_every_timing_run_is_bound_to_its_own_output_behavior():
    baseline = _baseline()["gates"]["pass"]
    corrupted = _output_behavior(
        value=3,
        one_pixel=True,
        diagnostic_hash="fast-corruption",
    )
    output_runs = [copy.deepcopy(corrupted) for _ in range(3)]
    output_runs.append(_output_behavior())
    evaluated = bench._evaluate_gate(
        _measurement(steady_ms=1.0),
        output_runs,
        baseline,
        fraction=0.05,
    )
    assert evaluated["timing_pass"] is True
    assert [run["pass"] for run in evaluated["behavior_runs"]] == [
        False,
        False,
        False,
        True,
    ]
    assert evaluated["behavior_pass"] is False
    assert evaluated["pass"] is False


def test_margin_floor_is_derived_from_baseline_noise_not_a_constant():
    # The old fixed 2.0 ms/frame floor predated the steady-state protocol,
    # exceeded the entire steady passthrough baseline, and could bless a ~2x
    # plumbing regression on a fast chain.  The floor is now the baseline's
    # own recorded 3-sigma run spread.
    fast = _baseline()["gates"]["pass"]
    fast["measurement"]["median"]["steady_ms_per_frame"] = 1.6
    for run in fast["measurement"]["runs"]:
        run["steady_ms_per_frame"] = 1.6
    doubled = bench._evaluate_gate(
        _measurement(steady_ms=3.1),
        _output_runs(),
        fast,
        fraction=0.10,
    )
    assert doubled["allowed_margin_ms"] == pytest.approx(0.16)
    assert doubled["timing_pass"] is False

    noisy = _baseline()["gates"]["pass"]
    spread = [96.0, 100.0, 104.0, 100.0]
    for run, value in zip(noisy["measurement"]["runs"], spread, strict=True):
        run["steady_ms_per_frame"] = value
    expected = max(3.0 * statistics.stdev(spread), 0.05 * 100.0)
    noisy_eval = bench._evaluate_gate(
        _measurement(steady_ms=100.0 + expected - 0.01),
        _output_runs(),
        noisy,
        fraction=0.05,
    )
    assert noisy_eval["allowed_margin_ms"] == pytest.approx(round(expected, 3))
    assert noisy_eval["timing_pass"] is True


def test_recording_rejects_inconsistent_per_run_output_behavior():
    outputs = _output_runs()
    outputs[0] = _output_behavior(
        value=3,
        one_pixel=True,
        diagnostic_hash="inconsistent",
    )
    mismatches, runs = bench._compare_behavior_runs(outputs[-1], outputs)
    assert [run["pass"] for run in runs] == [False, True, True, True]
    assert any("runs[0].output_behavior.quality.samples[0]" in item for item in mismatches)


def test_report_records_both_revisions_without_fingerprinting_them(tmp_path):
    baseline = _baseline()
    current = {"commit": "d" * 40, "dirty": True, "diff_sha256": "e" * 64}
    report = bench._report_header(
        recording=False,
        clip=tmp_path / "clip.mp4",
        baseline_path=tmp_path / "baseline.json",
        current_revision=current,
        baseline=baseline,
        fingerprint=_fingerprint(),
        workloads={"pass": _workload()},
    )
    assert report["current_product_revision"] == current
    assert report["baseline_product_revision"] == baseline["product_revision"]
    assert "product_revision" not in report["fingerprint"]


def test_baseline_recording_can_never_return_a_passing_gate_status():
    assert bench._result_exit_code(recording=True, failures=[]) == bench.RECORD_ONLY_EXIT
    assert bench.RECORD_ONLY_EXIT != 0
    assert bench._result_exit_code(recording=False, failures=[]) == 0
    assert bench._result_exit_code(recording=False, failures=["pass"]) == 1
