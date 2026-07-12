"""Typed noise-map conditioning config shared across denoise families."""
from __future__ import annotations

import pytest

from kinovsr.processors.conditioning import (
    DeblockMapConfig,
    NoiseMapConfig,
    build_blockiness_tracker,
    build_conditioning,
    parse_deblock_map,
    parse_noise_map,
)

pytestmark = pytest.mark.unit


class TestParse:
    def test_defaults_to_constant_no_op(self):
        config = parse_noise_map({})
        assert config == NoiseMapConfig()
        assert config.mode == "constant"
        assert config.pulse is False

    def test_reads_all_flat_keys(self):
        config = parse_noise_map({
            "noise_map": "auto",
            "noise_map_gain": 1.5,
            "noise_map_refresh": 0,
            "noise_map_masking": 0.5,
            "noise_map_motion_cap": "loose",
            "noise_map_floor_mode": "flat",
            "noise_map_floor": 0.02,
            "noise_map_pulse": True,
        })
        assert config == NoiseMapConfig(
            mode="auto", gain=1.5, refresh=0, masking=0.5,
            motion_cap="loose", floor_mode="flat", floor=0.02, pulse=True)

    @pytest.mark.parametrize("raw,match", [
        ({"noise_map": "blur"}, "noise_map must"),
        ({"noise_map_gain": 0.0}, "gain must be > 0"),
        ({"noise_map_refresh": -1}, "refresh must be >= 0"),
        ({"noise_map_masking": 1.5}, r"masking must be in \[0, 1\]"),
        ({"noise_map_motion_cap": "hard"}, "motion_cap must"),
        ({"noise_map_floor_mode": "avg"}, "floor_mode must"),
        ({"noise_map_floor": 2.0}, r"floor must be in \[0, 1\]"),
    ])
    def test_open_time_validation(self, raw, match):
        with pytest.raises(ValueError, match=match):
            parse_noise_map(raw)


class TestBuild:
    def test_constant_no_pulse_builds_nothing(self):
        assert build_conditioning(NoiseMapConfig()) == (None, None)

    def test_auto_builds_tracker_with_the_configured_params(self):
        tracker, pulse = build_conditioning(NoiseMapConfig(
            mode="auto", gain=1.5, masking=0.5,
            motion_cap="loose", floor_mode="flat", pulse=False))
        assert pulse is None
        assert tracker is not None
        # gain lives on the tracker; the estimator kwargs pass through verbatim,
        # exactly as the harness constructed NoiseMapTracker.
        assert tracker.gain == 1.5
        assert tracker.est_kwargs == {
            "motion_cap": "loose", "masking": 0.5,
            "pulse_robust": False, "floor_mode": "flat"}

    def test_pulse_builds_a_pulse_gain_independent_of_mode(self):
        # pulse conditions in constant mode too (matching the flat CLI).
        tracker, pulse = build_conditioning(NoiseMapConfig(pulse=True))
        assert tracker is None
        assert pulse is not None
        assert hasattr(pulse, "update")

    def test_auto_with_pulse_builds_both_and_damps_the_map(self):
        tracker, pulse = build_conditioning(
            NoiseMapConfig(mode="auto", pulse=True))
        assert tracker is not None and pulse is not None
        # the tracker is told to damp whole-frame pulse spikes out of the base
        # map so pulse is not counted twice.
        assert tracker.est_kwargs["pulse_robust"] is True


class TestDeblockMap:
    def test_defaults_to_constant(self):
        assert parse_deblock_map({}) == DeblockMapConfig()
        assert parse_deblock_map({}).mode == "constant"

    def test_reads_the_keys(self):
        config = parse_deblock_map(
            {"deblock_map": "auto", "deblock_map_gain": 1.5})
        assert config == DeblockMapConfig(mode="auto", gain=1.5)

    @pytest.mark.parametrize("raw,match", [
        ({"deblock_map": "blur"}, "deblock_map must"),
        ({"deblock_map_gain": 0.0}, "gain must be > 0"),
    ])
    def test_open_time_validation(self, raw, match):
        with pytest.raises(ValueError, match=match):
            parse_deblock_map(raw)

    def test_constant_builds_no_tracker(self):
        assert build_blockiness_tracker(DeblockMapConfig()) is None

    def test_auto_builds_a_blockiness_tracker(self):
        from kinovsr.analysis.noise import estimate_blockiness_map

        tracker = build_blockiness_tracker(
            DeblockMapConfig(mode="auto", gain=1.5))
        assert tracker is not None
        assert tracker.gain == 1.5
        # a spatial estimate needs no temporal warm-up, and it estimates
        # blockiness (not sigma).
        assert tracker.min_frames == 1
        assert tracker.estimator is estimate_blockiness_map
