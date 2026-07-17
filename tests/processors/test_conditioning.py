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
            "noise_map_upsample": "box",
        })
        assert config == NoiseMapConfig(
            mode="auto", gain=1.5, refresh=0, masking=0.5,
            motion_cap="loose", floor_mode="flat", floor=0.02, pulse=True,
            upsample="box")

    @pytest.mark.parametrize("raw,match", [
        ({"noise_map": "blur"}, "noise_map must"),
        ({"noise_map_gain": 0.0}, "gain must be > 0"),
        ({"noise_map_refresh": -1}, "refresh must be >= 0"),
        ({"noise_map_masking": 1.5}, r"masking must be in \[0, 1\]"),
        ({"noise_map_motion_cap": "hard"}, "motion_cap must"),
        ({"noise_map_floor_mode": "avg"}, "floor_mode must"),
        ({"noise_map_floor": 2.0}, r"floor must be in \[0, 1\]"),
        ({"noise_map_upsample": "bilinear"}, "upsample must"),
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
            "pulse_robust": False, "floor_mode": "flat",
            "upsample": "edge"}

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


class TestDiagnosticsHelpers:
    """The shared end-of-run report helpers (harness line formats)."""

    def _driver(self, **attrs):
        from types import SimpleNamespace

        return SimpleNamespace(**attrs)

    def test_noise_map_lines(self):
        import mlx.core as mx

        from kinovsr.processors.conditioning import noise_map_diagnostics

        drv = self._driver(
            last_noise_map=mx.full((4, 4, 1), 0.08, dtype=mx.float32),
            SIGMA_MIN=5.0 / 255.0, SIGMA_MAX=55.0 / 255.0, _map_floor=0.0,
            _pulse=object(), _pulse_log=[1.0, 1.1, 1.5])
        lines = noise_map_diagnostics(drv)
        assert lines[0].startswith("[noise-map] estimated sigma: min 0.0800")
        assert "(floor 0.0196, ceil 0.2157)" in lines[1]
        assert "pulse gain over 3 frames" in lines[2]
        assert "(1 frames > 1.2)" in lines[2]

    def test_no_conditioning_reports_nothing(self):
        from kinovsr.processors.conditioning import noise_map_diagnostics

        assert noise_map_diagnostics(self._driver(last_noise_map=None,
                                                  _pulse=None)) == []

    def test_blockiness_lines_and_image(self):
        import mlx.core as mx

        from kinovsr.processors.conditioning import (
            blockiness_debug_image,
            blockiness_diagnostics,
        )

        bm = mx.broadcast_to(
            mx.array([0.2, 0.9])[:, None, None], (2, 4, 1)).astype(mx.float32)
        drv = self._driver(last_blockiness_map=bm)
        (line,) = blockiness_diagnostics(drv)
        assert line.startswith("[deblock-map] blockiness mask: median")
        assert "(50% of frame > 0.5)" in line
        image = blockiness_debug_image(drv)["blockmap"]
        assert image.shape == (2, 4)

    def test_noise_map_image_normalizes_by_015(self):
        import mlx.core as mx

        from kinovsr.processors.conditioning import noise_map_debug_image

        drv = self._driver(
            last_noise_map=mx.full((2, 2, 1), 0.15, dtype=mx.float32))
        image = noise_map_debug_image(drv)["noisemap"]
        assert float(image.max()) == 1.0


def test_tracker_receives_the_upsample_choice():
    from kinovsr.processors.conditioning import (
        NoiseMapConfig,
        build_conditioning,
    )

    tracker, _ = build_conditioning(NoiseMapConfig(mode="auto",
                                                   upsample="box"))
    assert tracker.est_kwargs["upsample"] == "box"
    tracker, _ = build_conditioning(NoiseMapConfig(mode="auto"))
    assert tracker.est_kwargs["upsample"] == "edge"
