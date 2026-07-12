"""Parser behavior: canonical flags, hidden aliases, exit codes, assembly."""

import argparse
from pathlib import Path

import pytest

from kinovsr.cli.args import build_parser, validate_args
from kinovsr.cli.config import assemble, normalize_chain
from kinovsr.config import ConfigError
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def parser() -> argparse.ArgumentParser:
    return build_parser()


BASE = ["--video", "clip.mp4", "--output-dir", "out"]


class TestAliases:
    def test_canonical_and_alias_hit_the_same_dest(self, parser):
        canon = parser.parse_args(
            [*BASE, "--upscale", "safmn", "--basicvsrpp-flow", "vt",
             "--fastdvdnet-profile", "standard",
             "--realesrgan-denoise-strength", "0.3",
             "--realbasicvsr-clean-threshold", "12"])
        legacy = parser.parse_args(
            [*BASE, "--spatial-mode", "safmn", "--basicvsrpp-flow-mode", "vt",
             "--fastdvd-variant", "standard",
             "--realesrgan-denoise", "0.3",
             "--realbasicvsr-dynamic-refine-thres", "12"])
        for dest in ("upscale", "basicvsrpp_flow", "fastdvdnet_profile",
                     "realesrgan_denoise_strength",
                     "realbasicvsr_clean_threshold"):
            assert getattr(canon, dest) == getattr(legacy, dest)

    def test_alias_absent_leaves_canonical_default(self, parser):
        args = parser.parse_args(BASE)
        assert args.upscale == "balanced"
        assert args.fastdvdnet_profile == "clipped"

    def test_help_shows_canonical_names_only(self, parser):
        text = parser.format_help()
        for hidden in ("--spatial-mode", "--basicvsrpp-flow-mode",
                       "--fastdvd-variant", "--fastdvd-weights",
                       "--realbasicvsr-dynamic-refine-thres",
                       "--deblock-weights"):
            assert hidden not in text, hidden
        for canonical in ("--upscale", "--fastdvdnet-profile",
                          "--stdf-weights", "--fbcnn-weights",
                          "--realesrgan-denoise-strength", "--base-config",
                          "--verbose", "--mc-flow "):
            assert canonical in text, canonical


class TestExitCodes:
    def test_help_exits_zero(self, parser, capsys):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0
        assert "--upscale" in capsys.readouterr().out

    def test_missing_video_exits_two(self, parser, capsys):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--output-dir", "out"])
        assert exc.value.code == 2
        capsys.readouterr()

    def test_bad_choice_exits_two(self, parser, capsys):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([*BASE, "--upscale", "enhance"])
        assert exc.value.code == 2
        capsys.readouterr()

    def test_output_dir_required_without_probe(self, parser, capsys):
        args = parser.parse_args(["--video", "clip.mp4"])
        with pytest.raises(SystemExit) as exc:
            validate_args(parser, args)
        assert exc.value.code == 2
        capsys.readouterr()

    def test_probe_noise_waives_output_dir(self, parser):
        args = parser.parse_args(["--video", "clip.mp4", "--probe-noise"])
        validate_args(parser, args)

    def test_realbasicvsr_window_bounds(self, parser, capsys):
        args = parser.parse_args(
            [*BASE, "--upscale", "realbasicvsr", "--realbasicvsr-window", "0"])
        with pytest.raises(SystemExit) as exc:
            validate_args(parser, args)
        assert exc.value.code == 2
        capsys.readouterr()


class TestChains:
    def test_family_alias_normalizes(self):
        assert normalize_chain("fastdvd,mc") == "fastdvdnet,mc"
        assert normalize_chain("off") == "off"
        assert normalize_chain("mc, bsvd") == "mc,bsvd"

    def test_assemble_normalizes_denoise_chain(self, parser):
        args = parser.parse_args([*BASE, "--denoise", "fastdvd,bsvd"])
        inv = assemble(args, base=Settings())
        assert inv.options.denoise == "fastdvdnet,bsvd"


class TestSettingsTrifecta:
    def test_toml_settings_then_cli_wins(self, parser, tmp_path: Path):
        toml = tmp_path / "s.toml"
        toml.write_text("[settings]\nquiet = true\nmlx_cache_limit_gb = 2.5\n")
        args = parser.parse_args(
            [*BASE, "--config", str(toml), "--mlx-cache-limit-gb", "0.5"])
        inv = assemble(args, base=Settings())
        assert inv.settings.quiet is True
        assert inv.settings.mlx_cache_limit_gb == 0.5

    def test_base_configs_apply_in_order(self, parser, tmp_path: Path):
        a = tmp_path / "a.toml"
        b = tmp_path / "b.toml"
        a.write_text("[settings]\nmlx_cache_limit_gb = 1.5\nquiet = true\n")
        b.write_text("[settings]\nmlx_cache_limit_gb = 3.0\n")
        args = parser.parse_args(
            [*BASE, "--base-config", str(a), "--base-config", str(b)])
        inv = assemble(args, base=Settings())
        assert inv.settings.mlx_cache_limit_gb == 3.0
        assert inv.settings.quiet is True

    def test_family_weights_flag_lands_in_settings(self, parser):
        args = parser.parse_args([*BASE, "--basicvsrpp-weights", "/w/b.st"])
        inv = assemble(args, base=Settings())
        assert inv.settings.basicvsrpp_weights == "/w/b.st"

    def test_unknown_settings_key_is_an_error(self, parser, tmp_path: Path):
        toml = tmp_path / "s.toml"
        toml.write_text("[settings]\nvebrose = true\n")
        args = parser.parse_args([*BASE, "--config", str(toml)])
        with pytest.raises(ConfigError, match="unknown key"):
            assemble(args, base=Settings())

    def test_wrong_type_is_an_error(self, parser, tmp_path: Path):
        toml = tmp_path / "s.toml"
        toml.write_text('[settings]\nquiet = "yes"\n')
        args = parser.parse_args([*BASE, "--config", str(toml)])
        with pytest.raises(ConfigError, match="boolean"):
            assemble(args, base=Settings())

    def test_stage_tables_ride_the_invocation(self, parser, tmp_path: Path):
        toml = tmp_path / "p.toml"
        toml.write_text('pipeline = ["denoise"]\n[denoise]\nprocessor = "bsvd"\n')
        args = parser.parse_args([*BASE, "--config", str(toml)])
        inv = assemble(args, base=Settings())
        assert inv.config["pipeline"] == ["denoise"]
        assert inv.config["denoise"]["processor"] == "bsvd"


class TestDeblockWeightsCompat:
    def test_fills_chained_families_without_family_value(self, parser):
        args = parser.parse_args(
            [*BASE, "--deblock", "stdf,fbcnn",
             "--deblock-weights", "/w/legacy.st"])
        inv = assemble(args, base=Settings())
        assert inv.settings.stdf_weights == "/w/legacy.st"
        assert inv.settings.fbcnn_weights == "/w/legacy.st"

    def test_family_flag_wins_over_legacy_fill(self, parser):
        args = parser.parse_args(
            [*BASE, "--deblock", "stdf,fbcnn",
             "--deblock-weights", "/w/legacy.st",
             "--fbcnn-weights", "/w/fb.st"])
        inv = assemble(args, base=Settings())
        assert inv.settings.stdf_weights == "/w/legacy.st"
        assert inv.settings.fbcnn_weights == "/w/fb.st"

    def test_no_fill_outside_the_chain(self, parser):
        args = parser.parse_args(
            [*BASE, "--deblock", "stdf", "--deblock-weights", "/w/legacy.st"])
        inv = assemble(args, base=Settings())
        assert inv.settings.stdf_weights == "/w/legacy.st"
        assert inv.settings.fbcnn_weights is None


class TestTypedSourceLayout:
    """The typed route decodes into a layout the chain's head accepts."""

    def _pick(self, config):
        from kinovsr.cli.commands.run import _source_layout

        return _source_layout(config)

    def test_native_head_gets_a_cv_layout(self):
        from kinovsr.processors import Layout

        config = {"pipeline": ["fps"],
                  "fps": {"processor": "videotoolbox", "profile": "normal",
                          "target_fps": 50}}
        assert self._pick(config) is Layout.CV_RGBA_HALF

    def test_same_family_different_capability_different_layout(self):
        # videotoolbox interpolate is CV-in, but its upscale is the MLX->CV
        # bridge - so the head's SPECIFIC capability decides the layout.
        from kinovsr.processors import Layout

        config = {"pipeline": ["up"],
                  "up": {"processor": "videotoolbox", "capability": "upscale",
                         "profile": "balanced"}}
        assert self._pick(config) is Layout.MLX_RGB_HWC

    def test_mlx_head_keeps_the_default(self):
        from kinovsr.processors import Layout

        config = {"pipeline": ["up"],
                  "up": {"processor": "metalfx", "scale": 2}}
        assert self._pick(config) is Layout.MLX_RGB_HWC

    def test_empty_and_unknown_fall_back(self):
        from kinovsr.processors import Layout

        assert self._pick({"pipeline": []}) is Layout.MLX_RGB_HWC
        assert self._pick({"pipeline": ["x"],
                           "x": {"processor": "nosuch"}}) is Layout.MLX_RGB_HWC


class TestPipelineOwnedFlags:
    """A [pipeline] config owns the whole chain; stage AND geometry flags
    alongside it are a config error, not silently ignored (parity trap C8)."""

    def _flags(self, **overrides):
        from types import SimpleNamespace

        from kinovsr.cli.commands.run import _pipeline_owned_flags

        base = {
            "upscale": "balanced", "denoise": "off", "deblock": "off",
            "restore": "off", "nafnet": "off", "deflicker": "off",
            "cut_detect": "off", "crop_bars": None, "crop_aspect": None,
            "square_pixels": False, "sanitize_edges": None, "snap_start": False,
            "gop_align": False}
        base.update(overrides)
        return _pipeline_owned_flags(SimpleNamespace(**base))

    def test_none_set_is_clean(self):
        assert self._flags() == []

    def test_stage_selector_is_flagged(self):
        assert "denoise" in self._flags(denoise="mc,bsvd")

    def test_geometry_flags_are_flagged(self):
        # C8: crop/anamorphic/sanitize flags were silently ignored alongside
        # a [pipeline] config before this.
        assert "crop_bars" in self._flags(crop_bars="auto")
        assert "crop_aspect" in self._flags(crop_aspect="16:9")
        assert "square_pixels" in self._flags(square_pixels=True)
        assert "sanitize_edges" in self._flags(sanitize_edges="2,0,0,0")

    def test_keyframe_windowing_is_run_level_not_rejected(self):
        # --snap-start / --gop-align are run-level orchestration (like
        # --start): the typed route threads them, so they compose WITH a
        # [pipeline] config instead of being rejected.
        assert self._flags(gop_align=True) == []
        assert self._flags(snap_start=True) == []
