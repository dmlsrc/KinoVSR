"""Parser behavior: canonical flags, retired spellings, exit codes, assembly."""

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

RETIRED_FLAGS = (
    "--spatial-mode",
    "--restore-flow-mode",
    "--basicvsrpp-flow-mode",
    "--realbasicvsr-flow-mode",
    "--realviformer-flow-mode",
    "--basicvsrpp-variant",
    "--bsvd-variant",
    "--toflow-variant",
    "--pvdd-variant",
    "--fastdvd-variant",
    "--fastdvd-weights",
    "--realesrgan-denoise",
    "--realbasicvsr-dynamic-refine-thres",
    "--deblock-weights",
)


class TestCanonicalVocabulary:
    def test_canonical_flags_land_on_expected_destinations(self, parser):
        args = parser.parse_args(
            [*BASE, "--upscale", "safmn", "--basicvsrpp-flow", "vt",
             "--fastdvdnet-profile", "standard",
             "--realesrgan-denoise-strength", "0.3",
             "--realbasicvsr-clean-threshold", "12"])
        assert args.upscale == "safmn"
        assert args.basicvsrpp_flow == "vt"
        assert args.fastdvdnet_profile == "standard"
        assert args.realesrgan_denoise_strength == 0.3
        assert args.realbasicvsr_clean_threshold == 12

    def test_canonical_defaults(self, parser):
        args = parser.parse_args(BASE)
        assert args.upscale == "balanced"
        assert args.fastdvdnet_profile == "clipped"
        assert args.overwrite is False

    def test_overwrite_is_explicit(self, parser):
        args = parser.parse_args([*BASE, "--overwrite"])
        assert args.overwrite is True

    def test_help_shows_canonical_names(self, parser):
        text = parser.format_help()
        for canonical in ("--upscale", "--fastdvdnet-profile",
                          "--stdf-weights", "--fbcnn-weights",
                          "--realesrgan-denoise-strength", "--base-config",
                          "--verbose", "--mc-flow "):
            assert canonical in text, canonical

    @pytest.mark.parametrize("flag", RETIRED_FLAGS)
    def test_retired_flags_are_rejected(self, parser, capsys, flag):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([*BASE, flag, "unused"])
        assert exc.value.code == 2
        capsys.readouterr()


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

    @pytest.mark.parametrize("flags", [
        ["--gop-min-window", "0"],
        ["--gop-max-window", "0"],
        ["--gop-min-window", "20", "--gop-max-window", "10"],
    ])
    def test_invalid_gop_windows_exit_two(self, parser, capsys, flags):
        args = parser.parse_args([*BASE, "--gop-align", *flags])
        with pytest.raises(SystemExit) as exc:
            validate_args(parser, args)
        assert exc.value.code == 2
        capsys.readouterr()

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_invalid_video_chunk_size_exits_two(
            self, parser, capsys, value):
        args = parser.parse_args([*BASE, "--video-chunk-size", value])
        with pytest.raises(SystemExit) as exc:
            validate_args(parser, args)
        assert exc.value.code == 2
        capsys.readouterr()

    def test_probe_noise_waives_output_dir(self, parser):
        args = parser.parse_args(["--video", "clip.mp4", "--probe-noise"])
        validate_args(parser, args)

    def test_flat_noise_probe_threads_the_validated_chunk(
            self, monkeypatch, tmp_path):
        from kinovsr.cli.commands import probe_noise as probe_module
        from kinovsr.cli.commands.run import run_video_command

        captured = {}

        def probe(video, *, start_spec, end_spec, reader, chunk_size):
            captured["chunk_size"] = chunk_size
            return 0

        monkeypatch.setattr(probe_module, "probe_noise", probe)
        assert run_video_command([
            "--video", str(tmp_path / "unused.mp4"),
            "--probe-noise",
            "--video-chunk-size", "1",
        ]) == 0
        assert captured["chunk_size"] == 1

    def test_realbasicvsr_window_bounds(self, parser, capsys):
        args = parser.parse_args(
            [*BASE, "--upscale", "realbasicvsr", "--realbasicvsr-window", "0"])
        with pytest.raises(SystemExit) as exc:
            validate_args(parser, args)
        assert exc.value.code == 2
        capsys.readouterr()


class TestChains:
    def test_chain_whitespace_normalizes(self):
        assert normalize_chain("off") == "off"
        assert normalize_chain("mc, bsvd") == "mc,bsvd"

    def test_assemble_normalizes_canonical_denoise_chain(self, parser):
        args = parser.parse_args([*BASE, "--denoise", "fastdvdnet, bsvd"])
        inv = assemble(args, base=Settings())
        assert inv.options.denoise == "fastdvdnet,bsvd"

    def test_removed_family_token_is_rejected(self, parser):
        args = parser.parse_args([*BASE, "--denoise", "fastdvd"])
        with pytest.raises(ConfigError, match="fastdvd"):
            assemble(args, base=Settings())


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

    @pytest.mark.parametrize(("flag", "token", "selector"), [
        ("--basicvsrpp-weights", "reds4", "--basicvsrpp-profile"),
        ("--restore-weights", "denoise", "--restore"),
        ("--nafnet-weights", "gopro", "--nafnet"),
    ])
    def test_profile_tokens_are_rejected_in_weight_flags(
            self, parser, flag, token, selector):
        args = parser.parse_args([*BASE, flag, token])
        with pytest.raises(ConfigError, match=selector):
            assemble(args, base=Settings())

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


class TestDeblockWeights:
    def test_family_weight_flags_are_independent(self, parser):
        args = parser.parse_args(
            [*BASE, "--deblock", "stdf,fbcnn",
             "--stdf-weights", "/w/stdf.st",
             "--fbcnn-weights", "/w/fb.st"])
        inv = assemble(args, base=Settings())
        assert inv.settings.stdf_weights == "/w/stdf.st"
        assert inv.settings.fbcnn_weights == "/w/fb.st"

    def test_family_weight_does_not_fill_another_family(self, parser):
        args = parser.parse_args(
            [*BASE, "--deblock", "stdf", "--stdf-weights", "/w/stdf.st"])
        inv = assemble(args, base=Settings())
        assert inv.settings.stdf_weights == "/w/stdf.st"
        assert inv.settings.fbcnn_weights is None


class TestTypedSourceLayout:
    """The typed route decodes into a layout the chain's head accepts."""

    def _pick(self, config, **kwargs):
        from kinovsr.cli.commands.run import _source_layout

        return _source_layout(config, **kwargs)

    def test_native_head_gets_a_cv_layout(self):
        from kinovsr.processors import Layout

        config = {"pipeline": ["fps"],
                  "fps": {"processor": "videotoolbox", "profile": "normal",
                          "target_fps": 50}}
        assert self._pick(config) is Layout.CV_RGBA_HALF

    def test_bare_upscale_head_decodes_zero_copy(self):
        # A bare videotoolbox-upscale head prefers its mode's own session
        # format so the decode feeds the scaler zero-copy - the harness's
        # mainstream path. fast is NV12, balanced/image are RGBAHalf.
        from kinovsr.processors import Layout

        balanced = {"pipeline": ["up"],
                    "up": {"processor": "videotoolbox",
                           "capability": "upscale", "profile": "balanced"}}
        fast = {"pipeline": ["up"],
                "up": {"processor": "videotoolbox",
                       "capability": "upscale", "profile": "fast"}}
        assert self._pick(balanced) is Layout.CV_RGBA_HALF
        assert self._pick(fast) is Layout.CV_NV12

    def test_forced_color_or_range_uses_the_mlx_decode_bridge(self):
        from kinovsr.processors import Layout

        balanced = {"pipeline": ["up"],
                    "up": {"processor": "videotoolbox",
                           "capability": "upscale", "profile": "balanced"}}
        assert self._pick(
            balanced, source_color="bt601") is Layout.MLX_RGB_HWC
        assert self._pick(
            balanced, source_range="full") is Layout.MLX_RGB_HWC

    def test_preprocessed_upscale_head_stays_mlx(self):
        # With an MLX stage ahead of the upscale, the head is that stage:
        # the decode goes MLX and the upscale takes the bridge.
        from kinovsr.processors import Layout

        config = {"pipeline": ["c", "up"],
                  "c": {"processor": "crop", "bars": "16,16,0,0"},
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
            "gop_align": False, "target_fps": None,
            "temporal_mode": "normal"}
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

    def test_target_fps_is_rejected_with_an_explicit_pipeline(self):
        assert "target_fps" in self._flags(target_fps=50.0)
        assert "temporal_mode" in self._flags(temporal_mode="high")

    def test_conform_cfr_is_rejected_with_an_explicit_pipeline(self):
        assert "conform_cfr" in self._flags(conform_cfr=30.0)

    def test_typed_command_returns_config_error_for_target_fps(self):
        from types import SimpleNamespace

        from kinovsr.cli.commands.run import _run_typed

        options = SimpleNamespace(
            output_dir="unused",
            target_fps=50.0,
            temporal_mode="normal",
        )
        invocation = SimpleNamespace(
            options=options,
            config={"pipeline": []},
        )
        assert _run_typed(invocation) == 2

    def test_typed_command_returns_config_error_for_conform_cfr(self):
        from types import SimpleNamespace

        from kinovsr.cli.commands.run import _run_typed

        options = SimpleNamespace(
            output_dir="unused",
            target_fps=None,
            temporal_mode="normal",
            conform_cfr=30.0,
        )
        invocation = SimpleNamespace(
            options=options,
            config={"pipeline": []},
        )
        assert _run_typed(invocation) == 2
