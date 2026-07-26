"""Flag -> [pipeline] config assembler: the harness semantics as mapping."""

from fractions import Fraction

import pytest

from kinovsr.cli.args import build_parser
from kinovsr.cli.assemble_pipeline import assemble_pipeline
from kinovsr.config import ConfigError
from kinovsr.pipeline import resolve_pipeline
from kinovsr.processors import (
    Geometry,
    StreamSpec,
    TimelineSpec,
    UnknownStageError,
    frame_spec_for_matrix,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

PARSER = build_parser()


def asm(*argv, width=640, height=480):
    args = PARSER.parse_args(
        ["--video", "x.mp4", "--output-dir", "/tmp/o", *argv])
    return assemble_pipeline(args, width=width, height=height)


def spec(w=640, h=480):
    return StreamSpec(
        frame=frame_spec_for_matrix("bt709", full_range=False,
                                    geometry=Geometry(w, h)),
        timeline=TimelineSpec(time_base=Fraction(1, 24000),
                              cadence=Fraction(25)))


class TestDefaultsAndOrder:
    def test_bare_invocation_selects_no_processors(self):
        assert asm()["pipeline"] == []

    def test_upscale_none_omits_the_stage(self):
        assert asm("--upscale", "none")["pipeline"] == []

    def test_vision_flow_routes_to_balanced_vt_sr(self):
        config = asm(
            "--upscale", "balanced", "--vt-sr-flow", "vision"
        )
        assert config["upscale"]["flow"] == "vision"

    @pytest.mark.parametrize("mode", ["none", "fast", "image", "metalfx"])
    def test_vt_sr_flow_rejects_nonbalanced_upscalers(self, mode):
        with pytest.raises(ConfigError, match="only valid.*balanced"):
            asm("--upscale", mode, "--vt-sr-flow", "vision")

    def test_default_slot_order(self):
        config = asm("--restore", "decompress_track1", "--deflicker", "on",
                     "--deblock", "stdf", "--denoise", "mc",
                     "--nafnet", "gopro", "--upscale", "balanced")
        assert config["pipeline"] == [
            "restore_decompress_track1", "deflicker", "deblock_stdf",
            "denoise_mc", "nafnet", "upscale"]

    def test_denoise_first_swaps_the_pair(self):
        config = asm("--deblock", "stdf", "--denoise", "mc",
                     "--denoise-first", "--upscale", "balanced")
        assert config["pipeline"] == ["denoise_mc", "deblock_stdf",
                                      "upscale"]

    def test_explicit_order_appends_omitted_in_default_order(self):
        config = asm("--denoise", "mc", "--deblock", "stdf",
                     "--nafnet", "sidd", "--preprocess-order", "denoise",
                     "--upscale", "balanced")
        assert config["pipeline"] == ["denoise_mc", "deblock_stdf",
                                      "nafnet", "upscale"]

    def test_chains_expand_in_order(self):
        config = asm("--denoise", "mc,bsvd")
        assert config["pipeline"] == ["denoise_mc", "denoise_bsvd"]
        assert config["denoise_mc"]["capability"] == "denoise"
        assert config["denoise_bsvd"]["capability"] == "denoise"

    @pytest.mark.parametrize("argv", [
        ["--denoise", "stdf"],
        ["--deblock", "bsvd"],
    ])
    def test_wrong_slot_family_fails_preflight(self, argv):
        with pytest.raises(UnknownStageError, match="does not offer"):
            resolve_pipeline(
                asm(*argv), input_spec=spec(), settings=Settings())

    def test_geometry_then_cut_precede_the_slots(self):
        config = asm("--crop-bars", "16,16,0,0", "--square-pixels",
                     "--cut-detect", "simple", "--denoise", "mc")
        assert config["pipeline"] == ["crop", "square", "cut", "denoise_mc"]

    def test_target_fps_appends_the_interpolate_stage(self):
        config = asm("--target-fps", "50", "--temporal-mode", "high")
        assert config["pipeline"] == ["fps"]
        assert config["fps"]["profile"] == "high"
        assert config["fps"]["target_fps"] == 50.0

    def test_conform_cfr_with_target_fps_is_a_config_error(self):
        with pytest.raises(ConfigError, match="pick one"):
            asm("--conform-cfr", "30", "--target-fps", "60")


class TestGeometry:
    def test_odd_explicit_bars_get_the_even_bump(self):
        config = asm("--crop-bars", "3,0,0,0")
        assert config["crop"]["bars"] == "3,1,0,0"

    def test_explicit_bars_and_trim_fold(self):
        config = asm("--crop-bars", "16,16,0,0", "--sanitize-edges",
                     "2,0,0,0", "--sanitize-edges-fill", "trim")
        assert config["crop"]["bars"] == "18,16,0,0"
        assert "trim" not in config["crop"]

    def test_auto_values_pass_through_to_the_probe(self):
        config = asm("--crop-bars", "auto", "--sanitize-edges", "auto",
                     "--sanitize-edges-fill", "trim")
        assert config["crop"]["bars"] == "auto"
        assert config["crop"]["trim"] == "auto"

    def test_sanitize_fill_becomes_a_stage(self):
        # the flag surface's default fill is restore (the harness default)
        config = asm("--sanitize-edges", "2,0,0,0")
        assert config["sanitize"]["fill"] == "restore"
        assert config["sanitize"]["edges"] == "2,0,0,0"
        config = asm("--sanitize-edges", "2,0,0,0",
                     "--sanitize-edges-fill", "extend")
        assert config["sanitize"]["fill"] == "extend"


class TestDialRouting:
    def test_shared_dial_fills_the_slot_family_overrides(self):
        config = asm("--denoise", "mc,bsvd", "--denoise-strength", "0.4",
                     "--bsvd-strength", "0.7")
        assert config["denoise_mc"]["strength"] == 0.4
        assert config["denoise_bsvd"]["strength"] == 0.7

    def test_shared_strength_list_distributes_positionally(self):
        config = asm("--denoise", "mc,bsvd",
                     "--denoise-strength", "0.3,0.7")
        assert config["denoise_mc"]["strength"] == 0.3
        assert config["denoise_bsvd"]["strength"] == 0.7

    def test_shared_strength_list_requires_matching_arity(self):
        with pytest.raises(ConfigError, match="3 values for 2"):
            asm("--denoise", "mc,bsvd",
                "--denoise-strength", "0.2,0.4,0.6")

    def test_noise_map_reaches_capable_denoisers(self):
        config = asm("--denoise", "spatial,bsvd", "--noise-map", "auto",
                     "--noise-map-gain", "1.2")
        assert config["denoise_bsvd"]["noise_map"] == "auto"
        assert config["denoise_bsvd"]["noise_map_gain"] == 1.2
        assert "noise_map" not in config["denoise_spatial"]

    def test_deblock_map_reaches_stdf_and_fbcnn(self):
        config = asm("--deblock", "stdf,fbcnn", "--deblock-map", "auto")
        assert config["deblock_stdf"]["deblock_map"] == "auto"
        assert config["deblock_fbcnn"]["deblock_map"] == "auto"

    def test_fbcnn_quality_number_survives_assembly(self):
        # the untyped dial floatifies "35"; the stage parser must accept it
        config = asm("--deblock", "fbcnn", "--fbcnn-quality", "35")
        assert config["deblock_fbcnn"]["quality"] == 35.0

    def test_noise_map_skips_blind_pvdd_with_a_warning(self, caplog):
        # the broadcast dial is capability-scoped: a blind PVDD stage runs
        # unconditioned (with a warning) instead of refusing the chain
        import logging

        with caplog.at_level(logging.WARNING):
            config = asm("--denoise", "mc,pvdd", "--noise-map", "auto",
                         "--noise-map-gain", "1.2")
        assert config["denoise_mc"]["noise_map"] == "auto"
        assert config["denoise_mc"]["noise_map_gain"] == 1.2
        assert not any(k.startswith("noise_map")
                       for k in config["denoise_pvdd"])
        assert any("does not apply" in r.message for r in caplog.records)

    def test_noise_map_reaches_a_level_pvdd(self):
        config = asm("--denoise", "mc,pvdd", "--pvdd-profile", "pvdd_level",
                     "--noise-map", "auto")
        assert config["denoise_mc"]["noise_map"] == "auto"
        assert config["denoise_pvdd"]["noise_map"] == "auto"

    def test_noise_map_with_no_capable_stage_is_a_config_error(self):
        with pytest.raises(ConfigError, match="applies to no stage"):
            asm("--denoise", "spatial", "--noise-map", "auto")

    def test_noise_map_on_a_lone_blind_pvdd_is_a_config_error(self, caplog):
        # the stage-level skip fires its warning, then the chain-level
        # check refuses: nothing would be conditioned at all
        import logging

        with caplog.at_level(logging.WARNING), \
                pytest.raises(ConfigError, match="applies to no stage"):
            asm("--denoise", "pvdd", "--noise-map", "auto")
        assert any("does not apply" in r.message for r in caplog.records)

    def test_unused_family_dials_are_ignored(self):
        config = asm("--bsvd-strength", "0.7")   # no bsvd in the chain
        assert config["pipeline"] == []

    def test_nafnet_value_selects_capability_and_profile(self):
        config = asm("--nafnet", "sidd32")
        assert config["nafnet"]["capability"] == "denoise"
        assert config["nafnet"]["profile"] == "sidd32"


class TestResolves:
    @pytest.mark.parametrize("argv", [
        [],
        ["--denoise", "mc,bsvd", "--noise-map", "auto"],
        ["--crop-bars", "16,16,0,0", "--square-pixels",
         "--cut-detect", "simple"],
        ["--deflicker", "on", "--deblock", "stdf"],
        ["--target-fps", "50"],
        ["--upscale", "metalfx", "--target-fps", "50"],
        ["--upscale", "fast"],
    ])
    def test_assembled_configs_preflight(self, argv):
        plan = resolve_pipeline(asm(*argv), input_spec=spec(),
                                settings=Settings())
        assert plan.output_spec is not None


class TestOffOnDialsAreBooleans:
    """An off|on choice pair is a boolean spelled for the command line.

    The stage parsers type these keys with ``typed_value(..., bool, ...)``,
    and ``_set_flags`` only routes a value that differs from the registry
    default - so the only routable value is the one that used to be handed
    to the family as the string 'on'/'off' and refused at preflight.
    """

    @pytest.mark.parametrize("argv, stage, key, expected", [
        (["--deflicker", "on", "--deflicker-jitter", "on"],
         "deflicker", "jitter", True),
        (["--deflicker", "on", "--deflicker-gop", "off"],
         "deflicker", "gop", False),
        (["--deblock", "fbcnn", "--fbcnn-gop", "off"],
         "deblock_fbcnn", "gop", False),
    ])
    def test_off_on_dial_reaches_the_stage_as_a_bool(
            self, argv, stage, key, expected):
        config = asm(*argv)
        assert config[stage][key] is expected

    @pytest.mark.parametrize("argv", [
        ["--deflicker", "on", "--deflicker-jitter", "on"],
        ["--deflicker", "on", "--deflicker-gop", "off"],
        ["--deblock", "fbcnn", "--fbcnn-gop", "off"],
    ])
    def test_off_on_dial_survives_preflight(self, argv):
        plan = resolve_pipeline(asm(*argv), input_spec=spec(),
                                settings=Settings())
        assert plan.output_spec is not None

    def test_a_wider_choice_set_stays_a_token(self):
        config = asm("--denoise", "bsvd", "--noise-map-motion-cap", "loose")
        assert config["denoise_bsvd"]["noise_map_motion_cap"] == "loose"


class TestRepeatedFamilyInOneSlot:
    """Each occurrence is an independent pass and needs its own table.

    Sharing one table made the positional dial list collapse to its last
    value while still reporting the right number of stages.
    """

    def test_repeat_gets_distinct_stage_names(self):
        config = asm("--denoise", "mc,mc")
        assert config["pipeline"] == ["denoise_mc", "denoise_mc_2"]
        assert config["denoise_mc"] is not config["denoise_mc_2"]

    def test_positional_dials_address_each_instance(self):
        config = asm("--denoise", "mc,mc", "--denoise-strength", "0.3,0.7")
        assert config["denoise_mc"]["strength"] == 0.3
        assert config["denoise_mc_2"]["strength"] == 0.7

    def test_three_of_a_kind_keep_ascending_suffixes(self):
        config = asm("--denoise", "mc,mc,mc",
                     "--denoise-strength", "0.1,0.2,0.3")
        assert config["pipeline"] == [
            "denoise_mc", "denoise_mc_2", "denoise_mc_3"]
        assert [config[n]["strength"] for n in config["pipeline"]] == [
            0.1, 0.2, 0.3]

    def test_mixed_chain_is_unchanged(self):
        config = asm("--denoise", "mc,bsvd", "--denoise-strength", "0.3,0.7")
        assert config["pipeline"] == ["denoise_mc", "denoise_bsvd"]
        assert config["denoise_mc"]["strength"] == 0.3
        assert config["denoise_bsvd"]["strength"] == 0.7

    def test_a_family_dial_still_reaches_every_instance(self):
        config = asm("--denoise", "mc,mc", "--mc-sigma", "0.09")
        assert config["denoise_mc"]["sigma"] == 0.09
        assert config["denoise_mc_2"]["sigma"] == 0.09

    def test_repeated_chain_preflights(self):
        plan = resolve_pipeline(
            asm("--denoise", "mc,mc", "--denoise-strength", "0.3,0.7"),
            input_spec=spec(), settings=Settings())
        assert plan.output_spec is not None


class TestNoiseMapUpsampleDial:
    def test_upsample_choice_reaches_capable_stages(self):
        config = asm("--denoise", "mc,bsvd", "--noise-map", "auto",
                     "--noise-map-upsample", "box")
        assert config["denoise_mc"]["noise_map_upsample"] == "box"
        assert config["denoise_bsvd"]["noise_map_upsample"] == "box"


class TestStrengthBroadcastScoping:
    def test_strength_skips_pvdd_with_a_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            config = asm("--denoise", "pvdd,mc", "--denoise-strength", "1")
        assert config["denoise_mc"]["strength"] == 1.0
        assert "strength" not in config["denoise_pvdd"]
        assert any("does not apply" in r.message for r in caplog.records)

    def test_positional_strengths_keep_chain_alignment_past_pvdd(self):
        config = asm("--denoise", "pvdd,mc", "--denoise-strength", "0.3,0.7")
        assert "strength" not in config["denoise_pvdd"]
        assert config["denoise_mc"]["strength"] == 0.7

    def test_strength_on_a_lone_pvdd_is_a_config_error(self):
        with pytest.raises(ConfigError, match="applies to no stage"):
            asm("--denoise", "pvdd", "--denoise-strength", "0.8")
