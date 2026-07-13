"""Flag -> [pipeline] config assembler: the harness semantics as mapping."""

from fractions import Fraction

import pytest

from kinovsr.cli.args import build_parser
from kinovsr.cli.assemble_pipeline import assemble_pipeline
from kinovsr.pipeline import resolve_pipeline
from kinovsr.processors import (
    Geometry,
    StreamSpec,
    TimelineSpec,
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
    def test_bare_invocation_is_the_balanced_upscale(self):
        config = asm()
        assert config["pipeline"] == ["upscale"]
        assert config["upscale"] == {"processor": "videotoolbox",
                                     "capability": "upscale",
                                     "profile": "balanced"}

    def test_upscale_none_omits_the_stage(self):
        assert asm("--upscale", "none")["pipeline"] == []

    def test_default_slot_order(self):
        config = asm("--restore", "decompress_track1", "--deflicker", "on",
                     "--deblock", "stdf", "--denoise", "mc",
                     "--nafnet", "gopro")
        assert config["pipeline"] == [
            "restore_decompress_track1", "deflicker", "deblock_stdf",
            "denoise_mc", "nafnet", "upscale"]

    def test_denoise_first_swaps_the_pair(self):
        config = asm("--deblock", "stdf", "--denoise", "mc",
                     "--denoise-first")
        assert config["pipeline"] == ["denoise_mc", "deblock_stdf",
                                      "upscale"]

    def test_explicit_order_appends_omitted_in_default_order(self):
        config = asm("--denoise", "mc", "--deblock", "stdf",
                     "--nafnet", "sidd", "--preprocess-order", "denoise")
        assert config["pipeline"] == ["denoise_mc", "deblock_stdf",
                                      "nafnet", "upscale"]

    def test_chains_expand_in_order(self):
        config = asm("--denoise", "mc,bsvd")
        assert config["pipeline"] == ["denoise_mc", "denoise_bsvd",
                                      "upscale"]

    def test_geometry_then_cut_precede_the_slots(self):
        config = asm("--crop-bars", "16,16,0,0", "--square-pixels",
                     "--cut-detect", "simple", "--denoise", "mc")
        assert config["pipeline"] == ["crop", "square", "cut", "denoise_mc",
                                      "upscale"]

    def test_target_fps_appends_the_interpolate_stage(self):
        config = asm("--target-fps", "50", "--temporal-mode", "high")
        assert config["pipeline"] == ["upscale", "fps"]
        assert config["fps"]["profile"] == "high"
        assert config["fps"]["target_fps"] == 50.0


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

    def test_unused_family_dials_are_ignored(self):
        config = asm("--bsvd-strength", "0.7")   # no bsvd in the chain
        assert config["pipeline"] == ["upscale"]

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
