"""Cross-family invariants for the M4 factory sweep.

Every learned family resolves through the catalog, declares profiles the
family manifest knows, parses an empty stage table to its documented
defaults, rejects unknown keys, and (for upscalers) rewrites geometry by
the profile's manifest-declared scale - all without loading weights.
"""

from fractions import Fraction

import pytest

from kinovsr.modeling.weights import load_registered
from kinovsr.processors import (
    Capability,
    Geometry,
    StreamSpec,
    TemporalMode,
    TimelineSpec,
    frame_spec_for_matrix,
    get_factory,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

SETTINGS = Settings()

# family -> capabilities the sweep exercises (first profile is default).
SWEPT = {
    "basicvsrpp": (Capability.UPSCALE, Capability.RESTORE),
    "esc": (Capability.UPSCALE,),
    "fastdvdnet": (Capability.DENOISE,),
    "fbcnn": (Capability.DEBLOCK,),
    "nafnet": (Capability.DEBLUR, Capability.DENOISE, Capability.RESTORE),
    "pvdd": (Capability.DENOISE,),
    "realbasicvsr": (Capability.UPSCALE,),
    "realesrgan": (Capability.UPSCALE,),
    "realviformer": (Capability.UPSCALE,),
    "safmn": (Capability.UPSCALE,),
    "stdf": (Capability.DEBLOCK,),
    "toflow": (Capability.DENOISE, Capability.DEBLOCK, Capability.UPSCALE),
}

CASES = [(family, capability)
         for family, capabilities in SWEPT.items()
         for capability in capabilities]
IDS = [f"{family}-{capability.value}" for family, capability in CASES]


def stream() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(192, 144)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)))


@pytest.mark.parametrize("family,capability", CASES, ids=IDS)
class TestSweep:
    def test_catalog_resolves_and_names_match(self, family, capability):
        factory = get_factory(family)
        assert factory.name == family
        assert capability in factory.capabilities

    def test_profiles_exist_in_the_manifest(self, family, capability):
        spec = get_factory(family).capabilities[capability]
        manifest = load_registered(family)
        missing = [p for p in spec.profiles if p not in manifest.profiles]
        assert not missing, f"profiles not in manifest: {missing}"

    def test_empty_table_parses_to_defaults(self, family, capability):
        factory = get_factory(family)
        config = factory.parse_config(
            {}, capability=capability, profile=None, settings=SETTINGS)
        assert config is not None

    def test_unknown_key_is_rejected(self, family, capability):
        factory = get_factory(family)
        with pytest.raises(ValueError, match="unknown"):
            factory.parse_config(
                {"nonsense_key": 1}, capability=capability, profile=None,
                settings=SETTINGS)

    def test_build_defers_weight_loading(self, family, capability):
        factory = get_factory(family)
        config = factory.parse_config(
            {}, capability=capability, profile=None, settings=SETTINGS)
        from kinovsr.processors import PipelineContext

        processor = factory.build(config, context=PipelineContext(SETTINGS))
        assert hasattr(processor, "process")


class TestUpscaleGeometry:
    @pytest.mark.parametrize("family,profile,scale", [
        ("esc", "gan", 4),
        ("safmn", "light", 4),
        ("safmn", "real2x", 2),
        ("realesrgan", "general", 4),
        ("realesrgan", "x2plus", 2),
        ("realviformer", "x4", 4),
        ("realbasicvsr", "x4", 4),
        ("basicvsrpp", "vimeo90k_bd", 4),
        ("toflow", "sr", 4),
    ])
    def test_produces_scales_by_the_profile(self, family, profile, scale):
        factory = get_factory(family)
        spec = factory.capabilities[Capability.UPSCALE]
        config = factory.parse_config(
            {}, capability=Capability.UPSCALE, profile=profile,
            settings=SETTINGS)
        out = spec.produces(stream(), config)
        assert out.frame.geometry.width == 192 * scale
        assert out.frame.geometry.height == 144 * scale

    def test_explicit_path_requires_scale(self):
        factory = get_factory("safmn")
        with pytest.raises(ValueError, match="state scale"):
            factory.parse_config(
                {"weights": "/somewhere/custom.safetensors"},
                capability=Capability.UPSCALE, profile=None,
                settings=SETTINGS)

    def test_restore_preserves_geometry(self):
        factory = get_factory("basicvsrpp")
        spec = factory.capabilities[Capability.RESTORE]
        config = factory.parse_config(
            {"strength": 0.5}, capability=Capability.RESTORE,
            profile="decompress_track1", settings=SETTINGS)
        out = spec.produces(stream(), config)
        assert out.frame.geometry == stream().frame.geometry


class TestCapabilityDispatch:
    def test_toflow_capability_picks_the_checkpoint(self):
        factory = get_factory("toflow")
        denoise = factory.parse_config(
            {}, capability=Capability.DENOISE, profile=None,
            settings=SETTINGS)
        deblock = factory.parse_config(
            {}, capability=Capability.DEBLOCK, profile=None,
            settings=SETTINGS)
        assert denoise.capability is Capability.DENOISE
        assert deblock.capability is Capability.DEBLOCK

    def test_toflow_sr_rejects_denoise_keys(self):
        factory = get_factory("toflow")
        with pytest.raises(ValueError, match="unknown"):
            factory.parse_config(
                {"passes": 3}, capability=Capability.UPSCALE, profile=None,
                settings=SETTINGS)

    def test_nafnet_capability_scopes_the_default_variant(self):
        factory = get_factory("nafnet")
        assert factory.parse_config(
            {}, capability=Capability.DEBLUR, profile=None,
            settings=SETTINGS).variant == "gopro"
        assert factory.parse_config(
            {}, capability=Capability.DENOISE, profile=None,
            settings=SETTINGS).variant == "sidd"
        assert factory.parse_config(
            {}, capability=Capability.RESTORE, profile=None,
            settings=SETTINGS).variant == "reds"

    def test_basicvsrpp_restore_takes_strength_not_history(self):
        factory = get_factory("basicvsrpp")
        config = factory.parse_config(
            {"strength": 0.5}, capability=Capability.RESTORE,
            profile="denoise", settings=SETTINGS)
        assert config.strength == 0.5
        with pytest.raises(ValueError, match="unknown"):
            factory.parse_config(
                {"history_gate": "improve"}, capability=Capability.RESTORE,
                profile="denoise", settings=SETTINGS)


class TestParseDetails:
    def test_fbcnn_quality_grammar(self):
        factory = get_factory("fbcnn")

        def parse(quality):
            return factory.parse_config(
                {"quality": quality}, capability=Capability.DEBLOCK,
                profile=None, settings=SETTINGS).quality

        assert parse("auto") == "auto"
        assert parse("blind") is None
        assert parse("35") == 35.0
        with pytest.raises(ValueError, match="quality"):
            parse("vivid")

    def test_pvdd_preset_maps_to_variance(self):
        factory = get_factory("pvdd")
        config = factory.parse_config(
            {"noise_preset": "M"}, capability=Capability.DENOISE,
            profile=None, settings=SETTINGS)
        assert config.noise_variance == pytest.approx(0.002191)
        off = factory.parse_config(
            {"noise_preset": "off"}, capability=Capability.DENOISE,
            profile=None, settings=SETTINGS)
        assert off.noise_variance is None

    def test_pvdd_trim_bound(self):
        factory = get_factory("pvdd")
        with pytest.raises(ValueError, match="trim"):
            factory.parse_config(
                {"window": 10, "trim": 5}, capability=Capability.DENOISE,
                profile=None, settings=SETTINGS)

    def test_temporal_shapes_match_the_taxonomy(self):
        centered = {
            "fastdvdnet": 2, "stdf": 3, "pvdd": 10,
            "realbasicvsr": 14,
        }
        for family, radius in centered.items():
            spec = next(iter(get_factory(family).capabilities.values()))
            assert spec.temporal_mode is TemporalMode.CENTERED, family
            assert spec.temporal_radius == radius, family
            assert spec.stateful, family
        viformer = get_factory("realviformer").capabilities[Capability.UPSCALE]
        assert viformer.temporal_mode is TemporalMode.CAUSAL
        per_frame = get_factory("esc").capabilities[Capability.UPSCALE]
        assert per_frame.temporal_mode is TemporalMode.PER_FRAME
        assert not per_frame.stateful
