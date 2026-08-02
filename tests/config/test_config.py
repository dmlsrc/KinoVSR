"""Config layer: merge rules, composition, typed --set, validation, precedence."""

from __future__ import annotations

import tomllib
from fractions import Fraction

import pytest

from kinovsr.config import (
    ConfigError,
    apply_set_overrides,
    compose_config,
    dump_config,
    load_config,
    merge_configs,
    parse_set_argument,
    split_stage_table,
    validate_config,
)
from kinovsr.pipeline import resolve_pipeline
from kinovsr.processors import (
    Geometry,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.settings import Settings

# The documented example pair from the planning target structure: a base
# config with a stage library and a clip overlay that restates the list.
BASE_TOML = """\
pipeline = ["denoise", "deblock", "restore", "upscale"]

[settings]
quiet = false
mlx_cache_limit_gb = 1.0

[input]
video = "clip.mp4"

[output]
output_dir = "out"
audio = true

[denoise]
processor = "bsvd"
strength = 0.03

[deblock]
processor = "toflow"
capability = "deblock"
passes = 1
flow_scale = "half"

[restore]
processor = "basicvsrpp"
profile = "deblur_gopro"

[upscale]
processor = "safmn"
profile = "light"
"""

CLIP_TOML = """\
pipeline = ["deblock", "restore", "upscale"]

[restore]
profile = "deblur_dvd"

[upscale]
processor = "realesrgan"
profile = "general"
denoise_strength = 0.35
"""


@pytest.fixture
def base_and_clip(tmp_path):
    base = tmp_path / "base.toml"
    clip = tmp_path / "clip.toml"
    base.write_text(BASE_TOML)
    clip.write_text(CLIP_TOML)
    return base, clip


# ---- merge rules -------------------------------------------------------------


def test_rule1_tables_merge_recursively():
    a = {"upscale": {"processor": "safmn", "profile": "light", "nested": {"x": 1}}}
    b = {"upscale": {"profile": "real", "nested": {"y": 2}}}
    out = merge_configs(a, b)
    assert out["upscale"] == {
        "processor": "safmn", "profile": "real", "nested": {"x": 1, "y": 2}}


def test_rule2_scalars_and_arrays_replace():
    a = {"x": 1, "arr": [1, 2, 3]}
    b = {"x": 2, "arr": [9]}
    out = merge_configs(a, b)
    assert out == {"x": 2, "arr": [9]}


def test_rule3_pipeline_is_restated_not_spliced():
    a = {"pipeline": ["denoise", "deblock", "upscale"]}
    b = {"pipeline": ["deblock", "upscale"]}
    assert merge_configs(a, b)["pipeline"] == ["deblock", "upscale"]


def test_table_replaced_by_scalar_and_back():
    assert merge_configs({"k": {"a": 1}}, {"k": 5}) == {"k": 5}
    assert merge_configs({"k": 5}, {"k": {"a": 1}}) == {"k": {"a": 1}}


def test_merge_does_not_mutate_inputs():
    a = {"t": {"x": 1}}
    b = {"t": {"y": 2}}
    merge_configs(a, b)
    assert a == {"t": {"x": 1}} and b == {"t": {"y": 2}}


# ---- composition (the documented base + clip example) -------------------------


def test_compose_base_plus_clip_matches_documented_semantics(base_and_clip):
    base, clip = base_and_clip
    cfg = compose_config([base], clip)

    # The overlay restated the list: denoise is out of the chain...
    assert cfg["pipeline"] == ["deblock", "restore", "upscale"]
    # ...but its tuned stage table survives as a library entry.
    assert cfg["denoise"] == {"processor": "bsvd", "strength": 0.03}
    # Table merge: overlay tweaked only the profile; processor survives.
    assert cfg["restore"] == {"processor": "basicvsrpp", "profile": "deblur_dvd"}
    # Overlay swapped the upscaler and added a setting.
    assert cfg["upscale"] == {
        "processor": "realesrgan", "profile": "general",
        "denoise_strength": 0.35}
    # Foundation tables pass through untouched.
    assert cfg["input"] == {"video": "clip.mp4"}
    assert cfg["output"] == {"output_dir": "out", "audio": True}
    validate_config(cfg)


def test_documented_example_preflights_through_family_parsers(base_and_clip):
    base, clip = base_and_clip
    config = compose_config([base], clip)
    stream = StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(640, 480)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)),
    )
    assert resolve_pipeline(
        config, input_spec=stream, settings=Settings()
    ).output_spec is not None


def test_compose_applies_bases_in_order(tmp_path):
    p1 = tmp_path / "a.toml"
    p2 = tmp_path / "b.toml"
    p1.write_text('[stage]\nprocessor = "safmn"\nv = 1\n')
    p2.write_text("[stage]\nv = 2\n")
    assert compose_config([p1, p2])["stage"]["v"] == 2
    assert compose_config([p2, p1])["stage"]["v"] == 1


def test_compose_with_no_files_is_empty():
    assert compose_config(None, None) == {}


def test_load_config_errors_name_the_file(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("not = valid = toml")
    with pytest.raises(ConfigError, match="bad.toml"):
        load_config(bad)
    with pytest.raises(ConfigError, match="missing.toml"):
        load_config(tmp_path / "missing.toml")


def test_dump_config_round_trips_resolved_data():
    config = {
        "pipeline": ["denoise-one", "upscale"],
        "settings": {"quiet": True, "mlx_cache_limit_gb": 1.5},
        "input": {"video": "a clip.mp4", "snap_start": False},
        "denoise-one": {
            "processor": "bsvd", "strength": 0.3,
            "tuning": {"passes": 2, "enabled": True},
        },
        "upscale": {"processor": "safmn", "scale": 4},
    }
    assert tomllib.loads(dump_config(config)) == config


# ---- typed --set ----------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("true", True),
    ("false", False),
    ("512", 512),
    ("-3", -3),
    ("0.35", 0.35),
    ("half", "half"),                      # bare word -> string
    ('"512"', "512"),                      # quoted numeric stays a string
    ("[1, 2, 3]", [1, 2, 3]),
    ('{ mode = "auto", k = 2 }', {"mode": "auto", "k": 2}),
    ("no", "no"),                          # TOML has no yes/no keywords
    ("1979-05-27", "1979-05-27"),          # bare local-date parses as TOML date;
])
def test_set_value_typing(raw, expected):
    path, value = parse_set_argument(f"stage.key={raw}")
    if raw == "1979-05-27":
        # tomllib parses this as datetime.date; assert the type is TOML-shaped
        import datetime
        assert value == datetime.date(1979, 5, 27)
    else:
        assert value == expected
    assert path == ["stage", "key"]


def test_set_value_may_contain_equals():
    _, value = parse_set_argument('stage.expr=a=b')
    assert value == "a=b"


def test_set_nested_key_path():
    cfg = apply_set_overrides({}, ["deblock.tuning.passes=2"])
    assert cfg == {"deblock": {"tuning": {"passes": 2}}}


def test_set_overrides_win_in_order():
    cfg = apply_set_overrides(
        {"denoise": {"strength": 0.03}},
        ["denoise.strength=0.5", "denoise.strength=0.35"])
    assert cfg["denoise"]["strength"] == 0.35


def test_set_does_not_mutate_input_config():
    original = {"denoise": {"strength": 0.03}}
    apply_set_overrides(original, ["denoise.strength=0.5"])
    assert original == {"denoise": {"strength": 0.03}}


def test_set_no_template_expansion(monkeypatch):
    monkeypatch.setenv("SHARED_TEMP_DIR", "/scratch")
    _, value = parse_set_argument("stage.path={{SHARED_TEMP_DIR}}/x")
    assert value == "{{SHARED_TEMP_DIR}}/x"


@pytest.mark.parametrize("bad", [
    "novalue",            # no '='
    "=5",                 # empty path
    "stage=5",            # no key under the table
    "stage..k=5",         # empty path segment
])
def test_set_malformed_arguments_error(bad):
    with pytest.raises(ConfigError, match="--set"):
        apply_set_overrides({}, [bad])


def test_set_through_scalar_errors():
    with pytest.raises(ConfigError, match="not a table"):
        apply_set_overrides({"stage": {"k": 5}}, ["stage.k.deep=1"])


# ---- validation --------------------------------------------------------------------


def test_validate_missing_stage_table_names_it():
    with pytest.raises(ConfigError, match=r"\[denose\]|'denose'"):
        validate_config({"pipeline": ["denose"]})


def test_validate_reserved_table_listed_as_stage():
    with pytest.raises(ConfigError, match="reserved"):
        validate_config({"pipeline": ["output"], "output": {}})


def test_validate_processor_required():
    with pytest.raises(ConfigError, match="processor"):
        validate_config({"pipeline": ["s"], "s": {"strength": 1}})


def test_validate_top_level_scalar_rejected():
    with pytest.raises(ConfigError, match="quiet"):
        validate_config({"quiet": True})


def test_validate_unknown_top_level_table_rejected():
    with pytest.raises(ConfigError, match="unknown top-level table.*setings"):
        validate_config({"setings": {"quiet": True}})


def test_validate_unlisted_stage_library_table_is_legal():
    validate_config({"unused": {"processor": "bsvd", "strength": 0.3}})


def test_validate_flag_path_may_defer_partial_stage_table():
    validate_config(
        {"denoise_bsvd": {"strength": 0.3}},
        allow_stage_fragments=True,
    )


def test_validate_pipeline_must_be_string_list():
    with pytest.raises(ConfigError, match="pipeline"):
        validate_config({"pipeline": "denoise"})
    with pytest.raises(ConfigError, match="pipeline"):
        validate_config({"pipeline": [1]})


def test_validate_duplicate_stage_names_are_legal():
    validate_config({
        "pipeline": ["deblock", "denoise", "deblock"],
        "deblock": {"processor": "fbcnn"},
        "denoise": {"processor": "bsvd"},
    })


def test_validate_selector_types():
    with pytest.raises(ConfigError, match="capability"):
        validate_config({
            "pipeline": ["s"], "s": {"processor": "toflow", "capability": 3}})


def test_split_stage_table_reserved_vs_family_keys():
    selector, settings = split_stage_table({
        "processor": "toflow", "capability": "deblock", "profile": "deblock",
        "passes": 1, "flow_scale": "half"})
    assert selector == {
        "processor": "toflow", "capability": "deblock", "profile": "deblock"}
    assert settings == {"passes": 1, "flow_scale": "half"}
