"""Media color-resolution tests."""

def test_source_range_resolve_override():
    from kinovsr.media import color

    src = {"primaries": None, "transfer": None, "matrix": None,
           "full_range": False, "tagged": False}
    # auto trusts the container flag
    assert color.resolve(src, "auto", "auto")[3] is False
    assert color.resolve(dict(src, full_range=True), "auto", "auto")[3] is True
    # forcing overrides the flag in both directions
    assert color.resolve(src, "auto", "full")[3] is True
    assert color.resolve(dict(src, full_range=True), "auto", "video")[3] is False
    # range override composes with a colorimetry override
    resolved = color.resolve(src, "bt601", "full")
    assert resolved[3] is True
    assert "range=full" in color.describe(resolved)
