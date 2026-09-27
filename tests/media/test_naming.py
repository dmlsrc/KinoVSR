"""Output-prefix sanitization for file-endpoint stems."""

import pytest

from kinovsr.media.naming import default_output_prefix, sanitize_output_prefix


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "kinovsr"),
        ("", "kinovsr"),
        ("   ", "kinovsr"),
        ("clip", "clip"),
        ("My Clip 01", "My_Clip_01"),
        ("a/b\\c:d", "a_b_c_d"),
        ("keep-under_score.ok", "keep-under_score.ok"),
        (".__leading", "leading"),
        ("trailing__.", "trailing"),
        ("...", "kinovsr"),
        ("///", "kinovsr"),
    ],
)
def test_sanitize_output_prefix(raw, expected):
    assert sanitize_output_prefix(raw) == expected


@pytest.mark.parametrize(
    ("video", "expected"),
    [
        ("clip.mp4", "vsr_clip"),
        ("/media/My Clip.mov", "vsr_My_Clip"),
        ("/media/part.one.mkv", "vsr_part.one"),
    ],
)
def test_default_output_prefix_uses_source_stem(video, expected):
    assert default_output_prefix(video) == expected
