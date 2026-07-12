"""Output-prefix sanitization for file-endpoint stems."""
from __future__ import annotations

import pytest

from kinovsr.media.naming import sanitize_output_prefix


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
