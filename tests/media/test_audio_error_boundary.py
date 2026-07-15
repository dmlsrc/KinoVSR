"""Audio fallback ignores media failures, not programming defects."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from kinovsr.media.audio import read_audio_track_from_video

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failure_type", [TypeError, ValueError, AssertionError])
def test_bounded_audio_reader_preserves_programmer_failure(failure_type):
    failure = failure_type("injected audio reader defect")

    class _Reader:
        @staticmethod
        def read_audio_track_window(*args, **kwargs):
            raise failure

    with pytest.raises(failure_type) as caught:
        read_audio_track_from_video(
            Path("source.mov"), _Reader(),
            start_sec=Fraction(0), end_sec=Fraction(1),
        )
    assert caught.value is failure


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError])
def test_bounded_audio_reader_treats_media_failure_as_no_audio(failure_type):
    class _Reader:
        @staticmethod
        def read_audio_track_window(*args, **kwargs):
            raise failure_type("unusable audio")

    assert read_audio_track_from_video(
        Path("source.mov"), _Reader(),
        start_sec=Fraction(0), end_sec=Fraction(1),
    ) is None
