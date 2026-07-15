"""Encode-loop failure must discard partial output, not finalize it."""

from __future__ import annotations

import pytest

from kinovsr.native.encode import _discard_failed_output

pytestmark = pytest.mark.unit


class _RecordingWriter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def cancel(self) -> None:
        self.calls.append("cancel")


class _WedgedWriter:
    def cancel(self) -> None:
        raise RuntimeError("native cancel wedged")


def test_discard_cancels_writer_and_removes_partial(tmp_path):
    writer = _RecordingWriter()
    partial = tmp_path / "out.mp4"
    partial.write_bytes(b"truncated")

    _discard_failed_output(writer, partial)

    assert writer.calls == ["cancel"]
    assert not partial.exists()


def test_discard_survives_cancel_failure_and_still_removes(tmp_path, caplog):
    partial = tmp_path / "out.mp4"
    partial.write_bytes(b"truncated")

    _discard_failed_output(_WedgedWriter(), partial)

    assert not partial.exists()


def test_discard_tolerates_missing_partial(tmp_path):
    writer = _RecordingWriter()

    _discard_failed_output(writer, tmp_path / "never-created.mp4")

    assert writer.calls == ["cancel"]
