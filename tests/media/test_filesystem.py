"""Atomic filesystem primitives used by artifact publication."""

from __future__ import annotations

import errno

import pytest

from kinovsr.media.filesystem import rename_exclusive

pytestmark = pytest.mark.unit


def test_exclusive_rename_preserves_existing_destination(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"new")
    destination.write_bytes(b"external")

    with pytest.raises(OSError) as caught:
        rename_exclusive(source, destination)

    assert caught.value.errno == errno.EEXIST
    assert source.read_bytes() == b"new"
    assert destination.read_bytes() == b"external"


def test_exclusive_rename_moves_file_and_directory(tmp_path):
    source_file = tmp_path / "source-file"
    destination_file = tmp_path / "destination-file"
    source_file.write_bytes(b"file")
    rename_exclusive(source_file, destination_file)
    assert destination_file.read_bytes() == b"file"

    source_directory = tmp_path / "source-directory"
    destination_directory = tmp_path / "destination-directory"
    source_directory.mkdir()
    (source_directory / "marker").write_bytes(b"directory")
    rename_exclusive(source_directory, destination_directory)
    assert (destination_directory / "marker").read_bytes() == b"directory"
