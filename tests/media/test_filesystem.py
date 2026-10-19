"""Atomic filesystem primitives used by artifact publication."""

import errno

import pytest

from kinovsr.media.filesystem import create_partial_output, rename_exclusive

pytestmark = pytest.mark.unit


def test_exclusive_rename_preserves_existing_destination(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"new")
    destination.write_bytes(b"external")

    with pytest.raises(FileExistsError) as caught:
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


def test_exclusive_rename_failure_names_both_paths(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"new")
    destination.write_bytes(b"external")

    with pytest.raises(FileExistsError) as caught:
        rename_exclusive(source, destination)

    message = str(caught.value)
    assert str(source) in message
    assert str(destination) in message


def test_partial_output_is_visible_and_never_overwrites(tmp_path):
    destination = tmp_path / "vsr_clip.mp4"
    first = create_partial_output(destination)
    first.write_bytes(b"an earlier run's frames")
    second = create_partial_output(destination)

    assert first == tmp_path / "vsr_clip.partial.mp4"
    assert second == tmp_path / "vsr_clip.partial-2.mp4"
    assert first.read_bytes() == b"an earlier run's frames"
    assert not destination.exists()


def test_partial_output_reports_when_every_name_is_taken(tmp_path):
    destination = tmp_path / "vsr_clip.mp4"
    create_partial_output(destination, attempts=1)
    with pytest.raises(FileExistsError):
        create_partial_output(destination, attempts=1)
