"""Artifact-graph preflight, reservation, and publication rollback."""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from kinovsr.pipeline.run import (
    _Artifact,
    _ArtifactPlan,
    _ComparisonTee,
    _OutputTransaction,
)
from kinovsr.processors.errors import MediaError
from kinovsr.processors.specs import (
    Geometry,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


def _build_plan(
    tmp_path: Path,
    *,
    comparison: Path | None = None,
    cut_log: Path | None = None,
    save_pre_frames: Path | None = None,
    save_post_frames: Path | None = None,
    overwrite: bool = False,
) -> _ArtifactPlan:
    source = tmp_path / "input.mp4"
    if not source.exists():
        source.write_bytes(b"source")
    return _ArtifactPlan.build(
        video=source,
        output=tmp_path / "post.mp4",
        comparison=comparison,
        cut_log=cut_log,
        save_audio_sidecar=False,
        save_pre_frames=save_pre_frames,
        save_post_frames=save_post_frames,
        skip_post_mp4=False,
        noise_map_debug=False,
        overwrite=overwrite,
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(shared_temp_dir=tmp_path / "scratch")


def _commit_with_second_failure(
    plan: _ArtifactPlan,
    settings: Settings,
    failure: type[BaseException],
    *,
    post_bytes: bytes,
    comparison_bytes: bytes,
) -> None:
    with _OutputTransaction(plan, settings) as transaction:
        post_temp = transaction.temp_file("post output")
        comparison_temp = transaction.temp_file("comparison output")
        post_temp.write_bytes(post_bytes)
        comparison_temp.write_bytes(comparison_bytes)

        def replace(source, destination):
            if source == comparison_temp:
                raise failure("injected second publication failure")
            source.replace(destination)

        transaction._replace = replace
        transaction.commit()


def test_every_pairwise_file_alias_is_rejected(tmp_path):
    labels = [
        "post output",
        "comparison output",
        "audio sidecar",
        "cut log",
        "noisemap debug image",
        "blockmap debug image",
    ]
    input_path = (tmp_path / "input.mp4").resolve()
    input_path.write_bytes(b"source")

    for left_index in range(len(labels) + 1):
        for right_index in range(left_index + 1, len(labels) + 1):
            paths = [
                (tmp_path / f"artifact-{index}").resolve()
                for index in range(len(labels))
            ]
            alias = input_path if left_index == 0 else paths[left_index - 1]
            paths[right_index - 1] = alias
            plan = _ArtifactPlan(
                input_path=input_path,
                output_path=paths[0],
                artifacts=tuple(
                    _Artifact(label, path)
                    for label, path in zip(labels, paths, strict=True)
                ),
                overwrite=False,
            )
            with pytest.raises(MediaError, match="artifact paths alias"):
                plan.validate()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_filesystem_aliases_preserve_source_bytes(tmp_path, link_kind):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"irreplaceable")
    output = tmp_path / "different-name.mp4"
    if link_kind == "symlink":
        output.symlink_to(source)
    else:
        os.link(source, output)

    with pytest.raises(MediaError, match="destroy the source"):
        _ArtifactPlan.build(
            video=source,
            output=output,
            comparison=None,
            cut_log=None,
            save_audio_sidecar=False,
            save_pre_frames=None,
            save_post_frames=None,
            skip_post_mp4=False,
            noise_map_debug=False,
            overwrite=True,
        )
    assert source.read_bytes() == b"irreplaceable"


def test_casefolded_uncreated_destinations_are_treated_as_aliases(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    with pytest.raises(MediaError, match="artifact paths alias"):
        _ArtifactPlan.build(
            video=source,
            output=tmp_path / "Result.mp4",
            comparison=tmp_path / "result.mp4",
            cut_log=None,
            save_audio_sidecar=False,
            save_pre_frames=None,
            save_post_frames=None,
            skip_post_mp4=False,
            noise_map_debug=False,
            overwrite=False,
        )


def test_invalid_parent_fails_before_any_artifact_mutation(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_bytes(b"sentinel")

    with pytest.raises(MediaError, match="parent is not a directory"):
        _ArtifactPlan.build(
            video=source,
            output=tmp_path / "post.mp4",
            comparison=None,
            cut_log=None,
            save_audio_sidecar=False,
            save_pre_frames=None,
            save_post_frames=not_a_directory / "frames",
            skip_post_mp4=False,
            noise_map_debug=False,
            overwrite=False,
        )
    assert not list(tmp_path.glob(".*.partial"))
    assert not_a_directory.read_bytes() == b"sentinel"


def test_existing_destination_requires_explicit_overwrite(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "post.mp4"
    output.write_bytes(b"existing")

    kwargs = {
        "video": source,
        "output": output,
        "comparison": None,
        "cut_log": None,
        "save_audio_sidecar": False,
        "save_pre_frames": None,
        "save_post_frames": None,
        "skip_post_mp4": False,
        "noise_map_debug": False,
    }
    with pytest.raises(MediaError, match="overwrite=True"):
        _ArtifactPlan.build(**kwargs, overwrite=False)
    assert output.read_bytes() == b"existing"
    assert _ArtifactPlan.build(**kwargs, overwrite=True).overwrite


def test_complete_graph_is_reserved_until_transaction_exit(tmp_path):
    plan = _build_plan(tmp_path, comparison=tmp_path / "comparison.mp4")
    settings = _settings(tmp_path)

    with (
        _OutputTransaction(plan, settings),
        pytest.raises(MediaError, match="already reserved"),
        _OutputTransaction(plan, settings),
    ):
        pass
    with _OutputTransaction(plan, settings):
        pass


def test_parent_child_destinations_conflict_across_transactions(tmp_path):
    bundle = tmp_path / "bundle"
    parent_plan = _build_plan(tmp_path, save_pre_frames=bundle)
    child_source = tmp_path / "child-input.mp4"
    child_source.write_bytes(b"source")
    child_plan = _ArtifactPlan.build(
        video=child_source,
        output=bundle / "child.mp4",
        comparison=None,
        cut_log=None,
        save_audio_sidecar=False,
        save_pre_frames=None,
        save_post_frames=None,
        skip_post_mp4=False,
        noise_map_debug=False,
        overwrite=False,
    )
    settings = _settings(tmp_path)

    with (
        _OutputTransaction(parent_plan, settings),
        pytest.raises(MediaError, match="hierarchy is already reserved"),
        _OutputTransaction(child_plan, settings),
    ):
        pass


def test_comparison_setup_failure_discards_unregistered_sink(monkeypatch, tmp_path):
    import mlx.core as mx

    from kinovsr.pipeline import run as run_module

    instances = []

    class _Sink:
        def __init__(self, *args, **kwargs):
            self.discarded = False
            instances.append(self)

        def discard(self):
            self.discarded = True

    spec = StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(16, 16)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)),
    )
    source = SimpleNamespace(spec=spec)
    monkeypatch.setattr(run_module, "FileSink", _Sink)
    monkeypatch.setattr(
        mx, "array", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected comparison setup failure")))

    with pytest.raises(MediaError, match="injected comparison") as caught:
        _ComparisonTee(tmp_path / "comparison.mp4", spec, source, quality=0.5)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert len(instances) == 1
    assert instances[0].discarded


@pytest.mark.parametrize("failure_type", [TypeError, AssertionError])
def test_comparison_setup_preserves_programmer_failure(
        monkeypatch, tmp_path, failure_type):
    import mlx.core as mx

    from kinovsr.pipeline import run as run_module

    class _Sink:
        def __init__(self, *args, **kwargs):
            self.discarded = False

        def discard(self):
            self.discarded = True

    spec = StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(16, 16)),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25)),
    )
    source = SimpleNamespace(spec=spec)
    failure = failure_type("injected programmer failure")
    monkeypatch.setattr(run_module, "FileSink", _Sink)
    monkeypatch.setattr(
        mx, "array", lambda *args, **kwargs: (_ for _ in ()).throw(failure))

    with pytest.raises(failure_type) as caught:
        _ComparisonTee(tmp_path / "comparison.mp4", spec, source, quality=0.5)
    assert caught.value is failure


def test_commit_publishes_files_and_directories_together(tmp_path):
    plan = _build_plan(
        tmp_path,
        comparison=tmp_path / "comparison.mp4",
        cut_log=tmp_path / "cuts.txt",
        save_pre_frames=tmp_path / "pre",
        save_post_frames=tmp_path / "post-frames",
    )
    with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
        transaction.temp_file("post output").write_bytes(b"post")
        transaction.temp_file("comparison output").write_bytes(b"comparison")
        transaction.temp_file("cut log").write_text("8\n", encoding="utf-8")
        (transaction.temp_directory("pre-frame directory") / "frame.png").write_bytes(
            b"pre")
        (transaction.temp_directory("post-frame directory") / "frame.png").write_bytes(
            b"post-frame")
        transaction.commit()

    assert plan.path("post output").read_bytes() == b"post"
    assert plan.path("comparison output").read_bytes() == b"comparison"
    assert plan.path("cut log").read_text(encoding="utf-8") == "8\n"
    assert (plan.path("pre-frame directory") / "frame.png").read_bytes() == b"pre"
    assert (plan.path("post-frame directory") / "frame.png").read_bytes() == b"post-frame"
    assert not list(tmp_path.glob(".*.partial"))


def test_commit_seals_materializer_replaced_temporary_identity(tmp_path):
    plan = _build_plan(tmp_path)

    with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
        temp = transaction.temp_file("post output")
        original_inode = temp.stat().st_ino
        replacement = tmp_path / "materializer-output"
        replacement.write_bytes(b"complete")
        replacement.replace(temp)
        assert temp.stat().st_ino != original_inode
        transaction.commit()

    assert plan.path("post output").read_bytes() == b"complete"
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize(
    ("failure", "expected"),
    [(OSError, MediaError), (KeyboardInterrupt, KeyboardInterrupt)],
)
def test_second_publish_failure_rolls_back_new_singleton(
        tmp_path, failure, expected):
    comparison = tmp_path / "comparison.mp4"
    plan = _build_plan(tmp_path, comparison=comparison)

    with pytest.raises(expected):
        _commit_with_second_failure(
            plan,
            _settings(tmp_path),
            failure,
            post_bytes=b"post",
            comparison_bytes=b"comparison",
        )

    assert not plan.path("post output").exists()
    assert not comparison.exists()
    assert not list(tmp_path.glob(".*.partial"))



def test_overwrite_rollback_restores_every_existing_destination(tmp_path):
    post = tmp_path / "post.mp4"
    comparison = tmp_path / "comparison.mp4"
    post.write_bytes(b"old-post")
    comparison.write_bytes(b"old-comparison")
    plan = _build_plan(tmp_path, comparison=comparison, overwrite=True)

    with pytest.raises(MediaError, match="injected"):
        _commit_with_second_failure(
            plan,
            _settings(tmp_path),
            OSError,
            post_bytes=b"new-post",
            comparison_bytes=b"new-comparison",
        )

    assert post.read_bytes() == b"old-post"
    assert comparison.read_bytes() == b"old-comparison"
    assert not list(tmp_path.glob(".*.partial"))
    assert not list(tmp_path.glob(".*.rollback"))


def test_directory_publish_failure_restores_existing_directory(tmp_path):
    post = tmp_path / "post.mp4"
    pre = tmp_path / "pre"
    post.write_bytes(b"old-post")
    pre.mkdir()
    (pre / "old.png").write_bytes(b"old-frame")
    plan = _build_plan(tmp_path, save_pre_frames=pre, overwrite=True)

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            transaction.temp_file("post output").write_bytes(b"new-post")
            pre_temp = transaction.temp_directory("pre-frame directory")
            (pre_temp / "new.png").write_bytes(b"new-frame")

            def replace(source, destination):
                if source == pre_temp:
                    raise OSError("injected directory publication failure")
                source.replace(destination)

            transaction._replace = replace
            transaction.commit()

    with pytest.raises(MediaError, match="directory publication failure"):
        commit()

    assert post.read_bytes() == b"old-post"
    assert (pre / "old.png").read_bytes() == b"old-frame"
    assert not (pre / "new.png").exists()
    assert not list(tmp_path.glob(".*.partial"))
    assert not list(tmp_path.glob(".*.rollback"))


def test_interrupt_after_publish_rename_removes_landed_singleton(tmp_path):
    plan = _build_plan(tmp_path)

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            temp = transaction.temp_file("post output")
            temp.write_bytes(b"new")

            def replace_then_interrupt(source, destination):
                source.replace(destination)
                raise KeyboardInterrupt("after rename")

            transaction._replace = replace_then_interrupt
            transaction.commit()

    with pytest.raises(KeyboardInterrupt):
        commit()

    assert not plan.path("post output").exists()
    assert not list(tmp_path.glob(".*.partial"))


def test_interrupt_after_backup_rename_restores_original(tmp_path):
    post = tmp_path / "post.mp4"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            temp = transaction.temp_file("post output")
            temp.write_bytes(b"new")

            def replace_then_interrupt(source, destination):
                source.replace(destination)
                if source == post:
                    raise KeyboardInterrupt("after backup rename")

            transaction._replace = replace_then_interrupt
            transaction.commit()

    with pytest.raises(KeyboardInterrupt):
        commit()

    assert post.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.partial"))
    assert not list(tmp_path.glob(".*.rollback"))


def test_failed_publication_does_not_remove_external_destination(tmp_path):
    post = tmp_path / "post.mp4"
    marker = post / "external.txt"
    plan = _build_plan(tmp_path)

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            temp = transaction.temp_file("post output")
            temp.write_bytes(b"new")

            def external_destination_then_fail(source, destination):
                destination.mkdir()
                marker.write_bytes(b"external")
                source.replace(destination)

            transaction._replace = external_destination_then_fail
            transaction.commit()

    with pytest.raises(MediaError):
        commit()

    assert marker.read_bytes() == b"external"
    assert not list(tmp_path.glob(".*.partial"))


def test_exclusive_publication_preserves_external_regular_file(tmp_path):
    post = tmp_path / "post.mp4"
    plan = _build_plan(tmp_path)

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            temp = transaction.temp_file("post output")
            temp.write_bytes(b"new")
            original_replace = transaction._replace

            def external_file_then_publish(source, destination):
                if source == temp:
                    destination.write_bytes(b"external")
                original_replace(source, destination)

            transaction._replace = external_file_then_publish
            transaction.commit()

    with pytest.raises(MediaError):
        commit()

    assert post.read_bytes() == b"external"
    assert not list(tmp_path.glob(".*.partial"))









def test_failed_backup_restore_retains_original_and_diagnostic(tmp_path):
    post = tmp_path / "post.mp4"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)
    failure = KeyboardInterrupt("backup returned after rename")
    restore_failure = OSError("injected backup restore failure")

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            transaction.temp_file("post output").write_bytes(b"new")

            def replace(source, destination):
                if source == post:
                    # Backup rename lands, then the interrupt arrives - the
                    # coarse Ctrl-C-at-a-phase-boundary shape.
                    source.replace(destination)
                    raise failure
                source.replace(destination)

            def failing_restore(backup, destination):
                raise restore_failure

            transaction._replace = replace
            transaction._restore_replace = failing_restore
            transaction.commit()

    with pytest.raises(KeyboardInterrupt) as caught:
        commit()

    assert caught.value is failure
    assert not post.exists()
    backups = list(tmp_path.glob(".*.rollback"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"old"
    assert any(
        "injected backup restore failure" in note
        for note in getattr(failure, "__notes__", ())
    )


def test_restore_that_completes_then_raises_is_not_retried(tmp_path):
    post = tmp_path / "post.mp4"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)
    failure = TypeError("injected publication defect")
    restore_failure = OSError("restore returned after rename")

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            temp = transaction.temp_file("post output")
            temp.write_bytes(b"new")

            def replace(source, destination):
                if source == temp:
                    raise failure
                source.replace(destination)

            def restore_then_raise(backup, destination):
                backup.replace(destination)
                raise restore_failure

            transaction._replace = replace
            transaction._restore_replace = restore_then_raise
            transaction.commit()

    with pytest.raises(TypeError) as caught:
        commit()

    assert caught.value is failure
    assert post.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.rollback"))
    assert any(
        "restore returned after rename" in note
        for note in getattr(failure, "__notes__", ())
    )


@pytest.mark.parametrize("directory", [False, True])
def test_backup_slot_path_conversion_failure_cleans_slot(
        monkeypatch, tmp_path, directory):
    from kinovsr.pipeline import run as run_module

    artifact = _Artifact("output", tmp_path / "output", directory=directory)
    failure = SystemExit("injected rollback Path conversion failure")

    def fail_path(raw):
        raise failure

    monkeypatch.setattr(run_module, "Path", fail_path)
    with pytest.raises(SystemExit) as caught:
        _OutputTransaction._backup_slot(artifact)

    assert caught.value is failure
    assert not list(tmp_path.glob(".*.rollback"))


def test_backup_slot_close_failure_cleans_path_without_retry(
        monkeypatch, tmp_path):
    from kinovsr.pipeline import run as run_module

    artifact = _Artifact("output", tmp_path / "output")
    failure = KeyboardInterrupt("injected rollback close interruption")
    acquired = []

    def fail_close(fd):
        acquired.append(fd)
        raise failure

    monkeypatch.setattr(run_module, "_close_created_fd", fail_close)
    with pytest.raises(KeyboardInterrupt) as caught:
        _OutputTransaction._backup_slot(artifact)

    assert caught.value is failure
    assert len(acquired) == 1
    assert not list(tmp_path.glob(".*.rollback"))
    os.close(acquired[0])


def test_backup_file_slot_failure_preserves_replacement(monkeypatch, tmp_path):
    from kinovsr.pipeline import run as run_module

    artifact = _Artifact("output", tmp_path / "output")
    failure = TypeError("injected rollback unlink defect")
    original_unlink = run_module.os.unlink

    def unlink_then_replace(raw):
        original_unlink(raw)
        Path(raw).write_bytes(b"external")
        raise failure

    monkeypatch.setattr(run_module.os, "unlink", unlink_then_replace)
    with pytest.raises(TypeError) as caught:
        _OutputTransaction._backup_slot(artifact)

    assert caught.value is failure
    backups = list(tmp_path.glob(".*.rollback"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"external"


def test_backup_directory_slot_failure_preserves_replacement(
        monkeypatch, tmp_path):
    from kinovsr.pipeline import run as run_module

    artifact = _Artifact("output", tmp_path / "output", directory=True)
    failure = TypeError("injected rollback rmdir defect")
    original_rmdir = run_module.os.rmdir

    def rmdir_then_replace(raw):
        original_rmdir(raw)
        Path(raw).mkdir()
        (Path(raw) / "external").write_bytes(b"external")
        raise failure

    monkeypatch.setattr(run_module.os, "rmdir", rmdir_then_replace)
    with pytest.raises(TypeError) as caught:
        _OutputTransaction._backup_slot(artifact)

    assert caught.value is failure
    backups = list(tmp_path.glob(".*.rollback"))
    assert len(backups) == 1
    assert (backups[0] / "external").read_bytes() == b"external"


def test_interrupt_after_backup_removal_does_not_delete_replacement(tmp_path):
    post = tmp_path / "post.mp4"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)
    failure = KeyboardInterrupt("during backup cleanup")
    replacement = b"external replacement"

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            transaction.temp_file("post output").write_bytes(b"new")
            original_remove = transaction._remove
            interrupted = False

            def interrupt_once(path):
                nonlocal interrupted
                if path.suffix == ".rollback" and not interrupted:
                    interrupted = True
                    original_remove(path)
                    path.write_bytes(replacement)
                    raise failure
                original_remove(path)

            transaction._remove = interrupt_once
            transaction.commit()

    with pytest.raises(KeyboardInterrupt) as caught:
        commit()

    assert caught.value is failure
    assert post.read_bytes() == b"new"
    backups = list(tmp_path.glob(".*.rollback"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == replacement


def test_operational_backup_cleanup_failure_is_not_retried(tmp_path):
    post = tmp_path / "post.mp4"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)
    calls = []

    with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
        transaction.temp_file("post output").write_bytes(b"new")
        original_remove = transaction._remove

        def fail_backup_remove(path):
            if path.suffix == ".rollback":
                calls.append(path)
                raise OSError("injected backup cleanup failure")
            original_remove(path)

        transaction._remove = fail_backup_remove
        transaction.commit()

    assert post.read_bytes() == b"new"
    assert len(calls) == 1
    backups = list(tmp_path.glob(".*.rollback"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"old"


@pytest.mark.parametrize("failure_type", [TypeError, AssertionError])
def test_programmer_failure_during_backup_cleanup_retains_backup(
        tmp_path, failure_type):
    post = tmp_path / "post.mp4"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)
    failure = failure_type("injected backup cleanup defect")

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            transaction.temp_file("post output").write_bytes(b"new")
            original_remove = transaction._remove
            failed = False

            def fail_once(path):
                nonlocal failed
                if path.suffix == ".rollback" and not failed:
                    failed = True
                    raise failure
                original_remove(path)

            transaction._remove = fail_once
            transaction.commit()

    with pytest.raises(failure_type) as caught:
        commit()
    assert caught.value is failure
    assert post.read_bytes() == b"new"
    backups = list(tmp_path.glob(".*.rollback"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"old"


@pytest.mark.parametrize(
    "failure_type", [TypeError, AssertionError, KeyboardInterrupt, SystemExit],
)
def test_rollback_cleanup_failure_does_not_replace_primary(
        tmp_path, failure_type):
    plan = _build_plan(tmp_path)
    settings = _settings(tmp_path)
    failure = failure_type("injected publication defect")
    cleanup_failure = OSError("injected rollback cleanup failure")
    cleanup_calls = 0

    def commit():
        with _OutputTransaction(plan, settings) as transaction:
            temp = transaction.temp_file("post output")
            temp.write_bytes(b"new")
            original_remove = transaction._remove

            def fail_publication(source, destination):
                raise failure

            def fail_cleanup(path):
                nonlocal cleanup_calls
                if path == temp:
                    cleanup_calls += 1
                    raise cleanup_failure
                original_remove(path)

            transaction._replace = fail_publication
            transaction._remove = fail_cleanup
            transaction.commit()

    with pytest.raises(failure_type) as caught:
        commit()
    assert caught.value is failure
    assert any(
        "artifact rollback also failed" in note
        for note in getattr(failure, "__notes__", ())
    )
    # One bounded attempt per rollback pass: the commit-failure rollback and
    # the exit discard.  No within-pass retry loops.
    assert cleanup_calls == 2
    partials = list(tmp_path.glob(".*.partial"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == b"new"


def test_directory_cleanup_failure_keeps_primary_and_residue(tmp_path):
    frames = tmp_path / "frames"
    plan = _build_plan(tmp_path, save_pre_frames=frames)
    failure = TypeError("injected publication defect")
    cleanup_failure = OSError("injected directory cleanup failure")
    cleanup_calls = 0

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            temp = transaction.temp_directory("pre-frame directory")
            (temp / "owned").write_bytes(b"owned")

            def fail_publication(source, destination):
                raise failure

            def fail_directory_cleanup(path):
                nonlocal cleanup_calls
                assert path == temp
                cleanup_calls += 1
                raise cleanup_failure

            transaction._replace = fail_publication
            transaction._remove = fail_directory_cleanup
            transaction.commit()

    with pytest.raises(TypeError) as caught:
        commit()

    assert caught.value is failure
    assert cleanup_calls == 2
    partials = list(tmp_path.glob(".*.partial"))
    assert len(partials) == 1
    assert (partials[0] / "owned").read_bytes() == b"owned"


def test_reservation_uses_one_account_shared_lock_file(tmp_path):
    plan = _build_plan(tmp_path)
    settings = _settings(tmp_path)

    with _OutputTransaction(plan, settings):
        scratch = settings.shared_temp_dir.expanduser().resolve()
        lock_file = scratch / "kinovsr-namespaces.lock"
        assert lock_file.exists()
        assert (lock_file.stat().st_mode & 0o666) == 0o666
        # No per-namespace lock-file litter: the byte-range design keeps
        # exactly one rendezvous file regardless of how many paths ran.
        assert [p.name for p in scratch.iterdir()] == [lock_file.name]


def test_reservation_excludes_other_processes(tmp_path):
    import subprocess
    import sys
    import textwrap

    plan = _build_plan(tmp_path)
    settings = _settings(tmp_path)
    probe = textwrap.dedent("""
        import sys
        from pathlib import Path
        from kinovsr.pipeline.run import _ArtifactPlan, _OutputTransaction
        from kinovsr.processors.errors import MediaError
        from kinovsr.settings import Settings

        tmp = Path(sys.argv[1])
        plan = _ArtifactPlan.build(
            video=tmp / "input.mp4", output=tmp / "post.mp4",
            comparison=None, cut_log=None, save_audio_sidecar=False,
            save_pre_frames=None, save_post_frames=None,
            skip_post_mp4=False, noise_map_debug=False, overwrite=False)
        settings = Settings(shared_temp_dir=tmp / "scratch")
        try:
            with _OutputTransaction(plan, settings):
                pass
        except MediaError as exc:
            assert "already reserved" in str(exc), str(exc)
            print("REFUSED")
        else:
            print("ACQUIRED")
    """)

    with _OutputTransaction(plan, settings):
        result = subprocess.run(
            [sys.executable, "-c", probe, str(tmp_path)],
            capture_output=True, text=True, timeout=60, check=True)
    assert result.stdout.strip() == "REFUSED", result.stderr
    # After release the same probe must succeed.
    result = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path)],
        capture_output=True, text=True, timeout=60, check=True)
    assert result.stdout.strip() == "ACQUIRED", result.stderr


def test_legacy_private_lock_file_is_reset(tmp_path):
    from kinovsr.pipeline.run import _ArtifactReservation

    lock = tmp_path / "legacy.lock"
    lock.touch()
    lock.chmod(0)

    fd = _ArtifactReservation._open_shared_lock(lock)
    try:
        assert (os.fstat(fd).st_mode & 0o666) == 0o666
    finally:
        os.close(fd)


def test_transaction_temps_honor_process_umask(tmp_path):
    from kinovsr.pipeline.run import _UMASK

    plan = _build_plan(
        tmp_path, save_pre_frames=tmp_path / "frames")
    settings = _settings(tmp_path)

    with _OutputTransaction(plan, settings) as transaction:
        temp_file = transaction.temp_file("post output")
        temp_dir = transaction.temp_directory("pre-frame directory")
        assert (temp_file.stat().st_mode & 0o777) == (0o666 & ~_UMASK)
        assert (temp_dir.stat().st_mode & 0o777) == (0o777 & ~_UMASK)
        transaction.discard()
