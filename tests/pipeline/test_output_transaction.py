"""Artifact-graph preflight, reservation, and publication rollback."""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from kinovsr.pipeline.run import (
    FileSink,
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


def test_rollback_quarantine_preserves_replaced_publication(tmp_path):
    post = tmp_path / "post.mp4"
    comparison = tmp_path / "comparison.mp4"
    plan = _build_plan(tmp_path, comparison=comparison)
    failure = TypeError("injected second publication defect")

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            post_temp = transaction.temp_file("post output")
            comparison_temp = transaction.temp_file("comparison output")
            post_temp.write_bytes(b"post")
            comparison_temp.write_bytes(b"comparison")
            original_replace = transaction._replace
            original_quarantine = transaction._quarantine_and_remove

            def fail_second(source, destination):
                if source == comparison_temp:
                    raise failure
                original_replace(source, destination)

            def replace_before_quarantine(path, *args, **kwargs):
                if path == post:
                    path.unlink()
                    path.write_bytes(b"external")
                return original_quarantine(path, *args, **kwargs)

            transaction._replace = fail_second
            transaction._quarantine_and_remove = replace_before_quarantine
            transaction.commit()

    with pytest.raises(TypeError) as caught:
        commit()

    assert caught.value is failure
    assert post.read_bytes() == b"external"
    assert not comparison.exists()


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


def test_successful_rename_rejects_replaced_temporary_identity(tmp_path):
    post = tmp_path / "post.mp4"
    stash = tmp_path / "owned-stash"
    plan = _build_plan(tmp_path)

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            temp = transaction.temp_file("post output")
            temp.write_bytes(b"owned")
            original_replace = transaction._replace

            def replace_temporary_before_publish(source, destination):
                if source == temp:
                    source.replace(stash)
                    source.write_bytes(b"external")
                original_replace(source, destination)

            transaction._replace = replace_temporary_before_publish
            transaction.commit()

    with pytest.raises(MediaError, match="published identity"):
        commit()

    assert post.read_bytes() == b"external"
    assert stash.read_bytes() == b"owned"


def test_transaction_discard_preserves_replaced_sink_temporary(tmp_path):
    post = tmp_path / "post.mp4"
    temp = tmp_path / ".post.mp4.sink.partial"
    stash = tmp_path / "owned-stash"
    temp.write_bytes(b"owned")
    plan = _build_plan(tmp_path)
    cancelled = []

    sink = FileSink.__new__(FileSink)
    sink._temp_path = temp
    sink._published = False
    sink._discarded = False
    sink._transaction_managed = False
    sink.writer = SimpleNamespace(cancel=lambda: cancelled.append(True))

    with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
        transaction.register_sink("post output", sink)
        temp.replace(stash)
        temp.write_bytes(b"external")

    assert cancelled == [True]
    assert temp.read_bytes() == b"external"
    assert stash.read_bytes() == b"owned"
    assert not post.exists()


def test_backup_move_detects_source_identity_replacement(tmp_path):
    post = tmp_path / "post.mp4"
    stash = tmp_path / "actor-stash"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            transaction.temp_file("post output").write_bytes(b"new")
            original_replace = transaction._replace

            def replace_source_before_backup(source, destination):
                if source == post:
                    source.replace(stash)
                    source.write_bytes(b"external")
                original_replace(source, destination)

            transaction._replace = replace_source_before_backup
            transaction.commit()

    with pytest.raises(MediaError, match="identity changed"):
        commit()

    assert post.read_bytes() == b"external"
    assert stash.read_bytes() == b"old"
    assert not list(tmp_path.glob(".*.rollback"))


def test_exclusive_restore_preserves_external_regular_file(tmp_path):
    post = tmp_path / "post.mp4"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)
    failure = TypeError("injected publication defect")

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            temp = transaction.temp_file("post output")
            temp.write_bytes(b"new")
            original_replace = transaction._replace

            def external_file_before_restore(source, destination):
                if source == temp:
                    raise failure
                if source.suffix == ".rollback" and destination == post:
                    destination.write_bytes(b"external")
                original_replace(source, destination)

            transaction._replace = external_file_before_restore
            transaction.commit()

    with pytest.raises(TypeError) as caught:
        commit()

    assert caught.value is failure
    assert post.read_bytes() == b"external"
    backups = list(tmp_path.glob(".*.rollback"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"old"


def test_backup_cleanup_preserves_replacement_identity(tmp_path):
    post = tmp_path / "post.mp4"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            transaction.temp_file("post output").write_bytes(b"new")
            original_cleanup = transaction._cleanup_committed_backup

            def replace_backup_then_cleanup(entry):
                assert entry.backup_path is not None
                entry.backup_path.unlink()
                entry.backup_path.write_bytes(b"external")
                return original_cleanup(entry)

            transaction._cleanup_committed_backup = replace_backup_then_cleanup
            transaction.commit()

    with pytest.raises(MediaError, match="identity changed"):
        commit()

    assert post.read_bytes() == b"new"
    backups = list(tmp_path.glob(".*.rollback"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"external"


def test_publication_replacement_after_rename_survives_rollback(tmp_path):
    post = tmp_path / "post.mp4"
    plan = _build_plan(tmp_path)
    failure = KeyboardInterrupt("publication returned after rename")

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            temp = transaction.temp_file("post output")
            temp.write_bytes(b"new")

            def replace_then_swap(source, destination):
                source.replace(destination)
                destination.unlink()
                destination.write_bytes(b"external")
                raise failure

            transaction._replace = replace_then_swap
            transaction.commit()

    with pytest.raises(KeyboardInterrupt) as caught:
        commit()

    assert caught.value is failure
    assert post.read_bytes() == b"external"
    assert any(
        "publication ownership is ambiguous" in note
        for note in getattr(failure, "__notes__", ())
    )


def test_backup_probe_failure_does_not_replace_primary(tmp_path):
    post = tmp_path / "post.mp4"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)
    failure = KeyboardInterrupt("backup returned after rename")
    probe_failure = OSError("injected backup probe failure")

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            transaction.temp_file("post output").write_bytes(b"new")
            original_inspect = transaction._inspect_identity
            probe_failed = False

            def fail_probe_once(path):
                nonlocal probe_failed
                if path == post and not probe_failed:
                    probe_failed = True
                    return None, probe_failure
                return original_inspect(path)

            def replace_then_interrupt(source, destination):
                source.replace(destination)
                if source == post:
                    raise failure

            transaction._inspect_identity = fail_probe_once
            transaction._replace = replace_then_interrupt
            transaction.commit()

    with pytest.raises(KeyboardInterrupt) as caught:
        commit()

    assert caught.value is failure
    assert post.read_bytes() == b"old"
    assert any(
        "injected backup probe failure" in note
        for note in getattr(failure, "__notes__", ())
    )
    assert not list(tmp_path.glob(".*.rollback"))


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
                    source.replace(destination)
                    raise failure
                if source.suffix == ".rollback":
                    raise restore_failure
                source.replace(destination)

            transaction._replace = replace
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
                if source.suffix == ".rollback":
                    source.replace(destination)
                    raise restore_failure
                source.replace(destination)

            transaction._replace = replace
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
    replacement = b"external replacement"
    cleanup_calls = 0

    def commit():
        with _OutputTransaction(plan, settings) as transaction:
            temp = transaction.temp_file("post output")
            temp.write_bytes(b"new")
            original_remove = transaction._remove

            def fail_publication(source, destination):
                raise failure

            def fail_cleanup_once(path):
                nonlocal cleanup_calls
                if path == temp:
                    cleanup_calls += 1
                    original_remove(path)
                    path.write_bytes(replacement)
                    raise cleanup_failure
                original_remove(path)

            transaction._replace = fail_publication
            transaction._remove = fail_cleanup_once
            transaction.commit()

    with pytest.raises(failure_type) as caught:
        commit()
    assert caught.value is failure
    assert any(
        "artifact rollback also failed" in note
        for note in getattr(failure, "__notes__", ())
    )
    assert cleanup_calls == 1
    partials = list(tmp_path.glob(".*.partial"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == replacement


def test_ambiguous_directory_cleanup_is_not_retried_on_exit(tmp_path):
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

            def replace_contents_then_fail(path):
                nonlocal cleanup_calls
                assert path == temp
                cleanup_calls += 1
                (path / "owned").unlink()
                (path / "external").write_bytes(b"external")
                raise cleanup_failure

            transaction._replace = fail_publication
            transaction._remove = replace_contents_then_fail
            transaction.commit()

    with pytest.raises(TypeError) as caught:
        commit()

    assert caught.value is failure
    assert cleanup_calls == 1
    partials = list(tmp_path.glob(".*.partial"))
    assert len(partials) == 1
    assert (partials[0] / "external").read_bytes() == b"external"


def test_reservation_lock_files_are_account_shared(tmp_path):
    plan = _build_plan(tmp_path)
    settings = _settings(tmp_path)

    with _OutputTransaction(plan, settings):
        lock_root = (settings.shared_temp_dir.expanduser().resolve()
                     / "kinovsr-artifact-locks")
        locks = list(lock_root.glob("*.lock"))
        assert locks
        for lock in locks:
            assert (lock.stat().st_mode & 0o666) == 0o666


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
