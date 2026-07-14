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

    with pytest.raises(RuntimeError, match="injected comparison"):
        _ComparisonTee(tmp_path / "comparison.mp4", spec, source, quality=0.5)
    assert len(instances) == 1
    assert instances[0].discarded


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


def test_interrupt_during_backup_cleanup_keeps_committed_output(tmp_path):
    post = tmp_path / "post.mp4"
    post.write_bytes(b"old")
    plan = _build_plan(tmp_path, overwrite=True)

    def commit():
        with _OutputTransaction(plan, _settings(tmp_path)) as transaction:
            transaction.temp_file("post output").write_bytes(b"new")
            original_remove = transaction._remove
            interrupted = False

            def interrupt_once(path):
                nonlocal interrupted
                if path.suffix == ".rollback" and not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt("during backup cleanup")
                original_remove(path)

            transaction._remove = interrupt_once
            transaction.commit()

    with pytest.raises(KeyboardInterrupt):
        commit()

    assert post.read_bytes() == b"new"
    assert not list(tmp_path.glob(".*.rollback"))
