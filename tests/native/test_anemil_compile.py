"""compile_package keeps the compiled model beside its package and nothing elsewhere,
runs that build the same cache entry at once end up sharing one copy, and Core ML's
own cache of what the tests load stays in the test home."""

import os
import shutil
import subprocess
import types
import uuid
from pathlib import Path

import mlx.core as mx
import pytest

from kinovsr.native.anemil import builder, runtime

pytestmark = pytest.mark.unit


def _temporary_directory() -> Path:
    """The user's temporary directory, where Core ML compiles a package whatever
    TMPDIR says."""
    getconf = subprocess.run(
        ["/usr/bin/getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True, check=True
    )
    return Path(getconf.stdout.strip())


def _probe_package(
    directory: Path,
    stem: str,
    description: str = "compile probe",
    weight: mx.array | None = None,
) -> Path:
    weight = mx.full((8, 8, 3, 3), 0.1) if weight is None else weight
    graph = builder.Graph()
    graph.register_input("x", (1, 8, 4, 4))
    output = graph.relu(graph.conv2d("x", weight, None, "c"), "y")
    model = graph.finish([("x", (1, 8, 4, 4))], [], [output], description)
    package = directory / f"{stem}.mlpackage"
    builder.write_package(package, model, graph.blob)
    return package


def _fill(directory: Path, writer: str) -> None:
    directory.mkdir()
    (directory / "writer").write_text(writer)


def test_compile_package_moves_core_mls_output_into_place(tmp_path: Path) -> None:
    # Core ML compiles into "<package stem>.mlmodelc", or "<stem>_<UUID>.mlmodelc"
    # when that name is taken; a unique stem tells this test's output from anyone else's.
    stem = f"compile_probe_{uuid.uuid4().hex}"
    package = _probe_package(tmp_path, stem)

    compiled = runtime.compile_package(package)

    assert compiled == tmp_path / f"{stem}.mlmodelc"
    assert any(compiled.iterdir())
    assert not list(_temporary_directory().glob(f"{stem}*"))
    # A second call finds the compiled model instead of compiling again.
    assert runtime.compile_package(package) == compiled


def test_compile_package_keeps_the_model_a_concurrent_run_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two runs compiling the same package at once: the other one publishes its
    # compiled model while this one is still compiling.
    stem = f"compile_race_{uuid.uuid4().hex}"
    package = _probe_package(tmp_path, stem)
    compiled_dir = tmp_path / f"{stem}.mlmodelc"
    compile_model = runtime.CoreML.MLModel.compileModelAtURL_error_

    def compile_while_another_run_publishes(url: object, error: object) -> object:
        theirs, _ = compile_model(url, error)
        shutil.move(theirs.path(), compiled_dir)
        return compile_model(url, error)

    monkeypatch.setattr(
        runtime,
        "CoreML",
        types.SimpleNamespace(
            MLModel=types.SimpleNamespace(
                compileModelAtURL_error_=compile_while_another_run_publishes
            )
        ),
    )

    assert runtime.compile_package(package) == compiled_dir
    assert any(compiled_dir.iterdir())
    # This run's own compiled model is gone, from the cache and from Core ML's directory.
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"{stem}.mlmodelc",
        f"{stem}.mlpackage",
    ]
    assert not list(_temporary_directory().glob(f"{stem}*"))


class TestPublish:
    """Runs that build the same cache entry at once each fill a staging
    directory of their own; the first to publish wins and the others keep its
    copy, and nothing half-written ever becomes visible."""

    def test_a_complete_directory_is_renamed_into_place(self, tmp_path: Path) -> None:
        target = tmp_path / "entry"
        assert runtime.publish(target, lambda staged: _fill(staged, "this run")) == target
        assert (target / "writer").read_text() == "this run"
        assert [path.name for path in tmp_path.iterdir()] == ["entry"]

    def test_a_missing_cache_directory_is_created(self, tmp_path: Path) -> None:
        # BSVD's scheduled phases convert into a cache directory nothing made yet.
        target = tmp_path / "cache" / "geometry" / "entry"
        assert runtime.publish(target, lambda staged: _fill(staged, "this run")) == target
        assert (target / "writer").read_text() == "this run"

    def test_a_later_run_keeps_the_copy_published_first(self, tmp_path: Path) -> None:
        target = tmp_path / "entry"
        _fill(target, "first run")
        assert runtime.publish(target, lambda staged: _fill(staged, "this run")) == target
        assert (target / "writer").read_text() == "first run"
        assert [path.name for path in tmp_path.iterdir()] == ["entry"]

    def test_a_failed_build_leaves_nothing_behind(self, tmp_path: Path) -> None:
        def fail(staged: Path) -> None:
            _fill(staged, "this run")
            raise OSError("No space left on device")

        with pytest.raises(OSError, match="No space left"):
            runtime.publish(tmp_path / "entry", fail)
        assert not list(tmp_path.iterdir())


def test_core_ml_caches_the_tests_builds_in_the_test_home(tmp_path: Path) -> None:
    # tests/conftest.py points Core ML's never-pruned E5 cache, under the home
    # directory Foundation reports, at a home of the test suite's own. Core ML
    # files this CPU-placed probe by content, so its weights are new every time.
    caches = Path(os.environ["CFFIXED_USER_HOME"]) / "Library" / "Caches"
    before = set(caches.glob("*/com.apple.e5rt.e5bundlecache/*/*"))
    weight = mx.random.normal((8, 8, 3, 3), key=mx.random.key(uuid.uuid4().int >> 64))
    package = _probe_package(tmp_path, "home_probe", "home probe", weight)

    runtime.ModelRunner(runtime.compile_package(package))

    assert set(caches.glob("*/com.apple.e5rt.e5bundlecache/*/*")) - before


def test_model_runner_passes_ane_compiler_options(tmp_path: Path) -> None:
    # Core ML takes ANE compiler options as one "KEY=VALUE,..." string through its
    # private configuration setter; the loaded model reports the string back.
    package = _probe_package(tmp_path, "options_probe", "options probe")
    compiled = runtime.compile_package(package)
    options = "EnableGlobalChannelSplitting=true"

    runner = runtime.ModelRunner(compiled, ane_compiler_options=options)
    plain = runtime.ModelRunner(compiled)

    assert runner._model.configuration().e5rtCustomANECompilerOptions() == options
    assert plain._model.configuration().e5rtCustomANECompilerOptions() is None
