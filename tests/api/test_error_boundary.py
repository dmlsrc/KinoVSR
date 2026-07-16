"""Operational failures stay behind the documented public error boundary."""

from __future__ import annotations

import contextlib
from fractions import Fraction

import pytest

from kinovsr.pipeline.builder import ResolvedStage
from kinovsr.pipeline.scheduler import run_chain
from kinovsr.processors import (
    Capability,
    CapabilitySpec,
    FrameUnit,
    Geometry,
    PipelineContext,
    StreamConstraint,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
    preserve_stream,
)
from kinovsr.processors.errors import MediaError
from kinovsr.processors.specs import Layout
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failure_type", [OSError])
def test_process_video_file_normalizes_operational_failure(
        monkeypatch, tmp_path, failure_type):
    import kinovsr.api as api
    import kinovsr.pipeline as pipeline

    failure = failure_type("injected endpoint failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(api, "_runtime_setup", lambda settings: None)
    monkeypatch.setattr(pipeline, "run_file", fail)

    source = tmp_path / "source.mov"
    output = tmp_path / "output.mp4"
    with pytest.raises(MediaError) as caught:
        api.process_video_file(
            {}, video=source, output=output,
            settings=Settings(mlx_cache_limit_gb=0),
        )

    assert caught.value.__cause__ is failure
    assert str(source) in str(caught.value)
    assert str(output) in str(caught.value)


def test_process_video_file_preserves_unowned_runtime_failure(
        monkeypatch, tmp_path):
    import kinovsr.api as api
    import kinovsr.pipeline as pipeline

    failure = RuntimeError("injected internal invariant failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(api, "_runtime_setup", lambda settings: None)
    monkeypatch.setattr(pipeline, "run_file", fail)

    with pytest.raises(RuntimeError) as caught:
        api.process_video_file(
            {}, video=tmp_path / "source.mov", output=tmp_path / "output.mp4",
            settings=Settings(mlx_cache_limit_gb=0),
        )
    assert caught.value is failure


def test_process_video_file_preserves_typed_failure(monkeypatch, tmp_path):
    import kinovsr.api as api
    import kinovsr.pipeline as pipeline

    failure = MediaError("already normalized")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(api, "_runtime_setup", lambda settings: None)
    monkeypatch.setattr(pipeline, "run_file", fail)

    with pytest.raises(MediaError) as caught:
        api.process_video_file(
            {}, video=tmp_path / "source.mov", output=tmp_path / "output.mp4",
            settings=Settings(mlx_cache_limit_gb=0),
        )
    assert caught.value is failure


@pytest.mark.parametrize(
    "failure_type", [TypeError, AssertionError, KeyboardInterrupt, SystemExit],
)
def test_process_video_file_preserves_non_operational_failure(
        monkeypatch, tmp_path, failure_type):
    import kinovsr.api as api
    import kinovsr.pipeline as pipeline

    failure = failure_type("injected non-operational failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(api, "_runtime_setup", lambda settings: None)
    monkeypatch.setattr(pipeline, "run_file", fail)

    with pytest.raises(failure_type) as caught:
        api.process_video_file(
            {}, video=tmp_path / "source.mov", output=tmp_path / "output.mp4",
            settings=Settings(mlx_cache_limit_gb=0),
        )
    assert caught.value is failure


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError])
def test_process_video_file_normalizes_runtime_setup_failure(
        monkeypatch, tmp_path, failure_type):
    import kinovsr.api as api

    failure = failure_type("injected MLX setup failure")

    def fail(settings):
        raise failure

    monkeypatch.setattr(api, "_runtime_setup", fail)
    with pytest.raises(MediaError) as caught:
        api.process_video_file(
            {}, video=tmp_path / "source.mov", output=tmp_path / "output.mp4",
            settings=Settings(mlx_cache_limit_gb=0),
        )
    assert caught.value.__cause__ is failure


def _file_spec() -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709",
            full_range=False,
            geometry=Geometry(16, 16),
            layout=Layout.CV_BGRA,
        ),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25),
        ),
    )


class _Run:
    def __init__(self, units):
        self._iterator = iter(units)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Session:
    def __init__(self, spec, *, failure=None):
        self.output_spec = spec
        self.failure = failure

    def _bind_terminal_output_pool(self, *args):
        pass

    def process(self, units, *, retain_outputs):
        if self.failure is None:
            return _Run(units)

        failure = self.failure

        class _Processor:
            def prepare(self, input_spec, context):
                pass

            def process(self, unit, context):
                raise failure
                yield unit

            def reset(self, boundary, context):
                pass

            def flush(self, context):
                return ()

            def close(self, context):
                pass

        stage = ResolvedStage(
            name="injected",
            position=0,
            family="injected",
            factory=None,
            capability=Capability.PREPROCESS,
            capability_spec=CapabilitySpec(
                capability=Capability.PREPROCESS,
                profiles=(),
                accepts=StreamConstraint(),
                produces=preserve_stream,
            ),
            profile=None,
            config=None,
            input_spec=self.output_spec,
            output_spec=self.output_spec,
        )
        context = PipelineContext(settings=Settings(mlx_cache_limit_gb=0))
        return run_chain(((stage, _Processor()),), units, context)

    def stage_diagnostics(self):
        return []

    def stage_debug_images(self):
        return {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _patch_public_file_path(monkeypatch, tmp_path, *, track=None, failure=None):
    from kinovsr.media import pixel_buffers
    from kinovsr.pipeline import run as run_module
    from kinovsr.pipeline import session as session_module

    spec = _file_spec()

    class _Source:
        def __init__(self, *args, **kwargs):
            self.spec = spec
            self.resolved_color = None
            self.transform = None
            self.source_cadence = Fraction(25)
            self.frame_count = 1

        def units(self):
            return iter([FrameUnit(payload=object(), pts=0, duration=960)])

        def audio_track(self, *, max_duration=None):
            return track

    monkeypatch.setattr(run_module, "FileSource", _Source)
    monkeypatch.setattr(run_module, "_probe_timing", lambda *args: None)
    monkeypatch.setattr(
        session_module,
        "open_pipeline",
        lambda *args, **kwargs: _Session(spec, failure=failure),
    )
    monkeypatch.setattr(pixel_buffers, "ci_cache_owner", contextlib.nullcontext)


class _Adaptor:
    @staticmethod
    def pixelBufferPool():
        return None


def _writer_type(failure, phase):
    class _Writer:
        accepts_mlx_rgb = False
        adaptor = _Adaptor()
        adaptor_pixel_format = 0
        adaptor_width = 16
        adaptor_height = 16

        def __init__(self, *args, **kwargs):
            self.frame_count = 0
            if phase == "setup":
                raise failure

        def append(self, *args, **kwargs):
            if phase == "append":
                raise failure
            self.frame_count += 1

        def finish(self):
            if phase == "finalize":
                raise failure

        def cancel(self):
            pass

    return _Writer


@pytest.mark.parametrize("phase", ["setup", "append", "finalize"])
@pytest.mark.parametrize("failure_type", [OSError, RuntimeError, ValueError])
def test_public_file_path_normalizes_writer_failure(
        monkeypatch, tmp_path, phase, failure_type):
    import kinovsr.api as api
    from kinovsr.native import writer as writer_module

    failure = failure_type(f"injected writer {phase} failure")
    _patch_public_file_path(monkeypatch, tmp_path)
    monkeypatch.setattr(writer_module, "AVWriter", _writer_type(failure, phase))
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"

    with pytest.raises(MediaError) as caught:
        api.process_video_file(
            {}, video=source, output=output,
            settings=Settings(
                mlx_cache_limit_gb=0,
                shared_temp_dir=tmp_path / "scratch",
            ),
        )
    assert caught.value.__cause__ is failure
    assert not output.exists()
    assert not list(tmp_path.glob(".*.partial"))


def test_public_file_path_normalizes_objc_writer_failure(monkeypatch, tmp_path):
    import objc

    import kinovsr.api as api
    from kinovsr.native import writer as writer_module

    failure = objc.error("injected AVFoundation bridge failure")
    _patch_public_file_path(monkeypatch, tmp_path)
    monkeypatch.setattr(
        writer_module, "AVWriter", _writer_type(failure, "setup"),
    )
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")

    with pytest.raises(MediaError) as caught:
        api.process_video_file(
            {}, video=source, output=tmp_path / "output.mp4",
            settings=Settings(
                mlx_cache_limit_gb=0,
                shared_temp_dir=tmp_path / "scratch",
            ),
        )
    assert caught.value.__cause__ is failure


def test_public_file_path_normalizes_publication_failure(monkeypatch, tmp_path):
    import kinovsr.api as api
    from kinovsr.native import writer as writer_module
    from kinovsr.pipeline.run import _OutputTransaction

    failure = OSError("injected publication failure")
    _patch_public_file_path(monkeypatch, tmp_path)
    monkeypatch.setattr(writer_module, "AVWriter", _writer_type(None, "none"))
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"
    original_replace = _OutputTransaction._replace

    def fail_publication(source_path, destination):
        if destination == output.resolve():
            raise failure
        original_replace(source_path, destination)

    monkeypatch.setattr(
        _OutputTransaction, "_replace", staticmethod(fail_publication),
    )
    with pytest.raises(MediaError) as caught:
        api.process_video_file(
            {}, video=source, output=output,
            settings=Settings(
                mlx_cache_limit_gb=0,
                shared_temp_dir=tmp_path / "scratch",
            ),
        )
    assert caught.value.__cause__ is failure
    assert not output.exists()
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError])
def test_public_file_path_normalizes_sidecar_failure(
        monkeypatch, tmp_path, failure_type):
    import kinovsr.api as api

    failure = failure_type("injected sidecar failure")

    class _Track:
        def save_wav(self, path):
            raise failure

    _patch_public_file_path(monkeypatch, tmp_path, track=_Track())
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    output = tmp_path / "output.mp4"
    with pytest.raises(MediaError) as caught:
        api.process_video_file(
            {}, video=source, output=output,
            settings=Settings(
                mlx_cache_limit_gb=0,
                shared_temp_dir=tmp_path / "scratch",
            ),
            audio=True,
            save_audio_sidecar=True,
            skip_post_mp4=True,
        )
    assert caught.value.__cause__ is failure
    assert not output.with_name("output_audio.wav").exists()
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize("failure_type", [TypeError, AssertionError])
def test_public_file_path_preserves_processor_programmer_failure(
        monkeypatch, tmp_path, failure_type):
    import kinovsr.api as api
    from kinovsr.native import writer as writer_module

    failure = failure_type("injected processor defect")
    _patch_public_file_path(monkeypatch, tmp_path, failure=failure)
    monkeypatch.setattr(writer_module, "AVWriter", _writer_type(None, "none"))
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")

    with pytest.raises(failure_type) as caught:
        api.process_video_file(
            {}, video=source, output=tmp_path / "output.mp4",
            settings=Settings(
                mlx_cache_limit_gb=0,
                shared_temp_dir=tmp_path / "scratch",
            ),
        )
    assert caught.value is failure
