"""Endpoint owners normalize operational failures without hiding defects."""

from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import pytest

from kinovsr.pipeline import run as run_module
from kinovsr.pipeline.run import (
    FileSink,
    FileSource,
    _ArtifactPlan,
    _OutputTransaction,
    _run_file_reserved,
    _save_frame_png,
)
from kinovsr.processors.errors import MediaError
from kinovsr.processors.specs import (
    Geometry,
    Layout,
    StreamSpec,
    TimelineSpec,
    frame_spec_for_matrix,
)
from kinovsr.processors.units import FrameUnit
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


def _spec(*, layout: Layout = Layout.CV_BGRA) -> StreamSpec:
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709",
            full_range=False,
            geometry=Geometry(16, 16),
            layout=layout,
        ),
        timeline=TimelineSpec(
            time_base=Fraction(1, 24000), cadence=Fraction(25),
        ),
    )


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError])
def test_source_probe_normalizes_operational_failure(
        tmp_path, failure_type):
    failure = failure_type("injected source probe failure")

    class _Reader:
        @staticmethod
        def probe_video(path):
            raise failure

    with pytest.raises(MediaError) as caught:
        FileSource(tmp_path / "source.mov", reader=_Reader())
    assert caught.value.__cause__ is failure
    assert str(tmp_path / "source.mov") in str(caught.value)


@pytest.mark.parametrize("failure_type", [TypeError, AssertionError])
def test_source_probe_preserves_programmer_failure(tmp_path, failure_type):
    failure = failure_type("injected source probe defect")

    class _Reader:
        @staticmethod
        def probe_video(path):
            raise failure

    with pytest.raises(failure_type) as caught:
        FileSource(tmp_path / "source.mov", reader=_Reader())
    assert caught.value is failure


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError, ValueError])
def test_writer_setup_normalizes_operational_failure(
        monkeypatch, tmp_path, failure_type):
    from kinovsr.native import writer as writer_module

    failure = failure_type("injected writer setup failure")

    class _Writer:
        def __init__(self, *args, **kwargs):
            raise failure

    monkeypatch.setattr(writer_module, "AVWriter", _Writer)
    output = tmp_path / "output.mp4"
    with pytest.raises(MediaError) as caught:
        FileSink(output, _spec())
    assert caught.value.__cause__ is failure
    assert str(output) in str(caught.value)
    assert not list(tmp_path.glob(".output.mp4.*.partial"))


@pytest.mark.parametrize(
    "failure_type", [TypeError, AssertionError, KeyboardInterrupt, SystemExit],
)
def test_writer_setup_preserves_non_operational_failure(
        monkeypatch, tmp_path, failure_type):
    from kinovsr.native import writer as writer_module

    failure = failure_type("injected writer setup defect")

    class _Writer:
        def __init__(self, *args, **kwargs):
            raise failure

    monkeypatch.setattr(writer_module, "AVWriter", _Writer)
    with pytest.raises(failure_type) as caught:
        FileSink(tmp_path / "output.mp4", _spec())
    assert caught.value is failure
    assert not list(tmp_path.glob(".output.mp4.*.partial"))


def test_writer_temporary_failure_is_typed_and_chained(monkeypatch, tmp_path):
    failure = OSError("injected ENOSPC")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(run_module.tempfile, "mkstemp", fail)
    with pytest.raises(MediaError) as caught:
        FileSink(tmp_path / "output.mp4", _spec())
    assert caught.value.__cause__ is failure
    assert "writer temporary" in str(caught.value)


@pytest.mark.parametrize(
    "failure_type", [TypeError, AssertionError, KeyboardInterrupt, SystemExit],
)
def test_writer_temporary_path_conversion_failure_cleans_partial(
        monkeypatch, tmp_path, failure_type):
    failure = failure_type("injected writer Path conversion failure")
    original_path = run_module.Path

    def fail_partial(raw):
        if isinstance(raw, str) and raw.endswith(".partial"):
            raise failure
        return original_path(raw)

    monkeypatch.setattr(run_module, "Path", fail_partial)
    with pytest.raises(failure_type) as caught:
        FileSink(tmp_path / "output.mp4", _spec())

    assert caught.value is failure
    assert not list(tmp_path.glob(".output.mp4.*.partial"))


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_writer_temporary_interrupt_does_not_retry_ambiguous_close(
        monkeypatch, tmp_path, failure_type):
    failure = failure_type("injected close interruption")
    original_close = run_module.os.close
    close_calls = []
    acquired_fd = None

    def interrupt_before_close(fd):
        nonlocal acquired_fd
        acquired_fd = fd
        close_calls.append(fd)
        raise failure

    monkeypatch.setattr(run_module.os, "close", interrupt_before_close)
    with pytest.raises(failure_type) as caught:
        FileSink(tmp_path / "output.mp4", _spec())
    assert caught.value is failure
    assert acquired_fd is not None
    assert close_calls == [acquired_fd]
    run_module.os.fstat(acquired_fd)
    original_close(acquired_fd)
    assert not list(tmp_path.glob(".output.mp4.*.partial"))


def _bare_sink(tmp_path: Path, failure: BaseException, method: str) -> FileSink:
    sink = FileSink.__new__(FileSink)
    sink.spec = _spec()
    sink._is_mlx = False
    sink._final_path = tmp_path / "output.mp4"
    sink._temp_path = tmp_path / ".output.mp4.partial"
    sink._published = False
    sink._finalized = False
    sink._discarded = False

    class _Writer:
        frame_count = 0

        def append(self, *args, **kwargs):
            if method == "append":
                raise failure

        def finish(self):
            if method == "finalize":
                raise failure

        def cancel(self):
            pass

    sink.writer = _Writer()
    return sink


def _invoke_sink_operation(sink: FileSink, method: str) -> None:
    if method == "append":
        sink.append(FrameUnit(payload=object(), pts=0, duration=960))
    else:
        sink.finalize()


@pytest.mark.parametrize("method", ["append", "finalize"])
@pytest.mark.parametrize("failure_type", [OSError, RuntimeError, ValueError])
def test_writer_operations_normalize_operational_failure(
        tmp_path, method, failure_type):
    failure = failure_type(f"injected writer {method} failure")
    sink = _bare_sink(tmp_path, failure, method)

    with pytest.raises(MediaError) as caught:
        _invoke_sink_operation(sink, method)
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("method", ["append", "finalize"])
@pytest.mark.parametrize(
    "failure_type", [TypeError, AssertionError, KeyboardInterrupt, SystemExit],
)
def test_writer_operations_preserve_non_operational_failure(
        tmp_path, method, failure_type):
    failure = failure_type(f"injected writer {method} defect")
    sink = _bare_sink(tmp_path, failure, method)

    with pytest.raises(failure_type) as caught:
        _invoke_sink_operation(sink, method)
    assert caught.value is failure


def test_writer_mlx_conversion_value_error_is_not_mislabeled(tmp_path):
    failure = ValueError("injected MLX conversion defect")
    sink = _bare_sink(tmp_path, failure, "none")
    sink.spec = _spec(layout=Layout.MLX_RGB_HWC)
    sink._is_mlx = True
    sink._direct_mlx_encode = False

    def fail_conversion(frame):
        raise failure

    sink._mlx_to_buffer = fail_conversion
    payload = type("Frame", (), {"shape": (16, 16, 3)})()

    with pytest.raises(ValueError) as caught:
        sink.append(FrameUnit(payload=payload, pts=0, duration=960))
    assert caught.value is failure


@pytest.mark.parametrize(
    "failure_type", [OSError, RuntimeError, ValueError])
def test_writer_direct_mlx_failure_is_typed(tmp_path, failure_type):
    # One boundary for the collapsed direct path: the conversion graph
    # builds lazily for every payload the shape gate admits (verified by
    # probing all admissible payload classes), so operational failures -
    # including the native layer's ValueErrors - surface from the single
    # append call and are typed there.
    failure = failure_type("injected direct MLX append failure")
    sink = _bare_sink(tmp_path, failure, "none")
    sink.spec = _spec(layout=Layout.MLX_RGB_HWC)
    sink._is_mlx = True
    sink._direct_mlx_encode = True

    class _Writer:
        frame_count = 0

        @staticmethod
        def append_mlx_rgb(*args, **kwargs):
            raise failure

    sink.writer = _Writer()
    payload = type("Frame", (), {"shape": (16, 16, 3)})()

    with pytest.raises(MediaError) as caught:
        sink.append(FrameUnit(payload=payload, pts=0, duration=960))
    assert caught.value.__cause__ is failure


def test_writer_cleanup_failure_does_not_replace_primary(
        monkeypatch, tmp_path):
    failure = AssertionError("injected writer defect")
    cleanup_failure = KeyboardInterrupt("injected unlink interruption")
    sink = _bare_sink(tmp_path, failure, "finalize")
    sink._temp_path.write_bytes(b"partial")
    original_unlink = Path.unlink
    interrupted = False
    replacement = b"external replacement"

    def unlink_then_replace(path, *args, **kwargs):
        nonlocal interrupted
        if path == sink._temp_path and not interrupted:
            interrupted = True
            original_unlink(path, *args, **kwargs)
            path.write_bytes(replacement)
            raise cleanup_failure
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_then_replace)
    with pytest.raises(AssertionError) as caught:
        sink.finalize()
    assert caught.value is failure
    assert sink._temp_path.read_bytes() == replacement


def _plan(tmp_path: Path, **kwargs) -> _ArtifactPlan:
    source = tmp_path / "source.mov"
    source.write_bytes(b"source")
    return _ArtifactPlan.build(
        video=source,
        output=tmp_path / "output.mp4",
        comparison=kwargs.get("comparison"),
        cut_log=kwargs.get("cut_log"),
        save_audio_sidecar=kwargs.get("save_audio_sidecar", False),
        save_pre_frames=kwargs.get("save_pre_frames"),
        save_post_frames=None,
        skip_post_mp4=kwargs.get("skip_post_mp4", False),
        noise_map_debug=False,
        overwrite=False,
    )


def test_transaction_temp_failure_is_typed_and_chained(monkeypatch, tmp_path):
    failure = OSError("injected artifact ENOSPC")
    plan = _plan(tmp_path)
    transaction = _OutputTransaction(
        plan, Settings(shared_temp_dir=tmp_path / "scratch"),
    )

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(run_module.tempfile, "mkstemp", fail)
    with pytest.raises(MediaError) as caught:
        transaction.temp_file("post output")
    assert caught.value.__cause__ is failure
    assert "post output" in str(caught.value)


@pytest.mark.parametrize("failure_type", [TypeError, AssertionError])
def test_transaction_temp_cleanup_preserves_programmer_failure(
        monkeypatch, tmp_path, failure_type):
    plan = _plan(tmp_path)
    transaction = _OutputTransaction(
        plan, Settings(shared_temp_dir=tmp_path / "scratch"),
    )
    failure = failure_type("injected close defect")
    original_close = run_module.os.close

    def close_then_fail(fd):
        original_close(fd)
        raise failure

    monkeypatch.setattr(run_module.os, "close", close_then_fail)
    with pytest.raises(failure_type) as caught:
        transaction.temp_file("post output")
    assert caught.value is failure
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_transaction_temp_interrupt_does_not_retry_ambiguous_close(
        monkeypatch, tmp_path, failure_type):
    plan = _plan(tmp_path)
    transaction = _OutputTransaction(
        plan, Settings(shared_temp_dir=tmp_path / "scratch"),
    )
    failure = failure_type("injected close interruption")
    original_close = run_module.os.close
    close_calls = []
    acquired_fd = None

    def interrupt_before_close(fd):
        nonlocal acquired_fd
        acquired_fd = fd
        close_calls.append(fd)
        raise failure

    monkeypatch.setattr(run_module.os, "close", interrupt_before_close)
    with pytest.raises(failure_type) as caught:
        transaction.temp_file("post output")
    assert caught.value is failure
    assert acquired_fd is not None
    assert close_calls == [acquired_fd]
    run_module.os.fstat(acquired_fd)
    original_close(acquired_fd)
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize(
    "failure_type", [TypeError, AssertionError, KeyboardInterrupt, SystemExit],
)
def test_transaction_temp_file_conversion_failure_cleans_partial(
        monkeypatch, tmp_path, failure_type):
    plan = _plan(tmp_path)
    transaction = _OutputTransaction(
        plan, Settings(shared_temp_dir=tmp_path / "scratch"),
    )
    failure = failure_type("injected file Path conversion failure")

    def fail(raw):
        raise failure

    monkeypatch.setattr(run_module, "Path", fail)
    with pytest.raises(failure_type) as caught:
        transaction.temp_file("post output")

    assert caught.value is failure
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize(
    "failure_type", [TypeError, AssertionError, KeyboardInterrupt, SystemExit],
)
def test_transaction_temp_directory_conversion_failure_cleans_partial(
        monkeypatch, tmp_path, failure_type):
    pre = tmp_path / "frames"
    plan = _plan(tmp_path, save_pre_frames=pre)
    transaction = _OutputTransaction(
        plan, Settings(shared_temp_dir=tmp_path / "scratch"),
    )
    failure = failure_type("injected path conversion failure")

    def fail(raw):
        raise failure

    monkeypatch.setattr(run_module, "Path", fail)
    with pytest.raises(failure_type) as caught:
        transaction.temp_directory("pre-frame directory")
    assert caught.value is failure
    assert not list(tmp_path.glob(".frames.*.partial"))


def test_writer_setup_normalizes_objc_error(monkeypatch, tmp_path):
    import objc

    from kinovsr.native import writer as writer_module

    failure = objc.error("injected AVFoundation bridge failure")

    class _Writer:
        def __init__(self, *args, **kwargs):
            raise failure

    monkeypatch.setattr(writer_module, "AVWriter", _Writer)
    with pytest.raises(MediaError) as caught:
        FileSink(tmp_path / "output.mp4", _spec())
    assert caught.value.__cause__ is failure
    assert not list(tmp_path.glob(".output.mp4.*.partial"))


@pytest.mark.parametrize("failure_type", [TypeError, AssertionError])
def test_transaction_rollback_preserves_programmer_failure(
        tmp_path, failure_type):
    comparison = tmp_path / "comparison.mp4"
    plan = _plan(tmp_path, comparison=comparison)
    settings = Settings(shared_temp_dir=tmp_path / "scratch")
    failure = failure_type("injected publication defect")

    def commit():
        with _OutputTransaction(plan, settings) as transaction:
            post_temp = transaction.temp_file("post output")
            comparison_temp = transaction.temp_file("comparison output")
            post_temp.write_bytes(b"post")
            comparison_temp.write_bytes(b"comparison")

            def replace(source, destination):
                if source == comparison_temp:
                    raise failure
                source.replace(destination)

            transaction._replace = replace
            transaction.commit()

    with pytest.raises(failure_type) as caught:
        commit()

    assert caught.value is failure
    assert not plan.path("post output").exists()
    assert not comparison.exists()
    assert not list(tmp_path.glob(".*.partial"))


class _ArtifactTransaction:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def temp_file(self, label: str) -> Path:
        return self.tmp_path / f".{label}.partial"


def _call_reserved(
    monkeypatch,
    tmp_path: Path,
    plan: _ArtifactPlan,
    *,
    track=None,
    audio: bool,
):
    from kinovsr.pipeline import session as session_module

    spec = _spec(layout=Layout.MLX_RGB_HWC)

    class _Source:
        def __init__(self, *args, **kwargs):
            self.spec = spec
            self.source_cadence = Fraction(25)
            self.frame_count = 1

        def audio_track(self, *, max_duration=None):
            return track

    class _Session:
        output_spec = spec

    monkeypatch.setattr(run_module, "FileSource", _Source)
    monkeypatch.setattr(
        session_module, "open_pipeline", lambda *args, **kwargs: _Session(),
    )
    return _run_file_reserved(
        {"pipeline": []},
        plan=plan,
        transaction=_ArtifactTransaction(tmp_path),
        settings=Settings(shared_temp_dir=tmp_path / "scratch"),
        t0=time.perf_counter(),
        reporter=None,
        layout=Layout.MLX_RGB_HWC,
        start=0,
        end=None,
        max_frames=None,
        max_output_frames=None,
        max_output_seconds=None,
        audio=audio,
        audio_codec="alac",
        quality=0.65,
        chunk_size=1,
        source_color="auto",
        source_range="auto",
        encode_chroma="auto",
        snap_start=False,
        gop_align=False,
        gop_min_window=16,
        gop_max_window=96,
        reader=None,
        timing=None,
    )


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError])
def test_audio_sidecar_failure_is_typed_and_chained(
        monkeypatch, tmp_path, failure_type):
    failure = failure_type("injected sidecar write failure")

    class _Track:
        def save_wav(self, path):
            raise failure

    plan = _plan(
        tmp_path, save_audio_sidecar=True, skip_post_mp4=True,
    )
    with pytest.raises(MediaError) as caught:
        _call_reserved(
            monkeypatch, tmp_path, plan, track=_Track(), audio=True,
        )
    assert caught.value.__cause__ is failure
    assert str(plan.path("audio sidecar")) in str(caught.value)


def test_cut_log_failure_is_typed_and_chained(monkeypatch, tmp_path):
    failure = OSError("injected cut-log write failure")
    plan = _plan(
        tmp_path, cut_log=tmp_path / "cuts.txt", skip_post_mp4=True,
    )
    temp = tmp_path / ".cut log.partial"
    original_write_text = Path.write_text

    def write_text(path, *args, **kwargs):
        if path == temp:
            raise failure
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    with pytest.raises(MediaError) as caught:
        _call_reserved(monkeypatch, tmp_path, plan, audio=False)
    assert caught.value.__cause__ is failure
    assert str(plan.path("cut log")) in str(caught.value)


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError])
def test_frame_image_failure_is_typed_and_chained(
        monkeypatch, tmp_path, failure_type):
    from kinovsr.media import images

    failure = failure_type("injected frame image failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(images, "save_image", fail)
    pb = type("PB", (), {"read_pixel_buffer_rgb": staticmethod(lambda value: value)})
    with pytest.raises(MediaError) as caught:
        _save_frame_png(object(), Layout.CV_BGRA, tmp_path, 3, pb)
    assert caught.value.__cause__ is failure
    assert str(tmp_path / "frame_00003.png") in str(caught.value)


@pytest.mark.parametrize("failure_type", [TypeError, AssertionError])
def test_frame_image_preserves_programmer_failure(
        monkeypatch, tmp_path, failure_type):
    from kinovsr.media import images

    failure = failure_type("injected frame image defect")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(images, "save_image", fail)
    pb = type("PB", (), {"read_pixel_buffer_rgb": staticmethod(lambda value: value)})
    with pytest.raises(failure_type) as caught:
        _save_frame_png(object(), Layout.CV_BGRA, tmp_path, 3, pb)
    assert caught.value is failure
