"""The CLI renders typed operational failures without hiding code defects."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from kinovsr.cli.args import build_parser
from kinovsr.processors.errors import MediaError
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit


def _invocation(tmp_path, *, reader: str = "native") -> SimpleNamespace:
    options = build_parser().parse_args([
        "--video", str(tmp_path / "source.mov"),
        "--output-dir", str(tmp_path / "out"),
        "--reader", reader,
    ])
    return SimpleNamespace(
        options=options,
        config={"pipeline": []},
        settings=Settings(
            mlx_cache_limit_gb=0,
            shared_temp_dir=tmp_path / "scratch",
        ),
    )


def _successful_probe(path):
    return 16, 16, 25.0, 10, None, (1, 1)


def test_cli_renders_one_typed_failure(monkeypatch, tmp_path, caplog):
    import kinovsr.api as api
    from kinovsr.cli.commands.run import _run_typed
    from kinovsr.media import video_reader

    root = OSError("disk full")
    failure = MediaError("output write failed: disk full")
    failure.__cause__ = root

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(video_reader, "probe_video", _successful_probe)
    monkeypatch.setattr(api, "process_video_file", fail)
    with caplog.at_level(logging.ERROR, logger="kinovsr.cli.commands.run"):
        result = _run_typed(_invocation(tmp_path))

    records = [
        record for record in caplog.records
        if record.name == "kinovsr.cli.commands.run"
        and record.levelno == logging.ERROR
    ]
    assert result == 2
    assert len(records) == 1
    assert records[0].getMessage() == (
        "processing failed: output write failed: disk full"
    )
    assert records[0].exc_info is None
    assert "Traceback" not in caplog.text


def test_cli_main_renders_cause_once_without_traceback(
        monkeypatch, tmp_path, capfd):
    import kinovsr.api as api
    from kinovsr.cli.main import main
    from kinovsr.media import video_reader

    failure = MediaError("output write failed: disk full")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(video_reader, "probe_video", _successful_probe)
    monkeypatch.setattr(api, "process_video_file", fail)
    result = main([
        "--video", str(tmp_path / "source.mov"),
        "--output-dir", str(tmp_path / "out"),
        "--reader", "native",
    ])
    stderr = capfd.readouterr().err

    assert result == 2
    assert stderr.count("processing failed") == 1
    assert stderr.count("disk full") == 1
    assert "Traceback" not in stderr


@pytest.mark.parametrize(
    "failure_type", [TypeError, AssertionError, KeyboardInterrupt, SystemExit],
)
def test_cli_preserves_non_operational_processing_failure(
        monkeypatch, tmp_path, failure_type):
    import kinovsr.api as api
    from kinovsr.cli.commands.run import _run_typed
    from kinovsr.media import video_reader

    failure = failure_type("injected non-operational failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(video_reader, "probe_video", _successful_probe)
    monkeypatch.setattr(api, "process_video_file", fail)
    with pytest.raises(failure_type) as caught:
        _run_typed(_invocation(tmp_path))
    assert caught.value is failure


@pytest.mark.parametrize("failure_type", [TypeError, AssertionError])
def test_cli_probe_preserves_programmer_failure(
        monkeypatch, tmp_path, failure_type):
    from kinovsr.cli.commands.run import _run_typed
    from kinovsr.media import video_reader

    failure = failure_type("injected probe defect")

    def fail(path):
        raise failure

    monkeypatch.setattr(video_reader, "probe_video", fail)
    with pytest.raises(failure_type) as caught:
        _run_typed(_invocation(tmp_path, reader="auto"))
    assert caught.value is failure


@pytest.mark.parametrize("failure_type", [OSError, RuntimeError])
def test_cli_probe_renders_operational_failure(
        monkeypatch, tmp_path, caplog, failure_type):
    from kinovsr.cli.commands.run import _run_typed
    from kinovsr.media import video_reader

    failure = failure_type("injected probe failure")

    def fail(path):
        raise failure

    monkeypatch.setattr(video_reader, "probe_video", fail)
    with caplog.at_level(logging.ERROR, logger="kinovsr.cli.commands.run"):
        result = _run_typed(_invocation(tmp_path))

    records = [
        record for record in caplog.records
        if record.name == "kinovsr.cli.commands.run"
        and record.levelno == logging.ERROR
    ]
    assert result == 2
    assert len(records) == 1
    assert "cannot open" in records[0].getMessage()
    assert "injected probe failure" in records[0].getMessage()
    assert records[0].exc_info is None


def test_cli_rejects_bad_time_spec_cleanly(monkeypatch, tmp_path, caplog):
    from kinovsr.cli.commands.run import _run_typed
    from kinovsr.media import video_reader

    monkeypatch.setattr(video_reader, "probe_video", _successful_probe)
    options = build_parser().parse_args([
        "--video", str(tmp_path / "source.mov"),
        "--output-dir", str(tmp_path / "out"),
        "--reader", "native",
        "--start", "0:5:x",
    ])
    invocation = SimpleNamespace(
        options=options,
        config={"pipeline": []},
        settings=Settings(
            mlx_cache_limit_gb=0,
            shared_temp_dir=tmp_path / "scratch",
        ),
    )
    with caplog.at_level(logging.ERROR, logger="kinovsr.cli.commands.run"):
        result = _run_typed(invocation)

    records = [
        record for record in caplog.records
        if record.name == "kinovsr.cli.commands.run"
        and record.levelno == logging.ERROR
    ]
    assert result == 2
    assert len(records) == 1
    assert "bad --start/--end value" in records[0].getMessage()
    assert records[0].exc_info is None
    assert "Traceback" not in caplog.text
