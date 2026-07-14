"""Retained native-output copy failures close their owning chain."""

from __future__ import annotations

import pytest

from kinovsr.pipeline.ownership import OwnedCvOutputs
from kinovsr.processors import FrameUnit

pytestmark = pytest.mark.unit


class _Run:
    def __init__(self, close_error: BaseException | None = None) -> None:
        self._yielded = False
        self.close_error = close_error
        self.close_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return FrameUnit(payload=object(), pts=0, duration=1)

    def close(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


def _failing_copy(_payload):
    raise ValueError("retained copy failed")


def test_copy_failure_closes_underlying_run():
    run = _Run()
    outputs = OwnedCvOutputs(run)
    outputs._copy = _failing_copy

    with pytest.raises(ValueError, match="retained copy failed"):
        next(outputs)

    assert run.close_count == 1


def test_copy_side_stop_iteration_is_a_failure_and_closes_run():
    run = _Run()
    outputs = OwnedCvOutputs(run)

    def stop_copy(_payload):
        raise StopIteration("copy stopped")

    outputs._copy = stop_copy
    with pytest.raises(StopIteration, match="copy stopped"):
        next(outputs)

    assert run.close_count == 1


def test_copy_failure_outranks_ordinary_close_failure_without_cycle():
    run = _Run(RuntimeError("run close failed"))
    outputs = OwnedCvOutputs(run)
    outputs._copy = _failing_copy

    with pytest.raises(ValueError, match="retained copy failed") as caught:
        next(outputs)

    assert run.close_count == 1
    assert isinstance(caught.value.__context__, RuntimeError)
    assert str(caught.value.__context__) == "run close failed"
    assert caught.value.__context__.__context__ is None


def test_close_interrupt_outranks_copy_failure():
    run = _Run(KeyboardInterrupt("run close interrupted"))
    outputs = OwnedCvOutputs(run)
    outputs._copy = _failing_copy

    with pytest.raises(KeyboardInterrupt, match="run close interrupted") as caught:
        next(outputs)

    assert run.close_count == 1
    assert isinstance(caught.value.__context__, ValueError)
    assert str(caught.value.__context__) == "retained copy failed"
    assert caught.value.__context__.__context__ is None
