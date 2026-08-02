import json
from pathlib import Path

import pytest

from kinovsr.modeling.window_buffer import WindowBuffer
from kinovsr.processors.protocol import GopWindowPolicy
from kinovsr.processors.units import FrameUnit, SourceFrameInfo


def _windows(keyframes, count, minimum, maximum, source_indices=None):
    edges, emitted = [], []

    def run(frames, tokens, emit_start, emit_end):
        edges.append((frames[0], frames[-1] + 1,
                      frames[emit_start], frames[emit_end - 1] + 1))
        return [(frames[i], tokens[i]) for i in range(emit_start, emit_end)]

    machine = WindowBuffer.gop(minimum, maximum, run)
    syncs = set(keyframes)
    for position, source_index in enumerate(
            range(count) if source_indices is None else source_indices):
        token = FrameUnit(
            None, position, 1,
            source=SourceFrameInfo(source_index, source_index in syncs))
        emitted.extend(machine.feed(position, token))
    emitted.extend(machine.flush())
    return edges, [token.pts for _, token in emitted]


_CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures/gop_window_edges.json").read_text())


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
def test_recorded_planner_edges(case):
    edges, emitted = _windows(
        case["keyframes"], case["count"], case["min"], case["max"])
    assert [list(edge) for edge in edges] == case["windows"]
    assert emitted == list(range(case["count"]))


def test_conform_duplicate_uses_only_the_first_sync_slot():
    edges, emitted = _windows(
        [0, 5], 9, 4, 8,
        [0, 0, 1, 2, 5, 5, 6, 7, 8])
    assert edges == [(0, 5, 0, 4), (4, 9, 4, 9)]
    assert emitted == list(range(9))


def test_dropped_sync_does_not_fabricate_an_anchor():
    edges, _ = _windows([0, 1], 3, 1, 8, source_indices=[0, 3, 6])
    assert edges == [(0, 3, 0, 3)]


def test_flush_then_reset_clamps_a_hard_cut():
    windows = []

    def run(frames, _tokens, start, end):
        windows.append((list(frames), start, end))
        return ()

    machine = WindowBuffer.gop(4, 8, run)
    for index in range(3):
        list(machine.feed(index))
    list(machine.flush())
    machine.reset()
    for index in range(100, 104):
        list(machine.feed(index))
    list(machine.flush())
    assert windows == [([0, 1, 2], 0, 3), ([100, 101, 102, 103], 0, 4)]


@pytest.mark.parametrize("window,trim,count,expected", [
    (1, 0, 3, [(0, 1), (1, 2), (2, 3)]),
    (8, 0, 0, []),
    (8, 0, 7, [(0, 7)]),
    (8, 0, 8, [(0, 8)]),
    (8, 0, 9, [(0, 8), (1, 9)]),
    (8, 2, 8, [(0, 8), (0, 8)]),
    (8, 2, 17, [(0, 8), (4, 12), (8, 16), (9, 17)]),
    (8, 2, 20, [(0, 8), (4, 12), (8, 16), (12, 20), (12, 20)]),
])
def test_fixed_sliding_processing_ranges(window, trim, count, expected):
    calls = []

    def run(frames, tokens, emit_start, emit_end):
        calls.append((frames[0], frames[-1] + 1))
        return [(frames[i], tokens[i]) for i in range(emit_start, emit_end)]

    machine = WindowBuffer(window, trim, run)
    runs = [machine.feed(index, index) for index in range(count)]
    tail = machine.flush()
    assert all(isinstance(ready, list) for ready in (*runs, tail))
    emitted = [item for ready in (*runs, tail) for item in ready]
    assert calls == expected
    assert [token for _, token in emitted] == list(range(count))


@pytest.mark.parametrize(
    "minimum,maximum", [(0, 8), (-1, 8), (8, 0), (9, 8), (True, 8), (4.0, 8)])
def test_policy_rejects_old_planner_bound_errors(minimum, maximum):
    with pytest.raises(ValueError):
        GopWindowPolicy(minimum, maximum)
