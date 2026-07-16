"""Nearest-slot conform mechanics: dup, drop, ledger, grid regeneration."""

from fractions import Fraction

import pytest

from kinovsr.processors.conform import (
    CfrConformProcessor,
    ConformStageConfig,
    _produces,
)
from kinovsr.processors.protocol import PipelineContext
from kinovsr.processors.specs import (
    Cardinality,
    Geometry,
    StreamSpec,
    TimelineSpec,
    TimestampPolicy,
    VariableCadence,
    frame_spec_for_matrix,
)
from kinovsr.processors.units import FrameUnit
from kinovsr.settings import Settings

pytestmark = pytest.mark.unit

TB = Fraction(1, 24000)


def _spec(cadence, nominal=None):
    return StreamSpec(
        frame=frame_spec_for_matrix(
            "bt709", full_range=False, geometry=Geometry(64, 48)),
        timeline=TimelineSpec(
            time_base=TB, cadence=cadence, nominal_cadence=nominal))


def _run(times_s, durations_s, fps):
    proc = CfrConformProcessor(ConformStageConfig(fps=Fraction(fps)))
    proc.prepare(_spec(VariableCadence.VFR, nominal=Fraction(30)),
                 PipelineContext(settings=Settings()))
    out = []
    units = [
        FrameUnit(payload=f"f{i}", pts=round(t / TB),
                  duration=round(d / TB))
        for i, (t, d) in enumerate(zip(times_s, durations_s, strict=True))
    ]
    for unit in units:
        out.extend(proc.process(unit, None))
    out.extend(proc.flush(None))
    return proc, out


def test_gaps_duplicate_and_grid_is_regenerated():
    # source slots 0, 1, 3, 4, 7 of a 30 fps grid (two gaps)
    times = [Fraction(s, 30) for s in (0, 1, 3, 4, 7)]
    durs = [Fraction(1, 30)] * 5
    proc, out = _run(times, durs, 30)
    # Nearest wins: slot 2 ties the (f1, f2) midpoint and rounds to the
    # earlier frame; slot 6 sits past the (f3, f4) midpoint -> f4.
    assert [u.payload for u in out] == [
        "f0", "f1", "f1", "f2", "f3", "f3", "f4", "f4"]
    assert [u.pts for u in out] == [round(Fraction(m, 30) / TB)
                                    for m in range(8)]
    assert proc._dups == 3 and proc._drops == 0
    line = proc.run_diagnostics()[0]
    assert "3 duplicated" in line and "0 dropped" in line


def test_jitter_burst_drops_the_nearer_loser():
    # two frames inside one slot's half-width: the farther one drops
    times = [Fraction(0), Fraction(1, 90), Fraction(1, 30), Fraction(2, 30)]
    durs = [Fraction(1, 90), Fraction(2, 90), Fraction(1, 30), Fraction(1, 30)]
    proc, out = _run(times, durs, 30)
    assert proc._drops == 1
    assert [u.payload for u in out][0] == "f0"


def test_non_increasing_times_are_refused():
    proc = CfrConformProcessor(ConformStageConfig(fps=Fraction(30)))
    proc.prepare(_spec(VariableCadence.VFR, nominal=Fraction(30)),
                 PipelineContext(settings=Settings()))
    a = FrameUnit(payload="a", pts=800, duration=800)
    list(proc.process(a, None))
    from kinovsr.processors.errors import MediaError

    with pytest.raises(MediaError, match="strictly increasing"):
        list(proc.process(a, None))


def test_produces_regenerates_uniform_timeline():
    spec = _spec(VariableCadence.VFR, nominal=Fraction(30))
    out = _produces(spec, ConformStageConfig(fps=None))
    assert out.timeline.cadence == Fraction(30)      # auto = nominal
    assert out.timeline.timestamp_policy is TimestampPolicy.REGENERATED
    assert out.timeline.nominal_cadence is None
    up = _produces(spec, ConformStageConfig(fps=Fraction(60)))
    assert up.timeline.cardinality is Cardinality.ONE_TO_MANY
    down = _produces(spec, ConformStageConfig(fps=Fraction(15)))
    assert down.timeline.cardinality is Cardinality.MANY_TO_ONE


def test_clock_jump_fanout_is_bounded():
    # A one-hour recorder discontinuity must refuse BEFORE flooding the
    # chain with ~100k duplicates, naming an in-tool path forward.
    from kinovsr.processors.errors import MediaError

    times = [Fraction(0), Fraction(1, 30), Fraction(3600)]
    durs = [Fraction(1, 30)] * 3
    with pytest.raises(MediaError, match="duplicates"):
        _run(times, durs, 30)
