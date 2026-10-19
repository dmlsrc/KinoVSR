"""CoreFoundation ownership in the HID temperature reader."""

import pytest

from kinovsr.native import sensors

pytestmark = pytest.mark.unit


class _Core:
    def __init__(self) -> None:
        self.released: list[int] = []

    def CFRelease(self, ref: int) -> None:
        self.released.append(ref)


class _IOKit:
    def __init__(self) -> None:
        self.events: list[int] = []

    def IOHIDServiceClientCopyEvent(self, service, kind, options, timestamp):
        del service, kind, options, timestamp
        event = 1000 + len(self.events)
        self.events.append(event)
        return event

    def IOHIDEventGetFloatValue(self, event, field):
        del event, field
        return 40.0


def test_every_copied_event_is_released():
    # IOHIDServiceClientCopyEvent hands over ownership. The reader never
    # released the events, which leaked about 144 bytes per sensor per sample.
    hid = object.__new__(sensors._HIDTemperatures)
    hid._core, hid._iokit = _Core(), _IOKit()
    hid._services = [(1, "cluster"), (2, "nand"), (3, "cluster")]

    for _ in range(2):
        assert hid.read() == {"cluster": 40.0, "nand": 40.0}
    assert len(hid._iokit.events) == 6
    assert hid._core.released == hid._iokit.events


def test_power_reasons_are_shares_of_the_time_the_gpu_ran():
    # PWRCTRL counts idle time as IDLE_OFF; a throttled GPU shows a limit reason
    # such as PPMTARGE in the rest.
    shares = sensors.power_reason_shares(
        {"IDLE_OFF": 500, "PERF": 300, "PPMTARGE": 200, "POWER": 0, "PRFBOOST": 0}
    )
    assert shares == {"PERF": 0.6, "PPMTARGE": 0.4}
    assert sensors.power_reason_shares({"IDLE_OFF": 10, "PERF": 0}) == {}


def test_gpu_power_comes_from_the_counter_that_moves_every_sample():
    # "GPU Energy" and "GPU0" count the same energy, but only "GPU Energy" moves
    # within a second; "GPU0" stands in where it is missing.
    assert sensors._engine("GPU Energy") == "gpu"
    assert sensors._engine("GPU0") == "gpu_parts"
    assert sensors._engine("PACC0_CPU") is None
    assert sensors._engine("CPU Energy") == "cpu"


def test_a_sampler_needs_at_least_one_group():
    with pytest.raises(ValueError, match="at least one IOReport group"):
        sensors.SensorSampler(groups=())


def test_system_power_reads_the_system_rail_alone():
    class _SMC:
        def __init__(self) -> None:
            self.keys: list[str] = []

        def read(self, key: str) -> float | None:
            self.keys.append(key)
            return 97.5

    power = object.__new__(sensors.SystemPower)
    power._smc = _SMC()
    assert power.read() == 97.5
    assert power._smc.keys == ["PSTR"]
