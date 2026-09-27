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
