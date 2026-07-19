"""Hardware sensors: per-engine power, utilization, temperature and fan speed.

Reads the same counters a desktop hardware monitor shows. Everything here is a
local read of this machine's own sensors; nothing is recorded or sent anywhere.

Apple ships no public framework for any of it, and the third-party monitors
that report it are wrappers over the same handful of system interfaces, so they
are bound directly here rather than taken as a dependency. None of them needs
root, unlike `/usr/bin/powermetrics`, which is what makes them usable from
inside a bench or a test. The same reasoning made `vimage.py` a ctypes binding
rather than a pyobjc one.

Four sources, because no single one covers the SoC:

* **IOReport** - per-engine energy counters (Energy Model) and performance
  state residency (CPU Stats, GPU Stats, PMP). Power is an energy delta over a
  timed interval; utilization is the fraction of an interval spent outside an
  idle state, which is why both need an interval and not a point read.
* **IOAccelerator** - `PerformanceStatistics`, the GPU utilization figure most
  monitors read. Kept as a second, independent path on the one number where
  two are cheaply available.
* **SMC** - fan speed, the per-cluster temperatures that give a clean CPU and
  GPU figure (`Tp`/`Te` and `Tg`), and whole-machine power off the system rail
  (`PSTR`), which is the only source here that sees past the SoC.
* **IOHIDEventSystem** - named die temperatures on the Apple vendor sensor
  page. The SMC's four-character keys carry no names, so this is the readable
  view, and the one where an ANE sensor would appear on a chip that has one.

Measured on M1 Max / macOS 26.5 rather than assumed, since channel names,
units and availability are all chip specific:

* Power is reliable for every engine, but the Energy Model channel list is a
  flattened TREE and summing it multiply counts - see `_ROOTS`. The SMC system
  rail is the check that catches this, since no sum of parts may exceed the
  whole machine. Units are also mixed within the one group (mJ for
  CPU/GPU/ANE/DRAM, uJ for PCIe, nJ for the duplicate GPU channel), so the
  unit label is read per channel.
* CPU and GPU utilization both work, by two independent routes that agree:
  performance-state residency (86 percent on a saturated GPU) and
  `PerformanceStatistics` (99 percent on the same load). The latter must be
  read through `IORegistryEntryCreateCFProperties`; the `ioreg` command line
  shows it as zero, which is a property of that tool and not of the counter.
  Residency is the primary because it is interval-scoped and covers the CPU
  too; the accelerator figure is carried alongside as a cross-check.
* **ANE utilization is not available.** `SoC Stats/ANE0` reports `ACT` 100
  percent of the time regardless of load and `PMP/ANE0`'s percent histogram
  never accumulates. ANE power is the working proxy: 0.00 W idle against about
  2.5 W while a model runs, holding that separation with the GPU saturated,
  which makes it a usable check that work really landed on the ANE.
* **ANE temperature is not available on this chip.** The Apple vendor
  temperature page exposes 64 sensors on M1 Max and none of them is the ANE.
  Later chips are reported to carry an "ANE MTR Temp Sensor"; `Reading.temps`
  is returned whole so such a sensor appears by name if the chip has one.
* **PCIe reads exactly zero, including under load.** The internal SSD really
  is NVMe over PCIe, so the counters ought to move, but "PCIe Port 0/1 Energy"
  stayed at 0 uJ across a 1.5 GB read that cost 5.2 W of CPU. The channels
  exist and are simply not populated on this chip. It is broken out rather
  than folded into `other_w` so that this stays visible instead of quietly
  inflating an unlabeled bucket.

A sample costs about 34 ms of CPU, nearly all of it IPC round trips rather
than work. Polled at one hertz from a background thread it measured 1.0002x on
a GPU-bound frame loop, because it runs on a core the loop was not using.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import struct
import time
from dataclasses import dataclass, field
from functools import cache, lru_cache

_UTF8 = 0x08000100
_Ref = ctypes.c_void_p
_CF_NUMBER_INT = 3
_CF_NUMBER_INT64 = 4

_GROUPS = ("Energy Model", "CPU Stats", "GPU Stats", "PMP")
_JOULES = {"mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9, "J": 1.0}
# Percent-named residency buckets appear in two spellings: exact ("0%", "5%")
# in PMP, ranges ("0-9%") in GPU Stats.
_PERCENT_BUCKET = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?%\s*$")
_STATE_FORMAT = 2

#: Sorts above every four-character key, so a failed read during the bisection
#: pushes the search left rather than corrupting the bound.
_AFTER_ALL_KEYS = "\uffff"

#: Whole-system rail and adapter input. Both tracked load cleanly here:
#: 7.3 W / 7.1 W idle against 61.4 W / 58.5 W with the GPU saturated.
_SMC_SYSTEM_POWER = "PSTR"
_SMC_DC_IN_POWER = "PDTR"

_SMC_KERNEL_INDEX = 2
_SMC_READ_BYTES = 5
_SMC_READ_KEY_INDEX = 8
_SMC_READ_KEY_INFO = 9

_HID_TEMPERATURE = 15
_HID_APPLE_VENDOR_PAGE = 0xFF00
_HID_TEMPERATURE_USAGE = 5


@dataclass
class Reading:
    """One interval. Power in watts, utilization and stalls in percent."""

    interval_s: float = 0.0
    cpu_w: float = 0.0
    gpu_w: float = 0.0
    ane_w: float = 0.0
    dram_w: float = 0.0
    pci_w: float = 0.0
    other_w: float = 0.0
    #: Sum of the SoC engine buckets above. This is the chip, not the machine:
    #: it excludes the display, the SSD and everything else on the board.
    total_w: float = 0.0
    #: Whole-machine power off the SMC system rail - the figure a hardware
    #: monitor shows as "Total Power". Unlike every other power field this is
    #: an INSTANTANEOUS reading, not an average over `interval_s`, so a single
    #: one taken just after a burst catches the rail mid-decay and can land
    #: below `total_w`. Compare medians over a sustained load, not samples.
    system_w: float | None = None
    #: Power drawn from the adapter, same instantaneous caveat.
    dc_in_w: float | None = None
    cpu_util: float | None = None
    gpu_util: float | None = None
    #: `PerformanceStatistics`, the conventional source. Agrees closely with
    #: `gpu_util` (99 against 98 percent on a saturated GPU) but is not
    #: interval-scoped, so it is a cross-check rather than the figure.
    gpu_util_accelerator: float | None = None
    #: There is deliberately no ANE utilization field: nothing reports one.
    #: Use `ane_w`.
    cpu_temp_c: float | None = None
    gpu_temp_c: float | None = None
    fan_rpm: tuple[float, ...] = ()
    #: Every named temperature sensor the chip exposes, so a sensor this
    #: machine lacks is visible on one that has it.
    temps: dict[str, float] = field(default_factory=dict)
    stalls: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------
# CoreFoundation, IOReport and IOKit plumbing
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _libs() -> tuple[ctypes.CDLL, ctypes.CDLL, ctypes.CDLL]:
    core = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    report = ctypes.CDLL(ctypes.util.find_library("IOReport") or "IOReport")
    iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
    for lib, name, restype, argtypes in (
        (core, "CFStringCreateWithCString", _Ref,
         [_Ref, ctypes.c_char_p, ctypes.c_uint32]),
        (core, "CFStringGetCString", ctypes.c_bool,
         [_Ref, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]),
        (core, "CFDictionaryGetValue", _Ref, [_Ref, _Ref]),
        (core, "CFDictionaryCreateMutable", _Ref,
         [_Ref, ctypes.c_long, _Ref, _Ref]),
        (core, "CFDictionarySetValue", None, [_Ref, _Ref, _Ref]),
        (core, "CFNumberCreate", _Ref, [_Ref, ctypes.c_int, _Ref]),
        (core, "CFNumberGetValue", ctypes.c_bool, [_Ref, ctypes.c_int, _Ref]),
        (core, "CFArrayGetCount", ctypes.c_long, [_Ref]),
        (core, "CFArrayGetValueAtIndex", _Ref, [_Ref, ctypes.c_long]),
        (core, "CFRelease", None, [_Ref]),
        (report, "IOReportCopyAllChannels", _Ref,
         [ctypes.c_uint64, ctypes.c_uint64]),
        (report, "IOReportCopyChannelsInGroup", _Ref,
         [_Ref, _Ref, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]),
        (report, "IOReportMergeChannels", None, [_Ref, _Ref, _Ref]),
        (report, "IOReportCreateSubscription", _Ref,
         [ctypes.c_void_p, _Ref, ctypes.POINTER(_Ref), ctypes.c_uint64, _Ref]),
        (report, "IOReportCreateSamples", _Ref, [_Ref, _Ref, _Ref]),
        (report, "IOReportCreateSamplesDelta", _Ref, [_Ref, _Ref, _Ref]),
        (report, "IOReportChannelGetGroup", _Ref, [_Ref]),
        (report, "IOReportChannelGetChannelName", _Ref, [_Ref]),
        (report, "IOReportChannelGetUnitLabel", _Ref, [_Ref]),
        (report, "IOReportChannelGetFormat", ctypes.c_int32, [_Ref]),
        (report, "IOReportSimpleGetIntegerValue", ctypes.c_int64,
         [_Ref, ctypes.c_int]),
        (report, "IOReportStateGetCount", ctypes.c_int32, [_Ref]),
        (report, "IOReportStateGetNameForIndex", _Ref, [_Ref, ctypes.c_int32]),
        (report, "IOReportStateGetResidency", ctypes.c_int64,
         [_Ref, ctypes.c_int32]),
        (iokit, "IOServiceMatching", _Ref, [ctypes.c_char_p]),
        (iokit, "IOServiceGetMatchingService", ctypes.c_uint32,
         [ctypes.c_uint32, _Ref]),
        (iokit, "IOServiceGetMatchingServices", ctypes.c_int,
         [ctypes.c_uint32, _Ref, ctypes.POINTER(ctypes.c_uint32)]),
        (iokit, "IOIteratorNext", ctypes.c_uint32, [ctypes.c_uint32]),
        (iokit, "IOObjectRelease", ctypes.c_int, [ctypes.c_uint32]),
        (iokit, "IORegistryEntryCreateCFProperties", ctypes.c_int,
         [ctypes.c_uint32, ctypes.POINTER(_Ref), _Ref, ctypes.c_uint32]),
        (iokit, "IOServiceOpen", ctypes.c_int,
         [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
          ctypes.POINTER(ctypes.c_uint32)]),
        (iokit, "IOConnectCallStructMethod", ctypes.c_int,
         [ctypes.c_uint32, ctypes.c_uint32, _Ref, ctypes.c_size_t, _Ref,
          ctypes.POINTER(ctypes.c_size_t)]),
        (iokit, "IOHIDEventSystemClientCreate", _Ref, [_Ref]),
        (iokit, "IOHIDEventSystemClientSetMatching", ctypes.c_int,
         [_Ref, _Ref]),
        (iokit, "IOHIDEventSystemClientCopyServices", _Ref, [_Ref]),
        (iokit, "IOHIDServiceClientCopyProperty", _Ref, [_Ref, _Ref]),
        (iokit, "IOHIDServiceClientCopyEvent", _Ref,
         [_Ref, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]),
        (iokit, "IOHIDEventGetFloatValue", ctypes.c_double,
         [_Ref, ctypes.c_int32]),
    ):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return core, report, iokit


@cache
def _cfstr(text: str) -> _Ref:
    """Cached: these are immortal constants, and creating one per sample in a
    polling loop leaks a CFString per call."""
    core, _, _ = _libs()
    return core.CFStringCreateWithCString(None, text.encode(), _UTF8)


_STRING_BUFFER = ctypes.create_string_buffer(256)


def _pystr(ref: _Ref) -> str:
    core, _, _ = _libs()
    if not ref:
        return ""
    if not core.CFStringGetCString(ref, _STRING_BUFFER, 256, _UTF8):
        return ""
    return _STRING_BUFFER.value.decode()


# Energy Model is a TREE, not a flat list, which is easy to miss because it is
# presented flat. Measured under a CPU-saturating load: "CPU Energy" 5.00 W
# contains "PACC0_CPU" 4.95 (the performance cluster) contains "PACC0_CPU0/1/2"
# 4.73 (the cores) which "PCPUDTL*" mirrors again per core - four views of the
# same watts. "GPU Energy" (nJ) likewise duplicates "GPU0" (mJ). Summing the
# channels therefore multiply counts; only these roots are taken, and every
# channel matching `_CHILD` is a descendant of one and deliberately skipped.
_ROOTS = (
    (re.compile(r"^CPU Energy$"), "cpu"),
    (re.compile(r"^GPU\d+$"), "gpu"),
    (re.compile(r"^ANE\d+$"), "ane"),
    # Memory is a tree too: "AMCC0" 11.42 W is the parent of "DRAM0" 6.59 plus
    # "DCS0" 4.85. Adding all three put the SoC sum at 65.9 W against an SMC
    # system total of 61.4 W for the whole machine, display included, which is
    # how the double count was caught - see `system_w`.
    (re.compile(r"^(DRAM|DCS)\d+$"), "dram"),
    (re.compile(r"^(PCIe|apciec)"), "pci"),
)
_CHILD = re.compile(r"^(PACC|EACC|PCPU|ECPU|AMCC)|^GPU Energy$")
#: Cluster aggregates, summed only as a fallback where no "CPU Energy" root
#: exists. Their sum matched the root to within 1 percent here.
_CPU_CLUSTER = re.compile(r"^(PACC\d*_CPU|EACC_CPU)$")


def _engine(channel: str) -> str | None:
    """Bucket for a channel, or `None` when it is a child already counted."""
    for pattern, bucket in _ROOTS:
        if pattern.match(channel):
            return bucket
    if _CHILD.match(channel):
        return None
    return "other"


# --------------------------------------------------------------------------
# SMC - fan speed
# --------------------------------------------------------------------------

class _SMCVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint8), ("minor", ctypes.c_uint8),
                ("build", ctypes.c_uint8), ("reserved", ctypes.c_uint8),
                ("release", ctypes.c_uint16)]


class _SMCLimit(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint16), ("length", ctypes.c_uint16),
                ("cpuPLimit", ctypes.c_uint32), ("gpuPLimit", ctypes.c_uint32),
                ("memPLimit", ctypes.c_uint32)]


class _SMCKeyInfo(ctypes.Structure):
    _fields_ = [("dataSize", ctypes.c_uint32), ("dataType", ctypes.c_uint32),
                ("dataAttributes", ctypes.c_uint8)]


class _SMCKeyData(ctypes.Structure):
    """80 bytes. Swift ports of this struct usually carry an explicit padding
    field that the C ABI supplies implicitly, which yields 84 and fails the
    call with `0xe00002c2`."""

    _fields_ = [("key", ctypes.c_uint32), ("vers", _SMCVersion),
                ("pLimitData", _SMCLimit), ("keyInfo", _SMCKeyInfo),
                ("result", ctypes.c_uint8), ("status", ctypes.c_uint8),
                ("data8", ctypes.c_uint8), ("data32", ctypes.c_uint32),
                ("bytes", ctypes.c_uint8 * 32)]


class _SMC:
    """Minimal `AppleSMC` client. Needs no root."""

    def __init__(self) -> None:
        _, _, iokit = _libs()
        self._iokit = iokit
        self._connection = ctypes.c_uint32(0)
        self._key_cache: dict[str, list[str]] = {}
        self._temp_keys: dict[str, list[str]] | None = None
        self._info_cache: dict[str, tuple[int, int]] = {}
        device = iokit.IOServiceGetMatchingService(
            0, iokit.IOServiceMatching(b"AppleSMC"))
        if device:
            task = ctypes.c_uint32.in_dll(ctypes.CDLL(None), "mach_task_self_")
            iokit.IOServiceOpen(device, task, 0, ctypes.byref(self._connection))
            iokit.IOObjectRelease(device)

    def _call(self, request: _SMCKeyData) -> tuple[int, _SMCKeyData]:
        response = _SMCKeyData()
        size = ctypes.c_size_t(ctypes.sizeof(_SMCKeyData))
        code = self._iokit.IOConnectCallStructMethod(
            self._connection, _SMC_KERNEL_INDEX, ctypes.byref(request),
            ctypes.sizeof(_SMCKeyData), ctypes.byref(response),
            ctypes.byref(size))
        return code, response

    def read(self, key: str) -> float | None:
        if not self._connection.value or len(key) != 4:
            return None
        request = _SMCKeyData()
        request.key = struct.unpack(">I", key.encode())[0]
        # A key's size and type never change, so the describe call is made
        # once per key; polling then costs one round trip instead of two.
        described = self._info_cache.get(key)
        if described is None:
            request.data8 = _SMC_READ_KEY_INFO
            code, info = self._call(request)
            if code != 0 or not info.keyInfo.dataSize:
                return None
            described = (info.keyInfo.dataSize, info.keyInfo.dataType)
            self._info_cache[key] = described
        size, data_type = described
        request.keyInfo.dataSize = size
        request.data8 = _SMC_READ_BYTES
        code, payload = self._call(request)
        if code != 0:
            return None
        raw = bytes(payload.bytes)[:size]
        kind = struct.pack(">I", data_type).decode(errors="replace")
        if kind == "flt " and len(raw) == 4:
            return float(struct.unpack("<f", raw)[0])
        if kind == "sp78" and len(raw) == 2:
            return struct.unpack(">h", raw)[0] / 256.0
        if kind.startswith("ui"):
            return float(int.from_bytes(raw, "big"))
        return None

    def fan_rpm(self) -> tuple[float, ...]:
        count = self.read("FNum")
        if not count:
            return ()
        speeds = []
        for index in range(int(count)):
            value = self.read(f"F{index}Ac")
            if value is not None:
                speeds.append(value)
        return tuple(speeds)

    def _key_at(self, index: int) -> str | None:
        request = _SMCKeyData()
        request.data8 = _SMC_READ_KEY_INDEX
        request.data32 = index
        code, response = self._call(request)
        if code != 0:
            return None
        return struct.pack(">I", response.key).decode(errors="replace")

    def _keys(self, prefix: str) -> list[str]:
        """Keys under `prefix`, found by binary search rather than a full pass.

        The SMC returns its roughly 2100 keys in sorted order, so a prefix is
        one contiguous run. Enumerating all of them to find it costs about a
        second; bisecting to the run costs a few milliseconds.
        """
        if prefix in self._key_cache:
            return self._key_cache[prefix]
        found: list[str] = []
        if self._connection.value:
            count = int(self.read("#KEY") or 0)
            low, high = 0, count
            while low < high:
                middle = (low + high) // 2
                if (self._key_at(middle) or _AFTER_ALL_KEYS) < prefix:
                    low = middle + 1
                else:
                    high = middle
            for index in range(low, count):
                key = self._key_at(index)
                if key is None or not key.startswith(prefix):
                    break
                found.append(key)
        self._key_cache[prefix] = found
        return found

    def engine_temps(self) -> tuple[float | None, float | None]:
        """Hottest CPU and GPU sensor, in degrees Celsius.

        `Tp` is the performance cluster and `Te` the efficiency cluster; `Tg`
        is the GPU. Discovered by enumeration rather than hardcoded, since the
        populated suffixes differ per chip (M1 Max exposes 30 `Tp`, 8 `Tg`).
        """
        if self._temp_keys is None:
            temperature = self._keys("T")
            self._temp_keys = {
                "cpu": [k for k in temperature if k[:2] in ("Tp", "Te")],
                "gpu": [k for k in temperature if k[:2] == "Tg"],
            }
        hottest = []
        for group in ("cpu", "gpu"):
            values = [v for v in (self.read(k) for k in self._temp_keys[group])
                      if v is not None and 0.0 < v < 150.0]
            hottest.append(max(values) if values else None)
        return hottest[0], hottest[1]


# --------------------------------------------------------------------------
# IOHIDEventSystem - named die temperatures
# --------------------------------------------------------------------------

class _HIDTemperatures:
    """Apple vendor temperature sensor page. Services are matched once."""

    def __init__(self) -> None:
        core, _, iokit = _libs()
        self._core, self._iokit = core, iokit
        self._services: list[tuple[_Ref, str]] = []
        client = iokit.IOHIDEventSystemClientCreate(None)
        if not client:
            return
        matching = core.CFDictionaryCreateMutable(None, 0, None, None)
        for name, value in (("PrimaryUsagePage", _HID_APPLE_VENDOR_PAGE),
                            ("PrimaryUsage", _HID_TEMPERATURE_USAGE)):
            number = ctypes.c_int32(value)
            core.CFDictionarySetValue(
                matching, _cfstr(name),
                core.CFNumberCreate(None, _CF_NUMBER_INT, ctypes.byref(number)))
        iokit.IOHIDEventSystemClientSetMatching(client, matching)
        services = iokit.IOHIDEventSystemClientCopyServices(client)
        if not services:
            return
        # Names are fixed for the life of the service, so they are resolved
        # once here rather than per sample.
        for index in range(core.CFArrayGetCount(services)):
            service = core.CFArrayGetValueAtIndex(services, index)
            name = _pystr(iokit.IOHIDServiceClientCopyProperty(
                service, _cfstr("Product")))
            if name:
                self._services.append((service, name))

    def read(self) -> dict[str, float]:
        """Sensor name to degrees Celsius. Several sensors share a name; the
        hottest wins, since these are per-cluster duplicates."""
        out: dict[str, float] = {}
        for service, name in self._services:
            event = self._iokit.IOHIDServiceClientCopyEvent(
                service, _HID_TEMPERATURE, 0, 0)
            if not event:
                continue
            value = float(self._iokit.IOHIDEventGetFloatValue(
                event, _HID_TEMPERATURE << 16))
            if 0.0 < value < 150.0:
                out[name] = max(out.get(name, 0.0), value)
        return out


def _accelerator_utilization() -> float | None:
    """`PerformanceStatistics`, the path most GPU monitors take."""
    core, _, iokit = _libs()
    iterator = ctypes.c_uint32(0)
    if iokit.IOServiceGetMatchingServices(
            0, iokit.IOServiceMatching(b"IOAccelerator"),
            ctypes.byref(iterator)) != 0:
        return None
    best: float | None = None
    while True:
        entry = iokit.IOIteratorNext(iterator)
        if not entry:
            break
        properties = _Ref()
        if iokit.IORegistryEntryCreateCFProperties(
                entry, ctypes.byref(properties), None, 0) == 0 and properties:
            stats = core.CFDictionaryGetValue(
                properties, _cfstr("PerformanceStatistics"))
            if stats:
                for key in ("Device Utilization %", "GPU Activity(%)"):
                    value = core.CFDictionaryGetValue(stats, _cfstr(key))
                    if not value:
                        continue
                    number = ctypes.c_int64(0)
                    if core.CFNumberGetValue(
                            value, _CF_NUMBER_INT64, ctypes.byref(number)):
                        best = max(best or 0.0, float(number.value))
                    break
            core.CFRelease(properties)
        iokit.IOObjectRelease(entry)
    iokit.IOObjectRelease(iterator)
    return best


# --------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------

class SensorSampler:
    """Subscribes once; each `sample()` covers the interval since the last.

    The first reading covers the time since construction, so discard it when
    that interval is not meaningful.
    """

    def __init__(self, *, temperatures: bool = True, fans: bool = True,
                 named_sensors: bool = False) -> None:
        """`named_sensors` fills `Reading.temps` from the HID sensor page. It
        is off by default because it costs about 70 ms a sample - one IPC
        round trip per sensor, 64 of them - against 9 ms for the SMC pair that
        `cpu_temp_c` and `gpu_temp_c` come from. Turn it on to inspect a chip,
        not to poll one.
        """
        core, report, _ = _libs()
        self._core, self._report = core, report
        # Scoped to the groups actually read. Subscribing to every channel on
        # the system instead costs about 140 ms a sample, since each one is
        # walked and its name marshalled back out of CoreFoundation.
        self._channels = None
        for group in _GROUPS:
            channels = report.IOReportCopyChannelsInGroup(
                _cfstr(group), None, 0, 0, 0)
            if not channels:
                continue
            if self._channels is None:
                self._channels = channels
            else:
                report.IOReportMergeChannels(self._channels, channels, None)
                core.CFRelease(channels)
        if self._channels is None:
            self._channels = report.IOReportCopyAllChannels(0, 0)
        subscribed = _Ref()
        self._subscription = report.IOReportCreateSubscription(
            None, self._channels, ctypes.byref(subscribed), 0, None)
        if not self._subscription:
            raise RuntimeError("IOReportCreateSubscription failed")
        self._subscribed = subscribed
        self._previous = report.IOReportCreateSamples(
            self._subscription, self._subscribed, None)
        self._stamp = time.perf_counter()
        self._temperatures = temperatures
        self._smc = _SMC() if (fans or temperatures) else None
        self._hid = _HIDTemperatures() if named_sensors else None

    def sample(self) -> Reading:
        core, report = self._core, self._report
        current = report.IOReportCreateSamples(
            self._subscription, self._subscribed, None)
        elapsed = max(time.perf_counter() - self._stamp, 1e-9)
        delta = report.IOReportCreateSamplesDelta(self._previous, current, None)
        core.CFRelease(self._previous)
        self._previous, self._stamp = current, time.perf_counter()

        reading = Reading(interval_s=elapsed)
        watts = {"cpu": 0.0, "gpu": 0.0, "ane": 0.0, "dram": 0.0,
                 "pci": 0.0, "other": 0.0}
        cpu_busy = cpu_total = 0
        cpu_clusters = 0.0

        items = core.CFDictionaryGetValue(delta, _cfstr("IOReportChannels"))
        for index in range(core.CFArrayGetCount(items)):
            item = core.CFArrayGetValueAtIndex(items, index)
            group = _pystr(report.IOReportChannelGetGroup(item))
            channel = _pystr(report.IOReportChannelGetChannelName(item))

            if group == "Energy Model":
                unit = _pystr(report.IOReportChannelGetUnitLabel(item)).strip()
                scale = _JOULES.get(unit)
                if scale is None:
                    continue
                joules = report.IOReportSimpleGetIntegerValue(item, 0) * scale
                if _CPU_CLUSTER.match(channel):
                    cpu_clusters += joules / elapsed
                bucket = _engine(channel)
                if bucket is not None:
                    watts[bucket] += joules / elapsed
                continue

            if report.IOReportChannelGetFormat(item) != _STATE_FORMAT:
                continue
            states = {
                _pystr(report.IOReportStateGetNameForIndex(item, i)):
                    report.IOReportStateGetResidency(item, i)
                for i in range(report.IOReportStateGetCount(item))
            }
            total = sum(states.values())
            if total <= 0:
                continue

            if group == "GPU Stats" and channel == "GPUPH":
                reading.gpu_util = 100.0 * (total - states.get("OFF", 0)) / total
            elif group == "CPU Stats" and "IDLE" in states:
                # Per-core channels only. Cluster channels carry no IDLE state,
                # so this cannot double count them.
                cpu_busy += total - states["IDLE"]
                cpu_total += total
            elif group == "PMP" and any(
                    _PERCENT_BUCKET.match(k) for k in states if k):
                weighted = 0.0
                for label, residency in states.items():
                    match = _PERCENT_BUCKET.match(label or "")
                    if match:
                        low = int(match.group(1))
                        high = int(match.group(2)) if match.group(2) else low
                        weighted += ((low + high) / 2) * residency
                reading.stalls[channel] = weighted / total

        core.CFRelease(delta)
        if cpu_total > 0:
            reading.cpu_util = 100.0 * cpu_busy / cpu_total
        reading.gpu_util_accelerator = _accelerator_utilization()
        if watts["cpu"] == 0.0:
            watts["cpu"] = cpu_clusters
        reading.cpu_w = watts["cpu"]
        reading.gpu_w = watts["gpu"]
        reading.ane_w = watts["ane"]
        reading.dram_w = watts["dram"]
        reading.pci_w = watts["pci"]
        reading.other_w = watts["other"]
        reading.total_w = sum(watts.values())

        if self._hid is not None:
            reading.temps = self._hid.read()
        if self._smc is not None:
            if self._temperatures:
                reading.cpu_temp_c, reading.gpu_temp_c = \
                    self._smc.engine_temps()
            reading.fan_rpm = self._smc.fan_rpm()
            reading.system_w = self._smc.read(_SMC_SYSTEM_POWER)
            reading.dc_in_w = self._smc.read(_SMC_DC_IN_POWER)
        return reading

    def close(self) -> None:
        if getattr(self, "_previous", None):
            self._core.CFRelease(self._previous)
            self._previous = None

    def __enter__(self) -> SensorSampler:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
