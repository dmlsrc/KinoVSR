"""ctypes bindings for IOReport: per-engine power, utilization and stalls.

Apple ships no public framework for per-engine power, and the third-party
tools that report it are thin wrappers over this same interface, so it is
bound directly here rather than taken as a dependency. `libIOReport` has no
on-disk dylib but resolves through the dyld shared cache, needs no root, and
the C API is plain CoreFoundation objects - the same situation that made
`vimage.py` a ctypes binding rather than a pyobjc one.

`/usr/bin/powermetrics` is the first-party alternative and reports the same
counters, but it requires root, which is awkward inside a bench or a test.

What this can and cannot see, measured on M1 Max / macOS 26.5 rather than
assumed, because both the channel names and the unit scaling are chip
specific:

* **Power** - reliable for every engine. The Energy Model group carries
  per-engine energy counters and power is their delta over a timed interval.
  Units are mixed within the one group (mJ for CPU/GPU/ANE/DRAM, uJ for PCIe,
  nJ for the duplicate "GPU Energy" channel), so the unit label is read per
  channel instead of assumed.
* **CPU and GPU utilization** - available, but derived rather than read.
  There is no usable "Utilization %" key: the `AGXAccelerator` IORegistry
  keys of that name exist and stay at zero under full load. Utilization comes
  from performance-state RESIDENCY - the fraction of an interval a unit spent
  outside its idle state - which is why it needs an interval and not a point
  read.
* **ANE utilization** - NOT available. `SoC Stats/ANE0` exposes `INACT/ACT`
  but reports `ACT` 100 percent of the time regardless of load, and the
  21-bucket percent histogram at `PMP/ANE0` never accumulates residency. ANE
  power is the working proxy: it reads 0.00 W idle against about 2.5 W while
  a model runs, and holds that separation with the GPU saturated.
* **Stalls** - the PMP group's percent histograms that do populate are memory
  subsystem stalls (DRAM, last-level cache, core block slot), not engine
  occupancy.

Sampling costs about 1.001x on a frame loop at one hertz: two sample calls
and a delta, touching no accelerator.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

_UTF8 = 0x08000100
_Ref = ctypes.c_void_p

# Energy Model mixes units within the single group.
_JOULES = {"mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9, "J": 1.0}
# Percent-named residency buckets come in both spellings: exact
# ("0%", "5%") in the PMP group, ranges ("0-9%") in GPU Stats.
_PERCENT_BUCKET = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?%\s*$")


@dataclass
class Reading:
    """One interval's worth of counters. Power in watts, the rest in percent."""

    interval_s: float = 0.0
    cpu_w: float = 0.0
    gpu_w: float = 0.0
    ane_w: float = 0.0
    dram_w: float = 0.0
    other_w: float = 0.0
    total_w: float = 0.0
    cpu_util: float | None = None
    gpu_util: float | None = None
    # ANE utilization is deliberately absent: no channel reports it. Use
    # `ane_w` instead, which separates cleanly between idle and running.
    stalls: dict[str, float] = field(default_factory=dict)


@lru_cache(maxsize=1)
def _libs() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    core = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    report = ctypes.CDLL(ctypes.util.find_library("IOReport") or "IOReport")
    for lib, name, restype, argtypes in (
        (core, "CFStringCreateWithCString", _Ref,
         [_Ref, ctypes.c_char_p, ctypes.c_uint32]),
        (core, "CFStringGetCString", ctypes.c_bool,
         [_Ref, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]),
        (core, "CFDictionaryGetValue", _Ref, [_Ref, _Ref]),
        (core, "CFArrayGetCount", ctypes.c_long, [_Ref]),
        (core, "CFArrayGetValueAtIndex", _Ref, [_Ref, ctypes.c_long]),
        (core, "CFRelease", None, [_Ref]),
        (report, "IOReportCopyAllChannels", _Ref,
         [ctypes.c_uint64, ctypes.c_uint64]),
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
    ):
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return core, report


def _cfstr(text: str) -> _Ref:
    core, _ = _libs()
    return core.CFStringCreateWithCString(None, text.encode(), _UTF8)


def _pystr(ref: _Ref) -> str:
    core, _ = _libs()
    if not ref:
        return ""
    buffer = ctypes.create_string_buffer(256)
    if not core.CFStringGetCString(ref, buffer, 256, _UTF8):
        return ""
    return buffer.value.decode()


def _engine(name: str) -> str:
    if name.startswith(("ECPU", "PCPU")):
        return "cpu"
    if name.startswith("GPU"):
        return "gpu"
    if name.startswith("ANE"):
        return "ane"
    if name.startswith(("DRAM", "DCS", "AMCC")):
        return "dram"
    return "other"


class IOReportSampler:
    """Subscribes once; each `sample()` returns the interval since the last.

    The first reading covers the time since construction, so discard it if
    that interval is not meaningful.
    """

    def __init__(self) -> None:
        core, report = _libs()
        self._core, self._report = core, report
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

    def sample(self) -> Reading:
        core, report = self._core, self._report
        current = report.IOReportCreateSamples(
            self._subscription, self._subscribed, None)
        elapsed = max(time.perf_counter() - self._stamp, 1e-9)
        delta = report.IOReportCreateSamplesDelta(self._previous, current, None)
        core.CFRelease(self._previous)
        self._previous, self._stamp = current, time.perf_counter()

        reading = Reading(interval_s=elapsed)
        watts = {"cpu": 0.0, "gpu": 0.0, "ane": 0.0, "dram": 0.0, "other": 0.0}
        cpu_busy, cpu_total = 0, 0

        items = core.CFDictionaryGetValue(delta, _cfstr("IOReportChannels"))
        for index in range(core.CFArrayGetCount(items)):
            item = core.CFArrayGetValueAtIndex(items, index)
            group = _pystr(report.IOReportChannelGetGroup(item))
            name = _pystr(report.IOReportChannelGetChannelName(item))

            if group == "Energy Model":
                # "GPU Energy" (nJ) duplicates GPU0 (mJ); count it once.
                if name == "GPU Energy":
                    continue
                unit = _pystr(report.IOReportChannelGetUnitLabel(item)).strip()
                scale = _JOULES.get(unit)
                if scale is None:
                    continue
                joules = report.IOReportSimpleGetIntegerValue(item, 0) * scale
                watts[_engine(name)] += joules / elapsed
                continue

            if report.IOReportChannelGetFormat(item) != 2:
                continue
            count = report.IOReportStateGetCount(item)
            states = {
                _pystr(report.IOReportStateGetNameForIndex(item, i)):
                    report.IOReportStateGetResidency(item, i)
                for i in range(count)
            }
            total = sum(states.values())
            if total <= 0:
                continue

            # GPU: fraction of the interval spent outside the OFF state.
            if group == "GPU Stats" and name == "GPUPH":
                reading.gpu_util = 100.0 * (total - states.get("OFF", 0)) / total
            # CPU: per-core channels carry an explicit IDLE state. Cluster
            # channels do not, so they are skipped rather than double counted.
            elif group == "CPU Stats" and "IDLE" in states:
                cpu_busy += total - states["IDLE"]
                cpu_total += total
            # PMP percent histograms that populate are memory stalls.
            elif group == "PMP" and any(
                    _PERCENT_BUCKET.match(k) for k in states if k):
                weighted = 0.0
                for label, residency in states.items():
                    match = _PERCENT_BUCKET.match(label or "")
                    if match:
                        low = int(match.group(1))
                        high = int(match.group(2)) if match.group(2) else low
                        weighted += ((low + high) / 2) * residency
                reading.stalls[name] = weighted / total

        core.CFRelease(delta)
        if cpu_total > 0:
            reading.cpu_util = 100.0 * cpu_busy / cpu_total
        reading.cpu_w = watts["cpu"]
        reading.gpu_w = watts["gpu"]
        reading.ane_w = watts["ane"]
        reading.dram_w = watts["dram"]
        reading.other_w = watts["other"]
        reading.total_w = sum(watts.values())
        return reading

    def close(self) -> None:
        if getattr(self, "_previous", None):
            self._core.CFRelease(self._previous)
            self._previous = None

    def __enter__(self) -> IOReportSampler:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
