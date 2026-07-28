"""Boolean None-flow mirror of the BSVD network's fill and drain schedule.

The product :class:`~kinovsr.processors.bsvd.BSVD` net propagates ``None``
through its bi-buffer units and skip queues while the stream fills and
drains.  Accelerator backends run a static graph that always computes, so
they replay this mirror step by step and read off which unit inputs were
``None`` (gates), which centers primed (state-write masks), which skip
pushes and pops were real (ring maintenance), and whether a frame is
emitted.  The Core ML backend (``ane.py``) and the MPSGraph backend
(``mps.py``) both drive it; keep it backend-neutral.

Kept equivalent to the real network by test: the schedule is derived from
instrumented product classes in ``tests/processors/bsvd/test_ane.py`` and
compared for stream lengths 1..48.
"""

from __future__ import annotations


class StepRecord:
    """What one mirrored step observed, in schedule terms."""

    __slots__ = ("out_real", "unprimed", "primes", "drained", "pushes",
                 "pops")

    def __init__(self):
        self.out_real = False
        self.unprimed = [False] * 16   # center held None at entry
        self.primes = [False] * 16     # center went None -> real this step
        self.drained = [False] * 16    # unit input was None this step
        self.pushes = [False] * 6      # skip line received a real push
        self.pops = [False] * 6        # skip line surrendered a real pop


class NoneBiBuffer:
    """Boolean mirror of ``_BiBufferConv``'s None propagation."""

    def __init__(self, index: int):
        self.index = index
        self.center_real = None  # None: slot holds None; True: a real tensor

    def __call__(self, right_real: bool, record: StepRecord) -> bool:
        record.drained[self.index] = not right_real
        record.unprimed[self.index] = self.center_real is None
        if self.center_real is None:
            if right_real:
                record.primes[self.index] = True
                self.center_real = True
            return False
        self.center_real = True if right_real else None
        return True


class NoneSkip:
    """Boolean mirror of ``_MemSkip``."""

    def __init__(self, index: int):
        self.index = index
        self.items = 0

    def push(self, real: bool, record: StepRecord) -> None:
        record.pushes[self.index] = real
        if real:
            self.items += 1

    def pop(self, trigger_real: bool, record: StepRecord) -> bool:
        if not trigger_real or self.items == 0:
            return False
        self.items -= 1
        record.pops[self.index] = True
        return True


class NoneDenBlock:
    """Boolean mirror of ``_DenBlock.__call__``'s None propagation."""

    def __init__(self, block_index: int):
        base = block_index * 8
        self.units = [NoneBiBuffer(base + i) for i in range(8)]
        skip_base = block_index * 3
        self.skip1 = NoneSkip(skip_base)
        self.skip2 = NoneSkip(skip_base + 1)
        self.skip3 = NoneSkip(skip_base + 2)

    def __call__(self, x_real: bool, record: StepRecord) -> bool:
        self.skip1.push(x_real, record)
        x0 = x_real                                     # inc
        self.skip2.push(x0, record)
        x1 = self.units[1](self.units[0](x0, record), record)   # down0
        self.skip3.push(x1, record)
        x2 = self.units[3](self.units[2](x1, record), record)   # down1
        x2 = self.units[5](self.units[4](x2, record), record)   # up2 mem
        merged = x2 and self.skip3.pop(x2, record)      # none_add
        m = self.units[7](self.units[6](merged, record), record)  # up1 mem
        y = m and self.skip2.pop(m, record)             # none_add -> out conv
        return y and self.skip1.pop(y, record)          # none_minus


class NoneFlowNet:
    """Boolean mirror of the product ``BSVD`` net's None propagation.

    The write/gate/push/pop/emit schedules the accelerator paths need are
    read off this mirror, so a graph that always computes reproduces the
    product's fill and drain behavior exactly.
    """

    def __init__(self):
        self.blocks = [NoneDenBlock(0), NoneDenBlock(1)]

    def step(self, real: bool) -> StepRecord:
        record = StepRecord()
        record.out_real = self.blocks[1](
            self.blocks[0](real, record), record)
        return record


__all__ = ["StepRecord", "NoneBiBuffer", "NoneSkip", "NoneDenBlock",
           "NoneFlowNet"]
