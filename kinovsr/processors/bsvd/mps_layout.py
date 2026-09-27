"""Layout of the BSVD MPSGraph build: units, skip lines, and state groups.

Shared by :mod:`.mps`, which runs the graph, and :mod:`.mps_phases`, which
schedules it in phases; neither imports the other for these facts.
"""

_UNITS = 16
_LINES = 6
_SKIP_DEPTHS = (8, 8, 4, 8, 8, 4)
_STATE_GROUPS = (
    (0, 1, 6, 7),
    (2, 3, 4, 5),
    (8, 9, 14, 15),
    (10, 11, 12, 13),
)
_STATE_GROUP_OF = {
    unit: (group, slot)
    for group, units in enumerate(_STATE_GROUPS)
    for slot, unit in enumerate(units)
}
# Skip lines 1..5 push graph outputs; line 0 pushes the frame's own RGB
# head, which the host already holds.
_PUSH_OUTPUTS = ((1, "n0.x0"), (2, "n0.x1"), (3, "n0.out3"), (4, "n1.x0"), (5, "n1.x1"))


def _net_keys(prefix: str) -> dict[str, str]:
    c = f"{prefix}.%s.convblock"
    return {
        "inc0": f"{c % 'inc'}.0",
        "inc3": f"{c % 'inc'}.3",
        "d0": f"{c % 'downc0'}.0",
        "u0": f"{c % 'downc0'}.3.c1.net",
        "u1": f"{c % 'downc0'}.3.c2.net",
        "d1": f"{c % 'downc1'}.0",
        "u2": f"{c % 'downc1'}.3.c1.net",
        "u3": f"{c % 'downc1'}.3.c2.net",
        "u4": f"{c % 'upc2'}.0.c1.net",
        "u5": f"{c % 'upc2'}.0.c2.net",
        "up2": f"{c % 'upc2'}.1",
        "u6": f"{c % 'upc1'}.0.c1.net",
        "u7": f"{c % 'upc1'}.0.c2.net",
        "up1": f"{c % 'upc1'}.1",
        "out0": f"{c % 'outc'}.0",
        "out3": f"{c % 'outc'}.3",
    }
