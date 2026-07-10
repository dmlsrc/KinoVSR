"""Runtime-only environment contract (M2 acceptance).

NumPy is a dev-only dependency and PyAV is the optional ffmpeg extra: a
clean runtime environment must import kinovsr and render ``kinovsr
--help`` without either. The subprocess installs a meta-path blocker
BEFORE any import, which is the reliable way to prove absence (the test
process itself already has numpy loaded).
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

_BLOCKED_PROBE = r"""
import sys

BLOCKED = {"numpy", "av", "cv2"}

class Blocker:
    def find_spec(self, name, path=None, target=None):
        top = name.split(".")[0]
        if top in BLOCKED:
            raise ImportError(f"{name} blocked: runtime-only environment")
        return None

sys.meta_path.insert(0, Blocker())

import kinovsr
from kinovsr.cli.args import build_parser

text = build_parser().format_help()
assert "--upscale" in text
assert "--fastdvdnet-profile" in text
offenders = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
assert not offenders, f"blocked modules imported anyway: {offenders}"
print("runtime-only import ok")
"""


def test_import_and_help_without_numpy_av_cv2():
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCKED_PROBE],
        capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "runtime-only import ok" in proc.stdout
