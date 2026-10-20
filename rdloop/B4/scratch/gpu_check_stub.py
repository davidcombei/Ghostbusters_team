"""
Scratch (implement_core): exercise b4wave.checks.check_gpu's lever-on branch with a stub post-DSP lever
(the core has no real lever yet). Runs on whatever device is visible (CPU in the implement step).
"""
import os
import sys
from pathlib import Path

SLS = Path(__file__).resolve().parents[3] / "SLS_setup"
os.chdir(SLS)
sys.path.insert(0, str(SLS))

import numpy as np  # noqa: E402

from b4wave import augment as A  # noqa: E402
from b4wave import checks as C  # noqa: E402
from utils.rtc_augment import finalize  # noqa: E402


def _lever__stub(self, out, sr):
    if self._get_rng4().random() < getattr(self.cfg, "stub_p", 0.0):
        self._count("stub_hit")
        return finalize(out * np.float32(0.5), out)
    return out


A.RTCAugmenterB4._lever__stub = _lever__stub
A.register(A.LeverSpec("_stub", "post_dsp", "aug_stub_p", "stub_p", meta_keys=("stub_hit",)))
C.check_gpu(C.BASE_CKPT, n_clips=2)
failed = [n for n, ok in C.RESULTS if not ok]
print("FAILED" if failed else "ALL PASS", failed)
sys.exit(1 if failed else 0)
