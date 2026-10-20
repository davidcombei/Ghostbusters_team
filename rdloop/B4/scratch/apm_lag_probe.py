"""F1.2 probe: CPU vs wall of dsp_apm.measure() and of 4 wrtc_ns calls right after (OpenBLAS spin).
Run from SLS_setup/: python -B ../rdloop/B4/scratch/apm_lag_probe.py"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from b4wave import dsp_apm  # noqa: E402

t0, c0 = time.time(), time.process_time()
vals = dsp_apm.measure()
t1, c1 = time.time(), time.process_time()
dsp_apm.verify(vals)
x = np.random.default_rng(0).standard_normal(16000 * 4).astype(np.float32) * 0.05
c2 = time.process_time()
for lvl in (6, 12, 18, 21):
    dsp_apm.wrtc_ns(x, lvl)
c3 = time.process_time()
print(f"measure wall {t1 - t0:.2f}s cpu {c1 - c0:.2f}s | 4 wrtc_ns after: cpu {c3 - c2:.3f}s | aligned {vals['aligned']}")
