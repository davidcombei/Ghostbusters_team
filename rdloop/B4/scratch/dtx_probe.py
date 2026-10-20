"""Probe webrtcvad on synthetic signals (scratch, rdloop B4 dtx_gate). Run from SLS_setup/."""
import time

import numpy as np
import webrtcvad

SR = 16000


def scaled(x, dbfs):
    x = np.asarray(x, np.float64)
    return x / np.sqrt(np.mean(x ** 2)) * 10 ** (dbfs / 20)


def harmonic(n, f0=150.0, dbfs=-20.0, nh=20):
    t = np.arange(n) / SR
    return scaled(sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, nh + 1)), dbfs)


def decisions(x, mode, frame_ms):
    v = webrtcvad.Vad(mode)
    n = SR * frame_ms // 1000
    q = np.clip(np.round(np.clip(x, -1, 1) * 32767), -32768, 32767).astype(np.int16)
    m = len(q) // n
    return np.array([v.is_speech(q[i * n:(i + 1) * n].tobytes(), SR) for i in range(m)])


for name, sig in [("harmonic -20", harmonic(4 * SR)), ("harmonic -35", harmonic(4 * SR, dbfs=-35)),
                  ("harmonic f0=220 -26", harmonic(4 * SR, 220, -26)),
                  ("sine440 -20", scaled(np.sin(2 * np.pi * 440 * np.arange(4 * SR) / SR), -20)),
                  ("white -50", scaled(np.random.default_rng(0).standard_normal(4 * SR), -50)),
                  ("white -30", scaled(np.random.default_rng(0).standard_normal(4 * SR), -30)),
                  ("zeros", np.zeros(4 * SR))]:
    row = []
    for mode in (1, 2, 3):
        for f in (10, 20, 30):
            d = decisions(sig, mode, f)
            row.append(f"m{mode}f{f}:{d.mean():.2f}" + (f"(first0 {np.flatnonzero(~d)[:3].tolist()})" if (~d).any() and d.mean() > 0.5 else ""))
    print(f"{name:22s} " + " ".join(row))

# bursts: 0.5 s harmonic / 0.5 s gap, gap = zeros or low noise
for gap_name, gap_db in [("zeros", None), ("white -60", -60), ("white -50", -50), ("brown -50", "b-50")]:
    n_on, n_off = SR // 2, SR // 2
    rng = np.random.default_rng(1)
    parts = []
    for i in range(4):
        parts.append(harmonic(n_on))
        if gap_db is None:
            parts.append(np.zeros(n_off))
        elif isinstance(gap_db, str):
            w = np.cumsum(rng.standard_normal(n_off)); w -= np.linspace(w[0], w[-1], n_off)
            parts.append(scaled(w, -50))
        else:
            parts.append(scaled(rng.standard_normal(n_off), gap_db))
    x = np.concatenate(parts)
    for mode in (1, 3):
        for f in (10, 30):
            d = decisions(x, mode, f)
            per = len(d) // 8
            segs = [d[i * per:(i + 1) * per] for i in range(8)]
            on_miss = [int((~s).sum()) for s in segs[0::2]]
            off_speech = [int(s.sum()) for s in segs[1::2]]
            print(f"burst gap={gap_name:9s} m{mode} f{f}: on-frames non-speech {on_miss} | gap frames speech {off_speech} of {per}")

x = harmonic(60 * SR)
t0 = time.process_time()
for f in (10, 20, 30):
    decisions(x, 3, f)
print(f"60 s x 3 frame sizes: {time.process_time() - t0:.3f} CPU-s")
