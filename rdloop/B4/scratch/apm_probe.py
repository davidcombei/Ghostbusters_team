"""Scratch probe (rdloop B4 apm_ns implement step): the real WebRTC NS on the plan's selftest signals."""
import numpy as np
from scipy.signal import welch
from pywebrtc_audio import NoiseSuppressor

SR, F, LAT, PRIME = 16000, 160, 96, 8000


def ns_run(x, level, align=True):
    q = np.clip(np.round(np.clip(x, -1, 1) * 32768.0), -32768, 32767).astype(np.int16)
    ns = NoiseSuppressor(sample_rate=SR, num_channels=1, level=level)
    p = q[:min(PRIME, len(q))]
    p = np.pad(p, (0, -len(p) % F))
    for i in range(0, len(p), F):
        ns.process(p[i:i + F])
    extra = LAT if align else 0
    n = len(q) + extra
    z = np.pad(q, (0, n - len(q) + (-n % F)))
    y = np.concatenate([ns.process(z[i:i + F]) for i in range(0, len(z), F)]).astype(np.float32) / 32768.0
    return y[extra:extra + len(q)]


def rms_db(x):
    return 10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20)


def harmonic(seconds=6.0, dbfs=-30.0):
    t = np.arange(int(seconds * SR)) / SR
    h = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 21))
    period, on = SR // 4, SR // 8
    ramp = int(0.010 * SR)
    env1 = np.zeros(period)
    env1[:on] = 1.0
    r = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
    env1[:ramp] = r
    env1[on - ramp:on] = r[::-1]
    env = np.tile(env1, int(np.ceil(len(t) / period)))[:len(t)]
    x = h * env
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (dbfs / 20)).astype(np.float32)


def white(seconds=6.0, dbfs=-30.0):
    x = np.random.default_rng(0).standard_normal(int(seconds * SR))
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (dbfs / 20)).astype(np.float32)


def band_att(x, y, lo, hi):
    f, px = welch(x, fs=SR, window="hann", nperseg=512, noverlap=256)
    _, py = welch(y, fs=SR, window="hann", nperseg=512, noverlap=256)
    m = (f >= lo) & (f <= hi)
    return 10 * np.log10(px[m].sum() / py[m].sum())


h, w = harmonic(), white()
for align in (True, False):
    print("align", align)
    for lvl in range(4):
        yh, yw = ns_run(h, lvl, align), ns_run(w, lvl, align)
        g = rms_db(yh[PRIME:]) - rms_db(h[PRIME:])
        rem = rms_db(w[PRIME:]) - rms_db(yw[PRIME:])
        lo = band_att(w[PRIME:], yw[PRIME:], 31.0, 94.0)
        mid = band_att(w[PRIME:], yw[PRIME:], 1000.0, 4000.0)
        print(f"  level {lvl}: harmonic gain {g:+.2f} dB; white removed {rem:.2f} dB; low-band att {lo:.2f} vs 1-4k {mid:.2f} -> diff {lo - mid:+.2f}")
