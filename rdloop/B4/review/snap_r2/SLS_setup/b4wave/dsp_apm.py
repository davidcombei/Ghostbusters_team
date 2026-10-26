"""
WebRTC noise suppression for the B4 lever `apm_ns` (rdloop/B4/plan_agreed.md section 2.1).

One pass of pywebrtc_audio's standalone `NoiseSuppressor` (16 kHz mono, NS only: no high-pass
filter, no AGC, no AEC). The op name is `wrtc_ns`. The suppression target is given in dB and mapped
to the backend level (LEVEL_DB: 6 / 12 / 18 / 21 dB = levels 0-3 = WebRTC kLow..kVeryHigh).

    wrtc_ns(audio, level_db, sr=16000) -> float32, same length
        clip to [-1, 1] -> int16; a fresh suppressor per call, primed on the first min(0.5 s, len)
        (that output is discarded); then the whole clip from the start in 10 ms int16 frames; back to
        float32, trimmed to the input length. The caller passes the result through finalize().

    measure() / verify(values) / selftest()
        the start-up selftest with named asserts (plan O2.2): [shape_finite], [no_gain] (one-sided),
        [monotone], [no_hpf], plus [aligned] (see the latency note). verify() raises
        AssertionError("apm selftest [<name>] level=<dB>: <measured> (<tolerance>)") on the first failing
        assert; checks apm prints every measured value, also when it passes.

Latency. WebRTC NS works on 10 ms frames with a 256-point FFT, so its output lags the input by
96 samples (6 ms) at 16 kHz (measured at every level, rdloop/B4/scratch/apm_probe.py). wrtc_ns
compensates it: the clip is run with >= 96 extra zero samples at the end and the first 96 output
samples are dropped, so the output is sample-aligned with the input. Without that, the first 6 ms
of every processed clip would be the primed suppressor's synthesis tail of the priming segment and
the last 6 ms of the clip would be lost. [aligned] checks the lag is 0 at every level.

pywebrtc_audio is imported lazily (inside the functions), only when the lever is on.
"""

import numpy as np

SR = 16000
FRAME = 160                     # 10 ms at 16 kHz: the suppressor's frame
LATENCY = 96                    # algorithmic delay of WebRTC NS at 16 kHz (256-pt FFT - 160-sample frame)
PRIME_SECONDS = 0.5
LEVEL_DB = {6: 0, 12: 1, 18: 2, 21: 3}      # dB target -> pywebrtc_audio NoiseSuppressor level
BACKEND = "pywebrtc_audio.NoiseSuppressor"

# selftest tolerances (plan section 2.1)
NO_GAIN_TOL_DB = 0.5            # output RMS <= input RMS + 0.5 dB (one-sided)
MONOTONE_TOL_DB = 1.0           # removed(level k+1) >= removed(level k) - 1 dB
NO_HPF_TOL_DB = 3.0             # |att(31.25-93.75 Hz) - att(1-4 kHz)| <= 3 dB
WELCH_N = 512


def _new_suppressor(level_db):
    """A fresh standalone NoiseSuppressor (no HPF / AGC / AEC: those exist only on AudioProcessor)."""
    from pywebrtc_audio import NoiseSuppressor
    return NoiseSuppressor(sample_rate=SR, num_channels=1, level=LEVEL_DB[level_db])


def _process_frame(ns, frame):
    """The binding's frame call: one 10 ms int16 frame in, one int16 frame out."""
    return ns.process(frame)


def _run(ns, q):
    """`q` (int16, length a multiple of FRAME) through `ns`, frame by frame."""
    out = np.empty_like(q)
    for i in range(0, len(q), FRAME):
        out[i:i + FRAME] = _process_frame(ns, q[i:i + FRAME])
    return out


def _pad_frames(q, extra=0):
    n = len(q) + extra
    return np.pad(q, (0, n - len(q) + (-n % FRAME)))


def check_level(level_db):
    if level_db not in LEVEL_DB:
        raise ValueError(f"wrtc_ns level {level_db!r} dB not in {sorted(LEVEL_DB)}")
    return int(level_db)


def wrtc_ns(audio, level_db, sr=SR):
    """One WebRTC NS pass over a mono clip; float32 output of the input's length (see module doc)."""
    level_db = check_level(level_db)
    if sr != SR:
        raise ValueError(f"wrtc_ns runs at {SR} Hz, got {sr}")
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return x.copy()
    q = np.clip(np.round(np.clip(x, -1.0, 1.0) * 32768.0), -32768, 32767).astype(np.int16)
    ns = _new_suppressor(level_db)
    n_prime = min(int(PRIME_SECONDS * SR), len(q))
    _run(ns, _pad_frames(q[:n_prime]))                      # prime the noise estimate; output discarded
    y = _run(ns, _pad_frames(q, LATENCY))
    return (y[LATENCY:LATENCY + len(q)].astype(np.float32) / np.float32(32768.0))


# ------------------------------------------------------------------------------------------ #
# selftest (plan section 2.1, O2.2): deterministic 16 kHz signals; every level on a fresh,
# primed suppressor (wrtc_ns); every measurement skips the primed first 0.5 s
# ------------------------------------------------------------------------------------------ #
def _scaled(x, dbfs):
    x = np.asarray(x, dtype=np.float64)
    return (x / np.sqrt(np.mean(x ** 2)) * 10.0 ** (dbfs / 20.0)).astype(np.float32)


def harmonic_signal(seconds=6.0, dbfs=-30.0):
    """150 Hz harmonic complex (20 harmonics, amplitudes 1/k) under a 4 Hz on / off envelope
    (50 % duty, 10 ms raised-cosine ramps), scaled to `dbfs` RMS."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    h = sum(np.sin(2 * np.pi * 150.0 * k * t) / k for k in range(1, 21))
    period, on, ramp = SR // 4, SR // 8, int(0.010 * SR)
    cell = np.zeros(period)
    cell[:on] = 1.0
    r = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
    cell[:ramp] = r
    cell[on - ramp:on] = r[::-1]
    env = np.tile(cell, -(-n // period))[:n]
    return _scaled(h * env, dbfs)


def white_signal(seconds=6.0, dbfs=-30.0, n=None):
    """White noise from default_rng(0), `n` samples (default seconds * SR), scaled to `dbfs` RMS."""
    return _scaled(np.random.default_rng(0).standard_normal(int(seconds * SR) if n is None else int(n)), dbfs)


def _rms_db(x):
    return float(10.0 * np.log10(float(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-20))


def _band_att_db(x, y, lo_hz, hi_hz):
    """Attenuation in dB of y vs x in [lo_hz, hi_hz]: Welch PSDs (512-pt Hann, 50 % overlap, mean over
    frames), band powers summed over the bins."""
    from scipy.signal import welch
    f, px = welch(np.asarray(x, np.float64), fs=SR, window="hann", nperseg=WELCH_N, noverlap=WELCH_N // 2)
    _, py = welch(np.asarray(y, np.float64), fs=SR, window="hann", nperseg=WELCH_N, noverlap=WELCH_N // 2)
    m = (f >= lo_hz) & (f <= hi_hz)
    return float(10.0 * np.log10(float(px[m].sum()) / max(float(py[m].sum()), 1e-30)))


def _lag(x, y, max_lag=200):
    """Lag (samples) of y relative to x maximising their correlation, in [-max_lag, max_lag].
    Sums of elementwise products, not np.dot: the dot product wakes numpy's multi-threaded OpenBLAS,
    whose spinning threads are billed to the process (review F1.2; as dsp_dtx.lpc)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    core = slice(max_lag, len(x) - max_lag)
    xc = x[core]
    best, best_v = 0, -np.inf
    for d in range(-max_lag, max_lag + 1):
        v = float(np.sum(xc * y[core.start + d:core.stop + d]))
        if v > best_v:
            best, best_v = d, v
    return best


def measure(levels=tuple(LEVEL_DB)):
    """Run the selftest signals at every level; return the measured values (no asserts)."""
    skip = int(PRIME_SECONDS * SR)
    sig = {"harmonic": harmonic_signal(), "white": white_signal(),
           "short": white_signal(n=1234, dbfs=-20.0)}           # < 0.5 s and not a multiple of 10 ms
    lo_bins = (SR / WELCH_N * 1, SR / WELCH_N * 3)                  # Welch bins 1-3: 31.25-93.75 Hz (DC excluded)
    vals = {"levels": list(levels), "shape_finite": {}, "no_gain": {}, "monotone": {}, "no_hpf": {}, "aligned": {}}
    for lvl in levels:
        out = {k: wrtc_ns(v, lvl) for k, v in sig.items()}
        vals["shape_finite"][lvl] = {k: (len(out[k]) == len(v), bool(np.all(np.isfinite(out[k]))))
                                     for k, v in sig.items()}
        h, yh = sig["harmonic"][skip:], out["harmonic"][skip:]
        w, yw = sig["white"][skip:], out["white"][skip:]
        vals["no_gain"][lvl] = round(_rms_db(yh) - _rms_db(h), 3)
        vals["monotone"][lvl] = round(_rms_db(w) - _rms_db(yw), 3)
        low = _band_att_db(w, yw, lo_bins[0] - 1e-6, lo_bins[1] + 1e-6)
        mid = _band_att_db(w, yw, 1000.0, 4000.0)
        vals["no_hpf"][lvl] = {"att_31_94Hz": round(low, 3), "att_1_4kHz": round(mid, 3), "diff": round(low - mid, 3)}
        vals["aligned"][lvl] = _lag(w, yw)                       # white noise: no periodic side peaks
    return vals


ASSERTS = ("shape_finite", "no_gain", "monotone", "no_hpf", "aligned")


def _fail(name, lvl, text):
    raise AssertionError(f"apm selftest [{name}] level={lvl}: {text}")


def verify(vals, names=ASSERTS):
    """Raise AssertionError("apm selftest [<name>] level=<dB>: <measured> (<tolerance>)") on the first
    failing assert among `names` (in ASSERTS order)."""
    levels = vals["levels"]
    if "shape_finite" in names:
        for lvl in levels:
            bad = {k: v for k, v in vals["shape_finite"][lvl].items() if not all(v)}
            if bad:
                _fail("shape_finite", lvl, f"(length ok, finite) per signal {bad} (both must hold)")
    if "no_gain" in names:
        for lvl in levels:
            g = vals["no_gain"][lvl]
            if not g <= NO_GAIN_TOL_DB:
                _fail("no_gain", lvl, f"output - input RMS {g:+.3f} dB (<= +{NO_GAIN_TOL_DB} dB)")
    if "monotone" in names:
        rem = [vals["monotone"][lvl] for lvl in levels]
        for a, b, la, lb in zip(rem, rem[1:], levels, levels[1:]):
            if not b >= a - MONOTONE_TOL_DB:
                _fail("monotone", lb, f"removed {b:.3f} dB < removed({la}) {a:.3f} dB - {MONOTONE_TOL_DB} dB")
        if 6 in levels and 21 in levels and not vals["monotone"][21] > vals["monotone"][6]:
            _fail("monotone", 21, f"removed {vals['monotone'][21]:.3f} dB <= removed(6) {vals['monotone'][6]:.3f} dB "
                                  f"(must be greater)")
    if "no_hpf" in names:
        for lvl in levels:
            d = vals["no_hpf"][lvl]
            if not abs(d["diff"]) <= NO_HPF_TOL_DB:
                _fail("no_hpf", lvl, f"attenuation 31.25-93.75 Hz {d['att_31_94Hz']:.3f} dB vs 1-4 kHz "
                                     f"{d['att_1_4kHz']:.3f} dB, diff {d['diff']:+.3f} (|diff| <= {NO_HPF_TOL_DB} dB)")
    if "aligned" in names:
        for lvl in levels:
            if vals["aligned"][lvl] != 0:
                _fail("aligned", lvl, f"output lag {vals['aligned'][lvl]} samples vs the input (0)")
    return True


def summary(vals):
    """One line of the measured selftest values (printed at start-up and by checks apm)."""
    lv = vals["levels"]
    return ("shape_finite ok | no_gain dB " + " ".join(f"{l}:{vals['no_gain'][l]:+.2f}" for l in lv)
            + " | monotone removed dB " + " ".join(f"{l}:{vals['monotone'][l]:.2f}" for l in lv)
            + " | no_hpf diff dB " + " ".join(f"{l}:{vals['no_hpf'][l]['diff']:+.2f}" for l in lv)
            + " | aligned lag " + " ".join(f"{l}:{vals['aligned'][l]}" for l in lv))


def selftest(levels=tuple(LEVEL_DB)):
    """measure() + verify(); returns the measured values (AssertionError with the assert name on failure)."""
    vals = measure(levels)
    verify(vals)
    return vals
