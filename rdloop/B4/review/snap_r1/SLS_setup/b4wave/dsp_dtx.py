"""
webrtcvad DTX gating for the B4 lever `dtx_gate` (rdloop/B4/plan_agreed.md section 2.2). Op name `wrtc_dtx`.

    wrtc_dtx(audio, rng, sr=16000, fill="mix") -> (out float32 same length, gated_frac, fill_used)
        per clip, from `rng` (the augmenter's rng4), in this order:
            mode      (1, 2, 3)[integers(3)]       webrtcvad aggressiveness 1..3
            frame_ms  (10, 20, 30)[integers(3)]    VAD frame
            hang_ms   uniform(40, 200)             forward hangover on the speech mask
            fill      "mix": "zero" if random() < 0.5 else "cng"; otherwise the given fill
            level_db  uniform(-6, 0)               CNG level vs the gated content (cng only)
        then gate(); a CNG fill also draws its excitation, standard_normal(len + WARM), when a
        sample is gated.

    gate(audio, mode, frame_ms, hang_ms, fill, level_db=None, rng=None, sr=16000) -> (out, info)
        1. webrtcvad.Vad(mode) on consecutive frames of the int16 clip (the last partial frame is
           zero-padded to a full frame; its decision covers its samples);
        2. the speech mask (sample level) is dilated forward by round(hang_ms * sr / 1000) samples
           (a sample is kept when a speech sample lies within the hangover before it);
        3. the other samples are gated: the fill weight w rises with a raised cosine over the first /
           last 5 ms of every gated run (distance to the nearest kept sample, 1..80 samples at
           16 kHz) and is 1 inside it; at a clip edge the run is gated from the first sample;
           out = (1 - w) x + w fill on gated samples, exactly the fill where w = 1, exactly x on kept
           samples, so the length is unchanged;
        4. fill "zero": exact zeros. Fill "cng": order-10 LPC comfort noise -- autocorrelation of the
           gated content for lags 0-10 (kept samples zeroed, so no product crosses a kept sample;
           lag 0 x (1 + 1e-6) for conditioning), scipy.linalg.solve_toeplitz, synthesis
           lfilter([1], A, white) after WARM discarded samples, scaled to RMS = gated-content RMS x
           10^(level_db / 20) over the gated samples, floor -70 dBFS. Digital-silence content or an
           unstable predictor falls back to a flat (white) spectrum at the target RMS.
        No gated sample -> the input itself (copy, float32), gated fraction 0.
        info: gated (bool mask), full (w == 1 mask), w, gated_frac, target_rms, lpc (A), white_fallback.

    gated_bin(frac) -> "dtx_g0" (< 1 %) | "dtx_g1_20" (1-20 %) | "dtx_g20_40" (20-40 %) | "dtx_g40p" (>= 40 %)

    measure() / verify(values) / selftest() / summary(values)
        the start-up selftest with named asserts (plan 2.2 equivalence test), see SELFTEST below.

Relation to the reserved utils/sim_ops.vad_gate (held-out op): the gating family and the zero fill are
shared; its detector (energy VAD, p10 + margin, frame hangover) and its white-noise CNG colour are not.

webrtcvad (package webrtcvad-wheels) and scipy are imported lazily (inside the functions), only when
the lever is on. webrtcvad keeps its own internal overhang after speech (about 100-330 ms in this
build, measured by rdloop/B4/scratch/dtx_probe.py); the hangover above adds to it.
"""

import numpy as np

SR = 16000
MODES = (1, 2, 3)
FRAMES_MS = (10, 20, 30)
HANG_MS = (40.0, 200.0)
CNG_LEVEL_DB = (-6.0, 0.0)
CNG_FLOOR_DBFS = -70.0
LPC_ORDER = 10
RAMP_MS = 5.0
WARM = 512                          # discarded synthesis-filter samples (start-up transient)
FILLS = ("mix", "zero", "cng")
VAD_RATES = (8000, 16000, 32000, 48000)
BINS = ("dtx_g0", "dtx_g1_20", "dtx_g20_40", "dtx_g40p")
BACKEND = "webrtcvad.Vad"


def check_fill(fill):
    if fill not in FILLS:
        raise ValueError(f"wrtc_dtx fill {fill!r} not in {list(FILLS)}")
    return fill


def gated_bin(frac):
    if frac < 0.01:
        return BINS[0]
    if frac < 0.20:
        return BINS[1]
    if frac < 0.40:
        return BINS[2]
    return BINS[3]


def draw_params(rng, fill="mix"):
    """The per-clip parameters from `rng`, in the documented order (CNG excitation drawn later, in gate)."""
    check_fill(fill)
    mode = int(MODES[int(rng.integers(len(MODES)))])
    frame_ms = int(FRAMES_MS[int(rng.integers(len(FRAMES_MS)))])
    hang_ms = float(rng.uniform(*HANG_MS))
    fill_used = fill if fill != "mix" else ("zero" if rng.random() < 0.5 else "cng")
    level_db = float(rng.uniform(*CNG_LEVEL_DB)) if fill_used == "cng" else None
    return {"mode": mode, "frame_ms": frame_ms, "hang_ms": hang_ms, "fill": fill_used, "level_db": level_db}


def vad_speech(x, mode, frame_ms, sr=SR):
    """Sample-level speech mask from webrtcvad.Vad(mode) on consecutive frame_ms frames of x (float)."""
    import webrtcvad
    if sr not in VAD_RATES:
        raise ValueError(f"webrtcvad runs at {VAD_RATES} Hz, got {sr}")
    if mode not in MODES or frame_ms not in FRAMES_MS:
        raise ValueError(f"mode {mode} / frame {frame_ms} ms not in {MODES} / {FRAMES_MS}")
    n = sr * frame_ms // 1000
    q = np.clip(np.round(np.clip(x, -1.0, 1.0) * 32767.0), -32768, 32767).astype(np.int16)
    m = -(-len(q) // n)
    q = np.pad(q, (0, m * n - len(q)))
    vad = webrtcvad.Vad(int(mode))
    dec = np.fromiter((vad.is_speech(q[i * n:(i + 1) * n].tobytes(), sr) for i in range(m)), dtype=bool, count=m)
    return np.repeat(dec, n)[:len(x)]


def _keep_mask(speech, hang):
    """speech dilated forward by `hang` samples."""
    idx = np.arange(len(speech))
    last = np.maximum.accumulate(np.where(speech, idx, -(10 ** 12)))
    return (idx - last) <= hang


def _fill_weight(gated, ramp):
    """0 on kept samples; on gated samples 0.5 - 0.5 cos(pi min(d, ramp) / ramp), d = distance to the nearest kept sample."""
    n = len(gated)
    idx = np.arange(n)
    big = 10 ** 12
    prev_keep = np.maximum.accumulate(np.where(~gated, idx, -big))
    next_keep = np.minimum.accumulate(np.where(~gated, idx, big)[::-1])[::-1]
    d = np.minimum(idx - prev_keep, next_keep - idx).astype(np.float64)
    w = 0.5 - 0.5 * np.cos(np.pi * np.minimum(d, ramp) / ramp)
    return np.where(gated, w, 0.0)


def lpc(xg, n_content, order=LPC_ORDER):
    """A(z) = [1, -a_1, ..., -a_order] from the autocorrelation of xg (float64, zeros outside the content,
    n_content content samples); None for digital silence (content RMS <= 1e-9) or an unstable predictor.
    The lags are sums of elementwise products, not np.dot: the dot product wakes numpy's multi-threaded
    OpenBLAS, whose spinning threads cost ~8x the CPU in a loader worker (rdloop/B4/impl/progress.md)."""
    from scipy.linalg import solve_toeplitz
    n = len(xg)
    r = np.array([float(np.sum(xg[:n - k] * xg[k:])) for k in range(order + 1)])
    if not r[0] > 1e-18 * max(int(n_content), 1):
        return None
    r[0] *= 1.0 + 1e-6
    try:
        a = solve_toeplitz((r[:order], r[:order]), r[1:order + 1])
    except np.linalg.LinAlgError:
        return None
    A = np.concatenate([[1.0], -a])
    if not np.all(np.isfinite(A)) or np.max(np.abs(np.roots(A))) >= 1.0:
        return None
    return A


def gated_mask(audio, mode, frame_ms, hang_ms, sr=SR):
    """Steps 1-2 of gate(): the sample-level mask of the samples DTX gates (webrtcvad + forward hangover)."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    return ~_keep_mask(vad_speech(x, mode, frame_ms, sr), int(round(hang_ms * sr / 1000.0)))


def gate(audio, mode, frame_ms, hang_ms, fill, level_db=None, rng=None, sr=SR):
    """Gate one clip with fixed parameters (see the module doc). Returns (out float32, info)."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if fill not in ("zero", "cng"):
        raise ValueError(f"gate fill must be zero or cng, got {fill!r}")
    gated = gated_mask(x, mode, frame_ms, hang_ms, sr)
    info = {"gated": gated, "gated_frac": float(gated.mean()) if len(x) else 0.0, "target_rms": None,
            "lpc": None, "white_fallback": False}
    if not gated.any():
        info.update(full=gated, w=np.zeros(len(x)))
        return x.copy(), info
    w = _fill_weight(gated, max(1, int(round(RAMP_MS * sr / 1000.0))))
    full = w >= 1.0
    x64 = x.astype(np.float64)
    if fill == "zero":
        f = np.zeros(len(x))
    else:
        if level_db is None or rng is None:
            raise ValueError("cng fill needs level_db and rng")
        content = x64[gated]
        target = max(float(np.sqrt(np.mean(content ** 2))) * 10.0 ** (level_db / 20.0),
                     10.0 ** (CNG_FLOOR_DBFS / 20.0))
        A = lpc(np.where(gated, x64, 0.0), int(gated.sum()))
        e = rng.standard_normal(len(x) + WARM)
        if A is None:
            f = e[WARM:]
            info["white_fallback"] = True
        else:
            from scipy.signal import lfilter
            f = lfilter([1.0], A, e)[WARM:]
        f = f * (target / max(float(np.sqrt(np.mean(f[gated] ** 2))), 1e-30))
        info.update(target_rms=target, lpc=A)
    out = x.copy()
    part = gated & ~full
    out[part] = ((1.0 - w[part]) * x64[part] + w[part] * f[part]).astype(np.float32)
    out[full] = f[full].astype(np.float32)
    info.update(full=full, w=w)
    return out, info


def wrtc_dtx(audio, rng, sr=SR, fill="mix"):
    """One DTX gating pass with per-clip parameters drawn from `rng`. Returns (out, gated_frac, fill_used)."""
    p = draw_params(rng, fill)
    out, info = gate(audio, p["mode"], p["frame_ms"], p["hang_ms"], p["fill"], p["level_db"], rng, sr)
    return out, info["gated_frac"], p["fill"]


# ------------------------------------------------------------------------------------------ #
# SELFTEST (plan 2.2): deterministic 16 kHz signals; every mode x frame x hangover {40, 200} ms
#   burst clip: 0.96 s gap + 4 x (0.96 s tone + 0.96 s gap)  (segments on the 60 ms grid of every frame
#   size); tone = 220 Hz harmonic complex (20 harmonics, 1/k) at -26 dBFS, gap = low-passed noise
#   (one-pole 0.9) at -50 dBFS; tone clip: the same tone for 4 s
#   [shape_finite]    output length == input length, finite (both fills, both clips)
#   [gated_frac]      burst clip gated fraction in [GATED_LO, gap share]
#   [speech_kept]     every tone sample of the burst clip is bit-identical to the input
#   [zero_exact]      zero fill: every fully gated sample (w = 1) is exactly 0
#   [cng_level]       cng fill: RMS over the fully gated samples within +-3 dB of the target (level 0 dB
#                     and -6 dB), and on a burst clip with digital-silence gaps the -70 dBFS floor
#   [cng_colour]      cng fill: (0.1-1 kHz) / (2-6 kHz) band-power ratio of the fill within 3 dB of the
#                     gap content's (LPC-shaped, not white)
#   [tone_unchanged]  tone clip: output bit-identical to the input (nothing gated)
# ------------------------------------------------------------------------------------------ #
SEG = int(0.96 * SR)
GATED_LO = 0.20
CNG_TOL_DB = 3.0
COLOUR_TOL_DB = 3.0
ASSERTS = ("shape_finite", "gated_frac", "speech_kept", "zero_exact", "cng_level", "cng_colour", "tone_unchanged")


def _scaled(x, dbfs):
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt(np.mean(x ** 2)) * 10.0 ** (dbfs / 20.0)


def tone(n, dbfs=-26.0, f0=220.0):
    t = np.arange(n) / SR
    return _scaled(sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 21)), dbfs)


def coloured_noise(n, dbfs=-50.0, seed=0):
    from scipy.signal import lfilter
    return _scaled(lfilter([1.0], [1.0, -0.9], np.random.default_rng(seed).standard_normal(n)), dbfs)


def burst_clip(gap="noise"):
    """(clip float32, tone mask): gap, then 4 x (tone, gap); gap = coloured noise or digital zeros."""
    parts, mask = [], []
    for i in range(9):
        on = i % 2 == 1
        if on:
            seg = tone(SEG)
        else:
            seg = coloured_noise(SEG, seed=i) if gap == "noise" else np.zeros(SEG)
        parts.append(seg)
        mask.append(np.full(SEG, on))
    return np.concatenate(parts).astype(np.float32), np.concatenate(mask)


def _band_ratio_db(x):
    from scipy.signal import welch
    f, p = welch(np.asarray(x, np.float64), fs=SR, nperseg=512)
    lo, hi = p[(f >= 100) & (f <= 1000)].sum(), p[(f >= 2000) & (f <= 6000)].sum()
    return float(10 * np.log10(lo / max(hi, 1e-30)))


def _rms_db(x):
    return float(10 * np.log10(float(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-30))


def measure():
    """Run the selftest grid; return the measured values (no asserts)."""
    burst, on = burst_clip("noise")
    silent, _ = burst_clip("zeros")
    tclip = tone(4 * SR).astype(np.float32)
    gap_share = float((~on).mean())
    content_ratio = _band_ratio_db(burst[~on])
    vals = {"gap_share": round(gap_share, 4), "content_ratio_db": round(content_ratio, 2), "runs": []}
    for mode in MODES:
        for frame in FRAMES_MS:
            for hang in HANG_MS:
                row = {"mode": mode, "frame": frame, "hang": hang}
                yz, iz = gate(burst, mode, frame, hang, "zero")
                rng = np.random.default_rng(0)
                yc, ic = gate(burst, mode, frame, hang, "cng", 0.0, rng)
                y6, i6 = gate(burst, mode, frame, hang, "cng", -6.0, np.random.default_rng(1))
                ys, is_ = gate(silent, mode, frame, hang, "cng", 0.0, np.random.default_rng(2))
                yt, _ = gate(tclip, mode, frame, hang, "zero")
                row["shape_finite"] = all(len(y) == len(x) and bool(np.all(np.isfinite(y)))
                                          for y, x in ((yz, burst), (yc, burst), (y6, burst), (ys, silent), (yt, tclip)))
                row["gated_frac"] = round(iz["gated_frac"], 4)
                row["speech_changed"] = int(np.sum(yz[on] != burst[on]) + np.sum(yc[on] != burst[on]))
                row["zero_max"] = float(np.max(np.abs(yz[iz["full"]]))) if iz["full"].any() else 0.0
                row["n_full"] = int(iz["full"].sum())
                lv = []
                for y, info, lvl in ((yc, ic, 0.0), (y6, i6, -6.0), (ys, is_, None)):
                    if info["full"].any():
                        lv.append(round(_rms_db(y[info["full"]]) - 20 * np.log10(info["target_rms"]), 3))
                    else:
                        lv.append(None)
                row["cng_level_db"] = lv            # measured - target, dB (level 0, level -6, floor)
                row["floor_target_dbfs"] = round(20 * np.log10(is_["target_rms"]), 2) if is_["target_rms"] else None
                row["cng_colour_db"] = (round(_band_ratio_db(yc[ic["full"]]) - content_ratio, 3)
                                        if ic["full"].sum() > 2048 else None)
                row["white_fallback"] = bool(ic["white_fallback"])
                row["tone_changed"] = int(np.sum(yt != tclip))
                vals["runs"].append(row)
    return vals


def _fail(name, row, text):
    where = f"mode={row['mode']} frame={row['frame']}ms hang={row['hang']:g}ms" if row else ""
    raise AssertionError(f"dtx selftest [{name}] {where}: {text}")


def verify(vals, names=ASSERTS):
    """Raise AssertionError("dtx selftest [<name>] mode=.. frame=.. hang=..: <measured> (<tolerance>)")
    on the first failing assert among `names` (in ASSERTS order)."""
    runs = vals["runs"]
    for name in ASSERTS:
        if name not in names:
            continue
        for row in runs:
            if name == "shape_finite" and not row["shape_finite"]:
                _fail(name, row, "length or finiteness broken")
            if name == "gated_frac" and not GATED_LO <= row["gated_frac"] <= vals["gap_share"]:
                _fail(name, row, f"gated fraction {row['gated_frac']:.4f} not in [{GATED_LO}, {vals['gap_share']:.4f}]")
            if name == "speech_kept" and row["speech_changed"]:
                _fail(name, row, f"{row['speech_changed']} tone samples changed (0)")
            if name == "zero_exact" and (row["zero_max"] != 0.0 or row["n_full"] == 0):
                _fail(name, row, f"max |fully gated sample| {row['zero_max']:g} over {row['n_full']} samples (exactly 0, n > 0)")
            if name == "cng_level" and not all(v is not None and abs(v) <= CNG_TOL_DB for v in row["cng_level_db"]):
                _fail(name, row, f"CNG RMS - target {row['cng_level_db']} dB (level 0, level -6, -70 dBFS floor; "
                                 f"within +-{CNG_TOL_DB})")
            if name == "cng_colour" and not (row["cng_colour_db"] is not None and abs(row["cng_colour_db"]) <= COLOUR_TOL_DB
                                             and not row["white_fallback"]):
                _fail(name, row, f"fill - content (0.1-1 kHz)/(2-6 kHz) ratio {row['cng_colour_db']} dB "
                                 f"(within +-{COLOUR_TOL_DB}; LPC, no white fallback)")
            if name == "tone_unchanged" and row["tone_changed"]:
                _fail(name, row, f"{row['tone_changed']} samples of the tone clip changed (0)")
    return True


def summary(vals):
    """One line of the measured selftest values (printed at start-up and by checks dtx)."""
    runs = vals["runs"]
    gf = [r["gated_frac"] for r in runs]
    lv = [v for r in runs for v in r["cng_level_db"] if v is not None]
    col = [r["cng_colour_db"] for r in runs if r["cng_colour_db"] is not None]
    return (f"{len(runs)} mode x frame x hangover runs | gated_frac {min(gf):.3f}..{max(gf):.3f} "
            f"(gap share {vals['gap_share']:.3f}) | speech_kept {sum(r['speech_changed'] for r in runs) == 0} | "
            f"zero_exact max {max(r['zero_max'] for r in runs):g} | cng_level dB {min(lv):+.2f}..{max(lv):+.2f} | "
            f"cng_colour dB {min(col):+.2f}..{max(col):+.2f} | tone_unchanged {sum(r['tone_changed'] for r in runs) == 0}")


def selftest():
    """measure() + verify(); returns the measured values (AssertionError with the assert name on failure)."""
    vals = measure()
    verify(vals)
    return vals
