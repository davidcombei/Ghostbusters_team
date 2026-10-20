"""
In-process libopus round trip with Gilbert-Elliott packet loss and decoder PLC, for the B4 lever `opus_plc`
(rdloop/B4/plan_agreed.md section 2.3). Op name `opus_plc`, codec `libopus`.

    draw_codec_params(rng, apps=None) -> {"app", "kbps", "cutoff", "frame_ms", "application"}
        the parent rng's draws of utils.rtc_augment.codecs_augm, in exactly its order:
            app       apps[integers(len(apps))]
            kbps      integers(lo, hi + 1)                 (CODEC_PROFILES[app][0])
            cutoff    cutoffs[integers(len(cutoffs))]
            frame_ms  frames[integers(len(frames))]
        so the parent stream ends where a real codecs_augm call leaves it (`checks codec_draw_order`).

    ffmpeg_opus(audio, params, sr=16000) -> (out, app)
        codecs_augm's ffmpeg pipeline on pre-drawn parameters (same command, same pass-through on an
        ffmpeg error): bit-identical to codecs_augm for the same draws. The lever's fallback.

    opus_loss_roundtrip(audio, rng, kbps, cutoff, frame_ms, application, loss_range, burst_range,
                        sr=16000, dec_sr=48000) -> (out float32 same length, realised_loss)
        1. draws from `rng` (the augmenter's rng4), in this order: L = uniform(*loss_range),
           B = uniform(*burst_range), then ge_mask(): one random() per packet;
        2. a libopus encoder at `sr` mono, configured like ffmpeg 4.2's libopusenc with the drawn profile:
           bitrate kbps * 1000, VBR on (unconstrained, ffmpeg -vbr on), complexity 10, expected loss 0,
           max bandwidth from the cutoff (4000 / 6000 / 8000 / 12000 / 20000 Hz -> NB / MB / WB / SWB / FB),
           the profile's application (voip / audio / lowdelay); in-band FEC and DTX stay off (libopus defaults);
        3. the clip, zero-padded to whole frames covering len + look-ahead, is encoded frame by frame;
        4. Gilbert-Elliott mask over the packets: p_bg = 1 / B, p_gb = L p_bg / (1 - L), start state bad
           with probability L (stationary), a packet is lost in the bad state;
        5. a libopus decoder at `dec_sr` (48 kHz, like ffmpeg's native decoder) decodes the received packets
           and conceals each lost one with the NULL-payload PLC call (opus_decode_float(dec, NULL, 0, ...));
        6. the first (dec_sr / sr) x look-ahead output samples are dropped (the Ogg pre-skip ffmpeg removes),
           the rest is resampled dec_sr -> sr with scipy.signal.resample_poly and cut / zero-padded to the
           input length.
        Encoder and decoder are created per call and destroyed in `finally`. realised_loss = lost / packets.

    loss_bin(realised_loss) -> "plc_l0_5" (< 0.05) | "plc_l5_10" (0.05-0.10) | "plc_l10p" (>= 0.10)

    measure() / verify(values) / selftest() / summary(values)
        the start-up selftest with named asserts (plan 2.3: library loads, ctl set/get round trip for bitrate and
        max bandwidth, a concealed gap is not all-zero, finite output of the right length), see SELFTEST below.

libopus is bound through ctypes (the OPUSLIB pattern, https://github.com/orion-labs/opuslib): find_library("opus"),
then {sys.prefix}/lib/libopus.so.0. The library is loaded on the first call of _lib() only (the lever's start-up),
never at import, so a levers-off run maps no libopus (`checks imports`). opus_encoder_ctl is variadic and ctypes
passes its arguments untyped; the selftest's set/get round trip guards that. A missing library raises at start-up
(OSError) instead of passing clips through silently.
"""

import ctypes
import ctypes.util
import os
import subprocess
import sys
from math import gcd

import numpy as np

from utils import rtc_augment as _ra
from utils.rtc_augment import CODEC_PROFILES

SR = 16000
DEC_SR = 48000
LOSS_MAX = 0.5                          # exclusive upper bound of the loss range (p_gb <= 1 needs L < 0.5 at B = 1)
BINS = ("plc_l0_5", "plc_l5_10", "plc_l10p")
MAX_PACKET = 4000                       # bytes; libopus's own maximum for a 120 ms packet is 3 x 1275 + 7

# opus_defines.h
OPUS_OK = 0
APPLICATIONS = {"voip": 2048, "audio": 2049, "lowdelay": 2051}
BANDWIDTH = {4000: 1101, 6000: 1102, 8000: 1103, 12000: 1104, 20000: 1105}     # cutoff Hz -> OPUS_BANDWIDTH_*
SET_BITRATE, GET_BITRATE = 4002, 4003
SET_MAX_BANDWIDTH, GET_MAX_BANDWIDTH = 4004, 4005
SET_VBR, GET_VBR = 4006, 4007
SET_COMPLEXITY, GET_COMPLEXITY = 4010, 4011
GET_INBAND_FEC = 4013
SET_PACKET_LOSS_PERC, GET_PACKET_LOSS_PERC = 4014, 4015
SET_VBR_CONSTRAINT, GET_VBR_CONSTRAINT = 4020, 4021
GET_LOOKAHEAD = 4027
GET_DTX = 4017
COMPLEXITY = 10

_LIB = None
_LIB_PATH = None


def _lib():
    """The ctypes handle of libopus (loaded once per process, on the first call)."""
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB
    candidates = [ctypes.util.find_library("opus"), os.path.join(sys.prefix, "lib", "libopus.so.0")]
    errors = []
    for name in [c for c in candidates if c]:
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
            continue
        _bind(lib)
        _LIB, _LIB_PATH = lib, name
        return lib
    raise OSError(f"libopus not found ({'; '.join(errors) or 'find_library(opus) returned None'}); "
                  f"install the system libopus or `conda install -n rtc-sdd -c conda-forge libopus`")


def _bind(lib):
    c_int_p = ctypes.POINTER(ctypes.c_int)
    f_p = ctypes.POINTER(ctypes.c_float)
    u8_p = ctypes.POINTER(ctypes.c_ubyte)
    lib.opus_get_version_string.restype = ctypes.c_char_p
    lib.opus_get_version_string.argtypes = []
    lib.opus_strerror.restype = ctypes.c_char_p
    lib.opus_strerror.argtypes = [ctypes.c_int]
    lib.opus_encoder_create.restype = ctypes.c_void_p
    lib.opus_encoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.c_int, c_int_p]
    lib.opus_encode_float.restype = ctypes.c_int32
    lib.opus_encode_float.argtypes = [ctypes.c_void_p, f_p, ctypes.c_int, u8_p, ctypes.c_int32]
    lib.opus_encoder_destroy.restype = None
    lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_decoder_create.restype = ctypes.c_void_p
    lib.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, c_int_p]
    lib.opus_decode_float.restype = ctypes.c_int
    lib.opus_decode_float.argtypes = [ctypes.c_void_p, u8_p, ctypes.c_int32, f_p, ctypes.c_int, ctypes.c_int]
    lib.opus_decoder_destroy.restype = None
    lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_encoder_ctl.restype = ctypes.c_int            # variadic: no argtypes, every argument wrapped by hand


def version():
    """libopus version string, e.g. 'libopus 1.3.1' (loads the library)."""
    return _lib().opus_get_version_string().decode()


def library_path():
    _lib()
    return _LIB_PATH


def _err(code):
    return _lib().opus_strerror(int(code)).decode()


def _ctl_set(enc, request, value):
    ret = _lib().opus_encoder_ctl(ctypes.c_void_p(enc), ctypes.c_int(request), ctypes.c_int32(int(value)))
    if ret != OPUS_OK:
        raise RuntimeError(f"opus_encoder_ctl({request}, {value}): {_err(ret)}")


def _ctl_get(enc, request):
    out = ctypes.c_int32(-12345)
    ret = _lib().opus_encoder_ctl(ctypes.c_void_p(enc), ctypes.c_int(request), ctypes.byref(out))
    if ret != OPUS_OK:
        raise RuntimeError(f"opus_encoder_ctl({request}, &out): {_err(ret)}")
    return int(out.value)


# --------------------------------------------------------------------------- #
def draw_codec_params(rng, apps=None):
    """The parent's codec draws, in exactly utils.rtc_augment.codecs_augm's order."""
    apps = list(apps or CODEC_PROFILES)
    app = apps[int(rng.integers(len(apps)))]
    kbps, cutoffs, frames, application = CODEC_PROFILES[app]
    return {"app": app,
            "kbps": int(rng.integers(kbps[0], kbps[1] + 1)),
            "cutoff": cutoffs[int(rng.integers(len(cutoffs)))],
            "frame_ms": frames[int(rng.integers(len(frames)))],
            "application": application}


def ffmpeg_opus(audio, params, sr=SR):
    """codecs_augm's ffmpeg encode + decode on pre-drawn parameters (bit-identical for the same draws)."""
    app = params["app"]
    raw = ["-f", "f32le", "-ac", "1", "-ar", str(sr)]
    try:
        encoded = _ra._ffmpeg([*raw, "-i", "pipe:0", "-c:a", "libopus",
                               "-b:a", f"{int(params['kbps'])}k",
                               "-cutoff", str(params["cutoff"]),
                               "-frame_duration", str(params["frame_ms"]),
                               "-application", params["application"], "-f", "ogg", "pipe:1"],
                              np.asarray(audio, dtype=np.float32).tobytes())
        decoded = _ra._ffmpeg(["-i", "pipe:0", *raw, "pipe:1"], encoded)
    except (OSError, subprocess.CalledProcessError) as exc:    # never kill a training step
        print(f"[rtc_augment] codec {app} failed ({exc}); passing through")
        return audio, app
    return np.frombuffer(decoded, dtype=np.float32), app


def check_ranges(loss_range, burst_range):
    """(lo, hi) floats; ValueError unless 0 <= loss lo <= hi < 0.5 and 1 <= burst lo <= hi."""
    lo, hi = (float(v) for v in loss_range)
    blo, bhi = (float(v) for v in burst_range)
    if not 0.0 <= lo <= hi < LOSS_MAX:
        raise ValueError(f"opus loss range must satisfy 0 <= lo <= hi < {LOSS_MAX}, got {lo:g} {hi:g}")
    if not 1.0 <= blo <= bhi:
        raise ValueError(f"opus burst range must satisfy 1 <= lo <= hi, got {blo:g} {bhi:g}")
    return (lo, hi), (blo, bhi)


def ge_mask(rng, n, loss, burst):
    """Gilbert-Elliott loss mask of n packets (True = lost): p_bg = 1/burst, p_gb = loss p_bg / (1 - loss),
    stationary start; one rng.random() per packet (u[0] picks the start state, u[i] the transition into i)."""
    p_bg = 1.0 / float(burst)
    p_gb = float(loss) * p_bg / (1.0 - float(loss))
    u = rng.random(int(n))
    lost = np.zeros(int(n), dtype=bool)
    bad = bool(u[0] < loss) if n else False
    for i in range(int(n)):
        if i:
            bad = bool(u[i] >= p_bg) if bad else bool(u[i] < p_gb)
        lost[i] = bad
    return lost


def loss_bin(realised):
    return BINS[0] if realised < 0.05 else (BINS[1] if realised < 0.10 else BINS[2])


def _encoder(sr, application, kbps, cutoff):
    lib = _lib()
    err = ctypes.c_int(0)
    enc = lib.opus_encoder_create(int(sr), 1, APPLICATIONS[application], ctypes.byref(err))
    if err.value != OPUS_OK or not enc:
        raise RuntimeError(f"opus_encoder_create({sr}, 1, {application}): {_err(err.value)}")
    try:                                    # ffmpeg 4.2 libopusenc's libopus_configure_encoder, same order
        _ctl_set(enc, SET_BITRATE, int(kbps) * 1000)
        _ctl_set(enc, SET_COMPLEXITY, COMPLEXITY)
        _ctl_set(enc, SET_VBR, 1)
        _ctl_set(enc, SET_VBR_CONSTRAINT, 0)
        _ctl_set(enc, SET_PACKET_LOSS_PERC, 0)
        _ctl_set(enc, SET_MAX_BANDWIDTH, BANDWIDTH[int(cutoff)])
    except Exception:
        lib.opus_encoder_destroy(enc)
        raise
    return enc


def _decoder(dec_sr):
    lib = _lib()
    err = ctypes.c_int(0)
    dec = lib.opus_decoder_create(int(dec_sr), 1, ctypes.byref(err))
    if err.value != OPUS_OK or not dec:
        raise RuntimeError(f"opus_decoder_create({dec_sr}, 1): {_err(err.value)}")
    return dec


def encode_packets(x, enc, frame):
    """Encode whole frames of float32 x (len = k * frame) -> list of packet bytes."""
    lib = _lib()
    f_p = ctypes.POINTER(ctypes.c_float)
    buf = (ctypes.c_ubyte * MAX_PACKET)()
    packets = []
    for i in range(len(x) // frame):
        chunk = np.ascontiguousarray(x[i * frame:(i + 1) * frame], dtype=np.float32)
        nb = lib.opus_encode_float(enc, chunk.ctypes.data_as(f_p), frame, buf, MAX_PACKET)
        if nb < 0:
            raise RuntimeError(f"opus_encode_float: {_err(nb)}")
        packets.append(bytes(buf[:nb]))
    return packets


def decode_packets(packets, lost, dec, frame_dec):
    """Decode packets (lost ones concealed by the NULL-payload PLC call) -> float32 of len(packets) * frame_dec."""
    lib = _lib()
    f_p = ctypes.POINTER(ctypes.c_float)
    u8 = ctypes.c_ubyte
    out = np.zeros(len(packets) * frame_dec, dtype=np.float32)
    base = out.ctypes.data
    step = frame_dec * out.itemsize
    for i, pkt in enumerate(packets):
        dst = ctypes.cast(base + i * step, f_p)
        if lost[i]:
            ret = lib.opus_decode_float(dec, None, 0, dst, frame_dec, 0)
        else:
            data = (u8 * len(pkt)).from_buffer_copy(pkt)
            ret = lib.opus_decode_float(dec, data, len(pkt), dst, frame_dec, 0)
        if ret != frame_dec:
            raise RuntimeError(f"opus_decode_float packet {i} (lost {bool(lost[i])}): "
                               f"{_err(ret) if ret < 0 else f'{ret} samples, expected {frame_dec}'}")
    return out


def roundtrip_mask(audio, lost_fn, kbps, cutoff, frame_ms, application, sr=SR, dec_sr=DEC_SR, info=None):
    """
    Encode `audio` at sr with the profile, apply the loss mask lost_fn(n_packets) -> bool array, decode at dec_sr
    with PLC, drop the pre-skip, resample to sr, cut / pad to len(audio). Returns (out float32, lost mask).
    `info` (dict), when given, receives lookahead, packets and the decoded 48 kHz signal.
    """
    from scipy.signal import resample_poly
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = len(x)
    frame = int(sr) * int(frame_ms) // 1000
    if frame * 1000 != int(sr) * int(frame_ms) or (int(dec_sr) * frame) % int(sr):
        raise ValueError(f"frame {frame_ms} ms at {sr} Hz / {dec_sr} Hz is not a whole number of samples")
    frame_dec = int(dec_sr) * frame // int(sr)
    lib = _lib()
    enc = dec = None
    try:
        enc = _encoder(sr, application, kbps, cutoff)
        la = _ctl_get(enc, GET_LOOKAHEAD)
        n_pk = -(-(n + la) // frame)
        padded = np.zeros(n_pk * frame, dtype=np.float32)
        padded[:n] = x
        packets = encode_packets(padded, enc, frame)
        lost = np.asarray(lost_fn(n_pk), dtype=bool)
        dec = _decoder(dec_sr)
        y_dec = decode_packets(packets, lost, dec, frame_dec)
    finally:
        if enc:
            lib.opus_encoder_destroy(enc)
        if dec:
            lib.opus_decoder_destroy(dec)
    skip = la * int(dec_sr) // int(sr)
    g = gcd(int(sr), int(dec_sr))
    y = resample_poly(y_dec[skip:], int(sr) // g, int(dec_sr) // g).astype(np.float32)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    if info is not None:
        info.update(lookahead=la, packets=packets, decoded=y_dec, skip=skip, frame=frame, frame_dec=frame_dec)
    return y[:n], lost


def opus_loss_roundtrip(audio, rng, kbps, cutoff, frame_ms, application, loss_range, burst_range,
                        sr=SR, dec_sr=DEC_SR, info=None):
    """The lever's codec call: GE loss (from rng) + libopus round trip with PLC. Returns (out, realised_loss)."""
    (lo, hi), (blo, bhi) = check_ranges(loss_range, burst_range)
    loss = float(rng.uniform(lo, hi))
    burst = float(rng.uniform(blo, bhi))
    if info is not None:
        info.update(loss_target=loss, burst=burst)
    y, lost = roundtrip_mask(audio, lambda n: ge_mask(rng, n, loss, burst), kbps, cutoff, frame_ms, application,
                             sr, dec_sr, info)
    return y, float(lost.mean()) if len(lost) else 0.0


# --------------------------------------------------------------------------- #
# SELFTEST (start-up, named asserts; a failure raises AssertionError("opus selftest [<name>] ...")):
#   loads          libopus loads through ctypes and reports its version string
#   ctl_roundtrip  for every app profile (min and max bitrate x every cutoff, its application): bitrate and max
#                  bandwidth read back exactly what was set; VBR 1, VBR constraint 0, complexity 10, expected loss 0,
#                  in-band FEC 0, DTX 0; look-ahead > 0
#   plc_not_zero   tone at -20 dBFS (220 Hz complex), wideband 24 kbps 20 ms voip, packets 20-24 lost: the first
#                  concealed 20 ms are not all-zero (max |x| > 1e-4) and their RMS is printed vs the tone's
#   shape_finite   every profile x loss 0 / 0.2 (burst 2): output length == input length (incl. a 17-sample clip and
#                  a length that is not a whole number of frames) and finite
ASSERTS = ("loads", "ctl_roundtrip", "plc_not_zero", "shape_finite")
GAP = (20, 25)                          # lost packets in the plc_not_zero test (20 ms each)
NOT_ZERO = 1e-4


def _tone(n, dbfs=-20.0, f0=220.0, sr=SR):
    t = np.arange(n) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 9))
    x = x / np.sqrt(np.mean(x ** 2)) * 10 ** (dbfs / 20)
    return x.astype(np.float32)


def _rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12))


def measure():
    vals = {"version": version(), "path": library_path()}
    rows = []
    for app, (kbps, cutoffs, frames, application) in CODEC_PROFILES.items():
        for k in (kbps[0], kbps[1]):
            for c in cutoffs:
                enc = _encoder(SR, application, k, c)
                try:
                    rows.append({"app": app, "kbps": k, "cutoff": c,
                                 "bitrate": _ctl_get(enc, GET_BITRATE), "max_bw": _ctl_get(enc, GET_MAX_BANDWIDTH),
                                 "vbr": _ctl_get(enc, GET_VBR), "vbr_constraint": _ctl_get(enc, GET_VBR_CONSTRAINT),
                                 "complexity": _ctl_get(enc, GET_COMPLEXITY), "loss_perc": _ctl_get(enc, GET_PACKET_LOSS_PERC),
                                 "fec": _ctl_get(enc, GET_INBAND_FEC), "dtx": _ctl_get(enc, GET_DTX),
                                 "lookahead": _ctl_get(enc, GET_LOOKAHEAD)})
                finally:
                    _lib().opus_encoder_destroy(enc)
    vals["ctl"] = rows
    x = _tone(SR)
    gap = np.zeros(64, dtype=bool)
    gap[GAP[0]:GAP[1]] = True
    y, lost = roundtrip_mask(x, lambda n: gap[:n], 24, 8000, 20, "voip")
    y0, _ = roundtrip_mask(x, lambda n: np.zeros(n, bool), 24, 8000, 20, "voip")
    seg = slice(GAP[0] * 320, (GAP[0] + 1) * 320)
    vals["plc"] = {"lost": int(lost.sum()), "max_abs_first": float(np.abs(y[seg]).max()),
                   "rms_first_db": _rms_db(y[seg]), "rms_gap_db": _rms_db(y[GAP[0] * 320:GAP[1] * 320]),
                   "rms_tone_db": _rms_db(x), "rms_noloss_db": _rms_db(y0[seg])}
    shapes = []
    rng = np.random.default_rng(0)
    for n_in in (SR + 123, 17):
        clip = _tone(n_in)
        for app, (kbps, cutoffs, frames, application) in CODEC_PROFILES.items():
            for f in frames:
                for loss in (0.0, 0.2):
                    y, lost = roundtrip_mask(clip, lambda n: ge_mask(rng, n, loss, 2.0), kbps[0], cutoffs[0], f, application)
                    shapes.append({"app": app, "frame": f, "n": n_in, "loss": loss, "len": len(y),
                                   "finite": bool(np.all(np.isfinite(y))), "lost": int(lost.sum()), "packets": len(lost)})
    vals["shapes"] = shapes
    return vals


def _fail(name, text):
    raise AssertionError(f"opus selftest [{name}]: {text}")


def verify(vals, names=ASSERTS):
    for name in names:
        if name == "loads":
            if not str(vals.get("version", "")).startswith("libopus"):
                _fail(name, f"version string {vals.get('version')!r}")
        elif name == "ctl_roundtrip":
            for r in vals["ctl"]:
                want = {"bitrate": r["kbps"] * 1000, "max_bw": BANDWIDTH[r["cutoff"]], "vbr": 1, "vbr_constraint": 0,
                        "complexity": COMPLEXITY, "loss_perc": 0, "fec": 0, "dtx": 0}
                bad = {k: (r[k], v) for k, v in want.items() if r[k] != v}
                if bad or r["lookahead"] <= 0:
                    _fail(name, f"{r['app']} {r['kbps']} kbps cutoff {r['cutoff']}: (got, set) {bad}; "
                                f"lookahead {r['lookahead']}")
        elif name == "plc_not_zero":
            p = vals["plc"]
            if p["lost"] != GAP[1] - GAP[0] or not p["max_abs_first"] > NOT_ZERO:
                _fail(name, f"lost {p['lost']} packets; first concealed 20 ms max |x| {p['max_abs_first']:.2e} "
                            f"(> {NOT_ZERO:g} needed)")
        elif name == "shape_finite":
            bad = [s for s in vals["shapes"] if s["len"] != s["n"] or not s["finite"]]
            if bad:
                _fail(name, f"{len(bad)} runs with a wrong length / non-finite output, e.g. {bad[0]}")
        else:
            raise KeyError(name)


def summary(vals):
    la = sorted({r["lookahead"] for r in vals["ctl"]})
    p = vals["plc"]
    return (f"{vals['version']} ({vals['path']}); ctl round trip {len(vals['ctl'])} encoder configs, lookahead {la} "
            f"samples at 16 kHz; first concealed 20 ms RMS {p['rms_first_db']:.1f} dBFS (tone {p['rms_tone_db']:.1f}, "
            f"no-loss decode {p['rms_noloss_db']:.1f}), 100 ms gap {p['rms_gap_db']:.1f}; "
            f"{len(vals['shapes'])} shape runs ok")


def selftest():
    """measure() + verify(); returns the measured values (AssertionError with the assert name on failure)."""
    vals = measure()
    verify(vals)
    return vals
