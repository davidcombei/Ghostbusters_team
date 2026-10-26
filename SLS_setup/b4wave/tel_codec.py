"""
Telephony codecs for the b4wave lever `tel_codec` (run B4e).

An encode + decode round trip through PyAV's bundled FFmpeg, in memory. Ported from
`utils/rtc_augment_expert.py::telephony_augm` on origin/feature/model-updates (2283dfd): same codec
table, same resample -> WAV-container encode -> decode -> resample path.

    TELEPHONY_CODECS   name -> (PyAV encoder, codec sample rate, bitrates to draw from; None = fixed)
    ALLOWED            the names training may use: g723_1 (6.3 kbps) and amr_nb (4.75-12.2 kbps).
                       G.711 (pcm_mulaw / pcm_alaw), G.722 and G.726 are reserved for the local proxy
                       sets (b3wave/heldout_reserve.py: sim_unseen_v1 / sim_heldout_v1). The lever's
                       init passes the names and their encoders to assert_training_ops, which refuses them.

    roundtrip(audio, rng, name, sr=16000) -> (float32 at sr, bitrate)
        resample to the codec rate (s16) -> encode into an in-memory WAV -> decode -> resample back.
        The bitrate is drawn from `rng`. Raises on any backend error (the lever counts it and falls
        back to the Opus path); the original printed and passed the clip through. The output is
        longer than the input (frame padding + codec delay): the caller trims it with finalize().

    measure(names) / verify(vals) / selftest(names) / summary(vals) / version()
        the start-up selftest, named asserts: [encoder] (the encoder exists in this PyAV build),
        [shape_finite], [length] (output >= input), [level] (RMS within LEVEL_TOL_DB of the input),
        [narrowband] (share of the spectral magnitude above NB_EDGE_HZ below NB_MAX_SHARE; the input
        has NB_INPUT_MIN_SHARE or more there, so the assert has teeth).

Needs PyAV 15.1.0 (`pip install av==15.1.0`): the 19.x wheels ship no libopencore_amrnb encoder
(12.3, 13.1 and 15.1 have it; checked 2026-10-26). PyAV is imported lazily, only when the lever is on.
"""

import io
import time

import numpy as np

SR = 16000

# name -> (PyAV encoder, codec sample rate, bitrates to draw from; None = fixed by the codec)
TELEPHONY_CODECS = {
    "g711_ulaw": ("pcm_mulaw", 8000, (None,)),
    "g711_alaw": ("pcm_alaw", 8000, (None,)),
    "g722":      ("g722", 16000, (None,)),
    "g723_1":    ("g723_1", 8000, (6300,)),
    "g726":      ("g726", 8000, (16000, 24000, 32000, 40000)),
    "amr_nb":    ("libopencore_amrnb", 8000, (4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200)),
}
ALLOWED = ("g723_1", "amr_nb")

# selftest
SELFTEST_SECONDS = 2.0
LEVEL_TOL_DB = 3.0
NB_EDGE_HZ = 4200.0
NB_MAX_SHARE = 0.05                 # measured 0.020-0.027 (PyAV 15.1.0); wideband G.722 gives 0.29
NB_INPUT_MIN_SHARE = 0.15           # the probe clip measures 0.25


def encoder(name):
    return TELEPHONY_CODECS[name][0]


def version():
    """'PyAV <version> libavcodec <x.y.z>' (imports av)."""
    import av
    return f"PyAV {av.__version__} libavcodec {'.'.join(str(v) for v in av.library_versions['libavcodec'])}"


def _resample(audio, sr_in, sr_out, fmt):
    import av
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio, np.float32).reshape(1, -1),
                                       format="flt", layout="mono")
    frame.sample_rate = sr_in
    resampler = av.AudioResampler(format=fmt, layout="mono", rate=sr_out)
    frames = resampler.resample(frame) + resampler.resample(None)
    return np.concatenate([f.to_ndarray().reshape(-1) for f in frames])


def roundtrip(audio, rng, name, sr=SR):
    """
    Encode + decode `audio` through the telephony codec `name`; (decoded float32 at sr, bitrate).
    The bitstream goes through an in-memory WAV so the decoder gets the codec parameters (G.726
    needs them).
    """
    import av
    enc, codec_sr, bitrates = TELEPHONY_CODECS[name]
    bitrate = bitrates[int(rng.integers(len(bitrates)))]
    pcm = _resample(audio, sr, codec_sr, "s16")
    buf = io.BytesIO()
    with av.open(buf, "w", format="wav") as container:
        stream = container.add_stream(enc, rate=codec_sr, layout="mono")
        if bitrate:
            stream.bit_rate = bitrate
        step = stream.codec_context.frame_size or 1024
        pcm = np.pad(pcm, (0, -len(pcm) % step))
        for start in range(0, len(pcm), step):
            frame = av.AudioFrame.from_ndarray(pcm[start:start + step].reshape(1, -1),
                                               format="s16", layout="mono")
            frame.sample_rate = codec_sr
            frame.pts = start
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    buf.seek(0)
    with av.open(buf, "r", format="wav") as container:
        frames = [f.to_ndarray().reshape(-1) for f in container.decode(audio=0)]
        is_int = container.streams.audio[0].format.name.startswith("s16")
    decoded = np.concatenate(frames).astype(np.float32)
    if is_int:
        decoded /= 32768.0
    return _resample(decoded, codec_sr, sr, "flt").astype(np.float32), bitrate


# --------------------------------------------------------------------------- #
# start-up selftest
# --------------------------------------------------------------------------- #
def _probe_clip(seconds=SELFTEST_SECONDS, sr=SR):
    """200 Hz harmonic complex (30 harmonics, 1/k, up to 6 kHz) under a 3 Hz envelope, about -20 dBFS,
    plus white noise at -50 dBFS: wideband, so [narrowband] can tell an 8 kHz codec from a pass-through."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * 200 * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 31))
    x = x * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = 0.1 * x / np.sqrt(np.mean(x ** 2)) + 10 ** (-50 / 20) * rng.standard_normal(len(t))
    return x.astype(np.float32)


def _hf_share(x, sr=SR):
    mag = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(mag[f > NB_EDGE_HZ].sum() / max(mag.sum(), 1e-12))


def _rms_db(x):
    return 10 * np.log10(max(float(np.mean(np.asarray(x, np.float64) ** 2)), 1e-20))


def measure(names=ALLOWED):
    """One round trip of the probe clip per codec (at its lowest and highest bitrate); the measured values."""
    import av
    x = _probe_clip()
    vals = {"names": list(names), "version": version(), "input_hf_share": _hf_share(x), "codecs": {}}
    for name in names:
        enc = encoder(name)
        try:
            av.codec.Codec(enc, "w")
        except Exception as exc:                    # verify() turns this into the [encoder] assert
            vals["codecs"][name] = {"encoder": enc, "missing": f"{type(exc).__name__}: {exc}"}
            continue
        runs = []
        for pick in sorted({0, len(TELEPHONY_CODECS[name][2]) - 1}):     # lowest and highest bitrate (tables ascend)
            t0 = time.perf_counter()
            y, used = roundtrip(x, _FixedPick(pick), name)
            ms = (time.perf_counter() - t0) * 1e3
            y_in = y[:len(x)]
            runs.append({"bitrate": used, "finite": bool(np.all(np.isfinite(y))), "len_in": len(x), "len_out": len(y),
                         "level_db": _rms_db(y_in) - _rms_db(x), "hf_share": _hf_share(y_in), "ms": ms})
        vals["codecs"][name] = {"encoder": enc, "runs": runs}
    return vals


class _FixedPick:
    """A stand-in rng whose integers() returns a fixed index (the selftest walks the bitrate table)."""

    def __init__(self, index):
        self.index = index

    def integers(self, n):
        return min(self.index, n - 1)


def verify(vals):
    """AssertionError("tel selftest [<name>] <codec> ...: <measured> (<tolerance>)") on the first failing assert."""
    assert vals["input_hf_share"] >= NB_INPUT_MIN_SHARE, (f"tel selftest [narrowband] probe: input share above "
                                                          f"{NB_EDGE_HZ:g} Hz {vals['input_hf_share']:.4f} (>= {NB_INPUT_MIN_SHARE})")
    for name in vals["names"]:
        c = vals["codecs"][name]
        assert "missing" not in c, (f"tel selftest [encoder] {name}: {c['encoder']} not in {vals['version']} "
                                    f"({c.get('missing')}); install av==15.1.0")
        for r in c["runs"]:
            tag = f"{name} @{r['bitrate'] or 'fixed'}"
            assert r["finite"], f"tel selftest [shape_finite] {tag}: non-finite output"
            assert r["len_out"] >= r["len_in"], f"tel selftest [length] {tag}: {r['len_out']} < {r['len_in']} samples"
            assert abs(r["level_db"]) <= LEVEL_TOL_DB, (f"tel selftest [level] {tag}: {r['level_db']:+.2f} dB "
                                                        f"(|.| <= {LEVEL_TOL_DB})")
            assert r["hf_share"] < NB_MAX_SHARE, (f"tel selftest [narrowband] {tag}: share above {NB_EDGE_HZ:g} Hz "
                                                  f"{r['hf_share']:.4f} (< {NB_MAX_SHARE})")


def summary(vals):
    """One line of the measured selftest values (printed at start-up and by checks tel)."""
    parts = []
    for name in vals["names"]:
        runs = vals["codecs"][name].get("runs", [])
        parts.append(f"{name}: " + ", ".join(f"{r['bitrate'] or 'fixed'} bps level {r['level_db']:+.2f} dB "
                                             f"hf {r['hf_share']:.4f} len +{r['len_out'] - r['len_in']} {r['ms']:.0f} ms"
                                             for r in runs))
    return f"{vals['version']}; probe hf {vals['input_hf_share']:.4f}; " + " | ".join(parts)


def selftest(names=ALLOWED):
    """measure() + verify(); returns the measured values (AssertionError with the assert name on failure)."""
    vals = measure(names)
    verify(vals)
    return vals
