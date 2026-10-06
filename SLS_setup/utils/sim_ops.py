"""
Extra simulator stages for the local proxy sets (scripts/build_dev_noisy_sim.py) and, later,
for augmentation v2. Same contract as rtc_augment: float32 mono numpy in, float32 out, an
explicit numpy Generator for every random choice.

The stage functions themselves are neutral; which codecs / filters may be used for *training*
is decided by utils/heldout_reserve.py (G.722, G.726, GSM, anlmdn and acompressor are
reserved for sim-heldout).
"""

import subprocess

import numpy as np
from scipy.signal import butter, sosfilt

from .rtc_augment import _ffmpeg, resample

RAW = ["-f", "f32le", "-ac", "1"]

# name -> (ffmpeg encoder, codec sample rate, bitrate choices in kbps or None)
CODECS = {
    "g722": ("g722", 16000, None),
    "g726": ("g726", 8000, (16, 24, 32, 40)),
    "gsm": ("libgsm_ms", 8000, None),
    "speex": ("libspeex", 16000, None),
    "g711_mulaw": ("pcm_mulaw", 8000, None),
    "g711_alaw": ("pcm_alaw", 8000, None),
}


def ffmpeg_filter(audio, sr, graph):
    """Run an ffmpeg -af filtergraph over the clip (length may change slightly)."""
    out = _ffmpeg([*RAW, "-ar", str(sr), "-i", "pipe:0", "-af", graph, *RAW, "-ar", str(sr), "pipe:1"],
                  np.asarray(audio, dtype=np.float32).tobytes())
    return np.frombuffer(out, dtype=np.float32).copy()


def ffmpeg_codec(audio, sr, codec, rng):
    """Encode + decode through one of CODECS (in a wav container); returns (audio, params)."""
    encoder, codec_sr, bitrates = CODECS[codec]
    params = {"codec": codec, "codec_sr": codec_sr}
    enc = [*RAW, "-ar", str(sr), "-i", "pipe:0", "-ar", str(codec_sr), "-c:a", encoder]
    if bitrates:
        kbps = int(bitrates[int(rng.integers(len(bitrates)))])
        enc += ["-b:a", f"{kbps}k"]
        params["kbps"] = kbps
    encoded = _ffmpeg([*enc, "-f", "wav", "pipe:1"], np.asarray(audio, dtype=np.float32).tobytes())
    decoded = _ffmpeg(["-i", "pipe:0", *RAW, "-ar", str(sr), "pipe:1"], encoded)
    return np.frombuffer(decoded, dtype=np.float32).copy(), params


def noise_suppress(audio, sr, rng, kinds=("afftdn", "anlmdn")):
    """A classical noise suppressor with a random strength; returns (audio, params)."""
    kind = kinds[int(rng.integers(len(kinds)))]
    if kind == "afftdn":
        nr = float(rng.uniform(6, 30))
        graph, params = f"afftdn=nr={nr:.1f}:nf=-{int(rng.integers(30, 60))}", {"ns": kind, "nr": round(nr, 1)}
    elif kind == "anlmdn":
        s = float(10 ** rng.uniform(-4, -2))
        graph, params = f"anlmdn=s={s:.5f}", {"ns": kind, "s": round(s, 5)}
    else:
        raise ValueError(kind)
    return ffmpeg_filter(audio, sr, graph), params


def agc(audio, sr, rng, kinds=("dynaudnorm", "acompressor")):
    """Automatic gain control / dynamics; returns (audio, params)."""
    kind = kinds[int(rng.integers(len(kinds)))]
    if kind == "dynaudnorm":
        f = int(rng.integers(50, 300))
        graph, params = f"dynaudnorm=f={f}:g={int(rng.integers(3, 16)) | 1}", {"agc": kind, "f": f}
    elif kind == "acompressor":
        thr, ratio = float(rng.uniform(0.03, 0.3)), float(rng.uniform(2, 8))
        graph = f"acompressor=threshold={thr:.3f}:ratio={ratio:.1f}:attack=5:release=80:makeup=2"
        params = {"agc": kind, "threshold": round(thr, 3), "ratio": round(ratio, 1)}
    else:
        raise ValueError(kind)
    return ffmpeg_filter(audio, sr, graph), params


def _frames(audio, n):
    usable = len(audio) // n * n
    return audio[:usable].reshape(-1, n), audio[usable:]


def vad_gate(audio, sr, rng, frame_ms=20):
    """
    Energy VAD with hangover, a proxy for RTC VAD / DTX: frames judged non-speech are replaced
    by silence or by low comfort noise. Returns (audio, params).
    """
    n = int(sr * frame_ms / 1000)
    frames, tail = _frames(np.asarray(audio, dtype=np.float32), n)
    if len(frames) < 3:
        return audio, {"vad": "skip"}
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
    margin = float(rng.uniform(6, 15))
    thr = np.percentile(db, 10) + margin
    hang = int(rng.integers(2, 11))
    active = db > thr
    keep = active.copy()
    for i in np.flatnonzero(active):                 # hangover after every active frame
        keep[i:i + hang + 1] = True
    fill = "cng" if rng.random() < 0.5 else "zero"
    out = frames.copy()
    floor = float(np.sqrt(np.mean(frames[~active] ** 2))) if (~active).any() else 0.0
    gated = ~keep
    if fill == "cng":
        out[gated] = rng.standard_normal((int(gated.sum()), n)).astype(np.float32) * np.float32(floor * 0.5)
    else:
        out[gated] = 0.0
    return np.concatenate([out.reshape(-1), tail]), {"vad": fill, "margin_db": round(margin, 1),
                                                       "hang": hang, "gated": round(float(gated.mean()), 3)}


def packet_loss(audio, sr, rng, frame_ms=20):
    """
    Gilbert-Elliott burst loss on 20 ms frames, concealed by repeating the last good frame
    with decaying gain (silence after a few lost frames). Returns (audio, params).
    """
    n = int(sr * frame_ms / 1000)
    frames, tail = _frames(np.asarray(audio, dtype=np.float32), n)
    p_gb, p_bg = float(rng.uniform(0.01, 0.08)), float(rng.uniform(0.3, 0.7))
    out, bad, last, run = frames.copy(), False, np.zeros(n, dtype=np.float32), 0
    lost = 0
    for i in range(len(frames)):
        bad = (rng.random() < p_gb) if not bad else (rng.random() >= p_bg)
        if bad:
            run += 1
            lost += 1
            out[i] = last * np.float32(0.5 ** run) if run <= 3 else 0.0
        else:
            run = 0
            last = frames[i]
    rate = lost / max(1, len(frames))
    return np.concatenate([out.reshape(-1), tail]), {"plc": "repeat", "loss": round(rate, 3)}


def bandlimit(audio, sr, rng, cutoffs=(3400, 4000, 6000, 7000)):
    """Low-pass at a random telephone-ish cutoff, or a round trip through 8 kHz."""
    if rng.random() < 0.3:
        return resample(resample(audio, sr, 8000), 8000, sr), {"bw": "resample8k"}
    fc = int(cutoffs[int(rng.integers(len(cutoffs)))])
    sos = butter(8, fc, btype="low", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32), {"bw": fc}


def aec_residual(echo, sr, rng):
    """What an imperfect echo canceller leaves of an echo: attenuated, distorted, high-passed."""
    atten_db = float(rng.uniform(15, 30))
    drive = float(rng.uniform(1, 5))
    x = np.tanh(drive * echo) / drive
    sos = butter(4, float(rng.uniform(200, 400)), btype="high", fs=sr, output="sos")
    x = sosfilt(sos, x).astype(np.float32) * np.float32(10 ** (-atten_db / 20))
    return x, {"aec_atten_db": round(atten_db, 1), "drive": round(drive, 2)}


def safe(fn, audio, *args, **kwargs):
    """Run a stage; on an ffmpeg failure keep the input and report it in the params."""
    try:
        return fn(audio, *args, **kwargs)
    except (OSError, subprocess.CalledProcessError) as exc:
        return audio, {"failed": f"{fn.__name__}: {exc}"}
