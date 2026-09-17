

import os
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from scipy import signal

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3")



@dataclass
class RTCAugConfig:
    noise_dirs: List[str] = field(default_factory=list)   # ESC-50, MUSAN/noise, BSD10k ...
    music_dirs: List[str] = field(default_factory=list)   # MUSAN/music, FMA-small ...
    rir_dirs: List[str] = field(default_factory=list)     # RIRS_NOISES, MIT RIR, Aachen ...

    p_apply: float = 0.8            
    max_stages: int = 4             

    p_speed: float = 0.15
    p_rir: float = 0.25
    p_noise: float = 0.35
    p_music: float = 0.20
    p_suppress: float = 0.30        
    p_drc: float = 0.30
    p_agc: float = 0.30
    p_bandlimit: float = 0.15
    p_resample8k: float = 0.15
    p_codec: float = 0.60
    p_packet_loss: float = 0.20
    p_peak_norm: float = 0.80
    p_fade: float = 0.20
    p_trim_pad: float = 0.30

    # ranges
    speed_range = (0.95, 1.05)
    noise_snr_range = (4.0, 25.0)
    music_snr_range = (8.0, 22.0)
    bandlimit_hz = (300.0, 3400.0)
    peak_dbfs_range = (-12.0, -1.0)
    agc_target_rms_db_range = (-28.0, -18.0)
    packet_frame_ms = (10, 40)
    packet_p_enter = (0.02, 0.08)   
    packet_p_stay = (0.3, 0.7) 

    # codecs: (ffmpeg encoder, mux format, demux format, bitrate list in kbps)
    codecs = {
        "opus": ("libopus", "ogg", "ogg", [6, 8, 12, 16, 24, 32]),
        "mp3": ("libmp3lame", "mp3", "mp3", [24, 32, 48, 64]),
        "aac": ("aac", "adts", "aac", [24, 32, 48, 64]),
        "amrwb": ("libvo_amrwbenc", "amr", "amr", [6.6, 8.85, 12.65, 15.85, 23.85]),
    }
    codec_weights = {"opus": 0.5, "mp3": 0.2, "aac": 0.2, "amrwb": 0.1}

    
    suppress_fn: Optional[Callable[[np.ndarray, int], np.ndarray]] = None

    seed: Optional[int] = None



def _list_audio(dirs: List[str]) -> List[str]:
    out = []
    for d in dirs:
        if not d:
            continue
        for p in Path(d).rglob("*"):
            if p.suffix.lower() in AUDIO_EXT:
                out.append(str(p))
    return out


def _load_audio(path: str, sr: int) -> np.ndarray:
    import librosa  
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def _fit_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def _random_crop_or_tile(x: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    if len(x) >= n:
        s = rng.randint(0, len(x) - n)
        return x[s:s + n]
    reps = n // max(len(x), 1) + 1
    return np.tile(x, reps)[:n]



def speed_perturb(x, sr, factor):
    n_out = int(round(len(x) / factor))
    return signal.resample(x, n_out).astype(np.float32)


def add_at_snr(x, noise, snr_db):
    if _rms(noise) < 1e-6:
        return x
    target_noise_rms = _rms(x) / (10 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / _rms(noise))
    return (x + noise).astype(np.float32)


def apply_rir(x, rir):
    rir = rir.astype(np.float32)
    k = int(np.argmax(np.abs(rir)))
    rir = rir[k:]
    rir = rir / (np.sqrt(np.sum(rir ** 2)) + 1e-8)
    y = signal.fftconvolve(x, rir, mode="full")[: len(x)]
    y = y * (_rms(x) / (_rms(y) + 1e-8))
    return y.astype(np.float32)


def bandlimit(x, sr, lo, hi, order=6):
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def resample_via_8k(x, sr):
    down = signal.resample_poly(x, 8000, sr)
    up = signal.resample_poly(down, sr, 8000)
    return _fit_len(up.astype(np.float32), len(x))


def dynamic_range_compress(x, sr, threshold_db=-20.0, ratio=4.0, attack_ms=5.0, release_ms=50.0,
                           makeup=True):
    eps = 1e-8
    att = np.exp(-1.0 / (sr * attack_ms / 1000.0))
    rel = np.exp(-1.0 / (sr * release_ms / 1000.0))
    env = np.zeros_like(x)
    e = 0.0
    ax = np.abs(x)
    for i in range(len(x)):          
        a = ax[i]
        e = att * e + (1 - att) * a if a > e else rel * e + (1 - rel) * a
        env[i] = e
    env_db = 20 * np.log10(env + eps)
    over = np.maximum(env_db - threshold_db, 0.0)
    gain_db = -over * (1 - 1 / ratio)
    y = x * (10 ** (gain_db / 20.0))
    if makeup:
        y = y * (_rms(x) / (_rms(y) + eps))
    return y.astype(np.float32)


def agc(x, target_rms_db):
    target = 10 ** (target_rms_db / 20.0)
    y = x * (target / (_rms(x) + 1e-8))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def peak_normalize(x, peak_dbfs):
    peak = np.max(np.abs(x)) + 1e-8
    return (x * (10 ** (peak_dbfs / 20.0)) / peak).astype(np.float32)


def fade(x, sr, fin_ms, fout_ms):
    y = x.copy()
    n_in = min(int(sr * fin_ms / 1000), len(y))
    n_out = min(int(sr * fout_ms / 1000), len(y))
    if n_in > 0:
        y[:n_in] *= np.linspace(0, 1, n_in, dtype=np.float32)
    if n_out > 0:
        y[-n_out:] *= np.linspace(1, 0, n_out, dtype=np.float32)
    return y


def trim_pad(x, sr, rng: random.Random, top_db=30):
    import librosa
    y, _ = librosa.effects.trim(x, top_db=top_db)
    if len(y) < sr // 2:         
        y = x
    lead = int(sr * rng.uniform(0, 0.4))
    trail = int(sr * rng.uniform(0, 0.5))
    return np.pad(y, (lead, trail)).astype(np.float32)


def packet_loss(x, sr, frame_ms, p_enter, p_stay, conceal=True):
    n = int(sr * frame_ms / 1000)
    y = x.copy()
    lost = False
    last_good = np.zeros(n, dtype=np.float32)
    for s in range(0, len(y) - n + 1, n):
        lost = (random.random() < p_stay) if lost else (random.random() < p_enter)
        if lost:
            if conceal:
                y[s:s + n] = last_good * 0.6
            else:
                y[s:s + n] = 0.0
        else:
            last_good = y[s:s + n].copy()
    return y



class FFmpegCodec:
    def __init__(self):
        self.ffmpeg = shutil.which("ffmpeg")
        self._available = {}

    def ok(self) -> bool:
        return self.ffmpeg is not None

    def encoder_available(self, enc: str) -> bool:
        if enc in self._available:
            return self._available[enc]
        if not self.ok():
            self._available[enc] = False
            return False
        try:
            out = subprocess.run([self.ffmpeg, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=10).stdout
            self._available[enc] = f" {enc} " in out or f" {enc}\n" in out
        except Exception:
            self._available[enc] = False
        return self._available[enc]

    def roundtrip(self, x: np.ndarray, sr: int, enc: str, fmt: str, kbps: float,
                  demux_fmt: Optional[str] = None) -> np.ndarray:
        pcm = x.astype(np.float32).tobytes()
        enc_sr = 16000 if enc == "libvo_amrwbenc" else sr
        enc_cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error",
                   "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                   "-ar", str(enc_sr), "-c:a", enc, "-b:a", f"{kbps}k"]
        if enc == "libopus":
            enc_cmd += ["-application", "voip", "-frame_duration", "20"]
        enc_cmd += ["-f", fmt, "pipe:1"]
        encoded = subprocess.run(enc_cmd, input=pcm, capture_output=True, timeout=30)
        if encoded.returncode != 0 or not encoded.stdout:
            raise RuntimeError(encoded.stderr.decode(errors="ignore"))
        dec_cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error",
                   "-f", demux_fmt or fmt, "-i", "pipe:0",
                   "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1"]
        decoded = subprocess.run(dec_cmd, input=encoded.stdout, capture_output=True, timeout=30)
        if decoded.returncode != 0 or not decoded.stdout:
            raise RuntimeError(decoded.stderr.decode(errors="ignore"))
        y = np.frombuffer(decoded.stdout, dtype=np.float32)
        y = _align(x, y, sr)
        return _fit_len(y, len(x))


def _align(ref, y, sr, max_lag_ms=60):
    n = min(len(ref), len(y), sr)
    if n < sr // 4:
        return y
    max_lag = int(sr * max_lag_ms / 1000)
    a, b = ref[:n], y[:n]
    corr = signal.correlate(b, a, mode="full", method="fft")
    lags = np.arange(-n + 1, n)
    mask = np.abs(lags) <= max_lag
    lag = int(lags[mask][np.argmax(corr[mask])])
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        y = np.pad(y, (-lag, 0))
    return y



class RTCAugmenter:
    def __init__(self, cfg: RTCAugConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.noise_files = _list_audio(cfg.noise_dirs)
        self.music_files = _list_audio(cfg.music_dirs)
        self.rir_files = _list_audio(cfg.rir_dirs)
        self.codec = FFmpegCodec()
        self._rir_cache = {}
        if not self.noise_files:
            print("[RTCAugmenter] no noise files found -> additive noise disabled")
        if not self.rir_files:
            print("[RTCAugmenter] no RIR files found -> reverberation disabled")
        if not self.codec.ok():
            print("[RTCAugmenter] ffmpeg not found -> codec augmentation disabled")

    def _noise(self, x, sr, files, snr_range):
        path = self.rng.choice(files)
        try:
            nz = _load_audio(path, sr)
        except Exception:
            return x
        nz = _random_crop_or_tile(nz, len(x), self.rng)
        return add_at_snr(x, nz, self.rng.uniform(*snr_range))

    def _rir(self, x, sr):
        path = self.rng.choice(self.rir_files)
        if path not in self._rir_cache:
            try:
                self._rir_cache[path] = _load_audio(path, sr)[: sr]  # ≤1 s tail is plenty
            except Exception:
                return x
        return apply_rir(x, self._rir_cache[path])

    def _codec(self, x, sr):
        names = list(self.cfg.codec_weights.keys())
        weights = [self.cfg.codec_weights[n] for n in names]
        for _ in range(3):  # retry with another codec if encoder is missing
            name = self.rng.choices(names, weights=weights, k=1)[0]
            enc, fmt, demux, brs = self.cfg.codecs[name]
            if not self.codec.encoder_available(enc):
                continue
            try:
                return self.codec.roundtrip(x, sr, enc, fmt, self.rng.choice(brs), demux)
            except Exception:
                continue
        return x

    def __call__(self, x: np.ndarray, sr: int = 16000) -> np.ndarray:
        c = self.cfg
        if self.rng.random() > c.p_apply:
            return x
        x = np.asarray(x, dtype=np.float32)

        # RTC-ordered candidate stages: (prob, fn)
        stages = [
            (c.p_speed, lambda a: speed_perturb(a, sr, self.rng.uniform(*c.speed_range))),
            (c.p_trim_pad, lambda a: trim_pad(a, sr, self.rng)),
            (c.p_rir if self.rir_files else 0.0, lambda a: self._rir(a, sr)),
            (c.p_noise if self.noise_files else 0.0,
             lambda a: self._noise(a, sr, self.noise_files, c.noise_snr_range)),
            (c.p_music if self.music_files else 0.0,
             lambda a: self._noise(a, sr, self.music_files, c.music_snr_range)),
            (c.p_suppress if c.suppress_fn else 0.0, lambda a: c.suppress_fn(a, sr)),
            (c.p_drc, lambda a: dynamic_range_compress(
                a, sr, threshold_db=self.rng.uniform(-30, -12), ratio=self.rng.uniform(2, 8))),
            (c.p_agc, lambda a: agc(a, self.rng.uniform(*c.agc_target_rms_db_range))),
            (c.p_bandlimit, lambda a: bandlimit(a, sr, *c.bandlimit_hz)),
            (c.p_resample8k, lambda a: resample_via_8k(a, sr)),
            (c.p_codec if self.codec.ok() else 0.0, lambda a: self._codec(a, sr)),
            (c.p_packet_loss, lambda a: packet_loss(
                a, sr, self.rng.randint(*c.packet_frame_ms),
                self.rng.uniform(*c.packet_p_enter), self.rng.uniform(*c.packet_p_stay))),
            (c.p_fade, lambda a: fade(a, sr, self.rng.uniform(20, 80), self.rng.uniform(50, 150))),
        ]

        chosen = [fn for p, fn in stages if self.rng.random() < p]
        if len(chosen) > c.max_stages:
            idx = sorted(self.rng.sample(range(len(chosen)), c.max_stages))
            chosen = [chosen[i] for i in idx]

        for fn in chosen:
            try:
                x = fn(x)
            except Exception as e:  
                print(f"[RTCAugmenter] stage failed: {e}")

        if self.rng.random() < c.p_peak_norm:
            x = peak_normalize(x, self.rng.uniform(*c.peak_dbfs_range))
        return np.clip(x, -1.0, 1.0).astype(np.float32)



def make_deepfilternet_suppressor():
    """Returns fn(audio, sr) -> audio, or None if DeepFilterNet isn't installed.
    Public weights, trained for enhancement (not spoofing) -> allowed under RTCFake rules."""
    try:
        import torch
        from df.enhance import enhance, init_df
        model, df_state, _ = init_df(log_level="ERROR")
        df_sr = df_state.sr()

        def _fn(x, sr):
            xt = torch.from_numpy(x).unsqueeze(0)
            if sr != df_sr:
                xt = torch.from_numpy(signal.resample_poly(x, df_sr, sr).astype(np.float32)).unsqueeze(0)
            y = enhance(model, df_state, xt).squeeze(0).numpy()
            if sr != df_sr:
                y = signal.resample_poly(y, sr, df_sr).astype(np.float32)
            return _fit_len(y, len(x))
        return _fn
    except Exception:
        return None
