"""
RTC-SDD training-time augmentation.

The challenge eval set holds a noisy subset whose conditions come from three
upstream sources; this module puts the same conditions on the training clips:

    stage       scenario    source
    ------------------------------------------------------------------
    office      S02         RNNoise background noise (Xiph)
    coffee      S03         RNNoise background noise (Xiph)
    echo        S04         CLAD's AddEchoes
    rain        S05         ESC-50
    footsteps   S06         ESC-50
    keyboard    S07         ESC-50
    reverb      reverb      measured RIRs shipped with the RNNoise data
    musan       optional    MUSAN non-speech pool (data/augm/musan)
    music       optional    any user-supplied music pool
    suppress    optional    DeepFilterNet noise suppression
    codec       in series   QQ / Zoom / WeChat / DingTalk / Lark / VooV / Telegram

    https://media.xiph.org/rnnoise/data/     (noise + measured_rirs-v3.tar.gz)
    https://github.com/CLAD23/CLAD           (AddEchoes, in DatasetUtils.py)
    https://github.com/karolpiczak/ESC-50    (rain / footsteps / keyboard_typing)

One augmentation per clip at most: with probability `p_apply` one stage is
drawn, every stage with the same chance, and applied on its own. Stages are
never chained, because the noisy eval clips carry a single condition each.
The one exception is the codec: with probability `codec_p_apply` the clip is
then passed through one RTC app's codec (codecs_augm), since every eval clip
went through an app whatever its acoustic condition.

No noise is synthesised here -- every sample mixed in is read from a file in
the pools above. Synthetic noise is RawBoost's job, and RawBoost is a separate
option (--use_rawboost / --no_rawboost).

Build the pools with scripts/prepare_rtc_aug_data.py, then train with

    --use_rtc_aug --aug_noise_dirs data/augm/noise --aug_rir_dirs data/augm/measured_rirs
"""

import os
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3")      # noise pools: decoded by soundfile
RIR_EXTS = AUDIO_EXTS + (".f32",)                   # RIRs may also be headerless float32
RIR_SR = 48000          # sample rate of the headerless measured_rirs .f32 files


# --------------------------------------------------------------------------- #
# reading files
# --------------------------------------------------------------------------- #
def list_audio(directory, recursive=False, exts=AUDIO_EXTS):
    """Audio files directly in `directory`, or anywhere under it."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    paths = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in exts)


def resample(audio, sr, target_sr):
    """Exact-ratio resampling: 48000 -> 16000 is /3, 44100 -> 16000 is 160/441."""
    if sr == target_sr or audio.size == 0:
        return np.asarray(audio, dtype=np.float32)
    step = np.gcd(int(sr), int(target_sr))
    return resample_poly(audio, target_sr // step, sr // step).astype(np.float32)


def read_window(path, n_samples, sr, rng):
    """
    `n_samples` from a random position in a noise file, tiled if it is shorter.
    Only the part that is needed is decoded, so a long MUSAN recording costs the
    same as a short ESC-50 clip.
    """
    info = sf.info(str(path))
    if info.frames == 0:
        return np.zeros(n_samples, dtype=np.float32)

    needed = int(np.ceil(n_samples * info.samplerate / sr)) + 64
    if info.frames > needed:
        start = int(rng.integers(0, info.frames - needed))
        block, file_sr = sf.read(str(path), start=start, frames=needed,
                                 dtype="float32", always_2d=False)
    else:
        block, file_sr = sf.read(str(path), dtype="float32", always_2d=False)

    if block.ndim == 2:
        block = block.mean(axis=1)
    block = resample(block, file_sr, sr)
    if block.size == 0:
        return np.zeros(n_samples, dtype=np.float32)
    if block.size < n_samples:
        block = np.tile(block, n_samples // block.size + 1)
    offset = int(rng.integers(0, block.size - n_samples + 1))
    return block[offset:offset + n_samples]


def read_rir(path, sr):
    """A whole impulse response: headerless float32 for .f32, else a normal read."""
    path = Path(path)
    if path.suffix.lower() == ".f32":
        rir, file_sr = np.fromfile(str(path), dtype="<f4"), RIR_SR
    else:
        rir, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if rir.ndim == 2:
            rir = rir.mean(axis=1)
    return resample(rir, file_sr, sr)


# --------------------------------------------------------------------------- #
# one function per augmentation: audio in, audio out
# --------------------------------------------------------------------------- #
def add_noise(audio, noise_files, rng, snr_db=(5.0, 20.0), sr=16000):
    """Mix one recording from `noise_files` under the speech at a random SNR."""
    path = noise_files[int(rng.integers(len(noise_files)))]
    noise = read_window(path, len(audio), sr, rng)

    noise_power = float(np.mean(noise ** 2))
    speech_power = float(np.mean(audio ** 2))
    if noise_power <= 1e-12 or speech_power <= 1e-12:
        return audio

    snr = 10.0 ** (float(rng.uniform(*snr_db)) / 10.0)
    gain = np.sqrt(speech_power / (snr * noise_power))
    return audio + noise * np.float32(gain)


def add_echo(audio, rng, delay_ms=(40.0, 160.0), strength=(0.15, 0.5), sr=16000):
    """CLAD's AddEchoes: one attenuated delayed copy of the clip, summed on top."""
    delay = int(float(rng.uniform(*delay_ms)) * sr / 1000.0)
    delay = max(1, min(delay, len(audio) - 1))
    out = audio.copy()
    out[delay:] += audio[:-delay] * np.float32(rng.uniform(*strength))
    return out


def add_reverb(audio, rir_files, rng, sr=16000, max_seconds=2.0):
    """Convolve with a measured RIR, keeping the clip's length and level."""
    rir = read_rir(rir_files[int(rng.integers(len(rir_files)))], sr)
    if rir.size < 2:
        return audio
    rir = rir[int(np.argmax(np.abs(rir))):]         # drop the propagation delay
    rir = rir[:int(max_seconds * sr)]
    norm = float(np.sqrt(np.sum(rir ** 2)))
    if rir.size < 2 or norm <= 1e-8:
        return audio
    wet = fftconvolve(audio, rir / norm)[:len(audio)]
    return match_level(wet, audio)


# RTC app -> (bitrate kbps range, bandwidth cutoffs Hz, frame ms, libopus application).
# Every app here ships Opus or SILK; ffmpeg has no SILK encoder, so QQ/WeChat use
# Opus at low rate + narrow cutoff, which forces Opus into its SILK layer.
CODEC_PROFILES = {
    "qq":       ((10, 24), (6000, 8000), (20,), "voip"),        # SILK
    "wechat":   ((8, 20), (4000, 8000), (20,), "voip"),         # SILK v3
    "zoom":     ((24, 48), (8000,), (20,), "voip"),
    "dingtalk": ((16, 32), (8000,), (20, 40), "voip"),
    "lark":     ((24, 40), (8000,), (20,), "voip"),             # WebRTC Opus
    "voov":     ((16, 32), (6000, 8000), (20,), "voip"),
    "telegram": ((16, 32), (8000,), (20, 60), "audio"),         # Opus voice notes
}


def _ffmpeg(args, stdin):
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
                          input=stdin, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


def codecs_augm(audio, rng, sr=16000, apps=None):
    """Encode + decode through one RTC app's codec setting (ffmpeg libopus)."""
    apps = list(apps or CODEC_PROFILES)
    app = apps[int(rng.integers(len(apps)))]
    kbps, cutoffs, frames, application = CODEC_PROFILES[app]
    raw = ["-f", "f32le", "-ac", "1", "-ar", str(sr)]
    try:
        encoded = _ffmpeg([*raw, "-i", "pipe:0", "-c:a", "libopus",
                           "-b:a", f"{int(rng.integers(kbps[0], kbps[1] + 1))}k",
                           "-cutoff", str(cutoffs[int(rng.integers(len(cutoffs)))]),
                           "-frame_duration", str(frames[int(rng.integers(len(frames)))]),
                           "-application", application, "-f", "ogg", "pipe:1"],
                          np.asarray(audio, dtype=np.float32).tobytes())
        decoded = _ffmpeg(["-i", "pipe:0", *raw, "pipe:1"], encoded)
    except (OSError, subprocess.CalledProcessError) as exc:    # never kill a training step
        print(f"[rtc_augment] codec {app} failed ({exc}); passing through")
        return audio, app
    return np.frombuffer(decoded, dtype=np.float32), app


def suppress(audio, suppress_fn, sr=16000):
    """Run a noise suppressor (DeepFilterNet); RTC endpoints do this too."""
    try:
        return suppress_fn(audio, sr)
    except Exception as exc:                        # never kill a training step
        print(f"[rtc_augment] suppression failed ({exc}); passing through")
        return audio


# --------------------------------------------------------------------------- #
# keeping the output well-behaved
# --------------------------------------------------------------------------- #
def match_level(out, ref):
    """Rescale to the reference RMS so a stage cannot drift the level."""
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    if out_rms <= 1e-8 or ref_rms <= 1e-8:
        return out
    return out * np.float32(ref_rms / out_rms)


def finalize(out, audio):
    """Same length as the input, finite, and not clipping."""
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    out = out[:len(audio)]
    if not np.all(np.isfinite(out)):
        return audio
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out = out * np.float32(0.99 / peak)
    return out


def make_rng(seed):
    """
    One RNG per process. DataLoader workers are forked, so mixing the pid in is
    what stops all of them replaying the same augmentations.
    """
    base = random.randrange(2 ** 31) if seed is None else int(seed)
    return np.random.default_rng((base * 1_000_003 + os.getpid()) % (2 ** 63))


# --------------------------------------------------------------------------- #
# config + dispatch
# --------------------------------------------------------------------------- #
@dataclass
class RTCAugConfig:
    """Pools and knobs for :class:`RTCAugmenter`. Every field is optional."""

    noise_dirs: Sequence[str] = field(default_factory=list)
    music_dirs: Sequence[str] = field(default_factory=list)
    rir_dirs: Sequence[str] = field(default_factory=list)
    # MUSAN roots, scanned recursively. Used as-is with no filtering, so point
    # this at the non-speech part only.
    musan_dirs: Sequence[str] = field(default_factory=list)
    with_echo: bool = True
    p_apply: float = 0.8
    # Kept for CLI compatibility: one augmentation per clip, so > 1 only warns.
    max_stages: int = 1
    seed: Optional[int] = None
    suppress_fn: Optional[Callable] = None
    sr: int = 16000
    # How loud the pool recording sits under the speech, in dB SNR. One range
    # for every stage that mixes in a file.
    snr_db: Tuple[float, float] = (5.0, 20.0)
    echo_delay_ms: Tuple[float, float] = (40.0, 160.0)
    echo_strength: Tuple[float, float] = (0.15, 0.5)
    rir_max_seconds: float = 2.0
    # RTC codec applied after the stage above (or on its own if none was drawn).
    with_codecs: bool = False
    codec_p_apply: float = 1.0
    codec_apps: Sequence[str] = field(default_factory=lambda: list(CODEC_PROFILES))
    verbose: bool = True


class RTCAugmenter:
    """
    Picks one augmentation per clip and calls it: ``audio = augmenter(audio, sr)``.

    float32 mono numpy in, float32 mono numpy of the same length out, so it
    drops in ahead of cropping/padding and RawBoost.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or RTCAugConfig()
        if self.cfg.max_stages > 1 and self.cfg.verbose:
            print(f"[rtc_augment] max_stages={self.cfg.max_stages} ignored: "
                  f"one augmentation per clip at most")

        self.noise_files: Dict[str, List[Path]] = {}     # stage -> files
        for directory in self.cfg.noise_dirs:
            self._add_noise_dir(directory)
        for directory in self.cfg.music_dirs:
            self._add_pool("music", list_audio(directory))
        for directory in self.cfg.musan_dirs:
            self._add_pool("musan", list_audio(directory, recursive=True))

        self.rir_files: List[Path] = []
        for directory in self.cfg.rir_dirs:
            self.rir_files += list_audio(directory, exts=RIR_EXTS)

        self.stages = sorted(self.noise_files)
        if self.cfg.with_echo:
            self.stages.append("echo")              # needs no files
        if self.rir_files:
            self.stages.append("reverb")
        if self.cfg.suppress_fn is not None:
            self.stages.append("suppress")

        self.stage_counts: Dict[str, int] = {}      # per-process tally, for logging
        self.rng = None
        self.rng_pid = None

        if self.cfg.verbose:
            print(f"[rtc_augment] {self.describe()}")

    def _add_noise_dir(self, directory):
        """A directory of audio files is one pool; a directory of directories is one each."""
        root = Path(directory)
        if not root.is_dir():
            if self.cfg.verbose:
                print(f"[rtc_augment] missing pool dir: {root}")
            return
        self._add_pool(root.name.lower(), list_audio(root))
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            self._add_pool(sub.name.lower(), list_audio(sub))

    def _add_pool(self, stage, files):
        if files:
            self.noise_files.setdefault(stage, []).extend(files)

    def describe(self):
        pools = ", ".join(f"{s}:{len(f)}" for s, f in sorted(self.noise_files.items()))
        chance = f"1/{len(self.stages)}" if self.stages else "n/a"
        low, high = self.cfg.snr_db
        codecs = (f"p={self.cfg.codec_p_apply} [{', '.join(self.cfg.codec_apps)}]"
                  if self.cfg.with_codecs else "off")
        return (f"p_apply={self.cfg.p_apply} one stage/clip, equal chance {chance} | "
                f"snr={low:g}..{high:g} dB | stages=[{', '.join(self.stages)}] | "
                f"pools=({pools}) | rirs={len(self.rir_files)} | then codec {codecs}")

    def _get_rng(self):
        if self.rng is None or self.rng_pid != os.getpid():
            self.rng = make_rng(self.cfg.seed)
            self.rng_pid = os.getpid()
        return self.rng

    def __call__(self, audio, sr=None):
        cfg = self.cfg
        sr = sr or cfg.sr
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < 16:
            return audio

        rng = self._get_rng()
        out = self._stage(audio, rng, sr)
        if cfg.with_codecs and rng.random() < cfg.codec_p_apply:
            out, app = codecs_augm(out, rng, sr, cfg.codec_apps)
            self._count(f"codec:{app}")
            out = finalize(out, audio)
        return out

    def _count(self, stage):
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

    def _stage(self, audio, rng, sr):
        """At most one of the acoustic stages, drawn with equal chance."""
        cfg = self.cfg
        if not self.stages or rng.random() >= cfg.p_apply:
            return audio

        stage = self.stages[int(rng.integers(len(self.stages)))]    # equal chance
        self._count(stage)

        if stage == "echo":
            out = add_echo(audio, rng, cfg.echo_delay_ms, cfg.echo_strength, sr)
        elif stage == "reverb":
            out = add_reverb(audio, self.rir_files, rng, sr, cfg.rir_max_seconds)
        elif stage == "suppress":
            out = suppress(audio, cfg.suppress_fn, sr)
        else:
            out = add_noise(audio, self.noise_files[stage], rng, cfg.snr_db, sr)
        return finalize(out, audio)


# --------------------------------------------------------------------------- #
# DeepFilterNet, loaded once per process on first use
# --------------------------------------------------------------------------- #
_DF_MODEL = None


def deepfilternet_suppress(audio, sr=16000):
    """Denoise one clip with DeepFilterNet, resampling to its rate and back."""
    global _DF_MODEL
    import torch
    import torchaudio
    from df.enhance import enhance, init_df

    if _DF_MODEL is None:
        model, df_state, _ = init_df(config_allow_defaults=True)
        _DF_MODEL = (model, df_state)
    model, df_state = _DF_MODEL

    df_sr = df_state.sr()
    clip = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    if sr != df_sr:
        clip = torchaudio.functional.resample(clip, sr, df_sr)
    clean = enhance(model, df_state, clip)
    if sr != df_sr:
        clean = torchaudio.functional.resample(clean, df_sr, sr)
    return clean.squeeze(0).cpu().numpy().astype(np.float32)


def make_deepfilternet_suppressor(sr=16000):
    """Return the suppressor, or None when DeepFilterNet is not installed."""
    try:
        import df.enhance  # noqa: F401
    except Exception:
        return None
    return deepfilternet_suppress
