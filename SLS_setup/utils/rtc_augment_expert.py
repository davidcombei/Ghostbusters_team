"""
RTC-SDD training-time augmentation for single-condition expert models.

Same building blocks as rtc_augment.py, but only ONE acoustic condition is
ever used, chosen up front:

    stage       scenario    source
    ------------------------------------------------------------------
    office      S02         RNNoise background noise (Xiph)
    coffee      S03         RNNoise background noise (Xiph)
    echo        S04         CLAD's AddEchoes
    rain        S05         ESC-50
    footsteps   S06         ESC-50
    keyboard    S07         ESC-50
    reverb      reverb      measured RIRs shipped with the RNNoise data
    musan_noise optional    MUSAN noise/ (free-sound + sound-bible), scanned recursively
    musan_music optional    MUSAN music/ (fma, jamendo, rfm, ...), scanned recursively

Per clip, two independent draws:
    - with probability `p_noise` (default 0.5) the expert's condition is applied;
    - with probability `p_codec` (default 0.5) one codec, drawn uniformly from
      EXPERT_CODECS unless narrowed, is applied after the condition when both fire.
So a clip ends up clean, condition only, codec only, or condition -> codec
(each 25% with the defaults).

Codecs:
    RTC apps    qq, wechat, zoom, dingtalk, lark, voov, telegram
                (Opus profiles, codecs_augm via the ffmpeg CLI)
    telephony   g711_ulaw, g711_alaw (64 kbps), g722 (64 kbps, 16 kHz),
                g723_1 (6.3 kbps), g726 (16/24/32/40 kbps),
                amr_nb (4.75..12.2 kbps); all narrowband at 8 kHz except G.722,
                round-tripped in memory through PyAV's bundled ffmpeg
    G.728, G.729 and iLBC are missing: neither ffmpeg build here can encode them.

The helpers (add_noise, add_echo, add_reverb, codecs_augm, ...) are imported
from rtc_augment.py, which is left untouched.
"""

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .rtc_augment import (CODEC_PROFILES, RIR_EXTS, add_echo, add_noise, add_reverb,
                          codecs_augm, finalize, list_audio, make_rng)

NOISE_STAGES = ("office", "coffee", "rain", "footsteps", "keyboard")   # read from noise pools
MUSAN_STAGES = {"musan_noise": "noise", "musan_music": "music"}      # stage -> subdir of musan_dir
EXPERT_STAGES = NOISE_STAGES + ("echo", "reverb") + tuple(MUSAN_STAGES)

# name -> (PyAV encoder, codec sample rate, bitrates to draw from; None = fixed by the codec)
TELEPHONY_CODECS = {
    "g711_ulaw": ("pcm_mulaw", 8000, (None,)),
    "g711_alaw": ("pcm_alaw", 8000, (None,)),
    "g722":      ("g722", 16000, (None,)),
    "g723_1":    ("g723_1", 8000, (6300,)),
    "g726":      ("g726", 8000, (16000, 24000, 32000, 40000)),
    "amr_nb":    ("libopencore_amrnb", 8000, (4750, 5150, 5900, 6700, 7400, 7950, 10200, 12200)),
}
EXPERT_CODECS = tuple(CODEC_PROFILES) + tuple(TELEPHONY_CODECS)


def _resample(audio, sr_in, sr_out, fmt):
    import av
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio, np.float32).reshape(1, -1),
                                       format="flt", layout="mono")
    frame.sample_rate = sr_in
    resampler = av.AudioResampler(format=fmt, layout="mono", rate=sr_out)
    frames = resampler.resample(frame) + resampler.resample(None)
    return np.concatenate([f.to_ndarray().reshape(-1) for f in frames])


def telephony_augm(audio, rng, sr, name):
    """
    Encode + decode through a telephony codec. The bitstream goes through an
    in-memory WAV so the decoder gets the codec parameters (G.726 needs them).
    """
    import av
    encoder, codec_sr, bitrates = TELEPHONY_CODECS[name]
    bitrate = bitrates[int(rng.integers(len(bitrates)))]
    try:
        pcm = _resample(audio, sr, codec_sr, "s16")
        buf = io.BytesIO()
        with av.open(buf, "w", format="wav") as container:
            stream = container.add_stream(encoder, rate=codec_sr, layout="mono")
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
        return _resample(decoded, codec_sr, sr, "flt")
    except Exception as exc:                        # never kill a training step
        print(f"[rtc_augment_expert] codec {name} failed ({exc}); passing through")
        return audio


@dataclass
class ExpertAugConfig:
    """Pools and knobs for :class:`ExpertAugmenter`."""

    stage: str = "office"
    noise_dirs: Sequence[str] = field(default_factory=list)
    rir_dirs: Sequence[str] = field(default_factory=list)
    musan_dir: Optional[str] = None
    p_noise: float = 0.5
    # Codec drawn independently of the condition: clean clips can get one, noisy ones can skip it.
    p_codec: float = 0.5
    seed: Optional[int] = None
    sr: int = 16000
    snr_db: Tuple[float, float] = (5.0, 20.0)
    echo_delay_ms: Tuple[float, float] = (40.0, 160.0)
    echo_strength: Tuple[float, float] = (0.15, 0.5)
    rir_max_seconds: float = 2.0
    codec_apps: Sequence[str] = field(default_factory=lambda: list(EXPERT_CODECS))
    verbose: bool = True


class ExpertAugmenter:
    """
    ``audio = augmenter(audio, sr)``: the expert's condition with probability
    p_noise, then, independently, one codec with probability p_codec.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or ExpertAugConfig()
        stage = self.cfg.stage.lower()
        if stage not in EXPERT_STAGES:
            raise ValueError(f"unknown expert stage '{stage}', choose from {EXPERT_STAGES}")
        self.stage = stage
        unknown = [a for a in self.cfg.codec_apps if a not in EXPERT_CODECS]
        if unknown or not self.cfg.codec_apps:
            raise ValueError(f"bad codec apps {unknown or '[]'}, choose from {list(EXPERT_CODECS)}")

        self.noise_files: List[Path] = []
        self.rir_files: List[Path] = []
        if stage in NOISE_STAGES:
            for directory in self.cfg.noise_dirs:
                self.noise_files += self._find_pool(directory, stage)
            if not self.noise_files:
                raise FileNotFoundError(f"no '{stage}' pool found under {list(self.cfg.noise_dirs)}")
        elif stage in MUSAN_STAGES:
            root = Path(self.cfg.musan_dir or "") / MUSAN_STAGES[stage]
            self.noise_files = list_audio(root, recursive=True)
            if not self.noise_files:
                raise FileNotFoundError(f"no MUSAN '{stage}' files under {root}")
        elif stage == "reverb":
            for directory in self.cfg.rir_dirs:
                self.rir_files += list_audio(directory, exts=RIR_EXTS)
            if not self.rir_files:
                raise FileNotFoundError(f"no RIRs found under {list(self.cfg.rir_dirs)}")

        self.stage_counts: Dict[str, int] = {}      # per-process tally, for logging
        self.rng = None
        self.rng_pid = None

        if self.cfg.verbose:
            print(f"[rtc_augment_expert] {self.describe()}")

    @staticmethod
    def _find_pool(directory, stage):
        """`directory/<stage>/`, or `directory` itself when it is that pool."""
        root = Path(directory)
        if root.name.lower() == stage:
            return list_audio(root)
        sub = root / stage
        return list_audio(sub) if sub.is_dir() else []

    def describe(self):
        if self.stage in NOISE_STAGES or self.stage in MUSAN_STAGES:
            low, high = self.cfg.snr_db
            pool = f"files={len(self.noise_files)} snr={low:g}..{high:g} dB"
        elif self.stage == "reverb":
            pool = f"rirs={len(self.rir_files)}"
        else:
            pool = (f"delay={self.cfg.echo_delay_ms[0]:g}..{self.cfg.echo_delay_ms[1]:g} ms "
                    f"strength={self.cfg.echo_strength[0]:g}..{self.cfg.echo_strength[1]:g}")
        return (f"expert={self.stage} p_noise={self.cfg.p_noise} ({pool}) "
                f"-> codec p_codec={self.cfg.p_codec} [{', '.join(self.cfg.codec_apps)}]")

    def _get_rng(self):
        if self.rng is None or self.rng_pid != os.getpid():
            self.rng = make_rng(self.cfg.seed)
            self.rng_pid = os.getpid()
        return self.rng

    def _count(self, stage):
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

    def __call__(self, audio, sr=None):
        cfg = self.cfg
        sr = sr or cfg.sr
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < 16:
            return audio

        rng = self._get_rng()
        apply_noise = rng.random() < cfg.p_noise
        apply_codec = rng.random() < cfg.p_codec
        if not (apply_noise or apply_codec):
            self._count("clean")
            return audio

        out = audio
        if apply_noise:
            self._count(self.stage)
            if self.stage == "echo":
                out = add_echo(out, rng, cfg.echo_delay_ms, cfg.echo_strength, sr)
            elif self.stage == "reverb":
                out = add_reverb(out, self.rir_files, rng, sr, cfg.rir_max_seconds)
            else:
                out = add_noise(out, self.noise_files, rng, cfg.snr_db, sr)
            out = finalize(out, audio)

        if apply_codec:
            app = cfg.codec_apps[int(rng.integers(len(cfg.codec_apps)))]
            if app in TELEPHONY_CODECS:
                out = telephony_augm(out, rng, sr, app)
            else:
                out, _ = codecs_augm(out, rng, sr, [app])
            self._count(f"codec:{app}")
            out = finalize(out, audio)
        return out
