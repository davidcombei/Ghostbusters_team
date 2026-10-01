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

Per clip, with probability `p_apply` (default 0.5) the expert's condition is
applied and then, always, one RTC app's codec (codecs_augm, every app in
CODEC_PROFILES unless narrowed). Otherwise the clip is left fully clean: no
condition and no codec.

The helpers (add_noise, add_echo, add_reverb, codecs_augm, ...) are imported
from rtc_augment.py, which is left untouched.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .rtc_augment import (CODEC_PROFILES, RIR_EXTS, add_echo, add_noise, add_reverb,
                          codecs_augm, finalize, list_audio, make_rng)

NOISE_STAGES = ("office", "coffee", "rain", "footsteps", "keyboard")   # read from noise pools
EXPERT_STAGES = NOISE_STAGES + ("echo", "reverb")


@dataclass
class ExpertAugConfig:
    """Pools and knobs for :class:`ExpertAugmenter`."""

    stage: str = "office"
    noise_dirs: Sequence[str] = field(default_factory=list)
    rir_dirs: Sequence[str] = field(default_factory=list)
    p_apply: float = 0.5
    seed: Optional[int] = None
    sr: int = 16000
    snr_db: Tuple[float, float] = (5.0, 20.0)
    echo_delay_ms: Tuple[float, float] = (40.0, 160.0)
    echo_strength: Tuple[float, float] = (0.15, 0.5)
    rir_max_seconds: float = 2.0
    # Codec always follows the condition (never applied to clean clips).
    codec_apps: Sequence[str] = field(default_factory=lambda: list(CODEC_PROFILES))
    verbose: bool = True


class ExpertAugmenter:
    """
    ``audio = augmenter(audio, sr)``: with probability p_apply the expert's
    condition + one codec in series, otherwise the clip untouched.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or ExpertAugConfig()
        stage = self.cfg.stage.lower()
        if stage not in EXPERT_STAGES:
            raise ValueError(f"unknown expert stage '{stage}', choose from {EXPERT_STAGES}")
        self.stage = stage
        unknown = [a for a in self.cfg.codec_apps if a not in CODEC_PROFILES]
        if unknown or not self.cfg.codec_apps:
            raise ValueError(f"bad codec apps {unknown or '[]'}, choose from {list(CODEC_PROFILES)}")

        self.noise_files: List[Path] = []
        self.rir_files: List[Path] = []
        if stage in NOISE_STAGES:
            for directory in self.cfg.noise_dirs:
                self.noise_files += self._find_pool(directory, stage)
            if not self.noise_files:
                raise FileNotFoundError(f"no '{stage}' pool found under {list(self.cfg.noise_dirs)}")
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
        if self.stage in NOISE_STAGES:
            low, high = self.cfg.snr_db
            pool = f"files={len(self.noise_files)} snr={low:g}..{high:g} dB"
        elif self.stage == "reverb":
            pool = f"rirs={len(self.rir_files)}"
        else:
            pool = (f"delay={self.cfg.echo_delay_ms[0]:g}..{self.cfg.echo_delay_ms[1]:g} ms "
                    f"strength={self.cfg.echo_strength[0]:g}..{self.cfg.echo_strength[1]:g}")
        return (f"expert={self.stage} p_apply={self.cfg.p_apply} ({pool}) "
                f"-> codec always [{', '.join(self.cfg.codec_apps)}] | else clean")

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
        if rng.random() >= cfg.p_apply:
            self._count("clean")
            return audio

        self._count(self.stage)
        if self.stage == "echo":
            out = add_echo(audio, rng, cfg.echo_delay_ms, cfg.echo_strength, sr)
        elif self.stage == "reverb":
            out = add_reverb(audio, self.rir_files, rng, sr, cfg.rir_max_seconds)
        else:
            out = add_noise(audio, self.noise_files, rng, cfg.snr_db, sr)
        out = finalize(out, audio)

        out, app = codecs_augm(out, rng, sr, cfg.codec_apps)
        self._count(f"codec:{app}")
        return finalize(out, audio)
