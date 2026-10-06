"""
Level augmentation: random gain + AGC / compressor / limiter (roadmap A2.7, P0 after E0.5).

Why: the E0.5 shortcut audit found spoof louder than bonafide in the training data (active
RMS / peak AUC 0.60-0.77), and the model uses it: +10 dB flips 30 % of bonafide clips to spoof.
Qwen's log-mel turns a gain into a constant offset of every input feature, so absolute level
is directly visible. Under noise, RTC platforms' AGC / compressors / limiters move level and
peaks freely, so the cue is unreliable at eval. This stage makes it uninformative in training.

Per clip, with probability `p_apply`:
    1. level: "absolute" (default) sets the active level to a random target in `target_db`, so the
       output level is independent of the input level; "relative" adds a gain in `gain_db`
       (keeps each clip's level offset, so the class gap only gets diluted, not removed)
    2. at most one dynamics stage: AGC | compressor | limiter | none
    3. overshoot (peak > 1): hard clip with probability `clip_p`, else the limiter

Numpy only (~1 ms per clip). The AGC / compressor / limiter are our own implementations;
sim-heldout uses ffmpeg dynaudnorm and the reserved acompressor (utils/heldout_reserve.py),
so it keeps measuring transfer to unseen AGC implementations. Class-independent by
construction: __call__ never sees the label.

    --use_level_aug [--level_p_apply 0.8 --level_gain_db -10 10 --level_agc_p 0.3
                     --level_comp_p 0.15 --level_limit_p 0.15 --level_clip_p 0.3]
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

from .rtc_augment import finalize

EPS = 1e-10


def _db(power):
    return 10.0 * np.log10(np.maximum(power, EPS))


def _frame_power(audio, n):
    """Mean power of consecutive n-sample frames (the tail joins the last frame)."""
    n_frames = max(1, len(audio) // n)
    usable = audio[:n_frames * n] if len(audio) >= n else np.pad(audio, (0, n - len(audio)))
    return np.mean(usable.reshape(n_frames, -1) ** 2, axis=1)


def _smooth(target, attack, release):
    """Asymmetric one-pole smoothing: `attack` coefficient when the value falls, `release` when it rises."""
    out = np.empty_like(target)
    state = target[0]
    for i, t in enumerate(target):
        coef = attack if t < state else release
        state = coef * state + (1.0 - coef) * t
        out[i] = state
    return out


def _apply_frame_gain(audio, gain_db, n):
    """Interpolate per-frame gains (dB, at frame centres) to samples and apply them."""
    centres = (np.arange(len(gain_db)) + 0.5) * n
    g = np.interp(np.arange(len(audio)), centres, gain_db)
    return (audio * np.power(10.0, g / 20.0)).astype(np.float32)


def _coef(tau_ms, step_ms):
    return float(np.exp(-step_ms / max(tau_ms, 1e-3)))


# --------------------------------------------------------------------------- #
# stages: float32 audio in, (float32 audio, params) out
# --------------------------------------------------------------------------- #
def active_level_db(audio, sr=16000, frame_ms=10.0, gate_db=35.0):
    """Mean power (dB) of the frames within `gate_db` of the loudest one: the speech level."""
    p = _frame_power(audio, int(sr * frame_ms / 1000))
    d = _db(p)
    return float(_db(np.mean(p[d > d.max() - gate_db])))


def apply_gain(audio, rng, gain_db=(-10.0, 10.0)):
    g = float(rng.uniform(*gain_db))
    return (audio * np.float32(10 ** (g / 20))).astype(np.float32), {"gain_db": round(g, 2)}


def set_level(audio, sr, rng, target_db=(-38.0, -12.0)):
    """Scale so the active level hits a random absolute target (input level is discarded)."""
    target = float(rng.uniform(*target_db))
    g = target - active_level_db(audio, sr)
    return (audio * np.float32(10 ** (g / 20))).astype(np.float32), {"target_db": round(target, 1)}


def agc(audio, sr, rng, target_db=(-28.0, -16.0), attack_ms=(5.0, 20.0), release_ms=(100.0, 500.0),
        max_gain_db=20.0, gate_db=40.0, frame_ms=10.0):
    """
    Digital AGC (WebRTC-like): pulls the active level toward a target with attack / release
    smoothing; frames more than `gate_db` below the loudest frame hold the current gain.
    """
    n = int(sr * frame_ms / 1000)
    fdb = _db(_frame_power(audio, n))
    target = float(rng.uniform(*target_db))
    active = fdb > fdb.max() - gate_db
    desired = np.clip(target - fdb, -max_gain_db, max_gain_db)
    if active.any():                                    # gated frames keep the previous gain
        idx = np.where(active, np.arange(len(fdb)), 0)
        np.maximum.accumulate(idx, out=idx)
        first = int(np.argmax(active))
        idx[:first] = first
        desired = desired[idx]
    att, rel = float(rng.uniform(*attack_ms)), float(rng.uniform(*release_ms))
    gain_db = _smooth(desired, _coef(att, frame_ms), _coef(rel, frame_ms))
    return _apply_frame_gain(audio, gain_db, n), {"stage": "agc", "target_db": round(target, 1),
                                                  "attack_ms": round(att), "release_ms": round(rel)}


def compress(audio, sr, rng, threshold_db=(-30.0, -12.0), ratio=(2.0, 8.0), attack_ms=(2.0, 10.0),
             release_ms=(50.0, 300.0), frame_ms=2.0):
    """Feed-forward compressor on a 2 ms RMS envelope, with make-up gain back to the input RMS."""
    n = int(sr * frame_ms / 1000)
    env_db = _db(_frame_power(audio, n))
    thr, r = float(rng.uniform(*threshold_db)), float(rng.uniform(*ratio))
    reduction = -np.maximum(env_db - thr, 0.0) * (1.0 - 1.0 / r)       # <= 0 dB
    att, rel = float(rng.uniform(*attack_ms)), float(rng.uniform(*release_ms))
    gain_db = _smooth(reduction, _coef(att, frame_ms), _coef(rel, frame_ms))
    out = _apply_frame_gain(audio, gain_db, n)
    in_rms, out_rms = np.sqrt(np.mean(audio ** 2)), np.sqrt(np.mean(out ** 2))
    if out_rms > EPS:
        out = (out * np.float32(in_rms / out_rms)).astype(np.float32)
    return out, {"stage": "compressor", "threshold_db": round(thr, 1), "ratio": round(r, 1),
                 "attack_ms": round(att, 1), "release_ms": round(rel)}


def limit(audio, sr, rng, ceiling_db=(-3.0, -0.1), lookahead_ms=5.0, release_ms=(30.0, 100.0)):
    """Look-ahead peak limiter: the output never exceeds the ceiling."""
    ceiling = float(10 ** (float(rng.uniform(*ceiling_db)) / 20))
    look = max(1, int(sr * lookahead_ms / 1000))
    rel_ms = float(rng.uniform(*release_ms))
    rel = max(1, int(sr * rel_ms / 1000))
    env = maximum_filter1d(np.abs(audio), size=2 * look + 1)
    need = np.minimum(1.0, ceiling / np.maximum(env, EPS))
    gain = uniform_filter1d(minimum_filter1d(need, size=2 * look + rel), size=look)
    out = np.clip(audio * gain, -ceiling, ceiling).astype(np.float32)   # residual overshoot only
    return out, {"stage": "limiter", "ceiling_db": round(20 * np.log10(ceiling), 2), "release_ms": round(rel_ms)}


def hard_clip(audio, level=1.0):
    return np.clip(audio, -level, level).astype(np.float32), {"overshoot": "clip"}


# --------------------------------------------------------------------------- #
# augmenter
# --------------------------------------------------------------------------- #
@dataclass
class LevelAugConfig:
    p_apply: float = 0.8
    gain_mode: str = "absolute"         # absolute: random target level | relative: random gain
    target_db: Tuple[float, float] = (-38.0, -12.0)
    gain_db: Tuple[float, float] = (-10.0, 10.0)
    p_agc: float = 0.30
    p_comp: float = 0.15
    p_limit: float = 0.15
    clip_p: float = 0.3                 # share of overshooting clips hard-clipped (else limited)
    seed: Optional[int] = None
    sr: int = 16000
    verbose: bool = True


class LevelAugmenter:
    """``audio = augmenter(audio, sr)``: float32 mono in, float32 mono of the same length out."""

    def __init__(self, cfg=None):
        self.cfg = cfg or LevelAugConfig()
        total = self.cfg.p_agc + self.cfg.p_comp + self.cfg.p_limit
        if total > 1.0 + 1e-9:
            raise ValueError(f"level_agc_p + level_comp_p + level_limit_p = {total} > 1")
        if self.cfg.gain_mode not in ("absolute", "relative"):
            raise ValueError(f"gain_mode must be absolute or relative, not {self.cfg.gain_mode}")
        self.stage_counts: Dict[str, int] = {}
        self.rng = None
        self.rng_pid = None
        if self.cfg.verbose:
            print(f"[level_augment] {self.describe()}")

    def describe(self):
        c = self.cfg
        p_none = 1.0 - c.p_agc - c.p_comp - c.p_limit
        level = (f"target={c.target_db[0]:g}..{c.target_db[1]:g} dBFS" if c.gain_mode == "absolute"
                 else f"gain={c.gain_db[0]:g}..{c.gain_db[1]:g} dB")
        return (f"level p_apply={c.p_apply} {level} | "
                f"agc={c.p_agc} comp={c.p_comp} limit={c.p_limit} none={p_none:.2f} | "
                f"overshoot clip_p={c.clip_p}")

    def _get_rng(self):
        # own stream: RTCAugmenter seeds make_rng(seed) with the same (seed, pid), so reusing it
        # would make both augmenters draw identical numbers
        if self.rng is None or self.rng_pid != os.getpid():
            base = random.randrange(2 ** 31) if self.cfg.seed is None else int(self.cfg.seed)
            self.rng = np.random.default_rng([base, os.getpid(), 0x1E7E1])
            self.rng_pid = os.getpid()
        return self.rng

    def _count(self, stage):
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

    def __call__(self, audio, sr=None):
        cfg = self.cfg
        sr = sr or cfg.sr
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        rng = self._get_rng()
        if len(audio) < 16 or rng.random() >= cfg.p_apply:
            return audio
        if cfg.gain_mode == "absolute":
            out, _ = set_level(audio, sr, rng, cfg.target_db)
        else:
            out, _ = apply_gain(audio, rng, cfg.gain_db)
        u = rng.random()
        if u < cfg.p_agc:
            out, _ = agc(out, sr, rng)
            self._count("agc")
        elif u < cfg.p_agc + cfg.p_comp:
            out, _ = compress(out, sr, rng)
            self._count("compressor")
        elif u < cfg.p_agc + cfg.p_comp + cfg.p_limit:
            out, _ = limit(out, sr, rng)
            self._count("limiter")
        else:
            self._count("gain_only")
        if np.abs(out).max() > 1.0:
            if rng.random() < cfg.clip_p:
                out, _ = hard_clip(out)
                self._count("overshoot:clip")
            else:
                out, _ = limit(out, sr, rng)
                self._count("overshoot:limit")
        return finalize(out, audio)


class ChainAugmenter:
    """Applies augmenters in order; describe() joins theirs (printed and saved by main_train)."""

    def __init__(self, augmenters):
        self.augmenters = [a for a in augmenters if a is not None]

    def __call__(self, audio, sr=None):
        for aug in self.augmenters:
            audio = aug(audio, sr)
        return audio

    def describe(self):
        return " -> ".join(a.describe() for a in self.augmenters)
