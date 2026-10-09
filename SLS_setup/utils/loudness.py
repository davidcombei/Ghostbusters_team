"""
Loudness normalisation: every clip to one integrated loudness (ITU-R BS.1770 LUFS, via pyloudnorm).

Deterministic and label-free, applied in SpoofAudioDataset._load right after the file is read and
before every augmentation (RTC, level, RawBoost) -- on train, dev and eval alike, so the model sees
one input level distribution at training and inference time. Complements the stochastic level
augmentation (utils/level_augment.py), which is training-only.

    --loudness_norm [--loudness_norm_lufs -23]      (main_train.py, main_eval.py, scripts/local_eval.py)

Clips shorter than one 400 ms gating block and clips the gate measures as silent (-inf LUFS) are
returned unchanged. The gain is computed here instead of through pyln.normalize.loudness (same
arithmetic) so the loader workers are not spammed by its "possible clipped samples" warning; the
peak is then handled by utils/rtc_augment.finalize like every other stage (scaled to 0.99 if > 1).
"""

import math

import numpy as np

from .rtc_augment import finalize


class LoudnessNormalizer:
    """``audio = norm(audio, sr)``: float32 mono in, float32 mono of the same length at `target_lufs` out."""

    def __init__(self, target_lufs=-23.0, sr=16000, verbose=True):
        import pyloudnorm as pyln                   # lazy: only a dependency when the flag is on
        self.target = float(target_lufs)
        self.sr = int(sr)
        self.meter = pyln.Meter(self.sr)            # K-weighting filters built once, reused per clip
        self.min_len = int(math.ceil(self.meter.block_size * self.sr))     # 6400 samples at 16 kHz
        self.skipped = 0
        if verbose:
            print(f"[loudness_norm] {self.describe()}")

    def describe(self):
        return f"loudness_norm target={self.target:g} LUFS"

    def __call__(self, audio, sr=None):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < self.min_len:               # shorter than one gating block: cannot be measured
            self.skipped += 1
            return audio
        lufs = self.meter.integrated_loudness(audio.astype(np.float64))
        if not np.isfinite(lufs):                   # silence (every block under the -70 LUFS gate)
            self.skipped += 1
            return audio
        out = audio * np.float32(10.0 ** ((self.target - lufs) / 20.0))
        return finalize(out, audio)
