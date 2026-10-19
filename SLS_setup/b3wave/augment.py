"""
RTCAugmenterB3: the training augmenter of the B3 wave, a subclass of utils.rtc_augment.RTCAugmenter.

New, each behind a config field that defaults to the original behaviour:

    snr_dist        "uniform" (original) | "beta": SNR drawn from a Beta on [lo, hi] with the mode
                    at `snr_beta_mode` dB (concentration 4), for every noise stage incl. MUSAN
    max_stages      1 (original) | k: U{1..k} *distinct* acoustic stages chained per clip, in the
                    order noise -> echo -> reverb -> suppress (the original ignored values > 1)
    codec_profiles  "fixed" (original CODEC_PROFILES draw) | "random": the app's cutoff and
                    application are kept but bitrate, frame size, VBR mode and the expected
                    packet-loss hint are randomised (this ffmpeg 4.2 build has no -fec / -dtx)

Reproduction guarantee: with the defaults the stage draw delegates to the parent's `_stage`, so
the numpy RNG is consumed in exactly the same order and the outputs are bit-identical to the
original augmenter (b3wave/checks.py::check_augmenter_repro). The reserved-ops assert
(b3wave/heldout_reserve.py) runs at start-up whatever the settings.

Per-clip telemetry: `meta_keys` names a fixed-size count vector (stages, codec apps, SNR bins)
that the dataset derives from the `stage_counts` delta around one call, so the trainer can log
realised stage / SNR histograms per epoch without any worker IPC.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.rtc_augment import (
    CODEC_PROFILES,
    RTCAugConfig,
    RTCAugmenter,
    _ffmpeg,
    add_echo,
    add_noise,
    add_reverb,
    finalize,
    suppress,
)

from .heldout_reserve import assert_training_ops

# SNR histogram bins (dB), upper edges; the last bin is open
SNR_BIN_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
SNR_BIN_NAMES = ("lt0", "0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "ge30")
NON_NOISE_STAGES = ("echo", "reverb", "suppress")      # canonical chain order after the noise stages
BETA_CONCENTRATION = 4.0


def snr_bin(snr_db):
    for name, edge in zip(SNR_BIN_NAMES, SNR_BIN_EDGES):
        if snr_db < edge:
            return name
    return SNR_BIN_NAMES[-1]


def beta_params(mode, lo, hi, concentration=BETA_CONCENTRATION):
    """(a, b) of a Beta on [lo, hi] whose mode sits at `mode` (clipped into the range)."""
    if hi <= lo:
        raise ValueError(f"empty SNR range {lo}..{hi}")
    m = min(1.0, max(0.0, (float(mode) - lo) / (hi - lo)))
    return 1.0 + m * concentration, 1.0 + (1.0 - m) * concentration


@dataclass
class RTCAugConfigB3(RTCAugConfig):
    snr_dist: str = "uniform"
    snr_beta_mode: float = 8.0
    codec_profiles: str = "fixed"
    codec_bitrates_kbps: tuple = (6, 8, 12, 16, 24)
    codec_frames_ms: tuple = (20, 40, 60)


def codecs_augm_random(audio, rng, sr=16000, apps=None, bitrates=(6, 8, 12, 16, 24), frames=(20, 40, 60)):
    """
    Like utils.rtc_augment.codecs_augm (same ffmpeg pipeline, same app list and cutoffs), but the
    bitrate, frame size, VBR mode (off / on / constrained) and the expected packet-loss hint
    (0 or 5-20 %, which makes libopus trade quality for robustness) are randomised instead of
    taken from the app profile.
    """
    apps = list(apps or CODEC_PROFILES)
    app = apps[int(rng.integers(len(apps)))]
    _, cutoffs, _, application = CODEC_PROFILES[app]
    kbps = int(bitrates[int(rng.integers(len(bitrates)))])
    frame = int(frames[int(rng.integers(len(frames)))])
    vbr = ("off", "on", "constrained")[int(rng.integers(3))]
    loss = int(rng.integers(5, 21)) if rng.random() < 0.5 else 0
    raw = ["-f", "f32le", "-ac", "1", "-ar", str(sr)]
    opts = ["-c:a", "libopus", "-b:a", f"{kbps}k",
            "-cutoff", str(cutoffs[int(rng.integers(len(cutoffs)))]),
            "-frame_duration", str(frame), "-application", application,
            "-vbr", vbr, "-packet_loss", str(loss)]
    try:
        encoded = _ffmpeg([*raw, "-i", "pipe:0", *opts, "-f", "ogg", "pipe:1"],
                          np.asarray(audio, dtype=np.float32).tobytes())
        decoded = _ffmpeg(["-i", "pipe:0", *raw, "pipe:1"], encoded)
    except (OSError, subprocess.CalledProcessError) as exc:            # never kill a training step
        print(f"[b3wave.augment] codec {app} (random profile) failed ({exc}); passing through")
        return audio, app
    return np.frombuffer(decoded, dtype=np.float32), app


class RTCAugmenterB3(RTCAugmenter):

    def __init__(self, cfg=None):
        cfg = cfg or RTCAugConfigB3()
        if not isinstance(cfg, RTCAugConfigB3):
            raise TypeError("RTCAugmenterB3 needs an RTCAugConfigB3")
        if cfg.max_stages < 1:
            raise ValueError("max_stages must be >= 1")
        if cfg.snr_dist not in ("uniform", "beta"):
            raise ValueError(f"snr_dist must be uniform or beta, got {cfg.snr_dist}")
        if cfg.codec_profiles not in ("fixed", "random"):
            raise ValueError(f"codec_profiles must be fixed or random, got {cfg.codec_profiles}")
        lo, hi = cfg.snr_db
        if hi <= lo:
            raise ValueError(f"empty SNR range {lo}..{hi}")

        verbose = cfg.verbose
        cfg.verbose = False                 # the parent would warn about max_stages > 1; we print below
        try:
            super().__init__(cfg)
        finally:
            cfg.verbose = verbose
        for d in list(cfg.noise_dirs) + list(cfg.music_dirs) + list(cfg.musan_dirs) + list(cfg.rir_dirs):
            if verbose and not Path(d).is_dir():
                print(f"[rtc_augment] missing pool dir: {Path(d)}")

        assert_training_ops(
            codecs=["libopus"] if cfg.with_codecs else [],
            filters=[],
            dirs=list(cfg.noise_dirs) + list(cfg.music_dirs) + list(cfg.musan_dirs) + list(cfg.rir_dirs),
        )
        # the parent path is only exact when nothing new is switched on
        self.delegate_stage = cfg.max_stages == 1 and cfg.snr_dist == "uniform"
        self.meta_keys = ([f"stage:{s}" for s in self.stages]
                          + [f"codec:{a}" for a in cfg.codec_apps]
                          + [f"snr:{b}" for b in SNR_BIN_NAMES])
        if verbose:
            print(f"[rtc_augment] {self.describe()}")

    # ------------------------------------------------------------------ #
    def describe(self):
        text = super().describe()
        cfg = self.cfg
        extra = []
        if cfg.snr_dist != "uniform":
            extra.append(f"snr_dist={cfg.snr_dist}(mode {cfg.snr_beta_mode:g} dB)")
        if cfg.max_stages > 1:
            extra.append(f"max_stages={cfg.max_stages} (chained, distinct)")
        if cfg.codec_profiles != "fixed":
            extra.append(f"codec_profiles={cfg.codec_profiles} "
                         f"kbps={list(cfg.codec_bitrates_kbps)} frames={list(cfg.codec_frames_ms)} vbr=rand loss=0|5-20")
        return text + ("".join(f" | {e}" for e in extra) if extra else "")

    def _draw_snr(self, rng):
        lo, hi = self.cfg.snr_db
        if self.cfg.snr_dist == "beta":
            a, b = beta_params(self.cfg.snr_beta_mode, lo, hi)
            return lo + (hi - lo) * float(rng.beta(a, b))
        return float(rng.uniform(lo, hi))

    def _apply_stage(self, audio, stage, rng, sr):
        cfg = self.cfg
        self._count(stage)
        if stage == "echo":
            return add_echo(audio, rng, cfg.echo_delay_ms, cfg.echo_strength, sr)
        if stage == "reverb":
            return add_reverb(audio, self.rir_files, rng, sr, cfg.rir_max_seconds)
        if stage == "suppress":
            return suppress(audio, cfg.suppress_fn, sr)
        snr = self._draw_snr(rng)
        self._count(f"snr:{snr_bin(snr)}")
        # (snr, snr): the parent's uniform draw inside add_noise collapses to this value
        return add_noise(audio, self.noise_files[stage], rng, (snr, snr), sr)

    def _stage(self, audio, rng, sr):
        if self.delegate_stage:
            return super()._stage(audio, rng, sr)
        cfg = self.cfg
        if not self.stages or rng.random() >= cfg.p_apply:
            return audio
        k = 1 if cfg.max_stages == 1 else 1 + int(rng.integers(cfg.max_stages))
        k = min(k, len(self.stages))
        chosen = [self.stages[int(i)] for i in rng.choice(len(self.stages), size=k, replace=False)]
        # canonical order: noise stages (as listed) -> echo -> reverb -> suppress
        noise_first = [s for s in self.stages if s in chosen and s not in NON_NOISE_STAGES]
        ordered = noise_first + [s for s in NON_NOISE_STAGES if s in chosen]
        out = audio
        for stage in ordered:
            out = finalize(self._apply_stage(out, stage, rng, sr), out)
        return out

    def __call__(self, audio, sr=None):
        if self.cfg.codec_profiles != "random":
            return super().__call__(audio, sr)
        cfg = self.cfg
        sr = sr or cfg.sr
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < 16:
            return audio
        rng = self._get_rng()
        out = self._stage(audio, rng, sr)
        if cfg.with_codecs and rng.random() < cfg.codec_p_apply:
            out, app = codecs_augm_random(out, rng, sr, cfg.codec_apps,
                                          cfg.codec_bitrates_kbps, cfg.codec_frames_ms)
            self._count(f"codec:{app}")
            out = finalize(out, audio)
        return out

    # ------------------------------------------------------------------ #
    def counts_vector(self, before, after=None):
        """Fixed-size int vector of the stage_counts delta between two snapshots (telemetry)."""
        after = self.stage_counts if after is None else after
        vec = np.zeros(len(self.meta_keys), dtype=np.int32)
        for i, key in enumerate(self.meta_keys):
            raw = key.split(":", 1)[1] if key.startswith("stage:") else key
            vec[i] = after.get(raw, 0) - before.get(raw, 0)
        return vec


def build_rtc_augmenter_b3(args):
    """
    RTCAugmenterB3 from argparse-style args, mirroring utils.data_utils._build_rtc_augmenter
    (same flags) plus the B3 fields: --aug_snr_db, --aug_snr_dist, --aug_snr_beta_mode,
    --aug_max_stages (functional), --aug_codec_profiles. None when neither --use_rtc_aug nor --musan.
    """
    use_rtc = getattr(args, "use_rtc_aug", False)
    use_musan = getattr(args, "musan", False)
    if not (use_rtc or use_musan):
        return None
    cfg = RTCAugConfigB3(
        noise_dirs=(getattr(args, "aug_noise_dirs", []) or []) if use_rtc else [],
        music_dirs=(getattr(args, "aug_music_dirs", []) or []) if use_rtc else [],
        rir_dirs=(getattr(args, "aug_rir_dirs", []) or []) if use_rtc else [],
        musan_dirs=[getattr(args, "musan_dir", "data/augm/musan")] if use_musan else [],
        with_echo=use_rtc,
        p_apply=getattr(args, "aug_p_apply", 0.8),
        max_stages=int(getattr(args, "aug_max_stages", 1)),
        seed=getattr(args, "seed", None),
        with_codecs=use_rtc and not getattr(args, "no_codec_aug", False),
        codec_p_apply=getattr(args, "aug_codec_p", 1.0),
        snr_db=tuple(float(v) for v in getattr(args, "aug_snr_db", (5.0, 20.0))),
        snr_dist=getattr(args, "aug_snr_dist", "uniform"),
        snr_beta_mode=float(getattr(args, "aug_snr_beta_mode", 8.0)),
        codec_profiles=getattr(args, "aug_codec_profiles", "fixed"),
    )
    if getattr(args, "aug_use_deepfilternet", False):
        from utils.rtc_augment import make_deepfilternet_suppressor
        cfg.suppress_fn = make_deepfilternet_suppressor()
        if cfg.suppress_fn is None:
            print("[data_utils] DeepFilterNet not available; suppression stage disabled")
    return RTCAugmenterB3(cfg)
