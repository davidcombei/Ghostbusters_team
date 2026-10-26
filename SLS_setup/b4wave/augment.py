"""
RTCAugmenterB4: the training augmenter of the B4 cycle, a subclass of b3wave.augment.RTCAugmenterB3.

Chain (rdloop/B4/plan_agreed.md section 2.0), fixed order:

    1. the parent's acoustic `_stage` (parent rng)            -- unchanged B2f / B3 draws
    2. `_post_dsp`: wrtc_ns -> wrtc_dtx                        -- B4 levers, rng4 only
    3. the codec: original `codecs_augm` (or b3's random profiles), or a codec-slot B4 lever
       (opus_plc) that makes the parent's codec draws itself              -- parent rng + rng4

Reproduction guarantee
    * every B4 flag at its default -> `build_rtc_augmenter_b4(args)` returns
      `build_rtc_augmenter_b3(args)`: the B3 object itself, no B4 code on the data path;
    * a forced `RTCAugmenterB4` with no lever on draws exactly the parent's numbers (its
      `__call__` mirrors RTCAugmenter / RTCAugmenterB3 line for line and `_post_dsp` returns the
      audio untouched without drawing), so the outputs are bit-identical (`checks augmenter`);
    * B4 draws come from a separate stream `rng4 = default_rng([seed, pid, 0xB4])`, created lazily
      (only when a lever is on) and re-created on a pid change like the parent's `_get_rng`, so the
      parent `rng` keeps making exactly the base stage / codec draws.

Lever registry
    Each approved lever registers one `LeverSpec` in `LEVERS` (fixed chain order) and implements
    `_lever_<id>` on RTCAugmenterB4:
        slot "post_dsp":  `_lever_<id>(self, out, sr) -> out`              (draws from rng4 only)
        slot "codec":     `_lever_<id>(self, out, rng, sr) -> (out, app)`  (replaces `_codec`)
    A lever is on when its probability field is > 0; nothing of an off lever is imported, drawn,
    counted or described. Per-clip counters go through `self._count(key)` (key listed in
    `LeverSpec.meta_keys`, so the dataset's meta vector and the trainer telemetry carry them).
"""

import os
import random
from dataclasses import dataclass, fields
from typing import Callable, Optional, Tuple

import numpy as np

from utils import rtc_augment as _ra
from utils.rtc_augment import finalize

from b3wave import augment as _b3aug
from b3wave.augment import RTCAugConfigB3, RTCAugmenterB3, build_rtc_augmenter_b3
from b3wave.heldout_reserve import assert_training_ops

RNG4_TAG = 0xB4                     # third word of the rng4 seed sequence
SLOTS = ("post_dsp", "codec")


@dataclass(frozen=True)
class LeverSpec:
    """One approved B4 lever, as the augmenter / trainer / checks see it."""
    id: str
    slot: str                                   # "post_dsp" | "codec"
    prob_arg: str                               # argparse dest of the lever's probability (0 = off)
    prob_field: str                             # RTCAugConfigB4 field it maps to
    arg_fields: Tuple[Tuple[str, str], ...] = ()    # (argparse dest, config field), read only when on
    meta_keys: Tuple[str, ...] = ()             # telemetry counters (meta vector), appended when on
    modules: Tuple[str, ...] = ()               # python modules imported lazily, only when on
    filters: Tuple[str, ...] = ()               # op names for assert_training_ops(filters=...)
    codecs: Tuple[str, ...] = ()                # codec names for assert_training_ops(codecs=...)
    describe: Optional[Callable] = None         # cfg -> " | " suffix of describe() when on
    header: Optional[Callable] = None           # args -> the lever's ONE startup header line (when on)


LEVERS = {}                         # id -> LeverSpec, in chain order (wrtc_ns -> wrtc_dtx -> codec)


def register(spec):
    if spec.slot not in SLOTS:
        raise ValueError(f"{spec.id}: slot must be one of {SLOTS}")
    LEVERS[spec.id] = spec
    return spec


# ---------------------------------------------------------------------------------------------
# lever registrations (one block per lever part, in chain order)
# ---------------------------------------------------------------------------------------------
APM_NS_LEVELS_DB = (6, 12, 18, 21)          # = b4wave.dsp_apm.LEVEL_DB keys (kept here: no lazy import needed)


def _apm_levels(levels):
    return [int(v) for v in levels]


register(LeverSpec(
    id="apm_ns", slot="post_dsp", prob_arg="aug_apm_p", prob_field="apm_p",
    arg_fields=(("aug_apm_ns_levels", "apm_ns_levels"),),
    meta_keys=tuple(f"wrtc_ns{db}" for db in APM_NS_LEVELS_DB) + ("fail_wrtc_ns",),
    modules=("pywebrtc_audio", "b4wave.dsp_apm"),
    filters=("wrtc_ns",),
    describe=lambda cfg: f"wrtc_ns p={cfg.apm_p:g} levels_db={_apm_levels(cfg.apm_ns_levels)}",
    header=lambda a: (f"B4 APM NS: on p={float(a.aug_apm_p):g} levels_db={_apm_levels(a.aug_apm_ns_levels)} "
                      f"backend=pywebrtc_audio.NoiseSuppressor hpf=off agc=off"),
))


DTX_FILLS = ("mix", "zero", "cng")          # = b4wave.dsp_dtx.FILLS (kept here: no lazy import needed)
DTX_BINS = ("dtx_g0", "dtx_g1_20", "dtx_g20_40", "dtx_g40p")     # = b4wave.dsp_dtx.BINS (gated-fraction bins)

register(LeverSpec(
    id="dtx_gate", slot="post_dsp", prob_arg="aug_dtx_p", prob_field="dtx_p",
    arg_fields=(("aug_dtx_fill", "dtx_fill"),),
    meta_keys=("wrtc_dtx_zero", "wrtc_dtx_cng") + DTX_BINS + ("fail_wrtc_dtx",),
    modules=("webrtcvad", "b4wave.dsp_dtx"),
    filters=("wrtc_dtx",),
    describe=lambda cfg: f"wrtc_dtx p={cfg.dtx_p:g} fill={cfg.dtx_fill}",
    header=lambda a: (f"B4 DTX gate: on p={float(a.aug_dtx_p):g} fill={a.aug_dtx_fill} modes=[1, 2, 3] "
                      f"frames_ms=[10, 20, 30] hangover_ms=40..200"),
))


OPUS_PLC_KEYS = ("opus_plc", "plc_l0_5", "plc_l5_10", "plc_l10p", "fail_opus_plc")   # bins = b4wave.opus_plc.BINS


def _opus_header(a):
    from b4wave import opus_plc                 # lazy: the header line is built only when the lever is on
    lo, hi = (float(v) for v in a.aug_opus_loss)
    blo, bhi = (float(v) for v in a.aug_opus_burst)
    return (f"B4 Opus PLC: on p={float(a.aug_opus_plc_p):g} loss={lo:.2f}..{hi:.2f} burst={blo:g}..{bhi:g} "
            f"decode=48k+resample_poly libopus={opus_plc.version()}")


register(LeverSpec(
    id="opus_plc", slot="codec", prob_arg="aug_opus_plc_p", prob_field="opus_plc_p",
    arg_fields=(("aug_opus_loss", "opus_loss"), ("aug_opus_burst", "opus_burst")),
    meta_keys=OPUS_PLC_KEYS,
    modules=("b4wave.opus_plc",),
    filters=("opus_plc",),
    codecs=("libopus",),
    describe=lambda cfg: (f"opus_plc p={cfg.opus_plc_p:g} loss={float(cfg.opus_loss[0]):.2f}..{float(cfg.opus_loss[1]):.2f} "
                          f"burst={float(cfg.opus_burst[0]):g}..{float(cfg.opus_burst[1]):g} decode=48k"),
    header=_opus_header,
))


def _prob(obj, name):
    value = getattr(obj, name, 0.0)
    return float(value) if value is not None else 0.0


def active_levers(args):
    """Ids of the levers switched on by argparse-style args (probability > 0), in chain order."""
    return [lid for lid, s in LEVERS.items() if _prob(args, s.prob_arg) > 0.0]


def active_levers_cfg(cfg):
    """Ids of the levers switched on in an RTCAugConfigB4, in chain order."""
    return [lid for lid, s in LEVERS.items() if _prob(cfg, s.prob_field) > 0.0]


def b4_header_lines(args):
    """The startup header line of every lever that is on (none when all are off)."""
    return [LEVERS[lid].header(args) if LEVERS[lid].header else f"B4 {lid}: on"
            for lid in active_levers(args)]


@dataclass
class RTCAugConfigB4(RTCAugConfigB3):
    """RTCAugConfigB3 + the B4 lever fields (each lever part adds its own; default = off)."""
    # apm_ns (plan 2.1): per-clip probability of one WebRTC NS pass, NS targets in dB
    apm_p: float = 0.0
    apm_ns_levels: Tuple[int, ...] = APM_NS_LEVELS_DB
    # dtx_gate (plan 2.2): per-clip probability of webrtcvad DTX gating, fill of the gated samples
    dtx_p: float = 0.0
    dtx_fill: str = "mix"
    # opus_plc (plan 2.3): per-codec-call probability of the in-process libopus path with Gilbert-Elliott loss + PLC,
    # per-clip target loss range, mean burst length range (packets)
    opus_plc_p: float = 0.0
    opus_loss: Tuple[float, float] = (0.01, 0.20)
    opus_burst: Tuple[float, float] = (1.0, 4.0)


def b4_config_from_args(args):
    """
    RTCAugConfigB4 from argparse-style args: the b3wave.augment.build_rtc_augmenter_b3 config,
    field for field, plus the fields of the levers that are on. None when neither --use_rtc_aug
    nor --musan (as b3wave).
    """
    use_rtc = getattr(args, "use_rtc_aug", False)
    use_musan = getattr(args, "musan", False)
    if not (use_rtc or use_musan):
        return None
    cfg = RTCAugConfigB4(
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
    for lid in active_levers(args):
        spec = LEVERS[lid]
        setattr(cfg, spec.prob_field, float(getattr(args, spec.prob_arg)))
        for arg, fld in spec.arg_fields:
            value = getattr(args, arg)
            setattr(cfg, fld, tuple(value) if isinstance(value, list) else value)
    return cfg


class RTCAugmenterB4(RTCAugmenterB3):

    def __init__(self, cfg=None):
        cfg = cfg or RTCAugConfigB4()
        if not isinstance(cfg, RTCAugConfigB4):
            raise TypeError("RTCAugmenterB4 needs an RTCAugConfigB4")
        # set before the parent init: RTCAugmenterB3 prints describe() at its end when verbose
        self.levers_on = active_levers_cfg(cfg)
        self.post_dsp_levers = [lid for lid in self.levers_on if LEVERS[lid].slot == "post_dsp"]
        codec_levers = [lid for lid in self.levers_on if LEVERS[lid].slot == "codec"]
        if len(codec_levers) > 1:
            raise ValueError(f"at most one codec-slot lever, got {codec_levers}")
        self.codec_lever = codec_levers[0] if codec_levers else None
        self.rng4 = None
        self.rng4_pid = None
        self.fail_printed = {}                      # op -> pid of the last process that printed its failure
        super().__init__(cfg)
        assert_training_ops(
            codecs=[c for lid in self.levers_on for c in LEVERS[lid].codecs],
            filters=[f for lid in self.levers_on for f in LEVERS[lid].filters],
        )
        self.meta_keys = list(self.meta_keys) + [k for lid in self.levers_on for k in LEVERS[lid].meta_keys]
        for lid in self.levers_on:                  # lever start-up (lazy imports, selftests)
            init = getattr(self, f"_init_{lid}", None)
            if init is not None:
                init()

    # ------------------------------------------------------------------ #
    def describe(self):
        text = super().describe()
        for lid in getattr(self, "levers_on", []):
            spec = LEVERS[lid]
            text += f" | {spec.describe(self.cfg) if spec.describe else lid}"
        return text

    def _get_rng4(self):
        """The B4 stream: default_rng([seed, pid, 0xB4]); seed None -> a random 31-bit base, as make_rng."""
        if self.rng4 is None or self.rng4_pid != os.getpid():
            base = random.randrange(2 ** 31) if self.cfg.seed is None else int(self.cfg.seed) % (2 ** 63)
            self.rng4 = np.random.default_rng([base, os.getpid(), RNG4_TAG])
            self.rng4_pid = os.getpid()
        return self.rng4

    def _lever_failed(self, key, op, exc, action="passing through"):
        """Count a worker-side lever failure (`key`) and print the first one per process and op:
        `[b4wave] <op> failed (<exc>); <action>` (review F1.3). Same pattern as the parent's
        `codec ... failed (...)` line, so the wave's early check (`failed \\(`) sees lever failures too."""
        self._count(key)
        pid = os.getpid()
        if self.fail_printed.get(op) != pid:
            self.fail_printed[op] = pid
            print(f"[b4wave] {op} failed ({type(exc).__name__}: {exc}); {action}", flush=True)

    def _post_dsp(self, out, sr):
        """Slot between the acoustic stage and the codec. No lever on: `out` itself, no draw."""
        for lid in self.post_dsp_levers:
            out = getattr(self, f"_lever_{lid}")(out, sr)
        return out

    # ------------------------------------------------------------------ #
    # lever apm_ns (plan 2.1): WebRTC NS (standalone NoiseSuppressor, no HPF / AGC / AEC)
    def _init_apm_ns(self):
        """Level check, lazy import of dsp_apm (+ pywebrtc_audio) and the start-up selftest (named asserts)."""
        levels = [int(v) for v in self.cfg.apm_ns_levels]
        bad = [v for v in levels if v not in APM_NS_LEVELS_DB]
        if not levels or bad:
            raise ValueError(f"apm_ns_levels must be a non-empty subset of {list(APM_NS_LEVELS_DB)}, got {levels}")
        self.cfg.apm_ns_levels = tuple(levels)
        from b4wave import dsp_apm
        self._apm = dsp_apm
        vals = dsp_apm.selftest()               # AssertionError("apm selftest [<name>] ...") stops the run
        print(f"B4 APM NS selftest: PASS {dsp_apm.summary(vals)}", flush=True)

    def _lever_apm_ns(self, out, sr):
        cfg = self.cfg
        rng4 = self._get_rng4()
        if not rng4.random() < cfg.apm_p:
            return out
        lvl = int(cfg.apm_ns_levels[int(rng4.integers(len(cfg.apm_ns_levels)))])
        try:
            processed = self._apm.wrtc_ns(out, lvl, sr)
        except Exception as exc:                # the clip passes through, counted; first one printed
            self._lever_failed("fail_wrtc_ns", "wrtc_ns", exc)
            return out
        self._count(f"wrtc_ns{lvl}")
        return finalize(processed, out)

    # ------------------------------------------------------------------ #
    # lever dtx_gate (plan 2.2): webrtcvad DTX gating, zero or order-10 LPC comfort-noise fill
    def _init_dtx_gate(self):
        """Fill check, lazy import of dsp_dtx (+ webrtcvad, scipy) and the start-up selftest (named asserts)."""
        if self.cfg.dtx_fill not in DTX_FILLS:
            raise ValueError(f"dtx_fill must be one of {list(DTX_FILLS)}, got {self.cfg.dtx_fill!r}")
        from b4wave import dsp_dtx
        self._dtx = dsp_dtx
        vals = dsp_dtx.selftest()               # AssertionError("dtx selftest [<name>] ...") stops the run
        print(f"B4 DTX gate selftest: PASS {dsp_dtx.summary(vals)}", flush=True)

    def _lever_dtx_gate(self, out, sr):
        cfg = self.cfg
        rng4 = self._get_rng4()
        if not rng4.random() < cfg.dtx_p:
            return out
        try:
            processed, frac, fill_used = self._dtx.wrtc_dtx(out, rng4, sr, cfg.dtx_fill)
        except Exception as exc:                # the clip passes through, counted; first one printed
            self._lever_failed("fail_wrtc_dtx", "wrtc_dtx", exc)
            return out
        self._count(f"wrtc_dtx_{fill_used}")
        self._count(self._dtx.gated_bin(frac))
        return finalize(processed, out)

    # ------------------------------------------------------------------ #
    # lever opus_plc (plan 2.3): in-process libopus round trip, Gilbert-Elliott packet loss, decoder PLC at 48 kHz
    def _init_opus_plc(self):
        """Profile / range checks, lazy import of opus_plc (loads libopus through ctypes) and the start-up selftest."""
        cfg = self.cfg
        if not cfg.with_codecs:
            raise ValueError("opus_plc needs the codec stage (--use_rtc_aug without --no_codec_aug)")
        if cfg.codec_profiles != "fixed":
            raise ValueError(f"opus_plc needs --aug_codec_profiles fixed (the parent's codecs_augm draws), "
                             f"got {cfg.codec_profiles!r}")
        from b4wave import opus_plc
        cfg.opus_loss, cfg.opus_burst = opus_plc.check_ranges(cfg.opus_loss, cfg.opus_burst)
        self._opus = opus_plc
        vals = opus_plc.selftest()              # AssertionError("opus selftest [<name>] ...") / OSError stops the run
        print(f"B4 Opus PLC selftest: PASS {opus_plc.summary(vals)}", flush=True)

    def _lever_opus_plc(self, out, rng, sr):
        """Codec slot. Picked (rng4.random() < p): the parent's codec draws (draw_codec_params(rng), codecs_augm's
        order), then the GE loss draws (rng4) and the in-process round trip; on an exception the clip goes through
        ffmpeg with the same draws (fail_opus_plc). Not picked: the original codecs_augm."""
        cfg = self.cfg
        rng4 = self._get_rng4()
        if not rng4.random() < cfg.opus_plc_p:
            return _ra.codecs_augm(out, rng, sr, cfg.codec_apps)
        params = self._opus.draw_codec_params(rng, cfg.codec_apps)
        try:
            processed, loss = self._opus.opus_loss_roundtrip(out, rng4, params["kbps"], params["cutoff"],
                                                             params["frame_ms"], params["application"],
                                                             cfg.opus_loss, cfg.opus_burst, sr)
        except Exception as exc:                # ffmpeg with the same draws, counted; first one printed
            self._lever_failed("fail_opus_plc", "opus_plc", exc, "ffmpeg fallback with the same profile")
            return self._opus.ffmpeg_opus(out, params, sr)
        self._count("opus_plc")
        self._count(self._opus.loss_bin(loss))
        return processed, params["app"]

    def _codec(self, out, rng, sr):
        """The codec call of the parent's __call__ (same draws), or the codec-slot lever's."""
        cfg = self.cfg
        if self.codec_lever is not None:
            return getattr(self, f"_lever_{self.codec_lever}")(out, rng, sr)
        if cfg.codec_profiles == "random":
            return _b3aug.codecs_augm_random(out, rng, sr, cfg.codec_apps,
                                             cfg.codec_bitrates_kbps, cfg.codec_frames_ms)
        return _ra.codecs_augm(out, rng, sr, cfg.codec_apps)

    def __call__(self, audio, sr=None):
        # mirrors RTCAugmenter.__call__ / RTCAugmenterB3.__call__ with the B4 slot inserted
        cfg = self.cfg
        sr = sr or cfg.sr
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < 16:
            return audio
        rng = self._get_rng()
        out = self._stage(audio, rng, sr)
        out = self._post_dsp(out, sr)
        if cfg.with_codecs and rng.random() < cfg.codec_p_apply:
            out, app = self._codec(out, rng, sr)
            self._count(f"codec:{app}")
            out = finalize(out, audio)
        return out


def build_rtc_augmenter_b4(args):
    """
    Every B4 flag at its default: `build_rtc_augmenter_b3(args)`, unchanged (the base object).
    Otherwise an RTCAugmenterB4 over the same B3 config plus the active levers' fields.
    """
    if not active_levers(args):
        return build_rtc_augmenter_b3(args)
    cfg = b4_config_from_args(args)
    if cfg is None:
        raise ValueError(f"B4 levers {active_levers(args)} need the RTC augmenter (--use_rtc_aug or --musan)")
    return RTCAugmenterB4(cfg)


def b3_field_names():
    """Names of the RTCAugConfigB3 fields (for the config-equivalence check)."""
    return [f.name for f in fields(RTCAugConfigB3)]
