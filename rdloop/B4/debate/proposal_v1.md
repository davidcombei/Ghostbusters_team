# rdloop B4 - proposal v1

> research-architect, PROPOSE v1 (attempt a0 after the owner unblock of 12:17), 2026-10-17. Base **B2f**, entry
> module `b3wave`, cycle focus **augmentation**. Owner notes: none (`rdloop/B4/user_notes.md` holds only the header).
>
> RUN = `SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623`. A pair written
> "merge / final" is always `local_eval/merge_top5` first, then `local_eval/epoch_8`.
>
> Provenance of the numbers:
> - copied from a named file: **AUD** = `rdloop/B4/01_audit.md`, **SOTA** = `rdloop/B4/02_sota.md`, RUN
>   `logs/train.log`, `data/dev_noisy_sim/*/meta.csv`;
> - **(computed)**: read-only `python3 -I` joins of `RUN/local_eval/<readout>/scores/sim_heldout_v1.txt` with
>   `data/dev_noisy_sim/heldout_v1/meta.csv`, half B only, or simple arithmetic on copied numbers. The score is
>   P(spoof); a bona clip is a false alarm (FA) at score > 0.5. This session's join reproduces AUD §3.4 exactly (all
>   bona 69.3 / 69.8 = AUD 69.31 / 69.82);
> - **(measured)**: in-memory `python -I` runs in this session in the `rtc-sdd` env. They wrote no files.
>
> No progress-split file was read.

## 0. Plan in one paragraph

B2f trains on one acoustic stage, then Opus, and nothing else (AUD §1, RUN `config.yaml` stage list). "No DSP
stage of any kind has ever been in training" (AUD G1). Every noisy RTC clip went through a platform's NS / VAD-DTX /
codec / packet-loss chain (RTCFake: "key distortion sources, including noise suppression, echo cancellation, codec
encoding and decoding, and packet loss"). About three quarters of the noisy bona FA are clips the model gets right
when clean. AUD §3.3 counts the noise-induced ones among the online-source bona FA, merge / final:
- matched: 35 / 37 of 50 / 50;
- heldout: 43 / 43 of 58 / 56.

The worst heldout slices are DSP slices (§1).

I propose **three single-component levers**. Each is a CPU-side stage at the point where a platform puts it: after
the acoustic stage, before or inside the codec. Each one:
- draws its randomness from a **separate RNG stream**, so the B2f stage and codec draws stay exactly as they are;
- runs at p = 0.3 per clip, independent of the class;
- uses an implementation that differs from the reserved `sim_ops` one. `docs/next_steps_plan.md` §2.4 lists
  exactly these as the allowed training implementations: "WebRTC APM NS/AGC2, `webrtcvad`, libopus in-band PLC, …".

The levers:
1. `apm_ns`: WebRTC noise suppression (`webrtc_noise_gain`) after the noise stage. Gap G1.
2. `dtx_gate`: WebRTC-VAD DTX gating (`webrtcvad`). The gated frames are filled with digital zeros or LPC comfort
   noise. Gaps G3 and G1.
3. `opus_plc`: an in-process libopus round trip with bursty packet loss and decoder PLC. It binds the system
   libopus 1.3.1 through `ctypes`, so it needs **no new package**. Gaps G1 and G2.

No lever changes the model, the forward pass or the state_dict. **Two levers need an owner install** (§7):
`pip install webrtc-noise-gain webrtcvad-wheels`. Without it, only `opus_plc` can be approved.

## 1. Summary

| lever | flags (default = B2f) | wave arm | gap (AUD) | targeted heldout half-B bona slice: n, B2f recall merge / final (computed) | est LOC | VRAM | epoch × | smoke min | new deps (import name) |
|---|---|---|---|---|---|---|---|---|---|
| `apm_ns` | `--aug_apm_p 0.0`, `--aug_apm_ns_levels 1 2 3 4` | `--aug_apm_p 0.3` | G1 | `ns:afftdn`: n = 74, 55.4 / 58.1 (no-DSP slice: n = 128, 72.7 / 72.7) | 160 | 0 | 1.05 | 4.5 | `webrtc_noise_gain` |
| `dtx_gate` | `--aug_dtx_p 0.0`, `--aug_dtx_fill mix` | `--aug_dtx_p 0.3` | G3, G1 | `vad:cng`: n = 34, 55.9 / 61.8; `vad:zero`: n = 39, 69.2 / 69.2. Also the 136 / 133 noisy bona FA at p ≥ 0.99 (AUD §3.2) | 200 | 0 | 1.03 | 4.5 | `webrtcvad` |
| `opus_plc` | `--aug_opus_plc_p 0.0`, `--aug_opus_loss 0.02 0.15`, `--aug_opus_burst 1.0 4.0` | `--aug_opus_plc_p 0.3` | G1, G2 | `plc`: n = 130, 66.2 / 66.9 (the largest DSP slice) | 260 | 0 | 1.02 | 4.5 | none (system libopus 1.3.1) |

**Smoke budget (LINT rule).** S0 4.5 + eval path 2.0 + 3 × 4.5 = **20.0 min ≤ 25** (computed). No lever adds
parameters, so there is no overfit job.

**Mechanism readout.** Each lever's slice signature is heldout half-B bona recall per `dsp_chain` stage, and
`metrics.json` has no such breakdown: its `breakdowns` hold family / snr_db / codec / src / lang (AUD §3.4). So I add
the read-only script `scripts/b4_dsp_slices.py` (§3). It must reproduce this B2f table (computed in this session):

| heldout half-B bona slice (`dsp_chain` stage) | n | B2f merge | B2f final |
|---|---|---|---|
| no DSP (`[]`) | 128 | 72.7 | 72.7 |
| `ns:afftdn` | 74 | 55.4 | 58.1 |
| `ns:anlmdn` | 72 | 77.8 | 76.4 |
| `agc:dynaudnorm` | 53 | 64.2 | 62.3 |
| `agc:acompressor` | 56 | 69.6 | 71.4 |
| `vad:cng` | 34 | 55.9 | 61.8 |
| `vad:zero` | 39 | 69.2 | 69.2 |
| `plc` (`repeat`) | 130 | 66.2 | 66.9 |
| `bw` (band-limit / 8 kHz) | 101 | 65.3 | 64.4 |
| all | 391 | 69.3 | 69.8 |

**Heldout half-B facts (computed from `meta.csv`).**
- n = 1,994 clips: 1,603 spoof + 391 bona.
- Clips per stage, both labels: no DSP 704; NS 676 (afftdn 346, anlmdn 330); PLC 620; band-limit 516; AGC 513
  (acompressor 258, dynaudnorm 255); VAD 313 (cng 160, zero 153).
- PLC loss rate over the 620 PLC clips: min 0.000, median 0.078, mean 0.085, max 0.297.
- VAD gated fraction over the 313 VAD clips: min 0.000, median 0.127, max 1.000.
- Codecs: gsm 676, g722 671, g726 647.
- Matched half B has **no** DSP stage: 0 of its 1,994 rows has a non-empty `dsp_chain`. Matched is noise + Opus
  (7 app profiles) only.
- In `scripts/build_dev_noisy_sim.py`, heldout NS / AGC / VAD / band-limit run only on offline sources (l.189).
  Repeat-PLC runs *after* the codec, on 30 % of all clips (l.199-200).

**What a promotion needs (computed from the base heldout confusion counts).**
- Heldout half B (merge) has 391 bona, 120 bona FA and 16 spoof misses (AUD §3.2), so TP_bona = 271 (computed).
  Turning one bona FA into a correct call raises F1_bona by 0.0018 and F1_spoof by 0.0003. Macro-F1 then rises
  0.105 points, which at weight 0.35 is +0.037 WF1. Matched is about the same.
- The wave's ±0.5 null band (`exploop/B3/wave.json` `rules.band`) therefore needs about 14 fewer noisy bona FA. The
  base has 213 / 210 (AUD §3.2).

No single targeted slice is big enough to deliver that alone, so every hypothesis states two things:
- a **slice signature** in clips, which tests the mechanism;
- a **spill-over claim**, which is the WF1 effect.

I give honest expected WF1 ranges. §5 pre-registers what happens when a lever is "null with the slice signature".

## 2. Levers

### 2.0 Rules shared by all three levers (details in §3)

**Hook and order.** Today `RTCAugmenter.__call__` (`SLS_setup/utils/rtc_augment.py` l.355-368) runs `_stage`, then
the codec. B4 overrides `__call__` in `RTCAugmenterB4(RTCAugmenterB3)` and inserts a `_post_dsp` step between them.
The new stages run in this fixed order:
1. the parent `_stage` (noise / echo / reverb, unchanged);
2. `wrtc_ns`;
3. `wrtc_dtx`;
4. the codec: the original `codecs_augm`, or the libopus PLC path for the clips `opus_plc` picks.

This is the RTCFake / TFCL order noise → front-end processing → codec (SOTA §3.0, cross-cutting rule 1). It is also
the order of the heldout builder, `scripts/build_dev_noisy_sim.py` l.189: "faithful order: noise -> platform DSP ->
codec". Augmentation runs on the whole clip before the random 4.04 s crop (`utils/data_utils.py` `_augment` →
`_featurize`), as in B2f.

**Separate RNG stream.**
- Every B4 draw comes from `rng4 = np.random.default_rng([base_seed, os.getpid(), 0xB4])`.
- `rng4` is created lazily per process and re-created on a pid change, as `RTCAugmenter._get_rng` does. The loader
  has `persistent_workers=True` (`b3wave/train.py` l.55), so each worker keeps its stream across epochs.
- `base_seed` is `cfg.seed`, which is 1234 in B2f. With `None` it is a random 31-bit draw, as in `make_rng`. This is
  the same pattern as `utils/level_augment.py`: `default_rng([base, os.getpid(), 0x1E7E1])`.
- The parent `rng` therefore makes exactly the stage and codec draws it makes in B2f. With a lever on, 70 % of clips
  see exactly the B2f distribution. The other 30 % keep their B2f stage and codec draws and only gain the new
  component. That is one attributable component.

**Lever off = base.**
- With every B4 flag at its default, `build_rtc_augmenter_b4(args)` returns `build_rtc_augmenter_b3(args)`
  unchanged, so the augmenter *is* the B2f/B3 object.
- Nothing new is imported: `webrtc_noise_gain`, `webrtcvad` and the libopus `ctypes` handle load lazily, only when
  their lever is on.
- `describe()`, `meta_keys` and the `config.yaml` augmenter keys keep their values.
- The trainer prints no lever line. It prints only the scaffold's `b4wave: <version>` line in place of
  `b3wave: <version>`, which the wave's early check expects (`rdlib.build_seed`: `^b4wave: `).

**How B4 avoids the B3 lesson-4 signature.** AUD §6 caution: "Every input-side change that made training harder cut
the late-training fit and moved heldout bona toward spoof". B4 avoids it this way:
- the dose is 0.3, on a separate stream;
- NS and DTX *remove* energy (noise, non-speech). They do not add a harder corruption;
- PLC damages ≤ 15 % of packets, on 30 % of clips;
- the codec app profiles stay fixed (B3d randomised them);
- the B3d / B3e losses were measured with JS consistency on (AUD §6). The B4 levers run on B2f without it.

**Shared under-fit falsifier** (pre-registered for every lever): ep1 TrainAcc < 85.36 (B2f's 87.36 − 2) or ep8
TrainLoss > 0.0752 (1.5 × B2f's 0.050114), *together with* heldout bona recall down. The B2f values are in RUN
`logs/train.log` line 1 (`Epoch 1/8 … TrainAcc=87.36%`) and line 15 (`Epoch 8/8 TrainLoss=0.050114`).

**Shared class-neutrality telemetry.** The augmenter cannot see the label. So `b4wave/train.py` also sums each
batch's `meta` count vector separately for y = bona and y = spoof. When a B4 lever is on, it logs the B4 keys per
class as `Epoch N B4 aug-by-class bona: … | spoof: …`, and `metrics.jsonl["b4"]["aug_by_class"]`. This makes the
realised per-class rate of every new stage visible (§2.2 risk).

---

### 2.1 `apm_ns`: WebRTC noise suppression between the noise stage and the codec

**Motivation.**
- *Audit evidence.*
  - The worst heldout DSP slice is the FFT suppressor: `ns:afftdn` bona recall 55.4 / 58.1 (n = 74), against 72.7
    with no DSP (computed; AUD §3.4). That is a 17.3 / 14.6-point drop (computed) that no lever has touched.
  - NS is in 676 of the 1,994 heldout half-B clips (computed).
  - B2f never trained on any suppressor. The DeepFilterNet `suppress` stage exists but was never enabled (AUD G1),
    and `deepfilternet` is not installed in the env (import check, this session).
- *Literature.*
  - TFCL simulates the RTC front end with "the standard implementations provided by WebRTC". Training on a mix of
    clean and front-end-processed audio takes XLSR-AASIST's full-chain EER from 22.20 to 17.39, with clean EER
    0.23 → 0.52 (Table 2, as read in SOTA §2.1): "The Mix strategy partially alleviates the degradation."
    (https://arxiv.org/html/2607.17761)
  - RTCFake names "built-in speech enhancement modules of communication platforms, such as noise suppression, echo
    cancellation, and automatic gain control" (https://arxiv.org/html/2604.23742v1).
  - WCR: "detectors can often identify that speech has been processed" (https://arxiv.org/abs/2603.14033).
  - WNG: "Tiny Python wrapper around webrtc-audio-processing for noise suppression and auto gain only". It takes
    16 kHz mono int16 in 10 ms blocks, with NS level 0-4 and auto-gain 0-31 dBFS
    (https://github.com/rhasspy/webrtc-noise-gain; SOTA §2.6).
  - PYWRTC is the fallback binding, with wheels for Python 3.10-3.14 and NS levels 0-3
    (https://github.com/strands-labs/pywebrtc-audio).

**Mechanism.**
- With probability `--aug_apm_p` (drawn from `rng4`), the post-stage audio goes through one WebRTC NS pass, with AGC
  off. The level is drawn uniformly from `--aug_apm_ns_levels`.
- The pass is independent of the class and of the acoustic stage drawn. It also runs on clips with no noise stage,
  because platforms run NS on every call.
- Both classes then show suppressor artefacts: residual and musical noise, spectral holes, speech-onset
  attenuation. So the artefact stops being a spoof cue.
- Because NS runs on (speech + noise), the model also sees the realistic "noise partly removed" version of
  in-distribution noise. That is a secondary route into G2.
- NS alone is one component. AGC2 is left out (§6).

**Flags.**
- `--aug_apm_p` (float, default **0.0**): per-clip probability of the NS pass.
- `--aug_apm_ns_levels` (int list, default `1 2 3 4`, choices 1-4): the WNG `noise_suppression_level` values to draw
  from. Level 0 (off) is rejected.

**Default equivalence.** With `--aug_apm_p 0.0`:
- `build_rtc_augmenter_b4` returns the B3 object, and no `rng4` exists or is drawn from;
- `webrtc_noise_gain` is never imported;
- `--aug_apm_ns_levels` is read only when p > 0;
- `describe()`, `meta_keys`, the header and the `config.yaml` augmenter keys are b3wave's;
- the model is untouched, so the forward pass and the checkpoint keys are identical.

**Implementation sketch.**
- New `SLS_setup/b4wave/dsp_apm.py` (~60 LOC):
  - `wrtc_ns(audio, level, sr=16000) -> np.float32`:
    1. clip to [−1, 1], convert to int16 and pad to a multiple of 160 samples;
    2. build a fresh `webrtc_noise_gain.AudioProcessor(0, level)` per clip (`auto_gain_dbfs` 0 = AGC off);
    3. prime the noise estimate on the first min(0.5 s, len) and discard that output;
    4. run `Process10ms(frame).audio` over the whole clip from the start;
    5. convert back to float32 and trim the pad.
  - `selftest()` on a synthetic 1 s tone plus white noise. It raises unless:
    - the output is finite and the same length;
    - the energy drop on a noise-only input is monotone in the level;
    - the speech-band RMS of a −30 dBFS tone at level 1 stays within ±3 dB, which shows that AGC is really off.

    It also prints the < 100 Hz energy change, in case the binding runs the APM high-pass filter.
- `SLS_setup/b4wave/augment.py`:
  - add `apm_p` and `apm_ns_levels` to `RTCAugConfigB4`;
  - in `RTCAugmenterB4._post_dsp`, when `apm_p > 0 and rng4.random() < apm_p`: draw
    `lvl = levels[rng4.integers(len(levels))]`, then `out = finalize(wrtc_ns(out, lvl), out)`;
  - count `wrtc_ns{lvl}`; on an exception, pass the clip through and count `fail_wrtc_ns`;
  - at start-up, call `assert_training_ops(filters=["wrtc_ns"])`, plus `dsp_apm.selftest()` when the lever is on.
- `SLS_setup/b4wave/train.py`:
  - add both flags to a "B4 wave" argparse group;
  - when the lever is on, print
    `B4 APM NS: on p=0.3 levels=[1, 2, 3, 4] agc=off backend=webrtc_noise_gain <version>`;
  - `meta_keys` gains `stage:wrtc_ns1..4` and `stage:fail_wrtc_ns`. They appear in the existing
    `Epoch N B3 aug ...` log line (`b3wave/train.py` l.776-778, written through `log_line`, which prints and logs),
    and in the per-class B4 line.
- `SLS_setup/b4wave/checks.py::check_apm`: runs the selftest, rejects level 0, and checks the per-level energy drop.

**Hypothesis.** B2f reads suppressor artefacts on noisy speech as synthesis. Training both classes through a
different suppressor family (WebRTC NS, levels 1-4) on 30 % of clips makes those artefacts class-neutral. The gain
transfers to the unseen `afftdn` suppressor and, through the generic "processed ⇒ spoof" bias (WCR), to other
heldout processing.

**Predicted signature** (`scripts/b4_dsp_slices.py` + `metrics.json`, both readouts against B2f):
- `ns:afftdn` bona recall up ≥ 8 points (≥ 6 of 74 clips), a larger gain than on the no-DSP slice (n = 128).
- `ns:anlmdn` down by no more than 4 clips.
- Heldout F1 up ≥ 1.0 (from 87.92 / 88.27, AUD §3.1); heldout EER down (from 12.03 / 11.11).
- Expected WF1 Δ: +0.3 to +0.8.
- Must not move:
  - clean F1 within 0.5 of 97.60 / 97.69;
  - R_spoof within 0.5 on every set;
  - matched F1 down by no more than 0.5 (from 91.22 / 91.24);
  - ep1 TrainAcc within 2 of 87.36.

**Falsifier.**
- `ns:afftdn` gain < 4 points (≤ 2 clips) **and** heldout F1 Δ < +0.5: WebRTC NS exposure does not transfer to an
  unseen suppressor. Drop.
- WF1 Δ < −0.5 **with** matched F1 down by more than 1.0: losing raw-noise exposure costs more than NS robustness
  buys.
  - Today noise reaches 0.8 × 6/8 = 60 % of clips (AUD G2(c)).
  - With the lever on, 0.6 × 0.7 = 42 % of clips keep un-suppressed noise (computed).
  - Drop at this dose. A lower p earns one retry, and only if `ns:afftdn` moved.
- The shared under-fit falsifier.

**Equivalence test** (`python -m b4wave.checks augmenter`, CPU):
1. All B4 flags at default: `build_rtc_augmenter_b4` returns an `RTCAugmenterB3`. 200 synthetic clips without the
   codec, plus 10 with it, are bit-identical to `build_rtc_augmenter_b3`'s output, with identical `stage_counts` and
   `describe()`. This mirrors `b3wave/checks.py::check_augmenter`.
2. A forced `RTCAugmenterB4` with default B4 fields is also bit-identical (the delegation path).
3. With `apm_p = 1.0`, the parent `stage:*` counts over 200 clips equal the base augmenter's (same seed, same pid).
   So `rng4` leaves the parent stream untouched.
4. `'webrtc_noise_gain' not in sys.modules` after the levers-off augmenter is built.

The supervisor's regression covers the trainer (`cycle.json` `regression`):
- a re-score of the B2f checkpoint on `sim_heldout_v1 / probe_orig / probe_noise_only` at tol_logit 1e-4;
- the levers-off micro-run against the b3wave micro-run.

**Cost.**
- 160 LOC; VRAM 0.
- Epoch × ≤ 1.05 (estimate; the package is not installed, so I could not time it). That is about 660
  `Process10ms` calls per clip (the mean train clip is 6.57 s over 24 random clips, measured), plus a 0.5 s prime,
  on 30 % of clips, in the loader workers.
- For scale, the ffmpeg codec stage that every clip already runs costs 270-596 ms per 6 s clip in this session
  (mean 367 ms, 8 calls, load average ~25; measured). The NS pass is C++ at 10 ms granularity, expected an order of
  magnitude below that.
- `checks timing` prints ms per clip per stage, and the smoke rate (S_apm_ns vs S0) measures the real factor.

**Smoke** (4.5 min, 1-epoch 1 % subset):
- The header matches `^B4 APM NS: on p=`.
- On the `Epoch 1 B3 aug` line, Σ `wrtc_ns1..4` / Σ `codec:*` is within 0.3 ± 3·√(0.21 / N_codec). At N ≈ 758
  that is [0.25, 0.35] (computed). `fail_wrtc_ns` = 0.
- In the per-class B4 line, the rate is within 0.3 ± 0.12 for each class. That is 3 SD at the ≈ 144 bona clips of
  the subset (computed).
- TrainLoss and DevLoss are finite.
- The rate is within 1.3× S0, and peak VRAM ≤ S0 + 0.1 GiB.

**Risks.**
- **Wheel.** `webrtc-noise-gain` needs a cp312 wheel (the env runs Python 3.12.14, measured). If there is none, the
  owner leaves a note, and `pywebrtc-audio` becomes the backend behind the same `wrtc_ns()` signature. Its NS levels
  are 0-3 (SOTA §2.6), mapped as level − 1, and the dependency name changes at the gate.
- **Level semantics.** `check_apm` must confirm the 1-4 mapping (a monotone energy drop).
- **Transfer.** TFCL trains and tests on the same WebRTC chain, so transfer to `afftdn` is not shown (SOTA I1). The
  falsifier tests exactly this.
- **Matched exposure.** Matched has raw noise and no NS, so it may lose up to the falsifier bound.
- **G6 labelling.** Heldout `ns:*` becomes "family seen, implementation unseen" and must be reported as such (§4).

---

### 2.2 `dtx_gate`: WebRTC-VAD DTX gating with zero or LPC comfort-noise fill

**Motivation.**
- *Audit evidence.*
  - With no speech evidence, the model calls spoof with certainty. On the bona-source probes (n = 187), `sil_only`
    has mean dP +0.8929 / +0.8985 and `noise_only` +0.8870 / +0.8965, with bona → spoof 100 % for both (AUD §3.5).
  - The saturated core is large: 136 of 213 (merge) and 133 of 210 (final) noisy bona FA sit at p ≥ 0.99 (AUD §3.2).
  - VAD-gated bona is flipped: `vad:cng` 55.9 / 61.8 (n = 34), `vad:zero` 69.2 / 69.2 (n = 39) (computed).
  - G3 is the only gap no B2 or B3 lever moved: "No LR or schedule moved that core" (AUD G3), and B3a kept 77 % of
    its FA at p ≥ 0.99 (AUD §3.7).
- *Literature.*
  - SIL: "removing the silence of test speech through Voice Activity Detection (VAD) can severely degrade
    performance". Training where "the silence frames are masked by zero (Sil-mask)" lowers ASVspoof 2021 DF EER for
    3 of 3 models: AASIST 16.92 → 16.06, LCNN 22.46 → 20.41, SENet 22.95 → 22.14. The LA results are mixed (Table VI,
    as read in SOTA §2.2) (https://arxiv.org/html/2309.11827, https://arxiv.org/abs/2309.11827).
  - SSG: "EER increases from 3.6% (with silence) to 15.5% (trimmed silence)" (https://arxiv.org/abs/2106.12914).
  - TFCL Table 1: VAD is the last and most damaging front-end stage (AGC 12.70 → VAD 22.20 EER)
    (https://arxiv.org/html/2607.17761).
  - WVAD: "A frame must be either 10, 20, or 30 ms in duration"; "0 is the least aggressive about filtering out
    non-speech, 3 is the most aggressive" (https://github.com/wiseman/py-webrtcvad).

**Mechanism.** With probability `--aug_dtx_p` (from `rng4`), the clip is gated after the NS slot and before the
codec:
1. Run `webrtcvad.Vad(mode)` per frame. The mode ∈ {1, 2, 3} and the frame ∈ {10, 20, 30} ms are drawn per clip.
   Mode 0 ("quality") is left out: DTX VADs are tuned to gate, and mode 0 would make many draws a no-op.
2. Dilate the speech mask forward by a DTX hangover of U(40, 200) ms.
3. Fill the non-speech frames with digital zeros or with LPC-shaped comfort noise. Under `mix` the fill is drawn per
   clip, 50 / 50.
4. Cross-fade with 5 ms raised-cosine ramps.

Both classes get the same treatment, so the content and spectrum of gated pauses stop carrying class information,
and gated stretches read as "no evidence" rather than "spoof". This is the RTC analogue of SIL's Sil-mask, applied
stochastically. It is one component, gating. Pause insertion and attenuation fill are left out (§6).

**Flags.**
- `--aug_dtx_p` (float, default **0.0**): per-clip probability of DTX gating.
- `--aug_dtx_fill` (choice `mix | zero | cng`, default `mix`): the fill of the gated frames.

**Default equivalence.** With `--aug_dtx_p 0.0`:
- the B3 augmenter object is used, and there are no `rng4` draws;
- `webrtcvad` is never imported;
- `--aug_dtx_fill` is read only when p > 0;
- the model is untouched.

**Implementation sketch.**
- New `SLS_setup/b4wave/dsp_dtx.py` (~100 LOC):
  - `wrtc_dtx(audio, rng4, sr, fill) -> (np.float32, gated_frac, fill_used)`:
    1. int16 frames → `Vad(mode).is_speech(frame_bytes, sr)` → a boolean mask (a trailing partial frame counts as
       speech);
    2. forward dilation by the hangover;
    3. build the fill:
       - `zero`: exact zeros;
       - `cng`: our own comfort-noise generator. An order-10 LPC fit of the gated content (autocorrelation,
         `scipy.linalg.solve_toeplitz`) shapes white noise from `rng4` through `scipy.signal.lfilter`. The RMS is
         the gated content's RMS × 10^(U(−6, 0)/20), floored at −70 dBFS when the gated content is digital silence;
    4. if nothing is gated, return the input unchanged (still counted, in bin `dtx_g0`).
  - This differs from the reserved `sim_ops.vad_gate` (`utils/sim_ops.py` l.86-112), which uses an energy VAD with
    the threshold at the 10th percentile + 6-15 dB and fills with *white* noise at 0.5 × the floor.
  - `selftest()` on a synthetic tone-burst + silence clip:
    - the gated fraction is within bounds;
    - zero frames are exactly 0;
    - CNG frames are within ±3 dB of the target;
    - an all-tone clip comes back unchanged;
    - the length is unchanged.
- `augment.py`:
  - add `dtx_p` and `dtx_fill` to `RTCAugConfigB4`;
  - in `_post_dsp`, after the NS slot, when `dtx_p > 0 and rng4.random() < dtx_p`: `out = finalize(gated, out)`;
  - counts:
    - `wrtc_dtx_{zero|cng}`;
    - a gated-fraction bin `dtx_g0 / dtx_g1_20 / dtx_g20_40 / dtx_g40p`;
    - `fail_wrtc_dtx` on an exception (the clip passes through);
  - call `assert_training_ops(filters=["wrtc_dtx"])`. The name avoids the reserved `vad_gate`.
- Header when on: `B4 DTX gate: on p=0.3 fill=mix modes=[1, 2, 3] frames_ms=[10, 20, 30] hangover_ms=40..200`.
- `checks.py`:
  - `check_dtx` runs the selftest;
  - `dtx_balance` gates 200 bona + 200 spoof training clips (stratified by src and lang, read-only, CPU, no codec)
    and prints the realised gated fraction per class (mean and quartiles).

**Hypothesis.** Part of the saturated core is "non-speech or gated content ⇒ spoof". Class-independent DTX gating
on 30 % of clips makes gated pauses class-neutral. The VAD-gated heldout bona and the saturated noisy FA then
shrink, at no cost on clean.

**Predicted signature** (vs B2f, both readouts):
- `vad:cng + vad:zero` bona recall (n = 73): up ≥ 6 clips.
- Saturated noisy bona FA (p ≥ 0.99, matched + heldout half B): down ≥ 10 from 136 / 133. Their share of the noisy
  FA drops below 63.8 % (136 / 213, computed).
- `probe_padded` bona flip not above 3.74 % / 5.35 % (AUD §3.5).
- `sil_only` bona mean dP below +0.8929 / +0.8985. This is a direction only: the probe has no correct answer.
- Expected WF1 Δ: 0 to +0.6.
- Must not move:
  - clean F1 within 0.5 of 97.60 / 97.69;
  - noisy spoof misses (24 / 23, AUD §3.2) up by at most 8;
  - R_spoof within 0.5.

**Falsifier.**
- VAD-slice gain < 3 clips **and** saturated FA within ±5 of 136 / 133: gated silence is not what drives the core.
  Drop.
- Clean F1 down by more than 0.5, or noisy spoof misses up by more than 8: the gating pattern carries class
  information (SIL's mixed LA results, and the class-rate risk below). Drop.
- The shared under-fit falsifier.

**Equivalence test.**
- Items 1, 2 and 4 of §2.1, with `webrtcvad` not imported.
- Item 3 with `dtx_p = 1.0`: the parent stage counts are identical to the base.
- `check_dtx` asserts that the output length equals the input length.

**Cost.**
- 200 LOC; VRAM 0.
- Epoch × ≤ 1.03 (estimate): about 220-660 `is_speech` calls plus one LPC fit per gated clip, on 30 % of clips.
  `checks timing` measures it.

**Smoke** (4.5 min):
- The header matches `^B4 DTX gate: on p=`.
- Σ `wrtc_dtx_*` / Σ `codec:*` within [0.25, 0.35]; `fail_wrtc_dtx` = 0.
- At least one non-`dtx_g0` bin is > 0, so the VAD really gates training audio.
- The per-class B4 line has both classes in the gated bins.
- Finite losses; rate within 1.3× S0.

**Risks.**
- **Class-dependent realised gating.** Any content-driven VAD gates more where there is more pause. Measured proxy
  (40 clips per class × src, this session): the share of 20 ms frames more than 30 dB below the clip's
  95th-percentile frame energy is:

  | class / src | offline | online |
  |---|---|---|
  | bona | 0.205 | 0.165 |
  | spoof | 0.153 | 0.155 |

  The gate will therefore fire on bona pauses a few points more often than on spoof pauses. A real platform does
  the same at eval time, but it is a potential "gated ⇒ bona" shortcut. Three safeguards:
  - `checks dtx_balance` quantifies it before the run;
  - the per-class B4 telemetry quantifies it during the run;
  - the spoof-miss guard in the falsifier catches the harm.
- **Mixed evidence.** Sil-mask hurts SENet / AASIST on LA19 unseen attacks (SOTA §2.2). Hence the clean and spoof
  guards.
- **Low-SNR speech.** Modes 2-3 may gate low-SNR speech segments. That is realistic front clipping, and it is
  counted in the `dtx_g40p` bin.
- **G6 labelling.** The heldout VAD slices become "family seen, implementation unseen".
- **Small n.** The VAD slices hold 73 bona clips, so the saturated-core count and the probes carry most of the
  mechanism evidence.
- **Wheel.** The pip package `webrtcvad-wheels` provides `import webrtcvad`.

---

### 2.3 `opus_plc`: in-process libopus with bursty packet loss and decoder PLC

**Motivation.**
- *Audit evidence.*
  - Loss + repeat-PLC is the largest heldout DSP slice. It is in 620 of the 1,994 half-B clips, and its bona recall
    is 66.2 / 66.9 (n = 130), against 72.7 without DSP (computed; AUD §3.4).
  - The heldout loss rate has median 0.078 and range 0-0.297 (computed from `meta.csv` `dsp_chain`).
  - B2f's codec stage never drops a packet. B3d's `-packet_loss` was only the encoder's expected-loss hint
    (`b3wave/augment.py` `codecs_augm_random`), and ffmpeg 4.2 has no FEC / DTX options (SOTA §2.3; `ffmpeg
    -version` = 4.2.7, this session).
  - "Packet loss + PLC" is on the never-run list (AUD §2, A2.6 / D4).
  - G2: the worst matched profiles are the low-rate narrow-band ones, qq 61.62 / 66.67 and wechat 65.38 / 71.15
    (AUD §3.4). The ADD-C loss model acts on exactly such codec frames.
- *Literature.*
  - ADD-C degrades six codecs at 0-20 % packet loss. Models trained on clean data degrade by 5.30 EER on average
    (baselines) and 2.33 (the authors' models). Trained on the codec + packet-loss augmented set: "The EER remains
    unchanged, with a slight improvement of 0.003% from C0 to C1" (https://arxiv.org/html/2504.12423v2).
  - RTCFake lists "codec encoding and decoding, and packet loss" among the "key distortion sources"
    (https://arxiv.org/html/2604.23742v1).
  - UR21 trains with "an acoustic simulator to apply transmission codec, compression codec, and convolutional
    impulse responses" (https://arxiv.org/abs/2107.12018).
  - OPUSLIB, "Python bindings to the libopus, IETF low-delay audio codec", is a thin `ctypes` layer over the same C
    calls this lever uses (https://github.com/orion-labs/opuslib,
    https://github.com/orion-labs/opuslib/blob/master/opuslib/classes.py). We follow its pattern without taking it
    as a dependency.

**Measured in this session** (in-memory `python -I` in the `rtc-sdd` env; nothing written):
- `ctypes.util.find_library("opus")` loads, and `opus_get_version_string()` returns `libopus 1.3.1`.
- **The ctl constants self-verify.**
  - `OPUS_SET_BITRATE(16000)` (4002) returns 0 and reads back 16000 through 4003.
  - `OPUS_SET_MAX_BANDWIDTH(1103 = WB)` (4004) reads back 1103 through 4005.
  - `OPUS_GET_LOOKAHEAD` (4027) is 104 samples at 16 kHz.
- **The PLC convention works.** `opus_decode_float(dec, NULL, 0, pcm, 320, 0)` returns 320 (one full 20 ms frame)
  for each of 3 consecutive lost packets on a 6 s training clip. Frame RMS around the burst: 0.0204 (received) →
  0.0196 → 0.0139 → 0.0086 (concealed) → 0.0024. libopus fades out.
- **Cost on a 6 s clip** (this host had load average ~25 with the GPU idle, so absolute times are inflated):
  - in-process encode + decode with 3 lost packets: 363 ms (mean of 8);
  - `utils.rtc_augment.codecs_augm` (two ffmpeg spawns): mean 367 ms over 8 calls (min 270, max 596).

  The ratio is ≈ 1.0. The a2 attempt of this step measured 129 vs 241 ms on a quieter host
  (`rdloop/B4/steps/propose_v1.a2.out.json`). The in-process path is not more expensive.
- **Fidelity against the base codec path.** Training clip `offline/en/Train_000800.wav`, first 6 s, zero loss.
  ffmpeg's output lags the input by 69-70 samples, and `finalize` truncates it. After integer-lag alignment:

  | profile | SNR(ffmpeg, in-process) | SNR(ffmpeg, input) | SNR(in-process, input) |
  |---|---|---|---|
  | 12 kbps, cutoff 6000, 20 ms, voip | 18.3 dB | 6.7 dB | 6.3 dB |
  | 32 kbps, cutoff 8000, 20 ms, voip | 17.4 dB | 8.5 dB | 8.5 dB |
  | 24 kbps, cutoff 8000, 60 ms, audio | 17.6 dB | 14.1 dB | 16.9 dB |

  On every profile the two codec paths are closer to each other than ffmpeg is to the input. The margin is smallest
  on the `audio` profile (3.5 dB), where the in-process path is 2.8 dB *closer* to the input. ffmpeg decodes with
  its native Opus decoder at 48 kHz and resamples to 16 kHz, so a residual fractional-delay / filter difference is
  expected.

**Mechanism.**
- The codec stage still fires on the parent's draw (p = 1.0 in B2f).
- For a clip the lever picks (`rng4.random() < --aug_opus_plc_p`), `draw_codec_params(rng)` makes the parent's draws
  in exactly `codecs_augm`'s order (`utils/rtc_augment.py` l.186-193: app, then `-b:a`, `-cutoff` and
  `-frame_duration`, evaluated left to right).
- The round trip then runs in process:
  1. Set up a libopus encoder (16 kHz mono) with the *same* profile values:
     - the bitrate;
     - the max bandwidth from the cutoff: 4000 → NB 1101, 6000 → MB 1102, 8000 → WB 1103 (ffmpeg's `-cutoff` maps
       it the same way);
     - the frame (20 / 40 / 60 ms);
     - application voip 2048 / audio 2049;
     - VBR on and complexity 10, as ffmpeg's defaults.
  2. Drop packets with a Gilbert-Elliott burst model. Draw the target loss L ~ U(`--aug_opus_loss`) and the mean
     burst B ~ U(`--aug_opus_burst`) from `rng4`. Then p_bg = 1/B and p_gb = L·p_bg/(1 − L), so the stationary
     loss is L. The start state is drawn from the stationary distribution.
  3. The decoder conceals every lost packet: a NULL payload triggers libopus PLC.
  4. Trim the encoder look-ahead and pad to the input length.
- Clips the lever does not pick call the **original** `codecs_augm(out, rng, …)` unchanged.
- The app mix and profiles stay fixed, unlike in the exhausted B3d. The only new component is real packet loss with
  codec-native concealment. It is PLC only; in-band FEC is left out (§6).
- The loss range U(0.02, 0.15) has mean 0.085, which equals the heldout mean 0.085 (computed). It stays inside
  ADD-C's 0-20 % and below the heldout maximum 0.297, so the heldout PLC slice still tests a different concealment
  (repeat with 0.5^run decay after a non-Opus codec, `utils/sim_ops.py` l.115-135), partly at harsher rates.

**Flags.**
- `--aug_opus_plc_p` (float, default **0.0**): probability that a codec call takes the libopus loss + PLC path.
- `--aug_opus_loss` (2 floats, default `0.02 0.15`): the range of the target packet-loss rate, with 0 ≤ lo ≤ hi < 0.5.
- `--aug_opus_burst` (2 floats, default `1.0 4.0`): the range of the mean burst length in packets, with 1 ≤ lo ≤ hi.

**Default equivalence.** With `--aug_opus_plc_p 0.0`:
- the B3 augmenter object is used, so every codec call is the original `codecs_augm` with the same parent draws;
- libopus is never loaded through `ctypes`;
- the loss and burst flags are read only when p > 0;
- the model is untouched.

**Implementation sketch.**
- New `SLS_setup/b4wave/opus_plc.py` (~150 LOC):
  - `_lib()`:
    - loads libopus lazily, once per process: `ctypes.util.find_library("opus")`, then
      `{sys.prefix}/lib/libopus.so.0`;
    - declares `argtypes` / `restype` for `opus_encoder_create / opus_decoder_create / opus_encode_float /
      opus_decode_float / opus_encoder_ctl / *_destroy / opus_get_version_string`;
    - uses the constants measured above.
  - `draw_codec_params(rng, apps)` replicates `codecs_augm`'s draw order exactly: `app =
    apps[rng.integers(len(apps))]`, then `rng.integers(kbps_lo, kbps_hi + 1)`, then the cutoff index, then the
    frame index.
  - `ffmpeg_opus(audio, sr, app, kbps, cutoff, frame)` is `codecs_augm`'s argv with pre-drawn values. It serves
    only as the fallback when the in-process path raises, so no parent draw is repeated.
  - `opus_loss_roundtrip(audio, rng4, kbps, cutoff, frame, application, loss_range, burst_range, sr) -> (out,
    realised_loss)`:
    - encodes frame by frame with `opus_encode_float` (the last frame zero-padded);
    - applies the Gilbert-Elliott mask;
    - calls `opus_decode_float(pkt)` for received packets and `opus_decode_float(NULL, 0)` for lost ones;
    - trims the `OPUS_GET_LOOKAHEAD` samples and returns the input length.
  - `selftest()` raises when:
    - the library cannot be loaded;
    - a set/get ctl round trip (bitrate, max bandwidth) fails;
    - an output is non-finite or the wrong length;
    - a 1-packet concealed gap decodes to all zeros.
- `augment.py`, in the codec branch of `RTCAugmenterB4.__call__` when `opus_plc_p > 0`:
  1. parent `rng.random() < codec_p_apply`, as in the base;
  2. `if rng4.random() < opus_plc_p`:
     1. `draw_codec_params(rng)`;
     2. `opus_loss_roundtrip(...)`;
     3. `_count(f"codec:{app}")` as the parent does, plus `opus_plc` and a realised-loss bin
        `plc_l0_5 / plc_l5_10 / plc_l10p`;
     4. on an exception, `ffmpeg_opus` with the same parameters, counting `fail_opus_plc`;
  3. else `codecs_augm(out, rng, sr, cfg.codec_apps)` exactly as the base;
  4. `finalize(out, audio)`.
- Start-up guards:
  - init raises if `opus_plc_p > 0` while `--aug_codec_profiles random` is set or the codec stage is off;
  - `assert_training_ops(codecs=["libopus"], filters=["opus_plc"])`. The name avoids the reserved `packet_loss`;
  - `opus_plc.selftest()` runs when the lever is on.
- Header when on: `B4 Opus PLC: on p=0.3 loss=0.02..0.15 burst=1..4 libopus=<opus_get_version_string>`.
- `checks.py::check_opus_plc` (below).

**Hypothesis.** Concealment artefacts (pitch-continued, faded or smeared frames) are read as synthesis. Exposing
both classes to libopus PLC under bursty loss, on 30 % of codec calls, makes concealment class-neutral. The gain
transfers to heldout's repeat-PLC (a different concealment, on non-Opus codecs) without hurting matched, which has
no loss.

**Predicted signature** (vs B2f, both readouts):
- `plc` bona recall up ≥ 5 points (≥ 7 of 130 clips), a larger gain than on the no-DSP slice.
- Heldout F1 up ≥ 0.8.
- Matched per-codec bona recall (full set, `breakdowns.sim_matched_v1.codec`) within ±3 points of qq 61.62 / 66.67
  and wechat 65.38 / 71.15 (AUD §3.4).
- Expected WF1 Δ: +0.2 to +0.6.
- Must not move: clean F1 within 0.5; R_spoof within 0.5; ep1 TrainAcc within 2 of 87.36.

**Falsifier.**
- `plc` gain < 2 points (≤ 2 clips) **and** heldout F1 Δ < +0.5: libopus PLC does not transfer to unseen
  concealment. Drop.
- WF1 Δ < −0.5 with ep1 TrainAcc < 85.36 or ep8 TrainLoss > 0.0752: a B3d-type codec-side under-fit. Drop.
- **Attribution control (pre-registered).** If the arm is positive, a control arm `--aug_opus_plc_p 0.3
  --aug_opus_loss 0 0` (in-process codec, no loss) runs before promotion. It is one coupled factor
  (`wave_lever.coupled`). If the control alone gives ≥ half of the arm's WF1 gain, the gain is credited to the
  decoder path, not to PLC.

**Equivalence test.**
- Items 1, 2 and 4 of §2.1, with no libopus `ctypes` handle created.
- Item 3 with `opus_plc_p = 1.0` and the codec on (10 clips, ffmpeg present): the parent `stage:*` **and**
  `codec:<app>` counts equal the base.
- From the same start state, `draw_codec_params` leaves `rng.bit_generator.state` exactly where a real `codecs_augm`
  call leaves it.
- `ffmpeg_opus` on pre-drawn parameters gives output bit-identical to `codecs_augm` with the same RNG.
- **Attribution (zero loss).** With `--aug_opus_loss 0 0`, on 3 training clips × the 7 app profiles, every profile
  must reach an aligned SNR(ffmpeg, in-process) ≥ max(12 dB, SNR(ffmpeg, input) + 2 dB). The values are printed.
  This session measured 17.4-18.3 dB against 6.7-14.1 dB (table above). The check bounds "in-process vs ffmpeg" as a
  second component, and the control arm measures it.

**Cost.**
- 260 LOC: binding ~50, round trip and Gilbert-Elliott ~90, draw / fallback split ~40, checks ~80. VRAM 0.
- Epoch × ≤ 1.02. The in-process path costs about the same as the ffmpeg path it replaces on 30 % of clips
  (measured ratio ≈ 1.0, above), plus a small Python per-packet loop.

**Smoke** (4.5 min):
- The header matches `^B4 Opus PLC: on p=`.
- `opus_plc` / Σ `codec:*` within [0.25, 0.35]; `fail_opus_plc` = 0.
- Every `plc_l*` bin is > 0.
- The codec app counts still cover the 7 apps.
- Finite losses; rate within 1.3× S0.

**Risks.**
- **libopus availability.** The lever uses the system `/usr/lib/x86_64-linux-gnu/libopus.so.0` (`ls`, this
  session). ffmpeg's codec stage needs the same library. A container restart wipes the system ffmpeg (MEMORY note;
  AUD §5 environment row), probably with libopus.
  - Unlike the ffmpeg stage, which then silently passes audio through, `selftest()` raises at start-up.
  - Optional owner hardening: conda-forge `libopus` in `rtc-sdd`, which `_lib()` tries second.
- **Variadic `opus_encoder_ctl` through `ctypes`.** This is the opuslib pattern. The set/get round trip in
  `selftest()` catches a broken call; both round trips passed in this session.
- **Transfer.** ADD-C's gain is in-distribution, and transfer to repeat-PLC is unproven.
- **Frame size.** 40 / 60 ms frames (dingtalk / telegram) conceal more audio per lost packet.
- **Fit.** B3d showed that codec-side changes can cut fit (−2.20). Hence the fixed profiles, the 30 % dose,
  loss ≤ 15 % and the under-fit falsifier.
- **Decoder path.** The in-process path is look-ahead-aligned (lag 0), while the ffmpeg path lags 69-70 samples
  (4.4 ms) and is slightly harsher (table above). A sub-5 ms shift is no cue under the random 4.04 s crop, and both
  classes get both paths. The control arm measures any remaining decoder-path effect.

## 3. Shared changes

| what | files |
|---|---|
| Config and augmenter classes: `RTCAugConfigB4(RTCAugConfigB3)` with the seven new fields, and `RTCAugmenterB4(RTCAugmenterB3)`.<br>The B4 `__call__` mirrors the parent's: the `len < 16` early return → `rng = _get_rng()` → `_stage` → `_post_dsp` (`wrtc_ns` → `wrtc_dtx`) → the codec branch (`codecs_augm`, or `codecs_augm_random` under `--aug_codec_profiles random`, or the PLC split).<br>It also adds:<br>• `rng4` (`default_rng([base_seed, pid, 0xB4])`, re-created on a pid change);<br>• B4 `meta_keys` and `describe()` suffixes, only for active levers;<br>• a start-up self-test of each active stage;<br>• `fail_*` counters;<br>• `assert_training_ops` for the new op names.<br>`build_rtc_augmenter_b4(args)` returns `build_rtc_augmenter_b3(args)` unchanged when every B4 flag is at its default. | `SLS_setup/b4wave/augment.py` |
| `build_augmenter_b4` / `build_dataset_b4`: a copy-adapt of `b3wave.data.build_augmenter_b3` / `build_dataset_b3` with the B4 factory. Dev data uses the original `SpoofAudioDataset`. Train reuses `b3wave.data.SpoofAudioDatasetB3`, which already turns `meta_keys` / `counts_vector` into telemetry. | `SLS_setup/b4wave/data.py` |
| The trainer, a scaffolded copy of `b3wave/train.py`:<br>• a "B4 wave" argparse group with the §2 flags and their value checks: p in [0, 1]; levels in 1-4; loss with 0 ≤ lo ≤ hi < 0.5; burst with 1 ≤ lo ≤ hi;<br>• `build_dataset_b4`;<br>• lever header lines, printed only when the lever is on;<br>• `config.yaml` gains `b4wave_version` and `b4_levers` (the active levers and their realised settings);<br>• the per-epoch `Epoch N B3 aug ...` line keeps its format, and B4 keys appear in it when a lever is on;<br>• `train_epoch` also sums `meta` per label. When a lever is on, it logs `Epoch N B4 aug-by-class ...` and `metrics.jsonl["b4"]`. | `SLS_setup/b4wave/train.py` |
| Checks:<br>• `cpu` = `augmenter` (equivalence items 1-4 for every lever), `codec_draw_order`, `reserve_names` (the new names pass; `vad_gate`, `packet_loss` and `afftdn` still raise), `apm`, `dtx`, `dtx_balance`, `opus_plc` (incl. the attribution check), `timing` (ms per clip per stage on 6 s clips, vs the ffmpeg path) and `dsp_slices` (the script below reproduces the §1 B2f table on RUN);<br>• `model --ckpt` and `optim` stay as in b3wave (the cycle's regression `base_checks`).<br>In `all`, a lever check whose package is not importable prints `SKIP <check>: <module> not installed`. The levers-off path must not need it. Invoking that check by name fails, and a lever-on run fails at its start-up self-test. | `SLS_setup/b4wave/checks.py` |
| Flag table, reproduction statement, telemetry keys, dependency install lines. | `SLS_setup/b4wave/README.md` |
| A read-only readout. For a `local_eval/<readout>` dir (and an optional `--ref` dir) it prints:<br>• half-B heldout bona recall and FA count per `dsp_chain` stage, labelled seen-family vs unseen;<br>• the matched + heldout saturated FA (p ≥ 0.99);<br>• the noisy spoof misses;<br>• the noise-induced FA (clean-correct online sources that flipped; AUD §3.3);<br>• the probe dP / flip rows.<br>Δ is given in clips. It needs no GPU and reads only `scores/*.txt` (P(spoof)), `probe_report.json` and `data/dev_noisy_sim/*/meta.csv`. | `scripts/b4_dsp_slices.py` |

No file outside `SLS_setup/b4wave/` and `scripts/b4_*` is touched. `b3wave`, `utils`, `model`, `main_*.py` and the old
scripts stay byte-identical.

## 4. Eval impact

- No `state_dict_change` and no new eval profile. Checkpoints are plain `ModelSLS` state_dicts. The wave scores them
  with the existing `sls_local17` path and `--loudness_norm --loudness_norm_lufs -23`, as for B2f.
- The new script `scripts/b4_dsp_slices.py` is the mechanism readout for each lever's predicted signature. The
  analyst runs it on both readouts with `--ref RUN/local_eval/<same readout>`.
- **G6 readability.** After these levers, heldout `ns:*`, `vad:*` and `plc` are "family seen in training,
  implementation unseen". The no-DSP, `agc:*`, `bw`, noise-family and codec slices stay fully unseen and are reported
  separately. `sim_unseen_v1` stays unread (music leak, AUD G6).
- Wave verdicts, if B4 keeps the B3 rules (`exploop/B3/wave.json` `rules`):
  - band ±0.5;
  - promote = CI low > 0 and clean drop ≤ 0.5;
  - positive = Δ > +0.5 with CI low > 0 and clean drop ≤ 0.5.

  The reference is B2f at 91.98 / 92.14 (AUD §1).
- Run length: 8 epochs × ~31 min (RUN `train.log` timestamps) × ≤ 1.05 ≈ 4.3 h per arm, well under the 10 h cap
  (computed).

## 5. Interactions and what comes after the wave (pre-registered)

- The three stages sit at disjoint points of the chain: post-noise DSP, gating, network / codec. Each draws from
  `rng4` only when it is on, so any combination is well defined, in the fixed order NS → DTX → codec / PLC.
- If two or more levers show their slice signature (the predicted clip gains) with WF1 null or better, the next run
  is one combined arm with all signature-positive levers on at p = 0.3. That is the "platform chain" TFCL trained
  on. Promotion needs about 14 fewer noisy FA (§1), more than any single slice can give.
- A lever whose falsifier fires is dropped with no dose sweep. There are two exceptions: the `apm_ns` lower-p retry
  named in its falsifier, and the `opus_plc` zero-loss control arm, which runs only if `opus_plc` is positive.

## 6. Ideas considered and not proposed

| idea | decision | why |
|---|---|---|
| AGC2 inside `apm_ns` (I1 as carded) | left out | One component per lever. AGC was already simulated in B2e (`--use_level_aug`, `p_agc 0.30`), and B2f vs B2e was null at −0.36 / −0.39 (AUD §2 row 9: "level treatment is not a lever"). NS is the untested part, and `ns:afftdn` (55.4) is worse than `agc:dynaudnorm` (64.2). |
| Pause insertion inside `dtx_gate` (I2 as carded) | left out | A second component: it changes duration and rhythm, while gating changes pause content. The heldout VAD slices are gating with cng / zero fill. |
| Attenuation fill in `dtx_gate` | left out | A VAD-driven −15 to −35 dB gate is the `agate` family, which is reserved for `sim_unseen_v1` (`b3wave/heldout_reserve.py`). Zeros and comfort noise are the DTX fills. |
| libopus DTX (`OPUS_SET_DTX`) as a dependency-free gate | declined (measured) | On 24 random training clips × 4 profiles (this session), libopus 1.3.1 DTX marked 7.1 % of frames on the 4 bona clips and 1.2 % on the 20 spoof clips, identical across bitrates and applications. It fires on near-digital silence only. That is too little exposure and is class-skewed. |
| In-band FEC in `opus_plc` (I3 as carded) | left out | A second decoder component (LBRR redundancy). It is a follow-up only if PLC shows its signature. |
| `opuslib` as the binding (I3 as carded) | replaced | The system libopus 1.3.1 already works through `ctypes` (measured, §2.3). A ~50-line binding inside the lever avoids a pip dependency and the py3.12 import risk of an old package. |
| I4 AugMix-style single-view mixing | declined | In the time domain, mixing noise chains with a skip to the clean clip gives `x + (1−m)·Σ wᵢ nᵢ`. That raises the SNR by E[−20·log10(1−m)] = 20/ln 10 ≈ 8.7 dB at m ~ U(0, 1) (computed), so the noise gets *milder* on top of the composition change. That makes it two components, one of them on the SNR axis that B3e (exhausted) probed from the other side. The evidence is vision-only (AUGMIX Table 4). Composition (`--aug_max_stages`) is already a b3wave flag the wave can screen without an rdloop build. |
| I5 echo suite (multi-tap + far-end AEC3) | declined | The `sim_echo_v1` self-echo family *is* 1-3 taps of `add_echo` at 30-300 ms / gain 0.1-0.6 (`scripts/build_dev_noisy_sim.py` l.205-209), so a multi-tap training echo would train on the eval recipe and turn the echo set into a matched set. Echo is not in WF1. The AEC3 half needs another binding, and far-end speech from training clips is only "likely allowed" (roadmap §2). There is no ADD paper (SOTA §2.4). |
| I6 DeepAugment-perturbed DeepFilterNet | declined | The evidence is vision-only, and `deepfilternet` is not installed (import check, this session). 48 kHz neural NS on CPU inside the loader workers risks the 1.3× cap. Neural processing of bona is a rule grey zone (SOTA I6). |

## 7. Dependencies (owner action at the gate; this step installed nothing)

An import check in the `rtc-sdd` env (`importlib.util.find_spec`, this session) finds none of `webrtc_noise_gain`,
`webrtcvad`, `pywebrtc_audio`, `opuslib`, `deepfilternet` / `df` or `noisereduce`. The env has Python 3.12.14 and
171 distributions, with no NS or VAD library among them (`importlib.metadata`, this session). The system libraries
are `libopus.so.0`, `libspeex.so.1` (codec only: `speex_preprocess_state_init` is absent, checked with `ctypes`) and
`libsoxr.so.0`.

The supervisor tests each lever's `dependencies` entry with `import <name>` (`scripts/rdloop/supervisor.py`
`importable`), so the entries are import names: `webrtc_noise_gain` (apm_ns) and `webrtcvad` (dtx_gate). `opus_plc`
lists none.

```
/root/.conda/envs/rtc-sdd/bin/python -m pip install webrtc-noise-gain webrtcvad-wheels
/root/.conda/envs/rtc-sdd/bin/python -c "import webrtc_noise_gain, webrtcvad; print('ok')"
# if webrtc-noise-gain has no cp312 wheel: install pywebrtc-audio and leave an owner note (apm_ns backend swap, §2.1 risks)
# optional hardening against container restarts (system ffmpeg + libopus get wiped):
# conda install -n rtc-sdd -c conda-forge libopus
```

All of these are local DSP or software codecs: "local WebRTC AEC / NS / AGC / VAD simulation explicitly allowed;
software codecs allowed" (roadmap §2, AUD §5).

## 8. Responses to the critic

None (v1).

## 9. Changes since the previous version

None (v1).

Process note: attempts a0-a2 of this step drafted the same three components, but the step settings denied every
write of this file (`events.jsonl`: "proposal_v1.md is missing or empty"). The owner fixed the settings and
re-ran the step. This version re-measured every number in this session. Compared with the a2 draft
(`steps/propose_v1.a2.out.json`):
- the cost measurement is redone under load (ratio ≈ 1.0), and `opus_plc` is now costed at 1.02, not 1.00;
- the non-PLC codec share calls the original `codecs_augm` directly; `ffmpeg_opus` is only the failure fallback;
- the per-class B4 telemetry and the `dtx_balance` check are new, backed by the measured low-energy-frame proxy;
- `checks all` now SKIPs a lever check whose package is absent;
- the libopus-DTX alternative is measured and declined;
- the echo decline now names the eval recipe overlap.
