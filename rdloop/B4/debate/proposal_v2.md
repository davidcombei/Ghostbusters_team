# rdloop B4 - proposal v2

> research-architect, REVISE v2 (attempt a0), 2026-10-17. Base **B2f**, entry module `b3wave`, cycle focus
> **augmentation**, levers 2-4. Owner notes: none (`rdloop/B4/user_notes.md` holds only the header).
> This version answers critique v1 (`debate/critique_v1.md`, ledger O1.1-O1.9, 2 majors + 7 minors) and its three
> alternatives. §8 has the point-by-point answers and §9 the change list.
>
> RUN = `SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623`.
> B2e = `SLS_setup/exp/B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234_epoch8_bs16_20261008185925`.
> A pair written "merge / final" is always `local_eval/merge_top5` first, then `local_eval/epoch_8`.
>
> Provenance of the numbers:
> - **copied** from a named file: AUD = `rdloop/B4/01_audit.md`, SOTA = `rdloop/B4/02_sota.md`, CRIT =
>   `debate/critique_v1.md`, RUN `logs/train.log`, `exploop/B3/wave.json`;
> - **(computed)**: read-only `python3 -I` joins of `RUN/local_eval/<readout>/scores/sim_*_v1.{txt,logits}` (and B2e
>   `local_eval/epoch_8`) with `data/dev_noisy_sim/{heldout,matched}_v1/meta.csv`, run in this session. They
>   reproduce the critic's numbers exactly (e.g. offline bona with NS, n = 146, 66.4 / 67.1; without, n = 55,
>   76.4 / 74.5) and AUD §3.2 (saturated noisy bona FA 136 / 133 of 213 / 210);
> - **(measured)**: in-memory runs in the `rtc-sdd` env in this session (CPU-seconds with `time.process_time` plus
>   `RUSAGE_CHILDREN` for the ffmpeg children, one process, `OMP_NUM_THREADS=1`). Nothing was written except this
>   file.
>
> No progress-split file was read.

## 0. Plan in one paragraph

B2f trains on one acoustic stage, then Opus, and nothing else. In the audit's words, "No DSP stage of any kind
has ever been in training" (AUD G1). RTCFake's training split does contain online clips sent through Zoom and QQ,
so it includes platform NS / VAD / codec, but only on *clean* input (CRIT O1.6). Accurate statement: **B2f never
saw noisy speech through a suppressor, and never saw a simulated DSP stage.**

About three quarters of the noisy bona FA are clips the model gets right when clean (AUD §3.3). The plan keeps
the **three single-component, CPU-side levers** of v1. Each one draws from a separate RNG stream at p = 0.3, with
an implementation outside the reserved `sim_ops` and no state_dict change:
1. **`apm_ns`**: WebRTC noise suppression **only**. It now runs on `pywebrtc-audio`'s standalone `NoiseSuppressor`,
   which has no high-pass filter (O1.1). Levels are given in suppression dB: 6 / 12 / 18 / 21.
2. **`dtx_gate`**: webrtcvad DTX gating with zero or LPC comfort-noise fill.
3. **`opus_plc`**: in-process libopus with Gilbert-Elliott loss and decoder PLC. Three changes from v1:
   - it decodes at 48 kHz and resamples, like ffmpeg (O1.9; measured to be closer to the base path);
   - the loss range is ADD-C's 1-20 % (O1.4);
   - it targets G1 only (O1.3).

What changed most is the **mechanism readout** (O1.2). It now uses:
- same-source controls;
- a difference-in-differences of the clipped spoof logit with a bootstrap CI, over the full heldout set (half A +
  half B, final readout primary);
- paired fixed / broken counts and exclusive-stage counts;
- two noise references: B2f merge vs final (within-run) and B2e vs B2f final (cross-run, WF1-null);
- thresholds set at 40 % of each stage's baseline gap;
- for `dtx_gate`, a non-speech-tertile readout of the saturated core.

**Cost is now measured in CPU-seconds** (O1.7). The B2f loader path costs **0.350 CPU-s per clip**, which at B2f's
throughput uses about 90 % of the container's 16-CPU quota. So a stage's CPU cost turns almost one-for-one into
epoch time. `opus_plc` *lowers* the CPU per clip. The NS stage gets a pre-registered timing stop.

The combined arm of v1 is now only a recommendation to the owner (O1.8).

## 1. Summary

| lever | flags (default = B2f) | wave arm | gap | primary mechanism contrast: heldout bona, offline/online as stated, full set, final readout (computed) | est LOC | VRAM | epoch × (saturated loader) | smoke min | new deps (import name) |
|---|---|---|---|---|---|---|---|---|---|
| `apm_ns` | `--aug_apm_p 0.0`, `--aug_apm_ns_levels 6 12 18 21` | `--aug_apm_p 0.3` | G1 | offline with NS vs offline without NS: n 276 / 106, recall 69.2 / 75.5, logit gap 0.85 | 190 | 0 | ≤ 1.06 (estimate; stop rule at 1.15) | 4.5 | `pywebrtc_audio` |
| `dtx_gate` | `--aug_dtx_p 0.0`, `--aug_dtx_fill mix` | `--aug_dtx_p 0.3` | G3, G1 | offline with VAD vs without: n 136 / 246, recall 66.2 / 73.6, logit gap 1.05; plus the saturated FA by non-speech tertile, 66 / 76 / 107 | 210 | 0 | ≤ 1.02 | 4.5 | `webrtcvad` |
| `opus_plc` | `--aug_opus_plc_p 0.0`, `--aug_opus_loss 0.01 0.20`, `--aug_opus_burst 1.0 4.0` | `--aug_opus_plc_p 0.3` | G1 | online with PLC vs online without: n 114 / 254, recall 67.5 / 75.6, logit gap 0.95 | 280 | 0 | 0.85-0.87 projected (measured inputs); 1.0 declared | 4.5 | none (system libopus 1.3.1) |

**Smoke budget (LINT rule).** S0 4.5 + eval path 2.0 + 3 × 4.5 = **20.0 min ≤ 25** (computed). No lever adds
parameters, so there is no overfit job.

### 1.1 The mechanism readout (rebuilt for O1.2)

**Why the v1 readout failed.**
- **Source confound.** Heldout NS / AGC / VAD / band-limit run only on offline sources
  (`scripts/build_dev_noisy_sim.py` l.189), so v1's no-DSP control was 124 online clips + 4 offline.
- **Overlap.** `ns:afftdn` has 8 exclusive bona clips on half B. It co-occurs with `bw` 42, `agc` 41, `plc` 26 and
  `vad` 24 times (computed).
- **Thresholds inside the noise.** The v1 falsifier bands were in clips at half-B n, which is inside the within-run
  noise.

**v2 design.** Every contrast compares *treated vs untreated within the same source*. The primary statistic is a
**difference-in-differences of the per-clip spoof logit**, clipped to ±8 (the `.logits` files). Write
Δᵢ = logitᵢ(run) − logitᵢ(B2f, same readout). Then

> DiD = mean_T(Δ) − mean_C(Δ), with a 2,000× bootstrap 95 % CI over clips.

A negative DiD means the treated slice moved toward bona relative to its control.

Secondary readouts:
- the clip-level DiD, (Δrecall_T − Δrecall_C) · n_T, in clips (positive = good);
- paired fixed / broken counts on T, with an exact two-sided sign test;
- exclusive-stage counts.

**Population.** The **full set (half A + half B) on the final readout** is primary:
- the final epoch is not selected;
- the selection mini-protocol is 1,000 half-A clips (`heldout_v1/protocol_mini.txt`, computed: all 1,000 are half
  A), which touches only the merge;
- the full set doubles n.

Half B, on both readouts, is secondary: the same sign is required. The WF1 promotion rule stays on half B,
unchanged.

**B2f baseline and noise references** (computed; heldout bona).

Columns:
- *gap G*: mean logit T − C on B2f final;
- *D*: n_T × (recall_C − recall_T) in clips, merge / final;
- *within-run*: B2f final vs B2f merge, as logit DiD ± SE / clip DiD / T fixed : broken;
- *cross-run*: B2f final vs B2e final. B2e and B2f are a WF1-null pair: −0.36 / −0.39 (AUD §1). They differ in
  level treatment.

| contrast (full set) | n_T / n_C | B2f final recall T / C | gap G (final) | D (clips) | within-run | cross-run |
|---|---|---|---|---|---|---|
| NS: offline with NS vs offline without | 276 / 106 | 69.2 / 75.5 | 0.85 | 30.1 / 17.3 | −0.22 ± 0.19 / +12.8 / 7 : 2 | −0.36 ± 0.33 / +10.0 |
| NS: `afftdn` vs `anlmdn` (offline) | 139 / 137 | 62.6 / 75.9 | 2.01 | 21.5 / 18.5 | −0.02 ± 0.22 / +3.0 / 5 : 1 | +0.81 ± 0.33 / −7.0 |
| VAD: offline with VAD vs without | 136 / 246 | 66.2 / 73.6 | 1.05 | 13.6 / 10.1 | +0.05 ± 0.19 / +3.6 / 5 : 2 | +0.74 ± 0.32 / −4.9 |
| PLC: online with PLC vs without | 114 / 254 | 67.5 / 75.6 | 0.95 | 11.1 / 9.2 | −0.08 ± 0.19 / +1.9 / 4 : 3 | +0.73 ± 0.36 / −7.2 |
| PLC: offline with PLC vs without | 112 / 270 | 69.6 / 71.5 | 0.37 | 1.2 / 2.1 | +0.27 ± 0.16 / −0.8 / 2 : 2 | +0.46 ± 0.30 / −2.3 |

The half-B rows reproduce CRIT O1.2 (computed):
- NS: 146 / 55, D 14.5 / 10.8;
- `afftdn` vs `anlmdn`: D 16.6 / 13.5;
- VAD: 73 / 128, D 7.0 / 3.9;
- PLC online: 66 / 124, D 4.4 / 2.4;
- PLC offline: 64 / 137, D 1.8 / 3.3.

Half-B within-run logit SEs are 0.20-0.29.

What the table shows:
1. **Clip counts are threshold-noisy.** Merge and final of the *same* run differ by +12.8 clips on the NS contrast,
   but only by −0.22 ± 0.19 in logit. The logit DiD is the stable statistic, so it is primary. Clip DiD is
   secondary, with its threshold set as a fraction of D.
2. **Cross-run variation is large.** B2e vs B2f moved three DSP contrasts by +0.73 to +0.81 logit, with B2f more
   spoof-ward on the treated slice. B2e trained with a simulated AGC on 30 % of clips (AUD §2 row 9), so this is
   *weak, confounded* support for "processing exposure in training reduces processed ⇒ spoof". B2f also normalises
   loudness at eval time, which is the confound. It also bounds what a one-run DiD can prove. Hence the three-way
   verdict below.
3. **The `offline PLC` contrast is below resolution.** Its gap is 0.37 with SE 0.16, so it is reported only.

**Pre-registered verdict per lever, on its primary contrast (full set, final readout):**
- **mechanism shown**: DiD ≤ −0.4 · G with a 95 % CI that excludes 0, the clip DiD ≥ +0.4 · D_final, and the
  half-B DiD negative on both readouts. −0.4 · G is −0.34 to −0.42 for the three primaries, about 2 within-run SEs,
  and beyond every within-run |DiD| of the primaries in the table (max 0.22).
- **mechanism refuted**: DiD ≥ 0.
- **inconclusive**: anything between. It is reported, and it does not by itself decide anything.

**Saturated-core readout for `dtx_gate`** (O1.2 (5), alternative I2). The bona clips of matched + heldout are
split into tertiles by the **non-speech fraction of the clean source clip** in the scored window:
- the eval scores the first 64,600 samples (`utils/data_utils.pad_audio`, `random_start=False`);
- non-speech = 20 ms frames more than 30 dB below the clip's 95th-percentile frame energy;
- cut points 0.159 / 0.249 are fixed from the B2f full-set distribution (computed).

Saturated noisy bona FA (p ≥ 0.99), merge / final (computed):

| tertile (source non-speech fraction) | full set n | full set FA | **full set saturated FA** | half-B saturated FA |
|---|---|---|---|---|
| low (≤ 0.159) | 518 | 102 / 97 | **63 / 66** | 43 / 44 |
| mid | 500 | 130 / 126 | **73 / 76** | 47 / 47 |
| high (> 0.249) | 482 | 159 / 154 | **109 / 107** | 46 / 42 |

The totals match AUD: full set 245 / 249 here; half B 136 / 133 = AUD §3.2. On the full set the saturated core is
concentrated in pause-rich clips: 107 vs 66 on the final readout. On half B alone it is flat. That is evidence for
the G3 hypothesis "no speech evidence ⇒ spoof" at the full-set n, and a warning that half B alone cannot read it.

The measure uses the clean source, not webrtcvad on the sim audio, for three reasons:
1. In the sim clip the pauses are filled with noise at 0-20 dB SNR, so a VAD there measures the noise as much as
   the pauses.
2. Grading `dtx_gate` with its own training VAD would be circular.
3. It needs no dependency.

**Promotion arithmetic** (v1, unchanged). On half B, one bona FA turned correct is worth +0.037 WF1, so the ±0.5
band (`exploop/B3/wave.json` `rules.band`) needs about 14 fewer noisy bona FA. The half-B stage deficits are 10.8
(NS), 3.9 (VAD) and 2.4 + 3.3 (PLC) clips on the final readout. So **no lever can clear the band through its own
slice alone.** Each WF1 claim is a spill-over claim, and the expected ranges below say so.

### 1.2 Cost in CPU-seconds (O1.7)

**The B2f loader path** (measured):
- **Sample.** 40 random training clips, mean 6.94 s. The augmenter is built with the B2f argv.
- **CPU per clip.** load + loudness 0.0268 s, augmenter self 0.0128 s, ffmpeg children 0.3059 s, featurize 0.0044
  s. **Total 0.350 CPU-s per clip.**
- **Throughput.** B2f runs at 40.5-41.1 clips/s (75,785 clips per 30.7-31.2 min epoch, AUD §1; computed). That is
  14.2-14.4 CPU-s/s.
- **Quota.** The quota is 16 CPUs (`/sys/fs/cgroup/cpu.max` = `1600000 100000`, read this session). `cpu.stat`
  shows `nr_throttled` 242,817 of 3,711,099 periods.

**The loader must be treated as saturated.** The projected epoch factor of a stage that costs c CPU-s per
processed clip at rate p is (0.350 + p·c) / 0.350. At p = 0.3, the 1.3 cap is reached at c = 0.35 CPU-s.

**The codec path, paired measurement** (24 clips, mean 6.3 s, 10 % loss, measured):

| path | CPU-s per clip |
|---|---|
| ffmpeg path (self + children) | 0.258 |
| in-process libopus, 16 kHz decode | 0.114 |
| in-process libopus, 48 kHz decode + `resample_poly` | **0.109** |

`opus_plc` at p = 0.3 therefore projects to 1 − 0.3 · 0.874 · (1 − 0.109 / 0.258) = **0.85** (computed). It is
cheaper than B2f if the loader binds, and equal if the GPU binds. Declared cost: 1.0.

**`dtx_gate`, measured parts:**
- order-10 LPC fit + `lfilter` comfort noise on 30 % of a 7 s clip: 0.0012 CPU-s;
- int16 framing into 350 frames: 0.00025 CPU-s.

webrtcvad itself is not installed. It is C at 10-30 ms granularity, estimated ≤ 0.01 CPU-s per clip. Projected
factor ≤ 1.01; declared 1.02.

**`apm_ns`**: not measurable (not installed). Estimate 0.02-0.07 CPU-s per clip, so 1.02-1.06; declared 1.06.
**Stop rule:** if `checks timing` projects > 1.15, the implementer reports it and the lever goes back to the owner
before its smoke.

The smoke-rate check (≤ 1.3 × S0) stays only as a coarse sanity check, because at 758 clips start-up dominates it
(CRIT O1.7).

## 2. Levers

### 2.0 Rules shared by all three levers (unchanged from v1 unless marked)

**Hook and order.** `RTCAugmenter.__call__` (`SLS_setup/utils/rtc_augment.py` l.355-368) runs `_stage`, then the
codec. B4 overrides `__call__` in `RTCAugmenterB4(RTCAugmenterB3)` and inserts `_post_dsp` between them. The fixed
order is:
1. the parent `_stage`;
2. `wrtc_ns`;
3. `wrtc_dtx`;
4. the codec: the original `codecs_augm`, or the libopus PLC path for the clips `opus_plc` picks.

This is noise → front-end → codec, the order of RTCFake / TFCL (SOTA §3.0). It is also the order of the heldout
builder: "faithful order: noise -> platform DSP -> codec" (`scripts/build_dev_noisy_sim.py` l.189). Augmentation
runs on the whole clip, before the random 4.04 s crop (`utils/data_utils.py` `_augment` → `_featurize`).

**Separate RNG stream.**
- Every B4 draw comes from `rng4 = np.random.default_rng([base_seed, os.getpid(), 0xB4])`.
- It is created lazily and re-created on a pid change, like `RTCAugmenter._get_rng`. The loader has
  `persistent_workers=True` (`b3wave/train.py` l.55).
- `base_seed` is `cfg.seed` (1234 in B2f). With `None` it is a random 31-bit draw, as in `make_rng`.
- The parent `rng` therefore makes exactly the B2f stage and codec draws.

**Lever off = base.**
- With every B4 flag at default, `build_rtc_augmenter_b4(args)` returns `build_rtc_augmenter_b3(args)` unchanged.
- No package or `ctypes` handle is loaded.
- `describe()`, `meta_keys` and the `config.yaml` augmenter keys are b3wave's.
- No lever line is printed. Only the scaffold's `b4wave: <version>` line replaces `b3wave: <version>`, which the
  wave's early check expects (`rdlib.build_seed`).

**B3 lesson 4** (AUD §6). Every input-side change that made training harder cut the late fit. B4 guards against it
in four ways:
- the dose is 0.3 on a separate stream;
- NS and DTX *remove* energy;
- PLC touches ≤ 20 % of packets on 30 % of clips;
- the app profiles stay fixed.

The B3d / B3e losses were measured with JS consistency on; the B4 levers run without it.

**Shared under-fit falsifier**, pre-registered for every lever: ep1 TrainAcc < 85.36 (B2f 87.36 − 2) **or** ep8
TrainLoss > 0.0752 (1.5 × B2f's 0.050114), **together with** heldout bona recall down. Source: RUN `logs/train.log`
l.1 "Epoch 1/8 TrainLoss=0.355795 TrainAcc=87.36%" and l.15 "Epoch 8/8 TrainLoss=0.050114".

**Shared class telemetry.** `b4wave/train.py` also sums each batch's `meta` vector separately for y = bona and
y = spoof. When a lever is on, it logs `Epoch N B4 aug-by-class bona: … | spoof: …` and writes
`metrics.jsonl["b4"]["aug_by_class"]`.

**Spoof-miss guards** (new, O1.5). The bound is the Poisson 99 % upper quantile of each baseline count (computed).
A lever whose realised effect is class-asymmetric in the harmful direction ("processed ⇒ bona") shows up as missed
spoofs on its own treated slice.

| guard | n | B2f misses merge / final | bound |
|---|---|---|---|
| full set, offline spoof with NS | 1,135 | 18 / 18 | ≤ 29 |
| full set, offline spoof with VAD | 493 | 15 / 13 | ≤ 25 / 22 |
| full set, spoof with PLC (offline 512 + online 504) | 1,016 | 16 / 15 | ≤ 26 / 25 |
| half B, noisy spoof misses (matched + heldout; AUD §3.2) | | 24 / 23 | ≤ 36 / 35 |

**G6 labelling** (O1.4). After B4, heldout slices must be reported as:

| heldout slice | label |
|---|---|
| `ns:*` | "NS family seen (WebRTC NS); implementation unseen (`afftdn` / `anlmdn`)" |
| `vad:*` | "gating family seen (webrtcvad DTX, zero / LPC-CNG fill); implementation unseen (energy VAD, white-noise fill)" |
| `plc` | "loss-pattern family seen (Gilbert-Elliott); codec + concealment unseen (repeat-PLC after G.722 / G.726 / GSM)" |
| no-DSP, `agc:*`, `bw`, noise families, codecs | fully unseen |

---

### 2.1 `apm_ns`: WebRTC noise suppression (NS only) between the noise stage and the codec

**Motivation.**
- *Audit evidence (computed).* Within offline sources, NS-treated bona trails untreated bona by 6.3 recall points on
  the full set (69.2 vs 75.5, final), or 17.3 clips; on half B it is 10.8. The whole deficit is `afftdn`:
  - `afftdn` 62.6 vs `anlmdn` 75.9 (full set, final; logit gap 2.01);
  - `anlmdn` bona is *above* the no-NS bona on half B (77.8 vs 76.4).

  NS is the largest stage-attributable deficit of the three (§1.1).
- *Training exposure.* B2f saw platform NS only on clean online clips (Zoom / QQ, CRIT O1.6), never on noisy
  speech, and never a simulated suppressor. The DeepFilterNet `suppress` stage was never enabled (AUD G1).
- *Literature.*
  - TFCL simulates the front end with "the standard implementations provided by WebRTC". Mix training takes
    XLSR-AASIST's full-chain EER from 22.20 to 17.39 (clean 0.23 → 0.52): "The Mix strategy partially alleviates
    the degradation." (https://arxiv.org/html/2607.17761)
  - RTCFake: "built-in speech enhancement modules of communication platforms, such as noise suppression, echo
    cancellation, and automatic gain control" (https://arxiv.org/html/2604.23742v1).
  - WCR: "detectors can often identify that speech has been processed" (https://arxiv.org/abs/2603.14033).
  - PYWRTC (the new backend): "HP filter -> AEC -> NS -> AGC" as separate components, with the standalone NS at
    levels 0-3 = 6 / 12 / 18 / 21 dB and wheels for Python 3.10-3.14 (https://github.com/strands-labs/pywebrtc-audio;
    SOTA §2.6; CRIT §4 citation check).

**Mechanism.**
- With probability `--aug_apm_p` (from `rng4`), the post-stage audio goes through one pass of `pywebrtc_audio`'s
  **standalone `NoiseSuppressor`** (16 kHz, mono). The pass uses **no high-pass filter, no AGC and no AEC** (O1.1).
- The suppression target is drawn uniformly from `--aug_apm_ns_levels`, in dB. 6 / 12 / 18 / 21 map to backend
  levels 0 / 1 / 2 / 3, the same four WebRTC settings kLow … kVeryHigh that `webrtc-noise-gain` calls 1-4.
- The pass is independent of the class and of the acoustic stage, and it also runs on clips without a noise stage,
  because platforms run NS on every call.
- Both classes then carry suppressor artefacts (residual or musical noise, spectral holes, onset attenuation), so
  these stop being a class cue.
- One component: NS. AGC2 and the HPF are left out (§6).

**Flags.**
- `--aug_apm_p` (float, default **0.0**): per-clip probability of the NS pass.
- `--aug_apm_ns_levels` (int list, default `6 12 18 21`, choices 6 / 12 / 18 / 21): suppression targets in dB,
  backend-neutral. Read only when p > 0.

**Default equivalence.** With `--aug_apm_p 0.0`:
- `build_rtc_augmenter_b4` returns the B3 object, and there are no `rng4` draws;
- `pywebrtc_audio` is never imported;
- `--aug_apm_ns_levels` is not read;
- `describe()`, `meta_keys`, the header and the config keys are b3wave's;
- the model is untouched.

**Implementation sketch.**
- New `SLS_setup/b4wave/dsp_apm.py` (~80 LOC):
  - `LEVEL_DB = {6: 0, 12: 1, 18: 2, 21: 3}`.
  - `wrtc_ns(audio, level_db, sr=16000) -> np.float32`:
    1. clip to [−1, 1], convert to int16 and pad to a multiple of 160;
    2. build a **fresh** `NoiseSuppressor(sample_rate=16000, num_channels=1, level=LEVEL_DB[level_db])` per clip;
    3. prime the noise estimate on the first min(0.5 s, len) and discard that output;
    4. run 10 ms frames over the whole clip from the start;
    5. convert back to float32 and trim the pad.
  - The exact frame call of the binding is wrapped in one `_process_frame()` helper, so a binding API detail stays
    local. The import is lazy.
  - `selftest()` **asserts**:
    - the output is finite and the same length;
    - on a noise-only input, the energy drop is monotone in the level;
    - a −30 dBFS 1 kHz tone at 6 dB stays within ±3 dB, so there is no gain stage;
    - **no HPF** (O1.1): on white noise at −30 dBFS, the < 100 Hz band attenuation is within 3 dB of the
      1-4 kHz band attenuation at every level. A WebRTC-style ~80 Hz HPF would add ≥ 6 dB at < 100 Hz.
- `SLS_setup/b4wave/augment.py`:
  - `RTCAugConfigB4` gains `apm_p` and `apm_ns_levels`;
  - in `_post_dsp`, when `apm_p > 0 and rng4.random() < apm_p`: `lvl = levels[rng4.integers(len(levels))]`, then
    `out = finalize(wrtc_ns(out, lvl), out)`;
  - counts: `wrtc_ns{6|12|18|21}`; on an exception the clip passes through and `fail_wrtc_ns` is counted;
  - at start-up: `assert_training_ops(filters=["wrtc_ns"])`, plus `dsp_apm.selftest()` when the lever is on.
- `train.py` header when on: `B4 APM NS: on p=0.3 levels_db=[6, 12, 18, 21] backend=pywebrtc_audio.NoiseSuppressor
  hpf=off agc=off`.
- `checks.py`:
  - **`check_apm`**: the selftest, plus rejection of a level outside {6, 12, 18, 21}.
  - **`apm_balance`** (new, O1.5):
    - **Sample.** 200 bona + 200 spoof training clips, 25 per class × src × lang cell.
    - **Conditions.** (a) no acoustic stage; (b) a forced training-pool noise stage at U(5, 20) dB.
    - **Run.** NS at each of the four levels.
    - **Prints** per class × condition × level: the realised removed energy in dB over the whole clip and over the
      input's low-energy frames (< p95 − 30 dB), as mean and quartiles, plus the class difference of means.
    - **WARN.** |Δ| > 2 dB in condition (b), or > 4 dB in (a), at any level, prints `APM_BALANCE WARN`. That line is
      carried into the implement report and the owner gate.
    - **Why a warning and not a block.** The eval-side suppressors also act on both classes with content-dependent
      realised effects. The kill rules are the run-time guards below.
    - **Measured context** (proxy, no NS installed; 100 clips per class × src). The median floor depth, p95 − p10
      frame energy, is:

      | class | offline | online |
      |---|---|---|
      | bona | 40.8 dB | 44.4 dB |
      | spoof | 35.1 dB | 41.2 dB |

      Spoof floors sit *closer* to speech, so NS will likely remove more from spoof on no-noise clips. That is the
      opposite direction from the one CRIT O1.5 guessed, so both directions are guarded.

**Hypothesis.** B2f reads suppressor artefacts on noisy speech as synthesis. Training both classes through WebRTC
NS on 30 % of clips makes those artefacts class-neutral, and the effect transfers to the unseen `afftdn` / `anlmdn`
suppressors.

**Predicted signature** (`scripts/b4_dsp_slices.py` vs B2f, same readout):
- **Mechanism shown:** on the NS contrast (offline with NS vs without), full set, final:
  - logit DiD ≤ **−0.34** (0.4 × 0.85), with a CI that excludes 0;
  - clip DiD ≥ **+6.9** (0.4 × 17.3);
  - half-B DiD < 0 on both readouts.

  Secondary: the `afftdn` vs `anlmdn` DiD < 0, and ≤ −0.80 (0.4 × 2.01) if strong.
- **Outcome:** heldout F1 up ≥ 0.8 (from 87.92 / 88.27, AUD §3.1). Expected WF1 Δ is **0 to +0.6**, which needs
  spill-over beyond the NS slice (§1.1).
- **Must not move:**
  - clean F1 within 0.5 of 97.60 / 97.69;
  - matched F1 down ≤ 0.5 (91.22 / 91.24);
  - R_spoof within 0.5 on every set;
  - NS spoof misses ≤ 29 (§2.0);
  - ep1 TrainAcc within 2 of 87.36.

**Falsifier.**
- **Mechanism refuted** (NS logit DiD ≥ 0) **and** heldout F1 Δ < +0.5 on both readouts: WebRTC NS exposure does
  not transfer. Drop.
- WF1 Δ < −0.5 **with** matched F1 down > 1.0: losing raw-noise exposure costs more than it buys (noise reaches 60 %
  of clips today, 42 % stay un-suppressed with the lever; AUD G2(c), computed). Drop. One lower-p retry is allowed,
  only if the mechanism was shown.
- NS spoof misses > 29 on either readout: the lever taught "NS ⇒ bona". Drop.
- The shared under-fit falsifier.

**Equivalence test** (`python -m b4wave.checks augmenter`, CPU):
1. Defaults: `build_rtc_augmenter_b4` returns an `RTCAugmenterB3`. 200 synthetic clips without the codec + 10 with
   it are bit-identical to `build_rtc_augmenter_b3`, with identical `stage_counts` and `describe()`.
2. A forced `RTCAugmenterB4` with default fields is bit-identical too.
3. With `apm_p = 1.0` and a stub NS (identity), the parent `stage:*` counts over 200 clips equal the base's. So
   `rng4` leaves the parent stream untouched.
4. `'pywebrtc_audio' not in sys.modules` after the levers-off build.

The supervisor regression adds a re-score of the B2f checkpoint (tol_logit 1e-4) and the levers-off micro-run vs
the b3wave micro-run.

**Cost.** ~190 LOC (`dsp_apm.py` 80, augment glue 30, checks incl. `apm_balance` 80). VRAM 0. Epoch ≤ 1.06
(estimate) with the 1.15 stop rule (§1.2).

**Smoke** (4.5 min):
- the header matches `^B4 APM NS: on p=`;
- Σ `wrtc_ns*` / Σ `codec:*` within [0.25, 0.35], and `fail_wrtc_ns` = 0;
- per-class B4 rate within 0.3 ± 0.12;
- selftest passed, including the no-HPF assert;
- finite losses; rate ≤ 1.3 × S0; peak VRAM ≤ S0 + 0.1 GiB.

**Risks.**
- **Binding API.** The binding's exact frame-call API is known only from the README (fetched by the critic). The
  `_process_frame` helper plus the selftest contain it. If the import name is not `pywebrtc_audio`, the owner
  notes it at the gate.
- **Transfer.** Transfer to `afftdn` is unproven: TFCL trains and tests on WebRTC.
- **Matched exposure.** Matched has raw noise and no NS, so it may lose some exposure benefit (falsifier bound).
- **Class asymmetry.** It is measured by `apm_balance` and guarded by the NS spoof misses and the DiD sign.

---

### 2.2 `dtx_gate`: webrtcvad DTX gating with zero or LPC comfort-noise fill (mechanism unchanged; readout rebuilt)

**Motivation.**
- *Audit evidence.*
  - With no speech evidence, the model calls spoof with certainty: the `sil_only` / `noise_only` bona probes are
    100 % spoof, with mean dP +0.8929 / +0.8985 and +0.8870 / +0.8965 (AUD §3.5).
  - 136 / 133 of the 213 / 210 half-B noisy bona FA sit at p ≥ 0.99 (AUD §3.2). No B2 or B3 lever moved that core
    (AUD G3).
  - **New, computed:**
    - on the full set the saturated core is concentrated in pause-rich source clips: 109 / 107 in the high
      non-speech tertile vs 63 / 66 in the low one (§1.1);
    - offline VAD-gated bona trails ungated offline bona, 66.2 vs 73.6 (full set, final; D 10.1 clips; logit gap
      1.05).
- *Literature.*
  - SIL: "removing the silence of test speech through Voice Activity Detection (VAD) can severely degrade
    performance". Sil-mask training lowers ASVspoof 2021 DF EER for 3 / 3 models (Table VI); LA is mixed
    (https://arxiv.org/html/2309.11827, https://arxiv.org/abs/2309.11827).
  - SSG: "EER increases from 3.6% (with silence) to 15.5% (trimmed silence)" (https://arxiv.org/abs/2106.12914).
  - TFCL: VAD is the last and most damaging front-end stage (https://arxiv.org/html/2607.17761).
  - WVAD: "A frame must be either 10, 20, or 30 ms in duration"; mode 3 is "the most aggressive"
    (https://github.com/wiseman/py-webrtcvad).

**Mechanism** (as v1). With probability `--aug_dtx_p` (from `rng4`), after the NS slot and before the codec:
1. Run `webrtcvad.Vad(mode)` per frame, with mode ∈ {1, 2, 3} and frame ∈ {10, 20, 30} ms drawn per clip.
2. Dilate the speech mask forward by a hangover of U(40, 200) ms.
3. Fill non-speech frames with exact zeros or with order-10 LPC-shaped comfort noise. The CNG level is the gated
   content's RMS × 10^(U(−6, 0)/20), with a −70 dBFS floor. Under `mix` the fill is drawn 50 / 50 per clip.
4. Cross-fade with 5 ms raised-cosine ramps. The length is unchanged.

Both classes are gated the same way. One component: gating. Pause insertion and attenuation fill are left out (§6).
The op name `wrtc_dtx` is off the reserve. This differs from the reserved `sim_ops.vad_gate`, which uses an energy
VAD and a white-noise fill.

**Flags.**
- `--aug_dtx_p` (float, default **0.0**).
- `--aug_dtx_fill` (choice `mix | zero | cng`, default `mix`; read only when p > 0).

**Default equivalence.** As §2.1: the B3 object, no `rng4` draws, `webrtcvad` never imported, the fill flag unread,
and the model untouched.

**Implementation sketch** (as v1).
- `SLS_setup/b4wave/dsp_dtx.py` (~110 LOC):
  - `wrtc_dtx(audio, rng4, sr, fill) -> (out, gated_frac, fill_used)`. The LPC autocorrelation is computed for lags
    0-10 only (dot products; measured 1.2 ms per clip), then `solve_toeplitz` + `lfilter`;
  - `selftest()`.
- `augment.py`: `dtx_p` and `dtx_fill`. Counts are `wrtc_dtx_{zero|cng}`, the gated-fraction bins
  `dtx_g0 / dtx_g1_20 / dtx_g20_40 / dtx_g40p`, and `fail_wrtc_dtx`. `assert_training_ops(filters=["wrtc_dtx"])`.
- Header: `B4 DTX gate: on p=0.3 fill=mix modes=[1, 2, 3] frames_ms=[10, 20, 30] hangover_ms=40..200`.
- `checks.py`:
  - `check_dtx`: the selftest;
  - `dtx_balance`: 200 + 200 training clips, realised gated fraction per class (mean and quartiles). It prints
    `DTX_BALANCE WARN` if the class means differ by > 0.05.

**Hypothesis.** Part of the saturated core is "gated or non-speech content ⇒ spoof". Class-independent DTX gating
on 30 % of clips makes gated pauses class-neutral, so VAD-gated bona and the pause-rich saturated FA shrink.

**Predicted signature.**
- **Mechanism shown, both of:**
  - (i) on the VAD contrast, full set, final: logit DiD ≤ **−0.42** (0.4 × 1.05) with a CI that excludes 0, clip
    DiD ≥ **+4.0** (0.4 × 10.1), and half-B DiD < 0 on both readouts;
  - (ii) the saturated-core drop concentrates in the high non-speech tertile: drop_high − drop_low ≥ **5** clips
    (full set, final; baseline 107 vs 66), and ≥ 0 on half B for both readouts.
- **Outcome** (direction only): the `sil_only` bona mean dP falls below +0.8929 / +0.8985. Expected WF1 Δ is
  **0 to +0.5**.
- **Must not move:**
  - clean F1 within 0.5;
  - noisy spoof misses ≤ 36 / 35;
  - VAD spoof misses ≤ 25 / 22 (§2.0);
  - `probe_padded` bona flip up by no more than 2 clips (1.07 points) from 3.74 / 5.35 %;
  - R_spoof within 0.5.

**Falsifier.**
- **Mechanism refuted on both readouts' primary** (VAD logit DiD ≥ 0 **and** drop_high ≤ drop_low, full set,
  final) with WF1 Δ < +0.5: gating is not what drives the core. Drop.
- Clean F1 down > 0.5, or a spoof-miss guard broken: the gating pattern carries class information. Drop.
- The shared under-fit falsifier.

**Equivalence test.** Items 1, 2 and 4 of §2.1 (with `webrtcvad` not imported), plus item 3 with `dtx_p = 1.0`.
`check_dtx` asserts that the length is unchanged and that zero-filled frames are exactly 0.

**Cost.** ~210 LOC. VRAM 0. Epoch ≤ 1.02 (§1.2).

**Smoke** (4.5 min):
- header `^B4 DTX gate: on p=`;
- Σ `wrtc_dtx_*` / Σ `codec:*` within [0.25, 0.35], and `fail_wrtc_dtx` = 0;
- at least one non-`dtx_g0` bin > 0;
- both classes in the gated bins;
- finite losses; rate ≤ 1.3 × S0.

**Risks.**
- **Class-dependent gating.** The v1 proxy measured the share of frames more than 30 dB below p95 at bona 0.205 /
  0.165 vs spoof 0.153 / 0.155 (offline / online). It is guarded by `dtx_balance`, the per-class telemetry and the
  spoof-miss guards.
- **Mixed literature.** Sil-mask is mixed on LA19.
- **Gating speech.** Modes 2-3 gate some low-SNR speech; this is counted in `dtx_g40p`.
- **Readout limits.** Half B alone cannot read the tertile pattern (it is flat there, §1.1), which is why the full
  set is primary.
- **Dependency.** Needs `webrtcvad-wheels`.

---

### 2.3 `opus_plc`: in-process libopus with bursty packet loss and decoder PLC (decoder parity, ADD-C range, G1 only)

**Motivation.**
- *Audit evidence (computed).*
  - Online PLC-treated bona trails online untreated bona, 67.5 vs 75.6 (full set, final; D 9.2 clips; logit gap
    0.95).
  - The offline PLC contrast is small (gap 0.37) and below resolution.
  - B2f's codec never drops a packet. B3d's `-packet_loss` was only the encoder's hint, and ffmpeg 4.2 has no FEC /
    DTX options (SOTA §2.3).
  - "Packet loss + PLC" is on the never-run list (AUD §2).
- *Literature.*
  - ADD-C degrades six codecs at **packet-loss rates 0 / 1 / 5 / 10 / 20 %**. Trained on clean data, its models
    degrade by 5.30 (baselines) and 2.33 (theirs) EER. Trained on codec + packet loss: "The EER remains unchanged,
    with a slight improvement of 0.003% from C0 to C1" (https://arxiv.org/html/2504.12423v2).
  - RTCFake lists "codec encoding and decoding, and packet loss" among the "key distortion sources"
    (https://arxiv.org/html/2604.23742v1).
  - UR21 (https://arxiv.org/abs/2107.12018).
  - OPUSLIB is the `ctypes` pattern followed here, without the dependency (https://github.com/orion-labs/opuslib,
    https://github.com/orion-labs/opuslib/blob/master/opuslib/classes.py).

**Measured in this session.**
- `libopus 1.3.1` loads through `ctypes.util.find_library("opus")`. The NULL-payload PLC call returns a full frame
  (v1 measurement, re-run here).
- **Decoder parity (O1.9, alternative I3).** ffmpeg decodes with its native Opus decoder at 48 kHz and resamples to
  16 kHz (`utils/rtc_augment.py` l.196). Mirroring that, in-process 48 kHz decode + `scipy.signal.resample_poly(·,
  1, 3)` is closer to the ffmpeg path than the v1 16 kHz decode. Zero loss, 3 training clips × 6 s, each profile at
  its lowest bitrate, first cutoff and longest frame, aligned SNR in dB:

  | profile | SNR(ffmpeg, in-16k) | **SNR(ffmpeg, in-48k)** | SNR(ffmpeg, input) | SNR(in-16k, input) | SNR(in-48k, input) |
  |---|---|---|---|---|---|
  | qq 10 kbps, 6000, 20 ms, voip | 21.6 | 21.6 | 9.9 | 9.4 | 9.4 |
  | wechat 8, 4000, 20, voip | 17.4 | 17.0 | 9.2 | 8.8 | 8.8 |
  | zoom 24, 8000, 20, voip | 20.4 | **26.9** | 12.4 | 11.9 | 11.9 |
  | dingtalk 16, 8000, 40, voip | 20.4 | **26.3** | 11.7 | 11.4 | 11.2 |
  | lark 24, 8000, 20, voip | 20.4 | **26.9** | 12.4 | 11.9 | 11.9 |
  | voov 16, 6000, 20, voip | 22.8 | 22.8 | 11.5 | 10.8 | 10.8 |
  | telegram 16, 8000, 60, audio | 20.3 | **26.0** | 15.5 | 17.3 | **14.9** |

  - On the wide-band profiles the 48 kHz path gains about 6 dB of agreement.
  - On telegram it also stops being "cleaner than ffmpeg" (14.9 vs 15.5 to the input, against 17.3 at 16 kHz).
  - The narrow / medium-band profiles (SILK layer) are unchanged at 17.0-22.8 dB, the residual of ffmpeg's native
    decoder vs libopus.
  - In every row the two codec paths are 7.8-14.5 dB closer to each other than ffmpeg is to the input.
- **Cost** (§1.2): 0.109 vs 0.258 CPU-s per clip, so the lever makes the loader cheaper.

**Mechanism.**
- The codec stage fires on the parent's draw, as in B2f.
- For a clip the lever picks (`rng4.random() < p`), `draw_codec_params(rng)` makes the parent's draws in exactly
  `codecs_augm`'s order (`utils/rtc_augment.py` l.186-193).
- A `ctypes` libopus 1.3.1 encoder (16 kHz mono) is configured with the *same* profile:
  - the bitrate;
  - the max bandwidth from the cutoff: 4000 / 6000 / 8000 → NB / MB / WB;
  - the frame;
  - the application;
  - VBR on, complexity 10.
- A Gilbert-Elliott mask drops packets: L ~ U(`--aug_opus_loss`), B ~ U(`--aug_opus_burst`), p_bg = 1/B,
  p_gb = L·p_bg/(1 − L), start state drawn from the stationary distribution.
- A **48 kHz** decoder decodes received packets and conceals lost ones with the NULL-payload PLC call.
- The output is trimmed by 3 × the 16 kHz look-ahead, resampled 48 → 16 kHz with `resample_poly`, and padded or cut
  to the input length.
- Encoder and decoder are created per clip and **destroyed in `finally`** (O1.9).
- Unpicked clips call the original `codecs_augm`.
- One component: real packet loss with codec-native concealment. FEC is left out.

**Loss range** (O1.4). The default `0.01 0.20` is the span of ADD-C's non-zero PLR grid (1, 5, 10, 20 %). No
eval-set statistic is used. The overall share of training packets lost is 0.3 × 0.105 ≈ 3 % (computed).

The burst range U(1, 4) packets (20-240 ms gaps at 20-60 ms frames) is a **design choice**: Gilbert-Elliott is the
standard two-state burst model, and no verified source gives RTC burst statistics. It overlaps the reserved
`sim_ops.packet_loss` GE family (p_bg U(0.3, 0.7), i.e. mean burst 1.4-3.3; `utils/sim_ops.py` l.122). Hence the
G6 label in §2.0: "loss-pattern family seen, codec + concealment unseen". A heldout `plc` gain is therefore **not**
read as transfer to an unseen loss pattern.

**Flags.**
- `--aug_opus_plc_p` (float, default **0.0**).
- `--aug_opus_loss` (2 floats, default `0.01 0.20`; 0 ≤ lo ≤ hi < 0.5; `0 0` = the in-process codec without loss,
  the attribution control).
- `--aug_opus_burst` (2 floats, default `1.0 4.0`; 1 ≤ lo ≤ hi).

**Default equivalence.** As v1:
- the B3 object, so every codec call is the original `codecs_augm` with the same parent draws;
- libopus is never loaded;
- the loss and burst flags are unread;
- the model is untouched.

**Implementation sketch.**
- `SLS_setup/b4wave/opus_plc.py` (~160 LOC):
  - `_lib()`: lazy `ctypes` binding with argtypes / restype, loading `find_library("opus")`, then
    `{sys.prefix}/lib/libopus.so.0`;
  - `draw_codec_params(rng, apps)`;
  - `ffmpeg_opus(...)`, the fallback on an exception, called with the pre-drawn values;
  - `opus_loss_roundtrip(audio, rng4, kbps, cutoff, frame, application, loss_range, burst_range, sr, dec_sr=48000)
    -> (out, realised_loss)`, with `try / finally` destroy;
  - `selftest()`: library loads, ctl set/get round trip (bitrate, max bandwidth), a concealed gap is not all-zero,
    finite output of the right length.
- `augment.py`, codec branch of `RTCAugmenterB4.__call__`, as v1:
  - counts `codec:{app}`, `opus_plc`, `plc_l0_5 / plc_l5_10 / plc_l10p` and `fail_opus_plc`;
  - init raises if `opus_plc_p > 0` together with `--aug_codec_profiles random` or with the codec off;
  - `assert_training_ops(codecs=["libopus"], filters=["opus_plc"])`.
- Header: `B4 Opus PLC: on p=0.3 loss=0.01..0.20 burst=1..4 decode=48k+resample_poly libopus=libopus 1.3.1`.
- `checks.py::check_opus_plc`: the selftest, `codec_draw_order`, fallback bit-identity and the zero-loss
  attribution check (below).

**Hypothesis.** Concealment artefacts are read as synthesis. Exposing both classes to libopus PLC under bursty loss
on 30 % of codec calls makes concealment class-neutral, and the gain shows on heldout's repeat-PLC after non-Opus
codecs.

**Predicted signature.**
- **Mechanism shown:** on the online PLC contrast, full set, final:
  - logit DiD ≤ **−0.38** (0.4 × 0.95) with a CI that excludes 0;
  - clip DiD ≥ **+3.7** (0.4 × 9.2);
  - half-B DiD < 0 on both readouts.

  The offline PLC contrast is reported only.
- **Outcome:** heldout F1 up ≥ 0.5. Expected WF1 Δ is **0 to +0.4**.
- **Must not move** (G2 is no longer a target, O1.3):
  - matched F1 down ≤ 0.5;
  - matched qq and wechat bona recall each down by no more than 6 points. Merge vs final of B2f already differ by
    5.05 (qq 61.62 / 66.67) and 5.77 (wechat 65.38 / 71.15) points (AUD §3.4);
  - PLC spoof misses ≤ 26 / 25;
  - clean F1 within 0.5;
  - ep1 TrainAcc within 2.

**Falsifier.**
- Online PLC logit DiD ≥ 0 **and** heldout F1 Δ < +0.5: libopus PLC does not carry over. Drop.
- WF1 Δ < −0.5 with the shared under-fit signature: a B3d-type codec-side under-fit. Drop.
- **Attribution control (pre-registered):** if the arm is positive, `--aug_opus_plc_p 0.3 --aug_opus_loss 0 0`
  runs before promotion. If it alone gives ≥ half of the arm's WF1 gain, the gain is credited to the decoder path,
  not to PLC.

**Equivalence test.**
- Items 1, 2 and 4 of §2.1 (no `ctypes` handle).
- Item 3 with `opus_plc_p = 1.0` and the codec on: the parent `stage:*` and `codec:<app>` counts equal the base.
- `codec_draw_order`: `draw_codec_params` leaves `rng.bit_generator.state` where `codecs_augm` leaves it.
- `ffmpeg_opus` on pre-drawn values is bit-identical to `codecs_augm`.
- **Zero-loss attribution:** on 3 training clips × 7 profiles, aligned SNR(ffmpeg, in-process) ≥
  max(15 dB, SNR(ffmpeg, input) + 5 dB) on every profile. Measured minimum margin: wechat, 17.0 vs 15.

**Cost.** ~280 LOC: binding 50, round trip + GE + 48 kHz decode / resample 110, draw / fallback 40, checks 80.
VRAM 0. Projected epoch factor 0.85 under a saturated loader; 1.0 declared.

**Smoke** (4.5 min):
- header `^B4 Opus PLC: on p=`;
- `opus_plc` / Σ `codec:*` within [0.25, 0.35], and `fail_opus_plc` = 0;
- every `plc_l*` bin > 0;
- 7 apps covered;
- the selftest passed;
- finite losses; rate ≤ 1.3 × S0.

**Risks.**
- **Restart wipe.** A container restart wipes the system ffmpeg and probably libopus. The selftest raises at
  start-up instead of passing audio through. Optional hardening: conda-forge `libopus`.
- **Variadic `ctl` through `ctypes`.** Guarded by the set/get round trip.
- **Transfer.** Transfer to repeat-PLC after G.7xx / GSM is unproven; ADD-C is in-distribution.
- **Decoder residual.** The SILK-layer residual of 17-22 dB vs ffmpeg's native decoder remains. It is bounded by the
  zero-loss check and the control arm.
- **Long frames.** 40 / 60 ms frames conceal more audio per lost packet.

## 3. Shared changes

| what | files |
|---|---|
| **Config and augmenter classes.**<br>• `RTCAugConfigB4(RTCAugConfigB3)` with the seven fields; `RTCAugmenterB4(RTCAugmenterB3)` whose `__call__` mirrors the parent: `len < 16` return → `rng = _get_rng()` → `_stage` → `_post_dsp` (`wrtc_ns` → `wrtc_dtx`) → codec branch (`codecs_augm`, `codecs_augm_random`, or the PLC split).<br>• `rng4 = default_rng([seed, pid, 0xB4])`, re-created on a pid change.<br>• B4 `meta_keys` / `describe()` suffixes only for active levers.<br>• Start-up selftests, `fail_*` counters, `assert_training_ops` for the new op names.<br>• `build_rtc_augmenter_b4` returns `build_rtc_augmenter_b3(args)` when every B4 flag is at default. | `SLS_setup/b4wave/augment.py` |
| `build_augmenter_b4` / `build_dataset_b4`, copy-adapted from b3wave. Dev uses `SpoofAudioDataset`; train reuses `SpoofAudioDatasetB3` (the meta vector). | `SLS_setup/b4wave/data.py` |
| **Trainer** (scaffolded copy of `b3wave/train.py`):<br>• a "B4 wave" argparse group with value checks: p in [0, 1]; levels ⊂ {6, 12, 18, 21}; loss 0 ≤ lo ≤ hi < 0.5; burst 1 ≤ lo ≤ hi;<br>• the `b4wave: <version>` line;<br>• lever headers only when on;<br>• `config.yaml` gains `b4wave_version` and `b4_levers`;<br>• B4 keys in the existing `Epoch N B3 aug` line;<br>• per-label meta sums → `Epoch N B4 aug-by-class` and `metrics.jsonl["b4"]`. | `SLS_setup/b4wave/train.py` |
| **Checks.**<br>• `cpu`: `augmenter` (items 1-4 for every lever), `codec_draw_order`, `reserve_names`, `apm` (incl. the no-HPF assert), `apm_balance`, `dtx`, `dtx_balance`, `opus_plc` (incl. zero-loss attribution).<br>• **`timing`** (O1.7): single process, OMP = 1, 40 random training clips. CPU-s per clip (`process_time` + `RUSAGE_CHILDREN`) for load + loudness, the B2f augmenter (stage + codec), featurize, and each B4 stage. It prints the projected epoch factor (0.350-style base + p · Δ) / base per lever under the saturated-loader assumption, and fails above 1.3 (above 1.15 for `apm_ns`: report to the owner).<br>• `dsp_slices`: the script below must reproduce the §1.1 B2f tables on RUN.<br>• `model --ckpt` / `optim` as in b3wave.<br>• In `all`, a lever check whose package is missing prints `SKIP`. | `SLS_setup/b4wave/checks.py` |
| Flag table, reproduction statement, telemetry keys, install lines, G6 labels. | `SLS_setup/b4wave/README.md` |
| **Mechanism readout (rebuilt, O1.2).**<br>`scripts/b4_dsp_slices.py --run <local_eval/readout> --ref <B2f same readout> [--ref_pair <B2f merge> <B2f final>] [--halves AB\|B] [--json_out f]`. For every contrast of §1.1 it prints:<br>• n_T / n_C, recall T / C (run, ref), gap G and D;<br>• the clipped-logit DiD with a 2,000× bootstrap CI;<br>• the clip DiD;<br>• T fixed / broken with an exact sign-test p;<br>• the verdict (shown / inconclusive / refuted, §1.1 thresholds);<br>• the within-run noise row from `--ref_pair`.<br>It also prints exclusive-stage counts, the saturated FA by non-speech tertile (cuts 0.159 / 0.249; source audio from the dev wav dir, energy rule of §1.1), the §2.0 spoof-miss guards, and the probe rows (`probe_report.json`). It reads only `scores/*.{txt,logits}`, `probe_report.json`, the sim `meta.csv` and the dev source wavs. No GPU. | `scripts/b4_dsp_slices.py` |

No file outside `SLS_setup/b4wave/` and `scripts/b4_*` changes. `b3wave`, `utils`, `model`, `main_*.py` and the old
scripts stay byte-identical.

## 4. Eval impact

- No state_dict change, no new eval profile. Checkpoints are plain `ModelSLS` state_dicts, scored with
  `sls_local17` and `--loudness_norm --loudness_norm_lufs -23`, as for B2f.
- The mechanism readout is `scripts/b4_dsp_slices.py`. The analyst runs it on both readouts with `--ref` = B2f's
  same readout and `--ref_pair` = B2f merge / final. The primary contrast is read on the **full set, final
  readout**; half B is the secondary sign check. The WF1 rule stays on half B.
- G6 labels as in §2.0. `sim_unseen_v1` stays unread (music leak, AUD G6).
- Wave rules as in B3 (`exploop/B3/wave.json` `rules`): band ±0.5; promote = CI low > 0 and clean drop ≤ 0.5.
  The reference is B2f at 91.98 / 92.14 (AUD §1).
- Run length: 8 × ~31 min × ≤ 1.06 ≈ 4.4 h per arm (computed), under the 10 h cap.

## 5. After the wave: a recommendation, not a pre-registered arm (O1.8)

- **Composition.** The three stages sit at disjoint points of the chain and draw from `rng4` only when on, so any
  combination is well defined (NS → DTX → codec / PLC).
- **One-factor rule.** Every wave arm is one lever against B2f.
- **Recommendation to the owner / exp-designer.** Suppose ≥ 2 levers reach "mechanism shown" with WF1 null or
  better. Then a combined "platform chain" run (the TFCL Mix analogue) is the natural next step, because promotion
  needs about 14 fewer FA (§1.1), more than any single slice gives. It should be scheduled as a coupled group or a
  separate wave, by decision of the owner or exp-designer. It is not part of this plan's pre-registration.
- **No dose sweeps.** A lever whose falsifier fires is dropped without a dose sweep, except for:
  - the `apm_ns` lower-p retry (only if its mechanism was shown);
  - the `opus_plc` zero-loss control (only if it is positive).

## 6. Ideas considered and not proposed

| idea | decision | why |
|---|---|---|
| AGC2 inside `apm_ns` (I1 as carded) | left out | One component per lever. Level treatment is "not a lever" (B2f vs B2e null, AUD §2 row 9). |
| WebRTC HPF (implicit in `webrtc-noise-gain`) | left out (O1.1) | A second component that can carry class information (low-band rumble). The backend switch removes it. |
| Pause insertion / attenuation fill in `dtx_gate` | left out | Pause insertion is a second component (rhythm). An attenuation gate is the `agate` family, which is reserved for `sim_unseen_v1`. |
| libopus DTX as the gate | declined (v1, measured) | It fired on 7.1 % of bona frames vs 1.2 % of spoof frames: too little, and class-skewed. |
| In-band FEC in `opus_plc` | left out | A second decoder component; a follow-up only if PLC shows its mechanism. |
| `opuslib` dependency | replaced | The system libopus 1.3.1 works through `ctypes` (measured). |
| I4 AugMix single-view mixing | declined | It raises the effective SNR by ≈ 8.7 dB on top of the composition change (computed in v1): two components, one on the exhausted SNR axis. The evidence is vision-only. |
| I5 echo suite | declined | Multi-tap self-echo *is* the `sim_echo_v1` recipe (`build_dev_noisy_sim.py` l.205-209). Echo is not in WF1, and there is no ADD paper. |
| I6 DeepAugment DeepFilterNet | declined | Vision-only evidence; `deepfilternet` is not installed; CPU risk under a loader already at ~90 % of quota (§1.2). |

## 7. Dependencies (owner action at the gate; this step installed nothing)

Import check in `rtc-sdd` (this session, `importlib.util.find_spec`): `webrtc_noise_gain`, `webrtcvad`,
`pywebrtc_audio` and `opuslib` are all absent. Python is 3.12.14. The supervisor tests `import <name>`, so the
entries are import names:
- `apm_ns` → `pywebrtc_audio`;
- `dtx_gate` → `webrtcvad`;
- `opus_plc` → none.

```
/root/.conda/envs/rtc-sdd/bin/python -m pip install pywebrtc-audio webrtcvad-wheels
/root/.conda/envs/rtc-sdd/bin/python -c "import pywebrtc_audio, webrtcvad; print('ok')"
# if the pywebrtc-audio import name differs, leave an owner note with the name (apm_ns dependency)
# optional hardening against container restarts (system ffmpeg + libopus get wiped):
# conda install -n rtc-sdd -c conda-forge libopus
```

These are all local DSP or software codecs: "local WebRTC AEC / NS / AGC / VAD simulation explicitly allowed;
software codecs allowed" (roadmap §2, AUD §5). `webrtc-noise-gain` is no longer needed.

## 8. Responses to the critic

| id | action | response |
|---|---|---|
| **O1.1** (major) | **accept** | **Backend.** It is now `pywebrtc_audio`'s standalone `NoiseSuppressor`: NS only, with no HPF, AGC or AEC (§2.1).<br>**Levels.** `--aug_apm_ns_levels` is backend-neutral, in suppression dB (6 / 12 / 18 / 21 = backend levels 0-3 = WNG 1-4).<br>**Dependency.** The entry is now `pywebrtc_audio` (§7).<br>**Check.** `check_apm` now *asserts* "no HPF": the < 100 Hz attenuation on white noise is within 3 dB of the 1-4 kHz attenuation at every level.<br>**Fallback.** `webrtc-noise-gain` is dropped, so no HPF control arm is needed. |
| **O1.2** (major) | **accept** (all five parts, with two additions) | (1) **Same-source controls**, as required: NS = offline with NS vs offline without, plus `afftdn` vs `anlmdn`; VAD = offline with VAD vs without; PLC = within-source, online primary.<br>(2) **Paired counts.** Fixed / broken with an exact sign test, and the B2f merge-vs-final row as the within-run noise reference. *Addition 1:* the B2e-vs-B2f final row as a cross-run reference.<br>(3) **Thresholds** are 40 % of the stage's baseline gap (logit) and deficit (clips). Falsifier: DiD ≥ 0 = refuted; in between = inconclusive.<br>(4) **Exclusive-stage counts** are printed.<br>(5) **Tertile readout.** Saturated FA by non-speech tertile, predicting drop_high − drop_low ≥ 5 (§2.2).<br>*Addition 2:* the primary statistic is the clipped-logit DiD on the **full set, final readout**. Clip counts are threshold-noisy: merge vs final differ by +12.8 clips but −0.22 ± 0.19 logit on the NS contrast. The final epoch is unselected, and the full set doubles n. Half B remains the sign check and the WF1 decision set.<br>*Modification of (5):* the non-speech fraction comes from the clean source clip's energy, not webrtcvad on the sim audio. The noise fills the sim pauses, grading with the training VAD would be circular, and it needs no dependency. With this measure the full-set baseline already shows the concentration (107 vs 66 saturated FA, final). |
| **O1.3** (minor) | **accept** | G2 is removed from `opus_plc`'s `gap_ids`. The matched qq / wechat rows are now a must-not-move guard. The ±3-point band is widened to 6 points, because B2f merge vs final already differ by 5.05 / 5.77 points (AUD §3.4). |
| **O1.4** (minor) | **accept** | **Range.** The loss range is now ADD-C's non-zero PLR span, `0.01 0.20`. The heldout-mean argument is deleted.<br>**Burst.** U(1, 4) is stated as a design choice.<br>**Label.** Heldout `plc` is labelled "loss-pattern family seen (GE); codec + concealment unseen", and a `plc` gain is not read as loss-pattern transfer (§2.0, §2.3). |
| **O1.5** (minor) | **accept** | **Check.** `checks apm_balance` (§2.1): 200 + 200 training clips, with and without a noise stage, realised removed dB per class and level, with `APM_BALANCE WARN` at a class gap > 2 dB (noise) / > 4 dB (no noise).<br>**Guards** (kill rules): NS spoof misses ≤ 29 (Poisson bound on 18 / 18), and the NS DiD sign.<br>**Measured proxy.** The asymmetry likely runs the *other* way: spoof floor depth 35.1 / 41.2 dB vs bona 40.8 / 44.4 dB (median p95 − p10, offline / online), so NS has more to remove from spoof. Hence both directions are guarded. |
| **O1.6** (minor) | **accept** | Rephrased everywhere: "never noisy speech through a suppressor, and no simulated DSP" (§0, §2.1). This also explains why NS is a near no-op on the clean online training clips. |
| **O1.7** (minor) | **accept** | **Measured** (§1.2): the B2f path is 0.350 CPU-s per clip (ffmpeg children 0.306). At 40.5-41.1 clips/s that is 14.2-14.4 of 16 CPUs, so the loader is treated as saturated.<br>**Projections:** `opus_plc` 0.85 (in-process 0.109 vs ffmpeg 0.258 CPU-s, paired); `dtx_gate` ≤ 1.01 (CNG 0.0012 CPU-s measured); `apm_ns` estimated 1.02-1.06, with a 1.15 stop rule.<br>**`checks timing`** now reports CPU-s per clip for the full path and per stage, plus the saturated-loader factor. The smoke rate stays only as a coarse check. |
| **O1.8** (minor) | **accept** | §5 now frames the combined run as a recommendation to the owner / exp-designer (a coupled group or a separate wave). It is not a pre-registered arm. |
| **O1.9** (minor) | **accept** | **Decode.** The in-process path decodes at 48 kHz and resamples with `resample_poly`, like ffmpeg. Measured agreement with ffmpeg: 26.0-26.9 dB on the WB profiles vs 20.3-20.4 at 16 kHz; unchanged 17.0-22.8 on the SILK-layer profiles. Telegram's SNR to the input drops from 17.3 to 14.9 (ffmpeg 15.5).<br>**Check.** The zero-loss check is kept and tightened to ≥ max(15 dB, SNR(ffmpeg, input) + 5 dB).<br>**Cleanup.** Encoder and decoder are destroyed in `finally`. |

**Alternatives.**

| critic alternative | action | detail |
|---|---|---|
| I1: `pywebrtc-audio` standalone NoiseSuppressor | **adopt** | It is the `apm_ns` backend (O1.1), at no extra cost. |
| I2: saturated FA by the eval clip's non-speech fraction | **adopt (modified)** | Measured on the clean source clip's scored window with an energy rule, not webrtcvad on the sim audio (reasons in O1.2 (5)). Cut points are fixed at 0.159 / 0.249. Baseline: full set 63 / 73 / 109 (merge) and 66 / 76 / 107 (final). Pre-registered drop_high − drop_low ≥ 5. |
| I3: 48 kHz decode + resample | **adopt** | Measured closer to ffmpeg on 4 / 7 profiles and equal on the rest (O1.9 table). |

## 9. Changes since v1

1. **`apm_ns` backend.** `webrtc_noise_gain` (NS + forced HPF) → `pywebrtc_audio.NoiseSuppressor` (NS only). The
   levels flag is now in dB, `check_apm` asserts no HPF, `apm_balance` is new, and the dependency name changed.
2. **Mechanism readout rebuilt** (§1.1):
   - same-source contrasts;
   - clipped-logit DiD with a bootstrap CI;
   - full set / final readout as primary;
   - within-run and cross-run noise rows;
   - 40 %-of-gap thresholds with a three-way verdict;
   - exclusive-stage counts;
   - non-speech tertiles.

   Every lever's signature and falsifier are restated on it.
3. **Spoof-miss guards** with Poisson 99 % bounds for each lever's treated slice (§2.0).
4. **`opus_plc`.**
   - 48 kHz decode + `resample_poly`;
   - loss range `0.01 0.20` (ADD-C);
   - G2 removed;
   - matched codec guard widened to 6 points;
   - `finally` destroy;
   - tighter zero-loss check.
5. **Cost measured in CPU-seconds** (§1.2). The loader is near its quota; `opus_plc` is projected cheaper than B2f
   (0.85); `apm_ns` gets a 1.15 stop rule. LOC: `apm_ns` 160 → 190, `dtx_gate` 200 → 210, `opus_plc` 260 → 280.
6. **Wording.** "Never noisy speech through a suppressor, and no simulated DSP" (O1.6). The combined arm is now a
   recommendation (O1.8). The expected WF1 ranges are restated honestly: no lever can clear the ±0.5 band through
   its own slice alone.
