# RTC-SDD: Experiments Roadmap for Closing the Noisy Gap

> Status: living document · Created 2026-10-03 · Owner: Ghostbusters team
> Challenge: **RTC-SDD (ICASSP 2027 Grand Challenge)**, dataset **RTCFake** ([arXiv 2604.23742](https://arxiv.org/abs/2604.23742))
> Hard deadlines: final eval data **2026-11-09**, submission **2026-11-16 (AoE)**.

---

## Table of contents

0. [TL;DR](#0-tldr)
1. [Diagnosis: why noisy is ~11 F1 behind clean](#1-diagnosis)
2. [Rule-compliance matrix](#2-rule-compliance-matrix)
3. [Phase 0 – Evaluation infrastructure (do first)](#3-phase-0--evaluation-infrastructure)
4. [Phase 1 – Quick wins](#4-phase-1--quick-wins)
5. [Phase 2 – Augmentation v2: RTC-faithful, domain-randomised simulator](#5-phase-2--augmentation-v2)
6. [Phase 3 – Training strategies](#6-phase-3--training-strategies)
7. [Phase 4 – Architecture & backbones](#7-phase-4--architecture--backbones)
8. [Phase 5 – Moonshots & cross-domain ideas](#8-phase-5--moonshots--cross-domain-ideas)
9. [Experiment catalogue (master table)](#9-experiment-catalogue)
10. [Methodology & reporting](#10-methodology--reporting)
11. [Timeline to 2026-11-16](#11-timeline)
12. [Recommended final stack (best current guess)](#12-recommended-final-stack)
13. [Implementation appendix](#13-implementation-appendix)
14. [References](#14-references)

---

## 0. TL;DR

- **The metric is 0.3·MacroF1(clean) + 0.7·MacroF1(noisy), both at a fixed 0.5 threshold on P(spoof).**
  - One noisy F1 point is worth 2.3 clean points.
  - At ~90 clean and ~79 noisy (83.2 weighted), closing **half** the noisy gap is worth about +3.9 weighted points. Clean has much less room.
- **The noisy subset is out-of-distribution by construction.**
  - Noise (office/coffee from RNNoise data, echo from CLAD, rain/footsteps/keyboard from ESC-50) is added **before** transmission.
  - Each platform's own NS / AEC / AGC / VAD / codec / PLC / BWE then processes speech and noise together.
  - Train has **no noisy audio**, and only 2 of the 7 platforms (Zoom, QQ). Echo is the hardest condition in the paper.
- **Five hypotheses for the gap.** Each one maps to a cheap test (see §1):
  - **H1 Calibration / prior collapse.** The 0.1/0.9 CE weights train the model as if spoof prevalence were ~32%; eval is ~83% spoof. Under noise, posteriors shrink toward that wrong prior, and noisy spoofs fall below 0.5.
  - **H2 Simulator mismatch.** Our augmentation has noise + Opus, but no NS/AEC/AGC/VAD/DTX/PLC/EQ, and never chains stages. The real pipeline is noise → processing → codec.
  - **H3 Fine-tuning erodes native robustness.** Qwen3-ASR was trained on huge noisy ASR data; full fine-tuning at a constant LR on clean-ish data overwrites that.
  - **H4 Head design.** A fixed 4.04 s first-crop + flatten-FC head is position-sensitive, length-locked, and over-parameterised (~5.8M params in `fc1`).
  - **H5 Unseen platforms / generators.** Needs diversity (domain randomisation) more than fidelity.
- **Order of attack:**
  1. **Build a local noisy proxy dev set and select by local WF1** (otherwise we're flying blind and/or tuning on progress, which the rules forbid).
  2. **Fix calibration** (hours).
  3. **RTC-faithful augmentation with WebRTC APM** (days).
  4. **Paired consistency learning** (the authors' own PCL/TFCL show the biggest documented gains).
  5. **LoRA / WiSE-FT / better optimisation.**
  6. **New pooling head + w2v-BERT 2.0 / WavLM backbones.**
  7. Moonshots if time allows.

---

## 1. Diagnosis

### 1.1 What we know

| Fact | Source | Implication |
|---|---|---|
| Score = 0.3·F1_clean + 0.7·F1_noisy, Macro-F1 at threshold 0.5 | Challenge site | Calibration matters as much as separability |
| Eval = online only; noisy = {office, coffee, echo, rain, footsteps, keyboard} before transmission | Paper §3, challenge site ("noisy subset further expanded") | We must simulate "noise → platform DSP → codec" |
| Train/dev platforms: P01 Zoom, P02 QQ (+P03 WeChat in dev); eval: P01–P07 | Paper Table 6 | 5 unseen platforms → need diversity |
| Unseen generators in eval: IndexTTS2, Doubao, SparkTTS, ChatterboxVC | Paper Table 6 | Some clean-subset errors too |
| Echo (S04) is the hardest condition; QQ / Telegram / Lark / VooV are the hardest platforms | Paper Tables 2–3 | Prioritise echo + AEC-residual augmentation |
| PCL (offline↔online consistency) lowers EER 7.33 → 5.81, with the best result on every noise type | Paper Tables 2–4 | Paired consistency is the strongest documented lever |
| TFCL: a WebRTC AEC→NS→AGC→VAD chain takes XLSR-AASIST from 0.23% to 22.2% EER; with TFCL, 9.78%. VAD is the most damaging stage | [2607.17761](https://arxiv.org/abs/2607.17761) | Simulate APM + VAD; use soft temporal alignment in the consistency loss |
| Official baseline (XLS-R + AASIST): 91.47 clean / 64.36 noisy / 72.50 | Baseline repo | We're +15 noisy already; remaining gains come from finer things |

### 1.2 What the code tells us (`SLS_setup/`)

| Observation | Location | Consequence |
|---|---|---|
| CE weights `[0.1 spoof, 0.9 bonafide]` | `main_train.py` `--ce_weights` | Effective training prior for spoof ≈ 0.1·61432 / (0.1·61432 + 0.9·14353) ≈ **0.32**. Eval prior ≈ 0.83 (paper eval counts). With weak evidence (noise), posteriors drift to the 0.32 prior, so spoofs get predicted as bonafide (**H1**). Commit `02e34fa` already suspects this. |
| Checkpoints selected by weighted dev CE loss; dev loss swings 0.08 ↔ 0.58 | `main_train.py` | Selection is noisy and on the wrong metric, and dev contains no noisy audio |
| Constant LR (3e-5 in Qwen logs), Adam, bs 16, no warmup / LLRD / clipping / EMA | `main_train.py` (optimizer block) | Unstable training; likely over-writing pretrained robustness (**H3**) |
| Head: `SLS` → BN2d → SELU → maxpool(3,3) → flatten → `fc1` sized by `n_frames(64600)` → `fc2` | `model/sls_model.py` `ModelSLS` | Length-locked, position-sensitive, no dropout (**H4**) |
| `SLS` = shared `Linear(D,1)` + sigmoid per layer, **unnormalised** sum | `model/sls_model.py` `SLS` | Fused scale depends on the number of layers; weak inductive bias |
| Eval reads only the first 4.04 s of each clip | `utils/data_utils.py` `pad_audio(random_start=False)` | Ignores most of the evidence. Leading-silence / onset handling is exactly what VAD/NS changes under noise. |
| Current augmenter: ≤1 acoustic stage, then 1 Opus round-trip | `utils/rtc_augment.py` `_stage`, `codecs_augm` | No NS/AEC/AGC/VAD/DTX/PLC/EQ/bandwidth stages, no chaining (**H2**) |
| Older multi-stage chain (DFN, DRC, AGC, bandlimit, PLC, …) removed | `git show 43765f5^:SLS_setup/utils/rtc_augment.py` | Reusable code |
| All logged runs had RawBoost algo 5 forced on | pre-`43765f5` argparse | Confounds every past comparison |
| Paired offline/online mode exists, unused | `utils/data_utils.py` `SpoofAudioDataset(pair_map=...)` | PCL/TFCL-style training is ~1 day of work |
| w2v-BERT 2.0 backbone is wired but never run | `model/sls_model.py` `SSLModel` (`kind == "w2v_bert"`) | Free experiment; the AT-ADD 2026 winner used it |
| Neither the 83.20 run's merge list nor its augmentation config is saved; `--arch` defaults to a missing `aasist` module | `main_eval.py` | Fix reproducibility before anything else |

### 1.3 Hypothesis → cheapest test

| ID | Hypothesis | Cheapest falsification test | Phase |
|---|---|---|---|
| H1 | Prior / calibration collapse under noise | On dev-noisy-sim: compare macro-F1 at 0.5 vs the oracle threshold, and compare EER (threshold-free) between clean and noisy. If the F1 gap ≫ the EER gap, it's calibration. | 0/1 |
| H2 | Simulator lacks platform DSP | Add WebRTC APM + VAD to the sim; train 8 epochs; compare noisy-sim F1 | 2 |
| H3 | Fine-tuning erases backbone robustness | WiSE-FT interpolation α∈{0.3…0.9} between pretrained and fine-tuned weights, with zero training | 1/3 |
| H4 | Head/crop limits | Multi-crop averaging with the current head (no retraining) | 1 |
| H5 | Unseen platform diversity | Held-out-noise / held-out-codec sim variant vs matched variant; a gap indicates overfitting to the sim | 0 |

---

## 2. Rule-compliance matrix

Summarised from the challenge site and FAQ. **When in doubt, email the organisers (rtcsddchallenge2027@gmail.com) and keep the reply.**

| Idea | Status | Note |
|---|---|---|
| Non-speech noise / music / RIR corpora (ESC-50, FSD50K non-voice, MUSAN noise+music, DEMAND, RIR sets) | ✅ Allowed | MUSAN **speech** subset is explicitly forbidden. Filter "human voice" classes out of FSD50K/AudioSet-style sets. |
| RNNoise "office/coffee" data | ⚠️ Verify | Coffee-shop recordings may contain babble (speech). Use only if the organisers confirm, since they use it themselves for eval. |
| Babble built from **our own training utterances** | ✅ Likely allowed | It's official training data, not external speech |
| Local WebRTC AEC / NS / AGC / VAD simulation | ✅ Explicitly allowed | Sending data through real RTC apps is forbidden |
| Software codecs (Opus, SILK, AMR, G.722, Speex, …) via ffmpeg / python | ✅ Allowed | Local simulation |
| Neural codec round-trip (EnCodec/DAC) as augmentation, label kept | ⚠️ Ask | Rule bans "resynthesising bonafide to create spoofs". Keeping the label avoids that, but neural codecs leave vocoder-like artifacts, so it is risky anyway. |
| Pretrained weights not trained on spoof data (Qwen3-ASR, XLS-R, w2v-BERT 2.0, WavLM, Whisper, MMS, BEATs, Dasheng, …) | ✅ Allowed | **Not allowed:** DF_Arena, AntiDeepfake post-trained models, anything fine-tuned for ADD |
| Ensembles at inference | ❌ Forbidden | Single model |
| Weight averaging of runs of the same model (soups, SWA, EMA) | ✅ Allowed | |
| Merging differently-trained LoRA experts into one model | ⚠️ Ask | Probably "weight averaging", but confirm |
| Distilling an ensemble/teacher into a single student | ⚠️ Ask | Inference is a single model; training-time ensembles are usually fine |
| Dual-encoder single network trained end-to-end | ⚠️ Ask | One forward pass, one model file; may be read as an ensemble |
| Fixed enhancement network as a pre-processor | ⚠️ Ask (and probably harmful) | Speech enhancement tends to erase artifacts |
| Multi-crop / sliding-window inference with one model | ⚠️ Ask (very likely fine) | Single model, deterministic aggregation |
| Using progress/eval audio for tuning, model selection, pseudo-labels, test-time adaptation (TENT), continued pretraining, or manual analysis | ❌ Forbidden | Also covers analysing our own progress-score distributions (e.g., per-subset spoof rates) to drive decisions. Use the local proxy instead (§3). |
| Generating new spoofs with open TTS/VC from training text or speakers | ❌ / ⚠️ | "No resynthesis of bonafide" plus "no external speech data". Treat as forbidden unless the organisers say otherwise. |

> **Process note.** The progress leaderboard gives feedback, but picking among many variants by progress score is both risky (it's only 20% of eval) and against the spirit of the "no model selection on progress" rule. Use progress submissions only as a periodic sanity check that the local proxy tracks reality. All decisions go through §3.

---

## 3. Phase 0 – Evaluation infrastructure

**Nothing else is trustworthy until this exists.** Budget: 2–3 days, mostly CPU.

### E0.1 Build `dev-noisy-sim`, a local proxy for the noisy subset

- **Source audio:** the **dev online** split (P01/P02/P03; P03 WeChat is already unseen in train, which is good), plus dev offline passed through our simulated platform.
- **Recipe**, mirroring the paper's eval pipeline:
  ```
  clean dev utt ──► add noise/echo (official families) ──► platform DSP (APM: AEC/NS/AGC/VAD)
                 ──► codec (+ optional packet loss / PLC) ──► resample 16 kHz ──► store .flac
  ```
- **Three frozen variants.** Generate each once, with a fixed seed, and store it so every model sees identical audio:

  | Variant | Noise | DSP / codec | Purpose |
  |---|---|---|---|
  | `sim-matched` | ESC-50 rain/footsteps/keyboard, office/coffee-like noise, CLAD-style echo; **noise files disjoint from the training pool** | Same simulator family as training | Is the model learning the sim? |
  | `sim-heldout` | Different corpora (DEMAND, FSD50K non-voice, DNS noise) | Codecs/DSP **not** used in training (e.g., AMR-WB, G.722, Speex, different NS) | Generalisation to unseen platforms (H5) |
  | `sim-echo` | Echo-only: self-echo + RIR far-end echo + AEC residual | APM AEC on/off | Hardest condition, tracked separately |

- **SNR grid:** {0, 5, 10, 15, 20} dB, uniform. Record per-file metadata (noise type, SNR, DSP, codec) for breakdowns.
- **Size:** ~3–5k utterances per variant, stratified by language and label.
- **Clean proxy:** dev online, unprocessed.

### E0.2 Local metric and diagnostics

- **Primary:** `WF1_local = 0.3·MacroF1(dev-online-clean) + 0.7·mean(MacroF1(sim-matched), MacroF1(sim-heldout))` at threshold 0.5.
- **Always also log:**
  - EER per subset (threshold-free separability)
  - oracle-threshold macro-F1
  - ECE
  - **spoof recall and bonafide recall** per subset
  - F1 per noise type, SNR bucket and codec
- **Gap decomposition:** `F1@0.5` vs `F1@oracle` tells calibration loss apart from separability loss. Track both over time.
- **Code:** generalise the metric from `scripts/layerwise_logreg_probe.py` into `SLS_setup/utils/metrics.py`. Snippet in §13.1.

### E0.3 Select checkpoints by local WF1, not dev loss

- Evaluate every epoch on dev-clean + a 1k-utterance `sim-matched` subset (fast). Keep `best_by_wf1.pth` and `best_by_noisy_eer.pth`.
- Restore the best-model logic (removed in `43765f5`) keyed on WF1.

### E0.4 Reproducibility hygiene

- **One YAML config per run**, dumped into `exp/<run>/config.yaml`, including the augmentation config, CE weights, RawBoost flag, seed and git SHA.
- **Make `--arch sls` the default** in `main_eval.py`.
- **Write `MERGE_CHECKPOINTS` to a file next to the scores.** Re-create the 83.20 model (clarify whether "Merging [1, 8, 13, 18, 23]" means epochs) and score it with the new local metric. It becomes the **reference REF-0**.
- **Keep an experiment registry** (`docs/experiment_log.md` or a CSV) with columns: ID, config hash, local clean F1, sim-matched F1, sim-heldout F1, EER(s), progress score (if submitted), notes.

### E0.5 Sanity / shortcut audit (train/dev only)

- **Duration, leading/trailing silence, RMS level and peak per class** on train/dev, offline vs online. If bonafide and spoof differ in silence or level, the model may be using shortcuts that NS / VAD / AGC destroy under noise. That alone would explain part of the gap ([Müller et al. 2021, "Speech is Silver, Silence is Golden"](https://arxiv.org/abs/2106.12914)).
- **Probe:** score dev clips with speech replaced by silence / noise only. If P(spoof) moves strongly, there's a shortcut.

---

## 4. Phase 1 – Quick wins

Hours to days. Little or no retraining.

### E1.1 Logit bias / temperature calibration folded into `fc2` (H1)

- **Why.** The metric uses a fixed 0.5 threshold, and our training prior is wrong (0.32 vs ~0.83). Bayes prior correction would shift the spoof-vs-bonafide logit difference by log(0.83/0.17) − log(0.32/0.68) ≈ **+2.3** nats. That optimises accuracy, not macro-F1, so sweep instead.
- **How:**
  1. On `sim-matched ∪ sim-heldout ∪ dev-clean`, fit `d' = d/T + β`, where `d = z_spoof − z_bona`. Choose (T, β) to maximise the **local WF1** (grid β∈[−1, 3], T∈[0.5, 3]).
  2. Fold the result into the weights: `fc2.weight /= T; fc2.bias /= T; fc2.bias[0] += β` (spoof is index 0). The submission stays a single model with a 0.5 threshold. See §13.2.
- **Robustness:** choose β on the **average** of variants, not on the best one. Check that the clean F1 drop is < 0.5.
- **Expected:** +1–5 noisy F1 if H1 holds. Cost: about 1 hour.

### E1.2 Class-weight and prior sweep at train time

- **Grid:** `--ce_weights` ∈ {[0.1, 0.9] (ref), [0.19, 0.81] (inverse frequency), [0.5, 0.5] + class-balanced sampler, [0.5, 0.5] plain}.
- **Also try logit-adjusted loss** ([Menon et al. 2007.07314](https://arxiv.org/abs/2007.07314)): train balanced, then apply the prior offset at inference (equivalent to E1.1 but principled).
- Always pair this with E1.1 calibration afterwards. Compare F1@oracle (separability), not only F1@0.5.

### E1.3 Multi-crop / full-utterance inference (H4)

- **With the current head:** slide 4.04 s windows with 50% overlap and average the **logits** (or the mean of the top-k spoof logits). Compare first-crop / center-crop / mean / median.
- **With a new head (E4.1):** feed the full utterance directly.
- **Rules:** confirm with the organisers that multi-crop by one model is fine (it is not an ensemble).
- **Expected:** +0.5–2 F1. Bigger on noisy, because transients (keyboard, footsteps) are localised. Cost: under a day.

### E1.4 Proper weight averaging

- **Greedy soup** ([Wortsman et al. 2203.05482](https://arxiv.org/abs/2203.05482)):
  1. Sort checkpoints by local WF1.
  2. Add each one if it improves the soup's local WF1.
- **Recompute BN stats after averaging.** Run ~500 train batches in train mode with no grad. `first_bn` is a `BatchNorm2d(1)`, and averaged running stats are not the stats of the averaged model.
- **Uniform SWA** over the last K epochs of a cosine schedule ([Izmailov et al. 1803.05407](https://arxiv.org/abs/1803.05407)).
- **EMA of weights during training** (decay 0.999–0.9999). Cheap, and usually ≥ the best single epoch.
- **Code:** extend `main_eval.py` `merge_checkpoints`, adding greedy selection, a BN-recompute hook, and a list file instead of the hardcoded list.

### E1.5 WiSE-FT: interpolate fine-tuned with pretrained weights (H3)

- **What:** `θ = (1−α)·θ_pretrained + α·θ_finetuned` for the backbone; the head is taken from the fine-tuned model. α ∈ {0.3, 0.5, 0.7, 0.9}. ([Wortsman et al. 2109.01903](https://arxiv.org/abs/2109.01903))
- **Why:** this is the vision recipe for OOD robustness at zero training cost. It tests H3 directly.
- **Expected:** uncertain; anywhere from +0 to +3 noisy F1. Cost: hours.

---

## 5. Phase 2 – Augmentation v2

This is the biggest expected lever: an RTC-faithful, domain-randomised simulator.

**Principle (borrowed from vision corruption robustness).** A diverse, *compositional* random chain plus a consistency loss generalises to unseen corruptions better than any single matched augmentation (AugMix, DeepAugment, PixMix). For robotics it's *domain randomisation*: don't simulate the 7 platforms exactly; randomise over a family that contains them.

### 5.1 Target pipeline (order matters)

```
            ┌── noise   (ESC-50, FSD50K non-voice, MUSAN noise/music, DEMAND, RNNoise-noise*)  SNR 0–25 dB
x_clean ──► ├── echo    (self-echo; RIR far-end echo of another *training* utt of the same label)
            ├── reverb  (measured + pyroomacoustics RIRs; random room/mic distance)
            └── babble  (mix of 3–8 training utterances at low level)*                        [0..2 of these]
       ──► PLATFORM DSP  [random subset, random order constraints]
            AEC (WebRTC AEC3 / speexdsp)  →  NS (WebRTC NS levels 0–3 | RNNoise | DeepFilterNet | ffmpeg afftdn/anlmdn)
            →  AGC (WebRTC AGC2 | ffmpeg dynaudnorm/acompressor)  →  VAD gating / DTX + comfort noise
       ──► CODEC  Opus (voip/audio, 6–48 kbps, FEC, DTX, 10/20/40/60 ms) | SILK (pysilk) | AMR-NB/WB | G.722 | Speex | G.711
       ──► NETWORK  packet loss (Gilbert–Elliott bursts, 0–15%) + decoder PLC; jitter-buffer time-warp (±2%)
       ──► RECEIVER  band-limit / resample 8–48 kHz, bandwidth-extension proxy, EQ (random parametric), limiter, gain ±10 dB
```
`*` = check the rule status (§2).

**Key design rules:**
1. **Class-independent.** Every stage is sampled with the same probability for bonafide and spoof; never leak the label through augmentation. Assert this in the sampler.
2. **Chain depth.** Draw a random depth of 1–4 stages per category, in the order above (AugMix-style). The current "one stage then codec" becomes the special case depth=1.
3. **Noisy ratio.** Since 70% of the score is noisy, ~60–70% of training samples should take the noise branch and ~30% stay clean-online-like (the codec only).
4. **SNR distribution:** skewed toward hard cases, e.g. Beta-shaped over [0, 25] dB with mode ~8 dB.
5. **Both offline and online training audio go through the sim.** Offline + sim gives the cleanest analogue of the eval pipeline; online + sim gives "double processing", which adds diversity.

### 5.2 Experiments

| ID | Change | Hypothesis | Notes |
|---|---|---|---|
| A2.1 | **Chained augmentation** (depth 1–4) vs current single-stage | H2/H5 | Port the stages from the old chain (`43765f5^`) into the new class |
| A2.2 | **+ WebRTC APM** (AEC3/NS/AGC2) after the noise stage | H2 | Explicitly allowed. Highest prior for noisy F1 |
| A2.3 | **+ VAD gating, DTX, comfort noise** | H2 | TFCL found VAD the most damaging stage; Opus DTX replaces silence with CNG |
| A2.4 | **Echo suite:** self-echo (CLAD-like), far-end RIR echo + AEC residual, double-talk | Echo is the hardest condition | Far-end = training utterance of the **same label**, −25…−5 dB, delay 20–300 ms |
| A2.5 | **NS diversity:** WebRTC NS levels, RNNoise (`ffmpeg arnndn`), DeepFilterNet, `afftdn` | Unseen platforms | Platforms use different suppressors; musical noise and speech distortion differ |
| A2.6 | **Codec zoo + packet loss/PLC** (Opus FEC/DTX/frame sizes, SILK, AMR-WB, G.722, Speex) | H5 | QQ/WeChat use SILK-derived codecs; `pysilk` gives real SILK |
| A2.7 | **Receiver chain:** random EQ, band-limit 3.4/4/6/7/8 kHz, resample, AGC/limiter, ±10 dB gain | H5 | Covers bandwidth-extension and AGC differences across platforms. **Level part (gain/AGC/compressor/limiter/clipping) promoted to P0 as A2.7a** after the E0.5 audit found a level/clipping shortcut (`docs/audit/E0.5_shortcut_audit.html`): `--use_level_aug`, `utils/level_augment.py` |
| A2.8 | **Platform transfer-function augmentation (sim2real from our own pairs)** | H5 | Estimate per-platform long-term spectral ratio `|X_online(f)| / |X_offline(f)|` from paired train data (P01, P02). Apply random perturbations or interpolations of these curves as EQ. Cheap and data-driven. |
| A2.9 | **Silence / VAD robustness:** random trim, insert or replace silence; CNG | Shortcut removal (E0.5) | Especially if the E0.5 audit finds silence or duration cues |
| A2.10 | **SpecAugment / mel-band dropout on the Qwen log-mel input** (GPU, free) | Regularisation | Time masks of 2×≤40 frames, freq masks of 2×≤16 bins; also random high-band attenuation |
| A2.11 | **RawBoost on top** (algo 3 or 5, p=0.3) vs off | Confounder check | All past runs had it on |
| A2.12 | **Augmentation strength curriculum:** ramp p(noisy) and min-SNR over epochs | Optimisation | vs constant strength |
| A2.13 | **Babble from training speech** | Office/coffee realism | Rule-safe only because it is training data; use same-label utterances or bonafide-only babble. Keep the level low. |

**Ablation protocol.** Starting from A2.1, add stages one at a time (A2.2 → A2.7). Screen each with the short schedule (§10) on `sim-matched` and `sim-heldout`. Keep a stage only if it improves **sim-heldout** (that's the generalisation signal) without hurting clean by more than 0.5.

### 5.3 Engineering: make augmentation fast

- **Today:** two ffmpeg subprocesses per clip per sample is too slow for chained DSP.
- **Pre-rendered cache.** Render K=6–10 augmented variants per training utterance offline (multiprocessing on CPU nodes), store them as 16 kHz FLAC with JSON metadata, and sample one variant per epoch. Rough size: 75k utts × 8 variants × ~5 s ≈ 50–100 GB FLAC.
- **Online, GPU-side:** gain, EQ, SpecAugment, mixing of pre-rendered noise. Use `torch-audiomentations` or custom code.
- **Mixed:** 50% cached heavy chains + 50% online light augmentation, to keep diversity high.
- **Simulator fidelity check:** for P01/P02, compare `sim(offline)` against the real `online` pair in backbone-embedding space (cosine or a Fréchet distance on layer means). Tune the simulator to shrink that distance. This uses only training data and gives a principled simulator-tuning signal.

---

## 6. Phase 3 – Training strategies

### T3.1 Paired consistency learning (highest-evidence method)

- **Pairs available:**
  1. offline ↔ online (real platform, train data)
  2. clean ↔ sim(noisy) (our simulator)
  3. sim view A ↔ sim view B (AugMix-style)
- **Loss:**
  ```
  L = CE(x_a) + CE(x_b)
    + λ_emb · D_emb(h_a, h_b)     # embedding-level consistency
    + λ_js  · JS(p_a, p_b[, p_c]) # prediction-level (AugMix JSD)
  ```
- **`D_emb` variants:**

  | Variant | Description | Notes |
  |---|---|---|
  | (a) utterance-level | Cosine/MSE on attentive-pooled embeddings | Robust to VAD time shifts |
  | (b) frame-level, soft-aligned | Frame-level loss after soft attention alignment (TFCL temporal term) | Required when VAD/PLC shift time |
  | (c) phoneme/word-level | PCL; pooled within aligned segments (Qwen3-ForcedAligner, or CTC from an ASR model on *train* data) | |

  Start with (a) + JS. It's simplest and survives time misalignment.
- **Clean-teacher variant:** stop the gradient on the clean branch, or use an **EMA teacher** (mean teacher / BYOL-like), so the noisy view is pulled toward the clean representation rather than both collapsing.
- **λ:** start at 0.3, matching TFCL; sweep {0.1, 0.3, 1.0}.
- **Code:** enable `pair_map` in `build_dataset_from_protocol`; add a `--consistency` flag to `main_train.py`. Snippet in §13.4.
- **Expected:** +2–5 noisy F1 (the paper's PCL takes 7.33 → 5.81 EER overall and improves every noise type). Cost: 1–2 days of code, plus about 1.5× compute per step.

### T3.2 Optimisation recipe (stability first)

| Knob | Proposal |
|---|---|
| Optimiser | AdamW, wd 0.01 on non-norm/non-bias params |
| LR | Backbone 1e-5…3e-5, head 1e-3 (separate param groups) |
| **LLRD** | Layer-wise decay 0.85–0.9 from top to bottom; lower layers stay close to pretrained (protects H3) |
| Schedule | Linear warmup 5% → cosine to 0 over 20–30 epochs (short, with heavy augmentation) |
| Clipping | Global norm 1.0 |
| Precision | bf16 autocast; enables a bigger batch |
| Batch | Effective batch 64–128 via gradient accumulation. Replace `BatchNorm2d` in the head with LayerNorm if the batch stays small. |
| EMA | Decay 0.9995; evaluate the EMA weights |
| Early stop | On local WF1 (E0.3) |

Expected: lower epoch-to-epoch variance, more reproducible gains, and the precondition for trusting any other ablation.

### T3.3 Parameter-efficient and robust fine-tuning (H3)

| ID | Method | Rationale |
|---|---|---|
| T3.3a | **LoRA / DoRA** on audio-tower attention + FFN (r=16–64, α=2r, dropout 0.05), head fully trained | Keeps the pretrained noise robustness; less overfitting to P01/P02 |
| T3.3b | **Staged LoRA → full FT**: train the LoRA and head 5–10 epochs, merge, then full FT at a low LR with LLRD | The AT-ADD 2026 winner's recipe (WaveShield) |
| T3.3c | **Freeze the bottom k layers** (k = 4/8/12) + full FT on the rest | Low layers hold acoustic detail; their pretraining is precious |
| T3.3d | **LP-FT**: linear-probe (frozen) first, then full FT ([Kumar et al. 2202.10054](https://arxiv.org/abs/2202.10054)) | Avoids feature distortion from a random head |
| T3.3e | **WiSE-FT** on the result (E1.5) | Free robustness knob |
| T3.3f | **MoE-LoRA**: K LoRA experts + a router conditioned on the degradation (AMULET / MoE-LoRA) | One model, specialised paths for echo / noise / codec |

Note the evidence from our own table: the Qwen 0.6B and XLS-R 300M finetuned runs **lost** to their frozen versions. Our full fine-tuning recipe is fragile, which is a strong reason to try a/b/c/d.

### T3.4 Sharpness-aware minimisation

- **SAM / ASAM** (ρ = 0.05 / 0.5). [Huang et al., Interspeech 2025 (2506.11532)](https://arxiv.org/abs/2506.11532) show that sharpness predicts the domain-shift gap in anti-spoofing; code at `nii-yamagishilab/SAM-AntiSpoofing`.
- About 2× compute. Try it after T3.2 is stable.

### T3.5 Losses

- **Label smoothing** (ε = 0.05–0.1). Better calibration, so E1.1 transfers more reliably.
- **Focal loss** (γ = 1–2), as an alternative to class weights.
- **AM-softmax / OC-softmax.** Compact bonafide class with a margin; used by the AT-ADD winner. Use it with care under a fixed threshold, and always recalibrate (E1.1).
- **Supervised contrastive (SupCon) across augmentation views.** Pulls the same-class / different-condition embeddings together.

### T3.6 Worst-condition objectives

- **GroupDRO** ([Sagawa et al. 1911.08731](https://arxiv.org/abs/1911.08731)) over augmentation groups (clean, noise, echo, NS-heavy, low-bitrate, packet loss). It upweights the worst group, which on our proxy will be echo.
- **Worst-of-k augmentation.** Sample k=2–4 views and backprop the highest-loss view (cheap adversarial augmentation; [Gontijo-Lopes et al.](https://arxiv.org/abs/2002.08973)).
- **DANN** with gradient reversal on platform / augmentation-type labels (we have P01/P02/offline + sim metadata). Gives condition-invariant embeddings. AT-ADD #3 used DANN + GroupDRO.

### T3.7 Data mixture

- **Paper finding:** Off-only gives 13.79 EER online, On-only 8.35, Mix 8.57, PCL 6.77.
- **Grid:** {online only + sim, mix + sim, offline + sim only (offline → simulated platform), mix with online upweighted 2×}.
- **Language balance:** progress is ~74% zh. Check that train language ratios are similar, and reweight if not.

### T3.8 Longer context and crops

- **Train on random 4–8 s crops** with the new pooling head (E4.1); evaluate on the full utterance.
- **Random crop position** in training (already done). Also try **random multi-segment concatenation** of the same utterance, to break onset shortcuts.

---

## 7. Phase 4 – Architecture & backbones

### M4.1 Replace the flatten-FC head (H4)

| Head | Description | Notes |
|---|---|---|
| **ASP**: attentive statistics pooling | Frame attention → weighted mean + std → MLP (dropout 0.2–0.3) | Simplest; variable length; ~1–2M params |
| **MHFA** | Multi-head factorised attentive pooling over layers × frames ([2409.15234](https://arxiv.org/abs/2409.15234)) | Learns layer and frame attention jointly; strong in recent ADD |
| **AASIST** back-end | Graph attention over spectro-temporal | Official baseline back-end; heavier |
| **Small transformer** (2 layers, CLS token) | Contextual pooling | Good with longer crops |
| **Nes2Net-X** | Nested Res2Net without dimensionality reduction ([2504.05657](https://arxiv.org/abs/2504.05657)) | Did worse than AASIST on noisy for the organisers; low priority |

Expected: +1–3 noisy F1 (it removes position sensitivity and allows full-utterance inference). Cost: about 1 day.

### M4.2 Layer fusion

- **Fix `SLS`:** global softmax-normalised layer weights, optionally plus per-sample attention, plus **layer dropout** (p=0.1) on the fusion.
- **Layer probing on `sim-matched`/`sim-heldout`.** Re-run `scripts/layerwise_logreg_probe.py` with frozen Qwen3-ASR features on the **noisy sim**, not on clean dev. Find which layers keep separability under noise, then restrict the fusion or drop the top layers. Literature: low/mid layers carry the artifact cues ([2406.10283](https://arxiv.org/abs/2406.10283), [2509.12003](https://arxiv.org/abs/2509.12003)).
- **Truncate the backbone** at the best depth: faster training, less overfitting.

### M4.3 Backbones to test

All are public and spoof-free; verify each license and training data.

| Priority | Backbone | Why | Status in code |
|---|---|---|---|
| P0 | **w2v-BERT 2.0** (600M, 4.5M h, 143 langs) | AT-ADD 2026 winner backbone (3× w2v-BERT 2.0 + LoRA → FT) | ✅ wired (`kind == "w2v_bert"`) |
| P1 | **WavLM-Large** | Denoising pretraining (utterance mixing + noise); robust to noise | Add (wav2vec2-like path) |
| P1 | **Whisper-large-v3 / v3-turbo encoder** | ASR encoder like Qwen3-ASR; 5M h of noisy data | Add (similar to the Qwen mel path) |
| P1 | **Chinese-strong ASR encoders**: SenseVoice, FireRedASR-AED encoder, Paraformer | ~74% of progress is zh; trained on noisy Mandarin | Add; check licenses |
| P2 | **Qwen3-Omni / Qwen2.5-Omni audio encoder** | Bigger AuT-style encoder from the same family | Add |
| P2 | **MMS-1B** | Multilingual wav2vec2 | Wired via the wav2vec2 path |
| P2 | **Dasheng / BEATs / EAT** (general audio) | Noise-type awareness; dual-domain fusion helps ADD ([2608.29021](https://arxiv.org/abs/2608.29021)) | As a second branch only (M4.4) |

**Protocol:** freeze the backbone, run the layer probe on `sim-heldout`, and promote only the top-2 to full training.

### M4.4 Single-model multi-branch fusion (⚠️ check the ensemble rule first)

- **ASR-encoder + acoustic-SSL** (Qwen3-ASR + w2v-BERT/WavLM) with cross-attention fusion and one head, trained jointly. Complementary: semantic robustness plus acoustic artifacts.
- **Prosody branch:** F0 / voicing / energy / pause-duration contours (pyworld or a DSP pitch tracker) → small TCN → concatenate. TTS prosody signatures survive codecs better than spectral detail.
- **High-band branch:** LFCC/CQCC on 4–8 kHz with a band-presence mask. Artifacts there are often wiped by codecs, but strong when present.

### M4.5 Degradation-aware conditioning

- **Auxiliary heads:** predict the augmentation type, SNR bucket and codec family (multi-task, from sim metadata).
- **FiLM:** condition the classifier on the predicted degradation embedding. It's "condition-aware calibration" inside one model, and helps the fixed-0.5 threshold problem because bias can depend on the estimated condition.
- **Fallback:** the plain auxiliary task alone, with no FiLM, as regularisation.

### M4.6 Phoneme- / word-level pooling (PCL-style)

- Use Qwen3-ForcedAligner (verify availability and license) or the Qwen3-ASR decoder timestamps **on training data** to get segments.
- Pool backbone frames per phone/word and classify with attention over segments. Combine with T3.1(c).
- The paper's ablation shows phoneme-level beats frame-level for consistency under RTC.

---

## 8. Phase 5 – Moonshots & cross-domain ideas

Higher risk. Pick 1–2 that fit the remaining time after Phase 3.

| ID | Idea | Origin | Sketch | Risk |
|---|---|---|---|---|
| X5.1 | **JEPA / "world-model" latent prediction** | V-JEPA, A-JEPA, data2vec, BYOL | A predictor maps the student's latents (noisy view, masked) to an EMA teacher's latents (clean view). Auxiliary loss during FT, or a continued-pretraining stage on **training audio only** before FT. The predictor is dropped at inference, so it's still a single model with the same cost. | Medium: tuning collapse / EMA |
| X5.2 | **Task-aware feature-space enhancement** | DKDSSD ([2310.08869](https://arxiv.org/abs/2310.08869)) | A small module after the backbone maps noisy features to clean-view features (L2 to the paired clean features) and is trained jointly with the detection loss. Unlike waveform SE, it's optimised to *keep* artifacts. | Medium |
| X5.3 | **DeepAugment for audio** | DeepAugment ([2006.16241](https://arxiv.org/abs/2006.16241)) | Run DeepFilterNet / RNNoise / a small neural codec with **randomly perturbed weights** (noise, zeroing, sign flips) as augmentation. Gives novel nonlinear distortions resembling unseen platform DSP. | Low–medium; cheap to try |
| X5.4 | **Learned channel model (sim2real)** | Robotics sim2real | Train a small wave-U-net / Demucs-lite on paired offline→online train audio (P01/P02). Use it, plus weight-perturbed or interpolated versions, as augmentation. Apply it to both classes. | ⚠️ Rule question (neural processing of bonafide); generator artifacts → label noise |
| X5.5 | **Adversarial augmentation search** | Adversarial AutoAugment, AdvProp | Sample simulator parameters to maximise loss, or run feature-level FGSM on the fused features (ε small), with separate norm stats for adversarial samples (AdvProp). | Medium |
| X5.6 | **Ensemble → single-model distillation** | Born-again networks, model compression | Train 2–3 strong but diverse models (backbones/seeds). Distil their averaged soft labels on noisy-sim views into one student. | ⚠️ Rule question (training-time ensemble only) |
| X5.7 | **Same-text bonafide/spoof contrastive pairs** | Counterfactual / contrastive learning (NLP) | If metadata links spoofs to bonafide transcripts/speakers, build pairs that differ only in "realness". A pairwise margin loss forces non-content cues. | Low; depends on metadata |
| X5.8 | **Semantic / ASR-confidence cues** | Audio LLMs | Use the Qwen3-ASR decoder's token entropy / log-prob trajectory as an extra feature, or LoRA-tune the full Qwen3-ASR as a QA classifier ("real or fake?"). ALLM4ADD ([2505.11079](https://arxiv.org/abs/2505.11079)) shows in-domain viability. | High compute; low prior |
| X5.9 | **Leave-one-generator-out validation** | OOD evaluation practice | Measure method robustness to unseen generators (4 are unseen in eval) by holding out G03/G09 in training. Use it for tie-breaks between methods. | Compute only |
| X5.10 | **Frequency-occlusion analysis** | Explainability (vision occlusion maps) | On `sim-heldout`, mask mel bands / time regions and measure ΔP(spoof). Learn which evidence survives RTC, then design the head or augmentations to focus there. | Analysis only |

**Explicitly not recommended:**
- **Diffusion purification / speech-enhancement front-ends at inference.** The literature shows enhancement erases spoof artifacts ([2603.14767](https://arxiv.org/abs/2603.14767), [2503.17577](https://arxiv.org/abs/2503.17577)), and it's a rule grey area.
- **Test-time adaptation on progress/eval audio.** Forbidden.

---

## 9. Experiment catalogue

Priority: **P0** = do now, **P1** = high value, **P2** = if time.
"Exp. Δ" = prior guess of the gain in noisy F1 on the local proxy, *not* a promise.
Cost is in Qwen-1.7B GPU-days at our current throughput; "s" = short-schedule screen.

| ID | Name | Hypothesis | Files to touch | Cost | Exp. Δ | Prio | Depends | Keep if |
|---|---|---|---|---|---|---|---|---|
| E0.1 | dev-noisy-sim (3 variants) | — | new `scripts/build_dev_noisy_sim.py`, reuse `utils/rtc_augment.py` | CPU 1–2 d | — | P0 | — | Exists, frozen |
| E0.2 | Local WF1 + diagnostics | — | new `utils/metrics.py`; `main_eval.py` | 0.5 d | — | P0 | E0.1 | — |
| E0.3 | Select ckpt by local WF1 | — | `main_train.py` dev loop | 0.5 d | +0–2 | P0 | E0.2 | — |
| E0.4 | Reproducibility + REF-0 | — | configs, `main_eval.py` | 0.5 d | — | P0 | — | REF-0 scored locally |
| E0.5 | Shortcut audit | H2 | notebook / script | CPU 0.5 d | — | P0 | — | — |
| E1.1 | Calibration fold-in | H1 | `main_eval.py` (`--calib T β`) | hours | +1–5 | **P0** | E0.2 | ΔWF1 ≥ +0.3 |
| E1.2 | CE weight / logit adj. sweep | H1 | `main_train.py` | 4×s | +0–3 | P0 | E0.3 | F1@oracle ↑ |
| E1.3 | Multi-crop inference | H4 | `main_eval.py`, `utils/data_utils.py` | hours | +0.5–2 | P0 | E0.2 | ΔWF1 ≥ +0.3 |
| E1.4 | Greedy soup + BN recompute + EMA | — | `main_eval.py` `merge_checkpoints`, `main_train.py` | 0.5 d | +0.5–2 | P0 | E0.2 | ΔWF1 ≥ +0.3 |
| E1.5 | WiSE-FT α sweep | H3 | new `scripts/wise_ft.py` | hours | 0–3 | P1 | E0.2 | ΔWF1 ≥ +0.3 |
| A2.1 | Chained aug (depth 1–4) | H2/H5 | `utils/rtc_augment.py` | s + 1 GPU-d | +1–3 | P0 | E0.1 | heldout ↑ |
| A2.2 | + WebRTC APM | H2 | `utils/rtc_augment.py` | s | +1–4 | **P0** | A2.1 | heldout ↑ |
| A2.3 | + VAD / DTX / CNG | H2 | `utils/rtc_augment.py` | s | +0.5–2 | P1 | A2.2 | heldout ↑ |
| A2.4 | Echo suite | echo | `utils/rtc_augment.py` | s | +1–3 (echo) | **P0** | A2.1 | sim-echo ↑ |
| A2.5 | NS diversity | H5 | `utils/rtc_augment.py` | s | +0.5–2 | P1 | A2.2 | heldout ↑ |
| A2.6 | Codec zoo + PLC | H5 | `utils/rtc_augment.py` | s | +0.5–2 | P1 | A2.1 | heldout ↑ |
| A2.7a | Level aug: random target level + AGC / compressor / limiter / clipping | shortcut (E0.5) | `utils/level_augment.py` | s | 0–1.5 | **P0** | E0.5 | level_flip_bona ↓ (+10 dB clip < 5 %), WF1 ≥ ref − 0.3 |
| A2.7 | Receiver chain (EQ/BW) | H5 | `utils/rtc_augment.py` | s | +0.5–1.5 | P1 | A2.1 | heldout ↑ |
| A2.8 | Platform transfer-function EQ | H5 | new script + `rtc_augment.py` | 1 d + s | +0.5–2 | P1 | E0.1 | heldout ↑ |
| A2.9 | Silence / VAD robustness | shortcut | `utils/rtc_augment.py` | s | 0–2 | P1 (P0 if E0.5 flags) | E0.5 | noisy ↑ |
| A2.10 | Mel SpecAugment | reg. | `model/sls_model.py` | s | +0–1 | P1 | — | any ↑ |
| A2.11 | RawBoost on/off | confounder | flags | 2×s | ±1 | P1 | A2.1 | — |
| A2.12 | Aug curriculum | optim. | `utils/rtc_augment.py` | s | 0–1 | P2 | A2.x | — |
| A2.13 | Babble from train speech | office/coffee | `utils/rtc_augment.py` | s | 0–1.5 | P2 | rules | — |
| T3.1 | Paired consistency (offline↔online, clean↔sim, JS) | H2/H5 | `main_train.py`, `utils/data_utils.py` | 1–2 d + 1.5× compute | +2–5 | **P0** | E0.3, A2.1 | noisy ↑ ≥ 1 |
| T3.2 | Optim recipe (AdamW, LLRD, warmup-cos, clip, bf16, EMA) | stability | `main_train.py` | 1 d + s | +1–3 | **P0** | E0.3 | var ↓, WF1 ↑ |
| T3.3a | LoRA / DoRA | H3 | `model/sls_model.py` (peft) | s | 0–3 | P1 | T3.2 | heldout ↑ |
| T3.3b | Staged LoRA → full FT | H3 | same | 1.5× | +1–3 | P1 | T3.3a | heldout ↑ |
| T3.3c | Freeze bottom-k | H3 | `model/sls_model.py` | 3×s | 0–2 | P1 | T3.2 | heldout ↑ |
| T3.3d | LP-FT | H3 | `main_train.py` | s | 0–2 | P2 | T3.2 | — |
| T3.3f | MoE-LoRA by degradation | H5 | model | 3 d | 0–3 | P2 | T3.3a | — |
| T3.4 | SAM / ASAM | sharpness | `main_train.py` | 2× | +0–2 | P1 | T3.2 | heldout ↑ |
| T3.5 | Label smoothing / focal / AM-softmax | calib. | `main_train.py` | 3×s | 0–1.5 | P1 | E1.1 | after calib ↑ |
| T3.6 | GroupDRO / worst-of-k / DANN | worst group | `main_train.py` | s each | +0.5–2 | P1 | A2.x metadata | echo ↑ |
| T3.7 | Data mixture grid | — | protocols | 4×s | 0–2 | P1 | A2.1 | — |
| T3.8 | Longer crops (4–8 s) | H4 | `utils/data_utils.py` | s | +0.5–1.5 | P1 | M4.1 | — |
| M4.1 | ASP / MHFA head | H4 | `model/sls_model.py` | 1 d + s | +1–3 | **P0** | — | WF1 ↑ |
| M4.2 | Layer fusion fix + noisy layer probe | — | `SLS`, `scripts/layerwise_logreg_probe.py` | 1 d | 0–2 | P1 | E0.1 | — |
| M4.3 | w2v-BERT 2.0 | backbone | flags only | 1–2 GPU-d | ? (+0–4) | **P0** | T3.2 | WF1 ≥ Qwen−1 |
| M4.3b | WavLM-L / Whisper-v3 / SenseVoice / FireRedASR | backbone | `model/sls_model.py` | probe + 1–2 GPU-d each | ? | P1 | M4.2 probe | — |
| M4.4 | Multi-branch fusion | complementarity | model | 3 d | 0–3 | P2 | rules | — |
| M4.5 | Degradation-aware aux + FiLM | H1/H5 | model, trainer | 1–2 d | 0–2 | P2 | A2 metadata | — |
| M4.6 | Phoneme / word pooling | PCL | model, aligner | 3 d | 0–2 | P2 | T3.1 | — |
| X5.1 | JEPA latent prediction | — | trainer | 3–5 d | ? | P2 | T3.1 | — |
| X5.2 | Feature-space enhancement | — | model | 2 d | ? | P2 | T3.1 | — |
| X5.3 | DeepAugment-audio | H5 | aug | 1–2 d | ? | P2 | A2.5 | heldout ↑ |
| X5.6 | Distillation → single model | — | trainer | 3 d | +0–2 | P2 | rules | — |
| X5.9 | Leave-one-generator-out | H5 | protocols | compute | — | P2 | — | — |

---

## 10. Methodology & reporting

1. **Fixed reference.** REF-0 = re-created best Qwen-1.7B config, scored on the local proxy. Every experiment changes **one** factor relative to the current reference. When a change wins, it becomes the new reference (REF-1, REF-2, …). Log each promotion.
2. **Short-schedule screening ("s").**
   - 30–40% stratified train subset, 8–10 epochs of warmup-cosine, eval on dev-clean + sim-matched + sim-heldout.
   - Promote to a full run only if Δ local noisy F1 ≥ +1.0, or ≥ +0.5 consistently across 2 seeds.
   - Check once that screening rank correlates with full-run rank (Spearman on 4–5 configs).
3. **Seeds.** Any decision within ±1 WF1 needs 2 seeds. Report mean ± range.
4. **Always report:**
   - local WF1 (at 0.5 and after calibration)
   - clean / sim-matched / sim-heldout / sim-echo macro-F1
   - EERs
   - spoof and bonafide recall per subset
   - GPU-hours
5. **Progress submissions.** At most ~1 per major milestone (after Phase 1, after Phase 2+3 combined, final candidate), to check that the local proxy correlates with the leaderboard. Never choose between near-tied variants with progress.
6. **Kill criteria.** Drop an idea after 2 failed screens unless there's a clear bug hypothesis. Time-box moonshots to ≤ 3 days each.

**Run report template** (append to the registry):
```
ID / date / owner / git SHA / config hash
Change vs REF-k: ...
Local: clean F1 | sim-matched F1 | sim-heldout F1 | sim-echo F1 | WF1@0.5 | WF1@calib | EER clean/noisy
Recall (spoof/bona) noisy: ...
Cost: GPU-h ...
Decision: promote / drop / revisit — why
```

---

## 11. Timeline

Today is 2026-10-03. Final eval data comes out 2026-11-09; submission is due 2026-11-16.

| Week | Dates | Goals | Exit criterion |
|---|---|---|---|
| W1 | Oct 3–10 | E0.1–E0.5, REF-0, E1.1, E1.3, E1.4, E1.5; start A2.1/A2.2/A2.4 code; T3.2 recipe | Local proxy exists; REF-0 + calibration measured |
| W2 | Oct 10–17 | Screens A2.1→A2.7; T3.2 full run; M4.1 head; M4.3 w2v-BERT 2.0 screen | Augmentation v2 frozen (best chain) |
| W3 | Oct 17–24 | T3.1 consistency; T3.3a/b/c PEFT; E1.2 CE sweep on the new stack; T3.4 SAM | REF-n with consistency + PEFT |
| W4 | Oct 24–31 | Backbone shoot-out (M4.3b), M4.2 layer probe, T3.6/T3.7, 1 moonshot (X5.1 or X5.3) | Final backbone + head chosen |
| W5 | Oct 31–Nov 7 | Combine winners; 2–3 seeds of the final recipe; greedy soup; calibration; inference run-book dry run on progress-sized data | Frozen final model candidate(s) |
| W6 | Nov 7–16 | Final inference on eval (Nov 9+), sanity checks, submission; write the 2-page system description draft | Submitted by Nov 14 (2-day buffer) |

---

## 12. Recommended final stack

Best current guess, to be validated:

```
Backbone   : Qwen3-ASR-1.7B audio tower (or w2v-BERT 2.0 if it wins M4.3), bottom-k frozen or LLRD 0.85
Fusion     : softmax-normalised layer weights over probe-selected layers
Head       : attentive statistics pooling (or MHFA), dropout 0.2, LayerNorm
Training   : staged LoRA → full FT, AdamW, warmup-cosine, clip 1.0, bf16, EMA 0.9995
Loss       : CE (balanced or inverse-freq) + 0.3·consistency(clean↔sim, offline↔online) + JS over 2 aug views
Augment    : chained RTC simulator v2 (noise/echo → APM AEC/NS/AGC/VAD → codec zoo + PLC → receiver EQ/BW/gain),
             p(noisy)=0.65, class-independent, cached heavy chains + GPU light augmentations
Selection  : local WF1 on dev-clean + sim-matched + sim-heldout
Averaging  : greedy soup of EMA checkpoints (+ seeds) with BN/LN recompute; optional WiSE-FT α
Calibration: (T, β) fitted on the local proxy and folded into the final linear layer → threshold 0.5
Inference  : full utterance (or 4 s multi-crop mean-logit if the head stays fixed-length)
```

---

## 13. Implementation appendix

### 13.1 Local WF1 (`SLS_setup/utils/metrics.py`)

```python
import numpy as np
from sklearn.metrics import f1_score, roc_curve

def macro_f1(y_true_spoof, p_spoof, thr=0.5):
    """y_true_spoof: 1 = spoof, 0 = bonafide; p_spoof in [0, 1]."""
    return f1_score(y_true_spoof, (p_spoof >= thr).astype(int), average="macro")

def eer(y_true_spoof, p_spoof):
    fpr, tpr, _ = roc_curve(y_true_spoof, p_spoof)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2

def wf1(clean, noisy_list, thr=0.5):
    """clean / each noisy item: (y_true_spoof, p_spoof)."""
    f_clean = macro_f1(*clean, thr)
    f_noisy = np.mean([macro_f1(*n, thr) for n in noisy_list])
    return 0.3 * f_clean + 0.7 * f_noisy, f_clean, f_noisy
```

### 13.2 Fold calibration into `fc2` (spoof = index 0)

```python
# d = z_spoof - z_bona ; calibrated d' = d / T + beta  →  P(spoof) = sigmoid(d')
with torch.no_grad():
    model.fc2.weight /= T
    model.fc2.bias /= T
    model.fc2.bias[0] += beta      # shifting only the spoof logit shifts d by beta
torch.save(model.state_dict(), "calibrated.pth")
```
Fit (T, β) by grid search maximising `wf1(...)` on the local proxy. Compute the logits once and reuse them.

### 13.3 Multi-crop inference with the current head

```python
def multicrop_logits(model, wav, win=64600, hop=32300):
    if len(wav) < win:
        wav = np.tile(wav, int(np.ceil(win / len(wav))))[:win]
    starts = list(range(0, len(wav) - win + 1, hop)) or [0]
    crops = torch.stack([torch.from_numpy(wav[s:s + win]) for s in starts]).float()
    # NOTE: for qwen3_asr/w2v_bert, run each crop through the same AutoFeatureExtractor as data_utils._featurize
    with torch.no_grad():
        z = model(crops.to(model.input_device))         # (n_crops, 2)
    return z.mean(0)                                    # mean logits; also try median / top-k spoof
```

### 13.4 Paired consistency step (sketch for `main_train.py`)

```python
# batch from SpoofAudioDataset(pair_map=...) → (x_a, x_b, y)   (x_b = online twin or sim view)
z_a, h_a = model(x_a, return_emb=True)   # add return_emb to ModelSLS.forward (pooled embedding)
z_b, h_b = model(x_b, return_emb=True)
p_a, p_b = z_a.softmax(-1), z_b.softmax(-1)
m = 0.5 * (p_a + p_b)
js = 0.5 * (F.kl_div(m.log(), p_a, reduction="batchmean") + F.kl_div(m.log(), p_b, reduction="batchmean"))
emb = 1 - F.cosine_similarity(h_a.detach() if clean_teacher else h_a, h_b, dim=-1).mean()
loss = ce(z_a, y) + ce(z_b, y) + lam_emb * emb + lam_js * js
```

### 13.5 Layer-wise LR decay param groups

```python
def llrd_groups(model, base_lr, head_lr, decay=0.85, wd=0.01):
    layers = model.ssl_model.layers            # list of transformer blocks (adapt per backbone)
    n = len(layers)
    groups = []
    for i, blk in enumerate(layers):
        groups.append({"params": [p for p in blk.parameters() if p.requires_grad],
                       "lr": base_lr * decay ** (n - 1 - i), "weight_decay": wd})
    head = [p for m in (model.sls, model.first_bn, model.fc1, model.fc2) for p in m.parameters()]
    groups.append({"params": head, "lr": head_lr, "weight_decay": wd})
    return groups   # also add embeddings / conv front-end with the lowest LR
```

### 13.6 LoRA via `peft`

```python
from peft import LoraConfig, inject_adapter_in_model
cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                 target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])  # verify names: print(audio_tower)
model.ssl_model.model = inject_adapter_in_model(cfg, model.ssl_model.model)
# freeze everything but LoRA + head; later: merge_and_unload() → full FT stage
```
Careful: the head also has `fc1`/`fc2`. Restrict `target_modules` to the backbone (inject only into `ssl_model.model`, as above).

### 13.7 WiSE-FT

```python
pre = ModelSLS(args, device).state_dict()            # pretrained backbone + (unused) random head
ft = torch.load("finetuned.pth", map_location="cpu")
mix = {k: ((1 - a) * pre[k].float() + a * ft[k].float()).to(ft[k].dtype)
       if k.startswith("ssl_model.") and ft[k].is_floating_point() else ft[k]
       for k in ft}
```

### 13.8 Attentive statistics pooling head

```python
class ASPHead(nn.Module):
    def __init__(self, d, hidden=256, p=0.2):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(p), nn.Linear(2 * d, 256),
                                 nn.GELU(), nn.Dropout(p), nn.Linear(256, 2))
    def forward(self, x, mask=None):                 # x: (bs, T, d) from SLS fusion
        a = self.att(x).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e4)
        w = a.softmax(-1).unsqueeze(-1)
        mu = (w * x).sum(1)
        sd = ((w * (x - mu.unsqueeze(1)) ** 2).sum(1)).clamp_min(1e-6).sqrt()
        h = torch.cat([mu, sd], -1)
        return self.out(h), h                        # logits, embedding (for consistency loss)
```

### 13.9 Simulator building blocks: libraries & commands

Verify versions and licenses before use.

| Stage | Option(s) |
|---|---|
| WebRTC APM (AEC3/NS/AGC2/HPF) | `livekit` Python SDK `rtc.AudioProcessingModule`; `webrtc-audio-processing` (pip, older APM); or a C++ CLI built from the WebRTC `audioproc_f` tool |
| VAD | `webrtcvad` (py-webrtcvad), Silero VAD (check: trained on speech data, but used only as a tool, not training data) |
| AEC alternative | `speexdsp` Python bindings (EchoCanceller + preprocess denoise/AGC) |
| NS | ffmpeg `arnndn` (RNNoise models), `afftdn`, `anlmdn`; DeepFilterNet (`deepfilternet` pip) |
| AGC / dynamics | ffmpeg `dynaudnorm`, `acompressor`, `alimiter`, `loudnorm` |
| Codecs | ffmpeg `libopus` (`-application voip -b:a 12k -frame_duration 20 -packet_loss 10 -fec 1 -dtx 1`), `libspeex`, `g722`, `libopencore_amrnb`, `libvo_amrwbenc`, `pcm_mulaw/alaw`, `libgsm`; SILK via `pysilk` / `pilk` |
| Packet loss + PLC | `opuslib` encode → drop packets with a Gilbert–Elliott model → decode with FEC / PLC (libopus conceals when the decoder gets a lost frame) |
| RIR / echo | `pyroomacoustics` (ShoeBox, random geometry), measured RIR sets (BUT ReverbDB, OpenAIR, AIR) |
| EQ / band-limit | ffmpeg `equalizer`, `highpass`, `lowpass`, `aresample`; `torchaudio.functional` biquads (GPU) |
| GPU augmentation | `torch-audiomentations` (gain, bandpass, colored noise, background noise) |
| Noise corpora | ESC-50, FSD50K (drop "Human voice" ontology classes), MUSAN noise+music, DEMAND, DNS-Challenge noise (check for speech), RNNoise noise (⚠️ verify babble) |

**Code hooks in this repo:**
- `utils/rtc_augment.py`
  - `RTCAugmenter._stage` → becomes `_chain`, applying the stages in pipeline order.
  - Extend `CODEC_PROFILES` with a `codec` key, and add `packet_loss` / `fec` / `dtx`.
  - Keep `codecs_augm` as the codec stage.
  - Write per-sample metadata so GroupDRO / DANN / aux-head labels are available.
- `utils/data_utils.py`
  - `SpoofAudioDataset._augment`: return augmentation metadata.
  - Pass `pair_map` through `build_dataset_from_protocol` for train.
- `model/sls_model.py`
  - `SLS`: add softmax normalisation and layer dropout.
  - `ModelSLS`: add a `head={flatfc, asp, mhfa}` option and `return_emb`.
- `main_train.py`: optimizer / scheduler block → AdamW + LLRD + warmup-cosine + EMA + clip + bf16; select by local WF1.
- `main_eval.py`: `merge_checkpoints` → greedy soup + BN recompute; `--calib T beta`; `--multicrop`; default `--arch sls`.

---

## 14. References

IDs marked † were recalled from memory or only seen as abstracts; verify before citing in the system description.

**Challenge / dataset**
- RTCFake: Speech Deepfake Detection in Real-Time Communication (Findings of ACL 2026). [2604.23742](https://arxiv.org/abs/2604.23742)
- RTC-SDD challenge site: <https://www.junxue.tech/rtc-sdd-challenge> · baseline: <https://github.com/JunXue-tech/RTC-SDD> · data: <https://huggingface.co/datasets/JunXueTech/RTCFake>
- TFCL: Time-Frequency Consistency Learning under WebRTC processing. [2607.17761](https://arxiv.org/abs/2607.17761) · <https://github.com/JunXue-tech/TFCL>
- Noise sources: RNNoise data <https://media.xiph.org/rnnoise/data/> · CLAD <https://github.com/CLAD23/CLAD> · ESC-50 (Piczak 2015)

**SSL front-ends / back-ends**
- XLS-R + AASIST + RawBoost (Tak et al.). [2202.12233](https://arxiv.org/abs/2202.12233) · RawBoost [2111.04433](https://arxiv.org/abs/2111.04433)
- SLS (Zhang et al., ACM MM 2024). DOI 10.1145/3664647.3681345
- Attentive merging of hidden embeddings (Pan et al., IS 2024). [2406.10283](https://arxiv.org/abs/2406.10283)
- Layer selection & fusion for OOD ADD. [2509.12003](https://arxiv.org/abs/2509.12003) · Probing-guided layer selection [2606.30791](https://arxiv.org/abs/2606.30791)
- MHFA / CA-MHFA. [2409.15234](https://arxiv.org/abs/2409.15234), [2512.12851](https://arxiv.org/abs/2512.12851)
- Nes2Net. [2504.05657](https://arxiv.org/abs/2504.05657) · XLSR-Mamba [2411.10027](https://arxiv.org/abs/2411.10027)
- WavLM-based ASVspoof5 systems. [2409.05032](https://arxiv.org/abs/2409.05032), [2408.07414](https://arxiv.org/abs/2408.07414)
- AT-ADD 2026 benchmark / challenge summary (WaveShield: w2v-BERT 2.0, LoRA → FT). [2608.23437](https://arxiv.org/abs/2608.23437), [2608.14249](https://arxiv.org/abs/2608.14249)
- Wavelet prompt tuning (WPT-SSL, AAAI 2026). [2504.06753](https://arxiv.org/abs/2504.06753)
- Whisper features for ADD. [2306.01428](https://arxiv.org/abs/2306.01428) · ALLM4ADD [2505.11079](https://arxiv.org/abs/2505.11079) · MLLMs for ADD [2601.00777](https://arxiv.org/abs/2601.00777)
- Dual-domain SSL fusion. [2608.29021](https://arxiv.org/abs/2608.29021) · Dasheng [2406.06992](https://arxiv.org/abs/2406.06992)† · BEATs [2212.09058](https://arxiv.org/abs/2212.09058)† · EAT [2401.03497](https://arxiv.org/abs/2401.03497)†
- SUPERB-style SSL benchmark for ADD. [2603.01482](https://arxiv.org/abs/2603.01482)†

**Robustness: codecs, channels, noise**
- ASVspoof 5 overview / database. [2601.03944](https://arxiv.org/abs/2601.03944), [2408.08739](https://arxiv.org/abs/2408.08739)
- ADD-C codec + packet-loss benchmark. [2504.12423](https://arxiv.org/abs/2504.12423) · MGAA [2508.01467](https://arxiv.org/abs/2508.01467)
- Robustness of ADD under corruptions. [2503.17577](https://arxiv.org/abs/2503.17577) · Speech enhancement vs ADD [2603.14767](https://arxiv.org/abs/2603.14767)
- DKDSSD dual-branch KD for noise-robust detection. [2310.08869](https://arxiv.org/abs/2310.08869)
- SNR / noise-aware ADD benchmark. [2512.13744](https://arxiv.org/abs/2512.13744)†
- CLAD contrastive anti-manipulation. [2404.15854](https://arxiv.org/abs/2404.15854)
- DeePen (echo / time-stretch attacks). [2502.20427](https://arxiv.org/abs/2502.20427)
- Frequency-mask augmentation (ASVspoof5). [2408.06922](https://arxiv.org/abs/2408.06922)†
- Codec augmentation in SSL pretraining. [2501.05545](https://arxiv.org/abs/2501.05545)†
- Channel-robust anti-spoofing (adversarial channel augmentation). [2104.01320](https://arxiv.org/abs/2104.01320)
- Codecfake / CodecFake+. [2405.04880](https://arxiv.org/abs/2405.04880), [2501.08238](https://arxiv.org/abs/2501.08238)
- Silence shortcut ("Speech is Silver, Silence is Golden"). [2106.12914](https://arxiv.org/abs/2106.12914)

**Training strategies**
- SAM for speech deepfake detection (IS 2025). [2506.11532](https://arxiv.org/abs/2506.11532) · SAM [2010.01412](https://arxiv.org/abs/2010.01412) · multi-dataset + SAM [2305.19953](https://arxiv.org/abs/2305.19953)
- MoE-LoRA / MoLEx / AMULET. [2509.13878](https://arxiv.org/abs/2509.13878), [2509.09175](https://arxiv.org/abs/2509.09175), [2503.12010](https://arxiv.org/abs/2503.12010) · SSL → MoE [2606.14639](https://arxiv.org/abs/2606.14639)
- Wav2DF-TSL. [2509.04161](https://arxiv.org/abs/2509.04161)
- DANN [1505.07818](https://arxiv.org/abs/1505.07818) · GroupDRO [1911.08731](https://arxiv.org/abs/1911.08731) · IDFE [2603.18657](https://arxiv.org/abs/2603.18657)†
- OC-softmax [2010.13995](https://arxiv.org/abs/2010.13995) · SAMO [2211.02718](https://arxiv.org/abs/2211.02718) · Logit adjustment [2007.07314](https://arxiv.org/abs/2007.07314)
- LoRA [2106.09685](https://arxiv.org/abs/2106.09685) · DoRA [2402.09353](https://arxiv.org/abs/2402.09353) · LP-FT [2202.10054](https://arxiv.org/abs/2202.10054)
- Model soups [2203.05482](https://arxiv.org/abs/2203.05482) · WiSE-FT [2109.01903](https://arxiv.org/abs/2109.01903) · SWA [1803.05407](https://arxiv.org/abs/1803.05407) · R²M merging for deepfakes [2509.24367](https://arxiv.org/abs/2509.24367)† · TIES [2306.01708](https://arxiv.org/abs/2306.01708) · DARE [2311.03099](https://arxiv.org/abs/2311.03099)

**Cross-domain**
- AugMix [1912.02781](https://arxiv.org/abs/1912.02781) · ImageNet-C [1903.12261](https://arxiv.org/abs/1903.12261) · DeepAugment / ImageNet-R [2006.16241](https://arxiv.org/abs/2006.16241) · PixMix [2112.05135](https://arxiv.org/abs/2112.05135)
- Worst-of-k / affinity-diversity of augmentation [2002.08973](https://arxiv.org/abs/2002.08973)† · AdvProp [1911.09665](https://arxiv.org/abs/1911.09665)†
- V-JEPA [2404.08471](https://arxiv.org/abs/2404.08471)† · A-JEPA [2311.15830](https://arxiv.org/abs/2311.15830)† · Audio-JEPA [2507.02915](https://arxiv.org/abs/2507.02915)† · data2vec [2202.03555](https://arxiv.org/abs/2202.03555)† · BYOL [2006.07733](https://arxiv.org/abs/2006.07733)
- AudioPure (diffusion purification) [2303.01507](https://arxiv.org/abs/2303.01507)† (not recommended here)

**Recent challenges**
- ASVspoof 5 systems: Vicomtech-UGR [2408.10361](https://arxiv.org/abs/2408.10361) · XMUspeech [2509.18102](https://arxiv.org/abs/2509.18102)†
- RADAR 2026 [2605.09568](https://arxiv.org/abs/2605.09568)† · SAFE challenge [2510.03387](https://arxiv.org/abs/2510.03387)† · Deepfake-Eval-2024 [2503.02857](https://arxiv.org/abs/2503.02857)
