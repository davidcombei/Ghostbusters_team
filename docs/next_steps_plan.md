# RTC-SDD: Next-Steps Plan (robustness to noise, unseen platforms, unseen generators)

> Status: living document · Created 2026-10-05 · Companion to `docs/experiments_roadmap.md` (IDs such as E1.3, A2.x, T3.x refer to it)

## Context

Phase 0 of `docs/experiments_roadmap.md` is done: frozen local proxy sets, `utils/metrics.py`, `scripts/local_eval.py`, the shortcut audit and per-epoch WF1 selection. The two screens so far, B0 (RTC aug) and L1 (B0 + level aug, A2.7a), give the first evidence-based picture. That picture changes the priorities in the roadmap.

| Finding (from `exp/B0_*`, `exp/L1_*` local_eval + metrics.jsonl) | Consequence |
|---|---|
| **Calibration gap is small (0.9–1.5 F1); the separability gap is large (12–15 F1).** Noisy EER is 20–23%. | H1 (prior collapse) is mostly refuted for RTC-aug models. Calibration (E1.1/E1.2) is demoted to a free final step, not a lever. |
| **The dominant error is bonafide false alarms under unfamiliar degradation.** Heldout R_bon is 61% (B0) / 45% (L1), while R_spf is ~90%. Worst slices: GSM 50%, vacuum / washing machine / clock alarm / door creaks ≈ 50%, English bona 50–60%. In the `noise_only` probe, 67% of bona-source clips are scored as spoof. | The model has learned "unfamiliar processing ⇒ spoof". This is the #1 target. It matches 2603.14033: detectors detect processing more reliably than they detect fakeness. |
| **Epoch-to-epoch noise is larger than the effects we are measuring.** B0 per-epoch WF1 ranges from 74.3 to 81.6; heldout R_bon from 33% to 84%; clean R_bon from 53% to 93% (constant LR, no EMA, bs 16). | B0 vs L1 (−0.56 WF1) **is not a result**: it is inside the noise. No ablation is readable until variance is reduced and CIs are reported. |
| L1 removed the level shortcut (level_flip_bona 0.119 → 0.032) but created a **silence ⇒ spoof** behaviour (`sil_only` P≥.5 = 1.00 for both classes; B0: 0.02 / 0.06). | Level aug did what it was meant to do mechanistically. Keep it as a candidate, but re-test it under the stable recipe and watch the silence probe. |
| There is **no unseen-generator proxy**, and the metadata has no generator labels. Train has only 1.5k English offline bona vs 7.5k spoof. | Build a pseudo-generator hold-out. Bonafide diversity (especially English) is scarce. |
| Infrastructure: container restarts kill runs, and there is no resume. Compute: 4090 + 5090 locally, plus a teammate's GPUs. | Resume support is needed. Two seeds per decision is affordable if each screen is short. |

**Goal.** A systematic loop in which every change has a stated hypothesis, a predicted signature in the diagnostics, and a statistically readable outcome. The loop targets (1) bonafide false alarms under noise/DSP, (2) unseen platforms, and (3) unseen generators.

---

## 1. Experimental protocol (applies to everything below)

- **Reference ladder.** REF-1 is defined by the stability recipe (Wave 1). Each experiment changes one factor against the current REF. A winner becomes REF-k+1, and the promotion is logged in `docs/experiment_log.csv`.
- **Screen (S).** 50% stratified train subset (label × lang × src), 6 epochs, warmup-cosine, EMA. Target ≈1.5 h.
  - **2 seeds per arm:** seed A on the 4090 and seed B on the 5090 for *every* arm, so the GPU type is balanced across arms.
  - Once, check that screen rank matches full-run rank (Spearman on 4 configs).
- **Primary readout.** `half_B` WF1@0.5 of the **EMA weights at the final epoch**. There is no checkpoint selection, so no selection noise. `best_by_wf1` is secondary.
- **Decision rule.** Adopt only if the **paired bootstrap 95% CI of ΔWF1 (half_B, utterance-level resampling, both seeds pooled) excludes 0**, and clean F1 drops by no more than 0.5. Otherwise the result is "no effect". Kill an idea after 2 null screens unless there is a bug hypothesis.
- **Diagnostic signature, required for every run.** Before the run, write down which of these should move:
  - calibration gap vs separability gap (F1@oracle, EER);
  - heldout R_bon vs R_spf. A pure threshold shift moves them in opposite directions with EER flat, and that does **not** count as an improvement;
  - per-family / per-codec / per-language slices;
  - probes: `noise_only` bona→spoof rate, `sil_only`, level flips.
- **Integrity.** Training augmentation never uses anything in `utils/heldout_reserve.py` (extended in Wave 1, see 2.4). The progress leaderboard is used only for correlation checks, at most once per milestone.

---

## 2. Wave 1 (Oct 5–11): make results readable. Implement while the GPUs run zero-training evals.

### 2.1 Infrastructure (CPU / implementation)

| Item | What | Files |
|---|---|---|
| I1 Resume | Save model + optimizer + scheduler + EMA + epoch + RNG every epoch; add `--resume <run_dir>` | `SLS_setup/main_train.py` |
| I2 Paired bootstrap / CIs | `paired_bootstrap(scores_a, scores_b, sets, n=2000)` → ΔWF1 CI; per-set CIs; McNemar per set. Also a `compare` mode in `scripts/local_eval.py` that reads two cached score dirs. | `SLS_setup/utils/metrics.py`, `scripts/local_eval.py` |
| I3 Stability recipe flags | `--optim adamw`, `--warmup_frac`, `--lr_scheduler cosine`, `--grad_clip`, `--ema_decay` (evaluate and save the EMA weights), `--head_lr`, `--llrd`, `--grad_accum`, `--bf16`, `--train_subset_frac` (stratified) | `main_train.py`; reuse the §13.5 LLRD snippet in the roadmap |
| I4 Augmentation metadata | `RTCAugmenter` / `LevelAugmenter` return per-sample metadata (family, SNR, codec, DSP), passed through the dataset; aggregated `stage_counts` logged per epoch. This is a prerequisite for consistency, aux heads and GroupDRO. | `utils/rtc_augment.py`, `utils/level_augment.py`, `utils/data_utils.py` |
| I5 Pseudo-generator proxy (UG-proxy) | Mean-pool frozen Qwen3-ASR mid-layer features (≈ layers 8–12) of **offline spoofs** in train + dev. K-means, K≈10, after per-language centring, so clusters are not just language. Map online twins via `*_offline_online_pairs.csv`. Check that clusters are not speaker- or duration-driven (cluster vs duration / RMS / lang tables). Write `data/ug_proxy/clusters.csv` and a LOGO protocol: train without clusters {a, b}; evaluate on dev spoofs of {a, b} + all dev bona, clean and through the frozen matched/heldout recipes. | new `scripts/build_ug_proxy.py`; reuse `scripts/layerwise_logreg_probe.py` feature extraction and `scripts/build_dev_noisy_sim.py` rendering |
| I6 Score the 83.20 model | The teammate scores the 83.20 merge with `scripts/local_eval.py --names <17 sets>`. This gives the first local↔progress anchor. | — |

### 2.2 Zero-training GPU evals (on the B0 / L1 checkpoints, hours)

- **Z1 Multi-crop inference** (E1.3): first crop vs centre crop vs 50%-overlap mean logit vs top-k. Hypothesis H4: localised transients. Signature: heldout keyboard / door / clock slices improve.
- **Z2 WiSE-FT α ∈ {0.3, 0.5, 0.7, 0.9}** (E1.5, roadmap §13.7). Hypothesis H3. Signature: heldout R_bon ↑ with clean flat.
- **Z3 Uniform soup** of the last 3–5 epochs + BN recompute (E1.4). This also estimates how much EMA will buy.
- **Z4 Calibration fold-in** (E1.1). Record the gain only to confirm it is ≤ 1.5 as measured.

### 2.3 GPU screens

- **R1 = B0 + stability recipe** (AdamW, warmup 5% → cosine, clip 1.0, EMA 0.999, head LR 10×, bf16), 2 seeds. Measure the per-epoch WF1 standard deviation, heldout R_bon range and seed-to-seed Δ. Exit criterion: seed Δ < 1 WF1. R1 becomes REF-1, and its two seeds give the **noise floor** used by every later decision.
- **R1+L** (level aug on REF-1), 2 seeds. This is the properly powered re-test of A2.7a.

### 2.4 Heldout integrity before any DSP is added to training

`sim_heldout_v1` uses `sim_ops` energy-VAD, repeat-PLC, `afftdn` / `anlmdn`, `dynaudnorm` / `acompressor`, Butterworth band-limit and G.722 / G.726 / GSM. Training DSP (Wave 2) must use **different implementations**:
- WebRTC APM NS/AGC2, `webrtcvad`, libopus in-band PLC, RNNoise `arnndn`, DeepFilterNet, speexdsp;
- add the `sim_ops` VAD / PLC / NS / AGC functions and `afftdn` / `dynaudnorm` to `heldout_reserve.py`, and assert this in the augmenter.

Then heldout stays a "DSP family never seen" test. Install new dependencies into the `rtc-sdd` conda env, never the system, because restarts wipe the container layer.

---

## 3. Wave 2 (Oct 12–18): attack bonafide false alarms under degradation (core track)

Each item is run as an S-screen, 2 seeds, against REF-1.

| ID | Change | Hypothesis | Predicted signature | Implementation notes |
|---|---|---|---|---|
| C1 | **Class-balanced sampler** (bona ≈ 50% of each batch, language-balanced), CE 0.5/0.5, augmentation still class-independent | False alarms come from too few distinct *degraded bona* views (only 14k bona; 1.5k en offline) | heldout + en R_bon ↑, EER ↓ (not just a threshold shift) | `WeightedRandomSampler` in `main_train.py`; keep the CE-weight arm as the control |
| C2 | **Paired consistency** (T3.1): two views of the same utterance (clean-ish ↔ RTC-aug, plus offline ↔ online twin). Loss: CE + λ_js·JS + λ_emb·cos to a stop-grad/EMA "clean" view. λ ∈ {0.3, 1.0}. | Forces invariance to degradation, so degradation can't be used as a spoof cue | noisy EER ↓, `noise_only` bona→spoof ↓, clean ≈ flat | `SpoofAudioDataset(pair_map=...)` exists; extend `train_epoch` to 2-view batches; add `return_emb` to `ModelSLS.forward` (`model/sls_model.py`) |
| C3 | **Factorised processing-status auxiliary.** (a) Multi-task head predicting the augmentation family (from I4 metadata); (b) the same head behind a gradient-reversal layer (DANN). | Separating "processed" from "fake" (2603.14033; 4-class supervision in 2512.13744) | (b) should beat (a) on heldout R_bon if the shortcut hypothesis holds | Small head on the pooled embedding; inference unchanged (single model) |
| C4 | **SNR range extension**: 5–20 → Beta-skewed over [−5, 25] dB | SNR 0 is the worst matched slice (67.6) and was never trained | gain concentrated in the 0/5 dB slices | `RTCAugConfig.snr_db` + a flag |

The winner(s) are combined into REF-2. Before promotion, run a 2×2 factorial check on C1 × C2, because they may be redundant.

---

## 4. Wave 3 (Oct 19–25): augmentation v2 for unseen platforms

Run as add-one-in screens on REF-2. The final stack then gets a **leave-one-out** confirmation (more robust to interactions than add-one-in alone). Keep a stage only if it improves **heldout or echo**, not matched.

| ID | Stage (training-only implementations, see 2.4) | Roadmap ref |
|---|---|---|
| D1 | Chained acoustic stages, depth 1–3, p(noisy) 0.65 | A2.1 |
| D2 | Platform DSP: WebRTC APM NS levels / AGC2 / HPF, RNNoise, DeepFilterNet; noise → DSP → codec order | A2.2, A2.5 |
| D3 | VAD gating / DTX / CNG (`webrtcvad`); random silence trim/insert. Also fixes the L1 silence probe. | A2.3, A2.9 |
| D4 | Codec zoo: Opus FEC/DTX/frame-size variants + libopus PLC under burst loss, SILK (`pysilk`), AMR-NB/WB, Speex, G.711 | A2.6 |
| D5 | Receiver EQ from the paired-data platform transfer functions + band-limit | A2.7, A2.8 |
| D6 | Mel SpecAugment / band dropout on the Qwen input (GPU, free) | A2.10 |

**Throughput.** If D2–D4 make the loader the bottleneck, pre-render K=6 heavy-chain variants per utterance on CPU (roadmap §5.3) while the GPUs run Wave 2.

---

## 5. Wave 3–4 (Oct 19–Nov 1): unseen generators and fine-tuning style

- **G0 Baseline LOGO gap.** Train REF-2 on the I5 LOGO split (2 clusters out), once. Gap = in-distribution spoof recall − held-out-cluster spoof recall. This becomes the generator-generalisation metric for G1–G4. Use LOGO runs only for these items, because they double cost.
- **G1 Cluster-balanced spoof sampling** (DOSS-Weight-style, 2512.18210: diversity beats volume). Equalise pseudo-generator clusters inside the spoof half of the batch.
- **G2 Preserve pretrained features** (H3): LLRD 0.85 vs freeze bottom-8 vs LoRA r=32 (`peft`, roadmap §13.6) → staged LoRA → FT. Signature: LOGO gap ↓ and heldout ↑, with the frozen-0.6B > finetuned-0.6B evidence as the prior.
- **G3 SAM/ASAM** (ρ 0.05) on the best G2 arm. Sharpness predicts the domain gap (2506.11532).
- **G4 Backbone shoot-out, cheap first.** Frozen linear probe on matched / heldout / LOGO (`scripts/layerwise_logreg_probe.py`) for w2v-BERT 2.0 (already wired), WavLM-L and Whisper-v3 encoder vs Qwen3-ASR-1.7B. Only the top 1 gets an S-screen. The teammate's GPUs can take this track in parallel.
- **M1 ASP head + full-utterance / 4–8 s crops** (M4.1, T3.8). Do this only if Z1 shows that multi-crop helps; otherwise it is low priority.

---

## 6. Wave 5–6 (Nov 2–16): consolidation

- Combine the winners: a full schedule, 3 seeds, a greedy soup of the EMA checkpoints across seeds (with BN recompute), optional WiSE-FT α, then calibration fold-in last.
- One progress submission of the final candidate, checked against the local↔progress anchor (I6).
- Freeze by Nov 7; run inference on eval Nov 9+; submit by Nov 14.

---

## 7. GPU allocation

| Slot | Local (4090 = seed A, 5090 = seed B) | Teammate GPUs |
|---|---|---|
| Wave 1 | Z1–Z4 (hours), then R1 ×2, R1+L ×2 | I6 scoring of 83.20; G4 frozen probes |
| Wave 2 | C1, C2, C3a/b, C4 (×2 seeds each ≈ 12 h of screens) | G0 LOGO baseline |
| Wave 3 | D1–D6 add-one-in, then leave-one-out | G1/G2 on LOGO |
| Wave 4 | G2/G3 on the main track, M1 | backbone S-screen |

---

## 8. Critical files

- `SLS_setup/main_train.py`: resume, optimiser recipe, EMA, sampler, 2-view consistency loop, aux/GRL head losses, subset flag.
- `SLS_setup/model/sls_model.py`: `return_emb`, aux head + GRL, LoRA injection, ASP head.
- `SLS_setup/utils/data_utils.py`: pair views, metadata passthrough, balanced sampler weights.
- `SLS_setup/utils/rtc_augment.py`, `utils/level_augment.py`, new `utils/rtc_dsp.py` (training-only APM/VAD/PLC/codec stages, kept separate from `utils/sim_ops.py`, which stays heldout-reserved).
- `SLS_setup/utils/heldout_reserve.py`: extend the reserved list.
- `SLS_setup/utils/metrics.py`, `scripts/local_eval.py`: paired bootstrap, compare mode.
- New `scripts/build_ug_proxy.py`; `docs/experiment_log.csv` (add `seed`, `ci_low`, `ci_high`, `half_b_wf1` columns).

## 9. Verification

- **I1.** Kill a screen mid-epoch, `--resume`, and confirm the loss curve continues and the epoch count is preserved.
- **I2.** Bootstrap CI of a run against itself contains 0, with width ≈ the expected ±1 WF1. B0 vs L1 should come out non-significant, which also validates the earlier conclusion.
- **I3.** R1's two seeds differ by < 1 half_B WF1, and per-epoch WF1 std < 1.
- **2.4.** Augmenter start-up raises if a reserved op is configured. Grep the training logs for `failed (` and check `which ffmpeg` after restarts (a container restart wipes the system ffmpeg, and the codec stage then silently passes audio through).
- **I5.** A cluster sanity table (size, lang, duration, RMS per cluster). Every LOGO protocol has no spoof from held-out clusters in train.
- **Every run.** `scripts/local_eval.py --names <the 17 report sets>`, a probe report, and a registry row with seed and CI.

## References (new in this plan)

- Bokkahalli Satish et al., "What Counts as Real? Speech Restoration and Voice Quality Conversion Pose New Challenges to Deepfake Detection". [2603.14033](https://arxiv.org/abs/2603.14033)
- "Toward Noise-Aware Audio Deepfake Detection: Survey, SNR-Benchmarks, and Practical Recipes" (four-class supervision). [2512.13744](https://arxiv.org/abs/2512.13744)
- Huang, Mao, Qian, "A Data-Centric Approach to Generalizable Speech Deepfake Detection" (DOSS, ACL 2026). [2512.18210](https://arxiv.org/abs/2512.18210)
- SAM for speech deepfake detection. [2506.11532](https://arxiv.org/abs/2506.11532)
