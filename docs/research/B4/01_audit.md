# rdloop B4 - gap audit of B2f

> Written 2026-10-17 by the rdloop B4 AUDIT step (research-auditor, attempt a0). Every number is copied from a file
> read in this session, and the file is named next to it. Numbers marked **(computed)** are simple arithmetic on
> copied numbers, or counts I made with read-only `python3 -I` over local score files (`scores/*.txt`) joined with the
> sim `meta.csv` (half B only). Hypotheses are labelled **hypothesis**. I did not read any progress-split score
> (`model_merging_*/scores.txt`, `progress.txt`). The brief's "clean vs noisy gap seen on Codabench" is measured here
> only on the local proxy.
>
> **Files read.** Cycle: `rdloop/B4/{cycle.json,brief.md,user_notes.md}` (no owner notes yet).
> Base run `SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623/` (= **RUN**):
> `config.yaml`, `metrics.jsonl`, `logs/train.log`, `top5_by_wf1/merge_list.txt`,
> `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,scores/*.txt}`,
> `local_eval/merge_top5/{compare_all_readouts.txt,compare_vs_B2e-top5.txt}`. There is no `compare_ci*.json` in RUN.
> B3a: `SLS_setup/exp/B3a_…_20261009214846/local_eval/{merge_top5,epoch_8}/metrics.json` and
> `merge_top5/compare_ci_vs_B2f-top5.json`. Sim metadata: `data/dev_noisy_sim/{matched_v1,heldout_v1,echo_v1}/meta.csv`
> and `dev_online_clean/protocol.txt`.
> Registry `docs/experiment_log.csv`. Waves: `exploop/B3/{loop_state.json,wave.json,wave_table.md,plan_snapshot.md,user_notes.md}`.
> Reports: `docs/experiments-exploration/b3-experiments/20261012_012259_B3_wave_summary.md` (**B3S**), the B3d and B3e
> reports §5.5 / §5.5-5.6, `docs/experiments-exploration/20261009_175355_B2f_…md` (**B2fR**),
> `docs/experiments-exploration/20261008_232331_B2_wave_summary_and_full_run_recommendation.md` (**B2S**) §3 and §5.
> Context: `docs/experiments_roadmap.md` §0-2, §5, §11; `docs/next_steps_plan.md`; `docs/dev_splits.md`;
> `docs/augmentations.md` §1-3, §10; `docs/research/{lever_ledger.md,literature.md,README.md}`;
> `SLS_setup/b3wave/{README.md,heldout_reserve.py,augment.py (l.110-140)}`; `SLS_setup/utils/heldout_reserve.py`;
> `scripts/build_probe_sets.py` (header); `scripts/b3_build_unseen_set.py` (header).
>
> Abbreviations: **M** = `RUN/local_eval/merge_top5/metrics.json`, **E** = `RUN/local_eval/epoch_8/metrics.json`,
> **PM / PE** = the matching `probe_report.json`, **SC** = my counts from `RUN/local_eval/<readout>/scores/<set>.txt`
> joined with `data/dev_noisy_sim/<set>/meta.csv` rows with `half == B`. "merge / final" pairs are always merge first.

## 1. Base

| item | value | source |
|---|---|---|
| label | **B2f** (registry `B2f-loudnorm-screen` = merge, `B2f-loudnorm-ep8` = final) | `rdloop/B4/cycle.json` |
| run dir | `SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623` | `cycle.json` |
| trainer that produced it | `main_train.py` (original trainer), git `00da485c…`, `git_dirty: true` | RUN `config.yaml` |
| cycle entry module | `b3wave` (`python -m b3wave.train`; with every B3 flag at default it reproduces the original trainer) | `cycle.json` `base_module`; `b3wave/README.md` "Reproduction statement" |
| merged epochs | 8, 5, 7, 3, 4 | RUN `top5_by_wf1/merge_list.txt` |
| final checkpoint | `ckpt/epoch_8_dev_loss_0.144168.pth` | RUN `merge_list.txt`, `metrics.jsonl` |
| run time | 4.14 GPU-h; epochs 30.7-31.2 min | registry `gpu_h`; B2fR §1 |
| augmentation actually used | `p_apply=0.8 one stage/clip, equal chance 1/8 \| snr=5..20 dB \| stages=[coffee, footsteps, keyboard, musan, office, rain, echo, reverb] \| pools=(coffee:60, footsteps:32, keyboard:32, musan:1590, office:993, rain:32) \| rirs=64 \| then codec p=1.0 [qq, wechat, zoom, dingtalk, lark, voov, telegram]`; `rawboost: 'off'`; no `suppress` stage | RUN `config.yaml` |

**Flag diff vs the previous best (B2e).** `--use_level_aug` replaced by `--loudness_norm --loudness_norm_lufs -23`
(read-time BS.1770 normalisation on train, dev and eval), plus `--keep_epochs 8` (no effect) (B2fR §1, registry
`change_vs_ref`).

**Readouts on half B.**

| readout | WF1@0.5 | WF1@oracle | F1 clean | F1 noisy | EER noisy | source |
|---|---|---|---|---|---|---|
| merge (`merge_top5`) | **91.98** (`half_B.wf1.wf1` 0.9197880641320395) | 92.59 | 97.60 | 89.57 | 10.09 | M `half_B.wf1` |
| final (`epoch_8`) | **92.14** (0.9213685698540242) | 92.82 | 97.69 | 89.76 | 9.66 | E `half_B.wf1` |

**The clean-noisy gap the brief asks about:** 97.60 - 89.57 = **8.03 F1** (merge) and 97.69 - 89.76 = **7.93**
(final). At the 0.7 noisy weight that is **5.62 / 5.55 WF1 points** (computed).

**Base-choice facts the plan must keep in mind** (no judgement, only the record):
- B2f is not the numerically best run. The B3 wave's `best` is **B3a** (merge 93.06, final 92.94; Δ vs B2f merge +1.08 [+0.52, +1.67]) (`exploop/B3/loop_state.json` `best`, `wave_table.md`). B2e is also above B2f on half B (92.34 / 92.53; B2f vs B2e = -0.36 / -0.39, "null") (registry; B2fR §6).
- B3a's recipe (paired consistency, `--batch_size 8` pairs) costs 7.87 GPU-h against B2f's 4.14 (registry `gpu_h`); its epochs take 57.7-59.8 min (B3S §4.9).
- **Hypothesis:** B2f was chosen because the cycle's `max_epoch_time_frac 1.3` and `max_run_hours 10` are measured against the base, and a consistency base would not leave room (7.87 × 1.3 = 10.2 h, computed). Consequence for the wave: B4 levers are screened against **91.98 / 92.14**, and B3a's +1.08 is the best measured effect size of anything that closed part of the clean-noisy gap.

## 2. What has been tried

Δ is half_B WF1@0.5 against the reference named; CI = paired bootstrap of the change where logged (registry
`ci_low/ci_high`, `exploop/B3/wave_table.md`). Status: **exhausted** = closed by a report, **open** = positive or
not finished, **untested** = never run.

| # | lever | wave / run | ref | Δ merge | Δ final | CI (merge; final) | verdict | report | status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | RTC aug (noise + echo + reverb + Opus) | B0 | local recipe | merge 83.61 | - | - | promote | registry `B0-merge-top5` | in recipe |
| 2 | level aug (A2.7a) | L1 | B0 | 79.90 vs 80.94 (screen, best_by_wf1) | - | - | "not a result, inside the noise" | `docs/next_steps_plan.md` Context | superseded by #8 |
| 3 | stability recipe + label smoothing 0.05 × CE 0.1/0.9 | B1 | B0-merge | 81.11 (-2.50) | - | - | drop | `20261007_222414_00_B0_vs_B1_postmortem.md`; B2S §3 | **exhausted** |
| 4 | no label smoothing, 8-ep annealed cosine, lr 3e-5 | B2a | B0-merge | 84.95 (+1.33) | - | - | promote | registry | in recipe |
| 5 | lr 1e-5 | B2b | B2a | 89.59 (+4.64) | 90.34 | - | promote | registry | in recipe |
| 6 | lr 5e-6 | B2c | B2b | 91.73 (+2.14) | 92.26 | - | promote | registry | in recipe |
| 7 | 10 epochs | B2d | B2c | 91.57 (-0.16) | 91.84 (-0.42) | - | drop | registry; B2S §3 | **exhausted** |
| 8 | lr 2.5e-6 (LR series turned over) | B2e | B2c | 92.34 (+0.61) | 92.53 (+0.26) | - | revisit | registry; B2S §3 | **exhausted** (LR saturates 5e-6..2.5e-6) |
| 9 | loudness norm -23 LUFS instead of level aug | **B2f** | B2e | 91.98 (-0.36) | 92.14 (-0.39) | - | null (inside ±0.5) | B2fR §6 | **exhausted** (level treatment is not a lever) |
| 10 | paired JS consistency λ 0.3 (`--batch_size 8` pairs) | B3a | B2f | **93.06 (+1.08)** | 92.94 (+0.80) | [+0.52, +1.67]; [+0.22, +1.42] | promote / positive; final revisit (clean drop 0.63) | B3S §2-4 | **open** (only positive lever; confounded with 2× steps, B3S lesson 1) |
| 11 | class + language balanced sampler, CE re-derived | B3b | B3a | 92.23 (-0.83) | 92.36 (-0.58) | [-1.46, -0.25]; [-1.21, +0.02] | negative / drop | B3S lesson 3 | **exhausted** |
| 12 | JS dose λ 1.0 | B3c | B3a | 92.29 (-0.77) | 92.44 (-0.50) | [-1.31, -0.22]; [-1.05, +0.05] | negative / drop | B3S lesson 2 | **exhausted** (above 0.3) |
| 13 | random Opus profiles (`--aug_codec_profiles random`) | B3d | B3a | 90.86 (-2.20) | 90.72 (-2.22) | [-2.90, -1.57]; [-2.99, -1.52] | negative / drop | B3S lesson 4; B3d report §5.5 | **exhausted** |
| 14 | SNR range 0-20 dB, uniform (`--aug_snr_db 0 20`) | B3e | B3a | 91.99 (-1.06) | 92.36 (-0.58) | [-1.65, -0.49]; [-1.13, -0.05] | negative / drop | B3S lesson 4; B3e report §5.6 | **exhausted** (range widening) |
| 15 | cross-LR weight average (B2c-ep8 + B2e-ep8) | B2 zero-training | parents 92.26 / 92.53 | 91.87 | - | - | loses | B2S §3 | **exhausted** |
| 16 | global re-threshold / calibration | B2f zero-training | B2f | best single threshold 92.37 / 92.54 | - | - | ceiling ~0.4 | B2fR §5.2; B2S §3 | ceiling-limited |

**Never run** (from `b3wave/README.md` flags, B3S §6, B2S §5, roadmap §5.2, next_steps_plan §3-4). Status **untested**:

| direction | where it is described | what is in code today |
|---|---|---|
| `--aug_max_stages 2+` (chaining, A2.1 / D1) | roadmap §5.2; B3S §6 "deprioritised" | b3wave flag exists |
| platform DSP in training: NS / AGC / VAD-DTX-CNG / PLC with implementations disjoint from the reserve (A2.2, A2.3, A2.5, D2, D3) | roadmap §5.2; next_steps_plan §2.4, §4 | not in code. The DeepFilterNet `suppress` stage exists in `utils/rtc_augment.py` (`--aug_use_deepfilternet`) but was never enabled (RUN `config.yaml` stage list) |
| echo suite: far-end echo + AEC residual, double-talk (A2.4) | roadmap §5.2 | training `echo` = one delayed copy, delay 40-160 ms, gain 0.15-0.5 (`docs/augmentations.md` §3.2) |
| silence / VAD robustness: trim, insert, replace silence; CNG (A2.9) | roadmap §5.2 | none |
| platform transfer-function EQ from the paired train data, receiver EQ (A2.7, A2.8, D5) | roadmap §5.2 | none |
| babble from training speech (A2.13; rule "likely allowed") | roadmap §2, §5.2 | none |
| SNR shape: `--aug_snr_dist beta` inside 5-20 dB (B3e changed only the range) | b3wave README | b3wave flag exists |
| augmentation-strength curriculum (A2.12) | roadmap §5.2 | none |
| `--specaug`, `--llrd 0.9`, `--consistency_clean_view`, `--consistency_lambda 0` control | B3S §6 ranks 1-4 | b3wave flags exist |
| RawBoost re-check (A2.11) | roadmap §5.2 | `--use_rawboost` exists (off in B2f) |
| multi-crop inference, WiSE-FT | B2S §5 Z-E / Z-F "unknown" | `b3wave.eval --multicrop` exists |

The rdloop ledger's "Built by rdloop cycles" table is empty: B4 is the first rdloop cycle (`docs/research/lever_ledger.md`).

## 3. Error anatomy of the base

### 3.1 Where the missing WF1 is (half B)

| set (role, weight) | F1 merge / final | missing WF1 points = weight × (100 - F1), merge / final (computed) | R_bona | R_spoof | EER | source |
|---|---|---|---|---|---|---|
| `dev_online_clean` (clean, 0.3) | 97.60 / 97.69 | 0.72 / 0.69 | 92.84 / 92.91 | 99.90 / 99.95 | 3.42 / 3.33 | M / E `half_B.sets` |
| `sim_matched_v1` (0.35) | 91.22 / 91.24 | 3.07 / 3.07 | 76.21 / 76.47 | 99.50 / 99.44 | 8.15 / 8.21 | M / E |
| `sim_heldout_v1` (0.35) | 87.92 / 88.27 | **4.23 / 4.11** | **69.31 / 69.82** | 99.00 / 99.13 | 12.03 / 11.11 | M / E |
| `sim_echo_v1` (not in WF1) | 94.16 / 94.53 | - | 83.38 / 84.65 | 99.75 / 99.69 | 4.71 / 4.86 | M / E |

Heldout carries 53 % of the missing 8.02 WF1 points of the merge, matched 38 %, clean 9 % (computed). Spoof recall is
≥ 99 on every set: the gap is bonafide recall.

### 3.2 Error types (half B, counts; SC)

| set | bona n | bona FA @0.5 merge / final | of which p ≥ 0.99 | p ≥ 0.999 | spoof n | spoof misses merge / final |
|---|---|---|---|---|---|---|
| clean | 1,397 | 100 / 99 | 75 / 74 | - | 6,068 | 6 / 3 |
| matched | 391 | 93 / 92 | 57 / 53 | 32 / 35 | 1,603 | 8 / 9 |
| heldout | 391 | 120 / 118 | 79 / 80 | 36 / 37 | 1,603 | 16 / 14 |
| echo | 391 | 65 / 60 | 41 / 39 | 19 / 21 | 1,603 | 4 / 5 |
| **matched + heldout** | 782 | **213 / 210** | **136 / 133** | | 3,206 | 24 / 23 |

- Bona false alarms are 213 of 237 noisy errors of the merge (90 %, computed); 136 of the 213 (64 %) sit at p ≥ 0.99. B2fR §5.2 reports the same 27.2 % FA / 17.4 % saturated share for the merge.
- Oracle gap: 0.61 (merge) / 0.69 (final); one global logit shift recovers 0.39 / 0.40; the best single threshold is 0.965 / 0.980, worth 92.37 / 92.54 (B2fR §5.2). **The remaining error is ranking, not threshold.**

### 3.3 Are the noisy false alarms caused by the degradation? (online-source half-B bona; SC)

Each sim clip with `src == online` has its source clip in `dev_online_clean`, so the same audio can be compared clean vs degraded.

| set | online-source bona | noisy FA merge / final | also FA on the clean source | **noise-induced FA** (clean correct, degraded wrong), merge / final (computed) |
|---|---|---|---|---|
| matched | 190 | 50 / 50 | 15 / 13 | **35 / 37** (70 % / 74 %) |
| heldout | 190 | 58 / 56 | 15 / 13 | **43 / 43** (74 % / 77 %) |
| echo | 190 | 37 / 36 | 13 / 13 | **24 / 23** (65 % / 64 %) |

Every clean false alarm stays a false alarm after degradation. About three quarters of the noisy bona false alarms
are bona clips the model gets right when clean: **the clean-noisy gap is degradation flipping correct bona to spoof.**

### 3.4 Slices (bona recall @0.5, merge / final)

Full sim sets (4,000 clips each) from M / E `breakdowns`, except the DSP rows, which are half B from SC.

| slice | worst values | best / reference values | source |
|---|---|---|---|
| heldout **DSP stage** (half B, n bona) | `afftdn` NS (74) **55.4 / 58.1**; low-pass band-limit (63) 58.7 / 58.7; VAD + CNG (34) 55.9 / 61.8; `dynaudnorm` AGC (53) 64.2 / 62.3; packet loss + repeat-PLC (130) 66.2 / 66.9; chain depth 4 (32) 50.0 / 50.0 | no DSP (128) 72.7 / 72.7; `anlmdn` NS (72) 77.8 / 76.4; 8 kHz resample (38) 76.3 / 73.7 | SC (`heldout_v1/meta.csv` `dsp_chain`) |
| heldout noise family | vacuum_cleaner 60.00 / 57.14; washing_machine 62.82 / 65.38; wind 68.42 / 72.37; pouring_water 69.44 / 69.44 | glass_breaking 85.94 / 82.81; can_opening 81.97 / 81.97 | M / E `breakdowns.sim_heldout_v1.family` |
| heldout codec | gsm 65.85 / 65.45 (EER 14.23 / 15.02); g726 68.29 / 70.33 | g722 81.01 / 79.84 | M / E `.codec` |
| heldout SNR | 0 dB 63.27 / 61.22 (EER 17.49 / 17.07) | 20 dB 79.17 / 75.69 | M / E `.snr_db` |
| matched SNR (training range 5-20 dB) | 0 dB **62.39 / 65.81**; 5 dB 71.19 / 73.73; 10 dB 75.81 / 75.00 | 20 dB 82.68 / 83.46; no noise (echo family) 82.58 / 84.52 | M / E `breakdowns.sim_matched_v1.snr_db` |
| matched noise family (all training families) | rain 66.96 / 67.83; coffee 73.68 / 76.32 | echo 82.58 / 84.52 | M / E `.family` |
| matched Opus profile | qq **61.62 / 66.67**; wechat 65.38 / 71.15 (the two low-rate narrow-band profiles) | zoom 83.02 / 83.96; dingtalk 82.79 / 81.97 | M / E `.codec`; profiles in `docs/augmentations.md` §3.3 |
| source (double processing) | matched online 72.83 / 74.73; echo online 80.43 / 81.25 | matched offline 79.06 / 80.63; echo offline 86.13 / 87.70 | M / E `.src` |
| echo family | self_echo 79.40 / 79.83 (EER 8.05 / 8.41) | farend_aec 87.22 / 88.35 | M / E `breakdowns.sim_echo_v1.family` |
| language, full sets | en: matched 63.57 / 63.18, heldout 59.69 / 60.47, echo 70.93 / 72.48 | zh: 82.52 / 85.37, 78.25 / 78.05, 89.84 / 90.85 | M / E `.lang` |
| language, half B counts (FA / bona) | en: clean 70 / 479 and 65 / 479; matched 43 / 129 and 46 / 129; heldout 52 / 129 and 53 / 129 | zh: matched 50 / 262 and 46 / 262; heldout 68 / 262 and 65 / 262 | SC |

Even matched noise of the training families at training SNRs (10-20 dB) leaves bona recall 75-83, against 92.8 on
clean (rows above). The gap is not only an out-of-support problem.

### 3.5 Probes (187 bona-source + 813 spoof-source clips; one bona clip = 0.53 points)

| probe | bona → spoof, merge / final | mean dP (bona), merge / final | source |
|---|---|---|---|
| `orig` (unmodified) | 10.70 % / 10.16 % | - | PM / PE `pred_spoof_orig` |
| `noise_only` (clip replaced by held-out noise at the clip's RMS) | **100 % / 100 %** | +0.8870 / +0.8965 | PM / PE |
| `sil_only` (speech region zeroed) | **100 % / 100 %** | +0.8929 / +0.8985 | PM / PE |
| `padded` (1 s digital silence at both ends) | flip 3.74 % / 5.35 % | +0.0180 / +0.0182 | PM / PE |
| `trimmed` | flip 3.74 % / 3.74 % | +0.0041 / +0.0042 | PM / PE |
| level summary (`level_flip_bona`) | 1.54 % / 1.74 % | - | PM / PE `summary`. Under `--loudness_norm` the pure-gain probes are nearly identities (B2fR §5.4), so this row does not measure a level shortcut for B2f |

Probe definitions: `scripts/build_probe_sets.py` header. Clips with no speech evidence are called spoof with
near-certainty. That fits "spoof is the default when bona evidence is missing or masked". This is the
**hypothesis** behind the saturated core (§3.2).

### 3.6 Per-epoch behaviour (half-A mini sets, EMA; `RUN/metrics.jsonl`, `RUN/logs/train.log`)

| epoch | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| mini WF1 | 88.34 | 91.47 | 91.90 | 92.49 | 92.80 |
| matched mini R_bona / R_spoof | 81.56 / 93.67 | 77.09 / 98.42 | 74.86 / 99.39 | 78.77 / 99.27 | 80.45 / 99.15 |
| heldout mini R_bona / R_spoof | 78.57 / 92.29 | 76.02 / 96.64 | 71.43 / 98.76 | 71.94 / 99.25 | 73.47 / 98.88 |
| heldout mini EER | 13.92 | 12.90 | **11.84** | 12.53 | 12.60 |
| dev loss / TrainAcc | 0.0927 / 87.36 % | 0.0982 / 97.13 % | 0.1139 / 98.09 % | 0.1178 / 98.96 % | 0.1442 / 99.44 % |

Heldout ranking stops improving at epoch 3 (best mini EER). After that, training buys spoof recall and confidence,
not noisy-bona recall, while dev loss rises 55 % (B3S §2.2). Clean mini R_bona stays 90.45-92.70.

### 3.7 What the best lever (B3a consistency) moved, vs B2f-top5 (`B3a…/merge_top5/compare_ci_vs_B2f-top5.json`, half B, n = 2000)

| set | Δ F1 | 95 % CI |
|---|---|---|
| WF1 | +0.0108 | [+0.0052, +0.0167] |
| clean | -0.0012 | [-0.0044, +0.0020] |
| matched | +0.0090 | [-0.0019, +0.0203] |
| **heldout** | **+0.0228** | **[+0.0110, +0.0347]** |
| echo | +0.0071 | [-0.0039, +0.0177] |

B3a merge half-B R_bona: matched 79.03, heldout 74.94, clean 92.34 (B3a `merge_top5/metrics.json`). Its gain is
noisy-bona recall at unchanged spoof recall. The saturated core did not move: 77 % of its false alarms sit at
p ≥ 0.99 (B3S §3, lesson 6).

## 4. Gaps

Ranked for this cycle's focus (augmentation; brief: close the clean-noisy F1 gap).

### G1 (priority 1). Bona flipped to spoof by unseen platform DSP, noise and codecs (sim_heldout)

- **Evidence:** heldout F1 87.92 / 88.27, R_bona 69.31 / 69.82, EER 12.03 / 11.11 (M / E). Missing WF1 4.23 / 4.11 (computed). DSP-stage R_bona: `afftdn` 55.4 / 58.1, band-limit 58.7, VAD+CNG 55.9 / 61.8, `dynaudnorm` 64.2 / 62.3, loss+PLC 66.2 / 66.9, chain depth 4 50.0, against 72.7 with no DSP (SC). gsm 65.85 / 65.45 and g726 68.29 / 70.33 (M / E). vacuum_cleaner 60.00 / 57.14, washing_machine 62.82 / 65.38 (M / E). 74-77 % of heldout online-source FA are noise-induced (SC §3.3).
- **Size:** 120 / 118 of the 213 / 210 noisy FA. Closing heldout to matched's F1 would add 0.35 × (91.22 - 87.92) = 1.16 WF1 on the merge (computed).
- **Causes (hypotheses):** (a) the training simulator has no platform DSP. B2f trains on acoustic stage + Opus only (RUN `config.yaml`), so any suppressor, AGC, VAD gate or PLC artefact is out of support and scored as "processing ⇒ spoof" (roadmap H2; next_steps_plan Context). (b) Unseen noise families are stationary or broadband (vacuum, washing machine, wind), while the training ESC-50 pools hold 32 files per class (RUN `config.yaml` pools), so noise diversity is low. (c) Non-Opus narrow-band codecs (gsm / g726) remove the cues the model uses for bona.
- **Already tried:** random Opus profiles, B3d: -2.20, and the targeted g726 / gsm bona got worse (B3S §3). SNR 0-20 dB, B3e: -1.06, with heldout losing R_bona at every SNR (B3e report §5.6). Consistency, B3a: heldout +2.28 F1 [+1.10, +3.47] (B3a CI json). **No DSP stage of any kind has ever been in training.** Chaining (`--aug_max_stages`) has never been run.
- **Addressable by:** augmentation (DSP stages with implementations disjoint from the reserve: WebRTC APM NS / AGC2, `webrtcvad` gating, RNNoise `arnndn`, DeepFilterNet, speexdsp, libopus PLC; next_steps_plan §2.4), loss (consistency across views).
- **Caveats:** B3 lesson 4 says "Every input-side change that made training harder cut the late-training fit and moved heldout bona toward spoof" (B3S §4). Those runs had JS consistency on, and the B3d / B3e reports partly blame the JS pairing of views with independent draws (B3d §5.5, B3e §5.6). Widening the augmentation on the B2f base without consistency is untested. A DSP stage that resembles a reserved one (band-limit, PLC, VAD) can make heldout less "unseen" even when the implementation differs (§5).

### G2 (priority 2). Additive noise flips bona even in-distribution (sim_matched, SNR, low-rate Opus)

- **Evidence:** matched F1 91.22 / 91.24, R_bona 76.21 / 76.47 against clean 92.84 / 92.91 (M / E), with the training noise families. R_bona by SNR: 0 dB 62.39 / 65.81, 5 dB 71.19 / 73.73, 10 dB 75.81 / 75.00, 20 dB 82.68 / 83.46. qq 61.62 / 66.67, wechat 65.38 / 71.15, against zoom 83.02 / 83.96. Online-source (double-processed) 72.83 / 74.73, against offline 79.06 / 80.63 (M / E breakdowns). 70-74 % of matched online-source FA are noise-induced (SC §3.3). Per epoch, matched mini R_bona falls from 81.56 (ep1) to 74.86 (ep3) while R_spoof rises from 93.67 to 99.39 (`metrics.jsonl`).
- **Size:** missing WF1 3.07 / 3.07 (computed); 93 / 92 FA.
- **Causes (hypotheses):** (a) the model's bona evidence is a fragile low-level cue that additive noise masks, and spoof is the default class (noise_only / sil_only 100 % spoof, §3.5; train prior 81 % spoof: 61,432 spoof vs 14,353 bona, roadmap §1.2). (b) Noise and a low-rate narrow-band codec compound (qq / wechat profiles are 8-24 kbps with a 4-8 kHz cutoff, `docs/augmentations.md` §3.3). (c) The training exposure is shallow: one stage per clip, noise on 0.8 × 6/8 = 60 % of clips (computed from the 8-stage list), SNR uniform 5-20 dB.
- **Already tried:** SNR 0-20 (B3e): matched SNR-0 R_bona +2.56 but net -1.06 (B3S §3). Level aug vs loudness norm: null (B2fR). Consistency: matched +0.90 [-0.19, +2.03] (B3a CI json). `--aug_snr_dist beta` inside 5-20 dB, noise mixing with more than one source, and per-clip noise-level schedules are untested.
- **Addressable by:** augmentation, loss.

### G3 (priority 3). The saturated core: confident bona false alarms, and "no speech evidence ⇒ spoof"

- **Evidence:** 136 / 133 of the noisy FA at p ≥ 0.99; clean 75 / 74 of 100 / 99 (SC). `noise_only` and `sil_only` bona-source probes are 100 % spoof, mean dP +0.89 / +0.90 (PM / PE). Saturated noisy-bona share across every B3 readout: 15.7-21.6 % (B3S lesson 5). "No LR or schedule moved that core" (B2S §3). Heldout VAD-gated bona: R_bona 55.9 / 61.8 with CNG and 69.2 / 69.2 with zeroed gaps (SC).
- **Size:** 64 % of the noisy bona FA (computed). A global threshold recovers only 0.39 / 0.40 (B2fR §5.2), so the core is a ranking problem.
- **Causes (hypotheses):** (a) training never shows bona with long non-speech stretches, gated silence, comfort noise or noise-dominated crops. The random 4.04 s crop can land on near-silence, and augmentation is applied before cropping (`docs/augmentations.md` §2, §10.13). (b) The class prior makes spoof the fallback when the bona cue is absent (§3.5).
- **Already tried:** none targeted it directly. Consistency did not shrink it (B3a 77 % at p ≥ 0.99; B3c grew it to 21.2 %) (B3S §3). Silence / VAD robustness augmentation (A2.9) and VAD-DTX-CNG (A2.3) are untested.
- **Addressable by:** augmentation, loss.

### G4 (priority 4). Echo conditions (in the brief's noisy measure, not in WF1)

- **Evidence:** echo F1 94.16 / 94.53, R_bona 83.38 / 84.65 (M / E); self_echo R_bona 79.40 / 79.83 (EER 8.05 / 8.41), farend 82.87 / 84.86, farend_aec 87.22 / 88.35 (M / E breakdowns). 65 / 60 half-B FA, 64-65 % of online-source ones noise-induced (SC). The roadmap names echo as the hardest condition in the paper (§0, §1.1).
- **Size:** not in WF1 locally. 65 / 60 FA of 391 bona.
- **Causes (hypotheses):** the training echo is one delayed copy (40-160 ms, gain 0.15-0.5) with no far-end talker, AEC residual or double-talk (`docs/augmentations.md` §3.2). Self-echo with several taps is the worst echo family.
- **Already tried:** none targeted it. Consistency: echo +0.71 [-0.39, +1.77] (B3a CI json).
- **Addressable by:** augmentation.

### G5 (priority 5). English bona

- **Evidence:** full-set en R_bona: matched 63.57 / 63.18, heldout 59.69 / 60.47, echo 70.93 / 72.48. zh: 82.52 / 85.37, 78.25 / 78.05, 89.84 / 90.85 (M / E). Half-B en FA: clean 70 / 65 of 479, matched 43 / 46 of 129, heldout 52 / 53 of 129 (SC). English is 33 % of noisy bona and holds 47 % of the final readout's matched + heldout FA (computed: 99 / 210).
- **Size:** large share of FA (above), also on clean.
- **Causes (hypotheses):** few distinct English bona recordings (about 1.5k English offline bona, `docs/next_steps_plan.md` Context). Augmentation multiplies channels, not speakers.
- **Already tried:** balanced + language sampler, B3b: English got worse (matched R_bona 67.4 → 64.3), -0.83 (B3S lesson 3). Exposure is refuted.
- **Addressable by:** data, augmentation (indirectly), loss.

### G6 (priority 5). Evaluation blind spots that make augmentation results hard to read

- **Evidence:** (a) The `sim_unseen_v1` noise is MUSAN music, which B2f trained on. RUN `config.yaml` `musan:1590` = 930 noise + 660 music files under `data/augmentation/musan/{noise,music}` (computed by `find`), and `wave.json` `lever_notes` says "sim_unseen_v1 is therefore not read in this loop (music leak)". `assert_training_ops` only flags pool dirs at or under a reserved dir (`b3wave/heldout_reserve.py` `reserved_dir_hits`), so the parent `musan/` passes. (b) Seed band unmeasured: "B3a's 93.06 rests on one seed" (B3S lesson 8); noise band ~0.2-0.5 final, up to ~1.4 per set (`wave.json` `lever_notes`). (c) The clean term is not half-filtered: 5,753 of the 7,465 clean clips are in the dev-loss set (`docs/dev_splits.md` §5).
- **Size:** decides whether the wave can tell a +0.5 lever from noise.
- **Causes:** see evidence.
- **Already tried:** paired bootstrap CIs exist (`b3_compare_evals.py --ci`); no second seed; no clean unseen set.
- **Addressable by:** eval, data.

## 5. Constraints

| kind | constraint | source |
|---|---|---|
| challenge rules | Metric 0.3·MacroF1(clean) + 0.7·MacroF1(noisy), threshold 0.5. **Single model at inference: ensembles forbidden.** Weight averaging (soups, SWA, EMA) of the same model is allowed. **Score averaging of two checkpoints (B2fR §7, B3S §5 Z-A) is an inference-time ensemble, so it cannot be a submission under this row** | roadmap §0, §2 |
| challenge rules | Non-speech noise / music / RIR corpora allowed; **MUSAN speech forbidden**; RNNoise office/coffee "verify"; local WebRTC AEC / NS / AGC / VAD simulation **explicitly allowed**; software codecs allowed; neural-codec round trip "ask"; babble from own training utterances "likely allowed"; fixed enhancement pre-processor "ask"; no progress / eval audio for tuning, selection or analysis; no new TTS / VC spoofs | roadmap §2 |
| eval-only reserve | Never train on: codecs G.722 / G.726 / GSM, speex, G.711; filters `anlmdn`, `acompressor`, `afftdn`, `dynaudnorm`, `vad_gate`, `packet_loss`, `bandlimit`, `resample8k`, `compand`, `agate`, `crystalizer`, `speechnorm`; ESC-50 classes wind, pouring_water, door_wood_knock, mouse_click, door_wood_creaks, can_opening, washing_machine, vacuum_cleaner, clock_alarm, clock_tick, glass_breaking; dirs `data/augmentation_eval`, `data/augmentation/musan/music`; dev audio and half B. New augmentation code must call `assert_training_ops(codecs=…, filters=…, dirs=…)` itself. The guard checks names only (`docs/dev_splits.md` §7) | `b3wave/heldout_reserve.py`; `utils/heldout_reserve.py`; `docs/dev_splits.md` §7-8 |
| reserve design | Training DSP must use implementations **different** from `sim_ops` (WebRTC APM NS / AGC2, `webrtcvad`, libopus PLC, RNNoise `arnndn`, DeepFilterNet, speexdsp), so heldout stays a "DSP family never seen" test | `docs/next_steps_plan.md` §2.4 |
| known leak | B2f-lineage runs trained on MUSAN music, so `sim_unseen_v1` cannot be a clean check. Keeping `--musan_dir ../data/augmentation/musan` is a fixed B3 flag | G6; `exploop/B3/wave.json` `search_space.fixed`, `lever_notes` |
| preprocessing | `--loudness_norm --loudness_norm_lufs -23` runs in `_load` before any augmentation, on train, dev and eval. Every eval must pass it | `docs/augmentations.md` §2.1; B2fR §5.4 |
| compute (cycle) | One RTX 4090 (24 GB). Extra VRAM ≤ 4.0 GiB, peak ≤ 22.0 GiB, epoch time ≤ 1.3 × base, ≤ 400 LOC per lever, ≤ 10 h per run. Base epochs are 30.7-31.2 min, so the cap is ≈ 40.6 min per epoch and ≈ 5.4 h per 8-epoch run (computed). **A two-view consistency lever doubles the epoch (57.7-59.8 min, B3S §4.9) and breaks the 1.3× cap against B2f** | `rdloop/B4/cycle.json` `limits`; B2fR §1 |
| compute (CPU) | Augmentation runs on the fly in 64 DataLoader workers. Each codec call spawns two ffmpeg subprocesses, and heavy chains can make training data-bound. ffmpeg is 4.2 (no Opus FEC / DTX options) | `docs/augmentations.md` §1, §10.5; `b3wave/README.md` |
| environment | A container restart wipes system ffmpeg, and the codec stage then silently passes audio through (`[rtc_augment] codec … failed`). New dependencies go into the `rtc-sdd` conda env | `docs/next_steps_plan.md` §9; B3S §5 pre-launch checks |
| protocol | One seed (1234) per configuration; ≤ 10 epochs; `--keep_topk_by_wf1 7 --keep_epochs 8`; both readouts logged; null band ±0.5 | `exploop/B3/wave.json` |
| data | Train 75,785 clips: 61,432 spoof + 14,353 bona; about 1.5k English offline bona. Train has no noisy audio and only 2 of the 7 platforms | roadmap §0, §1.2; next_steps_plan Context |
| deadline | Final eval data 2026-11-09; submission due 2026-11-16 (about 30 days from 2026-10-17, computed) | roadmap §11 |

## 6. Exhausted directions (do not re-propose without new evidence)

| direction | closed by | evidence |
|---|---|---|
| label smoothing with asymmetric `--ce_weights` | B1 postmortem; B2S §3 | -2.50 WF1@0.5, oracle only -0.45, thr_oracle 0.69 |
| LR below 2.5e-6, or more epochs (10) | B2S §3 (B2d, B2e) | LR series turned over (+0.26 final); 10 epochs -0.16 / -0.42 |
| level treatment (level aug ↔ loudness norm) | B2fR §6 | null -0.36 / -0.39; same worst slices |
| JS consistency weight above 0.3 | B3S lesson 2 (B3c) | -0.77 [-1.31, -0.22]; saturated tail grew |
| class / language balanced sampler (English exposure) | B3S lesson 3 (B3b) | -0.83 [-1.46, -0.25]; English worse |
| random Opus bitrate / frame / VBR profiles | B3S lesson 4 (B3d) | -2.20 [-2.90, -1.57]; g726 / gsm bona -13.42 / -11.38 |
| widening the uniform SNR range to 0-20 dB | B3S lesson 4 (B3e) | -1.06 [-1.65, -0.49]; heldout SNR-0 R_bona -4.09 |
| cross-LR weight averaging | B2S §3 | 91.87 vs parents 92.26 / 92.53 |
| threshold / calibration / CE-prior tuning as a main lever | B2S §3; B2fR §5.2 | ceiling 0.2-0.5; global shift recovers 0.39-0.40 |

**Caution for the plan.** Every B3 augmentation change ran on top of consistency at λ 0.3, so "input-side widening
hurts" is established only with JS pairing on. A new augmentation lever on B2f should state up front how it avoids
the B3d / B3e signature: worse fit (ep8 CE ≥ 1.5× the reference) and heldout bona moving toward spoof (B3S lesson 4).
