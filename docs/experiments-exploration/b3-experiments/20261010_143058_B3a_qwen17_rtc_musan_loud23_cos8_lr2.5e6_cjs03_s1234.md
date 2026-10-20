# Experiment exploration B3-01: B3a (`B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234`) - paired JS consistency on top of B2f

> Written 2026-10-10 by the exploop B3 analysis agent (ANALYZE step, wave B3, iteration 1, attempt a0).
>
> **Sources.** Every number is copied from a file read in this session, or computed in this session from the half_B rows of the local `scores/*.txt` / `*.logits` files. Nothing is quoted from memory. The files:
> - run config and logs: `config.yaml` (both runs), `logs/train.log` (both runs), `metrics.jsonl` (both runs), `top5_by_wf1/{merge_list.txt,select_topk.log}`, the `ckpt/` listing;
> - local evals: `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,compare_vs_B2f-*.txt,ci_vs_B2f-*.txt,compare_ci_vs_B2f-*.json}`, `local_eval/merge_top5/compare_all_readouts.txt`, `local_eval/merge_top5.log`;
> - stdout logs: B3a `SLS_setup/B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_20261009214839.log` and B2f `SLS_setup/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_20261009132522.log` (header lines only);
> - loop files: `exploop/B3/iter01/{verdict.json,design.result.json}`, `exploop/B3/{wave.json,plan_snapshot.md,STATUS.md}`, `scripts/run_B3a.sh`;
> - registry and code: `docs/experiment_log.csv`, `SLS_setup/b3wave/{README.md,train.py}` (flag definitions), `SLS_setup/utils/local_eval.py` (half split, WF1 definition);
> - training labels: `/mnt/paml-research/RTCFake/data/train_label.txt` (class counts for the CE-weight derivation in section 7).
>
> Progress-split scores were not read.
>
> **Previous reports.** `docs/experiments-exploration/20261009_175355_B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234.md` (the reference run; also the source of the B2e numbers quoted in 2.5) and `docs/experiments-exploration/20261008_232331_B2_wave_summary_and_full_run_recommendation.md`.
>
> **Readouts.**
> - `-top5` is the weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5/`). It is the wave objective.
> - `-ep8` is the final-epoch EMA checkpoint alone (`local_eval/epoch_8/`).
> - half_B is the half of each sim set that is never used for selection: n = 1,994 per set, 1,603 spoof, so 391 bona (`metrics.json`). Clean is the full 7,465-clip `dev_online_clean`, with 1,397 bona.
> - WF1 = 0.3 clean + 0.35 matched + 0.35 heldout (macro-F1 at p_spoof 0.5). Echo is reported but is not in WF1 (`configs/local_eval_sets.yaml`, `SLS_setup/utils/local_eval.py`).
> - Threshold 0.5 on the score files reproduces all four half_B WF1 values of `metrics.json` exactly: 91.98 / 93.06 / 92.14 / 92.94.
> - Every B2f and B3a evaluation ran with `--loudness_norm -23`: the `[loudness_norm] loudness_norm target=-23 LUFS` line is in `merge_top5.log`, and `metrics.json` has `"loudness_norm": -23.0`.

Run directories (under `SLS_setup/exp/`):
- **B3a** `B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_epoch8_bs8_20261009214846` (this report)
- **B2f** `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623` (reference; current best at the start of B3: merge 91.98, final 92.14)

## 1. Setup

**B3a = B2f with one coupled factor changed: paired consistency training, `--consistency js --consistency_lambda 0.3 --consistency_pairs both`, and `--batch_size 8` so that a step still holds 16 clips.** Every training clip is seen as two views in one forward pass:
- view a is the clip with its usual RTC/MUSAN augmentation;
- view b is the clip's online RTC twin if one exists, otherwise the same clip augmented a second time, independently.

The loss is the mean CE over both views plus 0.3 × the Jensen-Shannon divergence between the two posteriors (`b3wave/README.md`, `train.py` help, `scripts/run_B3a.sh`).

Flag diff (`args` of both `config.yaml`; nothing else differs):

| flag / field | B2f | **B3a** | note |
|---|---|---|---|
| trainer | `main_train.py`, git `00da485`, dirty | **`b3wave.train`** (b3wave 0.1.0), git `df3dff2`, clean | The README states bit-identical behaviour with every B3 flag at its default. |
| `--consistency` | (absent = none) | **`js`** | the lever |
| `--consistency_lambda` | - | **0.3** | |
| `--consistency_pairs` | - | **`both`** | twin if one exists, else aug2 |
| `--batch_size` | 16 | **8** (8 pairs = 16 clips/step) | coupled with consistency (`wave.json` lever_notes) |
| `consistency_emb_lambda` / `consistency_clean_view` | - | 0.3 / false | inactive with `js` |
| other B3 flags | - | defaults: `sampler shuffle`, `aug_snr_db 5 20 uniform`, `aug_codec_profiles fixed`, `aug_max_stages 1`, `llrd 1.0`, `freeze_bottom_k 0`, `specaug false` | Header lines below confirm these. |
| everything else | identical | identical | `--num_epochs 8 --lr 2.5e-6 --lr_min 5e-7 --warmup_frac 0.05 warmup_cosine adamw wd 0.01 head x10 grad_clip 1 ema 0.9995`, `ce_weights 0.1 0.9`, `label_smoothing 0`, RTC aug + MUSAN, `--loudness_norm -23`, `--keep_topk_by_wf1 7 --keep_epochs 8`, seed 1234 |

**Header lines proving the change took effect** (B3a stdout log; B2f stdout log for comparison):
- `Consistency: js lambda_js=0.3 lambda_emb=0.3 pairs=both clean_view=False | pairs found {'rows': 38660, 'blank_online': 1535, 'found': 37125} dropped_without_twin=0 | 16 clips per step (8 pairs)`. This matches the design's `expect` regex `^Consistency: js lambda_js=0.3`.
- **`Warmup-cosine: 3790 warmup / 75792 steps, floor 5.0e-07`**, against B2f's `1895 warmup / 37896 steps`. **B3a took twice as many optimiser steps.** The LR at each epoch end is identical (2.469e-6 … 5.000e-7 in both `train.log`s).
- Unchanged lines: `Sampler: shuffle`, `SNR: 5..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: fixed`, `LLRD: 1`, `SpecAugment: off`, `Loudness norm: loudness_norm target=-23 LUFS`, `CE weights: spoof=0.1 bonafide=0.9`, `Parameters: 320268933`, `Train trials: 75785`.
- Per-epoch augmentation counts (`train.log`, epoch 1):
  - The seven codec counts sum to 151,570 = 2 × 75,785. Each clip went through the codec twice, once per view.
  - The eight stage counts sum to 121,242, i.e. 0.80 of the views, which matches `p_apply=0.8`.
- The stdout log has no `Traceback`, `out of memory` or `failed (` line (grep count 0).

**Run time.** `config.yaml` created 21:48:55; epoch 8 done at 05:40:48; last local eval at 05:41:12. That is **7 h 52 min** (registry `gpu_h` 7.87, against B2f's 4.14). Epochs took 58.3-59.5 min, against B2f's 30.7-31.2 min (`train.log` timestamps). Post-train timestamps: `merge_list.txt` 14:08, `merge_top5/metrics.json` 14:14, `epoch_8/metrics.json` 14:19, `model_merging_top5/submission.zip` 14:29.

**Checkpoints.** Nothing was pruned: there is no `Pruned` line, and all 8 epoch files are present at 1,281,235,691 bytes each. The symlinks are:
- `best_by_dev_loss` → epoch 1 (0.099016);
- `best_by_noisy_eer` → epoch 6;
- `best_by_wf1` → epoch 6.

**Merged top-5 by half-A mini WF1 = epochs 6, 5, 7, 8, 2** (92.87, 92.52, 92.36, 92.14, 91.64; mean 92.31; `select_topk.log`). Excluded: 4 (91.52), 3 (91.39), 1 (90.17). B2f's merge was 8, 5, 7, 3, 4.

## 2. Results vs references (half_B)

Sources: `compare_all_readouts.txt`, `compare_vs_B2f-*.txt`, `ci_vs_B2f-*.txt` / `compare_ci_vs_B2f-*.json` (paired bootstrap, n = 2,000, half B, threshold 0.5), and the `half_B` block of each `metrics.json`. Deltas use unrounded values.

### 2.1 Wave iterations and fixed comparators, both readouts

| half_B | B2f-top5 | B2f-ep8 | **B3a-top5** | **B3a-ep8** | B3a-top5 − B2f-top5 [95 % CI] | B3a-ep8 − B2f-ep8 [95 % CI] |
|---|---|---|---|---|---|---|
| **WF1@0.5** | 91.98 | 92.14 | **93.06** | **92.94** | **+1.08 [+0.52, +1.67]**, p(not better) 0.000 | **+0.80 [+0.22, +1.42]**, p 0.004 |
| WF1@oracle | 92.59 | 92.82 | 93.24 | 93.45 | +0.65 | +0.63 |
| clean F1@0.5 | 97.60 | 97.69 | 97.48 | 97.06 | −0.12 [−0.44, +0.20] | **−0.63 [−1.02, −0.26]** |
| noisy F1@0.5 (mean matched, heldout) | 89.57 | 89.76 | 91.16 | 91.17 | +1.59 | +1.42 |
| mean noisy EER | 10.09 | 9.66 | 9.16 | 9.01 | −0.93 | −0.65 |
| calibration gap noisy | 0.66 | 0.71 | 0.11 | 0.42 | −0.55 | −0.29 |
| separability gap | 7.87 | 7.86 | 6.56 | 6.20 | −1.31 | −1.66 |
| oracle gap (WF1@or − WF1@.5) | 0.61 | 0.69 | 0.18 | 0.51 | −0.43 | −0.18 |

Per-set paired CIs:

| set | merge Δ [CI] | ep8 Δ [CI] |
|---|---|---|
| clean | −0.12 [−0.44, +0.20] | −0.63 [−1.02, −0.26] |
| matched | +0.90 [−0.19, +2.03] | +0.79 [−0.25, +1.89] |
| heldout | **+2.28 [+1.10, +3.47]** | **+2.05 [+0.69, +3.42]** |
| echo | +0.71 [−0.39, +1.77] | −0.14 [−1.21, +0.87] |

Weighted contributions (0.3 / 0.35 / 0.35):
- merge: −0.04 clean + 0.31 matched + 0.80 heldout = +1.08. **Heldout carries three quarters of the gain.**
- final: −0.19 + 0.28 + 0.72 = +0.80.

**On other halves of the same sets** (the registry columns are full sets; half A was computed from `scores/*.txt`):

| readout | full-set WF1 (B2f → B3a) | half_A WF1 (B2f → B3a) |
|---|---|---|
| merge | 92.25 → 93.05 (+0.80) | 92.52 → 93.04 (+0.52) |
| final | 92.53 → 92.78 (+0.25) | 92.94 → 92.61 (**−0.33**) |

**The merge gain shows on both halves. The final-epoch gain is specific to half B.** Half A is the selection half, so the merge numbers on half A are optimistic for both runs equally.

### 2.2 Per set (R = recall at 0.5; thr = p_spoof threshold; `metrics.json` half_B)

| set | metric | B2f-top5 | **B3a-top5** | B2f-ep8 | **B3a-ep8** |
|---|---|---|---|---|---|
| dev_online_clean | F1@.5 / F1@or / thr_or | 97.60 / 98.10 / 0.997 | **97.48** / 97.83 / 0.996 | 97.69 / 98.33 / 0.999 | **97.06** / 97.79 / 1.000 |
| | EER / ECE | 3.42 / 1.42 | 3.32 / 1.47 | 3.33 / 1.33 | 3.20 / 1.72 |
| | R_spoof / R_bona | 99.90 / 92.84 | 99.93 / **92.34** | 99.95 / 92.91 | 99.92 / **91.12** |
| sim_matched_v1 | F1@.5 / F1@or / thr_or | 91.22 / 91.53 / 0.846 | **92.12** / 92.25 / 0.995 | 91.24 / 92.05 / 0.991 | **92.03** / 92.49 / 0.998 |
| | EER / ECE | 8.15 / 4.75 | 7.89 / 4.51 | 8.21 / 4.86 | 7.74 / 4.55 |
| | R_spoof / R_bona | 99.50 / 76.21 | 99.38 / **79.03** | 99.44 / 76.47 | 99.69 / **77.75** |
| sim_heldout_v1 | F1@.5 / F1@or / thr_or | 87.92 / 88.94 / 0.957 | **90.20** / 90.30 / 0.971 | 88.27 / 88.89 / 0.949 | **90.32** / 90.69 / 0.993 |
| | EER / ECE | 12.03 / 6.46 | 10.42 / 5.54 | 11.11 / 6.38 | 10.29 / 5.57 |
| | R_spoof / R_bona | 99.00 / 69.31 | 99.06 / **74.94** | 99.13 / 69.82 | 99.00 / **75.45** |
| sim_echo_v1 | F1@.5 / F1@or / thr_or | 94.16 / 94.91 / 0.983 | **94.87** / 95.03 / 0.998 | 94.53 / 95.41 / 0.983 | **94.39** / 94.96 / 0.999 |
| | EER / ECE | 4.71 / 3.37 | 4.61 / 3.00 | 4.86 / 3.20 | **5.62** / 3.28 |
| | R_spoof / R_bona | 99.75 / 83.38 | 99.81 / 85.17 | 99.69 / 84.65 | 99.94 / **83.38** |

Reading:
- **The gain is bona recall on the noisy sets at unchanged spoof recall.** Heldout R_bona rose +5.63 in both readouts and matched +2.81 / +1.28, while R_spoof stayed within ±0.25 on every set.
- **Clean and echo lose on the final epoch only.** Clean R_bona fell −0.50 (merge) / −1.79 (ep8). Echo EER rose +0.76 at ep8.
- Paired per clip (half_B score files, B3a vs B2f):

  | set | class | merge fixed / broken / wrong in both | ep8 fixed / broken / wrong in both |
  |---|---|---|---|
  | heldout | bona | 27 / 5 / 93 | 31 / 9 / 87 |
  | matched | bona | 21 / 10 / 72 | 16 / 11 / 76 |
  | echo | bona | 18 / 11 / 47 | 12 / 17 / 48 |
  | clean | bona | 18 / 25 / 82 | **17 / 42** / 82 |
  | any | spoof | 3-7 per set either way | 3-7 per set either way |

- Logit correlation of B3a vs B2f: 0.960-0.967 per set (merge) and 0.954-0.967 (ep8). These are the least correlated pairs measured in the lineage; the B2f report gave 0.960-0.990 for B2-wave pairs.

### 2.3 Probes (`probe_report.json`, probe sets of `metrics.json`; 187 bona + 813 spoof; one bona probe clip = 0.53 points)

Under `--loudness_norm` the pure-gain probes (`gain_down`, `gain_m20`, `gain_m6`) are identical to `probe_orig` by construction (B2f report, section 2.3). They flip 0.00 % in all four readouts here.

| probe (bona unless stated) | B2f-top5 | **B3a-top5** | B2f-ep8 | **B3a-ep8** |
|---|---|---|---|---|
| level_flip_bona (mean of 8) / max | 1.54 / 4.81 | **0.80** / 2.67 | 1.74 / 4.81 | **2.07** / 7.49 |
| clipped / limited: gain_up / gain_p6 / gain_p10_limit flip % | 4.28 / 2.14 / 4.81 | 2.67 / 1.07 / 2.67 | 4.81 / 2.67 / 3.21 | 4.81 / 2.14 / **7.49** |
| dynamics: agc_dynaudnorm / loudnorm flip % | 0.53 / 0.53 | 0.00 / 0.00 | 2.14 / 1.07 | 0.00 / 2.14 |
| level_flip_spoof | 0.03 | 0.02 | 0.03 | 0.00 |
| probe_orig: bona pred spoof / spoof pred spoof | 10.70 / 100.00 | **12.30** / 100.00 | 10.16 / 100.00 | **11.76** / 100.00 |
| **noise_only: bona-source / spoof-source pred spoof** | 100.00 / 100.00 | **100.00** / 100.00 | 100.00 / 100.00 | **100.00** / 100.00 |
| noise_only mean dP(spoof), bona-source | +0.887 | +0.876 | +0.897 | +0.882 |
| sil_only: bona-source / spoof-source pred spoof | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| trimmed / padded bona flip % | 3.74 / 3.74 | 2.14 / 3.21 | 3.74 / 5.35 | 3.21 / 2.67 |

Reading:
- **Consistency did not touch the non-speech failure.** Bona-source `noise_only` and `sil_only` clips are still 100 % spoof, with mean dP down by only 0.01-0.015.
- The merge is the most level- and dynamics-robust readout of the lineage (level_flip_bona 0.80; agc / loudnorm 0 flips).
- The final epoch is the least robust on the limiter probe: `gain_p10_limit` flipped 14 of 187 bona clips, against B2f-ep8's 6.
- `probe_orig` bona false alarms rose by 3 clips in both readouts. This is consistent with the clean-bona loss in 2.2.

### 2.4 Slices that moved most, B3a vs B2f (full sim sets, `metrics.json → breakdowns`, 63 slice rows)

Counts of slice rows that improved:

| readout | F1 up | EER down | R_bona up | R_spoof up |
|---|---|---|---|---|
| merge | **55 of 63** (0 ties) | 48 | 55 | 22 |
| final | 41 of 63 (2 ties) | 35 | 34 | 32 |

F1 up per set, merge / ep8: matched 21 / 15 of 23 rows, heldout 20 / 18 of 23, echo 14 / 8 of 17.

Minimum slice R_spoof: B3a-top5 97.02 (heldout / washing_machine), B3a-ep8 96.69 (same slice); B2f-top5 97.31, B2f-ep8 97.68.

| set / slice | n (bona) | F1 merge B2f → B3a | F1 ep8 B2f → B3a | EER ep8 | R_bona ep8 (merge) |
|---|---|---|---|---|---|
| heldout / vacuum_cleaner | 373 (70) | 83.27 → 87.09 +3.82 | 82.01 → 87.99 **+5.98** | 18.36 → 9.46 | 57.1 → 71.4 (60.0 → 67.1) |
| heldout / codec gsm | 1346 (246) | 85.67 → 88.18 +2.51 | 85.39 → 88.26 +2.87 | 15.02 → 13.43 | 65.4 → 73.6 (65.9 → 72.8) |
| heldout / pouring_water | 343 (72) | 88.19 → 90.28 +2.09 | 88.19 → 91.36 +3.17 | 11.05 → 10.21 | 69.4 → 76.4 (69.4 → 73.6) |
| heldout / door_wood_creaks | 361 (64) | 86.83 → 90.88 **+4.05** | 88.70 → 88.11 −0.60 | 10.18 → 9.06 | 73.4 → 71.9 (71.9 → 78.1) |
| matched / opus:qq | 572 (99) | 84.50 → 87.68 +3.19 | 86.04 → 86.61 +0.57 | 10.95 → 10.95 | 66.7 → 64.6 (61.6 → 68.7) |
| matched / opus:wechat | 585 (104) | 86.62 → 89.04 +2.41 | 88.94 → 89.22 +0.28 | 13.21 → 12.07 | 71.2 → 71.2 (65.4 → 72.1) |
| heldout / lang zh | 2462 (492) | 90.86 → 92.54 +1.68 | 91.15 → 92.89 +1.74 | 8.68 → 7.14 | 78.0 → 83.3 (78.3 → 82.5) |
| matched / snr 5 | 656 (118) | 88.33 → 90.34 +2.00 | 89.31 → 90.58 +1.27 | 10.20 → 9.55 | 73.7 → 77.1 (71.2 → 77.1) |
| heldout / snr 20 | 760 (144) | 92.16 → 92.21 +0.05 | 90.87 → 93.20 +2.34 | 10.38 → 8.49 | 75.7 → 82.6 (79.2 → 79.9) |
| heldout / snr 0 | 812 (147) | 84.68 → 85.72 +1.04 | 83.82 → 85.07 +1.25 | 17.07 → 15.00 | 61.2 → 64.6 (63.3 → 65.3) |
| matched / lang en | 1538 (258) | 86.76 → 87.34 +0.58 | 86.71 → 86.38 −0.33 | **12.68 → 14.36** | 63.2 → 63.2 (63.6 → 65.5) |
| heldout / lang en | 1538 (258) | 84.45 → 84.74 +0.29 | 84.47 → 84.26 −0.21 | **16.34 → 17.63** | 60.5 → 60.5 (59.7 → 60.9) |
| echo / lang en | 1538 (258) | 88.96 → 89.40 +0.44 | 89.33 → 88.66 −0.67 | **10.54 → 12.06** | 72.5 → 68.2 (70.9 → 70.9) |
| matched / snr 10 | 653 (124) | 91.50 → 91.80 +0.30 | 90.95 → 89.62 −1.33 | 7.22 → 7.41 | 75.0 → 71.0 (75.8 → 76.6) |
| heldout / glass_breaking | 346 (64) | 94.49 → 92.85 −1.64 | 92.94 → 92.29 −0.66 | 6.32 → 8.34 | 82.8 → 79.7 (85.9 → 81.2) |

Pattern:
- **The gain is unseen-DSP heldout (gsm, vacuum_cleaner, pouring_water) and Chinese.** Heldout zh R_bona rose +5.3 at ep8. Heldout gsm R_bona rose +8.2 at ep8, i.e. one of the two reserved codecs improved by generalisation.
- **English did not gain, and its separability got worse at the final epoch.** Full-set en EER at ep8 rose on all three sets (above). At the merge it was mixed (matched 12.76 → 13.66, heldout 17.78 → 17.04, echo 11.64 → 11.28), while zh EER fell (matched 7.29 → 5.56, heldout 9.22 → 7.82).
- **SNR 0 moved only +0.9 to +3.4 R_bona.** Matched snr 0 at the merge: 62.4 → 65.8.

### 2.5 Lever response (one factor per row)

| run | change vs previous reference | half_B merge WF1 (Δ) | final-epoch WF1 (Δ) | merge oracle (Δ) | final oracle (Δ) | merge noisy EER (Δ) | final EER (Δ) | merge clean (Δ) | final clean (Δ) | verdict (verdict.json) |
|---|---|---|---|---|---|---|---|---|---|---|
| B2e | `--lr 2.5e-6 --lr_min 5e-7` (B2 wave; B2f report) | 92.34 | 92.53 | 92.88 | 92.97 | 9.68 | 9.27 | 97.67 | 98.09 | (B2 wave) |
| B2f | `--loudness_norm -23` replaces `--use_level_aug` (B2f report) | 91.98 (−0.36) | 92.14 (−0.39) | 92.59 | 92.82 | 10.09 | 9.66 | 97.60 | 97.69 | null (B2f report) |
| **B3a** | **+ `--consistency js --consistency_lambda 0.3 --consistency_pairs both`, `--batch_size 8`** | **93.06 (+1.08)** | **92.94 (+0.80)** | 93.24 (+0.65) | 93.45 (+0.63) | 9.16 (−0.93) | 9.01 (−0.65) | 97.48 (−0.12) | 97.06 (−0.63) | merge: **positive / promote**; final: above band, clean F1 dropped / revisit |

B3a is the largest one-factor gain since the LR series of the B2 wave. Its merge Δ (+1.08) is above the near-replicate band quoted in `wave.json` lever_notes ("~0.2-0.5 WF1 on the final readout"). The CI excludes 0 on both readouts. B3a's half_B merge (93.06) also exceeds B2e's (92.34), the best B2-wave merge.

## 3. Per-epoch behaviour, B3a vs B2f (aligned by epoch = same LR at each epoch end; B3a ran 2× the steps per epoch)

Sources: `metrics.jsonl` (EMA weights; 2,000 clean clips + `sim_matched_mini` + `sim_heldout_mini`, half A), `logs/train.log`, and `STATUS.md` (Δ vs B2f at the same epoch). For B3a, `TrainLoss = CE + 0.3·JS` (epoch 1: 0.3945 + 0.3 × 0.0344 = 0.4048). The CE column is the per-view mean CE, which is the quantity comparable to B2f's train loss.

| ep | LR end | B3a mini WF1 | B2f mini WF1 | Δ | B3a WF1@or | B3a EER noisy (B2f) | B3a R_bona cl/ma/he | B2f R_bona cl/ma/he | B3a CE / JS | B2f train loss | B3a / B2f train acc | B3a / B2f dev loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.47e-6 | 90.17 | 88.34 | +1.83 | 90.55 | 13.14 (13.40) | 91.0/73.7/65.8 | 92.7/81.6/78.6 | 0.3945 / 0.0344 | 0.3558 | 90.16 / 87.36 | 0.0990 / 0.0927 |
| 2 | 2.29e-6 | 91.64 | 91.47 | +0.17 | 92.27 | 11.56 (11.78) | 91.6/76.5/70.9 | 91.6/77.1/76.0 | 0.1867 / 0.0214 | 0.2075 | 97.87 / 97.13 | 0.1232 / 0.0982 |
| 3 | 1.98e-6 | 91.39 | 91.90 | −0.51 | 92.14 | 11.49 (10.38) | 91.6/76.5/66.8 | 91.6/74.9/71.4 | 0.1176 / 0.0151 | 0.1546 | 98.63 / 98.09 | 0.1514 / 0.1139 |
| 4 | 1.58e-6 | 91.52 | 91.69 | −0.17 | 92.13 | 10.80 (10.64) | 89.9/76.5/69.9 | 91.3/75.4/71.4 | 0.0842 / 0.0113 | 0.1108 | 99.01 / 98.62 | 0.1657 / 0.1242 |
| 5 | 1.18e-6 | 92.52 | 92.49 | +0.03 | 93.17 | 11.16 (10.46) | 91.6/81.0/71.9 | 90.4/78.8/71.9 | 0.0642 / 0.0080 | 0.0879 | 99.32 / 98.96 | 0.1472 / 0.1178 |
| 6 | 8.23e-7 | **92.87** | 91.66 | **+1.21** | **93.51** | **10.09** (10.63) | 91.9/82.7/71.9 | 91.3/74.9/70.4 | 0.0452 / 0.0065 | 0.0725 | 99.47 / 99.18 | 0.1356 / 0.1321 |
| 7 | 5.84e-7 | 92.36 | 92.10 | +0.26 | 93.07 | 10.40 (10.69) | 91.3/80.4/71.4 | 91.3/76.0/73.0 | 0.0328 / 0.0051 | 0.0540 | 99.59 / 99.36 | 0.1584 / 0.1406 |
| 8 | 5.00e-7 | 92.14 | **92.80** | **−0.66** | 93.03 | 10.28 (10.94) | 89.9/78.8/71.9 | 91.3/80.4/73.5 | 0.0283 / 0.0038 | 0.0501 | **99.71** / 99.44 | **0.1722** / 0.1442 |

Reading:
- **The best epoch is 6, and the curve is falling at the end.** Mini WF1 peaks at 92.87 / oracle 93.51 / noisy EER 10.09 at epoch 6, then drops 0.51 and 0.73 at epochs 7-8. B2f was the other way round, with its best epoch at 8 (92.80). On half A, B3a beats B2f at 5 of 8 epochs but loses at the last one. This matches the half-A final-epoch result in 2.1 (92.61 vs 92.94).
- **B3a fits faster.** Twice the steps and two views per clip give a lower CE from epoch 2 on: 0.1867 → 0.0283 against 0.2075 → 0.0501. Train acc reaches 99.71 % against 99.44 %.
- **Dev loss rises further.** B3a: 0.0990 → 0.1722 (+74 %). B2f: 0.0927 → 0.1442 (+55 %). B3a's dev loss is above B2f's at every epoch, with a dip at epochs 5-6 (0.1472, 0.1356) that coincides with the mini-WF1 peak.
- **The JS term is small.** It falls 0.0344 → 0.0038, so the weighted term 0.3·JS is 2.5 % of the train loss at epoch 1 and 3.9 % at epoch 8. The two views already agree closely under plain CE, so the gain cannot be attributed to the divergence term alone: B3a also doubles the optimiser steps and shows each clip in two views. The λ dose step (proposal 2) separates the two.
- **Half-A operating point.** B3a's mini heldout `thr_oracle` sits at 0.86-0.99 from epoch 1. B2f's went from 0.14 to 0.90 over the run.
- **The merge beats the final epoch: 93.06 vs 92.94 (+0.12 for the merge) on half B, and 93.04 vs 92.61 on half A.** This is the first time in the lineage; the B2f report notes the final epoch beat the merge in 6 of 6 B2 runs.
  - The merge contains the epoch-6 peak, and epochs 7-8 are where the clean and echo losses concentrate (2.2, 5.1).
  - The weakest merged epoch by mini WF1 is epoch 2 (91.64). No merged epoch was evaluated alone on half B, so its half-B drag is not measured.

## 4. What went OK (evidence)

1. **The pre-registered promote rule fired cleanly on the objective.**
   - Merge half_B: +1.08, CI [+0.52, +1.67], p(not better) 0.000.
   - Clean drop: 0.12, within the 0.5 limit.
   - `verdict.json` merge: `decision promote`, `verdict positive`, `new_best true`.
   - The final readout is also above the band: +0.80, CI [+0.22, +1.42].
2. **It is a ranking gain, not a threshold gain.**
   - Oracle +0.65 / +0.63.
   - Separability gap 7.87 → 6.56 (merge) / 7.86 → 6.20 (ep8).
   - Noisy EER −0.93 / −0.65.
   - Heldout EER −1.61 / −0.82.
   - The merge's operating point is now essentially optimal at 0.5: calibration gap noisy 0.11, and a global logit shift recovers only 0.02 (5.2).
3. **The heldout gain is significant on its own**: +2.28 [+1.10, +3.47] (merge) and +2.05 [+0.69, +3.42] (ep8). The paired flips are lopsided:
   - heldout bona: 27 fixed / 5 broken (merge), 31 / 9 (ep8);
   - matched bona: 21 / 10 (merge).

   Generalisation to unseen DSP is the targeted property, and it moved:
   - heldout gsm R_bona +8.2 and vacuum_cleaner +14.3 at ep8;
   - heldout F1 rose in 20 of 23 heldout slice rows at the merge and 18 of 23 at ep8.
4. **Broad, not slice-specific.** At the merge, F1 rose in 55 of 63 slice rows and R_bona in 55 of 63, with spoof recall held (min slice R_spoof 97.02). Noisy bona false alarms on half B fell from 213 to 180 clips (merge) and from 210 to 183 (ep8).
5. **Clean experiment.**
   - One coupled factor, confirmed by the header.
   - Twin pairs found for 37,125 clips, none dropped.
   - The codec counts prove two views per clip.
   - No traceback or codec failure; nothing pruned.
   - Git clean (`df3dff2`).
   - Both readouts were evaluated with the loudness flag, and both registry rows exist with CI filled: `B3a-cjs03-screen` (0.52 / 1.67) and `B3a-cjs03-ep8`.
6. **Complementary to B2f.** Logit correlation is 0.954-0.967. The zero-training score average (mean p_spoof) of B2f-ep8 + B3a-ep8 gives half_B WF1 **93.40** (clean 97.72, matched 92.46, heldout 90.63), and B2f-top5 + B3a-top5 gives 93.32. This is not a trainer flag; it is noted for the submission step.

## 5. What went wrong (evidence)

### 5.1 The final epoch loses clean bona (the reason the final readout is `revisit`)

- Clean F1 at ep8 fell −0.63, CI [−1.02, −0.26], which excludes 0. `verdict.json` final: `clean_drop 0.63`, `decision revisit`, `verdict "above band, clean F1 dropped"`.
- Clean R_bona fell 92.91 → 91.12: 17 fixed and 42 broken of 1,397 bona. The ECE rose 1.33 → 1.72, and clean `thr_oracle` is 1.000.
- The share of clean bona at p_spoof ≥ 0.99 rose 5.3 → 8.3 % (≥ 0.999: 4.3 → 6.1 %).
- The merge loses only −0.12 [−0.44, +0.20] (18 fixed / 25 broken).
- Echo shows the same final-epoch over-confidence: EER 4.86 → 5.62, and echo bona at ≥ 0.99 rose 10.0 → 15.3 %.

### 5.2 Oracle-gap decomposition (global threshold vs set-specific)

Method: a sweep of one shared logit threshold over the half_B `scores/*.logits` (step 0.05, −6 to +12). The p-grid sweep (0.005-0.995 step 0.005, plus 1e-4 … 0.9999) gives the same picture.

| readout | WF1@0.5 | best global (logit / p) | WF1 at best global | per-set oracle WF1 | gap | recovered by one global shift | set-specific remainder |
|---|---|---|---|---|---|---|---|
| B2f-top5 | 91.98 | +3.10 / 0.957 | 92.39 | 92.59 | 0.61 | 0.41 (67 %) | 0.20 |
| B2f-ep8 | 92.14 | +4.40 / 0.988 | 92.58 | 92.82 | 0.69 | 0.44 (64 %) | 0.24 |
| **B3a-top5** | 93.06 | **−0.15 / 0.463** | 93.08 | 93.24 | **0.18** | **0.02** | 0.16 |
| **B3a-ep8** | 92.94 | +4.95 / 0.993 | 93.10 | 93.45 | 0.51 | 0.16 (31 %) | 0.35 |

**B3a removed the spoof-side operating-point bias that B2f had.** B2f's best global threshold sat at logit +3 to +4. B3a's merge is optimal at 0.5, so any calibration / CE-weight / threshold lever is now worth at most 0.02 on the objective readout. **What remains is ranking.**

### 5.3 The saturated bona tail did not shrink, and on the final epoch it grew

Share of half_B bona with p_spoof ≥ 0.5 / ≥ 0.9 / ≥ 0.99 / ≥ 0.999, and spoof with p < 0.5:

| set (n bona) | B2f-top5 | **B3a-top5** | B2f-ep8 | **B3a-ep8** |
|---|---|---|---|---|
| clean (1,397) | 7.2 / 6.9 / 5.4 / 4.4 | 7.7 / 7.6 / 6.4 / 4.6 | 7.1 / 6.3 / 5.3 / 4.3 | 8.9 / 8.9 / **8.3** / 6.1 |
| matched (391) | 23.8 / 22.3 / 14.6 / 8.2 | 21.0 / 21.0 / 16.4 / 8.2 | 23.5 / 21.5 / 13.6 / 9.0 | 22.3 / 22.3 / **19.2** / 12.0 |
| heldout (391) | 30.7 / 28.6 / 20.2 / 9.2 | 25.1 / 25.1 / 18.9 / 10.2 | 30.2 / 28.4 / 20.5 / 9.5 | 24.6 / 24.3 / 20.5 / 13.0 |
| echo (391) | 16.6 / 15.6 / 10.5 / 4.9 | 14.8 / 14.8 / 12.3 / 5.4 | 15.3 / 14.8 / 10.0 / 5.4 | 16.6 / 16.4 / **15.3** / 9.5 |
| noisy bona (782): FA % / ≥ 0.99 % / saturated share of FA | 27.2 / 17.4 / 64 % | 23.0 / 17.6 / **77 %** | 26.9 / 17.0 / 63 % | 23.4 / 19.8 / **85 %** |
| spoof p < 0.5 (clean / matched / heldout / echo) | 6 / 8 / 16 / 4 | 4 / 10 / 15 / 3 | 3 / 9 / 14 / 5 | 5 / 5 / 16 / 1 |

- **Consistency fixed the movable band (p 0.5-0.99) but not the saturated core.** In B3a the 0.5 → 0.9 band is nearly empty: matched 21.0 / 21.0 and heldout 25.1 / 25.1 at the merge.
- Every remaining false alarm is a confident one. 77-85 % of noisy bona false alarms sit at p ≥ 0.99, against 63-64 % in B2f.
- This core (16-20 % of noisy bona) has not moved since B2c (B2 wave summary). The B3a-ep8 core is the largest of the lineage.

### 5.4 The worst slices for bona recall are English and SNR 0

- **English carries half of what is left.** On half_B, English is 258 of the 782 noisy bona but 91 of the 180 false alarms (merge; ep8: 93 of 183).
  - B3a removed 4 English false alarms (95 → 91) and 29 Chinese ones (118 → 89).
  - English R_bona on half B, merge: matched 67.4 / heldout 62.0 / echo 73.6, against zh 84.7 / 81.3 / 90.8.
  - The share of English bona at ≥ 0.99 rose: matched 20.9 → 26.4 %, echo 16.3 → 23.3 % (merge); matched 20.9 → 31.8 % and echo 16.3 → 28.7 % (ep8).
- Worst slices by R_bona, full sets, B3a-top5 (ep8 in brackets):

  | slice | R_bona merge (ep8) |
  |---|---|
  | heldout en | 60.9 (60.5) |
  | heldout snr 0 | 65.3 (64.6) |
  | matched en | 65.5 (63.2) |
  | matched snr 0 | 65.8 (66.7) |
  | heldout washing_machine | 66.7 (69.2) |
  | matched opus:qq | 68.7 (64.6) |
  | heldout wind | 71.1 (72.4) |
  | heldout g726 | 71.1 (72.4) |
  | matched opus:wechat | 72.1 (71.2) |
  | matched rain | 72.2 (68.7) |
  | heldout gsm | 72.8 (73.6) |

- SNR series of R_bona at the merge (full set):

  | set | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB | none |
  |---|---|---|---|---|---|---|
  | matched | 85.8 | 82.6 | 76.6 | 77.1 | **65.8** | 86.5 |
  | heldout | 79.9 | 74.2 | 78.3 | 77.6 | **65.3** | - |

### 5.5 Overfitting signs and the confound

- **Overfitting signs.** Dev loss rose 74 % while train acc reached 99.71 %. The mini WF1 fell after epoch 6, and the final epoch lost to the merge on both halves. Clean and echo bona saturation grew at ep8, and limiter-probe flips rose 6 → 14 clips. Twice the optimiser steps at the same LR schedule over-trained the tail of the cosine.
- **The confound.** The lever doubles the steps and the views per clip as well as adding the JS term, which is 2.5-3.9 % of the loss. This run cannot say which of the three produced the +1.08.
- **Unchanged.** `noise_only` / `sil_only` bona-source clips are still 100 % spoof (2.3), which was part of the predicted signature.

## 6. Verdict on the pre-registered rule

From `exploop/B3/iter01/design.result.json`:

- **Falsifier:** "delta top-5-merge half_B WF1 vs B2f merge > +0.5 with the paired-bootstrap CI excluding 0 and clean F1 drop <= 0.5 -> promote; within +-0.5 -> null; below -0.5 -> negative".
- **Predicted signature:** "noisy EER down, noise_only bona->spoof down, heldout / matched bona recall up, clean flat".

`exploop/B3/iter01/verdict.json` (computed by the supervisor; quoted, not recomputed):

| readout | run | base | delta | CI | clean_drop | new_best | decision | verdict |
|---|---|---|---|---|---|---|---|---|
| merge (objective) | 93.05618353351335 | 91.97880641320396 | 1.077377120309393 | [0.518028201906214, 1.6680022268861796] | 0.11957519224326063 | true | **promote** | **positive** |
| final | 92.94058874557216 | 92.13685698540242 | 0.8037317601697396 | [0.21590309146058556, 1.4240957065638762] | 0.6303024637734467 | true | revisit | above band, clean F1 dropped |

**The promote branch fired on the objective readout.**

The predicted signature, item by item:

| predicted | merge | final | appeared? |
|---|---|---|---|
| noisy EER down | 10.09 → 9.16 (matched −0.25, heldout −1.61, echo −0.10) | 9.66 → 9.01 (matched −0.47, heldout −0.82, echo **+0.76**) | **yes** on the WF1 sets; echo up on the final epoch |
| noise_only bona → spoof down | 100.00 → 100.00 (mean dP 0.887 → 0.876) | 100.00 → 100.00 (0.897 → 0.882) | **no** |
| heldout bona recall up | 69.31 → 74.94 (+5.63) | 69.82 → 75.45 (+5.63) | **yes** |
| matched bona recall up | 76.21 → 79.03 (+2.81) | 76.47 → 77.75 (+1.28) | **yes** |
| clean flat | 97.60 → 97.48 (−0.12, CI spans 0) | 97.69 → 97.06 (−0.63, CI excludes 0) | **yes** on the merge, **no** on the final epoch |

Three of the four predicted items appeared on the objective readout. The one that did not (`noise_only`) shows that consistency between two augmented speech views does not teach the model what to do with non-speech input.

## 7. Decision and next-experiment proposals

**Decision.** Per `verdict.json`, **B3a is the new best** (merge `new_best true`, `decision promote`). The supervisor moves the reference to B3a: merge 93.06 and final 92.94 on half_B; registry ids `B3a-cjs03-screen` and `B3a-cjs03-ep8`. Every proposal below is one factor (or one coupled group) against **B3a's flags**: B2f + `--consistency js --consistency_lambda 0.3 --consistency_pairs both --batch_size 8`. Each keeps the ~8 h runtime of a consistency run. Every falsifier uses the same rule as this iteration, measured against B3a-top5 = 93.06 on half_B:
- promote: Δ merge > +0.5 with the CI excluding 0 and a clean drop of at most 0.5;
- null: Δ within ±0.5;
- negative: Δ below −0.5.

**The largest remaining error.** Confident noisy bona false alarms, of which English carries half:
- 91 of 180 half_B noisy bona false alarms are English, from 258 of 782 bona;
- 77 % of noisy bona false alarms sit at p ≥ 0.99;
- full-set en EER did not improve, and rose at ep8.

Calibration is no longer a lever: one global shift recovers 0.02 on the merge.

**Ranked proposals (configuration first, seed repeat last):**

1. **Class- and language-balanced sampler, CE weights re-derived to keep the effective prior.**
   - Flags: `--sampler balanced --sampler_lang_balance --sampler_bona_frac 0.5 --ce_weights 0.3223 0.6777`.
   - CE-weight derivation (from `train_label.txt`): 14,353 bona of 75,785, i.e. 18.94 % bona. With 0.1 / 0.9 the effective bona:spoof weight is 2.103 : 1. At a 0.5 bona share the same ratio gives `0.3223 0.6777` (order `[spoof, bonafide]`, `wave.json` lever_notes).
   - Targets: English bona, 2,922 of the 75,785 training clips (`config.yaml` `train_info`). With lang balance each English bona clip is drawn ~6.5× per epoch.
   - Hypothesis: the en-zh gap of 17-19 R_bona points on half B is an exposure problem that consistency could not fix, because the views of an English clip agree with each other while being wrong together.
   - Predicted signature: half_B English R_bona up by ≥ 3 points on matched and heldout (merge 67.4 / 62.0); the English share at ≥ 0.99 down (26.4 / 28.7 %); full-set en EER down (13.66 / 17.04); zh and clean flat; `thr_oracle` stays near 0.5 (merge best global logit −0.15).
   - Falsifier: Δ merge within ±0.5 **and** English R_bona within ±3 means exposure is not the cause. A clean drop > 0.5, or a best global logit moving beyond ±1, means the prior derivation failed. Dev loss above B3a's 0.172 at epoch 8 means the repeated English draws over-fit.
2. **Consistency dose: `--consistency_lambda 1.0`** (`wave.json` lever_queue `consistency_dose`, "if B3a's consistency helps").
   - Hypothesis: if the JS divergence is what buys invariance, then at 0.3 it is under-weighted (2.5-3.9 % of the loss, and JS fell to 0.0038 by epoch 8). Tripling it should cut the saturated core, which the movable-band fix left untouched.
   - Predicted signature: merge noisy EER below 9.16; heldout R_bona above 74.94; the noisy bona share at ≥ 0.99 below 17.6 %; JS at epoch 8 lower than 0.0038; clean flat.
   - Falsifier: Δ merge within ±0.5 means the B3a gain came from the paired views and the doubled steps, not the divergence weight, so stop dosing and keep 0.3. A clean drop > 0.5 means over-regularised.
   - The alternative in the queue, `--consistency js_emb`, ranks below this as a second dose step.
3. **SNR range to 0-20 dB: `--aug_snr_db 0 20`** (lever_queue `snr_0_20`).
   - Targets: SNR 0 is the worst SNR slice on both sets (full-set merge R_bona 65.8 / 65.3, against 85.8 / 79.9 at 20 dB). Consistency moved it by only +0.9 to +3.4, and training never sees 0 dB.
   - Hypothesis: the SNR-0 bona false alarms are out-of-range inputs, not a lack of invariance.
   - Predicted signature: SNR-0 R_bona up by ≥ 3 on both sets; SNR-0 EER down (ep8 16.37 / 15.00); slices at ≥ 10 dB flat; clean flat.
   - Falsifier: Δ merge within ±0.5 and SNR-0 R_bona within ±3 means SNR range is not the limit. A drop of > 2 R_bona at 15-20 dB means the uniform 0-20 draw diluted the high-SNR mass.
4. **SpecAugment: `--specaug`** (lever_queue `specaug`).
   - Targets: the over-fitting tail. Dev loss rose 0.099 → 0.172, mini WF1 fell after epoch 6, the final epoch lost to the merge for the first time, and clean and echo bona saturation grew at ep8 (5.1, 5.5).
   - Hypothesis: with twice the steps per epoch, B3a needs a regulariser that does not touch the class prior.
   - Predicted signature: dev loss at epoch 8 well below 0.172; mini WF1 not falling over epochs 6-8; final ≥ merge on half B; clean ep8 F1 not below 97.48; clean bona share at ≥ 0.99 at ep8 back to ≤ 6.4 %.
   - Falsifier: Δ merge within ±0.5 and dev loss at epoch 8 ≥ 0.17 means masking does not regularise this backbone.
5. **Seed repeat of B3a (`--seed 1235`, last).**
   - Purpose: measures the seed band of the +1.08. It is the first lever above the near-replicate band, decided on one seed.
   - Predicted signature: merge within ±0.5 of 93.06, with the heldout gain (> +1.5 vs B2f) reproduced.
   - Falsifier: a repeat merge below 92.48 (B2f + 0.5) means B3a's gain is inside seed noise.

**Recommended full run if the wave stopped now.**
- Flags: B3a's exact `config.yaml` command, i.e. `scripts/run_B3a.sh`, at 2-3 seeds (1234, 1235, 1236).
- Readout to submit: **the top-5 merge**. It is the wave objective, it beat the final epoch on half B (93.06 vs 92.94) and on half A (93.04 vs 92.61), and it does not carry the final epoch's clean drop. Keep `--keep_topk_by_wf1 7 --keep_epochs 8` so the merge candidates and the final epoch both exist.
- Score-average the seeds (or B2f + B3a: 93.40 on half_B, section 4) before deciding on a weight average.
- Pre-launch checks:
  - `bash -n scripts/run_B3a.sh`;
  - `which ffmpeg` (a restart wipes it and the codec stage silently no-ops);
  - `pyloudnorm` importable;
  - an empty `pgrep -af "b3wave.train|main_train"`;
  - an idle GPU.
- Post-launch checks (epoch 1):
  - the `Consistency: js lambda_js=0.3 … found': 37125 … 16 clips per step (8 pairs)` header;
  - `Warmup-cosine: 3790 warmup / 75792 steps`;
  - epoch-1 `CE≈0.39 JS≈0.034`, mini WF1 ≈ 90.2;
  - codec counts summing to 2 × 75,785.
- Expect ~7 h 52 min per seed.
- Every evaluation needs `--loudness_norm --loudness_norm_lufs -23`.
