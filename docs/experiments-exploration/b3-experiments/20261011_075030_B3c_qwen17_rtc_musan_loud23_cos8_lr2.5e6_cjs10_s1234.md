# Experiment exploration B3-03: B3c (`B3c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs10_s1234`) - JS consistency dose step (λ 0.3 → 1.0) on top of B3a

> Written 2026-10-11 by the exploop B3 analysis agent (ANALYZE step, wave B3, iteration 3, attempt a0).
>
> **Sources.** Every number below is either copied from a file read in this session or computed in this session from the half_B (and, where stated, half_A) rows of the local `scores/*.txt` / `*.logits` files, joined with the protocol and meta files of the local sets. Nothing is quoted from memory. The files:
> - run config and logs: `config.yaml` (B3c and B3a), `git.diff` (B3c; file headers only), `logs/train.log` (both runs), `metrics.jsonl` (both runs), `top5_by_wf1/{select_topk.log,merge_list.txt}`, `local_eval/merge_top5/merge_list.txt`, the `ckpt/` listing;
> - local evals: `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,compare_vs_B3a-*.txt,ci_vs_B3a-*.txt,compare_ci_vs_B3a-*.json}`, `local_eval/merge_top5/compare_all_readouts.txt`, `local_eval/merge_top5.log`, `local_eval/epoch_8.log` (summary lines); the same `metrics.json` / `probe_report.json` / `scores/` of B3a (`merge_top5`, `epoch_8`) and B2f; the `half_B.wf1` block of B3b's two `metrics.json`;
> - stdout log: `SLS_setup/B3c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs10_s1234_20261010234154.log` (header lines only);
> - loop files: `exploop/B3/iter03/{verdict.json,design.result.json}`, `exploop/B3/{wave.json,plan_snapshot.md,STATUS.md,user_notes.md}`, `scripts/run_B3c.sh`;
> - registry and code: `docs/experiment_log.csv`, `SLS_setup/b3wave/train.py` (train loss = CE + consistency term; flag help texts), `SLS_setup/b3wave/losses.py` (`consistency_loss`), `SLS_setup/utils/local_eval.py` (half filter), `SLS_setup/utils/metrics.py` (macro-F1, spoof if `p >= thr`), `configs/local_eval_sets.yaml`, `data/dev_noisy_sim/*/{protocol.txt,meta.csv}`.
>
> Progress-split scores were not read (`model_merging_top5/scores.txt` was not opened).
>
> **Previous reports.** `docs/experiments-exploration/b3-experiments/20261010_143058_B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234.md` (the reference run; numbers marked "B3a report"), `docs/experiments-exploration/b3-experiments/20261010_231745_B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_sampbal_s1234.md` (iteration 2; "B3b report"), and `docs/experiments-exploration/20261009_175355_B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234.md` (template, fixed comparator).
>
> **Readouts.**
> - `-top5` is the weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5/`). It is the wave objective.
> - `-ep8` is the final-epoch EMA checkpoint alone (`local_eval/epoch_8/`).
> - half_B is the half of each sim set that is never used for selection: n = 1,994 per set, 391 bona (`metrics.json`). Clean is the full 7,465-clip `dev_online_clean` (1,397 bona; 479 English, 918 Chinese).
> - WF1 = 0.3 clean + 0.35 matched + 0.35 heldout (macro-F1 at p_spoof 0.5). Echo is reported but is not in WF1.
> - Threshold 0.5 on the score files reproduces all six half_B WF1 values of the `metrics.json` files exactly: B2f 91.98 / 92.14, B3a 93.06 / 92.94, B3c 92.29 / 92.44 (top5 / ep8).
> - Every evaluation ran with the loudness flag: `[loudness_norm] loudness_norm target=-23 LUFS` is in `merge_top5.log` and `epoch_8.log`, and every `metrics.json` has `"loudness_norm": -23.0`.

Run directories (under `SLS_setup/exp/`):
- **B3c** `B3c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs10_s1234_epoch8_bs8_20261010234200` (this report)
- **B3a** `B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_epoch8_bs8_20261009214846` (reference = current best: merge 93.06, final 92.94)
- **B3b** `B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_sampbal_s1234_epoch8_bs8_20261010145615` (iteration 2, dropped: 92.23 / 92.36)
- **B2f** `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623` (fixed comparator: 91.98 / 92.14)

## 1. Setup

**B3c = B3a with one flag changed: `--consistency_lambda 0.3 → 1.0`** (the JS consistency dose step; `design.result.json` `change`). Everything else is identical, including `--consistency js --consistency_pairs both --batch_size 8`. The aim was to isolate the divergence weight from B3a's step/view confound (B3a report §5.5).

Flag diff (`config.yaml` of both runs, `diff` output; nothing else differs):

| flag / field | B3a | **B3c** | note |
|---|---|---|---|
| `--consistency_lambda` | 0.3 | **1.0** | the lever |
| `--track` | `..._cjs03_s1234` | `..._cjs10_s1234` | name only |
| git | `df3dff2`, `git_dirty: false` | `df3dff2`, `git_dirty: true` | `git.diff` touches only `.claude/skills/post-train-eval/SKILL.md` and `docs/experiment_log.csv`; no trainer code |
| everything else | identical | identical | `--consistency js --consistency_pairs both`, `--batch_size 8` (16 clips/step), 8 epochs, `--lr 2.5e-6 --lr_min 5e-7` warmup-cosine, `--ce_weights 0.1 0.9`, sampler shuffle, RTC aug + MUSAN, `--loudness_norm -23`, `--keep_topk_by_wf1 7 --keep_epochs 8`, seed 1234 |

**What the loss is.** `train.py` computes `batch_loss = ce + cons`, where `cons = lam_js * JS(p_a, p_b)` (`losses.py`, mode `js`, `clean_view=False`, so both views receive the JS gradient). So `TrainLoss = CE + 1.0·JS` here. Epoch 1: 0.4003 + 0.0332 = 0.4335, the logged `TrainLoss=0.433465`.

**Header lines proving the change took effect** (stdout log). They match all three `expect` regexes of the design:
- `Consistency: js lambda_js=1 lambda_emb=0.3 pairs=both clean_view=False | pairs found {'rows': 38660, 'blank_online': 1535, 'found': 37125} dropped_without_twin=0 | 16 clips per step (8 pairs)`
- `Sampler: shuffle`
- `CE weights: spoof=0.1 bonafide=0.9`
- Unchanged: `Warmup-cosine: 3790 warmup / 75792 steps, floor 5.0e-07` (same as B3a/B3b), `SNR: 5..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: fixed`, `LLRD: 1`, `SpecAugment: off`, `Loudness norm: loudness_norm target=-23 LUFS`, `Train trials: 75785`, `Parameters: 320268933`.
- Per-epoch augmentation counts (`metrics.jsonl`, computed): the codec counts sum to 151,570 = 2 × 75,785 in every epoch (two views per clip), and the stage counts are 0.7988-0.8014 of the views (`p_apply=0.8`).
- The stdout log has no `Traceback`, `out of memory`, `failed (`, `Pruned`, `Error` or `WARNING` line (grep count 0 each).

**Run time.** `config.yaml` created 23:42:10; epoch 8 done 07:27:22; last local eval 07:27:46 = **7 h 45 min** (registry `gpu_h` 7.76; B3a 7.87). Epoch-end to epoch-end spacing is 57.8-58.2 min (`train.log`). Post-train file times: `top5_by_wf1/merge_list.txt` 07:28:27, `merge_top5/metrics.json` 07:33:50, `epoch_8/metrics.json` 07:39:04, `model_merging_top5/submission.zip` 07:49:30. `verdict.json` was written at 07:50:30.

**Checkpoints.** Nothing was pruned: all 8 epoch files are present at 1,281,235,691 bytes each. Symlinks: `best_by_dev_loss` → epoch 1 (0.093498), `best_by_noisy_eer` → epoch 3, `best_by_wf1` → epoch 7.

**Merged top-5 by half-A mini WF1 = epochs 7, 8, 5, 6, 4** (92.92, 92.81, 92.63, 92.28, 92.03; mean 92.54; `select_topk.log`). Excluded: 2 (91.58), 3 (91.51), 1 (89.86). B3a's merge was 6, 5, 7, 8, 2 (mean 92.31; B3a `select_topk.log`).

## 2. Results vs references (half_B)

Sources: `compare_all_readouts.txt`, `compare_vs_B3a-*.txt`, `ci_vs_B3a-*.txt` / `compare_ci_vs_B3a-*.json` (paired bootstrap n = 2,000, half B, threshold 0.5), the `half_B` block of each `metrics.json`, and B3b's `metrics.json` for its column.

### 2.1 Wave iterations and fixed comparators, both readouts

| half_B | B2f-top5 | B2f-ep8 | B3a-top5 | B3a-ep8 | B3b-top5 | B3b-ep8 | **B3c-top5** | **B3c-ep8** | B3c-top5 − B3a-top5 [95 % CI] | B3c-ep8 − B3a-ep8 [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| **WF1@0.5** | 91.98 | 92.14 | 93.06 | 92.94 | 92.23 | 92.36 | **92.29** | **92.44** | **−0.77 [−1.31, −0.22]**, p(not better) 0.999 | **−0.50 [−1.05, +0.05]**, p 0.963 |
| WF1@oracle | 92.59 | 92.82 | 93.24 | 93.45 | 92.96 | 92.86 | 92.63 | 92.73 | −0.61 | −0.72 |
| clean F1@0.5 | 97.60 | 97.69 | 97.48 | 97.06 | 96.96 | 96.77 | 97.25 | 97.20 | −0.24 [−0.52, +0.02] | +0.14 [−0.20, +0.47] |
| noisy F1@0.5 | 89.57 | 89.76 | 91.16 | 91.17 | 90.20 | 90.47 | 90.17 | 90.40 | −0.99 | −0.77 |
| mean noisy EER | 10.09 | 9.66 | 9.16 | 9.01 | 10.15 | 10.17 | **9.96** | **9.71** | **+0.80** | **+0.70** |
| calibration gap noisy | 0.66 | 0.71 | 0.11 | 0.42 | 0.62 | 0.21 | 0.28 | 0.12 | +0.17 | −0.30 |
| separability gap | 7.87 | 7.86 | 6.56 | 6.20 | 7.12 | 7.27 | 7.29 | 7.36 | +0.73 | +1.16 |

Per-set paired CIs (`ci_vs_B3a-*.txt`):

| set | merge Δ [CI] | ep8 Δ [CI] |
|---|---|---|
| clean | −0.24 [−0.52, +0.02] | +0.14 [−0.20, +0.47] |
| matched | −0.34 [−1.42, +0.76] | −0.08 [−1.18, +0.93] |
| heldout | **−1.65 [−2.81, −0.55]** | **−1.46 [−2.68, −0.28]** |
| echo | **−0.96 [−1.86, −0.07]** | +0.06 [−0.84, +1.01] |

Weighted contributions (0.3 / 0.35 / 0.35 × the unrounded deltas in `compare_ci_vs_B3a-*.json`, computed):
- merge: −0.07 clean − 0.12 matched − 0.58 heldout = −0.77. **Heldout carries 75 % of the merge loss.**
- final: +0.04 − 0.03 − 0.51 = −0.50. **Heldout carries all of the final-epoch loss.**

The same set moved most for B3a, in the other direction: heldout carried three quarters of B3a's +1.08 over B2f (B3a report §2.1).

**Other halves of the same sets:**

| readout | full-set WF1 B3a → B3c (`metrics.json` `full.wf1`) | half_A WF1 B3a → B3c (computed from `scores/*.txt`) |
|---|---|---|
| merge | 93.05 → 92.64 (−0.41) | 93.04 → 93.01 (−0.03) |
| final | 92.78 → 92.73 (−0.05) | 92.61 → **93.02 (+0.41)** |

**The WF1 loss is specific to half B. The ranking loss is not.** On half A, B3c ties B3a on the merge and beats it on the final epoch. The heldout EER is worse on both halves, though (computed, same EER definition as `metrics.json`, which it reproduces on half B):

| set / half | B3a-top5 → B3c-top5 EER | B3a-ep8 → B3c-ep8 EER | F1 merge / ep8 |
|---|---|---|---|
| heldout A | 10.44 → **11.96** | 11.10 → **12.11** | 89.82 → 89.91 / 89.85 → 89.85 |
| heldout B | 10.42 → **11.56** | 10.29 → **11.21** | 90.20 → 88.55 / 90.32 → 88.86 |
| matched A | 8.00 → 8.77 | 8.40 → 8.39 | 92.45 → 92.46 / 91.55 → 92.60 |
| matched B | 7.89 → 8.37 | 7.74 → 8.21 | 92.12 → 91.78 / 92.03 → 91.95 |

Half A is the selection half, so its merge numbers are optimistic for both runs equally (B3a report §2.1). The final epoch has no selection, though, and it swings by 0.91 between the halves (+0.41 on A, −0.50 on B). `wave.json` lever_notes give the B2 near-replicate noise band as "~0.2-0.5 WF1 on the final readout, up to ~1.4 per set". The F1@0.5 deltas of B3c are therefore close to that band. The heldout EER loss of +0.9 to +1.5 on both halves and both readouts is the robust part of the result.

### 2.2 Per set (R = recall at 0.5; thr = p_spoof threshold; `metrics.json` half_B)

| set | metric | B3a-top5 | **B3c-top5** | B3a-ep8 | **B3c-ep8** |
|---|---|---|---|---|---|
| dev_online_clean | F1@.5 / F1@or / thr_or | 97.48 / 97.83 / 0.996 | 97.25 / 97.73 / 1.000 | 97.06 / 97.79 / 1.000 | 97.20 / 97.88 / 1.000 |
| | EER / ECE | 3.32 / 1.47 | **3.15** / 1.60 | 3.20 / 1.72 | **3.10** / 1.64 |
| | R_spoof / R_bona | 99.93 / 92.34 | 99.97 / **91.48** | 99.92 / 91.12 | 99.93 / 91.48 |
| sim_matched_v1 | F1@.5 / F1@or / thr_or | 92.12 / 92.25 / 0.995 | 91.78 / 91.93 / **0.150** | 92.03 / 92.49 / 0.998 | 91.95 / 92.04 / 0.848 |
| | EER / ECE | 7.89 / 4.51 | **8.37** / 4.63 | 7.74 / 4.55 | **8.21** / 4.69 |
| | R_spoof / R_bona | 99.38 / 79.03 | 99.56 / **77.49** | 99.69 / 77.75 | 99.31 / 78.77 |
| sim_heldout_v1 | F1@.5 / F1@or / thr_or | 90.20 / 90.30 / 0.971 | **88.55** / 88.96 / 0.992 | 90.32 / 90.69 / 0.993 | **88.86** / 89.00 / 0.992 |
| | EER / ECE | 10.42 / 5.54 | **11.56 / 6.39** | 10.29 / 5.57 | **11.21 / 6.27** |
| | R_spoof / R_bona | 99.06 / 74.94 | 99.19 / **70.33** | 99.00 / 75.45 | 99.19 / **71.10** |
| sim_echo_v1 | F1@.5 / F1@or / thr_or | 94.87 / 95.03 / 0.998 | **93.91** / 94.66 / 0.991 | 94.39 / 94.96 / 0.999 | 94.45 / 94.82 / 0.990 |
| | EER / ECE | 4.61 / 3.00 | **6.06** / 3.54 | 5.62 / 3.28 | 5.31 / 3.28 |
| | R_spoof / R_bona | 99.81 / 85.17 | 99.63 / 83.12 | 99.94 / 83.38 | 99.63 / 84.65 |

Reading:
- **Heldout bona recall fell 4.6 / 4.35 points at unchanged spoof recall** (R_spoof +0.12 / +0.19), with EER up 1.13 / 0.92. This is a ranking loss, not a threshold move.
- **Spoof recall held everywhere.** The largest R_spoof drop is −0.37 (matched, ep8). Clean EER even improved slightly (3.32 → 3.15, 3.20 → 3.10).
- **Matched did not significantly lose:** −0.34 / −0.08 F1, CIs spanning 0. Matched R_bona was −1.53 at the merge and +1.02 at ep8.
- Paired per clip on half_B, B3c vs B3a (computed; fixed / broken / wrong in both; by language: fixed / broken):

  | set | class | merge | merge en · zh | ep8 | ep8 en · zh |
  |---|---|---|---|---|---|
  | clean | bona | 10 / **22** / 97 | 2 / 11 · 8 / 11 | 25 / 20 / 99 | 13 / 8 · 12 / 12 |
  | matched | bona | 10 / 16 / 72 | **1 / 11** · 9 / 5 | 16 / 12 / 71 | 6 / 8 · 10 / 4 |
  | heldout | bona | 6 / **24** / 92 | 2 / 8 · **4 / 16** | 8 / **25** / 88 | 4 / 10 · **4 / 15** |
  | echo | bona | 6 / 14 / 52 | 3 / 8 · 3 / 6 | 15 / 10 / 50 | 8 / 4 · 7 / 6 |
  | matched | spoof | 5 / 2 / 5 | 2 / 0 · 3 / 2 | 0 / 6 / 5 | 0 / 0 · 0 / 6 |
  | heldout | spoof | 4 / 2 / 11 | 0 / 0 · 4 / 2 | 6 / 3 / 10 | 2 / 2 · 4 / 1 |
  | echo | spoof | 1 / 4 / 2 | 1 / 1 · 0 / 3 | 0 / 5 / 1 | 0 / 2 · 0 / 3 |

  **Summed over the four sets, bona: English 8 fixed / 38 broken and Chinese 24 / 38 (merge); English 31 / 30 and Chinese 33 / 37 (ep8).** The heldout bona loss is lopsided in both readouts (6 / 24 and 8 / 25), and two thirds of it is Chinese.
- Logit correlation B3c vs B3a (computed, half_B): 0.974-0.985 per set on the merge, 0.969-0.982 on ep8. B3c is a smaller perturbation of B3a than B3b was (0.938-0.948, B3b report §2.2) and than B3a was of B2f (0.954-0.967, B3a report §4).

### 2.3 Probes (`probe_report.json`; 187 bona + 813 spoof; one bona probe clip = 0.53 points)

| probe (bona unless stated) | B3a-top5 | **B3c-top5** | B3a-ep8 | **B3c-ep8** |
|---|---|---|---|---|
| level_flip_bona (mean) / max | 0.80 / 2.67 | **1.80 / 6.42** | 2.07 / 7.49 | 1.67 / 6.42 |
| gain_up / gain_p6 / gain_p10_limit flip % | 2.67 / 1.07 / 2.67 | **6.42** / 1.07 / **5.35** | 4.81 / 2.14 / 7.49 | 6.42 / 2.14 / 4.28 |
| agc_dynaudnorm / loudnorm flip % | 0.00 / 0.00 | 0.00 / 1.60 | 0.00 / 2.14 | 0.53 / 0.00 |
| level_flip_spoof | 0.02 | 0.00 | 0.00 | 0.00 |
| probe_orig: bona pred spoof % | 12.30 | 12.30 | 11.76 | 12.30 |
| noise_only: bona-source pred spoof %, mean dP | 100.00, +0.876 | 100.00, +0.877 | 100.00, +0.882 | 100.00, +0.877 |
| sil_only: bona-source pred spoof %, mean dP | 100.00, +0.877 | 100.00, +0.877 | 100.00, +0.882 | 100.00, +0.877 |
| trimmed / padded bona flip % | 2.14 / 3.21 | 2.14 / 4.28 | 3.21 / 2.67 | 3.21 / 4.28 |

Reading:
- Under loudness norm the gain-shape probes `gain_down`, `gain_m20` and `gain_m6` give 0.00 % flips in all four readouts, as in every loudness-norm run.
- **The merge lost part of B3a-top5's level robustness:** `gain_up` flips went from 5 to 12 bona clips (2.67 → 6.42 %). The final epoch is flat to slightly better (level_flip_bona 2.07 → 1.67).
- `noise_only` / `sil_only` bona-source clips are still 100 % spoof in every readout. A higher JS weight does nothing for non-speech input.

### 2.4 Slices that moved most, B3c vs B3a (full sim sets, `metrics.json → breakdowns`, 63 slice rows)

| readout | F1 up | EER down | R_bona up | R_spoof up | min slice R_spoof (B3a → B3c) |
|---|---|---|---|---|---|
| merge | **13 of 63** (3 ties) | **10** | **11** | 36 | 97.02 → 98.01 (heldout washing_machine) |
| final | 28 of 63 (4 ties) | 18 | 31 | 14 | 96.69 → 97.35 (heldout washing_machine) |

Largest moves (full sets; F1, EER, R_bona; B3a → B3c):

| readout | set / slice | n (bona) | F1 (Δ) | EER | R_bona |
|---|---|---|---|---|---|
| merge | heldout / door_wood_creaks | 361 (64) | 90.88 → 87.50 (−3.38) | 7.44 → **14.10** | 78.12 → 70.31 |
| merge | heldout / snr 5 | 778 (156) | 90.39 → 87.53 (−2.86) | 10.11 → 11.00 | 77.56 → **69.23** |
| merge | matched / opus:voov | 510 (89) | 94.58 → 92.59 (−1.99) | 6.93 → 6.70 | 84.27 → 78.65 |
| merge | heldout / lang en | 1538 (258) | 84.74 → 83.01 (−1.73) | 17.04 → **20.27** | 60.85 → 56.20 |
| merge | heldout / washing_machine | 380 (78) | 84.59 → 82.94 (−1.66) | 15.64 → 14.31 | 66.67 → 60.26 |
| merge | heldout / codec g726 | 1274 (246) | 88.81 → 87.36 (−1.45) | 10.74 → **13.37** | 71.14 → 66.67 |
| merge | matched / snr 5 | 656 (118) | 90.34 → 92.11 (+1.77) | 10.10 → 11.18 | 77.12 → 81.36 |
| merge | matched / opus:telegram | 552 (111) | 91.58 → 93.44 (+1.85) | 6.10 → 7.46 | 79.28 → 82.88 |
| final | heldout / snr 5 | 778 (156) | 91.10 → 88.06 (−3.04) | 11.00 → 12.12 | 79.49 → **72.44** |
| final | heldout / door_wood_knock | 353 (54) | 91.84 → 89.84 (−2.00) | 10.32 → 7.81 | 79.63 → 74.07 |
| final | heldout / codec g726 | 1274 (246) | 89.29 → 87.46 (−1.83) | 11.33 → 12.08 | 72.36 → 67.48 |
| final | heldout / lang en | 1538 (258) | 84.26 → 82.90 (−1.35) | 17.63 → 19.18 | 60.47 → 57.36 |
| final | matched / snr 10 | 653 (124) | 89.62 → 91.25 (+1.63) | 7.41 → 7.98 | 70.97 → 75.81 |
| final | matched / footsteps | 683 (133) | 91.69 → 93.29 (+1.60) | 6.19 → 7.40 | 78.20 → 81.95 |

SNR 0 (full sets; the design's prediction was for the merge):

| set | merge EER B3a → B3c | merge R_bona | ep8 EER B3a → B3c |
|---|---|---|---|
| matched snr 0 | 14.67 → 15.35 | 65.81 → 66.67 | 16.37 → 14.17 |
| heldout snr 0 | 15.08 → **19.07** | 65.31 → 62.59 | 15.00 → **19.60** |

**Pattern: the losses are mostly on unseen conditions; the gains are on seen ones.**
- Of the largest drops in the table, all but one (matched opus:voov at the merge) are heldout slices: unseen noise families, the unseen g726 codec, English, SNR 5.
- The largest gains are matched slices: training noise families, SNR 5 / 10, the training Opus profiles.
- Heldout SNR-0 EER worsened by 4.0 (merge) and 4.6 (final) points.

### 2.5 Lever response (one factor per row)

| run | change vs reference | half_B merge WF1 (Δ) | final WF1 (Δ) | merge oracle | final oracle | merge noisy EER | final noisy EER | merge clean | final clean | verdict (verdict.json) |
|---|---|---|---|---|---|---|---|---|---|---|
| B2f | (wave start) | 91.98 | 92.14 | 92.59 | 92.82 | 10.09 | 9.66 | 97.60 | 97.69 | - |
| B3a | + `--consistency js --consistency_lambda 0.3 --consistency_pairs both`, `--batch_size 8` (vs B2f) | 93.06 (+1.08) | 92.94 (+0.80) | 93.24 | 93.45 | 9.16 | 9.01 | 97.48 | 97.06 | merge positive / promote |
| B3b | + `--sampler balanced --sampler_lang_balance --sampler_bona_frac 0.5 --ce_weights 0.3223 0.6777` (vs B3a) | 92.23 (−0.83) | 92.36 (−0.58) | 92.96 | 92.86 | 10.15 | 10.17 | 96.96 | 96.77 | negative / drop (both) |
| **B3c** | **`--consistency_lambda 0.3 → 1.0` (vs B3a)** | **92.29 (−0.77)** | **92.44 (−0.50)** | 92.63 | 92.73 | 9.96 | 9.71 | 97.25 | 97.20 | merge **negative / drop**; final **null (inside ±0.5) / drop** |

The consistency weight has a non-monotone dose response, read off the noisy EER:
- 0 → 0.3, with views and steps doubled at the same time: 10.09 → 9.16 (merge) and 9.66 → 9.01 (final);
- 0.3 → 1.0, a pure weight change: 9.16 → 9.96 and 9.01 → 9.71.

Only the second step is a clean single-factor contrast.

## 3. Per-epoch behaviour, B3c vs B3a (same steps, same LR at each epoch end)

Sources: `metrics.jsonl` (EMA weights; 2,000 clean clips + `sim_matched_mini` + `sim_heldout_mini`, half A), `logs/train.log`, `STATUS.md` (Δ). CE weights are identical in the two runs, so CE, train acc and dev loss are directly comparable; JS is unweighted. The JS share is λ·JS / TrainLoss (computed).

| ep | LR end | B3c mini WF1 | B3a | Δ | B3c WF1@or | B3c EER noisy (B3a) | B3c R_bona cl/ma/he | B3a R_bona cl/ma/he | B3c thr_or ma/he | B3c CE / JS | B3a CE / JS | JS share B3c / B3a | train acc B3c / B3a | dev loss B3c / B3a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.469e-6 | 89.86 | 90.17 | −0.31 | 90.95 | 13.25 (13.14) | 92.4/74.3/60.7 | 91.0/73.7/65.8 | 0.445/0.861 | 0.4003 / 0.0332 | 0.3945 / 0.0344 | 7.7 / 2.5 % | 89.84 / 90.16 | 0.0935 / 0.0990 |
| 2 | 2.289e-6 | 91.58 | 91.64 | −0.06 | 91.88 | 11.50 (11.56) | 91.0/77.1/69.4 | 91.6/76.5/70.9 | 0.721/0.939 | 0.1915 / 0.0212 | 0.1867 / 0.0214 | 10.0 / 3.3 % | 97.87 / 97.87 | 0.1147 / 0.1232 |
| 3 | 1.976e-6 | 91.51 | 91.39 | +0.12 | 92.17 | **11.19** (11.49) | 90.2/77.7/66.8 | 91.6/76.5/66.8 | 0.903/0.954 | 0.1252 / 0.0152 | 0.1176 / 0.0151 | 10.8 / 3.7 % | 98.64 / 98.63 | 0.1431 / 0.1514 |
| 4 | 1.583e-6 | 92.03 | 91.52 | +0.51 | 92.80 | 11.29 (10.80) | 89.3/76.5/70.9 | 89.9/76.5/69.9 | 0.989/0.212 | 0.0902 / 0.0109 | 0.0842 / 0.0113 | 10.8 / 3.9 % | 99.07 / 99.01 | 0.1485 / 0.1657 |
| 5 | 1.175e-6 | 92.63 | 92.52 | +0.11 | 93.06 | 11.70 (11.16) | 91.0/79.3/72.4 | 91.6/81.0/71.9 | 0.939/0.136 | 0.0621 / 0.0084 | 0.0642 / 0.0080 | 11.9 / 3.6 % | 99.31 / 99.32 | 0.1489 / 0.1472 |
| 6 | 8.227e-7 | 92.28 | **92.87** | −0.59 | 93.02 | 11.67 (**10.09**) | 88.8/81.6/69.4 | 91.9/82.7/71.9 | 0.027/0.977 | 0.0489 / 0.0067 | 0.0452 / 0.0065 | 12.0 / 4.2 % | 99.50 / 99.47 | **0.1908** / 0.1356 |
| 7 | 5.842e-7 | **92.92** | 92.36 | +0.56 | 93.27 | 11.48 (10.40) | 90.4/83.2/72.4 | 91.3/80.4/71.4 | 0.907/0.984 | 0.0351 / 0.0051 | 0.0328 / 0.0051 | 12.6 / 4.5 % | 99.64 / 99.59 | 0.1894 / 0.1584 |
| 8 | 5.000e-7 | 92.81 | 92.14 | +0.67 | 93.30 | 11.26 (10.28) | 90.4/81.0/74.0 | 89.9/78.8/71.9 | 0.007/0.882 | 0.0339 / **0.0044** | 0.0283 / 0.0038 | 11.4 / 3.9 % | 99.66 / 99.71 | **0.1773** / 0.1722 |

Reading:
- **The weight tripled, but the divergence it penalises did not fall.**
  - Unweighted JS is within ±0.0012 of B3a's at every epoch: lower at epochs 1, 2 and 4, equal at 7, higher at 3, 5, 6 and 8. It ends at 0.0044 against 0.0038.
  - CE is higher than B3a's at 7 of 8 epochs; at epoch 8 it is 0.0339 against 0.0283.
  - The JS term rose from 2.5-4.5 % to 7.7-12.6 % of the loss (the design predicted 8-13.5 %). That extra gradient bought no extra agreement between views and slowed the CE fit.
- **The selection metric did not see the ranking loss.**
  - Mini WF1@0.5 is above B3a's at 5 of 8 epochs (3, 4, 5, 7, 8), and the top-5 mean is 92.54 against 92.31.
  - Mini noisy EER is worse than B3a's at 6 of 8 epochs (better only at 2 and 3) and ends at 11.26 against 10.28.
  - On half B, EER was the readout that transferred; mini WF1 did not.
- **The curve no longer falls at the end.** The best epoch is 7 (92.92) and epoch 8 is 92.81. B3a peaked at epoch 6 and fell 0.73 by epoch 8 (B3a report §3).
- **The noisy operating point is unstable across epochs.** The matched-mini `thr_or` drops to 0.027 (ep 6) and 0.007 (ep 8), and the heldout-mini `thr_or` to 0.212 / 0.136 (ep 4 / 5). B3a's mini `thr_or` sat at 0.863-0.992 throughout. A near-zero oracle threshold means the F1 curve is flat at the bona-heavy end, which fits a tail of saturated scores.
- **Dev loss rises from epoch 1 by 90 %** (0.0935 → 0.1773, peak 0.1908 at epoch 6), against B3a's 74 % (0.0990 → 0.1722). The minimum is at epoch 1 in both runs (`best_by_dev_loss`).
- **Final vs merge:**
  - On half B the final epoch beats the merge (92.44 vs 92.29, +0.15); on half A they tie (93.02 vs 93.01).
  - The merge includes epoch 6 (lowest mini clean F1 of the run, 96.42; clean R_bona 88.8; dev-loss peak) and epoch 4 (mini clean F1 96.61).
  - These are the likeliest drags, but no merged epoch was evaluated alone on half B.

## 4. What went OK (evidence)

1. **A clean, single-flag experiment that answers the dose question.**
   - Only `consistency_lambda` differs in `config.yaml`.
   - The header shows `lambda_js=1`, with the same 75,792 steps, 37,125 twin pairs and two views per clip as B3a.
   - No traceback, codec failure or pruning, and no trainer code in `git.diff`.
   - One 7 h 45 min run shows that raising the JS weight above 0.3 hurts: merge −0.77 with CI [−1.31, −0.22].
2. **It closes the "more consistency" branch.** The design's falsifier 1 ("weight is not the active ingredient") did not fire: noisy EER moved +0.80, well outside ±0.3 of 9.16. The weight is active, and above 0.3 it is harmful. Per the design, "keep 0.3 and deprioritise js_emb" applies a fortiori:
   - a further dose-up is ruled out;
   - so is a second consistency term (`js_emb`), whose mechanism is the same output/embedding agreement pressure.
3. **No over-regularisation of spoof or clean** (falsifier 2 not fired).
   - R_spoof at the merge: matched +0.19, heldout +0.12; worst drop −0.37 (matched, ep8).
   - Clean F1 −0.24 at the merge and +0.14 at ep8; clean EER 3.32 → 3.15 and 3.20 → 3.10.
   - Noisy spoof misses on half B: 25 → 20 at the merge (computed).
   - Min slice R_spoof rose (97.02 → 98.01 merge, 96.69 → 97.35 final).
4. **Seen-distribution slices did not suffer.**
   - Matched is −0.34 / −0.08 with both CIs spanning 0.
   - On the merge, matched snr 5 gained +1.77 F1 and opus:telegram +1.85. On the final epoch, matched snr 10 gained +1.63 and footsteps +1.60 (2.4).
   - Echo ep8 +0.06.
5. **The late-epoch decline of B3a is gone.** The best epoch moved from 6 to 7 and epoch 8 is within 0.11 of it. The final epoch beats the merge on half B (+0.15).

## 5. What went wrong (evidence)

### 5.1 Heldout ranking got worse, on both halves and both readouts

- Heldout half_B F1 −1.65 [−2.81, −0.55] (merge) and −1.46 [−2.68, −0.28] (ep8). These are the only per-set CIs that exclude 0 in both readouts.
- Heldout EER: +1.13 / +0.92 on half B and +1.52 / +1.01 on half A (2.1).
- Heldout bona fixed / broken: 6 / 24 (merge) and 8 / 25 (ep8), two thirds of them Chinese (2.2).
- Unseen-condition slices moved most: heldout SNR-0 EER 15.08 → 19.07, g726 10.74 → 13.37, door_wood_creaks 7.44 → 14.10, English 17.04 → 20.27 (merge, 2.4).
- Echo also lost on the merge: −0.96 [−1.86, −0.07], EER 4.61 → 6.06.

### 5.2 Oracle-gap decomposition (global threshold vs set-specific)

Method: one shared logit threshold swept over the half_B `scores/*.logits` (step 0.05, −6 to +12), as in the B3a and B3b reports; the per-set oracle comes from `metrics.json` `wf1_oracle` (computed).

| readout | WF1@0.5 | best global (logit / p) | WF1 at best global | per-set oracle WF1 | gap | recovered by one global shift | set-specific remainder |
|---|---|---|---|---|---|---|---|
| B3a-top5 | 93.06 | −0.15 / 0.463 | 93.08 | 93.24 | 0.18 | 0.02 | 0.16 |
| B3a-ep8 | 92.94 | +4.95 / 0.993 | 93.10 | 93.45 | 0.51 | 0.16 | 0.35 |
| **B3c-top5** | 92.29 | **+4.75 / 0.991** | 92.39 | 92.63 | 0.34 | 0.10 | 0.24 |
| **B3c-ep8** | 92.44 | +0.60 / 0.646 | 92.47 | 92.73 | 0.29 | 0.03 | 0.26 |

- **The loss is ranking, not calibration.** Even at its per-set oracle (92.63), B3c-top5 stays below B3a-top5 at 0.5 (93.06), and a global shift recovers only 0.10.
- The merge's best global logit moved from −0.15 to +4.75, beyond ±1. The WF1 curve is flat there: the move is worth 0.10.

### 5.3 The saturated bona tail grew, and its most confident part grew most

Share of half_B bona with p_spoof ≥ 0.5 / ≥ 0.9 / ≥ 0.99 / ≥ 0.999 (computed):

| set (n bona) | B3a-top5 | **B3c-top5** | B3a-ep8 | **B3c-ep8** |
|---|---|---|---|---|
| clean (1,397) | 7.7 / 7.6 / 6.4 / 4.6 | 8.5 / 8.4 / 7.7 / 6.0 | 8.9 / 8.9 / 8.3 / 6.1 | 8.5 / 8.4 / 7.8 / 6.2 |
| matched (391) | 21.0 / 21.0 / 16.4 / 8.2 | 22.5 / 22.3 / 18.2 / **12.8** | 22.3 / 22.3 / 19.2 / 12.0 | 21.2 / 21.0 / 18.2 / 13.3 |
| heldout (391) | 25.1 / 25.1 / 18.9 / 10.2 | **29.7 / 29.7 / 24.3 / 16.6** | 24.6 / 24.3 / 20.5 / 13.0 | **28.9 / 28.4 / 24.0 / 17.1** |
| echo (391) | 14.8 / 14.8 / 12.3 / 5.4 | 16.9 / 16.6 / 12.5 / 8.2 | 16.6 / 16.4 / 15.3 / 9.5 | 15.3 / 15.3 / 12.3 / 8.2 |

Noisy bona (matched + heldout, 782):

| readout | false alarms (FA %) | share ≥ 0.99 | share ≥ 0.999 | saturated share of FA | FA en / zh |
|---|---|---|---|---|---|
| B3a-top5 | 180 (23.0 %) | 17.6 % | 9.2 % | 77 % | 91 / 89 |
| **B3c-top5** | **204 (26.1 %)** | **21.2 %** | **14.7 %** | 81 % | **107** / 97 |
| B3a-ep8 | 183 (23.4 %) | 19.8 % | 12.5 % | 85 % | 93 / 90 |
| **B3c-ep8** | **196 (25.1 %)** | **21.1 %** | **15.2 %** | 84 % | 101 / 95 |

- The ≥ 0.999 share of noisy bona rose by 5.5 points on the merge and 2.7 on the final epoch. Heldout carries most of it: 10.2 → 16.6 and 13.0 → 17.1.
- **Interpretation (not proven by these files):** two posteriors saturated on the same class have JS ≈ 0. A heavier JS weight therefore rewards agreeing *confidently*, whether the agreed answer is right or wrong. That fits three observations:
  - the ≥ 0.999 tail grew;
  - the unweighted JS did not fall (section 3);
  - the damage is on conditions absent from training. There, both views of a training clip never resemble the test input, so agreement learned on the training augmentations does not transfer.

### 5.4 Worst slices for bona recall, B3c-top5 (full sets, `merge_top5.log`)

| slice | R_bona | R_spoof | EER |
|---|---|---|---|
| heldout en | **56.20** | 99.38 | 20.27 |
| heldout washing_machine | 60.26 | 98.01 | 14.31 |
| matched en | 62.02 | 99.77 | 14.71 |
| heldout snr 0 | 62.59 | 98.80 | **19.07** |
| heldout vacuum_cleaner | 64.29 | 99.34 | 15.12 |
| matched opus:qq | 64.65 | 99.15 | 10.84 |
| heldout g726 | 66.67 | 99.42 | 13.37 |
| matched snr 0 | 66.67 | 98.82 | 15.35 |
| heldout wind | 67.11 | 99.02 | 15.76 |
| echo en | 68.60 | 99.61 | 12.45 |

English, SNR 0 and the heldout codecs remain the worst axes, as in B3a (B3a report §5.4) and B3b (B3b report §5.5).

**Where the reference's (B3a-top5) 180 half_B noisy bona false alarms sit** (computed from B3a `scores/*.txt` + meta), which drives the proposals:
- **By codec** (each heldout clip has exactly one of g722 / g726 / gsm):
  - heldout g726: 36 of 129 bona (28 %);
  - heldout gsm: 38 of 127 (30 %);
  - heldout g722: 24 of 135 (18 %);
  - matched opus:qq: 21 of 60 (35 %);
  - matched opus:wechat: 17 of 50 (34 %);
  - together g726, gsm, qq and wechat hold **112 of the 180**.
- **By SNR:** SNR 0 has 40 of 123 (32.5 %; matched 21 of 63, heldout 19 of 60), against 23.0 % for all noisy bona.
- **By language:** English 91, Chinese 89 (English is 258 of 782 noisy bona).

### 5.5 Overfitting signs

- Dev loss is +90 % from epoch 1 and ends at 0.1773, above B3a's 0.1722 with identical CE weights. Train acc is 99.66 %.
- The merge's `gain_up` bona flips rose from 2.67 % to 6.42 %.
- The heldout-vs-matched split in 2.4 points the same way: the heavier invariance term fits the training augmentation distribution, matched improves or holds, and unseen DSP degrades.

## 6. Verdict on the pre-registered rule

From `exploop/B3/iter03/design.result.json`, the **falsifier**:

> "Wave rule against B3a-top5 = 93.06 on half_B:
> - Promote / positive: delta merge > +0.5, the paired-bootstrap CI excluding 0, and a clean F1 drop <= 0.5.
> - Null: delta within +-0.5.
> - Negative: delta below -0.5.
>
> Mechanism-level refutations:
> 1. Weight is not the active ingredient: delta within +-0.5 AND merge noisy EER within +-0.3 of 9.16. Then B3a's gain came from the paired views and the doubled steps, not the divergence weight; keep 0.3 and deprioritise js_emb.
> 2. Over-regularised: clean F1 drop > 0.5, or R_spoof down > 0.5 on matched or heldout.
> 3. JS cannot reach the core: noisy bona share at p >= 0.99 not below 17.6 % while epoch-8 JS falls below 0.0038. The core is agree-but-wrong, so pursue a different mechanism (SNR range / codec / SpecAugment), not js_emb.
> 4. Operating point shifted rather than ranking improved: best global logit beyond +-1 together with noisy EER not below 9.16."

`exploop/B3/iter03/verdict.json` (computed by the supervisor; quoted, not recomputed):

| readout | run | base | delta | CI | clean_drop | new_best | decision | verdict |
|---|---|---|---|---|---|---|---|---|
| merge (objective) | 92.28951037321083 | 93.05618353351335 | −0.766673160302517 | [−1.3067677768417916, −0.21743762825827398] | 0.23785041605697188 | false | **drop** | **negative** |
| final | 92.44187146959624 | 92.94058874557216 | −0.4987172759759204 | [−1.045254749087946, 0.053382479857357466] | −0.1398982072285122 | false | **drop** | **null (inside +-0.5)** |

**On the objective (merge) the negative branch fired. On the final epoch the verdict is null (inside ±0.5).**

Mechanism-level refutations (my reading of the evidence against the design's own conditions; the verdict above is the supervisor's):

| refutation | evidence | fired? |
|---|---|---|
| 1. weight not active | Δ −0.77 (merge) and −0.50 (ep8); noisy EER 9.96 (+0.80) and 9.71 (+0.70 vs 9.01): neither within ±0.3 | **no.** The run lands on the far side of falsifier 1: the weight is active and λ 1.0 is *worse* than 0.3. B3a's step/view confound below 0.3 is still open |
| 2. over-regularised | clean drop 0.24 / −0.14; R_spoof matched +0.19 / −0.37, heldout +0.12 / +0.19 | **no** |
| 3. JS cannot reach the core | the ≥ 0.99 share is 21.2 % (not below 17.6 %), but epoch-8 JS is 0.0044, not below 0.0038 | **not as stated**: JS never fell, so the condition's second half fails. The core grew while the training divergence stayed flat. That is consistent with an agree-but-wrong core and points the same way as the falsifier's advice: a different mechanism (SNR range / codec / SpecAugment), not `js_emb` |
| 4. operating point shifted | best global logit +4.75 (beyond ±1) with noisy EER 9.96 (not below 9.16) on the merge; +0.60 on ep8 | **yes on the merge** by its letter, but the shift is worth only 0.10 WF1 (5.2). Ranking got worse as well |

The predicted signature, item by item (merge vs B3a-top5; ep8 vs B3a-ep8 in brackets):

| predicted | observed | appeared? |
|---|---|---|
| mean noisy EER below 9.16 by ≥ 0.3 (ep8 below 9.01) | 9.96 (9.71) | **no**, up |
| full-set SNR-0 EER below 14.67 matched / 15.08 heldout | 15.35 / 19.07 | **no** |
| noisy bona share at p ≥ 0.99 below 17.6 % (19.8 %) | 21.2 % (21.1 %) | **no**, up |
| noisy bona FA below 180 of 782 | 204 (196 vs 183) | **no** |
| English noisy FA not above 91 | 107 (101 vs 93) | **no** |
| heldout R_bona above 74.94, matched above 79.03 | 70.33 / 77.49 (71.10 / 78.77) | **no** |
| R_spoof within 0.5 of 99.38 / 99.06 | 99.56 / 99.19 (99.31 / 99.19) | yes |
| clean F1 within 0.5 of 97.48 | 97.25 (97.20) | yes |
| best global logit within ±1 of 0 | +4.75 (+0.60) | no on the merge, yes on ep8 |
| unweighted JS below B3a's at most epochs, below 0.0038 at epoch 8 | below at 3 of 8; 0.0044 at epoch 8 | **no** |
| CE per epoch roughly at B3a's level | higher at 7 of 8 epochs (0.0339 vs 0.0283 at epoch 8) | partly |
| epoch-8 dev loss not above 0.1722 | 0.1773 | **no** |
| merge half_B WF1 above 93.06, gain on heldout and matched | 92.29; heldout −1.65, matched −0.34 | **no** |

Only the "unchanged" items (spoof recall, clean F1) appeared. Every ranking, tail and telemetry item went the opposite way.

## 7. Decision and next-experiment proposals

**Decision.** Per `verdict.json`, **B3c is not a new best** (`new_best false` on both readouts, `decision drop`; merge `negative`, final `null (inside +-0.5)`).
- The reference stays **B3a** (merge 93.06, final 92.94; registry `B3a-cjs03-screen` / `B3a-cjs03-ep8`).
- B3c's registry rows are `B3c-cjs10-screen` (CI −1.31 / −0.22) and `B3c-cjs10-ep8` (−1.05 / 0.05).

Every proposal below is one factor (or one coupled group) against **B3a's flags**: B2f + `--consistency js --consistency_lambda 0.3 --consistency_pairs both --batch_size 8`, sampler `shuffle`, `--ce_weights 0.1 0.9`. Each keeps the ~8 h runtime. Every falsifier uses the wave rule against B3a-top5 = 93.06 on half_B:
- promote: Δ merge > +0.5 with the CI excluding 0 and a clean drop of at most 0.5;
- null: Δ within ±0.5;
- negative: Δ below −0.5.

**The largest remaining error** is unchanged and is now better localised: confident noisy bona false alarms.
- B3a merge: 180 of 782 half_B noisy bona, 77 % of them at p ≥ 0.99.
- 112 of the 180 sit in four codecs: heldout g726 / gsm, matched opus:qq / wechat.
- 40 of the 180 sit at SNR 0, at a 32.5 % false-alarm rate.

The three B3 iterations narrow down what kind of error this is:
- B3b: it is not exposure;
- B3b and B3c: it is not the operating point;
- B3c: more invariance on the same augmentation distribution does not help. It sharpened the tail and hurt unseen conditions.

What remains is the **coverage of the training degradation distribution** and regularisation. Do not pursue a higher `--consistency_lambda` or `--consistency js_emb`.

**Ranked proposals (configuration first, seed repeat last):**

1. **Codec zoo: `--aug_codec_profiles random`** (`wave.json` lever_queue `codec_random`, the next queued lever).
   - Targets:
     - heldout, the set that carried three quarters of both B3a's gain and B3c's loss;
     - the codec slices with the highest false-alarm rates: B3a-top5 half_B heldout g726 36 / 129 and gsm 38 / 127, matched opus:qq 21 / 60 and opus:wechat 17 / 50.
   - Hypothesis:
     - B3c shows that heldout errors do not yield to more invariance on the existing 7 fixed Opus app profiles.
     - Per-clip random Opus bitrate / frame size / VBR / packet-loss (`train.py` help) widens the channel distribution itself.
     - Under the paired setup, JS at 0.3 then enforces invariance across codec parameters as well. That should reach unseen g726 / gsm by generalisation, and the low-bitrate qq / wechat profiles directly.
   - Predicted signature:
     - heldout half_B F1 above 90.20 and R_bona above 74.94;
     - full-set heldout g726 / gsm R_bona ≥ 3 points above 71.14 / 72.76, with EER below 10.74 / 13.91;
     - half_B false alarms in g726 + gsm below 74 and in qq + wechat below 38;
     - R_spoof within 0.5 of 99.38 / 99.06; clean within 0.5 of 97.48;
     - noisy EER below 9.16.
   - Falsifier:
     - Δ merge within ±0.5 and g726 / gsm R_bona within ±3 means codec diversity is not the limit.
     - R_spoof down > 0.5 on matched or heldout means random profiles blur a spoof cue.

2. **SNR range 0-20 dB: `--aug_snr_db 0 20`** (lever_queue `snr_0_20`; uniform distribution unchanged).
   - Targets: SNR 0, the highest false-alarm-rate SNR. On the B3a merge it is 40 of 123 half_B noisy bona (32.5 % vs 23.0 % overall), with full-set R_bona 65.81 / 65.31 and EER 14.67 / 15.08. Training never draws below 5 dB (`SNR: 5..20 dB` header). B3c made heldout SNR-0 EER worse (15.08 → 19.07).
   - Hypothesis: SNR-0 false alarms are out-of-range inputs. Covering 0-5 dB in training gives both views of a pair a low-SNR condition, so consistency can act on it.
   - Predicted signature:
     - full-set SNR-0 EER below 14.67 / 15.08;
     - SNR-0 R_bona up ≥ 3 on both sets at R_spoof within 0.5 of 98.65 / 98.50;
     - half_B SNR-0 noisy false alarms below 40;
     - 15-20 dB slices within 2 R_bona of B3a: matched 82.57 / 85.83, heldout 74.17 (B3b report §7) / 79.86;
     - clean flat.
   - Falsifier:
     - Δ merge within ±0.5 and SNR-0 R_bona within ±3 means the SNR range is not the limit.
     - A drop of > 2 R_bona at 15-20 dB means the uniform 0-20 draw diluted the high-SNR mass.

3. **SpecAugment: `--specaug`** (lever_queue `specaug`; default masks `--specaug_time 2 40 --specaug_freq 2 16`).
   - Targets: overconfidence and overfitting.
     - Dev loss rises from epoch 1 in B3a (+74 %), B3b (+83 %, B3b report) and B3c (+90 %).
     - Train acc reaches 99.66-99.71 %.
     - The ≥ 0.999 noisy bona share is 9.2 % on the B3a merge, and B3c showed that output-agreement pressure grows it.
   - Hypothesis: input masking regularises without rewarding confident agreement. It blocks clip-specific memorisation and should thin the saturated tail, not sharpen it.
   - Predicted signature:
     - epoch-8 dev loss below 0.1722;
     - mini WF1 not falling after its peak;
     - half_B noisy bona ≥ 0.999 share below 9.2 % and ≥ 0.99 share below 17.6 % (merge);
     - noisy EER below 9.16;
     - clean within 0.5 of 97.48;
     - R_spoof within 0.5.
   - Falsifier:
     - Δ merge within ±0.5 and epoch-8 dev loss ≥ 0.17 means masking does not regularise this backbone.
     - R_spoof down > 0.5 on matched or heldout means the masks remove a spoof cue.

4. **Consistency weight to zero: `--consistency_lambda 0.0`** (a one-flag dose step down inside the consistency group; views, pairs, `--batch_size 8` and steps unchanged).
   - Targets: the recipe question that B3c left open, namely B3a's confound (B3a report §5.5). B3c shows the dose response above 0.3 is negative and that the extra weight bought no reduction in JS. If the JS term at 0.3 is also inert, B3a's +1.08 came from the paired views and doubled steps. The full run would then not need the divergence term at all.
   - Hypothesis (null form): with λ = 0 the run keeps B3a's two views per clip and 75,792 steps, and only the JS gradient is removed.
   - Predicted signature if JS is inert: merge within ±0.5 of 93.06 and noisy EER within ±0.3 of 9.16, with the logged (unweighted) JS higher than B3a's at most epochs.
   - Falsifier:
     - Δ < −0.5 with noisy EER ≥ 9.46 means the JS term at 0.3 is an active ingredient: keep 0.3, since the optimum lies between 0 and 1.0.
     - Δ > +0.5 means JS hurts at any weight here: drop it from the recipe.
   - Ranked below the augmentation levers because its expected WF1 gain is smaller. It is mainly a recipe-clarifying control.

5. **Seed repeat of B3a: `--seed 1235`** (last).
   - Purpose: B3a's 93.06 is the only above-band result of the wave, on one seed. B3b and B3c both land about 0.8 below it on half_B (92.23 / 92.29), but B3c ties B3a on half_A (93.01 vs 93.04 merge) and beats it on the final epoch (93.02 vs 92.61). The seed band of the consistency recipe is unmeasured. `wave.json` gives ~0.2-0.5 WF1 on the final readout for the B2 near-replicates.
   - Predicted signature: merge within ±0.5 of 93.06, with the heldout gain over B2f (> +1.5) reproduced.
   - Falsifier: a repeat merge below 92.48 (B2f-top5 91.98 + 0.5) means B3a's gain over B2f is inside seed noise, and the B3b / B3c deltas must be re-read against that band.

**Recommended full run if the wave stopped now.** Unchanged from the B3a and B3b reports:
- Flags: B3a's exact `config.yaml` command (`scripts/run_B3a.sh`, `--consistency_lambda 0.3`) at 2-3 seeds (1234, 1235, 1236). Not λ 1.0.
- Readout to submit: the top-5 merge. It is the wave objective and B3a's best readout: 93.06 vs 92.94 on half B, 93.04 vs 92.61 on half A.
- Pre-launch checks: `bash -n scripts/run_B3a.sh`, `which ffmpeg`, `pyloudnorm` importable, an empty `pgrep -af "b3wave.train|main_train"`, an idle GPU.
- Post-launch checks:
  - `Consistency: js lambda_js=0.3 lambda_emb=0.3 pairs=both … 'found': 37125 … 16 clips per step (8 pairs)`;
  - `Sampler: shuffle`, `CE weights: spoof=0.1 bonafide=0.9`;
  - `Warmup-cosine: 3790 warmup / 75792 steps`;
  - epoch-1 `CE=0.3945 JS=0.0344`-level values with mini WF1 ≈ 90.2 (B3a `train.log`);
  - codec counts summing to 2 × 75,785.
- Every evaluation needs `--loudness_norm --loudness_norm_lufs -23`.
