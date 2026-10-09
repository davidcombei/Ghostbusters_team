# Experiment exploration 05: B2e (`B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234`) - iteration 5 of 5 of the B2 wave

> Written 2026-10-08 23:23 by the iteration-5 analysis agent. Every number is read from the run directories named below (`config.yaml`, `logs/train.log`, `metrics.jsonl`, `local_eval/*/metrics.json`, `probe_report.json`, `compare_vs_*.txt`, `local_eval/*.log`, `local_eval/*/scores/*.txt`, the stdout log `SLS_setup/B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234_20261008185918.log`, `docs/experiment_log.csv`); nothing is quoted from memory. Previous reports: `20261008_185750_B2d_...md` (iteration 4), `20261008_131754_B2c_...md` (3), `20261008_084435_B2b_...md` (2), `20261008_040609_B2a_...md` (1), `20261007_222414_00_B0_vs_B1_postmortem.md`. Protocol: plan section 3 and `docs/next_steps_plan.md` section 1. The wave-level summary and the full-run recommendation are in `20261008_232331_B2_wave_summary_and_full_run_recommendation.md`.

Run directories (under `SLS_setup/exp/`):
- **B2e** `B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234_epoch8_bs16_20261008185925` (this report)
- **B2c** `B2c_qwen17_rtc_musan_lvl_cos8_lr5e6_s1234_epoch8_bs16_20261008085023` (reference: half_B merge 91.73, final-epoch 92.26)
- **B2d** `B2d_qwen17_rtc_musan_lvl_cos10_lr5e6_s1234_epoch10_bs16_20261008132750` (merge 91.57, final 91.84); **B2b** `..._lr1e5_..._20261008040736` (89.59 / 90.34); **B2a** `..._cos8_s1234_epoch8_bs16_20261007231952` (84.95 / 85.18); **B0** `B0_qwen17_rtcaug_s1234_epoch10_bs16_20261003201121` (merge 83.61)

Readouts: `-top5` = weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5/`); `-ep8` = the final-epoch EMA checkpoint alone (`local_eval/epoch_8/`); `avg-B2c8-B2e8` = zero-training weight average of B2c's and B2e's `epoch_8` checkpoints (`local_eval/avg_B2c8_B2e8/`, `merge_list_in.txt`). half_B = the 1,994-clip half of each sim set never used for selection (391 bona + 1,603 spoof per set); clean = the full 7,465-clip `dev_online_clean` (1,397 bona). Scores in `scores/*.txt` are p_spoof; threshold 0.5 reproduces every half_B number of all five `metrics.json` files exactly (section 5.2). Registry rows: `B2e-lr2.5e6-screen` (92.34, revisit) and `B2e-lr2.5e6-ep8` (92.53, revisit).

## 1. Setup

B2e = B2c with **one factor changed: the whole LR schedule halved, `--lr 5e-6 -> 2.5e-6` and `--lr_min 1e-6 -> 5e-7`** (`lr_min` is applied as the ratio `lr_min/lr`, so the cosine shape and the 20 % floor are identical; head LR 2.5e-5 via `--head_lr_mult 10`). `config.yaml` diff vs B2c: only `created`, `--lr`, `--lr_min`, `--track`. Git `23bd5a3`, not dirty, created 2026-10-08 18:59:35. Stdout header: `Warmup-cosine: 1895 warmup / 37896 steps, floor 5.0e-07` (B2c: same steps, floor 1.0e-06), `LR: 2.5e-06 schedule=warmup_cosine`, `Optimizer: adamw weight_decay=0.01 head_lr=2.5e-05 (x10)`, `Grad clip: 1.0`, `EMA: decay=0.9995`, `CE weights: spoof=0.1 bonafide=0.9`, `Train trials: 75785`, `Dev trials: 11739`. Same step count as B2c, so the two runs are aligned by epoch index and the LR is exactly half of B2c's at every step.

| flag | B2c | **B2e** | everything else |
|---|---|---|---|
| `--lr` / `--lr_min` | 5e-6 / 1e-6 | **2.5e-6 / 5e-7** | identical: `--num_epochs 8 --warmup_frac 0.05 --lr_scheduler warmup_cosine --optim adamw --weight_decay 0.01 --head_lr_mult 10 --grad_clip 1.0 --ema_decay 0.9995 --earlystop_metric wf1 --earlystop_epoch 99 --keep_topk_by_wf1 7 --seed 1234`, RTC+MUSAN+level aug (`p_apply=0.8`, codec p=1.0, level p=0.8), `ce_weights 0.1 0.9`, `label_smoothing 0.0` |
| LR at the end of epoch 1 ... 8 | 4.94, 4.58, 3.95, 3.17, 2.35, 1.65, 1.17, 1.00 e-6 | 2.47, 2.29, 1.98, 1.58, 1.18, 0.82, 0.58, 0.50 e-6 | B2e's epoch 4 runs at the LR of B2c's epoch 7-8 |

Run time (`logs/train.log`, `config.yaml`): started 18:59:35, epoch 8 done 23:03:58, last local eval 23:04:16 -> **4 h 05 min** (B2c 4 h 06 min, B2d 5 h 07 min). GPU not shared. Epoch durations 30.3-30.8 min; local eval 18-24 s. Post-train: `top5_by_wf1/merge_list.txt` 23:04, `merge_top5/metrics.json` 23:08, `epoch_8/metrics.json` 23:13, `avg_B2c8_B2e8/metrics.json` 23:15. Registry GPU-h 4.08.

Checkpoints: **nothing pruned** (no `Pruned` line in `train.log`; all 8 epoch files present, 10.3 GB). Epoch 1 (mini WF1 88.07, the lowest) is `best_by_dev_loss.pth` (dev loss 0.0822) and therefore protected; `best_by_noisy_eer.pth` -> `epoch_6` (mini noisy EER 10.63); `best_by_wf1.pth` -> `epoch_8` (same md5). **Merged top-5 by half-A mini WF1 = epochs 8, 4, 6, 5, 7** (92.64, 92.19, 91.92, 91.90, 91.75); epoch 2 missed the cut by 0.67 (91.08), epoch 3 by 0.74 (91.01). First merge of the wave without an early-plateau epoch (B2c's merge was 7, 8, 6, 5, 3; B2d's 10, 8, 9, 3, 6).

## 2. Results vs references (half_B)

Source: `half_B` block of each `metrics.json`; deltas as printed in `compare_vs_B2c-top5.txt`, `compare_vs_B0-top5.txt`, `compare_vs_B2c-ep8.txt` (the compare script prints deltas from unrounded values, e.g. 92.53 - 92.26 prints as +0.26).

### 2.1 All iterations, both readouts

| half_B | B0-top5 | B1-top5 | B2a-top5 | B2b-top5 | B2c-top5 | B2d-top5 | **B2e-top5** | B2a-ep8 | B2b-ep8 | B2c-ep8 | B2d-ep10 | **B2e-ep8** | B2e-top5 - B2c-top5 | B2e-ep8 - B2c-ep8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **WF1@0.5** | 83.61 | 81.11 | 84.95 | 89.59 | 91.73 | 91.57 | **92.34** | 85.18 | 90.34 | 92.26 | 91.84 | **92.53** | **+0.61** | **+0.26** |
| WF1@oracle | 84.71 | 84.26 | 87.04 | 91.27 | 92.60 | 92.36 | 92.88 | 87.09 | 92.00 | 92.74 | 92.47 | 92.97 | +0.28 | +0.23 |
| clean F1@0.5 | 92.37 | 87.89 | 92.46 | 95.10 | 96.82 | 96.73 | 97.67 | 92.98 | 95.59 | 97.42 | 97.18 | 98.09 | +0.85 | +0.67 |
| noisy F1@0.5 (mean matched, heldout) | 79.86 | 78.21 | 81.73 | 87.23 | 89.55 | 89.36 | 90.05 | 81.84 | 88.09 | 90.05 | 89.55 | 90.14 | +0.51 | +0.09 |
| mean noisy EER | 18.01 | 19.07 | 15.77 | 12.00 | 10.11 | 10.44 | 9.68 | 15.63 | 11.26 | 10.06 | 9.96 | 9.27 | -0.43 | -0.79 |
| calibration gap noisy | 1.41 | 2.96 | 2.17 | 1.58 | 0.81 | 0.64 | 0.54 | 2.03 | 1.48 | 0.50 | 0.55 | 0.53 | -0.27 | +0.03 |
| separability gap | 11.45 | 10.33 | 10.47 | 8.20 | 7.45 | 7.88 | 7.62 | 10.73 | 8.10 | 7.28 | 7.90 | 7.65 | +0.17 | +0.37 |
| oracle gap (WF1@or - WF1@.5) | 1.09 | 3.15 | 2.09 | 1.68 | 0.87 | 0.79 | 0.54 | 1.91 | 1.66 | 0.48 | 0.63 | 0.44 | -0.33 | -0.04 |

B2e is above B2c on every aggregate in both readouts, every delta inside the near-replicate band (+-0.5 on WF1, section 3). Vs B0: merge +8.72, oracle +8.17, EER -8.33, clean +5.30, noisy +10.19 (B2c was +8.12 / +7.89 / -7.90 / +4.45 / +9.69). The final epoch is the best single checkpoint of the wave on half_B (92.53) and the best mini epoch of the wave (92.64).

### 2.2 Per set (R = recall at 0.5; thr = p_spoof threshold)

| set | metric | B0-top5 | B2b-top5 | B2c-top5 | B2d-top5 | **B2e-top5** | B2c-ep8 | B2d-ep10 | **B2e-ep8** | avg-B2c8-B2e8 |
|---|---|---|---|---|---|---|---|---|---|---|
| dev_online_clean | F1@.5 / F1@or / thr_or | 92.37 / 92.72 / 0.435 | 95.10 / 97.01 / 0.999 | 96.82 / 97.81 / 0.997 | 96.73 / 97.88 / 0.998 | **97.67** / 98.21 / 0.995 | 97.42 / 97.83 / 0.995 | 97.18 / 98.00 / 1.000 | **98.09** / 98.32 / 0.991 | 97.27 / 97.90 / 0.990 |
| | EER / ECE | 6.80 / 1.80 | 3.14 / 2.79 | 2.63 / 1.83 | 3.17 / 1.88 | **3.51** / 1.33 | 2.51 / 1.51 | 2.86 / 1.65 | **3.35** / 1.13 | 2.90 / 1.56 |
| | R_spoof / R_bona | 97.20 / 87.40 | 99.92 / 85.33 | 99.95 / 90.26 | 99.93 / 90.05 | 99.90 / **93.06** | 99.88 / 92.34 | 99.92 / 91.48 | 99.97 / **94.06** | 99.93 / 91.70 |
| sim_matched_v1 | F1@.5 / F1@or / thr_or | 82.18 / 83.90 / 0.377 | 88.65 / 90.40 / 0.975 | 90.11 / 91.19 / 0.960 | 90.81 / 91.22 / 0.979 | **91.60** / 92.09 / **0.874** | 90.64 / 91.35 / 0.987 | 90.89 / 91.15 / 0.986 | **91.90** / 92.18 / **0.841** | 90.66 / 91.76 / 0.958 |
| | EER / ECE | 14.08 / 7.25 | 10.51 / 5.90 | 8.94 / 5.42 | 9.16 / 4.94 | 8.27 / 4.68 | 8.97 / 5.24 | 8.40 / 4.98 | **7.67** / 4.64 | 8.43 / 4.94 |
| | R_spoof / R_bona | 89.96 / 79.54 | 99.44 / 69.82 | 99.31 / 73.91 | 99.56 / 74.94 | 98.88 / **79.28** | 99.19 / 75.70 | 99.31 / 75.96 | **98.81** / **80.31** | 99.44 / 74.94 |
| sim_heldout_v1 | F1@.5 / F1@or / thr_or | 77.54 / 78.64 / 0.458 | 85.81 / 87.22 / 0.942 | 88.98 / 89.53 / 0.904 | 87.91 / 88.77 / 0.957 | **88.50** / 89.09 / 0.967 | 89.46 / 89.76 / 0.963 | 88.20 / 89.06 / 0.898 | **88.39** / 89.17 / 0.968 | 88.45 / 89.06 / 0.786 |
| | EER / ECE | 21.94 / 9.14 | 13.48 / 7.00 | 11.28 / 5.86 | 11.72 / 6.35 | 11.09 / 6.12 | 11.15 / 5.69 | 11.52 / 6.19 | **10.86** / 6.40 | 12.10 / 5.70 |
| | R_spoof / R_bona | 87.96 / 71.61 | 99.38 / 63.17 | 99.13 / 71.61 | 99.25 / 68.54 | 99.06 / **70.59** | 99.19 / 72.63 | 98.81 / 70.59 | **98.63** / **71.61** | 99.19 / 70.08 |
| sim_echo_v1 | F1@.5 / F1@or / thr_or | 87.65 / 88.13 / 0.309 | 91.12 / 93.71 / 0.988 | 92.83 / 93.75 / 0.997 | 93.32 / 94.65 / 0.985 | **94.88** / 95.16 / 0.936 | 93.08 / 94.21 / 0.996 | 94.52 / 94.91 / 0.942 | **94.79** / 95.13 / 0.966 | 93.40 / 94.50 / 0.948 |
| | EER / ECE | 10.67 / 3.96 | 6.25 / 4.86 | 5.71 / 4.06 | 5.68 / 3.82 | **4.92** / 2.98 | 5.43 / 3.97 | 5.40 / 3.19 | 5.62 / 3.03 | 4.45 / 3.55 |
| | R_spoof / R_bona | 93.51 / 85.17 | 99.81 / 74.94 | 99.50 / 80.56 | 99.81 / 80.82 | 99.75 / **85.42** | 99.63 / 80.82 | 99.75 / 84.40 | 99.75 / **85.17** | 99.88 / 80.82 |

Direction per set, same in both readouts: **clean up** (+0.85 merge / +0.67 ep; R_bona +2.79 / +1.72 = 39 / 24 clips of 1,397), **matched up** (+1.49 / +1.26; R_bona +5.37 / +4.60 = 21 / 18 of 391), **echo up** (+2.05 / +1.71; R_bona +4.86 / +4.35 = 19 / 17), **heldout down** (-0.48 / -1.08; R_bona -1.02 / -1.02 = 4 clips, R_spoof -0.06 / -0.56 = 1 / 9 of 1,603). WF1 = 0.3 clean + 0.35 matched + 0.35 heldout, so the ep delta is +0.20 + 0.44 - 0.38 = +0.26. EER moved with F1 on matched (-0.66 / -1.29) and heldout (-0.19 / -0.29, i.e. the heldout F1 loss is not a ranking loss), against it on clean (+0.87 / +0.84: `thr_eer` is 0.99998 on clean, so the EER lives in the 5-6 % of clean bona scored above the top spoofs, which neither run moves) and on echo-ep (+0.19). Paired per clip (ep8 vs B2c-ep8, section 5.2): clean bona 39 fixed / 15 broken, matched 31 / 13, echo 24 / 7, heldout 9 / 13; heldout spoof 7 fixed / 16 broken, matched spoof 7 / 13. **For the first time in the wave the spoof side moved**: matched / heldout R_spoof fell to 98.81 / 98.63 (19 / 22 misses of 1,603; B2c 13 / 13) while the matched oracle threshold fell from 0.987 to 0.841 - the spoof prior shrank (section 5.2).

### 2.3 Probes (`probe_report.json`; probe sets of `metrics.json`, 187 bona + 813 spoof; one bona probe clip = 0.53 points)

| probe | B0-top5 | B2c-top5 | B2d-top5 | **B2e-top5** | B2c-ep8 | B2d-ep10 | **B2e-ep8** |
|---|---|---|---|---|---|---|---|
| level_flip_bona / max over gain probes | 7.29 / 12.83 (gain_up) | 1.80 / 3.74 (gain_m20, gain_p10_limit) | 2.27 / 6.42 | **1.40** / 2.67 (gain_up) | 2.54 / 5.88 (gain_up) | 2.41 / 5.88 | **1.87** / 4.81 (gain_up) |
| level_flip_spoof | 2.09 | 0.02 | 0.02 | 0.03 | 0.08 | 0.02 | 0.02 |
| probe_orig: bona pred spoof / spoof pred spoof | 14.97 / 97.17 | 13.90 / 100.00 | 14.97 / 100.00 | **10.70** / 100.00 | 11.23 / 100.00 | 13.90 / 100.00 | **8.02** / 100.00 |
| noise_only: bona-source / spoof-source pred spoof | 54.55 / 58.67 | 100.00 / 99.88 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 | 100.00 / 100.00 |
| noise_only mean dP(spoof) on bona-source | +0.34 | +0.86 | +0.84 | +0.89 | +0.88 | +0.85 | +0.91 |
| sil_only: bona-source / spoof-source pred spoof | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| trimmed / padded bona flip rate | 4.28 / 9.63 | 1.60 / 4.28 | 2.14 / 4.81 | 1.07 / 3.21 | 3.74 / 2.67 | 1.60 / 4.81 | 1.07 / 2.67 |

Level flips -0.40 (merge) / -0.67 (ep), the lowest of the wave (one clip each); `probe_orig` bona false alarms -6 clips (merge) / -6 clips (ep), also the lowest. Non-speech probes unchanged at 100 % with the mean dP marginally up (+0.03): the "spoof is the default class" behaviour (B2c report 5.3) is untouched by the LR scale.

### 2.4 Slices that moved, B2e vs B2c (full sim sets, `metrics.json -> breakdowns`, 63 slice rows)

Merge: F1 up in **46 of 63** rows, EER down in 38, R_bona up in 52. Final epoch: F1 up in 38, EER down in 36, R_bona up in 44. Minimum R_spoof: matched / snr 0 **96.13** (ep; B2c 98.15), heldout / washing_machine 96.36 (merge).

| set / slice | n | F1 B2c -> B2e merge | F1 B2c-ep8 -> B2e-ep8 | EER ep | R_bona ep (merge) | R_spoof ep |
|---|---|---|---|---|---|---|
| heldout / wind | 381 | 86.30 -> 84.23 -2.07 | 86.63 -> 84.06 **-2.57** | 16.26 -> 14.29 | 67.1 -> 64.5 (64.5 -> 63.2) | 98.69 -> 97.38 |
| heldout / snr 5 | 778 | 89.61 -> 88.99 -0.62 | 90.23 -> 87.71 -2.52 | 11.96 -> 10.27 | 75.6 -> 74.4 | 98.87 -> 97.11 |
| heldout / snr 0 | 812 | 84.08 -> 84.60 +0.52 | 86.66 -> 84.25 -2.41 | 15.46 -> 17.30 | 66.7 -> 65.3 (60.5 -> 63.9) | 98.80 -> 97.29 |
| heldout / clock_tick | 383 | 92.33 -> 91.63 -0.69 | 93.90 -> 91.63 -2.27 | 7.25 -> 9.29 | 82.5 -> 76.2 | 99.69 -> 99.69 |
| heldout / door_wood_creaks | 361 | 91.21 -> 90.88 -0.33 | 90.31 -> 88.16 -2.15 | 8.90 -> 10.69 | 76.6 -> 76.6 | 98.65 -> 96.97 |
| heldout / door_wood_knock | 353 | 90.01 -> 88.95 -1.06 | 90.52 -> 88.45 -2.08 | 9.31 -> 11.91 | 75.9 -> 70.4 | 99.00 -> 99.00 |
| matched / snr 0 | 711 | 83.43 -> 83.58 +0.15 | 84.07 -> 82.19 -1.88 | 15.35 -> 15.27 | 62.4 -> 64.1 (59.8 -> 65.0) | 98.15 -> **96.13** |
| heldout / washing_machine | 380 | 83.18 -> 84.93 +1.75 | 84.59 -> 83.36 -1.23 | 15.80 -> 14.98 | 66.7 -> 65.4 (64.1 -> 69.2) | 97.02 -> 96.36 |
| heldout / lang zh | 2462 | 91.55 -> 91.10 -0.45 | 92.29 -> 91.18 -1.11 | 8.38 -> 7.95 | 80.5 -> 81.7 | 99.09 -> 97.77 |
| heldout / lang en | 1538 | 83.52 -> 83.98 +0.46 | 84.01 -> 82.97 -1.03 | 17.12 -> 18.99 | 60.9 -> 56.6 (58.5 -> 58.1) | 98.59 -> 99.22 |
| heldout / g726 | 1274 | 87.65 -> 87.39 -0.26 | 88.24 -> 87.36 -0.87 | 11.93 -> 12.71 | 69.1 -> 68.7 | 99.32 -> 98.74 |
| heldout / gsm | 1346 | 86.32 -> 86.00 -0.32 | 86.96 -> 86.31 -0.65 | 14.64 -> 13.77 | 69.9 -> 69.1 | 98.00 -> 97.73 |
| matched / lang en | 1538 | 85.34 -> 86.21 +0.88 | 86.91 -> 86.48 -0.42 | 12.65 -> 13.27 | 65.5 -> 64.0 (61.2 -> 63.6) | 99.30 -> 99.45 |
| echo / lang en | 1538 | 87.32 -> 88.59 +1.27 | 88.46 -> 88.77 +0.30 | 10.39 -> 12.41 | 69.4 -> 70.2 | 99.30 -> 99.30 |
| matched / lang zh | 2462 | 93.15 -> 94.20 +1.05 | 93.30 -> 94.51 +1.21 | 7.11 -> 6.17 | 81.9 -> 89.0 (81.5 -> 87.6) | 99.49 -> 98.43 |
| echo / lang zh | 2462 | 96.17 -> 97.19 +1.02 | 95.95 -> 97.53 +1.58 | 2.95 -> 2.44 | 88.0 -> 93.1 | 99.90 -> 99.80 |
| matched / opus:qq | 572 | 83.46 -> 86.97 +3.51 | 86.33 -> 88.02 +1.70 | 11.66 -> 8.67 | 64.6 -> 71.7 (58.6 -> 69.7) | 99.15 -> 98.31 |
| matched / keyboard | 647 | 89.95 -> 92.60 +2.65 | 91.33 -> 93.27 +1.94 | 8.87 -> 5.58 | 74.8 -> 82.5 | 99.82 -> 99.26 |
| matched / office | 672 | 88.60 -> 90.75 +2.14 | 89.06 -> 91.04 +1.98 | 9.50 -> 10.17 | 73.1 -> 77.7 | 98.71 -> 98.89 |
| matched / snr 20 | 642 | 92.87 -> 94.57 +1.69 | 93.16 -> 95.36 +2.20 | 6.45 -> 6.55 | 81.1 -> 86.6 | 99.61 -> 99.81 |
| matched / snr 10 | 653 | 89.62 -> 91.25 +1.63 | 89.46 -> 91.56 +2.10 | 6.25 -> 6.75 | 71.8 -> 76.6 | 99.43 -> 99.62 |
| echo / opus:dingtalk | 554 | 88.77 -> 92.27 +3.49 | 88.77 -> 92.64 **+3.86** | 7.36 -> 6.84 | 70.8 -> 81.2 | 99.13 -> 99.13 |
| echo / self_echo | 1278 | 91.20 -> 92.63 +1.43 | 91.17 -> 92.96 +1.79 | 8.69 -> 7.69 | 76.8 -> 82.8 | 99.23 -> 98.95 |
| echo / farend | 1339 | 92.60 -> 94.67 +2.07 | 92.96 -> 94.53 +1.57 | 5.16 -> 5.55 | 79.7 -> 84.1 | 99.82 -> 99.82 |

Pattern, the same as B2d's but with a positive aggregate: the gainers are the matched families (keyboard, office), the training-time codecs (qq, dingtalk, zoom, lark), Chinese and SNR >= 10; the losers are heldout transients (wind, clock_tick, door_*), SNR 0-5 on both sim sets, and English on heldout. Both halvings of the LR after B2b moved the in-distribution slices; the unseen-DSP slices and English did not follow (section 5.3).

### 2.5 Lever-response table (all iterations; one factor each from B2a on)

| run | change vs previous reference | half_B merge WF1 (d) | final-epoch WF1 (d) | merge oracle (d) | final oracle (d) | merge noisy EER (d) | final EER | merge clean F1 (d) | final clean |
|---|---|---|---|---|---|---|---|---|---|
| B0 | baseline (RTC aug, constant lr 3e-5, 10 ep) | 83.61 | - (best_by_wf1 ep 6: 80.94) | 84.71 | - | 18.01 | - | 92.37 | - |
| B1 | + MUSAN + level + recipe with label smoothing, cos30 stopped at 13 | 81.11 (-2.50) | - | 84.26 (-0.45) | - | 19.07 (+1.06) | - | 87.89 (-4.48) | - |
| B2a | B1 aug + recipe, no LS, cos8 at lr 3e-5 | 84.95 (+1.34 vs B0) | 85.18 | 87.04 (+2.33) | 87.09 | 15.77 (-2.24) | 15.63 | 92.46 (+0.09) | 92.98 |
| B2b | `--lr 1e-5` | 89.59 (**+4.64**) | 90.34 (**+5.16**) | 91.27 (+4.23) | 92.00 (+4.91) | 12.00 (-3.77) | 11.26 (-4.37) | 95.10 (+2.64) | 95.59 (+2.61) |
| B2c | `--lr 5e-6` | 91.73 (**+2.14**) | 92.26 (**+1.92**) | 92.60 (+1.33) | 92.74 (+0.74) | 10.11 (-1.89) | 10.06 (-1.20) | 96.82 (+1.72) | 97.42 (+1.83) |
| B2d | `--num_epochs 10` | 91.57 (-0.16) | 91.84 (-0.42) | 92.36 (-0.24) | 92.47 (-0.27) | 10.44 (+0.33) | 9.96 (-0.10) | 96.73 (-0.09) | 97.18 (-0.24) |
| **B2e** | `--lr 2.5e-6 --lr_min 5e-7` | 92.34 (**+0.61**) | 92.53 (**+0.26**) | 92.88 (+0.28) | 92.97 (+0.23) | 9.68 (-0.43) | 9.27 (-0.79) | 97.67 (+0.85) | 98.09 (+0.67) |

Per ln(LR) the series is 4.22 -> 3.09 -> **0.88** (merge) and 4.70 -> 2.77 -> **0.38** (final) at threshold 0.5; at the oracle 3.85 -> 1.92 -> 0.40 (merge) and 4.47 -> 1.07 -> 0.33 (final). The B2d report's geometric continuation predicted +1.0 (merge) / +0.7 (final) at 0.5 and +0.4 / +0.1 at the oracle; the operating-point part came in at 60 % / 37 % of that, the oracle part as predicted. The lever has saturated: the third halving is inside the near-replicate band on every aggregate.

## 3. Per-epoch behaviour, B2e vs B2c (aligned by epoch index = aligned by step; LR is exactly half)

Source: `metrics.jsonl` (EMA weights; 2,000 clean clips + `sim_matched_mini` + `sim_heldout_mini`, half A) and `logs/train.log`. cl/ma/he = clean / matched-mini / heldout-mini.

**B2e** (`1895 warmup / 37896 steps, floor 5.0e-07`)

| ep | LR end | WF1 | WF1@or | F1 clean | F1 noisy | EER noisy | thr_or cl/ma/he | ECE cl/ma/he | R_bona cl/ma/he | R_spoof cl/ma/he | EER cl/ma/he | train acc | train loss | dev loss | dev acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.47e-6 | 88.07 | 89.72 | 96.63 | 84.40 | 13.88 | **0.48/0.04/0.21** | .013/.064/.055 | **93.0/81.6/73.5** | 99.2/92.7/93.8 | 4.03/13.40/14.36 | 87.19 | 0.358 | **0.082** | 97.86 |
| 2 | 2.29e-6 | 91.08 | 91.55 | 97.26 | 88.44 | 12.50 | 0.99/0.76/0.70 | .015/.045/.070 | 91.6/76.0/68.9 | 99.9/99.0/97.8 | 4.19/11.65/13.35 | 97.08 | 0.219 | 0.101 | 98.43 |
| 3 | 1.98e-6 | 91.01 | 91.88 | 96.90 | 88.48 | 11.78 | 0.99/0.40/0.87 | .017/.046/.067 | 90.5/76.5/69.4 | 99.9/98.8/97.8 | 4.25/10.91/12.66 | 98.00 | 0.170 | **0.124** | 98.34 |
| 4 | 1.58e-6 | **92.19** | 92.56 | 97.26 | 90.01 | 10.95 | 1.00/0.39/0.03 | .015/.044/.061 | 91.6/79.3/73.5 | 99.9/98.8/98.1 | 3.78/9.56/12.34 | 98.50 | 0.126 | 0.115 | 98.65 |
| 5 | 1.18e-6 | 91.90 | 92.47 | 97.35 | 89.57 | 10.99 | 1.00/0.36/0.03 | .015/.045/.067 | 91.9/78.8/72.4 | 99.9/98.8/97.9 | 3.93/9.13/12.85 | 98.82 | 0.102 | 0.120 | 98.62 |
| 6 | 8.23e-7 | 91.92 | 92.68 | 97.81 | 89.40 | **10.63** | 1.00/0.01/0.87 | .012/.048/.066 | 93.3/79.3/71.9 | 99.9/98.4/98.0 | 4.30/9.04/12.22 | 99.03 | 0.083 | 0.119 | 98.78 |
| 7 | 5.84e-7 | 91.75 | 92.39 | 97.62 | 89.23 | 11.13 | 1.00/0.01/0.97 | .014/.045/.069 | 92.4/79.3/69.9 | 100.0/98.7/98.1 | 4.17/10.30/11.97 | 99.29 | 0.064 | 0.123 | 98.72 |
| 8 | 5.00e-7 | **92.64** | **92.97** | **97.89** | **90.38** | 11.17 | 1.00/0.01/0.82 | .012/.042/.066 | 93.3/81.0/73.5 | 100.0/98.8/98.3 | 4.30/9.50/12.83 | 99.42 | 0.049 | 0.126 | **98.89** |

**B2c** (reference, floor 1.0e-06; same columns)

| ep | LR end | WF1 | WF1@or | F1 clean | F1 noisy | EER noisy | thr_or cl/ma/he | ECE cl/ma/he | R_bona cl/ma/he | R_spoof cl/ma/he | EER cl/ma/he | train acc | train loss | dev loss | dev acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.94e-6 | 87.69 | 89.89 | 96.24 | 84.03 | 12.94 | 0.37/0.15/0.17 | .015/.083/.090 | 93.8/87.2/82.7 | 98.7/90.7/89.9 | 4.55/11.59/14.29 | 87.94 | 0.361 | 0.080 | 97.66 |
| 2 | 4.58e-6 | 91.28 | 91.78 | 97.26 | 88.72 | 10.92 | 0.99/0.69/0.85 | .016/.042/.065 | 91.3/76.0/69.9 | 100.0/99.1/97.8 | 4.53/9.50/12.34 | 96.97 | 0.195 | 0.081 | 98.55 |
| 3 | 3.95e-6 | 91.31 | 92.00 | 96.24 | 89.20 | 11.79 | 0.99/0.90/0.44 | .020/.046/.063 | 88.2/74.9/71.9 | 100.0/99.1/98.3 | 4.85/10.61/12.98 | 98.01 | 0.151 | 0.105 | 98.30 |
| 4 | 3.17e-6 | 91.01 | 91.90 | 96.15 | 88.81 | 11.85 | 1.00/0.81/0.91 | .020/.047/.069 | 88.2/76.0/68.9 | 99.9/99.1/98.3 | 4.50/10.48/13.22 | 98.60 | 0.114 | 0.111 | 98.44 |
| 5 | 2.35e-6 | 91.83 | 92.56 | 96.61 | 89.78 | 11.38 | 1.00/0.94/0.94 | .019/.047/.062 | 89.3/75.4/72.4 | 100.0/99.3/98.8 | 3.10/10.61/12.15 | 98.90 | 0.083 | 0.119 | 98.47 |
| 6 | 1.65e-6 | 91.92 | 92.80 | 96.80 | 89.83 | **10.07** | 1.00/0.95/0.85 | .018/.043/.061 | 89.9/76.5/71.9 | 100.0/99.5/98.4 | 3.13/8.49/11.65 | 99.21 | 0.065 | 0.120 | 98.50 |
| 7 | 1.17e-6 | **92.44** | **92.92** | 97.44 | 90.29 | 10.38 | 1.00/0.93/0.94 | .014/.040/.064 | 91.9/77.7/70.9 | 100.0/99.6/99.0 | 3.07/8.85/11.91 | 99.38 | 0.051 | 0.120 | 98.58 |
| 8 | 1.00e-6 | 92.33 | 92.77 | 97.26 | 90.22 | 10.25 | 0.99/0.78/0.94 | .016/.042/.064 | 91.3/77.7/70.9 | 100.0/99.6/98.9 | 2.54/8.67/11.84 | 99.50 | 0.042 | 0.112 | 98.59 |

Reading:
- **Same fit, same timing, half the LR.** Train acc crosses 95 % at epoch 2 in both (97.08 / 96.97 %) and the clean `thr_oracle` jumps from 0.48 / 0.37 to 0.99 at the same epoch: the spoof offset is installed at epoch 2 by fit level at every peak LR of the wave (3e-5: 0.95, 1e-5: 0.96, 5e-6: 0.99, 2.5e-6: 0.99). Epoch 1 is not more under-fit at half LR (88.07 vs 87.69 mini WF1, train acc 87.19 vs 87.94 %, dev loss 0.082 vs 0.080); the end state is marginally less fitted (train acc 99.42 vs 99.50, train loss 0.049 vs 0.042, dev loss 0.126 vs 0.112) with the highest dev acc of the wave (98.89 %).
- **Where the plateau is.** B2e reaches 92.19 at epoch 4 (LR 1.58e-6, the LR at which B2c was at epochs 6-7) and then stays within 92.19 / 91.90 / 91.92 / 91.75 / 92.64 (epochs 4-8, range 0.89, sd 0.33); B2c's plateau is epochs 5-8 (91.83-92.44, range 0.61). The useful part of both trajectories is again the LR <= 1.6e-6 tail; B2e spends 5 epochs there instead of 2 and gains 0.2 on the last epoch (92.64 vs 92.44 best; 92.64 vs 92.33 at epoch 8), the same order as the B2d-vs-B2c difference (-0.30 at epoch 10 vs 8).
- **Dev loss.** 0.082 -> 0.101 -> **0.124 (+23 %, epoch 3)** -> 0.115 -> 0.120 -> 0.119 -> 0.123 -> 0.126. The bump is one epoch later than B2c's (epoch 3 vs 2-3) and the same size as B2c's (+30 %) and smaller than B2d's (+35 % at epoch 4); afterwards the loss is flat at 0.115-0.126 and ends 0.014 above B2c's. The predicted "dev loss <= 0.11 and not rising after epoch 4" did not come out (section 6).
- **Half A and half B disagree on the EER.** On the mini sets B2e's noisy EER is worse than B2c's at every epoch from 5 on (10.95-11.17 vs 10.07-10.38; matched-mini 9.0-9.5 vs 8.5-8.9, heldout-mini 12.0-12.8 vs 11.7-11.9), yet on half_B it is better by 0.79 (9.27 vs 10.06). The matched-mini `thr_oracle` collapses to 0.005-0.008 at epochs 6-8 (a few spoof clips scored near 0, R_spoof 98.4-98.8 on the mini set), the half-A signature of the spoof-recall loss seen on half_B (section 2.2). The two halves of a 1,000-clip mini set disagree at the 0.5-1.0 EER level, which is the size of the effect being read.
- **Final vs merge.** The final EMA (92.53) beats the merge (92.34) by **+0.19** (B2a +0.23, B2b +0.75, B2c +0.53, B2d +0.27: five of five), by +0.09 at the oracle (92.97 vs 92.88) and, for the first time, also on EER (9.27 vs 9.68). `best_by_wf1` = epoch 8; `best_by_noisy_eer` = epoch 6. The merge has no early-plateau epoch (8, 4, 6, 5, 7) and its deficit shrank accordingly.
- **Noise band from the near-replicate pairs** (same seed, same data order, one schedule factor apart): B2d vs B2c -0.42 final / -0.16 merge, max per-set |d| 1.26 (heldout) / 1.44 (echo); B2e vs B2c +0.26 / +0.61, max |d| 1.08 (heldout) / 2.05 (echo). Two one-factor tweaks around 5e-6 move the final readout by 0.3-0.4 and single sets by 1-2 points. A true second seed has still not been run; this is a lower bound.

## 4. What went OK (evidence)

1. **Every aggregate improved, both readouts**: merge +0.61 / oracle +0.28 / clean +0.85 / noisy +0.51 / EER -0.43; final +0.26 / +0.23 / +0.67 / +0.09 / -0.79. 12 of 12 aggregate comparisons in B2e's favour, 46 of 63 slices up at F1 (merge), 52 of 63 at R_bona. Nothing is outside the band, but the sign is uniform.
2. **Best single checkpoint of the wave**: half_B final EMA 92.53 (B2c 92.26), mini WF1 92.64 (B2c 92.44 at epoch 7), mini WF1@oracle 92.97, dev acc 98.89 %. `best_by_wf1` is the final epoch, as in B2d.
3. **The spoof prior shrank.** Optimal global threshold on the half_B scores 0.84 (logit +1.66) for the final epoch, 0.88 (+1.99) for the merge, vs 0.96 (+3.18) for both B2c readouts; matched `thr_oracle` 0.841 / 0.874 vs 0.987 / 0.960; noisy-bona false alarms 24.0 % / 25.1 % (B2c 25.8 / 27.2); tail at p >= 0.99 16.1 / 14.2 % (17.1 / 15.7). The lower LR leaves less for a later calibration or ensembling step to undo.
4. **Probes at their best values of the wave**: level_flip_bona 1.40 / 1.87 (B0 7.29), `probe_orig` bona false alarms 10.70 / 8.02 % (B0 14.97, B2c 13.90 / 11.23), trimmed / padded flips 1.07 / 2.67-3.21 %.
5. **Merge composition clean**: epochs 8, 4, 6, 5, 7 - no epoch <= 3; `--keep_topk_by_wf1 7` pruned nothing (epoch 1 protected as `best_by_dev_loss`); the merge-vs-final deficit fell to 0.19.
6. **The pre-registered rule produced a clean decision** (section 6): case (a), both conditions met with margin (0.26 of 0.5, 0.23 of 0.3).
7. **Prediction check of the B2d report's "predicted signature"**: train acc crosses 95 % at epoch 2 (met), plateau mini WF1 >= 91.5 from epoch 4 (met: 92.19), best epoch in 6-8 (met: 8), final-epoch train acc 98.6-99.3 (99.42, just above), train loss 0.05-0.08 (0.049, just below), epoch-1 train acc 84-87 % (87.19, at the edge), merge >= 92.2 (met: 92.34), mean noisy EER <= 9.6 (met: 9.27), clean F1 >= 97.4 (met: 98.09). Not met: epoch-1 mini WF1 85.5-87.5 (88.07, higher), dev loss <= 0.11 (0.126), epochs 6-8 within 0.5 (0.89), **final WF1 >= 92.8 (92.53)**, **oracle >= 93.0 (92.97)**, heldout EER <= 10.7 (10.86), heldout R_bona >= 75 (71.61), tail <= 15 % (16.1), English heldout R_bona >= 65 (59.7). The hypothesis "the ranking should improve by more than the geometric extrapolation" is refuted: oracle +0.23 / +0.28, exactly the extrapolation.

## 5. What went wrong / open problems (evidence)

### 5.1 Why the third halving gave less than the first two

- **The fit is identical.** Train acc at epoch 2 / 8: 97.08 / 99.42 (B2e) vs 96.97 / 99.50 (B2c) vs 96.76 / 99.56 (B2d). The 3e-5 -> 1e-5 step changed the epoch-1 state (B2a 83.90 % train acc, mini WF1 83.89 -> B2b 88.01 %, 87.98), the 1e-5 -> 5e-6 step still did (B2c 87.94 %, 87.69, clean R_bona 93.8), the 5e-6 -> 2.5e-6 step did not (87.19 %, 88.07, 93.0): the "feature preservation" part of the LR lever was exhausted at 5e-6, as the B2d report noted. What remains is the operating-point part, and that is 0.3-0.6 (section 5.2).
- **Heldout did not follow.** Heldout F1 -1.08 (ep) is 13 bona clips broken vs 9 fixed and 16 spoof broken vs 7 fixed; the losses sit in transients and low SNR (wind -2.57, snr 5 -2.52, snr 0 -2.41, clock_tick -2.27, door_wood_creaks -2.15, door_wood_knock -2.08) and in English (-1.03; R_bona 60.9 -> 56.6 on the full set, 62.0 -> 59.7 on half_B). Exactly the B2d losers; two different one-factor changes at ~5e-6 moved these slices by 1-3 points in both directions, so they are the noisy and un-addressed part of the metric (section 5.3).
- **Spoof recall moved for the first time**: matched 99.19 -> 98.81 (6 clips), heldout 99.19 -> 98.63 (9), min slice R_spoof matched / snr 0 96.13. Not a problem at this level (floor 98.5 on the sets), but the lower LR has started trading spoof margin for bona recall, which is the direction in which a further halving would start to cost.

### 5.2 Oracle-gap decomposition and the saturated tail

Sweep of one shared threshold over the half_B scores (`local_eval/*/scores/*.txt`, p_spoof; grid 0.005 to 0.995 step 0.005 plus 1e-4, 5e-4, 1e-3, 0.999, 0.9995, 0.9999; threshold 0.5 reproduces 92.34 / 97.67 / 91.60 / 88.50 / 94.88, 92.53 / 98.09 / 91.90 / 88.39 / 94.79, 91.87 / 97.27 / 90.66 / 88.45 / 93.40 and B2c's 91.73 / 92.26 exactly):

| readout | global thr | logit | WF1 | clean | matched | heldout | echo | R_bona cl/ma/he/ec | R_spoof cl/ma/he/ec |
|---|---|---|---|---|---|---|---|---|---|
| B2e-ep8 | 0.5 | 0 | 92.53 | 98.09 | 91.90 | 88.39 | 94.79 | 94.1/80.3/71.6/85.2 | 100.0/98.8/98.6/99.8 |
| | 0.9 | +2.20 | 92.66 | 98.13 | 91.95 | 88.66 | 94.65 | | |
| | **0.840 (best single)** | **+1.66** | **92.70** | 98.13 | 92.18 | 88.56 | 94.65 | 94.3/81.1/72.6/85.4 | 100.0/98.8/98.4/99.6 |
| | 0.99 | +4.60 | 91.84 | 98.27 | 91.20 | 86.97 | 94.58 | | |
| | per-set oracle (`metrics.json`) | | 92.97 | 98.32 (.991) | 92.18 (.841) | 89.17 (.968) | 95.13 (.966) | | |
| B2e-top5 | 0.5 | 0 | 92.34 | 97.67 | 91.60 | 88.50 | 94.88 | 93.1/79.3/70.6/85.4 | 99.9/98.9/99.1/99.8 |
| | **0.880 (best single)** | **+1.99** | **92.65** | 97.86 | 91.93 | 88.89 | 95.06 | 93.8/80.8/72.9/85.9 | 99.9/98.7/98.6/99.8 |
| | per-set oracle | | 92.88 | 98.21 (.995) | 92.09 (.874) | 89.09 (.967) | 95.16 (.936) | | |
| B2c-ep8 / B2c-top5 (same grid) | best single 0.960 / 0.960 | +3.18 / +3.18 | 92.46 / 92.17 | | | | | 93.1/77.2/76.5/82.9 (ep) | |
| avg-B2c8-B2e8 | 0.5 | 0 | 91.87 | 97.27 | 90.66 | 88.45 | 93.40 | 91.7/74.9/70.1/80.8 | 99.9/99.4/99.2/99.9 |
| | **0.955 (best single)** | **+3.06** | **92.48** | 97.72 | 91.48 | 88.97 | 94.35 | 93.3/81.8/78.5/85.7 | 99.9/98.0/96.9/99.3 |
| | per-set oracle | | 92.65 | 97.90 (.990) | 91.76 (.958) | 89.06 (.786) | 94.50 (.948) | | |

**Decomposition**: oracle gap 0.44 (ep8) / 0.54 (merge). One global logit shift recovers **0.17 (39 %) / 0.31 (57 %)**; the set-specific remainder is 0.27 / 0.23. B2c-ep8: 0.48 gap, 0.20 global. The sweep is flat between 0.5 and 0.9 (92.53 -> 92.66 for ep8), so there is still no cheap re-threshold, and the ceiling of any operating-point lever on the primary readout is now **0.17 (global) to 0.44 (per-set oracle)** - the smallest of the wave.

Share of half_B bona clips with p_spoof >= 0.5 / >= 0.9 / >= 0.99 / >= 0.999 (and spoof with p < 0.5):

| set (n bona) | B2c-top5 | **B2e-top5** | B2c-ep8 | **B2e-ep8** | avg-B2c8-B2e8 | spoof p < 0.5: B2c-ep8 / B2e-ep8 / avg |
|---|---|---|---|---|---|---|
| clean (1,397) | 9.7 / 8.9 / 6.7 / 5.1 | 6.9 / 6.2 / 5.2 / 4.1 | 7.7 / 7.3 / 6.0 / 4.8 | **5.9** / 5.7 / 5.0 / 4.4 | 8.3 / 7.4 / 5.5 / 3.9 | 0.12 / 0.03 / 0.07 % |
| matched (391) | 26.1 / 24.6 / 16.1 / 9.2 | 20.7 / 18.9 / 12.8 / 7.9 | 24.3 / 24.3 / 17.4 / 11.0 | **19.7** / 18.9 / 14.8 / 8.4 | 25.1 / 21.2 / 12.0 / 5.9 | 0.81 / **1.19** / 0.56 % |
| heldout (391) | 28.4 / 26.6 / 15.3 / 8.2 | 29.4 / 26.9 / 15.6 / 8.4 | 27.4 / 26.6 / 16.9 / 9.2 | **28.4** / 27.1 / 17.4 / 9.7 | 29.9 / 25.8 / 12.8 / 5.1 | 0.81 / **1.37** / 0.81 % |
| echo (391) | 19.4 / 18.7 / 12.8 / 6.4 | 14.6 / 14.1 / 9.7 / 5.6 | 19.2 / 18.4 / 13.0 / 6.9 | **14.8** / 14.6 / 10.7 / 6.6 | 19.2 / 16.4 / 9.2 / 3.6 | 0.37 / 0.25 / 0.12 % |
| noisy bona (782): FA / at >= 0.99 / share | 27.2 / 15.7 / 58 % | 25.1 / 14.2 / 57 % | 25.8 / 17.1 / 66 % | 24.0 / 16.1 / 67 % | 27.5 / 12.4 / 45 % | |

The matched and echo tails shrank by 10 and 9 clips at p >= 0.99 (ep), the clean tail by 14; **the heldout tail did not shrink at all** (17.4 % vs 16.9 %, +2 clips; 9.7 % vs 9.2 % at >= 0.999). The saturated core (two thirds of the noisy false alarms at p >= 0.99) is where the metric is stuck; no LR moved it.

**Why the cross-LR weight average (`avg-B2c8-B2e8`, 91.87) is worse than either parent (92.26 / 92.53).** Facts from the scores: (i) the parents' logits are correlated 0.990 / 0.969 / 0.968 / 0.977 (clean / matched / heldout / echo); (ii) per clip the average is wrong on 12 / 2 / 6 / 3 clips that both parents get right, right on 0 / 1 / 1 / 1 clips that both get wrong, and on the 59 / 64 / 45 / 39 clips where the parents disagree it sides with the right one only 21 / 28 / 24 / 16 times (36-53 %); (iii) at 0.5 it is **more spoof-biased than either parent** (R_bona 91.7 / 74.9 / 70.1 / 80.8 vs B2e 94.1 / 80.3 / 71.6 / 85.2 and B2c 92.3 / 75.7 / 72.6 / 80.8; R_spoof 99.4-99.9, the highest; noisy-bona FA 27.5 %) while its saturated tail is the smallest (12.4 % at p >= 0.99, 3.6-5.9 % at >= 0.999); (iv) its oracle gap is 0.78, of which 0.61 (78 %) is one global shift (best threshold 0.955, logit +3.06, -> 92.48, i.e. B2c's level); at the per-set oracle it is 92.65, within 0.1 / 0.3 of the parents. So the averaged weights keep the parents' ranking but land at a worse operating point: the two checkpoints sit at different distances from the shared init (one schedule is 2x the other) and different confidence scalings (optimal logit offsets +3.18 vs +1.66), and the midpoint in weight space does not inherit the midpoint of the offsets. **The score average (mean p_spoof of the two `epoch_8` score files) gives half_B WF1@0.5 = 92.90** (clean 97.97, matched 92.01, heldout 89.45, echo 94.79; best global threshold 0.91 -> 93.20): it keeps B2c's heldout level and B2e's matched / echo / clean level, +0.37 over the better parent with no fitted parameter. The parents are complementary per set; weight averaging across LR scales throws that away, score averaging keeps it. Consequence for the full run: average seeds at the score level by default, test the weight average per pair and keep it only if it beats both parents on half_B.

### 5.3 Where the bona false alarms sit (B2e-ep8; B2c-ep8 in brackets)

- **Language** (half_B, 129 en / 262 zh bona per set; one en clip = 0.78 points): R_bona@0.5 en **65.9 / 59.7 / 71.3** (65.9 / 62.0 / 71.3) vs zh **87.4 / 77.5 / 92.0** (80.5 / 77.9 / 85.5) on matched / heldout / echo. English did not move (0 / -3 / 0 clips), Chinese gained 18 / -1 / 17; the en-zh gap widened from 14.6 / 15.9 / 14.2 to **21.5 / 17.8 / 20.7**. en tail at p >= 0.99: 24.8 / 28.7 / 20.9 % (24.8 / 25.6 / 19.4); zh 9.9 / 11.8 / 5.7 (13.7 / 12.6 / 9.9). Full-set heldout en F1 82.97 (84.01), EER 18.99 (17.12). Across the whole wave, heldout English bona recall went 61.2 (B0 merge) -> 58.1 (B2e merge) and matched English 64.3 -> 63.6: **the +8.7 WF1 of the wave contains no English-bona gain at all.**
- **Worst slices, B2e-ep8** (full sets): matched snr 0 F1 **82.19** (R_bona 64.1, R_spoof 96.13), heldout en 82.97 (56.6, EER 18.99), washing_machine 83.36 (65.4), wind 84.06 (64.5), heldout snr 0 84.25 (65.3, 17.30), vacuum_cleaner 86.03 (71.4), gsm 86.31 (69.1), matched en 86.48 (64.0), g726 87.36 (68.7), opus:wechat 87.59 (71.2), heldout snr 5 87.71 (74.4), matched rain 87.91 (73.0). The same list as B2c and B2d; B0's R_bona on these was 73.5 / 61.2 / 53.8 / 68.4 / 62.6 / 65.7 / 68.7 / 64.3 / 65.9 / 76.9 / 75.0 / 69.6, i.e. unseen-codec, English and SNR-0 bona recall is at or below B0's level after five iterations while their spoof recall went from 87-90 to 96-99.
- **SNR** (R_bona ep): matched 90.3 (none) -> 86.6 (20) -> 82.6 (15) -> 76.6 (10) -> 78.8 (5) -> **64.1 (0)** [B2c 83.9 / 81.1 / 80.7 / 71.8 / 75.4 / 62.4]; heldout 77.8 (20) -> 74.2 (15) -> 73.7 (10) -> 74.4 (5) -> **65.3 (0)** [80.6 / 72.8 / 73.0 / 75.6 / 66.7]. SNR 0 unchanged, SNR >= 10 on matched up 4-5 points: the lower LR fitted the trained SNR band better and the untrained one not at all (training SNR is 5-20 dB).
- **Non-speech probes**: `noise_only` / `sil_only` bona-source clips 100 % spoof in both readouts, mean dP +0.89 / +0.91 - unchanged since B2b. Not what WF1 measures; it is the C2 target in `docs/next_steps_plan.md`.

### 5.4 Other open items

- **One seed per configuration, five runs.** The near-replicate pairs give 0.26-0.42 on the final readout and 1-2 per set; the formal +-1.5 rule was never challenged by a B2 result after B2c and the real band for a second seed is unknown. Two seeds of the chosen configuration are the first thing the full run must deliver.
- Registry rows `B2e-lr2.5e6-screen` (92.34) and `B2e-lr2.5e6-ep8` (92.53) carry `revisit`: inside the band, not promoted, best readouts of the wave.
- Half A (mini sets) and half B disagree on the EER sign (section 3); per-epoch selection by mini WF1 picked the right final epoch here, but mini noisy EER would have picked epoch 6.
- Heldout R_spoof 98.63 is the lowest final-epoch value of the wave (22 misses); still above the 98.5 floor used in the signatures, to be watched if the LR is lowered further (not proposed).

## 6. Verdict on the pre-registered falsification rule

The B2d report (section 6) pre-registered, on the final-epoch half_B readout vs B2c-ep8 (92.26, oracle 92.74):
- **(a) ep8 within +-0.5 of 92.26 and oracle within +-0.3 of 92.74 -> the LR series has turned over; the recipe's LR is 5e-6 and the full run uses B2c's flags.** Measured: **ep8 92.53 (+0.26, inside +-0.5); oracle 92.97 (+0.23, inside +-0.3). Both conditions hold: case (a) fired.**
- (b) ep8 < 91.8 with train acc < 98.5 %, train loss > 0.08 and best mini epoch = 8 by > 0.3 over epoch 7 (under-fit): ep8 92.53, train acc 99.42 %, train loss 0.049 - not fired (epoch 8 is the best mini epoch by 0.45 over epoch 4 and 0.89 over epoch 7, but with the fit complete, which is not the under-fit signature).
- (c) ep8 >= 92.8 with EER down on >= 3 of 4 sets and R_spoof >= 98.5: ep8 92.53 (not >= 92.8); EER down on 2 of 4 (matched -1.29, heldout -0.29; clean +0.84, echo +0.19); R_spoof >= 98.63 - not fired.
- Formal promotion (> 93.76 on ep8 or > 93.23 on the merge with clean F1 >= 96.92): not met (92.53 / 92.34).

**What it implies for the LR choice.** The LR series 3e-5 -> 1e-5 -> 5e-6 -> 2.5e-6 gave +4.64 / +2.14 / +0.61 (merge) and +5.16 / +1.92 / +0.26 (final); the third step is inside the near-replicate band on every aggregate and 5e-6 and 2.5e-6 are **statistically equivalent on one seed**. The series has not reversed (the sign is uniformly positive, 12 of 12 aggregates), it has saturated; a fourth halving (1.25e-6) is not proposed: the oracle part is 0.2-0.3 per halving and falling, the spoof side has started to move (R_spoof 98.6-98.8), and the loop is over. The rule's literal prescription is "the full run uses B2c's flags (5e-6)". Section 7 argues that, the two being equivalent, the tie-breakers favour 2.5e-6, and records the choice as a tie-break, not as a measured gain.

## 7. Decision on the baseline

- **B2e does not replace B2c as the formal reference.** Rule: > +1.5 half_B WF1@0.5 with clean F1 down by no more than 0.5. B2e: merge 92.34 - 91.73 = **+0.61**, final 92.53 - 92.26 = **+0.26**; clean +0.85 / +0.67. Not promoted; registry rows carry `revisit`. B2c's `local_eval/merge_top5` (91.73) and `local_eval/epoch_8` (92.26) remain the `--baseline` targets for any further `/post-train-eval` in this lineage.
- **Recommended configuration for the full run: B2e's flags (`--lr 2.5e-6 --lr_min 5e-7`), as a tie-break.** Pre-registration says the two are equivalent (case (a)); with no measured difference, the choice goes on the ensemble of evidence, and that ensemble is one-sided: B2e wins on WF1 (+0.26 / +0.61), oracle (+0.23 / +0.28), noisy EER (-0.79 / -0.43), clean F1 (+0.67 / +0.85), noisy F1 (+0.09 / +0.51), oracle gap (0.44 vs 0.48, 0.54 vs 0.87), spoof prior (optimal logit +1.66 vs +3.18), level_flip_bona (1.87 vs 2.54, 1.40 vs 1.80), `probe_orig` false alarms (8.0 vs 11.2 %), tail at p >= 0.99 (16.1 vs 17.1 %), best mini epoch (92.64 vs 92.44), dev acc (98.89 vs 98.59), and merge composition (no early epoch); same cost (4 h 05 vs 4 h 06). B2c wins on heldout F1 (89.46 vs 88.39 ep; 88.98 vs 88.50 merge), heldout R_spoof (99.19 vs 98.63), heldout R_bona (72.63 vs 71.61) and clean EER (2.51 vs 3.35) - all inside the per-set band that B2d (same LR, two more epochs) showed to be 1.3 on heldout. **If the user prefers the literal pre-registration, 5e-6 (B2c's flags) is equally defensible and the expected difference is below 0.5 either way**; what matters more than the LR is running 2-3 seeds and combining them at the score level (section 5.2). The full-run command line, launcher body, expected numbers and the one-seed caveat are in `20261008_232331_B2_wave_summary_and_full_run_recommendation.md`, section 4.
- **Readout for the full run: final-epoch EMA checkpoint, half_B WF1@0.5**, protected by `--keep_epochs 8` (it ranked 1st on half A in B2d and B2e, 2nd in B2c, 3rd in B2b; a new seed can rank it 8th and `--keep_topk_by_wf1 7` would then delete it). Floors to beat with the first full-run seed: 92.53 (B2e-ep8); band +-0.5; formal 93.76 / 93.23.
