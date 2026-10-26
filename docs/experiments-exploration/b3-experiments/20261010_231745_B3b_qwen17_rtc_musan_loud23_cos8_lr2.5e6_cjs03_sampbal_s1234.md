# Experiment exploration B3-02: B3b (`B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_sampbal_s1234`) - class- and language-balanced sampler on top of B3a

> Written 2026-10-10 by the exploop B3 analysis agent (ANALYZE step, wave B3, iteration 2, attempt a0).
>
> **Sources.** Every number is copied from a file read in this session, or computed in this session from the half_B rows of the local `scores/*.txt` / `*.logits` files (plus the protocol / meta files of the local sets). Nothing is quoted from memory. The files:
> - run config and logs: `config.yaml` and `git.diff` (B3b), `config.yaml` (B3a), `logs/train.log` (both runs), `metrics.jsonl` (both runs), `top5_by_wf1/{merge_list.txt,select_topk.log}`, `local_eval/merge_top5/merge_list.txt`, the `ckpt/` listing;
> - local evals: `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,compare_vs_B3a-*.txt,ci_vs_B3a-*.txt,compare_ci_vs_B3a-*.json}`, `local_eval/merge_top5/compare_all_readouts.txt`, `local_eval/{merge_top5,epoch_8}.log`; the same `metrics.json` / `probe_report.json` / `scores/` of B3a (`merge_top5`, `epoch_8`) and B2f;
> - stdout log: `SLS_setup/B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_sampbal_s1234_20261010145607.log` (header lines only);
> - loop files: `exploop/B3/iter02/{verdict.json,design.result.json}`, `exploop/B3/{wave.json,plan_snapshot.md,STATUS.md,user_notes.md}`, `scripts/run_B3b.sh`;
> - registry and code: `docs/experiment_log.csv`, `SLS_setup/b3wave/train.py` (CE / dev-loss criterion), `SLS_setup/b3wave/sampler.py` (draw rule), `SLS_setup/utils/local_eval.py` and `SLS_setup/utils/metrics.py` (half split, WF1 definition, `p >= thr`).
>
> Progress-split scores were not read.
>
> **Previous reports.** `docs/experiments-exploration/b3-experiments/20261010_143058_B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234.md` (the reference run; source of the B3a numbers marked "B3a report"), `docs/experiments-exploration/20261009_175355_B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234.md` (template, fixed comparator) and `docs/experiments-exploration/20261008_232331_B2_wave_summary_and_full_run_recommendation.md`.
>
> **Readouts.**
> - `-top5` is the weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5/`). It is the wave objective.
> - `-ep8` is the final-epoch EMA checkpoint alone (`local_eval/epoch_8/`).
> - half_B is the half of each sim set that is never used for selection: n = 1,994 per set, 391 bona. Clean is the full 7,465-clip `dev_online_clean` (1,397 bona; 479 English, 918 Chinese).
> - WF1 = 0.3 clean + 0.35 matched + 0.35 heldout (macro-F1 at p_spoof 0.5). Echo is reported but is not in WF1.
> - Threshold 0.5 on the score files reproduces all six half_B WF1 values of the `metrics.json` files exactly: B2f 91.98 / 92.14, B3a 93.06 / 92.94, B3b 92.23 / 92.36 (top5 / ep8).
> - Every evaluation ran with `--loudness_norm -23`: `[loudness_norm] loudness_norm target=-23 LUFS` is in `merge_top5.log`, and every `metrics.json` has `"loudness_norm": -23.0`.

Run directories (under `SLS_setup/exp/`):
- **B3b** `B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_sampbal_s1234_epoch8_bs8_20261010145615` (this report)
- **B3a** `B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_epoch8_bs8_20261009214846` (reference = current best: merge 93.06, final 92.94)
- **B2f** `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623` (fixed comparator: merge 91.98, final 92.14)

## 1. Setup

**B3b = B3a with one coupled factor changed: the class- and language-balanced sampler, `--sampler balanced --sampler_lang_balance --sampler_bona_frac 0.5`, with `--ce_weights 0.3223 0.6777` re-derived so that the effective bona:spoof weight of 0.1/0.9 at 18.94 % bona (2.1028 : 1) is unchanged** (`design.result.json` `derivation`). The sampler draws `len(dataset)` = 75,785 items per epoch with replacement, so a draw is bona with probability 0.5, and within each class half the draws are English and half Chinese (`b3wave/sampler.py`). Expected draws per clip per epoch (`derivation`): en bona 6.48×, en spoof 1.29×, zh bona 1.66×, zh spoof 0.40× (shuffle = 1.00×).

Flag diff (`args` of both `config.yaml`; nothing else differs):

| flag / field | B3a | **B3b** | note |
|---|---|---|---|
| `--sampler` | `shuffle` | **`balanced`** | the lever |
| `--sampler_lang_balance` | false | **true** | |
| `--sampler_bona_frac` | 0.5 (inactive) | **0.5** | |
| `--ce_weights` (spoof, bona) | 0.1 0.9 | **0.3223 0.6777** | coupled; same effective prior (`wave.json` lever_notes) |
| `sampler_expected_shares` | null | bona 0.5, en 0.5 | |
| git | `df3dff2`, clean | `df3dff2`, `git_dirty: true` | `git.diff` touches only `.claude/skills/post-train-eval/SKILL.md` and `docs/experiment_log.csv`; no trainer code |
| everything else | identical | identical | consistency `js` λ 0.3 pairs `both`, `--batch_size 8` (16 clips/step), 8 epochs, `--lr 2.5e-6 --lr_min 5e-7` warmup-cosine, RTC aug + MUSAN, `--loudness_norm -23`, `--keep_topk_by_wf1 7 --keep_epochs 8`, seed 1234 |

**Header lines proving the change took effect** (stdout log). They match all three `expect` regexes of the design:
- `Sampler: balanced bona_frac=0.5 lang_balance=True -> expected shares bona=0.500 en=0.500 (75785 draws/epoch with replacement)`
- `CE weights: spoof=0.3223 bonafide=0.6777`
- `Consistency: js lambda_js=0.3 lambda_emb=0.3 pairs=both clean_view=False | pairs found {'rows': 38660, 'blank_online': 1535, 'found': 37125} dropped_without_twin=0 | 16 clips per step (8 pairs)`
- Unchanged: `Warmup-cosine: 3790 warmup / 75792 steps, floor 5.0e-07` (same as B3a), `SNR: 5..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: fixed`, `LLRD: 1`, `SpecAugment: off`, `Loudness norm: loudness_norm target=-23 LUFS`, `Train trials: 75785`, `Parameters: 320268933`.
- **Realised sampler shares** (`train.log`, per epoch): bona 0.499-0.503, en 0.497-0.503.
- Per-epoch augmentation counts (`metrics.jsonl`, computed): the codec counts sum to 151,570 = 2 × 75,785 in every epoch, and the stage counts are 0.7988-0.8016 of the views (`p_apply=0.8`).
- The stdout log has no `Traceback`, `out of memory`, `failed (` or `Pruned` line (grep count 0 each).

**Run time.** `config.yaml` created 14:56:24; epoch 8 done 22:54:44; last local eval 22:55:10 = **7 h 59 min** (registry `gpu_h` 7.98; B3a 7.87). Epoch-end to epoch-end spacing 59.6-59.8 min (`train.log`). Post-train file times: `top5_by_wf1/merge_list.txt` 22:55, `merge_top5/metrics.json` 23:01, `epoch_8/metrics.json` 23:06, `model_merging_top5/` 23:16; `verdict.json` written 23:17:45.

**Checkpoints.** Nothing was pruned; all 8 epoch files are present at 1,281,235,691 bytes each. Symlinks: `best_by_dev_loss` → epoch 1 (0.094959), `best_by_noisy_eer` → epoch 3, `best_by_wf1` → epoch 7.

**Merged top-5 by half-A mini WF1 = epochs 7, 6, 8, 3, 4** (92.21, 91.93, 91.91, 91.76, 91.14; mean 91.79; `select_topk.log`). Excluded: 2 (90.53), 5 (91.09), 1 (89.68). B3a's merge was 6, 5, 7, 8, 2 (mean 92.31).

## 2. Results vs references (half_B)

Sources: `compare_all_readouts.txt`, `compare_vs_B3a-*.txt`, `ci_vs_B3a-*.txt` / `compare_ci_vs_B3a-*.json` (paired bootstrap n = 2,000, half B, threshold 0.5) and the `half_B` block of each `metrics.json`.

### 2.1 Wave iterations and fixed comparators, both readouts

| half_B | B2f-top5 | B2f-ep8 | B3a-top5 | B3a-ep8 | **B3b-top5** | **B3b-ep8** | B3b-top5 − B3a-top5 [95 % CI] | B3b-ep8 − B3a-ep8 [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| **WF1@0.5** | 91.98 | 92.14 | 93.06 | 92.94 | **92.23** | **92.36** | **−0.83 [−1.46, −0.25]**, p(not better) 0.998 | **−0.58 [−1.21, +0.02]**, p 0.973 |
| WF1@oracle | 92.59 | 92.82 | 93.24 | 93.45 | 92.96 | 92.86 | −0.28 | −0.59 |
| clean F1@0.5 | 97.60 | 97.69 | 97.48 | 97.06 | 96.96 | 96.77 | **−0.52 [−0.82, −0.23]** | −0.29 [−0.65, +0.06] |
| noisy F1@0.5 | 89.57 | 89.76 | 91.16 | 91.17 | 90.20 | 90.47 | −0.96 | −0.70 |
| mean noisy EER | 10.09 | 9.66 | 9.16 | 9.01 | 10.15 | 10.17 | +0.99 | +1.15 |
| calibration gap noisy | - | - | 0.11 | 0.42 | 0.62 | 0.21 | +0.51 | −0.21 |
| separability gap | - | - | 6.56 | 6.20 | 7.12 | 7.27 | +0.56 | +1.07 |

Per-set paired CIs (`ci_vs_B3a-*.txt`):

| set | merge Δ [CI] | ep8 Δ [CI] |
|---|---|---|
| clean | −0.52 [−0.82, −0.23] | −0.29 [−0.65, +0.06] |
| matched | −1.12 [−2.34, +0.03] | −0.25 [−1.51, +1.02] |
| heldout | −0.80 [−2.03, +0.36] | −1.16 [−2.48, +0.00] |
| echo | **−1.20 [−2.22, −0.21]** | −0.72 [−1.67, +0.27] |

Weighted contributions (0.3 / 0.35 / 0.35, unrounded deltas from `compare_ci_vs_B3a-*.json`):
- merge: −0.16 clean − 0.39 matched − 0.28 heldout = −0.83. **The loss is spread over all three WF1 sets.**
- final: −0.09 − 0.09 − 0.41 = −0.58. **Heldout carries 70 % of the final-epoch loss** (English heldout bona, 2.4 and 5.1).

**Other halves of the same sets:**

| readout | full-set WF1 B3a → B3b (registry `wf1@0.5`) | half_A WF1 B3a → B3b (computed from `scores/*.txt`) |
|---|---|---|
| merge | 93.05 → 92.36 (−0.69) | 93.04 → 92.50 (−0.54) |
| final | 92.78 → 92.33 (−0.45) | 92.61 → 92.29 (−0.32) |

The loss shows on both halves, so it is not a half-B artefact.

### 2.2 Per set (R = recall at 0.5; thr = p_spoof threshold; `metrics.json` half_B)

| set | metric | B3a-top5 | **B3b-top5** | B3a-ep8 | **B3b-ep8** |
|---|---|---|---|---|---|
| dev_online_clean | F1@.5 / F1@or / thr_or | 97.48 / 97.83 / 0.996 | **96.96** / 97.94 / 1.000 | 97.06 / 97.79 / 1.000 | **96.77** / 97.95 / 1.000 |
| | EER / ECE | 3.32 / 1.47 | 3.51 / 1.75 | 3.20 / 1.72 | 3.68 / 1.86 |
| | R_spoof / R_bona | 99.93 / 92.34 | 99.97 / **90.62** | 99.92 / 91.12 | 99.97 / **90.05** |
| sim_matched_v1 | F1@.5 / F1@or / thr_or | 92.12 / 92.25 / 0.995 | **91.00** / 91.95 / 0.975 | 92.03 / 92.49 / 0.998 | **91.78** / 92.08 / **0.019** |
| | EER / ECE | 7.89 / 4.51 | **9.00** / 5.13 | 7.74 / 4.55 | **8.84** / 4.82 |
| | R_spoof / R_bona | 99.38 / 79.03 | **98.07** / 80.31 | 99.69 / 77.75 | **98.57** / 80.82 |
| sim_heldout_v1 | F1@.5 / F1@or / thr_or | 90.20 / 90.30 / 0.971 | **89.40** / 89.69 / 0.822 | 90.32 / 90.69 / 0.993 | **89.16** / 89.28 / 0.177 |
| | EER / ECE | 10.42 / 5.54 | **11.30** / 5.93 | 10.29 / 5.57 | **11.49** / 6.04 |
| | R_spoof / R_bona | 99.06 / 74.94 | **98.32** / 75.19 | 99.00 / 75.45 | 98.69 / **73.40** |
| sim_echo_v1 | F1@.5 / F1@or / thr_or | 94.87 / 95.03 / 0.998 | **93.66** / 94.67 / 0.996 | 94.39 / 94.96 / 0.999 | 93.68 / 94.69 / 1.000 |
| | EER / ECE | 4.61 / 3.00 | 4.83 / 3.71 | 5.62 / 3.28 | 5.50 / 3.72 |
| | R_spoof / R_bona | 99.81 / 85.17 | 99.50 / **82.86** | 99.94 / 83.38 | 99.44 / 83.12 |

Reading:
- **The sampler traded spoof recall for noisy bona recall at worse ranking.** Matched R_spoof −1.31 / −1.12 and heldout −0.75 / −0.31, against matched R_bona +1.28 / +3.07. EER rose on matched (+1.10 / +1.11) and heldout (+0.88 / +1.20), so the bona gain is an operating-point shift, not separation.
- **Clean and echo bona fell in both readouts** (clean R_bona −1.72 / −1.07; echo −2.30 / −0.26).
- The operating point became set-dependent: B3b-ep8 per-set `thr_or` spans 0.019 (matched) to 1.000 (clean, echo); B3a-ep8 spans 0.993-1.000.
- Paired per clip, half_B, B3b vs B3a (fixed / broken / wrong in both; by language: fixed / broken):

  | set | class | merge | merge en / zh | ep8 | ep8 en / zh |
  |---|---|---|---|---|---|
  | clean | bona | 7 / **31** / 100 | **0 / 21** · 7 / 10 | 20 / 35 / 104 | **6 / 26** · 14 / 9 |
  | matched | bona | 16 / 11 / 66 | 5 / 9 · **11 / 2** | 24 / 12 / 63 | 6 / 10 · **18 / 2** |
  | heldout | bona | 16 / 15 / 82 | 5 / 6 · 11 / 9 | 12 / 20 / 84 | 2 / 11 · 10 / 9 |
  | echo | bona | 9 / 18 / 49 | 3 / 14 · 6 / 4 | 12 / 13 / 53 | 4 / 11 · 8 / 2 |
  | matched | spoof | 2 / **23** / 8 | 1 / 10 · 1 / 13 | 0 / **18** / 5 | 0 / 5 · 0 / 13 |
  | heldout | spoof | 2 / 14 / 13 | 1 / 6 · 1 / 8 | 5 / 10 / 11 | 3 / 3 · 2 / 7 |
  | echo | spoof | 1 / 6 / 2 | 1 / 2 · 0 / 4 | 0 / 8 / 1 | 0 / 2 · 0 / 6 |

  **Summed over the four sets, English bona: 13 fixed / 50 broken (merge), 18 / 58 (ep8). Chinese bona: 35 / 25 (merge), 50 / 22 (ep8).** The sampler helped the language it under-weighted and hurt the one it over-sampled.
- Logit correlation B3b vs B3a: 0.938-0.948 per set (merge), 0.928-0.952 (ep8). B3a vs B2f was 0.954-0.967 (B3a report).

### 2.3 Probes (`probe_report.json`; 187 bona + 813 spoof; one bona probe clip = 0.53 points)

| probe (bona unless stated) | B3a-top5 | **B3b-top5** | B3a-ep8 | **B3b-ep8** |
|---|---|---|---|---|
| level_flip_bona (mean of 8) / max | 0.80 / 2.67 | **1.74 / 6.95** | 2.07 / 7.49 | 2.27 / 8.56 |
| gain_up / gain_p6 / gain_p10_limit flip % | 2.67 / 1.07 / 2.67 | 4.81 / 1.07 / **6.95** | 4.81 / 2.14 / 7.49 | 5.35 / 1.60 / 8.56 |
| agc_dynaudnorm / loudnorm flip % | 0.00 / 0.00 | 1.07 / 0.00 | 0.00 / 2.14 | 1.60 / 1.07 |
| level_flip_spoof | 0.02 | 0.03 | 0.00 | 0.02 |
| probe_orig: bona pred spoof % | 12.30 | **13.90** | 11.76 | **13.37** |
| noise_only: bona-source / spoof-source pred spoof % | 100.00 / 100.00 | 99.47 / 99.88 | 100.00 / 100.00 | 99.47 / 100.00 |
| noise_only mean dP(spoof), bona-source | +0.876 | +0.853 | +0.882 | +0.855 |
| sil_only: bona-source pred spoof %, mean dP | 100.00, +0.877 | 100.00, +0.853 | 100.00, +0.882 | 100.00, +0.814 |
| trimmed / padded bona flip % | 2.14 / 3.21 | 1.60 / 3.74 | 3.21 / 2.67 | 3.74 / 4.28 |

Reading:
- The gain-shape probes are identical to `probe_orig` by construction under loudness norm (`gain_down`, `gain_m20`, `gain_m6`: 0.00 % flips in all four readouts).
- **B3b-top5 loses the merge's level robustness**: limiter flips 5 → 13 bona clips (2.67 → 6.95 %), and level_flip_bona more than doubles.
- `noise_only` / `sil_only` bona-source clips are still (almost) all spoof: one `noise_only` clip of 187 moved. More bona draws did not teach the model about non-speech input.

### 2.4 Slices that moved most, B3b vs B3a (full sim sets, `metrics.json → breakdowns`, 63 slice rows)

| readout | F1 up | EER down | R_bona up | R_spoof up | min slice R_spoof (B3a → B3b) |
|---|---|---|---|---|---|
| merge | **13 of 63** (2 ties) | 14 | 31 | **2** | 97.02 (heldout washing_machine) → **94.28 (matched snr 0)** |
| final | 18 of 63 (1 tie) | 20 | 30 | 1 | 96.69 → 96.36 (heldout washing_machine) |

Language rows (full sets; F1 / EER / R_spoof / R_bona):

| set / lang | B3a-top5 | **B3b-top5** | B3a-ep8 | **B3b-ep8** |
|---|---|---|---|---|
| matched en | 87.34 / 13.66 / 99.61 / 65.50 | 86.10 / 13.27 / **98.59** / 65.89 | 86.38 / 14.36 / 99.61 / 63.18 | 85.67 / 13.93 / 99.22 / 62.79 |
| matched zh | 94.65 / 5.56 / 99.19 / 86.79 | 94.11 / **6.07** / **97.92** / **89.63** | 94.40 / 5.71 / 99.39 / 85.37 | 94.86 / 6.17 / 98.38 / **90.24** |
| heldout en | 84.74 / 17.04 / 99.14 / 60.85 | **82.56 / 17.74** / 98.12 / **58.91** | 84.26 / 17.63 / 98.91 / 60.47 | **80.97 / 18.13** / 98.91 / **53.10** |
| heldout zh | 92.54 / 7.82 / 98.68 / 82.52 | 92.46 / 8.18 / 98.12 / 84.15 | 92.89 / 7.14 / 98.73 / 83.33 | 92.52 / 7.36 / 98.17 / 84.15 |
| echo en | 89.40 / 11.28 / 99.53 / 70.93 | 87.96 / 11.21 / 99.38 / **67.83** | 88.66 / 12.06 / 99.77 / 68.22 | 87.34 / 12.45 / 99.61 / 65.50 |
| echo zh | 97.19 / 2.44 / 99.85 / 91.87 | 96.81 / 2.59 / 99.54 / 91.87 | 97.18 / 2.46 / 99.90 / 91.67 | 97.01 / 2.23 / 99.54 / 92.48 |

Largest moves (F1, B3a → B3b):

| set / slice | n (bona) | merge F1 (Δ) | merge EER | merge R_bona / R_spoof |
|---|---|---|---|---|
| matched / rain | 671 (115) | 89.02 → 85.41 (−3.61) | 12.20 → 14.59 | 72.2 → 76.5 / 98.92 → **94.78** |
| matched / snr 0 | 711 (117) | 86.15 → 82.61 (−3.54) | 14.67 → **17.05** | 65.8 → 70.9 / 98.65 → **94.28** |
| heldout / door_wood_creaks | 361 (64) | 90.88 → 87.42 (−3.46) | 7.44 → 11.19 | 78.1 → 73.4 / 98.65 → 97.31 |
| matched / opus:wechat | 585 (104) | 89.04 → 86.15 (−2.88) | 12.28 → 14.49 | 72.1 → 72.1 / 98.96 → 96.67 |
| matched / opus:voov | 510 (89) | 94.58 → 91.84 (−2.73) | 6.93 → 6.66 | 84.3 → 82.0 / 99.76 → 98.34 |
| heldout / lang en | 1538 (258) | 84.74 → 82.56 (−2.18) | 17.04 → 17.74 | 60.9 → 58.9 / 99.14 → 98.12 |
| heldout / clock_tick | 383 (63) | 92.21 → 93.99 (+1.78) | 6.94 → 6.30 | 77.8 → 84.1 / 99.69 → 99.38 |
| heldout / door_wood_knock | 353 (54) | 90.68 → 91.84 (+1.17) | 7.47 → 8.48 | 77.8 → 79.6 / 98.66 → 99.00 |

| set / slice (final epoch) | n (bona) | ep8 F1 (Δ) | ep8 EER | ep8 R_bona |
|---|---|---|---|---|
| heldout / lang en | 1538 (258) | 84.26 → 80.97 (**−3.29**) | 17.63 → 18.13 | 60.5 → **53.1** |
| heldout / glass_breaking | 346 (64) | 92.29 → 89.78 (−2.50) | 8.34 → 11.32 | 79.7 → 76.6 |
| heldout / washing_machine | 380 (78) | 85.28 → 82.83 (−2.45) | 15.62 → 13.84 | 69.2 → 64.1 |
| heldout / codec gsm | 1346 (246) | 88.26 → 86.39 (−1.87) | 13.43 → 13.39 | 73.6 → 70.7 |
| matched / opus:telegram | 552 (111) | 91.25 → 93.53 (+2.28) | 8.25 → 8.14 | 78.4 → 84.7 |
| matched / opus:dingtalk | 589 (122) | 93.18 → 94.38 (+1.20) | 6.70 → 6.70 | 82.8 → 86.9 |

SNR 0 at the merge (full sets): R_bona rose (matched 65.81 → 70.94, heldout 65.31 → 68.71), but EER rose more (14.67 → 17.05, 15.08 → 18.92) and R_spoof fell (98.65 → 94.28, 98.50 → 96.54). **At SNR 0 the sampler moved the threshold, not the ranking.**

### 2.5 Lever response (one factor per row)

| run | change vs previous reference | half_B merge WF1 (Δ) | final WF1 (Δ) | merge oracle | final oracle | merge noisy EER | final noisy EER | merge clean | final clean | verdict (verdict.json) |
|---|---|---|---|---|---|---|---|---|---|---|
| B2f | (wave start; B2f report) | 91.98 | 92.14 | 92.59 | 92.82 | 10.09 | 9.66 | 97.60 | 97.69 | - |
| B3a | + `--consistency js --consistency_lambda 0.3 --consistency_pairs both`, `--batch_size 8` | 93.06 (+1.08) | 92.94 (+0.80) | 93.24 | 93.45 | 9.16 | 9.01 | 97.48 | 97.06 | merge positive / promote; final revisit (B3a report) |
| **B3b** | **+ `--sampler balanced --sampler_lang_balance --sampler_bona_frac 0.5 --ce_weights 0.3223 0.6777`** | **92.23 (−0.83)** | **92.36 (−0.58)** | 92.96 (−0.28) | 92.86 (−0.59) | 10.15 (+0.99) | 10.17 (+1.15) | 96.96 (−0.52) | 96.77 (−0.29) | merge **negative / drop**; final **negative / drop** |

B3b is the first lever of the wave below the band; the merge CI excludes 0. It lands just above the fixed comparator B2f on the merge (92.23 vs 91.98) and roughly level with it on the final epoch (92.36 vs 92.14), i.e. it erased most of B3a's +1.08.

## 3. Per-epoch behaviour, B3b vs B3a (same steps, same LR at each epoch end)

Sources: `metrics.jsonl` (EMA weights; 2,000 clean clips + `sim_matched_mini` + `sim_heldout_mini`, half A), `logs/train.log`, `STATUS.md`. **Caveats:** B3b's `CE` and `DevLoss` use the re-derived class weights (the dev loss is computed with the same weighted criterion, `train.py` `evaluate_dev(..., criterion)`), and B3b's training draws are 50 % bona, so CE, train acc and dev loss are not strictly comparable across the two runs. JS is unweighted and comparable.

| ep | LR end | B3b mini WF1 | B3a mini WF1 | Δ | B3b WF1@or | B3b EER noisy (B3a) | B3b R_bona cl/ma/he | B3a R_bona cl/ma/he | B3b R_spoof ma/he (B3a) | B3b thr_or ma/he | B3b CE / JS | B3a JS | B3b / B3a train acc | B3b / B3a dev loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.47e-6 | 89.68 | 90.17 | −0.49 | 91.17 | 12.23 (13.14) | 92.1/82.7/78.1 | 91.0/73.7/65.8 | 94.8/93.5 (98.4/98.3) | 0.030/0.027 | 0.2337 / 0.0417 | 0.0344 | 91.12 / 90.16 | 0.0950 / 0.0990 |
| 2 | 2.29e-6 | 90.53 | 91.64 | −1.11 | 91.95 | 12.14 (11.56) | 93.0/84.4/77.6 | 91.6/76.5/70.9 | 95.1/94.7 (99.0/98.3) | 0.006/0.007 | 0.1019 / 0.0246 | 0.0214 | 97.62 / 97.87 | 0.1059 / 0.1232 |
| 3 | 1.98e-6 | **91.76** | 91.39 | **+0.37** | 92.55 | **11.17** (11.49) | 91.0/83.2/76.0 | 91.6/76.5/66.8 | 97.0/97.1 (99.3/98.8) | 0.027/0.019 | 0.0681 / 0.0172 | 0.0151 | 98.57 / 98.63 | 0.1137 / 0.1514 |
| 4 | 1.58e-6 | 91.14 | 91.52 | −0.38 | 92.12 | 12.03 (10.80) | 87.9/84.4/73.5 | 89.9/76.5/69.9 | 96.7/97.1 (99.0/98.6) | 0.007/0.918 | 0.0478 / 0.0123 | 0.0113 | 98.99 / 99.01 | 0.1556 / 0.1657 |
| 5 | 1.18e-6 | 91.09 | 92.52 | −1.43 | 92.47 | 11.85 (11.16) | 91.0/81.6/75.5 | 91.6/81.0/71.9 | 96.6/96.5 (98.5/99.1) | 0.009/0.011 | 0.0357 / 0.0090 | 0.0080 | 99.24 / 99.32 | 0.1430 / 0.1472 |
| 6 | 8.23e-7 | 91.93 | **92.87** | −0.94 | 92.82 | 11.39 (10.09) | 89.3/80.4/73.5 | 91.9/82.7/71.9 | 98.8/97.6 (99.0/98.9) | 0.680/0.904 | 0.0276 / 0.0076 | 0.0065 | 99.43 / 99.47 | 0.1691 / 0.1356 |
| 7 | 5.84e-7 | **92.21** | 92.36 | −0.15 | 92.77 | 11.28 (10.40) | 89.9/82.7/73.0 | 91.3/80.4/71.4 | 98.1/98.5 (98.9/98.8) | 0.006/0.835 | 0.0194 / 0.0048 | 0.0051 | 99.63 / 99.59 | 0.1730 / 0.1584 |
| 8 | 5.00e-7 | 91.91 | 92.14 | −0.23 | 92.87 | 11.49 (10.28) | 89.9/81.0/69.9 | 89.9/78.8/71.9 | 98.5/98.8 (99.0/98.8) | 0.003/0.893 | 0.0179 / 0.0047 | 0.0038 | 99.67 / 99.71 | **0.1743** / 0.1722 |

Reading:
- **B3b is below B3a at 7 of 8 epochs on half A** (above only at epoch 3, +0.37). Its best epoch is 7 (92.21), 0.66 below B3a's best (92.87 at epoch 6). Noisy mini EER is worse than B3a at 6 of 8 epochs (better at epochs 1 and 3).
- **The early bona shift eroded as training went on.** Mini heldout R_bona starts 12.3 points above B3a (78.1 vs 65.8 at epoch 1) and ends 2.0 below (69.9 vs 71.9 at epoch 8), falling almost monotonically (78.1, 77.6, 76.0, 73.5, 75.5, 73.5, 73.0, 69.9). This is what a model memorising its repeated bona draws looks like: the operating-point push is largest before the repeated clips are fitted.
- **The noisy operating point leaned bona throughout.** Matched-mini `thr_or` is ≤ 0.030 at 7 of 8 epochs (0.680 at epoch 6) and mini R_spoof is 93.5-98.8 against B3a's 98.3-99.3. B3a's mini `thr_or` sat at 0.86-0.99 (B3a report). The CE re-derivation kept the class prior on average, but not per condition: on noisy inputs B3b calls spoof too rarely, while on clean (`thr_or` 0.983-1.000) it calls spoof too often.
- **JS is higher than B3a's at 7 of 8 epochs** (0.0417 → 0.0047 vs 0.0344 → 0.0038): the two views disagree more under the balanced draw.
- Dev loss rises from epoch 1 in both runs: B3b 0.0950 → 0.1743 (+83 %), B3a 0.0990 → 0.1722 (+74 %). The minimum is at epoch 1 in both (`best_by_dev_loss`).
- **Final vs merge:** on half B the final epoch beats the merge (92.36 vs 92.23, +0.13); on half A the merge wins (92.50 vs 92.29). The merge includes epoch 4 (lowest mini clean F1 of the run, 96.14; clean R_bona 87.9) and epoch 3 (from the early, strongly bona-leaning regime; `thr_or` 0.027 / 0.019). These are the likeliest drags, but no merged epoch was evaluated alone on half B.

## 4. What went OK (evidence)

1. **Clean, decisive experiment.** One coupled factor, confirmed by three header lines matching the design's `expect` regexes. Realised shares were bona 0.499-0.503 / en 0.497-0.503, with the same 75,792 steps and cosine schedule as B3a. The codec counts prove two views per clip in every epoch. There was no traceback, codec failure or pruning, and the git diff touches no trainer code. The merge CI [−1.46, −0.25] excludes 0, so one 8 h run closes the sampler branch of the plan (`plan_snapshot.md` "Sampler") with a clear negative.
2. **The mechanism did what a sampler does, for Chinese.** Chinese bona drawn 1.66× per clip improved:
   - half_B noisy zh bona false alarms 89 → 78 (merge) and 90 → 73 (ep8);
   - zh matched R_bona 84.7 → 88.2 (merge) and 84.0 → 90.1 (ep8);
   - Chinese bona fixed / broken over the four sets: 35 / 25 (merge), 50 / 22 (ep8).
3. **The saturated noisy-bona share fell slightly on the merge.** Noisy bona at p ≥ 0.99 went 17.6 → 15.7 %, and the saturated share of the false alarms 77 → 71 %. The final epoch did not move: 19.8 → 19.4 %, 85 → 85 % (5.4).
4. **It answers the open question of the B2 wave.** The B2 summary's follow-up C1 (class-balanced sampler, because English bona recall never moved) is now tested: exposure is not the English bottleneck (5.1). That removes a candidate from the queue and redirects the English problem to generalisation and regularisation levers.

## 5. What went wrong (evidence)

### 5.1 English bona got worse, the opposite of the hypothesis

Half_B English bona per set (computed from `scores/*.txt` + meta `lang`; R_bona, false alarms, share at p ≥ 0.99):

| set (en bona n) | B3a-top5 | **B3b-top5** | B3a-ep8 | **B3b-ep8** |
|---|---|---|---|---|
| clean (479) | 85.6, FA 69, 13.4 % | **81.2, FA 90**, 16.7 % | 81.8, FA 87, 17.5 % | **77.7, FA 107**, 19.2 % |
| matched (129) | 67.4, FA 42, 26.4 % | **64.3**, FA 46, 25.6 % | 65.1, FA 45, 31.8 % | **62.0**, FA 49, 34.1 % |
| heldout (129) | 62.0, FA 49, 28.7 % | 61.2, FA 50, 31.0 % | 62.8, FA 48, 31.0 % | **55.8, FA 57, 38.8 %** |
| echo (129) | 73.6, FA 34, 23.3 % | **65.1, FA 45**, 29.5 % | 69.0, FA 40, 28.7 % | 63.6, FA 47, 33.3 % |

Chinese bona over the same sets (R_bona): clean 95.9 → 95.5 / 96.0 → 96.5; matched 84.7 → 88.2 / 84.0 → 90.1; heldout 81.3 → 82.1 / 81.7 → 82.1; echo 90.8 → 91.6 / 90.5 → 92.7.

- **English noisy bona false alarms rose 91 → 96 (merge) and 93 → 106 (ep8)**, from 258 noisy English bona; Chinese ones fell 89 → 78 / 90 → 73. English is now 96 of 174 noisy false alarms (55 %) on the merge and 106 of 179 (59 %) on the final epoch, against 91 of 180 (51 %) for B3a.
- **The clean set shows it most cleanly:** 0 English bona fixed and 21 broken on the merge (2.2), and English clean R_bona −4.4 / −4.1 points.
- Each of the 2,922 English bona training clips (`config.yaml` `train_info`: 1,500 offline + 1,422 online) was drawn about 6.5× per epoch, in two views. Train acc reached 99.67 %. The decline of mini heldout R_bona over the epochs (section 3) together with the clean-English collapse points to **memorisation of a small English bona pool, not under-exposure**. The English gap is a generalisation problem; repetition makes it worse.

### 5.2 Spoof under-exposure cost more than the bona gain

- Half_B spoof misses on the noisy sets (matched + heldout) rose 25 → 58 on the merge (en 5 → 19, zh 20 → 39) and 21 → 44 on the final epoch (en 5 → 10, zh 16 → 34). Chinese spoof was drawn 0.40× per clip.
- Spoof with p < 0.5 per set (clean / matched / heldout / echo): B3a-top5 4 / 10 / 15 / 3 → B3b-top5 2 / 31 / 27 / 8; B3a-ep8 5 / 5 / 16 / 1 → B3b-ep8 2 / 23 / 21 / 9.
- Net on the noisy WF1 sets at the merge: 6 fewer bona false alarms (180 → 174) for 33 more spoof misses.
- R_spoof rose in only 2 of 63 slice rows on the merge, and the minimum slice R_spoof fell from 97.02 to 94.28 (matched snr 0). Matched rain fell to 94.78.

### 5.3 Oracle-gap decomposition (global threshold vs set-specific)

Method: one shared logit threshold swept over the half_B `scores/*.logits` (step 0.05, −6 to +12), as in the B3a report; per-set oracle from `metrics.json` `wf1_oracle`.

| readout | WF1@0.5 | best global (logit / p) | WF1 at best global | per-set oracle WF1 | gap | recovered by one global shift | set-specific remainder |
|---|---|---|---|---|---|---|---|
| B3a-top5 | 93.06 | −0.15 / 0.463 | 93.08 | 93.24 | 0.18 | 0.02 | 0.16 |
| B3a-ep8 | 92.94 | +4.95 / 0.993 | 93.10 | 93.45 | 0.51 | 0.16 | 0.35 |
| **B3b-top5** | 92.23 | **+2.60 / 0.931** | 92.58 | 92.96 | 0.73 | 0.35 | **0.38** |
| **B3b-ep8** | 92.36 | **−1.70 / 0.154** | 92.41 | 92.86 | 0.50 | 0.05 | **0.45** |

- **The re-derived CE weights did not keep the merge's operating point.** B3a's merge was optimal at 0.5 (logit −0.15). B3b's merge wants +2.60 (fewer spoof calls, driven by clean, where `thr_or` is 1.000), and B3b-ep8 wants −1.70 (more spoof calls, driven by matched `thr_or` 0.019).
- **The set-specific remainder grew** (0.16 → 0.38 merge, 0.35 → 0.45 final). The sampler pushed clean and noisy conditions in opposite directions, so no single threshold fits both.
- Even at its best global threshold, B3b-top5 (92.58) stays below B3a-top5 at 0.5 (93.06). The loss is ranking as well as calibration: noisy EER is +0.99 / +1.15.

### 5.4 The saturated bona tail

Share of half_B bona with p_spoof ≥ 0.5 / ≥ 0.9 / ≥ 0.99 / ≥ 0.999 (computed):

| set (n bona) | B3a-top5 | **B3b-top5** | B3a-ep8 | **B3b-ep8** |
|---|---|---|---|---|
| clean (1,397) | 7.7 / 7.6 / 6.4 / 4.6 | 9.4 / 8.8 / **7.9** / 6.2 | 8.9 / 8.9 / 8.3 / 6.1 | 9.9 / 9.5 / 8.4 / 6.7 |
| matched (391) | 21.0 / 21.0 / 16.4 / 8.2 | 19.7 / 17.9 / 13.6 / 9.7 | 22.3 / 22.3 / 19.2 / 12.0 | 19.2 / 18.9 / 16.6 / 11.5 |
| heldout (391) | 25.1 / 25.1 / 18.9 / 10.2 | 24.8 / 24.0 / 17.9 / 11.0 | 24.6 / 24.3 / 20.5 / 13.0 | 26.6 / 25.8 / **22.3** / 13.6 |
| echo (391) | 14.8 / 14.8 / 12.3 / 5.4 | 17.1 / 16.9 / 13.0 / 7.2 | 16.6 / 16.4 / 15.3 / 9.5 | 16.9 / 16.1 / 14.3 / 10.0 |
| noisy bona (782): FA % / ≥ 0.99 % / saturated share of FA | 23.0 / 17.6 / 77 % | 22.3 / 15.7 / 71 % | 23.4 / 19.8 / 85 % | 22.9 / 19.4 / 85 % |

The saturated core is unchanged in size (15.7-19.4 % of noisy bona at ≥ 0.99 across all four readouts). The ≥ 0.999 share grew in every set on both readouts except matched ep8. The English share of that core grew (5.1).

### 5.5 Worst slices for bona recall, B3b-top5 (full sets, `merge_top5.log`)

| slice | R_bona | R_spoof | EER |
|---|---|---|---|
| heldout en | **58.91** | 98.12 | 17.74 |
| matched en | 65.89 | 98.59 | 13.27 |
| heldout washing_machine | 66.67 | 95.36 | 13.84 |
| heldout snr 0 | 68.71 | 96.54 | 18.92 |
| heldout wind | 69.74 | 98.03 | 14.29 |
| heldout g726 | 70.73 | 98.74 | 12.78 |
| matched snr 0 | 70.94 | **94.28** | 17.05 |
| heldout clock_alarm | 71.43 | 99.64 | 10.04 |
| matched opus:qq | 71.72 | 97.25 | 11.05 |
| matched opus:wechat | 72.12 | 96.67 | 14.49 |

English and SNR 0 remain the two worst axes, as in B3a (B3a report 5.4). At SNR 0 the EER rose from 14.67 / 15.08 (B3a-top5) to 17.05 / 18.92 (2.4).

### 5.6 Overfitting signs

- Train acc reached 99.67 %, while dev loss rose 83 % from epoch 1 and mini heldout R_bona fell from 78.1 to 69.9 over the run.
- The English bona recall on every set fell from merge to final (clean 81.2 → 77.7, heldout 61.2 → 55.8), while Chinese held.
- `gain_p10_limit` bona flips: 2.67 → 6.95 % (merge).
- The run repeats a 2,922-clip pool about 52 times over 8 epochs (6.48 × 8, computed), in two views each. That is the most likely driver of the English collapse; more of the same exposure would not fix it.

## 6. Verdict on the pre-registered rule

From `exploop/B3/iter02/design.result.json`:

- **Falsifier (wave rule):** "Wave rule against B3a-top5 = 93.06 on half_B: Promote / positive: delta merge > +0.5, the paired-bootstrap CI excluding 0, and a clean F1 drop of at most 0.5. Null: delta within +-0.5. Negative: delta below -0.5."
- **Mechanism-level refutations** (same file): "Exposure is not the cause (null on the mechanism): delta within +-0.5 AND half_B English R_bona within +-3 on both matched and heldout. The prior derivation failed: the best global logit threshold moves beyond +-1, or the clean F1 drop exceeds 0.5. Spoof under-exposure costs more than the English gain: R_spoof falls by more than 0.5 on matched or heldout (zh spoof is drawn only 0.40x per clip per epoch). The repeated English bona draws over-fit: epoch-8 dev loss above B3a's 0.1722, together with English R_bona not up."

`exploop/B3/iter02/verdict.json` (computed by the supervisor; quoted, not recomputed):

| readout | run | base | delta | CI | clean_drop | new_best | decision | verdict |
|---|---|---|---|---|---|---|---|---|
| merge (objective) | 92.22948759382432 | 93.05618353351335 | −0.8266959396890314 | [−1.4572939566645768, −0.24906615943764313] | 0.5216040573340024 | false | **drop** | **negative** |
| final | 92.36138188102834 | 92.94058874557216 | −0.5792068645438206 | [−1.207759840001305, 0.019473884712772296] | 0.29026939722122336 | false | **drop** | **negative** |

**The negative branch fired on both readouts.**

Mechanism-level refutations (my reading of the evidence against the design's own conditions; the verdict above is the supervisor's):

| refutation | evidence | fired? |
|---|---|---|
| exposure is not the cause | Δ is −0.83, outside ±0.5, so the stated condition does not apply literally. English R_bona moved −3.1 (matched) / −0.8 (heldout) at the merge and −3.1 / −7.0 at ep8: **down, not up** | stronger than the null: exposure made English worse |
| prior derivation failed | best global logit +2.60 (merge) / −1.70 (ep8), both beyond ±1; clean drop 0.52 on the merge (> 0.5) | **yes** |
| spoof under-exposure | R_spoof −1.31 matched / −0.75 heldout (merge); −1.12 / −0.31 (ep8) | **yes** (both sets on the merge; matched on the final) |
| repeated English draws over-fit | epoch-8 dev loss 0.1743 vs 0.1722, with English R_bona down. Caveat: dev loss uses different CE weights in the two runs (section 3) | **yes** on the stated terms; the English R_bona fall is the stronger evidence |

The predicted signature, item by item (merge vs B3a-top5; ep8 in brackets):

| predicted | observed | appeared? |
|---|---|---|
| English R_bona up ≥ 3 on matched and heldout (67.4 / 62.0) | 64.3 / 61.2 (ep8 62.0 / 55.8 vs 65.1 / 62.8) | **no**, down |
| English noisy bona FA below 91 of 180 | 96 of 174 (ep8 106 of 179) | **no** |
| English share at p ≥ 0.99 on matched below 26.4 % | 25.6 % (33 vs 34 clips) (ep8 34.1 % vs 31.8 %) | marginal on the merge (one clip); no on ep8 |
| full-set English EER below 13.66 matched / 17.04 heldout | 13.27 / 17.74 | matched yes, heldout no |
| zh R_bona flat (84.7 / 81.3) | 88.2 / 82.1 | no; moved up on matched |
| clean F1 within 0.5 of 97.48 | 96.96 (−0.52) | **no** (just outside) |
| R_spoof within ~0.5 of 99.38 / 99.06 | 98.07 / 98.32 | **no** |
| best global logit near 0, not beyond ±1 | +2.60 | **no** |
| sampler shares near bona 0.5 / en 0.5 | 0.499-0.503 / 0.497-0.503 | yes |
| merge half_B WF1 above 93.06 | 92.23 | **no** |

Only the telemetry item appeared cleanly. The core of the hypothesis, that the en-zh gap is an exposure problem, is refuted, and refuted in the opposite direction: more English bona draws made English bona worse.

## 7. Decision and next-experiment proposals

**Decision.** Per `verdict.json`, **B3b is not a new best** (`new_best false` on both readouts, `decision drop`, `verdict negative`). The reference stays **B3a** (merge 93.06, final 92.94; registry `B3a-cjs03-screen` / `B3a-cjs03-ep8`). B3b's registry rows are `B3b-sampbal-screen` (CI −1.46 / −0.25) and `B3b-sampbal-ep8` (−1.21 / 0.02). Every proposal below is one factor (or one coupled group) against **B3a's flags**: B2f + `--consistency js --consistency_lambda 0.3 --consistency_pairs both --batch_size 8`, sampler `shuffle`, `--ce_weights 0.1 0.9`. Each keeps the ~8 h runtime. Every falsifier uses the wave rule against B3a-top5 = 93.06 on half_B:
- promote: Δ merge > +0.5 with the CI excluding 0 and a clean drop of at most 0.5;
- null: Δ within ±0.5;
- negative: Δ below −0.5.

**The largest remaining error** is still confident noisy bona false alarms (B3a merge: 180 of 782 half_B noisy bona, 77 % of them at p ≥ 0.99), with English carrying half (91 of 180). B3b shows what that error is not:
- not exposure: English got worse with 6.5× draws;
- not the operating point: B3a's merge is optimal at 0.5, and a global shift recovers 0.02.

What remains is ranking and generalisation. **Do not pursue more sampler variants** (class-only balance, other `bona_frac`). Every sampler setting changes the per-condition operating point and spoof exposure together (5.2, 5.3).

**Ranked proposals (configuration first, seed repeat last):**

1. **Consistency dose: `--consistency_lambda 1.0`** (`wave.json` lever_queue `consistency_dose`, "if B3a's consistency helps"; it helped).
   - Targets: the saturated noisy-bona core and noisy separability. Consistency is the only lever of the wave that moved ranking (B3a: noisy EER 10.09 → 9.16, heldout EER −1.61).
   - Hypothesis: at λ 0.3 the JS term is 2.5-3.9 % of the loss (B3a report), and JS fell to 0.0038 by epoch 8. Tripling the weight enforces more view invariance and should reach the saturated core, which B3a's movable-band fix left untouched.
   - Predicted signature:
     - merge noisy EER below 9.16 and heldout R_bona above 74.94, with R_spoof within 0.5 of 99.38 / 99.06;
     - noisy bona share at p ≥ 0.99 below 17.6 %;
     - JS at epoch 8 below 0.0038;
     - clean F1 within 0.5 of 97.48, and the best global logit within ±1.
   - Falsifier: Δ merge within ±0.5 and noisy EER within ±0.3 of 9.16 means the B3a gain came from the paired views and the doubled steps, not the divergence weight, so keep 0.3. A clean drop > 0.5 means over-regularised.
2. **SpecAugment: `--specaug`** (lever_queue `specaug`).
   - Targets: memorisation and over-fitting.
     - Dev loss rises from epoch 1 in both B3a (0.0990 → 0.1722) and B3b.
     - B3a's mini WF1 falls after epoch 6.
     - B3b showed that English bona errors grow when the model can memorise the small English pool (English bona 13 fixed / 50 broken on the merge, 5.1).
   - Hypothesis: masking time / frequency bands blocks clip-specific memorisation without touching the class prior or the draw. English bona (a small, repeatedly seen pool even under shuffle) should then generalise better, and the final epoch should stop losing clean bona.
   - Predicted signature:
     - epoch-8 dev loss below 0.1722 (same CE weights as B3a, so comparable);
     - mini WF1 not falling over epochs 6-8;
     - half_B English R_bona not below B3a merge's 67.4 / 62.0, and English noisy FA below 91;
     - final-epoch clean F1 at least 97.06;
     - best global logit within ±1.
   - Falsifier: Δ merge within ±0.5 and epoch-8 dev loss ≥ 0.17 means masking does not regularise this backbone. R_spoof down > 0.5 on matched or heldout means masking removes the spoof cue.
3. **SNR range 0-20 dB: `--aug_snr_db 0 20`** (lever_queue `snr_0_20`).
   - Targets: SNR 0, the worst SNR slice on both sets. B3a-top5 full-set R_bona is 65.81 matched / 65.31 heldout and EER 14.67 / 15.08, against R_bona 85.83 / 79.86 at 20 dB. Training never sees 0 dB (5..20 uniform).
   - B3b showed that SNR-0 bona recall can be bought by the threshold only at a cost: R_bona +5.1 / +3.4, but EER +2.4 / +3.8 and R_spoof down to 94.28 / 96.54. A ranking fix is needed.
   - Hypothesis: SNR-0 false alarms are out-of-range inputs; covering 0-5 dB in training separates them.
   - Predicted signature:
     - full-set SNR-0 EER below 14.67 / 15.08;
     - SNR-0 R_bona up ≥ 3 on both sets at R_spoof within 0.5 of 98.65 / 98.50;
     - 15-20 dB slices within 2 R_bona of 82.57 / 85.83 (matched) and 74.17 / 79.86 (heldout);
     - clean flat.
   - Falsifier: Δ merge within ±0.5 and SNR-0 R_bona within ±3 means SNR range is not the limit. A drop of > 2 R_bona at 15-20 dB means the uniform 0-20 draw diluted the high-SNR mass.
4. **Codec zoo: `--aug_codec_profiles random`** (lever_queue `codec_random`).
   - Targets: the reserved heldout codecs. B3a-top5 full-set R_bona: g726 71.14 and gsm 72.76, against g722 81.01; EER 10.74 / 13.91. The matched opus:qq / opus:wechat slices are 68.69 / 72.12.
   - Hypothesis: randomised codec parameters widen the channel distribution, which generalises to g726 / gsm without seeing them.
   - Predicted signature: heldout g726 and gsm R_bona up ≥ 3 with EER down, R_spoof within 0.5; heldout F1 up; matched and clean flat.
   - Falsifier: Δ merge within ±0.5 and g726 / gsm R_bona within ±3 means codec diversity is not the limit.
5. **Seed repeat of B3a: `--seed 1235`** (last).
   - Purpose: B3a's +1.08 is the only above-band gain of the wave and rests on one seed. B3b's −0.83 shows the size of swing one lever can cause, but not the seed band.
   - Predicted signature: merge within ±0.5 of 93.06, with the heldout gain over B2f (> +1.5) reproduced.
   - Falsifier: a repeat merge below 92.48 (B2f-top5 91.98 + 0.5) means B3a's gain is inside seed noise.

**Recommended full run if the wave stopped now.** Unchanged from the B3a report:
- Flags: B3a's exact `config.yaml` command (`scripts/run_B3a.sh`), at 2-3 seeds (1234, 1235, 1236). Not B3b's sampler flags.
- Readout to submit: the top-5 merge. It is the wave objective and B3a's best readout: 93.06 vs 92.94 on half B, 93.04 vs 92.61 on half A.
- Pre-launch checks: `bash -n scripts/run_B3a.sh`, `which ffmpeg`, `pyloudnorm` importable, an empty `pgrep -af "b3wave.train|main_train"`, an idle GPU.
- Post-launch checks:
  - `Sampler: shuffle` and `CE weights: spoof=0.1 bonafide=0.9` (so the B3b flags did not leak in);
  - `Consistency: js lambda_js=0.3 … found': 37125 … 16 clips per step (8 pairs)`;
  - `Warmup-cosine: 3790 warmup / 75792 steps`;
  - epoch-1 `CE≈0.39 JS≈0.034`, mini WF1 ≈ 90.2;
  - codec counts summing to 2 × 75,785.
- Every evaluation needs `--loudness_norm --loudness_norm_lufs -23`.
