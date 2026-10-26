# Experiment exploration B3-05: B3e (`B3e_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_snr0to20_s1234`) - training SNR range 0-20 dB (`--aug_snr_db 0 20`) on top of B3a

> Written 2026-10-12 by the exploop B3 analysis agent (ANALYZE step, wave B3, iteration 5 of 5, attempt a0).
>
> **Sources.** Every number below is either copied from a file read in this session or computed in this session from the half_B (and, where stated, half_A) rows of the local `scores/*.txt` / `*.logits` files, joined with the local sets' `protocol.txt` / `meta.csv`. "Computed" marks the second kind. Nothing is quoted from memory. The files:
> - run config and logs: `config.yaml` (B3e and B3a; `args` diff), `git.diff` (B3e; file headers only), `logs/train.log` (both runs), `metrics.jsonl` (both runs), `top5_by_wf1/{select_topk.log,merge_list.txt}` (both runs), the `ckpt/` listing with timestamps;
> - local evals: `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,compare_vs_B3a-*.txt,ci_vs_B3a-*.txt,compare_ci_vs_B3a-*.json,compare_ci.json}`, `local_eval/merge_top5/compare_all_readouts.txt`, `local_eval/merge_top5.log` (breakdown tables); the same `metrics.json` / `probe_report.json` / `scores/` files of B3a (`merge_top5`, `epoch_8`); the `half_B.wf1` and `full.wf1` blocks of the B2f, B3b, B3c and B3d `metrics.json` files;
> - stdout log: `SLS_setup/B3e_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_snr0to20_s1234_20261011170024.log` (header lines and grep counts only);
> - loop files: `exploop/B3/iter05/{verdict.json,design.result.json}`, `exploop/B3/{wave.json,plan_snapshot.md,user_notes.md}`;
> - registry and code: `docs/experiment_log.csv` (B2f and B3 rows), `SLS_setup/b3wave/train.py` (flag help texts, `train_epoch` docstring), `SLS_setup/b3wave/data.py` (`__getitem__`, the clean-view path), `SLS_setup/utils/local_eval.py` (half filter, `NOISY_ROLES`), `SLS_setup/utils/metrics.py` (`wf1_local`, `CLEAN_WEIGHT = 0.3`, spoof if `p >= thr`), `configs/local_eval_sets.yaml`, `data/dev_noisy_sim/{dev_online_clean,matched_v1,heldout_v1,echo_v1}/{protocol.txt,meta.csv}`.
>
> Progress-split scores were not read: `model_merging_top5/scores.txt` was listed by `ls` and never opened.
>
> **Previous reports.**
> - `docs/experiments-exploration/b3-experiments/20261011_163014_B3d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_codecrand_s1234.md` (iteration 4; "B3d report"). Its §7 ranked this lever first.
> - `.../20261011_075030_B3c_..._cjs10_s1234.md` (iteration 3), `.../20261010_231745_B3b_..._sampbal_s1234.md` (iteration 2), `.../20261010_143058_B3a_..._cjs03_s1234.md` (the reference run).
> - `docs/experiments-exploration/20261009_175355_B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234.md` (template, fixed comparator) and `docs/experiments-exploration/20261008_232331_B2_wave_summary_and_full_run_recommendation.md`.
>
> **Readouts.**
> - `-top5` is the weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5/`). It is the wave objective.
> - `-ep8` is the final-epoch EMA checkpoint alone (`local_eval/epoch_8/`).
> - half_B is the half of each sim set that is never used for selection: n = 1,994 per set, 391 bona (`metrics.json`). Clean is the full 7,465-clip `dev_online_clean` (1,397 bona), which has no `half` column (`local_eval.py` keeps every row of a set without one).
> - WF1 = 0.3 clean + 0.35 matched + 0.35 heldout, macro-F1 at p_spoof 0.5 (`metrics.py`, `NOISY_ROLES = ("matched", "heldout")`). Echo is reported but is not in WF1.
> - Threshold 0.5 on the score files reproduces the four half_B WF1 values of the `metrics.json` files exactly: B3a 93.06 / 92.94, B3e 91.99 / 92.36 (top5 / ep8; computed).
> - Every evaluation ran with loudness normalisation: every `metrics.json` has `"loudness_norm": -23.0`, and `merge_top5.log` line 2 is `[loudness_norm] loudness_norm target=-23 LUFS`.

Run directories (under `SLS_setup/exp/`):
- **B3e** `B3e_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_snr0to20_s1234_epoch8_bs8_20261011170031` (this report)
- **B3a** `B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_epoch8_bs8_20261009214846` (reference = current best: merge 93.06, final 92.94)
- **B3b** `B3b_..._cjs03_sampbal_s1234_epoch8_bs8_20261010145615`, **B3c** `B3c_..._cjs10_s1234_epoch8_bs8_20261010234200`, **B3d** `B3d_..._cjs03_codecrand_s1234_epoch8_bs8_20261011081946` (iterations 2-4, all dropped)
- **B2f** `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623` (fixed comparator: 91.98 / 92.14)

## 1. Setup

**B3e = B3a with one flag added: `--aug_snr_db 0 20`** (default `5 20`; `design.result.json` `change`). It sits in the SNR coupled group. `--aug_snr_dist` stays `uniform`. Everything else is identical, including `--consistency js --consistency_lambda 0.3 --consistency_pairs both --batch_size 8` and `--aug_codec_profiles fixed`.

**What changed in training** (the design's `derivation`, computed from the header's stage list and `p_apply=0.8`):
- 0.600 of the training views get a noise stage (6 of the 8 stages draw an SNR; echo and reverb do not).
- 5..20 dB (B3a): 0.000 of noise draws in 0-5 dB, 0.333 in 15-20 dB, mean SNR 12.5 dB.
- 0..20 dB (B3e): 0.250 of noise draws (0.150 of views) in 0-5 dB, 0.250 in 15-20 dB, mean SNR 10.0 dB.
- The local sim sets draw SNR from `SNR_GRID = (0, 5, 10, 15, 20)` (the derivation cites `scripts/build_dev_noisy_sim.py` line 55). The `breakdowns.snr_db` keys of `metrics.json` are `0, 5, 10, 15, 20, (none)`. So 0 dB was in the eval support and outside B3a's training support.

Flag diff (`config.yaml` `args` of both runs, `difflib` output; nothing else differs):

| flag / field | B3a | **B3e** | note |
|---|---|---|---|
| `--aug_snr_db` | absent (`args.aug_snr_db: [5.0, 20.0]`) | **`0 20`** (`[0.0, 20.0]`) | the lever |
| `rtc_augmentation` field | `... \| snr=5..20 dB \| ...` | `... \| snr=0..20 dB \| ...` | the augmenter really changed |
| `--track` | `..._cjs03_s1234` | `..._cjs03_snr0to20_s1234` | name only |
| git | `df3dff2`, `git_dirty: false` | `df3dff2`, `git_dirty: true` | `git.diff` touches only `.claude/skills/post-train-eval/SKILL.md` and `docs/experiment_log.csv`; no trainer code |
| everything else | identical | identical | `--consistency js --consistency_lambda 0.3 --consistency_pairs both`, `--batch_size 8` (16 clips/step), 8 epochs, `--lr 2.5e-6 --lr_min 5e-7` warmup-cosine, `--ce_weights` default 0.1 0.9, sampler shuffle, RTC aug + MUSAN, `aug_max_stages 1`, `aug_codec_profiles fixed`, `--loudness_norm -23`, `--keep_topk_by_wf1 7 --keep_epochs 8`, seed 1234 |

**Header lines proving the change took effect** (stdout log). They match all six `expect` regexes of the design:
- `SNR: 0..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: fixed`
- `RTC augmentation: p_apply=0.8 one stage/clip, equal chance 1/8 | snr=0..20 dB | stages=[coffee, footsteps, keyboard, musan, office, rain, echo, reverb] | ... | then codec p=1.0 [qq, wechat, zoom, dingtalk, lark, voov, telegram]`
- `Consistency: js lambda_js=0.3 lambda_emb=0.3 pairs=both clean_view=False | pairs found {'rows': 38660, 'blank_online': 1535, 'found': 37125} dropped_without_twin=0 | 16 clips per step (8 pairs)`
- `Sampler: shuffle`, `CE weights: spoof=0.1 bonafide=0.9`, `SpecAugment: off`
- Unchanged: `Warmup-cosine: 3790 warmup / 75792 steps, floor 5.0e-07`, `LLRD: 1`, `Loudness norm: loudness_norm target=-23 LUFS`, `Train trials: 75785`, `Parameters: 320268933`.

**Telemetry** (`metrics.jsonl` `b3.aug_counts`, computed). SNR bins are not counted on this path (design `eval_note`), so the SNR change is proven by the header only.
- The codec counts sum to 151,570 = 2 × 75,785 in every epoch (two views per clip).
- Stage counts are 0.7992-0.8003 of the views (`p_apply=0.8`). The noise stages (all but echo and reverb) are 0.5996-0.6015 of the views, against the derivation's 0.600.
- The stdout log has 0 lines matching `Traceback`, `out of memory`, `failed (`, `Error` or `WARNING`. The one `Warning:` line is the HF Hub unauthenticated-request notice.

**Run time.**
- `config.yaml` created 2026-10-11 17:00:42; epoch 8 done 00:43:53; last local eval 00:44:16. That is **7 h 43 min** (registry `gpu_h` 7.73).
- Epoch-end to epoch-end spacing is 57.7-58.5 min (`train.log`, computed). The wider SNR range costs no time.
- Post-train file times (`ls`): `top5_by_wf1/merge_list.txt` 00:44:52, `merge_top5/metrics.json` 00:49:53, `epoch_8/metrics.json` 00:55:27, `model_merging_top5/submission.zip` 01:05:55. `verdict.json` was written at 01:06:55.

**Checkpoints.**
- Nothing was pruned: the stdout log has no `Pruned` line, and all 8 epoch files are present at 1,281,235,691 bytes each, plus `last_state.pt`.
- Symlinks: `best_by_dev_loss` → epoch 1 (0.105079), `best_by_noisy_eer` → epoch 7, `best_by_wf1` → epoch 5.

**Merged top-5 by half-A mini WF1 = epochs 5, 7, 6, 8, 4** (92.32, 92.13, 91.90, 91.89, 91.65; mean 91.98; `select_topk.log`).
- Excluded: 3 (91.60), 2 (91.37), 1 (90.21).
- B3a's merge was 6, 5, 7, 8, 2 (mean 92.31; B3a `select_topk.log`).

## 2. Results vs references (half_B)

Sources: `compare_all_readouts.txt`, `compare_vs_B3a-*.txt`, `ci_vs_B3a-*.txt` / `compare_ci_vs_B3a-*.json` (paired bootstrap n = 2,000, half B, threshold 0.5), and the `half_B.wf1` block of each run's `metrics.json`.

### 2.1 Wave iterations and fixed comparators, both readouts

| half_B | B2f-top5 | B2f-ep8 | B3a-top5 | B3a-ep8 | B3b-top5 | B3b-ep8 | B3c-top5 | B3c-ep8 | B3d-top5 | B3d-ep8 | **B3e-top5** | **B3e-ep8** | B3e-top5 − B3a-top5 [95 % CI] | B3e-ep8 − B3a-ep8 [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **WF1@0.5** | 91.98 | 92.14 | 93.06 | 92.94 | 92.23 | 92.36 | 92.29 | 92.44 | 90.86 | 90.72 | **91.99** | **92.36** | **−1.06 [−1.65, −0.49]**, p(not better) 1.000 | **−0.58 [−1.13, −0.05]**, p 0.984 |
| WF1@oracle | 92.59 | 92.82 | 93.24 | 93.45 | 92.96 | 92.86 | 92.63 | 92.73 | 92.00 | 91.84 | 92.61 | 92.69 | −0.63 | −0.76 |
| clean F1@0.5 | 97.60 | 97.69 | 97.48 | 97.06 | 96.96 | 96.77 | 97.25 | 97.20 | 96.66 | 96.17 | 96.92 | 96.97 | **−0.56 [−0.87, −0.26]** | −0.09 [−0.44, +0.24] |
| noisy F1@0.5 | 89.57 | 89.76 | 91.16 | 91.17 | 90.20 | 90.47 | 90.17 | 90.40 | 88.38 | 88.38 | 89.88 | 90.39 | −1.28 | −0.78 |
| mean noisy EER | 10.09 | 9.66 | 9.16 | 9.01 | 10.15 | 10.17 | 9.96 | 9.71 | 11.05 | 10.99 | 9.77 | 9.84 | +0.61 | +0.83 |
| calibration gap noisy | 0.66 | 0.71 | 0.11 | 0.42 | 0.62 | 0.21 | 0.28 | 0.12 | 1.35 | 1.14 | 0.56 | 0.22 | +0.45 | −0.20 |
| separability gap | 7.87 | 7.86 | 6.56 | 6.20 | 7.12 | 7.27 | 7.29 | 7.36 | 7.58 | 7.73 | 7.25 | 6.95 | +0.69 | +0.75 |

- B3e is below B3a on WF1, oracle WF1, clean and noisy F1 and noisy EER in both readouts.
- B3e-top5 lands on B2f-top5 (91.99 vs 91.98): the merge gives back all of B3a's gain over B2f. B3e-ep8 is +0.22 over B2f-ep8 (computed from `compare_all_readouts.txt`).
- Unlike B3a, **B3e's final epoch beats its merge** on half B (92.36 vs 91.99).

Per-set paired CIs (`ci_vs_B3a-*.txt`):

| set | merge Δ F1 [CI] | ep8 Δ F1 [CI] |
|---|---|---|
| clean | **−0.56 [−0.87, −0.26]** | −0.09 [−0.44, +0.24] |
| matched | −0.88 [−1.98, +0.21] | −0.60 [−1.57, +0.40] |
| heldout | **−1.68 [−2.93, −0.49]** | −0.98 [−2.20, +0.19] |
| echo | −0.53 [−1.37, +0.36] | −0.55 [−1.49, +0.46] |

Every point estimate is negative. On the merge, the clean and heldout CIs exclude 0. On ep8 no per-set CI excludes 0; only the WF1 aggregate does.

Weighted contributions (0.3 / 0.35 / 0.35 × the unrounded deltas in `compare_ci_vs_B3a-*.json`, computed):
- merge: −0.17 clean − 0.31 matched − 0.59 heldout = −1.06. Heldout carries 55 %, matched 29 %, clean 16 %.
- final: −0.03 − 0.21 − 0.34 = −0.58. Heldout 59 %, matched 36 %, clean 5 %.

Heldout moved most again, as in B3c and B3d (B3d report §2.1: heldout 61 % of B3d's loss).

**Other halves of the same sets.** Unlike B3d's, this loss depends on the half:

| readout | full-set WF1 B3a → B3e (`metrics.json` `full.wf1`) | half_A WF1 B3a → B3e (computed from `scores/*.txt`) | half_A matched / heldout F1 B3a → B3e (computed) |
|---|---|---|---|
| merge | 93.05 → 92.27 (−0.78) | 93.04 → 92.55 (**−0.49**) | 92.45 → 92.29 / 89.82 → 89.08 |
| final | 92.78 → 92.39 (−0.39) | 92.61 → 92.41 (**−0.20**) | 91.55 → 92.11 / 89.85 → 88.82 |

On the selection half A, both deltas are inside the ±0.5 band. The B2 near-replicate noise band in `wave.json` lever_notes is "~0.2-0.5 WF1 on the final readout, up to ~1.4 per set". Heldout is lower on both halves; matched is half-dependent. half_B is the objective, and the verdict below is on half_B.

### 2.2 Per set (R = recall at 0.5; thr = p_spoof threshold; `metrics.json` half_B)

| set | metric | B3a-top5 | **B3e-top5** | B3a-ep8 | **B3e-ep8** |
|---|---|---|---|---|---|
| dev_online_clean | F1@.5 / F1@or / thr_or | 97.48 / 97.83 / 0.996 | **96.92** / 97.69 / 0.998 | 97.06 / 97.79 / 1.000 | 96.97 / 97.55 / 0.999 |
| | EER / ECE | 3.32 / 1.47 | 3.15 / 1.78 | 3.20 / 1.72 | **2.43** / 1.79 |
| | R_spoof / R_bona | 99.93 / 92.34 | 99.92 / **90.69** | 99.92 / 91.12 | 99.92 / 90.84 |
| sim_matched_v1 | F1@.5 / F1@or / thr_or | 92.12 / 92.25 / 0.995 | **91.24** / 91.59 / 0.979 | 92.03 / 92.49 / 0.998 | 91.43 / 91.65 / 0.0095 |
| | EER / ECE | 7.89 / 4.51 | 8.18 / 4.97 | 7.74 / 4.55 | 8.40 / 4.85 |
| | R_spoof / R_bona | 99.38 / 79.03 | 99.44 / **76.47** | 99.69 / 77.75 | 99.44 / 76.98 |
| sim_heldout_v1 | F1@.5 / F1@or / thr_or | 90.20 / 90.30 / 0.971 | **88.52** / 89.28 / 0.964 | 90.32 / 90.69 / 0.993 | 89.34 / 89.56 / 0.918 |
| | EER / ECE | 10.42 / 5.54 | **11.37** / 6.29 | 10.29 / 5.57 | **11.27** / 5.90 |
| | R_spoof / R_bona | 99.06 / 74.94 | **99.50** / **69.31** | 99.00 / 75.45 | 99.50 / **71.36** |
| sim_echo_v1 | F1@.5 / F1@or / thr_or | 94.87 / 95.03 / 0.998 | 94.34 / 94.73 / 0.994 | 94.39 / 94.96 / 0.999 | 93.85 / 94.50 / 0.997 |
| | EER / ECE | 4.61 / 3.00 | **6.25** / 3.29 | 5.62 / 3.28 | 5.46 / 3.60 |
| | R_spoof / R_bona | 99.81 / 85.17 | 99.75 / 83.89 | 99.94 / 83.38 | 99.50 / 83.38 |

(The B3e-ep8 matched `thr_oracle` of 0.0095 is copied as written. Its F1@or is only 0.22 above F1@0.5, so the F1 curve is flat over a wide threshold range.)

Reading:
- **Bona recall fell on every set** at the merge: clean −1.65, matched −2.56, **heldout −5.63**, echo −1.28. On ep8 (computed vs B3a-ep8): clean −0.28, matched −0.77, **heldout −4.09**, echo 0.00.
- **Spoof recall rose on heldout** (99.06 → 99.50 at both readouts) and was flat elsewhere at the merge. The operating point moved toward spoof.
- **EER rose on matched, heldout and echo at the merge** (+0.28, +0.94, +1.64): some of the loss is ranking (section 5.2). Clean EER improved in both readouts (3.32 → 3.15; 3.20 → 2.43 on ep8) while clean F1@0.5 fell. The clean loss is an operating-point loss, not a ranking loss.
- Paired per clip on half_B, B3e vs B3a (computed; fixed / broken / wrong in both; then en fixed / broken · zh fixed / broken):

  | set | class | merge | merge en · zh | ep8 | ep8 en · zh |
  |---|---|---|---|---|---|
  | clean | bona | 8 / **31** / 99 | 2 / 11 · 6 / 20 | 21 / 25 / 103 | 12 / 2 · 9 / 23 |
  | matched | bona | 9 / 19 / 73 | 4 / 9 · 5 / 10 | 10 / 13 / 77 | 4 / 3 · 6 / 10 |
  | heldout | bona | 5 / **27** / 93 | 2 / 8 · **3 / 19** | 7 / **23** / 89 | 4 / 8 · 3 / 15 |
  | echo | bona | 7 / 12 / 51 | 4 / 5 · 3 / 7 | 12 / 12 / 53 | 8 / 4 · 4 / 8 |
  | matched | spoof | 5 / 4 / 5 | 1 / 2 · 4 / 2 | 1 / 5 / 4 | 0 / 2 · 1 / 3 |
  | heldout | spoof | 9 / 2 / 6 | 2 / 1 · 7 / 1 | 10 / 2 / 6 | 2 / 2 · 8 / 0 |
  | echo | spoof | 2 / 3 / 1 | 1 / 2 · 1 / 1 | 1 / 8 / 0 | 0 / 5 · 1 / 3 |

  **Summed over the four sets, bona: 29 fixed / 89 broken (merge) and 50 / 73 (ep8).** Noisy spoof: 14 fixed / 6 broken at the merge (matched + heldout). For comparison, B3d broke 127 bona and fixed 24 (B3d report §2.2). Heldout zh bona broke most (19 of the 27 heldout breaks at the merge).
- Logit correlation B3e vs B3a (computed, half_B): 0.977-0.992 per set on the merge and 0.971-0.987 on ep8, lowest on heldout. B3e is a small perturbation of B3a, like B3c (0.974-0.985) and much smaller than B3d (0.925-0.960; both from the B3d report §2.2).

### 2.3 Probes (`probe_report.json`; 187 bona + 813 spoof; one bona probe clip = 0.53 points)

| probe (bona unless stated) | B3a-top5 | **B3e-top5** | B3a-ep8 | **B3e-ep8** |
|---|---|---|---|---|
| level_flip_bona (mean) / max | 0.80 / 2.67 | **2.34 / 6.42** | 2.07 / 7.49 | 1.94 / 9.09 |
| gain_up / gain_p6 / gain_p10_limit flip % | 2.67 / 1.07 / 2.67 | 5.35 / 2.14 / **6.42** | 4.81 / 2.14 / 7.49 | 4.28 / 0.53 / 9.09 |
| gain_down / gain_m20 / gain_m6 flip % | 0.00 / 0.00 / 0.00 | 0.53 / 0.53 / 0.53 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| agc_dynaudnorm / loudnorm flip % | 0.00 / 0.00 | 1.60 / 1.60 | 0.00 / 2.14 | 1.07 / 0.53 |
| level_flip_spoof | 0.02 | 0.00 | 0.00 | 0.02 |
| probe_orig: bona pred spoof % | 12.30 | 13.90 | 11.76 | 13.37 |
| noise_only: bona-source pred spoof %, mean dP | 100.00, +0.876 | 100.00, +0.861 | 100.00, +0.882 | 100.00, +0.865 |
| sil_only: bona-source pred spoof %, mean dP | 100.00, +0.877 | 100.00, +0.860 | 100.00, +0.882 | 100.00, +0.859 |
| trimmed / padded bona flip % | 2.14 / 3.21 | 2.14 / 2.14 | 3.21 / 2.67 | 2.14 / 2.67 |

Reading:
- `noise_only` / `sil_only` bona-source clips are still 100 % spoof in every readout, as in every B3 run. These are the probes closest to 0-dB bona (design `eval_note`). Training at 0 dB did not move them; mean dP fell only 0.015-0.023.
- Level robustness got worse on the merge (mean flip 0.80 → 2.34; the limiter probe 2.67 → 6.42 %, 5 → 12 clips). It is roughly flat on ep8 (2.07 → 1.94).

### 2.4 Slices that moved most, B3e vs B3a (full sim sets, `metrics.json → breakdowns`)

Over 60 slice rows (the echo `aec` axis is left out because it duplicates the echo `family` axis; computed):

| readout | F1 up | EER down | R_bona up | R_spoof up / down |
|---|---|---|---|---|
| merge | 10 of 60 | 11 | **6** | 35 / 10 |
| final | 20 of 60 | 25 | 16 | 26 / 26 |

**SNR slices, the target of the design** (full sets, merge; ep8 R_bona in the last column; `merge_top5.log` and `metrics.json`):

| set / SNR | n (bona) | F1 B3a → B3e | EER B3a → B3e | R_bona B3a → B3e | R_spoof B3a → B3e | ep8 R_bona B3a → B3e |
|---|---|---|---|---|---|---|
| matched 0 | 711 (117) | 86.15 → 86.94 | 14.67 → **15.94** | 65.81 → **68.38 (+2.56)** | 98.65 → 98.48 | 66.67 → 67.52 |
| matched 5 | 656 (118) | 90.34 → 91.07 | 10.10 → 10.10 | 77.12 → 77.12 | 98.51 → 99.07 | 77.12 → 74.58 |
| matched 10 | 653 (124) | 91.80 → 90.26 | 7.41 → **6.53** | 76.61 → 72.58 (−4.03) | 99.81 → 99.81 | 70.97 → 73.39 |
| matched 15 | 660 (109) | 93.57 → 93.08 | 6.94 → 9.40 | 82.57 → 78.90 (−3.67) | 99.46 → 100.00 | 78.90 → 79.82 |
| matched 20 | 642 (127) | 95.08 → 94.01 | 6.35 → 5.87 | 85.83 → 83.46 (−2.36) | 99.81 → 99.61 | 85.04 → 85.04 |
| matched (none) = echo family | 678 (155) | 95.38 → 94.21 | 4.84 → 4.36 | 86.45 → 83.23 (−3.23) | 100.00 → 100.00 | 85.16 → 83.87 |
| heldout 0 | 812 (147) | 85.72 → 84.57 | 15.08 → **18.92** | 65.31 → **61.22 (−4.08)** | 98.50 → 98.95 | 64.63 → 64.63 |
| heldout 5 | 778 (156) | 90.39 → 88.61 | 10.11 → 11.24 | 77.56 → **70.51 (−7.05)** | 98.39 → 99.20 | 79.49 → 73.08 |
| heldout 10 | 815 (152) | 91.13 → 90.31 | 10.29 → **8.57** | 78.29 → 73.03 (−5.26) | 98.79 → 99.70 | 76.32 → 74.34 |
| heldout 15 | 835 (151) | 90.42 → 89.20 | 9.31 → 11.74 | 74.17 → 71.52 (−2.65) | 99.42 → 99.27 | 74.17 → 70.86 |
| heldout 20 | 760 (144) | 92.21 → 91.07 | 8.98 → 10.32 | 79.86 → 75.69 (−4.17) | 99.19 → 99.51 | 82.64 → 75.69 |

(Bona counts per slice are n − n_spoof from `metrics.json`. Matched SNR-0 F1@oracle went 86.84 → 87.88. Heldout SNR-0 F1@oracle went 86.00 → 85.62.)

- **Matched SNR 0 moved in the predicted direction, but less than predicted.** R_bona rose +2.56 (the prediction was ≥ +3) and F1@oracle rose +1.04, but EER got worse.
- **Heldout SNR 0, the larger half of the SNR-0 target, got worse** on R_bona (−4.08) and EER (+3.84).
- **Heldout bona recall fell at every SNR** (−2.65 to −7.05). In matched it fell at 10-20 dB and held at 0-5 dB.
- The one consistent gain is EER at SNR 10 on both sets (7.41 → 6.53; 10.29 → 8.57). Bona recall at SNR 10 still fell, so that gain is lost to the operating point.

Other large moves (full sets, merge; F1, EER, R_bona; B3a → B3e):

| set / slice | n (bona) | F1 (Δ) | EER | R_bona |
|---|---|---|---|---|
| heldout / wind | 381 (76) | 88.23 → 84.79 (−3.44) | 13.46 → 13.30 | 71.05 → **61.84** (−9.21) |
| heldout / mouse_click | 369 (78) | 89.92 → 87.23 (−2.68) | 11.44 → 12.13 | 75.64 → 67.95 (−7.69) |
| heldout / door_wood_creaks | 361 (64) | 90.88 → 88.98 (−1.90) | 7.44 → 11.02 | 78.12 → 71.88 (−6.25) |
| heldout / g726 | 1274 (246) | 88.81 → 86.85 (−1.96) | 10.74 → 11.05 | 71.14 → 65.45 (−5.69) |
| heldout / lang en | 1538 (258) | 84.74 → 82.97 (−1.77) | 17.04 → 17.08 | 60.85 → **55.43** (−5.43) |
| heldout / gsm | 1346 (246) | 88.18 → 87.12 (−1.05) | 13.91 → 15.75 | 72.76 → 67.48 (−5.28) |
| heldout / vacuum_cleaner | 373 (70) | 87.09 → 87.09 (0.00) | 8.80 → **17.32** | 67.14 → 67.14 |
| matched / opus:lark | 608 (119) | 94.71 → 93.19 (−1.51) | 8.50 → 7.67 | 84.03 → 79.83 (−4.20) |
| matched / opus:voov | 510 (89) | 94.58 → 92.99 (−1.58) | 6.93 → 7.61 | 84.27 → 79.78 (−4.49) |
| matched / opus:zoom (largest gain) | 584 (106) | 93.82 → 94.77 (+0.95) | 7.90 → 7.43 | 82.08 → 83.96 (+1.89) |
| matched / rain | 671 (115) | 89.02 → 89.84 (+0.83) | 12.20 → 13.45 | 72.17 → 73.04 (+0.87) |

**Pattern: a mild, broad decline concentrated on the unseen conditions.**
- The heldout noise families, codecs and English lost most.
- Matched slices (training noise families, Opus apps) moved between +2.56 and −4.49 R_bona points. Matched SNR 0 and 5 held or rose.
- This is the B3d pattern at about half the size. B3d's merge lost 2.20; its heldout g726 R_bona lost 13.42 against 5.69 here (B3d report §2.4).

### 2.5 Lever response (one factor per row)

| run | change vs reference | half_B merge WF1 (Δ) | final WF1 (Δ) | merge oracle | final oracle | merge noisy EER | final noisy EER | merge clean | final clean | verdict (verdict.json / registry) |
|---|---|---|---|---|---|---|---|---|---|---|
| B2f | (wave start) | 91.98 | 92.14 | 92.59 | 92.82 | 10.09 | 9.66 | 97.60 | 97.69 | - |
| B3a | + `--consistency js --consistency_lambda 0.3 --consistency_pairs both`, `--batch_size 8` (vs B2f) | 93.06 (+1.08) | 92.94 (+0.80) | 93.24 | 93.45 | 9.16 | 9.01 | 97.48 | 97.06 | merge positive / promote |
| B3b | + `--sampler balanced --sampler_lang_balance --sampler_bona_frac 0.5 --ce_weights 0.3223 0.6777` (vs B3a) | 92.23 (−0.83) | 92.36 (−0.58) | 92.96 | 92.86 | 10.15 | 10.17 | 96.96 | 96.77 | negative / drop |
| B3c | `--consistency_lambda 0.3 → 1.0` (vs B3a) | 92.29 (−0.77) | 92.44 (−0.50) | 92.63 | 92.73 | 9.96 | 9.71 | 97.25 | 97.20 | merge negative / drop; final null / drop |
| B3d | + `--aug_codec_profiles random` (vs B3a) | 90.86 (−2.20) | 90.72 (−2.22) | 92.00 | 91.84 | 11.05 | 10.99 | 96.66 | 96.17 | negative / drop |
| **B3e** | **+ `--aug_snr_db 0 20` (vs B3a)** | **91.99 (−1.06)** | **92.36 (−0.58)** | 92.61 | 92.69 | 9.77 | 9.84 | 96.92 | 96.97 | **negative / drop (both)** |

**All four single-factor changes on top of B3a were negative on the merge.** The two augmentation levers that widened or shifted the training distribution (B3d, B3e) are the two largest merge losses. B3e's merge loss is about half of B3d's.

## 3. Per-epoch behaviour, B3e vs B3a (same steps, same LR at each epoch end)

Sources: `metrics.jsonl` (EMA weights; 2,000 clean clips + `sim_matched_mini` + `sim_heldout_mini`, half A) and `logs/train.log`. CE weights are identical in the two runs, so CE, train acc and dev loss are directly comparable. JS is unweighted.

| ep | LR end | B3e mini WF1 | B3a | Δ | B3e WF1@or (B3a) | B3e EER noisy (B3a) | B3e R_bona cl/ma/he | B3a R_bona cl/ma/he | B3e CE / JS | B3a CE / JS | train acc B3e / B3a | dev loss B3e / B3a | dev acc B3e / B3a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.469e-6 | 90.21 | 90.17 | +0.03 | 90.57 (90.55) | 13.24 (13.14) | 91.0/75.4/**68.9** | 91.0/73.7/65.8 | 0.4165 / 0.0380 | 0.3945 / 0.0344 | **89.26 / 90.16** | 0.1051 / 0.0990 | 98.12 / 98.35 |
| 2 | 2.289e-6 | 91.37 | 91.64 | −0.27 | 91.84 (92.27) | 11.18 (11.56) | 89.6/75.4/67.9 | 91.6/76.5/70.9 | 0.2119 / 0.0257 | 0.1867 / 0.0214 | 97.43 / 97.87 | 0.1607 / 0.1232 | 98.00 / 98.41 |
| 3 | 1.976e-6 | 91.60 | 91.39 | +0.22 | 92.45 (92.14) | 10.76 (11.49) | 89.0/78.2/70.9 | 91.6/76.5/66.8 | 0.1462 / 0.0192 | 0.1176 / 0.0151 | 98.27 / 98.63 | 0.1572 / 0.1514 | 98.28 / 98.53 |
| 4 | 1.583e-6 | 91.65 | 91.52 | +0.14 | 92.56 (92.13) | 10.43 (10.80) | 89.9/74.9/69.4 | 89.9/76.5/69.9 | 0.1056 / 0.0146 | 0.0842 / 0.0113 | 98.78 / 99.01 | 0.1654 / 0.1657 | 98.38 / 98.33 |
| 5 | 1.175e-6 | **92.32** | 92.52 | −0.20 | 93.01 (93.17) | 11.16 (11.16) | 90.7/79.3/70.4 | 91.6/81.0/71.9 | 0.0811 / 0.0115 | 0.0642 / 0.0080 | 99.07 / 99.32 | 0.1678 / 0.1472 | 98.42 / 98.69 |
| 6 | 8.227e-7 | 91.90 | **92.87** | **−0.97** | 92.79 (93.51) | 11.13 (10.09) | 89.9/78.8/68.4 | 91.9/82.7/71.9 | 0.0636 / 0.0095 | 0.0452 / 0.0065 | 99.21 / 99.47 | 0.1958 / 0.1356 | 98.31 / 98.70 |
| 7 | 5.842e-7 | 92.13 | 92.36 | −0.23 | 92.88 (93.07) | **10.18** (10.40) | 89.0/80.4/71.4 | 91.3/80.4/71.4 | 0.0514 / 0.0076 | 0.0328 / 0.0051 | 99.39 / 99.59 | 0.2006 / 0.1584 | 98.19 / 98.58 |
| 8 | 5.000e-7 | 91.89 | 92.14 | −0.24 | 92.35 (93.03) | 10.93 (10.28) | 89.6/78.2/69.9 | 89.9/78.8/71.9 | **0.0426 / 0.0063** | 0.0283 / 0.0038 | **99.49 / 99.71** | **0.1985** / 0.1722 | 98.18 / 98.44 |

Reading:
- **B3e starts level with B3a and falls behind in the second half of training.**
  - Mini WF1 is within ±0.27 of B3a's at epochs 1-5 (+0.03, −0.27, +0.22, +0.14, −0.20). It is −0.97 at B3a's best epoch (6) and −0.23 / −0.24 at epochs 7-8. B3e is below B3a at 5 of 8 epochs.
  - At epoch 1, heldout-mini bona recall was **higher** than B3a's (68.9 vs 65.8; matched 75.4 vs 73.7). The 0-dB exposure helped bona early. From epoch 5 on, B3e's heldout-mini R_bona is at or below B3a's (68.4-71.4 vs 71.4-71.9).
- **The model fits the training data less well as training goes on** (ratios computed from `train.log`).
  - CE is 1.06× B3a's at epoch 1, rising steadily to **1.51×** at epoch 8 (0.0426 vs 0.0283).
  - Unweighted JS is 1.10× → **1.67×** (0.0063 vs 0.0038). Two views of a clip that draw independent SNRs in 0..20 disagree more.
  - Train acc is 89.26 vs 90.16 at epoch 1 (−0.90, inside the design's 2-point under-fit guard) and 99.49 vs 99.71 at epoch 8. It is below B3a's at every epoch.
- **Dev loss and dev acc.**
  - Dev loss is lowest at epoch 1 (0.1051) and rises **+89 %** to 0.1985 (B3a: +74 %, 0.0990 → 0.1722; computed). It is above B3a's at 7 of 8 epochs (epoch 4: 0.1654 vs 0.1657).
  - Dev acc is below B3a's at 7 of 8 epochs (98.00-98.42 against 98.33-98.70).
  - Less fit on train and worse on dev is the signature of a harder training distribution, the same as B3d's but milder (B3d: CE 2.77×, JS 2.89×, train acc 84.75 at epoch 1; B3d report §3). It is not over-fitting.
- **Curve shape.** The best epoch is 5 (92.32; `best_by_wf1` → epoch 5). The curve then plateaus at 91.89-92.13. Mini noisy EER is best at epoch 7 (10.18). The curve is not still rising at the end.
- **Final vs merge.** On half B the final epoch beats the merge by 0.37 (92.36 vs 91.99). On half A the merge wins (92.55 vs 92.41, computed). Per-epoch half_B evaluations do not exist, so which merged epoch drags the merge down cannot be measured. What the files do show is that the merge's extra loss sits in moderate-confidence noisy bona false alarms: 65 at p in [0.5, 0.99) vs 38 for ep8 (section 5.3). That is an operating-point effect of averaging epochs 4-8.

## 4. What went OK (evidence)

1. **A clean, single-flag experiment.**
   - Only `aug_snr_db` (plus `track`) differs in `config.yaml` `args`, and all six header regexes matched.
   - The noise-stage share of views (0.5996-0.6015) matches the derivation's 0.600.
   - No traceback, OOM, `failed (` or `Error` line; no trainer code in `git.diff`.
   - The merge result is decisive (CI [−1.65, −0.49]), and the final readout's CI also excludes 0 ([−1.13, −0.05]).
2. **Training at 0 dB did not blur the spoof cue** (design falsifier 3 cleanly excluded):
   - half_B R_spoof rose on heldout (99.06 → 99.50) and held on matched (99.38 → 99.44);
   - full-set SNR-0 R_spoof was 98.48 / 98.95 against 98.65 / 98.50;
   - half_B noisy spoof misses fell 25 → 17 (merge) and 21 → 17 (ep8) (computed).
3. **The in-pool low-SNR condition moved the predicted way, if weakly.** Matched SNR 0: R_bona 65.81 → 68.38, F1 86.15 → 86.94, F1@oracle 86.84 → 87.88. EER at SNR 10 improved on both sets (7.41 → 6.53; 10.29 → 8.57).
4. **A small perturbation, unlike B3d.**
   - Logit correlation with B3a is 0.977-0.992 (merge).
   - The loss is about half of B3d's, and on the selection half A it is −0.49 / −0.20 (inside the band).
   - The final readout's clean drop is only 0.09, and its clean EER improved (3.20 → 2.43).
5. **The early under-fit check worked as intended.** Epoch-1 train acc 89.26 (vs 90.16) and CE 0.4165 (vs 0.3945) were within the design's guard, and the run did not collapse the way B3d did (84.75 % at epoch 1).
6. **No time cost:** 57.7-58.5 min per epoch, 7 h 43 min in total.

## 5. What went wrong (evidence)

### 5.1 Every set lost, heldout most, and the main target got worse

- WF1 −1.06 (merge) / −0.58 (ep8). On the merge the clean and heldout CIs exclude 0 (section 2.1).
- Bona recall (merge): heldout −5.63, matched −2.56, clean −1.65, echo −1.28.
- **The design's target was the SNR-0 bona false alarms on matched and heldout.**
  - Heldout SNR 0 got worse: R_bona 65.31 → 61.22, EER 15.08 → 18.92.
  - half_B SNR-0 false alarms rose 40 → 44 (section 5.3).
  - Matched SNR-0 R_bona rose +2.56, short of the +3 the design asked for.
- **Clean dropped 0.56 F1 on the merge**, above the wave's 0.5 clean guard (`verdict.json` `clean_drop` 0.5637). It is an operating-point loss: clean EER improved (3.32 → 3.15), while clean bona recall fell 92.34 → 90.69, with 31 bona clips broken and 8 fixed.

### 5.2 Oracle-gap decomposition (global threshold vs set-specific)

Method: one shared logit threshold swept over the half_B `scores/*.logits` (step 0.05, −6 to +12), as in the B3a-B3d reports. The per-set oracle comes from `metrics.json` `wf1_oracle` (computed).

| readout | WF1@0.5 | best global (logit / p) | WF1 at best global | per-set oracle WF1 | gap | recovered by one global shift | set-specific remainder |
|---|---|---|---|---|---|---|---|
| B3a-top5 | 93.06 | −0.15 / 0.463 (flat) | 93.08 | 93.24 | 0.18 | 0.02 | 0.16 |
| B3a-ep8 | 92.94 | +4.95 / 0.993 | 93.10 | 93.45 | 0.51 | 0.16 | 0.35 |
| **B3e-top5** | 91.99 | **+4.20 / 0.985** | **92.22** | 92.61 | **0.62** | **0.23** | 0.39 |
| **B3e-ep8** | 92.36 | +2.40 / 0.917 | 92.48 | 92.69 | 0.33 | 0.12 | 0.21 |

- **The merge loss is mostly ranking.** At their best global thresholds, B3e-top5 is 0.86 below B3a-top5 (92.22 vs 93.08). The rest is B3e's operating point (0.23) minus B3a's own (0.02). That gives 1.07, matching the −1.06 delta up to rounding (computed).
- **The final-epoch loss is all ranking:** 93.10 − 92.48 = 0.62 at the best global thresholds, against a −0.58 delta at 0.5.
- The calibration gap noisy (`metrics.json`) rose 0.11 → 0.56 on the merge and fell 0.42 → 0.22 on ep8.

### 5.3 Noisy bona false alarms grew in the moderate band; the saturated tail did not shrink

Share of half_B bona with p_spoof ≥ 0.5 / ≥ 0.9 / ≥ 0.99 / ≥ 0.999 (computed):

| set (n bona) | B3a-top5 | **B3e-top5** | B3a-ep8 | **B3e-ep8** |
|---|---|---|---|---|
| clean (1,397) | 7.7 / 7.6 / 6.4 / 4.6 | 9.3 / 9.1 / 7.4 / 5.9 | 8.9 / 8.9 / 8.3 / 6.1 | 9.2 / 9.2 / 8.4 / 6.4 |
| matched (391) | 21.0 / 21.0 / 16.4 / 8.2 | 23.5 / 23.5 / 16.6 / 10.0 | 22.3 / 22.3 / 19.2 / 12.0 | 23.0 / 22.5 / 19.4 / 13.6 |
| heldout (391) | 25.1 / 25.1 / 18.9 / 10.2 | **30.7 / 30.7 / 21.0 / 12.0** | 24.6 / 24.3 / 20.5 / 13.0 | 28.6 / 27.9 / 22.5 / 14.8 |
| echo (391) | 14.8 / 14.8 / 12.3 / 5.4 | 16.1 / 15.9 / 12.8 / 7.7 | 16.6 / 16.4 / 15.3 / 9.5 | 16.6 / 16.1 / 14.1 / 9.5 |

Noisy bona (matched + heldout, 782; computed):

| readout | false alarms (FA %) | FA at p ≥ 0.99 / in [0.5, 0.99) | share ≥ 0.99 | share ≥ 0.999 | saturated share of FA | FA en / zh | noisy spoof misses (of 3,206) |
|---|---|---|---|---|---|---|---|
| B3a-top5 | 180 (23.0 %) | 138 / 42 | 17.6 % | 9.2 % | 77 % | 91 / 89 | 25 |
| **B3e-top5** | **212 (27.1 %)** | 147 / **65** | **18.8 %** | **11.0 %** | 69 % | 102 / **110** | **17** |
| B3a-ep8 | 183 (23.4 %) | 155 / 28 | 19.8 % | 12.5 % | 85 % | 93 / 90 | 21 |
| **B3e-ep8** | **202 (25.8 %)** | 164 / 38 | **21.0 %** | **14.2 %** | 81 % | 96 / 106 | 17 |

- The merge has 32 more false alarms: 9 saturated and 23 moderate. The moderate ones are what the +0.23 global shift of 5.2 partly recovers.
- The saturated tail (p ≥ 0.99) did not shrink in either readout. It grew by 9 clips in each, and the ≥ 0.999 share rose 9.2 → 11.0 % (merge) and 12.5 → 14.2 % (ep8).
- Spoof misses fell (25 → 17) while bona false alarms rose (180 → 212): the decision moved toward spoof.

**Where the false alarms sit, half_B noisy bona** (computed from `scores/*.txt` + `meta.csv`):

| split (n bona) | B3a-top5 | **B3e-top5** | B3a-ep8 | **B3e-ep8** |
|---|---|---|---|---|
| SNR 0 (123: matched 63, heldout 60) | 40 (21 + 19) | **44** (21 + 23) | 40 | 42 |
| SNR 5 (155) | 34 (13 + 21) | **40** (10 + 30) | 34 | 39 |
| SNR 10 (142) | 42 (21 + 21) | **48** (25 + 23) | 46 | 46 |
| SNR 15 (148) | 32 (11 + 21) | **40** (14 + 26) | 34 | 39 |
| SNR 20 (134) | 24 (8 + 16) | 27 (9 + 18) | 20 | 25 |
| matched no-SNR echo family (80) | 8 | 13 | 9 | 11 |
| heldout g722 / g726 / gsm (135 / 129 / 127) | 24 / 36 / 38 | **31 / 45 / 44** | 26 / 36 / 34 | 31 / 42 / 39 |
| matched opus:qq / wechat (60 / 50) | 21 / 17 | 22 / 18 | 24 / 17 | 20 / 19 |
| matched other 5 apps (281) | 44 | 52 | 46 | 51 |
| heldout en / zh (129 / 262) | 49 / 49 | 55 / **65** | 48 / 48 | 52 / 60 |
| matched en / zh (129 / 262) | 42 / 40 | 47 / 45 | 45 / 42 | 44 / 46 |
| **matched / heldout total** | 82 / 98 | 92 / **120** | 87 / 96 | 91 / 111 |

- **The extra false alarms are spread over every SNR, not concentrated at 0 dB.** SNR 0 gained +4 (merge); SNR 15 gained +8, and SNR 5 and 10 +6 each.
- Heldout took 22 of the 32 extra merge false alarms (98 → 120), and heldout Chinese bona took 16 of those (49 → 65).

### 5.4 Worst slices for bona recall, B3e-top5 (full sets, `metrics.json` breakdowns)

| slice | n (bona) | R_bona | R_spoof | EER | B3a-top5 R_bona |
|---|---|---|---|---|---|
| heldout en | 1538 (258) | **55.43** | 99.61 | 17.08 | 60.85 |
| heldout snr 0 | 812 (147) | 61.22 | 98.95 | **18.92** | 65.31 |
| heldout wind | 381 (76) | 61.84 | 99.02 | 13.30 | 71.05 |
| heldout washing_machine | 380 (78) | 62.82 | 98.68 | 16.61 | 66.67 |
| matched en | 1538 (258) | 64.73 | 99.53 | 13.58 | 65.50 |
| heldout g726 | 1274 (246) | 65.45 | 99.42 | 11.05 | 71.14 |
| matched opus:qq | 572 (99) | 66.67 | 98.94 | 10.55 | 68.69 |
| heldout vacuum_cleaner | 373 (70) | 67.14 | 99.01 | 17.32 | 67.14 |
| heldout gsm | 1346 (246) | 67.48 | 98.91 | 15.75 | 72.76 |
| heldout mouse_click | 369 (78) | 67.95 | 98.97 | 12.13 | 75.64 |

These are the same axes as in B3a-B3d: English, SNR 0, the heldout noise families and codecs, and qq. Every one is at or below B3a.

### 5.5 Training dynamics: harder, not over-fit

- CE ratio to B3a grows monotonically: 1.06, 1.13, 1.24, 1.25, 1.26, 1.41, 1.57, 1.51. JS ratio grows too: 1.10 → 1.67 (computed from `train.log`).
- Train acc is lower at every epoch, and dev acc lower at 7 of 8 epochs (section 3). Dev loss rising while train acc passes 99 % (from epoch 5: 99.07 %) happens in both runs. B3e's rise is larger (+89 % vs +74 %) at a *lower* train acc, so the extra dev-loss rise does not come from fitting the training set harder.
- Epoch-8 CE is 0.0426, just above the design's 1.5 × 0.0283 bound (ratio 1.51, computed).

### 5.6 Why it failed (interpretation, not proven by these files)

- The matched set uses the same noise families as the training stages: coffee, footsteps, keyboard, office, rain and echo appear both in the `meta.csv` family names and in the header's stage list. The heldout set uses reserved ESC-50 families (`esc50_reserved` in `meta.csv` `noise_file`), extra DSP chains, and g722/g726/gsm.
- At 0-5 dB the noise carries as much energy as the speech or more. Training there gives both CE and JS more signal about the *noise* of the training pools. That plausibly helped matched SNR 0 (F1@oracle +1.04, R_bona +2.56) and SNR-10 ranking on both sets.
- It did not carry over to unseen noises. Heldout lost R_bona at every SNR (−2.65 to −7.05) and EER at 0, 5, 15 and 20 dB.
- The paired JS term at 0.3 asks two views with independent SNR draws (for example 0 vs 20 dB) to agree. JS rose to 1.67× B3a's. The representation that met both CE and JS on this harder distribution pushes more degraded bona toward spoof, while spoof recall rose. That is the same direction as B3d, at half the size.
- Lesson, consistent with B3d's: at this budget (8 epochs, LR 2.5e-6), widening the training augmentation, even toward the eval support, costs fit (CE 1.51×) and moves the unseen-condition bona toward spoof. The SNR-0 false alarms are not an out-of-support problem that more 0-dB data fixes.

## 6. Verdict on the pre-registered rule

From `exploop/B3/iter05/design.result.json`, the **falsifier**:

> "**Wave rule** against B3a-top5 = 93.06 on half_B:
> - promote / positive: delta merge > +0.5, the paired-bootstrap CI excluding 0, and a clean F1 drop <= 0.5;
> - null: delta within +-0.5;
> - negative: delta below -0.5.
>
> **Mechanism-level refutations:**
> 1. **The SNR range is not the limit.** Delta within +-0.5 AND full-set SNR-0 R_bona within +-3 of 65.81 (matched) / 65.31 (heldout). The SNR-0 errors are not an out-of-support problem.
> 2. **Covered, but traded away.** SNR-0 R_bona up >= 3 while 15-20 dB R_bona falls > 2 on matched or heldout, or the clean drop is > 0.5. Spending a quarter of the noise draws below 5 dB cost the high-SNR mass, and the net is null or negative.
> 3. **0 dB masks the spoof cue.** Full-set SNR-0 R_spoof down > 0.5 from 98.65 / 98.50, or half_B R_spoof down > 0.5 on matched or heldout. At 0 dB the synthesis artefacts are drowned, and training there teaches the model to wave noisy spoof through.
> 4. **B3d-like under-fit: any harder training distribution hurts at this budget.** Epoch-1 train acc more than 2 below 90.16, epoch-8 CE above 1.5 x 0.0283, and every set down (R_bona down > 2 on the 15-20 dB slices). The recipe stays B3a, and augmentation levers are closed for the full run."

`exploop/B3/iter05/verdict.json` (computed by the supervisor; quoted, not recomputed):

| readout | run | base | delta | CI | clean_drop | new_best | decision | verdict |
|---|---|---|---|---|---|---|---|---|
| merge (objective) | 91.99173229893125 | 93.05618353351335 | −1.0644512345820942 | [−1.6519077179139843, −0.4863646902834367] | 0.5636644068549543 | false | **drop** | **negative** |
| final | 92.36001923894966 | 92.94058874557216 | −0.5805695066224956 | [−1.1327474070824266, −0.04929360958760198] | 0.09463208972404402 | false | **drop** | **negative** |

**The negative branch fired on both readouts.** On the merge, the clean drop (0.56) would also have blocked a promotion.

Mechanism-level refutations (my reading of the evidence against the design's own conditions; the verdict above is the supervisor's):

| refutation | evidence (full sets unless stated; merge, ep8 in brackets) | fired? |
|---|---|---|
| 1. SNR range not the limit | Δ −1.06, not within ±0.5. SNR-0 R_bona: matched 68.38 (+2.56, within ±3), heldout 61.22 (−4.08, outside) | **no, by its letter** (the delta left the null band, downward). In substance, SNR 0 did not improve: matched within ±3, heldout down, half_B SNR-0 FA 40 → 44 |
| 2. covered, but traded away | SNR-0 R_bona did **not** rise ≥ 3 (+2.56 / −4.08). 15-20 dB R_bona fell > 2 on all four slices (matched −3.67 / −2.36, heldout −2.65 / −4.17). Clean drop 0.56 | **the "traded away" half fired, the "covered" half did not.** The high-SNR mass and clean paid, without the SNR-0 gain |
| 3. 0 dB masks the spoof cue | SNR-0 R_spoof 98.48 / 98.95 vs 98.65 / 98.50 (−0.17 / +0.45). half_B R_spoof matched 99.44 (+0.06), heldout 99.50 (+0.44) [ep8: 99.44 vs 99.69 (−0.25), 99.50 vs 99.00 (+0.50)] | **no**: cleanly excluded |
| 4. B3d-like under-fit | Epoch-1 train acc 89.26 (−0.90; leg **not** met). Epoch-8 CE 0.0426, above 1.5 × 0.0283 (ratio 1.51; leg met). Every set down at the merge (all four half_B F1 deltas negative) with R_bona down > 2 on every 15-20 dB slice (leg met) | **2 of 3 legs, so not by its letter.** The run is a milder version of the B3d pattern: a fit that weakens later in training rather than an epoch-1 collapse |

The predicted signature, item by item (merge vs B3a-top5; ep8 vs B3a-ep8 in brackets):

| predicted | observed | appeared? |
|---|---|---|
| SNR 0 matched: R_bona ≥ 3 above 65.81, EER below 14.67 | 68.38 (+2.56), EER 15.94 | **no** (R_bona close; EER the opposite) |
| SNR 0 heldout: R_bona ≥ 3 above 65.31, EER below 15.08 | 61.22, EER 18.92 | **no**, the opposite |
| SNR 0 R_spoof within 0.5 of 98.65 / 98.50 | 98.48 / 98.95 | yes |
| SNR 5: matched / heldout EER ≤ 10.10 / 10.11, R_bona not below 77.12 / 77.56 | EER 10.10 / 11.24; R_bona 77.12 / 70.51 | matched yes; **heldout no** |
| SNR 15-20 R_bona within 2 of B3a (matched 82.57 / 85.83, heldout 74.17 / 79.86) | 78.90 / 83.46, 71.52 / 75.69 | **no** on all four |
| half_B SNR-0 noisy bona FA below 40 of 123 | 44 (42 vs 40) | **no** |
| total noisy bona FA below 180 of 782 | 212 (202 vs 183) | **no** |
| noisy bona share at p ≥ 0.99 not above 17.6 % | 18.8 % (21.0 % vs 19.8 %) | **no** |
| matched and heldout half_B F1 up, heldout above 90.20 | 91.24 / 88.52 (91.43 / 89.34) | **no** |
| mean noisy EER below 9.16 | 9.77 (9.84 vs 9.01) | **no** |
| clean F1 within 0.5 of 97.48 | 96.92 (96.97 vs 97.06) | **no** on the merge (−0.56); yes on ep8 (−0.09) |
| echo within its noise band | F1 −0.53 [−1.37, +0.36] (−0.55 [−1.49, +0.46]); EER 4.61 → 6.25 | F1 yes (CI includes 0); EER worse |
| epoch-1 train acc within 2 of 90.16; CE near 0.3945, slightly higher | 89.26; 0.4165 | yes |
| epoch-8 CE below 1.5 × 0.0283 | 0.0426 (ratio 1.51) | **no** (just above) |
| mini WF1 not below B3a's at most epochs | below at 5 of 8 (2, 5, 6, 7, 8) | **no** |
| merge half_B WF1 above 93.06 | 91.99 | **no** |

Only spoof recall, the epoch-1 checks and echo F1 matched the prediction. Every bona-recall, false-alarm, set-level and headline item went the other way.

## 7. Decision and next-experiment proposals

**Decision.** Per `verdict.json`, **B3e is not a new best**: `new_best false` and `decision drop` on both readouts, verdict `negative`.
- The reference stays **B3a** (merge 93.06, final 92.94; registry `B3a-cjs03-screen` / `B3a-cjs03-ep8`).
- B3e's registry rows are `B3e-snr0to20-screen` (half_B 91.99, CI −1.65 / −0.49) and `B3e-snr0to20-ep8` (92.36, CI −1.13 / −0.05), both `drop`.
- `--aug_snr_db 0 20` should not enter the recipe.
- This was iteration 5 of `max_iterations 5`, so the wave ends here. The proposals below are for the wave summary and the next wave.

Every proposal below is one factor (or one coupled group) against **B3a's flags**: B2f + `--consistency js --consistency_lambda 0.3 --consistency_pairs both --batch_size 8`, sampler `shuffle`, `--ce_weights 0.1 0.9`, `--aug_snr_db 5 20` uniform, `--aug_codec_profiles fixed`, `--aug_max_stages 1`. Each keeps the ~8 h runtime. Every falsifier uses the wave rule against B3a-top5 = 93.06 on half_B:
- promote: Δ merge > +0.5 with the CI excluding 0 and a clean drop of at most 0.5;
- null: Δ within ±0.5;
- negative: Δ below −0.5.

**The largest remaining error** is unchanged: B3a-top5's confident noisy bona false alarms.
- 180 of 782 half_B noisy bona (23.0 %), 138 of them (77 %) at p ≥ 0.99.
- Heldout holds 98 of them: g726 36 / 129, gsm 38 / 127, g722 24 / 135. English heldout bona are flagged at twice the Chinese rate (49 / 129 vs 49 / 262).
- By SNR the errors are spread: 40 / 123 at 0 dB, but also 42 / 142 at 10 dB and 32 / 148 at 15 dB.
- No B3 lever shrank the saturated part. B3e has 147 saturated against B3a's 138.

What the five B3 iterations rule out:
- B3b: exposure and the operating point.
- B3c: more invariance on the same augmentation distribution.
- B3d: Opus-parameter coverage, which moves training away from the eval distribution.
- B3e: SNR coverage. Extending training onto the eval SNR grid helped only the in-pool 0-dB slice a little and cost the unseen conditions everywhere.
- Common thread: **every input-side change that made training harder cut the late-training fit (CE 1.51× here, 2.77× in B3d) and lowered heldout bona recall most.**

**Ranked proposals (configuration first, seed repeat last):**

1. **Layer-wise LR decay: `--llrd 0.9`** (lever_queue `llrd_0.9`; `wave.json` lever_notes: "0.9 is the proposed value").
   - Targets: heldout ranking, the set that carried 55 % of B3e's merge loss and 61 % of B3d's. Within heldout it targets the unseen-condition slices: English R_bona 60.85, wind 71.05, g726 71.14, gsm 72.76 (B3a-top5 full sets).
   - Hypothesis: robustness to unseen noise, DSP and codecs lives in the pretrained lower layers of the Qwen3-ASR encoder. Every B3 change that altered the training signal moved heldout most. `--llrd 0.9` changes no input, so it cannot create a train/eval mismatch the way B3d and B3e did. The bottom layer trains at 0.9^23 = 0.089× the LR and the stem at 0.9^24 = 0.080× (computed from the `train.py` help formula `--lr * llrd^(L-1-i)`, L = 24). That slows the lower-layer drift that dev loss shows from epoch 1, while the head and upper layers still adapt.
   - Predicted signature:
     - heldout half_B F1 above 90.20, EER below 10.42, R_bona above 74.94;
     - dev loss rising less than B3a's +74 % from epoch 1;
     - clean within 0.5 of 97.48;
     - merge above 93.06.
   - Falsifier:
     - Δ within ±0.5 and heldout EER within ±0.3 of 10.42 means lower-layer drift is not the limit.
     - Δ below −0.5 with epoch-8 CE above 1.5 × 0.0283 means the lower layers must adapt: an under-fit like B3d/B3e, through the optimiser instead of the data.

2. **SpecAugment: `--specaug`** (lever_queue `specaug`; default masks `--specaug_time 2 40 --specaug_freq 2 16`).
   - Targets: the saturated tail and overconfidence.
     - B3a-top5 has 9.2 % of half_B noisy bona at p ≥ 0.999 and 17.6 % at ≥ 0.99.
     - Dev loss rises from epoch 1 in every B3 run (B3a +74 %, B3e +89 %).
   - Hypothesis: time/frequency masking on the log-mel input regularises without adding a channel or noise condition. B3d's codec profiles and B3e's 0-dB mixtures both added such a condition, and both moved bona toward spoof. Masks hit both classes the same way, so no new artefact can become class-owned. The saturated tail should thin.
   - Predicted signature:
     - epoch-8 dev loss below 0.1722;
     - half_B noisy bona ≥ 0.999 share below 9.2 % and ≥ 0.99 share below 17.6 % (merge);
     - noisy EER below 9.16;
     - clean within 0.5 of 97.48; R_spoof within 0.5;
     - epoch-1 train acc not more than 2 below 90.16, and epoch-8 CE not above 1.5 × 0.0283.
   - Falsifier:
     - Δ within ±0.5 and epoch-8 dev loss ≥ 0.17 means masking does not regularise this backbone.
     - Δ below −0.5 with epoch-8 CE above 1.5 × 0.0283 and every set down means masked inputs under-fit like B3d/B3e at this budget. Close the input-side levers for the full run.
   - Ranked second because B3d and B3e both show that a harder training task costs fit at 8 epochs.

3. **Clean teacher view: `--consistency_clean_view`** (one flag inside the consistency coupled group; js, 0.3, both and `--batch_size 8` unchanged).
   - Targets: noisy bona flipped by degradation. B3a-top5 half_B bona recall is 92.34 on clean but 79.03 / 74.94 on matched / heldout.
   - Hypothesis: with this flag, view a is the un-augmented clip and acts as a stop-gradient teacher. `data.py` featurises the raw audio for view a; `losses.py` detaches view a in JS. The augmented view is then pulled toward the clean view's prediction, which is mostly correct for bona, instead of both views meeting halfway as in B3a/B3c. This attacks the bona → spoof drift directly, without changing the augmentation distribution of view b.
   - Risk (from the code): CE is taken over both views (`train_epoch` docstring), so half the CE views become raw clips with no codec. No local eval set contains raw offline audio, so this is partly a move off the eval support.
   - Predicted signature:
     - half_B noisy bona FA below 180 with the saturated count below 138;
     - heldout R_bona above 74.94;
     - R_spoof within 0.5 on matched / heldout;
     - clean within 0.5 of 97.48;
     - merge above 93.06.
   - Falsifier:
     - Δ within ±0.5 with noisy bona FA within ±10 of 180 means the teacher does not transfer bona evidence to degraded views.
     - Δ below −0.5 with matched / heldout EER up means halving the augmented CE exposure costs more than the teacher gives.

4. **Consistency weight to zero: `--consistency_lambda 0.0`** (a one-flag dose step down inside the consistency group; views, pairs, `--batch_size 8` and steps unchanged).
   - Targets: the recipe question still open. B3a's gain over B2f could come from the paired views and doubled steps rather than the JS term. B3c showed the dose response above 0.3 is negative. B3d and B3e showed JS grows with view disagreement (2.89× and 1.67× at epoch 8) without helping.
   - Predicted signature if JS is inert: merge within ±0.5 of 93.06 and noisy EER within ±0.3 of 9.16, with the logged (unweighted) JS higher than B3a's at most epochs.
   - Falsifier:
     - Δ < −0.5 with noisy EER ≥ 9.46 means JS at 0.3 is an active ingredient: keep it.
     - Δ > +0.5 means JS hurts at any weight here: drop it from the recipe.
   - Ranked below the error-targeting levers because its expected gain is small. It is mainly a recipe-clarifying control.

5. **Seed repeat of B3a: `--seed 1235`** (last).
   - Purpose: B3a's 93.06 is the only above-band result of the wave, on one seed. All four changes around it landed on −0.77 to −2.20 on half_B, while B3e's loss on the selection half A is only −0.49 / −0.20. The seed band of the consistency recipe is unmeasured. If B3a sits at the top of its seed band, every B3 comparison is biased negative.
   - Predicted signature: merge within ±0.5 of 93.06, with heldout F1 again above B2f-top5's 87.92 by more than 1.5.
   - Falsifier: a repeat merge below 92.48 (B2f-top5 91.98 + 0.5) means B3a's gain over B2f is inside seed noise. The B3b, B3c and B3e deltas (but not B3d's −2.20) would then have to be re-read against that band.

Deprioritised by this run's evidence:
- `--aug_max_stages 2`: it chains stages, a harder and off-support training distribution like B3d's and B3e's.
- `--aug_snr_dist beta` / other SNR ranges: B3e shows the SNR axis is not where the false alarms are. They are spread over every SNR, and 0-dB coverage did not transfer to unseen noises.
- `--use_level_aug`: the B3a merge's limiter probe flips only 2.67 % of bona. The B3e merge's level flips rose (2.34 mean) but stay small.

**Recommended full run if the wave stopped now.** Unchanged from the B3a-B3d reports:
- Flags: B3a's exact `config.yaml` command (`scripts/run_B3a.sh`; `--aug_snr_db` left at `5 20`, `--aug_codec_profiles` at `fixed`), at 2-3 seeds (1234, 1235, 1236).
- Readout to submit: the top-5 merge. It is the wave objective and B3a's best readout (93.06 vs 92.94 on half B).
- Pre-launch checks: `bash -n scripts/run_B3a.sh`, `which ffmpeg`, `pyloudnorm` importable, an empty `pgrep -af "b3wave.train|main_train"`, an idle GPU.
- Post-launch checks:
  - `SNR: 5..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: fixed` (B3a's `rtc_augmentation` field reads `snr=5..20 dB`);
  - `Consistency: js lambda_js=0.3 lambda_emb=0.3 pairs=both … 'found': 37125 … 16 clips per step (8 pairs)`;
  - `Sampler: shuffle`, `CE weights: spoof=0.1 bonafide=0.9`;
  - `Warmup-cosine: 3790 warmup / 75792 steps`;
  - epoch-1 values at B3a's level, `TrainAcc=90.16% CE=0.3945 JS=0.0344` with mini WF1 90.17 (B3a `train.log`). Train acc near 85 % (B3d) or CE drifting above B3a's by epoch 5 (B3e, 1.26×) signals the wrong augmentation;
  - codec counts summing to 2 × 75,785.
- Every evaluation needs `--loudness_norm --loudness_norm_lufs -23`.
