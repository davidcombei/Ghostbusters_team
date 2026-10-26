# Experiment exploration B3-04: B3d (`B3d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_codecrand_s1234`) - random Opus codec profiles (`--aug_codec_profiles random`) on top of B3a

> Written 2026-10-11 by the exploop B3 analysis agent (ANALYZE step, wave B3, iteration 4, attempt a0).
>
> **Sources.** Every number below is either copied from a file read in this session or computed in this session from the half_B (and, where stated, half_A) rows of the local `scores/*.txt` / `*.logits` files, joined with the protocol and meta files of the local sets. Nothing is quoted from memory. The files:
> - run config and logs: `config.yaml` (B3d and B3a, `diff`), `git.diff` (B3d; file headers only), `logs/train.log` (both runs), `metrics.jsonl` (both runs), `top5_by_wf1/{select_topk.log,merge_list.txt}` (both runs), `local_eval/merge_top5/merge_list.txt`, the `ckpt/` listing;
> - local evals: `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,compare_vs_B3a-*.txt,ci_vs_B3a-*.txt,compare_ci_vs_B3a-*.json}`, `local_eval/merge_top5/compare_all_readouts.txt`; the same `metrics.json` / `probe_report.json` / `scores/` of B3a (`merge_top5`, `epoch_8`); the `half_B.wf1` block of the B2f, B3b and B3c `metrics.json` files;
> - stdout log: `SLS_setup/B3d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_codecrand_s1234_20261011081939.log` (header lines and grep counts only);
> - loop files: `exploop/B3/iter04/{verdict.json,design.result.json}`, `exploop/B3/{wave.json,plan_snapshot.md,STATUS.md,user_notes.md}`, `scripts/run_B3d.sh`;
> - registry and code: `docs/experiment_log.csv`, `SLS_setup/b3wave/train.py` (flag help texts), `SLS_setup/b3wave/augment.py` (`codecs_augm_random`), `SLS_setup/utils/rtc_augment.py` (`CODEC_PROFILES` lines 163-174, `codecs_augm` lines 183-194), `scripts/build_dev_noisy_sim.py` (lines 22, 52, 57, 183, 223), `SLS_setup/utils/local_eval.py` (half filter), `SLS_setup/utils/metrics.py` (macro-F1, spoof if `p >= thr`), `configs/local_eval_sets.yaml`, `data/dev_noisy_sim/*/{protocol.txt,meta.csv}`.
>
> Progress-split scores were not read (`model_merging_top5/scores.txt` was not opened).
>
> **Previous reports.** `docs/experiments-exploration/b3-experiments/20261011_075030_B3c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs10_s1234.md` (iteration 3; "B3c report"), `.../20261010_231745_B3b_..._sampbal_s1234.md` (iteration 2), `.../20261010_143058_B3a_..._cjs03_s1234.md` (the reference run), and `docs/experiments-exploration/20261009_175355_B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234.md` (template, fixed comparator).
>
> **Readouts.**
> - `-top5` is the weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5/`). It is the wave objective.
> - `-ep8` is the final-epoch EMA checkpoint alone (`local_eval/epoch_8/`).
> - half_B is the half of each sim set that is never used for selection: n = 1,994 per set, 391 bona (`metrics.json`). Clean is the full 7,465-clip `dev_online_clean` (1,397 bona), which has no `half` column.
> - WF1 = 0.3 clean + 0.35 matched + 0.35 heldout (macro-F1 at p_spoof 0.5). Echo is reported but is not in WF1.
> - Threshold 0.5 on the score files reproduces the four half_B WF1 values of the `metrics.json` files exactly: B3a 93.06 / 92.94, B3d 90.86 / 90.72 (top5 / ep8; computed).
> - Every evaluation ran with loudness normalisation: every `metrics.json` has `"loudness_norm": -23.0`.

Run directories (under `SLS_setup/exp/`):
- **B3d** `B3d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_codecrand_s1234_epoch8_bs8_20261011081946` (this report)
- **B3a** `B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_epoch8_bs8_20261009214846` (reference = current best: merge 93.06, final 92.94)
- **B3b** `B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_sampbal_s1234_epoch8_bs8_20261010145615` (iteration 2, dropped: 92.23 / 92.36)
- **B3c** `B3c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs10_s1234_epoch8_bs8_20261010234200` (iteration 3, dropped: 92.29 / 92.44)
- **B2f** `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623` (fixed comparator: 91.98 / 92.14)

## 1. Setup

**B3d = B3a with one flag added: `--aug_codec_profiles random`** (`design.result.json` `change`). Everything else is identical, including `--consistency js --consistency_lambda 0.3 --consistency_pairs both --batch_size 8`.

**What changed in training.** Every training clip still goes through one of the 7 Opus app profiles (`then codec p=1.0`), keeping the app's cutoff and application (`train.py` help). Only the encoder parameters differ:

| parameter | fixed (B3a; `rtc_augment.py` `CODEC_PROFILES` + `codecs_augm`) | random (B3d; `augment.py` `codecs_augm_random`) |
|---|---|---|
| bitrate | per app: qq 10-24, wechat 8-20, zoom 24-48, dingtalk 16-32, lark 24-40, voov 16-32, telegram 16-32 kbps | {6, 8, 12, 16, 24} kbps for every app |
| frame size | 20 ms (dingtalk 20/40, telegram 20/60) | {20, 40, 60} ms |
| VBR mode | not set (encoder default; no `-vbr` in `codecs_augm`) | off / on / constrained |
| packet-loss hint | not set (no `-packet_loss`) | 0 w.p. 0.5, else U{5..20} % |

The design's `derivation`, computed from the two tables:
- share of codec draws at or below 12 kbps: fixed 0.084, random 0.600;
- share above 24 kbps: fixed 0.473, random 0.000;
- mean bitrate: fixed 24.43 kbps, random 13.20 kbps.

**The local sim sets use the fixed table.** `scripts/build_dev_noisy_sim.py` imports `codecs_augm` from `utils.rtc_augment` (line 52) and calls it at lines 183 and 223; its header comment (line 22) calls this chain "== today's training sim". The heldout set swaps in the reserved codecs `("g722", "g726", "gsm")` (line 57). B3d therefore trained on a codec distribution that **no longer matches** the one `sim_matched_v1` and `sim_echo_v1` were built with.

Flag diff (`config.yaml` of both runs, `diff` output; nothing else differs):

| flag / field | B3a | **B3d** | note |
|---|---|---|---|
| `--aug_codec_profiles` | absent (`args.aug_codec_profiles: fixed`) | **`random`** | the lever |
| `rtc_augmentation` header | `... then codec p=1.0 [qq, wechat, zoom, dingtalk, lark, voov, telegram]` | same + `\| codec_profiles=random kbps=[6, 8, 12, 16, 24] frames=[20, 40, 60] vbr=rand loss=0\|5-20` | the augmenter really changed |
| `--track` | `..._cjs03_s1234` | `..._cjs03_codecrand_s1234` | name only |
| git | `df3dff2`, `git_dirty: false` | `df3dff2`, `git_dirty: true` | `git.diff` touches only `.claude/skills/post-train-eval/SKILL.md` and `docs/experiment_log.csv`; no trainer code |
| everything else | identical | identical | `--consistency js --consistency_lambda 0.3 --consistency_pairs both`, `--batch_size 8` (16 clips/step), 8 epochs, `--lr 2.5e-6 --lr_min 5e-7` warmup-cosine, `--ce_weights` default 0.1 0.9, sampler shuffle, RTC aug + MUSAN, SNR 5-20, `--loudness_norm -23`, `--keep_topk_by_wf1 7 --keep_epochs 8`, seed 1234 |

**Header lines proving the change took effect** (stdout log). They match all five `expect` regexes of the design:
- `SNR: 5..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: random`
- `RTC augmentation: p_apply=0.8 one stage/clip, ... then codec p=1.0 [qq, wechat, zoom, dingtalk, lark, voov, telegram] | codec_profiles=random kbps=[6, 8, 12, 16, 24] frames=[20, 40, 60] vbr=rand loss=0|5-20`
- `Consistency: js lambda_js=0.3 lambda_emb=0.3 pairs=both clean_view=False | pairs found {'rows': 38660, 'blank_online': 1535, 'found': 37125} dropped_without_twin=0 | 16 clips per step (8 pairs)`
- `Sampler: shuffle`
- `CE weights: spoof=0.1 bonafide=0.9`
- Unchanged: `Warmup-cosine: 3790 warmup / 75792 steps, floor 5.0e-07`, `LLRD: 1`, `SpecAugment: off`, `Loudness norm: loudness_norm target=-23 LUFS`, `Train trials: 75785`, `Parameters: 320268933`.

**Telemetry** (`train.log` `B3 aug` lines and `metrics.jsonl` `b3.aug_counts`, computed):
- The codec counts sum to 151,570 = 2 × 75,785 in every epoch (two views per clip).
- Per-app shares are 0.1409-0.1454 at epoch 1 and 0.1416-0.1438 at epoch 8, against 1/7 = 0.1429: the 7 apps stay balanced.
- The stage counts are 0.7981-0.8006 of the views (`p_apply=0.8`).
- The stdout log has no `Traceback`, `out of memory`, `failed (`, `random profile`, `Error` or `WARNING` line (grep count 0 each). The one `Warning:` line is the HF Hub unauthenticated-request notice. The one `Pruned` line is epoch 2.

**Run time.**
- `config.yaml` created 08:19:55; epoch 8 done 16:06:45; last local eval 16:07:07. That is **7 h 47 min** (registry `gpu_h` 7.79; B3a 7.87).
- Epoch-end to epoch-end spacing is 57.8-58.7 min, against B3a's 58.3-59.5 (`train.log`, computed). The random profile costs no time.
- Post-train file times (`ls`): `top5_by_wf1/merge_list.txt` 16:08, `merge_top5/metrics.json` 16:13, `epoch_8/metrics.json` 16:18, `model_merging_top5/` 16:29. `verdict.json` was written at 16:30:14.

**Checkpoints.**
- Epoch 2 was pruned (`Pruned epoch_2_dev_loss_0.120876.pth (WF1=87.68, outside top-7)`). The other 7 epoch files are present at 1,281,235,691 bytes each, plus `last_state.pt`.
- Symlinks: `best_by_dev_loss` → epoch 1 (0.081188), `best_by_noisy_eer` → epoch 8, `best_by_wf1` → epoch 7.

**Merged top-5 by half-A mini WF1 = epochs 7, 6, 8, 3, 4** (90.29, 89.86, 89.74, 89.58, 89.06; mean 89.71; `select_topk.log`).
- Excluded: 5 (89.05, 0.01 below epoch 4), 1 (88.57), 2 (87.68).
- B3a's merge was 6, 5, 7, 8, 2 (mean 92.31; B3a `select_topk.log`).

## 2. Results vs references (half_B)

Sources: `compare_all_readouts.txt`, `compare_vs_B3a-*.txt`, `ci_vs_B3a-*.txt` / `compare_ci_vs_B3a-*.json` (paired bootstrap n = 2,000, half B, threshold 0.5), and the `half_B.wf1` block of each run's `metrics.json`.

### 2.1 Wave iterations and fixed comparators, both readouts

| half_B | B2f-top5 | B2f-ep8 | B3a-top5 | B3a-ep8 | B3b-top5 | B3b-ep8 | B3c-top5 | B3c-ep8 | **B3d-top5** | **B3d-ep8** | B3d-top5 − B3a-top5 [95 % CI] | B3d-ep8 − B3a-ep8 [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **WF1@0.5** | 91.98 | 92.14 | 93.06 | 92.94 | 92.23 | 92.36 | 92.29 | 92.44 | **90.86** | **90.72** | **−2.20 [−2.90, −1.57]**, p(not better) 1.000 | **−2.22 [−2.99, −1.52]**, p 1.000 |
| WF1@oracle | 92.59 | 92.82 | 93.24 | 93.45 | 92.96 | 92.86 | 92.63 | 92.73 | 92.00 | 91.84 | −1.24 | −1.61 |
| clean F1@0.5 | 97.60 | 97.69 | 97.48 | 97.06 | 96.96 | 96.77 | 97.25 | 97.20 | 96.66 | 96.17 | **−0.83 [−1.18, −0.49]** | **−0.89 [−1.29, −0.48]** |
| noisy F1@0.5 | 89.57 | 89.76 | 91.16 | 91.17 | 90.20 | 90.47 | 90.17 | 90.40 | 88.38 | 88.38 | −2.78 | −2.79 |
| mean noisy EER | 10.09 | 9.66 | 9.16 | 9.01 | 10.15 | 10.17 | 9.96 | 9.71 | **11.05** | **10.99** | **+1.89** | **+1.97** |
| calibration gap noisy | 0.66 | 0.71 | 0.11 | 0.42 | 0.62 | 0.21 | 0.28 | 0.12 | **1.35** | **1.14** | +1.24 | +0.72 |
| separability gap | 7.87 | 7.86 | 6.56 | 6.20 | 7.12 | 7.27 | 7.29 | 7.36 | 7.58 | 7.73 | +1.02 | +1.53 |

B3d is the lowest readout of the wave on every row except the separability gap. Both readouts are also below the fixed comparator B2f: −1.12 (merge) and −1.42 (ep8) WF1, computed from `compare_all_readouts.txt`.

Per-set paired CIs (`ci_vs_B3a-*.txt`). **Every CI excludes 0, in both readouts:**

| set | merge Δ [CI] | ep8 Δ [CI] |
|---|---|---|
| clean | −0.83 [−1.18, −0.49] | −0.89 [−1.29, −0.48] |
| matched | −1.72 [−2.97, −0.58] | −1.95 [−3.27, −0.58] |
| heldout | **−3.84 [−5.39, −2.44]** | **−3.64 [−5.26, −2.11]** |
| echo | −1.49 [−2.50, −0.56] | −1.56 [−2.64, −0.48] |

Weighted contributions (0.3 / 0.35 / 0.35 × the unrounded deltas in `compare_ci_vs_B3a-*.json`, computed):
- merge: −0.25 clean − 0.60 matched − 1.35 heldout = −2.20. Heldout carries 61 %, matched 27 %, clean 11 %.
- final: −0.27 − 0.68 − 1.27 = −2.22. Heldout 57 %, matched 31 %, clean 12 %.

Heldout again moved most. It also carried three quarters of B3a's gain over B2f and of B3c's loss (B3c report §2.1).

**Other halves of the same sets (the loss is not specific to half B):**

| readout | full-set WF1 B3a → B3d (`metrics.json` `full.wf1`) | half_A WF1 B3a → B3d (computed from `scores/*.txt`) | half_A matched / heldout F1 B3a → B3d (computed) |
|---|---|---|---|
| merge | 93.05 → 90.61 (−2.44) | 93.04 → 90.33 (**−2.71**) | 92.45 → 89.54 / 89.82 → 85.69 |
| final | 92.78 → 90.37 (−2.41) | 92.61 → 89.98 (**−2.63**) | 91.55 → 89.00 / 89.85 → 85.65 |

`wave.json` lever_notes give the B2 near-replicate noise band as "~0.2-0.5 WF1 on the final readout, up to ~1.4 per set". B3d's loss is 4-5 times the upper end of that band, and it is larger on the selection half A than on half B. Unlike B3c's (B3c report §2.1), it does not depend on the half.

### 2.2 Per set (R = recall at 0.5; thr = p_spoof threshold; `metrics.json` half_B)

| set | metric | B3a-top5 | **B3d-top5** | B3a-ep8 | **B3d-ep8** |
|---|---|---|---|---|---|
| dev_online_clean | F1@.5 / F1@or / thr_or | 97.48 / 97.83 / 0.996 | **96.66** / 97.30 / 0.998 | 97.06 / 97.79 / 1.000 | **96.17** / 97.26 / 0.999 |
| | EER / ECE | 3.32 / 1.47 | **3.92** / 1.93 | 3.20 / 1.72 | **3.78** / 2.20 |
| | R_spoof / R_bona | 99.93 / 92.34 | 99.93 / **89.84** | 99.92 / 91.12 | 99.95 / **88.33** |
| sim_matched_v1 | F1@.5 / F1@or / thr_or | 92.12 / 92.25 / 0.995 | **90.39** / 91.11 / 0.927 | 92.03 / 92.49 / 0.998 | **90.08** / 90.89 / 0.992 |
| | EER / ECE | 7.89 / 4.51 | **9.79** / 5.31 | 7.74 / 4.55 | **10.04** / 5.61 |
| | R_spoof / R_bona | 99.38 / 79.03 | 99.06 / **75.45** | 99.69 / 77.75 | 99.13 / **74.42** |
| sim_heldout_v1 | F1@.5 / F1@or / thr_or | 90.20 / 90.30 / 0.971 | **86.36** / 88.34 / 0.978 | 90.32 / 90.69 / 0.993 | **86.68** / 88.15 / 0.996 |
| | EER / ECE | 10.42 / 5.54 | **12.31 / 7.14** | 10.29 / 5.57 | **11.94 / 7.15** |
| | R_spoof / R_bona | 99.06 / 74.94 | 99.38 / **64.45** | 99.00 / 75.45 | 99.38 / **65.22** |
| sim_echo_v1 | F1@.5 / F1@or / thr_or | 94.87 / 95.03 / 0.998 | **93.38** / 94.09 / 0.987 | 94.39 / 94.96 / 0.999 | **92.83** / 93.51 / 0.981 |
| | EER / ECE | 4.61 / 3.00 | **6.44** / 3.82 | 5.62 / 3.28 | **6.95** / 4.20 |
| | R_spoof / R_bona | 99.81 / 85.17 | 99.56 / **81.84** | 99.94 / 83.38 | 99.50 / **80.56** |

Reading:
- **Bona recall fell on every set, in both readouts:** clean −2.51 / −2.79, matched −3.58 / −3.32, **heldout −10.49 / −10.23**, echo −3.32 / −2.81.
- **Spoof recall barely moved.** At the merge it is −0.31 on matched, +0.31 on heldout and −0.25 on echo. The largest drop is −0.56 (matched, ep8).
- **EER rose on every set** (+0.60 to +2.30). This is a ranking loss, not just a threshold move. ECE rose on every set too.
- Paired per clip on half_B, B3d vs B3a (computed; fixed / broken / wrong in both; by language: fixed / broken):

  | set | class | merge | merge en · zh | ep8 | ep8 en · zh |
  |---|---|---|---|---|---|
  | clean | bona | 6 / **41** / 101 | **0 / 23** · 6 / 18 | 14 / **53** / 110 | 6 / 17 · 8 / **36** |
  | matched | bona | 7 / 21 / 75 | 3 / 12 · 4 / 9 | 13 / 26 / 74 | 4 / 9 · 9 / 17 |
  | heldout | bona | 5 / **46** / 93 | 1 / 17 · **4 / 29** | 9 / **49** / 87 | 3 / 17 · **6 / 32** |
  | echo | bona | 6 / 19 / 52 | 4 / 11 · 2 / 8 | 10 / 21 / 55 | 6 / 7 · 4 / 14 |
  | matched | spoof | 6 / 11 / 4 | 2 / 4 · 4 / 7 | 2 / 11 / 3 | 1 / 5 · 1 / 6 |
  | heldout | spoof | 8 / 3 / 7 | 3 / 2 · 5 / 1 | 10 / 4 / 6 | 4 / 2 · 6 / 2 |
  | echo | spoof | 1 / 5 / 2 | 1 / 3 · 0 / 2 | 0 / 7 / 1 | 0 / 4 · 0 / 3 |

  **Summed over the four sets, bona: 24 fixed / 127 broken (merge) and 46 / 149 (ep8).** The one-sidedness is far stronger than B3c's (B3c report §2.2: 32 / 76 merge). The clean set alone breaks 41 bona clips at the merge, every English one of them broken and none fixed (0 / 23).
- Logit correlation B3d vs B3a (computed, half_B): 0.925-0.960 per set on the merge and 0.917-0.963 on ep8, lowest on heldout. B3d is a larger perturbation of B3a than B3c was (0.974-0.985) and similar to B3b (0.938-0.948; both from the B3c report §2.2).

### 2.3 Probes (`probe_report.json`; 187 bona + 813 spoof; one bona probe clip = 0.53 points)

| probe (bona unless stated) | B3a-top5 | **B3d-top5** | B3a-ep8 | **B3d-ep8** |
|---|---|---|---|---|
| level_flip_bona (mean) / max | 0.80 / 2.67 | **3.21 / 12.83** | 2.07 / 7.49 | **3.34 / 15.51** |
| gain_up / gain_p6 / gain_p10_limit flip % | 2.67 / 1.07 / 2.67 | 5.88 / 2.67 / **12.83** | 4.81 / 2.14 / 7.49 | 6.42 / 2.67 / **15.51** |
| gain_down / gain_m20 / gain_m6 flip % | 0.00 / 0.00 / 0.00 | 0.53 / 1.07 / 1.07 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| agc_dynaudnorm / loudnorm flip % | 0.00 / 0.00 | 0.00 / 1.60 | 0.00 / 2.14 | 0.53 / 1.60 |
| level_flip_spoof | 0.02 | 0.00 | 0.00 | 0.00 |
| probe_orig: bona pred spoof % | 12.30 | 13.37 | 11.76 | 14.97 |
| noise_only: bona-source pred spoof %, mean dP | 100.00, +0.876 | 100.00, +0.862 | 100.00, +0.882 | 100.00, +0.850 |
| sil_only: bona-source pred spoof %, mean dP | 100.00, +0.877 | 100.00, +0.864 | 100.00, +0.882 | 100.00, +0.851 |
| trimmed / padded bona flip % | 2.14 / 3.21 | 2.14 / 2.67 | 3.21 / 2.67 | 6.42 / 4.81 |

Reading:
- **Level robustness got worse.** The +10 dB-with-limiter probe flips 24 bona clips at the merge (12.83 %) against 5 for B3a (2.67 %), and 29 against 14 on ep8. Limiter and clipping artefacts now push bona toward spoof much more easily.
- `noise_only` / `sil_only` bona-source clips are still 100 % spoof in every readout, as in every B3 run.

### 2.4 Slices that moved most, B3d vs B3a (full sim sets, `metrics.json → breakdowns`)

Over 60 slice rows (the echo `aec` axis is left out because it duplicates the echo `family` axis; computed):

| readout | F1 up | EER down | R_bona up | R_spoof up / down | min slice R_spoof (B3a → B3d) |
|---|---|---|---|---|---|
| merge | **1 of 60** (echo opus:voov +0.34) | **2** (matched opus:zoom 7.90 → 6.44, coffee 10.27 → 9.75) | **1** | 22 / 32 | 97.02 (heldout washing_machine) → 97.47 (**matched snr 0**) |
| final | 1 of 60 (matched snr 15 +0.33) | 6 | 1 | 22 / 33 | 96.69 (heldout washing_machine) → **96.46 (matched snr 0)** |

**Codec slices, the target of the design** (full sets; R_bona / EER; the design's `predicted_signature` used the merge):

| set / codec | fixed-profile kbps | n (bona) | merge R_bona B3a → B3d | merge EER B3a → B3d | ep8 R_bona B3a → B3d | ep8 EER | merge R_spoof B3a → B3d |
|---|---|---|---|---|---|---|---|
| matched opus:qq | 10-24 | 572 (99) | 68.69 → **60.61** (−8.08) | 9.20 → **13.23** | 64.65 → 63.64 | 10.95 → 13.01 | 98.94 → 98.52 |
| matched opus:wechat | 8-20 | 585 (104) | 72.12 → **67.31** (−4.81) | 12.28 → 13.70 | 71.15 → 66.35 | 12.07 → 14.11 | 98.96 → 98.75 |
| matched opus:telegram | 16-32 | 552 (111) | 79.28 → 74.77 (−4.51) | 6.10 → 9.04 | 78.38 → 72.97 | 8.25 → 10.17 | 98.87 → 98.64 |
| matched opus:voov | 16-32 | 510 (89) | 84.27 → 79.78 (−4.49) | 6.93 → 10.28 | 80.90 → 74.16 | 6.58 → 9.81 | 99.76 → 99.05 |
| matched opus:dingtalk | 16-32 | 589 (122) | 84.43 → 79.51 (−4.92) | 6.38 → 9.00 | 82.79 → 79.51 | 6.70 → 10.16 | 99.14 → 99.57 |
| matched opus:lark | 24-40 | 608 (119) | 84.03 → **78.15** (−5.88) | 8.50 → 9.22 | 82.35 → 78.15 | 8.50 → 8.51 | 100.00 → 99.39 |
| matched opus:zoom | 24-48 | 584 (106) | 82.08 → **78.30** (−3.78) | 7.90 → **6.44** | 82.08 → 77.36 | 5.34 → 6.23 | 99.79 → 99.58 |
| heldout g722 | - | 1380 (258) | 81.01 → **72.87** (−8.14) | 6.96 → 8.63 | 80.23 → 72.48 | 6.56 → 8.04 | 99.38 → 99.47 |
| heldout g726 | - | 1274 (246) | 71.14 → **57.72** (−13.42) | 10.74 → **14.36** | 72.36 → 56.91 | 11.33 → 13.07 | 99.12 → 99.42 |
| heldout gsm | - | 1346 (246) | 72.76 → **61.38** (−11.38) | 13.91 → **15.36** | 73.58 → 63.41 | 13.43 → 15.88 | 98.09 → 98.64 |

- **The targeted low-bitrate apps lost the most bona recall in matched:** qq −8.08, against zoom −3.78. Zoom is the only matched codec whose EER improved.
- **All three unseen codecs lost 8-13 bona-recall points.** Low-rate Opus coverage did not transfer to narrowband G.72x / GSM; it made them worse.

Other large moves (full sets, merge unless stated; F1, EER, R_bona; B3a → B3d):

| set / slice | n (bona) | F1 (Δ) | EER | R_bona |
|---|---|---|---|---|
| heldout / door_wood_creaks | 361 (64) | 90.88 → 83.15 (−7.73) | 7.44 → 12.31 | 78.12 → **57.81** |
| heldout / wind | 381 (76) | 88.23 → 81.46 (−6.77) | 13.46 → 14.94 | 71.05 → 55.26 |
| heldout / lang en | 1538 (258) | 84.74 → 78.48 (−6.26) | 17.04 → 19.10 | 60.85 → **46.12** |
| heldout / snr 10 | 815 (152) | 91.13 → 85.82 (−5.31) | 10.29 → 10.62 | 78.29 → 63.82 |
| heldout / clock_alarm | 348 (70) | 91.28 → 86.13 (−5.15) | 11.11 → 14.88 | 77.14 → 64.29 |
| matched / snr 0 | 711 (117) | 86.15 → 81.38 (−4.77) | 14.67 → 15.77 | 65.81 → 58.12 (R_spoof 98.65 → 97.47) |
| matched / snr 0 (ep8) | 711 (117) | 86.73 → 80.49 (−6.23) | 16.37 → 14.67 | 66.67 → 58.97 (R_spoof 98.82 → **96.46**) |
| heldout / snr 0 | 812 (147) | 85.72 → 81.24 (−4.48) | 15.08 → **19.71** | 65.31 → 55.78 |
| heldout / lang zh | 2462 (492) | 92.54 → 89.45 (−3.09) | 7.82 → 9.19 | 82.52 → 73.58 |
| matched / lang en | 1538 (258) | 87.34 → 83.81 (−3.53) | 13.66 → 14.79 | 65.50 → 57.75 |

**Pattern: a broad decline, not a localised one.**
- Every axis lost: codec, SNR, noise family, language, source.
- Heldout lost most, and within heldout the unseen noise families and English lost most.
- The harshest matched condition (SNR 0) is the one place where spoof recall also fell by more than 1 point: 98.65 → 97.47 (merge) and 98.82 → 96.46 (ep8).

### 2.5 Lever response (one factor per row)

| run | change vs reference | half_B merge WF1 (Δ) | final WF1 (Δ) | merge oracle | final oracle | merge noisy EER | final noisy EER | merge clean | final clean | verdict (verdict.json / registry) |
|---|---|---|---|---|---|---|---|---|---|---|
| B2f | (wave start) | 91.98 | 92.14 | 92.59 | 92.82 | 10.09 | 9.66 | 97.60 | 97.69 | - |
| B3a | + `--consistency js --consistency_lambda 0.3 --consistency_pairs both`, `--batch_size 8` (vs B2f) | 93.06 (+1.08) | 92.94 (+0.80) | 93.24 | 93.45 | 9.16 | 9.01 | 97.48 | 97.06 | merge positive / promote |
| B3b | + `--sampler balanced --sampler_lang_balance --sampler_bona_frac 0.5 --ce_weights 0.3223 0.6777` (vs B3a) | 92.23 (−0.83) | 92.36 (−0.58) | 92.96 | 92.86 | 10.15 | 10.17 | 96.96 | 96.77 | negative / drop (both) |
| B3c | `--consistency_lambda 0.3 → 1.0` (vs B3a) | 92.29 (−0.77) | 92.44 (−0.50) | 92.63 | 92.73 | 9.96 | 9.71 | 97.25 | 97.20 | merge negative / drop; final null / drop |
| **B3d** | **+ `--aug_codec_profiles random` (vs B3a)** | **90.86 (−2.20)** | **90.72 (−2.22)** | 92.00 | 91.84 | 11.05 | 10.99 | 96.66 | 96.17 | **negative / drop (both)** |

**All three single-factor changes on top of B3a were negative, and B3d is the largest loss of the wave.** Its merge loss is 2.6-2.9 times B3b's (−0.83) or B3c's (−0.77) (computed).

## 3. Per-epoch behaviour, B3d vs B3a (same steps, same LR at each epoch end)

Sources: `metrics.jsonl` (EMA weights; 2,000 clean clips + `sim_matched_mini` + `sim_heldout_mini`, half A), `logs/train.log`, `STATUS.md` (Δ). CE weights are identical in the two runs, so CE, train acc and dev loss are directly comparable. JS is unweighted.

| ep | LR end | B3d mini WF1 | B3a | Δ | B3d WF1@or (B3a) | B3d EER noisy (B3a) | B3d R_bona cl/ma/he | B3a R_bona cl/ma/he | B3d thr_or ma/he | B3d CE / JS | B3a CE / JS | train acc B3d / B3a | dev loss B3d / B3a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.469e-6 | 88.57 | 90.17 | −1.60 | 89.00 (90.55) | 13.29 (13.14) | 90.4/69.3/62.8 | 91.0/73.7/65.8 | 0.428/0.832 | 0.4730 / 0.0431 | 0.3945 / 0.0344 | **84.75 / 90.16** | 0.0812 / 0.0990 |
| 2 | 2.289e-6 | **87.68** | 91.64 | **−3.96** | 89.89 (92.27) | 13.86 (11.56) | 88.2/65.9/**55.1** | 91.6/76.5/70.9 | 0.974/0.976 | 0.2980 / 0.0350 | 0.1867 / 0.0214 | 95.77 / 97.87 | 0.1209 / 0.1232 |
| 3 | 1.976e-6 | 89.58 | 91.39 | −1.81 | 90.64 (92.14) | 12.62 (11.49) | 88.2/72.6/61.2 | 91.6/76.5/66.8 | 0.965/0.967 | 0.2240 / 0.0279 | 0.1176 / 0.0151 | 97.02 / 98.63 | 0.1522 / 0.1514 |
| 4 | 1.583e-6 | 89.06 | 91.52 | −2.46 | 90.94 (92.13) | 12.99 (10.80) | 88.8/70.9/56.1 | 89.9/76.5/69.9 | 0.884/0.991 | 0.1649 / 0.0224 | 0.0842 / 0.0113 | 97.71 / 99.01 | 0.1737 / 0.1657 |
| 5 | 1.175e-6 | 89.05 | 92.52 | −3.47 | 90.58 (93.17) | 12.11 (11.16) | 86.8/69.8/60.2 | 91.6/81.0/71.9 | 0.982/0.992 | 0.1412 / 0.0189 | 0.0642 / 0.0080 | 98.23 / 99.32 | 0.1807 / 0.1472 |
| 6 | 8.227e-7 | 89.86 | **92.87** | −3.01 | 90.87 (93.51) | 13.06 (**10.09**) | 88.2/73.2/61.2 | 91.9/82.7/71.9 | 0.885/0.993 | 0.1123 / 0.0157 | 0.0452 / 0.0065 | 98.59 / 99.47 | 0.1775 / 0.1356 |
| 7 | 5.842e-7 | **90.29** | 92.36 | −2.07 | 91.15 (93.07) | 12.71 (10.40) | 89.6/73.2/61.2 | 91.3/80.4/71.4 | 0.988/0.990 | 0.0920 / 0.0128 | 0.0328 / 0.0051 | 98.92 / 99.59 | 0.1792 / 0.1584 |
| 8 | 5.000e-7 | 89.74 | 92.14 | −2.40 | 90.69 (93.03) | **11.91** (10.28) | 87.6/72.1/61.2 | 89.9/78.8/71.9 | 0.992/0.994 | **0.0784 / 0.0110** | 0.0283 / 0.0038 | **99.09 / 99.71** | **0.1874** / 0.1722 |

Reading:
- **B3d is below B3a at every epoch, from the first.**
  - The gap opens at −1.60 at epoch 1, peaks at −3.96 at epoch 2 (a dip below epoch 1) and is still −2.40 at epoch 8.
  - The heldout-mini bona recall stays at 55.1-62.8 against 65.8-71.9.
  - The per-epoch selection metric saw the regression all along. The result is not a selection or merge artefact.
- **The model fits the training data less well, at the same steps** (ratios computed from `train.log`).
  - CE is 1.20× B3a's at epoch 1, rising to 2.77× at epoch 8 (0.0784 vs 0.0283).
  - Unweighted JS is 1.25× → 2.89× (0.0110 vs 0.0038). Two views of a clip that draw independent random codec profiles disagree more.
  - Train acc is 84.75 vs 90.16 at epoch 1 and 99.09 vs 99.71 at epoch 8.
  - The JS share of the loss (0.3·JS / TrainLoss) is unchanged at 2.7-4.0 % vs 2.5-4.5 %, because CE and JS grew together.
- **Dev loss.**
  - It is lowest at epoch 1 (0.0812, below B3a's 0.0990) and then rises by 131 % to 0.1874 (B3a: +74 %).
  - DevAcc is below B3a's at every epoch: 97.79-97.98 against 98.33-98.70.
  - So the run is less fit on train and also worse on dev. That is the signature of a harder or mismatched training distribution, not of over-fitting.
- **The curve is flat and noisy.** The best epoch is 7 (90.29); epoch 8 is 89.74. Mini noisy EER is best at epoch 8 (11.91; `best_by_noisy_eer` → epoch 8) but stays above B3a's at every epoch (by 0.15 at epoch 1 and 1.63 at epoch 8).
- **Final vs merge.** The merge beats the final epoch on half B (90.86 vs 90.72) and on half A (90.33 vs 89.98, computed). This is unlike B3c, where the final epoch won on half B. No merged epoch was evaluated alone on half B.

## 4. What went OK (evidence)

1. **A clean, single-flag experiment with an unambiguous answer.**
   - Only `aug_codec_profiles` differs in `config.yaml`, and all five header regexes matched.
   - The 7 apps stay balanced (shares 0.1409-0.1454). There is no codec failure, traceback or `(random profile) failed` line, and no trainer code in `git.diff`.
   - The result is decisive in both readouts (CIs [−2.90, −1.57] and [−2.99, −1.52]).
   - It replicates on half A (−2.71 / −2.63) and on the full sets (−2.44 / −2.41), so it is not a half-B or seed-band effect.
2. **It closes the "Opus parameter coverage" hypothesis for the dominant error.**
   - Falsifiers 1 and 2 of the design are cleanly excluded (section 6): the run moved far outside the null band, and the seen low-bitrate codecs got worse, not better.
   - Codec coverage of this kind cannot reduce the qq / wechat / g726 / gsm false alarms at this budget.
3. **Spoof recall held on the objective readout.**
   - Set level, merge: matched −0.31, heldout +0.31, echo −0.25. Clean is unchanged (99.93).
   - Noisy spoof misses on half B: 25 → 25 at the merge, 21 → 24 at ep8 (computed).
   - Synthetic-speech cues survived 6-kbps Opus in training (except at matched SNR 0; section 5).
4. **The per-epoch selection metric tracked the half-B result.** Mini WF1 was below B3a's at all 8 epochs. Unlike B3c (B3c report §3), the cheap metric did not mislead here.
5. **No time cost:** 57.8-58.7 min per epoch, 7 h 47 min in total.

## 5. What went wrong (evidence)

### 5.1 Every set lost, heldout most, and the targeted slices got worse

- WF1 −2.20 (merge) / −2.22 (ep8), with the per-set CIs excluding 0 for all four sets (section 2.1).
- Bona recall: heldout −10.49 / −10.23, matched −3.58 / −3.32, clean −2.51 / −2.79, echo −3.32 / −2.81.
- The design targeted four codecs: g726, gsm, opus:qq and opus:wechat. Their full-set R_bona fell by 13.42, 11.38, 8.08 and 4.81 points (2.4).
- **Clean dropped 0.83 / 0.89 F1**, above the wave's 0.5 clean guard. The clean set is the `online/` dev clips (`dev_online_clean/protocol.txt`), which never pass through a simulated codec. Its English bona broke 23 times and were fixed 0 times at the merge (2.2).

### 5.2 Oracle-gap decomposition (global threshold vs set-specific)

Method: one shared logit threshold swept over the half_B `scores/*.logits` (step 0.05, −6 to +12), as in the B3a, B3b and B3c reports. The per-set oracle comes from `metrics.json` `wf1_oracle` (computed).

| readout | WF1@0.5 | best global (logit / p) | WF1 at best global | per-set oracle WF1 | gap | recovered by one global shift | set-specific remainder |
|---|---|---|---|---|---|---|---|
| B3a-top5 | 93.06 | −0.05 / 0.488 (flat; the B3c report found −0.15 for the same 93.08) | 93.08 | 93.24 | 0.18 | 0.02 | 0.16 |
| B3a-ep8 | 92.94 | +4.95 / 0.993 | 93.10 | 93.45 | 0.51 | 0.16 | 0.35 |
| **B3d-top5** | 90.86 | **+3.25 / 0.963** | **91.70** | 92.00 | **1.14** | **0.84** | 0.30 |
| **B3d-ep8** | 90.72 | +5.40 / 0.996 | 91.59 | 91.84 | 1.12 | 0.87 | 0.25 |

- **The loss has two parts.**
  - B3d's operating point moved toward bona-as-spoof: a global shift to p = 0.963 recovers 0.84 (merge).
  - Even at its best global threshold (91.70), B3d-top5 is 1.38 below B3a-top5 at its own (93.08): ranking.
  - Decomposed: 2.20 = 1.38 ranking + 0.84 operating point − 0.02 (B3a's own shift).
- The calibration gap noisy (`metrics.json`) went from 0.11 to 1.35 on the merge, the largest of the wave.

### 5.3 Noisy bona false alarms grew, in both the moderate and the saturated parts

Share of half_B bona with p_spoof ≥ 0.5 / ≥ 0.9 / ≥ 0.99 / ≥ 0.999 (computed):

| set (n bona) | B3a-top5 | **B3d-top5** | B3a-ep8 | **B3d-ep8** |
|---|---|---|---|---|
| clean (1,397) | 7.7 / 7.6 / 6.4 / 4.6 | 10.2 / 9.7 / 8.2 / 6.2 | 8.9 / 8.9 / 8.3 / 6.1 | 11.7 / 11.0 / 9.9 / 7.0 |
| matched (391) | 21.0 / 21.0 / 16.4 / 8.2 | 24.6 / 22.5 / 18.2 / 12.8 | 22.3 / 22.3 / 19.2 / 12.0 | 25.6 / 24.0 / 21.0 / 13.3 |
| heldout (391) | 25.1 / 25.1 / 18.9 / 10.2 | **35.5 / 32.5 / 25.1 / 14.8** | 24.6 / 24.3 / 20.5 / 13.0 | **34.8 / 33.2 / 27.1 / 17.1** |
| echo (391) | 14.8 / 14.8 / 12.3 / 5.4 | 18.2 / 17.1 / 14.6 / 8.4 | 16.6 / 16.4 / 15.3 / 9.5 | 19.4 / 19.2 / 16.4 / 9.5 |

Noisy bona (matched + heldout, 782):

| readout | false alarms (FA %) | FA at p ≥ 0.99 / in [0.5, 0.99) | share ≥ 0.99 | share ≥ 0.999 | saturated share of FA | FA en / zh | noisy spoof misses |
|---|---|---|---|---|---|---|---|
| B3a-top5 | 180 (23.0 %) | 138 / 42 | 17.6 % | 9.2 % | 77 % | 91 / 89 | 25 |
| **B3d-top5** | **235 (30.1 %)** | **169 / 66** | **21.6 %** | **13.8 %** | 72 % | **116 / 119** | 25 |
| B3a-ep8 | 183 (23.4 %) | 155 / 28 | 19.8 % | 12.5 % | 85 % | 93 / 90 | 21 |
| **B3d-ep8** | **236 (30.2 %)** | **188 / 48** | **24.0 %** | **15.2 %** | 80 % | 112 / 124 | 24 |

- 55 more false alarms at the merge: 31 saturated and 24 moderate. The moderate ones are what the global threshold shift of 5.2 recovers. The saturated ones are not recoverable by any threshold below p = 0.99.
- Unlike B3c, the extra errors are not mostly in the saturated tail. The saturated share of FA fell (77 % → 72 %) while the count rose.

**Where the false alarms sit, half_B noisy bona, B3a-top5 → B3d-top5** (computed from `scores/*.txt` + `meta.csv`):

| split | B3a-top5 | B3d-top5 | B3a-ep8 | B3d-ep8 |
|---|---|---|---|---|
| heldout g722 (135) | 24 | **36** | 26 | 36 |
| heldout g726 (129) | 36 | **53** | 36 | **56** |
| heldout gsm (127) | 38 | **50** | 34 | 44 |
| matched opus:qq (60) | 21 | 25 | 24 | 23 |
| matched opus:wechat (50) | 17 | 19 | 17 | 19 |
| other 5 matched apps (281) | 44 | 52 | 46 | 58 |
| **g726 + gsm** | 74 | **103** | 70 | 100 |
| **qq + wechat** | 38 | **44** | 41 | 42 |
| SNR 0 (123: matched 63, heldout 60) | 40 (21 + 19) | **55** (27 + 28) | 40 | 51 |
| heldout en / zh (129 / 262) | 49 / 49 | 65 / **74** | 48 / 48 | 62 / **74** |
| matched en / zh (129 / 262) | 42 / 40 | 51 / 45 | 45 / 42 | 50 / 50 |

The heldout codecs took 41 of the 55 extra merge false alarms (98 → 139); the matched apps took 14 (82 → 96).

### 5.4 Worst slices for bona recall, B3d-top5 (full sets, `metrics.json` breakdowns)

| slice | R_bona | R_spoof | EER | B3a-top5 R_bona |
|---|---|---|---|---|
| heldout en | **46.12** | 99.61 | 19.10 | 60.85 |
| heldout washing_machine | 53.85 | 98.34 | 18.72 | 66.67 |
| heldout wind | 55.26 | 98.69 | 14.94 | 71.05 |
| heldout snr 0 | 55.78 | 98.20 | **19.71** | 65.31 |
| heldout g726 | 57.72 | 99.42 | 14.36 | 71.14 |
| matched en | 57.75 | 99.45 | 14.79 | 65.50 |
| heldout door_wood_creaks | 57.81 | 98.99 | 12.31 | 78.12 |
| matched snr 0 | 58.12 | 97.47 | 15.77 | 65.81 |
| heldout vacuum_cleaner | 58.57 | 98.35 | 15.78 | 67.14 |
| matched opus:qq | 60.61 | 98.52 | 13.23 | 68.69 |

These are the same axes as in B3a, B3b and B3c (English, SNR 0, the heldout codecs and noise families), all lower.

### 5.5 Why it failed: mismatch and difficulty, not over-fitting (interpretation)

The evidence points to a training distribution that became harder **and** moved away from the evaluation distribution. Over-fitting does not fit it:
- **Harder:** CE 2.77× and JS 2.89× B3a's at epoch 8; train acc 99.09 vs 99.71; train acc 84.75 vs 90.16 already at epoch 1 (section 3). At 75,792 steps the model reached a worse fit of a harder task.
- **Mismatched:**
  - The matched and echo sets are encoded with the fixed `CODEC_PROFILES` (`build_dev_noisy_sim.py` lines 22, 52, 183, 223).
  - The random profile never draws above 24 kbps, where zoom (24-48) and lark (24-40) live. It draws ≤ 8 kbps 40 % of the time (design derivation), where only wechat's fixed range (8-20) reaches.
  - So the "matched" set stopped being matched. The clean `online/` clips never pass a simulated codec at all.
- **Not over-fitting:** dev acc is lower and train acc is lower than B3a's at every epoch. The dev-loss rise (+131 %) comes with a worse fit, not a better one.
- **Interpretation, not proven by these files:**
  - With both classes passed through 6-12 kbps Opus most of the time, the bona evidence the model relied on in degraded conditions became weaker.
  - The JS term at 0.3 then asks two views with independently drawn bitrates (for example 6 vs 24 kbps) to agree.
  - The run settled on a representation that pushes more degraded bona toward spoof. That is consistent with R_bona falling on every slice while R_spoof held.
  - The limiter probe (+10 dB, `gain_p10_limit` 2.67 → 12.83 %) shows the same sensitivity to new distortion artefacts.

## 6. Verdict on the pre-registered rule

From `exploop/B3/iter04/design.result.json`, the **falsifier**:

> "Wave rule against B3a-top5 = 93.06 on half_B:
> - Promote / positive: delta merge > +0.5, the paired-bootstrap CI excluding 0, and a clean F1 drop <= 0.5.
> - Null: delta within +-0.5.
> - Negative: delta below -0.5.
>
> Mechanism-level refutations:
> 1. Codec coverage is not the limit: delta within +-0.5 AND full-set g726 / gsm R_bona within +-3 of 71.14 / 72.76 AND qq / wechat R_bona within +-3 of 68.69 / 72.12. Move to the SNR range (snr_0_20) or SpecAugment.
> 2. Covers seen codecs but does not generalise: qq / wechat R_bona up >= 3 while g726 / gsm move less than 3. The heldout codec errors are not explained by Opus parameter coverage.
> 3. Blurs a spoof cue: R_spoof down > 0.5 on matched or heldout. Random low-rate profiles mask the synthesis artefacts.
> 4. Lost the high-bitrate mass: matched opus:zoom / opus:lark R_bona down > 3, or a clean F1 drop > 0.5. Removing every > 24 kbps draw hurt high-quality channels. A fixed/random mix would need code, so it is out of this wave."

`exploop/B3/iter04/verdict.json` (computed by the supervisor; quoted, not recomputed):

| readout | run | base | delta | CI | clean_drop | new_best | decision | verdict |
|---|---|---|---|---|---|---|---|---|
| merge (objective) | 90.86005229210235 | 93.05618353351335 | −2.1961312414109955 | [−2.9045067785726686, −1.5702440399509021] | 0.8273483686928813 | false | **drop** | **negative** |
| final | 90.7181471372302 | 92.94058874557216 | −2.2224416083419527 | [−2.9877017372502626, −1.5202101360673237] | 0.8885865097993246 | false | **drop** | **negative** |

**The negative branch fired on both readouts.** The clean drop (0.83 / 0.89) would also have blocked a promotion.

Mechanism-level refutations (my reading of the evidence against the design's own conditions; the verdict above is the supervisor's):

| refutation | evidence (full sets unless stated; merge, ep8 in brackets) | fired? |
|---|---|---|
| 1. coverage not the limit | Δ −2.20, not within ±0.5; g726 / gsm R_bona −13.42 / −11.38; qq / wechat −8.08 / −4.81 | **no.** The run went far outside the null band, and in the wrong direction |
| 2. covers seen codecs, no generalisation | qq / wechat R_bona went **down** (68.69 → 60.61, 72.12 → 67.31), not up ≥ 3 | **no**: neither seen nor unseen codecs improved |
| 3. blurs a spoof cue | half_B R_spoof matched 99.38 → 99.06 (−0.31), heldout 99.06 → 99.38 (+0.31) [ep8: matched 99.69 → 99.13, **−0.56**] | **not on the objective; by the letter on ep8 matched** (−0.56). Full-set matched SNR 0 R_spoof 98.65 → 97.47 [98.82 → 96.46] is the clearest local sign |
| 4. lost the high-bitrate mass | zoom R_bona 82.08 → 78.30 (−3.78), lark 84.03 → 78.15 (−5.88); clean drop 0.83 | **yes, by its letter.** But the loss is not specific to high-bitrate channels: low-bitrate qq lost more (−8.08) than zoom, and zoom is the only matched codec whose EER improved (7.90 → 6.44) |

The predicted signature, item by item (merge vs B3a-top5; ep8 vs B3a-ep8 in brackets):

| predicted | observed | appeared? |
|---|---|---|
| g726 / gsm R_bona ≥ 3 above 71.14 / 72.76, EER below 10.74 / 13.91 | 57.72 / 61.38, EER 14.36 / 15.36 | **no**, the opposite |
| qq / wechat R_bona ≥ 3 above 68.69 / 72.12, EER below 9.20 / 12.28 | 60.61 / 67.31, EER 13.23 / 13.70 | **no**, the opposite |
| g722 roughly flat (81.01, EER 6.96) | 72.87, EER 8.63 | **no** |
| half_B FA g726 + gsm below 74 | 103 | **no** |
| half_B FA qq + wechat below 38 | 44 | **no** |
| total noisy bona FA below 180 of 782 | 235 (236 vs 183) | **no** |
| noisy bona share at p ≥ 0.99 not above 17.6 % | 21.6 % (24.0 % vs 19.8 %) | **no** |
| heldout half_B F1 above 90.20, R_bona above 74.94 | 86.36 / 64.45 (86.68 / 65.22) | **no** |
| matched R_bona above 79.03 | 75.45 (74.42) | **no** |
| mean noisy EER below 9.16 | 11.05 (10.99 vs 9.01) | **no** |
| R_spoof within 0.5 of 99.38 / 99.06 | 99.06 / 99.38 (99.13 / 99.38 vs 99.69 / 99.00) | yes on the merge; ep8 matched −0.56 |
| clean F1 within 0.5 of 97.48 | 96.66 (96.17 vs 97.06) | **no** |
| zoom / lark R_bona within 3 of 82.08 / 84.03 | 78.30 / 78.15 | **no** |
| per-epoch codec counts balanced, no `(random profile) failed` lines | shares 0.1409-0.1454; grep count 0 | yes |
| merge half_B WF1 above 93.06 | 90.86 | **no** |

Only the telemetry items and spoof recall on the objective readout appeared. Every bona-recall, false-alarm, ranking and clean item went the opposite way.

## 7. Decision and next-experiment proposals

**Decision.** Per `verdict.json`, **B3d is not a new best**: `new_best false` and `decision drop` on both readouts, verdict `negative`.
- The reference stays **B3a** (merge 93.06, final 92.94; registry `B3a-cjs03-screen` / `B3a-cjs03-ep8`).
- B3d's registry rows are `B3d-codecrand-screen` (CI −2.90 / −1.57) and `B3d-codecrand-ep8` (−2.99 / −1.52).
- `--aug_codec_profiles random` should not enter the recipe. A fixed/random mix would need code (design falsifier 4), so it is outside this wave.

Every proposal below is one factor (or one coupled group) against **B3a's flags**: B2f + `--consistency js --consistency_lambda 0.3 --consistency_pairs both --batch_size 8`, sampler `shuffle`, `--ce_weights 0.1 0.9`, `--aug_codec_profiles fixed`. Each keeps the ~8 h runtime. This is iteration 5 of `max_iterations 5`, so the first proposal is the wave's last run. Every falsifier uses the wave rule against B3a-top5 = 93.06 on half_B:
- promote: Δ merge > +0.5 with the CI excluding 0 and a clean drop of at most 0.5;
- null: Δ within ±0.5;
- negative: Δ below −0.5.

**The largest remaining error** is unchanged: B3a-top5's confident noisy bona false alarms.
- 180 of 782 half_B noisy bona (23.0 %), 77 % of them at p ≥ 0.99.
- By codec: heldout g726 36 / 129 and gsm 38 / 127; matched opus:qq 21 / 60 and wechat 17 / 50.
- By SNR: SNR 0 holds 40 / 123. By language: English has 91 of the 180 while it is 258 of the 782.

The four B3 iterations narrow it down:
- B3b: it is not exposure or the operating point.
- B3c: it is not more invariance on the same augmentation distribution.
- B3d: it is not Opus-parameter coverage either. Moving the training codec distribution away from what the eval sets use cost 2.2 WF1 across the board.

**The lesson for augmentation levers:** extend training toward the evaluation support, never away from it. Watch the epoch-1 train acc and CE: B3d's 84.75 % / CE 0.4730 against B3a's 90.16 % / 0.3945 flagged the problem from the first epoch.

**Ranked proposals (configuration first, seed repeat last):**

1. **SNR range 0-20 dB: `--aug_snr_db 0 20`** (`wave.json` lever_queue `snr_0_20`; SNR coupled group, uniform distribution unchanged).
   - Targets: SNR 0, the highest-FA SNR, and the one condition the eval sets have but training never draws (`SNR: 5..20 dB` header; matched / heldout slices at 0, 5, 10, 15, 20).
     - B3a-top5: 40 of 123 half_B SNR-0 noisy bona are false alarms. Full-set R_bona is 65.81 / 65.31 and EER 14.67 / 15.08 (matched / heldout).
     - B3d made SNR 0 worse (55 false alarms; heldout EER 19.71).
   - Hypothesis: unlike B3d, this change extends the training distribution to cover the eval support rather than moving it off it. Only the noise stages' SNR range changes. 75 % of noise draws stay in 5-20 dB (computed), and the codec and noise pools are untouched. SNR-0 false alarms are out-of-range inputs that paired JS at 0.3 can then act on.
   - Predicted signature:
     - full-set SNR-0 EER below 14.67 / 15.08, and R_bona up ≥ 3 from 65.81 / 65.31 with R_spoof within 0.5 of 98.65 / 98.50;
     - half_B SNR-0 false alarms below 40;
     - 15-20 dB R_bona within 2 of B3a: matched 82.57 / 85.83, heldout 74.17 / 79.86;
     - clean within 0.5 of 97.48;
     - epoch-1 train acc within 2 points of B3a's 90.16, and epoch-8 CE below 1.5 × 0.0283;
     - merge above 93.06.
   - Falsifier:
     - Δ merge within ±0.5 and SNR-0 R_bona within ±3 means the SNR range is not the limit.
     - A B3d-like broad decline (R_bona down > 2 on 15-20 dB slices, epoch-8 CE above 1.5× B3a's, clean drop > 0.5) means any harder training distribution hurts at this budget. In that case stop the augmentation levers.

2. **SpecAugment: `--specaug`** (lever_queue `specaug`; default masks `--specaug_time 2 40 --specaug_freq 2 16`).
   - Targets: overconfidence and the dev-loss rise.
     - Dev loss rises from epoch 1 in every B3 run (B3a +74 %, B3d +131 %).
     - B3a's train acc reaches 99.71 %.
     - The ≥ 0.999 noisy bona share is 9.2 % on the B3a merge.
   - Hypothesis: time/frequency masking regularises without introducing a new channel artefact that one class could own, which is what B3d's codec profiles and the limiter probe show the model latches onto. It should thin the saturated tail.
   - Predicted signature:
     - epoch-8 dev loss below 0.1722;
     - half_B noisy bona ≥ 0.999 share below 9.2 % and ≥ 0.99 share below 17.6 % (merge);
     - noisy EER below 9.16;
     - clean within 0.5 of 97.48;
     - R_spoof within 0.5;
     - epoch-1 train acc not more than 2 points below 90.16.
   - Falsifier:
     - Δ merge within ±0.5 and epoch-8 dev loss ≥ 0.17 means masking does not regularise this backbone.
     - R_spoof down > 0.5 on matched or heldout means the masks remove a spoof cue.
     - A B3d-like under-fit (epoch-8 CE above 1.5× 0.0283 with every set down) means masked inputs are too hard at this budget.

3. **Layer-wise LR decay: `--llrd 0.9`** (lever_queue `llrd_0.9`; `wave.json` lever_notes: "0.9 is the proposed value").
   - Targets: heldout ranking, the set that carried 61 % of B3d's loss and 75 % of B3c's (B3c report). Within heldout it targets the unseen-condition slices: English R_bona 60.85, washing_machine 66.67, g726 71.14 (B3a-top5 full sets).
   - Hypothesis: robustness to unseen DSP and noise lives in the pretrained lower layers of the Qwen3-ASR encoder. Every B3 change that altered the training signal moved heldout most. With `--llrd 0.9` the bottom layer trains at 0.9^23 = 0.089× the LR and the stem at 0.080× (computed from the `train.py` help formula), so the lower-layer drift that dev loss shows from epoch 1 is slowed, while the head and upper layers still adapt.
   - Predicted signature:
     - heldout half_B F1 above 90.20, EER below 10.42 and R_bona above 74.94;
     - dev loss rising less than B3a's +74 % from epoch 1;
     - clean within 0.5;
     - merge above 93.06.
   - Falsifier:
     - Δ within ±0.5 and heldout EER within ±0.3 of 10.42 means lower-layer drift is not the limit.
     - Δ below −0.5 with epoch-8 CE above 1.5× 0.0283 means the lower layers must adapt: an under-fit like B3d's, through the optimiser instead of the data.

4. **Consistency weight to zero: `--consistency_lambda 0.0`** (a one-flag dose step down inside the consistency group; views, pairs, `--batch_size 8` and steps unchanged).
   - Targets: the recipe question still open from B3a (B3a's gain could come from the paired views and doubled steps rather than the JS term; B3c report §7).
     - B3c showed the dose response above 0.3 is negative.
     - B3d showed JS grows with view disagreement (2.89× at epoch 8) without helping.
   - Predicted signature if JS is inert: merge within ±0.5 of 93.06 and noisy EER within ±0.3 of 9.16, with the logged (unweighted) JS higher than B3a's at most epochs.
   - Falsifier:
     - Δ < −0.5 with noisy EER ≥ 9.46 means JS at 0.3 is an active ingredient: keep it.
     - Δ > +0.5 means JS hurts at any weight here: drop it from the recipe.
   - Ranked below the levers that target the error, because its expected gain is small. It is mainly a recipe-clarifying control.

5. **Seed repeat of B3a: `--seed 1235`** (last).
   - Purpose: B3a's 93.06 is the only above-band result of the wave, on one seed. B3b, B3c and B3d all land below it, and the seed band of the consistency recipe is unmeasured.
   - Predicted signature: merge within ±0.5 of 93.06, with heldout F1 again above B2f-top5's 87.92 by more than 1.5.
   - Falsifier: a repeat merge below 92.48 (B2f-top5 91.98 + 0.5) means B3a's gain over B2f is inside seed noise. The B3b / B3c deltas (but not B3d's −2.2) would then have to be re-read against that band.

Deprioritised by this run's evidence:
- `--aug_max_stages 2`: it chains stages, but the matched set is single-stage ("today's training sim"), so it repeats B3d's move away from the eval distribution.
- `--use_level_aug`: the level probes worsened in B3d, but on B3a the limiter probe flips only 2.67 % of bona.

**Recommended full run if the wave stopped now.** Unchanged from the B3a, B3b and B3c reports:
- Flags: B3a's exact `config.yaml` command (`scripts/run_B3a.sh`, `--aug_codec_profiles` left at `fixed`), at 2-3 seeds (1234, 1235, 1236).
- Readout to submit: the top-5 merge. It is the wave objective and B3a's best readout: 93.06 vs 92.94 on half B (B3c report: 93.04 vs 92.61 on half A).
- Pre-launch checks: `bash -n scripts/run_B3a.sh`, `which ffmpeg`, `pyloudnorm` importable, an empty `pgrep -af "b3wave.train|main_train"`, an idle GPU.
- Post-launch checks:
  - `SNR: 5..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: fixed`;
  - `Consistency: js lambda_js=0.3 lambda_emb=0.3 pairs=both … 'found': 37125 … 16 clips per step (8 pairs)`;
  - `Sampler: shuffle`, `CE weights: spoof=0.1 bonafide=0.9`;
  - `Warmup-cosine: 3790 warmup / 75792 steps`;
  - epoch-1 `TrainAcc≈90.2% CE=0.3945 JS=0.0344`-level values with mini WF1 ≈ 90.2 (B3a `train.log`). A train acc near 85 % at epoch 1, as in B3d, signals the wrong augmentation;
  - codec counts summing to 2 × 75,785.
- Every evaluation needs `--loudness_norm --loudness_norm_lufs -23`.
