# Experiment exploration B4-02: B4b (`B4b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_opusplc03_s1234`) - in-process libopus packet loss + decoder PLC (opus_plc) on top of B2f

> Written 2026-10-19 by the exploop B4 analysis agent (ANALYZE step, wave B4, iteration 2, attempt a0).
>
> **Sources.** Every number below was copied from a file read in this session, or computed in this session from the half_B rows of the local `scores/*.txt` / `*.logits` files. Nothing is quoted from memory. The files:
> - run config and logs: `config.yaml` and `git.diff` (B4b), `config.yaml` (B2f), `logs/train.log` and `metrics.jsonl` (both runs), `top5_by_wf1/{merge_list.txt,select_topk.log}`, the `ckpt/` listing;
> - local evals: `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,compare_vs_B2f-*.txt,ci_vs_B2f-*.txt,compare_ci_vs_B2f-*.json}` and `local_eval/merge_top5/compare_all_readouts.txt` (B4b); the `metrics.json`, `probe_report.json` and `scores/` of the B2f and B4a readouts;
> - stdout log, header lines only: `SLS_setup/B4b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_opusplc03_s1234_20261018201130.log`;
> - loop files: `exploop/B4/iter02/{verdict.json,design.result.json,log_merge.txt,log_final.txt}`, `exploop/B4/{wave.json,plan_snapshot.md,STATUS.md,user_notes.md}`, `scripts/run_B4b.sh`;
> - the pre-registered mechanism readout `scripts/b4_dsp_slices.py --lever opus_plc`. It was run in this session on `local_eval/epoch_8` (primary) and `local_eval/merge_top5`, with the default `--ref` (B2f, same readout), the pinned null panel, and `--placebo` = B4a's same readout (the only other B4 arm so far). There was no `--json_out`. The rules are in `SLS_setup/b4wave/README.md` ("Mechanism readout and verdict rules"). **Process slip:** the analyst captured the script's stdout in two scratch files outside the repository, `/tmp/b4b_an/final.txt` and `/tmp/b4b_an/merge.txt`. Apart from this report, nothing inside the repository was written;
> - registry and code: `docs/experiment_log.csv` (rows `B2f-loudnorm-*`, `B4a-apmns-*`, `B4b-opusplc-*`); `SLS_setup/b4wave/{augment.py,opus_plc.py (docstring),train.py (flag definitions),README.md}`; `SLS_setup/utils/local_eval.py` and `configs/local_eval_sets.yaml` (half split, WF1); `data/dev_noisy_sim/*/meta.csv` (half, lang, src, codec, `dsp_chain`).
>
> Progress-split scores (`model_merging_top5/scores.txt`, `progress.txt`) were not read.
>
> **Previous reports.** `docs/experiments-exploration/b4-experiments/20261018_194212_B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234.md` (B4a, iteration 1). Per the user note of 2026-10-18 21:38, B4a's registry decisions are now merge = revisit and final = promote. The "drop" labels inside that report are an artefact of the old screen setting. The structure here follows `docs/experiments-exploration/b3-experiments/20261010_143058_B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234.md`.
>
> **Readouts.**
> - `-top5` is the weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5/`). It is the wave objective.
> - `-ep8` is the final-epoch EMA checkpoint alone (`local_eval/epoch_8/`).
> - half_B is the half of each sim set that is never used for selection: n = 1,994 per sim set (1,603 spoof, 391 bona). Clean is the full 7,465-clip `dev_online_clean` (6,068 spoof, 1,397 bona; `metrics.json`).
> - WF1 = 0.3 clean + 0.35 matched + 0.35 heldout, using macro-F1 at p_spoof 0.5 (`configs/local_eval_sets.yaml`, `utils/metrics.py`). Echo is reported but is not in WF1.
> - Threshold 0.5 on the half_B score rows reproduces all six half_B WF1 values of `metrics.json` exactly: B2f 91.98 / 92.14, B4a 92.55 / 93.15, B4b 92.22 / 92.26.
> - All readouts were evaluated with `"loudness_norm": -23.0` (every `metrics.json`).

Run directories (under `SLS_setup/exp/`):
- **B4b** `B4b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_opusplc03_s1234_epoch8_bs16_20261018201137` (this report)
- **B2f** `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623`: the reference for the whole wave (`wave.json` `rules.keep_best` true = screen design). Its half_B scores are merge 91.98 and final 92.14.
- **B4a** `B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234_epoch8_bs16_20261018150134`: iteration 1, used as the placebo arm.

## 1. Setup

**B4b = B2f with one factor changed: `--aug_opus_plc_p 0.3`.**
- The codec stage fires on every clip (p = 1.0, as in B2f). On 30 % of those codec calls, picked from a separate RNG stream (rng4), the ffmpeg Opus encode is replaced by an in-process libopus 1.3.1 round trip.
- The round trip uses the same app / bitrate / cutoff / frame draws as the ffmpeg path, adds Gilbert-Elliott bursty packet loss, and runs the decoder's packet-loss concealment at 48 kHz, followed by `resample_poly`.
- The coupled flags stay at their defaults: target loss `--aug_opus_loss 0.01 0.20`, mean burst `--aug_opus_burst 1 4`.
- The lever applies to both classes.

Sources: `scripts/run_B4b.sh`, `b4wave/README.md` "Lever `opus_plc`", `b4wave/augment.py` `_lever_opus_plc`.

Flag diff (`args` of both `config.yaml`, parsed and compared in this session):

| flag / field | B2f | **B4b** | note |
|---|---|---|---|
| trainer | `main_train.py`, git `00da485`, dirty | **`b4wave/train.py`** (b4wave 0.1.0 on b3wave 0.1.0), git `8ea996c`, dirty | `git.diff` touches only `.claude/skills/post-train-eval/SKILL.md` and `docs/experiment_log.csv` (docs / registry). `module_sha256` `b6498bc4a5af…`, the same prefix as in B4a's header. |
| `--aug_opus_plc_p` | (absent = 0) | **0.3** | the lever |
| `--aug_opus_loss` / `--aug_opus_burst` | - | `[0.01, 0.2]` / `[1.0, 4.0]` (defaults) | coupled with `--aug_opus_plc_p` (`wave.json` `coupled`) |
| other B4 flags | - | `aug_apm_p 0.0`, `aug_dtx_p 0.0` (off) | |
| B3 flags (written by the b3wave parser) | - | defaults: `sampler shuffle`, `aug_snr_db 5 20 uniform`, `aug_codec_profiles fixed`, `consistency none`, `llrd 1.0`, `freeze_bottom_k 0`, `specaug False`, `train_subset_frac 1.0` | same behaviour as B2f |
| everything else | identical | identical | no other `args` key differs except `track`: `--batch_size 16 --num_epochs 8 --lr 2.5e-6 --lr_min 5e-7 --warmup_frac 0.05 warmup_cosine adamw wd 0.01 head x10 grad_clip 1 ema 0.9995`, `ce_weights 0.1 0.9`, `label_smoothing 0`, RTC aug + MUSAN, `--loudness_norm -23`, `--keep_topk_by_wf1 7 --keep_epochs 8`, seed 1234 |

**Header lines proving the change took effect** (stdout log):
- `B4 Opus PLC: on p=0.3 loss=0.01..0.20 burst=1..4 decode=48k+resample_poly libopus=libopus 1.3.1`. This matches the design's first `expect` regex.
- `B4 Opus PLC selftest: PASS libopus 1.3.1 (/root/.conda/envs/rtc-sdd/lib/libopus.so.0); ctl round trip 20 encoder configs, lookahead [104] samples at 16 kHz; first concealed 20 ms RMS -21.5 dBFS (tone -20.0, no-loss decode -19.9), 100 ms gap -27.3; 36 shape runs ok`.
  - The library was loaded from the rtc-sdd conda env. That is the owner's fallback install, the same 1.3.1 version as the system library (`plan_snapshot.md`, owner decisions). The README names `/usr/lib/x86_64-linux-gnu/libopus.so.0` as the default path.
- The `RTC augmentation:` line ends with `| opus_plc p=0.3 loss=0.01..0.20 burst=1..4 decode=48k`.
- `b3wave: 0.1.0`, `b4wave: 0.1.0`, `b4wave sha256: b6498bc4a5af`.
- Unchanged lines:
  - `Sampler: shuffle`;
  - `SNR: 5..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: fixed`;
  - `Consistency: none`, `LLRD: 1`, `SpecAugment: off`;
  - `Loudness norm: loudness_norm target=-23 LUFS`;
  - `CE weights: spoof=0.1 bonafide=0.9`;
  - `Parameters: 320268933`, `Train trials: 75785`;
  - `Warmup-cosine: 1895 warmup / 37896 steps, floor 5.0e-07`.
- The stdout log has no `Traceback`, `out of memory` or `failed (` line (grep count 0), and no `[b4wave] ` lever-failure line (count 0).

**Realised dose** (`metrics.jsonl` `b3.aug_counts` and `b4.aug_by_class`, i.e. the `Epoch N B3 aug` / `B4 aug-by-class` lines of `train.log`; shares computed in this session):

| epoch | `opus_plc` / codec calls | bona share (n = 14,353) | spoof share (n = 61,432) | realised-loss bins <5 % / 5-10 % / ≥10 % | `fail_opus_plc` |
|---|---|---|---|---|---|
| 1 | 22,778 / 75,785 = 0.3006 | 0.3039 | 0.2998 | 0.250 / 0.249 / 0.501 | 0 (key absent) |
| 2-8 range | 22,587-23,034 = 0.2980-0.3039 | 0.2970-0.3102 | 0.2973-0.3056 | 0.243-0.249 / 0.247-0.255 / 0.501-0.507 | 0 every epoch |

- The dose matches p = 0.3, it is class-balanced, and the bins are populated on both classes. Bona's ≥10 % share is 0.490-0.513 and spoof's is 0.501-0.505.
- `augment.py` counts `fail_opus_plc` only when the round trip raises and the clip falls back to ffmpeg. The key never appears in any epoch.
- **The arm was dosed exactly as designed.** Falsifier (4), "not dosed", does not apply.

**Run time.**
- `config.yaml` was created 20:11:46. The epoch-8 local eval finished at 23:44:49 (`train.log`), i.e. **3 h 33 min**. The registry `gpu_h` is 3.55, against B2f's 4.14 and B4a's 4.21.
- Epochs 2-8 took 26:05-26:23 min each, against B2f's 30:24-30:35 (eval-end to epoch-end timestamps in both `train.log`s).
  - The in-process path replaces the ffmpeg encode on 30 % of codec calls, and the run was about 4 min per epoch faster.
  - This is consistent with fewer ffmpeg calls, but the cause was not measured.
- Post-train timestamps: `top5_by_wf1/merge_list.txt` 23:45, `merge_top5/metrics.json` 23:55, `epoch_8/metrics.json` 00:00. `verdict.json` was written at 00:12:03.

**Checkpoints.**
- Epoch 1 was pruned: `Pruned epoch_1_dev_loss_0.102553.pth (WF1=88.08, outside top-7)` (`train.log`). Epochs 2-8 are kept at 1,281,235,691 bytes each.
- The symlinks are:
  - `best_by_dev_loss` → epoch 2 (0.096874);
  - `best_by_noisy_eer` → epoch 6;
  - `best_by_wf1` → epoch 8.

**Merged top-5 by half-A mini WF1: epochs 8, 7, 6, 4, 3** (92.85, 92.49, 92.35, 92.24, 91.77; mean 92.34; `select_topk.log`).
- Excluded: 5 (91.64), 2 (91.08) and 1 (88.08).
- B2f's merge was epochs 8, 5, 7, 3, 4 (registry row `B2f-loudnorm-screen`).

## 2. Results vs references (half_B)

Sources:
- `compare_all_readouts.txt` and `compare_vs_B2f-*.txt`;
- `ci_vs_B2f-*.txt` / `compare_ci_vs_B2f-*.json` (paired bootstrap, n = 2,000, half B, threshold 0.5);
- the `half_B` block of each `metrics.json`;
- B4a's CIs from its registry rows.

`compare_all_readouts.txt` gives every delta against its first column (B2f-top5). The B4b-ep8 − B2f-ep8 deltas therefore come from `compare_vs_B2f-ep8.txt` and `compare_ci_vs_B2f-ep8.json`.

### 2.1 Wave iterations and fixed comparators, both readouts

| half_B | B2f-top5 | B2f-ep8 | B4a-top5 | B4a-ep8 | **B4b-top5** | **B4b-ep8** | B4b-top5 − B2f-top5 [95 % CI] | B4b-ep8 − B2f-ep8 [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| **WF1@0.5** | 91.98 | 92.14 | 92.55 | 93.15 | **92.22** | **92.26** | **+0.24 [−0.30, +0.78]**, p(not better) 0.195 | **+0.12 [−0.48, +0.71]**, p 0.344 |
| (B4a vs B2f CI, registry) | | | [−0.01, +1.17] | [+0.43, +1.60] | | | | |
| WF1@oracle | 92.59 | 92.82 | 92.98 | 93.26 | 92.47 | 92.50 | **−0.13** | **−0.32** |
| clean F1@0.5 | 97.60 | 97.69 | 97.72 | 98.05 | 97.70 | 97.71 | +0.09 [−0.20, +0.38] | +0.02 [−0.25, +0.28] |
| noisy F1@0.5 (mean matched, heldout) | 89.57 | 89.76 | 90.34 | 91.06 | 89.88 | 89.92 | +0.31 | +0.16 |
| mean noisy EER | 10.09 | 9.66 | 9.53 | 9.16 | 10.66 | 9.68 | **+0.57** (worse) | +0.02 |
| calibration gap noisy | 0.66 | 0.71 | 0.49 | 0.05 | 0.18 | 0.21 | | |
| separability gap | 7.87 | 7.86 | 7.18 | 7.17 | 8.04 | 7.90 | | |

Per-set paired CIs (`ci_vs_B2f-*.txt`):

| set | merge Δ [CI], p(not better) | ep8 Δ [CI], p(not better) |
|---|---|---|
| clean | +0.09 [−0.20, +0.38], 0.2845 | +0.02 [−0.25, +0.28], 0.4595 |
| matched | **+0.86 [−0.17, +1.90], 0.0455** | +0.19 [−0.94, +1.29], 0.3695 |
| heldout | **−0.25 [−1.46, +0.89]**, 0.6515 | +0.13 [−1.12, +1.38], 0.404 |
| echo | −0.10 [−1.01, +0.81], 0.6155 | −0.04 [−0.97, +0.87], 0.5585 |

Weighted contributions (0.3 / 0.35 / 0.35 × unrounded deltas from `compare_ci_vs_B2f-*.json`):
- merge: +0.03 clean + 0.30 matched − 0.09 heldout = +0.24;
- final: +0.01 + 0.07 + 0.05 = +0.12.

Reading:
- **The small merge gain is matched, and the targeted set, heldout, went down on the merge.**
- **WF1@oracle fell on both readouts (−0.13 / −0.32) while WF1@0.5 rose.**
  - The calibration gap shrank from 0.66 / 0.71 to 0.18 / 0.21.
  - The separability gap grew from 7.87 / 7.86 to 8.04 / 7.90.
  - So B4b's +0.24 / +0.12 is an operating-point shift, not better ranking.
- **Full sets** (registry rows, `log_merge.txt` / `log_final.txt`):
  - merge WF1 92.43 vs B2f 92.25 (+0.18), with matched +0.47 and heldout −0.02;
  - final 92.65 vs 92.53 (+0.12), with heldout +0.27.

### 2.2 Per set (R = recall at 0.5; thr = p_spoof threshold; `metrics.json` half_B)

| set | metric | B2f-top5 | **B4b-top5** | B2f-ep8 | **B4b-ep8** |
|---|---|---|---|---|---|
| dev_online_clean | F1@.5 / F1@or / thr_or | 97.60 / 98.10 / 0.997 | 97.70 / 98.10 / 0.996 | 97.69 / 98.33 / 0.999 | 97.71 / 98.03 / 0.998 |
| | EER / ECE | 3.42 / 1.42 | 3.23 / 1.34 | 3.33 / 1.33 | 3.10 / 1.34 |
| | R_spoof / R_bona | 99.90 / 92.84 | 99.92 / 93.06 | 99.95 / 92.91 | 99.97 / 92.91 |
| sim_matched_v1 | F1@.5 / F1@or / thr_or | 91.22 / 91.53 / 0.846 | **92.08** / 92.18 / 0.537 | 91.24 / 92.05 / 0.991 | 91.43 / 91.66 / **0.099** |
| | EER / ECE | 8.15 / 4.75 | 8.75 / 4.44 | 8.21 / 4.86 | 8.28 / 4.81 |
| | R_spoof / R_bona | 99.50 / 76.21 | 99.50 / **78.52** | 99.44 / 76.47 | **99.13** / 78.01 |
| sim_heldout_v1 | F1@.5 / F1@or / thr_or | 87.92 / 88.94 / 0.957 | **87.67** / 87.93 / 0.967 | 88.27 / 88.89 / 0.949 | 88.40 / 88.60 / **0.019** |
| | EER / ECE | 12.03 / 6.46 | 12.57 / 6.45 | 11.11 / 6.38 | 11.08 / 6.28 |
| | R_spoof / R_bona | 99.00 / 69.31 | 99.31 / **67.77** | 99.13 / 69.82 | 99.06 / 70.33 |
| sim_echo_v1 | F1@.5 / F1@or / thr_or | 94.16 / 94.91 / 0.983 | 94.06 / 94.92 / 0.973 | 94.53 / 95.41 / 0.983 | 94.50 / 94.80 / 0.965 |
| | EER / ECE | 4.71 / 3.37 | 5.43 / 3.40 | 4.86 / 3.20 | 5.53 / 3.21 |
| | R_spoof / R_bona | 99.75 / 83.38 | 99.81 / 82.86 | 99.69 / 84.65 | 99.88 / 83.89 |

Reading:
- **Matched bona recall rose on both readouts** (+2.30 / +1.53).
- **Heldout bona recall fell on the merge (−1.53) and rose only +0.51 on the final**, against the design's "R_bona rising" prediction.
- **Echo EER rose on both readouts** (+0.72 / +0.66), and echo bona recall slipped (−0.51 / −0.77).
- **The final readout overshot on matched and heldout.** Its per-set oracle thresholds are 0.099 (matched) and 0.019 (heldout), i.e. the best operating point lies on the spoof-miss side. Matched spoof misses on half B rose from 9 to 14.

Paired per clip (half_B score files, B4b vs B2f at the same readout; computed):

| set | class | merge fixed / broken / wrong in both | ep8 fixed / broken / wrong in both |
|---|---|---|---|
| clean | bona | 20 / 17 / 80 | 16 / 16 / 83 |
| matched | bona | 18 / 9 / 75 | 17 / 11 / 75 |
| heldout | bona | **9 / 15** / 111 | 15 / 13 / 103 |
| echo | bona | 9 / 11 / 56 | 9 / 12 / 51 |
| heldout | spoof | 11 / 6 / 5 | 8 / 9 / 6 |
| matched | spoof | 3 / 3 / 5 | 3 / **8** / 6 |
| clean / echo | spoof | 2 / 1, 4 / 3 | 2 / 1, 5 / 2 |

Half-B spoof misses (p < 0.5; computed):

| set (n spoof) | B2f-top5 | B4a-top5 | **B4b-top5** | B2f-ep8 | B4a-ep8 | **B4b-ep8** |
|---|---|---|---|---|---|---|
| clean (6,068) | 6 | 9 | 5 | 3 | 7 | 2 |
| matched (1,603) | 8 | 8 | 8 | 9 | 12 | **14** |
| heldout (1,603) | 16 | 23 | **11** | 14 | 19 | 15 |
| echo (1,603) | 4 | 4 | 3 | 5 | 6 | 2 |

B4b undid B4a's rise in heldout spoof misses on the merge (11 vs B2f's 16 and B4a's 23).

### 2.3 Probes (`probe_report.json`; 187 bona + 813 spoof; one bona probe clip = 0.53 points)

Under `--loudness_norm`, the pure-gain probes (`gain_down`, `gain_m20`, `gain_m6`) flip 0.00 % in all four readouts.

| probe (bona unless stated) | B2f-top5 | **B4b-top5** | B2f-ep8 | **B4b-ep8** |
|---|---|---|---|---|
| level_flip_bona (mean) / max | 1.54 / 4.81 | **0.80** / 2.67 | 1.74 / 4.81 | **0.87** / 3.74 |
| gain_up / gain_p6 / gain_p10_limit flip % | 4.28 / 2.14 / 4.81 | 2.67 / 0.00 / 2.67 | 4.81 / 2.67 / 3.21 | 2.67 / 0.53 / 3.74 |
| agc_dynaudnorm / loudnorm flip % | 0.53 / 0.53 | 0.00 / 1.07 | 2.14 / 1.07 | 0.00 / 0.00 |
| level_flip_spoof | 0.03 | 0.02 | 0.03 | 0.02 |
| probe_orig bona pred spoof % (`pred_spoof_orig`) | 10.70 | 11.23 | 10.16 | 11.23 |
| **noise_only bona pred spoof % / mean dP** | 100.00 / +0.8870 | 99.47 / +0.8692 | 100.00 / +0.8965 | 100.00 / +0.8821 |
| **sil_only bona pred spoof % / mean dP** | 100.00 / +0.8929 | 100.00 / +0.8835 | 100.00 / +0.8985 | 100.00 / +0.8861 |
| trimmed / padded bona flip % | 3.74 / 3.74 | 1.60 / 2.67 | 3.74 / 5.35 | 2.14 / 3.74 |

Reading:
- Level sensitivity halved: level_flip_bona 1.54 → 0.80 (merge) and 1.74 → 0.87 (final). B4a also lowered it, to 0.87 / 1.40 (B4a report).
- **The non-speech failure is untouched.**
  - `noise_only` and `sil_only` bona-source clips are still 99.47-100 % spoof.
  - Mean dP is slightly lower (+0.8835 vs +0.8929 merge on `sil_only`).
- The clean probe source lost one bona clip on the merge and two on the final (`pred_spoof_orig` 10.70 → 11.23 and 10.16 → 11.23).

### 2.4 Slices that moved most, B4b vs B2f (full sim sets, `metrics.json → breakdowns`, 63 slice rows; computed)

Counts of slice rows that improved:

| readout | F1 up | EER down | R_bona up | R_spoof up |
|---|---|---|---|---|
| merge | 35 of 63 | **26** | 31 | 27 |
| final | 41 of 63 | **22** | 36 | 27 |

- F1 up per set, merge / ep8: matched 17 / 14 of 23, **heldout 9 / 13 of 23**, echo 9 / 14 of 17.
- Minimum slice R_spoof:
  - B4b-top5 98.35 (heldout / vacuum_cleaner);
  - B4b-ep8 97.36 (heldout / vacuum_cleaner);
  - B2f-top5 97.31 (heldout / door_wood_creaks);
  - B2f-ep8 97.68 (heldout / washing_machine).
- **EER did not improve in most slices** (37 / 41 of 63 rows), consistent with the WF1@oracle drop.

| set / slice | n | F1 merge B2f → B4b | F1 ep8 B2f → B4b | EER ep8 | R_bona merge / ep8 |
|---|---|---|---|---|---|
| heldout / door_wood_creaks | 361 | 86.83 → 91.78 **+4.95** | 88.70 → 90.88 +2.18 | 10.18 → 10.69 | 71.9 → 78.1 / 73.4 → 78.1 |
| matched / opus:qq | 572 | 84.50 → 86.87 +2.38 | 86.04 → 86.59 +0.55 | 10.95 → 13.23 | 61.6 → 66.7 / 66.7 → 66.7 |
| heldout / snr 15 | 835 | 88.08 → 90.10 +2.02 | 89.78 → 90.61 +0.83 | 10.71 → 11.11 | 68.2 → 72.8 / 71.5 → 74.2 |
| heldout / snr 0 | 812 | 84.68 → 85.63 +0.94 | 83.82 → 85.44 +1.62 | 17.07 → 16.13 | 63.3 → 61.9 / 61.2 → 63.3 |
| matched / echo family (no noise) | 678 | 93.76 → 95.20 +1.43 | 94.48 → 95.43 +0.95 | 4.84 → 4.07 | 82.6 → 87.1 / 84.5 → 87.7 |
| matched / opus:lark | 608 | 93.60 → 94.75 +1.15 | 93.55 → 94.45 +0.90 | 6.64 → 6.63 | 82.4 → 84.9 / 81.5 → 84.0 |
| heldout / src offline | 2036 | 87.84 → 88.66 +0.83 | 88.19 → 88.90 +0.71 | 12.16 → 12.41 | 70.4 → 70.4 / 70.9 → 72.0 |
| heldout / lang en | 1538 | 84.45 → 84.59 +0.14 | 84.47 → 85.25 +0.78 | 16.34 → 17.00 | 59.7 → 61.2 / 60.5 → 62.8 |
| **heldout / src online** | 1964 | 89.78 → 88.88 **−0.90** | 89.76 → 89.57 −0.19 | 10.47 → 11.43 | 73.4 → 70.7 / 73.1 → 73.6 |
| **heldout / snr 5** | 778 | 88.99 → 87.45 −1.53 | 89.72 → 87.99 **−1.73** | 10.83 → 11.08 | 74.4 → 68.6 / 76.3 → 71.8 |
| heldout / wind | 381 | 87.17 → 85.74 −1.43 | 88.36 → 87.64 −0.73 | 13.14 → 13.63 | 68.4 → 63.2 / 72.4 → 65.8 |
| heldout / glass_breaking | 346 | 94.49 → 92.49 −2.00 | 92.94 → 92.58 −0.36 | 6.32 → 6.14 | 85.9 → 82.8 / 82.8 → 84.4 |
| matched / opus:wechat | 585 | 86.62 → 86.60 −0.02 | 88.94 → 86.72 **−2.22** | 13.21 → 12.07 | 65.4 → 67.3 / 71.2 → 68.3 |
| echo / opus:dingtalk | 554 | 91.60 → 91.14 −0.46 | 93.00 → 91.74 −1.26 | 5.33 → 8.32 | 79.2 → 77.1 / 82.3 → 77.1 |
| matched / footsteps | 683 | 91.46 → 91.24 −0.22 | 92.14 → 90.74 −1.40 | 7.02 → 8.59 | 78.2 → 78.2 / 78.2 → 77.4 |
| heldout / lang zh | 2462 | 90.86 → 90.88 +0.02 | 91.15 → 91.21 +0.06 | 8.68 → 8.91 | 78.3 → 75.4 / 78.0 → 78.0 |

**The lever was aimed at online sources. Heldout online got worse while heldout offline gained.**
- Online F1 −0.90 (merge) / −0.19 (final), with EER 10.47 → 11.43 (final).
- Offline F1 +0.83 / +0.71.
- The largest slice losses are heldout snr 5 (−1.53 / −1.73), glass_breaking (−2.00 merge) and wind (−1.43 / −0.73).

Half-B splits of heldout bona and spoof (computed from `scores/*.txt` + `meta.csv`; PLC = `dsp_chain` has a `plc` stage, which is heldout's `repeat` concealment after G.722 / G.726 / GSM; NS = an `ns` stage; VAD = a `vad` stage):

| half-B heldout slice | n bona / spoof | R_bona merge B2f → B4b (B4a) | R_bona ep8 B2f → B4b (B4a) | spoof misses merge B2f → B4b (B4a) | spoof misses ep8 B2f → B4b (B4a) |
|---|---|---|---|---|---|
| all | 391 / 1603 | 69.3 → 67.8 (71.4) | 69.8 → 70.3 (73.4) | 16 → 11 (23) | 14 → 15 (19) |
| **online** | 190 / 807 | 69.5 → **65.3** (70.5) | 70.5 → 68.9 (73.7) | 5 → 4 (11) | 4 → 6 (9) |
| online, PLC | 66 / 253 | 65.2 → **62.1** (72.7) | 68.2 → 68.2 (77.3) | 3 → 1 (5) | 3 → 1 (4) |
| online, no PLC | 124 / 554 | 71.8 → 66.9 (69.4) | 71.8 → 69.4 (71.8) | 2 → 3 (6) | 1 → 5 (5) |
| offline | 201 / 796 | 69.2 → 70.1 (72.1) | 69.2 → 71.6 (73.1) | 11 → 7 (12) | 10 → 9 (10) |
| offline, PLC | 64 / 237 | 67.2 → 71.9 (75.0) | 65.6 → 71.9 (75.0) | 0 → 1 (2) | 1 → 2 (1) |
| offline, no PLC | 137 / 559 | 70.1 → 69.3 (70.8) | 70.8 → 71.5 (72.3) | 11 → 6 (10) | 9 → 7 (9) |
| offline, NS | 146 / 530 | 66.4 → 67.8 (68.5) | 67.1 → 67.8 (69.9) | 9 → 4 (10) | 9 → 6 (8) |
| offline, VAD | 73 / 240 | 63.0 → 63.0 (71.2) | 65.8 → 63.0 (71.2) | 7 → 1 (7) | 7 → 3 (6) |
| en | 129 / 631 | 59.7 → 60.5 (56.6) | 58.9 → 62.0 (58.9) | 3 → 7 (10) | 5 → 5 (9) |
| zh | 262 / 972 | 74.0 → 71.4 (78.6) | 75.2 → 74.4 (80.5) | 13 → 4 (13) | 9 → 10 (10) |
| codec gsm | 127 / 549 | 63.0 → 63.0 | 63.8 → 63.8 | 10 → 8 | 11 → 10 |
| codec g726 | 129 / 518 | 64.3 → 62.8 | 65.9 → 66.7 | 3 → 3 | 1 → 5 |
| codec g722 | 135 / 536 | 80.0 → 77.0 | 79.3 → 80.0 | 3 → 0 | 2 → 0 |

- **On half B, B4a (which never saw a dropped packet) raised PLC-treated heldout bona more than B4b did**:
  - online PLC 72.7 / 77.3 against B4b's 62.1 / 68.2;
  - offline PLC 75.0 / 75.0 against 71.9 / 71.9.
- **B4b moved heldout errors from the spoof side to the bona side.** Spoof misses fell (16 → 11 merge; Chinese 13 → 4), while Chinese bona recall fell (74.0 → 71.4).

### 2.5 Lever response (one factor per row)

| run | change vs B2f | half_B merge WF1 (Δ) | final-epoch WF1 (Δ) | merge oracle (Δ) | final oracle (Δ) | merge noisy EER (Δ) | final EER (Δ) | merge clean (Δ) | final clean (Δ) | verdict (`verdict.json`) | mechanism (`b4_dsp_slices.py`, provisional) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B2f | reference (start of wave) | 91.98 | 92.14 | 92.59 | 92.82 | 10.09 | 9.66 | 97.60 | 97.69 | - | - |
| B4a | + `--aug_apm_p 0.3` | 92.55 (+0.57) | 93.15 (+1.02) | 92.98 (+0.39) | 93.26 (+0.43) | 9.53 (−0.55) | 9.16 (−0.51) | 97.72 (+0.12) | 98.05 (+0.35) | merge: above band, CI includes 0 (revisit); final: positive (promote) | NS: REFUTED final / DIRECTIONAL merge (B4a report) |
| **B4b** | **+ `--aug_opus_plc_p 0.3`** | **92.22 (+0.24)** | **92.26 (+0.12)** | 92.47 (−0.13) | 92.50 (−0.32) | 10.66 (+0.57) | 9.68 (+0.02) | 97.70 (+0.09) | 97.71 (+0.02) | merge: **null (inside +-0.5)** / revisit; final: **null (inside +-0.5)** / revisit; new_best false | PLC_online: **REFUTED** on both readouts [placebos 1]; run anomaly flagged (VAD) |

### 2.6 Mechanism readout (pre-registered; `scripts/b4_dsp_slices.py --lever opus_plc --placebo <B4a same readout>`, full sets, arm vs B2f same readout)

The statistic is DiD: the mean clipped-logit change on the treated bona slice minus that on the control bona slice. Negative means the treated slice moved toward bona. It is calibrated on the 9-run null panel (DiD* = DiD − panel mean, z = DiD* / panel SD).

| contrast | readout | arm recall T/C (B2f final) | DiD [CI] | DiD* / z | clip DiD | half-B DiD merge / final | placebo B4a DiD (arm rank) | criteria (a)-(e) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **PLC_online** (target; n 114/254; panel mu −0.57 sd 0.41; refute line −0.57, shown bar −1.39) | **final (primary)** | 71.1 / 74.8 (67.5 / 75.6) | **−0.17 [−0.79, +0.44]** | +0.40 / +0.99 | +4.9 | +0.16 / +0.11 | −0.50 (2/2) | n n n n n | **REFUTED (provisional)** |
| PLC_online | merge | 64.9 / 73.2 | −0.06 [−0.62, +0.49] | +0.59 / +1.68 | +1.6 | +0.16 / +0.11 | −0.61 (2/2) | n n n n n | **REFUTED (provisional)** |
| PLC_offline (report; n 112/270) | final / merge | 69.6/73.0, 68.8/71.1 (69.6/71.5) | +0.17 [−0.49, +0.83] / +0.34 [−0.18, +0.87] | +0.52 / +1.70, +0.44 / +1.42 | | −0.20 / −0.38 | −0.40 / −0.25 (2/2) | | REFUTED / REFUTED |
| NS (non-target; n 276/106) | final / merge | 69.2/79.2, 68.1/76.4 | +0.31 [−0.26, +0.90] / −0.30 [−0.81, +0.20] | −0.02 / −0.07, −0.32 / −1.01 | | −0.14 / +0.81 | +0.49 / −0.03 (1/2) | n n n n Y | DIRECTIONAL / DIRECTIONAL |
| NS_afftdn_vs_anlmdn (secondary) | final / merge | 63.3/75.2, 62.6/73.7 | −0.36 [−1.01, +0.26] / −0.22 [−0.85, +0.37] | +0.47 / +1.10, +0.56 / +1.39 | | −0.63 / −0.48 | −1.39 / −1.14 (2/2) | n | REFUTED / REFUTED |
| **VAD** (non-target, dtx_gate's contrast; n 136/246) | final / merge | 64.0/76.4, 63.2/74.4 (66.2/73.6) | **+0.36 [−0.20, +0.95] / +0.22 [−0.36, +0.82]** | +0.92 / **+3.82**, +0.77 / **+4.25** | −6.9 / −1.6 | +0.39 / +0.72 | −0.22 / −0.54 (2/2) | n | REFUTED / REFUTED |
| VAD_zero / VAD_cng (report) | final | 65.3/76.4, 62.3/76.4 | +0.38 / +0.34 | z +2.58 / +2.05 | | | −0.49 / +0.12 | | REFUTED |

Bona by realised PLC loss on heldout (full set; "logit" = mean clipped logit, negative = bona-ward; B2f same readout in brackets):

| src, realised loss | n | final logit (B2f) | final recall (B2f) | merge logit (B2f) | merge recall (B2f) |
|---|---|---|---|---|---|
| offline, none | 270 | −3.87 (−3.76) | 73.0 (71.5) | −3.63 (−3.46) | 71.1 (70.7) |
| offline, < 0.05 | 35 | −2.92 (−2.59) | 65.7 (62.9) | −2.88 (−2.64) | 65.7 (62.9) |
| offline, 0.05-0.10 | 44 | −4.24 (−3.80) | 75.0 (70.5) | −3.93 (−3.75) | 72.7 (72.7) |
| **offline, ≥ 0.10** | 33 | **−2.54 (−3.67)** | **66.7 (75.8)** | **−2.54 (−3.58)** | 66.7 (72.7) |
| online, none | 254 | −4.13 (−4.19) | 74.8 (75.6) | −3.88 (−4.14) | 73.2 (76.4) |
| online, < 0.05 | 30 | −4.66 (−3.54) | 80.0 (66.7) | −3.91 (−3.52) | 70.0 (70.0) |
| online, 0.05-0.10 | 48 | −2.96 (−2.94) | 68.8 (66.7) | −2.66 (−2.56) | 64.6 (62.5) |
| **online, ≥ 0.10** | 36 | **−2.77 (−3.37)** | **66.7 (69.4)** | **−2.41 (−3.50)** | **61.1 (69.4)** |

**The high-loss heldout bona moved toward spoof in all four src × readout cells.** That is the opposite of the lever's hypothesis.
- B4b trained both classes on libopus concealment with up to 20 % target loss, half of its PLC calls in the ≥ 10 % bin.
- Yet heldout bona at ≥ 10 % realised repeat-PLC loss ended more spoof-like than under B2f.
- Low-loss online bona moved bona-ward on the final readout (−3.54 → −4.66, n = 30).

Other blocks of the same output:
- **Lever verdict (opus_plc): `REFUTED (provisional) [placebos 1]` on both readouts.**
- **Outcome (O3.2):**
  - final: heldout 88.40 (Δ +0.13); the panel mean is 88.50, SD 1.35, and 5 of 9 panel runs clear ≥ +0.5;
  - merge: heldout 87.67 (Δ −0.25); the panel mean is 88.25, SD 1.39, and 6 of 9 clear ≥ +0.5;
  - `falsifier (1) outcome escape (heldout F1 delta >= +0.5 on merge_top5 AND final): False`.
- **opus_plc decision lines:**
  - `falsifier (1) … FIRES (escape False)`;
  - `falsifier (2) … does not fire`;
  - must-not-move, all held:
    - matched F1 Δ +0.86 / +0.19;
    - matched qq bona recall 66.67 (B2f 61.62, +5.05) merge and 66.67 (B2f 66.67, +0.00) final;
    - matched wechat bona recall 67.31 (B2f 65.38, +1.92) merge and 68.27 (B2f 71.15, −2.88) final;
    - PLC spoof misses 7 / 10;
    - clean F1 +0.09 / +0.02;
    - ep1 TrainAcc Δ −0.81;
  - `attribution control (3): not given`;
  - script line: `decision: DROP: falsifier (1) fires (libopus PLC does not carry over)`. This is the script's mechanism decision line, not the supervisor verdict (section 6).
- **Spoof-miss guards (Poisson 99 % bound), all held:**

  | guard | merge B2f → B4b | final B2f → B4b | bound merge / final |
  |---|---|---|---|
  | NS_offline | 18 → 9 | 18 → 13 | ≤ 29 / ≤ 29 |
  | VAD_offline | 15 → 2 | 13 → 6 | ≤ 25 / ≤ 22 |
  | PLC | 16 → 7 | 15 → 10 | ≤ 26 / ≤ 25 |
  | halfB_noisy | 24 → 19 | 23 → 29 | ≤ 36 / ≤ 35 |

- **Run anomaly (O3.1): `ANOMALOUS`.**
  - It is flagged because `|z|>3 on non-target primary {'VAD': 3.82}` (final) and `{'VAD': 4.25}` (merge).
  - Under-fit False, clean-F1 guard False.
  - Per the README, a flagged run reports a would-be "shown" verdict as "confounded (run anomaly)". The verdict here is REFUTED, so the flag does not change it. It still means **B4b pushed offline VAD-gated heldout bona toward spoof by more than any null-panel run**: recall T/C 64.0 / 76.4 against B2f's 66.2 / 73.6 (final).
- **Pinned tertiles (dtx_gate's criterion (ii)).**
  - B4b final, full set: saturated noisy bona FA 67 / 73 / 106 (low / mid / high tertile) against B2f's 66 / 76 / 107.
  - drop_high − drop_low = +2 (need ≥ +5; panel mean −10.2, max +0; z +2.12).
  - Half B: −11 (final) and −3 (merge), against "need ≥ +0".
  - B4a scored +10 on the full set (B4a report).
- **Placebo use for the other arms (SUMMARY re-issues the verdicts):**
  - apm_ns: B4a's NS DiD (+0.49 final, −0.03 merge) sits above B4b's (+0.31 / −0.30). On these numbers B4a ranks 2 of 2 on its own placebo criterion (e).
  - B4a's afftdn-vs-anlmdn DiD (−1.39 / −1.14) is below B4b's (−0.36 / −0.22).

## 3. Per-epoch behaviour, B4b vs B2f (same LR at each epoch end: 2.469e-6 … 5.000e-7 in both `train.log`s)

Sources:
- `logs/train.log` (both runs);
- `metrics.jsonl` (EMA weights; 2,000 clean clips + `sim_matched_mini` + `sim_heldout_mini`, half A);
- `STATUS.md` (Δ = B4b − B2f mini WF1).

| ep | B4b mini WF1 | B2f mini WF1 | Δ | B4b WF1@or | B4b EER noisy (B2f) | B4b R_bona cl/ma/he | B2f R_bona cl/ma/he | mini heldout F1 B4b / B2f | train loss B4b / B2f | train acc B4b / B2f | dev loss B4b / B2f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 88.08 | 88.34 | −0.26 | 89.92 | 13.90 (13.40) | 93.3/82.7/78.1 | 92.7/81.6/78.6 | 83.31 / 84.10 | 0.3691 / 0.3558 | 86.55 / 87.36 | 0.1026 / 0.0928 |
| 2 | 91.08 | 91.47 | −0.39 | 91.87 | 11.53 (11.78) | 91.6/76.5/**67.9** | 91.6/77.1/76.0 | 86.31 / 87.78 | 0.2166 / 0.2075 | 97.02 / 97.13 | 0.0969 / 0.0982 |
| 3 | 91.77 | 91.90 | −0.13 | 92.38 | 11.38 (10.38) | 91.6/77.1/68.9 | 91.6/74.9/71.4 | 87.75 / 88.47 | 0.1687 / 0.1546 | 97.87 / 98.09 | 0.0990 / 0.1139 |
| 4 | 92.24 | 91.69 | +0.55 | 92.93 | 11.81 (10.64) | 92.4/77.1/70.9 | 91.3/75.4/71.4 | 88.86 / 88.32 | 0.1215 / 0.1108 | 98.47 / 98.62 | 0.1113 / 0.1242 |
| 5 | 91.64 | 92.49 | −0.85 | 92.36 | 11.22 (10.46) | 91.3/76.0/**67.3** | 90.4/78.8/71.9 | 87.87 / 89.27 | 0.1011 / 0.0879 | 98.83 / 98.96 | 0.1151 / 0.1178 |
| 6 | 92.35 | 91.66 | +0.69 | 93.15 | **10.49** (10.63) | 92.1/78.2/72.4 | 91.3/74.9/70.4 | 89.02 / 88.36 | 0.0800 / 0.0725 | 99.07 / 99.18 | 0.1072 / 0.1321 |
| 7 | 92.49 | 92.10 | +0.39 | 93.05 | 11.45 (10.69) | 91.6/78.8/73.0 | 91.3/76.0/73.0 | 89.37 / 89.37 | 0.0622 / 0.0540 | 99.23 / 99.36 | **0.1486** / 0.1406 |
| 8 | **92.85** | 92.80 | +0.05 | **93.25** | 11.05 (10.94) | 92.1/79.3/75.5 | 91.3/80.4/73.5 | **90.05** / 89.41 | 0.0571 / 0.0501 | 99.32 / 99.44 | 0.1420 / 0.1442 |

Reading:
- **The best epoch is the last** (mini WF1 92.85, `best_by_wf1` epoch 8). The curve rose +0.36 from epoch 7 to 8, so it is still rising, as in B2f (+0.70) and B4a (B4a report). The best noisy EER is at epoch 6 (10.49).
- **B4b ends level with B2f on the selection half** (+0.05 at epoch 8). It alternated around B2f from epoch 4 on (+0.55, −0.85, +0.69, +0.39).
  - Epoch 1 is −0.26, with TrainAcc 86.55 against 87.36: inside the design's ±2 band and above the 85.36 under-fit line.
- **Early heldout bona dipped.** Mini heldout bona recall at epochs 2-5 is 67.9 / 68.9 / 70.9 / 67.3 against B2f's 76.0 / 71.4 / 71.4 / 71.9. It recovers to 75.5 at epoch 8, above B2f's 73.5.
- **Train loss is higher than B2f's at every epoch** (epoch 8: 0.0571 vs 0.0501, inside the 0.0752 under-fit line).
- **Dev loss.**
  - It is lower than B2f's at epochs 2-6 and 8, and higher at epochs 1 and 7.
  - Epoch 1 → 8 it rose ×1.38 (0.102553 → 0.141961) against B2f's ×1.55 (0.092750 → 0.144168) (computed).
  - It jumped at epoch 7 (0.1072 → 0.1486).
- **The final epoch is level with the merge on half B**: 92.26 vs 92.22 (+0.04). B2f's final beat its merge by 0.16, B4a's by 0.60.
  - The merge has the better matched (92.08 vs 91.43); the final has the better heldout (88.40 vs 87.67) and noisy EER (9.68 vs 10.66).
  - The weakest merged epochs by mini WF1 are 3 (91.77) and 4 (92.24). Epoch 3 also has the low mini heldout bona recall (68.9).
  - No merged epoch was evaluated alone on half B, so its half-B drag is not measured.

## 4. What went OK (evidence)

1. **Clean experiment, dosed exactly as designed.**
   - One factor, confirmed by the header lines and `B4 Opus PLC selftest: PASS`.
   - Realised `opus_plc` share 0.2980-0.3039 of codec calls per epoch, class-balanced (bona 0.2970-0.3102, spoof 0.2973-0.3056).
   - All three loss bins populated, with ≥ 10 % loss on about half of the calls.
   - `fail_opus_plc` 0; no traceback, OOM or `failed (` line.
2. **Every pre-registered guard held** (`b4_dsp_slices.py` decision lines, `metrics.json`):
   - clean F1 +0.09 / +0.02;
   - matched F1 +0.86 / +0.19;
   - matched qq / wechat bona recall Δ +5.05 / +1.92 (merge) and +0.00 / −2.88 (final), all ≥ −6;
   - full-set PLC spoof misses 7 / 10 (bounds 26 / 25; B2f 16 / 15);
   - half-B online heldout spoof misses 4 / 6, below B4a's 11 / 9;
   - ep1 TrainAcc 86.55 (Δ −0.81);
   - ep8 TrainLoss 0.0571 ≤ 0.0752.
3. **Spoof misses went down where B4a had raised them.**
   - Half-B heldout spoof misses 11 on the merge (B2f 16, B4a 23).
   - Every spoof-miss guard count is below B2f's on the merge: full-set NS 9 vs 18, VAD 2 vs 15 and PLC 7 vs 16; half-B noisy 19 vs 24.
4. **Matched improved on the merge** (+0.86 [−0.17, +1.90], p(not better) 0.0455).
   - Matched bona recall rose +2.30 / +1.53.
   - The hardest Opus app rose too: matched opus:qq F1 +2.38 merge, bona recall 61.6 → 66.7.
5. **Level shortcut halved.** level_flip_bona 1.54 → 0.80 (merge) and 1.74 → 0.87 (final); `gain_p6` flips 2.14 → 0.00 (merge).
6. **Fewer saturated noisy bona false alarms on the merge**: p ≥ 0.99 on half-B matched + heldout fell from 136 to 115 (section 5.3).
7. **Cheaper.** 3.55 GPU-h against B2f's 4.14 (registry). Epochs ran 26:05-26:23 min against 30:24-30:35.

## 5. What went wrong (evidence)

### 5.1 The hypothesised mechanism did not appear, and the target slice moved the wrong way

- **PLC_online is REFUTED on both readouts.**
  - DiD −0.17 [−0.79, +0.44] (final) and −0.06 [−0.62, +0.49] (merge), both above the −0.57 refute line (DiD* +0.40 / +0.59).
  - Both are also above the placebo B4a's −0.50 / −0.61: B4b ranks 2 of 2, i.e. the arm that trained on packet loss moved PLC bona less than the arm that never saw it.
  - Half-B DiD is +0.16 / +0.11.
- **The heldout outcome did not appear.**
  - Heldout F1 −0.25 (merge) and +0.13 (final), against the predicted ≥ +0.5.
  - Heldout bona recall 69.31 → 67.77 (merge) and 69.82 → 70.33 (final).
  - Both heldout CIs include 0: [−1.46, +0.89] and [−1.12, +1.38].
- **High-loss heldout bona moved toward spoof** (section 2.6 table).
  - At ≥ 10 % realised loss: offline final recall 75.8 → 66.7 and online merge 69.4 → 61.1.
  - The logit moved spoof-ward in all four src × readout cells.
- **Heldout online, the lever's target source, got worse.**
  - Full-set online F1 −0.90 (merge); online EER 10.47 → 11.43 (final).
  - Half-B online bona recall 69.5 → 65.3 (merge) and 70.5 → 68.9 (final).
- **Non-target anomaly.** VAD-gated offline heldout bona moved spoof-ward: VAD DiD +0.36 / +0.22, z +3.82 / +4.25, `Run anomaly (O3.1): ANOMALOUS`. Something in the in-process codec path made gated or zero-filled pauses look more synthetic. This run cannot tell which part.

### 5.2 Oracle-gap decomposition (global threshold vs set-specific)

Method:
- one shared logit threshold swept over the half_B `scores/*.logits` (step 0.05, −6 to +12; computed in this session);
- the per-set oracle WF1 is `half_B.wf1.wf1_oracle`.

| readout | WF1@0.5 | best global (logit / p) | WF1 at best global | per-set oracle WF1 | gap | recovered by one global shift | set-specific remainder |
|---|---|---|---|---|---|---|---|
| B2f-top5 | 91.98 | +3.10 / 0.957 | 92.39 | 92.59 | 0.61 | 0.41 | 0.20 |
| B2f-ep8 | 92.14 | +4.40 / 0.988 | 92.58 | 92.82 | 0.69 | 0.44 | 0.24 |
| B4a-ep8 | 93.15 | +0.45 / 0.611 | 93.16 | 93.26 | 0.10 | 0.01 | 0.10 |
| **B4b-top5** | 92.22 | **+0.05 / 0.512** | 92.25 | 92.47 | 0.25 | **0.03** | 0.22 |
| **B4b-ep8** | 92.26 | **−4.00 / 0.018** | 92.39 | 92.50 | 0.24 | 0.13 | 0.11 |

- B4b removed most of B2f's spoof-side operating-point bias. What one global threshold shift can recover fell from 0.41 / 0.44 (B2f) to 0.03 / 0.13.
- It did this without adding separability. Per-set oracle WF1 fell from 92.59 / 92.82 to 92.47 / 92.50.
- On the final readout the best shared threshold sits on the other side (p 0.018): spoof misses are now the cheaper error, which is why matched spoof misses rose 9 → 14.
- **What remains is ranking.**

### 5.3 The saturated bona tail

Share of half_B bona with p_spoof ≥ 0.5 / ≥ 0.9 / ≥ 0.99 / ≥ 0.999 (computed from `scores/*.txt`):

| set (n bona) | B2f-top5 | **B4b-top5** | B2f-ep8 | **B4b-ep8** |
|---|---|---|---|---|
| clean (1,397) | 7.2 / 6.9 / 5.4 / 4.4 | 6.9 / 6.7 / 5.7 / 4.4 | 7.1 / 6.3 / 5.3 / 4.3 | 7.1 / 6.8 / **6.2 / 5.2** |
| matched (391) | 23.8 / 22.3 / 14.6 / 8.2 | 21.5 / 20.5 / 11.8 / 7.9 | 23.5 / 21.5 / 13.6 / 9.0 | 22.0 / 21.2 / 15.1 / **10.5** |
| heldout (391) | 30.7 / 28.6 / 20.2 / 9.2 | 32.2 / 30.9 / 17.6 / 9.0 | 30.2 / 28.4 / 20.5 / 9.5 | 29.7 / 28.6 / 19.7 / **10.7** |
| echo (391) | 16.6 / 15.6 / 10.5 / 4.9 | 17.1 / 16.1 / 10.2 / 5.6 | 15.3 / 14.8 / 10.0 / 5.4 | 16.1 / 15.6 / 13.0 / **8.7** |

Noisy bona (matched + heldout half B, n = 782):

| readout | FA (p ≥ 0.5) | saturated (p ≥ 0.99) | saturated share of FA | FA en / zh (en n = 258) | saturated en / zh |
|---|---|---|---|---|---|
| B2f-top5 | 213 | 136 | 64 % | 95 / 118 | 65 / 71 |
| B4a-top5 | 192 | 114 | 59 % | 95 / 97 | 61 / 53 |
| **B4b-top5** | 210 | **115** | **55 %** | 92 / 118 | 51 / 64 |
| B2f-ep8 | 210 | 133 | 63 % | 99 / 111 | 67 / 66 |
| B4a-ep8 | 177 | 123 | 69 % | 91 / 86 | 68 / 55 |
| **B4b-ep8** | 202 | **136** | **67 %** | 91 / 111 | 62 / 74 |

- On the merge, B4b moved bona out of the saturated core (136 → 115) without removing false alarms: 213 → 210 FA.
- **On the final readout the extreme tail grew.** The p ≥ 0.999 share rose on every set: clean 4.3 → 5.2, matched 9.0 → 10.5, heldout 9.5 → 10.7, echo 5.4 → 8.7.
- B4b removed no Chinese noisy bona false alarm on either readout (118 → 118, 111 → 111). B4a removed 21 / 25.

### 5.4 The worst slices for bona recall

Full sets, B4b-top5 (B4b-ep8 in brackets), `metrics.json` breakdowns:

| slice | R_bona merge (ep8) | B2f merge (ep8) |
|---|---|---|
| heldout vacuum_cleaner | 58.6 (64.3) | 60.0 (57.1) |
| heldout washing_machine | 59.0 (61.5) | 62.8 (65.4) |
| heldout lang en | 61.2 (62.8) | 59.7 (60.5) |
| heldout snr 0 | 61.9 (63.3) | 63.3 (61.2) |
| matched snr 0 | 62.4 (66.7) | 62.4 (65.8) |
| heldout wind | 63.2 (65.8) | 68.4 (72.4) |
| matched lang en | 64.7 (65.9) | 63.6 (63.2) |
| heldout gsm | 65.4 (67.5) | 65.9 (65.4) |
| matched opus:qq | 66.7 (66.7) | 61.6 (66.7) |
| heldout pouring_water | 66.7 (68.1) | 69.4 (69.4) |
| matched opus:wechat | 67.3 (68.3) | 65.4 (71.2) |
| heldout g726 | 67.5 (70.3) | 68.3 (70.3) |

- English is still the largest single block: half-B heldout English bona recall 60.5 / 62.0 against Chinese 71.4 / 74.4.
- English accounts for 92 of 210 (merge) and 91 of 202 (final) noisy bona false alarms, from 258 of 782 bona.

### 5.5 Overfitting signs

- Train accuracy reached 99.32 % while dev loss rose 0.102553 → 0.141961 (×1.38; B2f ×1.55). The over-fitting slope is no steeper than B2f's.
- The dev-loss jump at epoch 7 (0.1072 → 0.1486) coincides with mini heldout `thr_oracle` 0.02 at that epoch (`metrics.jsonl`). Epoch 7 is in the merge.
- The mini WF1 is still rising at epoch 8. Final and merge are level on half B (+0.04), so the earlier merged epochs neither help nor hurt much.

## 6. Verdict on the pre-registered rule

From `exploop/B4/iter02/design.result.json`:

- **Falsifier:** "(1) Mechanism refuted: online PLC DiD >= -0.57 (DiD* >= 0, no better than an average null retrain) AND heldout F1 delta < +0.5 on either readout: libopus PLC exposure does not carry over to heldout concealment. Drop. (2) half_B WF1 delta < -0.5 (below the ±0.5 band) with the shared under-fit signature (ep1 TrainAcc < 85.36 or ep8 TrainLoss > 0.0752): a B3d-type codec-side under-fit. Drop. (3) Guard break: full-set PLC spoof misses > 26 (merge) / > 25 (final), matched qq or wechat bona recall down > 6, or clean F1 down > 0.5: concealment became a bona cue, or matched codec fidelity was lost. Drop. (4) A high fail_opus_plc count in the train log: the arm was not dosed as designed (confounded, not refuted). Wave verdict rule: positive = merge half_B WF1 delta > +0.5 with CI low > 0 and clean drop <= 0.5. If positive, iteration d runs the zero-loss control `--aug_opus_plc_p 0.3 --aug_opus_loss 0 0`; …"
- **Predicted signature (abridged):** "online PLC-vs-no-PLC bona DiD < -0.57 (panel mu) on the final readout and below B4a's -0.50 (final) / -0.61 (merge) … Outcome: sim_heldout_v1 F1 up >= 0.5 from 87.92 / 88.27 (merge / final), driven by R_bona rising from 69.31 / 69.82 with R_spoof near 99.00 / 99.13. Expected half_B WF1 delta vs B2f 0 to +0.4 … against 91.98 / 92.14." The must-not-move items are listed in the table below.

`exploop/B4/iter02/verdict.json` (computed by the supervisor; quoted, not recomputed):

| readout | run | base | delta | CI | clean_drop | new_best | decision | verdict |
|---|---|---|---|---|---|---|---|---|
| merge (objective) | 92.22124842701986 | 91.97880641320396 | 0.24244201381590358 | [-0.29674842122580397, 0.7810814332812609] | -0.09175723980615658 | false | **revisit** | **null (inside +-0.5)** |
| final | 92.25616908859993 | 92.13685698540242 | 0.11931210319751528 | [-0.4777043557624472, 0.7078398363140838] | -0.021959619682121456 | false | **revisit** | **null (inside +-0.5)** |

Rules (`verdict.json`):
- `new_best_delta` 0.0, `keep_best` true (screen: B2f stays the reference);
- band 0.5;
- promote = ci_low > 0 and clean_drop ≤ 0.5;
- positive = delta > 0.5 with ci_low > 0 and clean_drop ≤ 0.5.

**Both readouts are null (inside ±0.5), with decision `revisit` and `new_best` false.** The arm is not "positive" on the WF1 CI rule, so the pre-registered zero-loss control for iteration d is not triggered (`wave.json` lever_notes).

Falsifiers:

| falsifier | evidence | fired? |
|---|---|---|
| (1) PLC DiD ≥ −0.57 AND heldout F1 Δ < +0.5 on either readout | PLC_online DiD −0.17 (final) / −0.06 (merge); heldout Δ −0.25 / +0.13; script: `FIRES (escape False)` | **yes**: the mechanism is REFUTED (provisional; 1 placebo arm) |
| (2) WF1 Δ < −0.5 with the under-fit signature | WF1 Δ +0.24 / +0.12; ep1 TrainAcc 86.55 ≥ 85.36; ep8 TrainLoss 0.0571 ≤ 0.0752 | no |
| (3) guard break | PLC spoof misses 7 / 10 (≤ 26 / 25); qq / wechat Δ ≥ −2.88; clean Δ +0.09 / +0.02 | no |
| (4) high `fail_opus_plc` | 0 in every epoch | no: the arm was dosed as designed, so the null is a real null, not a confound |

The predicted signature, item by item:

| predicted | merge | final | appeared? |
|---|---|---|---|
| online PLC DiD < −0.57 | −0.06 | −0.17 | **no** |
| PLC DiD below B4a's −0.61 / −0.50 | −0.06 (rank 2/2) | −0.17 (rank 2/2) | **no** |
| heldout F1 up ≥ 0.5 | −0.25 (87.92 → 87.67) | +0.13 (88.27 → 88.40) | **no** |
| driven by heldout R_bona rising | 69.31 → 67.77 | 69.82 → 70.33 | **no** / marginal |
| heldout R_spoof near 99.00 / 99.13 | 99.31 | 99.06 | yes |
| half_B WF1 Δ 0 to +0.4 | +0.24 | +0.12 | yes (inside the predicted range) |
| dev_online_clean F1 within 0.5 of 97.60 / 97.69 | 97.70 | 97.71 | yes |
| sim_matched F1 down ≤ 0.5 from 91.22 / 91.24 | 92.08 | 91.43 | yes (rose) |
| matched qq / wechat bona recall down ≤ 6 | +5.05 / +1.92 | +0.00 / −2.88 | yes |
| full-set PLC spoof misses ≤ 26 / 25 | 7 | 10 | yes |
| online heldout spoof misses (half B) not above 11 / 9 | 4 | 6 | yes |
| ep1 TrainAcc within 2 of 87.36 | 86.55 | - | yes |
| opus_plc counts ≈ 0.3 × 75,785, balanced, bins populated, `fail_opus_plc` ≈ 0 | 0.2980-0.3039, balanced, populated, 0 | | yes |
| report: offline PLC contrast | DiD +0.34 | DiD +0.17 | refuted (report only) |

**Summary.**
- Clean null: dosed as designed, no guard broken, WF1 inside the band on both readouts.
- The PLC mechanism was refuted, with the placebo arm (B4a) doing better on the target contrast.
- Heldout high-loss bona moved spoof-ward, against the hypothesis.
- The small WF1@0.5 gain is an operating-point shift with lower WF1@oracle.
- The run is flagged ANOMALOUS on the non-target VAD contrast (z +3.82 / +4.25).
- The mechanism verdict is provisional until the wave SUMMARY re-runs `b4_dsp_slices.py --summary` with all arms.

## 7. Decision and next-experiment proposals

**Decision.** Per `verdict.json`, **B4b is not the new best**: `new_best false`, decision `revisit`, verdict `null (inside +-0.5)` on both readouts.
- **B2f stays the reference**: merge 91.98, final 92.14 on half_B; registry ids `B2f-loudnorm-screen` / `B2f-loudnorm-ep8`.
- Every proposal below is one factor (or one coupled group) against **B2f's flags**: `scripts/run_B4b.sh` without `--aug_opus_plc_p`, with the `b4wave.train` entry, `--batch_size 16` and 8 epochs.
- The wave rule is unchanged, against B2f-top5 = 91.98:
  - positive = Δ merge > +0.5 with CI low > 0 and clean drop ≤ 0.5;
  - null = Δ within ±0.5;
  - negative = Δ below −0.5.

**Largest remaining error.** Confident noisy heldout bona false alarms, a ranking problem rather than a threshold problem:
- half-B heldout bona recall 67.77 / 70.33;
- 210 / 202 of 782 noisy bona are false alarms, 55 % / 67 % of them at p ≥ 0.99;
- English heldout bona recall 60.5 / 62.0 against Chinese 71.4 / 74.4;
- one global threshold recovers only 0.03 / 0.13 of the 0.25 / 0.24 oracle gap;
- WF1@oracle fell −0.13 / −0.32 against B2f.

**Ranked proposals (configuration first, seed repeat last):**

1. **`--aug_dtx_p 0.3` (dtx_gate; the queued iteration c, pre-registered in `wave.json` `lever_queue`).**
   - Targets:
     - the saturated noisy bona false alarms (half B p ≥ 0.99: 115 / 136 on B4b, 136 / 133 on B2f);
     - the pause-rich tertile (B2f full final saturated FA 66 / 76 / 107);
     - `sil_only` / `noise_only` bona at 99.47-100 % spoof;
     - the offline VAD-gated heldout bona (B2f final recall T/C 66.2 / 73.6).
   - **B4b sharpens the placebo bar.** B4b, which never gated anything, moved VAD-gated bona spoof-ward (VAD DiD +0.36 final / +0.22 merge, z +3.82 / +4.25); B4a moved it bona-ward (−0.22 / −0.54). dtx_gate must beat both.
   - Predicted signature (from `lever_queue`, with the two placebos):
     - VAD DiD < −0.57 and below B4a's −0.22 (final) / −0.54 (merge);
     - criterion (ii) drop_high − drop_low ≥ +5 on the full set (final), above both placebos (B4a +10, B4b +2), and ≥ 0 on half B for both readouts;
     - `sil_only` bona mean dP below +0.8929 / +0.8985;
     - clean F1 within 0.5;
     - half-B noisy spoof misses ≤ 36 / 35;
     - WF1 Δ 0 to +0.5.
   - Falsifier:
     - clean F1 down > 0.5, half-B noisy spoof misses > 36 / 35, or VAD spoof misses > 25 / 22 → drop;
     - VAD DiD ≥ −0.57 AND criterion (ii) ≤ −10.2 with WF1 Δ < +0.5 → mechanism refuted. Under the README dose rule this reads "inconclusive (under-dosed)", never "refuted".
2. **`--aug_apm_p 0.6` (apm_ns dose; candidate for iteration d).** opus_plc is not positive, so the plan's d slot is "the analyst's best-supported B4 follow-up".
   - Hypothesis: B4a's gain is a dose-responsive effect of the NS pass, but not suppressor-artefact neutralisation.
     - B4a is the only B4 arm with a positive verdict: final +1.02 [+0.43, +1.60], and its matched set rose in 23 of 23 slices at the merge (B4a report).
     - B4b is an equally dosed 30 % extra augmentation on another slot, and it gave only +0.12 [−0.48, +0.71] (final). "Any extra augmentation" therefore does not explain B4a.
     - B4a's own NS contrast is refuted, and it ranks 2 of 2 against the B4b placebo (NS DiD +0.49 vs +0.31, final).
   - Predicted signature:
     - merge Δ vs B2f > +0.5 with CI low > 0 (B4a's merge CI low was −0.01);
     - final Δ ≥ +0.5;
     - matched F1 ≥ 92.46 / 92.80 (B4a);
     - clean F1 within 0.5;
     - full-set offline-NS spoof misses ≤ 29;
     - ep1 TrainAcc within 2 of 87.36.
   - Falsifier:
     - final Δ vs B2f < +0.5 → no dose response, so B4a's +1.02 was not dose-driven (or was seed noise); stop the apm_ns line;
     - half-B heldout spoof misses above B4a's 23 / 19 → more NS trades heldout spoof recall for bona recall.
3. **`--aug_opus_plc_p 0.3 --aug_opus_loss 0 0` (one coupled group: the in-process libopus decoder path without packet loss).**
   - Hypothesis: B4b combines two opposite effects.
     - The in-process decoder path (aligned decode, 48 kHz + `resample_poly`; the README alignment note says the ffmpeg path decodes about 69 samples late) acts as codec diversity. It would explain the matched gain: +0.86 merge, matched bona recall +2.30 / +1.53, opus:qq +2.38.
     - The packet-loss part pushed high-loss heldout PLC bona spoof-ward (≥ 10 % bin: offline final recall 75.8 → 66.7, online merge 69.4 → 61.1).
     - Removing the loss should keep the first effect and drop the second. This is the plan's attribution control. It is run here as a diagnostic, because the arm was not positive.
   - Predicted signature:
     - matched F1 Δ ≥ +0.5 vs B2f on both readouts;
     - heldout ≥ 10 % realised-loss bona logit not above B2f's (−3.67 offline / −3.37 online, final);
     - heldout F1 Δ ≥ 0 on both readouts;
     - WF1 Δ ≥ B4b's +0.24 / +0.12;
     - VAD DiD not spoof-ward of the panel (z < +3).
   - Falsifier:
     - matched Δ < +0.5 on both readouts and WF1 within ±0.5 → the decoder path is worth nothing either, so close the opus family;
     - the ≥ 10 % bins move spoof-ward again with zero loss, or VAD z > +3 again → the shift comes from the decoder path, not from loss.
4. **`--aug_apm_p 0.3 --aug_apm_ns_levels 18 21` (one coupled group: strong WebRTC suppression only).**
   - Hypothesis: the one NS-specific signal is the afftdn-vs-anlmdn contrast, and it survives the placebo. B4a scored −1.39 / −1.14 against B4b's −0.36 / −0.22.
     - afftdn is the harsher heldout suppressor (B2f final recall T/C 62.6 / 75.9, `b4_dsp_slices.py`).
     - Training only on kHigh / kVeryHigh (18 / 21 dB) may match its artefacts better than the uniform 6-21 dB draw.
   - Predicted signature:
     - afftdn-vs-anlmdn DiD ≤ −1.70 (the "strong" line);
     - NS DiD ≤ −0.22 (final);
     - heldout F1 Δ ≥ +0.5 on both readouts;
     - offline-NS spoof misses ≤ 29.
   - Falsifier: NS DiD ≥ +0.32 and afftdn-vs-anlmdn DiD > −0.84 → WebRTC NS does not transfer to the ffmpeg suppressors at any level; drop the lever.
5. **Seed repeat of the reference: `--seed 1235` on B2f (last).**
   - Purpose: measure B2f's own half-B seed band, so that B4a's +0.57 / +1.02 and B4b's +0.24 / +0.12 can be read against same-configuration noise instead of the 9-run mixed panel (heldout F1 SD 1.35 / 1.39).
   - **This is outside wave B4's search space.** `wave.json` pins seed 1234, `--seed` is not in `allowed`, and the owner declined a B2f replicate arm (`plan_snapshot.md`). It needs an owner decision.
   - Predicted signature: |Δ| ≤ 0.5 vs B2f on both readouts.
   - Falsifier: a repeat that differs from B2f by ≥ +0.57 (merge) or ≥ +1.02 (final) puts B4a's gains inside seed noise.

**Recommended full run if the wave stopped now.**
- Per `verdict.json`, the reference is still B2f, and the plan leaves the wave's best to the owner at wrapup.
- B4b adds nothing to promote: null on both readouts, mechanism refuted, ANOMALOUS on VAD.
- The strongest one-factor result in the wave so far is still B4a's final readout: positive, Δ +1.02 [+0.43, +1.60], clean +0.35, registry decision `promote`. Its mechanism is refuted and its merge CI includes 0.
- If the owner promotes B4a:
  - Flags: B4a's exact `config.yaml` command (`scripts/run_B4a.sh`) at 2-3 seeds (1234, 1235, 1236).
  - Readout to submit: the final-epoch EMA checkpoint (B4a: 93.15 vs merge 92.55 on half B).
  - Keep `--keep_topk_by_wf1 7 --keep_epochs 8`.
- Pre-launch checks:
  - `bash -n scripts/run_B4a.sh`;
  - `which ffmpeg` (a container restart wipes it and the codec stage silently no-ops);
  - `pyloudnorm` and `pywebrtc_audio` importable in the rtc-sdd env;
  - an empty `pgrep -af "b4wave.train|b3wave.train|main_train"`;
  - an idle GPU.
- Post-launch checks:
  - `B4 APM NS selftest: PASS`;
  - `B4 APM NS: on p=0.3 …`;
  - `Warmup-cosine: 1895 warmup / 37896 steps`;
  - epoch-1 TrainAcc within 2 of B4a's.
- Every evaluation needs `--loudness_norm --loudness_norm_lufs -23`.
