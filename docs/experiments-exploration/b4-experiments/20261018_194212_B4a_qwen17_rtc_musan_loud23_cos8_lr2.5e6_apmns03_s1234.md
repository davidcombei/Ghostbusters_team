# Experiment exploration B4-01: B4a (`B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234`) - WebRTC NS pass (apm_ns) on top of B2f

> Written 2026-10-18 by the exploop B4 analysis agent (ANALYZE step, wave B4, iteration 1, attempt a0).
>
> **Sources.** Every number below was copied from a file read in this session or computed in this session from the half_B rows of the local `scores/*.txt` / `*.logits` files. Nothing is quoted from memory. The files:
> - run config and logs: `config.yaml` and `git.diff` (B4a), `config.yaml` (B2f), `logs/train.log` and `metrics.jsonl` (both runs), `top5_by_wf1/{merge_list.txt,select_topk.log}`, the `ckpt/` listing;
> - local evals: `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,compare_vs_B2f-*.txt,ci_vs_B2f-*.txt,compare_ci_vs_B2f-*.json}`, `local_eval/merge_top5/compare_all_readouts.txt`, `local_eval/merge_top5.log` (B4a), and the B2f readouts `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,scores/}`;
> - stdout logs, header lines only: B4a `SLS_setup/B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234_20261018145739.log` and B2f `SLS_setup/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_20261009132522.log`;
> - loop files: `exploop/B4/iter01/{verdict.json,design.result.json,log_merge.txt,log_final.txt}`, `exploop/B4/{wave.json,plan_snapshot.md,STATUS.md,user_notes.md}`, `scripts/run_B4a.sh`;
> - the pre-registered mechanism readout `scripts/b4_dsp_slices.py --lever apm_ns`. It was run in this session on `local_eval/epoch_8` (primary) and `local_eval/merge_top5`, with the default `--ref` (B2f, same readout) and the pinned null panel, and with no `--placebo`, because no other B4 arm exists yet. Output went to stdout only. The rules are in `SLS_setup/b4wave/README.md` ("Mechanism readout and verdict rules");
> - registry and code: `docs/experiment_log.csv` (rows `B2f-loudnorm-*`, `B4a-apmns-*`), `SLS_setup/b4wave/train.py` (B4 flag definitions), `SLS_setup/utils/local_eval.py` and `configs/local_eval_sets.yaml` (half split, WF1), `data/dev_noisy_sim/*/meta.csv` (half, lang, family, codec, `dsp_chain`).
>
> Progress-split scores (`model_merging_top5/scores.txt`, `progress.txt`) were not read.
>
> **Previous reports.** None in wave B4 (this is iteration 1). The structure follows `docs/experiments-exploration/b3-experiments/20261010_143058_B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234.md`.
>
> **Readouts.**
> - `-top5` is the weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5/`). It is the wave objective.
> - `-ep8` is the final-epoch EMA checkpoint alone (`local_eval/epoch_8/`).
> - half_B is the half of each sim set that is never used for selection: n = 1,994 per sim set, 1,603 spoof and 391 bona. Clean is the full 7,465-clip `dev_online_clean` with 1,397 bona (`metrics.json`).
> - WF1 = 0.3 clean + 0.35 matched + 0.35 heldout (macro-F1 at p_spoof 0.5). Echo is reported but is not in WF1.
> - Threshold 0.5 on the half_B score rows reproduces all four half_B WF1 values of `metrics.json` exactly: 91.98 / 92.55 / 92.14 / 93.15.
> - All four readouts were evaluated with loudness normalisation: `"loudness_norm": -23.0` in every `metrics.json`, and `[loudness_norm] loudness_norm target=-23 LUFS` in `merge_top5.log`.

Run directories (under `SLS_setup/exp/`):
- **B4a** `B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234_epoch8_bs16_20261018150134` (this report)
- **B2f** `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623`: the reference for the whole wave (`wave.json` `rules.new_best_delta` 100 = screen design). Its half_B scores are merge 91.98 and final 92.14.

## 1. Setup

**B4a = B2f with one factor changed: `--aug_apm_p 0.3`.** Each training clip has a 30 % chance of one pass through the standalone WebRTC NoiseSuppressor from `pywebrtc_audio`. Only NS runs (no HPF, AGC or AEC). The suppression level is drawn uniformly from `--aug_apm_ns_levels`, left at the default `6 12 18 21` dB. The pass runs after the acoustic noise/RIR stage and before the codec, on both classes, from a separate RNG stream (`b4wave/train.py` help; `scripts/run_B4a.sh`).

Flag diff (`config.yaml` of both runs, `diff`):

| flag / field | B2f | **B4a** | note |
|---|---|---|---|
| trainer | `main_train.py`, git `00da485`, dirty | **`b4wave/train.py`** (b4wave 0.1.0 on b3wave 0.1.0), git `8ea996c`, dirty | `git.diff` touches only `.claude/skills/post-train-eval/SKILL.md` (docs). `SLS_setup/b4wave/` is untracked in git, so the code is identified by its module hash (below). |
| `--aug_apm_p` | (absent = 0) | **0.3** | the lever |
| `--aug_apm_ns_levels` | - | 6 12 18 21 (default) | coupled with `--aug_apm_p` (`wave.json` `coupled`) |
| other B4 flags | - | `aug_dtx_p 0.0`, `aug_opus_plc_p 0.0` (off) | |
| B3 flags (written by the b3wave parser) | - | defaults: `sampler shuffle`, `aug_snr_db 5 20 uniform`, `aug_codec_profiles fixed`, `consistency none`, `llrd 1.0`, `freeze_bottom_k 0`, `specaug false` | equal to B2f's behaviour |
| everything else | identical | identical | `--batch_size 16 --num_epochs 8 --lr 2.5e-6 --lr_min 5e-7 --warmup_frac 0.05 warmup_cosine adamw wd 0.01 head x10 grad_clip 1 ema 0.9995`, `ce_weights 0.1 0.9`, `label_smoothing 0`, RTC aug + MUSAN, `--loudness_norm -23`, `--keep_topk_by_wf1 7 --keep_epochs 8`, seed 1234 |

**Header lines proving the change took effect** (B4a stdout log; the B2f stdout log is shown for comparison):
- `B4 APM NS: on p=0.3 levels_db=[6, 12, 18, 21] backend=pywebrtc_audio.NoiseSuppressor hpf=off agc=off`. This matches the design's `expect` regex.
- `B4 APM NS selftest: PASS shape_finite ok | no_gain dB 6:-6.02 12:-12.38 18:-18.40 21:-21.47 | monotone removed dB 6:5.99 12:11.89 18:17.53 21:20.83 | no_hpf … | aligned lag 6:0 12:0 18:0 21:0`.
- The `RTC augmentation:` line ends with `| wrtc_ns p=0.3 levels_db=[6, 12, 18, 21]`. B2f's line ends at the codec list.
- `b3wave: 0.1.0`, `b4wave: 0.1.0`, `b4wave sha256: b6498bc4a5af`.
  - `wave.json` records the rdloop code sha `d163f934…`. The two hashes use different definitions: rdloop's `code_sha` hashes the relative paths of the module and script files (`scripts/rdloop/rdlib.py`), while b4wave's `module_sha256` hashes the file names of the module's `.py` files (`b4wave/__init__.py`). So the two hex strings cannot be compared.
  - Every `b4wave/*.py` was last modified on 2026-10-17. The wave was created on 2026-10-18 11:34. There is no evidence of a code change.
- Unchanged lines: `Sampler: shuffle`, `SNR: 5..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: fixed`, `Consistency: none`, `LLRD: 1`, `SpecAugment: off`, `Loudness norm: loudness_norm target=-23 LUFS`, `CE weights: spoof=0.1 bonafide=0.9`, `Parameters: 320268933`, `Train trials: 75785`, `Warmup-cosine: 1895 warmup / 37896 steps, floor 5.0e-07`. The last line is identical to B2f's.
- The stdout log has no `Traceback`, `out of memory` or `failed (` line (grep count 0).

**Realised NS dose** (`metrics.jsonl` `b3.aug_counts` and `b4.aug_by_class`; `train.log` `B4 aug-by-class` lines):

| epoch | NS clips (sum of `wrtc_ns6/12/18/21`) / 75,785 | bona share (n = 14,353) | spoof share (n = 61,432) | stage share | codec calls |
|---|---|---|---|---|---|
| 1 | 22,655 = 0.2989 | 0.2974 | 0.2993 | 0.8005 | 75,785 |
| 2-8 range | 0.2982-0.3022 | 0.2923-0.3097 | 0.2976-0.3018 | 0.7967-0.8011 | 75,785 every epoch |

The dose matches p = 0.3, it is class-balanced, and the four levels are drawn about equally (epoch 1: 5,622 / 5,678 / 5,634 / 5,721).

**Run time.**
- `config.yaml` was created 15:01:52. The epoch-8 local eval finished at 19:14:38 (`train.log`), i.e. **4 h 12 min**. The registry `gpu_h` is 4.21, against B2f's 4.14.
- Epochs 2-8 took 31:01-31:12 min, against B2f's 30:24-30:35 (eval-end to epoch-end timestamps in both `train.log`s), i.e. about +0.6 min per epoch.
- Post-train timestamps: `top5_by_wf1/merge_list.txt` 19:15, `merge_top5/metrics.json` 19:26, `epoch_8/metrics.json` 19:30, `model_merging_top5/submission.zip` 19:41.

**Checkpoints.** Nothing was pruned: all 8 epoch files are present at 1,281,235,691 bytes each. The symlinks are:
- `best_by_dev_loss` → epoch 1 (0.075466);
- `best_by_noisy_eer` → epoch 6;
- `best_by_wf1` → epoch 8.

**Merged top-5 by half-A mini WF1: epochs 8, 6, 7, 4, 5** (93.28, 92.68, 92.58, 92.16, 92.09; mean 92.56; `select_topk.log`). Excluded: 3 (92.01), 2 (90.75), 1 (87.71). B2f's merge was 8, 5, 7, 3, 4.

## 2. Results vs references (half_B)

Sources: `compare_all_readouts.txt`, `compare_vs_B2f-*.txt`, `ci_vs_B2f-*.txt` / `compare_ci_vs_B2f-*.json` (paired bootstrap, n = 2,000, half B, threshold 0.5), and the `half_B` block of each `metrics.json`. Deltas are computed from unrounded values. `compare_all_readouts.txt` gives every delta against its first column (B2f-top5), so the B4a-ep8 − B2f-ep8 deltas below come from `compare_vs_B2f-ep8.txt`, `compare_ci_vs_B2f-ep8.json` and `metrics.json`.

### 2.1 Wave iterations and fixed comparators, both readouts

B4a is the only iteration so far.

| half_B | B2f-top5 | B2f-ep8 | **B4a-top5** | **B4a-ep8** | B4a-top5 − B2f-top5 [95 % CI] | B4a-ep8 − B2f-ep8 [95 % CI] |
|---|---|---|---|---|---|---|
| **WF1@0.5** | 91.98 | 92.14 | **92.55** | **93.15** | **+0.57 [−0.01, +1.17]**, p(not better) 0.029 | **+1.02 [+0.43, +1.60]**, p 0.001 |
| WF1@oracle | 92.59 | 92.82 | 92.98 | 93.26 | +0.39 | +0.43 |
| clean F1@0.5 | 97.60 | 97.69 | 97.72 | 98.05 | +0.12 [−0.20, +0.45] | **+0.35 [+0.05, +0.65]** |
| noisy F1@0.5 (mean matched, heldout) | 89.57 | 89.76 | 90.34 | 91.06 | +0.77 | +1.30 |
| mean noisy EER | 10.09 | 9.66 | 9.53 | 9.16 | −0.55 | −0.51 |
| calibration gap noisy | 0.66 | 0.71 | 0.49 | 0.05 | −0.17 | −0.66 |
| separability gap | 7.87 | 7.86 | 7.18 | 7.17 | −0.69 | −0.70 |
| oracle gap (WF1@or − WF1@.5) | 0.61 | 0.69 | 0.43 | 0.10 | −0.19 | −0.59 |

Per-set paired CIs:

| set | merge Δ [CI] | ep8 Δ [CI] |
|---|---|---|
| clean | +0.12 [−0.20, +0.45] | +0.35 [+0.05, +0.65] |
| matched | **+1.24 [+0.19, +2.39]** | **+1.56 [+0.52, +2.65]** |
| heldout | +0.30 [−0.84, +1.51] | +1.04 [−0.20, +2.33] |
| echo | +0.63 [−0.27, +1.56] | +0.10 [−0.94, +1.01] |

Weighted contributions (0.3 / 0.35 / 0.35):
- merge: +0.04 clean + 0.43 matched + 0.10 heldout = +0.57;
- final: +0.11 + 0.55 + 0.36 = +1.02.

**Matched carries three quarters of the merge gain and half of the final gain.** The targeted set, heldout, carries 0.10 / 0.36, and its CI includes 0 on both readouts.

**On the other halves of the same sets** (the registry columns are full sets; half A was computed from `scores/*.txt`):

| readout | full-set WF1 (B2f → B4a) | half_A WF1 (B2f → B4a) | half_A heldout F1 (B2f → B4a) |
|---|---|---|---|
| merge | 92.25 → 92.67 (+0.42) | 92.52 → 92.78 (+0.26) | 89.69 → 88.46 (**−1.23**) |
| final | 92.53 → 93.23 (+0.70) | 92.94 → 93.32 (+0.38) | 89.67 → 89.92 (+0.25) |

The gain is smaller on half A, which is the selection half. On the merge, the half-B heldout gain (+0.30) turns into a loss on half A. That loss is also why the registry's full-set merge heldout is down (`log_merge.txt`: "heldout -0.45").

### 2.2 Per set (R = recall at 0.5; thr = p_spoof threshold; `metrics.json` half_B)

| set | metric | B2f-top5 | **B4a-top5** | B2f-ep8 | **B4a-ep8** |
|---|---|---|---|---|---|
| dev_online_clean | F1@.5 / F1@or / thr_or | 97.60 / 98.10 / 0.997 | 97.72 / 98.00 / 0.996 | 97.69 / 98.33 / 0.999 | **98.05** / 98.27 / 0.999 |
| | EER / ECE | 3.42 / 1.42 | **2.11** / 1.32 | 3.33 / 1.33 | **2.21** / 1.15 |
| | R_spoof / R_bona | 99.90 / 92.84 | 99.85 / 93.41 | 99.95 / 92.91 | 99.88 / **94.27** |
| sim_matched_v1 | F1@.5 / F1@or / thr_or | 91.22 / 91.53 / 0.846 | **92.46** / 92.74 / 0.765 | 91.24 / 92.05 / 0.991 | **92.80** / 92.80 / 0.534 |
| | EER / ECE | 8.15 / 4.75 | 7.42 / 4.24 | 8.21 / 4.86 | 7.54 / 4.19 |
| | R_spoof / R_bona | 99.50 / 76.21 | 99.50 / **79.54** | 99.44 / 76.47 | 99.25 / **81.33** |
| sim_heldout_v1 | F1@.5 / F1@or / thr_or | 87.92 / 88.94 / 0.957 | 88.21 / 88.91 / 0.858 | 88.27 / 88.89 / 0.949 | **89.31** / 89.41 / 0.743 |
| | EER / ECE | 12.03 / 6.46 | 11.65 / 6.45 | 11.11 / 6.38 | 10.77 / 5.95 |
| | R_spoof / R_bona | 99.00 / 69.31 | **98.57** / 71.36 | 99.13 / 69.82 | 98.81 / **73.40** |
| sim_echo_v1 | F1@.5 / F1@or / thr_or | 94.16 / 94.91 / 0.983 | 94.79 / 95.22 / 0.977 | 94.53 / 95.41 / 0.983 | 94.63 / 94.83 / 0.985 |
| | EER / ECE | 4.71 / 3.37 | 4.86 / 2.98 | 4.86 / 3.20 | 4.61 / 3.16 |
| | R_spoof / R_bona | 99.75 / 83.38 | 99.75 / 85.17 | 99.69 / 84.65 | 99.63 / 85.17 |

Reading:
- **Bona recall rose on every set and both readouts.** Matched rose +3.32 / +4.86, heldout +2.05 / +3.58, clean +0.57 / +1.36 and echo +1.79 / +0.51.
- **Spoof recall slipped slightly on the noisy sets.** Heldout fell −0.44 / −0.31 and matched 0.00 / −0.19, all within the 0.5 guard.
- **Clean EER fell by more than a point** (−1.31 / −1.12), but clean F1@oracle did not move (−0.10 / −0.06). The ranking gain on clean is at the EER operating point, not at the F1 optimum.
- Paired per clip (half_B score files, B4a vs B2f; logit correlation over all half-B clips of the set):

  | set | class | merge fixed / broken / wrong in both | ep8 fixed / broken / wrong in both | logit corr merge / ep8 |
  |---|---|---|---|---|
  | clean | bona | 23 / 15 / 77 | 28 / 9 / 71 | 0.992 / 0.992 |
  | matched | bona | 23 / 10 / 70 | 23 / 4 / 69 | 0.983 / 0.979 |
  | heldout | bona | 18 / 10 / 102 | 23 / 9 / 95 | 0.976 / 0.968 |
  | echo | bona | 15 / 8 / 50 | 13 / 11 / 47 | 0.986 / 0.982 |
  | heldout | spoof | 3 / **10** | 5 / **10** | |
  | clean / matched / echo | spoof | 5 / 8, 4 / 4, 2 / 2 | 2 / 6, 3 / 6, 2 / 3 | |

- Half-B spoof misses (p < 0.5), B2f → B4a:
  - heldout 16 → **23** (merge) and 14 → **19** (ep8);
  - clean 6 → 9 and 3 → 7;
  - matched 8 → 8 and 9 → 12;
  - echo 4 → 4 and 5 → 6.

### 2.3 Probes (`probe_report.json`; 187 bona + 813 spoof; one bona probe clip = 0.53 points)

Under `--loudness_norm`, the pure-gain probes (`gain_down`, `gain_m20`, `gain_m6`) are near-identical to `probe_orig` by construction. Their flips are 0.00 %, except B4a-ep8 `gain_m20` / `gain_m6`, which flip 0.53 % (one clip) each.

| probe (bona unless stated) | B2f-top5 | **B4a-top5** | B2f-ep8 | **B4a-ep8** |
|---|---|---|---|---|
| level_flip_bona (mean) / max | 1.54 / 4.81 | **0.87** / 2.14 | 1.74 / 4.81 | **1.40** / 3.74 |
| gain_up / gain_p6 / gain_p10_limit flip % | 4.28 / 2.14 / 4.81 | 2.14 / 1.60 / 1.60 | 4.81 / 2.67 / 3.21 | 2.67 / 2.14 / 3.74 |
| agc_dynaudnorm / loudnorm flip % | 0.53 / 0.53 | 1.07 / 0.53 | 2.14 / 1.07 | 0.53 / 1.07 |
| level_flip_spoof | 0.03 | 0.05 | 0.03 | 0.03 |
| probe_orig bona pred spoof % (`pred_spoof_orig`) | 10.70 | 10.70 | 10.16 | **8.02** |
| **noise_only bona-source pred spoof % / mean dP** | 100.00 / +0.8870 | 100.00 / +0.8921 | 100.00 / +0.8965 | 100.00 / **+0.9202** |
| **sil_only bona-source pred spoof % / mean dP** | 100.00 / +0.8929 | 100.00 / +0.8911 | 100.00 / +0.8985 | 100.00 / **+0.9203** |
| trimmed / padded bona flip % | 3.74 / 3.74 | 2.67 / 2.67 | 3.74 / 5.35 | 2.14 / 2.14 |

Reading:
- The NS pass made the model slightly less level-sensitive (level_flip_bona 1.54 → 0.87 merge) and less sensitive to padding (padded flips 5.35 → 2.14 ep8).
- **It did not touch the non-speech failure.** Bona-source `noise_only` and `sil_only` clips are still 100 % spoof, and at the final epoch they are more confident (mean dP +0.8985 → +0.9203 on `sil_only`).

### 2.4 Slices that moved most, B4a vs B2f (full sim sets, `metrics.json → breakdowns`, 63 slice rows)

Counts of slice rows that improved:

| readout | F1 up | EER down | R_bona up | R_spoof up |
|---|---|---|---|---|
| merge | 45 of 63 | 46 | 49 | **15** |
| final | 45 of 63 | 48 | 50 | **15** |

- F1 up per set, merge / ep8: **matched 23 / 19 of 23**, heldout 7 / 16 of 23, echo 15 / 10 of 17.
- Minimum slice R_spoof:
  - B4a-top5 96.97 (heldout / door_wood_creaks);
  - B4a-ep8 96.63 (same slice);
  - B2f-top5 97.31 (same slice);
  - B2f-ep8 97.68 (heldout / washing_machine).

| set / slice | n | F1 merge B2f → B4a | F1 ep8 B2f → B4a | EER ep8 | R_bona merge / ep8 |
|---|---|---|---|---|---|
| heldout / vacuum_cleaner | 373 | 83.27 → 86.68 +3.41 | 82.01 → 87.43 **+5.42** | 18.36 → 17.15 | 60.0 → 67.1 / 57.1 → 70.0 |
| matched / snr 5 | 656 | 88.33 → 91.75 +3.42 | 89.31 → 91.63 +2.32 | 10.20 → 10.01 | 71.2 → 79.7 / 73.7 → 81.4 |
| matched / opus:qq | 572 | 84.50 → 88.08 **+3.58** | 86.04 → 87.68 +1.64 | 10.95 → 8.67 | 61.6 → 69.7 / 66.7 → 68.7 |
| matched / opus:telegram | 552 | 90.42 → 92.84 +2.42 | 91.25 → 93.89 +2.64 | 6.78 → 8.59 | 74.8 → 82.0 / 78.4 → 86.5 |
| echo / opus:qq | 603 | 91.48 → 93.93 +2.45 | 91.91 → 93.67 +1.76 | 5.09 → 5.92 | 76.7 → 83.3 / 79.2 → 83.3 |
| matched / echo family (no noise) | 678 | 93.76 → 96.30 +2.53 | 94.48 → 95.84 +1.37 | 4.84 → 4.74 | 82.6 → 89.0 / 84.5 → 87.7 |
| matched / snr 0 | 711 | 84.97 → 87.18 +2.20 | 86.38 → 87.58 +1.19 | 14.67 → 15.44 | 62.4 → 68.4 / 65.8 → 71.8 |
| matched / lang en | 1538 | 86.76 → 88.53 +1.77 | 86.71 → 87.77 +1.06 | 12.68 → 11.44 | 63.6 → 69.0 / 63.2 → 69.0 |
| heldout / lang zh | 2462 | 90.86 → 91.68 +0.82 | 91.15 → 93.00 +1.85 | 8.68 → 7.64 | 78.3 → 80.5 / 78.0 → 83.1 |
| **heldout / lang en** | 1538 | 84.45 → 81.44 **−3.01** | 84.47 → 82.66 **−1.81** | 16.34 → 17.08 | 59.7 → 56.6 / 60.5 → 58.9 |
| **heldout / wind** | 381 | 87.17 → 83.10 **−4.07** | 88.36 → 86.26 −2.11 | 13.14 → 14.61 | 68.4 → 60.5 / 72.4 → 67.1 |
| heldout / codec g726 | 1274 | 87.79 → 86.10 −1.69 | 88.85 → 87.57 −1.27 | 11.18 → 9.80 | 68.3 → 64.2 / 70.3 → 67.5 |
| heldout / snr 5 | 778 | 88.99 → 87.44 −1.55 | 89.72 → 88.43 −1.29 | 10.83 → 10.27 | 74.4 → 71.8 / 76.3 → 72.4 |
| heldout / washing_machine | 380 | 83.34 → 81.44 −1.90 | 84.78 → 84.96 +0.19 | 16.45 → 14.00 | 62.8 → 57.7 / 65.4 → 64.1 |

Pattern:
- **The gain is the matched set, i.e. the training-distribution noises and Opus apps, and Chinese.**
- **Heldout English, wind, g726 and snr 5 got worse.**
- No eval set contains WebRTC NS (grep of each `meta.csv`):
  - `matched_v1` and `echo_v1` have no NS stage at all (0 `afftdn`, `anlmdn` or `webrtc` rows);
  - `heldout_v1` has 702 `afftdn` and 709 `anlmdn` rows, all ffmpeg suppressors.

Half-B splits of heldout bona (computed from `scores/*.txt` + `meta.csv`; NS = `dsp_chain` contains `afftdn` or `anlmdn`):

| half-B heldout bona slice | n | R_bona merge B2f → B4a | R_bona ep8 B2f → B4a | spoof misses merge / ep8 (B2f → B4a) |
|---|---|---|---|---|
| offline, NS | 146 | 66.4 → 68.5 (+2.1) | 67.1 → 69.9 (+2.8) | 9 → 10 / 9 → 8 |
| offline, no NS | 55 | 76.4 → 81.8 (**+5.4**) | 74.5 → 81.8 (**+7.3**) | 2 → 2 / 1 → 2 |
| online (no NS in this set) | 190 | 69.5 → 70.5 | 70.5 → 73.7 | 5 → **11** / 4 → **9** |
| afftdn | 74 | 55.4 → 59.5 | 58.1 → 62.2 | 4 → 5 / 4 → 4 |
| anlmdn | 72 | 77.8 → 77.8 | 76.4 → 77.8 | 5 → 5 / 5 → 4 |
| en | 129 | 59.7 → **56.6** | 58.9 → 58.9 | 3 → **10** / 5 → **9** |
| zh | 262 | 74.0 → 78.6 | 75.2 → 80.5 | 13 → 13 / 9 → 10 |
| gsm | 127 | 63.0 → 68.5 | 63.8 → 71.7 | |
| g726 | 129 | 64.3 → 62.8 | 65.9 → 65.1 | |

**The untreated offline bona gained more than the NS-treated offline bona.** This is the opposite of the lever's hypothesis; section 2.6 has the calibrated test.

### 2.5 Lever response (one factor per row)

| run | change vs B2f | half_B merge WF1 (Δ) | final-epoch WF1 (Δ) | merge oracle (Δ) | final oracle (Δ) | merge noisy EER (Δ) | final EER (Δ) | merge clean (Δ) | final clean (Δ) | verdict (`verdict.json`) | mechanism (`b4_dsp_slices.py`, provisional) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B2f | reference (start of wave) | 91.98 | 92.14 | 92.59 | 92.82 | 10.09 | 9.66 | 97.60 | 97.69 | - | - |
| **B4a** | **+ `--aug_apm_p 0.3`** | **92.55 (+0.57)** | **93.15 (+1.02)** | 92.98 (+0.39) | 93.26 (+0.43) | 9.53 (−0.55) | 9.16 (−0.51) | 97.72 (+0.12) | 98.05 (+0.35) | merge: above band, CI includes 0 / drop; final: **positive** / drop; new_best false (screen) | final: **REFUTED**; merge: DIRECTIONAL (0 placebo arms) |

Null-panel context, from the `b4_dsp_slices.py` "Outcome" block (9 earlier runs, half B):
- The heldout F1 panel has mean 88.50 and SD 1.35 at the final readout, and mean 88.25 and SD 1.39 at the merge.
- A heldout F1 gain of ≥ +0.8 over B2f is reached by 4 of 9 panel runs (final) and 3 of 9 (merge).
- B4a's heldout +1.04 (final) is therefore within the panel's spread.

### 2.6 Mechanism readout (pre-registered; `scripts/b4_dsp_slices.py --lever apm_ns`, full sets, arm vs B2f same readout)

Statistic: DiD = the mean clipped-logit change on the treated bona slice minus that on the control bona slice. Negative means the treated slice moved toward bona. It is calibrated on the 9-run null panel.

| contrast | readout | arm recall T/C (B2f final 69.2/75.5 for NS) | DiD [CI] | DiD* / z | clip DiD | half-B DiD merge / final | criteria (a)-(e) | verdict |
|---|---|---|---|---|---|---|---|---|
| **NS** (offline NS vs no NS, n 276/106; refute line +0.32, shown bar −0.22) | **final (primary)** | 70.7 / 80.2 | **+0.49 [−0.21, +1.22]** | +0.17 / +0.61 | −9.0 | +0.23 / +0.62 | n n n n − | **REFUTED (provisional)** |
| NS | merge | 68.8 / 77.4 | −0.03 [−0.63, +0.57] | −0.05 / −0.15 | +6.6 | +0.23 / +0.62 | n n n n − | DIRECTIONAL (provisional) |
| NS_afftdn_vs_anlmdn (secondary, n 139/137; pred. < −0.84, strong ≤ −1.70) | final | 69.8 / 71.5 | **−1.39 [−2.14, −0.69]** | −0.56 / −1.30 | +16.1 | −0.76 / −0.51 | n n Y n − | DIRECTIONAL |
| NS_afftdn_vs_anlmdn | merge | 66.2 / 71.5 | −1.14 [−1.83, −0.49] | −0.36 / −0.88 | +14.1 | −0.76 / −0.51 | n n n n − | DIRECTIONAL |
| VAD (non-target; dtx_gate's contrast) | final / merge | 69.1/75.6, 69.9/72.0 | −0.22 / −0.54 | +0.35 / +0.02 | | | | REFUTED / REFUTED |
| PLC_online (non-target; opus_plc's contrast) | final / merge | 74.6/77.2, 71.1/74.4 | −0.50 [−1.20, +0.18] / −0.61 [−1.09, −0.18] | +0.07 / +0.03 | | | | REFUTED / REFUTED |

Other blocks of the same output:
- **Lever verdict (apm_ns), final readout: `REFUTED (provisional) [placebos 0] lower-p retry allowed: False`.** Merge: `DIRECTIONAL (provisional)`.
- **Falsifier (1) outcome escape:** `heldout F1 delta >= +0.5 on merge_top5 AND final: False` (heldout +0.30 merge, +1.04 final).
- **Run anomaly (O3.1): none.** No |z| > 3 on a non-target primary contrast. Under-fit False. Clean-F1 guard False.
- **Spoof-miss guards (Poisson 99 % bound), all held:**

  | guard | merge B2f → B4a | final B2f → B4a | bound merge / final |
  |---|---|---|---|
  | NS_offline | 18 → 20 | 18 → 15 | ≤ 29 / ≤ 29 |
  | VAD_offline | 15 → 14 | 13 → 14 | ≤ 25 / ≤ 22 |
  | PLC | 16 → 23 | 15 → 19 | ≤ 26 / ≤ 25 |
  | halfB_noisy | 24 → 31 | 23 → 31 | ≤ 36 / ≤ 35 |

- **Pinned tertiles (dtx_gate's criterion (ii)).** On the full set, final readout, B4a has saturated noisy bona FA of 67 / 62 / 98 (low / mid / high tertile), against B2f's 66 / 76 / 107. drop_high − drop_low = **+10**, against the anchor's "need ≥ +5" and a null panel of mean −10.2 / max +0 (z +3.51). On half B it is −8 (final) and −5 (merge), against "need ≥ +0". **B4a, which never saw DTX, already passes the full-set part of dtx_gate's anchor.**

## 3. Per-epoch behaviour, B4a vs B2f (same LR at each epoch end: 2.469e-6 … 5.000e-7 in both `train.log`s)

Sources: `logs/train.log` (both runs) and `metrics.jsonl` (EMA weights; 2,000 clean clips + `sim_matched_mini` + `sim_heldout_mini`, half A). Δ is B4a − B2f on the `train.log` values, as in `STATUS.md`.

| ep | B4a mini WF1 | B2f mini WF1 | Δ | B4a WF1@or | B4a EER noisy (B2f) | B4a R_bona cl/ma/he | B2f R_bona cl/ma/he | mini heldout F1 B4a / B2f | train loss B4a / B2f | train acc B4a / B2f | dev loss B4a / B2f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 87.71 | 88.34 | −0.63 | 90.01 | 13.34 (13.40) | 94.9/83.2/78.1 | 92.7/81.6/78.6 | 82.82 / 84.10 | 0.3609 / 0.3558 | 86.88 / 87.36 | 0.0755 / 0.0928 |
| 2 | 90.75 | 91.47 | −0.72 | 91.51 | 11.89 (11.78) | 91.9/76.0/71.9 | 91.6/77.1/76.0 | 86.65 / 87.78 | 0.2196 / 0.2075 | 96.93 / 97.13 | 0.0819 / 0.0982 |
| 3 | 92.01 | 91.90 | +0.11 | 92.40 | 11.10 (10.38) | 92.7/79.9/71.4 | 91.6/74.9/71.4 | 87.59 / 88.47 | 0.1584 / 0.1546 | 97.98 / 98.09 | 0.0860 / 0.1139 |
| 4 | 92.16 | 91.69 | +0.47 | 92.74 | 11.50 (10.64) | 91.3/79.3/71.9 | 91.3/75.4/71.4 | 88.52 / 88.32 | 0.1225 / 0.1108 | 98.56 / 98.62 | 0.1040 / 0.1242 |
| 5 | 92.09 | 92.49 | −0.40 | 92.67 | 10.80 (10.46) | 91.3/80.4/**68.4** | 90.4/78.8/71.9 | 87.84 / 89.27 | 0.1013 / 0.0879 | 98.82 / 98.96 | 0.1051 / 0.1178 |
| 6 | 92.68 | 91.66 | **+1.02** | 93.22 | **9.55** (10.63) | 92.4/81.6/73.5 | 91.3/74.9/70.4 | 88.97 / 88.36 | 0.0746 / 0.0725 | 99.11 / 99.18 | 0.1037 / 0.1321 |
| 7 | 92.58 | 92.10 | +0.48 | 92.86 | 9.78 (10.69) | 92.7/80.4/75.5 | 91.3/76.0/73.0 | 89.16 / 89.37 | 0.0592 / 0.0540 | 99.34 / 99.36 | 0.1204 / 0.1406 |
| 8 | **93.28** | 92.80 | +0.48 | **93.60** | 9.63 (10.94) | 93.8/81.6/76.0 | 91.3/80.4/73.5 | **90.39** / 89.41 | 0.0562 / 0.0501 | 99.36 / 99.44 | 0.1166 / 0.1442 |

Reading:
- **The best epoch is the last, and the curve is still rising.** Mini WF1 rose +0.70 from epoch 7 to 8 (93.28, oracle 93.60). `best_by_wf1` is epoch 8, and the best noisy EER is at epoch 6 (9.55). B2f also peaked at epoch 8.
- **B4a starts behind and leads from epoch 3, except at epoch 5.**
  - Epoch 1 is −0.63, with TrainAcc 86.88 against 87.36. That is within the design's ±2 band.
  - The lead is mostly mini matched. Mini heldout F1 trails B2f at epochs 1, 2, 3, 5 and 7, and leads only at 4, 6 and 8.
- **Train loss is higher than B2f's at every epoch, and dev loss is lower at every epoch.** At epoch 8: train loss 0.0562 vs 0.0501, dev loss 0.1166 vs 0.1442.
  - Both runs' dev loss rises by the same ratio over the run: 0.0755 → 0.1166 (×1.55) against 0.0928 → 0.1442 (×1.55).
  - The NS pass makes training slightly harder and dev loss uniformly lower. It does not change the over-fitting slope.
- **Half-A operating point.** B4a's mini matched `thr_oracle` fell to 0.00-0.01 at epochs 6-8 (B2f: 0.72 / 0.50 / 0.88), and its mini matched spoof recall was 98.8 / 98.7 / 98.7 against B2f's 99.0 / 98.9 / 99.1. At the margin, the matched-mini error is now on the spoof side.
- **The final epoch beats the merge**:
  - half B: 93.15 vs 92.55 (+0.60);
  - half A: 93.32 vs 92.78 (+0.54);
  - half-B heldout: 89.31 vs 88.21; heldout spoof misses 19 vs 23.

  B2f's final also beat its merge (+0.16). The weakest merged epochs by mini WF1 are 5 (92.09) and 4 (92.16). Epoch 5 has the lowest mini heldout bona recall of epochs 3-8 (68.4). No merged epoch was evaluated alone on half B, so its half-B drag is not measured.

## 4. What went OK (evidence)

1. **The final readout is positive on the wave rule** (`verdict.json` final): Δ +1.02, CI [+0.43, +1.60], clean drop −0.35 (i.e. clean rose), `verdict "positive"`. The merge is +0.57 with p(not better) 0.029, i.e. above the band, but its CI low is −0.01.
2. **Clean improved instead of paying.**
   - Clean F1 +0.12 / +0.35 (the final's CI excludes 0).
   - Clean EER 3.42 → 2.11 (merge) and 3.33 → 2.21 (final).
   - Clean bona: 23 fixed / 15 broken (merge), 28 / 9 (final).
   - No guard fired: clean F1, matched F1, R_spoof within 0.5 on every set, NS-offline spoof misses 20 / 15 ≤ 29, ep1 TrainAcc 86.88 ≥ 85.36, ep8 TrainLoss 0.0562 ≤ 0.0752.
3. **The matched set gained broadly and significantly.**
   - +1.24 [+0.19, +2.39] / +1.56 [+0.52, +2.65].
   - F1 rose in 23 of 23 matched slice rows at the merge.
   - Matched bona recall rose +3.32 / +4.86 at unchanged or −0.19 spoof recall.
   - The hardest Opus app moved (matched opus:qq F1 +3.58 merge), and so did matched SNR 0 (+2.20 / +1.19).
4. **The operating point is now right at 0.5 on the final readout.** The calibration gap noisy is 0.05 (B2f 0.71), and one global shift recovers only 0.01 (section 5.2). Noisy EER fell −0.55 / −0.51.
5. **Fewer noisy bona false alarms.** On half B (matched + heldout): 213 → 192 (merge), 210 → 177 (final). Saturated ones (p ≥ 0.99): 136 → 114 and 133 → 123.
6. **Clean experiment.**
   - One factor, confirmed by the header lines and the selftest PASS.
   - Realised NS dose 0.298-0.302 per epoch, class-balanced.
   - No traceback, OOM or `failed (` line; nothing pruned.
   - Both registry rows exist with CI filled: `B4a-apmns-screen` (−0.01 / 1.17) and `B4a-apmns-ep8` (0.43 / 1.60).
   - Cost +0.07 GPU-h (4.21 vs 4.14).

## 5. What went wrong (evidence)

### 5.1 The hypothesised mechanism did not appear; the gain is elsewhere

- **The NS contrast is REFUTED on the primary (final) readout.**
  - NS DiD is +0.49 [−0.21, +1.22], above the +0.32 refute line (null-panel mean). Criteria (a)-(d) all fail on both readouts (section 2.6).
  - On half B, untreated offline heldout bona rose +5.4 / +7.3 recall points against +2.1 / +2.8 for the NS-treated ones.
  - Falsifier (1) fires: the mechanism is refuted and the outcome escape is False (heldout +0.30 on the merge).
- **The only NS-specific signal is secondary.** afftdn-vs-anlmdn DiD is −1.39 [−2.14, −0.69] (final), with 13 fixed / 3 broken (p 0.0213), but its half-B DiD* is positive (+0.11 merge, +0.10 final), so it is directional only.
- **The WF1 gain sits on matched and clean**, two sets with no suppressor stage:
  - 0.43 of the +0.57 and 0.55 of the +1.02 come from matched (section 2.1);
  - the heldout delta is +0.30 [−0.84, +1.51] (merge) and +1.04 [−0.20, +2.33] (final), both CIs including 0;
  - on half A the merge's heldout F1 fell −1.23.
- **The predicted outcome on heldout appeared only on the final readout.** The heldout F1 target was ≥ +0.8; it reached +0.30 on the merge and +1.04 on the final. That size of heldout gain is reached by 4 of 9 null-panel runs (final), so it is not unusual by itself.

### 5.2 Oracle-gap decomposition (global threshold vs set-specific)

Method: one shared logit threshold swept over the half_B `scores/*.logits` (step 0.05, −6 to +12). The per-set oracle WF1 is `half_B.wf1.wf1_oracle`.

| readout | WF1@0.5 | best global (logit / p) | WF1 at best global | per-set oracle WF1 | gap | recovered by one global shift | set-specific remainder |
|---|---|---|---|---|---|---|---|
| B2f-top5 | 91.98 | +3.10 / 0.957 | 92.39 | 92.59 | 0.61 | 0.41 | 0.20 |
| B2f-ep8 | 92.14 | +4.40 / 0.988 | 92.58 | 92.82 | 0.69 | 0.44 | 0.24 |
| **B4a-top5** | 92.55 | **+1.50 / 0.818** | 92.79 | 92.98 | 0.43 | 0.24 | 0.19 |
| **B4a-ep8** | 93.15 | **+0.45 / 0.611** | 93.16 | 93.26 | **0.10** | **0.01** | 0.10 |

B4a removed most of B2f's spoof-side operating-point bias. On the final readout, threshold or calibration levers are worth at most 0.01. **What remains is ranking.**

### 5.3 The saturated bona tail shrank only a little

Share of half_B bona with p_spoof ≥ 0.5 / ≥ 0.9 / ≥ 0.99 / ≥ 0.999 (computed from `scores/*.txt`):

| set (n bona) | B2f-top5 | **B4a-top5** | B2f-ep8 | **B4a-ep8** |
|---|---|---|---|---|
| clean (1,397) | 7.2 / 6.9 / 5.4 / 4.4 | 6.6 / 6.4 / 5.1 / 3.8 | 7.1 / 6.3 / 5.3 / 4.3 | 5.7 / 5.4 / 5.0 / 3.7 |
| matched (391) | 23.8 / 22.3 / 14.6 / 8.2 | 20.5 / 19.7 / 12.0 / 6.6 | 23.5 / 21.5 / 13.6 / 9.0 | 18.7 / 17.6 / 14.1 / 8.2 |
| heldout (391) | 30.7 / 28.6 / 20.2 / 9.2 | 28.6 / 26.9 / 17.1 / 8.4 | 30.2 / 28.4 / 20.5 / 9.5 | 26.6 / 25.8 / 17.4 / 9.5 |
| echo (391) | 16.6 / 15.6 / 10.5 / 4.9 | 14.8 / 14.1 / 10.7 / 5.1 | 15.3 / 14.8 / 10.0 / 5.4 | 14.8 / 14.6 / 11.5 / 5.9 |
| noisy bona (782): FA % / ≥ 0.99 % / saturated share of FA | 27.2 / 17.4 / 64 % | 24.6 / 14.6 / 59 % | 26.9 / 17.0 / 63 % | 22.6 / 15.7 / **69 %** |

- On the final readout the movable band (0.5-0.99) shrank more than the saturated core: noisy FA −33 clips, saturated FA −10.
- **69 % of the remaining noisy bona false alarms are confident ones.**
- `noise_only` / `sil_only` bona are still 100 % spoof (section 2.3).

### 5.4 The worst slices: heldout English, spoof misses on heldout

- **English carries half of what is left, and B4a did not reduce it.**
  - On half B, English is 258 of the 782 noisy bona (matched + heldout).
  - English accounts for 95 of 192 noisy bona FAs (merge) and 91 of 177 (final).
  - B4a removed 0 English FAs (95 → 95) and 21 Chinese ones (118 → 97) at the merge. At the final it removed 8 English (99 → 91) and 25 Chinese (111 → 86).
  - Heldout English bona recall on half B is 56.6 (merge, down from 59.7) and 58.9 (final, flat), against Chinese 78.6 / 80.5.
  - On the full set, heldout en F1 fell −3.01 / −1.81 and its EER rose 16.34 → 17.08 (ep8).
- **Heldout spoof misses rose:** half B 16 → 23 (merge) and 14 → 19 (final). On heldout English spoof they rose 3 → 10 and 5 → 9, and on online heldout spoof 5 → 11 and 4 → 9. The PLC spoof-miss guard is at 23 of its 26 bound on the merge.
- **Worst slices for bona recall**, full sets, B4a-top5 (B4a-ep8 in brackets; from `merge_top5.log` / `metrics.json` breakdowns):

  | slice | R_bona merge (ep8) |
  |---|---|
  | heldout en | 56.6 (58.9) |
  | heldout washing_machine | 57.7 (64.1) |
  | heldout wind | 60.5 (67.1) |
  | heldout g726 | 64.2 (67.5) |
  | heldout snr 0 | 64.6 (69.4) |
  | heldout vacuum_cleaner | 67.1 (70.0) |
  | matched snr 0 | 68.4 (71.8) |
  | matched lang en | 69.0 (69.0) |
  | matched opus:qq | 69.7 (68.7) |
  | matched rain | 69.6 (75.7) |

### 5.5 Overfitting signs

- Dev loss rose 0.0755 → 0.1166 (×1.55) while train accuracy reached 99.36 %. That is the same ratio as B2f (×1.55, 99.44 %), and B4a's dev loss is lower at every epoch. There is no extra over-fitting from the lever.
- The mini WF1 is still rising at epoch 8. The final epoch beats the merge on both halves (+0.60 B, +0.54 A).
- The run did not peak early, so the earlier merged epochs (4, 5) dilute the merge.

## 6. Verdict on the pre-registered rule

From `exploop/B4/iter01/design.result.json`:

- **Falsifier:** "(1) Mechanism refuted: NS DiD >= +0.32 (no better than an average null retrain) AND heldout F1 delta < +0.5 on both readouts -> drop the lever. (2) half_B WF1 delta < -0.5 (outside the ±0.5 band) with matched F1 down > 1.0 -> … drop … (3) Full-set offline-NS spoof misses > 29 on either readout -> the model learned 'NS => bona'; drop. (4) Shared under-fit: ep1 TrainAcc < 85.36 or ep8 TrainLoss > 0.0752 …, together with heldout bona recall down -> confounded. Verdict on the wave rule: positive = merge half_B WF1 delta > +0.5 with CI low > 0 and clean drop <= 0.5. Inside ±0.5 = null on the outcome; the mechanism verdict still stands."
- **Predicted signature (abridged):** mechanism "shown" requires DiD <= -0.22, CI upper < +0.32, clip DiD >= -1.3, half-B DiD < +0.57 (final) / < +0.23 (merge), plus the placebo condition. Secondary: afftdn-vs-anlmdn DiD < -0.84. "Outcome: sim_heldout_v1 F1 up >= 0.8 point from 87.92 / 88.27 (merge / final), driven by recall_bona rising from 69.31 / 69.82 with recall_spoof staying near 99.00 / 99.13. Expected half_B WF1 delta 0 to +0.6 vs 91.98 / 92.14." The must-not-move items are listed in the table below.

`exploop/B4/iter01/verdict.json` (computed by the supervisor; quoted, not recomputed):

| readout | run | base | delta | CI | clean_drop | new_best | decision | verdict |
|---|---|---|---|---|---|---|---|---|
| merge (objective) | 92.55227928958006 | 91.97880641320396 | 0.5734728763761012 | [-0.008530328907396795, 1.1655467607844505] | -0.12025151349281771 | false | **drop** | **above band, CI includes 0** |
| final | 93.15279926126122 | 92.13685698540242 | 1.015942275858805 | [0.4293057843536239, 1.6017211297401048] | -0.3522567851530738 | false | **drop** | **positive** |

Rules (`verdict.json`): `new_best_delta` 100.0 (screen: B2f stays the reference), band 0.5, positive = delta > 0.5 with ci_low > 0 and clean_drop ≤ 0.5.

**On the objective (merge) readout the run is above the band, but the CI includes 0. The final readout is positive. Both readouts read `drop`, because the screen keeps B2f as the reference.**

Falsifiers:

| falsifier | evidence | fired? |
|---|---|---|
| (1) NS DiD ≥ +0.32 AND no heldout escape | NS DiD +0.49 (final, primary); heldout Δ +0.30 merge / +1.04 final. Under owner rule O3.2 (README), escape needs ≥ +0.5 on **both** readouts; the script prints `outcome escape … False`. | **yes**: the lever's mechanism verdict is REFUTED (provisional; 0 placebo arms) |
| (2) WF1 Δ < −0.5 with matched down > 1.0 | +0.57 / +1.02; matched +1.24 / +1.56 | no |
| (3) offline-NS spoof misses > 29 | 20 / 15 | no |
| (4) shared under-fit | ep1 TrainAcc 86.88 (≥ 85.36); ep8 TrainLoss 0.0562 (≤ 0.0752) | no; O3.1 run anomaly: none |

The predicted signature, item by item:

| predicted | merge | final | appeared? |
|---|---|---|---|
| NS DiD ≤ −0.22, CI upper < +0.32 | −0.03, CI upper +0.57 | +0.49, CI upper +1.22 | **no** |
| clip DiD / half-B DiD criteria (c), (d) | script: n / n | script: n / n (half-B final +0.62 vs < +0.57; merge +0.23 vs < +0.23) | **no** |
| placebo (e) | not readable (0 arms) | not readable | deferred to SUMMARY |
| secondary afftdn-vs-anlmdn DiD < −0.84 | −1.14 | −1.39 | **yes** (directional, not ≤ −1.70) |
| heldout F1 up ≥ 0.8 | +0.30 (87.92 → 88.21) | +1.04 (88.27 → 89.31) | **no** / yes |
| driven by heldout recall_bona | 69.31 → 71.36 | 69.82 → 73.40 | yes (smaller than matched's rise) |
| heldout recall_spoof near 99.00 / 99.13 | 98.57 (−0.44) | 98.81 (−0.31) | yes, within 0.5 |
| WF1 Δ 0 to +0.6 | +0.57 | +1.02 | yes / above range |
| clean F1 within 0.5 of 97.60 / 97.69 | 97.72 | 98.05 | yes |
| matched F1 down ≤ 0.5 from 91.22 / 91.24 | 92.46 | 92.80 | yes (rose) |
| R_spoof within 0.5 on every set | worst −0.44 (heldout) | worst −0.31 (heldout) | yes |
| full-set offline-NS spoof misses ≤ 29 | 20 | 15 | yes |
| ep1 TrainAcc within 2 of 87.36 | 86.88 | - | yes |

**Summary.**
- Every guard held, and the outcome was positive on the final readout.
- The pre-registered mechanism did not appear: the NS contrast is refuted on the primary readout.
- The WF1 gain came mostly from the matched set and from clean-set ranking.
- The mechanism verdict is provisional until the wave SUMMARY re-runs `b4_dsp_slices.py --summary` with the opus_plc and dtx_gate arms as placebos.

## 7. Decision and next-experiment proposals

**Decision.** Per `verdict.json`, **B4a is not the new best**: `new_best false` and `decision drop` on both readouts, by the screen design (`new_best_delta` 100).
- **B2f stays the reference**: merge 91.98, final 92.14 on half_B; registry ids `B2f-loudnorm-screen` / `B2f-loudnorm-ep8`.
- Every proposal below is one factor (or one coupled group) against **B2f's flags**, i.e. `scripts/run_B4a.sh` without `--aug_apm_p`, with the `b4wave.train` entry, `--batch_size 16` and 8 epochs.
- The wave rule is unchanged, against B2f-top5 = 91.98:
  - positive = Δ merge > +0.5 with CI low > 0 and clean drop ≤ 0.5;
  - null = Δ within ±0.5;
  - negative = Δ below −0.5.

**Largest remaining error.** Confident noisy bona false alarms, half of them English, with a rising heldout spoof-miss count:
- English is 95 of 192 (merge) / 91 of 177 (final) half-B noisy bona FAs, from 258 of 782 bona;
- heldout English bona recall is 56.6 / 58.9, against 78.6 / 80.5 for Chinese;
- 59 % / 69 % of noisy bona FAs sit at p ≥ 0.99;
- heldout spoof misses rose 16 → 23 / 14 → 19.

On the final readout, threshold or calibration is worth 0.01.

**Ranked proposals (configuration first, seed repeat last):**

1. **`--aug_opus_plc_p 0.3` (opus_plc; the queued iteration b, pre-registered in `wave.json` `lever_queue`).**
   - Targets the heldout online-PLC bona contrast: B2f final recall T/C 67.5 / 75.6, G +0.95 (`b4_dsp_slices.py`). It also targets heldout F1 (87.92 / 88.27) and the online heldout spoof misses that B4a raised (half B 5 → 11 / 4 → 9).
   - Hypothesis (`lever_queue`): concealment artefacts are read as synthesis; exposing both classes to libopus PLC under bursty loss makes concealment class-neutral.
   - **B4a sets the placebo bar.** B4a never saw a dropped packet, yet its online-PLC DiD is already −0.50 (final) / −0.61 (merge), against a panel mean of −0.57. To satisfy (e), opus_plc must go below those values.
   - Predicted signature: online-PLC DiD < −0.57 and below B4a's −0.50 (final); heldout F1 Δ ≥ +0.5 on both readouts; matched qq / wechat bona recall each down ≤ 6 from B2f's 61.62 / 71.15 (README `dsp_slices` row); PLC spoof misses ≤ 26 / 25; clean F1 within 0.5.
   - Falsifier: online-PLC DiD ≥ −0.57 AND heldout F1 Δ < +0.5 on either readout → drop. WF1 Δ < −0.5 with ep1 TrainAcc < 85.36 or ep8 TrainLoss > 0.0752 → under-fit, drop.
2. **`--aug_dtx_p 0.3` (dtx_gate; the queued iteration c).**
   - Targets the pause-rich saturated noisy bona FAs (B2f full final 66 / 76 / 107 by non-speech tertile) and `sil_only` bona at 100 % spoof (mean dP +0.8929 / +0.8985).
   - **B4a weakens the anchor.** B4a already scores drop_high − drop_low = +10 on the full set (z +3.51 against the null panel), but −8 / −5 on half B. The full-set half of criterion (ii) is therefore not specific to gating; the dtx_gate arm has to beat B4a's +10 on the full set and reach ≥ 0 on half B.
   - Predicted signature: criterion (ii) ≥ +5 on the full set (final) and ≥ 0 on half B for both readouts; VAD DiD < −0.57 and below B4a's −0.22 / −0.54; `sil_only` bona mean dP below +0.8929 / +0.8985 (B4a: +0.8911 / +0.9203); clean F1 within 0.5.
   - Falsifier: clean F1 down > 0.5, half-B noisy spoof misses > 36 / 35, or VAD spoof misses > 25 / 22 → drop. A null is "inconclusive (under-dosed)" per the README dose rule, not "refuted".
3. **`--aug_apm_p 0.6` (apm_ns dose; candidate for iteration d if opus_plc is not positive).**
   - Hypothesis: B4a's gain is a generic augmentation effect of the NS pass, not suppressor-artefact neutralisation. The evidence:
     - the NS contrast is refuted (DiD +0.49);
     - untreated heldout offline bona gained more than NS-treated (+5.4 / +7.3 against +2.1 / +2.8);
     - matched has no NS stage, yet its F1 rose in 23 of 23 slices at the merge;
     - dev loss was lower than B2f's at every epoch.

     If the pass works as extra augmentation diversity, doubling the dose should raise the gain.
   - Predicted signature: merge Δ vs B2f > +0.5 with CI low > 0 (what B4a missed: CI low −0.01); final ≥ 93.15; matched F1 ≥ 92.46 / 92.80; clean within 0.5; ep8 dev loss ≤ 0.1166; offline-NS spoof misses ≤ 29.
   - Falsifier: final Δ vs B2f < +0.5, or matched F1 more than 1.0 below B4a's → no dose response, so B4a's gain was not dose-driven (or was seed noise); drop apm_ns. Heldout English half-B bona recall below 56.6 / 58.9 or heldout spoof misses above 23 / 19 → more NS exposure trades heldout for matched.
4. **`--aug_apm_p 0.3 --aug_apm_ns_levels 18 21` (one coupled group: strong suppression only).**
   - Hypothesis: the only NS-specific signal in B4a is the afftdn-vs-anlmdn contrast, DiD −1.39 [−2.14, −0.69] (final).
     - Half-B afftdn bona rose 58.1 → 62.2, while anlmdn moved 76.4 → 77.8.
     - afftdn is the harsher heldout suppressor (B2f final recall 62.6 against anlmdn's 75.9).
     - Training only on WebRTC kHigh / kVeryHigh (18 / 21 dB) may match its artefacts better than the uniform 6-21 dB draw.
   - Predicted signature: NS DiD ≤ −0.22 on the final readout; afftdn-vs-anlmdn DiD ≤ −1.70 (the "strong" line); heldout F1 Δ ≥ +0.5 on both readouts; offline-NS spoof misses ≤ 29.
   - Falsifier: NS DiD ≥ +0.32 again → WebRTC NS exposure does not transfer to ffmpeg suppressors at any level; drop the lever.
   - It ranks below the dose because falsifier (1) already fired for the default levels.
5. **Seed repeat of the reference: `--seed 1235` on B2f (last).**
   - Purpose: measure B2f's own half-B seed band, so that B4a's +0.57 / +1.02 can be read against same-configuration noise instead of the 9-run mixed panel (heldout F1 SD 1.35 / 1.39).
   - **This is outside wave B4's search space.** `wave.json` pins `seed` 1234, `--seed` is not in `allowed`, and the owner declined a B2f replicate arm at the handoff (`plan_snapshot.md`). It needs an owner decision.
   - Predicted signature: |Δ| ≤ 0.5 vs B2f on both readouts.
   - Falsifier: a repeat that differs from B2f by ≥ +0.57 (merge) or ≥ +1.02 (final) puts B4a's gains inside seed noise.

**Recommended full run if the wave stopped now.**
- Per `verdict.json`, the reference is still B2f, and the plan leaves the wave's best to the owner at wrapup.
- The strongest one-factor result so far is B4a's final readout: positive, Δ +1.02 [+0.43, +1.60], clean +0.35. Its mechanism is refuted and its merge CI includes 0.
- If the owner promotes it:
  - Flags: B4a's exact `config.yaml` command (`scripts/run_B4a.sh`) at 2-3 seeds (1234, 1235, 1236).
  - Readout to submit: **the final-epoch EMA checkpoint**. It beat the merge on half B (93.15 vs 92.55) and on half A (93.32 vs 92.78), and the curve was still rising at epoch 8.
  - Keep `--keep_topk_by_wf1 7 --keep_epochs 8`.
- Pre-launch checks:
  - `bash -n scripts/run_B4a.sh`;
  - `which ffmpeg` (a container restart wipes it and the codec stage silently no-ops);
  - `pyloudnorm` and `pywebrtc_audio` importable in the rtc-sdd env;
  - an empty `pgrep -af "b4wave.train|b3wave.train|main_train"`;
  - an idle GPU.
- Post-launch checks:
  - `B4 APM NS selftest: PASS …`;
  - `B4 APM NS: on p=0.3 levels_db=[6, 12, 18, 21] backend=pywebrtc_audio.NoiseSuppressor hpf=off agc=off`;
  - `Warmup-cosine: 1895 warmup / 37896 steps`;
  - epoch 1 TrainAcc ≈ 86.88 and LocalWF1 ≈ 87.71;
  - `wrtc_ns*` counts summing to ≈ 0.30 × 75,785.
- Expect about 4 h 13 min per seed. Every evaluation needs `--loudness_norm --loudness_norm_lufs -23`.
