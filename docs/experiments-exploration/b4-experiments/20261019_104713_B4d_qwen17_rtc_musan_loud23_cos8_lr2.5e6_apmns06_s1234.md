# Experiment exploration B4-04: B4d (`B4d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns06_s1234`) - apm_ns dose step (WebRTC NS on 60 % of clips) on top of B2f

> Written 2026-10-19 by the exploop B4 analysis agent (ANALYZE step, wave B4, iteration 4, attempt a0).
>
> **Sources.** Every number below was either copied from a file read in this session or computed in this session from the half_B rows (and, where marked "full set", all rows) of the local `scores/*.txt` / `*.logits` files. Nothing is quoted from memory. The files:
> - run config and logs: `config.yaml` and `git.diff` (B4d), `config.yaml` (B2f), `logs/train.log` and `metrics.jsonl` (B4d, B2f, B4a), `top5_by_wf1/{merge_list.txt,select_topk.log}`, `local_eval/merge_top5/merge_list.txt`, the `ckpt/` listing;
> - local evals: `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,compare_vs_B2f-*.txt,ci_vs_B2f-*.txt,compare_ci_vs_B2f-*.json,compare_ci.json}` and `local_eval/merge_top5/compare_all_readouts.txt` (B4d); the `metrics.json`, `probe_report.json` and `scores/` of the B2f, B4a, B4b and B4c readouts;
> - stdout log, header lines only: `SLS_setup/B4d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns06_s1234_20261019055117.log`;
> - loop files: `exploop/B4/iter04/{verdict.json,design.result.json,log_merge.txt,log_final.txt}`, `exploop/B4/{wave.json,plan_snapshot.md,STATUS.md}`, `scripts/run_B4d.sh`;
> - the pre-registered mechanism readout `scripts/b4_dsp_slices.py --lever apm_ns`. It was run in this session on `local_eval/epoch_8` (primary) and `local_eval/merge_top5`, with the default `--ref` (B2f, same readout), the pinned null panel, and `--placebo` = B4b's and B4c's same readout (2 placebo arms; B4a is read as the half-dose point, as the design asks). There was no `--json_out`; stdout was read directly and no scratch file was written;
> - registry: `docs/experiment_log.csv` (rows `B2f-loudnorm-*`, `B4a-apmns-*`, `B4b-opusplc-*`, `B4c-dtxgate-*`, `B4d-apmns06-*`);
> - data: `data/dev_noisy_sim/*/meta.csv` (half, lang, src, codec, family, `dsp_chain`), `data/dev_noisy_sim/dev_online_clean/protocol.txt`, `configs/local_eval_sets.yaml`, `SLS_setup/utils/metrics.py` (macro-F1, spoof if p ≥ 0.5).
>
> Progress-split scores (`model_merging_top5/scores.txt`, `progress.txt`) were not read.
>
> **Previous reports.**
> - `docs/experiments-exploration/b4-experiments/20261018_194212_B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234.md` (B4a, iteration 1, p = 0.3, the half-dose point of this run). Per the user note of 2026-10-18 21:38, B4a's registry decisions are merge = revisit and final = promote; the "drop" labels inside that report are an artefact of the old screen setting. B4a's mechanism numbers quoted below (NS DiD +0.49, afftdn-vs-anlmdn −1.39 / −1.14, heldout Δ +0.30 / +1.04) come from that report.
> - `docs/experiments-exploration/b4-experiments/20261019_001203_B4b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_opusplc03_s1234.md` (B4b, iteration 2).
> - `docs/experiments-exploration/b4-experiments/20261019_052003_B4c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_dtx03_s1234.md` (B4c, iteration 3; its rank-1 proposal is this run).
> - The structure follows `docs/experiments-exploration/b3-experiments/20261010_143058_B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234.md`.
>
> **Readouts.**
> - `-top5` is the weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5/`). It is the wave objective.
> - `-ep8` is the final-epoch EMA checkpoint alone (`local_eval/epoch_8/`).
> - half_B is the half of each sim set that is never used for selection: n = 1,994 per sim set (1,603 spoof, 391 bona). Clean is the full 7,465-clip `dev_online_clean` (6,068 spoof, 1,397 bona).
> - WF1 = 0.3 clean + 0.35 matched + 0.35 heldout, using macro-F1 at p_spoof 0.5. Echo is reported but is not in WF1.
> - Threshold 0.5 on the half_B score rows reproduces all ten half_B WF1 values exactly (computed): B2f 91.98 / 92.14, B4a 92.55 / 93.15, B4b 92.22 / 92.26, B4c 92.01 / 92.34, B4d 92.14 / 92.34.
> - All readouts were evaluated with `"loudness_norm": -23.0` (`metrics.json`).

Run directories (under `SLS_setup/exp/`):
- **B4d** `B4d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns06_s1234_epoch8_bs16_20261019055124` (this report)
- **B2f** `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623`: the reference for the whole wave (`wave.json` `rules.keep_best` true = screen design). Half_B merge 91.98, final 92.14.
- **B4a** `B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234_epoch8_bs16_20261018150134` (p = 0.3, the half-dose point), **B4b** `…opusplc03…_20261018201137` and **B4c** `…dtx03…_20261019004222` (equal-dose placebo arms).

## 1. Setup

**B4d = B2f with one factor changed: `--aug_apm_p 0.6`.** Each training clip, of either class, has a 60 % chance of one pass through the standalone WebRTC NoiseSuppressor (pywebrtc_audio, NS only), after the acoustic stage and before the codec, from the separate B4 RNG stream. The level is drawn from the default 6 / 12 / 18 / 21 dB (`--aug_apm_ns_levels` unchanged). It is the dose step of B4a (p = 0.3). Sources: `scripts/run_B4d.sh`, `design.result.json`.

Flag diff (`args` of both `config.yaml`, parsed and compared in this session):

| flag / field | B2f | **B4d** | note |
|---|---|---|---|
| trainer | `main_train.py`, git `00da485`, dirty | **`b4wave/train.py`** (b4wave 0.1.0 on b3wave 0.1.0), git `8ea996c`, dirty | `git.diff` touches only `.claude/skills/post-train-eval/SKILL.md` and `docs/experiment_log.csv`. `module_sha256` `b6498bc4a5af…`, the same prefix as the `b4wave sha256` header line (same as B4a-B4c) |
| `--aug_apm_p` | (absent = 0) | **0.6** | the lever (B4a: 0.3) |
| `--aug_apm_ns_levels` | - | `[6, 12, 18, 21]` (default) | coupled with `--aug_apm_p` (`wave.json` `coupled`) |
| other B4 flags | - | `aug_dtx_p 0.0`, `aug_opus_plc_p 0.0` (off); `aug_dtx_fill mix`, `aug_opus_loss [0.01, 0.2]`, `aug_opus_burst [1.0, 4.0]` unread | |
| B3 flags (written by the b3wave parser) | - | defaults: `sampler shuffle`, `aug_snr_db [5.0, 20.0]` uniform, `aug_codec_profiles fixed`, `consistency none`, `llrd 1.0`, `freeze_bottom_k 0`, `specaug False`, `train_subset_frac 1.0` | same behaviour as B2f |
| everything else | identical | identical | no other `args` key differs except `track` |

**Header lines proving the change took effect** (stdout log):
- `B4 APM NS: on p=0.6 levels_db=[6, 12, 18, 21] backend=pywebrtc_audio.NoiseSuppressor hpf=off agc=off`. This matches the design's first `expect` regex.
- `B4 APM NS selftest: PASS shape_finite ok | no_gain dB 6:-6.02 12:-12.38 18:-18.40 21:-21.47 | monotone removed dB 6:5.99 12:11.89 18:17.53 21:20.83 | no_hpf diff dB 6:-0.09 12:-0.49 18:-0.83 21:-0.92 | aligned lag 6:0 12:0 18:0 21:0`.
- The `RTC augmentation:` line ends with `| wrtc_ns p=0.6 levels_db=[6, 12, 18, 21]`.
- `b3wave: 0.1.0`, `b4wave: 0.1.0`, `b4wave sha256: b6498bc4a5af`.
- Unchanged lines: `Sampler: shuffle`; `SNR: 5..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: fixed`; `Consistency: none`; `LLRD: 1`; `SpecAugment: off`; `Loudness norm: loudness_norm target=-23 LUFS`; `CE weights: spoof=0.1 bonafide=0.9`; `Parameters: 320268933`; `Train trials: 75785`; `Warmup-cosine: 1895 warmup / 37896 steps, floor 5.0e-07`.
- The stdout log has no `Traceback`, `out of memory`, `failed (`, `[b4wave] ` or `fail_wrtc` line (grep count 0).

**Realised dose** (`metrics.jsonl` `b3.aug_counts` and `b4.aug_by_class`, i.e. the `Epoch N B3 aug` / `B4 aug-by-class` lines of `train.log`; shares computed in this session):

| run | `wrtc_ns6+12+18+21` per epoch (share of 75,785) | per level | bona share (n = 14,353) | spoof share (n = 61,432) | fail key |
|---|---|---|---|---|---|
| B4a (p 0.3) | 22,599-22,899 (0.2982-0.3022) | 5,507-5,856 | - | - | none |
| **B4d (p 0.6)** | **45,251-45,521 (0.5971-0.6007)** | 11,148-11,541 | 8,552-8,678 (0.5958-0.6046) | 36,628-36,891 (0.5962-0.6005) | none |

The design predicted about 45,471 clips per epoch (about 11,368 per level), bona about 8,612 and spoof about 36,859, with no `fail_wrtc_ns`. The run was dosed exactly as designed.

**Run time.**
- `config.yaml` was created at 05:51:33. The epoch-8 local eval finished at 10:07:57 (`train.log`), i.e. **4 h 16 min**. The registry `gpu_h` is 4.27, against B2f's 4.14, B4a's 4.21, B4b's 3.55 and B4c's 4.20.
- Epochs 2-8 took 31:31-31:41 min each, against B4a's 31:01-31:12 and B2f's 30:24-30:35 (eval-end to epoch-end timestamps in the three `train.log`s; computed). This is inside the design's "at most about 1 min longer than B4a's 31 min".
- Post-train file times: `top5_by_wf1/merge_list.txt` 10:09, `merge_top5/metrics.json` 10:29, `epoch_8/metrics.json` 10:35. `verdict.json` was written at 10:47:13.

**Checkpoints.**
- Epochs 2-8 are kept at 1,281,235,691 bytes each. `train.log`: `Pruned epoch_1_dev_loss_0.110319.pth (WF1=86.63, outside top-7)`.
- The symlinks are `best_by_dev_loss` → epoch 2 (0.085579), `best_by_noisy_eer` → epoch 7, `best_by_wf1` → epoch 5.

**Merged top-5 by half-A mini WF1: epochs 5, 8, 6, 4, 7** (92.33, 92.31, 92.12, 92.11, 91.97; mean 92.17; `select_topk.log`).
- Excluded: 3 (90.69), 2 (90.39) and 1 (86.63, deleted).
- B2f's merge was epochs 8, 5, 7, 3, 4 (registry row `B2f-loudnorm-screen`).

## 2. Results vs references (half_B)

Sources: `compare_all_readouts.txt`, `compare_vs_B2f-*.txt`, `ci_vs_B2f-*.txt` / `compare_ci_vs_B2f-*.json` (paired bootstrap, n = 2,000, half B, threshold 0.5), and the `half_B` block of every `metrics.json`. The CIs of B4a-B4c come from their registry rows.

### 2.1 Wave iterations and fixed comparators, both readouts

| half_B | B2f-top5 | B2f-ep8 | B4a-top5 | B4a-ep8 | B4b-top5 | B4b-ep8 | B4c-top5 | B4c-ep8 | **B4d-top5** | **B4d-ep8** | B4d-top5 − B2f-top5 [95 % CI] | B4d-ep8 − B2f-ep8 [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **WF1@0.5** | 91.98 | 92.14 | 92.55 | 93.15 | 92.22 | 92.26 | 92.01 | 92.34 | **92.14** | **92.34** | **+0.17 [−0.39, +0.74]**, p(not better) 0.283 | **+0.20 [−0.42, +0.84]**, p 0.247 |
| (CI vs B2f, registry) | | | [−0.01, +1.17] | [+0.43, +1.60] | [−0.30, +0.78] | [−0.48, +0.71] | [−0.47, +0.51] | [−0.39, +0.78] | | | | |
| WF1@oracle | 92.59 | 92.82 | 92.98 | 93.26 | 92.47 | 92.50 | 92.85 | 93.10 | 92.81 | 92.78 | +0.22 | **−0.04** |
| clean F1@0.5 | 97.60 | 97.69 | 97.72 | 98.05 | 97.70 | 97.71 | 97.68 | 98.02 | 97.47 | 97.28 | −0.14 [−0.47, +0.18] | **−0.42 [−0.74, −0.11]**, p(not better) 0.9945 |
| noisy F1@0.5 (mean matched, heldout) | 89.57 | 89.76 | 90.34 | 91.06 | 89.88 | 89.92 | 89.58 | 89.90 | 89.86 | 90.22 | +0.29 | +0.47 |
| mean noisy EER | 10.09 | 9.66 | 9.53 | 9.16 | 10.66 | 9.68 | 10.09 | 10.04 | 10.10 | 10.17 | +0.02 | **+0.51** (worse) |
| calibration gap noisy | 0.66 | 0.71 | 0.49 | 0.05 | 0.18 | 0.21 | 0.94 | 0.97 | 0.75 | 0.33 | | |
| separability gap | 7.87 | 7.86 | 7.18 | 7.17 | 8.04 | 7.90 | 7.76 | 7.41 | 7.31 | 7.44 | | |

Per-set paired CIs (`ci_vs_B2f-*.txt`):

| set | merge Δ [CI], p(not better) | ep8 Δ [CI], p(not better) |
|---|---|---|
| clean | −0.14 [−0.47, +0.18], 0.803 | **−0.42 [−0.74, −0.11], 0.9945** |
| matched | +0.40 [−0.69, +1.57], 0.2415 | −0.17 [−1.32, +0.99], 0.6 |
| heldout | +0.19 [−0.92, +1.29], 0.381 | **+1.10 [−0.24, +2.57], 0.0515** |
| echo | −0.29 [−1.27, +0.60], 0.7595 | −1.00 [−2.09, +0.03], 0.9735 |

Weighted contributions (0.3 / 0.35 / 0.35 × the unrounded deltas of `compare_ci_vs_B2f-*.json`; computed):
- merge: −0.04 clean + 0.14 matched + 0.07 heldout = +0.17;
- final: −0.12 clean − 0.06 matched + 0.39 heldout = +0.20.

Reading:
- **Neither readout moved WF1@0.5 outside the ±0.5 band.**
- **On the final, the heldout gain is as large as B4a's** (+1.10 vs B4a +1.04), **but clean and matched took back what B4a had gained**: clean −0.42 (B4a +0.35), matched −0.17 (B4a +1.56). The clean loss is the only per-set CI of B4d that excludes 0.
- **The final's ranking did not improve**: WF1@oracle −0.04, noisy EER +0.51. Its better WF1@0.5 is an operating-point shift (calibration gap 0.71 → 0.33), not better separation (section 5.4).
- **Full sets** (registry rows, `log_merge.txt` / `log_final.txt`):
  - merge WF1 92.52 vs B2f 92.25 (+0.27): clean −0.13, matched +0.78, heldout +0.11, echo −0.50;
  - final 92.48 vs 92.53 (−0.05): clean −0.41, matched −0.30, heldout +0.50, echo −0.90.

### 2.2 Per set (R = recall at 0.5; thr = p_spoof threshold; `metrics.json` half_B)

| set | metric | B2f-top5 | B4a-top5 | **B4d-top5** | B2f-ep8 | B4a-ep8 | **B4d-ep8** |
|---|---|---|---|---|---|---|---|
| dev_online_clean | F1@.5 / F1@or / thr_or | 97.60 / 98.10 / 0.997 | 97.72 / 98.00 / 0.996 | 97.47 / 97.92 / 0.990 | 97.69 / 98.33 / 0.999 | 98.05 / 98.27 / 0.999 | **97.28** / 97.99 / 0.998 |
| | EER / ECE | 3.42 / 1.42 | 2.11 / 1.32 | **2.37** / 1.46 | 3.33 / 1.33 | 2.21 / 1.15 | **2.01** / 1.59 |
| | R_spoof / R_bona | 99.90 / 92.84 | 99.85 / 93.41 | 99.85 / 92.63 | 99.95 / 92.91 | 99.88 / 94.27 | 99.88 / **91.91** |
| sim_matched_v1 | F1@.5 / F1@or / thr_or | 91.22 / 91.53 / 0.846 | 92.46 / 92.74 / 0.765 | 91.62 / 92.09 / 0.980 | 91.24 / 92.05 / 0.991 | 92.80 / 92.80 / 0.534 | 91.07 / 91.55 / 0.995 |
| | EER / ECE | 8.15 / 4.75 | 7.42 / 4.24 | 7.96 / 4.68 | 8.21 / 4.86 | 7.54 / 4.19 | 8.43 / 5.09 |
| | R_spoof / R_bona | 99.50 / 76.21 | 99.50 / 79.54 | 99.13 / 78.52 | 99.44 / 76.47 | 99.25 / 81.33 | **98.75** / 78.26 |
| sim_heldout_v1 | F1@.5 / F1@or / thr_or | 87.92 / 88.94 / 0.957 | 88.21 / 88.91 / 0.858 | 88.10 / 89.15 / 0.970 | 88.27 / 88.89 / 0.949 | 89.31 / 89.41 / 0.743 | **89.38** / 89.55 / **0.323** |
| | EER / ECE | 12.03 / 6.46 | 11.65 / 6.45 | 12.25 / 6.39 | 11.11 / 6.38 | 10.77 / 5.95 | 11.91 / 5.94 |
| | R_spoof / R_bona | 99.00 / 69.31 | 98.57 / 71.36 | 98.81 / 70.33 | 99.13 / 69.82 | 98.81 / 73.40 | **98.38** / **74.94** |
| sim_echo_v1 | F1@.5 / F1@or / thr_or | 94.16 / 94.91 / 0.983 | 94.79 / 95.22 / 0.977 | 93.87 / 95.28 / 0.989 | 94.53 / 95.41 / 0.983 | 94.63 / 94.83 / 0.985 | **93.53** / 94.79 / 0.990 |
| | EER / ECE | 4.71 / 3.37 | 4.86 / 2.98 | 5.59 / 3.45 | 4.86 / 3.20 | 4.61 / 3.16 | 4.96 / 3.69 |
| | R_spoof / R_bona | 99.75 / 83.38 | 99.75 / 85.17 | 99.81 / 82.35 | 99.69 / 84.65 | 99.63 / 85.17 | 99.69 / **81.84** |

Reading:
- **Heldout bona recall rose most on the final** (+5.12, 69.82 → 74.94; B4a +3.58). The heldout oracle threshold dropped to p 0.323: at p = 0.5 the final already sits on the bona-ward side of heldout's best cut.
- **Spoof recall fell on matched and heldout on the final** (−0.69 and −0.75; `compare_vs_B2f-ep8.txt`). That breaks the design's must-not-move item "R_spoof within 0.5 on every set" twice (section 6).
- **Clean and echo bona recall fell on the final**: clean −1.00 (92.91 → 91.91), echo −2.81 (84.65 → 81.84). B4a had raised both (94.27 / 85.17).
- **Clean EER fell on both readouts** (3.42 → 2.37, 3.33 → 2.01), as it did for B4a (2.11 / 2.21) and not for the placebos B4b (3.23 / 3.10) and B4c (3.36 / 3.14). Clean ranking improves under NS at both doses, but clean F1@0.5 does not follow, because the clean bona tail at p ≥ 0.99 grew (section 5.5).

Paired per clip (half_B score files, B4d vs B2f at the same readout; computed):

| set | class | merge fixed / broken / wrong in both | ep8 fixed / broken / wrong in both |
|---|---|---|---|
| clean | bona | 19 / 22 / 81 | **16 / 30** / 83 |
| matched | bona | 19 / 10 / 74 | 19 / 12 / 73 |
| heldout | bona | 14 / 10 / 106 | **28 / 8** / 90 |
| echo | bona | 9 / 13 / 56 | **10 / 21** / 50 |
| clean | spoof | 4 / 7 / 2 | 2 / 6 / 1 |
| matched | spoof | 3 / 9 / 5 | **1 / 12** / 8 |
| heldout | spoof | 5 / 8 / 11 | **6 / 18** / 8 |
| echo | spoof | 3 / 2 / 1 | 3 / 3 / 2 |

Half-B spoof misses (p < 0.5; computed):

| set (n spoof) | B2f-top5 | B4a-top5 | B4b-top5 | B4c-top5 | **B4d-top5** | B2f-ep8 | B4a-ep8 | B4b-ep8 | B4c-ep8 | **B4d-ep8** |
|---|---|---|---|---|---|---|---|---|---|---|
| clean (6,068) | 6 | 9 | 5 | 10 | 9 | 3 | 7 | 2 | 8 | 7 |
| matched (1,603) | 8 | 8 | 8 | 19 | 14 | 9 | 12 | 14 | 20 | **20** |
| heldout (1,603) | 16 | 23 | 11 | 12 | 19 | 14 | 19 | 15 | 13 | **26** |
| echo (1,603) | 4 | 4 | 3 | 6 | 3 | 5 | 6 | 2 | 11 | 5 |
| noisy = matched + heldout | 24 | 31 | 19 | 31 | 33 | 23 | 31 | 29 | 33 | **46** |

Where the new spoof misses sit (half B, computed; B2f-top5 / B4d-top5 / B2f-ep8 / B4d-ep8):
- heldout language: zh 13 / 14 / 9 / **21**; en 3 / 5 / 5 / 5;
- heldout source: offline 11 / 10 / 10 / 15; online 5 / 9 / 4 / 11;
- heldout codec: **gsm 10 / 10 / 11 / 18**; g726 3 / 5 / 1 / 5; g722 3 / 4 / 2 / 3;
- heldout spoof with an `ns` stage (530 of 1,603): 9 / 6 / 9 / 10, so heldout spoof without one: 7 / 13 / 5 / 16. The new final misses are mostly in clips **without** an NS stage;
- matched: zh 8 / 10 / 9 / 15, en 0 / 4 / 0 / 5; coffee 0 / 5 / 1 / 5, rain 2 / 3 / 3 / 6, office 4 / 3 / 4 / 6; opus:wechat 3 / 5 / 3 / 8, opus:zoom 0 / 3 / 0 / 3.

### 2.3 Probes (`probe_report.json`; 187 bona + 813 spoof; one bona probe clip = 0.53 points)

| probe (bona unless stated) | B2f-top5 | B4a-top5 | **B4d-top5** | B2f-ep8 | B4a-ep8 | **B4d-ep8** |
|---|---|---|---|---|---|---|
| level_flip_bona (mean) / max | 1.54 / 4.81 | 0.87 / 2.14 | **0.80** / 2.67 | 1.74 / 4.81 | 1.40 / 3.74 | 1.80 / **6.95** |
| gain_up / gain_p6 / gain_p10_limit flip % | 4.28 / 2.14 / 4.81 | 2.14 / 1.60 / 1.60 | 1.07 / 1.60 / 2.67 | 4.81 / 2.67 / 3.21 | 2.67 / 2.14 / 3.74 | 1.60 / 2.14 / **6.95** |
| agc_dynaudnorm / loudnorm flip % | 0.53 / 0.53 | 1.07 / 0.53 | 0.00 / 1.07 | 2.14 / 1.07 | 0.53 / 1.07 | 1.07 / 2.14 |
| gain_down / gain_m20 / gain_m6 flip % | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.53 / 0.53 | 0.00 / 0.53 / 0.00 |
| level_flip_spoof | 0.03 | 0.05 | 0.05 | 0.03 | 0.03 | 0.03 |
| probe_orig bona pred spoof % (`pred_spoof_orig`) | 10.70 | 10.70 | **12.83** | 10.16 | 8.02 | **12.30** |
| **noise_only bona pred spoof % / mean dP** | 100.00 / +0.8870 | 100.00 / +0.8921 | **99.47** / +0.8628 | 100.00 / +0.8965 | 100.00 / +0.9202 | **98.93** / +0.8615 |
| **sil_only bona pred spoof % / mean dP** | 100.00 / +0.8929 | 100.00 / +0.8911 | 100.00 / +0.8660 | 100.00 / +0.8985 | 100.00 / +0.9203 | 100.00 / +0.8685 |
| trimmed / padded bona flip % | 3.74 / 3.74 | 2.67 / 2.67 | 3.74 / 5.35 | 3.74 / 5.35 | 2.14 / 2.14 | 4.81 / **6.95** |
| noise_only spoof pred spoof % | 100.00 | 100.00 | 99.88 | 100.00 | 100.00 | 99.75 |

Reading:
- **First movement of the non-speech probes in the wave.** noise_only bona fell from 100 % spoof to 99.47 / 98.93 % (1 / 2 clips of 187), and the mean dP of both non-speech probes fell by about 0.03 (noise_only +0.8965 → +0.8615, sil_only +0.8985 → +0.8685 on the final). The core failure is unchanged: sil_only bona is still 100 % spoof.
- **More clean probe bona predicted spoof**: probe_orig 10.70 → 12.83 % (merge) and 10.16 → 12.30 % (final), 4 more clips of 187 each. B4a had lowered it on the final (8.02 %). This matches the clean bona recall loss in 2.2.
- **Level shortcut**: lower on the merge (1.54 → 0.80), unchanged on the final (1.74 → 1.80), where the limiter probe (gain_p10_limit) flips 6.95 % of bona (B2f 3.21).

### 2.4 Slices that moved most, B4d vs B2f (full sim sets, `metrics.json → breakdowns`, 63 slice rows; computed)

| readout | F1 up | EER down | R_bona up | R_spoof up | R_spoof down |
|---|---|---|---|---|---|
| merge | 34 of 63 | 27 | 37 | 17 | 35 |
| final | 26 of 63 | 25 | 34 | **9** | **43** |

Minimum slice R_spoof: B4d-top5 97.87 (heldout / glass_breaking), **B4d-ep8 96.99 (heldout / snr 0)**; B2f-top5 97.31 (heldout / door_wood_creaks), B2f-ep8 97.68 (heldout / washing_machine).

| set / slice | n | F1 merge B2f → B4d | F1 ep8 B2f → B4d | B4a merge / ep8 | EER ep8 | R_bona merge / ep8 | R_spoof merge / ep8 |
|---|---|---|---|---|---|---|---|
| heldout / vacuum_cleaner | 373 | 83.27 → 86.68 **+3.41** | 82.01 → 86.96 **+4.95** | 86.68 / 87.43 | 18.36 → 14.95 | 60.0 → 67.1 / 57.1 → 72.9 | 98.35 → 98.68 / 98.35 → 97.03 |
| matched / keyboard | 647 | 91.49 → 93.83 +2.35 | 91.84 → 93.21 +1.38 | 93.27 / 92.32 | 7.74 → 7.93 | 76.7 → 82.5 / 77.7 → 81.6 | 99.45 → 99.63 / 99.45 → 99.45 |
| heldout / mouse_click | 369 | 88.80 → 89.92 +1.12 | 88.80 → 90.89 +2.10 | 87.66 / 88.16 | 9.25 → 12.13 | 71.8 → 75.6 / 71.8 → 78.2 | 98.97 → 98.63 / 98.97 → 98.63 |
| matched / opus:telegram | 552 | 90.42 → 92.12 +1.70 | 91.25 → 92.24 +0.99 | 92.84 / 93.89 | 6.78 → 8.81 | 74.8 → 79.3 / 78.4 → 81.1 | 99.32 → 99.32 / 98.87 → 98.87 |
| heldout / lang zh | 2462 | 90.86 → 91.74 +0.88 | 91.15 → 92.40 +1.25 | 91.68 / 93.00 | 8.68 → 8.48 | 78.3 → 80.5 / 78.0 → 84.1 | 98.58 → 98.63 / 98.88 → **98.07** |
| **echo / lang en** | 1538 | 88.96 → 86.69 **−2.27** | 89.33 → 86.37 **−2.96** | 88.97 / 88.71 | 10.54 → 9.69 | **70.9 → 65.5 / 72.5 → 64.7** | 99.22 → 99.14 / 99.06 → 99.14 |
| heldout / glass_breaking | 346 | 94.49 → 91.60 −2.90 | 92.94 → 91.83 −1.12 | 93.21 / 94.10 | 6.32 → 5.96 | 85.9 → 82.8 / 82.8 → 79.7 | 99.29 → 97.87 / 98.94 → 98.94 |
| heldout / wind | 381 | 87.17 → 84.23 −2.94 | 88.36 → 87.75 −0.62 | 83.10 / 86.26 | 13.14 → 14.61 | 68.4 → 63.2 / 72.4 → 73.7 | 98.69 → 98.03 / 98.36 → 97.38 |
| heldout / lang en | 1538 | 84.45 → 82.90 −1.55 | 84.47 → 83.14 −1.33 | 81.44 / 82.66 | 16.34 → 16.69 | 59.7 → 57.4 / 60.5 → 58.1 | 99.30 → 98.91 / 99.06 → 98.83 |
| echo / opus:wechat | 536 | 93.49 → 92.06 −1.44 | 93.55 → 92.06 −1.49 | 93.90 / 93.55 | 7.23 → 8.67 | 81.4 → 77.5 / 82.4 → 77.5 | 99.77 → 99.77 / 99.54 → 99.77 |
| matched / opus:wechat | 585 | 86.62 → 86.48 −0.15 | 88.94 → 86.46 **−2.48** | 88.39 / 88.42 | 13.21 → 13.90 | 65.4 → 66.3 / 71.2 → 68.3 | 99.17 → 98.75 / 99.17 → 98.13 |

Half-B splits of heldout (computed from `scores/*.txt` + `meta.csv`; NS = `dsp_chain` has an `ns` stage, `afftdn` or `anlmdn`; PLC / VAD likewise):

| half-B heldout slice | n bona / spoof | R_bona merge B2f / B4a / **B4d** | R_bona ep8 B2f / B4a / **B4d** | spoof misses merge B2f / B4a / **B4d** | spoof misses ep8 B2f / B4a / **B4d** |
|---|---|---|---|---|---|
| all | 391 / 1603 | 69.3 / 71.4 / 70.3 | 69.8 / 73.4 / **74.9** | 16 / 23 / 19 | 14 / 19 / **26** |
| online | 190 / 807 | 69.5 / 70.5 / 68.4 | 70.5 / 73.7 / 73.2 | 5 / 11 / 9 | 4 / 9 / 11 |
| offline | 201 / 796 | 69.2 / 72.1 / 72.1 | 69.2 / 73.1 / 76.6 | 11 / 12 / 10 | 10 / 10 / 15 |
| **offline, NS** | 146 / 530 | 66.4 / 68.5 / 69.2 | 67.1 / 69.9 / **74.7** | 9 / 10 / 6 | 9 / 8 / 10 |
| offline, NS afftdn | 74 / 272 | 55.4 / 59.5 / 59.5 | 58.1 / 62.2 / **66.2** | 4 / 5 / 1 | 4 / 4 / 5 |
| offline, NS anlmdn | 72 / 258 | 77.8 / 77.8 / 79.2 | 76.4 / 77.8 / 83.3 | 5 / 5 / 5 | 5 / 4 / 5 |
| offline, no NS | 55 / 266 | 76.4 / 81.8 / 80.0 | 74.5 / 81.8 / 81.8 | 2 / 2 / 4 | 1 / 2 / 5 |
| online, PLC | 66 / 253 | 65.2 / 72.7 / 65.2 | 68.2 / 77.3 / 69.7 | 3 / 5 / 2 | 3 / 4 / 3 |
| offline, VAD | 73 / 240 | 63.0 / 71.2 / 69.9 | 65.8 / 71.2 / 72.6 | 7 / 7 / 8 | 7 / 6 / 10 |
| en | 129 / 631 | 59.7 / 56.6 / 58.1 | 58.9 / 58.9 / 60.5 | 3 / 10 / 5 | 5 / 9 / 5 |
| zh | 262 / 972 | 74.0 / 78.6 / 76.3 | 75.2 / 80.5 / 82.1 | 13 / 13 / 14 | 9 / 10 / **21** |
| codec gsm | 127 / 549 | 63.0 / 68.5 / 65.4 | 63.8 / 71.7 / 70.1 | 10 / 15 / 10 | 11 / 12 / **18** |
| codec g726 | 129 / 518 | 64.3 / 62.8 / 62.8 | 65.9 / 65.1 / 71.3 | 3 / 2 / 5 | 1 / 1 / 5 |
| codec g722 | 135 / 536 | 80.0 / 82.2 / 82.2 | 79.3 / 83.0 / 83.0 | 3 / 6 / 4 | 2 / 6 / 3 |

- **On the final, the NS-treated heldout bona rose monotonically with dose**: offline NS 67.1 → 69.9 → 74.7, afftdn 58.1 → 62.2 → 66.2 (p 0 → 0.3 → 0.6). On the merge the dose step added little (68.5 → 69.2; afftdn 59.5 → 59.5).
- English heldout bona (the design's watch item, B4a 56.6 / 58.9) is 58.1 / 60.5: not worse.

### 2.5 Lever response (one factor per row)

Values from `metrics.json` (deltas computed from unrounded values) and `verdict.json` / registry; mechanism verdicts of B4a-B4c from their reports (via the B4c report §2.5).

| run | change vs B2f | half_B merge WF1 (Δ) | final-epoch WF1 (Δ) | merge oracle (Δ) | final oracle (Δ) | merge noisy EER (Δ) | final EER (Δ) | merge clean (Δ) | final clean (Δ) | verdict (`verdict.json`) | mechanism (`b4_dsp_slices.py`, provisional) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B2f | reference (start of wave) | 91.98 | 92.14 | 92.59 | 92.82 | 10.09 | 9.66 | 97.60 | 97.69 | - | - |
| B4a | + `--aug_apm_p 0.3` | 92.55 (+0.57) | 93.15 (+1.02) | 92.98 (+0.39) | 93.26 (+0.43) | 9.53 (−0.55) | 9.16 (−0.51) | 97.72 (+0.12) | 98.05 (+0.35) | merge: above band, CI includes 0 (revisit); final: positive (promote) | NS: REFUTED final / DIRECTIONAL merge [placebos 0] |
| B4b | + `--aug_opus_plc_p 0.3` | 92.22 (+0.24) | 92.26 (+0.12) | 92.47 (−0.13) | 92.50 (−0.32) | 10.66 (+0.57) | 9.68 (+0.02) | 97.70 (+0.09) | 97.71 (+0.02) | null (inside +-0.5) / revisit, both | PLC_online: REFUTED both; ANOMALOUS on VAD |
| B4c | + `--aug_dtx_p 0.3` | 92.01 (+0.03) | 92.34 (+0.20) | 92.85 (+0.25) | 93.10 (+0.27) | 10.09 (−0.00) | 10.04 (+0.38) | 97.68 (+0.08) | 98.02 (+0.33) | null (inside +-0.5) / revisit, both | dtx_gate: NOT SHOWN, under-dosed |
| **B4d** | **+ `--aug_apm_p 0.6`** | **92.14 (+0.17)** | **92.34 (+0.20)** | 92.81 (+0.22) | 92.78 (−0.04) | 10.10 (+0.02) | 10.17 (+0.51) | 97.47 (−0.14) | 97.28 (−0.42) | merge: **null (inside +-0.5)** / revisit; final: **null (inside +-0.5)** / revisit; new_best false | apm_ns: **INCONCLUSIVE final / REFUTED merge** [placebos 2]; afftdn-vs-anlmdn DIRECTIONAL final / REFUTED merge; no run anomaly |

### 2.6 Dose response p = 0 / 0.3 / 0.6 (B2f / B4a / B4d), as the design's eval note asks

Sources: `metrics.json` half_B; FA, spoof misses and the threshold sweep computed from `scores/`; DiD from `b4_dsp_slices.py` (B4d, this session) and the B4a report (B4a).

| half_B | merge p 0 | merge p 0.3 | **merge p 0.6** | final p 0 | final p 0.3 | **final p 0.6** |
|---|---|---|---|---|---|---|
| WF1@0.5 | 91.98 | 92.55 | **92.14** | 92.14 | 93.15 | **92.34** |
| WF1@oracle | 92.59 | 92.98 | 92.81 | 92.82 | 93.26 | 92.78 |
| oracle gap (global-shift part / set-specific part) | 0.61 (0.41 / 0.20) | 0.43 (0.24 / 0.19) | 0.66 (0.55 / 0.11) | 0.69 (0.45 / 0.24) | 0.10 (0.01 / 0.09) | 0.44 (0.12 / 0.33) |
| clean F1 / matched / heldout / echo | 97.60 / 91.22 / 87.92 / 94.16 | 97.72 / 92.46 / 88.21 / 94.79 | 97.47 / 91.62 / 88.10 / 93.87 | 97.69 / 91.24 / 88.27 / 94.53 | 98.05 / 92.80 / 89.31 / 94.63 | 97.28 / 91.07 / **89.38** / 93.53 |
| R_spoof clean / matched / heldout / echo | 99.90 / 99.50 / 99.00 / 99.75 | 99.85 / 99.50 / 98.57 / 99.75 | 99.85 / 99.13 / 98.81 / 99.81 | 99.95 / 99.44 / 99.13 / 99.69 | 99.88 / 99.25 / 98.81 / 99.63 | 99.88 / **98.75 / 98.38** / 99.69 |
| clean EER | 3.42 | 2.11 | 2.37 | 3.33 | 2.21 | 2.01 |
| mean noisy EER | 10.09 | 9.53 | 10.10 | 9.66 | 9.16 | 10.17 |
| half-B noisy bona FA (of 782) / saturated p ≥ 0.99 | 213 / 136 | 192 / 114 | 200 / 130 | 210 / 133 | 177 / 123 | 183 / 135 |
| FA en / zh; saturated en / zh | 95 / 118; 65 / 71 | 95 / 97; 61 / 53 | 95 / 105; 67 / 63 | 99 / 111; 67 / 66 | 91 / 86; 68 / 55 | 95 / 88; 71 / 64 |
| half-B heldout spoof misses / noisy spoof misses | 16 / 24 | 23 / 31 | 19 / 33 | 14 / 23 | 19 / 31 | **26 / 46** |
| afftdn-vs-anlmdn DiD (full set) | - | −1.14 | −0.78 | - | −1.39 | −1.05 |
| heldout F1 Δ vs B2f | - | +0.30 | +0.19 | - | +1.04 | +1.10 |
| ep1 TrainAcc / ep8 TrainLoss | 87.36 / 0.0501 | 86.88 / 0.0562 | 86.13 / 0.0544 | | | |

Reading:
- **No dose response on the outcome.** WF1 rose from p 0 to p 0.3 and fell back from p 0.3 to p 0.6 on both readouts. Doubling the dose kept only the heldout gain of the final (+1.04 → +1.10).
- **The costs grew with dose.** Noisy spoof misses 23 → 31 → 46 (final), heldout R_spoof 99.13 → 98.81 → 98.38, and ep1 TrainAcc 87.36 → 86.88 → 86.13.
- **Two NS-specific signals hold at both doses, and the placebos do not show them.**
  - Clean EER falls: 3.42 / 3.33 → 2.11 / 2.21 (p 0.3) and 2.37 / 2.01 (p 0.6), against B4b 3.23 / 3.10 and B4c 3.36 / 3.14.
  - The final heldout F1 rises by +1.04 / +1.10, against B4b +0.13 and B4c +0.18.
- **The noisy bona false alarms are lowest in the two NS arms** (192 / 177 and 200 / 183, against B4b 210 / 202 and B4c 207 / 199), but p 0.6 removes fewer of them than p 0.3 does.

### 2.7 Mechanism readout (pre-registered; `scripts/b4_dsp_slices.py --lever apm_ns --placebo <B4b, B4c same readout>`, full sets, arm vs B2f same readout)

DiD = the mean clipped-logit change on the treated bona slice minus that on the control bona slice. Negative means the treated slice moved toward bona. It is calibrated on the 9-run null panel (DiD* = DiD − panel mean, z = DiD* / panel SD).

| contrast | readout | arm recall T/C (B2f final) | DiD [CI] | DiD* / z | clip DiD | half-B DiD merge / final | placebo DiD B4b / B4c (arm rank) | criteria (a)-(e) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **NS** (primary; n 276/106; panel mu +0.32 sd 0.27; bar −0.22) | **final** | 72.8 / 80.2 (69.2 / 75.5) | **+0.25 [−0.43, +0.94]** | −0.08 / −0.28 | −3.0 | +0.20 / +0.05 | +0.31 / −0.07 (2/3) | n n n Y n | **INCONCLUSIVE (provisional)** |
| NS | merge | 69.6 / 78.3 | +0.10 [−0.51, +0.67] | +0.08 / +0.24 | +6.0 | +0.20 / +0.05 | −0.30 / −0.00 (3/3) | n n n Y n | **REFUTED (provisional)** |
| NS_afftdn_vs_anlmdn (secondary; n 139/137; panel mu −0.84; strong ≤ −1.70) | final | 69.8 / 75.9 (62.6 / 75.9) | **−1.05 [−1.84, −0.27]** | −0.21 / −0.49 | +10.0 | −0.57 / −0.64 | −0.36 / −0.50 (1/3) | n n n n Y | DIRECTIONAL |
| NS_afftdn_vs_anlmdn | merge | 65.5 / 73.7 | −0.78 [−1.49, −0.12] | +0.01 / +0.02 | +10.0 | −0.57 / −0.64 | −0.22 / −0.44 (1/3) | n n n n Y | REFUTED |
| VAD (non-target; n 136/246) | final / merge | 72.1/76.4, 70.6/72.8 (66.2/73.6) | −0.50 [−1.18, +0.17] / −0.63 [−1.21, −0.09] | +0.07 / +0.28, −0.07 / −0.40 | +4.1 / +10.7 | −0.13 / +0.09 | +0.36 / −0.64 (2/3); +0.22 / −0.34 (1/3) | | REFUTED / DIRECTIONAL |
| PLC_online (non-target; n 114/254) | final / merge | 69.3/78.3, 66.7/76.0 (67.5/75.6) | −0.02 [−0.63, +0.60] / −0.07 [−0.59, +0.43] | +0.55 / +1.34, +0.57 / +1.63 | | −0.04 / −0.06 | −0.17 / −0.47 (3/3); −0.06 / −0.27 (2/3) | | REFUTED / REFUTED |

For comparison, the B4a report gives B4a's NS DiD +0.49 (final; REFUTED) and afftdn-vs-anlmdn −1.39 [−2.14, −0.69] (final) / −1.14 [−1.83, −0.49] (merge).

Other blocks of the same output:
- **Lever verdict (apm_ns):** final `INCONCLUSIVE (provisional)  [placebos 2]  lower-p retry allowed: False`; merge `REFUTED (provisional)  [placebos 2]  lower-p retry allowed: False`.
- **Outcome (O3.2):**
  - final: heldout 89.38 (Δ +1.10); the panel mean is 88.50, SD 1.35, and 5 of 9 panel runs clear ≥ +0.5;
  - merge: heldout 88.10 (Δ +0.19); the panel mean is 88.25, SD 1.39, and 6 of 9 clear ≥ +0.5;
  - `falsifier (1) outcome escape (heldout F1 delta >= +0.5 on merge_top5 AND final): False`.
- **Spoof-miss guards (Poisson 99 % bound of the B2f count):**

  | guard | merge B2f → B4d | final B2f → B4d | bound merge / final |
  |---|---|---|---|
  | NS_offline | 18 → 14 | 18 → 18 | ≤ 29 / ≤ 29 |
  | VAD_offline | 15 → 18 | 13 → **22** | ≤ 25 / ≤ 22 (at the bound) |
  | PLC | 16 → 14 | 15 → 22 | ≤ 26 / ≤ 25 |
  | **halfB_noisy** | 24 → 33 | **23 → 46** | ≤ 36 / ≤ 35, **BROKEN** on the final |

- **Pinned non-speech tertiles, saturated FA low / mid / high** (half B): B2f 43 / 47 / 46 (merge), 44 / 47 / 42 (final); B4d 38 / 46 / 46 (merge), 42 / 45 / 48 (final). Criterion (ii) is +3 on the full set (final), −8 / −5 on half B (final / merge). That is not this lever's criterion; it is reported for the wave SUMMARY.
- **Run anomaly (O3.1): none.** There is no |z| > 3 on a non-target primary contrast. Under-fit False; clean-F1 guard False.
- **Placebo note for the SUMMARY.** B4d, which never gated a frame, has a VAD DiD of −0.63 on the merge (DIRECTIONAL, rank 1/3), more negative than the gating arm B4c's −0.34. The VAD contrast therefore moves with NS exposure too, and is not gating-specific.

## 3. Per-epoch behaviour, B4d vs B2f (and B4a; same LR at each epoch end: 2.469e-6 … 5.000e-7 in all three `train.log`s)

Sources: `logs/train.log` (all three runs); `metrics.jsonl` (EMA weights; 2,000 clean clips + `sim_matched_mini` + `sim_heldout_mini`, half A). Δ = B4d − B2f mini WF1, from unrounded values.

| ep | B4d mini WF1 | B2f | Δ | B4a | B4d WF1@or | B4d EER noisy (B2f / B4a) | B4d R_bona cl/ma/he | B2f R_bona cl/ma/he | B4d R_spoof ma/he (B2f) | mini heldout F1 B4d / B2f / B4a | mini matched F1 B4d / B2f / B4a | train loss B4d / B2f / B4a | train acc B4d / B2f / B4a | dev loss B4d / B2f / B4a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 86.63 | 88.34 | **−1.71** | 87.71 | 89.26 | 14.10 (13.40 / 13.34) | 94.4/83.8/82.1 | 92.7/81.6/78.6 | **89.5/89.9** (93.7/92.3) | 83.04 / 84.10 / 82.82 | 82.52 / 86.11 / 84.70 | 0.3763 / 0.3558 / 0.3609 | **86.13** / 87.36 / 86.88 | 0.1103 / 0.0927 / 0.0755 |
| 2 | 90.39 | 91.47 | −1.09 | 90.75 | 91.20 | 12.25 (11.78 / 11.89) | 91.3/75.4/72.4 | 91.6/77.1/76.0 | 97.9/96.5 (98.4/96.6) | 86.29 / 87.78 / 86.65 | 88.96 / 90.20 / 89.63 | 0.2136 / 0.2075 / 0.2196 | 96.87 / 97.13 / 96.93 | **0.0856** / 0.0982 / 0.0819 |
| 3 | 90.69 | 91.90 | −1.21 | 92.01 | 91.42 | 12.32 (10.38 / 11.10) | 91.0/74.9/71.9 | 91.6/74.9/71.4 | 98.5/96.8 (99.4/98.8) | 86.37 / 88.47 / 87.59 | 89.53 / 90.65 / 91.69 | 0.1598 / 0.1546 / 0.1584 | 97.77 / 98.09 / 97.98 | 0.0915 / 0.1139 / 0.0860 |
| 4 | 92.11 | 91.69 | +0.42 | 92.16 | 92.78 | 11.49 (10.64 / 11.50) | 92.1/78.8/73.0 | 91.3/75.4/71.4 | 99.0/98.0 (99.0/98.6) | 88.18 / 88.32 / 88.52 | 91.61 / 90.38 / 91.65 | 0.1265 / 0.1108 / 0.1225 | 98.38 / 98.62 / 98.56 | 0.0975 / 0.1242 / 0.1040 |
| 5 | **92.33** | 92.49 | −0.16 | 92.09 | 92.71 | 10.96 (10.46 / 10.80) | 91.6/77.7/71.9 | 90.4/78.8/71.9 | 99.3/99.0 (99.3/99.3) | 88.97 / 89.27 / 87.84 | 91.53 / 91.94 / 92.05 | 0.0963 / 0.0879 / 0.1013 | 98.85 / 98.96 / 98.82 | 0.1080 / 0.1178 / 0.1051 |
| 6 | 92.12 | 91.66 | +0.46 | 92.68 | 92.65 | 11.20 (10.63 / 9.55) | 91.6/76.5/69.9 | 91.3/74.9/70.4 | 99.5/99.3 (99.0/99.0) | 88.46 / 88.36 / 88.97 | 91.44 / 90.17 / 92.29 | 0.0801 / 0.0725 / 0.0746 | 99.06 / 99.18 / 99.11 | 0.1184 / 0.1321 / 0.1037 |
| 7 | 91.97 | 92.10 | −0.14 | 92.58 | **93.09** | **10.89** (10.69 / 9.78) | 90.4/76.0/70.9 | 91.3/76.0/73.0 | 99.4/99.1 (98.9/99.0) | 88.71 / 89.37 / 89.16 | 91.07 / 90.42 / 91.73 | 0.0616 / 0.0540 / 0.0592 | 99.22 / 99.36 / 99.34 | 0.1307 / 0.1406 / 0.1204 |
| 8 | 92.31 | 92.80 | **−0.49** | 93.28 | 93.01 | 11.39 (10.94 / 9.63) | 91.3/78.2/73.0 | 91.3/80.4/73.5 | 98.9/98.9 (99.1/98.9) | 89.22 / 89.41 / 90.39 | 91.25 / 92.38 / 92.13 | 0.0544 / 0.0501 / 0.0562 | 99.34 / 99.44 / 99.36 | 0.1376 / 0.1442 / 0.1166 |

Reading:
- **The selection-half curve is flat from epoch 4.** Mini WF1 is 92.11 / 92.33 / 92.12 / 91.97 / 92.31 at epochs 4-8; the best is epoch 5 (`best_by_wf1`). WF1@oracle and noisy EER are best at epoch 7 (`best_by_noisy_eer`). B2f and B4a both end at their best epoch (92.80, 93.28).
- **B4d ends 0.49 below B2f on half A** (92.31 vs 92.80) and 0.97 below B4a. On half B it is +0.20 above B2f; the two halves disagree in sign.
- **Slow start, mostly on spoof.** At epoch 1, mini R_spoof was 89.5 / 89.9 (matched / heldout) against B2f's 93.7 / 92.3. The mini WF1 trailed B2f by 1.09-1.71 at epochs 1-3. Epoch-1 dev loss was 0.1103, against 0.0927 / 0.0755.
- **Training is slower but not under-fit by the pre-registered line.**
  - Train loss is above B2f's at every epoch (epoch 8: 0.0544 vs 0.0501; B4a 0.0562).
  - Epoch-1 TrainAcc is 86.13: above the 85.36 line, but the lowest of the three runs (87.36 / 86.88).
  - Epoch-8 TrainLoss 0.0544 is under the 0.0752 line.
- **Dev loss** was lowest at epoch 2 (0.085579) and rose ×1.61 to epoch 8 (0.137636). Measured from epoch 1 the rise is ×1.25, against B2f's ×1.55 and B4a's ×1.55 (computed); epoch 1's dev loss was unusually high.
- **The final epoch beats the merge by 0.20 on half B** (92.34 vs 92.14). B2f's margin was 0.16, B4a's 0.60, B4b's 0.04 and B4c's 0.33.
  - The final has the better heldout (89.38 vs 88.10).
  - The merge has the better clean (97.47 vs 97.28), matched (91.62 vs 91.07), echo F1 (93.87 vs 93.53) and the fewer noisy spoof misses (33 vs 46).
  - The weakest merged epoch by mini WF1 is 7 (91.97, mini clean F1 96.81). No merged epoch was evaluated alone on half B, so its half-B drag is not measured.

## 4. What went OK (evidence)

1. **Clean experiment, dosed exactly as designed.**
   - One factor, confirmed by the header lines and `B4 APM NS selftest: PASS`.
   - Realised `wrtc_ns` share 0.5971-0.6007 of clips per epoch, class-balanced (bona 0.5958-0.6046, spoof 0.5962-0.6005); 11,148-11,541 clips per level.
   - No fail key, traceback, OOM, `failed (` or `[b4wave] ` line. Epochs 31:31-31:41 min.
2. **The heldout gain of the final readout repeated at double dose**: heldout F1 +1.10 [−0.24, +2.57] (B4a +1.04).
   - Heldout bona recall 69.82 → 74.94; paired heldout bona 28 fixed / 8 broken.
   - The NS-treated heldout bona rose with dose on the final: offline NS 67.1 → 69.9 → 74.7, afftdn 58.1 → 62.2 → 66.2 (p 0 / 0.3 / 0.6).
3. **The afftdn-vs-anlmdn contrast is again rank 1 of 3** against both placebos: −1.05 [−1.84, −0.27] on the final (DIRECTIONAL), against B4b −0.36 and B4c −0.50.
4. **Clean ranking improved on both readouts**: clean EER 3.42 → 2.37 and 3.33 → 2.01. Only the two NS arms show this (B4a 2.11 / 2.21; placebos 3.10-3.36).
5. **Fewer noisy bona false alarms than B2f on both readouts**: 213 → 200 and 210 → 183 of 782 (half B).
6. **The non-speech probes moved for the first time in the wave.**
   - noise_only bona 100 → 99.47 / 98.93 % spoof.
   - Mean dP fell on both non-speech probes, e.g. sil_only +0.8985 → +0.8685 (final).
7. **Guards that held**:
   - clean F1 within 0.5 (−0.14 / −0.42);
   - full-set offline-NS spoof misses 14 / 18 (≤ 29);
   - ep1 TrainAcc 86.13 (≥ 85.36) and ep8 TrainLoss 0.0544 (≤ 0.0752);
   - English heldout bona recall 58.1 / 60.5 (B4a 56.6 / 58.9);
   - no run anomaly on VAD or PLC.

## 5. What went wrong (evidence)

### 5.1 No dose response: B4a's gain did not grow, and on the WF1 outcome it did not even hold

- **merge** +0.17 [−0.39, +0.74] against B4a's +0.57 [−0.01, +1.17]; **final** +0.20 [−0.42, +0.84] against B4a's +1.02 [+0.43, +1.60] (`verdict.json` of both iterations).
- The final is 0.82 below B4a's final gain, more than the ±0.5 band. Pre-registered falsifier (1) fires (section 6).
- **What was lost is clean and matched, not heldout.**
  - Final clean Δ: +0.35 at p 0.3, −0.42 at p 0.6. Final matched Δ: +1.56, −0.17.
  - Final heldout Δ: +1.04, +1.10.
- **Noisy EER lost B4a's gain completely**: 10.10 / 10.17 against B4a's 9.53 / 9.16 and B2f's 10.09 / 9.66. The heldout gain at p 0.6 is an operating-point gain, not a ranking gain: heldout EER 11.11 → 11.91 on the final, and heldout thr_oracle 0.323.
- **Two readings, which this run cannot separate:**
  - (a) an inverted-U dose curve, with the benefit saturated by p 0.3 and the costs growing with p;
  - (b) B4a's +0.57 / +1.02 was partly seed noise. Its merge CI touched 0, and the B4d half-A curve ends 0.97 below B4a's.
  - Per the pre-registered text, either reading means "do not promote B4a without a seed replicate".

### 5.2 Spoof recall eroded on the final; the half-B noisy spoof-miss guard broke

| | B2f-ep8 | B4a-ep8 | **B4d-ep8** | bound / design limit |
|---|---|---|---|---|
| half-B noisy spoof misses | 23 | 31 | **46** | ≤ 35 (`b4_dsp_slices.py`: BROKEN); design ≤ 35 |
| half-B heldout spoof misses | 14 | 19 | **26** | design ≤ B4a's 19 |
| matched / heldout R_spoof Δ vs B2f | - | −0.19 / −0.32 | **−0.69 / −0.75** | design: within 0.5 |
| full-set offline-VAD spoof misses | 13 | - | 22 | ≤ 22 (at the bound) |

- Paired, on the final: matched spoof 1 fixed / 12 broken, heldout spoof 6 / 18.
- **Where.** Heldout zh 9 → 21, heldout gsm 11 → 18, matched zh 9 → 15, matched opus:wechat 3 → 8.
- **Mostly in clips without an NS stage.** Heldout spoof misses with an `ns` stage went 9 → 10, those without one 5 → 16. This is not an "NS ⇒ bona" shortcut on NS'd spoof (offline-NS guard 18 → 18). It is a general bona-ward shift of the final's heldout operating point (thr_oracle 0.949 → 0.323).
- R_spoof rose in only 9 of 63 full-set slice rows on the final and fell in 43. The worst slice is heldout / snr 0 at 96.99 (B2f's worst: 97.68).
- The merge is milder: noisy 33 (≤ 36), heldout 19 (≤ 23), matched / heldout R_spoof −0.37 / −0.19.

### 5.3 Clean and echo bona lost on the final

- Clean F1 −0.42 [−0.74, −0.11] (p(not better) 0.9945). Clean bona recall 92.91 → 91.91, with 16 fixed / 30 broken clips; clean spoof misses 3 → 7.
- Echo F1 −1.00 [−2.09, +0.03]. Echo bona recall 84.65 → 81.84 (10 fixed / 21 broken). Full-set echo English bona recall 72.5 → 64.7 (B4a 71.7), F1 −2.96.
- Probe_orig bona predicted spoof rose 10.16 → 12.30 %.
- The clean guard holds (`clean_drop` 0.415 ≤ 0.5), with 0.085 to spare.

### 5.4 Oracle-gap decomposition (global threshold vs set-specific)

Method:
- one shared logit threshold swept over the half_B `scores/*.logits` (step 0.05, −6 to +12; ties → the highest logit; computed in this session);
- the per-set oracle WF1 is `half_B.wf1.wf1_oracle`.

| readout | WF1@0.5 | best global (logit / p) | WF1 at best global | per-set oracle WF1 | gap | recovered by one global shift | set-specific remainder |
|---|---|---|---|---|---|---|---|
| B2f-top5 | 91.98 | +3.10 / 0.957 | 92.39 | 92.59 | 0.61 | 0.41 | 0.20 |
| B2f-ep8 | 92.14 | +4.40 / 0.988 | 92.58 | 92.82 | 0.69 | 0.45 | 0.24 |
| B4a-top5 | 92.55 | +1.60 / 0.832 | 92.79 | 92.98 | 0.43 | 0.24 | 0.19 |
| B4a-ep8 | 93.15 | +0.75 / 0.679 | 93.16 | 93.26 | 0.10 | 0.01 | 0.09 |
| B4b-top5 | 92.22 | +0.10 / 0.525 | 92.25 | 92.47 | 0.25 | 0.03 | 0.21 |
| B4b-ep8 | 92.26 | −4.00 / 0.018 | 92.39 | 92.50 | 0.24 | 0.14 | 0.11 |
| B4c-top5 | 92.01 | +2.95 / 0.950 | 92.58 | 92.85 | 0.84 | 0.56 | 0.27 |
| B4c-ep8 | 92.34 | +3.35 / 0.966 | 92.95 | 93.10 | 0.76 | 0.61 | 0.15 |
| **B4d-top5** | 92.14 | **+3.95 / 0.981** | 92.70 | 92.81 | **0.66** | **0.55** | 0.11 |
| **B4d-ep8** | 92.34 | +2.95 / 0.950 | 92.46 | 92.78 | 0.44 | 0.12 | **0.33** |

- **The merge is mis-placed globally**: a shift to p 0.981 would recover 0.55 of its 0.66 gap.
- **The final has the largest set-specific remainder of the wave (0.33).** Heldout wants a low cut (thr_oracle 0.323), while clean, matched and echo want p ≥ 0.990-0.998. One threshold cannot serve both, so the final's +1.10 on heldout and its losses on clean, matched and echo are the same shift.
- The design's limit "oracle gap on the final no larger than B4a-top5's 0.43" is missed by 0.01 (0.44).
- These thresholds are picked on half B, so this is a diagnostic only, not a usable readout.

### 5.5 The saturated bona tail did not shrink; on the final it grew

Share of half_B bona with p_spoof ≥ 0.5 / ≥ 0.9 / ≥ 0.99 / ≥ 0.999 (computed from `scores/*.txt`):

| set (n bona) | B2f-top5 | B4a-top5 | **B4d-top5** | B2f-ep8 | B4a-ep8 | **B4d-ep8** |
|---|---|---|---|---|---|---|
| clean (1,397) | 7.2 / 6.9 / 5.4 / 4.4 | 6.6 / 6.4 / 5.1 / 3.8 | 7.4 / 6.8 / 5.1 / 4.2 | 7.1 / 6.3 / 5.3 / 4.3 | 5.7 / 5.4 / 5.0 / 3.7 | 8.1 / 7.5 / **6.6** / 4.7 |
| matched (391) | 23.8 / 22.3 / 14.6 / 8.2 | 20.5 / 19.7 / 12.0 / 6.6 | 21.5 / 19.9 / 14.8 / 8.4 | 23.5 / 21.5 / 13.6 / 9.0 | 18.7 / 17.6 / 14.1 / 8.2 | 21.7 / 20.5 / **15.9** / **10.0** |
| heldout (391) | 30.7 / 28.6 / 20.2 / 9.2 | 28.6 / 26.9 / 17.1 / 8.4 | 29.7 / 27.6 / 18.4 / 10.0 | 30.2 / 28.4 / 20.5 / 9.5 | 26.6 / 25.8 / 17.4 / 9.5 | 25.1 / 24.6 / 18.7 / **11.3** |
| echo (391) | 16.6 / 15.6 / 10.5 / 4.9 | 14.8 / 14.1 / 10.7 / 5.1 | 17.6 / 15.6 / 10.5 / 6.1 | 15.3 / 14.8 / 10.0 / 5.4 | 14.8 / 14.6 / 11.5 / 5.9 | 18.2 / 17.1 / **13.0** / **8.4** |

Noisy bona (matched + heldout half B, n = 782; en n = 258):

| readout | FA (p ≥ 0.5) | saturated (p ≥ 0.99) | saturated share of FA | FA en / zh | saturated en / zh |
|---|---|---|---|---|---|
| B2f-top5 | 213 | 136 | 64 % | 95 / 118 | 65 / 71 |
| B4a-top5 | 192 | 114 | 59 % | 95 / 97 | 61 / 53 |
| **B4d-top5** | 200 | 130 | 65 % | 95 / 105 | 67 / 63 |
| B2f-ep8 | 210 | 133 | 63 % | 99 / 111 | 67 / 66 |
| B4a-ep8 | 177 | 123 | 69 % | 91 / 86 | 68 / 55 |
| **B4d-ep8** | 183 | **135** | **74 %** | 95 / 88 | **71** / 64 |

- **On the final, the 27 false alarms that B4d removed all came from the movable band** (p 0.5-0.99). The saturated core grew (133 → 135); it is now 74 % of all noisy false alarms, the highest share in the wave.
- **English barely moved**: English false alarms 95 → 95 (merge) and 99 → 95 (final), and English saturated FA rose on the final (67 → 71). Most of the reduction is Chinese: 118 → 105 on the merge (all 13 removed FA) and 111 → 88 on the final (23 of 27).
- The bona tail at p ≥ 0.999 grew on every noisy set on the final: matched 9.0 → 10.0, heldout 9.5 → 11.3, echo 5.4 → 8.4.

### 5.6 The worst slices for bona recall

Full sets, B4d-top5 (B4d-ep8 in brackets), `metrics.json` breakdowns:

| slice | R_bona B4d merge (ep8) | B2f merge (ep8) | B4a merge (ep8) |
|---|---|---|---|
| **heldout lang en** | **57.4 (58.1)** | 59.7 (60.5) | 56.6 (58.9) |
| heldout snr 0 | 62.6 (67.3) | 63.3 (61.2) | 64.6 (69.4) |
| heldout wind | 63.2 (73.7) | 68.4 (72.4) | 60.5 (67.1) |
| matched lang en | 64.0 (62.8) | 63.6 (63.2) | 69.0 (69.0) |
| matched snr 0 | 64.1 (68.4) | 62.4 (65.8) | 68.4 (71.8) |
| heldout washing_machine | 64.1 (66.7) | 62.8 (65.4) | 57.7 (64.1) |
| **echo lang en** | **65.5 (64.7)** | 70.9 (72.5) | 72.1 (71.7) |
| heldout g726 | 66.3 (71.5) | 68.3 (70.3) | 64.2 (67.5) |
| matched opus:wechat | 66.3 (68.3) | 65.4 (71.2) | 71.2 (74.0) |
| matched opus:qq | 66.7 (66.7) | 61.6 (66.7) | 69.7 (68.7) |

- English is still the largest single block of errors. 95 of the 200 / 183 noisy bona false alarms are English, from 258 of 782 bona. Half-B heldout English bona recall is 58.1 / 60.5, against Chinese 76.3 / 82.1.
- B4d added a new weak slice: echo English, which fell from 70.9 / 72.5 to 65.5 / 64.7.

### 5.7 Fit and overfitting signs

- Train accuracy reached 99.34 % while dev loss rose ×1.61 from its epoch-2 minimum. The slope is not steeper than B2f's or B4a's (×1.55 from epoch 1).
- **Slower fit, scaling with p**: epoch-1 TrainAcc 87.36 → 86.88 → 86.13, and train loss above B2f's at all 8 epochs.
  - The pre-registered under-fit line was not crossed. Falsifier (5) also needs heldout bona recall down, and it rose.
- The half-A mini WF1 plateaued from epoch 4 and ended 0.49 below B2f's. A longer schedule is not indicated: the best mini WF1 was epoch 5.

## 6. Verdict on the pre-registered rule

From `exploop/B4/iter04/design.result.json`:

- **Falsifier:** "(1) No dose response: a final delta vs B2f < +0.5, or a merge delta below B4a's +0.57 together with a final below B4a's +1.02 by more than the ±0.5 band. Either means B4a's gain was not dose-driven (or was seed noise); close the flag-level apm_ns line and do not promote B4a without a seed replicate. (2) Trade-off: half-B heldout spoof misses above B4a's 23 / 19, matched or heldout R_spoof down > 0.5, or full-set offline-NS spoof misses > 29. More NS buys bona recall with spoof recall ('NS => bona'); drop the higher dose. (3) Lost raw-noise exposure: WF1 delta < -0.5 with matched F1 down > 1.0 vs B2f; drop. (4) Clean guard: dev_online_clean F1 down > 0.5; drop. (5) Shared under-fit: ep1 TrainAcc < 85.36 or ep8 TrainLoss > 0.0752, together with heldout bona recall down; confounded. (6) Mechanism (secondary): afftdn-vs-anlmdn DiD > -0.84 (the panel mean) on the final; the harsh-suppressor signal does not scale with dose. Verdict rule (wave): positive = merge half_B WF1 delta > +0.5 with CI low > 0 and clean drop <= 0.5; |delta| <= 0.5 = null."
- **Hypothesis:** "B4a's half-B gain comes from the WebRTC NS pass and grows with exposure. … At p = 0.6 the gain should hold or grow, and the merge CI low should clear 0."

`exploop/B4/iter04/verdict.json` (computed by the supervisor; quoted, not recomputed):

| readout | run | base | delta | CI | clean_drop | new_best | decision | verdict |
|---|---|---|---|---|---|---|---|---|
| merge (objective) | 92.1440345440219 | 91.97880641320396 | 0.16522813081795107 | [-0.3913489154983185, 0.7387105180237062] | 0.13564095575543433 | false | **revisit** | **null (inside +-0.5)** |
| final | 92.34047897744139 | 92.13685698540242 | 0.20362199203897546 | [-0.42464648654905646, 0.8352062230253479] | 0.4150305722901777 | false | **revisit** | **null (inside +-0.5)** |

Rules (`verdict.json`): `new_best_delta` 0.0, `keep_best` true (screen: B2f stays the reference); band 0.5; promote = ci_low > 0 and clean_drop ≤ 0.5; positive = delta > 0.5 with ci_low > 0 and clean_drop ≤ 0.5.

**Both readouts are null (inside ±0.5), with decision `revisit` and `new_best` false.**

Falsifiers:

| falsifier | evidence | fired? |
|---|---|---|
| (1) no dose response | final Δ +0.20 < +0.5; merge +0.17 < B4a's +0.57 with final +0.20 below B4a's +1.02 by 0.82 (> 0.5) | **YES, both branches.** Pre-registered consequence: close the flag-level apm_ns line; do not promote B4a without a seed replicate |
| (2) trade-off | half-B heldout spoof misses 19 / **26** (B4a 23 / 19); matched R_spoof −0.37 / **−0.69**, heldout −0.19 / **−0.75**; full-set offline-NS spoof misses 14 / 18 (≤ 29) | **YES on the final** (two of three conditions); not on the merge. Pre-registered consequence: drop the higher dose |
| (3) lost raw-noise exposure | WF1 Δ +0.17 / +0.20; matched −0.17 at worst | no |
| (4) clean guard | clean −0.14 / −0.42 | no (the final has 0.08 to spare; its CI [−0.74, −0.11] excludes 0) |
| (5) shared under-fit | ep1 TrainAcc 86.13 ≥ 85.36; ep8 TrainLoss 0.0544 ≤ 0.0752; heldout bona recall rose | no |
| (6) mechanism (secondary) | afftdn-vs-anlmdn DiD −1.05 (final) ≤ −0.84 | no. But it is less negative than B4a's −1.39, so it did not scale with dose either |

The predicted signature, item by item:

| predicted | merge | final | appeared? |
|---|---|---|---|
| merge Δ > +0.5 with CI low > 0 | +0.17 [−0.39, +0.74] | - | **no** |
| final Δ ≥ +0.5 | - | +0.20 | **no** |
| matched F1 Δ ≥ +1.0 on both | +0.40 | −0.17 | **no** |
| mean noisy EER ≤ 9.53 / 9.16 | 10.10 | 10.17 | **no** (B2f 10.09 / 9.66) |
| half-B noisy bona FA ≤ 192 / 177 | 200 | 183 | **no** (better than B2f's 213 / 210) |
| saturated FA below 136 / 133 | 130 | 135 | merge only |
| final oracle gap ≤ 0.43 | - | 0.44 | **no** (by 0.01) |
| afftdn-vs-anlmdn DiD below B4a's −1.14 / −1.39 | −0.78 | −1.05 | **no** |
| afftdn-vs-anlmdn still rank 1 vs B4b / B4c | 1/3 | 1/3 | yes |
| primary NS contrast not 'shown' | REFUTED | INCONCLUSIVE | yes (as expected) |
| dev_online_clean F1 within 0.5 of 97.60 / 97.69 | 97.47 | 97.28 | yes |
| R_spoof within 0.5 on every set | max −0.37 | matched −0.69, heldout −0.75 | **no (final)** |
| full-set offline-NS spoof misses ≤ 29 | 14 | 18 | yes |
| half-B noisy spoof misses ≤ 36 / 35 | 33 | **46** | **no (final)** |
| half-B heldout spoof misses ≤ 23 / 19 | 19 | **26** | **no (final)** |
| ep1 TrainAcc ≥ 85.36 | 86.13 | | yes |
| ep8 TrainLoss ≤ 0.0752 | 0.0544 | | yes |
| wrtc_ns ≈ 45,471 / epoch, ≈ 11,368 per level, class-balanced, no fail_wrtc_ns | 45,251-45,521; 11,148-11,541; bona 0.5958-0.6046 / spoof 0.5962-0.6005; none | | yes |
| epochs ≤ about 1 min longer than B4a's 31 min | 31:31-31:41 (B4a 31:01-31:12) | | yes |

**Summary.**
- Clean null on the wave rule, dosed exactly as designed.
- The pre-registered "no dose response" falsifier fires on both branches, and the "trade-off" falsifier fires on the final.
- The only parts of the hypothesis that held are the heldout gain of the final and the rank-1 afftdn contrast. Neither scales with dose in a way that lifts WF1.
- Mechanism verdicts are provisional until the wave SUMMARY re-runs `b4_dsp_slices.py --summary` with all arms.

## 7. Decision and next-experiment proposals

**Decision.** Per `verdict.json`, **B4d is not the new best**: `new_best false`, decision `revisit`, verdict `null (inside +-0.5)` on both readouts.
- **B2f stays the reference**: merge 91.98, final 92.14 on half_B; registry ids `B2f-loudnorm-screen` / `B2f-loudnorm-ep8`.
- Every proposal below is one factor (or one coupled group) against **B2f's flags**: `scripts/run_B4d.sh` without `--aug_apm_p`, with the `b4wave.train` entry, `--batch_size 16` and 8 epochs.
- This was the last iteration of the wave (`max_iterations` 4); the SUMMARY follows, so these proposals are inputs for the SUMMARY and the owner.
- By the pre-registered falsifier (1), the flag-level apm_ns line is closed, and **B4a must not be promoted without a seed replicate**. The p = 0.6 dose is dropped (falsifier (2)).
- `b4_dsp_slices.py` prints `lower-p retry allowed: False`, so a lower-p apm_ns run (for example 0.15) is not proposed.

**Largest remaining error.** Confident noisy bona false alarms, worst for English, that no B4 arm has removed:
- 200 / 183 of 782 half-B noisy bona are false alarms (B4d merge / final), 130 / 135 of them at p ≥ 0.99 (65 % / 74 %);
- English false alarms are 95 on both readouts, and half-B heldout English bona recall is 58.1 / 60.5, against Chinese 76.3 / 82.1;
- sil_only bona is still 100 % spoof;
- for B4d specifically, the final's bona-ward heldout shift cost spoof recall: 46 half-B noisy spoof misses, against B2f's 23.

**Ranked proposals (configuration first, seed repeat last).**

1. **`--aug_opus_plc_p 0.3 --aug_opus_loss 0 0` (one coupled group: the in-process libopus decoder path without packet loss; the plan's pre-registered attribution control).**
   - Hypothesis: B4b's merge-only matched gain (+0.86, matched bona recall 76.21 → 78.52; `metrics.json`) comes from the aligned libopus decode, not from loss and concealment.
   - It is also the fourth equal-dose DSP placebo for the NS contrasts. That tests whether B4a's and B4d's rank-1 afftdn-vs-anlmdn signal (−1.39 / −1.05 final) and the NS-arm clean-EER drop (3.33 → 2.21 / 2.01) are NS-specific.
   - Targets: matched (opus-codec) bona false alarms. Half-B matched R_bona is 76.21 / 76.47, and matched opus:qq / wechat are among the worst bona slices (66.7 / 66.3 in B4d).
   - Predicted signature:
     - matched F1 Δ ≥ +0.5 vs B2f on both readouts;
     - WF1 Δ ≥ B4b's +0.24 / +0.12;
     - afftdn-vs-anlmdn DiD > −0.84 (placebo-like), so the NS arms stay rank 1;
     - clean EER ≥ 3.0 (no NS-like clean-ranking gain);
     - half-B noisy spoof misses ≤ 36 / 35;
     - VAD z < +3 (B4b was ANOMALOUS on VAD).
   - Falsifier:
     - matched Δ < +0.5 on both readouts with WF1 inside ±0.5 → the decoder path is worth nothing either; close the opus family;
     - afftdn-vs-anlmdn DiD ≤ −1.05 or clean EER ≤ 2.4 → the "NS-specific" signals are generic DSP exposure.
2. **`--aug_dtx_p 1.0` (the only dose knob of dtx_gate that flags allow).**
   - Hypothesis: B4c's VAD gain on 1-40 %-gated bona is exposure-limited. p = 1.0 gives about 3.3× as many processed clips (B4c report §7). The per-clip gating stays under-dosed (ratio near 0.03), so the run is a cap: if it still does not move VAD, the flag-level dtx line is closed.
   - New constraint from this run: B4d, which never gated a frame, reached VAD DiD −0.63 on the merge. A gating effect must therefore beat the NS arms too.
   - Targets: the saturated noisy bona core (130 / 135 at p ≥ 0.99), especially the pause-rich high non-speech tertile (B4d half B 46 / 48), and sil_only bona (100 % spoof).
   - Predicted signature:
     - VAD DiD ≤ −1.05 on the final, rank 1 against B4a, B4b and B4d;
     - criterion (ii) ≥ 0 on half B for both readouts;
     - sil_only bona mean dP below +0.8929 / +0.8985;
     - half-B saturated noisy FA below 136 / 133.
   - Falsifier:
     - half-B noisy spoof misses > 36 / 35, or matched R_spoof down > 0.5 (B4c −0.69) → gating carries class information; drop;
     - VAD DiD ≥ −0.64 (no better than B4c) → exposure is not the limit; close dtx.
3. **`--aug_apm_p 0.3 --aug_apm_ns_levels 18 21` (one coupled group: strong WebRTC suppression only, at B4a's dose). Conditional: run only if the owner re-opens the apm_ns line after a B4a seed replicate confirms the 0.3 gain.**
   - Hypothesis: the NS effect that survives both doses and both placebos sits on the harsh ffmpeg suppressor.
     - afftdn-vs-anlmdn is rank 1 at both doses (−1.39 / −1.05 final).
     - Half-B afftdn bona recall rose 58.1 → 62.2 → 66.2 with p on the final.
     - Raising p costs matched, clean and spoof recall (section 2.6). Concentrating the 30 % budget on the 18 / 21 dB levels may buy the afftdn gain without those costs.
   - Targets: heldout afftdn-suppressed bona, the lowest NS slice (B2f half B 55.4 / 58.1).
   - Predicted signature:
     - afftdn-vs-anlmdn DiD ≤ −1.70 (strong) on the final;
     - heldout F1 Δ ≥ +0.5 on both readouts;
     - merge WF1 Δ > +0.5 with CI low > 0;
     - matched F1 Δ ≥ 0;
     - half-B heldout spoof misses ≤ 23 / 19;
     - R_spoof within 0.5 everywhere.
   - Falsifier:
     - afftdn-vs-anlmdn DiD > −0.84 on the final → the suppressor level is not the active ingredient;
     - half-B noisy spoof misses > 36 / 35 or matched / heldout R_spoof down > 0.5 → harsh suppression buys bona recall with spoof recall; drop.
4. **Seed repeat of the reference: `--seed 1235` on B2f (last).**
   - Purpose: measure B2f's own half-B seed band. B4a's +0.57 / +1.02 and B4d's failure to reproduce it can then be read against same-configuration noise instead of the 9-run mixed panel (heldout F1 SD 1.35 / 1.39).
   - **This is outside wave B4's search space.** `wave.json` pins seed 1234, `--seed` is not in `allowed`, and the owner declined a B2f replicate arm (`plan_snapshot.md`). It needs an owner decision.
   - Predicted signature: |Δ| ≤ 0.5 vs B2f on both readouts.
   - Falsifier: a repeat that differs from B2f by ≥ +0.57 (merge) or ≥ +1.02 (final) puts B4a's gains inside seed noise.
   - The more direct test, a B4a replicate (`--aug_apm_p 0.3 --seed 1235`), is two factors against B2f and therefore belongs to the owner's wrapup decision, below.

**Recommended full run if the wave stopped now.**
- Per `verdict.json`, the reference is still B2f, and the plan leaves the wave's best to the owner at wrapup.
- B4d adds nothing to promote. It is null on both readouts, two pre-registered falsifiers fired, and the noisy spoof-miss guard broke on the final.
- B4a's final readout is still the wave's only positive verdict (Δ +1.02 [+0.43, +1.60], registry decision `promote`). But B4d's falsifier (1) now requires a seed replicate before any promotion.
- Recommended:
  - **Flags:** B4a's exact command (`scripts/run_B4a.sh`) at seeds 1235 and 1236. This is both the replicate and the full run; seed 1234 already exists.
  - **Decision rule:** promote B4a over B2f only if both replicates' final readouts beat B2f-ep8 (92.14) on half_B, with the paired-bootstrap CI excluding 0 and clean drop ≤ 0.5. Otherwise keep B2f.
  - **Readout to submit:** the final-epoch EMA checkpoint. The final beat the merge in every run of the wave: B2f +0.16, B4a +0.60, B4b +0.04, B4c +0.33, B4d +0.20.
  - Keep `--keep_topk_by_wf1 7 --keep_epochs 8`.
- Pre-launch checks:
  - `bash -n scripts/run_B4a.sh`;
  - `which ffmpeg` (a container restart wipes it, and the codec stage then silently no-ops);
  - `pyloudnorm` and `pywebrtc_audio` importable in the rtc-sdd env;
  - an empty `pgrep -af "b4wave.train|b3wave.train|main_train"`;
  - an idle GPU.
- Post-launch checks:
  - `B4 APM NS selftest: PASS`;
  - `B4 APM NS: on p=0.3 …`;
  - `Warmup-cosine: 1895 warmup / 37896 steps`;
  - epoch-1 TrainAcc within 2 of B4a's 86.88.
- Every evaluation needs `--loudness_norm --loudness_norm_lufs -23`.
