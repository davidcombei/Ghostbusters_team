# Experiment exploration B4-03: B4c (`B4c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_dtx03_s1234`) - webrtcvad DTX gating (dtx_gate) on top of B2f

> Written 2026-10-19 by the exploop B4 analysis agent (ANALYZE step, wave B4, iteration 3, attempt a0).
>
> **Sources.** Every number below was copied from a file read in this session, or computed in this session from the half_B rows (and, where marked "full set", all rows) of the local `scores/*.txt` / `*.logits` files. Nothing is quoted from memory. The files:
> - run config and logs: `config.yaml` and `git.diff` (B4c), `config.yaml` (B2f), `logs/train.log` and `metrics.jsonl` (both runs), `top5_by_wf1/{merge_list.txt,select_topk.log}`, `local_eval/merge_top5/merge_list.txt`, the `ckpt/` listing;
> - local evals: `local_eval/{merge_top5,epoch_8}/{metrics.json,probe_report.json,compare_vs_B2f-*.txt,ci_vs_B2f-*.txt,compare_ci_vs_B2f-*.json,compare_ci.json}` and `local_eval/merge_top5/compare_all_readouts.txt` (B4c); the `metrics.json`, `probe_report.json` and `scores/` of the B2f, B4a and B4b readouts;
> - stdout log, header lines only: `SLS_setup/B4c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_dtx03_s1234_20261019004215.log`;
> - loop files: `exploop/B4/iter03/{verdict.json,design.result.json,log_merge.txt,log_final.txt}`, `exploop/B4/{wave.json,plan_snapshot.md,STATUS.md}`, `scripts/run_B4c.sh`;
> - the pre-registered mechanism readout `scripts/b4_dsp_slices.py --lever dtx_gate`. It was run in this session on `local_eval/epoch_8` (primary) and `local_eval/merge_top5`, with the default `--ref` (B2f, same readout), the pinned null panel, and `--placebo` = B4a's and B4b's same readout (2 placebo arms). There was no `--json_out`, and stdout was read directly in the session. No scratch file was written. The rules are in `SLS_setup/b4wave/README.md` (sections "Lever `dtx_gate`" and "Mechanism readout and verdict rules");
> - registry: `docs/experiment_log.csv` (rows `B2f-loudnorm-*`, `B4a-apmns-*`, `B4b-opusplc-*`, `B4c-dtxgate-*`);
> - data: `data/dev_noisy_sim/*/meta.csv` (half, lang, src, codec, family, `dsp_chain`), `data/dev_noisy_sim/dev_online_clean/protocol.txt`, `configs/local_eval_sets.yaml`, `SLS_setup/utils/metrics.py` (macro-F1, spoof if p ≥ 0.5).
>
> Progress-split scores (`model_merging_top5/scores.txt`, `progress.txt`) were not read.
>
> **Previous reports.**
> - `docs/experiments-exploration/b4-experiments/20261018_194212_B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234.md` (B4a, iteration 1). Per the user note of 2026-10-18 21:38, B4a's registry decisions are merge = revisit and final = promote. The "drop" labels inside that report are an artefact of the old screen setting.
> - `docs/experiments-exploration/b4-experiments/20261019_001203_B4b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_opusplc03_s1234.md` (B4b, iteration 2).
> - The structure follows `docs/experiments-exploration/b3-experiments/20261010_143058_B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234.md`.
>
> **Readouts.**
> - `-top5` is the weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5/`). It is the wave objective.
> - `-ep8` is the final-epoch EMA checkpoint alone (`local_eval/epoch_8/`).
> - half_B is the half of each sim set that is never used for selection: n = 1,994 per sim set (1,603 spoof, 391 bona). Clean is the full 7,465-clip `dev_online_clean` (6,068 spoof, 1,397 bona).
> - WF1 = 0.3 clean + 0.35 matched + 0.35 heldout, using macro-F1 at p_spoof 0.5. Echo is reported but is not in WF1.
> - Threshold 0.5 on the half_B score rows reproduces all eight half_B WF1 values exactly (computed): B2f 91.98 / 92.14, B4a 92.55 / 93.15, B4b 92.22 / 92.26, B4c 92.01 / 92.34.
> - All readouts were evaluated with `"loudness_norm": -23.0` (`metrics.json`).

Run directories (under `SLS_setup/exp/`):
- **B4c** `B4c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_dtx03_s1234_epoch8_bs16_20261019004222` (this report)
- **B2f** `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623`: the reference for the whole wave (`wave.json` `rules.keep_best` true = screen design). Its half_B scores are merge 91.98 and final 92.14.
- **B4a** `B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234_epoch8_bs16_20261018150134` and **B4b** `B4b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_opusplc03_s1234_epoch8_bs16_20261018201137`: iterations 1-2, used here as placebo arms.

## 1. Setup

**B4c = B2f with one factor changed: `--aug_dtx_p 0.3`.**
- On 30 % of training clips (rng4, a separate B4 RNG stream, both classes), `webrtcvad` gates the audio in the `post_dsp` slot, after the NS slot (off here) and before the codec.
- Mode 1-3 and frame 10 / 20 / 30 ms are drawn per clip, with a forward hangover of U(40, 200) ms.
- Gated samples are filled with exact zeros or with order-10 LPC comfort noise, 50/50 per clip (`--aug_dtx_fill mix`, the default of the coupled flag).

Sources: `scripts/run_B4c.sh`, `b4wave/README.md` "Lever `dtx_gate`".

Flag diff (`args` of both `config.yaml`, parsed and compared in this session):

| flag / field | B2f | **B4c** | note |
|---|---|---|---|
| trainer | `main_train.py`, git `00da485`, dirty | **`b4wave/train.py`** (b4wave 0.1.0 on b3wave 0.1.0), git `8ea996c`, dirty | `git.diff` touches only `.claude/skills/post-train-eval/SKILL.md` and `docs/experiment_log.csv`. `module_sha256` `b6498bc4a5af…`, the same prefix as the `b4wave sha256` header line |
| `--aug_dtx_p` | (absent = 0) | **0.3** | the lever |
| `--aug_dtx_fill` | - | `mix` (default) | coupled with `--aug_dtx_p` (`wave.json` `coupled`) |
| other B4 flags | - | `aug_apm_p 0.0`, `aug_opus_plc_p 0.0` (off); `aug_apm_ns_levels [6, 12, 18, 21]`, `aug_opus_loss [0.01, 0.2]`, `aug_opus_burst [1.0, 4.0]` unread | |
| B3 flags (written by the b3wave parser) | - | defaults: `sampler shuffle`, `aug_snr_db 5 20 uniform`, `aug_codec_profiles fixed`, `consistency none`, `llrd 1.0`, `freeze_bottom_k 0`, `specaug False`, `train_subset_frac 1.0` | same behaviour as B2f |
| everything else | identical | identical | no other `args` key differs except `track` |

**Header lines proving the change took effect** (stdout log):
- `B4 DTX gate: on p=0.3 fill=mix modes=[1, 2, 3] frames_ms=[10, 20, 30] hangover_ms=40..200`. This matches the design's first `expect` regex.
- `B4 DTX gate selftest: PASS 18 mode x frame x hangover runs | gated_frac 0.343..0.491 (gap share 0.556) | speech_kept True | zero_exact max 0 | cng_level dB -0.01..+0.02 | cng_colour dB -0.14..+0.01 | tone_unchanged True`.
- The `RTC augmentation:` line ends with `| wrtc_dtx p=0.3 fill=mix`.
- `b3wave: 0.1.0`, `b4wave: 0.1.0`, `b4wave sha256: b6498bc4a5af`.
- Unchanged lines: `Sampler: shuffle`; `SNR: 5..20 dB dist=uniform | Chaining: max_stages=1 | Codec profiles: fixed`; `Consistency: none`, `LLRD: 1`, `SpecAugment: off`; `Loudness norm: loudness_norm target=-23 LUFS`; `CE weights: spoof=0.1 bonafide=0.9`; `Parameters: 320268933`, `Train trials: 75785`; `Warmup-cosine: 1895 warmup / 37896 steps, floor 5.0e-07`.
- The stdout log has no `Traceback`, `out of memory` or `failed (` line (grep count 0), and no `[b4wave] ` lever-failure line (count 0).

**Realised dose.** Sources: `metrics.jsonl` `b3.aug_counts` / `b4.aug_by_class` (the `Epoch N B3 aug` / `B4 aug-by-class` lines of `train.log`), shares computed in this session; the pooled block printed by `b4_dsp_slices.py`.

| epoch | `wrtc_dtx_zero + wrtc_dtx_cng` / 75,785 | bona share (n = 14,353) | spoof share (n = 61,432) | zero-fill share | gated-fraction bins g0 (< 1 %) / g1_20 | ≥ 20 % gated (`dtx_g20_40 + dtx_g40p`) | `fail_wrtc_dtx` |
|---|---|---|---|---|---|---|---|
| 1 | 22,647 = 0.2988 | 0.2996 | 0.2987 | 0.498 | 0.437 / 0.553 | 224 = 0.0099 | 0 |
| 1-8 range | 22,603-22,853 = 0.2983-0.3016 | 0.2931-0.3042 | 0.2969-0.3031 | 0.497-0.509 | 0.428-0.437 / 0.552-0.562 | 210-255 = 0.0093-0.0112 | 0 every epoch |

`b4_dsp_slices.py` dose block, summed over epochs 1-8:

| | processed | g0 | g1_20 | g20_40 | g40p | ≥ 20 % gated |
|---|---|---|---|---|---|---|
| arm, bona | 34,253 / 114,824 (29.8 %) | 38.1 % | 60.4 % | 1.5 % | 0.0 % | 1.5 % |
| arm, spoof | 147,411 / 491,456 (30.0 %) | 44.6 % | 54.4 % | 0.9 % | 0.1 % | 0.9 % |
| arm, all | 181,664 / 606,280 (30.0 %) | 43.4 % | 55.6 % | 1.0 % | 0.1 % | 1.0 % |
| heldout `vad_gate`, all (n 629) | - | 1.3 % | 66.6 % | 15.7 % | 16.4 % | 32.1 % |

- **dose_ratio = 0.032** (bona 0.047, spoof 0.029). The rule says < 0.25 is UNDER-DOSED. The `lever_notes` pre-registered "~0.027".
- The clip-level dose matches the design (about 22,736 clips per epoch, 50/50 fill, class-balanced, `fail_wrtc_dtx` 0). Falsifier (5), "not dosed", therefore does not fire.
- The per-clip gating is the weak part. 43.4 % of the processed clips had less than 1 % of their samples gated, and only 1.0 % had 20 % or more, against 32.1 % of the heldout VAD clips.

**Run time.**
- `config.yaml` was created at 00:42:32. The epoch-8 local eval finished at 04:54:20 (`train.log`), i.e. **4 h 12 min**. The registry `gpu_h` is 4.20, against B2f's 4.14, B4a's 4.21 and B4b's 3.55.
- Epochs 2-8 took 30:56-31:04 min each, against B2f's 30:24-30:35 (eval-end to epoch-end timestamps in both `train.log`s; computed).
- Post-train file times: `top5_by_wf1/merge_list.txt` 04:54, `merge_top5/metrics.json` 05:03, `epoch_8/metrics.json` 05:08. `verdict.json` was written at 05:20:03.

**Checkpoints.**
- All 8 epochs are kept at 1,281,235,691 bytes each. `train.log` has no `Pruned` line. Epoch 1 ranks 8th by mini WF1 but is the `best_by_dev_loss` target (0.083560).
- The symlinks are `best_by_dev_loss` → epoch 1, `best_by_noisy_eer` → epoch 8 and `best_by_wf1` → epoch 8.

**Merged top-5 by half-A mini WF1: epochs 8, 7, 6, 5, 3** (92.83, 92.81, 92.30, 91.90, 91.64; mean 92.30; `select_topk.log`).
- Excluded: 4 (91.48), 2 (89.84) and 1 (88.45).
- B2f's merge was epochs 8, 5, 7, 3, 4 (registry row `B2f-loudnorm-screen`).

## 2. Results vs references (half_B)

Sources: `compare_all_readouts.txt`, `compare_vs_B2f-*.txt`, `ci_vs_B2f-*.txt` / `compare_ci_vs_B2f-*.json` (paired bootstrap, n = 2,000, half B, threshold 0.5), and the `half_B` block of every `metrics.json`. B4a's and B4b's CIs come from their registry rows. `compare_all_readouts.txt` gives every delta against its first column (B2f-top5), so the B4c-ep8 − B2f-ep8 deltas come from `compare_vs_B2f-ep8.txt` / `compare_ci_vs_B2f-ep8.json`.

### 2.1 Wave iterations and fixed comparators, both readouts

| half_B | B2f-top5 | B2f-ep8 | B4a-top5 | B4a-ep8 | B4b-top5 | B4b-ep8 | **B4c-top5** | **B4c-ep8** | B4c-top5 − B2f-top5 [95 % CI] | B4c-ep8 − B2f-ep8 [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| **WF1@0.5** | 91.98 | 92.14 | 92.55 | 93.15 | 92.22 | 92.26 | **92.01** | **92.34** | **+0.03 [−0.47, +0.51]**, p(not better) 0.431 | **+0.20 [−0.39, +0.78]**, p 0.240 |
| (CI vs B2f, registry) | | | [−0.01, +1.17] | [+0.43, +1.60] | [−0.30, +0.78] | [−0.48, +0.71] | | | | |
| WF1@oracle | 92.59 | 92.82 | 92.98 | 93.26 | 92.47 | 92.50 | 92.85 | 93.10 | **+0.25** | **+0.27** |
| clean F1@0.5 | 97.60 | 97.69 | 97.72 | 98.05 | 97.70 | 97.71 | 97.68 | 98.02 | +0.08 [−0.18, +0.33] | **+0.33 [+0.05, +0.59]**, p 0.006 |
| noisy F1@0.5 (mean matched, heldout) | 89.57 | 89.76 | 90.34 | 91.06 | 89.88 | 89.92 | 89.58 | 89.90 | +0.01 | +0.15 |
| mean noisy EER | 10.09 | 9.66 | 9.53 | 9.16 | 10.66 | 9.68 | 10.09 | 10.04 | −0.00 | **+0.38** (worse) |
| calibration gap noisy | 0.66 | 0.71 | 0.49 | 0.05 | 0.18 | 0.21 | **0.94** | **0.97** | | |
| separability gap | 7.87 | 7.86 | 7.18 | 7.17 | 8.04 | 7.90 | 7.76 | 7.41 | | |

Per-set paired CIs (`ci_vs_B2f-*.txt`):

| set | merge Δ [CI], p(not better) | ep8 Δ [CI], p(not better) |
|---|---|---|
| clean | +0.08 [−0.18, +0.33], 0.266 | **+0.33 [+0.05, +0.59], 0.006** |
| matched | −0.17 [−1.18, +0.84], 0.6215 | +0.12 [−0.91, +1.20], 0.404 |
| heldout | +0.20 [−0.81, +1.16], 0.339 | +0.18 [−1.14, +1.54], 0.395 |
| echo | +0.30 [−0.58, +1.14], 0.2465 | +0.07 [−0.94, +1.03], 0.4555 |

Weighted contributions (0.3 / 0.35 / 0.35 × the unrounded deltas of `compare_ci_vs_B2f-*.json`; computed):
- merge: +0.02 clean − 0.06 matched + 0.07 heldout = +0.03;
- final: +0.10 clean + 0.04 matched + 0.06 heldout = +0.20.

Reading:
- **Neither readout moved WF1@0.5 outside the ±0.5 band.** Half of the final's +0.20 is the clean set, and the clean delta is the only per-set CI that excludes 0.
- **WF1@oracle rose on both readouts (+0.25 / +0.27) while WF1@0.5 barely moved.** The noisy calibration gap grew from 0.66 / 0.71 to 0.94 / 0.97. B4c ranks a little better than B2f, but its 0.5 operating point is worse placed (section 5.3). This is the opposite of B4b, whose +0.24 / +0.12 was an operating-point shift with a lower oracle.
- **Full sets** (registry rows, `log_merge.txt` / `log_final.txt`):
  - merge WF1 92.33 vs B2f 92.25 (+0.08), with matched +0.22, heldout −0.04 and echo +0.24;
  - final 92.49 vs 92.53 (−0.04), with heldout −0.38.

### 2.2 Per set (R = recall at 0.5; thr = p_spoof threshold; `metrics.json` half_B)

| set | metric | B2f-top5 | **B4c-top5** | B2f-ep8 | **B4c-ep8** |
|---|---|---|---|---|---|
| dev_online_clean | F1@.5 / F1@or / thr_or | 97.60 / 98.10 / 0.997 | 97.68 / 98.28 / 0.996 | 97.69 / 98.33 / 0.999 | **98.02** / 98.28 / 0.997 |
| | EER / ECE | 3.42 / 1.42 | 3.36 / 1.33 | 3.33 / 1.33 | 3.14 / 1.15 |
| | R_spoof / R_bona | 99.90 / 92.84 | 99.84 / 93.34 | 99.95 / 92.91 | 99.87 / **94.27** |
| sim_matched_v1 | F1@.5 / F1@or / thr_or | 91.22 / 91.53 / 0.846 | 91.05 / 91.64 / 0.974 | 91.24 / 92.05 / 0.991 | 91.36 / 91.70 / 0.957 |
| | EER / ECE | 8.15 / 4.75 | **7.74** / 4.94 | 8.21 / 4.86 | 8.37 / 4.92 |
| | R_spoof / R_bona | 99.50 / 76.21 | **98.81** / 78.01 | 99.44 / 76.47 | **98.75** / 79.03 |
| sim_heldout_v1 | F1@.5 / F1@or / thr_or | 87.92 / 88.94 / 0.957 | 88.11 / 89.40 / 0.952 | 88.27 / 88.89 / 0.949 | 88.45 / **90.04** / 0.973 |
| | EER / ECE | 12.03 / 6.46 | 12.44 / 6.29 | 11.11 / 6.38 | 11.72 / 6.33 |
| | R_spoof / R_bona | 99.00 / 69.31 | 99.25 / 69.05 | 99.13 / 69.82 | 99.19 / 70.08 |
| sim_echo_v1 | F1@.5 / F1@or / thr_or | 94.16 / 94.91 / 0.983 | 94.45 / 95.04 / 0.962 | 94.53 / 95.41 / 0.983 | 94.60 / 94.92 / 0.961 |
| | EER / ECE | 4.71 / 3.37 | 5.24 / 3.24 | 4.86 / 3.20 | 5.15 / 3.18 |
| | R_spoof / R_bona | 99.75 / 83.38 | 99.63 / 84.65 | 99.69 / 84.65 | 99.31 / 86.19 |

Reading:
- **Bona recall rose on clean, matched and echo, on both readouts**: clean +0.50 / +1.36, matched +1.79 / +2.56, echo +1.28 / +1.53 (`compare_vs_B2f-*.txt`).
- **Heldout, the set the lever targets, barely moved**: R_bona −0.26 / +0.26, F1 +0.20 / +0.18. Heldout F1@oracle did rise, by +0.46 / +1.15.
- **Matched spoof recall fell by 0.69 on both readouts** (99.50 → 98.81, 99.44 → 98.75). That breaks the design's must-not-move item "R_spoof within 0.5 on every set" (section 6). Echo R_spoof fell by 0.37 on the final.

Paired per clip (half_B score files, B4c vs B2f at the same readout; computed):

| set | class | merge fixed / broken / wrong in both | ep8 fixed / broken / wrong in both |
|---|---|---|---|
| clean | bona | 18 / 11 / 82 | **25 / 6** / 74 |
| matched | bona | 14 / 7 / 79 | 16 / 6 / 76 |
| heldout | bona | 9 / 10 / 111 | 18 / 17 / 100 |
| echo | bona | 11 / 6 / 54 | 13 / 7 / 47 |
| clean | spoof | 0 / 4 / 6 | 0 / 5 / 3 |
| matched | spoof | **1 / 12** / 7 | **3 / 14** / 6 |
| heldout | spoof | 7 / 3 / 9 | 5 / 4 / 9 |
| echo | spoof | 3 / 5 / 1 | 3 / **9** / 2 |

Half-B spoof misses (p < 0.5; computed):

| set (n spoof) | B2f-top5 | B4a-top5 | B4b-top5 | **B4c-top5** | B2f-ep8 | B4a-ep8 | B4b-ep8 | **B4c-ep8** |
|---|---|---|---|---|---|---|---|---|
| clean (6,068) | 6 | 9 | 5 | **10** | 3 | 7 | 2 | **8** |
| matched (1,603) | 8 | 8 | 8 | **19** | 9 | 12 | 14 | **20** |
| heldout (1,603) | 16 | 23 | 11 | 12 | 14 | 19 | 15 | 13 |
| echo (1,603) | 4 | 4 | 3 | 6 | 5 | 6 | 2 | **11** |

Where the new matched spoof misses sit (half B, computed; B2f-top5 / B4c-top5 / B2f-ep8 / B4c-ep8):
- language: zh 8 / 16 / 9 / 14; en 0 / 3 / 0 / 6;
- noise family: coffee 0 / 5 / 1 / 5; rain 2 / 6 / 3 / 6; office 4 / 5 / 4 / 5; footsteps 1 / 2 / 0 / 3; keyboard 1 / 1 / 1 / 1; echo (no noise) 0 / 0 / 0 / 0;
- codec: wechat 3 / 7 / 3 / 8; qq 4 / 6 / 5 / 4; dingtalk 0 / 2 / 0 / 3; the other four apps 0-2 each;
- echo set: self_echo 3 / 5 / 4 / 10.

**B4c doubled the matched spoof misses.** The new misses sit in clips with continuous background noise (coffee, rain) and are absent from the noise-free echo family.

### 2.3 Probes (`probe_report.json`; 187 bona + 813 spoof; one bona probe clip = 0.53 points)

| probe (bona unless stated) | B2f-top5 | **B4c-top5** | B2f-ep8 | **B4c-ep8** |
|---|---|---|---|---|
| level_flip_bona (mean) / max | 1.54 / 4.81 | **1.27** / 4.28 | 1.74 / 4.81 | 1.60 / 3.74 |
| gain_up / gain_p6 / gain_p10_limit flip % | 4.28 / 2.14 / 4.81 | 2.67 / 1.07 / 4.28 | 4.81 / 2.67 / 3.21 | 3.74 / 2.14 / 3.74 |
| agc_dynaudnorm / loudnorm flip % | 0.53 / 0.53 | 0.53 / 1.60 | 2.14 / 1.07 | 1.60 / 1.60 |
| gain_down / gain_m20 / gain_m6 flip % | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| level_flip_spoof | 0.03 | 0.03 | 0.03 | 0.05 |
| probe_orig bona pred spoof % (`pred_spoof_orig`) | 10.70 | 11.76 | 10.16 | **8.56** |
| **noise_only bona pred spoof % / mean dP** | 100.00 / +0.8870 | 100.00 / +0.8814 | 100.00 / +0.8965 | 100.00 / **+0.9130** |
| **sil_only bona pred spoof % / mean dP** | 100.00 / +0.8929 | 100.00 / **+0.8825** | 100.00 / +0.8985 | 100.00 / **+0.9134** |
| trimmed / padded bona flip % | 3.74 / 3.74 | 3.21 / **4.81** | 3.74 / 5.35 | 2.67 / 2.14 |

Reading:
- **The non-speech failure is untouched.** `noise_only` and `sil_only` bona are still 100 % spoof on every readout.
  - On the merge, mean dP is slightly lower (sil_only +0.8825 vs +0.8929).
  - On the final, it is higher (+0.9134 vs +0.8985), against the design's prediction.
- The padded-probe bona flip rose by 2 clips on the merge (3.74 → 4.81), exactly the design's limit ("up by at most 2 clips"). On the final it fell (5.35 → 2.14).
- Level sensitivity fell slightly: level_flip_bona 1.54 → 1.27 (merge) and 1.74 → 1.60 (final).

### 2.4 Slices that moved most, B4c vs B2f (full sim sets, `metrics.json → breakdowns`, 63 slice rows; computed)

| readout | F1 up | EER down | R_bona up | R_spoof up |
|---|---|---|---|---|
| merge | 38 of 63 | 35 | 43 | **18** |
| final | 34 of 63 | 36 | 40 | **9** |

- F1 up per set, merge / final: matched 13 / 13 of 23, heldout 10 / 11 of 23, echo 15 / 10 of 17.
- **R_spoof fell in most slices**: it rose in only 18 / 9 of 63 rows.
- Minimum slice R_spoof:
  - B4c-top5 97.35 (heldout / washing_machine), B4c-ep8 **97.14 (matched / snr 0)**;
  - B2f-top5 97.31 (heldout / door_wood_creaks), B2f-ep8 97.68 (heldout / washing_machine).

| set / slice | n | F1 merge B2f → B4c | F1 ep8 B2f → B4c | EER ep8 | R_bona merge / ep8 | R_spoof merge / ep8 |
|---|---|---|---|---|---|---|
| heldout / door_wood_creaks | 361 | 86.83 → 90.03 **+3.20** | 88.70 → 89.29 +0.59 | 10.18 → 8.28 | 71.9 → 73.4 / 73.4 → 75.0 | 97.31 → 99.33 / 98.32 → 98.32 |
| matched / keyboard | 647 | 91.49 → 93.55 +2.07 | 91.84 → 93.88 +2.05 | 7.74 → 8.30 | 76.7 → 82.5 / 77.7 → 83.5 | 99.45 → 99.45 / 99.45 → 99.45 |
| matched / opus:qq | 572 | 84.50 → 86.17 +1.67 | 86.04 → 87.00 +0.96 | 10.95 → 9.83 | 61.6 → 67.7 / 66.7 → 67.7 | 98.73 → 98.10 / 98.31 → 98.73 |
| heldout / snr 15 | 835 | 88.08 → 89.65 +1.57 | 89.78 → 90.10 +0.32 | 10.71 → 10.56 | 68.2 → 72.2 / 71.5 → 72.8 | 99.42 → 99.42 / 99.71 → 99.56 |
| matched / echo family (no noise) | 678 | 93.76 → 94.71 +0.95 | 94.48 → 95.64 +1.16 | 4.84 → 3.69 | 82.6 → 85.2 / 84.5 → 87.7 | 99.81 → 99.81 / 99.81 → 99.81 |
| **heldout / lang en** | 1538 | 84.45 → 82.97 **−1.48** | 84.47 → 82.35 **−2.12** | 16.34 → 16.77 | 59.7 → 57.8 / 60.5 → 56.6 | 99.30 → 98.83 / 99.06 → 98.75 |
| **heldout / snr 5** | 778 | 88.99 → 87.85 −1.13 | 89.72 → 87.12 **−2.60** | 10.83 → 9.71 | 74.4 → 70.5 / 76.3 → 70.5 | 98.23 → 98.55 / 98.23 → 97.91 |
| **matched / snr 0** | 711 | 84.97 → 83.54 −1.44 | 86.38 → 83.80 **−2.59** | 14.67 → 13.99 | 62.4 → 63.2 / 65.8 → 65.0 | 98.82 → 97.47 / 98.82 → 97.14 |
| heldout / mouse_click | 369 | 88.80 → 86.85 −1.94 | 88.80 → 87.00 −1.79 | 9.25 → 10.28 | 71.8 → 67.9 / 71.8 → 69.2 | 98.97 → 98.63 / 98.97 → 98.28 |
| heldout / wind | 381 | 87.17 → 85.17 −2.00 | 88.36 → 87.40 −0.96 | 13.14 → 12.81 | 68.4 → 61.8 / 72.4 → 67.1 | 98.69 → 99.34 / 98.36 → 99.34 |
| heldout / washing_machine | 380 | 83.34 → 83.34 +0.00 | 84.78 → 82.23 −2.55 | 16.45 → 16.78 | 62.8 → 62.8 / 65.4 → 60.3 | 97.35 → 97.35 / 97.68 → 97.35 |
| matched / opus:wechat | 585 | 86.62 → 86.20 −0.43 | 88.94 → 85.94 **−3.00** | 13.21 → 11.17 | 65.4 → 68.3 / 71.2 → 68.3 | 99.17 → 97.92 / 99.17 → 97.71 |
| matched / coffee | 649 | 90.54 → 89.61 −0.93 | 91.28 → 89.69 −1.58 | 8.97 → 8.69 | 73.7 → 74.6 / 76.3 → 75.4 | 99.63 → 98.69 / 99.44 → 98.50 |

Half-B splits of heldout bona and spoof (computed from `scores/*.txt` + `meta.csv`; PLC = `dsp_chain` has a `plc` stage; NS = an `ns` stage; VAD = a `vad` stage, which only offline clips carry: 629 offline, 0 online):

| half-B heldout slice | n bona / spoof | R_bona merge B2f / B4a / B4b / **B4c** | R_bona ep8 B2f / B4a / B4b / **B4c** | spoof misses merge B2f / B4a / B4b / **B4c** | spoof misses ep8 B2f / B4a / B4b / **B4c** |
|---|---|---|---|---|---|
| all | 391 / 1603 | 69.3 / 71.4 / 67.8 / **69.1** | 69.8 / 73.4 / 70.3 / **70.1** | 16 / 23 / 11 / 12 | 14 / 19 / 15 / 13 |
| online | 190 / 807 | 69.5 / 70.5 / 65.3 / 68.9 | 70.5 / 73.7 / 68.9 / 68.9 | 5 / 11 / 4 / 4 | 4 / 9 / 6 / 4 |
| online, PLC | 66 / 253 | 65.2 / 72.7 / 62.1 / 66.7 | 68.2 / 77.3 / 68.2 / 71.2 | 3 / 5 / 1 / 3 | 3 / 4 / 1 / 4 |
| online, no PLC | 124 / 554 | 71.8 / 69.4 / 66.9 / 70.2 | 71.8 / 71.8 / 69.4 / 67.7 | 2 / 6 / 3 / 1 | 1 / 5 / 5 / 0 |
| offline | 201 / 796 | 69.2 / 72.1 / 70.1 / 69.2 | 69.2 / 73.1 / 71.6 / 71.1 | 11 / 12 / 7 / 8 | 10 / 10 / 9 / 9 |
| offline, NS | 146 / 530 | 66.4 / 68.5 / 67.8 / 65.8 | 67.1 / 69.9 / 67.8 / 68.5 | 9 / 10 / 4 / 5 | 9 / 8 / 6 / 6 |
| **offline, VAD** | 73 / 240 | 63.0 / 71.2 / 63.0 / **64.4** | 65.8 / 71.2 / 63.0 / **69.9** | 7 / 7 / 1 / 4 | 7 / 6 / 3 / 5 |
| offline, VAD zero | 39 / 114 | 69.2 / 76.9 / 66.7 / 66.7 | 69.2 / 76.9 / 66.7 / 71.8 | 5 / 5 / 0 / 2 | 5 / 4 / 1 / 3 |
| offline, VAD cng | 34 / 126 | 55.9 / 64.7 / 58.8 / 61.8 | 61.8 / 64.7 / 58.8 / 67.6 | 2 / 2 / 1 / 2 | 2 / 2 / 2 / 2 |
| offline, no VAD | 128 / 556 | 72.7 / 72.7 / 74.2 / 71.9 | 71.1 / 74.2 / 76.6 / 71.9 | 4 / 5 / 6 / 4 | 3 / 4 / 6 / 4 |
| en | 129 / 631 | 59.7 / 56.6 / 60.5 / **58.1** | 58.9 / 58.9 / 62.0 / **57.4** | 3 / 10 / 7 / 5 | 5 / 9 / 5 / 4 |
| zh | 262 / 972 | 74.0 / 78.6 / 71.4 / 74.4 | 75.2 / 80.5 / 74.4 / 76.3 | 13 / 13 / 4 / 7 | 9 / 10 / 10 / 9 |
| codec gsm | 127 / 549 | 63.0 / 68.5 / 63.0 / 59.8 | 63.8 / 71.7 / 63.8 / 62.2 | 10 / 15 / 8 / 8 | 11 / 12 / 10 / 9 |
| codec g726 | 129 / 518 | 64.3 / 62.8 / 62.8 / 66.7 | 65.9 / 65.1 / 66.7 / 66.7 | 3 / 2 / 3 / 2 | 1 / 1 / 5 / 2 |
| codec g722 | 135 / 536 | 80.0 / 82.2 / 77.0 / 80.0 | 79.3 / 83.0 / 80.0 / 80.7 | 3 / 6 / 0 / 2 | 2 / 6 / 0 / 2 |

- On the target slice (half-B offline VAD bona), B4c's final rose 65.8 → 69.9, mostly in the CNG half (61.8 → 67.6). It still trails B4a (71.2), which never saw a gated frame.

### 2.5 Lever response (one factor per row)

B4a and B4b values are from the B4b report's §2.5. B4c values are from `compare_vs_B2f-*.txt` and `verdict.json`.

| run | change vs B2f | half_B merge WF1 (Δ) | final-epoch WF1 (Δ) | merge oracle (Δ) | final oracle (Δ) | merge noisy EER (Δ) | final EER (Δ) | merge clean (Δ) | final clean (Δ) | verdict (`verdict.json`) | mechanism (`b4_dsp_slices.py`, provisional) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B2f | reference (start of wave) | 91.98 | 92.14 | 92.59 | 92.82 | 10.09 | 9.66 | 97.60 | 97.69 | - | - |
| B4a | + `--aug_apm_p 0.3` | 92.55 (+0.57) | 93.15 (+1.02) | 92.98 (+0.39) | 93.26 (+0.43) | 9.53 (−0.55) | 9.16 (−0.51) | 97.72 (+0.12) | 98.05 (+0.35) | merge: above band, CI includes 0 (revisit); final: positive (promote) | NS: REFUTED final / DIRECTIONAL merge (B4a report) |
| B4b | + `--aug_opus_plc_p 0.3` | 92.22 (+0.24) | 92.26 (+0.12) | 92.47 (−0.13) | 92.50 (−0.32) | 10.66 (+0.57) | 9.68 (+0.02) | 97.70 (+0.09) | 97.71 (+0.02) | null (inside +-0.5) / revisit, both readouts | PLC_online: REFUTED both [placebos 1]; ANOMALOUS on VAD |
| **B4c** | **+ `--aug_dtx_p 0.3`** | **92.01 (+0.03)** | **92.34 (+0.20)** | 92.85 (+0.25) | 93.10 (+0.27) | 10.09 (−0.00) | 10.04 (+0.38) | 97.68 (+0.08) | 98.02 (+0.33) | merge: **null (inside +-0.5)** / revisit; final: **null (inside +-0.5)** / revisit; new_best false | dtx_gate: **NOT SHOWN** (criterion (ii) fails; VAD DIRECTIONAL final / REFUTED merge) [placebos 2]; **under-dosed** (ratio 0.032) → reads "inconclusive (under-dosed)"; no run anomaly |

### 2.6 Mechanism readout (pre-registered; `scripts/b4_dsp_slices.py --lever dtx_gate --placebo <B4a, B4b same readout>`, full sets, arm vs B2f same readout)

The statistic is DiD: the mean clipped-logit change on the treated bona slice minus that on the control bona slice. Negative means the treated slice moved toward bona. It is calibrated on the 9-run null panel (DiD* = DiD − panel mean, z = DiD* / panel SD).

| contrast | readout | arm recall T/C (B2f final) | DiD [CI] | DiD* / z | clip DiD | half-B DiD merge / final | placebo DiD B4a / B4b (arm rank) | criteria (a)-(e) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **VAD** (target; n 136/246; panel mu −0.57 sd 0.24; bar −1.05) | **final (primary)** | 69.9 / 72.4 (66.2 / 73.6) | **−0.64 [−1.27, −0.06]** | −0.08 / −0.32 | +6.7 | −0.13 / −0.36 | −0.22 / +0.36 (1/3) | n n n n Y | **DIRECTIONAL (provisional)** |
| VAD | merge | 66.9 / 72.0 | −0.34 [−0.83, +0.10] | +0.21 / +1.18 | +6.8 | −0.13 / −0.36 | −0.54 / +0.22 (2/3) | n n n n n | **REFUTED (provisional)** |
| VAD_zero (report; n 75/246) | final / merge | 73.3/72.4, 69.3/72.0 (68.0/73.6) | −0.88 [−1.64, −0.18] / −0.24 [−0.87, +0.33] | −0.16 / −0.39, +0.28 / +0.76 | | +0.41 / −0.43 | −0.49 / +0.38 (1/3); −0.56 / +0.47 (2/3) | | DIRECTIONAL / REFUTED |
| VAD_cng (report; n 61/246) | final / merge | 65.6/72.4, 63.9/72.0 (63.9/73.6) | −0.35 [−1.19, +0.45] / −0.46 [−1.17, +0.11] | +0.03 / +0.08, +0.13 / +0.65 | | −0.75 / −0.29 | +0.12 / +0.34 (1/3); −0.51 / −0.09 (2/3) | | REFUTED / REFUTED |
| NS (non-target; n 276/106) | final / merge | 70.3/74.5, 67.8/76.4 (69.2/75.5) | −0.07 [−0.77, +0.63] / −0.00 [−0.54, +0.54] | −0.40 / −1.46, −0.02 / −0.08 | +5.6 / +6.2 | +0.31 / +0.33 | +0.49 / +0.31 (1/3); −0.03 / −0.30 (3/3) | | DIRECTIONAL / INCONCLUSIVE |
| NS_afftdn_vs_anlmdn (secondary) | final / merge | 65.5/75.2, 62.6/73.0 (62.6/75.9) | −0.50 [−1.06, +0.04] / −0.44 [−0.88, −0.02] | +0.33 / +0.77, +0.34 / +0.84 | | −0.50 / −0.15 | −1.39 / −0.36 (2/3); −1.14 / −0.22 (2/3) | | REFUTED / REFUTED |
| PLC_online (non-target) | final / merge | 69.3/72.0, 66.7/74.4 (67.5/75.6) | −0.47 [−1.13, +0.22] / −0.27 [−0.75, +0.21] | +0.10 / +0.25, +0.38 / +1.08 | | −0.32 / −0.72 | −0.50 / −0.17 (2/3); −0.61 / −0.06 (2/3) | | REFUTED / REFUTED |

**Pinned non-speech tertiles: dtx_gate's criterion (ii).** Saturated noisy bona FA (p ≥ 0.99), low / mid / high tertile:

| | full set | half B |
|---|---|---|
| B2f merge | 63 / 73 / 109 | 43 / 47 / 46 (= 136) |
| B2f final | 66 / 76 / 107 | 44 / 47 / 42 (= 133) |
| **B4c merge** | 65 / 59 / 97 | 37 / 38 / 44 (= 119) |
| **B4c final** | 71 / 69 / 99 | 42 / 44 / 47 (= 133) |

| criterion (ii) drop_high − drop_low | B4c | need | null panel mu / sd / max, z | B4a (B4a report) | B4b (B4b report) |
|---|---|---|---|---|---|
| full set, final | **+13** | ≥ +5 and above both placebos | −10.2 / 5.8 / +0, z +4.03 | +10 | +2 |
| half B, final | **−7** | ≥ 0 | −13.4 / 3.5 / −5, z +1.86 | −8 | −11 |
| half B, merge | **−4** | ≥ 0 | −6.8 / 3.6 / −2, z +0.78 | −5 | −3 |
| (full set, merge; computed from the tertile counts, not a rule) | +14 | | | | |

- The full-set part passes (+13 ≥ +5) and beats both placebos, but only by 3 clips over B4a's +10.
- **The half-B part fails on both readouts.** So criterion (ii) fails.
- All three B4 arms sit above the null panel's maximum (+0) on the full set (+10 / +2 / +13). The full-set part is therefore not specific to gating.

**Heldout offline bona by realised VAD gated fraction** (full set; mean clipped logit, recall in brackets; computed from `scores/sim_heldout_v1.logits` + `meta.csv` `gated`):

| gated fraction | n | merge B2f / B4a / B4b / **B4c** | final B2f / B4a / B4b / **B4c** |
|---|---|---|---|
| no VAD stage | 246 | −3.78 (74.0) / −3.92 (72.0) / −3.94 (74.4) / −3.75 (72.0) | −4.02 (73.6) / −4.19 (75.6) / −4.21 (76.4) / −3.82 (72.4) |
| 1-20 % | 87 | −3.38 (69.0) / −4.17 (77.0) / −3.68 (71.3) / **−3.80 (72.4)** | −3.72 (73.6) / −4.05 (75.9) / −3.68 (71.3) / **−4.22 (75.9)** |
| 20-40 % | 27 | −2.91 (63.0) / −3.68 (70.4) / −2.90 (66.7) / **−3.54 (70.4)** | −2.87 (63.0) / −3.68 (70.4) / −3.19 (66.7) / **−3.71 (70.4)** |
| **≥ 40 %** | 22 | −0.25 (45.5) / −0.33 (40.9) / +1.29 (27.3) / **+0.26 (40.9)** | −0.13 (40.9) / −0.24 (40.9) / +1.12 (31.8) / **+0.12 (45.5)** |

- **B4c's VAD gain is confined to the lightly and moderately gated clips (1-40 %), and B4a, which never gated anything, moved those bins by as much.**
- The heavily gated bona (≥ 40 %, n = 22) moved spoof-ward under B4c on both readouts (logit −0.25 → +0.26 merge, −0.13 → +0.12 final). Training put 0.1 % of the processed clips in that bin.

Other blocks of the same output:
- **Lever verdict (dtx_gate):**
  - final: `NOT SHOWN (CRITERION II FAILS, VAD DIRECTIONAL) (provisional)  [placebos 2]  dose: under-dosed`;
  - merge: `NOT SHOWN (CRITERION II FAILS, VAD REFUTED) (provisional)  [placebos 2]  dose: under-dosed`.
- **VAD by fill (O2.3 report):**
  - final: `vad:zero DiD -0.88 DiD* -0.16 (near-in-distribution bar <= -0.42); vad:cng DiD -0.35 DiD* +0.03 (transfer needs < 0) -> no fill-specific gain`;
  - merge: zero −0.24 (DiD* +0.28) and cng −0.46 (DiD* +0.13), the same reading.
- **Outcome (O3.2):**
  - final: heldout 88.45 (Δ +0.18); the panel mean is 88.50, SD 1.35, and 5 of 9 panel runs clear ≥ +0.5;
  - merge: heldout 88.11 (Δ +0.20); the panel mean is 88.25, SD 1.39, and 6 of 9 clear ≥ +0.5;
  - `falsifier (1) outcome escape (heldout F1 delta >= +0.5 on merge_top5 AND final): False`.
- **Spoof-miss guards (Poisson 99 % bound of the B2f count), all held:**

  | guard | merge B2f → B4c | final B2f → B4c | bound merge / final |
  |---|---|---|---|
  | NS_offline | 18 → 12 | 18 → 15 | ≤ 29 / ≤ 29 |
  | VAD_offline | 15 → 7 | 13 → 11 | ≤ 25 / ≤ 22 |
  | PLC | 16 → 14 | 15 → 17 | ≤ 26 / ≤ 25 |
  | **halfB_noisy** | **24 → 31** | **23 → 33** | ≤ 36 / ≤ 35 |

- **Run anomaly (O3.1): none.** There is no |z| > 3 on a non-target primary contrast. Under-fit False, clean-F1 guard False.
- **dtx_gate decision lines:**
  - `falsifier (1) … does not fire (mechanism null False; WF1 delta final +0.20, merge +0.03; escape False); dose under-dosed (ratio 0.032)`;
  - `falsifier (2) … does not fire`;
  - `falsifier (3) [shared under-fit]: does not fire`;
  - `decision: no falsifier fires; decide on the lever verdict and the wave's WF1 CI rule`.
- **Placebo use for the other arms (the wave SUMMARY re-issues these):**
  - On the NS contrast, B4c (no NS) ranks 1 of 3 on the final (−0.07 vs B4a +0.49, B4b +0.31). B4a's NS mechanism is now beaten by two placebos on the final.
  - On NS_afftdn_vs_anlmdn, B4a stays rank 1 of 3 (−1.39 / −1.14 vs B4c −0.50 / −0.44 and B4b −0.36 / −0.22).
  - On PLC_online, B4b ranks 3 of 3 on the final (B4a −0.50, B4c −0.47, B4b −0.17).

## 3. Per-epoch behaviour, B4c vs B2f (same LR at each epoch end: 2.469e-6 … 5.000e-7 in both `train.log`s)

Sources: `logs/train.log` (both runs); `metrics.jsonl` (EMA weights; 2,000 clean clips + `sim_matched_mini` + `sim_heldout_mini`, half A). Δ = B4c − B2f mini WF1, computed from the unrounded `metrics.jsonl` values. `STATUS.md` shows the same deltas from rounded values, e.g. +0.71 at epoch 7.

| ep | B4c mini WF1 | B2f mini WF1 | Δ | B4c WF1@or | B4c EER noisy (B2f) | B4c R_bona cl/ma/he | B2f R_bona cl/ma/he | mini heldout F1 B4c / B2f | mini matched F1 B4c / B2f | train loss B4c / B2f | train acc B4c / B2f | dev loss B4c / B2f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 88.45 | 88.34 | +0.11 | 89.49 | 13.24 (13.40) | 93.3/80.4/**71.4** | 92.7/81.6/78.6 | 84.28 / 84.10 | 85.45 / 86.11 | 0.3595 / 0.3558 | 87.41 / 87.36 | 0.0836 / 0.0928 |
| 2 | 89.84 | 91.47 | **−1.63** | 90.78 | 11.39 (11.78) | 90.7/76.0/**64.3** | 91.6/77.1/76.0 | 84.55 / 87.78 | 89.01 / 90.20 | 0.2039 / 0.2075 | 97.19 / 97.13 | 0.0945 / 0.0982 |
| 3 | 91.64 | 91.90 | −0.26 | 92.32 | 11.08 (10.38) | 90.7/75.4/70.9 | 91.6/74.9/71.4 | 88.56 / 88.47 | 90.06 / 90.65 | 0.1535 / 0.1546 | 98.10 / 98.09 | 0.1074 / 0.1139 |
| 4 | 91.48 | 91.69 | −0.21 | 92.46 | 11.82 (10.64) | 92.1/75.4/67.3 | 91.3/75.4/71.4 | 87.57 / 88.32 | 90.22 / 90.38 | 0.1142 / 0.1108 | 98.62 / 98.62 | 0.1196 / 0.1242 |
| 5 | 91.90 | 92.49 | −0.59 | 92.99 | 11.97 (10.46) | 92.4/76.5/67.9 | 90.4/78.8/71.9 | 87.93 / 89.27 | 90.95 / 91.94 | 0.0863 / 0.0879 | 98.95 / 98.96 | 0.1268 / 0.1178 |
| 6 | 92.30 | 91.66 | +0.64 | 93.08 | 11.81 (10.63) | 92.7/78.2/70.9 | 91.3/74.9/70.4 | 88.71 / 88.36 | 91.25 / 90.17 | 0.0660 / 0.0725 | 99.23 / 99.18 | 0.1535 / 0.1321 |
| 7 | 92.81 | 92.10 | +0.70 | **93.50** | 10.77 (10.69) | 93.3/79.9/73.5 | 91.3/76.0/73.0 | **89.56** / 89.37 | 91.69 / 90.42 | 0.0529 / 0.0540 | 99.35 / 99.36 | **0.1589** / 0.1406 |
| 8 | **92.83** | 92.80 | +0.03 | 93.40 | **10.69** (10.94) | 93.8/81.6/70.9 | 91.3/80.4/73.5 | 88.56 / 89.41 | **92.61** / 92.38 | 0.0498 / 0.0501 | 99.44 / 99.44 | 0.1273 / 0.1442 |

Reading:
- **The best epoch is the last** (mini WF1 92.83, `best_by_wf1` and `best_by_noisy_eer` both epoch 8).
  - The curve has flattened: +0.02 from epoch 7 to 8, against B2f's +0.70.
  - The best WF1@oracle (93.50) and the best mini heldout F1 (89.56) are at epoch 7.
- **B4c ends level with B2f on the selection half** (+0.03 at epoch 8). It trailed at epochs 2-5 and led at 6-7.
- **Early heldout bona dip, as in B4b.**
  - Mini heldout bona recall at epochs 1-2 is 71.4 / 64.3, against B2f's 78.6 / 76.0. Epoch 2 is the −1.63 mini-WF1 deficit.
  - At epoch 8 it ends at 70.9, against B2f's 73.5. The mini heldout F1 at epoch 8 is below B2f's (88.56 vs 89.41), while the mini matched F1 is above (92.61 vs 92.38).
- Training curves track B2f closely: epoch-1 TrainAcc 87.41 vs 87.36 (inside the ±2 band); epoch-8 TrainLoss 0.0498 vs 0.0501 (under the 0.0752 under-fit line).
- **Dev loss.**
  - Epoch 1 → 8 it rose ×1.52 (0.083560 → 0.127304), against B2f's ×1.55 (0.092750 → 0.144168) (computed).
  - It peaked at epoch 7 (0.158925) and fell at epoch 8.
- **The final epoch beats the merge by 0.33 on half B** (92.34 vs 92.01). B2f's margin was 0.16, B4a's 0.60 and B4b's 0.04.
  - The final has the better clean (98.02 vs 97.68), matched F1 (91.36 vs 91.05) and heldout (88.45 vs 88.11).
  - The merge has the better matched EER (7.74 vs 8.37).
  - The weakest merged epochs by mini WF1 are 3 (91.64) and 5 (91.90). Epoch 5 also has a low mini heldout bona recall (67.9).
  - No merged epoch was evaluated alone on half B, so its half-B drag is not measured.

## 4. What went OK (evidence)

1. **Clean experiment, dosed in clip count exactly as designed.**
   - One factor, confirmed by the header lines and `B4 DTX gate selftest: PASS`.
   - Realised `wrtc_dtx` share 0.2983-0.3016 of clips per epoch, class-balanced (bona 0.2931-0.3042, spoof 0.2969-0.3031); zero-fill share 0.497-0.509.
   - `fail_wrtc_dtx` 0; no traceback, OOM, `failed (` or `[b4wave] ` line.
2. **No falsifier fires and no run anomaly** (`b4_dsp_slices.py` decision lines).
   - Clean F1 +0.08 / +0.33.
   - Spoof-miss guards: half-B noisy 31 / 33 (≤ 36 / 35), offline VAD 7 / 11 (≤ 25 / 22), offline NS 12 / 15 (≤ 29), PLC 14 / 17 (≤ 26 / 25).
   - Under-fit: ep1 TrainAcc 87.41 (≥ 85.36), ep8 TrainLoss 0.0498 (≤ 0.0752).
3. **Clean improved on the final, with a CI that excludes 0**: +0.33 [+0.05, +0.59], p(not better) 0.006.
   - Clean bona recall rose 92.91 → 94.27; paired clean bona 25 fixed / 6 broken.
   - It is the only per-set CI in B4c that excludes 0. B4a's final clean was +0.35 (B4b report §2.5).
4. **Ranking improved a little on both readouts**: WF1@oracle +0.25 / +0.27.
   - Heldout F1@oracle rose from 88.94 / 88.89 to 89.40 / 90.04, and matched EER on the merge from 8.15 to 7.74.
   - Bona recall rose on clean, matched and echo on both readouts (matched +1.79 / +2.56, echo +1.28 / +1.53).
5. **The VAD contrast is directional on the primary (final) readout**: DiD −0.64 [−1.27, −0.06], rank 1 of 3 against both placebos.
   - Offline VAD-gated heldout bona recall rose 66.2 → 69.9 (full set, final); the half-B offline VAD recall rose 65.8 → 69.9.
   - The VAD-gated heldout spoof misses fell 15 → 7 / 13 → 11.
6. **Fewer saturated noisy bona false alarms on the merge**: half-B p ≥ 0.99 fell 136 → 119 (section 5.4).
7. **Level shortcut slightly lower**: level_flip_bona 1.54 → 1.27 (merge) and 1.74 → 1.60 (final).
8. **The arm did its placebo job for the wave SUMMARY.**
   - It beats B4a on B4a's own NS contrast on the final (−0.07 vs +0.49).
   - It leaves B4a's afftdn-vs-anlmdn signal as the only NS-specific one (B4a −1.39 vs B4c −0.50 / B4b −0.36).
   - It shows that B4a's WF1 gain is not "any 30 % DSP augmentation": B4c is +0.03 / +0.20 against B4a's +0.57 / +1.02.

## 5. What went wrong (evidence)

### 5.1 The pre-registered mechanism did not appear, and the arm was under-dosed

- **Criterion (ii) fails.** The full set passes (+13 ≥ +5, above B4a's +10 and B4b's +2). Half B fails on both readouts: −7 (final) and −4 (merge), where ≥ 0 is needed.
  - All three B4 arms beat the null panel's maximum on the full set: B4a +10, B4b +2, B4c +13.
  - B4c clears B4a's placebo bar by only 3 clips.
- **The VAD contrast is directional on the final only.**
  - On the merge it is REFUTED: DiD −0.34, DiD* +0.21, rank 2 of 3 behind B4a's −0.54.
  - On the final it stays above the "shown" bar of −1.05, and its CI upper bound (−0.06) is not below the panel mean (−0.57).
- **No fill-specific effect.**
  - Zero fill: DiD* −0.16 (final), short of the near-in-distribution bar (≤ −0.42).
  - CNG fill: DiD −0.35, short of the transfer bar (< −0.38).
- **The heavily gated heldout bona moved spoof-ward**: in the ≥ 40 % bin (n = 22) the logit went −0.25 → +0.26 (merge) and −0.13 → +0.12 (final). B4a moved the 1-40 % bins bona-ward by as much as B4c did.
- **The non-speech probes did not move the way the design predicted.**
  - `sil_only` / `noise_only` bona are still 100 % spoof.
  - On the final, mean dP rose: sil_only +0.8985 → +0.9134, noise_only +0.8965 → +0.9130.
- **Cause, as measured: dose.**
  - dose_ratio 0.032 (rule: < 0.25 = under-dosed). 43.4 % of the "processed" clips had less than 1 % of their samples gated, and 1.0 % had 20 % or more, against 32.1 % of the heldout VAD clips.
  - The README names webrtcvad's internal overhang (about 100-330 ms) plus the drawn 40-200 ms hangover as the cause. Only pauses longer than about 0.2-0.5 s get gated.
  - Under the pre-registered dose rule, this null reads **"inconclusive (under-dosed)"**, not "refuted".

### 5.2 Spoof recall eroded, mostly on matched (a must-not-move item broke)

- Matched R_spoof fell 0.69 on both readouts (99.50 → 98.81, 99.44 → 98.75). The design's "R_spoof within 0.5 on every set" fails on matched.
- Half-B spoof misses:
  - matched 8 → 19 / 9 → 20;
  - echo 4 → 6 / 5 → 11;
  - clean 6 → 10 / 3 → 8.
- Paired per clip, on matched spoof: 1 fixed / 12 broken (merge) and 3 / 14 (final).
- The half-B noisy guard holds (31 / 33 against ≤ 36 / 35), but the final sits 2 clips under its bound. B2f had 24 / 23.
- **Where.**
  - Matched zh 8 → 16 / 9 → 14.
  - Noise families with continuous background: coffee 0 → 5 / 1 → 5, rain 2 → 6 / 3 → 6.
  - Codec: wechat 3 → 7 / 3 → 8.
  - Echo self_echo 3 → 5 / 4 → 10.
  - R_spoof rose in only 18 / 9 of 63 full-set slice rows, and the minimum slice R_spoof on the final is matched snr 0 at 97.14 (B2f 97.68 worst slice).
- **Reading (hypothesis, not tested here).** Gating both classes made "pause content" less informative. Some spoof clips whose pauses are filled with continuous noise lost a cue that B2f used.

### 5.3 Oracle-gap decomposition (global threshold vs set-specific)

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
| **B4c-top5** | 92.01 | **+2.95 / 0.950** | 92.58 | 92.85 | **0.84** | **0.56** | 0.27 |
| **B4c-ep8** | 92.34 | **+3.35 / 0.966** | 92.95 | 93.10 | **0.76** | **0.61** | 0.15 |

- **B4c has the largest oracle gap in the wave, and most of it is one global shift.**
  - Its best shared threshold is at p 0.950 / 0.966: too many bona sit between p 0.5 and p 0.95.
  - At that threshold B4c would score 92.58 / 92.95, against B2f's 92.39 / 92.58 at B2f's own best threshold.
  - The threshold is picked on half B, so this is a diagnostic only, not a usable readout.
- B4c's better ranking (oracle +0.25 / +0.27) is not converted at p = 0.5. B4a converts almost all of it (gap 0.10 on the final).

### 5.4 The saturated bona tail

Share of half_B bona with p_spoof ≥ 0.5 / ≥ 0.9 / ≥ 0.99 / ≥ 0.999 (computed from `scores/*.txt`):

| set (n bona) | B2f-top5 | **B4c-top5** | B2f-ep8 | **B4c-ep8** |
|---|---|---|---|---|
| clean (1,397) | 7.2 / 6.9 / 5.4 / 4.4 | 6.7 / 6.2 / 5.2 / 4.4 | 7.1 / 6.3 / 5.3 / 4.3 | 5.7 / 5.4 / 4.9 / 4.2 |
| matched (391) | 23.8 / 22.3 / 14.6 / 8.2 | 22.0 / 20.7 / 13.3 / 7.4 | 23.5 / 21.5 / 13.6 / 9.0 | 21.0 / 20.2 / 14.3 / 9.0 |
| heldout (391) | 30.7 / 28.6 / 20.2 / 9.2 | 30.9 / 29.2 / 17.1 / 9.7 | 30.2 / 28.4 / 20.5 / 9.5 | 29.9 / 28.1 / 19.7 / **10.5** |
| echo (391) | 16.6 / 15.6 / 10.5 / 4.9 | 15.3 / 14.3 / 10.5 / 5.6 | 15.3 / 14.8 / 10.0 / 5.4 | 13.8 / 13.0 / 11.0 / 6.1 |

Noisy bona (matched + heldout half B, n = 782):

| readout | FA (p ≥ 0.5) | saturated (p ≥ 0.99) | saturated share of FA | FA en / zh (en n = 258) | saturated en / zh |
|---|---|---|---|---|---|
| B2f-top5 | 213 | 136 | 64 % | 95 / 118 | 65 / 71 |
| B4a-top5 | 192 | 114 | 59 % | 95 / 97 | 61 / 53 |
| B4b-top5 | 210 | 115 | 55 % | 92 / 118 | 51 / 64 |
| **B4c-top5** | 207 | **119** | 57 % | 98 / 109 | 54 / 65 |
| B2f-ep8 | 210 | 133 | 63 % | 99 / 111 | 67 / 66 |
| B4a-ep8 | 177 | 123 | 69 % | 91 / 86 | 68 / 55 |
| B4b-ep8 | 202 | 136 | 67 % | 91 / 111 | 62 / 74 |
| **B4c-ep8** | 199 | **133** | 67 % | 96 / 103 | 59 / 74 |

- On the merge, B4c moved bona out of the saturated core (136 → 119) but removed only 6 false alarms (213 → 207).
- **On the final, the saturated core did not shrink at all (133 → 133),** against the design's "down from 136 / 133". The heldout p ≥ 0.999 share rose 9.5 → 10.5.
- English noisy bona false alarms rose on the merge (95 → 98) and fell by 3 on the final (99 → 96). Chinese ones fell by 9 / 8 (118 → 109, 111 → 103), against B4a's 21 / 25.

### 5.5 The worst slices for bona recall: English heldout got worse

Full sets, B4c-top5 (B4c-ep8 in brackets), `metrics.json` breakdowns:

| slice | R_bona B4c merge (ep8) | B2f merge (ep8) |
|---|---|---|
| **heldout lang en** | **57.8 (56.6)** | 59.7 (60.5) |
| heldout vacuum_cleaner | 61.4 (60.0) | 60.0 (57.1) |
| heldout snr 0 | 61.2 (62.6) | 63.3 (61.2) |
| heldout wind | 61.8 (67.1) | 68.4 (72.4) |
| heldout washing_machine | 62.8 (60.3) | 62.8 (65.4) |
| heldout gsm | 63.0 (65.4) | 65.9 (65.4) |
| matched snr 0 | 63.2 (65.0) | 62.4 (65.8) |
| matched lang en | 64.3 (66.3) | 63.6 (63.2) |
| heldout g726 | 67.5 (66.7) | 68.3 (70.3) |
| matched opus:qq | 67.7 (67.7) | 61.6 (66.7) |
| heldout mouse_click | 67.9 (69.2) | 71.8 (71.8) |
| matched opus:wechat | 68.3 (68.3) | 65.4 (71.2) |

- English is still the largest single block, and B4c made it worse.
  - Full-set heldout English F1 fell −1.48 / −2.12 and R_bona 59.7 → 57.8 / 60.5 → 56.6.
  - Half-B heldout English bona recall is 58.1 / 57.4, against Chinese 74.4 / 76.3.
- English accounts for 98 of 207 (merge) and 96 of 199 (final) noisy bona false alarms, from 258 of 782 bona.

### 5.6 Overfitting signs

- Train accuracy reached 99.44 % while dev loss rose 0.083560 → 0.127304 (×1.52; B2f ×1.55). The over-fitting slope is no steeper than B2f's.
- Dev loss peaked at epoch 7 (0.158925), the epoch with the best mini WF1@oracle. Both epochs 6 and 7 are in the merge.
- The mini-WF1 curve is flat from epoch 7 to 8 (+0.02). The final beats the merge by 0.33 on half B, so the earlier merged epochs (3, 5) cost a little.

## 6. Verdict on the pre-registered rule

From `exploop/B4/iter03/design.result.json`:

- **Falsifier:** "(1) Mechanism not supported: VAD DiD >= -0.57 AND drop_high - drop_low <= -10.2 on the full set, final, with half_B WF1 delta < +0.5. Under the README dose rule this reads 'inconclusive (under-dosed)', never 'refuted'. (2) Guard break: dev_online_clean F1 down > 0.5, half-B noisy spoof misses > 36 / 35, or full-set offline-VAD spoof misses > 25 / 22. The gating then carries class information, so drop. (3) Shared under-fit: ep1 TrainAcc < 85.36 or ep8 TrainLoss > 0.0752, together with heldout bona recall down. Drop. (4) Criterion (ii) passes (>= +5) but stays at or below B4a's placebo +10: not specific to gating, at most 'directional'. (5) A high fail_wrtc_dtx count, or wrtc_dtx counts far from ~0.3 of the clips: confounded (not dosed as designed). Verdict rule: positive = merge half_B WF1 delta > +0.5 with CI low > 0 and clean drop <= 0.5; |delta| <= 0.5 = null."
- **Predicted signature (abridged):** "(ii) Anchor: saturated-FA (p >= 0.99) drop_high - drop_low >= +5 clips … and above both placebos (B4a +10, B4b +2), plus >= 0 on half B for both readouts … (i') VAD contrast at least directional: DiD < -0.57 … and below B4a's -0.22 (final) / -0.54 (merge) and B4b's +0.36 / +0.22." The outcome and must-not-move items are listed in the table below.

`exploop/B4/iter03/verdict.json` (computed by the supervisor; quoted, not recomputed):

| readout | run | base | delta | CI | clean_drop | new_best | decision | verdict |
|---|---|---|---|---|---|---|---|---|
| merge (objective) | 92.01180869050039 | 91.97880641320396 | 0.03300227729643268 | [-0.4706137214361495, 0.5092148158168027] | -0.07518078006337703 | false | **revisit** | **null (inside +-0.5)** |
| final | 92.3387959325349 | 92.13685698540242 | 0.20193894713247573 | [-0.3857722202960731, 0.7757870738223756] | -0.3303503954475815 | false | **revisit** | **null (inside +-0.5)** |

Rules (`verdict.json`): `new_best_delta` 0.0, `keep_best` true (screen: B2f stays the reference); band 0.5; promote = ci_low > 0 and clean_drop ≤ 0.5; positive = delta > 0.5 with ci_low > 0 and clean_drop ≤ 0.5.

**Both readouts are null (inside ±0.5), with decision `revisit` and `new_best` false.**

Falsifiers:

| falsifier | evidence | fired? |
|---|---|---|
| (1) VAD DiD ≥ −0.57 AND (ii) ≤ −10.2 with WF1 Δ < +0.5 | VAD DiD −0.64 (final); (ii) +13 (full, final); WF1 Δ +0.03 / +0.20; script: `does not fire (mechanism null False …)` | **no**. The lever verdict is still NOT SHOWN, and under the dose rule (ratio 0.032) it reads **inconclusive (under-dosed)** |
| (2) guard break | clean +0.08 / +0.33; half-B noisy spoof misses 31 / 33 (≤ 36 / 35); offline-VAD 7 / 11 (≤ 25 / 22) | no (the final is 2 clips under its noisy bound) |
| (3) shared under-fit | ep1 TrainAcc 87.41 ≥ 85.36; ep8 TrainLoss 0.0498 ≤ 0.0752 | no |
| (4) (ii) ≥ +5 but ≤ B4a's +10 | +13 > +10 (by 3 clips) | no; but the half-B part of (ii) fails, so the anchor fails anyway |
| (5) high `fail_wrtc_dtx` or counts far from 0.3 | 0; 0.2983-0.3016 of clips per epoch | no for clip counts. The per-clip gating (1.0 % ≥ 20 % gated vs 32.1 % on heldout) is the pre-registered under-dose, not a confound |

The predicted signature, item by item:

| predicted | merge | final | appeared? |
|---|---|---|---|
| (ii) full set ≥ +5, above B4a +10 / B4b +2 | (+14, not a rule) | +13 | yes (by 3 clips over B4a) |
| (ii) half B ≥ 0 | −4 | −7 | **no** |
| VAD DiD < −0.57 | −0.34 | −0.64 | **final only** |
| VAD DiD below B4a (−0.54 / −0.22) and B4b (+0.22 / +0.36) | rank 2/3 | rank 1/3 | **final only** |
| transfer: CNG-fill DiD < −0.38 | −0.46 | −0.35 | merge only (merge VAD verdict REFUTED) |
| sil_only bona mean dP below +0.8929 / +0.8985 | +0.8825 | +0.9134 | **merge only** |
| noise_only bona below 99.47-100 % spoof | 100.00 | 100.00 | **no** |
| half-B saturated noisy bona FA down from 136 / 133 | 119 | 133 | **merge only** |
| offline VAD-gated heldout bona recall up from 66.2 (final) | 66.9 | 69.9 | yes (final) |
| half_B WF1 Δ 0 to +0.5 | +0.03 | +0.20 | yes |
| dev_online_clean F1 within 0.5 of 97.60 / 97.69 | 97.68 | 98.02 | yes (rose) |
| half-B noisy spoof misses ≤ 36 / 35 | 31 | 33 | yes |
| full-set offline-VAD spoof misses ≤ 25 / 22 | 7 | 11 | yes |
| probe_padded bona flip up ≤ 2 clips from 3.74 / 5.35 % | 4.81 (+2 clips) | 2.14 | yes (merge at the limit) |
| R_spoof within 0.5 on every set | matched −0.69 | matched −0.69 | **no** |
| ep1 TrainAcc within 2 of 87.36 | 87.41 | - | yes |
| wrtc_dtx ≈ 22,736 / epoch, ~50/50 zero / cng, `dtx_g*` populated, `fail_wrtc_dtx` ≈ 0 | 22,603-22,853, 0.497-0.509 zero, g20_40 / g40p sparse (1.0 % / 0.1 %), 0 | | yes (with the under-dose) |

**Summary.**
- Clean null: dosed as designed in clip count, no falsifier fired, WF1 inside the band on both readouts.
- The dtx_gate mechanism is NOT SHOWN: criterion (ii) fails on half B, and VAD is directional on the final only. By the pre-registered dose rule (ratio 0.032) this is **inconclusive (under-dosed)**, not refuted.
- Side effects:
  - clean F1 up on the final, with a CI that excludes 0;
  - better ranking (WF1@oracle +0.25 / +0.27), but a worse operating point (largest oracle gap of the wave);
  - matched spoof recall −0.69 (a must-not-move item broke);
  - English heldout worse.
- The verdicts are provisional until the wave SUMMARY re-runs `b4_dsp_slices.py --summary` with all arms.

## 7. Decision and next-experiment proposals

**Decision.** Per `verdict.json`, **B4c is not the new best**: `new_best false`, decision `revisit`, verdict `null (inside +-0.5)` on both readouts.
- **B2f stays the reference**: merge 91.98, final 92.14 on half_B; registry ids `B2f-loudnorm-screen` / `B2f-loudnorm-ep8`.
- Every proposal below is one factor (or one coupled group) against **B2f's flags**: `scripts/run_B4c.sh` without `--aug_dtx_p`, with the `b4wave.train` entry, `--batch_size 16` and 8 epochs.
- Iteration d is the wave's last slot (`max_iterations` 4). opus_plc was not positive, so per `wave.json` lever_notes it is "the analyst's best-supported B4 follow-up (B4 flags only)".
- The wave rule is unchanged, against B2f-top5 = 91.98:
  - positive = Δ merge > +0.5 with CI low > 0 and clean drop ≤ 0.5;
  - null = Δ within ±0.5;
  - negative = Δ below −0.5.

**Largest remaining error.** Confident noisy bona false alarms on heldout, worst for English, that no B4 arm has removed:
- 207 / 199 of 782 half-B noisy bona are false alarms (B4c merge / final), 119 / 133 of them at p ≥ 0.99;
- half-B heldout English bona recall 58.1 / 57.4, against Chinese 74.4 / 76.3;
- `sil_only` / `noise_only` bona are still 100 % spoof;
- B4c additionally has the wave's largest threshold-recoverable gap (0.56 / 0.61 of a 0.84 / 0.76 oracle gap), so its ranking gain is not converted at p = 0.5.

**Ranked proposals (configuration first, seed repeat last):**

1. **`--aug_apm_p 0.6` (apm_ns dose step; recommended for iteration d).**
   - Hypothesis: B4a's gain is specific to the WebRTC NS pass and dose-responsive.
     - B4a is the only B4 arm with a positive verdict: final +1.02 [+0.43, +1.60]; merge +0.57 [−0.01, +1.17].
     - Two equally dosed (30 %) augmentation arms on other DSP slots are now null: B4b +0.24 / +0.12 and B4c +0.03 / +0.20. So "any extra 30 % DSP augmentation" does not explain B4a.
     - B4a also has the smallest oracle gap (0.10 on the final) and the most negative afftdn-vs-anlmdn DiD of the three arms (−1.39 / −1.14).
   - Targets: noisy bona false alarms (half-B noisy FA B2f 213 / 210, B4a 192 / 177) and noisy EER (B2f 10.09 / 9.66, B4a 9.53 / 9.16).
   - Predicted signature:
     - merge Δ vs B2f > +0.5 with CI low > 0 (B4a's merge CI low was −0.01);
     - final Δ ≥ +0.5;
     - noisy EER ≤ 9.53 / 9.16;
     - half-B noisy bona FA ≤ 192 / 177;
     - clean F1 within 0.5;
     - full-set offline-NS spoof misses ≤ 29;
     - matched R_spoof within 0.5;
     - ep1 TrainAcc within 2 of 87.36.
   - Falsifier:
     - final Δ vs B2f < +0.5 → no dose response, so B4a's +1.02 was not dose-driven (or was seed noise); stop the apm_ns line;
     - half-B heldout spoof misses above B4a's 23 / 19, or matched R_spoof down > 0.5 → more NS trades spoof recall for bona recall; drop.
2. **`--aug_apm_p 0.3 --aug_apm_ns_levels 18 21` (one coupled group: strong WebRTC suppression only).**
   - Hypothesis: the only NS-specific signal left after two placebo arms is the harsh-suppressor contrast.
     - B4a's afftdn-vs-anlmdn DiD (−1.39 final / −1.14 merge) ranks 1 of 3 against B4b (−0.36 / −0.22) and B4c (−0.50 / −0.44).
     - The main NS contrast is beaten by the placebo B4c (−0.07 vs B4a's +0.49, final).
     - afftdn is the harsher heldout suppressor (B2f final recall T/C 62.6 / 75.9). Training only at kHigh / kVeryHigh (18 / 21 dB) may match its artefacts better than the uniform 6-21 dB draw.
   - Predicted signature:
     - afftdn-vs-anlmdn DiD ≤ −1.70 (the "strong" line), still rank 1 against the three arms;
     - heldout F1 Δ ≥ +0.5 on both readouts;
     - merge WF1 Δ ≥ +0.57 (B4a);
     - offline-NS spoof misses ≤ 29;
     - clean within 0.5.
   - Falsifier:
     - afftdn-vs-anlmdn DiD > −0.84 (the panel mean) on the final → the suppressor level is not the active ingredient; drop the level idea;
     - matched F1 down > 0.5 → losing raw-noise exposure costs more than it buys.
3. **`--aug_opus_plc_p 0.3 --aug_opus_loss 0 0` (one coupled group: the in-process libopus decoder path without packet loss).**
   - Hypothesis: B4b's matched gain (+0.86 merge, matched bona recall +2.30 / +1.53) comes from the aligned libopus decode, not from packet loss.
     - B4c, another 30 % DSP lever, did not reproduce it (matched −0.17 / +0.12), so the gain was not generic augmentation.
     - This is the plan's attribution control, run as a diagnostic because opus_plc was not positive.
   - Predicted signature:
     - matched F1 Δ ≥ +0.5 vs B2f on both readouts;
     - heldout F1 Δ ≥ 0;
     - WF1 Δ ≥ B4b's +0.24 / +0.12;
     - VAD z < +3 (B4b's +3.82 / +4.25 was ANOMALOUS).
   - Falsifier:
     - matched Δ < +0.5 on both readouts and WF1 within ±0.5 → the decoder path is worth nothing either; close the opus family;
     - VAD z > +3 again → the decoder path itself pushes gated pauses spoof-ward.
4. **`--aug_dtx_p 1.0` (the only dose knob of dtx_gate that flags allow).**
   - Hypothesis: B4c's directional VAD gain on 1-40 %-gated bona is exposure-limited.
     - p = 1.0 gives about 3.3× as many processed clips, so the ≥ 20 %-gated training clips go from about 230 per epoch to about 760.
     - The per-clip gating distribution is unchanged, so dose_ratio stays near 0.03 and the README rule still reads "under-dosed". A real dose variant (hangover U(0, 200) ms) is a code change for the owner / rdloop, not a loop flag.
     - The value of this run is a cap: if 3.3× exposure still does not move VAD or criterion (ii), the flag-level dtx line is closed.
   - Predicted signature:
     - VAD DiD ≤ −1.05 on the final (shown), rank 1 against B4a / B4b;
     - criterion (ii) ≥ 0 on half B for both readouts;
     - ≥ 40 %-gated heldout bona logit below B2f's −0.25 / −0.13;
     - sil_only bona dP below +0.8929 / +0.8985.
   - Falsifier:
     - half-B noisy spoof misses > 36 / 35, or matched R_spoof down > 0.5 again (B4c −0.69) → gating carries class information; drop the dtx line;
     - VAD DiD ≥ −0.64 (no better than B4c) → exposure is not the limit.
5. **Seed repeat of the reference: `--seed 1235` on B2f (last).**
   - Purpose: measure B2f's own half-B seed band, so that B4a's +0.57 / +1.02 and the three null arms can be read against same-configuration noise instead of the 9-run mixed panel (heldout F1 SD 1.35 / 1.39).
   - **This is outside wave B4's search space.** `wave.json` pins seed 1234, `--seed` is not in `allowed`, and the owner declined a B2f replicate arm (`plan_snapshot.md`). It needs an owner decision.
   - Predicted signature: |Δ| ≤ 0.5 vs B2f on both readouts.
   - Falsifier: a repeat that differs from B2f by ≥ +0.57 (merge) or ≥ +1.02 (final) puts B4a's gains inside seed noise.

**Recommended full run if the wave stopped now.**
- Per `verdict.json`, the reference is still B2f, and the plan leaves the wave's best to the owner at wrapup.
- B4c adds nothing to promote: null on both readouts, mechanism not shown (under-dosed), and matched spoof recall eroded.
- The strongest one-factor result in the wave is still B4a's final readout: positive, Δ +1.02 [+0.43, +1.60], clean +0.35, registry decision `promote`. Its merge CI includes 0, and its NS mechanism is beaten by two placebo arms.
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
