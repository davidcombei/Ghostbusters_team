# B2 wave summary (iterations 1-5) and the recommended full-length training strategy

> Written 2026-10-08 23:23 by the iteration-5 analysis agent, after B2e. Every number comes from the run directories under `SLS_setup/exp/` (`config.yaml`, `logs/train.log`, `metrics.jsonl`, `local_eval/*/metrics.json`, `probe_report.json`, `compare_vs_*.txt`, `local_eval/*/scores/*.txt`), `docs/experiment_log.csv` and the five per-run reports in this folder; nothing is typed from memory. half_B = the 1,994-clip half of each sim set never used for selection (391 bona + 1,603 spoof); clean = the full 7,465-clip `dev_online_clean`. "merge" = weight merge of the top-5 epochs by half-A mini WF1 (`local_eval/merge_top5`), "final" = the final-epoch EMA checkpoint alone (`local_eval/epoch_8` or `epoch_10`). Seed 1234 everywhere; one seed per configuration.

## 1. Starting point (from `20261007_222414_00_B0_vs_B1_postmortem.md`)

- **B0** (`B0_qwen17_rtcaug_s1234_epoch10_bs16_20261003201121`): Qwen3-ASR-1.7B + RTC augmentation, constant lr 3e-5, 10 epochs, no EMA. Merge of epochs 6, 5, 3, 2, 7: half_B WF1@0.5 **83.61**, oracle 84.71, noisy EER 18.01, clean F1 92.37. Per-epoch mini WF1 74.3-81.6 (range 7.3): epoch-to-epoch noise larger than any effect being measured.
- **B1** (`B1_qwen17_rtc_musan_lvl_stab_s1234_epoch30_bs16_20261006183011`): B0 + MUSAN + level aug + stability recipe (AdamW, 10x head LR, warmup-cosine over 30 epochs, grad clip, EMA 0.9995, label smoothing 0.05), early-stopped at epoch 13. Merge of epochs 1, 3, 2, 10, 4: **81.11** (-2.50 vs B0), oracle 84.26, clean 87.89, bona recall at 0.5 down 20-27 points on every set, oracle threshold ~0.69 on every set (one global logit offset toward spoof). Best epoch = the warmup epoch 1 (mini 84.75), then monotone decline.
- Misconfigurations found: **M1** `--label_smoothing 0.05` x `--ce_weights 0.1 0.9` (PyTorch weights the smoothing term by the class weights: per-sample optimum p_spoof 0.81 for spoof, 0.003 for bona -> shifted boundary, dev loss floor 0.20); **M2** 30-epoch cosine + early stop at 12 + warmup 3 % (the cosine never annealed, no final-epoch readout); **M3** peak lr 3e-5 / head 3e-4 too high for full fine-tuning (train acc 96.7 %, noisy EER 16.8 -> 23.2 over epochs); **M4** `--keep_topk_by_wf1 7` deleted the last epoch; **M5** three factors changed at once (MUSAN, level aug, recipe); **M6** top-5 merge mixing warmup weights with an overfit epoch.
- The loop: at most 5 iterations, one training flag per iteration against the current reference, 8-10 epochs, `/post-train-eval` + a per-run analysis report, promotion only above +1.5 half_B WF1 with clean F1 down by no more than 0.5.

## 2. The five iterations

GPU time = `config.yaml: created` to the last `logs/train.log` line (unshared RTX 4090, 30-33 min per epoch). Verdict per the promotion rule (> +1.5) and the pre-registered rules in the reports.

| it. | track (`SLS_setup/exp/...`) | single change vs previous reference | hypothesis (<= 12 words) | merge WF1@.5 (d) | final WF1@.5 (d) | oracle merge / final | noisy EER merge / final | clean F1 merge / final | noisy F1 merge / final | heldout F1 merge / final | GPU h | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | B0 `..._20261003201121` | - | - | 83.61 | - (best_by_wf1 ep 6: 80.94) | 84.71 / - | 18.01 / - | 92.37 / - | 79.86 / - | 77.54 / - | 5.03 (10 ep) | baseline |
| ref | B1 `..._stab_s1234_epoch30_bs16_20261006183011` | + MUSAN, level aug, recipe with label smoothing | stability recipe cuts variance and improves | 81.11 (-2.50) | - (stopped at 13) | 84.26 / - | 19.07 / - | 87.89 / - | 78.21 / - | 76.80 / - | 6.70 (13 ep) | dropped (M1-M6) |
| 1 | B2a `..._cos8_s1234_epoch8_bs16_20261007231952` | B1 minus label smoothing, 8-ep annealed cosine, warmup 5 %, no early stop | M1 + M2 explain the bias and the early peak | 84.95 (+1.34 vs B0) | 85.18 | 87.04 / 87.09 | 15.77 / 15.63 | 92.46 / 92.98 | 81.73 / 81.84 | 80.39 / 79.85 | 4.40 | promoted (separability: oracle +2.33, EER -2.24) |
| 2 | B2b `..._cos8_lr1e5_s1234_epoch8_bs16_20261008040736` | `--lr 1e-5` (head 1e-4) | peak-LR epochs install the spoof offset; lower LR keeps epoch-1 balance | 89.59 (**+4.64**) | 90.34 (**+5.16**) | 91.27 / 92.00 | 12.00 / 11.26 | 95.10 / 95.59 | 87.23 / 88.09 | 85.81 / 86.13 | 4.27 | promoted |
| 3 | B2c `..._cos8_lr5e6_s1234_epoch8_bs16_20261008085023` | `--lr 5e-6` (head 5e-5) | halving again buys more separability in 8 epochs | 91.73 (**+2.14**) | 92.26 (**+1.92**) | 92.60 / 92.74 | 10.11 / 10.06 | 96.82 / 97.42 | 89.55 / 90.05 | 88.98 / 89.46 | 4.10 | promoted (reference) |
| 4 | B2d `..._cos10_lr5e6_s1234_epoch10_bs16_20261008132750` | `--num_epochs 10` | curve still rising at epoch 8; longer schedule helps | 91.57 (-0.16) | 91.84 (-0.42) | 92.36 / 92.47 | 10.44 / 9.96 | 96.73 / 97.18 | 89.36 / 89.55 | 87.91 / 88.20 | 5.11 | null (dropped) |
| 5 | B2e `..._cos8_lr2.5e6_s1234_epoch8_bs16_20261008185925` | `--lr 2.5e-6 --lr_min 5e-7` (schedule halved) | LR optimum at or below 5e-6; ranking still improves | 92.34 (+0.61) | 92.53 (+0.26) | 92.88 / 92.97 | 9.68 / 9.27 | 97.67 / 98.09 | 90.05 / 90.14 | 88.50 / 88.39 | 4.08 | turned over (case (a); revisit) |

Total: 22.0 GPU h for B2a-e (27.0 with B0, 33.7 with B1). Wave result vs B0 merge: **+8.72 WF1@0.5, +8.17 oracle, -8.33 noisy EER, +5.30 clean F1, +10.19 noisy F1, +10.96 heldout F1** (B2e merge, `compare_vs_B0-top5.txt`).

Per-epoch mini WF1 (half A: 2,000 clean + `sim_matched_mini` + `sim_heldout_mini`; EMA weights for B1-B2e; `metrics.jsonl`):

| run | ep 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | best (ep) | range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 (lr 3e-5 const) | 74.27 | 76.93 | 79.15 | 74.45 | 80.41 | **81.55** | 77.57 | 75.20 | 76.83 | 76.92 | | | | 81.55 (6) | 7.28 |
| B1 (cos30, LS) | **84.75** | 82.98 | 83.61 | 82.72 | 80.72 | 81.38 | 81.69 | 81.67 | 81.74 | 82.83 | 81.00 | 81.97 | 81.51 | 84.75 (1) | 4.03 |
| B2a (3e-5, cos8) | 83.89 | 84.04 | 83.17 | 84.88 | 84.67 | 86.63 | 86.45 | **87.09** | | | | | | 87.09 (8) | 3.92 |
| B2b (1e-5) | 87.98 | 90.34 | 89.22 | 90.09 | 90.49 | 89.81 | **90.59** | 90.48 | | | | | | 90.59 (7) | 2.61 |
| B2c (5e-6) | 87.69 | 91.28 | 91.31 | 91.01 | 91.83 | 91.92 | **92.44** | 92.33 | | | | | | 92.44 (7) | 4.75 (ep 2-8: 1.43) |
| B2d (5e-6, 10 ep) | 87.72 | 90.41 | 91.39 | 90.59 | 90.85 | 91.26 | 90.88 | 91.99 | 91.80 | **92.03** | | | | 92.03 (10) | 4.31 (ep 2-10: 1.62) |
| B2e (2.5e-6) | 88.07 | 91.08 | 91.01 | 92.19 | 91.90 | 91.92 | 91.75 | **92.64** | | | | | | 92.64 (8) | 4.57 (ep 2-8: 1.63) |

## 3. What we learned (each with the number that proves it)

- **The loss misconfiguration was worth ~2.5 points of pure threshold bias.** B1 vs B0: WF1@0.5 -2.50 but oracle only -0.45; oracle threshold 0.69 on all four sets; clean R_bona 66.6 vs 87.4. B2a (same augmentation, no label smoothing) restored the clean oracle threshold to 0.995 only after epoch 2 and the dev loss to 0.11-0.21 (B1: 0.53-0.56). Never combine `--label_smoothing` with asymmetric `--ce_weights`.
- **Annealing matters; the final epoch is the model.** B1's cosine stopped at 64 % of peak with the warmup epoch as the best (84.75 then decline). Every annealed run has its best or second-best mini epoch in the last two (B2a ep 8, B2b ep 7, B2c ep 7, B2d ep 10, B2e ep 8) and the final EMA checkpoint beats the top-5 merge on half_B in **every iteration: +0.23 (B2a), +0.75 (B2b), +0.53 (B2c), +0.27 (B2d), +0.19 (B2e)**, with equal or better oracle (+0.05, +0.73, +0.14, +0.11, +0.09). The merge loses on the operating point (it admits early-plateau epochs: B2c's and B2d's merges contain epoch 3, thr_oracle heldout 0.44), not on the ranking. `--keep_epochs <last>` must protect the final epoch (B1 lost it; it ranked 2nd-3rd on half A in B2b / B2c).
- **The LR is the lever, and it saturates between 5e-6 and 2.5e-6.** 3e-5 -> 1e-5 -> 5e-6 -> 2.5e-6: merge **+4.64 / +2.14 / +0.61**, final **+5.16 / +1.92 / +0.26**, oracle +4.23 / +1.33 / +0.28 (merge) and +4.91 / +0.74 / +0.23 (final), noisy EER -3.77 / -1.89 / -0.43 (merge). Per ln(LR): 4.22 / 3.09 / 0.88 (merge), 4.70 / 2.77 / 0.38 (final). The first two halvings changed the epoch-1 state (train acc 83.9 -> 88.0 -> 87.9 %, mini WF1 83.89 -> 87.98 -> 87.69) and the plateau; the third changed neither (87.2 %, 88.07; train acc at epoch 8 99.42 vs 99.50 %). B2e's pre-registered case (a) fired: ep8 +0.26 (inside +-0.5), oracle +0.23 (inside +-0.3). 5e-6 and 2.5e-6 are equivalent on one seed; 2.5e-6 is ahead on 12 of 12 aggregates, 5e-6 on heldout (-1.08 / -0.48).
- **Epochs 8 vs 10 is null.** B2d: merge -0.16, final -0.42, oracle -0.24 / -0.27, EER +0.33 / -0.10, clean -0.09 / -0.24; mini WF1 at epochs 8-10 flat (91.99 / 91.80 / 92.03); dev loss 0.141-0.160 vs B2c's 0.112-0.120 at equal LR with equal train acc. Two more epochs at LR >= 1.9e-6 bought confidence, not separation. 8 epochs is the length.
- **The fit is complete at epoch 2 at every LR <= 1e-5**: train acc 96.3 / 97.0 / 96.8 / 97.1 % (B2b / B2c / B2d / B2e) and clean thr_oracle 0.96-0.99 from epoch 2 on. Everything after is +-1 of drift in the LR <= 1.6e-6 tail, which is where both 5e-6 runs' and B2e's best epochs are.
- **Cross-LR weight averaging fails; score averaging works.** The weight average of B2c-ep8 and B2e-ep8 scores 91.87 (parents 92.26 / 92.53; oracle 92.65 vs 92.74 / 92.97): it keeps the parents' ranking (logit correlation 0.97-0.99) but lands at a worse operating point (best global threshold 0.955, logit +3.06; R_bona at 0.5 below both parents on clean, matched and echo; oracle gap 0.78 of which 0.61 is one global shift). The score average (mean p_spoof) of the same two checkpoints scores **92.90** (clean 97.97, matched 92.01, heldout 89.45, echo 94.79), +0.37 over the better parent with no fitted parameter: the parents are complementary per set (B2c better on heldout, B2e on the other three). Same-LR seed averaging is untested; default to score averaging and test the weight average per pair.
- **The residual spoof bias and its ceiling.** Oracle thresholds 0.84-0.99 on every set since B2a; R_spoof 98.6-100, R_bona 71-94. The oracle gap shrank from 2.09 (B2a merge) to **0.44 (B2e final)**, of which a single global threshold recovers 0.17; two thirds of the noisy bona false alarms sit at p_spoof >= 0.99 (16-17 % of noisy bona in every readout since B2c, 9-11 % at >= 0.999) and no LR or schedule moved that core. Any calibration / `--ce_weights` / head-LR lever has a ceiling of 0.2-0.5 on the primary readout.
- **The noise band at this operating point, from the two near-replicate pairs** (same seed, same data order, one schedule factor apart): B2d vs B2c -0.42 final / -0.16 merge, per-set up to 1.26 (heldout) / 1.44 (echo); B2e vs B2c +0.26 / +0.61, per-set up to 1.08 (heldout) / 2.05 (echo). One schedule tweak moves the final readout by 0.3-0.4 and single sets by 1-2 points; a second seed will be at least this large. The formal +-1.5 rule (set from B0's 7.3-point epoch spread) was never challenged after B2c and is probably conservative on WF1 and about right per set.
- **What did NOT move, and remains for the next phase** (B0 merge -> B2e merge / final, full sets): English bona recall - heldout **61.2 -> 58.1 / 56.6**, matched 64.3 -> 63.6 / 64.0 (zh heldout 78.5 -> 79.9 / 81.7, zh matched 86.2 -> 87.6 / 89.0: the en-zh gap on half_B is now 18-22 points); unseen codecs - g726 R_bona 65.9 -> 67.9 / 68.7, gsm 68.7 -> 66.7 / 69.1; SNR 0 - matched **73.5 -> 65.0 / 64.1**, heldout 62.6 -> 63.9 / 65.3 (vs 86.6 / 77.8 at 20 dB; training SNR is 5-20 dB); `noise_only` / `sil_only` probes - bona-source clips 54.6 / 100 % spoof in B0 -> 100 / 100 % in every B2 run (mean dP +0.86 to +0.91). Set-level noisy bona recall is at B0's level (matched 79.54 -> 79.28, heldout 71.61 -> 70.59, echo 85.17 -> 85.42); **the wave's +8.7 WF1 is spoof recall (89.96 / 87.96 / 93.51 -> 98.88 / 99.06 / 99.75) and clean bona (87.40 -> 93.06)** traded against the prior, with EER halved (14.08 / 21.94 / 10.67 -> 8.27 / 11.09 / 4.92). Bona false alarms under unfamiliar degradation, English and low SNR are the next phase's targets (C1, C2, C4, D-wave in `docs/next_steps_plan.md`), not the optimiser's.

## 4. Recommended full-length training strategy

"Full-length" = seeds, not epochs: the screens already use the full 75,785-clip train set and 8 epochs is the tested length (B2d). Chosen configuration: **B2e's flags**, as a tie-break between two equivalent LRs (B2e report section 7): 2.5e-6 wins every aggregate on both readouts (WF1 +0.26 / +0.61, oracle +0.23 / +0.28, EER -0.79 / -0.43, clean +0.67 / +0.85, noisy +0.09 / +0.51), has the smaller spoof prior (optimal logit +1.66 vs +3.18; matched thr_oracle 0.84 vs 0.99), the best probes of the wave (level_flip_bona 1.87 vs 2.54, `probe_orig` false alarms 8.0 vs 11.2 %), the best single checkpoint (92.53 / mini 92.64) and the same cost; 5e-6 wins heldout (89.46 vs 88.39) inside the 1.3-point per-set band B2d measured. If the user prefers the literal pre-registration ("case (a) -> B2c's flags"), replace `--lr 2.5e-6 --lr_min 5e-7` by `--lr 5e-6 --lr_min 1e-6`; the expected difference is below 0.5 either way, and the seeds matter more than the LR.

**Exact command line** (= `B2e/config.yaml: command` with `--keep_epochs 8` added and seed / track per run; run from `SLS_setup/`):

```
main_train.py \
  --train_data_path /mnt/paml-research/RTCFake/data/wav/train \
  --train_protocol /mnt/paml-research/RTCFake/data/train_label.txt \
  --dev_data_path /mnt/paml-research/RTCFake/data/wav/dev \
  --dev_protocol ../data/dev_noisy_sim/dev_label_excl_halfB.txt \
  --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24 \
  --batch_size 16 --num_epochs 8 --num_workers 64 \
  --use_rtc_aug \
  --aug_noise_dirs ../data/augmentation_train/rnnoise ../data/augmentation_train/esc50 \
  --aug_rir_dirs ../data/augmentation_train/rirs \
  --musan --musan_dir ../data/augmentation/musan \
  --use_level_aug \
  --optim adamw --lr 2.5e-6 --head_lr_mult 10 --weight_decay 0.01 \
  --lr_scheduler warmup_cosine --warmup_frac 0.05 --lr_min 5e-7 \
  --grad_clip 1.0 --ema_decay 0.9995 \
  --local_eval_sets ../configs/local_eval_sets.yaml \
  --earlystop_metric wf1 --earlystop_epoch 99 --keep_topk_by_wf1 7 --keep_epochs 8 \
  --seed <SEED> --track F1_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s<SEED>
```

Every flag and what it does: `--ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24` backbone (full fine-tune, `freeze_ssl` off); `--batch_size 16 --num_epochs 8` (4,737 steps / epoch, 37,896 steps; `Warmup-cosine: 1895 warmup / 37896 steps, floor 5.0e-07` must appear in the stdout log); `--use_rtc_aug --aug_noise_dirs ... --aug_rir_dirs ...` RTC augmentation (`p_apply=0.8`, one stage per clip from coffee / footsteps / keyboard / musan / office / rain / echo / reverb, SNR 5-20 dB, then codec p=1.0 from qq / wechat / zoom / dingtalk / lark / voov / telegram); `--musan --musan_dir` MUSAN as one of the 8 stages; `--use_level_aug` level augmentation (p=0.8, target -38..-12 dBFS, agc 0.3 / comp 0.15 / limit 0.15 / none 0.40, clip 0.3); `--optim adamw --weight_decay 0.01`; `--lr 2.5e-6 --head_lr_mult 10` (head 2.5e-5); `--lr_scheduler warmup_cosine --warmup_frac 0.05 --lr_min 5e-7` (floor ratio 0.2, applied to both groups); `--grad_clip 1.0`; `--ema_decay 0.9995` (EMA weights are what is evaluated and saved); `--ce_weights` left at its default `0.1 0.9`; **no `--label_smoothing`** (default 0.0); `--local_eval_sets ../configs/local_eval_sets.yaml` per-epoch half-A mini evaluation; `--earlystop_metric wf1 --earlystop_epoch 99` no early stop; `--keep_topk_by_wf1 7` prunes at most one epoch; **`--keep_epochs 8`** protects the final epoch whatever its half-A rank; `--seed`, `--track` per run. Needs ffmpeg on PATH (container restarts remove it; the codec stage would silently no-op) and `--resume <run_dir>` after a restart with the same arguments.

**Seeds**: 2 seeds minimum (`2345`, `3456`), 3 if the 4090 is free for 12 h; B2e (`1234`) is the third seed at this configuration and already on disk. ~4 h 05 min per seed, sequential on one GPU (both runs on the 4090 and the 5090 in parallel if the latter is available; the GPU type was not balanced in this wave). Do not split seeds across LRs: one seed per LR is not readable.

**Checkpoint to submit** (user's choice on half_B; never computed on progress scores):
1. Primary readout per seed: the **final-epoch EMA checkpoint** (`ckpt/epoch_8_*.pth`), scored with `scripts/local_eval.py --names <the 17 report sets>` into `local_eval/epoch_8/`; compare with B2e's `local_eval/epoch_8` (92.53).
2. Combination: the **score average (mean p_spoof) of the final-epoch checkpoints of all seeds** - the measured cross-LR pair gave 92.90 vs 92.53 / 92.26 for the parents. Then test the **weight average** of the same checkpoints (`--merge_list`) and keep it only if it beats every seed on half_B; the cross-LR weight average lost 0.4-0.7, so it is not safe to assume.
3. Not the top-5 merge (five of five iterations below the final epoch); the last-3 merge (epochs 6, 7, 8) is a cheap alternative to check per seed (ceiling: final + 0.2).
4. Keep the `scores/*.txt`; a global re-threshold at ~0.84-0.91 is worth +0.17 to +0.30 if the competition metric allows a free threshold.

**Expected half_B numbers** (one seed, from B2c / B2d / B2e): final EMA WF1@0.5 **92.3-92.8** (92.26 / 91.84 / 92.53; band +-0.5), oracle 92.7-93.0, mean noisy EER 9.3-10.1, clean F1 97.4-98.1, noisy F1 89.6-90.2, heldout F1 88.2-89.5; merge 0.2-0.5 below the final epoch. Score average of 2-3 seeds: 92.9-93.2 expected (one measured pair: 92.90), bounded by the oracle gap (0.44) plus per-set complementarity. **One-seed caveat**: every number in this document is one seed; two near-replicates differ by 0.26-0.42 on the final readout and 1-2 points per set, and a true second seed has never been run. The first two full-run seeds are also the first measurement of the real band; a seed below 92.0 is inside what the near-replicates allow and is not a failure of the configuration.

**Proposed `scripts/run_full.sh` body** (text only; a copy of `scripts/run_B2e.sh` with a `SEED` variable, `--keep_epochs 8`, and the track renamed; usage `SEED=2345 bash scripts/run_full.sh`):

```bash
#!/usr/bin/env bash
# Full-length run of the B2 wave's recommended strategy = B2e's flags + --keep_epochs 8, one seed per launch.
# Evidence: docs/experiments-exploration/20261008_232331_B2_wave_summary_and_full_run_recommendation.md
#   lr 2.5e-6 / head 2.5e-5, warmup 5 % -> cosine to 5e-7 over 8 epochs, AdamW wd 0.01, grad clip 1,
#   EMA 0.9995, RTC + MUSAN + level aug, ce_weights 0.1/0.9, no label smoothing, no early stop,
#   top-7 ckpts kept with the final epoch protected. Primary readout: local_eval/epoch_8 on half_B.
#   SEED=2345 bash scripts/run_full.sh            # fresh run (SEED defaults to 2345)
#   SEED=2345 bash scripts/run_full.sh --resume exp/F1_qwen17_..._<ts>    # after a container restart
#   CUDA_VISIBLE_DEVICES=1 SEED=3456 bash scripts/run_full.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/root/.conda/envs/rtc-sdd/bin/python}"
SEED="${SEED:-2345}"
TRACK="F1_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s${SEED}"
RTCFAKE="/mnt/paml-research/RTCFake/data"
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found on PATH: the RTC codec augmentation would silently no-op. Reinstall it first." >&2; exit 1
fi
if [ ! -x "$PY" ]; then echo "python not found: $PY (set PY=... to override)" >&2; exit 1; fi
cd "$REPO/SLS_setup"
LOG="${TRACK}_$(date +%Y%m%d%H%M%S).log"
nohup "$PY" -u main_train.py \
    --train_data_path "$RTCFAKE/wav/train" \
    --train_protocol  "$RTCFAKE/train_label.txt" \
    --dev_data_path   "$RTCFAKE/wav/dev" \
    --dev_protocol    ../data/dev_noisy_sim/dev_label_excl_halfB.txt \
    --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24 \
    --batch_size 16 --num_epochs 8 --num_workers 64 \
    --use_rtc_aug \
    --aug_noise_dirs ../data/augmentation_train/rnnoise ../data/augmentation_train/esc50 \
    --aug_rir_dirs ../data/augmentation_train/rirs \
    --musan --musan_dir ../data/augmentation/musan \
    --use_level_aug \
    --optim adamw --lr 2.5e-6 --head_lr_mult 10 --weight_decay 0.01 \
    --lr_scheduler warmup_cosine --warmup_frac 0.05 --lr_min 5e-7 \
    --grad_clip 1.0 --ema_decay 0.9995 \
    --local_eval_sets ../configs/local_eval_sets.yaml \
    --earlystop_metric wf1 --earlystop_epoch 99 --keep_topk_by_wf1 7 --keep_epochs 8 \
    --seed "$SEED" --track "$TRACK" \
    "$@" > "$LOG" 2>&1 &
PID=$!
echo "Started $TRACK (pid $PID)"; echo "Log:     $REPO/SLS_setup/$LOG"
echo "Follow:  tail -f $REPO/SLS_setup/$LOG"; echo "Stop:    kill $PID"
```

Post-run per seed: `/post-train-eval SLS_setup/exp/<run> --k 5 --baseline SLS_setup/exp/B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234_epoch8_bs16_20261008185925/local_eval/merge_top5`, then the final-epoch checkpoint alone into `local_eval/epoch_8/` with `scripts/compare_evals.py` against B2e's `local_eval/epoch_8`; `/log-experiment` both readouts. Checks before launch: `bash -n`, `which ffmpeg`, `pgrep -af main_train.py` empty, `nvidia-smi` idle; after epoch 1: dev loss ~0.08 and mini WF1 ~88 in `logs/train.log`.

## 5. Follow-ups

**Zero-training** (each ~10 min on the 17 report sets; ceilings from the B2e report, section 5.2):

| id | what | expected / ceiling on half_B WF1@0.5 |
|---|---|---|
| Z-A | **Same-LR seed averaging** once the full-run seeds exist: score average first (measured on the cross-LR pair: 92.90 vs 92.53 / 92.26, +0.37), then the weight average per pair kept only if it beats both parents (cross-LR: -0.39 vs the worse parent) | +0.3 to +0.6 over the best seed; bounded by the oracle gap 0.44 plus per-set complementarity |
| Z-B | Last-3 merge (B2e: epochs 6, 7, 8) instead of top-5 | closes the 0.19-0.75 merge deficit; ceiling = final + 0.2 |
| Z-C | Global re-threshold on the `scores/*.txt` | +0.17 (B2e-ep8 at 0.84), +0.31 (B2e merge at 0.88), +0.30 (score average at 0.91) - only if the metric allows a free threshold |
| Z-D | Calibration fold-in (temperature / bias fitted on half A; E1.1) | <= oracle gap 0.44 (final) / 0.54 (merge), of which 0.2-0.3 global |
| Z-E | Multi-crop inference (Z1 of the plan) | unknown; targets the heldout transient slices that moved by 2-3 points between near-replicates (wind 84.06, clock_tick 91.63, door_wood_creaks 88.16, door_wood_knock 88.45 in B2e-ep8) |
| Z-F | WiSE-FT (alpha-interpolation with the pretrained backbone, Z2) | unknown; the LR series says moving less from the init helped until 5e-6, so alpha in 0.7-0.9 is the range to probe |

**Next-phase training levers** (from `docs/next_steps_plan.md`, sections 3-5), in priority order, one line each:

1. **C1 class-balanced sampler** (bona ~50 % per batch, language-balanced, CE 0.5 / 0.5): English bona recall never moved in the wave (heldout 61.2 -> 56.6, matched 64.3 -> 64.0) with 14,353 bona of 75,785 train clips and English the scarcer language; the en-zh gap on half_B is 18-22 points.
2. **C2 paired consistency** (clean-ish <-> RTC-aug view, offline <-> online twin; JS + embedding cosine): `noise_only` / `sil_only` bona-source clips are 100 % spoof with mean dP +0.9 in every B2 run - "unfamiliar processing => spoof" is intact after five optimiser iterations.
3. **C4 SNR range extension** (5-20 -> Beta-skewed over [-5, 25] dB): SNR 0 is the worst slice on both sim sets (R_bona 64.1 / 65.3 vs 86.6 / 77.8 at 20 dB) and was never trained.
4. **D-wave DSP augmentation v2** (WebRTC APM / RNNoise / DeepFilterNet / VAD-DTX / codec zoo with implementations disjoint from `sim_heldout_v1`): g726 / gsm bona recall 68.7 / 69.1 and heldout transients are the slices the LR did not reach; keep a stage only if heldout or echo improves.
5. **G2 LLRD 0.85 / freeze bottom-8 / LoRA** (feature preservation): the +7.3 WF1 of the LR series came from moving the backbone less; a per-layer decay is the next way to move it less without under-fitting (B2e's fit was complete at epoch 2 with train acc 99.42 at the end).
6. **C3 processing-status auxiliary head / DANN** after C2, and the **I5 pseudo-generator LOGO proxy** before any generator-side claim.

All of these need the paired-bootstrap CI (`I2`) and two seeds per arm to be readable: the wave's smallest decisions (B2d, B2e) were made at the 0.3-0.6 level, below the +-1.5 rule and at the edge of the near-replicate band.

## 6. Artefacts

- Reports (`docs/experiments-exploration/`): `20261007_222414_00_B0_vs_B1_postmortem.md`, `20261008_040609_B2a_qwen17_rtc_musan_lvl_cos8_s1234.md`, `20261008_084435_B2b_qwen17_rtc_musan_lvl_cos8_lr1e5_s1234.md`, `20261008_131754_B2c_qwen17_rtc_musan_lvl_cos8_lr5e6_s1234.md`, `20261008_185750_B2d_qwen17_rtc_musan_lvl_cos10_lr5e6_s1234.md`, `20261008_232331_B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234.md`, this file.
- Launchers: `scripts/run_B1.sh`, `scripts/run_B2a.sh`, `scripts/run_B2b.sh`, `scripts/run_B2c.sh`, `scripts/run_B2d.sh`, `scripts/run_B2e.sh` (each header records the hypothesis); the proposed `scripts/run_full.sh` body is in section 4 (not created).
- Registry `docs/experiment_log.csv` rows: `B0-merge-top5` (83.61, promote), `B1-stab-screen` (81.11, drop), `B2a-cos8-screen` (84.95, promote), `B2b-lr1e5-screen` (89.59, promote), `B2b-lr1e5-ep8` (90.34, revisit), `B2c-lr5e6-screen` (91.73, promote), `B2c-lr5e6-ep8` (92.26, promote), `B2d-cos10-screen` (91.57, drop), `B2d-cos10-ep10` (91.84, drop), `B2e-lr2.5e6-screen` (92.34, revisit), `B2e-lr2.5e6-ep8` (92.53, revisit).
- Run directories (`SLS_setup/exp/`): `B0_qwen17_rtcaug_s1234_epoch10_bs16_20261003201121`, `B1_qwen17_rtc_musan_lvl_stab_s1234_epoch30_bs16_20261006183011`, `B2a_qwen17_rtc_musan_lvl_cos8_s1234_epoch8_bs16_20261007231952`, `B2b_qwen17_rtc_musan_lvl_cos8_lr1e5_s1234_epoch8_bs16_20261008040736`, `B2c_qwen17_rtc_musan_lvl_cos8_lr5e6_s1234_epoch8_bs16_20261008085023`, `B2d_qwen17_rtc_musan_lvl_cos10_lr5e6_s1234_epoch10_bs16_20261008132750`, `B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234_epoch8_bs16_20261008185925`; each with `local_eval/merge_top5/` (metrics, probes, scores, `compare_vs_*.txt`) and `local_eval/epoch_8/` or `epoch_10/` (B2a-e); B2e also has `local_eval/avg_B2c8_B2e8/`.
- Submissions (top-5 merge, progress split): `SLS_setup/exp/B2a_.../model_merging_top5/submission.zip`, `.../B2b_.../model_merging_top5/submission.zip`, `.../B2c_.../model_merging_top5/submission.zip`, `.../B2d_.../model_merging_top5/submission.zip`, `.../B2e_.../model_merging_top5/submission.zip` (B0 / B1: no zip under `exp/`; B0's progress score 83.45 is in the registry notes). **No statistics were ever computed on progress scores; the user chooses what to submit on half_B evidence.** The final-epoch checkpoints (`ckpt/epoch_8_*.pth`) have no submission zip yet; packing one for B2e-ep8 or for the B2c-ep8 + B2e-ep8 score average is a 10-minute step (`/post-train-eval` steps 4-5 with a one-line merge list, or the score-level mean of two `scores.txt`).
