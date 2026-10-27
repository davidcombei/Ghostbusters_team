# B4a vs B4e: what the telephony-codec lever added and why it did not help

> Written 2026-10-27 after the B4e post-training evaluation. Every number is read from the two run directories below
> (`config.yaml`, `logs/train.log`, `local_eval/*/metrics.json`, `local_eval/*/probe_report.json`,
> `local_eval/*/ci_vs_*.txt`, `local_eval/*/compare_ci_vs_*.json`, `local_eval/merge_top5/compare_all_readouts.txt`,
> `ties_k50_lam1_merge.log`), from `scripts/run_B4e.sh` and from `SLS_setup/b4wave/{tel_codec.py,README.md}`.
> The B4a run is described in `docs/experiments-exploration/b4-experiments/20261018_194212_B4a_...md`.

Run directories (under `SLS_setup/exp/`):
- **B4a** `B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234_epoch8_bs16_20261018150134` (reference, best run so far)
- **B4e** `B4e_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_tel25_s1234_epoch8_bs16_20261026204847`

Readouts (same three for both runs):
- **top-5 average**: plain weight average of the 5 best epochs by half-A WF1 (`local_eval/merge_top5/`).
  B4a merges epochs 8, 6, 7, 4, 5; B4e merges epochs 7, 5, 8, 6, 3.
- **final epoch 8**: the last EMA checkpoint alone (`local_eval/epoch_8/`).
- **TIES k50 lam1**: TIES-merge of the same 5 epochs, top-50% trim, lambda 1 (B4a: `local_eval/ties_sweep/k50_lam1/`,
  B4e: `local_eval/ties_k50_lam1/`).

All numbers are on **half B** (the half never used for epoch selection) unless marked otherwise: 1,994 clips per sim set
(391 bona fide + 1,603 spoof) and the full 7,465-clip `dev_online_clean` (1,397 bona fide). CIs are paired bootstraps
(n = 2000) of B4e minus B4a on the same clips.

## TL;DR

- **B4e = B4a + one change**: on 25% of the training clips, the codec round-trip goes through an old phone-network
  codec (G.723.1 or AMR-NB, 8 kHz narrowband) **instead of** one of the 7 Opus video-call app profiles.
- **Why**: B4a misses many real voices on phone-quality audio (bona fide recall ~71% on `sim_heldout_v1`). The bet
  was that seeing phone codecs on both classes teaches the model that "sounds like a phone line" is not a spoof cue.
- **Result: negative on all three readouts.** Top-5 average 91.46 vs 92.55 (−1.09, CI [−1.73, −0.47]), final epoch
  −1.57, TIES −0.46. The pre-registered rule says negative and the falsifier is hit, so the lever is dropped.
- **Why it lost**: the Opus app profiles, which match the target channel, lost 25% of their exposure, and every one of
  them lost F1. The narrowband clips made training harder in the same 8-epoch budget. The model now flags more real
  wideband voices as fake. The intended gain was small and narrow: only GSM improved, G.722 got worse, held-out F1 flat.
- **TIES still works inside B4e**: +1.56 over B4e's own plain average (CI [+1.09, +2.05]), the third run where it holds.

## 1. The two runs

| | B4a | B4e |
|---|---|---|
| recipe | B2f + WebRTC noise suppression on 30% of clips (`--aug_apm_p 0.3`) | **B4a + `--aug_tel_p 0.25 --aug_tel_codecs g723_1 amr_nb`** |
| seed, LR schedule, epochs, batch, model | 1234, warmup-cosine 2.5e-6 → 5e-7, 8, 16, Qwen3-ASR-1.7B 24 layers | identical |
| launcher | `scripts/run_B4a.sh` | `scripts/run_B4e.sh` (commit `886c827` on `feature/b4wave-tel-codec`) |
| b4wave module | 0.1.0 | 0.2.0 (adds `tel_codec`; off by default) |
| wall clock | 15:01 → 19:14 on 2026-10-18 (4 h 13 min; ~31.5 min/epoch) | 20:49 → 02:42 on 2026-10-26/27 (5 h 54 min) |

The `config.yaml` `args` diff between the two runs is only `track`, `aug_tel_p: 0.25` and `aug_tel_codecs: [g723_1, amr_nb]`.
The acoustic augmentation draws (noise stage, Opus parameters, WebRTC NS) are the same as B4a's: the lever draws from
its own random stream, and the parent's Opus draws are made whether or not the clip goes to a telephony codec.

B4e's epochs 1-4 took 52-58 min each, epochs 5-8 took 33-34 min (B4a: ~31.5 min each). The smoke test showed no
slowdown, and the cause of the slow first half was not investigated. It does not change the results: both runs did the
same number of steps.

## 2. What B4e adds: the `tel_codec` lever

Every training clip in this recipe goes through a codec round-trip (`--aug_codec_p 1.0`). In B4a that is always one of
7 Opus profiles imitating DingTalk, Lark, QQ, Telegram, VooV, WeChat and Zoom: wideband audio (content up to ~8 kHz).
B4e's lever (`SLS_setup/b4wave/tel_codec.py`, ported from David's `utils/rtc_augment_expert.py::telephony_augm`) takes
25% of those codec calls and sends the clip through a telephony codec instead, using PyAV 15.1.0 in memory:

| codec | bitrate | sample rate | what it is |
|---|---|---|---|
| G.723.1 | 6.3 kbps | 8 kHz | old VoIP / video-conferencing speech codec (CELP family) |
| AMR-NB | 4.75-12.2 kbps (drawn per clip) | 8 kHz | the classic GSM / 3G mobile-phone voice codec (ACELP) |

Both are **narrowband**: they cut everything above 4 kHz and smooth the voice heavily at these bitrates. The start-up
self-test in the B4e log measures it: the share of spectral energy above 4.2 kHz goes from 0.25 on the probe clip to
0.020-0.027 after the round-trip.

The other telephony codecs in the same table are refused at start-up, because they are used to build the local proxy
sets: G.722 and G.726 are in `sim_heldout_v1`, and G.711 is in `sim_unseen_v1`.

Where it sits in the augmentation chain: noise / reverb stage → WebRTC NS (B4a's lever, 30%) → **codec slot: Opus app
profile (75%) or telephony codec (25%)**.

Training exposure per epoch (75,785 clips, epoch-8 counts from `logs/train.log`):

| codec calls per epoch | B4a | B4e |
|---|---|---|
| 7 Opus app profiles, total | 75,785 | 56,887 |
| each Opus app profile | 10,672-10,994 | 7,998-8,277 (−25%) |
| G.723.1 | 0 | 9,474 |
| AMR-NB | 0 | 9,424 |
| telephony share, bona fide / spoof | - | 25.2% / 24.9% |

The lever is class-balanced, so the codec itself does not carry label information.

## 3. Why we tried it (as pre-registered in `scripts/run_B4e.sh`)

- **Hypothesis**: B4a never sees narrowband phone audio, so it reads codec smoothing and the 4 kHz band edge on real
  speech as signs of synthesis. Training both classes through G.723.1 / AMR-NB should make those traces class-neutral,
  and that should transfer to the unseen G.722 / G.726 / GSM implementations in `sim_heldout_v1`.
- **Expected signature** (vs B4a, same readout): `sim_heldout_v1` bona fide recall up, driving held-out F1 up;
  `sim_matched_v1` F1 down by at most 0.5 (25% fewer Opus clips); `dev_online_clean` F1 within 0.5; spoof recall
  within 0.5 on every set; probe sets steady.
- **Decision rule** (top-5 average, half B WF1 vs B4a's 92.55): positive = Δ > +0.5 with CI low > 0 and clean F1 drop
  ≤ 0.5; within ±0.5 = null; below −0.5 = negative.
- **Falsifier**: held-out F1 Δ < +0.5 on both readouts while matched F1 drops > 0.5 → drop the lever.

## 4. Results

### 4.1 Headline (half B)

| readout | metric | B4a | B4e | Δ |
|---|---|---|---|---|
| top-5 average | WF1@0.5 | 92.55 | 91.46 | -1.09 |
| top-5 average | WF1@oracle | 92.98 | 92.56 | -0.42 |
| top-5 average | clean F1 | 97.72 | 96.78 | -0.94 |
| top-5 average | noisy F1 | 90.34 | 89.18 | -1.15 |
| top-5 average | noisy EER | 9.53 | 9.95 | +0.41 |
| final epoch 8 | WF1@0.5 | 93.15 | 91.58 | -1.57 |
| final epoch 8 | WF1@oracle | 93.26 | 92.73 | -0.53 |
| final epoch 8 | clean F1 | 98.05 | 96.71 | -1.34 |
| final epoch 8 | noisy F1 | 91.06 | 89.38 | -1.67 |
| final epoch 8 | noisy EER | 9.16 | 9.30 | +0.14 |
| TIES k50 lam1 | WF1@0.5 | 93.48 | 93.02 | -0.46 |
| TIES k50 lam1 | WF1@oracle | 93.70 | 93.24 | -0.46 |
| TIES k50 lam1 | clean F1 | 98.26 | 97.71 | -0.56 |
| TIES k50 lam1 | noisy F1 | 91.43 | 91.01 | -0.42 |
| TIES k50 lam1 | noisy EER | 9.08 | 9.43 | +0.35 |

### 4.2 Paired-bootstrap CIs (B4e − B4a)

| readout | metric | B4a | B4e | Δ | 95% CI |
|---|---|---|---|---|---|
| top-5 average | WF1 | 92.55 | 91.46 | -1.09 | [-1.73, -0.47] |
| top-5 average | dev_online_clean F1 | 97.72 | 96.78 | -0.94 | [-1.31, -0.60] |
| top-5 average | sim_matched_v1 F1 | 92.46 | 90.21 | -2.25 | [-3.55, -1.07] |
| top-5 average | sim_heldout_v1 F1 | 88.21 | 88.15 | -0.06 | [-1.33, +1.21] |
| top-5 average | sim_echo_v1 F1 | 94.79 | 93.65 | -1.14 | [-2.08, -0.30] |
| final epoch 8 | WF1 | 93.15 | 91.58 | -1.57 | [-2.18, -0.98] |
| final epoch 8 | dev_online_clean F1 | 98.05 | 96.71 | -1.34 | [-1.73, -0.99] |
| final epoch 8 | sim_matched_v1 F1 | 92.80 | 90.03 | -2.78 | [-4.09, -1.59] |
| final epoch 8 | sim_heldout_v1 F1 | 89.31 | 88.74 | -0.57 | [-1.86, +0.69] |
| final epoch 8 | sim_echo_v1 F1 | 94.63 | 93.25 | -1.38 | [-2.38, -0.41] |
| TIES k50 lam1 | WF1 | 93.48 | 93.02 | -0.46 | [-0.98, +0.08] |
| TIES k50 lam1 | dev_online_clean F1 | 98.26 | 97.71 | -0.56 | [-0.87, -0.27] |
| TIES k50 lam1 | sim_matched_v1 F1 | 93.71 | 92.17 | -1.54 | [-2.55, -0.57] |
| TIES k50 lam1 | sim_heldout_v1 F1 | 89.15 | 89.86 | +0.70 | [-0.48, +1.91] |
| TIES k50 lam1 | sim_echo_v1 F1 | 95.74 | 94.61 | -1.13 | [-2.10, -0.22] |

Every clean, matched and echo CI excludes zero on the losing side. No held-out CI excludes zero.

### 4.3 Per set (half B)

**dev_online_clean**

| metric | B4a top-5 | B4e top-5 | Δ | B4a ep8 | B4e ep8 | Δ | B4a TIES | B4e TIES | Δ |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 97.72 | 96.78 | -0.94 | 98.05 | 96.71 | -1.34 | 98.26 | 97.71 | -0.56 |
| eer | 2.11 | 2.53 | +0.42 | 2.21 | 2.42 | +0.21 | 1.95 | 2.22 | +0.28 |
| recall_spoof | 99.85 | 99.90 | +0.05 | 99.88 | 99.92 | +0.03 | 99.69 | 99.77 | +0.08 |
| recall_bona | 93.41 | 90.34 | -3.08 | 94.27 | 90.05 | -4.22 | 95.78 | 93.70 | -2.08 |

**sim_matched_v1** (the 7 Opus app profiles B4e trained on less)

| metric | B4a top-5 | B4e top-5 | Δ | B4a ep8 | B4e ep8 | Δ | B4a TIES | B4e TIES | Δ |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 92.46 | 90.21 | -2.25 | 92.80 | 90.03 | -2.78 | 93.71 | 92.17 | -1.54 |
| eer | 7.42 | 8.40 | +0.98 | 7.54 | 7.58 | +0.03 | 7.48 | 7.89 | +0.41 |
| recall_spoof | 99.50 | 98.75 | -0.75 | 99.25 | 99.00 | -0.25 | 98.88 | 98.50 | -0.37 |
| recall_bona | 79.54 | 75.96 | -3.58 | 81.33 | 74.68 | -6.65 | 85.17 | 82.10 | -3.07 |

**sim_heldout_v1** (G.722 / G.726 / GSM plus held-out DSP stages: the target of the change)

| metric | B4a top-5 | B4e top-5 | Δ | B4a ep8 | B4e ep8 | Δ | B4a TIES | B4e TIES | Δ |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 88.21 | 88.15 | -0.06 | 89.31 | 88.74 | -0.57 | 89.15 | 89.86 | +0.70 |
| eer | 11.65 | 11.49 | -0.16 | 10.77 | 11.02 | +0.25 | 10.67 | 10.96 | +0.28 |
| recall_spoof | 98.57 | 98.00 | -0.56 | 98.81 | 98.25 | -0.56 | 97.94 | 97.57 | -0.37 |
| recall_bona | 71.36 | 72.89 | +1.53 | 73.40 | 73.66 | +0.26 | 75.70 | 78.77 | +3.07 |

**sim_echo_v1**

| metric | B4a top-5 | B4e top-5 | Δ | B4a ep8 | B4e ep8 | Δ | B4a TIES | B4e TIES | Δ |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 94.79 | 93.65 | -1.14 | 94.63 | 93.25 | -1.38 | 95.74 | 94.61 | -1.13 |
| eer | 4.86 | 6.66 | +1.80 | 4.61 | 6.38 | +1.77 | 4.26 | 5.68 | +1.42 |
| recall_spoof | 99.75 | 99.56 | -0.19 | 99.63 | 99.69 | +0.06 | 99.31 | 99.25 | -0.06 |
| recall_bona | 85.17 | 82.61 | -2.56 | 85.17 | 81.07 | -4.09 | 89.51 | 86.45 | -3.07 |

### 4.4 Per codec (full set, from the `breakdowns` block of `metrics.json`)

These include half A, whose mini subsets were used to pick the top-5 epochs, and each codec has only ~90-260 bona fide
clips. Read them as diagnostics, not as decision numbers.

**sim_heldout_v1 by codec** (bona fide clips: G.722 258, G.726 246, GSM 246)

| codec | n | B4a top-5 F1 | B4e top-5 F1 | Δ F1 | Δ bona rec | B4a ep8 F1 | B4e ep8 F1 | Δ F1 | Δ bona rec | B4a TIES F1 | B4e TIES F1 | Δ F1 | Δ bona rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g722 (wideband, 16 kHz ADPCM) | 1380 | 92.87 | 91.29 | -1.59 | -3.49 | 93.41 | 91.40 | -2.02 | -5.04 | 94.29 | 92.59 | -1.70 | -2.71 |
| g726 (narrowband ADPCM) | 1274 | 86.10 | 86.15 | +0.04 | +1.22 | 87.57 | 87.20 | -0.38 | +0.81 | 88.61 | 87.39 | -1.23 | -1.63 |
| gsm (narrowband, LPC-based) | 1346 | 85.54 | 86.65 | +1.11 | +7.72 | 87.41 | 86.96 | -0.45 | +2.85 | 85.80 | 85.91 | +0.11 | +5.69 |

**sim_matched_v1 by codec**

| codec | n | B4a top-5 F1 | B4e top-5 F1 | Δ F1 | Δ bona rec | B4a ep8 F1 | B4e ep8 F1 | Δ F1 | Δ bona rec | B4a TIES F1 | B4e TIES F1 | Δ F1 | Δ bona rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opus:dingtalk | 589 | 94.27 | 92.98 | -1.29 | -0.82 | 94.92 | 94.31 | -0.61 | -2.46 | 95.07 | 93.40 | -1.68 | -4.92 |
| opus:lark | 608 | 94.71 | 92.84 | -1.87 | -1.68 | 94.41 | 92.48 | -1.93 | -2.52 | 95.07 | 93.73 | -1.35 | -1.68 |
| opus:qq | 572 | 88.08 | 85.77 | -2.31 | -5.05 | 87.68 | 85.49 | -2.19 | -4.04 | 89.18 | 88.24 | -0.94 | -4.04 |
| opus:telegram | 552 | 92.84 | 90.84 | -2.00 | -5.41 | 93.89 | 90.77 | -3.12 | -10.81 | 93.67 | 93.53 | -0.14 | -2.70 |
| opus:voov | 510 | 95.06 | 93.85 | -1.20 | -4.49 | 94.30 | 92.41 | -1.89 | -4.49 | 95.06 | 93.59 | -1.47 | -3.37 |
| opus:wechat | 585 | 88.39 | 87.69 | -0.70 | +0.96 | 88.42 | 88.12 | -0.30 | -2.88 | 89.58 | 88.19 | -1.39 | -0.96 |
| opus:zoom | 584 | 94.44 | 91.69 | -2.75 | -3.77 | 94.48 | 91.90 | -2.59 | -5.66 | 95.17 | 93.47 | -1.71 | -0.94 |

**sim_echo_v1 by codec**

| codec | n | B4a top-5 F1 | B4e top-5 F1 | Δ F1 | Δ bona rec | B4a ep8 F1 | B4e ep8 F1 | Δ F1 | Δ bona rec | B4a TIES F1 | B4e TIES F1 | Δ F1 | Δ bona rec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opus:dingtalk | 554 | 92.34 | 88.70 | -3.64 | -8.33 | 92.70 | 88.59 | -4.11 | -10.42 | 93.89 | 90.63 | -3.26 | -9.38 |
| opus:lark | 564 | 94.80 | 93.86 | -0.95 | -1.01 | 95.83 | 94.10 | -1.73 | -5.05 | 95.94 | 94.35 | -1.59 | -3.03 |
| opus:qq | 603 | 93.93 | 92.22 | -1.71 | -3.33 | 93.67 | 92.47 | -1.21 | -3.33 | 94.34 | 92.72 | -1.62 | -2.50 |
| opus:telegram | 597 | 94.59 | 94.32 | -0.28 | +0.00 | 94.87 | 93.34 | -1.53 | -2.80 | 96.48 | 95.58 | -0.90 | -1.87 |
| opus:voov | 601 | 94.34 | 94.09 | -0.25 | +0.00 | 93.84 | 93.50 | -0.33 | -1.61 | 93.92 | 94.16 | +0.25 | +0.00 |
| opus:wechat | 536 | 93.90 | 92.84 | -1.05 | -2.94 | 93.55 | 93.14 | -0.41 | -1.96 | 94.97 | 93.71 | -1.26 | -1.96 |
| opus:zoom | 545 | 96.91 | 96.91 | +0.00 | +0.00 | 97.85 | 96.27 | -1.58 | -3.92 | 97.26 | 97.55 | +0.30 | +0.00 |

### 4.5 Probes (level shortcut)

`level_flip_bona` is the share of real clips whose verdict flips when only the volume changes (lower is better).

| readout | B4a | B4e |
|---|---|---|
| top-5 average | 0.87% | 2.67% |
| final epoch 8 | 1.40% | 2.61% |
| TIES k50 lam1 | 0.87% | 1.80% |

`sil_only` and `noise_only` clips are predicted 100% spoof by every readout of both runs, the same as before. B4e did not
change that.

### 4.6 Training curves (half-A mini sets, from `logs/train.log`)

| epoch | B4a train loss | B4e train loss | B4a dev loss | B4e dev loss | B4a WF1 | B4e WF1 | B4a clean F1 | B4e clean F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.3609 | 0.3937 | 0.0755 | 0.0755 | 87.71 | 85.67 | 96.92 | 96.01 |
| 2 | 0.2196 | 0.2577 | 0.0819 | 0.0816 | 90.75 | 90.42 | 96.85 | 96.93 |
| 3 | 0.1584 | 0.2155 | 0.0860 | 0.0872 | 92.01 | 90.65 | 97.54 | 96.57 |
| 4 | 0.1225 | 0.1666 | 0.1040 | 0.1287 | 92.16 | 90.33 | 97.00 | 95.96 |
| 5 | 0.1013 | 0.1393 | 0.1051 | 0.1223 | 92.09 | 91.88 | 97.09 | 96.61 |
| 6 | 0.0746 | 0.1204 | 0.1037 | 0.1225 | 92.68 | 91.59 | 97.45 | 96.62 |
| 7 | 0.0592 | 0.0958 | 0.1204 | 0.1331 | 92.58 | 92.41 | 97.54 | 96.80 |
| 8 | 0.0562 | 0.0882 | 0.1166 | 0.1299 | 93.28 | 91.86 | 97.99 | 96.70 |

B4e's training loss is higher at every epoch, and its validation WF1 is lower at every epoch. Its clean F1 is lower at
every epoch except epoch 2.

### 4.7 TIES inside B4e

TIES k50 lam1 vs B4e's own top-5 plain average: WF1 91.46 → 93.02 (+1.56, CI [+1.09, +2.05]). All four report sets
improved: clean +0.93, matched +1.96, held-out +1.71, echo +0.96. The merge had 0.25% sign conflicts and merged 66% of
positions (`ties_k50_lam1_merge.log`). This is the third run where TIES k50 beats the plain average, after B2f (+1.18)
and B4a (+0.93). TIES narrows the B4a/B4e gap but does not close it.

## 5. Why B4e is not better

**1. Less training on the channel that matters.** Each Opus app profile went from ~10,800 to ~8,100 clips per epoch.
Those profiles are what `sim_matched_v1` is built from, and they imitate the real-time-communication apps the challenge
is about. Every one of the 7 apps lost F1 on all three readouts (section 4.4). `sim_matched_v1` F1 dropped
2.25 / 2.78 / 1.54, against a pre-registered allowance of 0.5. Echo, also Opus-coded, dropped too (−1.1 to −1.4).

**2. A harder task in the same budget.** On a quarter of the clips the model has to tell real from fake without the
4-8 kHz band, where much of the synthesis evidence in wideband audio lives. With the same 8 epochs and LR schedule,
training loss stayed higher at every epoch (0.088 vs 0.056 at epoch 8). Validation WF1 was lower at every epoch
(section 4.6). This is a run that sits lower throughout, not a bad pick of epochs.

**3. More real wideband voices called fake; the effect splits by bandwidth.** Spoof recall barely moved anywhere.
Bona fide recall fell on all wideband audio: clean −3.08 / −4.22 / −2.08, matched −3.58 / −6.65 / −3.07, echo −2.56 /
−4.09 / −3.07. Inside `sim_heldout_v1` the codec split is the clearest picture (section 4.4):
- **GSM** (narrowband, LPC-based like the trained codecs): bona fide recall **up** (+7.72 / +2.85 / +5.69).
- **G.726** (narrowband, waveform ADPCM): about flat.
- **G.722** (wideband): **down** (F1 −1.59 / −2.02 / −1.70, bona fide recall −3.49 / −5.04 / −2.71).

So the lever did what it was designed for on narrowband audio close to the training codecs, and cost us on everything
wideband. Part of the overall loss is a threshold shift: with the best possible threshold, the top-5 gap shrinks from
−1.09 to −0.42. Part of it is real loss of separation: EER rose on matched (+0.98 top-5) and echo (+1.42 to +1.80). One
possible reading, **not tested**: with a quarter of the training audio band-limited, the model learned to lean more on
cues that survive narrowband coding, and those cues are less reliable on wideband real speech.

**4. More reliance on loudness.** `level_flip_bona` roughly tripled for the plain average (0.87% → 2.67%) and doubled
under TIES (0.87% → 1.80%). Verdicts on real speech now depend more on volume, which is a shortcut, not real evidence.

**5. The intended gain was too small and too narrow to pay for this.** Held-out F1 was −0.06 (top-5), −0.57 (epoch 8)
and +0.70 (TIES), and every CI includes zero. The bona fide recall gain (+1.53 / +0.26 / +3.07) comes mostly from GSM,
and G.722 got worse. The launcher also flagged that `sim_heldout_v1` contains plain `bandlimit` / `resample8k` stages,
so even part of the GSM-side gain may be familiarity with narrowband audio rather than learned codec robustness.

**Net**: we gave up a quarter of the training on the right channel for a small, uncertain gain on a stand-in for phone
audio, and that gain is confined to narrowband codecs similar to the training ones.

## 6. Verdict against the pre-registered rule

| criterion | threshold | B4e | met? |
|---|---|---|---|
| top-5 Δ WF1 vs B4a | > +0.5, CI low > 0 | −1.09, CI [−1.73, −0.47] | no: **negative** (below −0.5) |
| clean F1 drop | ≤ 0.5 | −0.94 | no |
| matched F1 drop | ≤ 0.5 (expected) | −2.25 (top-5), −2.78 (ep8) | no |
| held-out F1 gain | ≥ +0.5 on a readout | −0.06 (top-5), −0.57 (ep8) | no |
| falsifier: held-out < +0.5 on both readouts AND matched drop > 0.5 | | both true | **hit → drop the lever** |

**Decision: drop `tel_codec` at 25% replacement.** B4a stays the reference.

## 7. Caveats

- **One seed per run.** The CIs cover the sampling of test clips, not training randomness. Every epoch and every
  readout points the same way, so a second seed is very unlikely to reverse the sign.
- **The per-codec breakdowns** (section 4.4) use the full sets, including half A, and only ~90-260 bona fide clips per
  codec. Treat them as a diagnosis of where the loss comes from, not as decision numbers.
- **Progress / Codabench** scores of the B4e submissions are not used here (challenge rules: no analysis or model
  selection on the progress split).

## 8. What next

- **Do not stack `tel_codec` 0.25 onto later runs.** B4a, and its TIES k50 merge (93.48 on half B), stay the best
  local candidates.
- If phone-channel robustness is still wanted, keep the Opus exposure intact. Each option is a one-flag run against B4a:
  - **Stack instead of replace**: Opus app profile *then* telephony codec on a fraction of clips, so every clip still
    gets an app profile. This needs a small code change, since the lever currently occupies the codec slot.
  - **Lower dose**: `--aug_tel_p 0.05-0.10`.
  - **Watch G.722 and the wideband sets**, not only total held-out F1: this run shows the gain and the loss sit on
    different codec families.

### Submissions built (not uploaded)

All three pass `scripts/make_submission.py` (30,279 progress ids, each scored once, scores in [0, 1]):
- `SLS_setup/exp/B4e_..._20261026204847/model_ties_k50_lam1/submission.zip`: B4e's best local readout (93.02)
- `SLS_setup/exp/B4e_..._20261026204847/model_merging_top5/submission.zip`
- `SLS_setup/exp/B4e_..._20261026204847/final_epoch8/submission.zip`

Choosing on half B, none of them beats the existing B4a TIES k50 zip
(`SLS_setup/exp/B4a_..._20261018150134/model_ties_k50_lam1/submission.zip`).

### Files

- B4e local evals: `local_eval/{merge_top5,epoch_8,ties_k50_lam1}/` (`metrics.json`, `probe_report.json`,
  `compare_vs_*.txt`, `ci_vs_*.txt`, `compare_ci_vs_*.json`).
- Six-way table: `local_eval/merge_top5/compare_all_readouts.txt`.
- TIES merge: `ties_k50_lam1/{merged.pth,ties_config.json,merged.sources.txt}`, log `ties_k50_lam1_merge.log`.
- Registry rows to add once the Codabench scores are back (`/log-experiment`): B4e top-5, ep8, ties_k50_lam1.
