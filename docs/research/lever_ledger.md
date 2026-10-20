# Lever ledger

Every lever tried on the RTC-SDD detector, and every lever an rdloop cycle built. The research team reads
this before proposing anything (rdloop AUDIT / SCOUT / PROPOSE) so exhausted directions are not re-opened
without new evidence.

- **Seeded 2026-10-11** by a script from `docs/experiment_log.csv` (the numbers are copied from it; the
  per-run reports in `docs/experiments-exploration/` hold the analysis). Readout = half_B WF1 of the
  top-5 merge (`-screen`) or the final EMA epoch (`-ep<N>`); the CI is the paired bootstrap of the
  change against that row's reference, where logged.
- **rdloop appends** one row per implemented / dropped lever at the end of each cycle (`wave result`
  = pending until the exploop wave reports; fill it in at `/exp-loop wrapup`).

## Tried in the exploration waves (from the registry)

| date | wave | registry id | change vs reference | half_B WF1 | CI of Δ | decision |
|---|---|---|---|---|---|---|
| 2026-10-03 | LOCAL | `LOCAL-qwen17-e2e-ep35` |  |  |  |  |
| 2026-10-03 | LOCAL | `LOCAL-qwen17-e2e-merge5` |  | 72.65 |  |  |
| 2026-10-03 | LOCAL | `LOCAL-qwen17-frozen-best` |  | 65.37 |  |  |
| 2026-10-04 | B0 | `B0-level-screen` | baseline (local recipe + RTC aug, train pools) | 80.94 |  |  |
| 2026-10-04 | L1 | `L1-level-screen` | B0 + level aug (A2.7a) | 79.90 |  |  |
| 2026-10-07 | B0 | `B0-merge-top5` | B0-level-screen run, merge of the top-5 epochs by half-A WF1 (6,5,3,2,7) | 83.61 |  | promote |
| 2026-10-07 | B1 | `B1-stab-screen` | B0-merge-top5 + MUSAN + level aug + stability recipe (AdamW, 10x head LR, warmup-cosine 30ep early-stopped at 13, clip 1, EMA 0.9995, label smoothing 0.05); mer | 81.11 |  | drop |
| 2026-10-08 | B2 | `B2a-cos8-screen` | B1 aug (RTC+MUSAN+level) + stability recipe WITHOUT label smoothing; 8-epoch annealed warmup(5%)-cosine, lr 3e-5/head 3e-4, EMA 0.9995, no early stop, top-7 kep | 84.95 |  | promote |
| 2026-10-08 | B2 | `B2b-lr1e5-screen` | B2a with --lr 1e-5 (head 1e-4) instead of 3e-5; merge top-5 (ep 7,5,8,2,4) | 89.59 |  | promote |
| 2026-10-08 | B2 | `B2b-lr1e5-ep8` | B2b final-epoch EMA checkpoint alone (no merge) | 90.34 |  | revisit |
| 2026-10-08 | B2 | `B2c-lr5e6-screen` | B2b with --lr 5e-6 (head 5e-5) instead of 1e-5; merge top-5 (ep 7,8,6,5,3) | 91.73 |  | promote |
| 2026-10-08 | B2 | `B2c-lr5e6-ep8` | B2c final-epoch EMA checkpoint alone (no merge) | 92.26 |  | promote |
| 2026-10-08 | B2 | `B2d-cos10-screen` | B2c with --num_epochs 10 instead of 8 (same warmup frac / cosine floor); merge top-5 (ep 10,8,9,3,6) | 91.57 |  | drop |
| 2026-10-08 | B2 | `B2d-cos10-ep10` | B2d final-epoch EMA checkpoint alone | 91.84 |  | drop |
| 2026-10-08 | B2 | `B2e-lr2.5e6-screen` | B2c with the LR schedule halved: --lr 2.5e-6 --lr_min 5e-7 (same cosine shape); merge top-5 (ep 8,4,6,5,7) | 92.34 |  | revisit |
| 2026-10-08 | B2 | `B2e-lr2.5e6-ep8` | B2e final-epoch EMA checkpoint alone | 92.53 |  | revisit |
| 2026-10-09 | B2 | `B2f-loudnorm-screen` | B2e with --use_level_aug replaced by --loudness_norm --loudness_norm_lufs -23 (deterministic BS.1770 loudness normalisation at read time, train+dev+eval, before | 91.98 |  |  |
| 2026-10-09 | B2 | `B2f-loudnorm-ep8` | B2f final-epoch EMA checkpoint alone (no merge) | 92.14 |  |  |
| 2026-10-10 | B3 | `B3a-cjs03-screen` | + --consistency js --consistency_lambda 0.3 --consistency_pairs both (--batch_size 8 = 8 pairs = 16 clips/step) | 93.06 | [0.52, 1.67] | promote |
| 2026-10-10 | B3 | `B3a-cjs03-ep8` | + --consistency js --consistency_lambda 0.3 --consistency_pairs both (--batch_size 8 = 8 pairs = 16 clips/step) | 92.94 | [0.22, 1.42] | revisit |
| 2026-10-10 | B3 | `B3b-sampbal-screen` | + --sampler balanced --sampler_lang_balance --sampler_bona_frac 0.5 --ce_weights 0.3223 0.6777 (coupled sampler group; CE re-derived to keep the 0.1/0.9 effecti | 92.23 | [-1.46, -0.25] | drop |
| 2026-10-10 | B3 | `B3b-sampbal-ep8` | + --sampler balanced --sampler_lang_balance --sampler_bona_frac 0.5 --ce_weights 0.3223 0.6777 (coupled sampler group; CE re-derived to keep the 0.1/0.9 effecti | 92.36 | [-1.21, 0.02] | drop |
| 2026-10-11 | B3 | `B3c-cjs10-screen` | --consistency_lambda 0.3 -> 1.0 (JS consistency dose step; everything else identical to B3a, incl. --consistency js --consistency_pairs both --batch_size 8) | 92.29 | [-1.31, -0.22] | drop |
| 2026-10-11 | B3 | `B3c-cjs10-ep8` | --consistency_lambda 0.3 -> 1.0 (JS consistency dose step; everything else identical to B3a, incl. --consistency js --consistency_pairs both --batch_size 8) | 92.44 | [-1.05, 0.05] | drop |
| 2026-10-11 | B3 | `B3d-codecrand-screen` | + --aug_codec_profiles random (per-clip random Opus bitrate {6,8,12,16,24} kbps / frame {20,40,60} ms / VBR mode / packet-loss hint; everything else identical t | 90.86 | [-2.90, -1.57] | drop |
| 2026-10-11 | B3 | `B3d-codecrand-ep8` | + --aug_codec_profiles random (per-clip random Opus bitrate {6,8,12,16,24} kbps / frame {20,40,60} ms / VBR mode / packet-loss hint; everything else identical t | 90.72 | [-2.99, -1.52] | drop |

## Built by rdloop cycles

| date | cycle | lever | flags | gaps | status | smoke | wave result | sources |
|---|---|---|---|---|---|---|---|---|
| 2026-10-18 | B4 | `apm_ns` | --aug_apm_p --aug_apm_ns_levels | G1 | implemented (b4wave) | pass 1.126 | pending | https://arxiv.org/html/2607.17761 https://arxiv.org/html/2604.23742v1 https://arxiv.org/abs/2603.14033 |
| 2026-10-18 | B4 | `dtx_gate` | --aug_dtx_p --aug_dtx_fill | G3, G1 | implemented (b4wave) | pass 0.928 | pending | https://arxiv.org/html/2309.11827 https://arxiv.org/abs/2309.11827 https://arxiv.org/abs/2106.12914 |
| 2026-10-18 | B4 | `opus_plc` | --aug_opus_plc_p --aug_opus_loss --aug_opus_burst | G1 | implemented (b4wave) | pass 0.913 | pending | https://arxiv.org/html/2504.12423v2 https://arxiv.org/html/2604.23742v1 https://github.com/orion-labs/opuslib |
