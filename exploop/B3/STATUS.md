# exploop wave B3 - DONE

Updated 2026-10-12 01:31:00 - supervisor pid None - heartbeat 2026-10-12 01:31:00 - container same boot - Claude spend $31.90 of $150

## Best so far

| label | run | merge (objective) | final epoch | registry ids | submission |
|---|---|---|---|---|---|
| B3a | `B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_epoch8_bs8_20261009214846` | 93.06 | 92.94 | B3a-cjs03-screen, B3a-cjs03-ep8 | SLS_setup/exp/B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234_epoch8_bs8_20261009214846/model_merging_top5/submission.zip |

## Iterations

| # | track | change | status | merge | delta | CI | final | delta | decision | verdict | report |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234` | + --consistency js --consistency_lambda 0.3 --consistency_pairs both (--batch_size 8 = 8 pairs = 16 clips/step) | done | 93.06 | +1.08 | [+0.52, +1.67] | 92.94 | +0.80 | promote | positive | [report](../../docs/experiments-exploration/b3-experiments/20261010_143058_B3a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_s1234.md) |
| 2 | `B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_sampbal_s1234` | + --sampler balanced --sampler_lang_balance --sampler_bona_frac 0.5 --ce_weights 0.3223 0.6777 (coupled sampler group; CE re-derived to keep the 0.1/0.9 effective class prior) | done | 92.23 | -0.83 | [-1.46, -0.25] | 92.36 | -0.58 | drop | negative | [report](../../docs/experiments-exploration/b3-experiments/20261010_231745_B3b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_sampbal_s1234.md) |
| 3 | `B3c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs10_s1234` | --consistency_lambda 0.3 -> 1.0 (JS consistency dose step; everything else identical to B3a, incl. --consistency js --consistency_pairs both --batch_size 8) | done | 92.29 | -0.77 | [-1.31, -0.22] | 92.44 | -0.50 | drop | negative | [report](../../docs/experiments-exploration/b3-experiments/20261011_075030_B3c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs10_s1234.md) |
| 4 | `B3d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_codecrand_s1234` | + --aug_codec_profiles random (per-clip random Opus bitrate {6,8,12,16,24} kbps / frame {20,40,60} ms / VBR mode / packet-loss hint; everything else identical to B3a, incl. --consistency js --consistency_lambda 0.3 --consistency_pairs both --batch_size 8) | done | 90.86 | -2.20 | [-2.90, -1.57] | 90.72 | -2.22 | drop | negative | [report](../../docs/experiments-exploration/b3-experiments/20261011_163014_B3d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_codecrand_s1234.md) |
| 5 | `B3e_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_snr0to20_s1234` | + --aug_snr_db 0 20 (default 5 20; SNR coupled group, --aug_snr_dist stays uniform; everything else identical to B3a, incl. --consistency js --consistency_lambda 0.3 --consistency_pairs both --batch_size 8, --aug_codec_profiles fixed) | done | 91.99 | -1.06 | [-1.65, -0.49] | 92.36 | -0.58 | drop | negative | [report](../../docs/experiments-exploration/b3-experiments/20261012_010655_B3e_qwen17_rtc_musan_loud23_cos8_lr2.5e6_cjs03_snr0to20_s1234.md) |

## Next

Wave finished. Run `/exp-loop wrapup B3` in an interactive Claude session (memory update, list of changed files), upload the best submission.zip by hand.

## Claude steps (inspect with `claude --resume <session>`)

- iter 1 analyze attempt 0: `0a1c3535-57d9-4a98-ab5e-7126495bb1ce` (ok, $4.66)
- iter 2 design attempt 0: `6069c1f8-f0e8-4ce7-bbb5-08f84edeb552` (ok, $0.99)
- iter 2 analyze attempt 0: `1b629709-d017-40a1-bcea-68f1d5a4050e` (ok, $3.71)
- iter 3 design attempt 0: `8d893f39-cf77-411c-9a73-16abd67ce0f1` (ok, $0.98)
- iter 3 analyze attempt 0: `28a8f0ab-2632-4804-b461-d526965370a2` (ok, $4.93)
- iter 4 design attempt 0: `446b05f9-d466-4faa-bfe3-493a14171ed3` (ok, $1.32)
- iter 4 analyze attempt 0: `ae98f3e2-94ea-4201-875d-03512b3f322a` (ok, $5.24)
- iter 5 design attempt 0: `85934c50-00a0-49ec-840b-b2b689339b26` (ok, $1.33)
- iter 5 analyze attempt 0: `a8c8096e-705f-41f7-b346-433d0f681af0` (ok, $5.56)
- iter 0 summary attempt 0: `a914dbbe-cc4f-4780-ab91-5e6a6168b978` (ok, $3.19)

## Commands

```
scripts/exploop/loopctl.sh status B3      # health + exit code
scripts/exploop/loopctl.sh tail B3        # recent events
scripts/exploop/loopctl.sh note B3 "..."  # steer the next design step
scripts/exploop/loopctl.sh pause B3       # no new GPU job (running ones continue)
scripts/exploop/loopctl.sh stop B3        # supervisor exits; running jobs continue
scripts/exploop/loopctl.sh resume B3      # after BLOCKED / DOWN / container restart
```
