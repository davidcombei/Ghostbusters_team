# exploop wave B4 - DONE

Updated 2026-10-19 11:20:29 - supervisor pid None - heartbeat 2026-10-19 11:20:29 - container same boot - Claude spend $32.58 of $1000000

## Best so far

| label | run | merge (objective) | final epoch | registry ids | submission |
|---|---|---|---|---|---|
| B2f | `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623` | 91.98 | 92.14 | B2f-loudnorm-screen, B2f-loudnorm-ep8 | SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623/model_merging_top5/submission.zip |

## Iterations

| # | track | change | status | merge | delta | CI | final | delta | decision | verdict | report |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234` | + --aug_apm_p 0.3 (WebRTC NS pass on 30% of clips; --aug_apm_ns_levels left at default 6 12 18 21) | done | 92.55 | +0.57 | [-0.01, +1.17] | 93.15 | +1.02 | revisit | above band, CI includes 0 | [report](../../docs/experiments-exploration/b4-experiments/20261018_194212_B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234.md) |
| 2 | `B4b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_opusplc03_s1234` | + --aug_opus_plc_p 0.3 (in-process libopus round trip with Gilbert-Elliott loss and decoder PLC on 30% of codec calls; --aug_opus_loss 0.01 0.20 and --aug_opus_burst 1 4 left at default) | done | 92.22 | +0.24 | [-0.30, +0.78] | 92.26 | +0.12 | revisit | null (inside +-0.5) | [report](../../docs/experiments-exploration/b4-experiments/20261019_001203_B4b_qwen17_rtc_musan_loud23_cos8_lr2.5e6_opusplc03_s1234.md) |
| 3 | `B4c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_dtx03_s1234` | + --aug_dtx_p 0.3 (webrtcvad DTX gating on 30% of clips; --aug_dtx_fill left at default mix) | done | 92.01 | +0.03 | [-0.47, +0.51] | 92.34 | +0.20 | revisit | null (inside +-0.5) | [report](../../docs/experiments-exploration/b4-experiments/20261019_052003_B4c_qwen17_rtc_musan_loud23_cos8_lr2.5e6_dtx03_s1234.md) |
| 4 | `B4d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns06_s1234` | + --aug_apm_p 0.6 (WebRTC NS pass on 60% of clips; --aug_apm_ns_levels left at default 6 12 18 21) | done | 92.14 | +0.17 | [-0.39, +0.74] | 92.34 | +0.20 | revisit | null (inside +-0.5) | [report](../../docs/experiments-exploration/b4-experiments/20261019_104713_B4d_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns06_s1234.md) |

## Next

Wave finished. Run `/exp-loop wrapup B4` in an interactive Claude session (memory update, list of changed files), upload the best submission.zip by hand.

## Claude steps (inspect with `claude --resume <session>`)

- iter 1 design attempt 0: `222bb1f9-5e34-4e2f-999a-d408b33f4134` (ok, $0.56)
- iter 1 analyze attempt 0: `dc7185c9-a8b3-4e93-992f-22bd00801c97` (ok, $5.74)
- iter 2 design attempt 0: `ed7b85c1-6c3c-4846-b1a1-a64a39f96b7a` (ok, $0.75)
- iter 2 analyze attempt 0: `74c84a24-d8eb-43b4-b70c-9c04ff055136` (ok, $5.58)
- iter 3 design attempt 0: `fd4aece7-d237-4c40-96de-ed6351f6c5c9` (ok, $0.87)
- iter 3 analyze attempt 0: `f4832176-fd2f-4934-8604-a098498d1c47` (ok, $5.97)
- iter 4 design attempt 0: `00bad5f6-4818-4ec7-ac6a-14db4fb934f1` (ok, $1.36)
- iter 4 analyze attempt 0: `07d221c9-d46e-4820-8cf6-69df14ef76cd` (ok, $6.50)
- iter 0 summary attempt 0: `5bb00878-1e65-4d03-8eb3-c89967333735` (ok, $5.24)

## Commands

```
scripts/exploop/loopctl.sh status B4      # health + exit code
scripts/exploop/loopctl.sh tail B4        # recent events
scripts/exploop/loopctl.sh note B4 "..."  # steer the next design step
scripts/exploop/loopctl.sh pause B4       # no new GPU job (running ones continue)
scripts/exploop/loopctl.sh stop B4        # supervisor exits; running jobs continue
scripts/exploop/loopctl.sh resume B4      # after BLOCKED / DOWN / container restart
```
