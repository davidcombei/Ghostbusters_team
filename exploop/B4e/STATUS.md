# exploop wave B4e - INITIALISED (iteration 1/1 B4e, phase TRAIN)

Updated 2026-10-26 21:06:18 - supervisor pid None - heartbeat None - container same boot - Claude spend $0.00 of $1000000

## Best so far

| label | run | merge (objective) | final epoch | registry ids | submission |
|---|---|---|---|---|---|
| B4a | `B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234_epoch8_bs16_20261018150134` | 92.55 | 93.15 | B4a-apmns-screen, B4a-apmns-ep8 | SLS_setup/exp/B4a_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_s1234_epoch8_bs16_20261018150134/model_merging_top5/submission.zip |

## Iterations

| # | track | change | status | merge | delta | CI | final | delta | decision | verdict | report |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `B4e_qwen17_rtc_musan_loud23_cos8_lr2.5e6_apmns03_tel25_s1234` | + --aug_tel_p 0.25 --aug_tel_codecs g723_1 amr_nb (telephony codecs on 25% of the codec calls) | TRAIN | – | – | – | – | – | – | – | – |

## Commands

```
scripts/exploop/loopctl.sh status B4e      # health + exit code
scripts/exploop/loopctl.sh tail B4e        # recent events
scripts/exploop/loopctl.sh note B4e "..."  # steer the next design step
scripts/exploop/loopctl.sh pause B4e       # no new GPU job (running ones continue)
scripts/exploop/loopctl.sh stop B4e        # supervisor exits; running jobs continue
scripts/exploop/loopctl.sh resume B4e      # after BLOCKED / DOWN / container restart
```
