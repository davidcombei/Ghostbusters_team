# exploop wave B4dry - INITIALISED

Updated 2026-10-18 11:34:39 - supervisor pid None - heartbeat None - container same boot - Claude spend $0.00 of $150

## Best so far

| label | run | merge (objective) | final epoch | registry ids | submission |
|---|---|---|---|---|---|
| B2f | `B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623` | 91.98 | 92.14 | B2f-loudnorm-screen, B2f-loudnorm-ep8 | SLS_setup/exp/B2f_qwen17_rtc_musan_loud23_cos8_lr2.5e6_s1234_epoch8_bs16_20261009132623/model_merging_top5/submission.zip |

## Iterations

| # | track | change | status | merge | delta | CI | final | delta | decision | verdict | report |
|---|---|---|---|---|---|---|---|---|---|---|---|

## Commands

```
scripts/exploop/loopctl.sh status B4dry      # health + exit code
scripts/exploop/loopctl.sh tail B4dry        # recent events
scripts/exploop/loopctl.sh note B4dry "..."  # steer the next design step
scripts/exploop/loopctl.sh pause B4dry       # no new GPU job (running ones continue)
scripts/exploop/loopctl.sh stop B4dry        # supervisor exits; running jobs continue
scripts/exploop/loopctl.sh resume B4dry      # after BLOCKED / DOWN / container restart
```
