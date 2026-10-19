#!/usr/bin/env bash
# Phase-0 smoke tests of the b3wave package (plan: /root/.claude/plans/..., section "Smoke tests").
#   bash scripts/b3_smoke.sh            # everything (GPU steps included; ~1 h)
#   bash scripts/b3_smoke.sh cpu        # only the CPU steps (1-6, 9)
# Each step prints PASS/FAIL; the exit code is non-zero when anything failed.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/root/.conda/envs/rtc-sdd/bin/python}"
MODE="${1:-all}"
B2E="$REPO/SLS_setup/exp/B2e_qwen17_rtc_musan_lvl_cos8_lr2.5e6_s1234_epoch8_bs16_20261008185925"
NAMES4="dev_online_clean sim_heldout_v1"
fail=0
# each step runs in a subshell started from the repo root, so a `cd` inside one step cannot leak into the next
step() { if ( cd "$REPO" && eval "$2" ) > /tmp/b3_smoke_step.log 2>&1; then echo "PASS $1"; else echo "FAIL $1 (see /tmp/b3_smoke_step.log)"; tail -5 /tmp/b3_smoke_step.log; fail=$((fail+1)); fi; }

cd "$REPO"
# 1 isolation: no pre-existing file modified, only new paths under the B3 locations
step "1 isolation (git diff empty, only new B3 paths)" '
  [ -z "$(git diff --name-only | grep -v "^configs/local_eval_sets.yaml$")" ] && ! git status --porcelain | grep -v "^?? " | grep -vq "configs/local_eval_sets.yaml" &&
  ! git status --porcelain | grep "^?? " | grep -vE "SLS_setup/(b3wave/|main_train_b3.py|exp_smoke|exp_smoke_train.log)|scripts/b3_|data/dev_noisy_sim/unseen_|\.claude/|\.vscode/" | grep -q .'
# 2 CLIs
step "2 CLI (b3wave.train --help; main_train.py unchanged)" '
  cd SLS_setup && "$PY" -m b3wave.train --help | grep -q "B3 wave" && ! "$PY" main_train.py --help | grep -qE "sampler|llrd|specaug|consistency"'
# 3-6 CPU self-checks
step "3-6 self-checks (augmenter repro, data, sampler, losses)" 'cd SLS_setup && "$PY" -m b3wave.checks all'
# 9 bootstrap
step "9 bootstrap (self CI [0,0]; B2d vs B2c contains 0; B2b vs B2a excludes 0)" '
  cd SLS_setup &&
  C=exp/B2c_qwen17_rtc_musan_lvl_cos8_lr5e6_s1234_epoch8_bs16_20261008085023/local_eval/merge_top5 &&
  D=exp/B2d_qwen17_rtc_musan_lvl_cos10_lr5e6_s1234_epoch10_bs16_20261008132750/local_eval/merge_top5 &&
  A=exp/B2a_qwen17_rtc_musan_lvl_cos8_s1234_epoch8_bs16_20261007231952/local_eval/merge_top5 &&
  B=exp/B2b_qwen17_rtc_musan_lvl_cos8_lr1e5_s1234_epoch8_bs16_20261008040736/local_eval/merge_top5 &&
  "$PY" -m b3wave.metrics_ci $C $C --n 200 | grep -q "wf1 .* +0.00  \[ +0.00,  +0.00\]" &&
  "$PY" -m b3wave.metrics_ci $C $D --n 500 --json /tmp/b3_ci_dc.json && "$PY" -c "
import json; r=json.load(open(\"/tmp/b3_ci_dc.json\"))[\"wf1\"]; assert r[\"ci_low\"] < 0 < r[\"ci_high\"], r; assert abs(r[\"point_a\"]-0.9173)<5e-4, r" &&
  "$PY" -m b3wave.metrics_ci $A $B --n 500 --json /tmp/b3_ci_ab.json && "$PY" -c "
import json; r=json.load(open(\"/tmp/b3_ci_ab.json\"))[\"wf1\"]; assert r[\"ci_low\"] > 0, r"'
if [ "$MODE" = "cpu" ]; then echo "cpu steps done: $fail failed"; exit $fail; fi

# 7-8 GPU self-checks
step "7 model (ModelSLSB3 == ModelSLS, key-exact ckpt, SpecAugment)" 'cd SLS_setup && "$PY" -m b3wave.checks model'
step "8 optim (LLRD groups, freeze_bottom_k)" 'cd SLS_setup && "$PY" -m b3wave.checks optim'
# 10 multi-crop: first reproduces B2e epoch_8 on two sets; mean runs
step "10 multicrop (first == local_eval; mean runs)" '
  OUT=/tmp/b3_mc && rm -rf $OUT && mkdir -p $OUT &&
  "$PY" scripts/b3_local_eval.py --model_path "$B2E/ckpt/epoch_8_dev_loss_0.126069.pth" --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24 --out $OUT/first --names '"$NAMES4"' --multicrop first &&
  "$PY" - <<PYE
import json
a=json.load(open("$OUT/first/metrics.json"))["full"]["sets"]; b=json.load(open("$B2E/local_eval/epoch_8/metrics.json"))["full"]["sets"]
for n in "'"$NAMES4"'".split():
    for k in ("f1","eer","recall_bona"):
        assert abs(a[n][k]-b[n][k])<1e-9, (n,k,a[n][k],b[n][k])
print("first-crop identical")
PYE
  "$PY" scripts/b3_local_eval.py --model_path "$B2E/ckpt/epoch_8_dev_loss_0.126069.pth" --ssl_name Qwen/Qwen3-ASR-1.7B --n_layers 24 --out $OUT/mean --names sim_heldout_v1 --multicrop mean | grep -q "crops/clip"'
# 11 trainer end to end (one-epoch subset runs, every lever)
step "11 trainer smoke runs (a-k)" 'bash scripts/b3_smoke_train.sh'
# 12 unseen set (smoke: 24 files into a scratch version)
step "12 unseen-set builder (--limit 24, version smoke)" '
  rm -rf /tmp/b3_unseen && "$PY" scripts/b3_build_unseen_set.py --limit 24 --version smoke --workers 8 --out /tmp/b3_unseen --no_yaml &&
  [ -f /tmp/b3_unseen/unseen_smoke/MANIFEST.json ] && [ "$(wc -l < /tmp/b3_unseen/unseen_smoke/protocol.txt)" -eq 24 ]'
echo "b3_smoke: $fail step(s) failed"
exit $fail
